"""GPU batch target generation for SATS on-the-fly GT training."""

from __future__ import annotations

import math

import torch

from .config import SATSConfig

GT_META_COLUMNS = ("diameter_mm", "x_mm", "y_mm", "z_depth_mm", "fz_n")


def _make_patch_template(radius: float, step: float, device: torch.device) -> torch.Tensor:
    if radius <= 0:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)
    n_half = int(radius / step)
    offsets = torch.arange(-n_half, n_half + 1, dtype=torch.float32, device=device) * float(step)
    gy, gx = torch.meshgrid(offsets, offsets, indexing="ij")
    inside = gx.square() + gy.square() <= float(radius * radius) + 1e-9
    pts = torch.stack((gx[inside], gy[inside]), dim=1)
    if pts.numel() == 0:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)
    return pts


class BatchGPUTargetGenerator:
    """Generate dense Boussinesq GT maps from compact batch metadata.

    The CPU on-the-fly path caches ``[center_y, center_x, H, W]`` lookup
    tables. That is fast at 41x41 but grows poorly for 0.25/0.1 mm grids.
    This class caches only the extended origin-centered base kernel per
    quantized contact radius and slices it on GPU for each batch row.
    """

    def __init__(self, cfg: SATSConfig, device: str | torch.device) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.grid_size = int(cfg.grid_size)
        self.grid_step_mm = float(cfg.grid_step_mm)
        self.grid_min_mm = float(cfg.grid_min_mm)
        self.grid_max_mm = float(cfg.grid_max_mm)
        self.ext_size = 2 * self.grid_size - 1
        self.ext_half = self.grid_size - 1
        self._base_cache: dict[tuple[float, float], torch.Tensor] = {}

        expected_size = int(round((self.grid_max_mm - self.grid_min_mm) / self.grid_step_mm)) + 1
        if expected_size != self.grid_size:
            raise ValueError(
                "grid_size must match grid_min/grid_max/grid_step: "
                f"expected {expected_size}, got {self.grid_size}"
            )

        rows = torch.arange(self.grid_size, device=self.device, dtype=torch.long)
        cols = torch.arange(self.grid_size, device=self.device, dtype=torch.long)
        self._row_offsets = rows.view(1, self.grid_size, 1)
        self._col_offsets = cols.view(1, 1, self.grid_size)

    def _radius_key(self, diameter_mm: torch.Tensor, z_depth_mm: torch.Tensor) -> torch.Tensor:
        sphere_radius = diameter_mm * 0.5
        radius = torch.sqrt(torch.clamp(sphere_radius * torch.clamp(z_depth_mm, min=0.0), min=0.0))
        radius = torch.clamp(radius, min=float(self.cfg.min_contact_radius_mm))
        radius = torch.minimum(radius, sphere_radius)
        step = float(self.cfg.contact_radius_step_mm)
        radius = torch.round(radius / step) * step
        radius = torch.clamp(radius, min=float(self.cfg.min_contact_radius_mm))
        radius = torch.minimum(radius, sphere_radius)
        return radius

    def _compute_base_kernel(self, radius: float) -> torch.Tensor:
        ext_half_width = self.grid_step_mm * self.ext_half
        ext_grid = torch.linspace(
            -ext_half_width,
            ext_half_width,
            self.ext_size,
            dtype=torch.float32,
            device=self.device,
        )
        patch = _make_patch_template(radius, float(self.cfg.on_the_fly_patch_step_mm), self.device)
        inv_r5_sum = torch.zeros((self.ext_size, self.ext_size), dtype=torch.float32, device=self.device)
        z_s = float(self.cfg.on_the_fly_z_s_mm)
        z_sq = z_s * z_s
        patch_step = float(self.cfg.on_the_fly_patch_step_mm)
        pre = (3.0 * (z_s ** 3) * (patch_step ** 2)) / (2.0 * math.pi ** 2 * radius ** 2)

        chunk = 256
        for start in range(0, patch.shape[0], chunk):
            pts = patch[start : start + chunk]
            px = pts[:, 0]
            py = pts[:, 1]
            r_sq = (
                (ext_grid[:, None, None] - py[None, None, :]).square()
                + (ext_grid[None, :, None] - px[None, None, :]).square()
                + z_sq
            )
            inv_r5_sum += torch.sum(torch.pow(r_sq, -2.5), dim=2)
        base = inv_r5_sum.mul(float(pre))

        # (B) β_geom force conservation: the base kernel approximates σ_zz / F,
        # so ∫∫ base dA must equal 1 (Boussinesq equilibrium). Coarse patch
        # discretization for small radii breaks this; rescale to restore it.
        if bool(getattr(self.cfg, "gt_beta_force_conservation", False)):
            integral = base.sum() * (self.grid_step_mm ** 2)
            if float(integral) > 1e-9:
                base = base.div(integral)
        return base.contiguous()

    def _base_kernel(self, diameter_mm: float, radius_mm: float) -> torch.Tensor:
        key = (float(diameter_mm), float(radius_mm))
        cached = self._base_cache.get(key)
        if cached is None or cached.device != self.device:
            cached = self._compute_base_kernel(float(radius_mm))
            self._base_cache[key] = cached
        return cached

    @torch.no_grad()
    def __call__(self, meta: torch.Tensor) -> torch.Tensor:
        """Return target maps for ``meta`` with columns ``GT_META_COLUMNS``."""

        if meta.ndim != 2 or meta.shape[1] != len(GT_META_COLUMNS):
            raise ValueError(f"expected meta [B, {len(GT_META_COLUMNS)}], got {tuple(meta.shape)}")
        meta = meta.to(device=self.device, dtype=torch.float32, non_blocking=True)
        diameter = meta[:, 0]
        x_mm = meta[:, 1]
        y_mm = meta[:, 2]
        z_depth = meta[:, 3]
        fz_n = meta[:, 4]

        B = meta.shape[0]
        out = torch.zeros((B, self.grid_size, self.grid_size), dtype=torch.float32, device=self.device)
        valid = (
            torch.isfinite(fz_n)
            & torch.isfinite(z_depth)
            & (fz_n > float(self.cfg.fz_min_abs_n))
            & (z_depth > float(self.cfg.z_depth_min_mm))
        )
        if not bool(valid.any()):
            return out

        ix = torch.round((x_mm - self.grid_min_mm) / self.grid_step_mm).to(torch.long)
        iy = torch.round((y_mm - self.grid_min_mm) / self.grid_step_mm).to(torch.long)
        ix = ix.clamp(0, self.grid_size - 1)
        iy = iy.clamp(0, self.grid_size - 1)
        radii = self._radius_key(diameter, z_depth)

        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        keys = torch.stack((diameter[valid_idx], radii[valid_idx]), dim=1)
        unique_keys = torch.unique(keys, dim=0)

        for key in unique_keys:
            d_val = float(key[0].item())
            r_val = float(key[1].item())
            sel = valid_idx[(diameter[valid_idx] == key[0]) & (radii[valid_idx] == key[1])]
            if sel.numel() == 0:
                continue

            base = self._base_kernel(d_val, r_val)
            row_idx = self.ext_half - iy[sel].view(-1, 1, 1) + self._row_offsets
            col_idx = self.ext_half - ix[sel].view(-1, 1, 1) + self._col_offsets
            maps = base[row_idx, col_idx]
            scale = fz_n[sel] * float(self.cfg.gt_scale)
            if str(getattr(self.cfg, "gt_beta_mode", "none")) == "poly2":
                scale = scale * self._beta_material(fz_n[sel], r_val)
            out[sel] = maps * scale.view(-1, 1, 1)

        return out

    def _beta_material(self, fz_n: torch.Tensor, radius_mm: float) -> torch.Tensor:
        """(C) β(p) rectification factor (paper S9), p = |fz|/(π·a²) in kPa.

        Mirrors generate_gt.compute_beta poly2 so on-the-fly GT matches the
        precomputed path. Material-specific coefficients (hyperelastic stiffening).
        """
        area = math.pi * float(radius_mm) * float(radius_mm)
        p_kpa = (fz_n.abs() / max(area, 1e-9)) * 1000.0   # N/mm² → kPa
        c0 = float(self.cfg.gt_beta_c0)
        c1 = float(self.cfg.gt_beta_c1)
        c2 = float(self.cfg.gt_beta_c2)
        beta = c0 + c1 * p_kpa + c2 * p_kpa * p_kpa
        return beta.clamp(float(self.cfg.gt_beta_min), float(self.cfg.gt_beta_max))
