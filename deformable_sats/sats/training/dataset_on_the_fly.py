"""
On-the-fly GT datasets for SATS training.

This module is intentionally separate from ``dataset.py`` so the legacy
precomputed ``*_targets.npy`` path remains available unchanged.
"""

from __future__ import annotations

import json
import logging
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from sats.preprocessing import generate_gt as gt
from sats.preprocessing.merged_bin import merged_bin_to_frame
from .config import SATSConfig, _parse_trial_id, filter_trial_ids
from .dataset import _load_baseline, _on_grid_mask, _resolve_merged_input, window_collate_fn
from .gt_gpu import GT_META_COLUMNS

log = logging.getLogger(__name__)

_SENSOR_COLS = [f"s{i}" for i in range(1, 17)]
_CACHE_FORMAT = "sats_on_the_fly_meta_cache_v1"


def spherical_contact_radius_mm(
    z_depth_mm: np.ndarray | float,
    sphere_radius_mm: float,
    min_radius_mm: float,
    max_radius_mm: float | None = None,
) -> np.ndarray:
    """Hertz-style spherical indenter contact radius ``a = sqrt(R * delta)``."""

    z = np.asarray(z_depth_mm, dtype=np.float64)
    radius = np.sqrt(np.maximum(float(sphere_radius_mm) * np.maximum(z, 0.0), 0.0))
    upper = float(sphere_radius_mm if max_radius_mm is None else max_radius_mm)
    return np.clip(radius, float(min_radius_mm), upper)


def _grid_indices(
    x_mm: float | np.ndarray,
    y_mm: float | np.ndarray,
    *,
    step: float,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = round((hi - lo) / step)
    ix = np.clip(np.round((np.asarray(x_mm, dtype=np.float64) - lo) / step), 0, n_steps).astype(np.int32)
    iy = np.clip(np.round((np.asarray(y_mm, dtype=np.float64) - lo) / step), 0, n_steps).astype(np.int32)
    return ix, iy


class OnTheFlyGTCache:
    """Small kernel cache for row-wise Boussinesq GT generation."""

    def __init__(
        self,
        *,
        z_s_mm: float = 2.0,
        patch_step_mm: float = 0.1,
        contact_radius_step_mm: float = 0.05,
        min_contact_radius_mm: float = 0.05,
        fz_min_n: float = 0.05,
        z_depth_min_mm: float = 0.001,
        grid_step_mm: float = 0.5,
        grid_min_mm: float = -10.0,
        grid_max_mm: float = 10.0,
        gt_scale: float = 100.0,
    ) -> None:
        self.z_s_mm = float(z_s_mm)
        self.patch_step_mm = float(patch_step_mm)
        self.contact_radius_step_mm = float(contact_radius_step_mm)
        self.min_contact_radius_mm = float(min_contact_radius_mm)
        self.fz_min_n = float(fz_min_n)
        self.z_depth_min_mm = float(z_depth_min_mm)
        self.grid_step_mm = float(grid_step_mm)
        self.grid_min_mm = float(grid_min_mm)
        self.grid_max_mm = float(grid_max_mm)
        self.gt_scale = float(gt_scale)
        self._kernel_cache: dict[tuple[float, float], np.ndarray] = {}
        self._zero = np.zeros((gt.GRID_SIZE, gt.GRID_SIZE), dtype=np.float32)

    def _radius_key(self, diameter_mm: float, z_depth_mm: float) -> float:
        sphere_radius = float(diameter_mm) / 2.0
        radius = spherical_contact_radius_mm(
            z_depth_mm,
            sphere_radius_mm=sphere_radius,
            min_radius_mm=self.min_contact_radius_mm,
            max_radius_mm=sphere_radius,
        )
        quantized = np.round(radius / self.contact_radius_step_mm) * self.contact_radius_step_mm
        return float(np.clip(quantized, self.min_contact_radius_mm, sphere_radius))

    def _all_kernels(self, diameter_mm: float, contact_radius_mm: float) -> np.ndarray:
        key = (float(diameter_mm), float(contact_radius_mm))
        if key not in self._kernel_cache:
            base = gt.compute_base_kernel(
                radius=float(contact_radius_mm),
                patch_step=self.patch_step_mm,
                z_s=self.z_s_mm,
            )
            self._kernel_cache[key] = gt.build_all_kernels(base)
        return self._kernel_cache[key]

    def map_for_row(
        self,
        *,
        diameter_mm: float,
        x_mm: float,
        y_mm: float,
        z_depth_mm: float,
        fz_n: float,
    ) -> np.ndarray:
        """Return a scaled ``41x41`` pressure map for one merged row."""

        if not np.isfinite(fz_n) or not np.isfinite(z_depth_mm):
            return self._zero.copy()
        if fz_n <= self.fz_min_n or z_depth_mm <= self.z_depth_min_mm:
            return self._zero.copy()

        ix, iy = _grid_indices(
            x_mm,
            y_mm,
            step=self.grid_step_mm,
            lo=self.grid_min_mm,
            hi=self.grid_max_mm,
        )
        radius = self._radius_key(diameter_mm, z_depth_mm)
        kernels = self._all_kernels(diameter_mm, radius)
        return (kernels[int(iy), int(ix)] * np.float32(fz_n * self.gt_scale)).astype(np.float32, copy=True)


def _load_merged_frame(path: Path, cfg: SATSConfig) -> Any:
    required = _SENSOR_COLS + ["x_mm", "y_mm", "z_depth_mm", "Fz"]
    optional = ["u_mm"]
    if path.suffix == ".bin":
        return merged_bin_to_frame(path, columns=required + optional)

    import pandas as pd

    header_cols = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in required if c not in header_cols]
    if missing:
        raise ValueError(f"{path}: missing columns for on-the-fly GT: {missing}")
    usecols = required + [c for c in optional if c in header_cols]
    df = pd.read_csv(path, usecols=usecols, dtype=np.float64)
    if "u_mm" not in df.columns:
        df["u_mm"] = 0.0
    return df


def _select_window_targets(
    fz_seq: np.ndarray,
    z_seq: np.ndarray,
    *,
    window_size: int,
    policy: str,
    loading_stride: int,
    plateau_stride: int,
    saturation_stride: int,
    saturation_fz_frac: float,
    saturation_z_frac: float,
    fz_bins: int,
    z_bins: int,
    z_bin_width_mm: float,
) -> np.ndarray:
    """Select target timesteps while preserving dynamic/static/saturation rows."""

    T = len(fz_seq)
    if T < window_size:
        return np.asarray([], dtype=np.int64)

    candidates = np.arange(window_size - 1, T, dtype=np.int64)
    fz_pos = np.maximum(fz_seq.astype(np.float64), 0.0)
    z_pos = np.maximum(z_seq.astype(np.float64), 0.0)
    peak_t = int(np.argmax(fz_pos))

    if policy == "all":
        return candidates
    if policy == "loading_only":
        return candidates[candidates <= peak_t]
    if policy != "balanced_contact":
        raise ValueError(f"unknown on-the-fly sampling policy: {policy}")

    selected: list[np.ndarray] = []

    loading = candidates[candidates <= peak_t]
    if len(loading):
        selected.append(loading[:: max(1, int(loading_stride))])

    dz = np.abs(np.diff(z_pos, prepend=z_pos[0]))
    df = np.abs(np.diff(fz_pos, prepend=fz_pos[0]))
    static = candidates[(dz[candidates] <= 1e-6) & (df[candidates] <= 0.02)]
    if len(static):
        selected.append(static[:: max(1, int(plateau_stride))])

    max_fz = float(np.max(fz_pos)) if len(fz_pos) else 0.0
    max_z = float(np.max(z_pos)) if len(z_pos) else 0.0
    saturation_mask = np.zeros(T, dtype=bool)
    if max_fz > 0:
        saturation_mask |= fz_pos >= (max_fz * float(saturation_fz_frac))
    if max_z > 0:
        saturation_mask |= z_pos >= (max_z * float(saturation_z_frac))
    saturation = candidates[saturation_mask[candidates]]
    if len(saturation):
        selected.append(saturation[:: max(1, int(saturation_stride))])

    # z/Fz 균형 샘플: 각 bin 조합에서 대표 시점 하나를 보존한다.
    contact = candidates[(fz_pos[candidates] > 0.0) & (z_pos[candidates] > 0.0)]
    if len(contact):
        if z_bin_width_mm > 0:
            z_min = float(z_pos[contact].min())
            z_max = float(z_pos[contact].max()) + 1e-9
            z_edges = np.arange(z_min, z_max + float(z_bin_width_mm), float(z_bin_width_mm))
            if len(z_edges) < 2:
                z_edges = np.array([z_min, z_max + float(z_bin_width_mm)], dtype=np.float64)
            z_bins_eff = len(z_edges) - 1
        else:
            z_edges = np.linspace(float(z_pos[contact].min()), float(z_pos[contact].max()) + 1e-9, max(2, z_bins + 1))
            z_bins_eff = z_bins
        f_edges = np.linspace(float(fz_pos[contact].min()), float(fz_pos[contact].max()) + 1e-9, max(2, fz_bins + 1))
        z_id = np.clip(np.digitize(z_pos[contact], z_edges) - 1, 0, z_bins_eff - 1)
        f_id = np.clip(np.digitize(fz_pos[contact], f_edges) - 1, 0, fz_bins - 1)
        by_bin: dict[tuple[int, int], int] = {}
        for t, zi, fi in zip(contact, z_id, f_id):
            by_bin.setdefault((int(zi), int(fi)), int(t))
        selected.append(np.fromiter(by_bin.values(), dtype=np.int64))

    if not selected:
        return np.asarray([], dtype=np.int64)
    return np.unique(np.concatenate(selected)).astype(np.int64)


def _load_trial_on_the_fly(trial_id: str, cfg: SATSConfig) -> list[dict] | None:
    log.info("[%s] on-the-fly trial 로드 시작", trial_id)
    paths = cfg.trial_paths(trial_id)
    merged_input = _resolve_merged_input(paths, cfg)
    merged_bin = paths.get("merged_bin")
    if merged_input is None:
        log.warning("[%s] merged BIN/CSV 없음", trial_id)
        return None

    baseline = _load_baseline(paths["baseline_json"], merged_bin=merged_bin)
    if baseline is None:
        log.warning("[%s] baseline 없음", trial_id)
        return None

    try:
        df = _load_merged_frame(merged_input, cfg)
    except ValueError as exc:
        log.warning("[%s] merged 로드 실패: %s", trial_id, exc)
        return None

    x_arr = df["x_mm"].to_numpy(dtype=np.float64)
    y_arr = df["y_mm"].to_numpy(dtype=np.float64)
    mask = _on_grid_mask(
        x_arr,
        y_arr,
        step=cfg.grid_step_mm,
        tol=cfg.grid_tol_mm,
        lo=cfg.grid_min_mm,
        hi=cfg.grid_max_mm,
    )
    if cfg.use_u_zero_only:
        mask &= np.abs(df["u_mm"].to_numpy(dtype=np.float64)) <= cfg.u_zero_tol_mm
    if not mask.any():
        log.warning("[%s] on-grid rows 없음", trial_id)
        return None

    df_grid = df[mask].reset_index(drop=True)
    s_raw = df_grid[_SENSOR_COLS].to_numpy(dtype=np.float64)
    s_norm = (((s_raw - baseline) / baseline) * 100.0).astype(np.float32)
    fz_grid = df_grid["Fz"].to_numpy(dtype=np.float32)
    z_grid = df_grid["z_depth_mm"].to_numpy(dtype=np.float32)

    ix, iy = _grid_indices(
        df_grid["x_mm"].to_numpy(dtype=np.float64),
        df_grid["y_mm"].to_numpy(dtype=np.float64),
        step=cfg.grid_step_mm,
        lo=cfg.grid_min_mm,
        hi=cfg.grid_max_mm,
    )
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row_i, key in enumerate(zip(ix.tolist(), iy.tolist())):
        groups[(int(key[0]), int(key[1]))].append(row_i)

    p = _parse_trial_id(trial_id)
    sequences: list[dict] = []
    for (x_idx, y_idx), idxs in groups.items():
        if len(idxs) < max(cfg.min_seq_len, cfg.window_size):
            continue
        idx_arr = np.asarray(idxs[: cfg.seq_len], dtype=np.int64)
        fz_seq = fz_grid[idx_arr]
        if np.max(fz_seq) <= cfg.fz_min_abs_n:
            continue
        z_seq = z_grid[idx_arr]
        targets = _select_window_targets(
            fz_seq,
            z_seq,
            window_size=cfg.window_size,
            policy=cfg.on_the_fly_sampling_policy,
            loading_stride=cfg.loading_stride,
            plateau_stride=cfg.plateau_stride,
            saturation_stride=cfg.saturation_stride,
            saturation_fz_frac=cfg.saturation_fz_frac,
            saturation_z_frac=cfg.saturation_z_frac,
            fz_bins=cfg.fz_balance_bins,
            z_bins=cfg.z_balance_bins,
            z_bin_width_mm=cfg.z_balance_bin_width_mm,
        )
        if len(targets) == 0:
            continue
        sequences.append(
            {
                "trial_id": trial_id,
                "diameter_mm": float(p["d"]),
                "x_idx": x_idx,
                "y_idx": y_idx,
                "sensor_seq": s_norm[idx_arr],
                "x_seq": df_grid["x_mm"].to_numpy(dtype=np.float32)[idx_arr],
                "y_seq": df_grid["y_mm"].to_numpy(dtype=np.float32)[idx_arr],
                "z_seq": z_seq,
                "fz_seq": fz_seq,
                "target_ts": targets,
            }
        )

    log.info("[%s] on-the-fly sequences=%d", trial_id, len(sequences))
    return sequences


def meta_cache_signature(cfg: SATSConfig) -> dict[str, Any]:
    """Return the preprocessing settings that define compact meta-cache content."""

    return {
        "grid_step_mm": float(cfg.grid_step_mm),
        "grid_tol_mm": float(cfg.grid_tol_mm),
        "grid_min_mm": float(cfg.grid_min_mm),
        "grid_max_mm": float(cfg.grid_max_mm),
        "fz_min_abs_n": float(cfg.fz_min_abs_n),
        "use_u_zero_only": bool(cfg.use_u_zero_only),
        "u_zero_tol_mm": float(cfg.u_zero_tol_mm),
        "prefer_merged_bin": bool(cfg.prefer_merged_bin),
        "seq_len": int(cfg.seq_len),
        "min_seq_len": int(cfg.min_seq_len),
        "window_size": int(cfg.window_size),
        "sampling_policy": str(cfg.on_the_fly_sampling_policy),
        "loading_stride": int(cfg.loading_stride),
        "plateau_stride": int(cfg.plateau_stride),
        "saturation_stride": int(cfg.saturation_stride),
        "saturation_fz_frac": float(cfg.saturation_fz_frac),
        "saturation_z_frac": float(cfg.saturation_z_frac),
        "fz_balance_bins": int(cfg.fz_balance_bins),
        "z_balance_bins": int(cfg.z_balance_bins),
        "z_balance_bin_width_mm": float(cfg.z_balance_bin_width_mm),
    }


def meta_cache_path(cfg: SATSConfig, trial_id: str) -> Path:
    sig_json = json.dumps(meta_cache_signature(cfg), sort_keys=True, separators=(",", ":"))
    sig_hash = hashlib.sha1(sig_json.encode("utf-8")).hexdigest()[:10]
    return Path(cfg.gt_meta_cache_dir) / f"{trial_id}_{sig_hash}_meta_cache.pt"


def save_trial_meta_cache(trial_id: str, cfg: SATSConfig, *, overwrite: bool = False) -> Path | None:
    """Build and save compact sensor/meta cache for one trial."""

    out_path = meta_cache_path(cfg, trial_id)
    if out_path.exists() and not overwrite:
        log.info("[%s] meta cache already exists: %s", trial_id, out_path)
        return out_path

    seqs = _load_trial_on_the_fly(trial_id, cfg)
    if not seqs:
        log.warning("[%s] meta cache 생성 실패: 유효 sequence 없음", trial_id)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": _CACHE_FORMAT,
        "trial_id": trial_id,
        "signature": meta_cache_signature(cfg),
        "sequences": seqs,
    }
    torch.save(payload, out_path)
    log.info("[%s] meta cache 저장: %s sequences=%d", trial_id, out_path, len(seqs))
    return out_path


def _load_trial_meta_cache(trial_id: str, cfg: SATSConfig) -> list[dict] | None:
    path = meta_cache_path(cfg, trial_id)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - defensive corruption path
        log.warning("[%s] meta cache 로드 실패: %s (%s)", trial_id, path, exc)
        return None
    if payload.get("format") != _CACHE_FORMAT:
        log.warning("[%s] meta cache format 불일치: %s", trial_id, path)
        return None
    if payload.get("trial_id") != trial_id:
        log.warning("[%s] meta cache trial_id 불일치: %s", trial_id, path)
        return None
    if payload.get("signature") != meta_cache_signature(cfg):
        log.warning("[%s] meta cache 설정 불일치, merged BIN fallback: %s", trial_id, path)
        return None
    seqs = payload.get("sequences")
    if not isinstance(seqs, list) or not seqs:
        log.warning("[%s] meta cache sequence 없음: %s", trial_id, path)
        return None
    log.info("[%s] meta cache 로드: %s sequences=%d", trial_id, path, len(seqs))
    return seqs


def _load_trial_cached_or_merged(trial_id: str, cfg: SATSConfig) -> list[dict] | None:
    if cfg.use_gt_meta_cache:
        cached = _load_trial_meta_cache(trial_id, cfg)
        if cached:
            return cached
    return _load_trial_on_the_fly(trial_id, cfg)


class SATSOnTheFlyWindowDataset(Dataset):
    """Window dataset that generates spherical-indenter GT maps on demand."""

    def __init__(
        self,
        trial_ids: list[str],
        cfg: SATSConfig,
        *,
        _preloaded: dict[str, Any] | None = None,
        return_gt_meta: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.return_gt_meta = bool(return_gt_meta)
        self._sequences: dict[str, list[dict]] = {}
        self._index: list[tuple[str, int, int]] = []
        self._gt_cache = None
        if not self.return_gt_meta:
            self._gt_cache = OnTheFlyGTCache(
                z_s_mm=cfg.on_the_fly_z_s_mm,
                patch_step_mm=cfg.on_the_fly_patch_step_mm,
                contact_radius_step_mm=cfg.contact_radius_step_mm,
                min_contact_radius_mm=cfg.min_contact_radius_mm,
                fz_min_n=cfg.fz_min_abs_n,
                z_depth_min_mm=cfg.z_depth_min_mm,
                grid_step_mm=cfg.grid_step_mm,
                grid_min_mm=cfg.grid_min_mm,
                grid_max_mm=cfg.grid_max_mm,
                gt_scale=cfg.gt_scale,
            )

        if _preloaded is not None:
            self._sequences = _preloaded["sequences_by_trial"]
        else:
            for tid in trial_ids:
                seqs = _load_trial_cached_or_merged(tid, cfg)
                if seqs:
                    self._sequences[tid] = seqs

        for tid, seqs in self._sequences.items():
            for seq_i, seq in enumerate(seqs):
                for t in seq["target_ts"]:
                    self._index.append((tid, seq_i, int(t)))

        if not self._index:
            raise RuntimeError("유효한 on-the-fly GT 윈도우가 없습니다.")
        log.info("OnTheFlyWindowDataset: %d windows", len(self._index))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tid, seq_i, t = self._index[idx]
        seq = self._sequences[tid][seq_i]
        w = self.cfg.window_size
        sensor_window = torch.from_numpy(seq["sensor_seq"][t - w + 1 : t + 1].copy())
        if self.return_gt_meta:
            gt_meta = np.array(
                [
                    float(seq["diameter_mm"]),
                    float(seq["x_seq"][t]),
                    float(seq["y_seq"][t]),
                    float(seq["z_seq"][t]),
                    float(seq["fz_seq"][t]),
                ],
                dtype=np.float32,
            )
            return sensor_window, torch.from_numpy(gt_meta)

        assert self._gt_cache is not None
        gt_map = self._gt_cache.map_for_row(
            diameter_mm=seq["diameter_mm"],
            x_mm=float(seq["x_seq"][t]),
            y_mm=float(seq["y_seq"][t]),
            z_depth_mm=float(seq["z_seq"][t]),
            fz_n=float(seq["fz_seq"][t]),
        )
        return sensor_window, torch.from_numpy(gt_map)


def gt_meta_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate ``sensor_window`` and compact GT metadata for GPU target generation."""

    sensor_wins, gt_metas = zip(*batch)
    B = len(sensor_wins)
    window_size = sensor_wins[0].shape[0]
    lengths = torch.full((B,), window_size, dtype=torch.int64)
    meta_batch = torch.stack(gt_metas)
    if meta_batch.shape[1] != len(GT_META_COLUMNS):
        raise ValueError(f"invalid GT metadata shape: {tuple(meta_batch.shape)}")
    return torch.stack(sensor_wins), meta_batch, lengths


def _all_trial_ids_from_index_or_raw(cfg: SATSConfig) -> list[str]:
    index_path = Path(cfg.dataset_index_path)
    if index_path.exists():
        with open(index_path) as f:
            idx = json.load(f)
        return [t["trial_id"] for t in idx["trials"]]

    root = Path(cfg.raw_dir)
    trial_ids: list[str] = []
    for merged in sorted(root.glob("**/*_merged.bin")):
        trial_ids.append(merged.stem.replace("_merged", ""))
    for merged in sorted(root.glob("**/*_merged.csv")):
        trial_id = merged.stem.replace("_merged", "")
        if trial_id not in trial_ids:
            trial_ids.append(trial_id)
    return trial_ids


def build_dataloaders_on_the_fly(
    cfg: SATSConfig,
    all_trial_ids: list[str] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders for the on-the-fly GT path."""

    return_gt_meta = cfg.gt_mode == "gpu_on_the_fly"
    collate_fn = gt_meta_collate_fn if return_gt_meta else window_collate_fn

    if all_trial_ids is None:
        all_trial_ids = _all_trial_ids_from_index_or_raw(cfg)

    if cfg.include_materials or cfg.exclude_diameters:
        before = len(all_trial_ids)
        all_trial_ids = filter_trial_ids(
            all_trial_ids,
            include_materials=cfg.include_materials,
            exclude_diameters=cfg.exclude_diameters,
        )
        log.info(
            "on-the-fly trial filters include_materials=%s exclude_diameters=%s: %d -> %d trials",
            cfg.include_materials, cfg.exclude_diameters, before, len(all_trial_ids),
        )
    log.info("on-the-fly GT trials=%d: %s", len(all_trial_ids), all_trial_ids)

    if cfg.val_ratio > 0:
        sequences_by_trial: dict[str, list[dict]] = {}
        for tid in all_trial_ids:
            seqs = _load_trial_cached_or_merged(tid, cfg)
            if seqs:
                sequences_by_trial[tid] = seqs
        full_ds = SATSOnTheFlyWindowDataset(
            [],
            cfg,
            _preloaded={"sequences_by_trial": sequences_by_trial},
            return_gt_meta=return_gt_meta,
        )
        n_val = max(1, int(len(full_ds) * cfg.val_ratio))
        n_train = len(full_ds) - n_val
        log.info(
            "on-the-fly random split: total_windows=%d train=%d val=%d",
            len(full_ds),
            n_train,
            n_val,
        )
        train_ds, val_ds = random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg.seed),
        )
    else:
        val_set = set(cfg.val_trials)
        train_ids = [t for t in all_trial_ids if t not in val_set]
        val_ids = [t for t in all_trial_ids if t in val_set]
        train_ds = SATSOnTheFlyWindowDataset(train_ids, cfg, return_gt_meta=return_gt_meta)
        val_ds = SATSOnTheFlyWindowDataset(val_ids, cfg, return_gt_meta=return_gt_meta)

    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        loader_kwargs["prefetch_factor"] = max(1, int(cfg.dataloader_prefetch_factor))

    return (
        DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            **loader_kwargs,
        ),
        DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
    )
