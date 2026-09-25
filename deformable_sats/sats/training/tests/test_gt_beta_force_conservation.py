"""Tests for β_geom force-conservation rectification (B) in GT generation.

The base kernel approximates σ_zz / F, so its integral over the plane must
equal 1 (Boussinesq equilibrium: ∫∫ σ_zz dA = F at any depth). Coarse patch
discretization for sub-patch-step contact radii breaks this. The force-
conservation flag rescales the kernel so the numeric integral returns to 1.
"""
import math

from sats.training.config import SATSConfig
from sats.training.gt_gpu import BatchGPUTargetGenerator


def _kernel_integral(gen: BatchGPUTargetGenerator, radius: float) -> float:
    base = gen._compute_base_kernel(float(radius))
    return float(base.sum().item()) * (gen.grid_step_mm ** 2)


def test_small_radius_kernel_over_conserves_without_flag():
    """A sub-patch-step radius integrates to noticeably more than 1 (the defect)."""
    cfg = SATSConfig(
        gt_mode="gpu_on_the_fly", grid_size=41, grid_step_mm=0.5,
        on_the_fly_patch_step_mm=0.1, contact_radius_step_mm=0.05,
        min_contact_radius_mm=0.05,
    )
    gen = BatchGPUTargetGenerator(cfg, device="cpu")
    # radius 0.05 with patch_step 0.1 assigns one point of area 0.01 for a disk
    # of area π·0.05² ≈ 0.00785 → integral ≈ 1.27.
    assert _kernel_integral(gen, 0.05) > 1.2


def test_force_conservation_flag_normalizes_integral_to_one():
    """With the flag on, the kernel integral is 1.0 across radii (small and large)."""
    cfg = SATSConfig(
        gt_mode="gpu_on_the_fly", grid_size=41, grid_step_mm=0.5,
        on_the_fly_patch_step_mm=0.1, contact_radius_step_mm=0.05,
        min_contact_radius_mm=0.05, gt_beta_force_conservation=True,
    )
    gen = BatchGPUTargetGenerator(cfg, device="cpu")
    for radius in (0.05, 0.5, 1.0, 2.0):
        assert math.isclose(_kernel_integral(gen, radius), 1.0, rel_tol=1e-3)


def test_flag_correction_is_small_for_physical_d10_lowforce_radii():
    """For d10 low-force radii (~0.7-1.6mm) the β_geom correction is only a few %
    (staircase-aliasing smoothing), far too small to explain the 3-4x d10 low-force
    over-prediction — a documented negative: force conservation is NOT the cause."""
    cfg_on = SATSConfig(
        gt_mode="gpu_on_the_fly", grid_size=41, grid_step_mm=0.5,
        on_the_fly_patch_step_mm=0.1, contact_radius_step_mm=0.05,
        min_contact_radius_mm=0.05, gt_beta_force_conservation=True,
    )
    cfg_off = SATSConfig(**{**cfg_on.__dict__, "gt_beta_force_conservation": False})
    gen_on = BatchGPUTargetGenerator(cfg_on, device="cpu")
    gen_off = BatchGPUTargetGenerator(cfg_off, device="cpu")
    for radius in (0.7, 1.0, 1.6):
        on = gen_on._compute_base_kernel(radius).max().item()
        off = gen_off._compute_base_kernel(radius).max().item()
        assert abs(on / off - 1.0) < 0.08
