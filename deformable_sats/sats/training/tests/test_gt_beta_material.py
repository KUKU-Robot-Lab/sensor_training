"""On-the-fly β(p) material rectification must match the precomputed path.

The precomputed generate_gt.compute_beta already implements β(p)=c0+c1·p+c2·p²
(paper S9). This ports it to the GPU on-the-fly generator; the two must agree so
GT is identical regardless of path.
"""
import math

import numpy as np
import torch

from sats.preprocessing.generate_gt import compute_beta
from sats.training.config import SATSConfig
from sats.training.gt_gpu import BatchGPUTargetGenerator

_KW = dict(
    gt_mode="gpu_on_the_fly", grid_size=41, grid_step_mm=0.5,
    on_the_fly_patch_step_mm=0.5, contact_radius_step_mm=0.05,
    min_contact_radius_mm=0.2, fz_min_abs_n=0.05, z_depth_min_mm=0.02,
)


def test_beta_none_is_noop():
    cfg = SATSConfig(**_KW, gt_beta_mode="none")
    gen = BatchGPUTargetGenerator(cfg, device="cpu")
    meta = torch.tensor([[5.0, 0.0, 0.0, 0.5, 8.0]], dtype=torch.float32)
    cfg_b = SATSConfig(**_KW, gt_beta_mode="poly2", gt_beta_c0=1.0, gt_beta_c1=0.0, gt_beta_c2=0.0)
    gen_b = BatchGPUTargetGenerator(cfg_b, device="cpu")
    assert torch.allclose(gen(meta), gen_b(meta), atol=1e-6)  # c1=c2=0 → β≡1


def test_onthefly_beta_matches_precomputed_compute_beta():
    c0, c1, c2 = 1.0, 0.0045, 1.5e-4
    cfg = SATSConfig(**_KW, gt_beta_mode="poly2",
                     gt_beta_c0=c0, gt_beta_c1=c1, gt_beta_c2=c2, gt_beta_max=5.0)
    gen = BatchGPUTargetGenerator(cfg, device="cpu")
    diameter, x, y, z_depth, fz = 5.0, 0.0, 0.0, 0.5, 8.0
    radius = float(gen._radius_key(torch.tensor([diameter]), torch.tensor([z_depth]))[0])

    meta = torch.tensor([[diameter, x, y, z_depth, fz]], dtype=torch.float32)
    off = SATSConfig(**_KW, gt_beta_mode="none")
    gen_off = BatchGPUTargetGenerator(off, device="cpu")
    ratio = float(gen(meta).max() / gen_off(meta).max())

    p_kpa = np.array([fz / (math.pi * radius * radius) * 1000.0])
    beta_ref = float(compute_beta(p_kpa, "poly2", (c0, c1, c2), 0.2, 5.0)[0])
    assert abs(ratio - beta_ref) < 1e-4
