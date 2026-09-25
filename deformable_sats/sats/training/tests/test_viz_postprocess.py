from __future__ import annotations

import numpy as np

from sats.inference.viz_postprocess import (
    apply_viz_threshold_nmm2,
    compute_gt_global_max_nmm2,
    to_nmm2,
)


def test_to_nmm2_converts_model_scale() -> None:
    src = np.array([[0.0, 10.0], [15.0, 153.6]], dtype=np.float32)
    out = to_nmm2(src)
    expected = np.array([[0.0, 0.1], [0.15, 1.536]], dtype=np.float32)
    assert np.allclose(out, expected, atol=1e-6)


def test_apply_viz_threshold_nmm2_clips_below_threshold() -> None:
    src = np.array([[0.099, 0.1], [0.101, 0.0]], dtype=np.float32)
    out = apply_viz_threshold_nmm2(src, threshold_nmm2=0.1)
    expected = np.array([[0.0, 0.1], [0.101, 0.0]], dtype=np.float32)
    assert np.allclose(out, expected, atol=1e-6)


def test_compute_gt_global_max_nmm2_scans_all_target_files(tmp_path) -> None:
    arr1 = np.zeros((2, 40, 40), dtype=np.float32)
    arr2 = np.zeros((2, 40, 40), dtype=np.float32)
    arr1[0, 0, 0] = 0.12
    arr2[1, 3, 4] = 0.1536

    np.save(tmp_path / "trial_a_targets.npy", arr1)
    np.save(tmp_path / "trial_b_targets.npy", arr2)
    np.save(tmp_path / "ignore.npy", arr2)

    compute_gt_global_max_nmm2.cache_clear()
    vmax = compute_gt_global_max_nmm2(str(tmp_path))
    assert abs(vmax - 0.1536) < 1e-8
