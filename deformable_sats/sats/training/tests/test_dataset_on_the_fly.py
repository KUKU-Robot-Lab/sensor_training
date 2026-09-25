import numpy as np
import torch

from sats.preprocessing.merged_bin import MERGED_DTYPE, SENSOR_COLS, write_merged_bin
from sats.training.config import SATSConfig
from sats.training.dataset_on_the_fly import (
    OnTheFlyGTCache,
    SATSOnTheFlyWindowDataset,
    meta_cache_path,
    save_trial_meta_cache,
    spherical_contact_radius_mm,
)
from sats.training.gt_gpu import BatchGPUTargetGenerator, GT_META_COLUMNS


def test_spherical_contact_radius_uses_hertz_depth_and_clamps():
    depths = np.array([0.0, 0.01, 0.5, 1.0, 2.5, 5.0], dtype=np.float64)

    radii = spherical_contact_radius_mm(
        z_depth_mm=depths,
        sphere_radius_mm=2.5,
        min_radius_mm=0.2,
        max_radius_mm=2.5,
    )

    assert np.allclose(radii[0], 0.2)
    assert np.allclose(radii[1], 0.2)
    assert np.allclose(radii[2], np.sqrt(2.5 * 0.5))
    assert np.allclose(radii[3], np.sqrt(2.5 * 1.0))
    assert np.allclose(radii[4], 2.5)
    assert np.allclose(radii[5], 2.5)


def test_on_the_fly_cache_returns_zero_for_noncontact_and_varies_with_depth():
    cache = OnTheFlyGTCache(
        z_s_mm=2.0,
        patch_step_mm=0.5,
        contact_radius_step_mm=0.05,
        min_contact_radius_mm=0.2,
        fz_min_n=0.05,
        z_depth_min_mm=0.02,
    )

    zero = cache.map_for_row(
        diameter_mm=5.0,
        x_mm=0.0,
        y_mm=0.0,
        z_depth_mm=0.0,
        fz_n=10.0,
    )
    shallow = cache.map_for_row(
        diameter_mm=5.0,
        x_mm=0.0,
        y_mm=0.0,
        z_depth_mm=0.5,
        fz_n=10.0,
    )
    deep = cache.map_for_row(
        diameter_mm=5.0,
        x_mm=0.0,
        y_mm=0.0,
        z_depth_mm=2.5,
        fz_n=10.0,
    )

    assert zero.shape == (41, 41)
    assert np.allclose(zero, 0.0)
    assert shallow.sum() > 0.0
    assert deep.sum() > 0.0
    assert not np.allclose(shallow, deep)


def test_on_the_fly_window_dataset_does_not_require_gt_npy(tmp_path):
    raw_dir = tmp_path / "raw_data"
    trial_dir = raw_dir / "ecomesh" / "d5" / "z_2.5mm" / "test1"
    trial_dir.mkdir(parents=True)

    rows = np.zeros(20, dtype=MERGED_DTYPE)
    rows["x_mm"] = 0.0
    rows["y_mm"] = 0.0
    rows["z_depth_mm"] = np.linspace(0.0, 2.5, len(rows), dtype=np.float32)
    rows["u_mm"] = 0.0
    rows["Fz"] = np.linspace(0.0, 10.0, len(rows), dtype=np.float32)
    baseline = {}
    for i, col in enumerate(SENSOR_COLS, start=1):
        value = 1000.0 + i
        rows[col] = value
        baseline[f"Skin{i}_mean"] = value

    trial_id = "ecomesh_d5_z2.5_test1"
    write_merged_bin(
        trial_dir / f"{trial_id}_merged.bin",
        rows,
        metadata={"baseline": baseline},
    )

    cfg = SATSConfig(
        raw_dir=str(raw_dir),
        gt_mode="on_the_fly",
        min_seq_len=3,
        seq_len=100,
        window_size=5,
        batch_size=4,
        on_the_fly_patch_step_mm=0.5,
        on_the_fly_sampling_policy="balanced_contact",
        plateau_stride=4,
        loading_stride=2,
        saturation_stride=1,
    )

    ds = SATSOnTheFlyWindowDataset([trial_id], cfg)

    assert len(ds) > 0
    sensor_window, gt_map = ds[0]
    assert sensor_window.shape == (5, 16)
    assert gt_map.shape == (41, 41)
    assert np.isfinite(gt_map.numpy()).all()


def test_on_the_fly_window_dataset_can_return_compact_gt_metadata(tmp_path):
    raw_dir = tmp_path / "raw_data"
    trial_dir = raw_dir / "ecomesh" / "d5" / "z_2.5mm" / "test1"
    trial_dir.mkdir(parents=True)

    rows = np.zeros(20, dtype=MERGED_DTYPE)
    rows["x_mm"] = 0.0
    rows["y_mm"] = 0.0
    rows["z_depth_mm"] = np.linspace(0.0, 2.5, len(rows), dtype=np.float32)
    rows["u_mm"] = 0.0
    rows["Fz"] = np.linspace(0.0, 10.0, len(rows), dtype=np.float32)
    baseline = {}
    for i, col in enumerate(SENSOR_COLS, start=1):
        value = 1000.0 + i
        rows[col] = value
        baseline[f"Skin{i}_mean"] = value

    trial_id = "ecomesh_d5_z2.5_test1"
    write_merged_bin(
        trial_dir / f"{trial_id}_merged.bin",
        rows,
        metadata={"baseline": baseline},
    )

    cfg = SATSConfig(
        raw_dir=str(raw_dir),
        gt_mode="gpu_on_the_fly",
        min_seq_len=3,
        seq_len=100,
        window_size=5,
        batch_size=4,
        on_the_fly_patch_step_mm=0.5,
        on_the_fly_sampling_policy="balanced_contact",
    )

    ds = SATSOnTheFlyWindowDataset([trial_id], cfg, return_gt_meta=True)

    sensor_window, gt_meta = ds[0]
    assert sensor_window.shape == (5, 16)
    assert gt_meta.shape == (len(GT_META_COLUMNS),)
    assert gt_meta[0].item() == 5.0
    assert np.isfinite(gt_meta.numpy()).all()


def test_batch_gpu_target_generator_matches_cpu_on_the_fly_cache():
    cfg = SATSConfig(
        gt_mode="gpu_on_the_fly",
        grid_size=41,
        grid_step_mm=0.5,
        on_the_fly_patch_step_mm=0.5,
        contact_radius_step_mm=0.05,
        min_contact_radius_mm=0.2,
        fz_min_abs_n=0.05,
        z_depth_min_mm=0.02,
    )
    cpu_cache = OnTheFlyGTCache(
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
    generator = BatchGPUTargetGenerator(cfg, device="cpu")
    meta = np.array(
        [
            [5.0, 0.0, 0.0, 0.5, 10.0],
            [5.0, 0.5, -0.5, 2.5, 7.0],
            [5.0, 0.0, 0.0, 0.0, 10.0],
        ],
        dtype=np.float32,
    )

    batch = generator(torch.from_numpy(meta)).numpy()
    expected0 = cpu_cache.map_for_row(diameter_mm=5.0, x_mm=0.0, y_mm=0.0, z_depth_mm=0.5, fz_n=10.0)
    expected1 = cpu_cache.map_for_row(diameter_mm=5.0, x_mm=0.5, y_mm=-0.5, z_depth_mm=2.5, fz_n=7.0)

    assert np.allclose(batch[0], expected0, rtol=1e-5, atol=1e-5)
    assert np.allclose(batch[1], expected1, rtol=1e-5, atol=1e-5)
    assert np.allclose(batch[2], 0.0)


def test_compact_meta_cache_can_be_built_and_loaded(tmp_path):
    raw_dir = tmp_path / "raw_data"
    cache_dir = tmp_path / "gt_meta_cache"
    trial_dir = raw_dir / "ecomesh" / "d5" / "z_2.5mm" / "test1"
    trial_dir.mkdir(parents=True)

    rows = np.zeros(20, dtype=MERGED_DTYPE)
    rows["x_mm"] = 0.0
    rows["y_mm"] = 0.0
    rows["z_depth_mm"] = np.linspace(0.0, 2.5, len(rows), dtype=np.float32)
    rows["u_mm"] = 0.0
    rows["Fz"] = np.linspace(0.0, 10.0, len(rows), dtype=np.float32)
    baseline = {}
    for i, col in enumerate(SENSOR_COLS, start=1):
        value = 1000.0 + i
        rows[col] = value
        baseline[f"Skin{i}_mean"] = value

    trial_id = "ecomesh_d5_z2.5_test1"
    write_merged_bin(
        trial_dir / f"{trial_id}_merged.bin",
        rows,
        metadata={"baseline": baseline},
    )

    cfg = SATSConfig(
        raw_dir=str(raw_dir),
        gt_mode="gpu_on_the_fly",
        gt_meta_cache_dir=str(cache_dir),
        use_gt_meta_cache=True,
        min_seq_len=3,
        seq_len=100,
        window_size=5,
        batch_size=4,
        on_the_fly_patch_step_mm=0.5,
        on_the_fly_sampling_policy="balanced_contact",
    )

    path = save_trial_meta_cache(trial_id, cfg, overwrite=True)
    assert path == meta_cache_path(cfg, trial_id)
    assert path is not None and path.exists()

    ds = SATSOnTheFlyWindowDataset([trial_id], cfg, return_gt_meta=True)
    sensor_window, gt_meta = ds[0]

    assert sensor_window.shape == (5, 16)
    assert gt_meta.shape == (len(GT_META_COLUMNS),)
