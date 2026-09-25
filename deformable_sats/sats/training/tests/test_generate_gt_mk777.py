"""mk555 GT generation contract tests."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sats.preprocessing import generate_gt as gt
from sats.preprocessing.merged_bin import MERGED_DTYPE, write_merged_bin


def test_gt_grid_constants_are_mk555():
    assert gt.GRID_SIZE == 41
    assert gt.GRID_MIN_MM == -10.0
    assert gt.GRID_MAX_MM == 10.0
    assert gt.GRID_STEP_MM == 0.5
    assert gt.EXT_SIZE == 81
    assert gt.EXT_HALF == 40


def test_cli_includes_u_rows_by_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["generate_gt.py"])
    args = gt.parse_args()
    assert args.include_shear_u is True


def test_cli_can_request_u_zero_only(monkeypatch):
    monkeypatch.setattr("sys.argv", ["generate_gt.py", "--u-zero-only"])
    args = gt.parse_args()
    assert args.include_shear_u is False


def test_process_trial_outputs_41_grid_and_includes_u_rows(tmp_path):
    raw_dir = tmp_path / "raw_data"
    trial_dir = raw_dir / "ecomesh" / "d5" / "z_2.5mm" / "test1"
    out_dir = tmp_path / "gt_output_mk555"
    trial_dir.mkdir(parents=True)
    out_dir.mkdir()

    csv_path = trial_dir / "ecomesh_d5_z2.5_test1_merged.csv"
    pd.DataFrame(
        {
            "x_mm": [0.0, 0.0],
            "y_mm": [0.0, 0.0],
            "z_depth_mm": [1.0, 1.5],
            "u_mm": [0.0, 0.5],
            "Fz": [1.0, 10.0],
        }
    ).to_csv(csv_path, index=False)

    meta = gt.process_trial(
        csv_path=csv_path,
        raw_dir=raw_dir,
        out_dir=out_dir,
        grid_x=np.linspace(gt.GRID_MIN_MM, gt.GRID_MAX_MM, gt.GRID_SIZE),
        grid_y=np.linspace(gt.GRID_MIN_MM, gt.GRID_MAX_MM, gt.GRID_SIZE),
        kernel_cache={},
        z_s=2.0,
        patch_step=2.5,
        fz_min_abs=0.0,
        fz_mode="positive_only",
        beta_mode="none",
        beta_coeffs=(1.0, 0.0, 0.0),
        beta_min=0.5,
        beta_max=2.0,
        z_comp_mode="xy_contact",
        z_contact_force_thresh=0.0,
        z0_estimator="first",
        z_k=1.0,
        z_min=1.5,
        z_max=3.5,
        z_cache_step=0.05,
        grid_tol_mm=0.05,
        drop_offgrid=True,
        include_shear_u=True,
        u_zero_tol_mm=1e-9,
    )

    assert meta is not None
    assert meta["gt_shape"] == [2, 41, 41]
    assert meta["n_nonzero_u_rows"] == 1
    assert meta["n_shear_u_rows"] == 0

    targets = np.load(out_dir / "ecomesh_d5_z2.5_test1_targets.npy")
    assert targets.shape == (2, 41, 41)

    with open(out_dir / "ecomesh_d5_z2.5_test1_gt_meta.json") as f:
        saved_meta = json.load(f)
    assert saved_meta["gt_shape"] == [2, 41, 41]


def test_process_trial_accepts_merged_bin_input(tmp_path):
    raw_dir = tmp_path / "raw_data"
    trial_dir = raw_dir / "ecomesh" / "d10" / "z_3.5mm" / "test1"
    out_dir = tmp_path / "gt_output_mk555"
    trial_dir.mkdir(parents=True)
    out_dir.mkdir()

    rows = np.zeros(2, dtype=MERGED_DTYPE)
    rows["x_mm"] = [0.0, 0.0]
    rows["y_mm"] = [0.0, 0.0]
    rows["z_depth_mm"] = [1.0, 1.0]
    rows["u_mm"] = [0.0, 0.5]
    rows["Fz"] = [1.0, 10.0]
    bin_path = trial_dir / "ecomesh_d10_z3.5_test1_merged.bin"
    write_merged_bin(bin_path, rows)

    meta = gt.process_trial(
        csv_path=bin_path,
        raw_dir=raw_dir,
        out_dir=out_dir,
        grid_x=np.linspace(gt.GRID_MIN_MM, gt.GRID_MAX_MM, gt.GRID_SIZE),
        grid_y=np.linspace(gt.GRID_MIN_MM, gt.GRID_MAX_MM, gt.GRID_SIZE),
        kernel_cache={},
        z_s=2.0,
        patch_step=5.0,
        fz_min_abs=0.0,
        fz_mode="positive_only",
        beta_mode="none",
        beta_coeffs=(1.0, 0.0, 0.0),
        beta_min=0.5,
        beta_max=2.0,
        z_comp_mode="xy_contact",
        z_contact_force_thresh=0.0,
        z0_estimator="first",
        z_k=1.0,
        z_min=1.5,
        z_max=3.5,
        z_cache_step=0.05,
        grid_tol_mm=0.05,
        drop_offgrid=True,
        include_shear_u=True,
        u_zero_tol_mm=1e-9,
    )

    assert meta is not None
    assert meta["source_format"] == "bin"
    assert meta["gt_shape"] == [2, 41, 41]
    assert (out_dir / "ecomesh_d10_z3.5_test1_targets.npy").exists()
