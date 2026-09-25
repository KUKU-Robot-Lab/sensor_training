import json

from sats.tools.compare_sats_runs import SUMMARY_COLUMNS, summarize_runs, write_summary_csv


def _write_run(run_dir, *, material, history):
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "include_materials": [material],
                "train_trials": [
                    f"{material}_d5_z2.5_test1",
                    f"{material}_d5_z2.5_test2",
                ],
                "val_trials": [f"{material}_d5_z2.5_test3"],
                "grid_step_mm": 0.5,
                "gt_mode": "gpu_on_the_fly",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")


def test_compare_sats_runs_writes_deterministic_summary_csv(tmp_path):
    run_dir = tmp_path / "xy1_d5_eco20_e2e_g05"
    _write_run(
        run_dir,
        material="eco20_xy1",
        history=[
            {"epoch": 1, "train_loss": 0.55, "val_rmse": 0.41},
            {"epoch": 2, "train_loss": 0.44, "val_rmse": 0.31},
        ],
    )

    rows = summarize_runs([run_dir])
    out_path = tmp_path / "summary.csv"
    write_summary_csv(rows, out_path)

    assert rows == [
        {
            "material": "eco20_xy1",
            "train_trials": "eco20_xy1_d5_z2.5_test1;eco20_xy1_d5_z2.5_test2",
            "val_trials": "eco20_xy1_d5_z2.5_test3",
            "best_epoch": "2",
            "best_val_rmse": "0.31",
            "final_train_loss": "0.44",
            "grid_step_mm": "0.5",
            "gt_mode": "gpu_on_the_fly",
            "run_dir": str(run_dir),
        }
    ]
    assert out_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(SUMMARY_COLUMNS)
