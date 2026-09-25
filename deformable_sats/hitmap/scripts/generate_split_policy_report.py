from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.utils.split_policy import (
    build_stratified_cv_folds,
    load_loading_trial_counts,
    render_split_policy_report,
    suggest_holdout_trials,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the split policy report for reconstruction checklist 7.2.")
    parser.add_argument("--zarr-path", type=Path, default=Path("preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr"))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("md/260415_split_policy_report.md"))
    args = parser.parse_args()

    trial_counts = load_loading_trial_counts(args.zarr_path)
    trial_ids = sorted(trial_counts)
    held_out_trials = suggest_holdout_trials(trial_ids)
    folds = build_stratified_cv_folds(trial_ids, args.cv_folds, held_out_trials)
    args.output.write_text(render_split_policy_report(trial_counts, held_out_trials, folds), encoding="utf-8")


if __name__ == "__main__":
    main()
