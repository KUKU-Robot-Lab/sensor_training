import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.pipelines.evaluate_comparison_heatmap import _resolve_checkpoint, _resolve_eval_indices


class FakeDataset:
    def __init__(self, sample_trial_ids: list[str]) -> None:
        self.sample_trial_ids = sample_trial_ids

    def __len__(self) -> int:
        return len(self.sample_trial_ids)


class HeatmapEvaluationTest(unittest.TestCase):
    def test_resolve_checkpoint_requires_explicit_tag_when_multiple_candidates_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fold_dir = Path(tmp) / "folds" / "fold_0"
            fold_dir.mkdir(parents=True)
            (fold_dir / "best_multi_head_field_stage1_point_xybce1_zoff_fzoff.pth").write_text("", encoding="utf-8")
            (fold_dir / "best_multi_head_field_stage3_dlabel-gaussian-hertz_xybce1_zhuber0p2_fzhuber0p2_decsoftargmax.pth").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "multiple checkpoints"):
                _resolve_checkpoint(Path(tmp), 0, "multi_head_field")

    def test_resolve_checkpoint_selects_exact_requested_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fold_dir = Path(tmp) / "folds" / "fold_0"
            fold_dir.mkdir(parents=True)
            stage1 = fold_dir / "best_multi_head_field_stage1_point_xybce1_zoff_fzoff.pth"
            stage3 = fold_dir / "best_multi_head_field_stage3_dlabel-gaussian-hertz_xybce1_zhuber0p2_fzhuber0p2_decsoftargmax.pth"
            stage1.write_text("", encoding="utf-8")
            stage3.write_text("", encoding="utf-8")

            resolved = _resolve_checkpoint(
                Path(tmp),
                0,
                "multi_head_field",
                checkpoint_tag="_stage3_dlabel-gaussian-hertz_xybce1_zhuber0p2_fzhuber0p2_decsoftargmax",
            )

            self.assertEqual(resolved, stage3)

    def test_resolve_eval_indices_uses_fold_validation_trials(self) -> None:
        ds = FakeDataset(["trial_a", "trial_a", "trial_b", "trial_c", "trial_b"])
        fold = {"val_trials": ["trial_b"]}

        indices = _resolve_eval_indices(ds, fold, eval_split="val")

        np.testing.assert_array_equal(indices, np.array([2, 4], dtype=np.int64))

    def test_resolve_eval_indices_all_returns_full_dataset(self) -> None:
        ds = FakeDataset(["trial_a", "trial_b", "trial_c"])
        fold = {"val_trials": ["trial_b"]}

        indices = _resolve_eval_indices(ds, fold, eval_split="all")

        np.testing.assert_array_equal(indices, np.array([0, 1, 2], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
