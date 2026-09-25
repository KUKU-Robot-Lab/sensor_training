import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from training.models.z_fz_sequence_regressor import ZFzSequenceRegressor
from training.pipelines.train_z_fz_regressor import (
    ScalarNormalizer,
    build_condition_features,
    metric_dict,
    _limit_samples_for_smoke,
    _resolve_fold_xy_checkpoint,
)


class ZFzRegressorTest(unittest.TestCase):
    def test_scalar_normalizer_round_trips_targets(self) -> None:
        values = torch.tensor([[1.0, 2.0], [3.0, 8.0], [5.0, 14.0]], dtype=torch.float32)

        normalizer = ScalarNormalizer.fit(values)
        restored = normalizer.denormalize(normalizer.normalize(values))

        torch.testing.assert_close(restored, values)
        self.assertGreater(normalizer.std[0].item(), 0.0)
        self.assertGreater(normalizer.std[1].item(), 0.0)

    def test_condition_features_use_xy_and_radius_scales(self) -> None:
        xy = torch.tensor([[5.0, -5.0]], dtype=torch.float32)
        radius = torch.tensor([[2.5]], dtype=torch.float32)

        cond = build_condition_features(xy, radius)

        torch.testing.assert_close(cond, torch.tensor([[0.5, -0.5, 0.5]], dtype=torch.float32))

    def test_z_fz_regressor_forward_shape(self) -> None:
        model = ZFzSequenceRegressor(seq_len=3)
        grid = torch.zeros((2, 3, 1, 4, 4), dtype=torch.float32)
        cond = torch.zeros((2, 3), dtype=torch.float32)

        pred = model(grid, cond)

        self.assertEqual(tuple(pred.shape), (2, 2))

    def test_metric_dict_reports_z_and_fz_mae_rmse(self) -> None:
        preds = torch.tensor([[1.0, 10.0], [2.0, 12.0]], dtype=torch.float32)
        targets = torch.tensor([[1.5, 9.0], [1.0, 14.0]], dtype=torch.float32)

        metrics = metric_dict(preds, targets)

        self.assertEqual(metrics["output_names"], ["z", "fz"])
        self.assertAlmostEqual(metrics["per_output"]["z"]["mae"], 0.75)
        self.assertAlmostEqual(metrics["per_output"]["fz"]["rmse"], (2.5 ** 0.5))

    def test_smoke_limit_preserves_multiple_trials(self) -> None:
        class FakeDataset:
            samples = [["a0"], ["a1"], ["a2"], ["b0"], ["b1"], ["b2"]]
            sample_trial_ids = ["a", "a", "a", "b", "b", "b"]

            def __len__(self) -> int:
                return len(self.samples)

        limited = _limit_samples_for_smoke(FakeDataset(), 4)

        self.assertEqual(len(limited.samples), 4)
        self.assertEqual(set(limited.sample_trial_ids), {"a", "b"})

    def test_resolve_fold_xy_checkpoint_rejects_reusing_single_base_checkpoint_for_other_fold(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            requested = base / "best_multi_head_field_stage2.pth"
            requested.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "Refusing to reuse one checkpoint across folds"):
                _resolve_fold_xy_checkpoint(str(requested), fold_index=1)

    def test_resolve_fold_xy_checkpoint_swaps_fold_segment(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            fold0 = base / "folds" / "fold_0"
            fold1 = base / "folds" / "fold_1"
            fold0.mkdir(parents=True)
            fold1.mkdir(parents=True)
            requested = fold0 / "best_multi_head_field_stage2.pth"
            resolved = fold1 / "best_multi_head_field_stage2.pth"
            requested.write_text("", encoding="utf-8")
            resolved.write_text("", encoding="utf-8")

            self.assertEqual(_resolve_fold_xy_checkpoint(str(requested), fold_index=1), resolved)


if __name__ == "__main__":
    unittest.main()
