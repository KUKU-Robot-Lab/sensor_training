import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from training.pipelines.train_comparison import (
    _build_multi_head_target_map,
    _build_soft_heatmap,
    _effective_batch_size,
    _combine_multi_head_loss,
    _multi_head_metric_tensors,
    _multi_head_run_tag,
    _save_overlay,
    _write_fz_summary_csv,
    build_metric_report,
    calculate_metrics,
)


class MultiHeadFzMetricsTest(unittest.TestCase):
    def test_calculate_metrics_records_4d_schema_and_fz_values(self) -> None:
        preds = np.array(
            [
                [1.0, 2.0, 0.5, 10.0],
                [3.0, 4.0, 1.5, 14.0],
            ],
            dtype=np.float32,
        )
        targets = np.array(
            [
                [1.0, 1.0, 0.0, 8.0],
                [5.0, 4.0, 1.0, 11.0],
            ],
            dtype=np.float32,
        )

        metrics = calculate_metrics(preds, targets)

        self.assertEqual(metrics["output_names"], ["x", "y", "z", "fz"])
        self.assertAlmostEqual(metrics["mae"][3], 2.5)
        self.assertAlmostEqual(metrics["per_output"]["fz"]["mae"], 2.5)
        self.assertIn("fz", metrics["metric_schema"]["outputs"])

    def test_multi_head_metric_tensors_keep_fz_prediction_and_target(self) -> None:
        args = SimpleNamespace(
            apply_linear_calib=False,
            decode_xy="none",
            heatmap_size=2,
        )
        scalar_pred = torch.tensor([[1.25, 7.5], [1.75, 8.5]], dtype=torch.float32)
        fmap = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
        targets = torch.tensor(
            [
                [-1.0, -2.0, 1.2, 7.0],
                [1.0, 2.0, 1.8, 9.0],
            ],
            dtype=torch.float32,
        )

        metric_preds, metric_targets = _multi_head_metric_tensors(scalar_pred, fmap, targets, args)

        self.assertEqual(tuple(metric_preds.shape), (2, 4))
        self.assertEqual(tuple(metric_targets.shape), (2, 4))
        torch.testing.assert_close(metric_preds[:, 2], scalar_pred[:, 0])
        torch.testing.assert_close(metric_preds[:, 3], scalar_pred[:, 1])
        torch.testing.assert_close(metric_targets[:, 2], targets[:, 2])
        torch.testing.assert_close(metric_targets[:, 3], targets[:, 3])

    def test_save_overlay_writes_annotated_png_from_fake_batch(self) -> None:
        fmap_logits = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
        fmap_logits[0, 0, 1, 2] = 8.0
        fmap_logits[1, 0, 2, 1] = 8.0
        target_map = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
        target_map[0, 0, 1, 1] = 1.0
        target_map[1, 0, 2, 2] = 1.0
        pred_values = torch.tensor(
            [
                [-8.75, -9.25, 0.8, 11.0],
                [-9.25, -8.75, 1.4, 14.0],
            ],
            dtype=torch.float32,
        )
        target_values = torch.tensor(
            [
                [-9.25, -9.25, 0.5, 10.0],
                [-8.75, -8.75, 1.0, 12.0],
            ],
            dtype=torch.float32,
        )

        with tempfile.TemporaryDirectory() as tmp:
            _save_overlay(
                0,
                fmap_logits,
                target_map,
                tmp,
                prefix="smoke",
                max_samples=1,
                pred_values=pred_values,
                target_values=target_values,
            )

            png_path = Path(tmp) / "smoke_overlay_b0_i0.png"
            self.assertTrue(png_path.exists())
            self.assertGreater(png_path.stat().st_size, 0)

    def test_fz_summary_csv_writes_z_fz_prediction_and_target_columns(self) -> None:
        preds = np.array([[1.0, 2.0, 0.5, 10.0]], dtype=np.float32)
        targets = np.array([[1.5, 2.5, 0.75, 12.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "fz_summary.csv"

            _write_fz_summary_csv(csv_path, preds, targets)

            text = csv_path.read_text(encoding="utf-8")
            self.assertIn("target_z", text)
            self.assertIn("pred_fz", text)
            self.assertIn("target_fz", text)
            self.assertIn("-2.0", text)

    def test_multi_head_effective_batch_size_is_capped_for_lstm_memory(self) -> None:
        self.assertEqual(_effective_batch_size("multi_head_field", 16384), 1024)
        self.assertEqual(_effective_batch_size("multi_head_field", 128), 128)

    def test_depth_aware_flag_off_uses_point_target_map(self) -> None:
        args = SimpleNamespace(
            use_depth_aware_label=False,
            heatmap_size=4,
            depth_radius_model="hertz",
            depth_label_kernel="gaussian",
            normalize_heatmap=False,
            indenter_radius_mm=2.5,
            depth_fallback_mm=1.0,
            heatmap_sigma_scale=1.0,
        )
        targets = torch.tensor([[-9.75, -9.75, 1.0, 3.0]], dtype=torch.float32)

        target_map = _build_multi_head_target_map(targets, args)

        self.assertEqual(tuple(target_map.shape), (1, 1, 4, 4))
        self.assertEqual(torch.count_nonzero(target_map).item(), 1)
        self.assertEqual(target_map[0, 0, 0, 0].item(), 1.0)

    def test_depth_aware_flag_on_uses_soft_target_map(self) -> None:
        args = SimpleNamespace(
            use_depth_aware_label=True,
            heatmap_size=4,
            depth_radius_model="hertz",
            depth_label_kernel="gaussian",
            normalize_heatmap=False,
            indenter_radius_mm=2.5,
            depth_fallback_mm=1.0,
            heatmap_sigma_scale=1.0,
        )
        targets = torch.tensor([[-9.75, -9.75, 1.0, 3.0]], dtype=torch.float32)

        target_map = _build_multi_head_target_map(targets, args)

        self.assertGreater(torch.count_nonzero(target_map).item(), 1)
        self.assertGreater(target_map[0, 0, 0, 0].item(), target_map[0, 0, 0, 3].item())

    def test_soft_heatmap_accepts_per_sample_indenter_radius(self) -> None:
        x = torch.tensor([-9.75, -9.75], dtype=torch.float32)
        y = torch.tensor([-9.75, -9.75], dtype=torch.float32)
        depth = torch.tensor([1.0, 1.0], dtype=torch.float32)
        radius = torch.tensor([2.5, 5.0], dtype=torch.float32)

        target_map = _build_soft_heatmap(
            x,
            y,
            depth,
            heatmap_size=4,
            radius_model="hertz",
            kernel="gaussian",
            normalize=False,
            indenter_radius_mm=radius,
            fallback_depth_mm=1.0,
            sigma_scale=1.0,
        )

        self.assertEqual(tuple(target_map.shape), (2, 1, 4, 4))
        self.assertGreater(target_map[1, 0, 0, 3].item(), target_map[0, 0, 0, 3].item())

    def test_zero_scalar_lambdas_do_not_contribute_to_total_loss(self) -> None:
        args = SimpleNamespace(lambda_xy=1.0, lambda_z=0.0, lambda_fz=0.0)
        l_xy = torch.tensor(2.0)
        l_z = torch.tensor(100.0)
        l_fz = torch.tensor(200.0)

        loss = _combine_multi_head_loss(args, l_xy, l_z, l_fz)

        self.assertEqual(loss.item(), 2.0)

    def test_multi_head_checkpoint_tag_reflects_stage_and_loss_configuration(self) -> None:
        stage2_args = SimpleNamespace(
            use_depth_aware_label=True,
            depth_label_kernel="gaussian",
            depth_radius_model="hertz",
            loss_xy="bce",
            loss_z="huber",
            loss_fz="huber",
            lambda_xy=1.0,
            lambda_z=0.0,
            lambda_fz=0.0,
            decode_xy="softargmax",
            normalize_heatmap=True,
        )
        stage3_args = SimpleNamespace(
            **{
                **stage2_args.__dict__,
                "lambda_z": 0.2,
                "lambda_fz": 0.2,
            }
        )

        self.assertIn("stage2", _multi_head_run_tag("multi_head_field", stage2_args))
        self.assertIn("zoff_fzoff", _multi_head_run_tag("multi_head_field", stage2_args))
        self.assertIn("stage3", _multi_head_run_tag("multi_head_field", stage3_args))
        self.assertIn("zhuber0p2_fzhuber0p2", _multi_head_run_tag("multi_head_field", stage3_args))

    def test_build_metric_report_separates_raw_and_calibrated_metrics(self) -> None:
        args = SimpleNamespace(
            depth_bin_edges=[0.0, float("inf")],
            apply_linear_calib=True,
            calib_x_ax=2.0,
            calib_x_by=0.0,
            calib_x_bias=0.0,
            calib_y_ax=0.0,
            calib_y_by=1.0,
            calib_y_bias=0.0,
        )
        raw_preds = np.array([[1.0, 2.0, 0.5], [2.0, 1.0, 1.5]], dtype=np.float32)
        targets = np.array([[2.0, 2.0, 0.5], [4.0, 1.0, 1.5]], dtype=np.float32)

        report = build_metric_report(raw_preds, targets, args)

        self.assertEqual(report["metric_selection"]["primary"], "raw")
        self.assertAlmostEqual(report["metric_variants"]["raw"]["mae"][0], 1.5)
        self.assertAlmostEqual(report["metric_variants"]["calibrated"]["mae"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
