import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training.utils.reconstruction_baseline import (
    collect_baseline_rows,
    render_markdown_table,
    render_reclassification_notes,
)


class ReconstructionBaselineTest(unittest.TestCase):
    def test_collect_baseline_rows_reads_current_and_legacy_sources(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]

        rows = collect_baseline_rows(repo_root)

        run_names = {row.run_name for row in rows}
        self.assertIn("multi_head_field_stage3_cv5", run_names)
        self.assertIn("z_fz_regressor_gt_xy", run_names)
        self.assertIn("z_fz_regressor_predicted_xy", run_names)
        self.assertIn("min8_sats", run_names)
        self.assertIn("min10_sats_xy", run_names)
        self.assertIn("legacy_sr_ff_report_reference", run_names)

    def test_markdown_table_separates_upper_bound_rows(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]

        rows = collect_baseline_rows(repo_root)
        markdown = render_markdown_table(rows)

        self.assertIn("separated_upper_bound", markdown)
        self.assertIn("pred_xy+gt_radius", markdown)
        self.assertIn("legacy_direct_regression", markdown)

    def test_reclassification_notes_capture_7_1_policy(self) -> None:
        notes = render_reclassification_notes()

        self.assertIn("eval-split all", notes)
        self.assertIn("predicted_xy + GT radius", notes)
        self.assertIn("reference-only", notes)


if __name__ == "__main__":
    unittest.main()
