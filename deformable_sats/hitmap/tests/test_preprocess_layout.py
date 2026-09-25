import tempfile
import unittest
from pathlib import Path

from preprocessing.preprocess import (
    _normalize_workers,
    discover_merged_csvs,
    parse_args,
    parse_trial_csv_info,
    parse_trial_name,
)


class PreprocessLayoutTest(unittest.TestCase):
    def test_parse_nested_raw_merge_trial_id(self) -> None:
        info = parse_trial_name("ecomesh_d10_z1.5_test4_merged")

        self.assertEqual(info["material"], "ecomesh")
        self.assertEqual(info["diameter_mm"], 10.0)
        self.assertEqual(info["z_max_indentation_mm"], 1.5)
        self.assertEqual(info["trial_no"], 4)

    def test_discover_merged_csvs_finds_nested_test_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw_data"
            merged = raw_dir / "ecomesh" / "d10" / "z_1.5mm" / "test4" / "ecomesh_d10_z1.5_test4_merged.csv"
            merged.parent.mkdir(parents=True)
            merged.write_text("timestep_sec\n", encoding="utf-8")

            self.assertEqual(discover_merged_csvs(raw_dir, "**/*_merged.csv"), [merged])

    def test_parse_trial_csv_info_falls_back_to_nested_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw_data"
            merged = raw_dir / "ecomesh" / "d10" / "z_1.5mm" / "test4" / "test4_merged.csv"
            merged.parent.mkdir(parents=True)
            merged.write_text("timestep_sec\n", encoding="utf-8")

            trial_id, info = parse_trial_csv_info(merged, raw_dir)

            self.assertEqual(trial_id, "ecomesh_d10_z1.5_test4")
            self.assertEqual(info["material"], "ecomesh")
            self.assertEqual(info["diameter_mm"], 10.0)
            self.assertEqual(info["z_max_indentation_mm"], 1.5)
            self.assertEqual(info["trial_no"], 4)

    def test_normalize_workers_clamps_to_task_count(self) -> None:
        self.assertEqual(_normalize_workers(worker_count=8, num_tasks=3), 3)
        self.assertEqual(_normalize_workers(worker_count=0, num_tasks=3), 1)
        self.assertEqual(_normalize_workers(worker_count=8, num_tasks=0), 1)

    def test_parse_args_accepts_workers(self) -> None:
        args = parse_args(["--workers", "4"])

        self.assertEqual(args.workers, 4)


if __name__ == "__main__":
    unittest.main()
