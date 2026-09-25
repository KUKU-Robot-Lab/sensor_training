import tempfile
import unittest
from pathlib import Path

from preprocessing.raw_merge import (
    DUE_PATTERNS,
    discover_trial_dirs,
    find_single_file,
    parse_trial_dir_info,
)


class RawMergeLayoutTest(unittest.TestCase):
    def test_parse_nested_trial_metadata_includes_material_indenter_z_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw_data"
            trial_dir = raw_root / "ecomesh" / "d10" / "z_1.5mm" / "test4"
            trial_dir.mkdir(parents=True)

            info = parse_trial_dir_info(trial_dir, raw_root)

            self.assertEqual(info["trial_id"], "ecomesh_d10_z1.5_test4")
            self.assertEqual(info["material"], "ecomesh")
            self.assertEqual(info["indenter_diameter_mm"], 10.0)
            self.assertEqual(info["z_max_indentation_mm"], 1.5)
            self.assertEqual(info["experiment_no"], 4)

    def test_find_single_file_recurses_into_stream_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = Path(tmp) / "raw_data" / "ecomesh" / "d5" / "z_1.0mm" / "test2"
            due_path = trial_dir / "due_data" / "due_data_ecomesh_delay1.0_zig_2.csv"
            due_path.parent.mkdir(parents=True)
            due_path.write_text("Timestamp,Skin1\n", encoding="utf-8")

            self.assertEqual(find_single_file(trial_dir, DUE_PATTERNS), due_path)

    def test_discover_trial_dirs_finds_nested_test_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw_data"
            expected = [
                raw_root / "ecomesh" / "d10" / "z_1.0mm" / "test1",
                raw_root / "ecomesh" / "d10" / "z_1.5mm" / "test2",
            ]
            for trial_dir in expected:
                (trial_dir / "afd50_data").mkdir(parents=True)

            self.assertEqual(discover_trial_dirs(raw_root), expected)


if __name__ == "__main__":
    unittest.main()
