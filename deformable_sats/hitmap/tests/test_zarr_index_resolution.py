import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import zarr
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("zarr is not installed") from exc

from preprocessing.preprocess import export_to_zarr
from training.data.dataset_zarr import ZarrDataset, _compact_index_path
from training.pipelines.train_comparison import _resolve_zarr_path


def _feature_frame(trial_id: str, n_rows: int = 2) -> pd.DataFrame:
    data = {f"s_norm_{i}": np.full(n_rows, i / 100.0, dtype=np.float32) for i in range(1, 17)}
    data.update(
        {
            "z_depth_mm": np.linspace(0.1, 0.2, n_rows, dtype=np.float32),
            "diameter_mm": np.full(n_rows, 5.0, dtype=np.float32),
            "fz_bc": np.linspace(0.3, 0.4, n_rows, dtype=np.float32),
            "x_mm": np.linspace(-1.0, -0.5, n_rows, dtype=np.float32),
            "y_mm": np.linspace(1.0, 1.5, n_rows, dtype=np.float32),
            "trial_id": [trial_id] * n_rows,
            "phase": np.zeros(n_rows, dtype=np.int8),
        }
    )
    return pd.DataFrame(data)


def _write_minimal_zarr(path: Path, n_rows: int = 1) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["aux_last_field"] = "diameter_mm"
    root.create_dataset("tactile_lr_norm", data=np.zeros((n_rows, 16), dtype=np.float32), chunks=(10, 16))
    root.create_dataset("aux_feat", data=np.zeros((n_rows, 4), dtype=np.float32), chunks=(10, 4))
    root.create_dataset("fz", data=np.zeros((n_rows,), dtype=np.float32), chunks=(10,))
    root.create_dataset("cx", data=np.zeros((n_rows,), dtype=np.float32), chunks=(10,))
    root.create_dataset("cy", data=np.zeros((n_rows,), dtype=np.float32), chunks=(10,))
    root.create_dataset("depth_mm", data=np.ones((n_rows,), dtype=np.float32), chunks=(10,))
    root.create_dataset("x_bounds", data=np.zeros((n_rows, 2), dtype=np.float32), chunks=(10, 2))
    root.create_dataset("y_bounds", data=np.zeros((n_rows, 2), dtype=np.float32), chunks=(10, 2))


class ZarrIndexResolutionTest(unittest.TestCase):
    def test_export_to_zarr_writes_index_inside_each_zarr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zarr_path = Path(tmp) / "dataset_ecemesh.zarr"

            export_to_zarr(_feature_frame("ecemesh_d5_1"), zarr_path)

            self.assertTrue((zarr_path / "dataset_index.json").exists())
            self.assertFalse((zarr_path.parent / "dataset_index.json").exists())

    def test_resolve_zarr_path_rejects_ambiguous_multiple_zarrs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zarr_root = Path(tmp) / "zarr_data"
            zarr_root.mkdir()
            (zarr_root / "dataset_ecemesh.zarr").mkdir()
            (zarr_root / "dataset_ecomesh.zarr").mkdir()

            with self.assertRaisesRegex(RuntimeError, "multiple .zarr"):
                _resolve_zarr_path(tmp)

    def test_loader_rejects_index_that_references_a_different_zarr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a_zarr = root / "dataset_ecemesh.zarr"
            b_zarr = root / "dataset_ecomesh.zarr"
            _write_minimal_zarr(a_zarr)
            _write_minimal_zarr(b_zarr)
            with open(root / "dataset_index.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "samples": [
                            {
                                "trial_id": "ecomesh_d5_7",
                                "phase": "loading",
                                "zarr_path": str(b_zarr),
                                "zarr_index": 0,
                                "depth_bin_mm": 1.0,
                            }
                        ]
                    },
                    f,
                )

            with self.assertRaisesRegex(ValueError, "references a different zarr"):
                ZarrDataset(a_zarr, split="all", phase="all")

    def test_loader_rebuilds_stale_compact_index_when_row_count_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zarr_path = Path(tmp) / "dataset_ecemesh.zarr"
            export_to_zarr(_feature_frame("ecemesh_d5_1"), zarr_path)
            compact_path = _compact_index_path(zarr_path)
            np.savez(
                compact_path,
                trial_codes=np.array([0, 0, 0], dtype=np.int32),
                phase_codes=np.array([0, 0, 0], dtype=np.uint8),
                trial_vocab=np.array(["ecemesh_d5_1"], dtype=object),
            )

            ds = ZarrDataset(zarr_path, split="all", phase="all")

            self.assertEqual(len(ds), 2)


if __name__ == "__main__":
    unittest.main()
