import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import zarr
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("zarr is not installed") from exc

from preprocessing.preprocess import export_to_zarr
from training.data.dataset_zarr import ZarrDataset
from training.utils.contact_geometry import contact_radius, contact_radius_tensor


def _feature_frame(n_rows: int = 2) -> pd.DataFrame:
    data = {f"s_norm_{i}": np.full(n_rows, i / 100.0, dtype=np.float32) for i in range(1, 17)}
    data.update(
        {
            "z_depth_mm": np.linspace(0.5, 0.8, n_rows, dtype=np.float32),
            "z_stage_mm": np.linspace(1.0, 1.3, n_rows, dtype=np.float32),
            "z_contact_mm": np.linspace(0.2, 0.5, n_rows, dtype=np.float32),
            "diameter_mm": np.full(n_rows, 5.0, dtype=np.float32),
            "contact_radius_mm": np.linspace(1.0, 1.2, n_rows, dtype=np.float32),
            "fz_bc": np.linspace(0.3, 0.4, n_rows, dtype=np.float32),
            "x_mm": np.linspace(-1.0, -0.5, n_rows, dtype=np.float32),
            "y_mm": np.linspace(1.0, 1.5, n_rows, dtype=np.float32),
            "trial_id": ["trial_a"] * n_rows,
            "material": ["ecomesh"] * n_rows,
            "z_max_indentation_mm": np.full(n_rows, 1.5, dtype=np.float32),
            "phase": np.zeros(n_rows, dtype=np.int8),
        }
    )
    return pd.DataFrame(data)


class DepthContractTest(unittest.TestCase):
    def test_export_to_zarr_writes_depth_semantic_attrs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zarr_path = Path(tmp) / "dataset_ecomesh.zarr"
            export_to_zarr(_feature_frame(), zarr_path, aux_last_field="contact_radius_mm")

            root = zarr.open_group(str(zarr_path), mode="r")
            self.assertEqual(root.attrs["depth_source"], "z_contact_mm")
            self.assertEqual(root.attrs["target_depth_source"], "z_contact_mm")
            self.assertEqual(root.attrs["aux_depth_source"], "z_contact_mm")
            self.assertEqual(root.attrs["stage_depth_source"], "z_stage_mm")

    def test_loader_exposes_depth_semantic_attrs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zarr_path = Path(tmp) / "dataset_ecomesh.zarr"
            export_to_zarr(_feature_frame(), zarr_path, aux_last_field="contact_radius_mm")

            ds = ZarrDataset(zarr_path, split="all", phase="all")

            self.assertEqual(ds.depth_source, "z_contact_mm")
            self.assertEqual(ds.target_depth_source, "z_contact_mm")
            self.assertEqual(ds.aux_depth_source, "z_contact_mm")
            self.assertEqual(ds.stage_depth_source, "z_stage_mm")

    def test_export_to_zarr_uses_contact_depth_for_aux_and_depth_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            zarr_path = Path(tmp) / "dataset_ecomesh.zarr"
            frame = _feature_frame()
            export_to_zarr(frame, zarr_path, aux_last_field="contact_radius_mm")

            root = zarr.open_group(str(zarr_path), mode="r")
            np.testing.assert_allclose(np.asarray(root["aux_feat"])[:, 2], frame["z_contact_mm"].to_numpy(dtype=np.float32))
            np.testing.assert_allclose(np.asarray(root["depth_mm"]), frame["z_contact_mm"].to_numpy(dtype=np.float32))
            np.testing.assert_allclose(np.asarray(root["z_stage_mm"]), frame["z_stage_mm"].to_numpy(dtype=np.float32))
            np.testing.assert_allclose(np.asarray(root["z_contact_mm"]), frame["z_contact_mm"].to_numpy(dtype=np.float32))

    def test_contact_radius_accepts_geom_alias(self) -> None:
        scalar_geo = contact_radius(1.0, R_mm=2.5, model="geo")
        scalar_geom = contact_radius(1.0, R_mm=2.5, model="geom")
        tensor_geo = contact_radius_tensor(torch.tensor([1.0], dtype=torch.float32), R_mm=2.5, model="geo")
        tensor_geom = contact_radius_tensor(torch.tensor([1.0], dtype=torch.float32), R_mm=2.5, model="geom")

        self.assertAlmostEqual(scalar_geo, scalar_geom)
        np.testing.assert_allclose(np.asarray(tensor_geo), np.asarray(tensor_geom))


if __name__ == "__main__":
    unittest.main()
