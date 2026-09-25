import numpy as np

from sats.preprocessing.merged_bin import MERGED_DTYPE, SENSOR_COLS, write_merged_bin
from sats.training.config import SATSConfig
from sats.training.dataset import SATSSequenceDataset


def test_sequence_dataset_prefers_merged_bin_and_uses_header_baseline(tmp_path):
    raw_dir = tmp_path / "raw_data"
    gt_dir = tmp_path / "gt"
    trial_dir = raw_dir / "ecomesh" / "d5" / "z_2.5mm" / "test1"
    trial_dir.mkdir(parents=True)
    gt_dir.mkdir()

    rows = np.zeros(3, dtype=MERGED_DTYPE)
    rows["x_mm"] = [0.0, 0.0, 0.0]
    rows["y_mm"] = [0.0, 0.0, 0.0]
    rows["u_mm"] = [0.0, 0.05, 0.0]
    rows["Fz"] = [0.0, 10.0, 2.0]
    baseline = {}
    for i, col in enumerate(SENSOR_COLS, start=1):
        value = 1000.0 + i
        rows[col] = value
        baseline[f"Skin{i}_mean"] = value

    trial_id = "ecomesh_d5_z2.5_test1"
    write_merged_bin(
        trial_dir / f"{trial_id}_merged.bin",
        rows,
        metadata={"baseline": baseline},
    )
    np.save(gt_dir / f"{trial_id}_targets.npy", np.ones((2, 41, 41), dtype=np.float32))

    cfg = SATSConfig(
        raw_dir=str(raw_dir),
        gt_dir=str(gt_dir),
        min_seq_len=1,
        seq_len=10,
        prefer_merged_bin=True,
        use_u_zero_only=True,
    )
    ds = SATSSequenceDataset([trial_id], cfg)

    assert len(ds) == 1
    sensor_seq, gt_seq, length = ds[0]
    assert length == 2
    assert sensor_seq.shape == (2, 16)
    assert np.allclose(sensor_seq.numpy(), 0.0)
    assert np.allclose(gt_seq[0].numpy(), 0.0)
    assert np.allclose(gt_seq[1].numpy(), 100.0)
