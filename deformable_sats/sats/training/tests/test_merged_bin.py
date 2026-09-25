import numpy as np

from sats.preprocessing.merged_bin import (
    MERGED_DTYPE,
    SENSOR_COLS,
    export_merged_bin_csv,
    merged_bin_to_frame,
    open_merged_bin,
    write_merged_bin,
)


def _make_rows() -> np.ndarray:
    rows = np.zeros(3, dtype=MERGED_DTYPE)
    rows["timestep_sec"] = [0.0, 0.01, 0.02]
    for i, col in enumerate(SENSOR_COLS, start=1):
        rows[col] = 1000.0 + i
    rows["x_mm"] = [-10.0, -10.0, -9.5]
    rows["y_mm"] = [-10.0, -10.0, -10.0]
    rows["z_stage_mm"] = [0.0, 0.5, 0.0]
    rows["z_depth_mm"] = rows["z_stage_mm"]
    rows["u_mm"] = [0.0, 0.05, 0.0]
    rows["Fz"] = [0.0, 1.0, 2.0]
    rows["timestamp_due"] = rows["timestep_sec"]
    rows["timestamp_loadcell"] = rows["timestep_sec"]
    rows["timestamp_ethermotion"] = rows["timestep_sec"]
    return rows


def test_merged_bin_roundtrip_and_column_filter(tmp_path):
    path = tmp_path / "trial_merged.bin"
    write_merged_bin(path, _make_rows(), metadata={"baseline": {"Skin1_mean": 1001.0}})

    header, rows = open_merged_bin(path)
    assert header["row_count"] == 3
    assert rows.shape == (3,)
    assert rows["Fz"][2] == np.float32(2.0)

    df = merged_bin_to_frame(path, columns=["x_mm", "u_mm", "Fz"], u_zero_only=True)
    assert df.columns.tolist() == ["x_mm", "u_mm", "Fz"]
    assert len(df) == 2
    assert df["Fz"].tolist() == [0.0, 2.0]


def test_merged_bin_csv_export_is_optional_surface(tmp_path):
    bin_path = tmp_path / "trial_merged.bin"
    csv_path = tmp_path / "trial_merged.csv"
    write_merged_bin(bin_path, _make_rows())

    n = export_merged_bin_csv(bin_path, csv_path, columns=["timestep_sec", "Fz"], limit=2)

    assert n == 2
    text = csv_path.read_text()
    assert "timestep_sec,Fz" in text
