import json
import struct
import sys
from pathlib import Path

import numpy as np

RECORD_STRUCT = struct.Struct("<Qddddiiii")
DTYPE = np.dtype([
    ("elapsed_ns", "<u8"),
    ("x_cmd", "<f8"),
    ("y_cmd", "<f8"),
    ("z_cmd", "<f8"),
    ("u_cmd", "<f8"),
    ("x_lcmd", "<i4"),
    ("y_lcmd", "<i4"),
    ("z_lcmd", "<i4"),
    ("u_lcmd", "<i4"),
])


def read_header(f):
    magic = f.readline().decode("ascii").strip()
    header = json.loads(f.readline().decode("ascii"))
    end = f.readline()
    assert end == b"END_HEADER\n"
    return magic, header


def analyze(path):
    path = Path(path)
    with open(path, "rb") as f:
        magic, header = read_header(f)
        data_start = f.tell()
        f.seek(0, 2)
        total = f.tell()

    n_full = (total - data_start) // RECORD_STRUCT.size
    arr = np.memmap(path, dtype=DTYPE, mode="r", offset=data_start, shape=(n_full,))

    z_mm = arr["z_lcmd"] * 0.0001  # actual position pulses -> mm
    zcmd_mm = arr["z_cmd"] * 0.0001  # command pulses -> mm
    t_s = arr["elapsed_ns"] / 1e9

    print(f"=== {path.name} ===")
    print(f"records: {n_full}, duration: {t_s[-1]:.1f}s")
    print(f"z_lcmd range: {z_mm.min():.4f} .. {z_mm.max():.4f} mm")
    print(f"z_cmd  range: {zcmd_mm.min():.4f} .. {zcmd_mm.max():.4f} mm")

    # Detect plateaus (steady-state levels) in z_lcmd (actual encoder position)
    # round to avoid float jitter
    rounded = np.round(z_mm, 4)
    change_idx = np.where(np.diff(rounded) != 0)[0]

    # group into plateaus: find runs of constant value lasting more than some min length
    levels = []
    start = 0
    boundaries = np.concatenate(([0], change_idx + 1, [n_full]))
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        length = e - s
        if length >= 50:  # plateau must be at least 50 samples
            levels.append((rounded[s], length, t_s[s], t_s[e - 1]))

    print(f"number of plateaus (>=50 samples): {len(levels)}")

    # collapse consecutive plateaus with same rounded value (jitter)
    collapsed = []
    for val, length, t0, t1 in levels:
        if collapsed and abs(collapsed[-1][0] - val) < 0.0005:
            # merge
            prev = collapsed[-1]
            collapsed[-1] = (prev[0], prev[1] + length, prev[2], t1)
        else:
            collapsed.append([val, length, t0, t1])

    print(f"collapsed plateaus: {len(collapsed)}")
    plateau_vals = [c[0] for c in collapsed]
    print("plateau z values (mm):", [f"{v:.3f}" for v in plateau_vals[:40]])
    if len(plateau_vals) > 40:
        print(f"... ({len(plateau_vals)} total)")

    diffs = np.diff(plateau_vals)
    print("step diffs between consecutive plateaus (mm):")
    print([f"{d:.4f}" for d in diffs[:40]])
    if len(diffs) > 40:
        print(f"... ({len(diffs)} total)")

    nonzero_diffs = diffs[np.abs(diffs) > 1e-6]
    if len(nonzero_diffs) > 0:
        print(f"mean abs step: {np.mean(np.abs(nonzero_diffs)):.4f} mm")
        print(f"median abs step: {np.median(np.abs(nonzero_diffs)):.4f} mm")
        unique_steps = np.unique(np.round(np.abs(nonzero_diffs), 3))
        print(f"unique abs step sizes (rounded 0.001): {unique_steps}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
