import argparse
import csv
import os
import re
import struct
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Constants matching final_logger_gui.py
NUM_SENSORS = 16
FIFO_FRAMES = 10
DUE_PAYLOAD_SIZE = NUM_SENSORS * FIFO_FRAMES * 4
AFD50_FORCE_ID = 0x01A

def read_magic(f):
    line = f.readline().decode("ascii", errors="replace").strip()
    return line

def convert_due_v2(input_path, output_path):
    burst_count = 0
    row_count = 0
    with open(input_path, "rb") as f:
        magic = read_magic(f)
        if magic != "DUE_V2":
            print(f"Warning: {input_path} has unexpected magic {magic}")

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            headers = ["ns", "time_s", "burst_index", "frame_index"] + [f"S{i+1:02d}" for i in range(NUM_SENSORS)]
            writer.writerow(headers)

            record_size = 8 + DUE_PAYLOAD_SIZE
            while True:
                data = f.read(record_size)
                if len(data) < record_size: break

                ns = struct.unpack("<Q", data[:8])[0]
                payload = data[8:]
                values = struct.unpack("<" + ("I" * NUM_SENSORS * FIFO_FRAMES), payload)

                for frame_i in range(FIFO_FRAMES):
                    row = [values[sensor_i * FIFO_FRAMES + frame_i] for sensor_i in range(NUM_SENSORS)]
                    writer.writerow([ns, ns / 1e9, burst_count, frame_i] + row)
                    row_count += 1
                burst_count += 1
    return row_count

def convert_em_v2(input_path, output_path):
    row_count = 0
    with open(input_path, "rb") as f:
        magic = read_magic(f)
        if magic != "EM_V2":
            print(f"Warning: {input_path} has unexpected magic {magic}")

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ns", "time_s", "X", "Y", "Z", "U"])

            record_size = 8 + 32
            while True:
                data = f.read(record_size)
                if len(data) < record_size: break

                ns, x, y, z, u = struct.unpack("<Qdddd", data)
                writer.writerow([ns, ns / 1e9, x, y, z, u])
                row_count += 1
    return row_count

def convert_afd_v2(input_path, output_path, bias_samples=200):
    rows = []
    with open(input_path, "rb") as f:
        magic = read_magic(f)
        if magic != "AFD_V2":
            print(f"Warning: {input_path} has unexpected magic {magic}")

        record_size = 8 + 8
        while True:
            data = f.read(record_size)
            if len(data) < record_size: break

            ns = struct.unpack("<Q", data[:8])[0]
            raw_data = data[8:]

            raw_fx = (raw_data[0] << 8) | raw_data[1]
            raw_fy = (raw_data[2] << 8) | raw_data[3]
            raw_fz = (raw_data[4] << 8) | raw_data[5]

            fx_u = raw_fx / 300.0 - 100.0
            fy_u = raw_fy / 300.0 - 100.0
            fz_u = raw_fz / 300.0 - 100.0

            rows.append({"ns": ns, "raw_fx": raw_fx, "raw_fy": raw_fy, "raw_fz": raw_fz, "fx_u": fx_u, "fy_u": fy_u, "fz_u": fz_u})

    b_fx = sum(r["fx_u"] for r in rows[:bias_samples]) / max(1, min(len(rows), bias_samples))
    b_fy = sum(r["fy_u"] for r in rows[:bias_samples]) / max(1, min(len(rows), bias_samples))
    b_fz = sum(r["fz_u"] for r in rows[:bias_samples]) / max(1, min(len(rows), bias_samples))

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ns", "time_s", "raw_fx", "raw_fy", "raw_fz", "Fx", "Fy", "Fz"])
        for r in rows:
            fx = r["fx_u"] - b_fx
            fy = r["fy_u"] - b_fy
            fz = -(r["fz_u"] - b_fz)
            writer.writerow([r["ns"], r["ns"]/1e9, r["raw_fx"], r["raw_fy"], r["raw_fz"], fx, fy, fz])
    return len(rows)

def convert_lc_v2(input_path, output_path):
    VALUE_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
    row_count = 0
    with open(input_path, "rb") as f:
        magic = read_magic(f)
        if magic != "LC_V2":
            print(f"Warning: {input_path} has unexpected magic {magic}")

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ns", "time_s", "kg"])

            buffer = b""
            while True:
                header = f.read(12)
                if len(header) < 12: break
                ns, size = struct.unpack("<QI", header)
                payload = f.read(size)

                buffer += payload
                while b"\n" in buffer:
                    line, _, buffer = buffer.partition(b"\n")
                    line_str = line.decode("ascii", errors="replace").strip()
                    match = VALUE_PATTERN.search(line_str)
                    if match:
                        kg = float(match.group(0))
                        writer.writerow([ns, ns/1e9, kg])
                        row_count += 1
    return row_count

def find_bin_set(test_dir):
    """Find the 4 v2 bin files in a gui test folder."""
    due_bins = list(test_dir.glob("due_v2_*.bin"))
    em_bins  = list(test_dir.glob("em_v2_*.bin"))
    lc_bins  = list(test_dir.glob("lc_v2_*.bin"))
    afd_bins = list(test_dir.glob("afd_v2_*.bin"))
    if not (due_bins and em_bins and lc_bins and afd_bins):
        return None
    return {
        "due": due_bins[0],
        "em":  em_bins[0],
        "lc":  lc_bins[0],
        "afd": afd_bins[0],
    }


def find_target_test_dir(raw_data_dir):
    """Return the most recent gui_YYYYMMDD_testN folder without CSV files."""
    pattern = re.compile(r"^gui_\d{8}_test\d+$")
    candidates = sorted(
        [d for d in raw_data_dir.iterdir() if d.is_dir() and pattern.match(d.name)],
        key=lambda d: d.name,
    )
    for d in reversed(candidates):
        if not any(d.glob("*.csv")):
            return d
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias-samples", type=int, default=200)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    raw_data_dir = base_dir / "raw_data"

    if not raw_data_dir.exists():
        print(f"raw_data directory not found: {raw_data_dir}")
        return

    test_dir = find_target_test_dir(raw_data_dir)
    if test_dir is None:
        print("No gui test folder without CSV files found.")
        return

    files = find_bin_set(test_dir)
    if files is None:
        print(f"Incomplete bin set in {test_dir.name}. Expected: due_v2_*.bin, em_v2_*.bin, lc_v2_*.bin, afd_v2_*.bin")
        return

    print(f"Converting {test_dir.name} ...")

    c = convert_due_v2(files["due"], test_dir / "due_data.csv")
    print(f"  DUE: {c} rows")
    c = convert_em_v2(files["em"], test_dir / "ethermotion_data.csv")
    print(f"  EM: {c} rows")
    c = convert_lc_v2(files["lc"], test_dir / "loadcell_data.csv")
    print(f"  LC: {c} rows")
    c = convert_afd_v2(files["afd"], test_dir / "afd50_data.csv", bias_samples=args.bias_samples)
    print(f"  AFD: {c} rows")


if __name__ == "__main__":
    main()
