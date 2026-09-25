import argparse
import csv
import json
import os
import re
import struct
import sys
from datetime import datetime
from pathlib import Path


NUM_SENSORS = 16
FIFO_FRAMES = 10
DUE_PAYLOAD_SIZE = NUM_SENSORS * FIFO_FRAMES * 4
DUE_RECORD_STRUCT = struct.Struct("<Q")
DUE_PAYLOAD_STRUCT = struct.Struct("<" + ("I" * NUM_SENSORS * FIFO_FRAMES))
AFD50_RECORD_STRUCT = struct.Struct("<QH8s")
ETHERMOTION_RECORD_STRUCT = struct.Struct("<Qddddiiii")
LOADCELL_RECORD_STRUCT = struct.Struct("<QI")

DUE_MAGIC = "DUE_RAW_BURST_BIN_V1"
AFD50_MAGIC = "AFD50_CAN_RAW_BIN_V1"
ETHERMOTION_MAGIC = "ETHERMOTION_ENCODER_BIN_V1"
LOADCELL_MAGIC = "LOADCELL_BIN_V1"
AFD50_FORCE_ID = 0x01A
LCMD_MM_PER_PULSE = 0.0001


def read_bin_header(binfile):
    magic = binfile.readline().decode("ascii", errors="replace").strip()
    if not magic:
        raise ValueError("missing binary magic header")

    header_line = binfile.readline()
    if not header_line:
        raise ValueError(f"{magic}: missing JSON header")

    try:
        header = json.loads(header_line.decode("ascii"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{magic}: invalid JSON header: {exc}") from exc

    end_header = binfile.readline()
    if end_header != b"END_HEADER\n":
        raise ValueError(f"{magic}: missing END_HEADER marker")

    return magic, header


def payload_to_rows(payload):
    values = DUE_PAYLOAD_STRUCT.unpack(payload)
    return [
        [
            values[sensor_i * FIFO_FRAMES + frame_i]
            for sensor_i in range(NUM_SENSORS)
        ]
        for frame_i in range(FIFO_FRAMES)
    ]


def iter_due_records(path):
    with open(path, "rb") as binfile:
        magic, header = read_bin_header(binfile)
        if magic != DUE_MAGIC:
            raise ValueError(f"{path}: expected {DUE_MAGIC}, got {magic}")

        record_bytes = int(header.get("record_bytes", DUE_RECORD_STRUCT.size + DUE_PAYLOAD_SIZE))
        if record_bytes != DUE_RECORD_STRUCT.size + DUE_PAYLOAD_SIZE:
            raise ValueError(f"{path}: unsupported DUE record size {record_bytes}")

        record_i = 0
        while True:
            elapsed_bytes = binfile.read(DUE_RECORD_STRUCT.size)
            if not elapsed_bytes:
                break
            if len(elapsed_bytes) != DUE_RECORD_STRUCT.size:
                raise ValueError(f"{path}: truncated DUE elapsed_ns at record {record_i}")

            payload = binfile.read(DUE_PAYLOAD_SIZE)
            if len(payload) != DUE_PAYLOAD_SIZE:
                raise ValueError(f"{path}: truncated DUE payload at record {record_i}")

            elapsed_ns = DUE_RECORD_STRUCT.unpack(elapsed_bytes)[0]
            yield record_i, elapsed_ns, payload
            record_i += 1


def convert_due_bin(input_path, output_path):
    row_count = 0
    burst_count = 0
    headers = ["elapsed_ns", "time_s", "burst_index", "frame_index"]
    headers.extend([f"Skin{i}" for i in range(1, NUM_SENSORS + 1)])

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for burst_i, elapsed_ns, payload in iter_due_records(input_path):
            burst_count += 1
            for frame_i, row in enumerate(payload_to_rows(payload)):
                writer.writerow([elapsed_ns, elapsed_ns / 1_000_000_000, burst_i, frame_i] + row)
                row_count += 1

    return {"bursts": burst_count, "rows": row_count}


def iter_ethermotion_records(path):
    with open(path, "rb") as binfile:
        magic, header = read_bin_header(binfile)
        if magic != ETHERMOTION_MAGIC:
            raise ValueError(f"{path}: expected {ETHERMOTION_MAGIC}, got {magic}")

        record_bytes = int(header.get("record_bytes", ETHERMOTION_RECORD_STRUCT.size))
        if record_bytes != ETHERMOTION_RECORD_STRUCT.size:
            raise ValueError(
                f"{path}: unsupported EtherMotion record size {record_bytes} "
                f"(expected {ETHERMOTION_RECORD_STRUCT.size})"
            )

        record_i = 0
        while True:
            record = binfile.read(ETHERMOTION_RECORD_STRUCT.size)
            if not record:
                break
            if len(record) != ETHERMOTION_RECORD_STRUCT.size:
                raise ValueError(f"{path}: truncated EtherMotion record {record_i}")
            yield record_i, ETHERMOTION_RECORD_STRUCT.unpack(record)
            record_i += 1


def convert_ethermotion_bin(input_path, output_path):
    row_count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "elapsed_ns", "time_s",
            "X", "Y", "Z", "U",
            "X_lCmd", "Y_lCmd", "Z_lCmd", "U_lCmd",
            "Z_lCmd_mm",
        ])
        for _, values in iter_ethermotion_records(input_path):
            elapsed_ns, x, y, z, u, xl, yl, zl, ul = values
            writer.writerow([
                elapsed_ns,
                elapsed_ns / 1_000_000_000,
                x, y, z, u,
                xl, yl, zl, ul,
                zl * LCMD_MM_PER_PULSE,
            ])
            row_count += 1
    return {"rows": row_count}


def to_u16_be(data, index):
    return (data[index * 2] << 8) | data[index * 2 + 1]


def force_from_raw(raw):
    return raw / 300.0 - 100.0


def iter_afd50_records(path):
    with open(path, "rb") as binfile:
        magic, header = read_bin_header(binfile)
        if magic != AFD50_MAGIC:
            raise ValueError(f"{path}: expected {AFD50_MAGIC}, got {magic}")

        record_bytes = int(header.get("record_bytes", AFD50_RECORD_STRUCT.size))
        if record_bytes != AFD50_RECORD_STRUCT.size:
            raise ValueError(f"{path}: unsupported AFD50 record size {record_bytes}")

        record_i = 0
        while True:
            record = binfile.read(AFD50_RECORD_STRUCT.size)
            if not record:
                break
            if len(record) != AFD50_RECORD_STRUCT.size:
                raise ValueError(f"{path}: truncated AFD50 record {record_i}")
            yield record_i, AFD50_RECORD_STRUCT.unpack(record)
            record_i += 1


def afd50_force_rows(input_path):
    rows = []
    for record_i, (elapsed_ns, arbitration_id, raw_data) in iter_afd50_records(input_path):
        if arbitration_id != AFD50_FORCE_ID:
            continue
        raw_fx = to_u16_be(raw_data, 0)
        raw_fy = to_u16_be(raw_data, 1)
        raw_fz = to_u16_be(raw_data, 2)
        rows.append(
            {
                "record_index": record_i,
                "elapsed_ns": elapsed_ns,
                "arbitration_id": arbitration_id,
                "raw_hex": raw_data.hex(),
                "raw_fx": raw_fx,
                "raw_fy": raw_fy,
                "raw_fz": raw_fz,
                "fx_unbiased": force_from_raw(raw_fx),
                "fy_unbiased": force_from_raw(raw_fy),
                "fz_unbiased": force_from_raw(raw_fz),
            }
        )
    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


def calculate_afd50_bias(rows, bias_samples):
    sample_rows = rows[:bias_samples] if bias_samples > 0 else []
    return {
        "fx": mean([row["fx_unbiased"] for row in sample_rows]),
        "fy": mean([row["fy_unbiased"] for row in sample_rows]),
        "fz": mean([row["fz_unbiased"] for row in sample_rows]),
        "samples": len(sample_rows),
    }


def convert_afd50_bin(input_path, output_path, bias_samples=200, invert_fz=True):
    rows = afd50_force_rows(input_path)
    bias = calculate_afd50_bias(rows, bias_samples)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "elapsed_ns", "time_s",
            "arbitration_id", "raw_hex",
            "raw_fx", "raw_fy", "raw_fz",
            "Fx", "Fy", "Fz",
        ])

        for row in rows:
            fx = row["fx_unbiased"] - bias["fx"]
            fy = row["fy_unbiased"] - bias["fy"]
            fz = row["fz_unbiased"] - bias["fz"]
            if invert_fz:
                fz = -fz

            writer.writerow([
                row["elapsed_ns"],
                row["elapsed_ns"] / 1_000_000_000,
                f"0x{row['arbitration_id']:03X}",
                row["raw_hex"],
                row["raw_fx"],
                row["raw_fy"],
                row["raw_fz"],
                fx, fy, fz,
            ])

    return {"rows": len(rows), "bias": bias, "invert_fz": invert_fz}


def iter_loadcell_records(path):
    with open(path, "rb") as binfile:
        magic, header = read_bin_header(binfile)
        if magic != LOADCELL_MAGIC:
            raise ValueError(f"{path}: expected {LOADCELL_MAGIC}, got {magic}")

        record_i = 0
        while True:
            record_header = binfile.read(LOADCELL_RECORD_STRUCT.size)
            if not record_header:
                break
            if len(record_header) != LOADCELL_RECORD_STRUCT.size:
                raise ValueError(f"{path}: truncated record header at record {record_i}")

            elapsed_ns, payload_size = LOADCELL_RECORD_STRUCT.unpack(record_header)
            payload = binfile.read(payload_size)
            if len(payload) != payload_size:
                raise ValueError(f"{path}: truncated payload at record {record_i}")

            yield record_i, elapsed_ns, payload
            record_i += 1


def read_loadcell_header(path):
    with open(path, "rb") as binfile:
        magic, header = read_bin_header(binfile)
        if magic != LOADCELL_MAGIC:
            raise ValueError(f"{path}: expected {LOADCELL_MAGIC}, got {magic}")
        return header


def convert_loadcell_bin(input_path, output_path):
    import re as _re
    VALUE_PATTERN = _re.compile(r"[-+]?\d+(?:\.\d+)?")

    def parse_kg(raw_line):
        match = VALUE_PATTERN.search(raw_line)
        if match is None:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    header = read_loadcell_header(input_path)
    baseline = header.get("baseline") or {}
    baseline_mean = baseline.get("mean_kg")

    rows = []
    buffer = b""
    elapsed_ns = 0
    for record_i, elapsed_ns, payload in iter_loadcell_records(input_path):
        buffer += payload
        while b"\n" in buffer:
            line, _, buffer = buffer.partition(b"\n")
            line_str = line.decode("ascii", errors="replace").strip()
            if not line_str:
                continue
            kg = parse_kg(line_str)
            if kg is None:
                continue
            kg_zeroed = kg - baseline_mean if baseline_mean is not None else None
            rows.append((elapsed_ns, kg, kg_zeroed))

    if buffer:
        line_str = buffer.decode("ascii", errors="replace").strip()
        if line_str:
            kg = parse_kg(line_str)
            if kg is not None:
                kg_zeroed = kg - baseline_mean if baseline_mean is not None else None
                rows.append((elapsed_ns, kg, kg_zeroed))

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["elapsed_ns", "time_s", "kg_raw", "kg_zeroed", "baseline_mean_kg"])
        for elapsed_ns, kg, kg_zeroed in rows:
            writer.writerow([
                elapsed_ns,
                elapsed_ns / 1_000_000_000,
                kg,
                "" if kg_zeroed is None else kg_zeroed,
                "" if baseline_mean is None else baseline_mean,
            ])

    return {"rows": len(rows), "baseline_mean_kg": baseline_mean}


def find_bin_set(test_dir):
    """Find supported bin files that exist in a test folder."""
    patterns = {
        "due": "due_raw_burst_*.bin",
        "afd50": "afd50_can_raw_*.bin",
        "ethermotion": "ethermotion_encoder_*.bin",
        "loadcell": "loadcell_raw_*.bin",
    }
    inputs = {}
    for name, pattern in patterns.items():
        bins = sorted(test_dir.glob(pattern))
        if bins:
            inputs[name] = bins[0]
    return inputs


def find_target_test_dir(raw_data_dir):
    """Return the most recent YYYYMMDD_testN folder that has no CSV files."""
    pattern = re.compile(r"^\d{8}_test\d+$")
    candidates = sorted(
        [d for d in raw_data_dir.iterdir() if d.is_dir() and pattern.match(d.name)],
        key=lambda d: d.name,
    )
    for d in reversed(candidates):
        if not any(d.glob("*.csv")):
            return d
    return None


def convert_set(inputs, output_dir, args):
    stats = {}

    if "due" in inputs:
        print(f"DUE -> {output_dir / 'due_data.csv'}", file=sys.stderr)
        stats["due"] = convert_due_bin(inputs["due"], output_dir / "due_data.csv")

    if "ethermotion" in inputs:
        print(f"EtherMotion -> {output_dir / 'ethermotion_data.csv'}", file=sys.stderr)
        stats["ethermotion"] = convert_ethermotion_bin(inputs["ethermotion"], output_dir / "ethermotion_data.csv")

    if "afd50" in inputs:
        print(f"AFD50 -> {output_dir / 'afd50_data.csv'}", file=sys.stderr)
        stats["afd50"] = convert_afd50_bin(
            inputs["afd50"],
            output_dir / "afd50_data.csv",
            bias_samples=args.bias_samples,
            invert_fz=not args.no_invert_fz,
        )

    if "loadcell" in inputs:
        print(f"Loadcell -> {output_dir / 'loadcell_data.csv'}", file=sys.stderr)
        stats["loadcell"] = convert_loadcell_bin(inputs["loadcell"], output_dir / "loadcell_data.csv")

    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert binary logs in the latest log test folder into CSV files."
    )
    default_base = Path(__file__).resolve().parent
    parser.add_argument("--base-dir", type=Path, default=default_base)
    parser.add_argument(
        "--bias-samples",
        type=int,
        default=200,
        help="Number of initial AFD50 samples used as zero-force bias. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-invert-fz",
        action="store_true",
        help="Do not invert AFD50 Fz after bias removal.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir.resolve()
    raw_data_dir = base_dir / "log"

    if not raw_data_dir.exists():
        print(f"log directory not found: {raw_data_dir}", file=sys.stderr)
        return 1

    test_dir = find_target_test_dir(raw_data_dir)
    if test_dir is None:
        print("No test folder without CSV files found in log/.", file=sys.stderr)
        return 1

    inputs = find_bin_set(test_dir)
    if not inputs:
        print(
            f"No supported bin files found in {test_dir.name}. "
            "Supported: due_raw_burst_*.bin, afd50_can_raw_*.bin, "
            "ethermotion_encoder_*.bin, loadcell_raw_*.bin",
            file=sys.stderr,
        )
        return 1

    print(f"Converting {test_dir.name} ...", file=sys.stderr)
    stats = convert_set(inputs, test_dir, args)
    summary = ", ".join(f"{name}_rows={stat['rows']}" for name, stat in stats.items())
    print(f"Done: {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
