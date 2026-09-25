import argparse
import csv
import math
import os


DEFAULT_SOURCE = r"C:\Program Files (x86)\PAIX\NMC\EtherMotion\Node\Conti\4linear_1.0sec_1mm.node"


def read_node_file(path):
    encodings = ("utf-8-sig", "cp949")
    last_error = None
    for encoding in encodings:
        try:
            with open(path, "r", newline="", encoding=encoding) as f:
                return list(csv.reader(f)), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def iter_xy_blocks(data_rows):
    for i in range(0, len(data_rows), 5):
        block = data_rows[i:i + 5]
        if len(block) != 5:
            continue
        try:
            xs = [float(row[2]) for row in block]
            ys = [float(row[3]) for row in block]
            zs = [float(row[4]) for row in block]
            us = [float(row[5]) for row in block]
        except (ValueError, IndexError):
            continue
        if len({round(x, 9) for x in xs}) != 1:
            continue
        if len({round(y, 9) for y in ys}) != 1:
            continue
        yield {
            "x": xs[0],
            "y": ys[0],
            "z": zs,
            "u": us,
            "rows": block,
        }


def select_block(data_rows, target_x, target_y, nearest, tolerance):
    candidates = []
    for block in iter_xy_blocks(data_rows):
        distance = math.hypot(block["x"] - target_x, block["y"] - target_y)
        candidates.append((distance, block))

    if not candidates:
        raise ValueError("No fixed-X/Y 5-row blocks were found in the node file.")

    candidates.sort(key=lambda item: item[0])
    distance, block = candidates[0]
    if nearest or distance <= tolerance:
        return block, distance

    raise ValueError(
        f"No X/Y block within tolerance {tolerance}. "
        f"Nearest is X={block['x']}, Y={block['y']}, distance={distance:.6f}."
    )


def make_linear_profile(block, sample_hz, z_duration, z_step_um, include_hold, hold_duration):
    dt = 1.0 / sample_hz
    output = []
    t = 0.0
    rows = block["rows"]

    # Commanded Z moves in this node pattern are row 1 -> 2 and row 4 -> 5.
    segments = [
        ("z_extend", rows[0], rows[1]),
        ("z_retract", rows[3], rows[4]),
    ]

    for name, start_row, end_row in segments:
        z0 = float(start_row[4])
        z1 = float(end_row[4])
        if z_step_um is None:
            steps = int(round(z_duration * sample_hz))
        else:
            z_step_mm = z_step_um / 1000.0
            steps = int(round(abs(z1 - z0) / z_step_mm))

        for step in range(steps + 1):
            if z_step_um is None:
                local_t = min(step * dt, z_duration)
                alpha = local_t / z_duration if z_duration > 0 else 1.0
            else:
                alpha = step / steps if steps > 0 else 1.0
                local_t = alpha * z_duration
            z_cmd = z0 + (z1 - z0) * alpha
            output.append([
                f"{t + local_t:.6f}",
                f"{block['x']:.6f}",
                f"{block['y']:.6f}",
                f"{z_cmd:.6f}",
                name,
            ])
        t += z_duration

        if include_hold and hold_duration > 0:
            hold_steps = int(round(hold_duration * sample_hz))
            for step in range(1, hold_steps + 1):
                local_t = step * dt
                output.append([
                    f"{t + local_t:.6f}",
                    f"{block['x']:.6f}",
                    f"{block['y']:.6f}",
                    f"{z1:.6f}",
                    f"{name}_hold",
                ])
            t += hold_duration

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Extract a fixed-X/Y Z command profile from an EtherMotion .node file."
    )
    parser.add_argument("--x", type=float, required=True, help="Target X coordinate.")
    parser.add_argument("--y", type=float, required=True, help="Target Y coordinate.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Source EtherMotion node file.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--sample-hz", type=float, default=200.0, help="Command profile sample rate.")
    parser.add_argument(
        "--z-step-um",
        type=float,
        help="Generate samples by Z position increment in micrometers instead of by sample rate.",
    )
    parser.add_argument(
        "--z-duration",
        type=float,
        default=1.0,
        help="Duration in seconds for each 1D Z move. Default matches 4linear_1.0sec_1mm naming.",
    )
    parser.add_argument("--nearest", action="store_true", help="Use nearest available X/Y grid point.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Exact-match X/Y tolerance.")
    parser.add_argument(
        "--include-hold",
        action="store_true",
        help="Insert constant-Z hold samples after each Z move.",
    )
    parser.add_argument("--hold-duration", type=float, default=0.0, help="Hold duration in seconds.")
    args = parser.parse_args()

    rows, _ = read_node_file(args.source)
    data_rows = [row for row in rows[2:] if row]
    data_rows = [row for row in data_rows if not (len(row) >= 6 and row[2:6] == ["0.0", "0.0", "0.0", "0.0"])]
    block, distance = select_block(data_rows, args.x, args.y, args.nearest, args.tolerance)
    profile = make_linear_profile(
        block,
        args.sample_hz,
        args.z_duration,
        args.z_step_um,
        args.include_hold,
        args.hold_duration,
    )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep_s", "x_cmd", "y_cmd", "z_cmd", "segment"])
        writer.writerows(profile)

    print(f"Created: {args.output}")
    print(f"Selected X={block['x']}, Y={block['y']}, samples={len(profile)}")
    if distance > 0:
        print(f"Requested point distance from selected grid point: {distance:.6f}")


if __name__ == "__main__":
    main()
