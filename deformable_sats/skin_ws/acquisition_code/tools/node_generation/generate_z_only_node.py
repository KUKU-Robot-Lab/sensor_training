import argparse
import csv
import math
import os


DEFAULT_SOURCE = r"C:\Program Files (x86)\PAIX\NMC\EtherMotion\Node\Conti\4linear_1.0sec_1mm.node"
DEFAULT_OUTPUT_DIR = r"C:\Program Files (x86)\PAIX\NMC\EtherMotion\Node\Conti"


def read_node_file(path):
    encodings = ("utf-8-sig", "cp949")
    last_error = None
    for encoding in encodings:
        try:
            with open(path, "r", newline="", encoding=encoding) as f:
                rows = list(csv.reader(f))
            return rows, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def find_matching_block(data_rows, target_x, target_y, tolerance, nearest):
    blocks = []
    for i in range(0, len(data_rows), 5):
        block = data_rows[i:i + 5]
        if len(block) < 5:
            continue
        try:
            xs = [float(row[2]) for row in block]
            ys = [float(row[3]) for row in block]
        except (ValueError, IndexError):
            continue
        if len({round(x, 9) for x in xs}) == 1 and len({round(y, 9) for y in ys}) == 1:
            distance = math.hypot(xs[0] - target_x, ys[0] - target_y)
            blocks.append((distance, block))

    if not blocks:
        raise ValueError("No 5-row Z scan blocks were found in the source node file.")

    blocks.sort(key=lambda item: item[0])
    distance, block = blocks[0]
    if nearest or distance <= tolerance:
        return block, distance

    raise ValueError(
        f"No exact grid point within tolerance {tolerance}. "
        f"Nearest is X={block[0][2]}, Y={block[0][3]} at distance {distance:.6f}."
    )


def renumber(rows):
    new_rows = []
    for idx, row in enumerate(rows, start=1):
        new_row = list(row)
        new_row[0] = str(idx)
        new_rows.append(new_row)
    return new_rows


def main():
    parser = argparse.ArgumentParser(
        description="Create an EtherMotion node file that scans full Z range at one fixed X/Y point."
    )
    parser.add_argument("--x", type=float, required=True, help="Target X coordinate.")
    parser.add_argument("--y", type=float, required=True, help="Target Y coordinate.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Source full-grid node file.")
    parser.add_argument("--output", help="Output node file path.")
    parser.add_argument(
        "--nearest",
        action="store_true",
        help="Use the nearest X/Y grid point when the requested point is not exactly in the source file.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Allowed X/Y distance for an exact match when --nearest is not used.",
    )
    parser.add_argument(
        "--return-origin",
        action="store_true",
        help="Append the source file's final return-to-origin row, if present.",
    )
    args = parser.parse_args()

    rows, encoding = read_node_file(args.source)
    if len(rows) < 3:
        raise ValueError("The source node file does not contain the expected two header rows plus data.")

    headers = rows[:2]
    data_rows = [row for row in rows[2:] if row]
    return_rows = [row for row in data_rows if len(row) >= 6 and row[2:6] == ["0.0", "0.0", "0.0", "0.0"]]
    scan_rows = [row for row in data_rows if not (len(row) >= 6 and row[2:6] == ["0.0", "0.0", "0.0", "0.0"])]

    block, distance = find_matching_block(scan_rows, args.x, args.y, args.tolerance, args.nearest)
    output_rows = list(block)
    if args.return_origin and return_rows:
        output_rows.append(return_rows[-1])
    output_rows = renumber(output_rows)

    if args.output:
        output_path = args.output
    else:
        selected_x = output_rows[0][2].replace("-", "m").replace(".", "p")
        selected_y = output_rows[0][3].replace("-", "m").replace(".", "p")
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"z_only_x{selected_x}_y{selected_y}.node")

    with open(output_path, "w", newline="", encoding=encoding) as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(headers)
        writer.writerows(output_rows)

    print(f"Created: {output_path}")
    print(f"Selected X={output_rows[0][2]}, Y={output_rows[0][3]}, rows={len(output_rows)}")
    if distance > 0:
        print(f"Requested point distance from selected grid point: {distance:.6f}")


if __name__ == "__main__":
    main()
