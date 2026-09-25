import argparse
import json
import os
import struct
import sys
import time
from datetime import datetime

try:
    import serial
except ImportError:
    serial = None

from serial_autodetect import resolve_port


DEFAULT_PORT = "COM10"  # fallback only; auto-detected in main() when possible
LOADCELL_PORT_VID_PID = {(0x067B, 0x2303)}  # Prolific USB-RS232 adapter
LOADCELL_PORT_KEYWORDS = ("prolific", "usb-to-serial", "usb serial")
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 0.01
DEFAULT_READ_SIZE = 8192
DEFAULT_RX_BUFFER_SIZE = 1024 * 1024
MAGIC = "LOADCELL_BIN_V1"
RECORD_STRUCT = struct.Struct("<QI")  # elapsed_ns, payload_size


def default_output_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(base_dir, "loadcell data")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"loadcell_raw_{timestamp}.bin")


def write_header(binfile, args):
    header = {
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        "port": args.port,
        "baud": args.baud,
        "bytesize": 8,
        "parity": "N",
        "stopbits": 1,
        "timeout_sec": args.timeout,
        "read_size": args.read_size,
        "record_format": "<QI + payload",
        "record_fields": ["elapsed_ns", "payload_size", "payload"],
        "timebase": "time.perf_counter_ns from first read loop",
        "payload": "raw RS232 bytes from indicator",
    }
    binfile.write((MAGIC + "\n").encode("ascii"))
    binfile.write(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    binfile.write(b"\nEND_HEADER\n")


def open_serial(args):
    if serial is None:
        raise RuntimeError(
            "pyserial is not installed. Install it with: python -m pip install pyserial"
        )

    ser = serial.Serial(
        port=args.port,
        baudrate=args.baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=args.timeout,
    )
    try:
        ser.set_buffer_size(rx_size=args.rx_buffer_size)
    except (AttributeError, OSError, ValueError):
        pass
    if not args.keep_input_buffer:
        ser.reset_input_buffer()
    return ser


def log_raw(args):
    output_path = os.path.abspath(args.output or default_output_path())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ser = None
    chunks = 0
    bytes_written = 0
    start_ns = 0
    next_status = time.perf_counter() + args.status_interval_sec
    next_flush = time.perf_counter() + args.flush_interval_sec

    try:
        ser = open_serial(args)
        print(f"Loadcell raw logger connected on {args.port} ({args.baud} baud, 8N1).")
        print(f"Saving BIN: {output_path}")
        print("Recording raw bytes only. Press Ctrl+C to stop.")

        with open(output_path, "wb", buffering=args.file_buffer_size) as binfile:
            write_header(binfile, args)
            start_ns = time.perf_counter_ns()
            read_buffer = bytearray(args.read_size)
            read_view = memoryview(read_buffer)

            while True:
                elapsed_s = (time.perf_counter_ns() - start_ns) / 1_000_000_000.0
                if args.duration_sec is not None and elapsed_s >= args.duration_sec:
                    break

                payload_size = ser.readinto(read_buffer)
                if not payload_size:
                    continue

                elapsed_ns = time.perf_counter_ns() - start_ns
                binfile.write(RECORD_STRUCT.pack(elapsed_ns, payload_size))
                binfile.write(read_view[:payload_size])
                chunks += 1
                bytes_written += payload_size

                now = time.perf_counter()
                if now >= next_flush:
                    binfile.flush()
                    next_flush = now + args.flush_interval_sec

                if now >= next_status:
                    rate = bytes_written / elapsed_s if elapsed_s > 0 else 0.0
                    print(
                        f"\r[RAW] chunks:{chunks} bytes:{bytes_written} "
                        f"rate:{rate:.0f} B/s",
                        end="",
                    )
                    next_status = now + args.status_interval_sec

    except KeyboardInterrupt:
        print("\nStop requested.")
    finally:
        if ser is not None and ser.is_open:
            ser.close()

    print(f"\nLoadcell raw logging complete: {output_path}")
    print(f"Chunks written: {chunks}")
    print(f"Payload bytes written: {bytes_written}")


def read_header(path):
    with open(path, "rb") as binfile:
        magic = binfile.readline().decode("ascii").rstrip("\n")
        if magic != MAGIC:
            raise ValueError(f"unsupported magic {magic!r}")
        header = json.loads(binfile.readline().decode("utf-8"))
        marker = binfile.readline()
        if marker != b"END_HEADER\n":
            raise ValueError("missing END_HEADER marker")
        return header, binfile.tell()


def self_check():
    print(f"magic: {MAGIC}")
    print(f"record_struct_size: {RECORD_STRUCT.size}")
    print("Self-check passed.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Record CI-400AL loadcell indicator RS232 bytes at full receive rate."
    )
    parser.add_argument(
        "--port",
        default=None,
        help=f"Serial port. Default: auto-detect, falling back to the LOADCELL_PORT "
        f"env var, then {DEFAULT_PORT}.",
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"Baud rate, default {DEFAULT_BAUD}.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Serial read timeout in seconds.")
    parser.add_argument("--read-size", type=int, default=DEFAULT_READ_SIZE, help="Maximum bytes per serial read.")
    parser.add_argument("--rx-buffer-size", type=int, default=DEFAULT_RX_BUFFER_SIZE, help="Requested serial RX buffer size.")
    parser.add_argument("--file-buffer-size", type=int, default=1024 * 1024, help="Binary file buffer size.")
    parser.add_argument("--flush-interval-sec", type=float, default=1.0, help="Flush interval for the binary file.")
    parser.add_argument("--status-interval-sec", type=float, default=1.0, help="Console status update interval.")
    parser.add_argument("--duration-sec", type=float, default=None, help="Optional automatic stop time in seconds.")
    parser.add_argument("--output", default=None, help="Optional BIN output path.")
    parser.add_argument("--keep-input-buffer", action="store_true", help="Do not clear existing serial bytes on start.")
    parser.add_argument("--self-check", action="store_true", help="Check binary format constants and exit.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return 0

    args.port = resolve_port(
        "Loadcell", args.port, os.environ.get("LOADCELL_PORT"), DEFAULT_PORT,
        LOADCELL_PORT_VID_PID, LOADCELL_PORT_KEYWORDS,
    )

    try:
        log_raw(args)
    except Exception as exc:
        print(f"Loadcell raw logger error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
