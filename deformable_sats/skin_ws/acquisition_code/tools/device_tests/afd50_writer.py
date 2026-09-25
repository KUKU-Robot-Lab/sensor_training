import json
import os
import struct
import sys
import time
from datetime import datetime

import can


CAN_INTERFACE = "ixxat"
CAN_CHANNEL = 0
CAN_BITRATE = 1000000
FORCE_ID = 0x01A

# Binary record: elapsed_ns(uint64), arbitration_id(uint16), raw_can_data(8 bytes)
RECORD_STRUCT = struct.Struct("<QH8s")
BUFFER_RECORDS = 8192


def make_output_path():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    skin_ws_dir = os.path.dirname(current_script_dir)
    save_dir = os.path.join(skin_ws_dir, "afd_50 data")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"afd50_can_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
    return os.path.join(save_dir, filename)


def write_header(binfile):
    header = {
        "format": "AFD50_CAN_RAW_BIN_V1",
        "record_struct": "<QH8s",
        "record_bytes": RECORD_STRUCT.size,
        "columns": ["elapsed_ns", "arbitration_id", "data"],
        "units": {
            "elapsed_ns": "nanoseconds since logging start",
            "arbitration_id": "CAN arbitration id",
            "data": "raw CAN data bytes",
        },
        "force_id": FORCE_ID,
        "can_interface": CAN_INTERFACE,
        "can_channel": CAN_CHANNEL,
        "can_bitrate": CAN_BITRATE,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    binfile.write(b"AFD50_CAN_RAW_BIN_V1\n")
    binfile.write(json.dumps(header, separators=(",", ":")).encode("ascii"))
    binfile.write(b"\nEND_HEADER\n")


def connect_bus():
    while True:
        try:
            print("AFD50 Writer: connecting CAN bus...", file=sys.stderr)
            bus = can.interface.Bus(
                interface=CAN_INTERFACE,
                channel=CAN_CHANNEL,
                bitrate=CAN_BITRATE,
            )
            print("AFD50 Writer: CAN bus connected.", file=sys.stderr)
            return bus
        except can.CanError as exc:
            print(f"AFD50 Writer: CAN bus connection failed: {exc}; retrying in 5 seconds.", file=sys.stderr)
            time.sleep(5)


def main():
    bus = connect_bus()
    filepath = make_output_path()
    print("AFD50_READY")
    sys.stdout.flush()
    print(f"AFD50 Writer: binary raw CAN logging to {filepath}", file=sys.stderr)
    print("AFD50 Writer: no physical conversion or bias correction is applied.", file=sys.stderr)

    buffer = bytearray(RECORD_STRUCT.size * BUFFER_RECORDS)
    buffer_offset = 0
    sample_count = 0
    error_count = 0
    start_ns = time.perf_counter_ns()
    last_report_ns = start_ns
    binfile = None

    try:
        binfile = open(filepath, "wb", buffering=1024 * 1024)
        write_header(binfile)

        while True:
            try:
                msg = bus.recv(timeout=0.1)
            except can.CanError:
                error_count += 1
                continue

            if msg is None or msg.arbitration_id != FORCE_ID:
                continue

            now_ns = time.perf_counter_ns()
            raw_data = bytes(msg.data)
            if len(raw_data) < 8:
                raw_data = raw_data.ljust(8, b"\x00")

            RECORD_STRUCT.pack_into(
                buffer,
                buffer_offset,
                now_ns - start_ns,
                msg.arbitration_id,
                raw_data[:8],
            )
            buffer_offset += RECORD_STRUCT.size
            sample_count += 1

            if buffer_offset >= len(buffer):
                binfile.write(buffer)
                buffer_offset = 0

            if now_ns - last_report_ns >= 1_000_000_000:
                if buffer_offset:
                    binfile.write(buffer[:buffer_offset])
                    buffer_offset = 0
                elapsed_s = (now_ns - start_ns) / 1_000_000_000
                avg_rate = sample_count / elapsed_s if elapsed_s > 0 else 0.0
                print(
                    f"AFD50 Writer: samples={sample_count}, avg_rate={avg_rate:.1f}Hz, errors={error_count}",
                    file=sys.stderr,
                )
                last_report_ns = now_ns

    except KeyboardInterrupt:
        print("AFD50 Writer: stop requested.", file=sys.stderr)
    finally:
        try:
            if binfile is not None and buffer_offset > 0:
                binfile.write(buffer[:buffer_offset])
            if binfile is not None:
                binfile.flush()
                binfile.close()
        except Exception as exc:
            print(f"AFD50 Writer: final flush failed: {exc}", file=sys.stderr)
        if bus is not None:
            bus.shutdown()
        print(f"AFD50 Writer: saved. samples={sample_count}, errors={error_count}", file=sys.stderr)


if __name__ == "__main__":
    main()
