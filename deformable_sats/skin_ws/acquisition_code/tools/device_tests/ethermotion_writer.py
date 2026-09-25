import json
import os
import struct
import sys
import time
from ctypes import *
from datetime import datetime


DLL_PATH = r"C:\Program Files (x86)\PAIX\NMC\Sample\Python\dist_64bit\NMC2.dll"
DEV_NO = 11

# Binary record: elapsed_ns(uint64), x_enc(double), y_enc(double), z_enc(double)
RECORD_STRUCT = struct.Struct("<Qddd")
BUFFER_RECORDS = 8192


class NMC_AXES_EXPR(Structure):
    _pack_ = 1
    _fields_ = [
        ("nBusy", c_short * 8),
        ("nError", c_short * 8),
        ("nNear", c_short * 8),
        ("nPLimit", c_short * 8),
        ("nMLimit", c_short * 8),
        ("nAlarm", c_short * 8),
        ("nEmer", c_short * 2),
        ("nSwPLimit", c_short * 8),
        ("nInpo", c_short * 8),
        ("nHome", c_short * 8),
        ("nEncZ", c_short * 8),
        ("nOrg", c_short * 8),
        ("nSReady", c_short * 8),
        ("nContStatus", c_short * 2),
        ("nDummy", c_short * 6),
        ("nSwMLimit", c_short * 8),
        ("lEnc", c_int * 8),
        ("lCmd", c_int * 8),
        ("dEnc", c_double * 8),
        ("dCmd", c_double * 8),
        ("dummy", c_char * 4),
    ]


def make_output_path():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    skin_ws_dir = os.path.dirname(current_script_dir)
    save_dir = os.path.join(skin_ws_dir, "ethermotion data")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"ethermotion_encoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
    return os.path.join(save_dir, filename)


def write_header(binfile):
    header = {
        "format": "ETHERMOTION_ENCODER_BIN_V1",
        "record_struct": "<Qddd",
        "record_bytes": RECORD_STRUCT.size,
        "columns": ["elapsed_ns", "x_enc", "y_enc", "z_enc"],
        "units": {
            "elapsed_ns": "nanoseconds since logging start",
            "x_enc": "controller encoder position unit",
            "y_enc": "controller encoder position unit",
            "z_enc": "controller encoder position unit",
        },
        "dev_no": DEV_NO,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    binfile.write(b"ETHERMOTION_ENCODER_BIN_V1\n")
    binfile.write(json.dumps(header, separators=(",", ":")).encode("ascii"))
    binfile.write(b"\nEND_HEADER\n")


def main():
    if not os.path.exists(DLL_PATH):
        print(f"EtherMotion Writer: DLL not found: {DLL_PATH}", file=sys.stderr)
        return

    try:
        nmc = cdll.LoadLibrary(DLL_PATH)
    except Exception as e:
        print(f"EtherMotion Writer: DLL load failed: {e}", file=sys.stderr)
        return

    print(f"EtherMotion Writer: connecting DEV_NO={DEV_NO} (192.168.0.{DEV_NO})...", file=sys.stderr)
    ret = nmc.nmc_PingCheck(DEV_NO, 100)
    if ret != 0:
        print(f"EtherMotion Writer: ping failed, ret={ret}; trying open anyway.", file=sys.stderr)

    ret = nmc.nmc_OpenDeviceEx(DEV_NO, 100)
    if ret != 0:
        print(f"EtherMotion Writer: open failed, ret={ret}.", file=sys.stderr)
        return

    filepath = make_output_path()
    print("ETHERMOTION_READY")
    sys.stdout.flush()
    print(f"EtherMotion Writer: binary encoder logging to {filepath}", file=sys.stderr)
    print("EtherMotion Writer: no fixed polling interval; press Ctrl+C to stop.", file=sys.stderr)

    axes_status = NMC_AXES_EXPR()
    sample_count = 0
    error_count = 0
    buffer = bytearray(RECORD_STRUCT.size * BUFFER_RECORDS)
    buffer_offset = 0
    start_ns = time.perf_counter_ns()
    last_report_ns = start_ns
    binfile = None

    try:
        binfile = open(filepath, "wb", buffering=1024 * 1024)
        write_header(binfile)

        while True:
            ret = nmc.nmc_GetAxesExpress(DEV_NO, byref(axes_status))
            if ret != 0:
                error_count += 1
                continue

            now_ns = time.perf_counter_ns()
            RECORD_STRUCT.pack_into(
                buffer,
                buffer_offset,
                now_ns - start_ns,
                axes_status.dEnc[0],
                axes_status.dEnc[1],
                axes_status.dEnc[2],
            )
            buffer_offset += RECORD_STRUCT.size
            sample_count += 1

            if buffer_offset >= len(buffer):
                binfile.write(buffer)
                buffer_offset = 0

            if now_ns - last_report_ns >= 1_000_000_000:
                elapsed_s = (now_ns - start_ns) / 1_000_000_000
                avg_rate = sample_count / elapsed_s if elapsed_s > 0 else 0.0
                print(
                    f"EtherMotion Writer: samples={sample_count}, avg_rate={avg_rate:.1f}Hz, errors={error_count}",
                    file=sys.stderr,
                )
                last_report_ns = now_ns

    except KeyboardInterrupt:
        print("EtherMotion Writer: stop requested.", file=sys.stderr)
    finally:
        try:
            if binfile is not None and buffer_offset > 0:
                binfile.write(buffer[:buffer_offset])
                binfile.flush()
            if binfile is not None:
                binfile.close()
        except Exception as e:
            print(f"EtherMotion Writer: final flush failed: {e}", file=sys.stderr)
        nmc.nmc_CloseDevice(DEV_NO)
        print(f"EtherMotion Writer: saved. samples={sample_count}, errors={error_count}", file=sys.stderr)


if __name__ == "__main__":
    main()
