#!/usr/bin/env python3
import can
import time
import sys
import statistics
import signal

# =========================================================
# AFD50-D15 sensor spec
# =========================================================
DATA_ID = 0x1A              # default CAN ID
CAL_SAMPLES = 1000          # number of samples for bias calibration
CONV_DIV = 300.0            # datasheet: Force[N] = raw/300 – 100
CONV_OFF = 100.0
RATE_HZ = 100               # 100 Hz update
BUS_TYPE = "ixxat"      # change to 'pcan' etc. if needed
CHANNEL = 0
BITRATE = 1000000

# =========================================================
# utility
# =========================================================
def to_u16(buf, i):
    return (buf[2*i] << 8) | buf[2*i + 1]

def force_from_raw(raw):
    return raw / CONV_DIV - CONV_OFF

# =========================================================
# graceful exit
# =========================================================
stop_flag = False
def signal_handler(sig, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, signal_handler)

# =========================================================
# CAN open
# =========================================================
bus = can.interface.Bus(channel=CHANNEL, bustype=BUS_TYPE, bitrate=BITRATE)
print("[AFD50] CAN opened.", file=sys.stderr)

# =========================================================
# bias calibration
# =========================================================
fx_samples, fy_samples, fz_samples = [], [], []
print("[AFD50] Starting bias calibration...", file=sys.stderr)

while len(fx_samples) < CAL_SAMPLES and not stop_flag:
    msg = bus.recv(timeout=1)
    if msg is None or msg.arbitration_id != DATA_ID:
        continue

    raw_fx = to_u16(msg.data, 0)
    raw_fy = to_u16(msg.data, 1)
    raw_fz = to_u16(msg.data, 2)

    fx = force_from_raw(raw_fx)
    fy = force_from_raw(raw_fy)
    fz = force_from_raw(raw_fz)

    fx_samples.append(fx)
    fy_samples.append(fy)
    fz_samples.append(fz)

    if len(fx_samples) <= 10:
        print(f"DBG raw=({raw_fx},{raw_fy},{raw_fz}) "
              f"F=({fx:.2f},{fy:.2f},{fz:.2f})", file=sys.stderr)

bias_fx = statistics.mean(fx_samples)
bias_fy = statistics.mean(fy_samples)
bias_fz = statistics.mean(fz_samples)
std_fx = statistics.pstdev(fx_samples)
std_fy = statistics.pstdev(fy_samples)
std_fz = statistics.pstdev(fz_samples)

print(f"[AFD50] Bias calibration complete. "
      f"Fx={bias_fx:.2f}±{std_fx:.2f}, "
      f"Fy={bias_fy:.2f}±{std_fy:.2f}, "
      f"Fz={bias_fz:.2f}±{std_fz:.2f}",
      file=sys.stderr)
print("AFD50_READY", flush=True)

# =========================================================
# continuous stream
# =========================================================
start_time = time.time()
print("[AFD50] Streaming (bias removed)...", file=sys.stderr)

while not stop_flag:
    msg = bus.recv(timeout=1)
    if msg is None or msg.arbitration_id != DATA_ID:
        continue

    raw_fx = to_u16(msg.data, 0)
    raw_fy = to_u16(msg.data, 1)
    raw_fz = to_u16(msg.data, 2)

    fx = force_from_raw(raw_fx) - bias_fx
    fy = force_from_raw(raw_fy) - bias_fy
    fz = force_from_raw(raw_fz) - bias_fz

    ##print(f"{time.time()-start_time:8.3f}s "
   ##     f"Fx={fx:+7.2f} Fy={fy:+7.2f} Fz={fz:+7.2f}")

print("\n[AFD50] stopped.", file=sys.stderr)
