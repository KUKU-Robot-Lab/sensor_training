import sys
import time

import can


CAN_INTERFACE = "ixxat"
CAN_CHANNEL = 0
CAN_BITRATE = 1000000
FORCE_ID = 0x01A


def connect_bus():
    while True:
        try:
            print("AFD50 Reader: connecting CAN bus...", file=sys.stderr)
            bus = can.interface.Bus(
                interface=CAN_INTERFACE,
                channel=CAN_CHANNEL,
                bitrate=CAN_BITRATE,
            )
            print("AFD50 Reader: CAN bus connected.", file=sys.stderr)
            return bus
        except can.CanError as exc:
            print(f"AFD50 Reader: CAN bus connection failed: {exc}; retrying in 5 seconds.", file=sys.stderr)
            time.sleep(5)


def main():
    bus = connect_bus()
    start_ns = time.perf_counter_ns()
    print("AFD50_READY")
    sys.stdout.flush()
    print("AFD50 Reader: streaming raw CAN frames; no physical conversion is applied.", file=sys.stderr)

    try:
        while True:
            try:
                msg = bus.recv(timeout=0.1)
            except can.CanError:
                print("AFD50 Reader: CAN receive error.", file=sys.stderr)
                time.sleep(2)
                continue

            if msg is None or msg.arbitration_id != FORCE_ID:
                continue

            elapsed_ns = time.perf_counter_ns() - start_ns
            raw_data = bytes(msg.data)
            print(f"elapsed_ns:{elapsed_ns},id:0x{msg.arbitration_id:03X},data:{raw_data.hex()}")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("AFD50 Reader: stop requested.", file=sys.stderr)
    finally:
        if bus is not None:
            bus.shutdown()
        print("AFD50 Reader: stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
