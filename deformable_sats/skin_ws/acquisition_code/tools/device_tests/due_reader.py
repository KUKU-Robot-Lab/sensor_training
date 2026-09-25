import os
import struct
import sys
import time

import serial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from serial_autodetect import resolve_port


DUE_COM_PORT = 'COM11'  # fallback only; auto-detected in main() when possible
DUE_PORT_VID_PID = {(0x2341, 0x003D), (0x2341, 0x003E)}
DUE_PORT_KEYWORDS = ("arduino", "due")
BAUD_RATE = 250000

NUM_SENSORS = 16
FIFO_FRAMES = 10
FRAME_HEADER = 0xAA
FRAME_FOOTER = 0x55
PAYLOAD_SIZE = NUM_SENSORS * FIFO_FRAMES * 4
PACKET_FORMAT = '<' + ('I' * NUM_SENSORS * FIFO_FRAMES)


def payload_to_rows(payload):
    values = struct.unpack(PACKET_FORMAT, payload)
    return [
        [
            values[sensor_i * FIFO_FRAMES + frame_j]
            for sensor_i in range(NUM_SENSORS)
        ]
        for frame_j in range(FIFO_FRAMES)
    ]


def read_burst_rows(ser):
    while True:
        header = ser.read(1)
        if not header:
            return None
        if header[0] == FRAME_HEADER:
            break

    payload = ser.read(PAYLOAD_SIZE)
    if len(payload) != PAYLOAD_SIZE:
        print("DUE Reader: incomplete packet discarded.", file=sys.stderr)
        return None

    footer = ser.read(1)
    if len(footer) != 1:
        print("DUE Reader: incomplete packet footer discarded.", file=sys.stderr)
        return None
    if footer[0] != FRAME_FOOTER:
        print("DUE Reader: malformed packet footer discarded.", file=sys.stderr)
        return None

    return payload_to_rows(payload)


def main():
    ser = None
    port = DUE_COM_PORT
    while True:
        try:
            if ser is None:
                port = resolve_port(
                    "DUE", None, os.environ.get("DUE_PORT"), DUE_COM_PORT,
                    DUE_PORT_VID_PID, DUE_PORT_KEYWORDS,
                )
                print("DUE Reader: connecting to port...", file=sys.stderr)
                ser = serial.Serial(port, BAUD_RATE, timeout=1)
                print("DUE Reader: port connected.", file=sys.stderr)

            rows = read_burst_rows(ser)
            if rows is None:
                continue

            for row in rows:
                print(','.join(str(value) for value in row))
            sys.stdout.flush()

        except serial.SerialException:
            print(f"DUE Reader: port({port}) connection failed. Retrying in 5 seconds...", file=sys.stderr)
            if ser:
                ser.close()
            ser = None
            time.sleep(5)
        except KeyboardInterrupt:
            print("DUE Reader: stopped.", file=sys.stderr)
            break

    if ser and ser.is_open:
        ser.close()


if __name__ == "__main__":
    main()
