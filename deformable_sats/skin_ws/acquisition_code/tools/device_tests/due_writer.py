import csv
import os
import struct
import sys
import time
from datetime import datetime

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
        print("DUE Writer: incomplete packet discarded.", file=sys.stderr)
        return None

    footer = ser.read(1)
    if len(footer) != 1:
        print("DUE Writer: incomplete packet footer discarded.", file=sys.stderr)
        return None
    if footer[0] != FRAME_FOOTER:
        print("DUE Writer: malformed packet footer discarded.", file=sys.stderr)
        return None

    return payload_to_rows(payload)


def timestamp_rows(rows, packet_time):
    return [
        [packet_time] + row
        for row in rows
    ]


def main():
    ser = None
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    skin_ws_dir = os.path.dirname(current_script_dir)
    save_dir = os.path.join(skin_ws_dir, "due data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f"due_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['Timestamp'] + [f's{i}' for i in range(1, NUM_SENSORS + 1)]
        writer.writerow(headers)
        csvfile.flush()
        print(f"DUE Writer: writing to {filepath}.", file=sys.stderr)

        while True:
            try:
                if ser is None:
                    port = resolve_port(
                        "DUE", None, os.environ.get("DUE_PORT"), DUE_COM_PORT,
                        DUE_PORT_VID_PID, DUE_PORT_KEYWORDS,
                    )
                    print("DUE Writer: connecting to port...", file=sys.stderr)
                    ser = serial.Serial(port, BAUD_RATE, timeout=1)
                    print("DUE Writer: port connected.", file=sys.stderr)

                rows = read_burst_rows(ser)
                if rows is None:
                    continue

                writer.writerows(timestamp_rows(rows, time.time()))
                csvfile.flush()

            except serial.SerialException:
                print("DUE Writer: connection failed. Retrying in 5 seconds...", file=sys.stderr)
                if ser:
                    ser.close()
                ser = None
                time.sleep(5)
            except KeyboardInterrupt:
                break

    if ser and ser.is_open:
        ser.close()


if __name__ == "__main__":
    main()
