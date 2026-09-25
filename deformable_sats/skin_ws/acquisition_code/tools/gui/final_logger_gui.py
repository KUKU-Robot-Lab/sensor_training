import argparse
import json
import os
import struct
import sys
import threading
import time
import csv
from ctypes import *
from datetime import datetime
from queue import Empty, Queue

# Enable ANSI on Windows
if sys.platform == "win32":
    os.system("")

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
    HAS_GUI = True
except ImportError:
    print("Error: pyqtgraph or Qt (PyQt5/PyQt6) not found.")
    HAS_GUI = False

try:
    import can
    HAS_CAN = True
except ImportError:
    HAS_CAN = False

# --- Hardware Constants ---
DLL_PATH = r"C:\Program Files (x86)\PAIX\NMC\DLL\x64\NMC2.dll"
DUE_PORT = "COM11"
DUE_BAUD_RATE = 250000
NUM_SENSORS = 16
FIFO_FRAMES = 10
TARGET_HZ = 200
POLLING_INTERVAL = 1.0 / TARGET_HZ

CAN_INTERFACE = "ixxat"
CAN_CHANNEL = 0
CAN_BITRATE = 1000000
AFD50_FORCE_ID = 0x01A

# --- Global State ---
is_running = True
logging_started = threading.Event()
due_ready = threading.Event()
afd_ready = threading.Event()
log_start_ns = 0
due_queue = Queue()
loadcell_queue = Queue()
afd_queue = Queue()
reader_errors = Queue()

def elapsed_ns():
    return time.perf_counter_ns() - log_start_ns

def payload_to_rows(payload):
    values = struct.unpack("<" + ("I" * NUM_SENSORS * FIFO_FRAMES), payload)
    return [[values[s * FIFO_FRAMES + f] for s in range(NUM_SENSORS)] for f in range(FIFO_FRAMES)]

class NMC_AXES_EXPR(Structure):
    _pack_ = 1
    _fields_ = [
        ("nBusy", c_short * 8), ("nError", c_short * 8), ("nNear", c_short * 8),
        ("nPLimit", c_short * 8), ("nMLimit", c_short * 8), ("nAlarm", c_short * 8),
        ("nEmer", c_short * 2), ("nSwPLimit", c_short * 8), ("nInpo", c_short * 8),
        ("nHome", c_short * 8), ("nEncZ", c_short * 8), ("nOrg", c_short * 8),
        ("nSReady", c_short * 8), ("nContStatus", c_short * 2), ("nDummy", c_short * 6),
        ("nSwMLimit", c_short * 8), ("lEnc", c_int * 8), ("lCmd", c_int * 8),
        ("dEnc", c_double * 8), ("dCmd", c_double * 8), ("dummy", c_char * 4),
    ]

# --- Readers ---
def due_reader():
    try:
        import serial
        ser = serial.Serial(DUE_PORT, DUE_BAUD_RATE, timeout=0.1)
        ser.reset_input_buffer()
        due_ready.set()
        while is_running:
            if not logging_started.wait(timeout=0.05): continue
            header = ser.read(1)
            if not header or header[0] != 0xAA: continue
            payload = ser.read(NUM_SENSORS * FIFO_FRAMES * 4)
            footer = ser.read(1)
            if len(payload) == 640 and footer and footer[0] == 0x55:
                due_queue.put((elapsed_ns(), payload))
        ser.close()
    except Exception as e: reader_errors.put(("DUE", str(e)))

def afd50_reader():
    if not HAS_CAN:
        reader_errors.put(("AFD50", "python-can not installed"))
        return
    try:
        bus = can.interface.Bus(interface=CAN_INTERFACE, channel=CAN_CHANNEL, bitrate=CAN_BITRATE)
        afd_ready.set()
        while is_running:
            if not logging_started.wait(timeout=0.05): continue
            msg = bus.recv(timeout=0.01)
            if msg and msg.arbitration_id == AFD50_FORCE_ID:
                afd_queue.put((elapsed_ns(), bytes(msg.data)))
        bus.shutdown()
    except Exception as e: reader_errors.put(("AFD50", str(e)))

def loadcell_reader():
    try:
        from loadcell_bin_logger import open_serial, DEFAULT_BAUD, DEFAULT_PORT, DEFAULT_READ_SIZE
        class Args: port=DEFAULT_PORT; baud=DEFAULT_BAUD; timeout=0.01; read_size=DEFAULT_READ_SIZE; rx_buffer_size=1024*1024; keep_input_buffer=False
        ser = open_serial(Args)
        if not ser: return
        read_buffer = bytearray(DEFAULT_READ_SIZE)
        while is_running:
            if not logging_started.wait(timeout=0.05): continue
            size = ser.readinto(read_buffer)
            if size: loadcell_queue.put((elapsed_ns(), bytes(read_buffer[:size])))
        ser.close()
    except Exception as e: reader_errors.put(("LOADCELL", str(e)))

class BinaryBuffer:
    def __init__(self, binfile, record_struct, payload_size=0, magic="RAW"):
        self.binfile = binfile
        self.record_struct = record_struct
        self.payload_size = payload_size
        self.offset = 0
        self.binfile.write(f"{magic}\n".encode("ascii"))
        self.buffer = bytearray((record_struct.size + payload_size) * 8192)

    def write(self, ns, payload=None):
        size = self.record_struct.size + (len(payload) if payload else self.payload_size)
        if self.offset + size > len(self.buffer): self.flush()

        struct.pack_into("<Q", self.buffer, self.offset, ns)

        if payload:
            self.buffer[self.offset + 8 : self.offset + 8 + len(payload)] = payload
            self.offset += (8 + len(payload))
        else:
            self.offset += 8

    def flush(self):
        if self.offset: self.binfile.write(self.buffer[:self.offset]); self.offset = 0

if HAS_GUI:
    class DashboardGUI(QtWidgets.QMainWindow):
        def __init__(self, paths, nmc, args):
            super().__init__()
            self.paths = paths
            self.nmc = nmc
            self.args = args
            self.baseline = None
            self.counts = {"DUE": 0, "EM": 0, "LC": 0, "AFD": 0}
            self.last_flush = time.perf_counter()
            self.initUI()

            # DUE
            self.f_due = open(paths["due"], "wb")
            self.due_bin = BinaryBuffer(self.f_due, struct.Struct("<Q"), 640, magic="DUE_V2")

            # Loadcell
            self.f_lc = open(paths["lc"], "wb")
            self.f_lc.write(b"LC_V2\n")

            # EtherMotion
            self.f_em = open(paths["em"], "wb")
            self.em_bin = BinaryBuffer(self.f_em, struct.Struct("<Q"), 32, magic="EM_V2")

            # AFD50
            self.f_afd = open(paths["afd"], "wb")
            self.afd_bin = BinaryBuffer(self.f_afd, struct.Struct("<Q"), 8, magic="AFD_V2")

            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_data)
            self.timer.start(50)

        def initUI(self):
            self.setWindowTitle("Tactile Sensor Dashboard v2.0 (Integrated AFD50)")
            self.resize(1100, 700)
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QVBoxLayout(central)
            status_layout = QtWidgets.QHBoxLayout()
            self.status_lbl = QtWidgets.QLabel("Status: Waiting for Motion...")
            self.status_lbl.setStyleSheet("font-weight: bold; font-size: 18px;")
            self.counts_lbl = QtWidgets.QLabel("DUE: 0 | EM: 0 | LC: 0 | AFD: 0")
            self.counts_lbl.setStyleSheet("font-family: monospace; font-size: 14px;")
            status_layout.addWidget(self.status_lbl); status_layout.addStretch(); status_layout.addWidget(self.counts_lbl)
            layout.addLayout(status_layout)

            grid = QtWidgets.QGridLayout()
            self.sensor_boxes = [None] * 16
            for r in range(4):
                for c in range(4):
                    idx = (r * 4) + c
                    box = QtWidgets.QLabel(f"S{idx+1:02d}\n0.0%")
                    box.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    box.setStyleSheet("border: 2px solid gray; font-size: 20px; background: #f0f0f0; border-radius: 5px;")
                    box.setMinimumSize(120, 90)
                    grid.addWidget(box, r, c)
                    self.sensor_boxes[idx] = box
            layout.addLayout(grid)
            btn_layout = QtWidgets.QHBoxLayout()
            self.zero_btn = QtWidgets.QPushButton("Zero Calibration (R Key)")
            self.zero_btn.clicked.connect(self.reset_baseline); self.zero_btn.setFixedHeight(50)
            btn_layout.addWidget(self.zero_btn); layout.addLayout(btn_layout)

        def reset_baseline(self): self.baseline = None

        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key.Key_R: self.reset_baseline()

        def update_data(self):
            global log_start_ns
            if not logging_started.is_set():
                if self.nmc:
                    axes = NMC_AXES_EXPR()
                    if self.nmc.nmc_GetAxesExpress(self.args.dev_no, byref(axes)) == 0:
                        if any(axes.nBusy[i] != 0 for i in range(4)) or axes.nContStatus[0] != 0:
                            log_start_ns = time.perf_counter_ns()
                            logging_started.set()
                            self.status_lbl.setText("Status: LOGGING"); self.status_lbl.setStyleSheet("color: red; font-weight: bold; font-size: 18px;")
                return

            # 1. EM
            if self.nmc:
                axes = NMC_AXES_EXPR()
                if self.nmc.nmc_GetAxesExpress(self.args.dev_no, byref(axes)) == 0:
                    payload = struct.pack("<dddd", axes.dCmd[0], axes.dCmd[1], axes.dCmd[2], axes.dCmd[3])
                    self.em_bin.write(elapsed_ns(), payload)
                    self.counts["EM"] += 1

            # 2. DUE
            latest_row = None
            while True:
                try:
                    ns, payload = due_queue.get_nowait()
                    self.due_bin.write(ns, payload)
                    rows = payload_to_rows(payload)
                    if self.baseline is None:
                        if not hasattr(self, 'baseline_init_count'): self.baseline_init_count = 0
                        self.baseline_init_count += 1
                        if self.baseline_init_count == 2:
                            self.baseline = [sum(col)/len(col) for col in zip(*rows)]
                            self.status_lbl.setText("Status: Baseline Set")
                    latest_row = rows[-1]; self.counts["DUE"] += 1
                except Empty: break

            # 3. LC
            while True:
                try:
                    ns, payload = loadcell_queue.get_nowait()
                    self.f_lc.write(struct.pack("<QI", ns, len(payload)) + payload)
                    self.counts["LC"] += 1
                except Empty: break

            # 4. AFD50
            while True:
                try:
                    ns, payload = afd_queue.get_nowait()
                    self.afd_bin.write(ns, payload)
                    self.counts["AFD"] += 1
                except Empty: break

            self.counts_lbl.setText(f"DUE: {self.counts['DUE']:6d} | EM: {self.counts['EM']:6d} | LC: {self.counts['LC']:4d} | AFD: {self.counts['AFD']:6d}")
            now = time.perf_counter()
            if now - self.last_flush >= 1.0:
                self.flush_logs()
                self.last_flush = now

            if latest_row and self.baseline:
                for i in range(16):
                    base_val = self.baseline[i] if self.baseline[i] != 0 else 1
                    percent = ((latest_row[i] - base_val) / base_val) * 100.0
                    self.sensor_boxes[i].setText(f"S{i+1:02d}\n{percent:+.1f}%")
                    val = min(255, int(abs(percent) * 10))
                    if percent > 1.0: self.sensor_boxes[i].setStyleSheet(f"background-color: rgb(255, {255-val}, {255-val}); border: 2px solid red; font-size: 20px;")
                    elif percent < -1.0: self.sensor_boxes[i].setStyleSheet(f"background-color: rgb({255-val}, {255-val}, 255); border: 2px solid blue; font-size: 20px;")
                    else: self.sensor_boxes[i].setStyleSheet("background-color: #f0f0f0; border: 2px solid gray; font-size: 20px;")

        def flush_logs(self):
            self.due_bin.flush()
            self.em_bin.flush()
            self.afd_bin.flush()
            self.f_due.flush()
            self.f_lc.flush()
            self.f_em.flush()
            self.f_afd.flush()

        def closeEvent(self, event):
            global is_running
            is_running = False
            self.flush_logs()
            self.f_due.close(); self.f_lc.close(); self.f_em.close(); self.f_afd.close()
            event.accept()

def _next_session_dir(raw_data_dir):
    import re
    date_prefix = datetime.now().strftime("%Y%m%d")
    os.makedirs(raw_data_dir, exist_ok=True)
    pattern = re.compile(rf"^gui_{re.escape(date_prefix)}_test(\d+)$")
    used = []
    for entry in os.scandir(raw_data_dir):
        if entry.is_dir():
            m = pattern.match(entry.name)
            if m:
                used.append(int(m.group(1)))
    next_n = max(used, default=0) + 1
    session_dir = os.path.join(raw_data_dir, f"gui_{date_prefix}_test{next_n}")
    os.makedirs(session_dir)
    return session_dir


def main():
    if not HAS_GUI: return
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-no", type=int, default=11)
    args = parser.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base, "raw_data")
    session_dir = _next_session_dir(raw_data_dir)
    paths = {
        "due": os.path.join(session_dir, f"due_v2_{ts}.bin"),
        "em":  os.path.join(session_dir, f"em_v2_{ts}.bin"),
        "lc":  os.path.join(session_dir, f"lc_v2_{ts}.bin"),
        "afd": os.path.join(session_dir, f"afd_v2_{ts}.bin"),
    }
    nmc = None
    if os.path.exists(DLL_PATH):
        try:
            nmc = WinDLL(DLL_PATH)
            nmc.nmc_OpenDeviceEx.argtypes = [c_short, c_int]
            nmc.nmc_GetAxesExpress.argtypes = [c_short, POINTER(NMC_AXES_EXPR)]
            nmc.nmc_OpenDeviceEx(args.dev_no, 100)
        except: pass
    threading.Thread(target=due_reader, daemon=True).start()
    threading.Thread(target=loadcell_reader, daemon=True).start()
    threading.Thread(target=afd50_reader, daemon=True).start()
    app = QtWidgets.QApplication(sys.argv)
    gui = DashboardGUI(paths, nmc, args)
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
