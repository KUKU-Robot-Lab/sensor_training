import can
import serial
import time
import threading
import csv
from datetime import datetime
import os
from collections import deque
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# --- 설정 --- #
MOTOR_SERIAL_PORT = 'COM14'
SENSOR_SERIAL_PORT = 'COM13'
BAUD_RATE = 460800

CAN_INTERFACE = 'ixxat'
CAN_CHANNEL = 0
CAN_BITRATE = 1000000
FORCE_ID, TORQUE_ID = 0x01A, 0x01B

# --- 자동화 및 안전 정지 설정 ---
FORCE_LIMIT = 50.0
TORQUE_LIMIT = 0.35
FULL_TEST_SEQUENCE = {
    'F': [{'axis': 'X'}, {'axis': 'Y'}, {'axis': 'Z'}],
    'T': [{'axis': 'X'}, {'axis': 'Y'}, {'axis': 'Z'}]
}

# --- 데이터 수집 및 상태 관리를 위한 공유 변수 ---
data_lock = threading.Lock()
global_data = {
    "timestamp": None,
    "afd_fx": 0.0, "afd_fy": 0.0, "afd_fz": 0.0,
    "afd_tx": 0.0, "afd_ty": 0.0, "afd_tz": 0.0,
    "vensor_raw": [0]*8,
    "is_running": True,
    "stop_sequence": False
}

# --- 플로팅을 위한 데이터 저장소 ---
MAX_DATA_POINTS = 100  # 플롯에 표시할 최대 데이터 포인트 수
plot_data = {
    'time': deque(maxlen=MAX_DATA_POINTS),
    'fx': deque(maxlen=MAX_DATA_POINTS), 'fy': deque(maxlen=MAX_DATA_POINTS), 'fz': deque(maxlen=MAX_DATA_POINTS),
    'tx': deque(maxlen=MAX_DATA_POINTS), 'ty': deque(maxlen=MAX_DATA_POINTS), 'tz': deque(maxlen=MAX_DATA_POINTS),
    'p': [deque(maxlen=MAX_DATA_POINTS) for _ in range(8)]
}

# --- 데이터 로깅 클래스 ---
class DataLogger:
    def __init__(self):
        self.file = None
        self.writer = None
        self.is_logging = False
        self.thread = None

    def start_logging(self, filename):
        self.stop_logging()
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Timestamp", "AFD_Fx", "AFD_Fy", "AFD_Fz", "AFD_Tx", "AFD_Ty", "AFD_Tz",
                               "VENSOR_P1", "VENSOR_P2", "VENSOR_P3", "VENSOR_P4",
                               "VENSOR_P5", "VENSOR_P6", "VENSOR_P7", "VENSOR_P8"])
        self.is_logging = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"로깅 시작: {filename}")

    def _run(self):
        while self.is_logging:
            with data_lock:
                if global_data["timestamp"]:
                    row_data = [
                        global_data["timestamp"],
                        global_data["afd_fx"], global_data["afd_fy"], global_data["afd_fz"],
                        global_data["afd_tx"], global_data["afd_ty"], global_data["afd_tz"],
                    ] + global_data["vensor_raw"]
                    self.writer.writerow(row_data)
            time.sleep(0.005)

    def stop_logging(self):
        if self.is_logging:
            self.is_logging = False
            if self.thread:
                self.thread.join(timeout=1)
            if self.file:
                self.file.close()
                print(f"로깅 중지 및 파일 저장 완료.")
            self.file = None
            self.writer = None
            self.thread = None

# --- 실시간 플로팅 클래스 ---
class RealTimePlotter:
    def __init__(self):
        if not MATPLOTLIB_AVAILABLE:
            return
        self.fig = plt.figure(figsize=(15, 8), tight_layout=True)
        gs = self.fig.add_gridspec(2, 2)
        self.ax_force = self.fig.add_subplot(gs[0, 0])
        self.ax_torque = self.fig.add_subplot(gs[1, 0])
        self.ax_pressure = self.fig.add_subplot(gs[:, 1])
        self.ani = None

    def setup_plots(self):
        if not MATPLOTLIB_AVAILABLE:
            return
        # Force Plot
        self.ax_force.set_title("Force Data (Fx, Fy, Fz)")
        self.ax_force.set_ylabel("Force (N)")
        self.force_lines = [self.ax_force.plot([], [], label=label)[0] for label in ['Fx', 'Fy', 'Fz']]
        self.ax_force.legend()
        self.ax_force.grid(True)

        # Torque Plot
        self.ax_torque.set_title("Torque Data (Tx, Ty, Tz)")
        self.ax_torque.set_xlabel("Time")
        self.ax_torque.set_ylabel("Torque (Nm)")
        self.torque_lines = [self.ax_torque.plot([], [], label=label)[0] for label in ['Tx', 'Ty', 'Tz']]
        self.ax_torque.legend()
        self.ax_torque.grid(True)

        # Pressure Plot
        self.ax_pressure.set_title("Vensor Pressure Data (P1-P8)")
        self.ax_pressure.set_xlabel("Time")
        self.ax_pressure.set_ylabel("Raw Value")
        self.pressure_lines = [self.ax_pressure.plot([], [], label=f'P{i+1}')[0] for i in range(8)]
        self.ax_pressure.legend(loc='upper left', bbox_to_anchor=(1, 1))
        self.ax_pressure.grid(True)

    def _update(self, frame):
        if not MATPLOTLIB_AVAILABLE:
            return []

        times = list(plot_data['time'])
        if not times or len(times) < 2:
            return self.force_lines + self.torque_lines + self.pressure_lines

        # Force
        self.ax_force.relim()
        self.ax_force.autoscale_view()
        for i, key in enumerate(['fx', 'fy', 'fz']):
            self.force_lines[i].set_data(times, list(plot_data[key]))

        # Torque
        self.ax_torque.relim()
        self.ax_torque.autoscale_view()
        for i, key in enumerate(['tx', 'ty', 'tz']):
            self.torque_lines[i].set_data(times, list(plot_data[key]))

        # Pressure
        self.ax_pressure.relim()
        self.ax_pressure.autoscale_view()
        for i in range(8):
            self.pressure_lines[i].set_data(times, list(plot_data['p'][i]))
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return self.force_lines + self.torque_lines + self.pressure_lines

    def start(self):
        if not MATPLOTLIB_AVAILABLE:
            print("\n경고: matplotlib 라이브러리를 찾을 수 없습니다. 플로팅 기능이 비활성화됩니다.")
            print("실시간 플로팅을 사용하려면 'pip install matplotlib'을 실행하세요.\n")
            return
        self.ani = FuncAnimation(self.fig, self._update, blit=False, interval=500, cache_frame_data=False)
        plt.show(block=False)
        plt.pause(0.1)

    def stop(self):
        if not MATPLOTLIB_AVAILABLE or self.ani is None:
            return
        plt.close(self.fig)

# --- 센서 리더 스레드 ---
class Afd50Reader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.bus = can.interface.Bus(bustype=CAN_INTERFACE, channel=CAN_CHANNEL, bitrate=CAN_BITRATE)
        self.bias_f = [0.0]*3; self.bias_t = [0.0]*3
        self.cnt_f = 0; self.cnt_t = 0
        self.done_f = False; self.done_t = False
        self.CAL_SAMPLES = 100
        self.calibration_done_event = threading.Event()

    def run(self):
        print("AFD50 바이어스 보정을 시작합니다...")
        while not (self.done_f and self.done_t) and global_data["is_running"]:
            msg = self.bus.recv(timeout=0.1)
            if msg is None: continue
            if msg.arbitration_id == FORCE_ID and not self.done_f:
                r = [self.to_u16(msg.data, i) for i in range(3)]
                f = [self.conv_f(val) for val in r]
                for i in range(3): self.bias_f[i] += f[i]
                self.cnt_f += 1
                if self.cnt_f >= self.CAL_SAMPLES:
                    for i in range(3): self.bias_f[i] /= self.cnt_f
                    self.done_f = True
                    print(">>> Force bias 계산 완료.")
            elif msg.arbitration_id == TORQUE_ID and not self.done_t:
                r = [self.to_u16(msg.data, i) for i in range(3)]
                t = [self.conv_t(val) for val in r]
                for i in range(3): self.bias_t[i] += t[i]
                self.cnt_t += 1
                if self.cnt_t >= self.CAL_SAMPLES:
                    for i in range(3): self.bias_t[i] /= self.cnt_t
                    self.done_t = True
                    print(">>> Torque bias 계산 완료.")
        if global_data["is_running"]:
            print("*** AFD50 바이어스 보정 완료! ***")
            self.calibration_done_event.set()
            
        while global_data["is_running"]:
            msg = self.bus.recv(timeout=0.1)
            if msg is None: continue
            with data_lock:
                global_data["timestamp"] = datetime.now().isoformat()
                if msg.arbitration_id == FORCE_ID:
                    r = [self.to_u16(msg.data, i) for i in range(3)]
                    f = [self.conv_f(val) for val in r]
                    global_data["afd_fx"] = f[0] - self.bias_f[0]
                    global_data["afd_fy"] = f[1] - self.bias_f[1]
                    global_data["afd_fz"] = -(f[2] - self.bias_f[2])
                elif msg.arbitration_id == TORQUE_ID:
                    r = [self.to_u16(msg.data, i) for i in range(3)]
                    t = [self.conv_t(val) for val in r]
                    global_data["afd_tx"] = t[0] - self.bias_t[0]
                    global_data["afd_ty"] = t[1] - self.bias_t[1]
                    global_data["afd_tz"] = t[2] - self.bias_t[2]
                
                current_time = datetime.now()
                plot_data['time'].append(current_time)
                plot_data['fx'].append(global_data["afd_fx"])
                plot_data['fy'].append(global_data["afd_fy"])
                plot_data['fz'].append(global_data["afd_fz"])
                plot_data['tx'].append(global_data["afd_tx"])
                plot_data['ty'].append(global_data["afd_ty"])
                plot_data['tz'].append(global_data["afd_tz"])

    def stop(self): self.bus.shutdown()
    def to_u16(self, buf, idx): return (buf[2 * idx] << 8) | buf[2 * idx + 1]
    def conv_f(self, raw): return raw / 300.0 - 100.0
    def conv_t(self, raw): return raw / 50000.0 - 0.6

class VensorReader(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)

    def run(self):
        while global_data["is_running"]:
            try:
                line = self.ser.readline().decode('ascii').strip()
                if line:
                    parts = [int(p) for p in line.split(',')]
                    if len(parts) == 8:
                        with data_lock:
                            global_data["vensor_raw"] = parts
                            for i in range(8):
                                plot_data['p'][i].append(parts[i])
            except (UnicodeDecodeError, ValueError):
                continue
    def stop(self): self.ser.close()

# --- 자동 측정 시퀀스 실행 함수 ---
def run_single_direction_test(motor_ser, axis_to_test, direction, data_logger, sensor_type, test_type):
    dir_str = "pos" if direction > 0 else "neg"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"Vensor-{sensor_type}_{test_type}_{axis_to_test}_{dir_str}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    output_filename = os.path.join(output_dir, filename)
    data_logger.start_logging(output_filename)

    steps_taken = 0
    try:
        print("\n" + "-" * 50)
        print(f"측정 시작: {sensor_type} 센서, {axis_to_test}축, {dir_str.upper()} 방향으로 이동")
        print(f"안전 한계: Force > {FORCE_LIMIT}N 또는 Torque > {TORQUE_LIMIT}Nm 시 정지")
        print("-" * 50)

        step_increment = 20 * direction
        move_delay = 800

        while not global_data["stop_sequence"]:
            motor_cmd = f"M,{axis_to_test},{step_increment},{move_delay}\n"
            motor_ser.write(motor_cmd.encode())
            steps_taken += step_increment
            time.sleep(0.05)

            with data_lock:
                fx, fy, fz = global_data["afd_fx"], global_data["afd_fy"], global_data["afd_fz"]
                tx, ty, tz = global_data["afd_tx"], global_data["afd_ty"], global_data["afd_tz"]
                vensor_values = global_data["vensor_raw"]

            if 0 in vensor_values:
                print("경고: VENSOR 센서 값 0 감지! 측정 중단.")
                break

            if abs(fx) > FORCE_LIMIT or abs(fy) > FORCE_LIMIT or abs(fz) > FORCE_LIMIT:
                print(f"안전 정지: 힘 한계({FORCE_LIMIT}N) 초과! (Fx: {fx:.2f}, Fy: {fy:.2f}, Fz: {fz:.2f})")
                break
            
            if abs(tx) > TORQUE_LIMIT or abs(ty) > TORQUE_LIMIT or abs(tz) > TORQUE_LIMIT:
                print(f"안전 정지: 토크 한계({TORQUE_LIMIT}Nm) 초과! (Tx: {tx:.2f}, Ty: {ty:.2f}, Tz: {tz:.2f})")
                break

    finally:
        data_logger.stop_logging()
        print("원점 복귀를 시작합니다...")
        return_cmd = f"M,{axis_to_test},{-steps_taken},{move_delay}\n"
        motor_ser.write(return_cmd.encode())
        return_duration_s = (abs(steps_taken) * (move_delay / 1000000.0)) + 2.0
        print(f"복귀에 약 {return_duration_s:.2f}초 소요 예상. 잠시 대기합니다.")
        time.sleep(return_duration_s)
        print("원점 복귀 완료.")
        print("-" * 50)

# --- 메인 제어 로직 ---
def main():
    while True:
        sensor_type = input("테스트할 센서 종류를 입력하세요 (FT 또는 S): ").upper()
        if sensor_type in ['FT', 'S']:
            break
        else:
            print(">> 잘못된 입력입니다. 'FT' 또는 'S'를 입력해주세요.")

    while True:
        axes_input = input("테스트할 축을 입력하세요 (예: X, XY, YZ, XYZ): ").upper()
        selected_axes = sorted(list(set([char for char in axes_input if char in ['X', 'Y', 'Z']])))
        if selected_axes:
            print(f">> 선택된 축: {', '.join(selected_axes)}")
            break
        else:
            print(">> 잘못된 입력입니다. X, Y, Z 중에서 하나 이상을 입력해주세요.")

    test_sequence = []
    for test_type_key, tests in FULL_TEST_SEQUENCE.items():
        for test in tests:
            if test['axis'] in selected_axes:
                new_test = test.copy()
                new_test['test_type'] = test_type_key
                test_sequence.append(new_test)

    if not test_sequence:
        print("실행할 테스트가 없습니다. 프로그램을 종료합니다.")
        return

    plotter = RealTimePlotter()
    plotter.setup_plots()
    
    try:
        motor_ser = serial.Serial(MOTOR_SERIAL_PORT, 115200, timeout=1)
        time.sleep(2)
        print(f"모터 제어 포트({MOTOR_SERIAL_PORT})에 연결되었습니다.")
    except serial.SerialException as e:
        print(f"모터 제어 포트({MOTOR_SERIAL_PORT}) 연결 실패: {e}")
        return

    afd50_reader = Afd50Reader()
    vensor_reader = VensorReader(SENSOR_SERIAL_PORT)
    data_logger = DataLogger()

    try:
        plotter.start()
        afd50_reader.start()
        vensor_reader.start()

        print("센서 스레드 시작. AFD50 바이어스 보정을 기다립니다...")
        if not afd50_reader.calibration_done_event.wait(timeout=30):
            print("오류: AFD50 바이어스 보정 시간 초과. 프로그램을 종료합니다.")
            raise SystemExit

        print("자동 보정 시퀀스를 시작합니다. (5초 후)")
        time.sleep(5)

        for test_params in test_sequence:
            if global_data["stop_sequence"]:
                break
            # 정방향 (+)
            run_single_direction_test(motor_ser, test_params['axis'], 1, data_logger, sensor_type, test_params['test_type'])
            if global_data["stop_sequence"]:
                break
            time.sleep(2)
            
            # 역방향 (-)
            run_single_direction_test(motor_ser, test_params['axis'], -1, data_logger, sensor_type, test_params['test_type'])
            if global_data["stop_sequence"]:
                break
            time.sleep(2)

    except (KeyboardInterrupt, SystemExit):
        print("\n프로그램이 중단되었습니다.")
    finally:
        print("모든 측정을 종료하고 장치를 정지합니다.")
        global_data["is_running"] = False
        if 'motor_ser' in locals() and motor_ser.is_open:
            motor_ser.write(b'S\n')
        
        data_logger.stop_logging()
        if afd50_reader.is_alive(): afd50_reader.join(timeout=1)
        if vensor_reader.is_alive(): vensor_reader.join(timeout=1)
        
        afd50_reader.stop()
        vensor_reader.stop()
        if 'motor_ser' in locals() and motor_ser.is_open:
            motor_ser.close()
        
        plotter.stop()
        print("모든 리소스가 정리되었습니다. 프로그램 종료.")

if __name__ == "__main__":
    main()