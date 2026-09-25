import can
import serial
import time
import threading
import csv
from datetime import datetime
import os

# --- 설정 --- #
MOTOR_SERIAL_PORT = 'COM14'
SENSOR_SERIAL_PORT = 'COM13'
BAUD_RATE = 460800

CAN_INTERFACE = 'ixxat'
CAN_CHANNEL = 0
CAN_BITRATE = 1000000
FORCE_ID, TORQUE_ID = 0x01A, 0x01B

# --- 모터 및 안전 설정 ---
STEPS_PER_MM = 640
MAX_TRAVEL_MM = 27.0
MAX_TRAVEL_STEPS = int(MAX_TRAVEL_MM * STEPS_PER_MM)

FORCE_LIMIT = 50.0
TORQUE_LIMIT = 0.35

# --- 데이터 수집 및 상태 관리를 위한 공유 변수 ---
data_lock = threading.Lock()
global_data = {
    "timestamp": None,
    "afd_fx": 0.0, "afd_fy": 0.0, "afd_fz": 0.0,
    "afd_tx": 0.0, "afd_ty": 0.0, "afd_tz": 0.0,
    "vensor_raw": [0]*8,
    "is_running": True,
    "stop_sequence": False,
    "test_phase": "Idle"
}

# --- 데이터 로깅 클래스 ---
class DataLogger(threading.Thread):
    def __init__(self, filename):
        super().__init__(daemon=True)
        self.filename = filename
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Timestamp", "Phase", "AFD_Fx", "AFD_Fy", "AFD_Fz", "AFD_Tx", "AFD_Ty", "AFD_Tz",
                               "VENSOR_P1", "VENSOR_P2", "VENSOR_P3", "VENSOR_P4",
                               "VENSOR_P5", "VENSOR_P6", "VENSOR_P7", "VENSOR_P8"])

    def run(self):
        while global_data["is_running"]:
            with data_lock:
                # 원점 복귀 중에는 기록하지 않음
                if global_data["timestamp"] and "returning" not in global_data["test_phase"]:
                    row_data = [
                        global_data["timestamp"],
                        global_data["test_phase"],
                        global_data["afd_fx"], global_data["afd_fy"], global_data["afd_fz"],
                        global_data["afd_tx"], global_data["afd_ty"], global_data["afd_tz"],
                    ] + global_data["vensor_raw"]
                    self.writer.writerow(row_data)
            time.sleep(0.005)

    def stop(self):
        if self.file:
            self.file.close()
            print(f"\n데이터가 '{self.filename}'에 최종 저장되었습니다.")

# --- 실시간 데이터 표시 스레드 ---
class LiveDisplay(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)

    def run(self):
        while global_data["is_running"]:
            os.system('cls' if os.name == 'nt' else 'clear')
            with data_lock:
                fx, fy, fz = global_data["afd_fx"], global_data["afd_fy"], global_data["afd_fz"]
                tx, ty, tz = global_data["afd_tx"], global_data["afd_ty"], global_data["afd_tz"]
                vensor = global_data["vensor_raw"]
                phase = global_data["test_phase"]
            
            print("--- 실시간 센서 데이터 ---")
            print(f"테스트 단계: {phase}")
            print("-" * 28)
            print(f" AFD Force (N)  : Fx={fx: >7.2f}, Fy={fy: >7.2f}, Fz={fz: >7.2f}")
            print(f" AFD Torque (Nm): Tx={tx: >7.2f}, Ty={ty: >7.2f}, Tz={tz: >7.2f}")
            print("-" * 28)
            print(" VENSOR Raw Data:")
            print(f" P1-P4: {vensor[0]: >5}, {vensor[1]: >5}, {vensor[2]: >5}, {vensor[3]: >5}")
            print(f" P5-P8: {vensor[4]: >5}, {vensor[5]: >5}, {vensor[6]: >5}, {vensor[7]: >5}")
            print("---------------------------")
            print("Ctrl+C를 눌러 언제든지 측정을 중단할 수 있습니다.")
            time.sleep(1)

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
            elif msg.arbitration_id == TORQUE_ID and not self.done_t:
                r = [self.to_u16(msg.data, i) for i in range(3)]
                t = [self.conv_t(val) for val in r]
                for i in range(3): self.bias_t[i] += t[i]
                self.cnt_t += 1
                if self.cnt_t >= self.CAL_SAMPLES:
                    for i in range(3): self.bias_t[i] /= self.cnt_t
                    self.done_t = True
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
            except (UnicodeDecodeError, ValueError):
                continue
    def stop(self): self.ser.close()

# --- 자동 측정 시퀀스 실행 함수 ---
def run_single_direction_test(motor_ser, axis_to_test, direction):
    dir_str = "pos" if direction > 0 else "neg"
    phase_name = f"{axis_to_test}_{dir_str}"
    with data_lock:
        global_data["test_phase"] = phase_name

    steps_taken = 0
    try:
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

            # Z축을 제외하고 이동 거리 제한 확인
            if axis_to_test != 'Z' and abs(steps_taken) > MAX_TRAVEL_STEPS:
                print(f"\n경고: 최대 이동 거리({MAX_TRAVEL_MM}mm) 도달! 측정 중단.")
                break

            if 0 in vensor_values:
                print("\n경고: VENSOR 센서 값 0 감지! 측정 중단.")
                break

            if abs(fx) > FORCE_LIMIT or abs(fy) > FORCE_LIMIT or abs(fz) > FORCE_LIMIT:
                print(f"\n안전 정지: 힘 한계({FORCE_LIMIT}N) 초과!")
                break
            
            if abs(tx) > TORQUE_LIMIT or abs(ty) > TORQUE_LIMIT or abs(tz) > TORQUE_LIMIT:
                print(f"\n안전 정지: 토크 한계({TORQUE_LIMIT}Nm) 초과!")
                break

    finally:
        with data_lock:
            global_data["test_phase"] = f"{axis_to_test}_returning"
        
        return_cmd = f"M,{axis_to_test},{-steps_taken},{move_delay}\n"
        motor_ser.write(return_cmd.encode())
        return_duration_s = (abs(steps_taken) * (move_delay / 1000000.0)) + 2.0
        time.sleep(return_duration_s)
        with data_lock:
            global_data["test_phase"] = "Idle"

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

    while True:
        direction_input = input("테스트할 방향을 입력하세요 (+: 정방향, -: 역방향, *: 양방향): ")
        if direction_input in ['+', '-', '*']:
            break
        else:
            print(">> 잘못된 입력입니다. '+, -, *' 중에서 입력해주세요.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"Vensor-{sensor_type}_{''.join(selected_axes)}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    output_filename = os.path.join(output_dir, filename)
    
    try:
        motor_ser = serial.Serial(MOTOR_SERIAL_PORT, 115200, timeout=1)
        time.sleep(2)
        print(f"모터 제어 포트({MOTOR_SERIAL_PORT})에 연결되었습니다.")
    except serial.SerialException as e:
        print(f"모터 제어 포트({MOTOR_SERIAL_PORT}) 연결 실패: {e}")
        return

    afd50_reader = Afd50Reader()
    vensor_reader = VensorReader(SENSOR_SERIAL_PORT)
    data_logger = DataLogger(output_filename)
    live_display = LiveDisplay()

    try:
        afd50_reader.start()
        vensor_reader.start()

        print("센서 스레드 시작. AFD50 바이어스 보정을 기다립니다...")
        if not afd50_reader.calibration_done_event.wait(timeout=30):
            print("오류: AFD50 바이어스 보정 시간 초과. 프로그램을 종료합니다.")
            raise SystemExit
        
        data_logger.start()
        live_display.start()

        print("자동 보정 시퀀스를 시작합니다. (5초 후)")
        time.sleep(5)

        for axis in selected_axes:
            if global_data["stop_sequence"]:
                break

            directions_to_run = []
            if direction_input == '+':
                directions_to_run.append(1)
            elif direction_input == '-':
                directions_to_run.append(-1)
            elif direction_input == '*':
                directions_to_run.extend([1, -1])

            for direction in directions_to_run:
                run_single_direction_test(motor_ser, axis, direction)
                if global_data["stop_sequence"]:
                    break
                time.sleep(2)
            if global_data["stop_sequence"]:
                break

    except (KeyboardInterrupt, SystemExit):
        print("\n프로그램이 중단되었습니다.")
    finally:
        print("\n모든 측정을 종료하고 장치를 정지합니다...")
        global_data["is_running"] = False
        if 'motor_ser' in locals() and motor_ser.is_open:
            motor_ser.write(b'S\n')
        
        if afd50_reader.is_alive(): afd50_reader.join(timeout=1)
        if vensor_reader.is_alive(): vensor_reader.join(timeout=1)
        if data_logger.is_alive(): data_logger.join(timeout=1)
        
        afd50_reader.stop()
        vensor_reader.stop()
        data_logger.stop()
        if 'motor_ser' in locals() and motor_ser.is_open:
            motor_ser.close()
        
        print("모든 리소스가 정리되었습니다. 프로그램 종료.")

if __name__ == "__main__":
    main()
