import serial
import time
import threading
import csv
import queue
from datetime import datetime

try:
    import can
    CAN_AVAILABLE = True
except ImportError:
    CAN_AVAILABLE = False

# --- 설정 ---#
SENSOR_SERIAL_PORT = 'COM6'
MOTOR_SERIAL_PORT = 'COM14'
BAUD_RATE = 115200
MOTOR_BAUD_RATE = 115200

# CAN 설정
CAN_INTERFACE = 'ixxat'
CAN_CHANNEL = 0
CAN_BITRATE = 1000000
FORCE_ID = 0x01A

# 모터 설정
STEPS_PER_MM = 8889

# --- 공유 데이터 구조 ---
data_lock = threading.Lock()
global_data = {
    "timestamp": None,
    "sensor_raw": 0,
    "afd_fz": 0.0,
    "motor_z_pos": 0.0,
    "is_running": True,
    "auto_returning": False,
    "is_test_started": False,
}
command_queue = queue.Queue()

# --- 데이터 로깅 클래스 ---
class DataLogger:
    def __init__(self):
        self.file = None
        self.writer = None
        self.is_logging = False

    def start_logging(self):
        if self.is_logging: return
        filename = f"polymer_test_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Timestamp", "Sensor_Raw_Value", "AFD_Fz", "Motor_Position_mm"])
        self.is_logging = True
        print(f"로깅 시작: {filename}")

    def log_data(self):
        if not self.is_logging: return
        with data_lock:
            if global_data["timestamp"]:
                row_data = [
                    global_data["timestamp"],
                    global_data["sensor_raw"],
                    global_data["afd_fz"],
                    global_data["motor_z_pos"],
                ]
                self.writer.writerow(row_data)

    def stop_logging(self):
        if self.is_logging:
            self.is_logging = False
            if self.file: 
                self.file.close()
                print("로깅 중지 및 파일 저장 완료.")

# --- 작업자 스레드 ---
class LoggerThread(threading.Thread):
    def __init__(self, data_logger, log_interval=0.05):
        super().__init__(daemon=True)
        self.log_interval = log_interval
        self.data_logger = data_logger

    def run(self):
        while global_data["is_running"]:
            if global_data["is_test_started"]:
                self.data_logger.log_data()
            time.sleep(self.log_interval)

class CLIThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)

    def run(self):
        print_manual()
        while global_data["is_running"]:
            try:
                cmd_input = input("명령 입력 (mm) > ").strip().lower()
                if cmd_input and global_data["is_running"]:
                    command_queue.put(cmd_input)
                    if cmd_input == 'exit':
                        break
            except (EOFError, KeyboardInterrupt):
                command_queue.put('exit')
                break

class PolymerSensorReader(threading.Thread):
    def __init__(self, port, baudrate, motor_controller):
        super().__init__(daemon=True)
        self.motor_controller = motor_controller
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            print(f"폴리머 센서 포트({port})에 연결되었습니다.")
        except serial.SerialException as e:
            print(f"오류: 폴리머 센서 포트({port}) 연결 실패 - {e}")
            self.ser = None

    def run(self):
        if not self.ser: return
        while global_data["is_running"]:
            try:
                line = self.ser.readline().decode('ascii').strip()
                if line:
                    sensor_value = int(line.split(',')[0])
                    with data_lock:
                        global_data["timestamp"] = datetime.now().isoformat()
                        global_data["sensor_raw"] = sensor_value

                    # 센서 스레드가 직접 원점 복귀를 실행
                    if sensor_value == 0 and global_data["motor_z_pos"] > 0.1 and not global_data["auto_returning"]:
                        print("\n경고: 센서 값 0 감지! 자동으로 원점 복귀를 실행합니다.")
                        self.motor_controller.return_to_origin()

            except (UnicodeDecodeError, ValueError):
                if global_data["is_running"]: continue

class Afd50Reader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        if not CAN_AVAILABLE: self.bus = None; return
        try:
            self.bus = can.interface.Bus(interface=CAN_INTERFACE, channel=CAN_CHANNEL, bitrate=CAN_BITRATE)
            self.bias_f = [0.0]*3; self.cnt_f = 0; self.done_f = False
            self.CAL_SAMPLES = 100
            print(f"AFD50 CAN 인터페이스({CAN_INTERFACE})에 연결되었습니다.")
        except Exception as e:
            print(f"오류: AFD50 CAN 연결 실패 - {e}")
            self.bus = None

    def run(self):
        if not self.bus: return
        print("AFD50 바이어스 보정을 시작합니다 (약 5초 소요)...")
        start_time = time.time()
        while not self.done_f and global_data["is_running"] and time.time() - start_time < 10:
            msg = self.bus.recv(timeout=0.1)
            if msg and msg.arbitration_id == FORCE_ID:
                f = [self.conv_f(self.to_u16(msg.data, i)) for i in range(3)]
                for i in range(3): self.bias_f[i] += f[i]
                self.cnt_f += 1
                if self.cnt_f >= self.CAL_SAMPLES:
                    for i in range(3): self.bias_f[i] /= self.cnt_f
                    self.done_f = True
        
        if not self.done_f:
            print("오류: AFD50 바이어스 보정 시간 초과 또는 실패. AFD50 센서를 비활성화합니다.")
            return

        print("*** AFD50 바이어스 보정 완료! ***")

        while global_data["is_running"]:
            msg = self.bus.recv(timeout=0.1)
            if msg and msg.arbitration_id == FORCE_ID:
                with data_lock:
                    f = [self.conv_f(self.to_u16(msg.data, i)) for i in range(3)]
                    global_data["afd_fz"] = -(f[2] - self.bias_f[2])

    def to_u16(self, buf, idx): return (buf[2 * idx] << 8) | buf[2 * idx + 1]
    def conv_f(self, raw): return raw / 300.0 - 100.0

# --- 모터 제어 클래스 ---
class MotorController:
    def __init__(self, port, baudrate):
        self.z_position_mm = 0.0
        self.last_delay = 800
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            print(f"모터 제어 포트({port})에 연결되었습니다.")
        except serial.SerialException as e:
            print(f"모터 제어 포트({port}) 연결 실패: {e}")
            self.ser = None

    def _execute_move(self, distance_mm, delay):
        # 이 함수는 명령을 직접 실행하므로, auto_returning 체크를 제거합니다.
        if not self.ser: return

        steps = int(distance_mm * STEPS_PER_MM)
        cmd = f"M,Z,{steps},{delay}\n"
        self.ser.write(cmd.encode())
        self.ser.flush()

        start_pos = self.z_position_mm
        target_pos = start_pos + distance_mm
        move_duration_s = (abs(steps) * (delay / 1000000.0))
        
        start_time = time.time()
        elapsed_time = 0

        while elapsed_time < move_duration_s:
            if not global_data["is_running"]: break
            elapsed_time = time.time() - start_time
            fraction_complete = min(elapsed_time / move_duration_s, 1.0)
            current_pos = start_pos + (distance_mm * fraction_complete)
            
            self.z_position_mm = current_pos
            with data_lock:
                global_data["motor_z_pos"] = current_pos
            
            time.sleep(0.02)

        self.z_position_mm = target_pos
        with data_lock:
            global_data["motor_z_pos"] = target_pos
        time.sleep(1.0)

    def move_mm(self, distance_mm, delay=800):
        # 사용자 명령 함수 입구에서 auto_returning 상태를 체크합니다.
        if global_data["auto_returning"]:
            print("경고: 자동 원점 복귀 중에는 수동 이동 명령을 처리할 수 없습니다.")
            return
        self.last_delay = delay
        print(f"이동 시작: {distance_mm}mm...")
        self._execute_move(distance_mm, delay)
        print("이동 완료.")

    def return_to_origin(self):
        if self.z_position_mm == 0: return
        with data_lock: global_data["auto_returning"] = True
        
        print(f"원점 복귀 시작 (현재 위치: {self.z_position_mm:.3f}mm)")
        return_dist_mm = -self.z_position_mm
        self._execute_move(return_dist_mm, self.last_delay)
        print("원점 복귀 완료.")

        with data_lock: global_data["auto_returning"] = False

    def move_until_sensor_zero(self, increment_mm=0.1):
        # 사용자 명령 함수 입구에서 auto_returning 상태를 체크합니다.
        if global_data["auto_returning"]:
            print("경고: 자동 원점 복귀 중에는 max 모드를 시작할 수 없습니다.")
            return
        if not self.ser: return
        
        print("센서 값이 0이 될 때까지 전진합니다...")
        with data_lock:
            sensor_val = global_data["sensor_raw"]
        if sensor_val == 0:
            print("센서 값이 이미 0입니다.")
            return

        while global_data["is_running"]:
            with data_lock:
                sensor_val = global_data["sensor_raw"]
            
            if sensor_val == 0:
                print("센서 값 0 감지. 전진을 멈춥니다.")
                break

            # _execute_move는 내부적으로 auto_returning을 체크하지 않으므로 호출 가능
            self._execute_move(increment_mm, self.last_delay)

        print("최대 지점 도달. 원점으로 복귀합니다.")
        self.return_to_origin()

    def close(self):
        if self.ser and self.ser.is_open: self.ser.close()

# --- 메인 로직 ---
def print_manual():
    print("\n--- Z축 모터 제어 매뉴얼 ---")
    print("명령어: 이동할 거리를 mm 단위로 입력하세요.")
    print("  'max'       - 센서 값이 0이 될 때까지 전진 후 원점 복귀")
    print("  'calibrate' - 스텝/mm 값 자동 보정 모드")
    print("예시:")
    print("  '10.5'      - Z축을 양의 방향으로 10.5mm 이동 후 원점 복귀")
    print("  'exit'      - 프로그램을 종료합니다.")
    print("--------------------------\n")

def process_command(cmd, motor_controller, data_logger):
    if cmd == 'exit':
        global_data["is_running"] = False
        return

    if cmd == 'calibrate':
        print("\n--- 모터 보정 시작 ---")
        if motor_controller.z_position_mm != 0:
            print("정확한 측정을 위해, 먼저 모터를 원점으로 이동합니다.")
            motor_controller.return_to_origin()
        
        print("1. 자(ruler) 또는 캘리퍼스를 준비해주세요.")
        print("2. 준비가 되면 Enter를 누르세요. 모터가 10mm 이동합니다...")
        command_queue.get() # Wait for enter

        commanded_dist_mm = 10.0
        current_steps_per_mm = STEPS_PER_MM
        motor_controller.move_mm(commanded_dist_mm)

        print("3. 모터가 실제로 이동한 거리를 mm 단위로 정확히 측정하여 입력해주세요: ")
        measured_dist_str = command_queue.get() # Wait for measurement
        try:
            measured_dist_mm = float(measured_dist_str)
            if measured_dist_mm <= 0:
                print("오류: 0보다 큰 값을 입력해야 합니다.")
                return
        except ValueError:
            print("오류: 숫자를 입력해주세요.")
            return
        
        commanded_steps = int(commanded_dist_mm * current_steps_per_mm)
        new_steps_per_mm = int(commanded_steps / measured_dist_mm)

        print("\n--- 보정 결과 ---")
        print(f"계산된 새로운 STEPS_PER_MM 값: {new_steps_per_mm}")
        print(f"업데이트를 위해, 스크립트 상단의 STEPS_PER_MM 값을 {new_steps_per_mm}으로 직접 수정해주세요.")
        print("업데이트 후 프로그램을 재시작해주세요.")
        motor_controller.return_to_origin()
        return

    if not global_data["is_test_started"]:
        print("첫 이동 명령 수신. 로깅을 시작합니다.")
        global_data["is_test_started"] = True
        data_logger.start_logging()
        time.sleep(1)

    if cmd == 'max':
        motor_controller.move_until_sensor_zero()
        return

    try:
        distance_mm = float(cmd)
        print(f"목표: {distance_mm}mm 이동 후 원점 복귀를 시작합니다.")
        motor_controller.move_mm(distance_mm)

        if motor_controller.z_position_mm != 0:
            print("목표 지점 도달 확인. 원점으로 복귀합니다.")
            motor_controller.return_to_origin()

    except ValueError:
        print(f"잘못된 입력입니다: '{cmd}'")

def main():
    # --- 객체 생성 ---
    motor_controller = MotorController(MOTOR_SERIAL_PORT, MOTOR_BAUD_RATE)
    data_logger = DataLogger()

    # --- 스레드 생성 및 시작 ---
    sensor_reader = PolymerSensorReader(SENSOR_SERIAL_PORT, BAUD_RATE, motor_controller)
    afd_reader = Afd50Reader()
    logger_thread = LoggerThread(data_logger)
    cli_thread = CLIThread()

    sensor_reader.start()
    afd_reader.start()
    logger_thread.start()
    cli_thread.start()

    try:
        while global_data["is_running"]:
            try:
                cmd_input = command_queue.get(timeout=0.5) # 0.5초마다 타임아웃
                process_command(cmd_input, motor_controller, data_logger)
                if cmd_input == 'exit':
                    break
            except queue.Empty:
                # 사용자 입력이 없을 때 다른 작업을 수행할 수 있음 (현재는 아무것도 안함)
                continue

    finally:
        print("프로그램 종료 절차 시작...")
        global_data["is_running"] = False
        if motor_controller: motor_controller.close()
        if data_logger: data_logger.stop_logging()
        print("모든 리소스 정리 완료. 프로그램 종료.")

if __name__ == "__main__":
    main()