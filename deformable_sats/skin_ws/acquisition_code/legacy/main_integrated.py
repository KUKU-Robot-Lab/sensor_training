import subprocess
import threading
from threading import Event
import serial
import time
import csv
import sys
import os
from datetime import datetime
from collections import deque
from ctypes import *

# ==============================================================================
# --- 전역 설정 ---
# ==============================================================================
BASE_DATA_DIR = r"C:\Users\SORO7\Downloads\acquisition_code-20260317T042906Z-3-001\acquisition_code\data"
NODE_FILE_PATH = r"C:\Program Files (x86)\PAIX\NMC\EtherMotion\Node\Conti\tactile_sensor_10.5~13.0.node"
DLL_64_PATH = r"C:\Program Files (x86)\PAIX\NMC\Sample\Python\dist_64bit\NMC2.dll"

DEV_NO = 11
LOGGING_FREQUENCY_HZ = 200
POLLING_INTERVAL = 1.0 / LOGGING_FREQUENCY_HZ

# ==============================================================================

# --- PAIX DLL 구조체 및 설정 ---
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

class NMCCONTISTATUS(Structure):
    _pack_ = 1
    _fields_ = [
        ("nContiRun", c_short * 2),
        ("nContiWait", c_short * 2),
        ("nContiRemainBuffNum", c_short * 2),
        ("nContiStopReason", c_short * 2),
        ("uiContiExecutionNum", c_uint * 2),
    ]

# --- 전역 변수 ---
procs, queues = {}, {}
lock = threading.Lock()
sensors_ready_event = {'afd50': Event()}
latest_sensor_data = {
    "due_s_values": [-1]*16, 
    "afd_fx": 0.0, "afd_fy": 0.0, "afd_fz": 0.0, 
    "real_x": 0.0, "real_y": 0.0, "real_z": 0.0
}
recording_active = False
nmc = None

# --- 센서 리더 관리 ---
def enqueue_output(process, queue, name):
    for line in iter(process.stdout.readline, ''):
        if name == 'afd50' and 'AFD50_READY' in line:
            sensors_ready_event['afd50'].set()
            continue
        queue.append(line)
    process.stdout.close()

def log_stderr(process, name):
    for line in iter(process.stderr.readline, ''):
        print(f"[{name.upper()} STDERR] {line.strip()}", file=sys.stderr)
    process.stderr.close()

def start_readers():
    print("센서 프로세스(DUE, AFD50) 시작 중...")
    py_executable = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    commands = {
        'due': [py_executable, '-u', os.path.join(script_dir, 'due_reader.py')],
        'afd50': [py_executable, '-u', os.path.join(script_dir, 'afd50_reader.py')]
    }

    for name, cmd in commands.items():
        procs[name] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        queues[name] = deque(maxlen=1)
        threading.Thread(target=enqueue_output, args=(procs[name], queues[name], name), daemon=True).start()
        threading.Thread(target=log_stderr, args=(procs[name], name), daemon=True).start()

    sensors_ready_event['afd50'].wait(timeout=15)
    print("센서 프로세스 준비 완료.")

def stop_readers():
    for proc in procs.values():
        if proc.poll() is None:
            proc.terminate()
    procs.clear()

# --- 데이터 통합 로깅 스레드 ---
def logging_thread(filename):
    global recording_active
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    full_path = os.path.join(BASE_DATA_DIR, filename)
    
    headers = ["Time", "X", "Y", "Z", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"] + [f"Skin{i+1}" for i in range(16)]
    axes_status = NMC_AXES_EXPR()
    
    with open(full_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        print(f"데이터 기록 시작: {full_path}")
        while recording_active:
            t_start = time.perf_counter()
            
            # 1. 이더모션 좌표 읽기
            if nmc:
                nmc.nmc_GetAxesExpress(DEV_NO, byref(axes_status))
                with lock:
                    latest_sensor_data["real_x"] = axes_status.dCmd[0]
                    latest_sensor_data["real_y"] = axes_status.dCmd[1]
                    latest_sensor_data["real_z"] = axes_status.dCmd[2]

            # 2. DUE/AFD50 데이터 업데이트
            try:
                line = queues['due'].pop()
                parts = [int(p) for p in line.strip().split(',') if p]
                if len(parts) == 16: latest_sensor_data["due_s_values"] = parts
            except: pass
            
            try:
                line = queues['afd50'].pop()
                if line.startswith("Fx:"):
                    parts = line.split(',')
                    latest_sensor_data["afd_fx"] = float(parts[0].split(':')[1])
                    latest_sensor_data["afd_fy"] = float(parts[1].split(':')[1])
                    latest_sensor_data["afd_fz"] = float(parts[2].split(':')[1])
            except: pass

            # 3. CSV 기록
            d = latest_sensor_data
            row = [
                datetime.now().strftime("%H:%M:%S.%f")[:-3],
                f"{d['real_x']:.3f}", f"{d['real_y']:.3f}", f"{d['real_z']:.3f}",
                f"{d['afd_fx']:.3f}", f"{d['afd_fy']:.3f}", f"{d['afd_fz']:.3f}",
                "0.000", "0.000", "0.000"
            ]
            row.extend(d["due_s_values"])
            writer.writerow(row)
            
            # 200Hz 맞춤
            elapsed = time.perf_counter() - t_start
            sleep_time = POLLING_INTERVAL - elapsed
            if sleep_time > 0: time.sleep(sleep_time)

    print("데이터 기록 종료.")

# --- 노드 파일 로드 및 실행 ---
def run_node_conti():
    global nmc, recording_active
    nmc = cdll.LoadLibrary(DLL_64_PATH)
    
    # 1. 연결
    ret = nmc.nmc_OpenDeviceEx(DEV_NO, 100)
    if ret != 0:
        print(f"이더모션 연결 실패 (에러:{ret})")
        return
    print("이더모션 연결 성공.")

    # 서보 ON
    for axis in [0, 1, 2]:
        nmc.nmc_SetServoOn(DEV_NO, axis, 1)
    time.sleep(0.5)

    GROUP_NO = 0
    nmc.nmc_ContiSetNodeClear(DEV_NO, GROUP_NO)
    
    # 노드 파일 전체 읽기
    print(f"노드 파일 읽는 중: {NODE_FILE_PATH}")
    node_list = []
    with open(NODE_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[2:]
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 10: continue
            try:
                # 0:Idx, 1:Func, 2:X, 3:Y, 4:Z, ..., 9:Vel
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                vel = float(parts[9]) if parts[9] else float(parts[6])
                acc, dec = float(parts[7]), float(parts[8])
                node_list.append((x, y, z, vel, acc, dec))
            except: continue

    if not node_list:
        print("로드할 노드가 없습니다.")
        return

    # 로깅 시작
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"tactile_data_{timestamp}.csv"
    recording_active = True
    log_thread = threading.Thread(target=logging_thread, args=(filename,))
    log_thread.start()

    print(f"총 {len(node_list)}개 노드 구동 시작...")
    
    # 초기 노드 채우기 (최대 500개)
    initial_fill = min(len(node_list), 500)
    for i in range(initial_fill):
        x, y, z, vel, acc, dec = node_list[i]
        nmc.nmc_ContiAddNodeLine3Axis(DEV_NO, GROUP_NO, c_double(x), c_double(y), c_double(z), c_double(vel), c_double(acc), c_double(dec))
    
    # 구동 시작
    nmc.nmc_ContiRunStop(DEV_NO, GROUP_NO, 1)
    
    current_idx = initial_fill
    conti_status = NMCCONTISTATUS()
    axes_status = NMC_AXES_EXPR()

    try:
        while True:
            # 상태 업데이트
            nmc.nmc_ContiGetStatus(DEV_NO, byref(conti_status))
            nmc.nmc_GetAxesExpress(DEV_NO, byref(axes_status))
            
            remain_buf = conti_status.nContiRemainBuffNum[GROUP_NO]
            is_running = axes_status.nContStatus[GROUP_NO]
            
            # 버퍼에 여유가 있고 남은 노드가 있다면 추가
            if remain_buf > 100 and current_idx < len(node_list):
                to_add = min(len(node_list) - current_idx, remain_buf - 50)
                for _ in range(to_add):
                    x, y, z, vel, acc, dec = node_list[current_idx]
                    ret = nmc.nmc_ContiAddNodeLine3Axis(DEV_NO, GROUP_NO, c_double(x), c_double(y), c_double(z), c_double(vel), c_double(acc), c_double(dec))
                    if ret == 0:
                        current_idx += 1
                    else:
                        break
            
            print(f"\r진행: {current_idx}/{len(node_list)} | 버퍼잔량: {remain_buf} | X:{axes_status.dCmd[0]:.2f} Y:{axes_status.dCmd[1]:.2f} Z:{axes_status.dCmd[2]:.2f}", end="")

            # 모든 노드가 들어갔고 구동이 멈췄으면 종료
            if current_idx >= len(node_list) and is_running == 0:
                print("\n모든 이동 완료.")
                break
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n중단 요청됨.")
        nmc.nmc_ContiRunStop(DEV_NO, GROUP_NO, 0)

    recording_active = False
    log_thread.join()
    nmc.nmc_CloseDevice(DEV_NO)

def main():
    try:
        start_readers()
        run_node_conti()
    except Exception as e:
        print(f"메인 루프 오류: {e}")
    finally:
        stop_readers()
        print("모든 프로세스 종료.")

if __name__ == "__main__":
    main()
