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
# 데이터 저장 폴더
BASE_DATA_DIR = r"C:\Users\SORO7\Downloads\acquisition_code-20260317T042906Z-3-001\acquisition_code\data"
# PAIX DLL 경로 (64비트)
DLL_64_PATH = r"C:\Program Files (x86)\PAIX\NMC\Sample\Python\dist_64bit\NMC2.dll"

DEV_NO = 11  # 컨트롤러 IP 번호 (기본 11)
LOGGING_FREQUENCY_HZ = 200 # 기록 주파수
POLLING_INTERVAL = 1.0 / LOGGING_FREQUENCY_HZ

# ==============================================================================

# --- PAIX DLL 구조체 ---
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

# --- 전역 변수 및 상태 관리 ---
procs, queues = {}, {}
lock = threading.Lock()
sensors_ready_event = {'afd50': Event()}
latest_sensor_data = {
    "due_s_values": [0]*16, 
    "afd_fx": 0.0, "afd_fy": 0.0, "afd_fz": 0.0, 
    "real_x": 0.0, "real_y": 0.0, "real_z": 0.0
}
running = True
nmc = None

# --- 센서 리더 관리 (DUE, AFD50) ---
def enqueue_output(process, queue, name):
    for line in iter(process.stdout.readline, ''):
        if name == 'afd50' and 'AFD50_READY' in line:
            sensors_ready_event['afd50'].set()
            continue
        queue.append(line)
    process.stdout.close()

def log_stderr(process, name):
    for line in iter(process.stderr.readline, ''):
        # 센서 데이터 수신 확인을 위해 stderr 출력 유지
        if "Fx=" in line or "Due Reader" in line:
            print(f"[{name.upper()}] {line.strip()}", file=sys.stderr)
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

    print("AFD50 센서 대기 및 바이어스 보정 중...")
    sensors_ready_event['afd50'].wait(timeout=15)
    print("모든 센서 준비 완료.")

def stop_readers():
    print("\n센서 프로세스 종료 중...")
    for proc in procs.values():
        if proc.poll() is None:
            proc.terminate()
    procs.clear()

# --- 통합 데이터 로깅 루프 ---
def run_logger():
    global nmc, running
    
    # 1. 이더모션 DLL 연결
    try:
        nmc = cdll.LoadLibrary(DLL_64_PATH)
        ret = nmc.nmc_OpenDeviceEx(DEV_NO, 100)
        if ret != 0:
            print(f"이더모션 연결 실패 (에러:{ret}) - 좌표 없이 센서값만 기록합니다.")
            nmc = None
        else:
            print("이더모션 연결 성공. 실시간 좌표 추적을 시작합니다.")
    except Exception as e:
        print(f"DLL 로드 실패: {e}")
        nmc = None

    # 2. 파일 생성
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"tactile_logger_{timestamp}.csv"
    full_path = os.path.join(BASE_DATA_DIR, filename)
    
    headers = ["Timestamp", "X", "Y", "Z", "Fx", "Fy", "Fz"] + [f"Skin{i+1}" for i in range(16)]
    axes_status = NMC_AXES_EXPR()
    
    print(f"\n" + "="*50)
    print(f"데이터 기록 시작!")
    print(f"저장 경로: {full_path}")
    print("종료하려면 Ctrl+C를 누르세요.")
    print("="*50 + "\n")

    with open(full_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        try:
            while running:
                t_start = time.perf_counter()
                
                # (A) 이더모션 좌표 읽기
                if nmc:
                    nmc.nmc_GetAxesExpress(DEV_NO, byref(axes_status))
                    # 0=X, 1=Y, 2=Z축 가정
                    latest_sensor_data["real_x"] = axes_status.dCmd[0]
                    latest_sensor_data["real_y"] = axes_status.dCmd[1]
                    latest_sensor_data["real_z"] = axes_status.dCmd[2]

                # (B) DUE 데이터 업데이트
                try:
                    line = queues['due'].pop()
                    parts = [int(p) for p in line.strip().split(',') if p]
                    if len(parts) == 16: latest_sensor_data["due_s_values"] = parts
                except IndexError: pass
                
                # (C) AFD50 데이터 업데이트
                try:
                    line = queues['afd50'].pop()
                    if line.startswith("Fx:"):
                        parts = line.split(',')
                        latest_sensor_data["afd_fx"] = float(parts[0].split(':')[1])
                        latest_sensor_data["afd_fy"] = float(parts[1].split(':')[1])
                        latest_sensor_data["afd_fz"] = float(parts[2].split(':')[1])
                except IndexError: pass

                # (D) 데이터 기록
                d = latest_sensor_data
                row = [
                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    f"{d['real_x']:.3f}", f"{d['real_y']:.3f}", f"{d['real_z']:.3f}",
                    f"{d['afd_fx']:.3f}", f"{d['afd_fy']:.3f}", f"{d['afd_fz']:.3f}"
                ]
                row.extend(d["due_s_values"])
                writer.writerow(row)
                
                # 화면에 실시간 표시 (10Hz 주기로)
                if int(t_start * 10) % 2 == 0:
                    print(f"\r[REC] X:{d['real_x']:>7.2f} Y:{d['real_y']:>7.2f} Z:{d['real_z']:>7.2f} | Fz:{d['afd_fz']:>6.2f} N", end="")

                # 주기 유지 (200Hz)
                elapsed = time.perf_counter() - t_start
                sleep_time = POLLING_INTERVAL - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n기록 중지 요청됨.")
        finally:
            running = False

    print(f"\n데이터 기록 완료: {full_path}")
    if nmc:
        nmc.nmc_CloseDevice(DEV_NO)

def main():
    try:
        start_readers()
        run_logger()
    except Exception as e:
        print(f"\n메인 오류: {e}")
    finally:
        stop_readers()
        print("프로그램 종료.")

if __name__ == "__main__":
    main()
