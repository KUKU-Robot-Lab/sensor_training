import sys
import time
from ctypes import *
import os

# PAIX NMC2.dll 설정 (64비트)
DLL_PATH = r"C:\Program Files (x86)\PAIX\NMC\Sample\Python\dist_64bit\NMC2.dll"
DEV_NO = 11  # 기본값 11 (IP 192.168.0.11)
POLLING_INTERVAL = 0.005  # 200Hz (0.005s)

class NMC_AXES_EXPR(Structure):
    _pack_ = 1
    _fields_ = [
        ("nBusy", c_short * 8),
        ("nError", c_short * 8),
        ("nNear", c_short * 8),
        ("nPLimit", c_short * 8),
        ("nMLimit", c_short * 8),
        ("nAlarm", c_short * 8),
        ("nEmer", c_short * 2),
        ("nSwPLimit", c_short * 8),
        ("nInpo", c_short * 8),
        ("nHome", c_short * 8),
        ("nEncZ", c_short * 8),
        ("nOrg", c_short * 8),
        ("nSReady", c_short * 8),
        ("nContStatus", c_short * 2),
        ("nDummy", c_short * 6),
        ("nSwMLimit", c_short * 8),
        ("lEnc", c_int * 8),
        ("lCmd", c_int * 8),
        ("dEnc", c_double * 8),
        ("dCmd", c_double * 8),
        ("dummy", c_char * 4),
    ]

def main():
    if not os.path.exists(DLL_PATH):
        print(f"EtherMotion Reader: DLL을 찾을 수 없습니다: {DLL_PATH}", file=sys.stderr)
        return

    try:
        nmc = cdll.LoadLibrary(DLL_PATH)
    except Exception as e:
        print(f"EtherMotion Reader: DLL 로드 실패: {e}", file=sys.stderr)
        return

    # 장치 연결
    print(f"EtherMotion Reader: 장치 {DEV_NO} (192.168.0.{DEV_NO}) 연결 시도 중...", file=sys.stderr)
    ret = nmc.nmc_PingCheck(DEV_NO, 100)
    if ret != 0:
        print(f"EtherMotion Reader: Ping 실패 (에러코드: {ret}). 네트워크 연결을 확인하세요.", file=sys.stderr)
        # 핑 실패해도 Open 시도 (가끔 핑만 안되는 경우 있음)
    
    ret = nmc.nmc_OpenDeviceEx(DEV_NO, 100)
    if ret != 0:
        print(f"EtherMotion Reader: 장치 연결 실패 (에러코드: {ret}).", file=sys.stderr)
        return
    
    print("EtherMotion Reader: 장치 연결 성공.", file=sys.stderr)
    print("ETHERMOTION_READY") # 메인 컨트롤러에 신호
    sys.stdout.flush()

    axes_status = NMC_AXES_EXPR()

    try:
        while True:
            t_start = time.perf_counter()
            
            # 상태 읽기
            ret = nmc.nmc_GetAxesExpress(DEV_NO, byref(axes_status))
            if ret == 0:
                # 0, 1, 2번 축의 Encoder Position 추출 (실제 위치)
                x = axes_status.dEnc[0]
                y = axes_status.dEnc[1]
                z = axes_status.dEnc[2]
                print(f"X:{x:.3f},Y:{y:.3f},Z:{z:.3f}")
                sys.stdout.flush()
            else:
                print(f"EtherMotion Reader: 데이터 읽기 실패 (에러코드: {ret})", file=sys.stderr)

            # 200Hz 주기를 맞추기 위한 대기
            t_elapsed = time.perf_counter() - t_start
            t_sleep = POLLING_INTERVAL - t_elapsed
            if t_sleep > 0:
                time.sleep(t_sleep)
                
    except KeyboardInterrupt:
        print("EtherMotion Reader: 종료됨.", file=sys.stderr)
    finally:
        nmc.nmc_CloseDevice(DEV_NO)

if __name__ == "__main__":
    main()
