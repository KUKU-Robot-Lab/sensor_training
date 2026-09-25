import serial
import time

# --- 설정 ---
MOTOR_SERIAL_PORT = 'COM14'  # 모터 컨트롤러 COM 포트
MOTOR_BAUD_RATE = 115200

# 10mm에 해당하는 스텝 수 (STEPS_PER_MM = 640)
TEST_STEPS = 640 * 10 

# 모터 이동 명령 (M,축,스텝,딜레이)
# 딜레이 값이 클수록 속도가 느려집니다.
command_to_send = f"M,Z,{TEST_STEPS},800\n"

# --- 테스트 실행 ---
print(f"모터 제어 포트({MOTOR_SERIAL_PORT})에 연결을 시도합니다...")

try:
    # 시리얼 포트 열기
    motor_ser = serial.Serial(MOTOR_SERIAL_PORT, MOTOR_BAUD_RATE, timeout=2)
    time.sleep(2) # 아두이노가 리셋되고 시리얼 통신을 준비할 시간을 줍니다. 
    print("포트 연결 성공!")

    # 명령어 전송
    print(f"다음 명령어를 전송합니다: {command_to_send.strip()}")
    motor_ser.write(command_to_send.encode('ascii'))
    
    # 아두이노가 명령을 처리할 시간을 줍니다.
    # 예상 이동 시간: (스텝 수 * (딜레이_us / 1,000,000))초. 여기서는 넉넉하게 대기합니다.
    estimated_time = (abs(TEST_STEPS) * (800 / 1000000)) + 1
    print(f"모터가 약 {estimated_time:.2f}초 동안 움직이는지 확인하세요...")
    time.sleep(estimated_time)

    print("테스트 명령 전송 완료.")

finally:
    # 시리얼 포트 닫기
    if 'motor_ser' in locals() and motor_ser.is_open:
        motor_ser.close()
        print("포트 연결을 종료했습니다.")
