import os

def generate_custom_sensor_node():
    # 1. 파일 저장 경로 설정 (다운로드 폴더)
    downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
    full_path = os.path.join(downloads_dir, "Sensor_Sequence_Test.node")

    # 2. 파라미터 설정
    target_x, target_y = 2.0, 0.0
    safe_z = 10.5        # 1차 접근 높이
    press_z = 12.0      # 최종 압입 높이
    
    node_data = []
    order = 1

    # [STEP 1] (0, 0, 0)에서 (0, 0, 10.5)로 쾌속 하강
    node_data.append([order, "4직선", target_x, target_y, safe_z, 0.0, 1, 10, 10, 10, "NONE"])
    order += 1

    # [STEP 2] 10.5 지점에서 3초 대기 (U=3.0, 속도=1.0)
    node_data.append([order, "4직선", target_x, target_y, safe_z, 3.0, 1.0, 1.0, 1.0, 1.0, "NONE"])
    order += 1
    
    # [STEP 3] 대기 종료 후 U축 초기화 (잠시 멈춤)
    node_data.append([order, "4직선", target_x, target_y, safe_z, 0.0, 1, 10, 10, 10, "NONE"])
    order += 1

    # [STEP 4] 10.5에서 12.0까지 천천히 하강 (속도 0.6667)
    node_data.append([order, "4직선", target_x, target_y, press_z, 0.0, 0.6667, 0.6667, 0.6667, 0.6667, "NONE"])
    order += 1

    # [STEP 5] 12.0 지점에서 3초 대기 (U=3.0, 속도=1.0)
    node_data.append([order, "4직선", target_x, target_y, press_z, 3.0, 1.0, 1.0, 1.0, 1.0, "NONE"])
    order += 1

    # [STEP 6] 대기 종료 후 U축 초기화
    node_data.append([order, "4직선", target_x, target_y, press_z, 0.0, 1, 10, 10, 10, "NONE"])
    order += 1

    # [STEP 7] 원점(0, 0, 0)으로 쾌속 복귀
    node_data.append([order, "4직선", 0.0, 0.0, 0.0, 0.0, 1, 10, 10, 10, "NONE"])
    order += 1

    # 3. 파일 저장 (CP949 인코딩)
    header1 = ["순서", "함수", "위치", "위치", "위치", "위치", "속도", "속도", "속도", "속도", "IO"]
    header2 = ["순서", "함수", "X", "Y", "Z", "U", "시작", "가속", "감속", "구동", "IO"]

    try:
        with open(full_path, 'w', encoding='cp949') as f:
            f.write(",".join(header1) + "\n")
            f.write(",".join(header2) + "\n")
            for row in node_data:
                line = ",".join(str(item) for item in row)
                f.write(line + "\n")
        
        print("-" * 50)
        print("✅ 노드 파일이 성공적으로 생성되었습니다!")
        print(f"저장 경로: {full_path}")
        print(f"동작 요약: 하강(10.5) -> 3초 대기 -> 천천히 하강(12.0) -> 3초 대기 -> 원점 복귀")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    generate_custom_sensor_node()