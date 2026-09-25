import os

def generate_n1_center_line_direct_plunge():
    # 1. 파일 저장 경로 설정
    save_dir = r"C:\Users\SM\Desktop\node_create"
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, "N1_CenterLine_10.0_to_-10.0.node")

    # 2. X, Y축 설정 (가운데 한 줄: X=0 고정, Y=10.0 ~ -10.0, 간격 -0.1mm)
    target_x = 0.0
    start_y = 10.0
    end_y = -10.0
    y_step = -0.1

    # 3. Z축 압입 파라미터 설정 (계단식 하강 없이 13.0 -> 15.0 한 번에)
    z_start = 13.0       # 시작 및 이동 시 안전 높이
    z_end = 15.0         # 목표 압입 깊이

    # 양수/음수 스텝을 모두 지원하는 범위 계산 함수
    def float_range(start, end, step_val):
        res = []
        curr = start
        if step_val < 0:
            while curr >= end - 0.0001:
                res.append(round(curr, 3))
                curr += step_val
        else:
            while curr <= end + 0.0001:
                res.append(round(curr, 3))
                curr += step_val
        return res

    # Y 좌표 리스트 생성
    y_coords = float_range(start_y, end_y, y_step)

    node_data = []
    order = 1

    # 4. 노드 로직 생성
    for y in y_coords:
        x = target_x
        
        # [STEP 1] 다음 XY 위치로 이동 및 Z 안전 높이(13.0) 유지
        node_data.append([order, "4직선", x, y, z_start, 0.0, 1, 10, 10, 10, "NONE"])
        order += 1
        # 13.0 높이에서 U축 대기
        node_data.append([order, "4직선", x, y, z_start, 3.0, 1.0, 1.0, 1.0, 1.0, "NONE"])
        order += 1
        # U축 리셋
        node_data.append([order, "4직선", x, y, z_start, 0.0, 1, 10, 10, 10, "NONE"])
        order += 1

        # [STEP 2] Z 목표 높이(15.0)로 한 번에 하강
        node_data.append([order, "4직선", x, y, z_end, 0.0, 0.6667, 0.6667, 0.6667, 0.6667, "NONE"])
        order += 1
        # 15.0 높이에서 U축 대기 (압축 유지)
        node_data.append([order, "4직선", x, y, z_end, 3.0, 1.0, 1.0, 1.0, 1.0, "NONE"])
        order += 1
        # U축 리셋
        node_data.append([order, "4직선", x, y, z_end, 0.0, 1, 10, 10, 10, "NONE"])
        order += 1

        # [STEP 3] 표면을 긁지 않도록 다음 포인트 이동 전 Z 안전 높이(13.0)로 쾌속 상승
        node_data.append([order, "4직선", x, y, z_start, 0.0, 1, 10, 10, 10, "NONE"])
        order += 1

    # [STEP 4] 루프 종료 후 마지막 위치에서 3초 대기 및 원점 복귀
    last_x = target_x
    last_y = y_coords[-1]
    
    # 3초 대기 (U값 30.0 / 속도 10.0)
    node_data.append([order, "4직선", last_x, last_y, z_start, 30.0, 10.0, 10.0, 10.0, 10.0, "NONE"])
    order += 1
    # U값 리셋
    node_data.append([order, "4직선", last_x, last_y, z_start, 0.0, 1, 10, 10, 10, "NONE"])
    order += 1
    # 최종 원점 복귀
    node_data.append([order, "4직선", 0.0, 0.0, 0.0, 0.0, 1, 10, 10, 10, "NONE"])
    order += 1

    # 5. CSV 형태로 저장
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
        print("✅ Y축 10.0 ~ -10.0 반복 & 직하강(13.0->15.0) 노드 생성 완료")
        print(f"저장 경로: {full_path}")
        print("-" * 50)
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    generate_n1_center_line_direct_plunge()