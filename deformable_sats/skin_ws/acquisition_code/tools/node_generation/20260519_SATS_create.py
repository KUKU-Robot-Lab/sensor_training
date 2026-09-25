import os

def generate_raster_node():
    # 1. 파일 저장 경로 설정 (바탕화면 node_create 폴더 유지)
    save_dir = r"C:\Users\SM\Desktop\node_create"
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, "Grid_Raster_13.0_to_15.0.node")

    # 2. X, Y축 시작 및 종료 위치 설정 (X: -10 ~ 10, Y: -10 ~ 10)
    start_x, end_x = -10.0, 10.0
    start_y, end_y = -10.0, 10.0
    xy_step = 0.5  # 간격 (필요시 0.1 등으로 수정)

    # 3. Z축 및 딜레이 설정 (안전 높이 13.0 -> 직하강 15.0)
    safe_z = 13.0
    target_z = 15.0
    u_wait = 3.0   # 압축 유지(대기) 시간

    def float_range(start, end, step_val):
        res = []
        curr = start
        if step_val > 0:
            while curr <= end + 0.0001:
                res.append(round(curr, 3))
                curr += step_val
        else:
            while curr >= end - 0.0001:
                res.append(round(curr, 3))
                curr += step_val
        return res

    # X, Y 모두 정방향으로 리스트 생성
    x_coords = float_range(start_x, end_x, xy_step)
    y_coords = float_range(start_y, end_y, xy_step)

    node_data = []
    order = 1

    # 4. 노드 로직 생성 (ㄹ자 제거 -> 한 방향으로만 찍기)
    for y in y_coords:
        # 역방향 뒤집기( [::-1] )를 제거하여 무조건 -10부터 10까지 이동하도록 수정
        for x in x_coords:
            
            # (1) XY 지점으로 쾌속 이동 (Z=13.0 유지)
            node_data.append([order, "4직선", x, y, safe_z, 0.0, 1, 10, 10, 10, "NONE"])
            order += 1
            
            # (2) Z축 목표 깊이(15.0)로 0.6667 속도로 직하강
            node_data.append([order, "4직선", x, y, target_z, 0.0, 0.6667, 0.6667, 0.6667, 0.6667, "NONE"])
            order += 1
            
            # (3) 측정 대기/동작 (Z=15.0 압축 유지 상태에서 U축 대기)
            node_data.append([order, "4직선", x, y, target_z, u_wait, 1.0, 1.0, 1.0, 1.0, "NONE"])
            order += 1
            
            # (4) 동작 종료 (U축 리셋)
            node_data.append([order, "4직선", x, y, target_z, 0.0, 1, 10, 10, 10, "NONE"])
            order += 1
            
            # (5) 표면 긁힘 방지를 위해 Z축 안전 높이(13.0)로 복귀 상승
            node_data.append([order, "4직선", x, y, safe_z, 0.0, 1, 10, 10, 10, "NONE"])
            order += 1

    # [STEP 5] 전체 완료 후 최종 원점 복귀
    node_data.append([order, "4직선", 0.0, 0.0, 0.0, 0.0, 1, 10, 10, 10, "NONE"])

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
        print("✅ 한 방향(Raster) 전체 영역 스캔 노드 생성 완료")
        print(f"저장 경로: {full_path}")
        print(f"시작 위치: X={node_data[0][2]}, Y={node_data[0][3]}")
        print("-" * 50)
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    generate_raster_node()