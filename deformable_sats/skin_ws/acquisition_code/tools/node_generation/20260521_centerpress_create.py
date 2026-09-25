import os

def generate_center_single_node():
    # 1. 파일 저장 경로 설정 (요청하신 바탕화면 경로)
    save_dir = r"C:\Users\SM\Desktop\node_create"
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, "Center_Single_Press_13.0_to_15.0.node")

    # 2. 파라미터 설정 (가운데 위치 및 Z축 범위)
    target_x = 0.0
    target_y = 0.0
    z_start = 12.0
    z_end = 14.0

    node_data = []
    order = 1

    # 3. 노드 로직 생성 (계단식 하강 없이 한 번에 이동 후동작)
    
    # [STEP 1] Z축 첫 번째 높이(13.0) 이동 및 U축 동작
    node_data.append([order, "4직선", target_x, target_y, z_start, 0.0, 1, 10, 10, 10, "NONE"])
    order += 1
    node_data.append([order, "4직선", target_x, target_y, z_start, 3.0, 1.0, 1.0, 1.0, 1.0, "NONE"])
    order += 1
    node_data.append([order, "4직선", target_x, target_y, z_start, 0.0, 1, 10, 10, 10, "NONE"])
    order += 1

    # [STEP 2] Z축 두 번째 높이(15.0) 이동 및 U축 동작 (지정해주신 속도 0.6667 적용)
    node_data.append([order, "4직선", target_x, target_y, z_end, 0.0, 0.6667, 0.6667, 0.6667, 0.6667, "NONE"])
    order += 1
    node_data.append([order, "4직선", target_x, target_y, z_end, 3.0, 1.0, 1.0, 1.0, 1.0, "NONE"])
    order += 1
    node_data.append([order, "4직선", target_x, target_y, z_end, 0.0, 1, 10, 10, 10, "NONE"])
    order += 1

    # [STEP 3] 최종 원점 복귀
    node_data.append([order, "4직선", 0.0, 0.0, 0.0, 0.0, 1, 10, 10, 10, "NONE"])
    order += 1

    # 4. CSV(Node) 형태로 저장
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
        print("✅ 가운데 단일 포인트 노드 생성 완료")
        print(f"저장 경로: {full_path}")
        print("-" * 50)
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    generate_center_single_node()