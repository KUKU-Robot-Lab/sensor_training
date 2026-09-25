import os

def generate_raster_node():
    save_dir = os.path.dirname(os.path.abspath(__file__))  # skin_ws/node 고정(환경 무관)
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, "SATS_d5_mk20_StairStep_05_Ufix_velfix.node")

    start_x, end_x = -10.0, 10.0
    start_y, end_y = -10.0, 10.0
    xy_step = 0.5

    safe_z = 13.0
    target_z = 15.5

    # [계단식 하강 파라미터]
    z_step = 0.5

    # [U축 미세 구동 파라미터]
    # 0.5 <-> 0 토글 (누적시키지 않고 항상 0.5폭으로만 이동시켜 대기시간을 동일하게 유지)
    u_step_val = 0.5

    # [고속 이동 파라미터 - X, Y축 이동 및 Z축 상승 복귀용]
    fast_acc = 10.0
    fast_dec = 10.0
    fast_vel = 10.0

    # [압입/접촉/멈춤 속도]
    press_acc = 1.0
    press_dec = 1.0
    press_vel = 1.0

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

    x_list = float_range(start_x, end_x, xy_step)
    y_list = float_range(start_y, end_y, xy_step)

    node_data = []
    order = 1

    prev_drive = 1.0

    def add_node(x, y, z, u, acc, dec, drive):
        nonlocal order, prev_drive
        start_vel = prev_drive
        node_data.append([order, "4직선", x, y, z, u, start_vel, acc, dec, drive, "NONE"])
        order += 1
        prev_drive = drive

    for y in y_list:
        for x in x_list:

            # ==========================================================
            # [Step 1] X, Y 좌표 이동 (안전 높이 safe_z)
            # ==========================================================
            prev_drive = 1.0
            add_node(x, y, safe_z, 0.0, fast_acc, fast_dec, fast_vel)

            # ==========================================================
            # [Step 2 & 3] Z축 계단식 하강 및 U축 구동 최적화
            # ==========================================================
            current_z = safe_z
            current_u = 0.0

            # Step1(구동=fast_vel) 직후 첫 압입 이동으로 넘어갈 때, 시작속도가
            # Step1의 구동속도(fast_vel)를 그대로 물려받으면 압입 구간의
            # 가속/감속/구동(press_*)과 맞지 않아 컨트롤러가 해당 노드를
            # 제대로 처리하지 못하고 건너뛰는 문제가 있었음.
            # -> 압입 구간으로 들어가기 전 시작속도를 press_vel로 리셋.
            prev_drive = press_vel

            # (1) 시작 높이에서 첫 번째 U축 구동 (0 -> 0.5)
            current_u = u_step_val
            add_node(x, y, current_z, current_u, press_acc, press_dec, press_vel)

            # (2) 계단식 하강 루프
            while current_z < target_z - 0.0001:
                next_z = round(current_z + z_step, 3)
                if next_z > target_z:
                    next_z = target_z

                # Z축 하강 (U축은 현재 위치 유지)
                add_node(x, y, next_z, current_u, press_acc, press_dec, press_vel)

                # U축 구동: 0.5 <-> 0 토글
                # (누적시키지 않고 항상 0.5폭으로만 이동시켜 대기시간을 동일하게 유지)
                current_u = 0.0 if current_u > 0.0001 else u_step_val
                add_node(x, y, next_z, current_u, press_acc, press_dec, press_vel)

                current_z = next_z

            # ==========================================================
            # [Step 4] U축 복귀 (해당 좌표의 모든 계단 완료 후)
            # 토글 방식이라 0.5에서 끝났을 때만 0으로 복귀 (이동량 0.5로 동일)
            # ==========================================================
            if current_u > 0.0001:
                add_node(x, y, target_z, 0.0, press_acc, press_dec, press_vel)

            # ==========================================================
            # [Step 5] Z축 단독 상승 복귀
            # ==========================================================
            prev_drive = 1.0
            add_node(x, y, safe_z, 0.0, fast_acc, fast_dec, fast_vel)

    # ==========================================================
    # [Step 6] 최종 원점 복귀
    # ==========================================================
    prev_drive = 1.0
    add_node(0.0, 0.0, 0.0, 0.0, fast_acc, fast_dec, fast_vel)

    with open(full_path, 'w', encoding='cp949') as f:
        header1 = ["순서", "함수", "위치", "위치", "위치", "위치", "속도", "속도", "속도", "속도", "IO"]
        header2 = ["순서", "함수", "X", "Y", "Z", "U", "시작", "가속", "감속", "구동", "IO"]
        f.write(",".join(header1) + "\n")
        f.write(",".join(header2) + "\n")
        for row in node_data:
            f.write(",".join(map(str, row)) + "\n")

    print(f"SATS_d5_mk20_StairStep_05_Ufix_velfix.node 파일이 생성되었습니다: {full_path}")

if __name__ == '__main__':
    generate_raster_node()
