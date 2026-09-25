"""xy 1mm 계단식 z node 생성 (d5·d10) — SATS_create_stair_0.5_Ufix_velfix.py 기반.

기존 검증된 stair 스크립트에서 **xy_step 0.5 → 1.0** 만 변경 + d5/d10 두 파일 생성.
U축 토글·시작속도(prev_drive) 관리 노하우(컨트롤러 노드 건너뜀 방지)는 그대로 보존.

SOP: 새 센서 취득 = xy 1mm 격자, z 0.5mm 계단, 최대 침투 2.5mm(target_z-safe_z).
z 절대좌표(safe_z=13.0 / target_z=15.5)는 기존 스크립트 검증값 유지 — 셋업 바뀌면 조정.
저장: **항상 이 스크립트가 있는 폴더(skin_ws/node/)에 고정** — 어디서 clone/실행하든 동일.
"""
import os


def generate(indenter: str, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, f"SATS_{indenter}_mk20_StairStep_xy1_05_Ufix_velfix.node")

    start_x, end_x = -10.0, 10.0
    start_y, end_y = -10.0, 10.0
    xy_step = 1.0            # ★ SOP: 1mm (기존 0.5 → 1.0)

    safe_z = 13.0
    target_z = 15.5         # safe_z + 2.5mm (최대 침투 마지노선)
    z_step = 0.5
    u_step_val = 0.5        # U축 0.5<->0 토글(대기시간 균일)

    fast_acc = fast_dec = fast_vel = 10.0
    press_acc = press_dec = press_vel = 1.0

    def float_range(start, end, step_val):
        res, curr = [], start
        while curr <= end + 0.0001:
            res.append(round(curr, 3))
            curr += step_val
        return res

    x_list = float_range(start_x, end_x, xy_step)
    y_list = float_range(start_y, end_y, xy_step)

    node_data, order, prev_drive = [], 1, 1.0

    def add_node(x, y, z, u, acc, dec, drive):
        nonlocal order, prev_drive
        node_data.append([order, "4직선", x, y, z, u, prev_drive, acc, dec, drive, "NONE"])
        order += 1
        prev_drive = drive

    for y in y_list:
        for x in x_list:
            # [1] XY 이동 (안전 높이)
            prev_drive = 1.0
            add_node(x, y, safe_z, 0.0, fast_acc, fast_dec, fast_vel)

            # [2-3] Z 계단식 하강 + U 토글 (시작속도 press_vel 리셋 — 노드 건너뜀 방지)
            current_z, current_u = safe_z, 0.0
            prev_drive = press_vel
            current_u = u_step_val
            add_node(x, y, current_z, current_u, press_acc, press_dec, press_vel)
            while current_z < target_z - 0.0001:
                next_z = min(round(current_z + z_step, 3), target_z)
                add_node(x, y, next_z, current_u, press_acc, press_dec, press_vel)
                current_u = 0.0 if current_u > 0.0001 else u_step_val
                add_node(x, y, next_z, current_u, press_acc, press_dec, press_vel)
                current_z = next_z

            # [4] U 복귀
            if current_u > 0.0001:
                add_node(x, y, target_z, 0.0, press_acc, press_dec, press_vel)

            # [5] Z 상승 복귀
            prev_drive = 1.0
            add_node(x, y, safe_z, 0.0, fast_acc, fast_dec, fast_vel)

    # [6] 원점 복귀
    prev_drive = 1.0
    add_node(0.0, 0.0, 0.0, 0.0, fast_acc, fast_dec, fast_vel)

    with open(full_path, "w", encoding="cp949") as f:
        f.write(",".join(["순서", "함수", "위치", "위치", "위치", "위치", "속도", "속도", "속도", "속도", "IO"]) + "\n")
        f.write(",".join(["순서", "함수", "X", "Y", "Z", "U", "시작", "가속", "감속", "구동", "IO"]) + "\n")
        for row in node_data:
            f.write(",".join(map(str, row)) + "\n")

    print(f"[{indenter}] {full_path}  ({len(x_list)}x{len(y_list)}={len(x_list)*len(y_list)}점, {len(node_data)}행)")
    return full_path


if __name__ == "__main__":
    # 저장 위치 = 이 스크립트 폴더(skin_ws/node/)로 고정. 환경 무관 상대경로.
    out = os.path.dirname(os.path.abspath(__file__))
    for ind in ("d5", "d10"):
        generate(ind, out)
    print(f"저장 위치: {out}  (xy 1mm, z 0.5 계단, 침투 최대 2.5mm)")
