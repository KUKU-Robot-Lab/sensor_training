"""spec 검사용 미세 스텝 node 생성 — 위치 분해능 실측(D10, 랜덤 순서).

SATS_create_stair_xy1_d5d10.py 의 검증 노하우(U 토글 dwell·prev_drive 관리) 유지.
분해능 취득 프로토콜(analyze_localization_resolution.py 와 짝):
  - 압입 깊이 1.5mm (학습 유효 분포 중심, 2.0mm 초과 금지)
  - 하강·복귀 모두 press_vel(1mm/s) — ★빠른 복귀 금지(점탄성 리바운드 방지)
  - 압입 사이 safe_z 회복 대기(U 토글) — 이전 압입 잔류 회복
  - ★방문 순서 랜덤(seed 고정) — 드리프트가 위치와 상관되는 것 차단
  - (0,0) 기준점을 시작/중간/끝 3회 방문 — 드리프트 체크
  - 시작 무접촉 대기 ≥5s — 분석 baseline

z 규약(★2026-08-08 실측 교정): v6_finestep 하강곡선에서 신호가 z≈13.25부터 상승
(13.0 이하 0.29% → 13.5 1.07% → 14.0 3.38%) → **실제 접촉 시작 z=13.25**.
12.5 가정 시 실제 압입은 0.75mm뿐이라 신호가 학습(1.5mm→33%)의 1/4(≤9%)에 그쳐
예측이 고정점으로 붕괴했다. contact_z=13.25 기준으로 press 14.75(깊이 1.5).
셋업(센서 교체·재장착)이 바뀌면 접촉 z를 다시 실측해 갱신할 것.
"""
import os
import random


def generate(save_dir: str, *, name: str = "SATS_d10_spec_finestep",
             axis: str = "x", span_mm: float = 1.5, step_mm: float = 0.1,
             repeats: int = 3, seed: int = 42,
             contact_z: float = 13.25, depth_mm: float = 1.5, safe_lift: float = 1.25,
             hold_toggles: int = 2, rest_toggles: int = 1,
             baseline_toggles: int = 6) -> str:
    """미세 스텝 랜덤 방문 node 파일 생성. 반환=파일 경로.

    axis     : 'x' 또는 'y' — 스윕 축(다른 축은 0 고정)
    span_mm  : ±span 구간, step_mm 스텝 → 점 개수 = 2*span/step+1
    repeats  : 전체 점 목록 반복 회수(라운드마다 랜덤 순서 재추첨)
    hold_toggles : 압입 유지 시간 = U 0.5mm 왕복(1mm/s≈1s) × 회수
    """
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, f"{name}.node")

    safe_z = round(contact_z - safe_lift, 3)       # xy 이동 높이(비접촉 여유)
    press_z = round(contact_z + depth_mm, 3)       # 압입 목표(깊이 1.5mm)
    u_step_val = 0.5

    fast_acc = fast_dec = fast_vel = 10.0
    press_acc = press_dec = press_vel = 1.0        # 하강·복귀 동일(느린 복귀)

    n = int(round(2 * span_mm / step_mm)) + 1
    pts = [round(-span_mm + i * step_mm, 4) for i in range(n)]
    rng = random.Random(seed)

    node_data, order, prev_drive = [], 1, 1.0

    def add_node(x, y, z, u, acc, dec, drive):
        nonlocal order, prev_drive
        node_data.append([order, "4직선", x, y, z, u, prev_drive, acc, dec, drive, "NONE"])
        order += 1
        prev_drive = drive

    def u_toggles(x, y, z, count):
        """z 유지·U 0.5↔0 왕복 — 시간 벌기(1회≈1s). 항상 U=0 으로 끝남."""
        nonlocal prev_drive
        u = 0.0
        for _ in range(count):
            u = u_step_val if u < 0.0001 else 0.0
            add_node(x, y, z, u, press_acc, press_dec, press_vel)
        if u > 0.0001:
            add_node(x, y, z, 0.0, press_acc, press_dec, press_vel)

    def press_at(v):
        """한 점 압입 사이클: xy 이동(safe) → 느린 하강 → hold → 느린 복귀 → 회복 대기."""
        nonlocal prev_drive
        x, y = (v, 0.0) if axis == "x" else (0.0, v)
        prev_drive = 1.0
        add_node(x, y, safe_z, 0.0, fast_acc, fast_dec, fast_vel)   # xy 이동
        prev_drive = press_vel                                      # 시작속도 리셋(건너뜀 방지)
        add_node(x, y, press_z, 0.0, press_acc, press_dec, press_vel)  # 하강 1mm/s
        u_toggles(x, y, press_z, hold_toggles)                      # ★hold ≈ toggles×1s
        add_node(x, y, safe_z, 0.0, press_acc, press_dec, press_vel)   # ★느린 복귀 1mm/s
        u_toggles(x, y, safe_z, rest_toggles)                       # 회복 대기

    # [0] 시작: (0,0) safe_z 무접촉 대기(분석 baseline, ≈ baseline_toggles s)
    prev_drive = 1.0
    add_node(0.0, 0.0, safe_z, 0.0, fast_acc, fast_dec, fast_vel)
    prev_drive = press_vel
    u_toggles(0.0, 0.0, safe_z, baseline_toggles)

    # [1] 라운드 반복 — 매 라운드 랜덤 순서, 라운드 경계에 (0,0) 기준점 방문
    for _ in range(repeats):
        press_at(0.0)                              # 기준점(드리프트 체크)
        order_pts = pts[:]
        rng.shuffle(order_pts)
        for v in order_pts:
            press_at(v)
    press_at(0.0)                                  # 마지막 기준점

    # [2] 원점 복귀
    prev_drive = 1.0
    add_node(0.0, 0.0, 0.0, 0.0, fast_acc, fast_dec, fast_vel)

    with open(full_path, "w", encoding="cp949") as f:
        f.write(",".join(["순서", "함수", "위치", "위치", "위치", "위치", "속도", "속도", "속도", "속도", "IO"]) + "\n")
        f.write(",".join(["순서", "함수", "X", "Y", "Z", "U", "시작", "가속", "감속", "구동", "IO"]) + "\n")
        for row in node_data:
            f.write(",".join(map(str, row)) + "\n")

    presses = repeats * (n + 1) + 1
    est_min = presses * (2 * (depth_mm + safe_lift) / press_vel + hold_toggles + rest_toggles + 2) / 60
    print(f"{full_path}")
    print(f"  {axis}축 ±{span_mm} @ {step_mm}mm = {n}점 × {repeats}라운드(랜덤) + 기준점 {repeats+1}회"
          f" = 압입 {presses}회, {len(node_data)}행, 예상 ~{est_min:.0f}분")
    print(f"  z: safe {safe_z} → contact {contact_z} → press {press_z} (깊이 {depth_mm}mm, 복귀 1mm/s)")
    return full_path


if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    # 코스 확인용(먼저 실행 권장): 1mm 스텝 5점 × 2라운드 — 추종(slope~1) 확인
    generate(out, name="SATS_d10_spec_coarse", span_mm=2.0, step_mm=1.0, repeats=2)
    # 파인 본측정: 0.1mm 스텝 31점 × 3라운드
    generate(out, name="SATS_d10_spec_finestep", span_mm=1.5, step_mm=0.1, repeats=3)
