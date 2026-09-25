# common/ — 두 패키지가 공유하는 기반

의존 방향은 한쪽뿐: `robot_skin → common ← deformable_sats`. `common` 은 numpy·PyYAML 만 쓰고
`robot_skin`·`sats`·`hitmap` 을 import 하지 않는다 (`tests/test_dependency_direction.py` 가 강제).

| 모듈 | 내용 |
|---|---|
| `signal.py` | `relative_change`: ΔS% = (raw − baseline)/baseline × 100 (**`sats/training/dataset.py:248` 과 동일 — 누르면 음수**), `PRESS_SIGN = −1`, `press_intensity`(= −ΔS, 누르면 양수), `estimate_baseline`(초기 무접촉 구간 median), `NormStats`(offset/scale, fit·apply·invert·json), `saturation_mask`(ADC 레일 + \|ΔS\| 상한) |
| `timeline.py` | `Stream`, `align_streams`: 여러 타임스탬프 스트림 → 마스터 클럭(200 Hz), linear / zero-order hold, intersection·union span |
| `layouts.py` + `layouts/*.yaml` | taxel 레이아웃 스키마(id, channel, parent, position, normal, groups, imu_sites), `load_layout`, `grid_layout` |

내장 레이아웃: `sats_4x4`(4×4, 6.5 mm, ±9.75 mm, SATS S1..S16 순서) · `glove_template`(손끝 5 + 손바닥 2×2,
parent = MANO 세그먼트, IMU 7개) · `robot_hand_template`(parent = URDF 링크, 이름은 실제 URDF 로 교체).

mk555 raw `.bin` 파서는 여기 없다 — 정본은 `deformable_sats/sats/preprocessing/bin_merge.py`.
