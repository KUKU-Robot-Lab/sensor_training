# SATS v6 Real-time Demo — Sensor & Inference Spec (D10)

전시/학회 데모용 스펙 시트. **모든 수치는 지름 10 mm 인덴터(D10) 기준.**
표는 포스터에 그대로 쓸 수 있게 영문 라벨로 정리. 주석은 한글.

> ⚠ `[measure@4090]` 표시 항목은 gitignore된 eval 산출물·GPU가 4090에만 있어
> `scripts/measure_demo_spec.py` 를 4090에서 실행해 채운다(추정치 날조 금지).

---

## 1. Raw sensing unit (per sensor)

| Item | Value | Source |
|---|---|---|
| Taxels | 16 (4×4) | vensor2.ino |
| Sampling rate | **200 Hz** | firmware |
| MCU link | Arduino **Due**, binary `0xAA … 0x55` | serial_reader |
| Baud rate | **250000** | vensor2.ino `Serial.begin(250000)` |
| Transducer | BMP384 absolute pressure | — |

## 2. SATS super-resolution output (D10)

| Item | Value | Source |
|---|---|---|
| Output map | **41 × 41** | deploy config `grid_size` |
| Grid pitch | **0.5 mm** | config `grid_step_mm` |
| Sensing area | **20 × 20 mm** (−10 … +10 mm) | config `grid_min/max_mm` |
| Super-resolution | taxel pitch **~6.5 mm → 0.5 mm map** (≈13× linear, 105× cells) | 16 → 1681 |
| Window size | 10 frames | config `window_size` |
| Size conditioning | ON, fixed **d = 10 mm** | `use_indenter_size_input` |
| Inference throughput | **129 Hz** (7.75 ms/frame, RTX GPU) | measured@4090 |
| Demo output rate | cap **20 Hz** (`--infer-max-fps`, 상향 가능) | run_demo |

## 3. Force / position ranges & resolution (size-conditioned, D5 & D10)

**★ 위치 분해능 — 모터 실측 확정(2026-08-08)**: 0.1μm 스테이지로 0.1 mm 스텝 31점을
랜덤 순서·3회 반복 압입(D10, 깊이 1.5 mm)해 측정(`scripts/analyze_localization_resolution.py
--v2-dir`, 데이터 `skin_ws/raw_data/v6_finestep_2`, 모델 `ecomesh_v6_deploy_g025`).

| 측정 항목 | 실측값 | 비고 |
|---|---|---|
| 반복 노이즈 σ | **0.028 mm** | 같은 점 반복 예측의 흔들림 |
| 0.1 mm 스텝 구분 성공률 | **40 %** | 인접 위치가 2σ를 넘어 구분되는 비율 |
| **실효 분해능 (플래토 폭)** | **median 0.20 mm, worst 0.50 mm** | ★ 실제로 분간 가능한 최소 간격 |
| 추종 gain | **1.80** | 모터 1 mm 이동 → 예측 1.8 mm (과대응답, §3c 보정) |
| 직선 잔차 RMS | 0.48 mm | 계통 비선형 |

- **σ(0.028 mm)를 분해능으로 읽으면 안 된다**: 응답이 계단형이라 플래토 안에서 안정적일 뿐
  위치를 구분하지 못한다. 정직한 스펙은 **실효 분해능 0.2 mm**.
- **★라벨 간격은 한계가 아니다**: 학습 라벨은 xy **1 mm** 격자인데 실효 분해능은 **0.2 mm** —
  라벨보다 **5× 미세**하다. 즉 warm-start(xy0.5 사전학습) + 보간이 의도대로 작동해
  라벨 사이를 채웠다는 실측 증거이며, "1 mm 라벨이 1 mm 한계를 강제한다"는 성립하지 않는다.
  (데이터 효율 전략 d5-2·d10-2 는 전이 목적에서 이미 검증됨 — 해상도는 독립적인 축.)
- **★★출력 격자도 한계가 아니다 — 물리 한계 확정(2026-08-08)**: 동일 v6 센서·동일
  finestep_2 데이터를 **세 해상도 모델**로 재분석한 결과, 격자를 **5× 미세화해도 실효
  분해능이 0.20 mm로 불변**:

| 모델 (출력 격자) | 실효 분해능 | 0.1 mm 구분률 | σ (mm) | gain | 잔차 RMS |
|---|---|---|---|---|---|
| `deploy_all4` (0.5 mm, 41²) | **0.20 mm** | 37 % | 0.074 | 1.55 | 0.345 |
| `deploy_g025` (0.25 mm, 81²) | **0.20 mm** | 40 % | **0.028** | 1.80 | 0.480 |
| `deploy_g01` (0.1 mm, 201²) | **0.20 mm** | 33 % | 0.037 | 2.00 | 0.614 |

  → 남은 0.2 mm 계단은 **물리 정보량 한계**(taxel 16 개, 엘라스토머 PSF 확산 폭, SNR)로
  **확정**. 격자·라벨을 아무리 미세화해도 넘지 못한다. 이는 claim boundary
  "output-grid expansion ≠ 물리 분해능"의 **정량적 실증**이다.
- **교차 확인(1 mm 스텝, `v6_coarse_2`)**: 세 해상도 모두 1 mm 스텝을 구분(0.5/0.25 mm =
  100 %, 0.1 mm = 75 %). 1 mm ≫ 0.20 mm 한계이므로 여유 구분 — 위 결과와 **일관**.
- **부수 관찰**: 격자가 미세할수록 gain(과대응답)과 잔차가 **오히려 커진다**(1.55→2.00,
  0.345→0.614). 반복 노이즈 σ는 **0.25 mm에서 최소**. 0.1 mm 모델은 1 mm 스텝조차
  75 %만 구분(worst 플래토 2 mm)해 **실용상 열등**. → 실용 권장 **0.25 mm**(σ 최소·
  분해능 동일), 계산 절약이 필요하면 0.5 mm(gain 오차 최소).
- ⚠ 기존 기록의 "위치 median 0.50(0.5 mm)→0.22 mm(0.1 mm) 개선"과 모순이 아니다. 그것은
  **라벨 대비 argmax 오차**(격자 양자화에 직접 좌우)이고, 여기 0.20 mm는 **모터 스텝을
  실제로 구분하는 능력**(격자 무관)이다. 서로 다른 지표이며 보완적이다.
- **taxel pitch ~6.5 mm → 0.2 mm 분간 = 약 32× super-resolution**.
- 측정 조건 한계: **x축 ±1.5 mm 구간·D10·깊이 1.5 mm**. 전면(±10 mm) 지도·y축은 미측정.

| Quantity | Range | Resolution | Source |
|---|---|---|---|
| **x, y** | −10 … +10 mm | **실효 0.20 mm** (σ 0.028 mm, grid 0.25/0.5 mm) | motor sweep measured |
| **z (D10)** | **0.48 … 2.0 mm** | LUT (peak→z), r² = 0.59 (coarse) | z_calibration_v6.json `"10.0"` |
| **z (D5)** | 0.72 … 2.0 mm | LUT, r² = 0.87 (더 신뢰) | z_calibration_v6.json `"5.0"` |
| **Fz (D10)** | **0 … 3.9 N** | 무접촉 노이즈 플로어 **p99 0.003 N** (사실상 0) | measured@4090 |
| **Fz (D5)** | 0 … 1.5 N | 노이즈 플로어 measured@4090 | GT-integral upper |

- **Fz 노이즈가 사실상 0**(p99 0.003 N) → 필터 임계는 *노이즈*가 아니라 *의미있는 접촉*
  기준(가벼운 손가락 ~0.2 N)으로 잡음: 기본 `fz_on 0.20 / fz_off 0.10`.
- **다중접촉(2·3)은 D5 권장**: D10 은 min-distance 10 mm 라 가까운 접촉이 병합됨(`[d]` 토글).
- Fz = Σ(clip≥0 pred) × taxel_area / 100 [N] (해상도 불변 적분, `inference_engine.get_fz`).

## 3c. Position gain correction (과대응답 보정)

실측 gain **1.80** = 접촉이 실제보다 중심에서 멀게 추정됨. 역보정:
`s = c + (p − c)/gain` (`demo_contacts.apply_position_gain`, 중심 c=(0,0)).

- 활성화: `--position-gain 1.8` (run_demo / run_dashboard). **기본 1.0 = off.**
- ⚠ 기본 off 인 이유: "중심에서는 정확하다"는 가정이 **x축 ±1.5 mm 한 구간에서만**
  검증됨. 전면 2D 격자로 gain map을 측정하기 전에는 전역 적용이 오히려 왜곡을 만들 수 있다.
- 보정해도 **분해능(0.2 mm)은 개선되지 않는다** — gain 은 계통 스케일 오차만 줄인다.

## 3b. Visualization filtering (contact_filter.py)

무접촉/접촉 안정화 — 임계는 위 분해능·노이즈 플로어로 튜닝:
- **히스테리시스**: 총 fz ≥ `fz_on`(0.30 N) → ON, < `fz_off`(0.15 N) → OFF (깜빡임 방지).
- **디바운스**: `on_frames`(2)/`off_frames`(5) 연속 만족 시 전환.
- **무접촉 리셋**: OFF 확정 시 표시 blank + 트랙 초기화(release).
- **위치 스무딩**: `pos_smooth`(3) median — 분해능 한계(≈0.5–1 mm) 내 떨림 제거.

## 4. Bending angle (θ) — jig curvature (D10)

| Item | Value | Source |
|---|---|---|
| Observable range | **~20 … 150°** (유효 관측창) | memory bending-observability |
| Accuracy (G1 leave-one-out) | **MAE 1.78°**, ρ ≈ 1.0 | memory v6 G1 |
| Per-frame resolution | std **~0.4 … 1.0°** | memory theta resolution |
| Dead-band | |θ| < 20° → 0 (관측 불가 → flat) | run_demo `--theta-deadband` |

## 5. Multi-sensor demo (3 sensors, one desktop)

- 통합 대시보드 1창(`run_dashboard.py`), 3열 패널: **contacts | theta | bending→SATS**.
- SATS 엔진 1개 공유(predict는 상태무관) → GPU 1회 로드로 3센서 처리.
- 포트 고정: **`/dev/serial/by-id/*`** symlink 사용(ttyACM 번호 재연결마다 뒤바뀜 방지).
- 부하: 소형 CNN 3-추론 + 3 × 200 Hz × 16 taxel @ 250000 baud(각 USB) → 여유.

---

_생성: `sats/inference/SPEC.md`. 수치 갱신은 `scripts/measure_demo_spec.py`(4090) 재실행._
