# G1 밴딩 취득 설계 (스테이지 압축 방식, 2026-07-28)

> 마스터 게이트: Notion "Bending-Aware Tactile Skin — NatComm" **G1 — Curvature Observability**.
> 이 문서는 **검증된 스테이지 압축(원호 좌굴) 방식**으로 G1을 충족하기 위한 취득 스펙이다.
> 전처리·곡률 정의·유효범위는 `sats/preprocessing/prepare_bending_data.py`, `sats/bending/geometry.py`.

## 배경 — 방식 확정과 지그 방식과의 차이

- 기존 G1/README는 **지그 각도 눈금(±40°, 10° 간격)** 을 GT로 전제했다.
- 실제 취득은 **양단 고정 + 스테이지 Y 압축 → 원호 좌굴**(L=35mm). δ(=Y 압축량)로부터
  곡률/각도를 계산: `(L−δ)/L = sinc(θ/2)`, `κ=θ/L`. **곡률이 연속량**이라 지그의 이산 각도와 다르다.
- **센서 자가추정 κ̂ 로 교차검증됨**: valid **δ≤10mm 에서만** κ̂↔κ_geo R²=0.82(δ>10mm 포화·붕괴).

### ★ 각도 정의·범위 정합 (필수)

원호 subtended 각 θ 와 지그 "tilt 각"은 다르다. δ↔각 대응(L=35):

| δ (mm) | κ (1/mm) | R (mm) | θ subtended | θ/2 end-tilt |
|---|---|---|---|---|
| 2 | 0.034 | 30 | 68° | 34° |
| 4 | 0.045 | 22 | ~92° | ~46° |
| 6 | 0.056 | 18 | 108° | 54° |
| 8 | 0.067 | 15 | ~130° | ~65° |
| **10 (상한)** | **0.076** | **13** | **157°** | **78°** |

- **결론**: 지그 "±40°"는 **end-tilt θ/2 기준 δ≈2.8mm** 에 해당. 우리 유효범위(δ≤10)는
  그보다 **훨씬 큰 곡률**까지 커버한다. → G1의 각도 축을 **곡률 κ(1/mm) 또는 δ(mm)** 로
  재정의하고, 보고 시 θ·θ/2·R 를 함께 표기(정의 명시). "±40°" 문구는 **곡률 연속 스윕**으로 대체.
- 라벨 축은 **signed κ** (Y>0=+, Y<0=−). 학습·평가·정렬 판정 모두 signed κ 기준.

## G1 충족 취득 스펙

### 1) 곡률 스윕 (지그 이산각 → 연속 δ)
- 각 세션에서 **δ: 0 → +10mm → 0 → −10mm → 0** 왕복(양방향 signed). **속도 저속**(준정적).
- 유효 상한 **δ=10mm 엄수**(포화 회피). 넘겨 취득 시 valid 마스크로 자동 배제되나 낭비.
- 무접촉(pure bending)만. Z는 접촉 안 되게 안전높이 고정(원점이동 값 무시).
- 곡률 격자 재현성 위해 **정지-홀드 스텝** 권장: δ = 0,2,4,6,8,10mm 각 지점 **홀드 ≥1s**,
  지점당 유효 프레임 확보(지그의 "각도별 10 set" 대응 = **δ 지점별 반복**).

### 2) Remounting ≥3 세션 (G1 핵심)
- 센서를 **탈거→재장착** 하여 **독립 3세션 이상**. 세션마다 위 스윕 1회 이상.
- 세션 간 클램프 재고정으로 **L·z_start 미세변동** 발생 → 이게 바로 G1이 요구하는
  mounting 변동 강건성. 세션별 L 실측 기록(달라지면 `--bend-length` 로 반영).

### 3) Held-out 평가 (2축)
- **Remounting-held-out**: 한 세션을 통째로 홀드아웃(현재 `--val-trials` 로 이미 지원).
- **Intermediate-δ holdout**: 학습에 안 쓴 중간 δ(예: 3,5,7,9mm)에서 평가(선택).

### 4) 관측성 판정 기준 (통과 조건)
- **signed κ 단조 정렬 유지**: 예측 κ̂ 가 δ 부호·크기와 단조. flat/bent **이진 분류로 붕괴 금지**
  (연속 회귀 R² 및 순서상관 ρ 보고).
- **remounting-held-out 세션**에서 곡률(또는 θ) 회귀 MAE 가 기준선(평균예측) 대비 유의 개선.
  현재 단일세션 예비치: val θ MAE ≈ 6.3°(기준선 29.9°). G1은 **remounting holdout**에서 재확인.
- **온도/drift 통제**: 세션마다 온도·시각 기록, 무접촉 baseline drift 가 κ̂ 를 설명하지 않음을 확인
  (δ=0 반복 프레임의 κ̂ 분산 < 곡률 신호).

### 5) 로깅 (G0 provenance)
세션별 필수 기록: sensor ID, mounting session #, L(mm), z_start, 온도, 취득 시각/retare,
δ 스윕 프로파일, 부호. trial_registry 유사 관리(밴딩용 registry 별도 권장).

## 산출·연결
- 취득 raw → `skin_ws/raw_data/bending/v1/{session}_testN` (v0 는 단일세션 예비).
- 전처리: `prepare_bending_data.py --raw-root ... --bend-length <세션 실측 L>` (양방향 지원 확장 필요:
  현재 (+)push만 → **±push 세그먼트로 확장** = TODO).
- 학습·평가: `sats.bending.train_bending --val-trials <holdout 세션>` (valid-only 기본).

## ★ v5 실측 — G1 통과 (2026-08-02)

첫 다세션 remounting 취득(ecomesh v5)으로 **G1 정식 통과**. 취득 rig가 v0와 달라짐:

### v5 취득 구조 (앞으로의 표준)
- raw: `skin_ws/raw_data/bending/v5/{0.1mm,2mm}/<session>` — **0.1mm(11세션)·2mm(10세션)** = δ 스텝 granularity. loadcell·afd50 추가 수집(전처리는 due+ether만 사용).
- **δ 정의 (핵심, v0와 다름)**: Z를 먼저 12mm까지 내린 뒤, **Y=18mm가 flat(δ=0), Y=33mm가 최대 밴딩** → **δ = Y − 18mm** (범위 0~15mm). Y는 절대 스테이지 위치(0.1mm세션 0→33, 2mm세션 18→33).
- **각 세션 = 독립 remounting**(탈착→재장착) — 사용자 확인.
- 전처리: `prepare_bending_data.py --raw-root .../v5/0.1mm --out-dir learning_data/bending/v5 --y-origin-mm 18 --z-active-min-mm 11.5 --delta-max-valid 15 --bend-length 35`. (신규 인자 `--y-origin-mm`·`--z-active-min-mm`, 세션 탐색은 due+ether CSV 존재로 판별 = 명명 무관.)
- **δ≤15mm 포화 없음**(v0는 10mm): κ̂↔κ_geo R²=0.877 (전 구간).

### G1 판정 결과 (leave-one-session-out, 11 세션)
- **평균 θ MAE = 2.97°** (기준선 24.8°), **Spearman ρ = 1.000**, 세션별 선형 R² ≈ 0.98.
- ⇒ (a) remounting-holdout 회귀 개선 ✅ (b) signed 순서 보존 ✅ (c) 이진 붕괴 없음(연속 R²) ✅ (d) ≥3 세션 ✅. **G1 Curvature Observability 통과.**
- 코드: `sats.bending.eval_g1 --data-dir learning_data/bending/v5 --epochs 60`. 배포 estimator: `sats/bending/runs/estimator_v5`.
- 잔여: drift 지표 nan(홀드아웃 세션의 |θ|<10° 프레임 희소) — 온도/drift 통제는 별도 로그로 보완 권장. 또 11세션 모두 동일 δ 스윕이라 "unseen intermediate 곡률"은 자명 커버 = **remounting 강건성**이 실증의 핵심(더 어려운 조건 통과).

## 현 코드 대비 TODO (G1 취득 전 보완)
1. ~~**양방향(±) 세그먼트**~~ ✅ **완료(2026-07-28)**: `select_segment(direction=positive/negative/both)` +
   signed κ·θ·κ̂. 단 **v0 both 실측: 음 브랜치 품질 낮음**(test3 음 R²=0.03, signed estimator val 불안정
   best 12.5°→발산) — v0는 대칭 스윕 아님. **신뢰 가능한 signed 데이터는 아래 대칭 취득 필요**.
2. **밴딩 trial registry**: 세션·L·온도 메타 고정 관리.
3. **δ 지점-홀드 세그먼트 옵션**: 준정적 홀드 구간 우선 추출(반복 신뢰도↑).
4. Notion G1 스펙 문구 갱신: "±40° 지그" → "signed κ 연속 스윕(δ≤10mm), 각 정의 명시".
