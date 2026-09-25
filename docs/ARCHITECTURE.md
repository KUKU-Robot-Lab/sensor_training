# Architecture — 촉각 글러브 데이터에서 로봇 제어까지

이 문서는 `robot_skin` 의 **왜**와 **어떻게 연결되는가**를 다룬다. 실행 방법은 [`TRAINING.md`](TRAINING.md),
정책 모델은 [`VTLA.md`](VTLA.md), 데이터 수집은 [`DATA_ACQUISITION.md`](DATA_ACQUISITION.md), 파일 포맷은
[`DATA_FORMAT.md`](DATA_FORMAT.md), 로봇 실행은 [`DEPLOYMENT.md`](DEPLOYMENT.md), 인용 근거는
[`REFERENCES.md`](REFERENCES.md) 에 있다. 모듈별 코드 지도는 [`robot_skin/README.md`](../robot_skin/README.md)
와 각 모듈의 `README.md` 다. 문서와 코드가 다르면 **코드가 기준**이다.

---

## 1. 목표

기압 taxel(mk555 계열) 글러브 + IMU 7개 + 카메라로 사람 손 데이터를 모으고, 그것으로 **vision + tactile +
language → action** 정책(VTLA)을 학습해 촉각 스킨을 붙인 로봇 핸드를 제어한다. 데이터는 두 종류다.

| 데이터셋 | 내용 | 무엇을 학습하나 |
|---|---|---|
| **D1 `motion`** | 물체 없이 손을 움직인다. 관절별 굽힘, 쥐었다 펴기, 손목 회전, 자유 동작, 물체 없는 쥐기 손모양(`air_grasp_*`), self-touch 세트(엄지–손가락 핀치, 주먹 등). 비전·IMU·촉각을 함께 기록 | ① IMU → 손 자세 모델, ② **움직임 때문에 생기는 무접촉 ΔS**(motion artefact) 예측기, ③ 접촉 판정 보정. self-touch 는 손 자세만으로 자동 라벨이 나오는 "공짜" 접촉 라벨이다 |
| **D2 `task`** | 물체 과제(잡아 옮기기, 붓기, 끼우기, 닦기 …)와 언어 지시문. 한 과제 반복 = 에피소드 하나 | VTLA 정책. D1 에서 학습한 모델로 촉각을 해석한 뒤 입력으로 쓴다 |

최종 사용처는 로봇 핸드다. 정책은 사람 손 행동(MANO)을 예측하고, 배포 때 로봇 관절로 리타게팅한다(§5.6).
로봇 원격조작 데이터가 생기면 로봇 관절 행동 공간(`robot_joint`)으로 바로 학습할 수도 있다.

## 2. 저장소 구조와 의존 방향

```
robot_skin  ──▶  common  ◀──  deformable_sats
```

- `common/` — 두 패키지가 공유하는 규약: ΔS% 신호·baseline·정규화·포화(`signal`), 스트림 시계 정렬
  (`timeline`), taxel 레이아웃(`layouts`). numpy·PyYAML 만 쓴다.
- `robot_skin/` — 이 문서의 대상. `common` 만 import 한다.
- `deformable_sats/` — 기존 평면 4×4 SATS 패드 연구 저장소 전체(sats, hitmap, …). 내부 import 가 저장소 루트
  기준 경로에 묶여 있어 **통째로 한 단계 아래로 옮기기만** 했다. 기존 명령은 `cd deformable_sats` 후 그대로 쓴다.

강제 규칙 (`common/tests/test_dependency_direction.py`): `common` 은 어느 쪽도 import 하지 않고, `robot_skin` 은
`sats`/`hitmap`/`cnn_lstm`/`deformable_sats` 를 import 하지 않는다. `deformable_sats` 의 결과(bending restorer,
대시보드 격리 규칙)는 **아이디어로** 가져와 일반화했고(§5.3, `contact/saturation_fsm.py`), mk555 `.bin` 파서의
정본은 `deformable_sats/sats/preprocessing/bin_merge.py` 로 남긴다(복사 금지, 로더 콜러블로 주입).

`robot_skin` 안에서도 방향이 있다: 아래 계층(geometry, pose, datasets, contact, representation …)은 위 계층
(stages, control, `__main__`)을 모른다. `import robot_skin.datasets`, `robot_skin.contact`, `robot_skin.control`,
`robot_skin.acquisition` 은 torch 를 즉시 불러오지 않는다(PEP 562 지연 export) — 수집 PC 에서 가볍게 쓰기 위해서다.

## 3. 전체 데이터 흐름

```
[수집]   acquisition: protocol (d1_motion | d2_task | robot_sweep) → Recorder (pressure, imu, camera_<name>,
         joint_state, events.jsonl) → 3-탭 싱크 → IMU 평손 보정 → QC            (DATA_ACQUISITION.md)
         오프라인 손 라벨: HaMeR / WiLoR → hand_pose.npz        합성 세션: datasets.synthetic (`synth`)
    │
    ▼  raw 세션  data/raw/<dataset>/<subject>/<session_id>/                        (DATA_FORMAT.md §1)
[전처리] datasets.build: 채널 → 레이아웃 순서, 200 Hz 마스터 시계, baseline → ΔS, 포화, IMU 보정 적용,
         손 라벨 평활, q / q̇ (인과 Savitzky–Golay), taxel pose (손 프레임), phase, self-touch, contact_label
    │
    ▼  Episode  data/processed/<dataset>/<episode_id>/ (arrays/, static/, derived/)  (DATA_FORMAT.md §2)
         + splits.json (datasets.splits, 기본 subject 단위) — 모든 stage 가 같은 파일을 쓴다
    │
[stage 1: 촉각 해석 — D1 로 학습, 모든 episode 에 derived 기록]
    ├─ imu_pose   IMU 창 → MANO 손가락 자세                    → derived/hand_finger_pose_imu
    ├─ baseline   q·q̇ 이력 + taxel pose → 무접촉 ΔS 평균·분산   → derived/baseline_pred, baseline_logvar,
    │                                                             residual = ΔS − 평균
    └─ contact    ResidualCalibrator → residual_z, contact_level;  ContactDetector → contact_prob;
                  D2 pseudo 라벨 → contact_label_pseudo;  파일: calibrator.json, contact_detector.pt
    │
[stage 2: 촉각 표현]  pretrain   tactile_value_features(z, level, sat) → TaxelEncoder,
                                 MAE 식 taxel 마스킹 재구성 → encoder_state.pt
    │
[stage 3: 정책]       vtla       D2 policy tick: 언어 + 카메라 + 촉각 토큰 (ContactGate) + proprio → action chunk
                                 → policy_bundle.pt (가중치·정규화기·특징 스펙·이미지 변환·타이밍·stage-1 참조)
    │
    ▼
[배포]   control / stages.deploy: OnlineTactileProcessor (= stage 1 을 틱 단위로) → PolicyRunner
         (20 Hz 추론, ACT 시간 앙상블, 200 Hz 제어) → hand_mano: FingertipRetargeter → SafetyFilter → 로봇
         └─ DeploymentLogger: 실행 기록을 raw 세션 포맷으로 → datasets.build 로 재투입 (dataset `other`)

곁가지:  transfer (글러브 ⇄ 로봇 taxel 대응, 로봇 q → MANO 역추정 = deploy 의 hand_state.estimate)
         sim (TaxelDomainRandomizer 구현; TouchGridEnv·RL 학습은 스텁) · eval (지표) · train (모든 stage 의 학습 엔진)
```

stage 들은 **파일과 episode 의 derived 배열로만** 연결된다. `python -m robot_skin pipeline` 이 순서
(imu_pose → baseline → contact → pretrain → vtla), 공유 splits, 산출물 연결을 맡는다([`TRAINING.md`](TRAINING.md) §3).
어떤 모델이 derived 배열을 썼는지는 episode 에 기록되지 않는다 — 파이프라인이 순서와 재실행 규칙(앞 stage 가 다시
돌면 뒤 stage 도 다시)으로 일관성을 보장하고, `<runs>/pipeline.json` 에 실행 기록을 남긴다.

| stage | 학습 데이터 | 주 산출물 (`<out>/<stage>/`) | 에피소드에 쓰는 derived 배열 |
|---|---|---|---|
| `imu_pose` | 손 라벨 있는 D1 (`data.datasets: [motion]`) | `imu_pose_model.pt`, `imu_stats.json` | `hand_finger_pose_imu` |
| `baseline` | D1 무접촉 프레임 (`contact_label == 0`) | `baseline_model.pt`, `joint_stats.json` | `baseline_pred`, `baseline_logvar`, `residual` |
| `contact` | D1 val 무접촉(보정) + D1 self-touch/무접촉 라벨(검출기) | `calibrator.json`, `contact_detector.pt` | `residual_z`, `contact_level`, `contact_prob`, D2 `contact_label_pseudo` |
| `pretrain` | D1 + D2, 라벨 없음 | `encoder_state.pt` | — |
| `vtla` | D2 과제 phase (reach…retreat) | `policy_bundle.pt` | — (선택: `vision_<key>_<cam>.npy` 특징 캐시) |
| `deploy` | — | `metrics.json`, `sessions/<id>/` (raw 세션) | — |

## 4. 두 가지 손: 글러브와 로봇 핸드

같은 코드가 글러브(사람)와 로봇 핸드를 모두 다룬다. 달라지는 것은 taxel 의 **부모 프레임 자세를 어디서 얻는가**뿐이다.

| | 글러브 | 로봇 핸드 |
|---|---|---|
| 레이아웃 부모 | MANO 세그먼트 (`common/layouts/glove_template.yaml`) | URDF 링크 (`robot_hand_template.yaml`) |
| 관절 상태 `q` | MANO 손가락 자세 45-D (비전 라벨, 또는 `imu_pose` 출력) | URDF 구동 관절 (`joint_state.npz`) |
| taxel pose | `pose.mano.taxel_poses_from_hand` (go = 0, 손목 원점) | `pose.robot_fk.taxel_poses_from_joints` (URDF 루트) |
| self-touch 라벨 | 캡슐 거리로 자동 (`self_touch_from_hand`) | 없음 (`contact` 의 `detector.bootstrap` 이 대신) |

taxel 은 채널 번호가 아니라 **3D pose(위치·법선)** 로 토큰화되므로(3D-ViTac 방식) 두 손이 같은 촉각 인코더를
쓸 수 있다. 레이아웃 간 대응이 필요하면 `transfer.align_layouts` / `map_taxel_values` 를 쓴다(손가락 그룹 안에서,
캡슐 골격 좌표로 매칭).

## 5. 핵심 설계 결정

### 5.1 ΔS 부호 규약 (SATS)

`ΔS% = (raw − baseline) / baseline × 100` (`common.signal.relative_change`, `deformable_sats/sats/training/dataset.py`
와 비트 단위로 같다). mk555 기압 taxel 은 **누르면 raw 가 줄어서** 접촉 ΔS 는 **음수**다(드롭아웃은 −100 % 쪽).
기존 SATS 모델에 그대로 넣을 수 있도록 부호를 뒤집지 않는다. 대신 `PRESS_SIGN = −1`, `press_intensity(ΔS) = −ΔS`
를 두어 접촉 쪽 로직(임계값, `residual_z`, 레벨)은 **누르면 양수**로 다룬다. 저장된 배열 중 `delta_pct`,
`baseline_pred`, `residual` 은 SATS 부호, `residual_z` 는 누름 양수다.

### 5.2 taxel pose 는 손(로봇 base) 프레임

글러브 taxel 위치·법선은 전처리에서 `global_orient = 0`, 손목 = 원점으로 계산한다
(`meta.preprocessing.taxel_frame = mano_wrist`, 버전 `robot_skin.datasets.build/2`). 로봇은 URDF 루트 프레임이다.
이유: 월드 프레임이면 합성 D2 에피소드에서 taxel 중심이 수십 cm 움직이고, 위치 특징이 "손이 방의 어디에 있나"를
인코딩하게 된다. 그러면 로봇 base 프레임 자세와 맞지 않는다. 월드 좌표가 필요한 곳(손–물체 근접 veto)은
`R(hand_global_orient)·p + hand_wrist_pos` 로 되돌린다(`contact.pseudo_label.taxel_world_positions`). 온라인
처리기(`control.online.glove_pose_fn`)도 같은 규약이다. build/1 로 만든 에피소드는 `--force` 로 다시 만들어야 하고,
stage 들은 `taxel_frame` 이 없는 글러브 에피소드에 경고한다.

위치 Fourier 특징의 기본값(최저 옥타브 주기 0.3 m, 6 옥타브 → 가장 짧은 주기 ≈ 9 mm)도 같은 이유로 정했다: 손 크기를
덮되 mm 단위 자세 라벨 잡음보다 짧은 주기는 쓰지 않는다(`test_fourier_pose_defaults_are_consistent_and_above_pose_noise`).

### 5.3 관측 ΔS 는 baseline 모델 입력에 절대 넣지 않는다

손 위의 taxel 은 접촉이 없어도 관절이 굽고 피부가 늘어나면 ΔS 가 변한다. `baseline` 은 이 무접촉 ΔS 를
**운동학 변수만으로** 예측한다: `TemporalBaselinePredictor.forward(q_hist, qd_hist, pos, nrm)` — ΔS 인자가 없고,
`test_observed_delta_is_never_an_input` 이 시그니처를 고정한다. 이것은 `deformable_sats/sats/bending/` 의 deg→offset
restorer 를 일반화한 것이다(조건 변수가 밴딩각 1개 → 관절 이력 + taxel pose). 거기서 얻은 교훈: 관측 ΔS 를 입력으로
보면 모델이 접촉까지 오프셋으로 학습해 **지운다**. 평균 헤드는 0-초기화라 학습 전에는 residual = 관측 ΔS 다.

모델은 인과적이다: 프레임 t 의 예측은 `q[t−W+1…t]`, `q̇[t−W+1…t]` 와 `pos[t]`, `nrm[t]` 만 본다(기본 W = 32,
TCN 수용 영역 31). 지연·속도에 따라 달라지는 아티팩트를 표현하기 위해서다. `q̇` 도 인과 Savitzky–Golay 미분이다.

### 5.4 heteroscedastic baseline + 보정된 z

baseline 은 평균과 함께 **입력에 따라 달라지는 분산**(log σ²)을 예측한다(Gaussian NLL, Kendall & Gal 2017).
기본 손실은 평균 = target-scale MSE, 분산 = stop-gradient NLL 이다(공동 NLL 은 평균 수렴이 크게 느렸다 —
`baseline/README.md`). `contact.ResidualCalibrator` 가 D1 **val** 무접촉 프레임에서 taxel 별 중심 c, robust σ
(MAD·1.4826 중 예측 분산이 설명하지 못한 부분), 이득 g(보정 프레임에서 z 의 MAD-σ 가 1 이 되게)를 맞춘다
(`contact/calibration.py` docstring):

```
p = press_intensity(residual) − c = −residual − c        (중심을 뺀 누름 %, taxel 별)
z = p / (g · sqrt(σ² + exp(baseline_logvar)))
WEAK   ⇔ z ≥ weak_z (3)   ∧ p ≥ weak_floor_pct (0.5 %)
STRONG ⇔ z ≥ strong_z (8) ∧ p ≥ strong_floor_pct (3 %)
포화 / SaturationFSM 미신뢰 → SATURATED
```

그래서 임계값이 taxel 마다, 자세마다 달라진다(모델이 불확실한 자세에서는 z 가 작아진다). % 하한은 σ 가 아주 작은
taxel 이 0.05 % 흔들림을 접촉으로 보고하지 못하게 막는다. σ·c·g·임계값·FSM 설정은 `calibrator.json` 하나에 들어가
온라인 처리기가 그대로 읽는다. 그 위에 학습형 `ContactDetector`(taxel 별 인과 z 이력 + 관절 속도 요약, focal loss)와
`HysteresisFilter` 가 있다. 한계는 §8: 분산은 aleatoric 만 표현한다.

### 5.5 ContactGate — 접촉이 없으면 촉각 토큰은 0

VTLA 에서 taxel 토큰은 Perceiver 식 쿼리 K 개로 압축된 뒤 `ContactGate` 를 지난다. `hard`: 샘플에 접촉 taxel
(`contact = level ≥ WEAK`, `data.contact_rule`)이 하나도 없으면 K 개 토큰 전체를 0 으로 만든다. `soft`: 거기에
`σ(w·접촉비율 + b)` 를 곱한다(무접촉 0 은 유지). 드리프트나 남은 아티팩트가 WEAK 문턱을 넘지 못하면 융합
트랜스포머에 **아예 들어가지 못한다**(상수 modality embedding 만 남는다). 촉각 환각을 구조로 막는 장치다.
주의: 기본 규칙은 SATURATED 도 접촉으로 센다 — 항상 포화인 죽은 채널이 있으면 게이트가 늘 열리므로 `taxel_pad`
로 가리거나 `weak_or_strong` 을 쓴다.

### 5.6 정준 행동 = 사람 손(MANO) + 리타게팅

정책의 기본 행동은 로봇과 무관한 **`hand_mano` 54-D**: 손목 위치 3 + 손목 회전 6D(회전행렬 첫 두 열, Zhou et al.
2019) + 손가락 15 관절 axis-angle 45 (MANO 관절 순서). D2 사람 시연에서 바로 계산되고, ACT 식 청크(H 스텝)로
예측한다. 배포 때 `action.retarget.FingertipRetargeter` 가 손끝 벡터(손목→손끝, 엄지→각 손가락)를 로봇 FK 에 맞춘다
(DexPilot / AnyTeleop 방식, projected Levenberg–Marquardt). 로봇 모델은 FK 콜러블 또는 URDF 만 바꿔 끼우면 된다.
장점: 한 데이터셋으로 여러 로봇 핸드를 지원한다. 비용: 리타게팅 오차, 그리고 로봇에는 사람 손 상태가 없어서
proprio 를 명령한 손 행동(또는 `transfer.RobotToManoEstimator` 역추정)으로 채운다. 로봇 텔레옵 데이터가 있으면
`action.kind: robot_joint` 로 바로 학습한다.

### 5.7 촉각 특징 함수는 하나: `tactile_value_features`

`representation.tactile_value_features(residual_z, level, saturated, obs_mode)` 가 pretrain, VTLA 학습, 온라인
제어가 쓰는 **유일한** 촉각 값 특징이다. `full` = `[asinh(clip(z, ±100)/2), level one-hot 4, sat]` (taxel 당 6),
`ordinal` = one-hot 4, `binary` = WEAK∨STRONG, `none` = 촉각 브랜치 없음. 시간 스태킹은 `TactileFeatureSpec`
(history k, stride s) 이 맡고, 그 스펙은 `encoder_state.pt` 와 `policy_bundle.pt` 에 저장된다. 온라인에서는
`TactileHistory(spec).push(...)` 가 오프라인 `spec.from_arrays(...)` 와 같은 값을 낸다(`test_history_stacking_offline_matches_online`).

### 5.8 오프라인 ≡ 온라인

배포 처리기(`control.online.OnlineTactileProcessor`)는 학습 데이터를 만든 함수를 **틱 단위로 그대로** 호출한다:
`relative_change`/`saturation_mask`(전처리와 같은 레일·|ΔS| ≥ 90 % 규칙), `joint_velocity` 의 인과 SG 계수,
손 프레임 taxel FK, `CausalBaselineStream`, `ResidualCalibrator` (+ 같은 설정의 `SaturationFSM`),
`CausalDetectorStream` + `HysteresisFilter`, `TactileHistory`, `vtla.contact_from_level`, 그리고 관측 조립은
`vtla.make_observation` + `collate_vtla` 로 학습 데이터셋과 같다. `test_online_processor_reproduces_offline_stage_outputs`
(`tests/test_control.py`, 글러브·로봇 두 경우)가 처리된 합성 에피소드에 드롭아웃을 넣어 틱마다 흘리고 모든 중간값을
오프라인 stage 함수 출력과 비교한다: ΔS·포화·레벨·미신뢰 플래그·히스테리시스 출력은 비트 단위로 같고, 연속값
(q̇, baseline 평균·log 분산, z, 촉각 특징, 검출 확률)은 1e-4 이내다. 설계상 남는 차이: 온라인 압력은 최신 샘플
(zero-order hold), 오프라인은 마스터 격자 선형 보간이고, 글러브 손 라벨의 비인과 평활(filtfilt)은 온라인에 없다
([`DEPLOYMENT.md`](DEPLOYMENT.md) §4).

### 5.9 누수 없는 평가: 하나의 splits.json

`splits.json` 하나(기본 subject 단위, 그룹은 split 을 넘지 않음)를 모든 stage 가 `data.splits` 로 받는다. 그래야
pretrain 이 VTLA test 에피소드로 학습하거나, calibrator 가 baseline 학습 에피소드에서 맞춰지는 일이 없다.
`splits.json` 없이 stage 를 돌리면 각 stage 가 자기 풀을 따로 나누고 경고한다(`stages.warn_unshared_split`).

## 6. 논문 매핑

서지 정보와 검증 범위는 [`REFERENCES.md`](REFERENCES.md) 가 기준이다(여기서는 관계만 요약한다). 목록에 없는 문헌은
인용하지 않는다.

| 논문 (REFERENCES.md 절) | robot_skin 이 가져온 것 | 코드 | robot_skin 과의 차이 |
|---|---|---|---|
| **VTLA** (Zhang et al., arXiv:2505.09577; §6) | vision + tactile + language → action 문제 정의와 이름, 삽입 과제, 선호 학습(DPO) | `vtla/`, `vtla/dpo.py`, `stages/vtla.py` | 행동을 토큰으로 분류하지 않고 연속 헤드(chunk 회귀 / flow matching)로 예측. 촉각 입력은 손 전체 기압 taxel 의 보정된 z·레벨을 pose 토큰으로 만들고 ContactGate 를 거친다. DPO 는 손실·우도 대용치만 있고 선호 쌍 수집은 스텁 |
| **3D-ViTac** (Huang et al., arXiv:2410.24091; §4) | 촉각 값을 3D 공간 점으로 두어 공간 관계 보존 | `representation/tokenizer.py`, `representation/encoder.py` | 점구름 + diffusion policy 대신 taxel 하나를 `MLP(값) + MLP([Fourier(위치), 법선])` 토큰으로 만들고 transformer 로 인코딩. 손 프레임이라 글러브·로봇이 인코더를 공유 |
| **ActionSense** (DelPreto et al., NeurIPS 2022 D&B; §1) | 웨어러블 다중 스트림 동시 기록: 스트림별 파일 + 매니페스트 + 이벤트 로그, 1인칭 `ego` + 고정 `third` 카메라 | `acquisition/` | 촉각 스킨 전용 프로토콜(D1 무접촉·self-touch 블록, D2 과제 카탈로그), 3-탭 싱크, 세션 QC 게이트를 더함 |
| **OSMO** (Yin et al., arXiv:2512.08920; §1) | 사람과 로봇이 같은 촉각 표현을 쓰면 embodiment gap 이 준다; 사람 시연 → 로봇 기술 이전; wipe 과제 | `representation/`, `transfer/`, `action/retarget.py` | OSMO 는 3축(법선+전단) 센서, mk555 는 법선 1축. robot_skin 은 로봇 핸드에 다른 배치의 스킨이 붙는 경우를 pose 토큰과 `transfer.align_layouts` 로 다룬다 |
| **ACT** (Zhao et al., arXiv:2304.13705; §5) | action chunking(H 스텝), temporal ensembling(w_i = exp(−k·i), i = 0 이 가장 오래된 예측), masked L1 | `action/chunking.py`, `vtla/heads.py` (`ChunkRegressionHead`), `control/runner.py` | 관측 → 청크를 한 번의 forward 로 결정적 회귀(학습 쿼리 H 개 + 디코더). 여러 해가 가능한 동작에는 flow head |
| **π0 / flow matching / rectified flow** (Black et al. arXiv:2410.24164; Lipman et al. arXiv:2210.02747; Liu et al. arXiv:2209.03003; §6) | 연속 action chunk 를 조건부 flow matching 으로 생성, 직선 경로, 적은 스텝 Euler | `vtla/heads.py` (`FlowMatchingHead`) | VLM 백본 없이 작은 융합 transformer 위의 헤드. τ = 0 노이즈 / τ = 1 데이터 규약을 명시(π0 표기와 다를 수 있음). VLM 확장은 [`VTLA.md`](VTLA.md) §10 |
| **Kendall & Gal** (arXiv:1703.04977; §3) | heteroscedastic aleatoric 분산의 Gaussian NLL | `baseline/temporal.py`, `contact/calibration.py` | 분산을 z-score 분모로 써서 taxel·자세별 적응 임계값을 만든다. epistemic 분산(MC dropout·앙상블)은 아직 없음(§8) |
| **MAE** (He et al., arXiv:2111.06377; §4) | 높은 비율 마스킹 + 가벼운 디코더 재구성 자기지도 학습 | `representation/pretrain.py`, `stages/pretrain.py` | 이미지 패치 대신 taxel. 가린 taxel 은 인코더 key 에서 제외, 디코더 mask 토큰이 `residual_z`(Huber)와 레벨(가중 CE)을 재구성. 레이아웃 그룹(손가락 단위) 마스킹 옵션 |
| **HaMeR** (Pavlakos et al., arXiv:2312.05251; §2) | 단안 영상 → MANO 파라미터 | `pose/vision_hand.py` (`HaMeREstimator` 는 절차만 적은 스텁) | 실시간이 아니라 **오프라인 라벨러**: `hand_pose.npz` 를 만들어 `imu_pose` 를 감독하고 D2 행동 타깃이 된다. WiLoR 도 같은 자리 |
| **AnyTeleop / DexPilot** (Qin et al. arXiv:2307.04577; Handa et al. arXiv:1910.03135; §7) | 손끝·엄지–손가락 상대 벡터 매칭 리타게팅(DexPilot), 로봇 모델 무관 설계(AnyTeleop) | `action/retarget.py`, `stages/deploy.py` | 텔레옵이 아니라 **정책 출력**(MANO 청크)을 로봇으로 옮기는 데 쓴다. 기본 솔버는 projected LM; scale 은 AnyTeleop 규약(사람 벡터에 곱함) |
| **VIHand / VIFNet-S** (Wang et al., ACM MM 2025; §2, §8) | 비전 감독으로 IMU 손 자세 모델을 학습하는 구조(D1 의 근거), 사용자 지정 사전학습 백본 후보 | `pose/imu_model.py` (`ImuHandPoseNet`), `pose/glove_imu2mano.py` | VIFNet-S 의 IMU 수·입출력 형식은 **미검증** — `load_vifnet_s` / `finetune_vifnet_s` 는 스텁. 같은 역할(IMU 창 → MANO 손가락 자세)의 사내 베이스라인 GRU/TCN 으로 대신한다 |
| **Yu et al. 2026, pose-aware artifact** (arXiv:2607.22964; §3) | 같은 문제의 병행 연구: 촉각 글러브의 자세 관련 아티팩트를 손 자세로 보정 | `baseline/temporal.py`, `stages/baseline.py` | 아래 §6.1 |

### 6.1 Yu et al. (2026) 과 robot_skin baseline 의 관계

REFERENCES.md 에 적힌 범위(초록 수준)에서 이 논문은: 유연 촉각 글러브가 접촉 없이도 손 자세에 따라 신호가 변하는
문제(pose-related artifact)를 다루고, 손 자세 정보를 쓰는 **잔차 예측 분기**로 이를 보정해 글러브 3종·사용자
15명에서 **최소 검출 힘(MDF)** 이 줄었다고 보고한다. robot_skin 의 D1 baseline 과 **문제 설정이 같다**. robot_skin
설계는 이 논문과 독립적으로 `deformable_sats` bending restorer 에서 나왔다. 원문 세부(모델 구조, 시간 정보 사용 여부,
학습 데이터 구성)는 확인하지 못했으므로, 아래 차이는 **robot_skin 쪽 설계**를 기준으로 적는다.

- **시간 인과 모델**: 현재 자세만이 아니라 관절 이력 `q`, `q̇` (160 ms 창)을 본다. 늘어남·공압 커플링의 지연과
  속도 의존성을 표현하려는 것이다(합성 데이터도 1차 지연 + 속도 항으로 만든다).
- **분산 예측 → 보정된 z**: 평균만 빼는 것이 아니라 예측 분산으로 z-score 를 만들어, 접촉 문턱이 taxel·자세마다
  달라진다(§5.4). 결과는 연속 z + 순서 레벨(NONE/WEAK/STRONG/SATURATED)이다.
- **taxel pose 조건 + 손 프레임**: 채널 id 가 아니라 taxel 위치·법선을 입력으로 쓰므로 같은 모델 구조가 로봇 핸드
  (URDF FK)에도 적용된다.
- **파이프라인 안의 한 단계**: 출력이 접촉 판정에서 끝나지 않고 pretrain·VTLA 의 촉각 입력이 되며, 배포 때 같은
  계산을 온라인으로 재현한다(§5.8).
- **데이터 설계**: D1 프로토콜의 무접촉 블록(`air_grasp_*` 포함)과 self-touch 자동 라벨로 학습·평가 라벨을 만든다.
- **평가 지표**: robot_skin 은 MDF 를 구현하지 않았다(힘 정답이 필요). 대신 무접촉 환각률, 움직임–접촉 분리도
  (AUROC·d′, baseline 차감 전/후), self-touch recall, 합성 데이터의 정답 아티팩트 오차를 쓴다. 힘 센서 벤치 데이터가
  생기면 MDF 로 이 논문과 직접 비교하는 것이 다음 단계다(§8).

## 7. 평가 지표

`eval/metrics.py`: `hallucination_rate`(실제 무접촉 frame/taxel 중 접촉 예측 비율), `motion_contact_separability`
(실접촉 vs 움직임만 구간의 AUROC·d′ — baseline 차감 전/후 비교가 핵심), `saturation_recovery_times`, `auroc`.
stage 별 `metrics.json` 키와 읽는 법은 [`TRAINING.md`](TRAINING.md) §12. 합성 데이터(`datasets.synthetic`)에는
정답(`gt_artefact_pct`, `gt_contact` …)이 있어 `gt_*` 지표가 추가된다. 합성 데이터의 촉각 모델은 현상학적이고
실제 mk555 에 맞춘 값이 아니다 — 합성 지표는 **배관(plumbing) 검증**이지 성능 주장이 아니다.

## 8. 알려진 한계와 다음 단계

| 한계 | 현재 상태 | 다음 단계 |
|---|---|---|
| **MANO 축·휴지 자세 미검증** | 기본 스켈레톤은 손으로 만든 근사 오른손(손가락 −x, 요측 +z, 손바닥 −y). 실제 MANO 템플릿과 대조하지 못함(라이선스 파일 없음) | `ManoSkeleton.from_mano_pkl(MANO_RIGHT.pkl)` 로 한 번 대조. 캡슐 반지름·self-touch 여유도 실제 글러브로 조정 |
| **VIFNet-S 입출력 미검증** | `load_vifnet_s` / `finetune_vifnet_s` 는 스텁. `ImuHandPoseNet` 이 대신 | 원문·공개 코드로 IMU 개수·입력 형식·출력 표현 확인 → 입력 어댑터(7 사이트 특징) + 출력 변환(→ `finger_pose[15,3]`) 래퍼 |
| **baseline OOD / epistemic σ 없음** | 분산은 aleatoric 만. D1 에 없던 자세(D2 파워 그립)에서 평균은 과소 예측, σ 는 작음 → z 가 커져 거짓 접촉. stage-1 리뷰(합성 데이터, `air_grasp_*` 블록·공유 장갑 도입 전)에서 D2 pseudo-label 정밀도가 학습 시드에 따라 0.30–0.70 이었다 | ① D1 `air_grasp_*` 블록으로 자세 범위 덮기(프로토콜·합성 데이터에 반영됨, 실제 데이터 효과는 미측정) ② 앙상블 epistemic σ (`baseline/README.md` 에 설계만 있음) ③ 손–물체 근접 veto(`pseudo_label.proximity_m`) |
| **동기식 추론** | 정책 추론·리타게팅이 200 Hz 제어 틱 안에서 동기로 돈다. CPU 에서는 정책 틱마다 overrun | 백그라운드 추론 스레드 + 지연 보상 앙상블. GPU 에서 `metrics.json` 의 `tick_p95_ms` 측정 |
| **사전학습 가중치 오프라인 미시험** | 이 개발 환경은 HF hub·download.pytorch.org 가 막혀 ResNet/DINOv2/SigLIP/CLIP 가중치를 받아 보지 못했다. 무작위·로컬 가중치로만 시험 | GPU 박스에서 `HF_HOME`/`TORCH_HOME` 캐시로 한 번 실행. 번들 재구성(`pretrained: false`)도 HF config 가 캐시에 있어야 한다 |
| **GPU 경로 미시험** | NCCL DDP, fused AdamW, CUDA 에서의 `torch.compile`, bf16 autocast, 노드 간 DDP 는 여기서 돌려 보지 못했다(CPU·gloo 로만 시험) | RTX 5090 에서 `python -m robot_skin env` → `torchrun --standalone --nproc_per_node=1 -m robot_skin train <stage>` 스모크 |
| 실제 장치 드라이버 | `ImuSource`, `RosJointStateSource`, 실제 `RobotHandInterface` 없음. mk555 `.bin` 로더 없음(`bin_merge.py` 래핑 필요) | 장비가 생기면 프로토콜에 맞춰 구현 |
| `q_source: hand_pose_imu` 온라인 경로 | IMU 자세로 학습한 baseline 을 온라인에서 쓰려면 IMU→자세 스트림이 필요한데 `CausalImuPoseStream` 이 없다(로봇 배포는 무관) | `CausalBaselineStream` 옆에 IMU 창 링버퍼 스트림 추가 |
| `hand_mano` proprio 의 손목 | 손목 위치·회전은 카메라/월드 프레임 절대값이라 팔 없는 로봇에서는 관측 불가 | 손목 없는/상대 proprio 옵션 |
| 로봇 스킨 stage 1 | 글러브로 학습한 stage-1 모델은 로봇 스킨에 맞지 않는다. deploy 는 시작 보정 + residual = ΔS 로 대신 | 로봇 D1(`robot_sweep`) 로 별도 baseline·contact 를 학습해 `stage1.*` 로 넘김 |
| 파생 배열의 출처 | stage 간 연결이 episode 의 derived 배열이라, 어떤 모델이 썼는지 stage 가 검증하지 못한다(파이프라인이 순서·재실행으로 보장) | derived 옆에 출처 기록 |
| sim / RL | `TaxelDomainRandomizer` 만 구현. `TouchGridEnv`, `policy/train_rl.py` 는 스텁 | 물리 시뮬레이터 연동 |
| MDF 비교 | 힘 정답이 없어 Yu et al. 과 같은 지표로 비교 불가 | 힘 센서 벤치 데이터 |

하이퍼파라미터(baseline 창 32, 보정 임계 3/8 z · 0.5/3 %, 마스크 비율 0.3, VTLA d_model 128·horizon 16 …)는
실제 mk555 데이터로 튜닝한 값이 아니다.

## 9. 경계 규칙 요약

| 규칙 | 이유 |
|---|---|
| `robot_skin` 은 `deformable_sats` 를 import 하지 않는다; 공유 규약은 `common/` 에, 테스트와 함께 | 두 패키지 드리프트 방지 |
| mk555 raw `.bin` 파서는 `deformable_sats/sats/preprocessing/bin_merge.py` 가 정본 | 포맷 두 벌 유지 금지 |
| 규약: ΔS SATS 부호(누르면 음수), 쿼터니언 wxyz, 6D = 회전행렬 첫 두 열, MANO 관절 순서, 시간 s, 위치 m | 모든 모듈·파일 포맷 공통 |
| 인용은 `docs/REFERENCES.md` 에 있는 문헌만 | 검증되지 않은 인용 금지 |
| `robot_skin/data`, `robot_skin/runs` 는 git-ignored (README 만 추적) | 대용량 산출물 |
| 합성 데이터(`data/synthetic`)는 실제 데이터와 섞지 않는다 | 합성 촉각 모델은 실제 센서에 맞춘 것이 아니다 |
