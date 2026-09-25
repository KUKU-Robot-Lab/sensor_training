# VTLA — Vision + Tactile + Language → Action 정책

D2 과제 에피소드로 학습하는 정책 모델, 데이터 샘플링, 배포 번들, 논문과의 관계, 사전학습 VLM 백본으로 넓히는 방법을
정리한다. 코드: `robot_skin/vtla/` (모델·헤드·손실·데이터셋·DPO), `robot_skin/stages/vtla.py` (stage 러너),
`robot_skin/configs/stages/vtla.yaml` (설정). 코드 지도는 [`robot_skin/vtla/README.md`](../robot_skin/vtla/README.md),
실행은 [`TRAINING.md`](TRAINING.md) §4.5, 로봇 실행은 [`DEPLOYMENT.md`](DEPLOYMENT.md), 전체 구조는
[`ARCHITECTURE.md`](ARCHITECTURE.md), 인용 근거는 [`REFERENCES.md`](REFERENCES.md).

---

## 1. 목적과 위치

파이프라인의 마지막 학습 stage 다. stage 1(`baseline`, `contact`)이 D2 에피소드의 촉각을 **보정된 잔차 z 와 접촉
레벨**로 바꿔 놓으면, VTLA 는 그것과 카메라·지시문·현재 손 상태를 받아 **앞으로 H 스텝의 행동(action chunk)** 을
예측한다. 결과물 `policy_bundle.pt` 하나에 배포에 필요한 모든 것이 들어간다.

## 2. 한눈에

| | 내용 (기본값) |
|---|---|
| 관측 | 지시문 1개, 카메라 `policy.cameras` (`[ego]`) 이미지 또는 캐시 특징, taxel 별 촉각 특징 `[N, F]` + taxel pose `[N, 3]`×2 + 접촉 플래그 `[N]`, proprio(현재 행동 공간 상태) |
| 출력 | 정규화된 action chunk `[H, A]` — `hand_mano` 이면 A = 54, H = `policy.horizon` (16) |
| 주기 | 관측·추론은 `policy.policy_hz` (20 Hz) tick, 데이터 마스터 시계는 200 Hz (stride = 10 프레임) |
| 헤드 | `chunk` (ACT 식 회귀, masked L1) 또는 `flow` (flow matching, Euler 10 스텝) |
| 학습 | `train.Trainer` (정밀도는 하드웨어 프로파일 — GPU 프로파일은 bf16; torchrun DDP; `train.ema_decay` 를 주면 EMA; 재개), best = `val/loss` (chunk 헤드: masked L1) |
| 산출물 | `<out>/vtla/policy_bundle.pt`, `metrics.json` |

## 3. 모델 구조 (`vtla/model.py` — `VTLAConfig`, `VTLAPolicy`)

```
지시문   ─ TextEncoder (language.build_text_encoder) ─ Linear ────────────────────┐ L 토큰
카메라 c ─ VisionEncoder (vision.build_vision_encoder) ─ Linear + camera_emb[c] ───┤ P × 카메라 (× obs_history)
           (또는 frozen 인코더의 캐시 특징 vision.feature_cache)                    │
taxel    ─ tactile_value_features → TaxelEncoder (stage 2 사전학습 가능, 동결 가능)   │
           → TactileTokenAdapter (Perceiver 식 쿼리 K 개) → ContactGate ───────────┤ K 토큰
proprio  ─ MLP (현재 행동 공간 상태, 정규화, × obs_history) ─────────────────────────┤ 1 토큰
readout  ─ 학습 토큰 ───────────────────────────────────────────────────────────┘ R 토큰
     + modality type embedding → pre-LN TransformerEncoder (fusion; key padding = 없음/드롭된 토큰)
     → head: ChunkRegressionHead | FlowMatchingHead → 정규화된 [B, H, A]
```

### 3.1 토큰

| modality | 만드는 곳 | 개수 | 설정 |
|---|---|---|---|
| language | `HashingTextEncoder` (기본: 단어 해싱 + 학습 임베딩, 의존성 없음) 또는 `HFTextEncoder` (`clip`, `siglip`, `t5`) | 지시문 토큰 L (≤ `max_len` 32, 위치 0 은 BOS) | `language.encoder`, `language.frozen` |
| vision | `TinyConvEncoder` (기본, `grid: [4, 4]` → 카메라당 16) / `ResNetEncoder` / `HFVisionEncoder` (`dinov2`, `siglip`, `clip`) → Linear + 카메라 임베딩 (+ `obs_history > 1` 이면 history 임베딩) | 카메라당 P × k | `vision.encoder`, `vision.frozen`, `vision.cache_features`, `image.*` |
| tactile | §3.2 | K = `model.n_tactile_tokens` (4) | `features.*`, `model.tactile_encoder`, `tactile.*` |
| proprio | 2층 MLP (입력 = k × A) | 1 | `policy.obs_history` |
| readout | 학습 파라미터 | R = `model.n_readout` (1) | |

모든 토큰에 modality type embedding 을 더하고 하나의 pre-LN transformer(`model.fusion_depth` 2, `fusion_heads` 4,
`d_model` 128)로 섞는다. 없는 입력(카메라 첫 프레임 이전, 드롭된 modality)은 key-padding 으로 가린다.

### 3.2 촉각 브랜치

1. **값 특징** — `representation.tactile_value_features(residual_z, contact_level, saturated, obs_mode)`: pretrain·VTLA·
   온라인 제어가 같이 쓰는 **유일한** 촉각 특징 함수. `full` = `[asinh(clip(z, ±100)/2), level one-hot 4, sat]`
   (taxel 당 6). `TactileFeatureSpec(history k, stride s)` 이면 과거 k 프레임을 쌓는다(200 Hz tick 기준, 인과).
2. **TaxelEncoder** — taxel 하나 = `LayerNorm(MLP(값) + MLP([Fourier(위치), 법선]))` 토큰(`TaxelTokenizer`; Fourier
   최저 옥타브 주기 0.3 m, 6 옥타브), 그 위에 pre-LN transformer. 위치는 **손/로봇 base 프레임**이라 글러브와 로봇
   핸드가 같은 인코더를 쓴다(taxel id 임베딩 `n_taxels` 는 기본으로 끔 — 켜면 레이아웃 공유가 깨진다). `tactile.pretrained` 로 pretrain stage 의
   `encoder_state.pt` 에서 시작할 수 있고(그 인코더의 특징 스펙이 우선), `tactile.freeze` 로 고정할 수 있다.
3. **TactileTokenAdapter** — 학습 쿼리 K 개가 taxel 토큰에 cross-attention (Perceiver 식) → taxel 수와 무관하게 K
   토큰 → `d_model` 로 투영. 섞인 레이아웃 배치는 `taxel_pad` 로 패딩 taxel 을 가린다.
4. **ContactGate** — `contact = level ≥ WEAK` (`data.contact_rule: level_ge_weak`, SATURATED 포함; `weak_or_strong`
   은 포화 제외). `model.tactile_gate: hard` 이면 샘플에 접촉 taxel 이 하나도 없을 때 K 토큰 전체가 0, `soft` 이면
   `σ(w·접촉비율 + b)` 를 곱한다(무접촉 0 은 유지). 무접촉 드리프트가 융합에 들어가지 못하게 하는 구조적 장치다.
   항상 포화인 죽은 채널은 게이트를 늘 열어 두므로 `taxel_pad` 로 가리거나 `weak_or_strong` 을 쓴다.

### 3.3 헤드 (`vtla/heads.py`)

- **`chunk` — `ChunkRegressionHead`** (ACT): H 개 학습 쿼리 슬롯이 pre-LN transformer 디코더에서 융합 토큰에
  cross-attention → `[B, H, A]`. 손실 = 유효 스텝만의 L1(`action_valid`). 한 번의 forward, 결정적.
- **`flow` — `FlowMatchingHead`** (조건부 flow matching, rectified flow 직선 경로). **τ = 0 이 노이즈, τ = 1 이
  데이터**: `x_τ = τ·a + (1−τ)·ε`, 목표 속도 `u = a − ε`, 손실 `‖v_θ(x_τ, τ, obs) − u‖²` (유효 스텝만). 추론은
  `x_0 = ε` 에서 `x_{k+1} = x_k + (1/K)·v_θ(x_k, k/K)` 로 K = `model.flow_steps` (10) 스텝 Euler. 학습 τ 는
  `uniform` 또는 `beta` (Beta(1, b), b > 1 이면 노이즈 쪽에 가중). eval 모드 손실은 `eval_seed` 로 매번 같은 (ε, τ)
  를 뽑아 epoch 간 비교가 가능하다. **이 τ 방향은 robot_skin 규약이며 π0 원문 표기와 다를 수 있다.**
- 두 헤드 모두 출력층이 0-초기화다(학습 전 chunk = 정규화 평균 0, flow 속도 = 0).

### 3.4 정규화와 학습 장치

- **Modality dropout** (`model.p_drop_tactile` 0.1, `p_drop_vision` 0, `p_drop_language` 0; 학습 때 샘플별):
  modality 의 토큰 전체를 key-padding 으로 가린다. 센서 하나가 없어도 행동하고 한 modality 에만 기대지 않게 한다.
  가린 토큰도 그래프에 남으므로(기울기 0) DDP 에 `find_unused_parameters` 가 필요 없다.
- **Aux contact head** (`model.aux_contact_weight > 0`): 촉각 인코더 토큰에서 taxel 별 접촉 logit → BCE.
  타깃 `data.aux_target`: `label` (contact stage 의 D2 pseudo 라벨 `contact_label_pseudo`, 없으면 전처리
  `contact_label`; −1 은 무시), `level` (촉각 규칙), `gt` (합성 정답). 촉각 표현을 접촉 중심으로 다듬는다.
- **동결**: `vision.frozen`, `language.frozen`, `tactile.freeze` — 고정된 인코더는 `train()` 중에도 eval 모드.
  천천히 미세조정하려면 `train.lr_mult` (0 = 동결). 접두사로 쓸 수 있는 모듈 이름: `text_encoder`, `text_proj`,
  `vision_encoder`, `vision_proj`, `camera_emb`, `history_emb`, `tactile_encoder`, `adapter`, `contact_head`,
  `proprio_mlp`, `readout`, `type_emb`, `fusion`, `head`.
- 토크나이저의 `[MASK]` 벡터는 pretrain 에서만 쓰므로 VTLA 에서는 `requires_grad=False`.

### 3.5 DPO 훅 (`vtla/dpo.py`)

VTLA 논문은 선호 학습(DPO)으로 연속 행동과 토큰 손실 사이의 간극을 메운다. robot_skin 에는:
`dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)` (일반 DPO 손실),
`preference_loss(policy, reference, batch)` — 연속 chunk 의 **우도 대용치**(chunk 헤드: 고정 σ 가우시안, flow 헤드:
공유 (ε, τ) 에서의 flow-matching 오차; 정확한 우도가 아니다). 각 모델은 관측을 한 번만 인코딩해 chosen/rejected 가
공유하고, `disable_dropout=True` 로 dropout·modality dropout 을 끈다 — 정책 = 참조이면 손실이 정확히 log 2 다.
`make_reference_policy` 로 참조를 고정한다. 선호 쌍 수집(성공/실패 rollout → `PreferencePair`)은
`build_preference_pairs` 가 절차를 설명하고 `NotImplementedError` 를 낸다 — 배포 세션에 `success` 를 기록해 모으는
경로는 [`DEPLOYMENT.md`](DEPLOYMENT.md) §9.

## 4. 데이터셋 샘플링 (`vtla/dataset.py` — `VTLADataset`)

### 4.1 policy tick

샘플 하나 = 200 Hz 마스터 시계의 policy tick `t` 하나. `stride = round(200 / policy_hz)` (20 Hz → 10), 샘플 간격은
`data.sample_stride` (null = stride; 1 = 모든 프레임, 데이터 증강 효과). 샘플링 구간은 `data.phases`:
`task` (기본 — reach, grasp, manipulate, release, retreat) | `all` (phase 가 있는 모든 프레임) | phase 이름 목록.
D2 에피소드 앞의 휴지 baseline·IMU 평손 보정·3-탭 싱크 블록을 **모방하지 않기 위해** 기본값이 `task` 다. 추가 필터:
`require_valid_state` (현재 손 상태가 측정값), `min_valid_steps`, `require_success` (`manifest.task.success`).

### 4.2 action chunk

`actions[i] = a[t + (chunk_offset + i)·stride]`, `i = 0 … H−1` (`chunk_offset` 1 → `actions[0]` 은 한 tick 뒤 상태).
에피소드 끝을 넘거나 손 라벨이 무효(`hand_pose_valid` 거짓)인 스텝은 마지막 유효 프레임으로 채우고
`action_valid = False` (ACT 의 패딩 마스크) — 손실과 지표에서 빠진다. 16 스텝 × 50 ms = 0.8 s 앞까지 예측한다.

### 4.3 행동 공간 (`action/space.py`)

| `action.kind` | 차원 | 내용 |
|---|---|---|
| `hand_mano` (기본) | 54 | `[0:3]` 손목 위치 (m) · `[3:9]` 손목 회전 6D (회전행렬 첫 두 열) · `[9:54]` 손가락 15 관절 axis-angle (MANO 순서 index1..3, middle1..3, pinky1..3, ring1..3, thumb1..3) |
| `robot_joint` | D | 로봇 관절 `q` (URDF 구동 관절 순서) — 로봇 텔레옵 데이터가 있을 때 |

`action.rel_mode`: `abs` | `delta` (기본 — `hand_mano` 는 손목 위치만 현재 손목 위치 기준, 회전·손가락은 절대;
`robot_joint` 는 `q − q_cur`) | `delta_pose` (`hand_mano` 만 — 위치와 회전을 현재 손목 프레임 기준으로).
`robot_joint` + `delta_pose` 는 거부한다. 정규화: `ActionNormalizer` 를 **train split 의 상대 chunk 유효 스텝**으로
맞춘다(`action.norm_method`: `std` | `robust` | `minmax` | `none`, 거의 상수인 차원은 `action.min_scale` 0.01 로 하한).

### 4.4 proprio

현재 행동 공간 상태(절대값: `hand_mano` 54-D 또는 로봇 `q`)의 최근 k = `policy.obs_history` policy tick
(오래된 것 먼저, 인과 edge-padding)을 이어 붙인 `[k·A]`. 두 번째 `ActionNormalizer` 를 train 샘플의 현재 상태로 맞춘다.
주의: `hand_mano` proprio 의 손목 위치·회전은 손 라벨 프레임(카메라/월드)의 절대값이다 — 팔 없는 로봇에서는 관측할 수
없어 배포 때 명령한 손 행동으로 채운다(§7, §11).

### 4.5 촉각·비전·언어 입력

- **촉각**: `data.tactile_source` — `derived` (contact stage 의 `residual_z`/`contact_level`, 실제 학습) | `auto`
  (기본: derived 가 없는 에피소드는 부트스트랩으로 대체 + 경고) | `bootstrap` (테스트 전용: 에피소드의 `no_contact`
  프레임 ΔS 중앙값을 정적 기준으로 한 z — 움직임 모델 없음). 번들에 사용한 원천이 기록되고, `bootstrap`/`mixed`
  번들은 deploy 가 거부한다(`PolicyBundle.check_deployable`; 시뮬레이션·bring-up 에서만 `--set allow_bootstrap=true`).
  파이프라인은 contact stage 결과가 있으면 `derived` 를 강제한다.
- **비전**: `image` 섹션 → `vision.build_transforms` — train 은 random resized crop(`scale` 0.8–1.0) + 밝기·대비
  jitter, val/test/배포는 같은 확대율의 eval 변환(`eval_crop_scale: null` = train 평균 crop 면적), 기본 `image_size`
  `[96, 128]`. `vision_valid` 는 카메라 첫 프레임 이전 False(빈 이미지로 채우고 key-padding). `vision.cache_features`
  면 frozen 인코더 특징을 미리 캐시하고 eval 변환만 쓴다([`TRAINING.md`](TRAINING.md) §13).
- **언어**: `meta.task.instruction` (작업 카탈로그 템플릿으로 생성, 운영자가 덮어쓸 수 있음). collate 가 인코더의
  피클 가능한 토크나이저(`get_tokenizer()`)로 `input_ids`/`text_pad_mask` 를 만든다.

`make_observation(...)` 이 관측 dict 를 만들고 `collate_vtla` 가 taxel 축을 패딩(`taxel_pad`)해 배치로 묶는다.
**온라인 제어도 같은 두 함수로** 단일 샘플을 만든다 — 학습과 배포 입력이 갈라질 수 없다.

## 5. 학습과 평가

```bash
python -m robot_skin pipeline --stages vtla                        # 파이프라인 안에서 (연결 자동)
python -m robot_skin train vtla --hardware rtx5090 --set data.splits=robot_skin/runs/splits.json \
    --set data.tactile_source=derived --set tactile.calibrator=robot_skin/runs/contact \
    --set tactile.baseline_model=robot_skin/runs/baseline --set tactile.pretrained=robot_skin/runs/pretrain
python -m robot_skin train vtla --set model.head=flow --set model.flow_steps=10    # flow 헤드
```

- split: `data.splits` (파이프라인과 같은 파일; 없으면 `data.split_by: subject` 로 자체 분할 + 경고).
- 설정 검증: `check_config` 가 `DEFAULTS` 에 없는 키를 최상위와 각 섹션 한 단계에서 거부한다(`train`, `image` 는
  열린 섹션). stage 가 `policy.*` 에서 유도하는 `model.horizon` 같은 키도 `model` 에 쓰면 오류다. `type` 이 있는
  `vision.encoder` / `language.encoder` 블록은 기본 블록을 **대체**하고, 일부 키만 주면 병합된다. `null` 은 그 입력을 끈다.
- 평가(`evaluate_policy`, best/EMA 가중치): `{val,test}/l1` (정규화 단위 masked L1), `l1_per_step[H]`, `l1_by_task`,
  `l1_raw/{wrist_pos, wrist_rot6d, finger_aa}` (원 단위 상대 행동), `n_samples`, `n_valid_steps`. 오프라인 chunk 오차만
  본다 — 폐루프 성공률은 deploy/실기에서 측정한다.

## 6. `policy_bundle.pt`

`save_policy_bundle` 가 원자적으로 쓰고 `torch.load(weights_only=True)` 로 읽힌다(`read_policy_bundle`). 섹션:

| 키 | 내용 |
|---|---|
| `format`, `version` | `robot_skin/vtla_policy_bundle`, 1 |
| `model_config`, `state_dict` | `VTLAConfig` + 최적(EMA 켜면 EMA) 가중치 (CPU 텐서) |
| `action` | `spec` (ActionSpec), `normalizer` (ActionNormalizer), `rel_mode`, `chunk_offset` |
| `proprio` | `normalizer`, `history`, `source` |
| `tactile` | `feature_spec`, `obs_mode`, `contact_rule`, `source` (derived / bootstrap / mixed), `bootstrap`, `calibrator` + `calibrator_state` (contact stage 의 `calibrator.json` 내용 내장), `baseline_model`, `pretrained_encoder`, `frozen`, `layouts` |
| `vision` | `cameras`, `encoder` cfg, `frozen`, `cached_features_key`, `eval_transform` 파라미터, `transform_config` |
| `language` | `encoder` cfg, `frozen` |
| `timing` | `policy_hz`, `source_hz`, `stride`, `horizon`, `obs_history`, `sample_stride` |
| `head`, `meta` | 헤드 종류·flow 스텝; split 별 에피소드, phase, best, 지표, `ensemble_k` |

`build_policy_from_bundle(bundle)` → eval 모드 `VTLAPolicy` (인코더는 `pretrained: false` 로 만들고 가중치는
`state_dict` 에서 — HF 인코더는 모델 **config** 가 로컬 캐시나 hub 에 있어야 한다). `bundle_components(bundle)` →
정책 + 행동/proprio 정규화기 + `TactileFeatureSpec` + contact 규칙 + `EvalTransform` + 토크나이저 + 타이밍.

## 7. 배포 경로 (요약)

`control` 이 번들을 로봇 루프에 연결한다([`DEPLOYMENT.md`](DEPLOYMENT.md)): 200 Hz 틱마다 `OnlineTactileProcessor`
(오프라인 stage 1 과 같은 계산) → `TactileHistory(feature_spec)` → `contact_from_level` → policy tick 마다
`make_observation` + `collate_vtla` → `policy.predict` → `action_normalizer.unnormalize` → `make_absolute(chunk,
state, spec, rel_mode)` → `TemporalEnsembler` (ACT, `meta.ensemble_k` 0.01) → `hand_mano` 면 손끝 → 
`FingertipRetargeter` → `SafetyFilter` → 로봇. 추론은 제어 틱 안에서 동기로 돈다. 로봇에는 사람 손 상태가 없어
`hand_mano` proprio 는 명령한 손 행동(또는 `transfer.RobotToManoEstimator` 역추정)으로 채우고, 손목 행동은
팔 제어기가 없으면 로그에만 남는다.

## 8. 절제(ablation) 스위치

| 무엇을 보나 | 설정 |
|---|---|
| 촉각 정보량 | `features.obs_mode`: `full` (6/taxel) · `ordinal` (레벨 one-hot 4) · `binary` (접촉 1) · `none` (촉각 브랜치 없음, 토큰 0) |
| 촉각 시간 문맥 | `features.history`, `features.stride` |
| 접촉 게이트 | `model.tactile_gate: hard | soft`, `data.contact_rule` |
| 모달리티 제거 | `vision.encoder: null`, `language.encoder: null`, `policy.cameras` |
| 모달리티 강건성 | `model.p_drop_tactile`, `p_drop_vision`, `p_drop_language` |
| 촉각 사전학습 효과 | `tactile.pretrained` 유/무, `tactile.freeze` |
| 헤드 | `model.head: chunk | flow`, `model.flow_steps`, `model.flow_tau` |
| 행동 표현 | `action.rel_mode`, `action.norm_method`, `policy.horizon`, `policy.policy_hz` |
| 보조 손실 | `model.aux_contact_weight`, `data.aux_target` |

비교할 때는 같은 splits.json 과 같은 stage-1 결과(같은 processed root)를 쓰고 `test/` 지표로 본다. 스윕은
[`TRAINING.md`](TRAINING.md) §11 (vtla 의 `--metric` 은 `val/l1` 또는 `best.value`).

## 9. 논문과의 관계

서지·검증 범위는 [`REFERENCES.md`](REFERENCES.md) 기준이다.

| 논문 | VTLA 에서 가져온 것 | 차이 |
|---|---|---|
| VTLA (Zhang et al., arXiv:2505.09577) | vision + tactile + language → action 문제 정의, 삽입 과제(`peg_insert`), 선호 학습(DPO) | 행동을 토큰 분류하지 않고 연속 헤드로 예측. DPO 는 훅(손실·우도 대용치)만 있고 선호 쌍 수집은 스텁 |
| Octo (Octo Model Team, arXiv:2405.12213) | 모듈형 토큰 설계: 언어 과제, 여러 카메라, proprio 관측, 여러 행동 공간을 한 트랜스포머가 처리 | robot_skin 은 modality type embedding + readout 토큰을 쓰고 촉각 modality 와 ContactGate 를 더했다. 작은 스크래치 융합 트랜스포머 |
| ACT (Zhao et al., arXiv:2304.13705) | action chunking, temporal ensembling(w_i = exp(−k·i)), masked L1, 패딩 마스크 | 관측 → chunk 결정적 회귀 헤드 |
| π0 (Black et al., arXiv:2410.24164), Flow Matching (Lipman et al., arXiv:2210.02747), Rectified Flow (Liu et al., arXiv:2209.03003) | flow-matching 행동 헤드, 직선 경로, 적은 스텝 Euler | 사전학습 VLM 백본 없이 작은 융합 모델 위의 헤드(§10). τ 방향 규약을 명시 |
| 3D-ViTac (Huang et al., arXiv:2410.24091) | 촉각을 3D 공간 점으로 두어 공간 관계 보존 | pose 토큰 + transformer 인코더, 손 프레임 |
| Perceiver (Jaegle et al., arXiv:2103.03206) | 학습 쿼리 K 개의 cross-attention 으로 큰 입력 압축 | `TactileTokenAdapter` — taxel 수 무관 |
| MAE (He et al., arXiv:2111.06377) | 촉각 인코더 사전학습(pretrain stage) | 이미지 대신 taxel, `tactile.pretrained` 로 초기화 |
| DPO (Rafailov et al., arXiv:2305.18290) | `dpo_loss` | 연속 chunk 에는 정확한 우도가 없어 대용치 사용 |
| Diffusion Policy (Chi et al., arXiv:2303.04137) | 비전 조건화(ResNet + GroupNorm, spatial softmax, 작은 random crop) | `vision/` 의 선택지 |
| OpenVLA, DINOv2, SigLIP, CLIP (arXiv:2406.09246, 2304.07193, 2303.15343, 2103.00020) | 고정 비전/텍스트 타워 선택지(`dinov2`, `siglip`, `clip`) | 타워를 VLM 으로 합치지 않고 따로 쓴다(§10) |

## 10. 사전학습 VLM 백본으로 넓히기

현재 모델은 작은 스크래치 융합 트랜스포머(`d_model` 128) 위에 인코더를 붙인 구조다. 사전학습 가중치를 쓰는 길은
단계가 있다. **10.1 은 지금 설정만으로 되고, 10.2–10.3 은 코드 변경이 필요한 설계 안내다(구현되어 있지 않다).**

### 10.1 지금 되는 것: 사전학습 비전·텍스트 타워

```bash
python -m robot_skin train vtla --set data.splits=robot_skin/runs/splits.json \
    --set vision.encoder='{type: siglip, frozen: true}' --set image.image_size='[224, 224]' \
    --set vision.cache_features=true \
    --set language.encoder='{type: siglip, frozen: true}' --set language.frozen=true
```

- `vision.encoder.type`: `dinov2` (`facebook/dinov2-small`), `siglip` (`google/siglip-base-patch16-224`), `clip`
  (`openai/clip-vit-base-patch32`), `hf` + `model_id` (다른 HF 모델), `resnet18/34/50` (torchvision). 텍스트:
  `clip`, `siglip`, `t5` (`t5-small`), `hf` + `model_id`. transformers / torchvision 이 필요하고, SigLIP·T5 토크나이저는
  sentencepiece 도 필요하다. (이 명령은 이 환경에서 설정 해석까지 확인했고, transformers 가 없어 인코더 생성 단계의
  `ImportError` 에서 멈췄다.)
- 고정 타워는 `vision.cache_features: true` 로 특징을 한 번만 계산한다(eval 변환 고정, 가중치 해시 키).
- 미세조정하려면 `frozen: false` + `train.lr_mult='{vision_encoder: 0.1}'`. 이때는 캐시를 쓸 수 없고 메모리·시간이
  커진다 — bf16 과 GPU 프로파일의 배치(5090 vtla 64 × 2)를 기준으로 조정한다.
- 번들은 인코더 가중치를 `state_dict` 에 담고 `pretrained: false` 로 재구성하므로, 로봇 PC 에 HF 모델 **config**
  (가중치 아님)가 캐시돼 있어야 한다(`HF_HOME`). 이 개발 환경에서는 hub 접근이 막혀 사전학습 가중치로 시험하지 못했다.

### 10.2 한 단계 더: VLM 을 "인코더"로 (가장 덜 침습적)

이미지 + 지시문을 한 VLM 에 넣어 나온 hidden state 를 언어·비전 토큰 대신 쓰는 방식. 나머지(촉각 브랜치,
ContactGate, proprio, readout, 융합, 헤드, 번들)는 그대로 둔다. 바꿀 곳:

1. `vtla/model.py` — `VTLAConfig` 에 VLM 설정 필드 추가(`from_dict` 는 모르는 키를 거부하므로 필드로 선언),
   `VTLAPolicy.__init__` 에 VLM 모듈 + `Linear(D_vlm → d_model)`, `encode()` 에서 `_language`/`_vision` 대신 VLM 호출 →
   `LANG`/`VISION` 타입 토큰으로 `parts` 에 추가. 헤드는 `memory [B,S,d]` 만 보므로 수정 불필요.
2. `vtla/dataset.py` / `collate_vtla` — VLM 프로세서(이미지 전처리 + 토크나이저)를 collate 에서 쓰려면 피클 가능한
   콜러블로 감싼다(`TextEncoder.get_tokenizer()` 와 같은 패턴). 이미지 변환은 VLM 의 mean/std·해상도에 맞춘다.
3. `stages/vtla.py` — `DEFAULTS` + `configs/stages/vtla.yaml` (테스트가 둘을 같게 유지), `_model_config` 의 설정 전달,
   번들의 `vision`/`language` 섹션에 VLM id·revision 기록.
4. `build_policy_from_bundle` — VLM 을 `pretrained: false` 로 재구성하고 `state_dict` 에서 가중치를 싣는 규칙을 따른다.
   번들 크기가 VLM 가중치만큼 커진다.

### 10.3 π0 식: VLM 이 융합 백본

촉각 K 토큰·proprio·readout 을 VLM 입력 시퀀스에 접두 임베딩으로 넣고(`Linear(d → D_vlm)`), 행동 헤드(chunk 쿼리
또는 flow 행동 전문가)가 VLM 의 hidden state 에 cross-attention 하는 방식. `fusion` 을 VLM 으로 대체한다.
고려할 점:

- **불변식 유지**: 촉각 입력은 계속 `tactile_value_features` + `TaxelEncoder` + ContactGate 를 거친다(무접촉이면 0).
  행동은 정규화된 chunk 이고, `make_observation` 은 온라인 제어와 공유한다.
- **attention mask**: 드롭·결측 토큰을 VLM 의 attention mask 로 가려야 modality dropout 이 유지된다. 일부 파라미터가
  step 마다 기울기를 못 받으면 `train.find_unused_parameters: true`.
- **학습 설정**: bf16(5090 프로파일), `train.lr_mult` 로 VLM 은 작은 lr(또는 0 = 동결). gradient checkpointing,
  LoRA 류 어댑터는 구현돼 있지 않다.
- **지연**: 추론은 20 Hz policy tick 에 동기로 돈다(50 ms 예산, 제어 틱 5 ms 를 넘으면 overrun). VLM 크기에 따라
  GPU 에서 `metrics.json` 의 `latency_p95_ms`·`overruns` 를 먼저 측정한다([`DEPLOYMENT.md`](DEPLOYMENT.md) §6).
  비동기 추론은 아직 없다.
- **데이터 양**: 사전학습 백본의 이점은 D2 데이터가 적을 때 크다. 같은 splits.json·test 지표로 스크래치 모델과
  비교한다(§8).

## 11. 한계

- 하이퍼파라미터(d_model 128, horizon 16 @ 20 Hz, `p_drop_tactile` 0.1, flow 10 스텝, 이미지 96×128, `rel_mode: delta`)는
  실제 데이터로 튜닝하지 않았다.
- `hand_mano` proprio 는 손목의 절대 자세(카메라/월드 프레임)를 포함한다 — 손목 없는/상대 proprio 옵션이 없다.
- 전처리가 손 라벨을 비인과로 평활하므로(filtfilt) 학습 proprio 는 온라인 추정보다 약간 매끄럽다.
- `data.phases: task` 인데 task phase 가 없는 에피소드는 라벨된 모든 프레임으로 대체하고 경고만 한다.
- 평가는 오프라인 chunk L1 만. DPO 선호 쌍 수집, 비동기 추론, VLM 백본(§10.2–10.3)은 구현되지 않았다.
- CUDA, 실제 DDP, `torch.compile`, HF/ResNet 인코더의 사전학습 가중치는 이 환경에서 실행하지 못했다.
