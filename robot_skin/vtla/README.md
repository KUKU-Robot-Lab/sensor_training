# vtla/ — Vision + Tactile + Language → Action (구현)

D2 과제 에피소드(비전 + 촉각 + 지시문 + 손 자세)로 **action chunk 정책**을 학습하고, 제어(`control`)가 그대로
다시 만들 수 있는 `policy_bundle.pt` 를 쓴다. 스테이지 러너: `robot_skin/stages/vtla.py`
(`python -m robot_skin.stages.vtla --config robot_skin/configs/stages/vtla.yaml --set model.head=flow`).

## 모듈

| 모듈 | 내용 |
|---|---|
| `adapter.py` | `TactileTokenAdapter` (Perceiver 식 K 쿼리 cross-attention, taxel 수 무관, `key_padding_mask`), `ContactGate` (`hard`: 접촉 없으면 0 / `soft`: σ(w·접촉비율+b), 패딩 taxel 은 비율 분모에서 제외) |
| `model.py` | `VTLAConfig`, `VTLAPolicy` (아래 구조), bundle: `save_policy_bundle` / `read_policy_bundle` / `build_policy_from_bundle` / `bundle_components` |
| `heads.py` | `ChunkRegressionHead` (ACT: H 개 학습 쿼리 → 디코더 → `[B,H,A]`, masked L1), `FlowMatchingHead` (flow matching / rectified flow, Euler K 스텝) |
| `losses.py` | `masked_l1` / `masked_mse` / `masked_step_sums` (패딩 스텝 제외), `contact_bce` (aux), `vtla_loss` (Trainer `loss_fn`) |
| `dataset.py` | `VTLADataset` (policy tick 샘플), `make_observation` (온라인 제어와 공유), `collate_vtla` / `VTLACollator`, 부트스트랩 촉각 레벨 |
| `dpo.py` | `dpo_loss` (Rafailov et al. 2023), `preference_loss` (정책 vs 고정 참조, 우도 대용치), `build_preference_pairs` = 문서화된 stub |

## 모델 구조

```
지시문  ─ TextEncoder (language.build_text_encoder) ─ Linear ───────────────────┐ L 토큰
카메라 c ─ VisionEncoder (vision.build_vision_encoder) ─ Linear + cam_emb[c] ────┤ P×카메라 (×obs_history)
          (또는 frozen 인코더의 캐시 특징 vision.feature_cache)                  │
taxel   ─ TaxelEncoder (stage-2 사전학습 가능, freeze 가능)                       │
          → TactileTokenAdapter (K 쿼리) → ContactGate ─────────────────────────┤ K 토큰
proprio ─ MLP (현재 손 action / 로봇 q, 정규화, ×obs_history) ──────────────────┤ 1 토큰
readout ─ 학습 토큰 ───────────────────────────────────────────────────────────┘ R 토큰
      + modality type embedding → pre-LN TransformerEncoder (key padding = 없음/드롭된 토큰)
      → head (chunk | flow) → 정규화된 action chunk [B,H,A]
```

- **촉각 값**: `representation.tactile_value_features` (`TactileFeatureSpec`) — 사전학습·VTLA·온라인 제어가 같은
  함수를 쓴다. `obs_mode` 절제: `full`(6) / `ordinal`(4) / `binary`(1) / `none` (촉각 분기 자체를 만들지 않음, 토큰 0개).
- **ContactGate**: `contact = level ≥ WEAK` (`data.contact_rule`) 가 하나도 없으면 촉각 토큰이 0 → 무접촉 드리프트가
  융합에 들어갈 수 없다 (상수 modality embedding 만 남는다). 주의: 기본 규칙(`level_ge_weak`)은 SATURATED 도 접촉으로
  센다. 전처리가 모든 프레임에서 saturated 로 표시하는 **죽은 채널**이 있으면 게이트가 항상 열리므로, 그런 taxel 은
  `taxel_pad` 로 가리거나 `weak_or_strong` 을 쓴다.
- **Modality dropout** (학습 시 샘플별): `p_drop_tactile` / `p_drop_vision` / `p_drop_language` 가 해당 모달리티
  토큰 전체를 key-padding 으로 가린다. 가려진 토큰도 그래프에 남으므로(기울기 0) DDP 에
  `find_unused_parameters` 가 필요 없다.
- **Aux contact head** (`aux_contact_weight > 0`): 촉각 인코더 토큰에서 taxel 별 접촉 logit → BCE
  (`data.aux_target`: `label` = contact_label ≥ 0, `level`, `gt` = 합성 정답).
- `lr_mult` 접두사: `text_encoder`, `vision_encoder`, `tactile_encoder`, `adapter`, `fusion`, `head` …

## Flow matching 규약 (`heads.py`)

**τ = 0 이 노이즈, τ = 1 이 데이터.** `x_τ = τ·a + (1−τ)·ε`, 목표 속도 `u = a − ε`, 손실 `‖v_θ(x_τ, τ, obs) − u‖²`
(유효 스텝만). 추론: `x_0 = ε` 에서 `x_{k+1} = x_k + (1/K)·v_θ(x_k, k/K)` 로 τ = 1 까지 Euler K 스텝.
학습 τ: `uniform` 또는 `beta` = Beta(1, b) (b > 1 이면 노이즈 쪽 가중). eval 모드의 손실(`Trainer.evaluate` 의
`val/loss`)은 `eval_seed` 로 매번 같은 (ε, τ) 를 뽑으므로 epoch 간 비교(최적 체크포인트 선택)에 샘플링 잡음이 없다.
π0 등 논문마다 τ 방향 규약이 다르므로
robot_skin 에서는 이 규약만 쓴다. 근거: Flow Matching (Lipman et al., arXiv:2210.02747), Rectified Flow
(Liu et al., arXiv:2209.03003), π0 (arXiv:2410.24164).

## 데이터셋 (`VTLADataset`)

- 샘플 = `data.phases` (기본 `task` = reach…retreat; 보정/싱크/기저선 블록은 모방하지 않음) 안의 policy tick
  `t` (200 Hz master 인덱스, `stride = round(200 / policy_hz)`; `data.sample_stride: 1` 이면 모든 프레임).
- `actions[H,A]`: `a[t + (offset+i)·stride]` (offset 1 → `actions[0]` 은 한 tick 뒤 상태), `rel_mode`
  (`delta` 기본: 손목 위치를 현재 상태 기준으로) → `ActionNormalizer` (train 에서 fit). `action_valid[H]` 는
  에피소드 끝을 넘거나 `hand_pose_valid` 가 거짓인 스텝에서 False (ACT `is_pad`).
- `proprio[k·A]`: 최근 k tick 의 현재 action 공간 상태 (정규화). `tactile_values[N,F]`, `taxel_pos/nrm[N,3]`
  (손/로봇 base frame), `contact[N]`, `images{cam: [(k,)3,h,w]}` 또는 `vision_feats{cam: [(k,)P,D]}`,
  `vision_valid{cam: [k]}` (첫 프레임 이전 False), `instruction`, `task_id`.
- **촉각 입력 원천**: contact 스테이지의 derived `residual_z` / `contact_level` 이 있으면 그것을 쓴다. 없으면
  (`tactile_source: auto`) **부트스트랩** — 에피소드의 `no_contact` 프레임 ΔS 중앙값을 정적 기저선으로, MAD σ,
  z 임계값 + % 하한 — 을 계산하고 경고한다. 움직임 아티팩트 모델이 없으므로 **테스트/파이프라인 점검 전용**이다.
- `make_observation(...)` 이 관측 dict 를 만든다. 온라인 제어도 같은 함수로 단일 샘플을 만들고
  `collate_vtla([obs], tokenizer)` → `policy.predict(batch)` 한다.

## policy_bundle.pt

스테이지 설정은 `DEFAULTS` 에 없는 키(최상위 및 각 섹션 한 단계; `train`/`image` 제외)를 거부한다
(`stages.vtla.check_config`) — 오타나 `model.horizon` 처럼 스테이지가 `policy.*` 에서 유도하는 키가 조용히 무시되지 않는다.

`torch.load(weights_only=True)` 로 읽힌다 (`read_policy_bundle`). `tactile.source` 가 `bootstrap` 또는 `mixed` 이면
부트스트랩 촉각 레벨로 학습된 것이므로 배포하지 않는다. 섹션:

| 키 | 내용 |
|---|---|
| `model_config`, `state_dict` | `VTLAConfig` + 최적(EMA) 가중치 |
| `action` | `spec` (ActionSpec), `normalizer` (ActionNormalizer), `rel_mode`, `chunk_offset` |
| `proprio` | `normalizer`, `history`, `source` (= action 공간 상태) |
| `tactile` | `feature_spec`, `contact_rule`, `source` (derived / bootstrap), `calibrator` / `baseline_model` / `pretrained_encoder` 참조, `layouts` |
| `vision` | `cameras`, `encoder` cfg, `eval_transform` 파라미터, `cached_features_key`, `transform_config` |
| `language`, `timing`, `head`, `meta` | 텍스트 인코더 cfg; `policy_hz`, `source_hz`, `stride`, `horizon`, `obs_history`; head 종류·flow 스텝; split·지표 |

`build_policy_from_bundle(bundle)` → eval 모드 `VTLAPolicy` (인코더는 `pretrained: false` 로 만들고 가중치는
state_dict 에서). `bundle_components(bundle)` → 정책 + 정규화기 + `TactileFeatureSpec` + `EvalTransform` +
토크나이저 + 타이밍. 실행: `predict` 출력(정규화) → `action_normalizer.unnormalize` → `make_absolute(chunk, state,
spec, rel_mode)` → `TemporalEnsembler` (ACT, `meta.ensemble_k`) → (hand_mano) retarget.

## DPO 훅

VTLA 논문(arXiv:2505.09577)은 삽입 과제에서 선호 학습(DPO)을 쓴다. `dpo_loss` 는 일반 DPO 손실,
`preference_loss(policy, reference, batch)` 는 연속 chunk 의 우도 대용치(chunk: 고정 σ 가우시안, flow: 공유
(ε, τ) 에서의 flow-matching 오차)로 계산한다. 각 모델은 관측을 한 번만 인코딩해 chosen/rejected 가 공유하고, 기본값
`disable_dropout=True` 로 정책의 dropout / modality dropout 을 끈다 (기울기는 유지) — 그래야 정책 = 참조일 때 손실이
정확히 log 2 이다. 선호 쌍 수집(성공/실패 롤아웃)은 아직 데이터가 없어
`build_preference_pairs` 가 `NotImplementedError` 와 절차 설명을 낸다.

## 테스트

`robot_skin/tests/test_vtla.py` (adapter), `test_vtla_model.py` (모델·헤드·손실·bundle·DPO),
`test_vtla_dataset.py` (합성 D2 → preprocess → dataset, chunk 정렬, 스테이지 end-to-end + bundle 재현).
