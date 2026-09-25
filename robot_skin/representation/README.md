# representation/ — 촉각 값 특징 · taxel 인코더 · 마스킹 사전학습

| 파일 | 상태 | 역할 |
|---|---|---|
| `encoder.py` | 구현 | `tactile_value_features` (촉각 값 특징의 **단일 기준 함수**), `TactileFeatureSpec` / `TactileHistory` (k 프레임 스태킹, 오프라인·온라인 동일), `TaxelEncoder` (토크나이저 + pre-LN transformer), `encoder_state.pt` 저장·로드 |
| `tokenizer.py` | 구현 | `TaxelTokenizer`: 값 MLP + pose(Fourier(pos) ⊕ normal) MLP (+ 선택적 id 임베딩) → LayerNorm 토큰 `[B,N,D]`. `mask` 로 값 부분만 `[MASK]` 치환 |
| `pretrain.py` | 구현 | MAE 방식 `MaskedTaxelPretrainer`, `sample_taxel_mask` (random / layout group / mixed), `TaxelPretrainDataset` + `collate_pretrain` (레이아웃 혼합 배치 패딩), `evaluate_reconstruction`. `random_taxel_mask` 는 기존 API 그대로 |
| `../stages/pretrain.py` | 구현 | stage runner `run(cfg)` → `encoder_state.pt` + `metrics.json` (설정: `configs/stages/pretrain.yaml`) |

pose 기반 토큰이라 taxel 수·배치가 다른 글러브/로봇 핸드가 같은 인코더를 공유한다(id 임베딩을 끄면
순열 등변, taxel 수 무관).

## 1. 촉각 값 특징 — `tactile_value_features` (사전학습·VTLA·온라인 제어 공통)

입력은 contact stage 출력: `residual_z` (`derived/residual_z`, 보정된 z, **누르면 +**), `level`
(`derived/contact_level`, `ContactLevel` 0..3), `saturated` (`arrays/saturated`, 선택).
모드 이름은 `policy.OBS_MODES` 와 같다.

| obs_mode | 차원/taxel | 특징 |
|---|---|---|
| `full` | 6 | `[zf, 1[NONE], 1[WEAK], 1[STRONG], 1[SATURATED], sat]` |
| `ordinal` | 4 | level one-hot (= `OrdinalQuantizer.one_hot`) |
| `binary` | 1 | `1[WEAK or STRONG]` (포화 제외, `policy` binary 와 동일) |
| `none` | 0 | `[...,N,0]` — **촉각 브랜치를 끈다** (`tactile_value_dim(mode) == 0` 이면 인코더를 만들지 않는다) |

* `zf = asinh(clip(z, ±z_clip) / z_scale)` (기본 `z_clip=100`, `z_scale=2`): 잡음 대역에서는 `≈ z/2`
  선형, 강한 누름은 로그 압축. 역변환 `z_from_feature`.
* 포화(`saturated` 또는 `level == SATURATED`) taxel 은 잔차를 믿을 수 없으므로 `zf = 0`, level 은
  SATURATED 로 강제, `sat = 1` (`policy.tactile_features` 와 같은 규칙). NaN z → `zf = 0`, level 이
  0..3 밖(예: -1)이면 one-hot 전부 0.
* `full` 은 RL 관측(`policy` 의 full = 잔차% + sat, 2/taxel)보다 풍부하다. ordinal/binary/none 은 동일.
* numpy 입력 → float32 numpy, torch 입력 → 같은 device 의 tensor (모델 안에서 GPU 로 계산 가능).

### 시간 스태킹 (k 프레임)

`TactileFeatureSpec(obs_mode, history=k, stride=s)`: taxel n 의 값 벡터 =
`[f(t-(k-1)s) | … | f(t)]` (오래된 것 먼저, `s` 는 200 Hz master tick), 에피소드 시작 전 인덱스는
프레임 0 반복(인과적 edge padding). 값 차원 `spec.dim = frame_dim × k`.

```python
from robot_skin.representation import TactileFeatureSpec, TactileHistory, load_pretrained_encoder

enc = load_pretrained_encoder("robot_skin/runs/pretrain", freeze=True)   # 디렉터리 또는 .pt
spec = enc.feature_spec                           # 사전학습에 쓴 특징 레시피 그대로
vals = spec.from_episode(episode, t_index)        # 오프라인 (VTLADataset): [..., N, spec.dim]
hist = TactileHistory(spec)                       # 온라인 (control): 틱마다 push
vals_t = hist.push(residual_z_t, level_t, saturated_t)   # == 오프라인 결과 (테스트로 보장)
tokens = enc(torch.as_tensor(vals)[None], torch.as_tensor(pos)[None],     # pos/nrm [N,3]: hand/robot
             torch.as_tensor(nrm)[None])                                  # base frame → [1,N,D]
```

## 2. `TaxelEncoder`

`TaxelTokenizer` → `nn.TransformerEncoder`(pre-LN, batch_first, GELU) → 최종 LayerNorm → `[B,N,D]`.
`forward(values, pos, nrm, mask=None, key_padding_mask=None)`:

* `mask` `[B,N]` — 해당 taxel 값을 토크나이저의 `[MASK]` 로 치환(pose 유지). 주의: MAE 사전학습에서는
  가려진 taxel 이 어텐션 key 가 되지 않으므로 이 `[MASK]` 의 gradient 는 0 이고 초기값 그대로다.
  사전학습된 인코더에서 죽은/격리된 taxel 을 빼려면 `key_padding_mask` 를 쓴다(또는 downstream 에서
  `[MASK]` 를 fine-tune).
* `key_padding_mask` `[B,N]` (True = 무시) — 어텐션 key 에서 제외하고 출력은 0. 레이아웃 혼합 배치의
  패딩, MAE 사전학습의 가려진 taxel 에 쓴다. 한 샘플의 모든 taxel 을 가리면 `ValueError`.
* 입력은 파라미터 dtype 으로 변환(bf16 autocast 에서도 Fourier 위상은 fp32).
* `config` / `from_config`, `encode(residual_z, level, saturated, pos, nrm)` (history=1 편의 함수).

`encoder_state.pt` = `{format: "robot_skin/taxel_encoder", version, config (feature_spec 포함),
state_dict, meta}` — `save_pretrained_encoder` (원자적 저장) / `read_encoder_state` /
`load_pretrained_encoder(path, freeze=False)` (`torch.load(weights_only=True)`).

## 3. 마스킹 사전학습 (MAE, He et al. CVPR 2022, arXiv:2111.06377)

한 프레임의 taxel 집합에서 `round(mask_ratio·N)` 개를 가리고(≥ 1 개는 항상 보이게):

* **mask_mode** `random` · `group` (layout group — 손끝/손가락/손바닥 — 하나를 통째로 가리고 k 까지
  무작위 보충; 가장 가까운 이웃 복사가 아니라 영역 간 추론을 강제) · `mixed` (샘플마다 `group_prob`
  확률로 group). `max_group_frac`(기본 0.5)보다 큰 그룹(`all`, 9개 중 5개인 `fingertip`)은 제외.
* **인코더**: 가려진 taxel 을 어텐션 key 에서 제거 → 보이는 taxel 의 토큰 = 보이는 부분집합에만 인코더를
  돌린 결과와 같다(MAE: 인코더에 mask token 없음; 테스트로 확인). 가려진 값은 `[MASK]` 로 치환돼
  네트워크에 들어가지 않는다.
* **디코더**: 가벼운 pre-LN transformer (`decoder_depth=1`): 보이는 토큰(투영) + 가려진 자리의 학습된
  decoder mask token, 모두 decoder pose 임베딩을 더한다.
* **목표**(가려진 taxel, 현재 프레임): `zf` 회귀 (Huber, 포화/NaN 제외) + contact level 분류 (CE,
  NONE 이 대부분이라 `level_class_weights: balanced` = freq^-0.5, 기대 가중치 1 로 정규화, 최대 10).
* `forward(batch)` (= `loss(batch)`) → `{loss, z_huber, level_ce, z_mae, level_acc, mask_frac}`; Trainer 의
  `loss_fn = pretrain_loss` (forward 만 호출 → DDP/compile 안전). 모든 학습 파라미터가 매 step
  gradient 를 받는다(`find_unused_parameters` 불필요; 토크나이저 `[MASK]` 는 그래프에 있지만 gradient 0).
  쓰이지 않는 target(보이는 taxel, `z_valid=False`; NaN 이어도)은 loss·gradient 에 들어가지 않는다.
* eval 모드 마스크는 호출마다 `eval_seed` 로 재시드 → epoch 간 val loss 비교 가능.

### Stage runner (`robot_skin/stages/pretrain.py`)

```bash
python -m robot_skin.stages.pretrain --config robot_skin/configs/stages/pretrain.yaml \
    --set train.max_epochs=50 --set hardware=rtx5090
```

1. `data.processed_root` × `data.datasets` (기본 motion + task) 에피소드 탐색. contact stage 의
   `residual_z`/`contact_level` 이나 taxel pose 가 없는 에피소드는 건너뛰고 metrics 에 기록.
2. 에피소드(또는 `split_by: subject`) 단위 분할: `data.splits` (splits.json, 항목은 에피소드 경로 —
   절대, `processed_root` 기준 또는 splits.json 디렉터리 기준 — 또는 id; test 는 `use_test: true` 가
   아니면 제외) 또는 seed 고정 `val_frac`. **다운스트림(VTLA) 평가를 누수 없이 하려면 VTLA 와 같은
   splits.json 을 `data.splits` 로 준다** — 기본 `val_frac` 분할은 VTLA 의 test 에피소드도 (라벨 없이)
   사전학습에 쓸 수 있다.
3. `frame_stride`(기본 4 → 50 Hz) 로 프레임 샘플링, `contact_repeat` 로 접촉 프레임 과표집.
   taxel pose 가 NaN/inf 인 프레임은 건너뛰고(경고 1회), taxel 중심이 0.15 m 넘게 움직이는 에피소드
   (world 좌표계 `taxel_pos` 로 의심)는 경고 1회로 모아 알린다.
4. `robot_skin.train.Trainer` (`train:` = `TrainConfig`, 프로파일 `suggest.pretrain` 배치).
   우선순위: YAML `train` < 하드웨어 프로파일 < 명시적 override(`--set`). `load_stage_config` 가
   YAML 과 override 사이에서 프로파일을 적용하고 `hardware_applied: true` 로 표시한다(프로파일 `env`
   도 CUDA 초기화 전에 export). `run(cfg)` 에 raw dict 를 넘기면 `hardware_applied` 가 false 일 때만
   적용한다(이때는 프로파일이 dict 의 `train` 값을 이긴다).
5. best 체크포인트(EMA 있으면 EMA) 가중치로 `encoder_state.pt` 저장 + `metrics.json`:
   `val/…` = `evaluate_reconstruction` (가려진 taxel 전체에 대해 개수 가중 정확 평균: `z_mae` vs
   `z_mae_zero`(0 예측 baseline), `level_acc` vs `level_acc_majority`, 클래스별 recall,
   `level_bal_acc`, `contact_precision/recall/f1`), 분할 크기, 건너뛴 에피소드.
   history 의 `val/loss` 는 Trainer 의 배치 평균이라 metrics.json 의 값과 조금 다를 수 있다.

VTLA 는 `load_pretrained_encoder(out_dir)` 로 촉각 브랜치를 초기화하고(선택적으로 `freeze=True`),
**같은 `enc.feature_spec`** 으로 데이터셋의 촉각 값을 만든다.

## 참고
MAE (He et al., CVPR 2022), Fourier features (Tancik et al., NeurIPS 2020, arXiv:2006.10739),
3D-ViTac (arXiv:2410.24091) — 전체 목록과 검증 상태는 `docs/REFERENCES.md`.
