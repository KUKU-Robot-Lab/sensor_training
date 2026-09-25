# Training — stage 실행, 설정, 하드웨어, 여러 GPU·머신

`python -m robot_skin` 으로 학습 stage 를 돌리는 방법이다. 설계 이유는 [`ARCHITECTURE.md`](ARCHITECTURE.md), 정책
모델은 [`VTLA.md`](VTLA.md), 데이터 수집·포맷은 [`DATA_ACQUISITION.md`](DATA_ACQUISITION.md) /
[`DATA_FORMAT.md`](DATA_FORMAT.md), 로봇 실행은 [`DEPLOYMENT.md`](DEPLOYMENT.md) 에 있다. 학습 엔진(`Trainer`,
AMP, DDP, EMA, 체크포인트)의 세부는 [`robot_skin/train/README.md`](../robot_skin/train/README.md) 가 기준이다.

```
raw 세션 ─ preprocess ─▶ processed Episode ─ splits ─▶ splits.json
   imu_pose ─▶ baseline ─▶ contact ─▶ pretrain ─▶ vtla ─▶ policy_bundle.pt ─ deploy ─▶ 로봇
   (stage 1: 촉각 해석)                (stage 2)    (stage 3)
```

이 문서의 명령은 모두 저장소 루트에서 실행한다. 여기 적은 CLI 명령은 이 저장소의 CPU VM(4 코어, GPU 없음)에서
작은 규모로 실제로 실행해 확인했다. 예외: `rsync`, `ssh`, `tailscale`, `nvidia-smi` 가 필요한 명령과 GPU 전용 경로
(NCCL, CUDA 에서의 `torch.compile`, bf16)는 이 환경에 해당 도구·장치가 없어 **실행하지 못했다**. 해당 절에 따로 표시한다.

---

## 1. 준비

### 1.1 환경

```bash
pip install -r requirements.txt           # deformable_sats 고정 버전 + PyYAML, pytest
python -m robot_skin env                  # torch/CUDA 버전, GPU, arch list, 추천 하드웨어 프로파일
```

선택 의존성(없어도 기본 경로는 모두 돈다): torchvision(ResNet 인코더), transformers(HF 비전/텍스트 인코더),
tensorboard / wandb(로그), optuna(스윕). RTX 5090 은 CUDA ≥ 12.8 torch 가 필요하다(§5.2).

### 1.2 데이터

```bash
python -m robot_skin preprocess                    # robot_skin/data/raw → robot_skin/data/processed
python -m robot_skin preprocess --raw <dir> --out <dir> --force   # 다른 경로, 이미 있는 episode 도 다시
```

전처리 설정은 `robot_skin/configs/stages/preprocess.yaml` (`--set KEY=V` 로 덮어쓰기, 모르는 키는 오류). 이미 만든
episode 는 건너뛰고, 설정·버전·raw 파일 구성이 바뀌었으면 `stale` 로 보고한다. `--force` 는 episode 의 `derived/`
(stage 1 결과)까지 지운다 — 그 뒤에는 stage 1 부터 다시 돌려야 한다.

### 1.3 하드웨어 없이 전체 경로 확인 (합성 데이터)

```bash
R=/tmp/rs
python -m robot_skin synth --out $R/raw                     # 합성 glove 세션 8개 (D1 4 + D2 4, 8 s, 피험자 s0–s3)
python -m robot_skin preprocess --raw $R/raw --out $R/processed
python -m robot_skin pipeline --processed $R/processed --out $R/runs --hardware cpu \
    --set train.max_steps=30 --set train.warmup_steps=5 --set vtla.image.image_size='[24, 32]'
python -m robot_skin deploy --set bundle=$R/runs/vtla --set duration_s=2 --set out_dir=$R/deploy
```

이 VM 에서 잰 시간: `synth` 4.0 s, `preprocess` 4.1 s, `pipeline` 17.4 s(다섯 stage), `deploy` 5.5 s — 합쳐 약 30 초.
모델이 작고 30 step 이라 지표는 의미가 없다 — 배관 점검용이다(예: contact 검출기는 AUROC 가 높아도 0.5 문턱 recall
이 0 이다, §4.3). `synth` 출력은 실제 데이터와 섞지 않는다(`--out` 을 생략하면 `robot_skin/data/synthetic`).

---

## 2. CLI 와 설정

| 명령 | 하는 일 |
|---|---|
| `python -m robot_skin preprocess ...` | `datasets.build` (raw → Episode) |
| `python -m robot_skin splits ...` | processed root 전체에 대한 `splits.json` 하나 |
| `python -m robot_skin train <stage> ...` | stage 하나: `imu_pose` \| `baseline` \| `contact` \| `pretrain` \| `vtla` |
| `python -m robot_skin pipeline ...` | splits(한 번) → 다섯 stage 를 순서대로, 산출물 연결, 재개 |
| `python -m robot_skin deploy ...` | `stages.deploy` (정책 번들을 로봇에서; `robot: fake` = 시뮬레이션) |
| `python -m robot_skin env` / `sweep ...` | `train.hardware` 환경 보고 / `train.sweep` 하이퍼파라미터 스윕 |
| `python -m robot_skin synth` / `record` / `postprocess` / `qc` | 합성 데이터 / 수집 ([`DATA_ACQUISITION.md`](DATA_ACQUISITION.md)) |

`python -m robot_skin <command> --help` 가 각 명령의 인자를 보여 준다. `train <stage>` 는
`python -m robot_skin.stages.<stage>` 와 같은 일을 하되 하드웨어 프로파일과 torchrun 정리를 더 해 준다.

### 2.1 설정 계층

| 파일 | 내용 |
|---|---|
| `robot_skin/configs/default.yaml` | 최상위 기본값만: `paths`(raw / processed / runs / synthetic 루트), 기본 `hardware`, stage → 설정 파일 매핑(`stages:`), `pipeline.stages`·`pipeline.splits`, `synthetic` |
| `robot_skin/configs/stages/<stage>.yaml` | 각 stage 의 모든 키(= 모듈의 `DEFAULTS`, 테스트로 동기화). **모르는 키는 오류** — 오타가 조용히 무시되지 않는다 |
| `robot_skin/configs/hardware/<name>.yaml` | 장치·정밀도·compile·worker 수, stage 별 배치 제안(`suggest.<stage>`), 환경 변수(`env`) |

적용 순서(뒤가 이긴다): **stage YAML `train:` < 프로파일 `suggest.<stage>` < 프로파일 `train:` < `--set`**.
`--set KEY=VALUE` 는 점 경로이고 값은 YAML 로 읽는다: `--set train.lr=1e-3`, `--set policy.cameras='[ego, third]'`,
`--set train.lr_mult='{vision_encoder: 0.1}'`. 어떤 프로파일을 쓰나: `train` 은 `--hardware` > `--set hardware=…` >
`default.yaml` `hardware` > stage YAML `hardware`. `pipeline` 은 `--set [<stage>.]hardware=…` > `--hardware` >
`default.yaml` `hardware` 라서 stage 마다 다른 프로파일을 줄 수 있다(예: `--hardware rtx4090 --set
vtla.hardware=rtx5090`). 프로파일의 `env`(예: `PYTORCH_CUDA_ALLOC_CONF`)는 첫 CUDA 호출 전에 export 된다.

`train <stage>` 의 데이터·출력 경로는 **stage YAML 기준**이다(기본 `data.processed_root: robot_skin/data/processed`,
`train.out_dir: robot_skin/runs/<stage>`) — `default.yaml` `paths` 를 읽지 않는다. 다른 경로는 `--set` 으로 준다.

---

## 3. 파이프라인 한 번에 (`pipeline`)

```bash
python -m robot_skin pipeline --hardware rtx5090          # robot_skin/data/processed → robot_skin/runs/<stage>
python -m robot_skin pipeline --processed <dir> --out <dir> --hardware cpu --set train.max_steps=200
python -m robot_skin pipeline --stages vtla --force       # 한 stage 만 다시 (splits.json 은 유지)
```

- **splits**: `<out>/splits.json` 을 처음 한 번 만든다(`datasets.splits.make_splits`, 기본 subject 단위, val 0.15 /
  test 0.15, `default.yaml` `pipeline.splits`; `--split-by`, `--val-frac`, `--test-frac`, `--split-seed` 로 변경).
  경로는 processed root 기준 상대 경로라 데이터를 옮겨도 쓸 수 있다. 같은 파일이 **모든 stage 의 `data.splits`** 로
  들어간다. 기존 파일을 쓰려면 `--splits <file>`, 다시 나누려면 파일을 지우고 `--force`.
- **순서와 연결**: imu_pose → baseline → contact → pretrain → vtla (`--stages` 로 부분집합; 순서는 항상 이대로).
  각 stage 는 `out_dir = <out>/<stage>`, `data.processed_root = --processed` 로 돈다. baseline → contact,
  contact → pretrain/vtla 는 episode 의 derived 배열로 이어지고, vtla 에는 `tactile.calibrator`
  (`<out>/contact/calibrator.json`), `tactile.baseline_model`, `tactile.pretrained`(`<out>/pretrain/encoder_state.pt`),
  `data.tactile_source: derived` 가 자동으로 들어간다(부트스트랩 촉각 대체 경로를 쓰지 않는다).
- **`--set` 라우팅**: `--set KEY=V` 는 그 키를 가진 선택된 stage **모두**에, `--set <stage>.KEY=V` 는 그 stage
  에만 간다(stage 지정이 이긴다). 어느 stage 에도 없는 키는 오류.
- **`--config`**: `STAGE=YAML` 또는 `<stage>.yaml` 파일들이 있는 디렉터리(반복 가능).
- **재개**: `metrics.json` 과 주 산출물이 있고, 같은 processed root 와 같은 splits.json(sha256)으로 학습됐고, 그
  stage 의 derived 배열이 episode 에 남아 있으면 건너뛴다. 한 stage 가 다시 돌면 뒤 stage 도 모두 다시 돈다.
  `--force` 는 선택된 stage 를 모두 다시 돌린다. 설정만 바꿔서는 재실행되지 않는다 — `--force` 를 준다.
- **기록**: `<out>/<stage>/pipeline_config.yaml`(실제로 쓴 설정 전체), `<out>/pipeline.json`(상태, splits 경로·
  sha256, upstream stage, 자동 연결된 키, 소요 시간). 기록 안의 경로는 절대 경로다 — processed root 경로가 다른
  머신에서 같은 runs 디렉터리로 재개하면 "다른 processed root" 로 보고 다시 학습한다.

주의 — **processed root 하나에는 stage-1 결과 한 벌**: stage 1 은 예측을 episode 의 `derived/` 에 쓴다. 같은
processed root 로 설정이 다른 파이프라인을 두 번 돌리면 뒤의 것이 앞의 derived 를 덮어쓴다. 비교 실험은 processed
root 를 복사해서 하거나, 스윕처럼 결과만 필요하면 `predict.write_derived=false` 를 준다(§11).

---

## 4. stage 별 실행과 산출물

모든 stage 는 `run(cfg) -> metrics` 이고, `<out_dir>/metrics.json`(엄격한 JSON)과 학습 run 파일(`config.json`,
`env.json`, `metrics.jsonl`, `history.json`, `ckpt_last.pt`, `ckpt_best.pt`, 끝나면 `summary.json`)을 쓴다.
단독 실행에서는 항상 파이프라인과 같은 `splits.json` 을 준다 — 없으면 stage 마다 따로 나누고 경고한다.

```bash
S=robot_skin/runs/splits.json
python -m robot_skin splits --out $S                                  # 한 번 (이미 있으면 --force 없이는 거부)
python -m robot_skin train imu_pose --hardware rtx5090 --set data.splits=$S
python -m robot_skin train baseline --hardware rtx5090 --set data.splits=$S
python -m robot_skin train contact  --hardware rtx5090 --set data.splits=$S
python -m robot_skin train pretrain --hardware rtx5090 --set data.splits=$S
python -m robot_skin train vtla     --hardware rtx5090 --set data.splits=$S --set data.tactile_source=derived \
    --set tactile.calibrator=robot_skin/runs/contact --set tactile.baseline_model=robot_skin/runs/baseline \
    --set tactile.pretrained=robot_skin/runs/pretrain
```

`splits` 옵션: `--by subject|session|object|task|dataset|kind|episode_id` (쉼표로 복합 키), `--holdout subject=S07`
(→ test) 또는 `--holdout val:task=pour`, `--datasets motion,task`.

### 4.1 `imu_pose` — IMU → MANO 손가락 자세

- 입력: 손 라벨(`hand_pose_valid`)이 있는 episode 의 IMU(`imu_quat/gyro/acc`). 학습 풀 `data.datasets: [motion]`.
- 모델: `pose.imu_model.ImuHandPoseNet` (GRU 기본, `model.arch: tcn` 가능), 인과 창 `model.window` 32 프레임,
  손실 = 측지 회전 + `loss.tip_weight`·손끝 거리.
- 산출물: `imu_pose_model.pt`, `imu_stats.json`; 모든 `predict_datasets` episode 에 derived `hand_finger_pose_imu`.
- 용도: 카메라 없는 글러브 경로(`baseline`/`contact` 의 `data.q_source: hand_pose_imu`).

### 4.2 `baseline` — 무접촉 ΔS(motion artefact) 예측

- 입력: D1 의 무접촉 프레임(`contact_label ∈ data.only_labels`, 기본 0), 창 전체에 측정된 `q`/`q̇`.
- 모델: `baseline.TemporalBaselinePredictor` (인과 TCN, 창 32, 평균 + log 분산). **관측 ΔS 는 입력이 아니다.**
- 산출물: `baseline_model.pt`(정규화 버퍼·`bundle_meta.qd` 포함, 온라인 재현용), `joint_stats.json`; 모든
  `predict_datasets` episode 에 derived `baseline_pred`, `baseline_logvar`, `residual`.
- 옵션: `data.q_source: q | hand_pose_imu`, `data.kind: glove | robot` (레이아웃 하나당 모델 하나 — 섞인 데이터면 지정).

### 4.3 `contact` — 보정 z, 레벨, 검출기, D2 pseudo 라벨

- 입력: baseline 의 derived `residual`/`baseline_logvar`. 보정은 D1 **val** split 의 무접촉 프레임
  (`calibration.split`), 검출기는 D1 라벨(self-touch 1, 무접촉 0).
- 산출물: `calibrator.json`(σ·c·g·임계값·FSM 설정), `contact_detector.pt`; 모든 episode 에 derived `residual_z`,
  `contact_level`, `contact_prob`, D2 에 `contact_label_pseudo`. 전처리 `contact_label` 은 수정하지 않는다.
- 옵션: `detector.bootstrap: auto` — train split 에 self-touch 라벨이 없으면(로봇 D1) 접촉 허용 phase 의 STRONG 을
  양성으로 자기학습. `fsm.enabled` (포화 복구 게이트), `pseudo_label.proximity_m` (손–물체 근접 veto).
- 검출기는 짧게 학습하면 AUROC 는 높아도 0.5 문턱 recall 이 낮다(합성 e2e 의 30 step 에서 recall ≈ 0). 100 step
  이상을 주고, 로봇 D1 bootstrap 은 더 길게 준다.

### 4.4 `pretrain` — 촉각 인코더 MAE 사전학습

- 입력: 모든 episode(D1 + D2)의 derived `residual_z`/`contact_level` + `saturated` → `tactile_value_features`.
- 모델: `representation.TaxelEncoder` + `MaskedTaxelPretrainer`(마스크 비율 0.3, `random | group | mixed`).
- 산출물: `encoder_state.pt` (`load_pretrained_encoder`; 특징 스펙 포함). vtla 의 `tactile.pretrained` 로 쓴다.
- `data.use_test: false` 가 기본 — VTLA test episode 로 사전학습하지 않도록 같은 splits.json 을 쓴다.

### 4.5 `vtla` — 정책

- 입력: D2 task episode, `data.phases: task` (reach…retreat) 안의 policy tick, 카메라, 지시문, 촉각(derived).
- 산출물: `policy_bundle.pt` — 배포에 필요한 모든 것([`VTLA.md`](VTLA.md) §6).
- `data.tactile_source` 기본값은 `auto`: derived 가 없는 episode 는 정적 **부트스트랩** 촉각으로 대체하고
  경고한다(움직임 모델 없음 — 테스트용). 실제 학습은 `derived` 로 고정한다(파이프라인은 자동). 번들의
  `tactile.source` 가 `bootstrap`/`mixed` 이면 deploy 가 거부한다.

### 4.6 산출물 요약

| stage | `<out>/<stage>/` 주 산출물 | episode `derived/` | 다음 소비자 |
|---|---|---|---|
| `imu_pose` | `imu_pose_model.pt`, `imu_stats.json` | `hand_finger_pose_imu` | baseline/contact (`q_source: hand_pose_imu`) |
| `baseline` | `baseline_model.pt`, `joint_stats.json` | `baseline_pred`, `baseline_logvar`, `residual` | contact; 번들 `tactile.baseline_model`; deploy `stage1.baseline_model` |
| `contact` | `calibrator.json`, `contact_detector.pt` | `residual_z`, `contact_level`, `contact_prob`, `contact_label_pseudo` | pretrain, vtla; 번들 `tactile.calibrator` |
| `pretrain` | `encoder_state.pt` | — | vtla `tactile.pretrained` |
| `vtla` | `policy_bundle.pt` | (선택) `vision_<key>_<camera>.npy` 특징 캐시 | deploy |

---

## 5. 하드웨어 프로파일 · RTX 5090

### 5.1 프로파일

| 프로파일 | GPU (compute capability) | 메모리 | precision | compile | workers | vtla batch × grad_accum |
|---|---|---|---|---|---|---|
| `rtx5090` | Blackwell (12.0, sm_120) | 32 GB | bf16 | true | 8 | 64 × 2 |
| `rtx4090` | Ada (8.9) | 24 GB | bf16 | true | 8 | 32 × 4 |
| `rtx3090` | Ampere (8.6) | 24 GB | bf16 | true | 8 | 32 × 4 |
| `a100` | Ampere (8.0) | 80 GB | bf16 | true | 16 | 128 × 1 |
| `cpu` | — | — | fp32 | false | 0 | 8 × 1 (디버그용) |

GPU 프로파일끼리는 stage 별 **유효 배치(batch_size × grad_accum × GPU 수)** 가 같도록 맞춰져 있어(테스트로 강제)
머신을 바꿔도 learning rate 를 다시 맞출 필요가 없다. `--hardware auto` 는 GPU 이름으로 프로파일을 고른다.
`cpu` 프로파일의 배치는 디버그용이라 GPU 유효 배치와 다르다.

### 5.2 RTX 5090 (Blackwell, sm_120)

- **torch 요구사항**: sm_120 커널은 **CUDA ≥ 12.8 로 빌드된 torch** 에만 있다. 오래된 wheel 도 import 는 되지만 첫
  CUDA 연산에서 `no kernel image is available for execution on the device` 로 죽는다. 이 저장소는
  `deformable_sats/requirements.txt` 에 `torch==2.9.0+cu128` 을 고정했다. 드라이버도 Blackwell 지원(R570 계열 이상)이어야 한다.
  ```bash
  pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
  ```
- **arch 확인**:
  ```bash
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"   # 'sm_120' 이 있어야 함
  python -m robot_skin env            # = python -m robot_skin.train.hardware; 문제가 있으면 경고 (--json 가능)
  ```
  `train.hardware.check_arch_support()` 는 GPU capability 가 torch arch list 에 없거나, Blackwell(cc ≥ 10)인데 torch 의
  CUDA 가 12.8 미만이면(PTX 항목이 있어도) 경고한다. Trainer 는 run 마다 `env.json` 에 같은 정보를 남긴다.
  (참고: 이 개발 VM 의 torch `2.14.0+cu130` 의 arch list 에도 `sm_120` 이 있다 — GPU 는 없다.)
- **bf16**: 프로파일 `precision: bf16`. bf16 은 fp32 와 지수 범위가 같아 GradScaler 가 필요 없다. `tf32: true` 로
  fp32 matmul 에도 TF32 텐서코어를 쓴다. `precision: auto` 도 cc ≥ 8.0 이면 bf16 을 고른다.
- **compile**: 프로파일 `compile: true` (`torch.compile`). 첫 epoch 와 입력 shape 가 바뀔 때 느리다 — 짧은 실험이나
  shape 가 자주 바뀌면 `--set train.compile=false`. Triton 오류가 나면 torch wheel 에 포함된 triton 을 쓰거나 끈다.
- **메모리**: `env.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. OOM 이면 `batch_size` ↓, `grad_accum` ↑
  (유효 배치 유지). vtla 는 frozen 비전 인코더의 특징 캐시(§13)로 크게 줄어든다.

증상별 조치는 [`robot_skin/train/README.md`](../robot_skin/train/README.md) §3 표.

---

## 6. 단일 GPU

```bash
python -m robot_skin pipeline --hardware rtx5090
python -m robot_skin train vtla --hardware rtx5090 --set data.splits=robot_skin/runs/splits.json
python -m robot_skin train vtla --hardware auto --set train.device=cuda:1      # 두 번째 GPU
```

`CUDA_VISIBLE_DEVICES=1 python -m robot_skin ...` 도 같다. 로그는 `metrics.jsonl` 과 `history.json` 이다.
`metrics.jsonl` 에는 `train.log_every` step 마다 `kind: step` 기록(`train/*`, `lr`, `train/grad_norm`,
`sys/samples_per_s`, CUDA 면 `sys/gpu_mem_gb`)과 epoch 마다 `kind: epoch` 기록(`train/*`, `val/*`)이 한 줄씩 쌓인다.
`train.tensorboard: true`, `train.wandb_project: <name>` 은 패키지가 설치돼 있을 때만 켜진다.

## 7. 한 머신 여러 GPU (torchrun)

```bash
torchrun --standalone --nproc_per_node=2 -m robot_skin train vtla --hardware rtx5090 \
    --set data.splits=robot_skin/runs/splits.json --set train.grad_accum=1
torchrun --standalone --nproc_per_node=2 -m robot_skin pipeline --hardware rtx5090 --set vtla.train.grad_accum=1
```

- Trainer 가 torchrun 환경 변수를 읽어 DDP 로 돈다(CUDA → NCCL, 아니면 gloo). rank 0 만 파일을 쓰고 출력한다.
  CLI 는 끝날 때 process group 을 정리한다.
- **유효 배치 유지**: 프로파일 배치는 GPU 1 개 기준이다. GPU n 개면 `grad_accum` 을 n 으로 나눈다(5090 vtla:
  64 × 2 × 1 GPU → 64 × 1 × 2 GPU).
- 검증은 패딩 없는 샤드 + all-reduce 라 지표가 GPU 수와 무관하게 정확하다.
- VTLA 의 modality dropout 은 key-padding 으로 가려 그래프를 유지하므로 `find_unused_parameters` 가 필요 없다.
  직접 만든 브랜치가 step 마다 gradient 를 못 받으면 `--set train.find_unused_parameters=true`.
- 소비자용 GPU 는 NVLink 가 없고 GPU P2P 도 보통 꺼져 있다(`rtx4090` 프로파일 노트). 한 머신 안에서 all-reduce 가
  멈추면 `NCCL_P2P_DISABLE=1`, 진단은 `NCCL_DEBUG=INFO`.

검증: `torchrun --standalone --nproc_per_node=2 -m robot_skin train imu_pose --hardware cpu --set data.splits=...`
를 이 VM 에서 CPU 2 rank(gloo)로 돌려 `config.json` 의 `world_size: 2` 를 확인했다(e2e 테스트는 `pipeline --stages
imu_pose,baseline` 도 torchrun 2 rank 로 돌린다). NCCL/GPU 경로는 실행하지 못했다.

## 8. 여러 머신 — Tailscale

> 이 절의 `ssh`/`rsync`/`tailscale`/노드 간 torchrun 명령은 이 환경에 해당 도구가 없어 실행하지 못했다.
> 로컬 복사로 같은 흐름(아래 §9 의 인계)을 흉내 내 검증했다.

**권장: 머신마다 독립된 단일 노드 학습(필요하면 그 안에서 DDP). Tailscale 은 ssh·rsync·작업 분배용.**
동기식 DDP 는 가장 느린 GPU 와 가장 느린 링크에 맞춰 돈다. 5090 과 4090 을 묶으면 5090 이 기다리고, Tailscale
(WireGuard) 터널은 PCIe/NVLink 보다 훨씬 느려 gradient all-reduce 가 병목이 된다. 그래서 머신마다 **다른 일**을 준다:

- stage 분담: 가벼운 stage 1 + pretrain 은 3090/4090 박스, vtla 는 5090 박스(§9 의 인계 절차).
- 스윕 샤드: 같은 space·seed 로 `--shard 0/2`, `--shard 1/2` 를 나눠 돌리고 `results.jsonl` 을 합친다(§11).
- seed 반복·ablation(`obs_mode`, `head`)을 머신별로.

```bash
tailscale status                                                    # MagicDNS 이름 / 100.x.y.z
ssh box5090 'cd ~/sensor_training && .venv/bin/python -m robot_skin env'
ssh box5090 'cd ~/sensor_training && tmux new -d -s vtla \
  ".venv/bin/python -m robot_skin train vtla --hardware auto --set data.splits=robot_skin/runs/splits.json --set train.resume=auto"'
ssh box5090 'tail -n 3 ~/sensor_training/robot_skin/runs/vtla/metrics.jsonl'
ssh box5090 'test -f ~/sensor_training/robot_skin/runs/vtla/summary.json' && \
  rsync -az box5090:~/sensor_training/robot_skin/runs/vtla/ robot_skin/runs/vtla_box5090/   # 끝난 run 만 회수
```

**노드 간 DDP over Tailscale (필요할 때만)**:

```bash
export NCCL_SOCKET_IFNAME=tailscale0 GLOO_SOCKET_IFNAME=tailscale0 NCCL_IB_DISABLE=1
MASTER=$(tailscale ip -4 <node0>)
ARGS="-m robot_skin train vtla --hardware auto --set data.splits=robot_skin/runs/splits.json"
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr=$MASTER --master_port=29500 $ARGS   # node 0
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 --master_addr=$MASTER --master_port=29500 $ARGS   # node 1
```

- `NCCL_SOCKET_IFNAME=tailscale0` 이 없으면 NCCL 이 LAN/도커 인터페이스를 골라 연결이 안 되거나 멈춘다.
  Tailscale 에는 InfiniBand 가 없으므로 `NCCL_IB_DISABLE=1`.
- `tailscale ping <host>` 가 `via DERP` 이면 직접 연결이 안 되고 릴레이를 거친다 — 사실상 쓸 수 없다.
- 모든 노드가 torch/CUDA/NCCL 버전, 코드, 데이터 경로가 같아야 한다. master port 가 tailnet ACL 에서 열려 있어야 한다.
- GPU 종류가 다르면 느린 쪽에 맞춰진다. `grad_accum` 을 키우면 all-reduce 횟수가 줄어 통신 비중이 작아진다.
- 느린 링크에서는 초기화·첫 collective 가 오래 걸릴 수 있다(`init_distributed` 기본 timeout 1800 s).

전체 절차는 [`robot_skin/train/README.md`](../robot_skin/train/README.md) §5.

## 9. 데이터 동기화

processed episode 를 다른 머신으로 옮길 때 알아야 할 것:

- **카메라 디렉터리는 기본이 절대 경로 symlink** 다(`cameras.copy_frames: symlink`, raw 세션을 가리킴). `rsync -a`
  는 symlink 를 그대로 옮기므로 받는 쪽에 같은 절대 경로의 raw 가 없으면 끊긴다. 방법은 셋 중 하나:
  `rsync -aL`(링크를 따라가 프레임 복사), 전처리 때 `--set cameras.copy_frames=copy`, 또는 raw 루트도 같은 경로로 동기화.
- **stage 1 결과는 episode 안(`derived/`)** 에 있다. stage 1 을 A 머신에서 돌리고 vtla 를 B 머신에서 돌리려면
  processed root 를 derived 까지 옮기고, `runs/` 의 `splits.json`, `contact/`, `baseline/`, `pretrain/` 도 옮긴다
  (파이프라인이 vtla 에 연결하는 파일). vision 특징 캐시(`derived/vision_*.npy`)도 같이 간다.
- `splits.json` 경로는 processed root 기준 상대라 그대로 쓸 수 있다. `pipeline.json` 의 경로는 절대 경로다(§3).

```bash
# A (stage 1 + pretrain) → B (vtla)
python -m robot_skin pipeline --stages imu_pose,baseline,contact,pretrain --hardware rtx4090
rsync -aL robot_skin/data/processed/ box5090:~/sensor_training/robot_skin/data/processed/
rsync -a robot_skin/runs/splits.json robot_skin/runs/baseline robot_skin/runs/contact robot_skin/runs/pretrain \
      box5090:~/sensor_training/robot_skin/runs/
ssh box5090 'cd ~/sensor_training && .venv/bin/python -m robot_skin pipeline --stages vtla --hardware rtx5090'
```

마지막 명령은 `<out>/splits.json` 을 그대로 쓰고, `contact/calibrator.json` 이 있으므로 vtla 에
`data.tactile_source: derived` 와 stage-1 참조를 연결한다. (이 흐름을 이 VM 에서 `cp -rL` 로 흉내 내 확인했다:
vtla 가 `tactile_source: derived` 로 돌고 `pipeline.json` 에 세 참조가 연결됐다.)

## 10. 재개

- **학습 중단 → 이어서**: `--set train.resume=auto` 이면 `out_dir/ckpt_last.pt` 에서 이어 간다(없으면 새로 시작).
  optimizer·scheduler·scaler·EMA·RNG·epoch 중간 위치까지 복원한다. Ctrl-C 에도 `ckpt_last.pt` 를 남긴다. 긴 원격
  학습은 `--set train.ckpt_every_steps=500` 처럼 중간 저장을 켠다. dropout·worker 안 augmentation 이 있으면 epoch
  중간 재개는 비트 단위로 같지 않고, DDP 재개는 rank 별 RNG 를 새로 seed 한다. `max_steps`/`max_epochs` 를 늘려서
  재개할 수 있다.
  ```bash
  python -m robot_skin train baseline --set train.resume=auto --set train.max_steps=20000
  ```
- **파이프라인**: 끝난 stage 를 건너뛰는 규칙은 §3. 중간 stage 가 실패하면 고친 뒤 같은 명령을 다시 실행한다.
- **추론용 로딩**: `Trainer.load_model_weights(model, path, use_ema=True)` (DDP/compile 접두사 자동 제거).

## 11. 스윕 · HPO

```yaml
# sweep_baseline.yaml
mode: grid            # grid | random
space:
  train.lr: [1.0e-3, 3.0e-3]
  model.hidden: [64, 128]
  data.splits: [robot_skin/runs/splits.json]     # 원소 하나짜리 리스트 = 고정값
  predict.write_derived: [false]                 # trial 이 episode 의 derived 를 덮어쓰지 않게
# random 예: train.lr: {log_uniform: [1.0e-4, 1.0e-3]}, model.n_layers: {int: [2, 6]}, 리스트 값 자체는 [[24, 32]]
```

```bash
python -m robot_skin sweep --fn robot_skin.stages.baseline:run --base robot_skin/configs/stages/baseline.yaml \
    --space sweep_baseline.yaml --metric best.value --out robot_skin/runs/sweeps/baseline --hardware rtx4090
python -m robot_skin sweep ... --shard 0/2      # 머신 A      (--shard 1/2 는 머신 B, 같은 space·seed)
```

- trial 마다 `<out>/trial_XXX/` 가 `train.out_dir` 로 들어가고, 끝날 때마다 `<out>/results.jsonl` 에 한 줄 추가된다.
  다시 실행하면 성공한 trial 은 건너뛴다. 결과 합치기: `robot_skin.train.load_results(dir_a, dir_b)`.
- **`--metric`** 은 `run(cfg)` 가 돌려주는 metrics 의 키(점 경로 가능)다. stage 마다 키가 다르다: **`best.value`** 는
  모든 stage 에 있는 monitor 값(기본 `val/loss`, 최소화)이다. 그 밖에 imu_pose `val/rot_deg`, baseline `val/nll` ·
  `val/mae_resid`, contact `val/z_auroc`(`--direction max`), pretrain `val/loss`, vtla `val/l1`. 최상위 `val/loss` 는
  **pretrain 에만** 있다 — 다른 stage 에 `--metric val/loss` 를 주면 모든 trial 이 `failed` 로 기록된다.
- stage 1 trial 은 `predict.write_derived: [false]` 로 episode 의 derived 배열을 보존한다.
- `optuna` 가 설치돼 있으면 `robot_skin.train.run_optuna(...)` 로 TPE/pruning, 여러 머신이 한 study 를 공유할 수 있다.

(이 VM 에서 합성 데이터로 `train.lr` 축 하나(2 trial, `train.max_steps: [20]`, `data.processed_root`·`data.splits` 는
space 에 고정값으로)만 위 명령 형태로 `--hardware cpu --metric best.value` 실행해 두 trial 모두 `status: ok` 를
확인했다.)

## 12. 평가 — `metrics.json` 읽기

모든 stage 공통: `best` (`{monitor, value, epoch}`), `steps`, `epochs`, `final_train_loss`, `n_episodes`
(split 별 개수, `skipped`), `n_samples`, `skipped`(건너뛴 episode 와 이유). split 접두사 `val/`, `test/` 는
splits.json 의 split 이다 — **튜닝·조기 종료에 쓴 val 이 아니라 test 를 최종 수치로 본다.** 합성 데이터에는 정답
기반 `gt_*` 지표가 붙는다.

| stage | 핵심 키 | 읽는 법 |
|---|---|---|
| `imu_pose` | `{val,test}/rot_deg`, `tip_mm`, 기준 `rot_deg_flat`, `tip_mm_flat` | 평평한 손(학습 전 출력)보다 얼마나 나은가 |
| `baseline` | `{val,test,task}/mae_raw`, `mae_resid`, `resid_reduction` (= 1 − mae_resid/mae_raw), `rmse_*`, `nll`, `coverage_2sigma`, `z_std`, `z_robust_std`, `sep_auroc_before/after`, `sep_dprime_before/after`; 합성 `gt_artefact_*` | 무접촉 ΔS 를 얼마나 지웠나. `coverage_2sigma` ≈ 0.95, `z_std` ≈ 1 이면 분산이 보정된 것. `sep_*_after` > `_before` 면 차감이 접촉과 움직임을 더 잘 가른다. `task/` = D2 무접촉 프레임(학습에 없던 자세) |
| `contact` | `{val,test}/{z,prob,hyst}_{hallucination_taxel, hallucination_frame, recall, precision, f1, auroc}`, 합성 `…_gt_auroc`, `…_gt_hallucination`, `…_gt_recall`; `pseudo/{n_contact, n_no_contact, n_unknown, gt_precision, gt_recall, …}`; `calibrator`, `detector_bootstrap` | `z` = 보정 규칙, `prob` = 검출기(0.5 문턱), `hyst` = 히스테리시스 후. 무접촉 환각률이 낮고 self-touch recall 이 높아야 한다. val 은 보정·조기 종료에 쓰여 낙관적이다 |
| `pretrain` | `val/loss`, `z_huber` vs `z_huber_zero`, `z_mae` vs `z_mae_zero`, `level_acc` vs `level_acc_majority`, `level_bal_acc`, `recall_{none,weak,strong,saturated}`, `contact_{precision,recall,f1}` | 가린 taxel 재구성이 "0 예측"·"다수 클래스"보다 나은가 |
| `vtla` | `{val,test}/l1` (정규화 단위), `l1_per_step[H]`, `l1_by_task`, `l1_raw/{wrist_pos, wrist_rot6d, finger_aa}` (원 단위, 상대 행동), `n_samples`, `n_valid_steps`; `tactile_source`, `head` | 오프라인 청크 오차만이다. `tactile_source` 는 `derived` 여야 한다. 성공률은 deploy/실기로 본다 |
| `deploy` | `loop_hz`, `loop_hz_wall`, `latency_p50_ms`/`p95_ms`, `tick_p50_ms`/`p95_ms`, `overruns`, `budget_ms`, `safety_counts`, `estop`, `contact_frac`, `retarget_ms`, `benchmark`, `notes` | [`DEPLOYMENT.md`](DEPLOYMENT.md) §6, §8 |

## 13. frozen 비전 인코더 특징 캐시

사전학습 비전 인코더를 고정(frozen)하면 매 step 이미지를 인코딩할 필요가 없다. vtla stage 가 한 번 캐시한다:

```bash
python -m robot_skin train vtla --set data.splits=robot_skin/runs/splits.json \
    --set vision.encoder='{type: dinov2, frozen: true}' --set vision.cache_features=true
```

- `vision.cache_features: true` 는 인코더를 고정으로 만들고, 학습 전에 train/val/test episode 의 모든 카메라 프레임을
  `derived/vision_<key>_<camera>.npy` (float16 `[F,P,D]`, 카메라 프레임 단위) + `.json` 사이드카로 쓴다. 키는
  `encoder.cache_key` + **인코더 가중치 해시**라 가중치가 바뀌면 옛 캐시를 쓰지 않는다. 이미 있는 유효한 캐시는 재사용한다.
- 캐시로 학습하면 **eval 변환**(augmentation 없음)을 쓴다. 번들은 `vision.cached_features_key` 와 eval 변환을 기록하고,
  배포에서는 캐시 대신 같은 인코더를 온라인으로 돌린다.
- 인코더만 따로 미리 캐시하는 CLI: `python -m robot_skin.vision.feature_cache --root <processed> --dataset task
  --encoder '{type: dinov2}' --image-size 224 224 [--bf16]`. 이 CLI 의 캐시 키는 `encoder.cache_key` 뿐이라(가중치 해시
  없음) vtla stage 는 그 파일을 재사용하지 **않는다** — 분석·다른 모델용이다. vtla 학습에는 `vision.cache_features` 를 쓴다.
- `dinov2`/`siglip`/`clip` 은 transformers, `resnet18/34/50` 은 torchvision 이 필요하다. 이 환경에서는 hub 접근이 막혀
  사전학습 가중치로는 시험하지 못했다 — 기본 `tiny` 인코더로 `cache_features=true` 경로만 확인했다.

## 14. 자주 겪는 문제

| 증상 | 원인 / 조치 |
|---|---|
| `UserWarning: warmup_steps=… >= total optimizer steps` | 짧은 스모크 run — `--set train.warmup_steps=5` |
| `data.splits is not set — splitting this stage's own episode pool` 경고 | `data.splits` 를 주지 않았다 — 파이프라인과 같은 splits.json 을 `--set data.splits=…` 로 준다 |
| `… entries without a usable episode, … usable episodes not listed (ignored)` 경고 | splits.json 이 이 processed root 에 없는 episode 를 가리키거나(다른 processed root, 지운 episode), splits.json 을 만든 뒤 추가된 episode 가 목록에 없다(학습에서 빠진다) — 새 데이터를 넣었으면 splits 를 다시 만들고 `--force`. 이 stage 풀 밖의 **존재하는** episode(다른 dataset)는 INFO 로만 기록된다 |
| stage 가 `taxel_frame` 경고 | build/1 로 만든 글러브 episode(월드 프레임 taxel) — `python -m robot_skin preprocess --force` 후 stage 1 부터 |
| vtla 가 "bootstrap" 경고, deploy 가 번들 거부 | contact stage 의 derived 가 없다 — contact 를 먼저 돌리고 `data.tactile_source=derived` |
| `--set` 키 오류 | stage YAML 에 없는 키(오타). `python -m robot_skin pipeline` 에서는 `<stage>.<key>` 로 한 stage 에만 |
| 스윕 trial 이 모두 `failed` | `--metric` 키가 그 stage metrics 에 없음(§11) — `results.jsonl` 의 `error` 확인 |
| `no kernel image is available` (RTX 5090) | sm_120 없는 torch → cu128 wheel (§5.2) |
| 파이프라인이 끝난 stage 를 다시 학습 | processed root 경로 또는 splits.json 이 바뀌었거나 derived 배열이 사라짐(`preprocess --force`) — §3 |
