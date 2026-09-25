# robot_skin/train — 공용 학습 인프라

모든 stage(imu_pose → baseline → contact → pretrain → vtla)가 공유하는 학습 엔진이다.
같은 stage config 하나로 **RTX 5090 워크스테이션, RTX 4090/3090 박스, A100 노드, CPU 노트북**
어디서든 돌도록 device·precision·batch 를 하드웨어 프로파일로 분리했다. 여러 머신은 **Tailscale**
로 묶어 ssh/rsync/작업 분배에 쓴다.

기존 SATS 학습(`deformable_sats/sats/training/train_e2e.py`)의 run 디렉터리 관례(`config.json`,
`history.json`, best/last 체크포인트, bf16 autocast)를 그대로 잇고, DDP·재개·스윕을 더했다.

| 모듈 | 내용 |
|---|---|
| `engine.py` | `TrainConfig`(dataclass, YAML `train:` 섹션) · `Trainer`(AMP, grad accumulation, clip, EMA, compile, DDP, best/last ckpt, 조기 종료, 재개) · `seed_everything` · `move_to_device` |
| `hardware.py` | `resolve_device`, `resolve_precision`(→`PrecisionPlan`), `enable_tf32`, 하드웨어 프로파일 로드/적용, `describe_environment`, `check_arch_support`(RTX 5090 sm_120 검사) |
| `distributed.py` | `init_distributed`(torchrun env, 없으면 no-op) · `wrap_ddp` · `make_sampler`/`make_eval_sampler` · `barrier` · `all_reduce_sum/mean` · `unwrap_model` |
| `optim.py` | `param_groups`(bias/norm/embedding/token/query 는 weight decay 제외, `lr_mult`) · `build_optimizer`(AdamW, CUDA fused 자동) · `build_scheduler`(warmup + cosine/linear/constant) · `EMA` |
| `checkpoint.py` | 원자적 저장(tmp → fsync → rename) · `load_checkpoint` · `find_last`/`find_best` |
| `logging_utils.py` | `JsonlLogger`(metrics.jsonl, TensorBoard/W&B 는 설치돼 있을 때만) |
| `sweep.py` | `expand_grid` / `sample_random` / `run_sweep` / `shard`(머신별 분할) / Optuna(선택) + CLI |
| `../configs/hardware/*.yaml` | `rtx5090`, `rtx4090`, `rtx3090`, `a100`, `cpu` 프로파일 |

---

## 1. 빠른 사용 (Python API)

```python
from robot_skin.train import Trainer, TrainConfig, maybe_apply_hw_profile

cfg = maybe_apply_hw_profile(stage_cfg, stage="vtla")    # stage_cfg["hardware"] = "rtx5090" | "auto" | ...

def loss_fn(model, batch):                  # 반드시 스칼라 "loss" 포함, 나머지 스칼라는 평균·로그
    out = model(batch)
    return {"loss": out["loss"], "l1": out["l1"]}

trainer = Trainer(model, loss_fn, TrainConfig.from_dict(cfg["train"]),
                  train_ds, val_ds, collate_fn=collate_vtla,
                  extra_state={"normalizers": norm.to_dict()})   # ckpt["extra"] 로 저장
history = trainer.fit()                    # epoch 별 dict 리스트

# 추론: EMA 가중치(있으면) 로드
Trainer.load_model_weights(model, "robot_skin/runs/vtla/ckpt_best.pt", use_ema=True)
```

* 학습 중 `loss_fn` 은 DDP/compile 래퍼를 받으므로 `model(...)`(forward)만 호출한다. 커스텀 메서드가
  필요하면 `robot_skin.train.unwrap_model(model)`. 평가 때는 EMA 가중치가 들어간 bare 모듈을 받는다.
* `callbacks=[fn]`: epoch 끝마다 `fn(trainer, row) -> dict|None` 호출, 반환 dict 는 history row 에
  합쳐져 `monitor` 로 쓸 수 있다(예: `val/auroc`, `monitor_mode: max`). `trainer.should_stop = True` 로 중단.

### 주요 `TrainConfig` 키 (YAML `train:`)

| 키 | 기본값 | 설명 |
|---|---|---|
| `max_epochs` / `max_steps` | 10 / null | `max_steps`(optimizer step) 가 있으면 우선 |
| `batch_size`, `grad_accum` | 64, 1 | **유효 배치 = batch_size × grad_accum × GPU 수** |
| `lr`, `weight_decay`, `optimizer`, `betas` | 3e-4, 0.05, adamw, [0.9, 0.999] | `lr_mult: {vision: 0.1}` 로 사전학습 인코더만 느리게 (0 = 동결) |
| `schedule`, `warmup_steps`, `min_lr_ratio` | cosine, 100, 0.1 | optimizer step 단위 |
| `grad_clip` | 1.0 | null 이면 끔 |
| `precision` | auto | auto / bf16 / fp16 / fp32 (아래 규칙) |
| `compile`, `compile_mode` | false, null | `torch.compile` opt-in |
| `ema_decay` | null | 예: 0.999 → 평가·best 체크포인트는 EMA 가중치 기준 |
| `monitor`, `monitor_mode`, `early_stop_patience` | val/loss, min, null | val 데이터가 없으면 `train/loss` 로 대체(경고) |
| `resume` | null | `auto` → `out_dir/ckpt_last.pt` 에서 이어서 (없으면 새로 시작) |
| `ckpt_every_steps` | null | 긴 원격 학습용 중간 저장(epoch 중간 재개 가능) |
| `num_workers`, `pin_memory` | 0, true | 프로파일이 덮어씀 |
| `device` | auto | auto / cpu / cuda / cuda:N / mps |

YAML 1.1 특성상 `lr: 3e-4` 는 **문자열**로 읽히는데 `TrainConfig.from_dict` 가 float 로 변환한다.
모르는 키는 경고 후 무시한다.

---

## 2. 하드웨어 프로파일

```bash
python -m robot_skin.train.hardware          # 이 머신의 GPU/torch/arch list 요약 + 추천 프로파일 + 경고
```

적용 우선순위: **stage YAML `train:` < 프로파일 `suggest.<stage>` < 프로파일 `train:` < CLI override**.
`hardware: auto` 는 GPU 이름의 모델 토큰으로 프로파일을 고른다(RTX 5090/4090/3090, A100; GPU 없으면 cpu).
노트북 변형(Laptop/Mobile/Max-Q)이나 메모리가 프로파일 `gpu.memory_gb` 의 90 % 미만인 GPU(A100 40 GB, `RTX A1000`
같은 이름만 비슷한 카드)는 프로파일 없이 stage 기본값을 쓴다(경고) — 큰 프로파일의 배치로 바로 OOM 나지 않게.

| 프로파일 | GPU (cc) | 메모리 | precision | compile | workers | vtla batch × accum |
|---|---|---|---|---|---|---|
| `rtx5090` | Blackwell (12.0, sm_120) | 32 GB | bf16 | ✔ | 8 | 64 × 2 |
| `rtx4090` | Ada (8.9) | 24 GB | bf16 | ✔ | 8 | 32 × 4 |
| `rtx3090` | Ampere (8.6) | 24 GB | bf16 | ✔ | 8 | 32 × 4 |
| `a100` | Ampere (8.0) | 80 GB | bf16 | ✔ | 16 | 128 × 1 |
| `cpu` | — | — | fp32 | ✘ | 0 | 8 × 1 (디버그용) |

GPU 프로파일끼리는 stage 별 **유효 배치를 동일하게** 맞춰 두었다(테스트로 강제). 그래서 어느 머신에서
돌려도 learning rate 를 다시 튜닝할 필요가 없다. 배치 값은 출발점이며, 로그의 `sys/gpu_mem_gb` 를 보고
조정한다(OOM → `batch_size` ↓, `grad_accum` ↑).

### Precision 규칙 (`resolve_precision`)

| 요청 | CUDA cc ≥ 8.0 (3090/4090/5090/A100) | CUDA cc < 8.0 (V100/T4) | CPU | MPS |
|---|---|---|---|---|
| `auto` | **bf16**, GradScaler 없음 | fp16 + GradScaler | fp32 | fp32 |
| `bf16` | bf16 | fp16 + GradScaler (경고) | bf16 autocast | fp32 (경고) |
| `fp16` | fp16 + GradScaler | fp16 + GradScaler | fp32 (경고) | fp32 (경고) |
| `fp32` | fp32 (+TF32 matmul) | fp32 | fp32 | fp32 |

bf16 은 fp32 와 지수 범위가 같아 loss scaling 이 필요 없다. CUDA 에서는 `tf32: true` 이면
`torch.set_float32_matmul_precision("high")` 로 fp32 matmul 에도 TF32 텐서코어를 쓴다
(torch ≥ 2.9 의 신규 `fp32_precision` API 와 섞으면 오류가 나므로 레거시 API 하나만 사용).

---

## 3. RTX 5090 (Blackwell, sm_120) 세팅

RTX 5090 은 compute capability **12.0 (sm_120)** 이다. **CUDA 12.8 이상으로 빌드된 torch** 에만
sm_120 커널이 들어 있다. 그보다 오래된 wheel 도 import 는 되지만 첫 CUDA 연산에서
`no kernel image is available for execution on the device` 로 죽는다.
이 repo 는 `deformable_sats/requirements.txt` 에 **`torch==2.9.0+cu128`** 을 고정해 두었다
(기존 SATS 학습용 `.venv` 를 그대로 써도 된다).

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy pyyaml pillow                    # robot_skin 기본 의존성
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
#   → CUDA 12.8, arch list 에 'sm_120' 이 있어야 함
python -m robot_skin.train.hardware                     # 경고가 없어야 정상
```

* NVIDIA 드라이버도 Blackwell 을 지원해야 한다(R570 계열 이상). `nvidia-smi` 가 GPU 를 보는데
  `torch.cuda.is_available()` 가 False 면 CPU 전용 wheel 이거나 드라이버/CUDA 불일치 —
  `check_arch_support()` 가 이 경우도 경고한다.
* Trainer 는 `env.json` 에 `describe_environment()` 결과(호스트명, torch/CUDA, GPU, arch list,
  경고)를 남긴다 → 머신마다 결과를 비교할 때 근거가 된다.
* `compile: true`: 첫 epoch(및 입력 shape 가 바뀔 때)는 컴파일 때문에 느리다. 짧은 실험·가변 길이
  입력이면 끄는 편이 낫다. Windows 는 Triton 이 없어 자동으로 건너뛴다(경고).
* 메모리 단편화: 프로파일 `env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (`apply_profile_env(profile)` 를 CUDA 초기화 전에 호출).

| 증상 | 원인 / 조치 |
|---|---|
| `no kernel image is available for execution on the device` | sm_120 없는 torch → cu128 wheel 설치 |
| CUDA 초기화 시 "... sm_120 is not compatible with the current PyTorch installation" 경고 | 위와 동일 |
| `torch.cuda.is_available() == False` (nvidia-smi 는 정상) | CPU wheel 또는 드라이버 불일치 |
| compile 중 Triton 오류 | torch wheel 에 포함된 triton 버전 사용(따로 설치한 triton 제거), 또는 `compile: false` |
| OOM | `batch_size` ↓ + `grad_accum` ↑ (유효 배치 유지), `vtla` 는 vision feature cache 사용 |

---

## 4. 한 머신 여러 GPU (DDP, torchrun)

```bash
# 2-GPU 한 대: 유효 배치를 유지하려면 grad_accum 을 GPU 수로 나눈다 (5090 vtla: 64×2×1 → 64×1×2)
torchrun --standalone --nproc_per_node=2 -m robot_skin train vtla \
    --config robot_skin/configs/stages/vtla.yaml --hardware rtx5090 --set data.splits=robot_skin/runs/splits.json
```
(`python -m robot_skin train <stage>` CLI 는 top-level `robot_skin/__main__.py` 가 제공한다.
직접 스크립트를 쓸 때도 `Trainer` 가 torchrun 환경변수를 읽어 알아서 DDP 로 동작한다.)

* `init_distributed()` 는 `WORLD_SIZE` 가 없거나 1 이면 아무것도 하지 않는다 → 같은 코드가 단일
  GPU/CPU 에서 그대로 돈다. CUDA 면 NCCL, 아니면 gloo.
* train sampler = `DistributedSampler`(epoch 마다 `set_epoch`), 검증은 패딩 없는 샤드
  (`ShardSampler`) + (합, 개수) all-reduce → **검증 지표가 GPU 수와 무관하게 정확**.
* gradient accumulation 중에는 `no_sync()` 로 all-reduce 를 optimizer step 당 1 회만 한다.
  각 micro-batch loss 는 그룹 내 샘플 비율로 가중되므로(마지막 micro-batch 가 짧아도) `accum × B` 가
  `accum·B` 한 배치와 정확히 같다(Trainer 가 만든 loader 기준; 사용자 `DataLoader`·`IterableDataset` 은
  micro-batch 크기를 미리 알 수 없어 동일 가중).
* 파일(ckpt/로그/history)은 rank 0 만 쓴다.
* modality dropout 처럼 step 마다 일부 파라미터가 gradient 를 못 받으면 `find_unused_parameters: true`.
* 소비자용 GPU(4090/5090)는 NVLink 가 없고 GPU P2P 도 보통 비활성이다. all-reduce 가 멈추면 `NCCL_P2P_DISABLE=1`,
  원인 확인은 `NCCL_DEBUG=INFO`.

---

## 5. 여러 머신 — Tailscale

### 권장: 머신마다 독립 single-node 학습, Tailscale 은 ssh/rsync/작업 분배용

동기식 DDP 는 가장 느린 GPU 와 가장 느린 링크에 맞춰 돈다. 5090 + 4090 을 묶으면 5090 이 4090 을
기다리고, Tailscale(WireGuard) 터널 대역폭은 PCIe/NVLink 보다 훨씬 낮아 gradient all-reduce 가
병목이 된다. 그래서 **각 머신이 서로 다른 job(다른 stage, 다른 seed, 스윕 샤드)을 돌리게** 한다.

```bash
# 0) 머신 확인 (MagicDNS 이름 또는 100.x.y.z)
tailscale status
ssh arm4090 'cd ~/sensor_training && .venv/bin/python -m robot_skin.train.hardware'

# 1) 코드·전처리 데이터 동기화 (run 산출물은 제외). processed episode 의 camera_<name> 은 기본이 raw 세션을
#    가리키는 절대 경로 symlink 라 -L(링크를 따라가 프레임 복사)로 보낸다 — docs/TRAINING.md §9
rsync -azL --exclude 'robot_skin/runs/' --exclude '.venv/' ~/sensor_training/ arm4090:~/sensor_training/

# 2) 원격 실행 — tmux/nohup 로 SSH 끊겨도 계속, resume=auto 로 재실행 시 이어서
ssh arm4090 'cd ~/sensor_training && tmux new -d -s vtla \
  ".venv/bin/python -m robot_skin train vtla --hardware auto --set data.splits=robot_skin/runs/splits.json --set train.resume=auto"'

# 3) 모니터링 (metrics.jsonl 은 한 줄 = 한 기록)
ssh arm4090 'tail -n 3 ~/sensor_training/robot_skin/runs/vtla/metrics.jsonl'

# 4) 결과 회수: summary.json 이 있는(=fit 완료) run 만 가져온다
ssh arm4090 'test -f ~/sensor_training/robot_skin/runs/vtla/summary.json' && \
  rsync -az arm4090:~/sensor_training/robot_skin/runs/vtla/ robot_skin/runs/vtla_arm4090/
```
(`--set` 형태의 override 문법은 top-level CLI 가 정한다. 핵심은 `train.resume: auto`.)
완주한 run 만 동기화하는 방식은 `deformable_sats/scripts/sync_models.sh` 와 같은 원칙이다
(여기서는 `summary.json` 을 완료 표시로 쓴다).

**스윕을 머신별로 나누기** — 같은 space·seed 로 trial 목록을 만들고 `--shard i/n` 으로 서로 다른
부분을 돌린 뒤 `results.jsonl` 을 합친다(trial 번호·디렉터리는 전역으로 유일):
```bash
# 5090
python -m robot_skin.train.sweep --fn robot_skin.stages.vtla:run --base robot_skin/configs/stages/vtla.yaml \
    --space sweep_vtla.yaml --metric val/l1 --out robot_skin/runs/sweeps/vtla --shard 0/2 --hardware rtx5090
# arm4090
python -m robot_skin.train.sweep ... --out robot_skin/runs/sweeps/vtla --shard 1/2 --hardware rtx4090
# 합치기 (원격 결과를 rsync 로 가져온 뒤)
python -c "from robot_skin.train import load_results; print(load_results('runs_5090/sweeps/vtla', 'runs_4090/sweeps/vtla')[:3])"
```
Optuna 가 설치돼 있으면 `run_optuna(..., storage="postgresql://<tailscale-host>/optuna")` 로 여러 머신이
한 study 를 공유할 수도 있다.

### 필요할 때만: 노드 간 DDP over Tailscale

```bash
# 두 머신 모두 (torch/CUDA/NCCL 버전, 코드, 데이터 경로 동일해야 함)
export NCCL_SOCKET_IFNAME=tailscale0 GLOO_SOCKET_IFNAME=tailscale0 NCCL_IB_DISABLE=1
MASTER=$(tailscale ip -4 <node0-hostname>)        # node 0 의 100.x.y.z

# node 0
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --master_addr=$MASTER --master_port=29500 \
    -m robot_skin train vtla --hardware auto --set data.splits=robot_skin/runs/splits.json
# node 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 --master_addr=$MASTER --master_port=29500 \
    -m robot_skin train vtla --hardware auto --set data.splits=robot_skin/runs/splits.json
```
* `NCCL_SOCKET_IFNAME=tailscale0` 이 없으면 NCCL 이 LAN/도커 인터페이스를 골라 연결이 안 되거나 멈춘다.
* **대역폭이 낮다고 가정**하라. `tailscale ping <host>` 가 `via DERP` 로 나오면(직접 연결 실패, 릴레이 경유)
  사실상 쓸 수 없다. `grad_accum` 을 키우면 all-reduce 횟수가 줄어(no_sync) 통신 비중이 작아진다.
* GPU 종류가 다르면 느린 쪽 속도로 맞춰진다 → 이 경우 이득이 거의 없다. 같은 GPU 가 여러 대 있는
  한 머신 안의 DDP(4절)가 훨씬 효율적이다.
* `init_distributed(timeout_s=1800)` — 느린 링크에서 초기화/첫 collective 가 오래 걸릴 수 있다.
* **재개(`train.resume=auto`)에는 모든 노드가 같은 체크포인트를 봐야 한다.** 체크포인트는 rank 0 만 쓰므로 노드마다
  로컬 `runs/` 이면 다른 노드에는 없다 → 공유 `out_dir` 을 쓰거나 node 0 의 `ckpt_last.pt` 를 다른 노드의 같은
  경로로 복사한다. rank 마다 재개 지점(체크포인트 유무, step/epoch)이 다르면 `Trainer.resume()` 이 모든 rank 에서
  `RuntimeError` 로 멈춘다(예전에는 replica 가 서로 다른 가중치로 조용히 갈라지거나 collective 가 멈췄다).

---

## 6. 체크포인트 · 재개

run 디렉터리(`train.out_dir`):

```
config.json    TrainConfig + world_size + effective_batch_size
env.json       describe_environment() (호스트, GPU, torch/CUDA, arch list, 경고)
metrics.jsonl  step 로그(kind=step: train/*, lr, grad_norm, samples/s, gpu_mem) + epoch 로그(kind=epoch)
history.json   epoch 별 row (train/*, val/*, callback 지표, lr, time_s)
ckpt_last.pt   재개 지점 (ckpt_every_epochs / ckpt_every_steps / 종료·중단 시)
ckpt_best.pt   monitor 기준 최고 (EMA 사용 시 EMA 가중치로 평가한 값 기준)
summary.json   fit 완료 표시 (finished, stopped_early, best, step, epoch, time)
previous/      재개가 아닌 새 run 이 이전 run 의 파일이 있는 out_dir 에서 시작하면, 이전 run 의
               ckpt_last/ckpt_best/metrics.jsonl/history.json 을 여기로 옮긴다 (한 벌만 유지)
```

체크포인트 키: `model, optimizer, scheduler, scaler, ema, step, epoch, config, extra`
(+ `best, bad_epochs, history, rng, batch_in_epoch`). 저장은 임시 파일 → fsync → rename 으로 원자적이라
저장 중 전원이 나가도 이전 체크포인트가 깨지지 않는다.

* `resume: auto` 는 RNG·데이터 위치(epoch 중간 포함)까지 복원해 **중단 없는 학습과 같은 결과**를 낸다
  (단일 프로세스에서 테스트로 검증). Ctrl-C 시에도 `ckpt_last.pt` 를 저장한다(마지막 optimizer step
  기준). 단, epoch 중간 재개에서 dropout 같은 확률적 layer 나 worker 안의 random augmentation 은 난수
  흐름이 달라져 bit 단위로 같지는 않다. DDP 재개는 rank 별 RNG 를 새로 seed 한다.
* 재개 시 LR schedule 은 *현재* config 로 계산된다(재개하면서 `max_epochs` 를 늘리는 것이 가능). Trainer 가 만든
  optimizer 면 최고 LR(`lr` × `lr_mult`)·`weight_decay`·`betas`·`eps` 도 현재 config 값을 다시 적용하고(바뀌었으면
  경고, moment 는 유지) 복원된 step 의 LR 을 그 값으로 다시 계산한다 — 낮춘 `lr` 로 이어 가기가 된다. `optimizer=`
  로 직접 넘긴 optimizer 는 체크포인트 값을 그대로 쓴다. `batch_size`/`grad_accum`/`seed` 를 바꾸거나 학습 데이터
  크기가 체크포인트와 다르면(다른 split·processed root) 경고한다. `extra_state` 를 넘기면 그 값이 유지된다(넘기지
  않았을 때만 체크포인트의 `extra` 를 복원).
* early stopping 으로 끝난 run 은 `resume: auto` 로 다시 띄워도 학습하지 않는다(`stopped_early` 가 체크포인트에
  있다). `early_stop_patience` 를 늘리거나 끄거나, budget(`max_epochs`/`max_steps`)을 늘리면 이어서 학습한다.
* `resume: auto` 는 `ckpt_last.pt`, 없으면 가장 최근 `ckpt*.pt` 만 찾는다 — 같은 디렉터리의 stage 산출물
  (`policy_bundle.pt`, `encoder_state.pt`, `*_model.pt`)은 재개 대상이 아니다(없으면 새로 시작).
* 재개가 아닌 새 run 은 이전 run 의 파일을 `previous/` 로 옮기고 시작한다. stage 들은 *이번* run 이 best 를 기록했을
  때만 `ckpt_best.pt` 를 내보내고, 유한한 monitor 값이 한 번도 없으면 EMA(없으면 마지막) 가중치를 쓴다(경고).
* 파이프라인(`python -m robot_skin pipeline`)은 끝난 stage 를 다시 학습할 때(`--force`, 다른 splits 등) `resume` 을
  무시하고 처음부터 학습한다 — `docs/TRAINING.md` §3.
* 추론 로딩: `Trainer.load_model_weights(model, path, use_ema=True)` — DDP/compile 접두사(`module.`,
  `_orig_mod.`)는 자동 제거.

---

## 7. 스윕 (`sweep.py`)

```yaml
# sweep_vtla.yaml
mode: random          # grid | random
n: 16
seed: 0
space:
  train.lr: {log_uniform: [1.0e-4, 1.0e-3]}
  train.weight_decay: {uniform: [0.0, 0.1]}
  model.fusion_depth: {int: [2, 6]}
  model.head: [chunk, flow]          # 리스트 = 선택지 (grid 에서는 축)
  train.betas: [[0.9, 0.95]]         # 리스트 자체를 고정값으로 쓰려면 한 번 더 감싼다
```

* `run_sweep(train_fn, base_cfg, overrides, out_dir)` — trial 마다 `out_dir/trial_XXX` 를
  `train.out_dir` 에 넣고(base 에 최상위 `out_dir` 이 있으면 — stage 는 그쪽을 먼저 본다, 예: 파이프라인의
  `pipeline_config.yaml` — 그것도 바꾼다), 끝날 때마다 `results.jsonl` 에 한 줄 추가. 다시 실행하면 성공한 trial 은
  건너뛴다. `run_optuna`·`suggest_from_space`·`trial_config` 도 `robot_skin.train` 에서 바로 import 된다. `train_fn` 은 float 또는 dict(stage `run(cfg)` 의 metrics) 를 반환, `--metric` 으로 키 지정.
* stage metrics 의 키는 stage 마다 다르다. `best.value`(monitor 값, 기본 `val/loss`)는 모든 stage 에 있고, 최상위
  `val/loss` 는 pretrain 에만 있다 — vtla 는 `val/l1`, baseline `val/nll`, imu_pose `val/rot_deg`, contact
  `val/z_auroc`(`--direction max`). 목록은 `docs/TRAINING.md` §11–12.
* stage 1(imu_pose/baseline/contact) trial 은 space 에 `predict.write_derived: [false]` 를 넣어 episode 의
  derived 배열(다음 stage 입력)을 덮어쓰지 않게 한다.
* 실패한 trial(없는 `--metric` 키 포함)은 `status: failed` + 에러로 기록되고 정렬에서 맨 뒤.
* 하드웨어 프로파일(`--hardware`, 없으면 base YAML 의 `hardware`, space 의 `hardware` 축도 가능)은 trial 마다
  `trial_config` 로 stage 의 `load_stage_config` 와 같은 순서로 적용된다: 프로파일 → trial override, 그리고
  `hardware_applied: true`. 그래서 `train.batch_size`·`precision`·`grad_accum`·`compile` 을 스윕하면 그 값으로
  학습한다(stage 의 `run()` 이 프로파일을 다시 덮어쓰지 않는다). `suggest.<stage>` 는 base 의 `stage` 키(없으면
  `--fn` 모듈의 `STAGE`)로 고른다.

---

## 8. 테스트

```bash
python -m pytest -q robot_skin/tests/test_train_engine.py robot_skin/tests/test_hardware.py \
    robot_skin/tests/test_sweep.py
```
CPU 전용·결정적. 검증 항목: 회귀 수렴, grad accumulation 동치(accum=2×B ≡ 1×2B, 마지막 그룹이 짧은 경우
포함), 재개 = 무중단 학습(epoch 경계 및 epoch 중간), best-by-monitor, EMA 평가, 조기 종료, 중첩 배치
device 이동, `TrainConfig.from_dict`, precision 규칙(capability mock), weight-decay 그룹, warmup+cosine 값,
fp16 GradScaler 경로(CPU fp16 autocast 로 대체 실행) 및 재개, torchrun env 없을 때 no-op,
**실제 2-프로세스 gloo DDP**(2 rank × batch 8 × accum 2 ≡ 1 프로세스 × batch 32, 샤딩된 검증 지표 정확;
IterableDataset 의 마지막 짧은 accumulation 그룹도 rank 평균 gradient; 노드 로컬 `out_dir` 재개는 모든 rank 에서
오류, 공유 `out_dir` 재개는 동일), 재개 시 현재 lr/weight decay 적용·early stop 된 run 재개는 no-op·다른 학습 데이터
경고·stage 산출물 무시, 새 run 의 `previous/` 이동, run 마다 TF32 설정, 프로파일 유효성/유효 배치 일치, GPU 이름 매칭
(`RTX A1000` ≠ A100, 노트북·메모리 부족 → 없음), sm_120 arch 검사, 스윕 grid/random/샤딩/`--hardware`(스윕한
batch/precision 이 실제 stage `resolve_config` 뒤에도 유지).
NCCL·CUDA 전용 경로(fused AdamW, CUDA 에서의 `torch.compile`, 노드 간 DDP)는 GPU 머신에서
`python -m robot_skin.train.hardware` 와 `torchrun --standalone --nproc_per_node=1` smoke run 으로 확인한다.
