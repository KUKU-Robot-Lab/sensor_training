# 16-Channel Tactile Intelligence Framework

16채널 기압 기반 tactile sensor array 데이터로 sparse-to-dense pressure map,
XY heatmap, contact 기준 Z/Fz 회귀를 실험하는 workspace입니다.

현재 mk555 SATS 데이터의 공식 경로는 `skin_ws/raw_data`의 raw BIN archive를
`learning_data`로 정리한 뒤 `sats` 학습을 돌리는 흐름입니다. 기존
CSV/Zarr 기반 `hitmap` 경로는 XY/Z/Fz 회귀 실험용으로 유지됩니다.

## 현재 상태 (2026-07-12)

- **최종 모델 = `train_e2e` + 인덴터 크기 입력(A, FiLM conditioning)**. β GT 보정은 인프라만 보존(무이득 확정).
  소재 서열(d10 상대오차): **ecomesh 0.182 < eco20 0.259 < eco50 0.336** / 위치오차 ecomesh_xy1 0.79 mm.
- 데이터: xy1 소재 3종(각 6 trial) + ecomesh xy0.5(13 trial) 병합 완료. 진단은 `sats/tools/eval_diagnostics.py`.
- **밴딩 보상 모듈** `sats/bending/` Phase 0 완료 — 데이터 취득 대기.
- **논문 워크스페이스 = `history/fig_data/`** (Figure·분석·투고 로드맵). 진행 관리:
  `history/fig_data/SUBMISSION_CHECKLIST.md` · 구조 색인: `history/fig_data/PROJECT_STRUCTURE.md`.
- 실험 러너·검증 스크립트는 `scripts/` (구 루트 scratchpad_* — `scripts/README.md` 참조).

## Current Official SATS Flow

```text
skin_ws/raw_data/sats/eco20 + mesh/d5/testN/
  ├── due_raw_burst_*.bin
  ├── ethermotion_encoder_*.bin
  └── loadcell_raw_*.bin

learning_data/
  ├── sensor_raw_bin/ecomesh/d5/z_2.5mm/testN/*_merged.bin
  └── gt/ecomesh_d5_z2.5_testN_targets.npy  # legacy/precomputed GT

sats/training/
  ├── train_lstm.py
  ├── train_attention.py
  ├── train_local_map.py
  └── train_cnn.py
```

## Data Alignment Policy

- DUE effective sensor stream is 200 Hz. One raw DUE burst contains 10 FIFO
  frames, so `bin_merge.py` expands bursts into the 200 Hz tactile stream.
- Loadcell is aligned to the same 200 Hz common timeline and converted with
  `Fz = (kg - kg_baseline) * 9.80665`.
- EtherMotion is logged much faster, around 1000 Hz or higher, and provides
  high-resolution `x/y/z/u` labels. It is interpolated onto the 200 Hz common
  timeline; the final training row count is not expanded to EtherMotion rate.
- `z_stage_mm` and `z_depth_mm` preserve EtherMotion command precision at about
  `0.0001 mm`; do not bin Z coarsely when the goal is `0.xx mm` behavior.
- `u_mm` is a node-internal wait/virtual axis. It is not physical shear and is
  not the depth source. SATS uses all rows by default.

This means the recommended learning row is:

```text
input[t] = DUE s1..s16 at 200 Hz
Fz[t]    = loadcell Fz interpolated to t
Z[t]     = EtherMotion z interpolated to t
GT[t]    = Boussinesq pressure map from x/y/Fz/diameter
```

## GT Modes

Two GT paths are available:

```text
precomputed  : legacy path, stores learning_data/gt/*_targets.npy
on_the_fly   : CPU worker path, generates dense GT during DataLoader fetch
gpu_on_the_fly: optimized path, sends compact GT metadata and builds batch GT on GPU
```

The default remains `precomputed`, so existing runs are unchanged. The new
on-the-fly modes avoid writing dense GT files of about 18 GB per d5 trial.
For `0.25 mm`, `0.2 mm`, or `0.1 mm` output grids, prefer
`gpu_on_the_fly`; the CPU `on_the_fly` mode is mainly a compatibility/debug
path for the original `41 x 41` grid.

Current on-the-fly GT assumptions:

```text
z_s_mm                  = 2.0 fixed
beta/FEM correction     = none
indenter model          = spherical
contact radius          = sqrt(R * z_depth_mm), R = diameter / 2
contact starts at d5    = z_depth_mm > 0.001
minimum contact radius  = 0.05 mm
z-depth sample bins     = 0.005 mm
pressure map grid       = 41 x 41, 0.5 mm XY spacing
```

The high-resolution EtherMotion Z signal is preserved in `merged.bin` at about
`0.0001 mm`, but training samples are balanced by `0.005 mm` z-depth bins to
avoid over-weighting repeated plateau rows.

Example on-the-fly training command:

First build compact metadata cache from merged BIN data. This does not save
dense pressure maps; it stores normalized sensor windows and compact
`diameter/x/y/z_depth/Fz` metadata.

```bash
python3 -m sats.training.build_gt_meta_cache \
  --raw-dir learning_data/sensor_raw_bin \
  --out-dir learning_data/gt_meta_cache \
  --exclude-diameters 10 \
  --grid-step-mm 0.5
```

Then train from that cache:

```bash
python3 -m sats.training.train_e2e \
  --gt-mode gpu_on_the_fly \
  --raw-dir learning_data/sensor_raw_bin \
  --gt-meta-cache-dir learning_data/gt_meta_cache \
  --z-depth-min-mm 0.001 \
  --z-balance-bin-width-mm 0.005 \
  --min-contact-radius-mm 0.05 \
  --exclude-diameters 10 \
  --run-name e2e_d5_onthefly
```

High-resolution grid examples:

```bash
# 0.25 mm virtual grid, 81 x 81 output
python3 -m sats.training.train_e2e \
  --gt-mode gpu_on_the_fly \
  --raw-dir learning_data/sensor_raw_bin \
  --gt-meta-cache-dir learning_data/gt_meta_cache \
  --exclude-diameters 10 \
  --grid-step-mm 0.25 \
  --batch-size 2048 \
  --num-workers 6 \
  --prefetch-factor 2 \
  --run-name e2e_d5_gpu_gt_g025

# 0.1 mm virtual grid, 201 x 201 output. Start smaller because target memory
# grows about 24x compared with 41 x 41.
python3 -m sats.training.train_e2e \
  --gt-mode gpu_on_the_fly \
  --raw-dir learning_data/sensor_raw_bin \
  --gt-meta-cache-dir learning_data/gt_meta_cache \
  --exclude-diameters 10 \
  --grid-step-mm 0.1 \
  --batch-size 1024 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --run-name e2e_d5_gpu_gt_g010
```

Use a denser virtual XY pressure map by changing the GT/model grid, not by
recollecting raw data. For example, `81 x 81` over the same `[-10, 10] mm` area
gives `0.25 mm` virtual taxel spacing. The current raw scan centers are still
collected every `0.5 mm`, so finer XY output improves map resolution but does
not create new sub-0.5mm press-center labels by itself.

In `train_e2e.py`, `--local-map-size 0` is the default. It automatically scales
the local decoder's physical footprint with `--grid-step-mm`: `0.5 mm -> 15`,
`0.25 mm -> 29`, `0.2 mm -> 37`, and `0.1 mm -> 71`.

## Build Learning Data

Run from repository root:

```bash
python3 sats/preprocessing/prepare_learning_data.py \
  --source-root skin_ws/raw_data \
  --learning-root learning_data
```

The script discovers raw BIN trial folders, assigns stable `testN` numbers via
`learning_data/trial_registry.json`, writes merged BIN files, and generates
Boussinesq pressure-map GT files under `learning_data/gt`.

To preview what will be processed without writing:

```bash
python3 sats/preprocessing/prepare_learning_data.py --dry-run --stage all
```

To process only one new trial, use the planned output from dry-run and run the
merge/GT tools against that specific trial, or run `prepare_learning_data.py`
after confirming the registry mapping. Existing `testN` numbers are append-only.

For on-the-fly GT training, dense `learning_data/gt/*_targets.npy` files are not
required. You still need the merged BIN and baseline artifacts under
`learning_data/sensor_raw_bin`.

## Raw BIN Sufficiency Check

Before running long training, inspect whether the raw BIN archive has enough
usable data and consistent coverage.

Quick file/record check:

```bash
python3 sats/tools/analyze_raw_bins.py \
  --source-root skin_ws/raw_data \
  --source-material "eco20 + mesh" \
  --diameter d5 \
  --out sats/tools/raw_bin_sufficiency_quick.csv
```

Full distribution check:

```bash
python3 sats/tools/analyze_raw_bins.py \
  --source-root skin_ws/raw_data \
  --source-material "eco20 + mesh" \
  --diameter d5 \
  --full \
  --out sats/tools/raw_bin_sufficiency_full.csv
```

The full report builds the same 200 Hz merged rows in memory and summarizes:

```text
source stream row counts and rates
merged row count and duration
covered XY cells
per-XY sequence length distribution
z_depth range
Fz distribution
active contact row ratio
```

Use this report to confirm that each trial reaches the expected `2.5 mm` d5
depth, has comparable force distribution, covers all `41 x 41` XY points, and
does not have abnormal sequence lengths or missing streams.

## Current Dataset (2026-07 기준)

`learning_data/sensor_raw_bin` 병합 완료 31 trials:

```text
eco20_xy1   : d5 x3 + d10 x3  (6)
eco50_xy1   : d5 x3 + d10 x3  (6)   # d10 test3 loadcell tare 교정됨 (retare_meta_cache)
ecomesh_xy1 : d5 x3 + d10 x3  (6)
ecomesh_xy0p5 : d5 x10 + d10 x3 (13)  # 최종 모델 학습 데이터
```

GT meta cache: `learning_data/gt_meta_cache_xy_d5d10_g05` (31개 + manifest, grid-step 0.5).
trial 번호 추적: `learning_data/trial_registry.json`, controlled 비교용 인덱스: `learning_data/trial_indices/`.

## SATS Training (현행 = train_e2e + 크기입력 A)

현행 학습은 **`train_e2e` 단일 커맨드** (위 on-the-fly GT 예시 참조)에
`--use-indenter-size-input`(A)을 켠 구성이 최종이다. 현행 run:

```text
sats/training/runs/size_input/           # 최종 flat 모델 (ecomesh xy0.5)
sats/training/runs/size_input_material/  # 소재 비교 대표 fold (eco20 f2 / eco50 f1 / ecomesh f3)
```

재현 러너는 `scripts/`(예: `scratchpad_rollout_A.sh`), 진단·figure 재생성은
`sats/tools/eval_diagnostics.py` + `history/fig_data/visualizing_scripts/`.

Default training follows the paper-style SATS data contract:

```text
raw cycle cap:  seq_len = 1000      # keeps loading peak around timestep 820-860
sample input:   sensor_window [B, 10, 16]
sample target:  pressure map  [B, 41, 41] at the window's last timestep
split:          random sequence-level train/val split, val_ratio = 0.2
```

`--use-window-dataset` is enabled by default. Use `--no-use-window-dataset`
only for the older peak-map experiment:

```text
old mode: sensor_seq [B, 1000, 16] -> peak pressure map [B, 41, 41]
```

### Legacy: 4단계 분리 학습 (train_lstm → attention → local_map → cnn)

초기 재현용으로 유지되는 단계별 파이프라인이다. 현행 실험은 모두 `train_e2e`를 사용한다.

```bash
# 예시 (legacy): 단계별 학습 체인
python3 -m sats.training.train_lstm      --run-name lstm_run --epochs 50 --batch-size 2048 --device cuda
python3 -m sats.training.train_attention --lstm-ckpt sats/training/runs/lstm_run/best_model.pt --run-name attn_run ...
python3 -m sats.training.train_local_map --attn-ckpt sats/training/runs/attn_run/best_model.pt --run-name local_run ...
python3 -m sats.training.train_cnn       --local-map-ckpt sats/training/runs/local_run/best_model.pt --run-name cnn_run ...
```

Batch tuning rule for 2 to 10 sets:

```text
default:       batch_size=2048, num_workers=2
RAM pressure:  batch_size=1024, num_workers=1 or 0
stable/idle:   batch_size=4096, num_workers=2
```

Judge tuning by epoch time and RAM stability, not by whether instantaneous GPU
utilization stays fixed at 90%.

## Legacy / Alternate Paths

- `hitmap/` contains the newer CSV/Zarr based XY heatmap and Z/Fz regressor
  experiments.
- `sats/preprocessing/raw_merge.py` and CSV exports are compatibility surfaces.
  For current mk555 SATS data, prefer raw BIN -> merged BIN.

## Directory Map

- `skin_ws/`: raw acquisition archive, node files, acquisition scripts.
- `learning_data/`: managed SATS merged BIN and GT meta cache workspace (대용량, git-ignored).
- `sats/`: SATS preprocessing, GT generation, training(e2e+A), inference, tools(eval_diagnostics), **bending/**(밴딩 보상 모듈).
- `scripts/`: 실험 러너·검증 스크립트 (구 루트 scratchpad_* — `scripts/README.md`).
- `history/fig_data/`: **논문 워크스페이스** — Figure(fig1~4)·supplementary·experiments_archive·투고 체크리스트.
- `hitmap/`: Zarr/heatmap/Z-Fz experimental training pipelines (legacy, 데이터 컨트랙트 다름).
- `cnn_lstm/`: 사이드 프로젝트.
