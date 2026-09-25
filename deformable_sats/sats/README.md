# SATS

SATS maps the 16 physical pressure sensors of a sparse tactile array to a dense
`41 x 41` virtual pressure map. The implementation follows the staged SATS
idea from *Super-resolution tactile sensor arrays with sparse units enabled by
deep learning*, adapted to the current mk555 raw BIN data.

> **현행 최종 구성 (2026-07-12)**: `train_e2e` + **인덴터 크기 입력(A, FiLM)** =
> `--use-indenter-size-input`. β GT 보정(`gt_beta_*` config)은 인프라만 보존(기본 off, d5-only에서만 사용).
> 출력 grid는 `--grid-step-mm`으로 자유(0.5 기본, 0.1까지 검증·AMP `--use-amp` 지원).
> 진단 = `sats/tools/eval_diagnostics.py`(d5/d10 분리·상대오차·`--dump-samples`).
> 밴딩 보상 프론트엔드 = `sats/bending/`(별도 README). 학습 환경은 반드시 `.venv`(RTX 5090).

## What This Pipeline Learns

Current `sats.training` learns a **dense pressure-map GT**:

```text
input  = normalized DUE s1..s16 sequence
target = Boussinesq 41 x 41 pressure map
```

This is different from scalar Z/Fz regression. Z and Fz are preserved in merged
BIN files, but the current SATS target npy files are pressure maps generated
from `(x_mm, y_mm, Fz, diameter)`.

## Sensor and Scan Geometry

Physical sensor layout:

```text
4 x 4 BMP384 sensors
sensor spacing = 6.5 mm
sensor coordinates = +/-9.75 mm and +/-3.25 mm
virtual grid = 41 x 41
virtual grid coordinate = [-10.0, -9.5, ..., 10.0] mm
```

The stage scans `x, y` from `-10.0` to `10.0` mm in `0.5 mm` steps. At each
grid point, mk555 d5 data runs `z_stage_mm` from `13.0` to `15.5`, so:

```text
d5 z_depth_mm = z_stage_mm - 13.0  # max 2.5 mm
d10 z_depth_mm = z_stage_mm - 12.0 # max 3.5 mm
```

The node holds at 0.5 mm depth points. Those hold/plateau rows are kept by
default, so each XY sequence includes both Z motion and stationary relaxation.

## Official Data Flow

```text
skin_ws/raw_data/sats/ecomesh/xy_0.5mm/d5/testN/
  ├── due_raw_burst_*.bin
  ├── ethermotion_encoder_*.bin
  └── loadcell_raw_*.bin

learning_data/sensor_raw_bin/ecomesh_xy0p5/d5/z_2.5mm/testN/
  ├── ecomesh_xy0p5_d5_z2.5_testN_merged.bin
  ├── ecomesh_xy0p5_d5_z2.5_testN_baseline.json
  └── ecomesh_xy0p5_d5_z2.5_testN_merge_summary.json

learning_data/gt_meta_cache/
  ├── ecomesh_xy0p5_d5_z2.5_testN_*_meta_cache.pt
  └── manifest.json
```

The current raw archive also includes `eco20`, `eco50`, and `ecomesh`
materials under `xy_0.5mm` or `xy_1mm`. `prepare_learning_data.py` encodes the
XY resolution into the output material key, for example `eco20_xy1` and
`ecomesh_xy0p5`.

Build/update merged BIN artifacts:

```bash
python3 sats/preprocessing/prepare_learning_data.py \
  --source-root skin_ws/raw_data \
  --source-material all \
  --learning-root learning_data \
  --stage merge
```

Preview:

```bash
python3 sats/preprocessing/prepare_learning_data.py \
  --source-root skin_ws/raw_data \
  --source-material all \
  --learning-root learning_data \
  --dry-run \
  --stage merge
```

Expected current dry-run: `planned trials: 31`.

## On-The-Fly GT Path

The legacy/default GT path stores dense `*_targets.npy` files. For d5 this is
about 18 GB per trial because every 200 Hz row stores one `41 x 41 float32`
pressure map.

The optional on-the-fly path keeps the original merged BIN data and generates
the target pressure map during training:

```text
sensor_window [10, 16]
merged row x/y/z_depth/Fz
  -> spherical-indenter Boussinesq GT [41, 41]
```

There are two on-the-fly modes:

```text
--gt-mode on_the_fly     # CPU/DataLoader worker generates dense maps
--gt-mode gpu_on_the_fly # DataLoader returns metadata; train step builds batch GT on GPU
```

Use `gpu_on_the_fly` for denser grids such as `81 x 81`, `101 x 101`, or
`201 x 201`. It avoids prefetching large dense target maps through worker
processes and avoids saving huge `*_targets.npy` files.

Current on-the-fly assumptions:

```text
z_s_mm                  = 2.0 fixed
beta/FEM correction     = none
indenter model          = spherical
contact radius          = sqrt(R * z_depth_mm), R = diameter / 2
d5 contact threshold    = z_depth_mm > 0.001
minimum contact radius  = 0.05 mm
z-depth balance bins    = 0.005 mm
pressure map grid       = 41 x 41, 0.5 mm XY spacing
```

The `0.005 mm` setting is for z-depth sample balancing, not XY map spacing. To
increase XY virtual taxel density, change the GT/model grid, for example from
`41 x 41` at `0.5 mm` spacing to `81 x 81` at `0.25 mm` spacing.

Example:

Build compact metadata cache once from the merged BIN data. The first training
pass should use d5 only; d10 remains mapped and can be included later by
removing `--exclude-diameters 10`.

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
  --run-name e2e_d5_mapped_all
```

High-resolution examples:

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

# 0.1 mm virtual grid, 201 x 201 output
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

The existing `precomputed` path remains the default:

```text
--gt-mode precomputed
```

In `train_e2e.py`, `--local-map-size 0` is the default. It keeps the local
decoder's physical footprint close to the original `0.5 mm / 15 cell` setting:
`0.25 mm -> 29`, `0.2 mm -> 37`, and `0.1 mm -> 71`.

## Alignment Policy

The final merged timeline is 200 Hz:

- DUE is expanded from raw bursts into an effective 200 Hz tactile stream.
- Loadcell is interpolated onto the 200 Hz timeline and converted to `Fz`.
- EtherMotion is high-rate, around 1000 Hz or higher, and is interpolated onto
  the same 200 Hz timeline for high-precision `x/y/z/u` labels.

Do not upsample training rows to EtherMotion rate. That creates many rows where
DUE/loadcell data are just interpolated duplicates. The correct row basis is
the meaningful 200 Hz tactile/loadcell timeline with high-resolution EtherMotion
labels attached.

Current precision/behavior:

```text
z_stage_mm / z_depth_mm precision: about 0.0001 mm
Fz rounding: disabled in current merged data
u_mm: virtual wait axis, not physical shear
```

## Current Dataset Mapping Status

As of the current raw archive:

```text
planned trials: 31
materials: eco20_xy1, eco50_xy1, ecomesh_xy0p5, ecomesh_xy1
diameters: d5, d10
```

The first recommended training pool is d5-only:

```text
--exclude-diameters 10
```

## Raw BIN Sufficiency Check

Use `sats/tools/analyze_raw_bins.py` before long training runs to inspect raw
trial coverage.

Quick file/record check:

```bash
python3 sats/tools/analyze_raw_bins.py \
  --source-root skin_ws/raw_data \
  --source-material sats/ecomesh/xy_0.5mm \
  --diameter d5 \
  --out sats/tools/raw_bin_sufficiency_quick.csv
```

Full distribution check:

```bash
python3 sats/tools/analyze_raw_bins.py \
  --source-root skin_ws/raw_data \
  --source-material sats/ecomesh/xy_0.5mm \
  --diameter d5 \
  --full \
  --out sats/tools/raw_bin_sufficiency_full.csv
```

The full report summarizes source rates, merged rows, XY coverage, per-XY
sequence lengths, z-depth range, Fz distribution, and active contact ratio.

## Training Data Semantics

`sats/training/dataset.py` groups rows by `(trial_id, x_mm, y_mm)`.

The default training sample follows the SATS paper-style sliding window:

```text
raw XY cycle:    up to seq_len=1000 rows
window sample:   sensor_window [10, 16]
target:          GT map [41, 41] at the window's last timestep
phase:           loading phase only, up to the GT-sum peak
```

This is implemented by `--use-window-dataset`, which is enabled by default in
`train_lstm.py`, `train_attention.py`, `train_local_map.py`, `train_cnn.py`, and
`train_e2e.py`.

The older peak-map experiment is still available with
`--no-use-window-dataset`:

```text
sensor_seq [B, 1000, 16] -> peak GT map [B, 41, 41]
```

Because the current mk555 node includes 0.5 mm hold/plateau segments:

```text
sequence length median ~= 1631
peak contact index     ~= 820-860
default seq_len         = 1000
window_size             = 10
```

## Smoke Test

```bash
python3 -m sats.training.train_lstm \
  --run-name smoke_lstm_window10 \
  --epochs 1 \
  --batch-size 256 \
  --num-workers 0 \
  --device cuda
```

Verified smoke checks:

```text
sensor batch = [B, 10, 16]
target       = [B, 41, 41]
forward/backward OK
```

## Recommended Training on RTX 4090

Default window-mode profile:

```text
GPU: RTX 4090 24GB
batch_size=2048
window_size=10
num_workers=2
```

Window mode is much lighter than the previous full-sequence GT mode because it
does not move `[B,1000,41,41]` GT tensors through the DataLoader. With the data
expected to grow to 10 sets, keep workers conservative and scale batch size
first.

Recommended LSTM command:

```bash
python3 -m sats.training.train_lstm \
  --run-name lstm_d5_test12_window10_bs2048 \
  --epochs 50 \
  --batch-size 2048 \
  --num-workers 2 \
  --device cuda
```

Tuning rule:

```text
default:       batch_size=2048, num_workers=2
RAM pressure:  batch_size=1024, num_workers=1 or 0
stable/idle:   batch_size=4096, num_workers=2
```

## Staged SATS Training (legacy — 현행은 train_e2e)

```bash
python3 -m sats.training.train_attention \
  --lstm-ckpt sats/training/runs/lstm_d5_test12_window10_bs2048/best_model.pt \
  --run-name attn_d5_test12_window10_bs2048 \
  --epochs 50 \
  --batch-size 2048 \
  --num-workers 2 \
  --device cuda

python3 -m sats.training.train_local_map \
  --attn-ckpt sats/training/runs/attn_d5_test12_window10_bs2048/best_model.pt \
  --run-name local_map_d5_test12_window10_bs2048 \
  --epochs 50 \
  --batch-size 2048 \
  --num-workers 2 \
  --device cuda

python3 -m sats.training.train_cnn \
  --local-map-ckpt sats/training/runs/local_map_d5_test12_window10_bs2048/best_model.pt \
  --run-name cnn_d5_test12_window10_bs2048 \
  --epochs 50 \
  --batch-size 2048 \
  --num-workers 2 \
  --device cuda
```

## Directory Map

```text
sats/
├── preprocessing/
│   ├── prepare_learning_data.py
│   ├── bin_merge.py
│   ├── merged_bin.py
│   └── generate_gt.py            # β(p) 물성보정 포함(compute_beta, 기본 off)
├── training/
│   ├── config.py                 # ablate_*, use_indenter_size_input, use_amp, gt_beta_* 노브
│   ├── cnn_module.py             # SATS 모듈(FiLM 크기 컨디셔닝 포함)
│   ├── dataset.py / dataset_on_the_fly.py / gt_gpu.py
│   ├── train_e2e.py              # ★ 현행 학습 엔트리 (end-to-end)
│   ├── train_lstm/attention/local_map/cnn.py   # legacy 4단계
│   ├── build_gt_meta_cache.py
│   └── tests/                    # TDD (contract·GT·size-input·β 등)
├── tools/
│   ├── eval_diagnostics.py       # ★ d5/d10 분리 진단 + --dump-samples
│   ├── compare_sats_runs.py / analyze_taxel_rmse.py
│   ├── retare_meta_cache.py      # loadcell 영점 교정 (eco50 test3 사례)
│   └── analyze_raw_bins.py
├── bending/                      # ★ 밴딩 보상 프론트엔드 (Phase 0 — 자체 README)
├── inference/                    # realtime 추론
└── ref/
```

## Notes

- `raw_merge.py` and CSV paths are legacy/compatibility surfaces.
- `preprocess.py` is useful for CSV/Zarr experiments but is not required for
  current SATS pressure-map training.
- For scalar Z/Fz learning, use the separate Z/Fz regression path rather than
  the SATS pressure-map target npy files.
