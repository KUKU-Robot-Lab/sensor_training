# SATS Learning Data

`learning_data` is the managed SATS training-data workspace. Raw acquisition
archives stay under `skin_ws/raw_data`; SATS GT generation and training should
read only the artifacts collected here.

## Layout

```text
learning_data/
├── sensor_raw_bin/                    # 병합 BIN (31 trials, git-ignored)
│   ├── eco20_xy1/{d5,d10}/testN/      # 각 3 rep
│   ├── eco50_xy1/{d5,d10}/testN/      # 각 3 rep (d10 test3 = tare 교정됨, .bak 보존)
│   ├── ecomesh_xy1/{d5,d10}/testN/    # 각 3 rep
│   └── ecomesh_xy0p5/{d5,d10}/testN/  # d5 10 + d10 3 (최종 모델 학습 데이터)
│       └── testN/ *_merged.bin · *_baseline.json · *_merge_summary.json
├── gt_meta_cache_xy_d5d10_g05/        # ★ 현행 GT meta cache (31개 + manifest, grid 0.5)
├── trial_indices/                     # controlled 비교용 curated dataset_index.json
│   └── {eco50_xy1, ecomesh_xy1_common, ecomesh_xy0p5_common, ecomesh_pool_d10, ...}
├── gt/                                # legacy dense GT (*_targets.npy) — 현행은 on-the-fly
├── trial_registry.json                # testN 고정 매핑 (tracked)
└── README.md
```

Large artifacts under `sensor_raw_bin/` and `gt/` are git-ignored.
`trial_registry.json` is small and tracked because it pins source folders to
stable `testN` numbers.

## Build

Build or update from the current mk555 source archive:

```bash
python3 sats/preprocessing/prepare_learning_data.py \
  --source-root skin_ws/raw_data \
  --learning-root learning_data
```

Preview without writing:

```bash
python3 sats/preprocessing/prepare_learning_data.py --dry-run --stage all
```

The script creates trial-level merged BIN files and GT files. Full CSV export is
optional and should be avoided unless debugging.

## Current Trials (2026-07 기준, 31 trials)

```text
eco20_xy1     : d5 test1-3, d10 test1-3
eco50_xy1     : d5 test1-3, d10 test1-3   # d10 test3 loadcell 영점 -2.269N → retare_meta_cache 로 교정
ecomesh_xy1   : d5 test1-3, d10 test1-3
ecomesh_xy0p5 : d5 test1-10, d10 test1-3
```

취득 프로토콜 차이(중요): **xy0.5 = 계단식 느린 하강**(고force), **xy1 = straight press**(저force).
점탄성 때문에 두 프로토콜은 사실상 다른 도메인 — pooling 이득 없음이 실증됨
(`history/fig_data/experiments_archive/pool_diag/pool_result.md`).

## Alignment Principles

- `*_merged.bin` is the official input for both GT and SATS training.
- The merged timeline is 200 Hz, matching the meaningful DUE/loadcell rate.
- DUE raw bursts are expanded into effective 200 Hz `s1..s16` rows.
- EtherMotion can be 1000 Hz or higher and is interpolated onto the 200 Hz
  timeline. It provides high-resolution `x_mm`, `y_mm`, `z_stage_mm`, and
  `u_mm`; it does not define the final row rate.
- `z_depth_mm = max(z_stage_mm - z_start_mm, 0)`, with mk555 d5
  `z_start_mm = 13.0`.
- Z precision in the merged BIN is about `0.0001 mm`, so intermediate depths
  between 0.5 mm hold points are preserved.
- Force is single-axis loadcell:

```text
Fz = (kg - kg_baseline) * 9.80665
```

- `force_round_dp` is `null` for the current merged data, so Fz is not rounded
  to 0.01 N.
- `u_mm` is a node-internal wait/virtual axis and is not a physical shear label.
  Current SATS GT/training includes all rows by default.

## Stable Test Numbering

`trial_registry.json` maps source folders to `testN` numbers. Known source
folders keep their assigned number and only new folders are appended as
`max + 1`. Adding an earlier-dated acquisition later does not renumber existing
trials.

Do not delete `trial_registry.json` unless you intentionally want to rebuild the
entire trial numbering scheme.

## Training Note

Current d5 sequences include Z motion and 0.5 mm hold/plateau segments. Each
XY sequence is about 1600 timesteps, and peak contact is around timestep
820-860. Current SATS training defaults follow the paper-style sliding-window
setup:

```text
seq_len = 1000      # preserve the loading peak inside each raw cycle
window_size = 10    # one training sample = 10 timesteps
batch_size = 2048   # RTX 4090 default for window mode
num_workers = 2     # conservative default for 2-10 sets
```

Use `--no-use-window-dataset` only for the older peak-map experiment. That mode
is heavier because it moves full `[B, seq_len, 41, 41]` GT sequences through the
DataLoader.
