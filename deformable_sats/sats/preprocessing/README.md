# SATS Preprocessing

This directory builds the official SATS training artifacts from mk555 raw BIN
logs. The current production path is:

```text
skin_ws/raw_data/**/*raw*.bin
  -> learning_data/sensor_raw_bin/**/*_merged.bin
  -> learning_data/gt_meta_cache_xy_d5d10_g05/   # 현행: compact meta cache + GPU on-the-fly GT
```

(legacy: dense `learning_data/gt/*_targets.npy` — precomputed 모드에서만 사용.
`generate_gt.py`에는 β(p) 물성보정 `compute_beta`가 있으나 기본 off.)

CSV export and CSV-based preprocessing are retained for inspection and legacy
experiments, but current SATS pressure-map training should use merged BIN files.

## Raw Inputs

Each raw trial directory should contain:

```text
due_raw_burst_*.bin          # DUE: 16 sensors x 10 FIFO frames per burst
ethermotion_encoder_*.bin    # stage command/log x, y, z, u
loadcell_raw_*.bin           # single-axis loadcell kg stream
```

Current source root:

```text
skin_ws/raw_data/sats/eco20/xy_1mm/d5/20260619_test5
skin_ws/raw_data/sats/eco50/xy_1mm/d5/20260620_test1
skin_ws/raw_data/sats/ecomesh/xy_0.5mm/d5/test1
skin_ws/raw_data/sats/ecomesh/xy_1mm/d5/20260622_test1
```

`prepare_learning_data.py` maps the source `xy_*` resolution into the output
material key:

```text
eco20/xy_1mm      -> eco20_xy1
eco50/xy_1mm      -> eco50_xy1
ecomesh/xy_0.5mm  -> ecomesh_xy0p5
ecomesh/xy_1mm    -> ecomesh_xy1
```

## Official Build Command

```bash
python3 sats/preprocessing/prepare_learning_data.py \
  --source-root skin_ws/raw_data \
  --source-material all \
  --learning-root learning_data \
  --stage merge
```

Dry-run:

```bash
python3 sats/preprocessing/prepare_learning_data.py \
  --source-root skin_ws/raw_data \
  --source-material all \
  --learning-root learning_data \
  --dry-run \
  --stage merge
```

The current archive should plan 31 trials and skip none.

The script:

1. discovers usable raw BIN trial folders,
2. assigns stable `testN` numbers using `learning_data/trial_registry.json`,
3. merges DUE/EtherMotion/loadcell into `*_merged.bin`,
4. writes baseline and merge summary JSON,
5. optionally generates Boussinesq `41 x 41` pressure-map GT when `--stage gt`
   or `--stage all` is requested.

For full training, prefer compact metadata caches and `--gt-mode gpu_on_the_fly`
instead of precomputing `*_targets.npy` for every raw row.

## Time Alignment

The final merged row rate is 200 Hz.

- DUE: effective 200 Hz after expanding 10 FIFO frames per raw burst.
- Loadcell: interpolated onto the common 200 Hz timeline.
- EtherMotion: logged much faster, commonly 1000 Hz or more, and interpolated
  onto the same 200 Hz timeline.

This avoids creating artificial samples at EtherMotion rate while preserving
high-resolution Z labels on every DUE/loadcell training row.

Merged row semantics:

```text
timestep_sec
s1..s16
x_mm, y_mm
z_stage_mm
z_depth_mm
u_mm
Fz
timestamp_due, timestamp_loadcell, timestamp_ethermotion
lag_*_abs_sec
```

`z_stage_mm` is EtherMotion command Z. `z_depth_mm` is the indentation depth
relative to the node start Z:

```text
d5  z_depth_mm = max(z_stage_mm - 13.0, 0)
d10 z_depth_mm = max(z_stage_mm - 12.0, 0)
```

`u_mm` is a node-internal wait/virtual axis. It is not used as physical shear or
depth. Current SATS training and GT include all rows by default.

## `bin_merge.py`

Direct raw BIN merger. `prepare_learning_data.py` calls this internally.

Useful direct command:

```bash
python3 sats/preprocessing/bin_merge.py \
  --raw-root "skin_ws/raw_data/sats/ecomesh/xy_0.5mm/d5/test2" \
  --target-hz 200 \
  --window-ms 10 \
  --window-agg median \
  --max-dt-ms 10
```

Key defaults:

| Option | Default | Meaning |
| --- | --- | --- |
| `--target-hz` | `200.0` | common merged timeline |
| `--window-ms` | `10.0` | smoothing window for DUE/loadcell |
| `--window-agg` | `median` | smoothing aggregation |
| `--max-dt-ms` | `10.0` | drop row if nearest source sample is too far |
| `--force-round-dp` | `-1` | disabled; keep Fz precision |
| `--no-stable-xy-filter` | off | keep only stable/on-grid XY rows by default |

For the current d5 test2 merge:

```text
merged_rows            = 2,743,017
target_hz              = 200.0
ethermotion_source_rows = 30,615,725
loadcell_source_rows    = 3,330,984
force_round_dp          = null
```

## `generate_gt.py`

Generates one `41 x 41` Boussinesq pressure map per merged row.

```bash
python3 -m sats.preprocessing.generate_gt \
  --raw-dir learning_data/sensor_raw_bin \
  --out-dir learning_data/gt \
  --input-format bin \
  --include-shear-u \
  --z-s 2.0 \
  --patch-step 0.1 \
  --fz-mode abs \
  --fz-min-abs 0.05
```

GT row policy must match SATS training dataset policy:

```text
drop off-grid rows: yes
include u != 0 rows: yes
grid: [-10.0, -9.5, ..., 10.0] mm
grid tolerance: 0.05 mm
```

Current GT is pressure-map GT:

```text
GT = sigma_zz(u, v, z_s | x, y, Fz, diameter)
```

It uses `(x_mm, y_mm, Fz, diameter)` for each row. The default
`z_comp_mode=none` means `z_stage_mm` is not used to vary `z_s`; Z remains
available in merged BIN for depth analysis or separate Z/Fz regression.

## Boussinesq Model Summary

For contact radius `a = diameter / 2` and total force `Fz`:

```text
p0 = Fz / (pi * a^2)
Fi = p0 * patch_step^2
R_i = sqrt((u - x_i)^2 + (v - y_i)^2 + z_s^2)
sigma_zz(u, v) = sum_i 3 * Fi * z_s^3 / (2*pi*R_i^5)
```

Because the map is linear in Fz, `generate_gt.py` caches unit kernels and scales
them by row force. Current d5 GT files are about 18 GB per trial.

## `preprocess.py` Legacy/Alternate Path

`preprocess.py` creates grid/features/Zarr artifacts and is useful for
CSV/Zarr-based experiments. It is not required for current `sats.training`
pressure-map training, which reads merged BIN and GT npy directly.

Use it only when a downstream pipeline explicitly expects:

```text
processed_data/grid/*.csv
processed_data/features/*.csv
processed_data/zarr_data/*.zarr
```

For `0.xx mm` Z/Fz experiments, avoid coarse Z binning:

```text
--z-bin-mm 0.001
```

or disable binning if the downstream code can consume continuous Z.

## Verification Checklist

Before training:

```text
GT rows == SATS on-grid rows
GT peak xy matches row x/y
seq_len reaches peak contact
```

Verified current d5 data:

```text
test1 on-grid rows = 2,743,978 = GT rows
test2 on-grid rows = 2,743,016 = GT rows
peak contact index ~= 820-860
recommended seq_len >= 1000
```
