#!/usr/bin/env bash
set -euo pipefail

# 9-combo grid sweep for:
# - preprocess: z binning + min-signal filtering
# - train/eval: SR zarr pipeline
#
# Usage:
#   bash scripts/sweep_sr_grid.sh
#   EPOCHS=100 bash scripts/sweep_sr_grid.sh
#   SKIP_RAW_MERGE=1 bash scripts/sweep_sr_grid.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RAW_ROOT="${RAW_ROOT:-preprocessing/raw_data}"
SWEEP_PREP_BASE="${SWEEP_PREP_BASE:-preprocessing/processed_sweeps}"
SWEEP_RUN_BASE="${SWEEP_RUN_BASE:-training/runs_sr_sweep}"
EPOCHS="${EPOCHS:-50}"                  # quick screening default
BATCH_SIZE="${BATCH_SIZE:-16384}"
LR="${LR:-1e-3}"
SKIP_RAW_MERGE="${SKIP_RAW_MERGE:-0}"

Z_BINS=(0.01 0.02 0.03)
MIN_SIGNALS=(0.015 0.02 0.025)

mkdir -p "$SWEEP_PREP_BASE" "$SWEEP_RUN_BASE"
SUMMARY_CSV="$SWEEP_RUN_BASE/summary.csv"
echo "run_id,z_bin_mm,min_signal,epochs,best_val_mae_mm,eval_mae_x_mm,eval_mae_y_mm,eval_mae_z_mm,eval_xy_euclidean_mm" > "$SUMMARY_CSV"

echo "[sweep] root: $ROOT_DIR"
echo "[sweep] preprocess out base: $SWEEP_PREP_BASE"
echo "[sweep] train out base: $SWEEP_RUN_BASE"
echo "[sweep] epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR"

if [[ "$SKIP_RAW_MERGE" != "1" ]]; then
  echo "[sweep] step 1/3: raw_merge"
  python3 preprocessing/raw_merge.py --raw-root "$RAW_ROOT"
else
  echo "[sweep] step 1/3: raw_merge skipped (SKIP_RAW_MERGE=1)"
fi

run_idx=0
for zbin in "${Z_BINS[@]}"; do
  for minsig in "${MIN_SIGNALS[@]}"; do
    run_idx=$((run_idx + 1))
    ztag="${zbin/./p}"
    stag="${minsig/./p}"
    run_id="z${ztag}_s${stag}"

    prep_out="$SWEEP_PREP_BASE/$run_id"
    run_out="$SWEEP_RUN_BASE/$run_id"
    zarr_path="$prep_out/zarr_data/dataset_ecomesh.zarr"
    train_log="$run_out/train.log"
    eval_log="$run_out/eval.log"

    mkdir -p "$prep_out" "$run_out"

    echo ""
    echo "[sweep][$run_idx/9] run_id=$run_id z_bin_mm=$zbin min_signal=$minsig"
    echo "[sweep] preprocess -> $prep_out"
    python3 preprocessing/preprocess.py \
      --raw-dir "$RAW_ROOT" \
      --out-dir "$prep_out" \
      --z-bin-mm "$zbin" \
      --min-signal "$minsig"

    echo "[sweep] train -> $run_out"
    python3 -m training.train_sr_zarr \
      --zarr-path "$zarr_path" \
      --out-dir "$run_out" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" | tee "$train_log"

    echo "[sweep] eval -> $run_out"
    python3 -m training.evaluate_zarr_sr \
      --zarr-path "$zarr_path" \
      --model-path "$run_out/best_model.pt" | tee "$eval_log"

    read -r best_mae mae_x mae_y mae_z xy_err < <(
      python3 - <<PY
import json, re
from pathlib import Path

run_out = Path("$run_out")
history = json.loads((run_out / "history.json").read_text())
best = min(history["val_mae_mm"]) if "val_mae_mm" in history and history["val_mae_mm"] else float("nan")

txt = (run_out / "eval.log").read_text()
m1 = re.search(r"MAE \\(mm\\)\\s*\\|\\s*X:\\s*([0-9.]+),\\s*Y:\\s*([0-9.]+),\\s*Z:\\s*([0-9.]+)", txt)
m2 = re.search(r"XY Euclidean Error:\\s*([0-9.]+)\\s*mm", txt)
if not m1 or not m2:
    raise SystemExit("failed_to_parse_eval_log")
print(best, m1.group(1), m1.group(2), m1.group(3), m2.group(1))
PY
    )

    echo "$run_id,$zbin,$minsig,$EPOCHS,$best_mae,$mae_x,$mae_y,$mae_z,$xy_err" >> "$SUMMARY_CSV"
    echo "[sweep] done $run_id | best_val_mae=$best_mae | eval_xy=$xy_err"
  done
done

echo ""
echo "[sweep] completed. summary: $SUMMARY_CSV"
echo "[sweep] top 5 by best_val_mae_mm:"
python3 - <<'PY'
import csv
from pathlib import Path
p = Path("training/runs_sr_sweep/summary.csv")
rows = list(csv.DictReader(p.open()))
rows.sort(key=lambda r: float(r["best_val_mae_mm"]))
for r in rows[:5]:
    print(
        f"{r['run_id']}: best={float(r['best_val_mae_mm']):.6f}, "
        f"X={float(r['eval_mae_x_mm']):.3f}, Y={float(r['eval_mae_y_mm']):.3f}, "
        f"Z={float(r['eval_mae_z_mm']):.3f}, XY={float(r['eval_xy_euclidean_mm']):.3f}"
    )
PY
