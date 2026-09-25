#!/usr/bin/env bash
# Experiment B: ecomesh xy_1mm vs xy_0.5mm controlled comparison (2 materials x 3 folds = 6 runs)
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1   # 저장소 루트(scripts/ 상위)로 이동, 환경 무관
PY=.venv/bin/python
OUT_DIR="sats/training/runs/ecomesh_resolution_controlled_d5d10"
CACHE_DIR="learning_data/gt_meta_cache_xy_d5d10_g05"

echo "==== Experiment B 시작: $(date '+%F %T') ===="
n=0
for MAT in ecomesh_xy1 ecomesh_xy0p5; do
  if [ "$MAT" = "ecomesh_xy1" ]; then
    GT_DIR="learning_data/trial_indices/ecomesh_xy1_common"
  else
    GT_DIR="learning_data/trial_indices/ecomesh_xy0p5_common"
  fi
  for FOLD in 1 2 3; do
    n=$((n+1))
    RUN="ecomesh_controlled_d5d10_${MAT}_fold${FOLD}_e2e_g05"
    echo "---- [$n/6] $RUN 시작 (gt-dir=$GT_DIR): $(date '+%F %T') ----"
    $PY -m sats.training.train_e2e \
      --gt-mode gpu_on_the_fly \
      --raw-dir learning_data/sensor_raw_bin \
      --gt-dir "$GT_DIR" \
      --gt-meta-cache-dir "$CACHE_DIR" \
      --include-materials "$MAT" \
      --val-ratio 0 \
      --val-trials "${MAT}_d5_z2.5_test${FOLD}" "${MAT}_d10_z3.5_test${FOLD}" \
      --grid-step-mm 0.5 \
      --epochs 50 \
      --seed 42 \
      --out-dir "$OUT_DIR" \
      --run-name "$RUN" 2>&1 | grep -vE "UserWarning|warnings.warn|it/s\]|it/s,"
    rc=${PIPESTATUS[0]}
    echo "---- [$n/6] $RUN 종료 (rc=$rc): $(date '+%F %T') ----"
  done
done
echo "==== Experiment B 완료: $(date '+%F %T') ===="
