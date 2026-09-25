#!/usr/bin/env bash
# Experiment A: xy_1mm 소재별 SATS 성능평가 (3 materials x 3 folds = 9 runs)
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1   # 저장소 루트(scripts/ 상위)로 이동, 환경 무관
PY=.venv/bin/python
OUT_DIR="sats/training/runs/xy1_material_d5d10"
CACHE_DIR="learning_data/gt_meta_cache_xy_d5d10_g05"

echo "==== Experiment A 시작: $(date '+%F %T') ===="
n=0
for MAT in eco20_xy1 eco50_xy1 ecomesh_xy1; do
  for FOLD in 1 2 3; do
    n=$((n+1))
    RUN="xy1_d5d10_${MAT}_fold${FOLD}_e2e_g05"
    echo "---- [$n/9] $RUN 시작: $(date '+%F %T') ----"
    $PY -m sats.training.train_e2e \
      --gt-mode gpu_on_the_fly \
      --raw-dir learning_data/sensor_raw_bin \
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
    echo "---- [$n/9] $RUN 종료 (rc=$rc): $(date '+%F %T') ----"
  done
done
echo "==== Experiment A 완료: $(date '+%F %T') ===="
