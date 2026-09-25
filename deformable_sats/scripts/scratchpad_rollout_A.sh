#!/usr/bin/env bash
# 최종 롤아웃: 소재 대표 fold를 크기입력(A)로 재학습. β 없음.
# 대표 fold = fig3/summary 가 쓰는 것: eco20 fold2, eco50 fold1, ecomesh fold3.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # 저장소 루트(scripts/ 상위)로 이동, 환경 무관
PY=.venv/bin/python
OUT_DIR="sats/training/runs/size_input_material"
CACHE_DIR="learning_data/gt_meta_cache_xy_d5d10_g05"

echo "==== A 롤아웃 시작: $(date '+%F %T') ===="
# (material, fold)
declare -a JOBS=("eco20_xy1 2" "eco50_xy1 1" "ecomesh_xy1 3")
n=0
for JOB in "${JOBS[@]}"; do
  set -- $JOB; MAT=$1; FOLD=$2
  n=$((n+1))
  RUN="sizeA_${MAT}_fold${FOLD}_e2e_g05"
  echo "---- [$n/3] $RUN 시작: $(date '+%F %T') ----"
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
    --use-indenter-size-input \
    --out-dir "$OUT_DIR" \
    --run-name "$RUN" 2>&1 | grep -vE "UserWarning|warnings.warn|it/s\]|it/s,"
  rc=${PIPESTATUS[0]}
  echo "---- [$n/3] $RUN 종료 (rc=$rc): $(date '+%F %T') ----"
done
echo "==== A 롤아웃 완료: $(date '+%F %T') ===="
