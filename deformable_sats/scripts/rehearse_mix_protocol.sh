#!/usr/bin/env bash
# xy1+xy0.5 혼합학습 리허설 (2026-07-19) — z/fz(magnitude) 관점에서 xy0.5 소량 보충의 실익 검증.
# 질문: xy1(위치)만 vs xy1+xy0.5 소량(고force 커버리지)이 fz/magnitude 개선하는가?
#       그 대가로 xy1 위치 성능이 저하되는가? (프로토콜 점탄성 도메인 혼합 리스크)
# 새 센서 없이 기존 ecomesh 데이터로 취득 전 검증. scratch(출발점 편향 없음), 크기입력 A.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # 저장소 루트(scripts/ 상위)로 이동, 환경 무관
PY=.venv/bin/python
OUT_DIR="sats/training/runs/mix_protocol"
CACHE_DIR="learning_data/gt_meta_cache_xy_d5d10_g05"

run_one() {  # $1=run명 $2=index디렉토리 $3..=val-trials
  local RUN=$1 IDX=$2; shift 2
  echo "---- $RUN 시작: $(date '+%F %T') ----"
  $PY -m sats.training.train_e2e \
    --gt-mode gpu_on_the_fly \
    --raw-dir learning_data/sensor_raw_bin \
    --gt-meta-cache-dir "$CACHE_DIR" \
    --gt-dir "learning_data/trial_indices/$IDX" \
    --val-ratio 0 --val-trials "$@" \
    --grid-step-mm 0.5 --epochs 50 --seed 42 \
    --use-indenter-size-input \
    --out-dir "$OUT_DIR" --run-name "$RUN" \
    2>&1 | grep -vE "UserWarning|warnings.warn|it/s\]|it/s,"
  echo "---- $RUN 종료 (rc=${PIPESTATUS[0]}): $(date '+%F %T') ----"
}

echo "==== mix_protocol 시작: $(date '+%F %T') ===="
# A) xy1-only: train = xy1 d5 t1,t2 + d10 t1,t2 (4 trial), holdout = xy1 t3 pair
run_one xy1only mix_xy1only \
  ecomesh_xy1_d5_z2.5_test3 ecomesh_xy1_d10_z3.5_test3
# B) 혼합: train = xy1 4 + xy0.5 d5 t1 + d10 t1 (6 trial), holdout = xy1 t3 pair + xy0.5 d5 t10 + d10 t3
run_one xy1_xy0p5 mix_xy1_xy0p5 \
  ecomesh_xy1_d5_z2.5_test3 ecomesh_xy1_d10_z3.5_test3 \
  ecomesh_xy0p5_d5_z2.5_test10 ecomesh_xy0p5_d10_z3.5_test3
echo "==== mix_protocol 완료: $(date '+%F %T') ===="
