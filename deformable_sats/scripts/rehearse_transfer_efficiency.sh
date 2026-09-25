#!/usr/bin/env bash
# 새 센서 대비 데이터 효율/전이 리허설 (2026-07-17, 노션 '새 센서 일반화 전략' 항목)
# 질문: 소량(xy1 1~2 pair)으로 충분한가? warm-start 가 이득인가? 교차-출발(유닛 편차 프록시)은?
# 홀드아웃 고정 = ecomesh_xy1 fold3 (d5 test3 + d10 test3) → 기존 sizeA_ecomesh_xy1_fold3(=scratch_2pair)와 직접 비교.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # 저장소 루트(scripts/ 상위)로 이동, 환경 무관
PY=.venv/bin/python
OUT_DIR="sats/training/runs/transfer_efficiency"
CACHE_DIR="learning_data/gt_meta_cache_xy_d5d10_g05"
WARM_XY0P5="sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3/best_model.pt"
WARM_ECO20="sats/training/runs/size_input_material/sizeA_eco20_xy1_fold2_e2e_g05/best_model.pt"
VAL_ARGS=(--val-ratio 0 --val-trials ecomesh_xy1_d5_z2.5_test3 ecomesh_xy1_d10_z3.5_test3)

run_one() {  # $1=run명 $2=gt-dir $3=init-ckpt("" 이면 scratch)
  local RUN=$1 GTD=$2 INIT=$3
  local EXTRA=()
  [ -n "$INIT" ] && EXTRA=(--init-ckpt "$INIT")
  echo "---- $RUN 시작: $(date '+%F %T') ----"
  $PY -m sats.training.train_e2e \
    --gt-mode gpu_on_the_fly \
    --raw-dir learning_data/sensor_raw_bin \
    --gt-meta-cache-dir "$CACHE_DIR" \
    --gt-dir "$GTD" \
    "${VAL_ARGS[@]}" \
    --grid-step-mm 0.5 --epochs 50 --seed 42 \
    --use-indenter-size-input \
    --out-dir "$OUT_DIR" --run-name "$RUN" "${EXTRA[@]}" \
    2>&1 | grep -vE "UserWarning|warnings.warn|it/s\]|it/s,"
  echo "---- $RUN 종료 (rc=${PIPESTATUS[0]}): $(date '+%F %T') ----"
}

IDX1="learning_data/trial_indices/ecomesh_xy1_1pair"     # train 1 pair(d5t1+d10t1) + 홀드아웃
IDX2="learning_data/trial_indices/ecomesh_xy1_common"    # train 2 pair(t1,2) + 홀드아웃 (6 trial)

echo "==== transfer_efficiency 시작: $(date '+%F %T') ===="
run_one scratch_1pair   "$IDX1" ""
run_one warm_1pair      "$IDX1" "$WARM_XY0P5"
run_one warm_2pair      "$IDX2" "$WARM_XY0P5"
run_one crosswarm_2pair "$IDX2" "$WARM_ECO20"

# 추가: xy1 취득 + fine 출력(0.25mm, 81^2) 검증 — "coarse 스캔 + 연속 GT → 임의 해상도" 증거
echo "---- xy1_g025 시작: $(date '+%F %T') ----"
$PY -m sats.training.train_e2e \
  --gt-mode gpu_on_the_fly \
  --raw-dir learning_data/sensor_raw_bin \
  --gt-meta-cache-dir "$CACHE_DIR" \
  --gt-dir "$IDX2" \
  "${VAL_ARGS[@]}" \
  --grid-step-mm 0.25 --epochs 50 --seed 42 \
  --use-indenter-size-input \
  --batch-size 1024 \
  --out-dir "$OUT_DIR" --run-name xy1_2pair_g025 \
  2>&1 | grep -vE "UserWarning|warnings.warn|it/s\]|it/s,"
echo "---- xy1_g025 종료 (rc=${PIPESTATUS[0]}): $(date '+%F %T') ----"
echo "==== transfer_efficiency 완료: $(date '+%F %T') ===="
# scratch_2pair 참조 = 기존 sizeA_ecomesh_xy1_fold3_e2e_g05 (재학습 불필요)
