#!/usr/bin/env bash
# d5-only 다해상도 생성: 출력 grid 1.0 / 0.5 / 0.25 mm. β(물성, 물리유도) 켜짐.
# d5 단일이라 크기입력 불필요(애매성 없음). holdout = d5 test10.
# β 계수 = FE-Ogden 압축 유도(c1=0.00244, c2=1.7e-4, clamp≤2.0).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # 저장소 루트(scripts/ 상위)로 이동, 환경 무관
PY=.venv/bin/python
OUT_DIR="sats/training/runs/d5_only_multires"
CACHE_DIR="learning_data/gt_meta_cache_xy_d5d10_g05"   # 0.5는 캐시, 1.0/0.25는 on-the-fly fallback

echo "==== d5 다해상도 생성 시작: $(date '+%F %T') ===="
n=0
for RES in 1.0 0.5 0.25; do
  n=$((n+1))
  TAG=$(echo "$RES" | tr '.' 'p')
  RUN="d5only_beta_g${TAG}"
  echo "---- [$n/3] $RUN (grid ${RES}mm) 시작: $(date '+%F %T') ----"
  $PY -m sats.training.train_e2e \
    --gt-mode gpu_on_the_fly --grid-step-mm "$RES" \
    --raw-dir learning_data/sensor_raw_bin \
    --gt-meta-cache-dir "$CACHE_DIR" --use-gt-meta-cache \
    --include-materials ecomesh_xy0p5 --exclude-diameters 10 \
    --val-ratio 0 --val-trials ecomesh_xy0p5_d5_z2.5_test10 \
    --epochs 15 --batch-size 2048 --lr 0.001 --weight-decay 1e-5 --dropout 0.1 \
    --gt-beta-mode poly2 --gt-beta-c0 1.0 --gt-beta-c1 0.00244 --gt-beta-c2 0.00017 --gt-beta-max 2.0 \
    --out-dir "$OUT_DIR" --run-name "$RUN" 2>&1 | grep -vE "UserWarning|warnings.warn|it/s\]|it/s,"
  rc=${PIPESTATUS[0]}
  echo "---- [$n/3] $RUN 종료 (rc=$rc): $(date '+%F %T') ----"
done
echo "==== d5 다해상도 생성 완료: $(date '+%F %T') ===="
