#!/bin/bash
# 목표: 실험 재진행(eco50 교정 재학습 + ecomesh pooled) 후 figure set 전체 재생성.
# 백그라운드 실행. 각 단계 로그 마커로 진행 추적.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # 저장소 루트(scripts/ 상위)로 이동, 환경 무관
PY=.venv/bin/python
CACHE=learning_data/gt_meta_cache_xy_d5d10_g05
mark(){ echo ""; echo "########## $(date +%H:%M:%S) $* ##########"; echo ""; }

# ── 1) eco50 fold1 재학습 (교정된 test3 캐시 자동 사용, 원 경로 덮어씀) ──────────
mark "STEP1 eco50 fold1 재학습 (교정 test3)"
$PY -m sats.training.train_e2e \
  --gt-mode gpu_on_the_fly --grid-step-mm 0.5 \
  --raw-dir learning_data/sensor_raw_bin \
  --gt-dir learning_data/trial_indices/eco50_xy1 \
  --gt-meta-cache-dir $CACHE --use-gt-meta-cache --val-ratio 0 \
  --val-trials eco50_xy1_d5_z2.5_test1 eco50_xy1_d10_z3.5_test1 \
  --epochs 50 --batch-size 2048 --lr 0.001 --weight-decay 1e-5 --dropout 0.1 \
  --out-dir sats/training/runs/xy1_material_d5d10 \
  --run-name xy1_d5d10_eco50_xy1_fold1_e2e_g05 || { echo "STEP1 FAIL"; exit 1; }

# ── 2) ecomesh pooled 재학습 (0p5 + xy1 d10) ────────────────────────────────
mark "STEP2 ecomesh pooled 재학습 (0p5 + xy1 d10)"
$PY -m sats.training.train_e2e \
  --gt-mode gpu_on_the_fly --grid-step-mm 0.5 \
  --raw-dir learning_data/sensor_raw_bin \
  --gt-dir learning_data/trial_indices/ecomesh_pool_d10 \
  --gt-meta-cache-dir $CACHE --use-gt-meta-cache --val-ratio 0 \
  --val-trials ecomesh_xy0p5_d5_z2.5_test10 ecomesh_xy0p5_d10_z3.5_test3 \
  --epochs 15 --batch-size 2048 --lr 0.001 --weight-decay 1e-5 --dropout 0.1 \
  --out-dir sats/training/runs/pool_d10 \
  --run-name ecomesh_pool_d10_val_d5t10_d10t3 || { echo "STEP2 FAIL"; exit 1; }

# ── 3) diag 재덤프 (xy1 material 전체: 교정 eco50 포함) + pooled ─────────────
mark "STEP3a xy1 material diag 재덤프 (fig3_diag)"
$PY -m sats.tools.eval_diagnostics --no-fig \
  --run-dirs sats/training/runs/xy1_material_d5d10/xy1_d5d10_*_e2e_g05 \
  --out-dir history/fig_data/experiments_archive/fig3_diag --dump-samples || echo "STEP3a WARN"

mark "STEP3b pooled diag 덤프 (pool_diag)"
$PY -m sats.tools.eval_diagnostics --no-fig \
  --run-dirs sats/training/runs/pool_d10/ecomesh_pool_d10_val_d5t10_d10t3 \
  --out-dir history/fig_data/experiments_archive/pool_diag --dump-samples || echo "STEP3b WARN"

# ── 4) figure set 재생성 ───────────────────────────────────────────────────
mark "STEP4a Fig3 xy1_material (A-F)"
$PY history/fig_data/visualizing_scripts/figure_set/generate_fig3_sats.py \
  --figset xy1_material --panels A B C D E F || echo "STEP4a WARN"
mark "STEP4b Fig3 xy1_material shared-axes"
$PY history/fig_data/visualizing_scripts/figure_set/generate_fig3_sats.py \
  --figset xy1_material --panels A B C D E F --shared-axes || echo "STEP4b WARN"

mark "STEP4c S20 localization (전 소재, 교정 eco50)"
$PY history/fig_data/visualizing_scripts/figure_set/generate_supp_localization.py || echo "STEP4c WARN"

mark "STEP4d 데이터 품질 재분석"
$PY history/fig_data/visualizing_scripts/figure_set/analyze_data_quality.py || echo "STEP4d WARN"

mark "ALL DONE — pooled figset(코드 편집 필요)은 별도 처리 예정"
