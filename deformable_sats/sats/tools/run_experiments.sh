#!/usr/bin/env bash
# SATS 실험 실행 스크립트
#
# 실행 전 위치 확인:
#   cd /home/user/sensor_training
#
# 단계별 실행:
#   bash sats/tools/run_experiments.sh [단계]
#   예) bash sats/tools/run_experiments.sh analyze_gt
#       bash sats/tools/run_experiments.sh local_map_v2
#       bash sats/tools/run_experiments.sh cnn_v2
#       bash sats/tools/run_experiments.sh visualize_all

set -e
STEP="${1:-help}"

case "$STEP" in

# ─── GT 진단 ─────────────────────────────────────────────────────────────────
analyze_gt)
    echo "=== GT 값 범위 진단 ==="
    python3 -m sats.tools.analyze_gt
    ;;

# ─── Local Map v2 (센서 매핑 버그 수정 후) ────────────────────────────────────
local_map_v2)
    echo "=== Local Map v2 학습 (버그 수정 후) ==="
    python3 -m sats.training.train_local_map \
        --attn-ckpt  sats/training/runs/attn_v1/best_model.pt \
        --run-name   local_map_v2 \
        --epochs     50
    ;;

# ─── CNN v2 (local_map_v2 기반) ───────────────────────────────────────────────
cnn_v2)
    echo "=== CNN v2 학습 (local_map_v2 기반) ==="
    python3 -m sats.training.train_cnn \
        --local-map-ckpt sats/training/runs/local_map_v2/best_model.pt \
        --run-name       cnn_v2 \
        --epochs         50
    ;;

# ─── 전체 재학습 (local_map_v2 → cnn_v2 순서) ────────────────────────────────
retrain_all)
    echo "=== Local Map v2 + CNN v2 순서 학습 ==="
    bash "$0" local_map_v2
    bash "$0" cnn_v2
    ;;

# ─── 시각화 (4단계 비교) ──────────────────────────────────────────────────────
visualize_all)
    echo "=== 4단계 비교 시각화 ==="
    for STAGE in lstm attn local_map cnn; do
        case "$STAGE" in
            lstm)      CKPT="sats/training/runs/lstm_v1/best_model.pt" ;;
            attn)      CKPT="sats/training/runs/attn_v1/best_model.pt" ;;
            local_map) CKPT="sats/training/runs/local_map_v2/best_model.pt" ;;
            cnn)       CKPT="sats/training/runs/cnn_v2/best_model.pt" ;;
        esac
        if [ -f "$CKPT" ]; then
            echo "--- $STAGE ---"
            python3 -m sats.tools.visualize --stage "$STAGE" --ckpt "$CKPT" --n-samples 6
        else
            echo "체크포인트 없음: $CKPT"
        fi
    done
    ;;

# ─── 기존 v1 시각화 (버그 있는 버전 확인용) ──────────────────────────────────
visualize_v1)
    echo "=== 기존 v1 시각화 (버그 버전 비교용) ==="
    for STAGE in lstm attn local_map cnn; do
        case "$STAGE" in
            lstm)      CKPT="sats/training/runs/lstm_v1/best_model.pt"; TAG="v1" ;;
            attn)      CKPT="sats/training/runs/attn_v1/best_model.pt"; TAG="v1" ;;
            local_map) CKPT="sats/training/runs/local_map_v1/best_model.pt"; TAG="v1" ;;
            cnn)       CKPT="sats/training/runs/cnn_v1/best_model.pt"; TAG="v1" ;;
        esac
        if [ -f "$CKPT" ]; then
            echo "--- $STAGE (v1) ---"
            python3 -m sats.tools.visualize \
                --stage "$STAGE" --ckpt "$CKPT" \
                --n-samples 6 \
                --out-dir "sats/tools/viz_output_v1"
        fi
    done
    ;;

help|*)
    echo "사용법: bash sats/tools/run_experiments.sh [단계]"
    echo ""
    echo "  analyze_gt      GT 값 범위 진단 (val_rmse 해석)"
    echo "  local_map_v2    Local Map 재학습 (센서 매핑 버그 수정)"
    echo "  cnn_v2          CNN 재학습 (local_map_v2 기반)"
    echo "  retrain_all     local_map_v2 + cnn_v2 연속 실행"
    echo "  visualize_all   4단계 예측 시각화 (v2 기반)"
    echo "  visualize_v1    기존 v1 시각화 (버그 버전 비교용)"
    ;;
esac
