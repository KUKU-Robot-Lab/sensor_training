# S19 / Table S2 — Ablation study (ecomesh, xy 1 mm)

> **2026-07-08 A 갱신**: 변형들을 **A(크기 입력) 베이스**로 재학습(`scripts/scratchpad_run_ablation_A.py` → `runs/ablation_ecomesh_A/`), full = `sizeA_ecomesh_xy1_fold3`. A 결과 overall_rel: **noAttention 0.403 > noLSTM 0.352 > noCNN 0.186 ≈ full 0.189** → attention 최핵심·LSTM 핵심, CNN 기여는 이 홀드아웃에선 미미. 해석 방법 동일.

논문 FigS19 / Table S2 대응. SATS의 각 모듈(LSTM·self-attention·CNN)을 제거한 변형을
**동일 데이터·split(ecomesh_xy1 fold3)**로 재학습해 기여도를 비교.

## Ablation 구현
- config 플래그 `ablate_lstm / ablate_attention / ablate_cnn` (config.py) → `SATSCNNStage.forward` 에서 모듈 pass-through.
  - noLSTM: LSTM → 비순환 통계 인코더(mean/max/last + 선형)
  - noAttention: self-attention 출력을 0 (이웃 정보 집계 제거)
  - noCNN: CNN refiner 생략 (merged_map 을 최종 출력)
- 각 변형 50 epoch 재학습(`scripts/scratchpad_run_ablation.py`), full = 기존 ecomesh_xy1 fold3.

## 결과 (상대오차, 낮을수록 좋음)

| variant | overall | d5 | d10 |
|---|---|---|---|
| **SATS (full)** | **0.158** | 0.353 | **0.149** |
| noCNN | 0.257 | 0.434 | 0.250 |
| noLSTM | 0.377 | 0.672 | 0.365 |
| noAttention | 0.423 | 0.746 | 0.410 |

- **모든 모듈이 기여**: full < noCNN < noLSTM < noAttention (논문 Table S2 서열과 동일).
- **self-attention 제거가 최대 악화**(0.158→0.423) → 다중유닛 공간 정보 집계가 핵심.
- CNN refiner는 상대적으로 작은 기여(smoothing), LSTM(시계열/이력) 중간.

## 코드 (재현)
```bash
# 1) ablation 재학습 (noLSTM/noAttention/noCNN)
.venv/bin/python scripts/scratchpad_run_ablation.py
# 2) 진단 평가 (full + 3 ablation)
.venv/bin/python -m sats.tools.eval_diagnostics --no-fig \
    --run-dirs sats/training/runs/xy1_material_d5d10/xy1_d5d10_ecomesh_xy1_fold3_e2e_g05 \
               sats/training/runs/ablation_ecomesh/{noLSTM,noAttention,noCNN} \
    --out-dir history/fig_data/supplementary/S19_ablation
# 3) 비교 그림
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_supp_ablation.py
```
