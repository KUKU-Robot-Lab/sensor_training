# FigS30 — 좌표/힘 회귀 (localization)

논문 FigS30 대응. SATS의 local-map/CNN 대신 **회귀 head**로 접촉 좌표(x,y)·힘(fz)을 직접 추론.
encoder(LSTM)+self-attention은 SATS와 동일 구조, 새로 학습.

## 모델·학습
- `RegressionSATS`: encoder + attention + 16노드 평균풀 + MLP(→3) = (x,y,fz)/scale.
- 데이터: ecomesh_xy1 fold3 split, 40 epoch, target = meta(x,y,fz) 정규화, MSE.

## 결과 (ecomesh, xy 1 mm)
- **위치오차 0.99 mm**, **힘오차 0.229 N** (접촉 샘플 기준).
- force↑ → 위치오차↓ (0-0.25N 1.3mm → 2-5N 0.77mm), 논문 FigS30A 경향과 동일.
- 위치오차 2D 맵: 감지면 **가장자리에서 오차↑** (수용영역 한계, 논문과 일치).

## SR scale factor 주의
- ε-기반 공식 α = S/(N·π·ε²) = 400/(16·π·0.99²) ≈ **8** — 우리 ε(0.99mm)가 논문 회귀(0.12mm)보다 크고
  감지면적(≈400mm²)이 작아 낮게 나옴. 논문의 19547은 훨씬 작은 ε·큰 면적 전제.
- 격자 기반 scale factor(N_v/N_r = 1681/16 ≈ **105**)가 우리 셋업엔 더 대표적(→ summary_metrics 참조).
- 참고: 맵 argmax localization(FigS20)이 0.78mm로 회귀(0.99mm)보다 오히려 좋음 — 회귀 head는 단순·소epoch.

## 코드 (재현)
```bash
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_supp_regression.py --epochs 40
# 평가만: --skip-train
```
스크립트: `generate_supp_regression.py` (`RegressionSATS`, `train_regression`, `evaluate`, `plot`).
체크포인트: `S30_regression/regression_ecomesh.pt`.
