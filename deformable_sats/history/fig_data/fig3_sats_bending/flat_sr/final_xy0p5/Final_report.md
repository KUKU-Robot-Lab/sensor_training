# Final (xy 0.5 mm) — Eco-mesh 최종 flat 데이터 성능

> **2026-07-08 A 갱신**: **크기 입력(A) 모델**(`runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3`)로 재생성. 크기 입력이 d10 blend 해소(pedestal 전구간↓). 순수 d5 성능 개선 트랙(다해상도)은 `../d5_final/README.md` 및 `experiments_archive/d5_multires_diag/README.md` 참조.

**최종 flat 데이터(xy 0.5 mm) data-rich 모델의 최종 성능** figure 세트. Fig3(xy 1mm 소재비교)와 별도.

## 모델

- run: `sats/training/runs/datarich_probe/ecomesh_xy0p5_datarich_val_d5test10_d10test3`
- 소재: **Eco-mesh 단일** (xy0.5는 ecomesh만 취득) → 소재 비교(패널 B)는 해당 없음
- 학습: train 11 trials(data-rich), 15 epoch, best@e14, val_rmse **0.283** (collapse 없음)
- val holdout: d5 test10 + d10 test3

## 진단 지표 (d5/d10 분리 + 상대오차)

| 구분 | RMSE | 상대오차 | n |
|---|---|---|---|
| overall | 0.283 | 0.417 | 2.88M |
| **d5-only** | **0.129** | **0.171** | 1.61M |
| d10-only | 0.400 | 0.710 | 1.27M |

→ **d5 상대오차 0.171** = 이전 기억 "~0.1" 성능 재현(최종 flat 데이터 대표 성능). d10은 반복취득(3rep)만이라 상대적으로 약함.

## 패널 (논문 Fig4 대응)

| 파일 | 논문 | 내용 |
|---|---|---|
| `FinalA_lineprofile_ecomesh.png` | Fig4A | 중앙선 압력 프로파일(SR, force 색맵) |
| `FinalC_pressure3d_ecomesh.png` | Fig4A/E | 3D 압력맵 GT/Pred (d5·d10) |
| `FinalD_poserror3d_ecomesh.png` | Fig4B | 위치별 오차 3D 막대(d5/d10) |
| `FinalE_error_hist_ecomesh.png` | Fig4C | 상대오차 히스토그램+KDE |
| `FinalF_force_error_ecomesh.png` | Fig4D | force별 오차 바이올린 |

(패널 B = 소재 비교는 단일 소재라 자동 skip)

## 코드 (재현)

동일 스크립트, figset만 전환:

```bash
# 1) 진단 재평가 + per-sample npz 덤프
.venv/bin/python -m sats.tools.eval_diagnostics \
    --run-dirs sats/training/runs/datarich_probe/ecomesh_xy0p5_datarich_val_d5test10_d10test3 \
    --out-dir history/fig_data/experiments_archive/final_xy0p5_diag --dump-samples

# 2) figure 생성 (figset=xy0p5_final)
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_fig3_sats.py \
    --figset xy0p5_final --panels A C D E F
```

figset 정의는 `generate_fig3_sats.py`의 `FIGSETS["xy0p5_final"]`.


> **⚠️ 2026-07-20 재평가 정정**: 이 문서의 rel RMSE 수치(특히 d10)는 **저force 분모 왜곡**을 포함한다. rel = rmse/target_RMS 는 저force(target 압력≈0)에서 폭발하며, 계단식 xy0.5·저force 홀드아웃이 특히 영향받았다. **map 품질 재평가(`experiments_archive/reeval/map_quality.md`)로는 위치 loc 0.5mm·peak 상관 높음 = 재구성 정확**. d10 "약함"은 특정 저force 홀드아웃 아티팩트이며 학습 실패 아님. 표준 지표는 loc+peak 상관+rel(저force 제외)+절대 rmse. rel 단독 인용 금지.
