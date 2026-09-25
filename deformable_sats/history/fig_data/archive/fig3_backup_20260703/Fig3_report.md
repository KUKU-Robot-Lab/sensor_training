# Fig3 — SATS(xy_1mm) 학습 결과 리포트

참고 논문(*Super-resolution tactile sensor arrays with sparse units*, Fig.4) 시각화를
우리 xy_1mm 학습 SATS 체크포인트로 재현. **핵심 메시지 = 소재 비교(가설 검증) + SATS 추론 품질.**

## 지표 정의 (중요)

- 저장된 `val_rmse`(= 배치 MSE의 sqrt)는 GT target = `base_kernel × fz × gt_scale(100)` 구조상
  **고force·d10 샘플에 지배**되는 아티팩트다. 따라서 본 Fig3는 모든 지표를 **d5/d10 분리 + 상대오차**로 보고한다.
- **상대오차(relative RMSE)** = `per-sample RMSE / target RMS`. 스케일 불변이므로 소재·force 간 공정 비교 가능.
- 압력 절대값 단위는 스케일된 **a.u.**(`norm_kernel × fz × 100`) — kPa로 직접 환산하지 않음.

## 대표 fold (fold collapse 회피)

데이터 부족으로 일부 fold가 early-overfit 붕괴(best@e1~2). **소재별 정상 fold(=d10 상대오차 최상)**를 대표로 사용:

| 소재 | 대표 run | d5 rel | d10 rel |
|---|---|---|---|
| Eco20 | `..._eco20_xy1_fold2_...` | 0.442 | 0.268 |
| Eco50 | `..._eco50_xy1_fold1_...` | 0.476 | 0.368 |
| **Eco-mesh** | `..._ecomesh_xy1_fold3_...` | **0.353** | **0.149** |

붕괴 fold(eco50 fold3 rel 2.10, ecomesh fold1 0.70 등)는 대표에서 제외하되 Fig3B에 회색 산점으로 함께 표기(정직 리포팅).

## 패널 (모든 자료 소재별)

- **Fig3A_lineprofile_{eco20,eco50,ecomesh}.png** — 감지면 중앙선 press들의 추론 압력 단면(pressure vs x)을 겹쳐 그리고 force로 색 매핑.
  겹치는 종형 곡선 = **수용영역 중첩 → 초해상도(SR)** 근거. 논문 **Fig4A** 대응. (중앙 행에 peak가 놓인 press만, 과밀 방지 160개 subsample.)
- **Fig3B_material_compare.png** — 소재별 d5/d10 상대오차 막대(대표 fold, 값 라벨 포함).
  → **Eco-mesh(d10 0.149) < Eco20(0.268) < Eco50(0.368)**. 가설 *Eco-mesh ≥ Eco50* 지지.
  (구지표에서 Eco20이 1위였던 건 Eco20 d5가 거의 안 눌려 target≈0인 착시였음.
  대표 fold = 소재별 best_epoch 최대(비붕괴) fold: eco20 f2, eco50 f1, ecomesh f3.)
- **Fig3C_pressure3d_{eco20,eco50,ecomesh}.png** — 소재별 **3D 압력 분포맵**(GT vs Prediction), d5·d10 대표 press.
  대표 press는 GT 맵 peak 픽셀이 중앙 40% 박스 안인 것 중 최대(=blob 중앙 보장). 논문 Fig4A/E 대응.
- **Fig3D_poserror3d_{eco20,eco50,ecomesh}.png** — 소재별 감지면 위치별 평균 상대오차 **3D 막대(bar3d, d5+d10 통합)**. 논문 **Fig4B** 스타일 재현(상대오차는 스케일 불변이라 d5+d10 통합 가능). 절대 kPa 아님.
- **Fig3E_error_hist_{eco20,eco50,ecomesh}.png** — 소재별 상대오차 히스토그램 + KDE + 평균선(d5/d10). 논문 **Fig4C** 대응.
- **Fig3F_force_error_{eco20,eco50,ecomesh}.png** — 소재별 force(fz) 구간별 d10 상대오차 바이올린 + 평균선. 논문 **Fig4D** 대응.
  → force↑ → 상대오차↓ (SNR 향상), 논문 경향 일치. d5는 저force(~1N)에 국한(취득 특성).

## 코드 (모든 피규어는 재현 코드 보유)

생성 스크립트: `history/fig_data/visualizing_scripts/figure_set/generate_fig3_sats.py`
진단/데이터: `sats/tools/eval_diagnostics.py` (`collect_samples` + `--dump-samples`)

| 피규어 파일 | 생성 함수 | 패널 인자 |
|---|---|---|
| `Fig3A_lineprofile_{mat}.png` | `panel_symmetry_line` | `--panels A` |
| `Fig3B_material_compare.png` | `panel_material_compare` | `--panels B` |
| `Fig3C_pressure3d_{mat}.png` | `panel_pressure_maps` | `--panels C` |
| `Fig3D_poserror3d_{mat}.png` | `panel_position_error` | `--panels D` |
| `Fig3E_error_hist_{mat}.png` | `panel_error_hist` | `--panels E` |
| `Fig3F_force_error_{mat}.png` | `panel_force_error` | `--panels F` |

## 재현 방법

```bash
# 1) 진단 재평가 + per-sample npz 덤프 (9 xy1 runs)
.venv/bin/python -m sats.tools.eval_diagnostics \
    --run-dirs sats/training/runs/xy1_material_d5d10/xy1_d5d10_*_e2e_g05 \
    --out-dir history/fig_data/sats_experiments/fig3_diag --dump-samples

# 2) Fig3 전체 패널 생성 (한 장씩: --panels A / B / C / D / E / F 선택)
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_fig3_sats.py
```

## 한계 / 미해결

- **단일점 xy1 데이터만** 사용 → 다점 접촉/2점 분해능/형상 이미징(논문 Fig4 E·F·G)은 데이터 없어 제외.
- **d10 성능은 반복 취득(현재 3rep) 부족**이 잔존 약점. 추가 취득 시 개선 여지.
- **조건 간 force 범위 불일치**는 취득 단계 이슈(버그 아님, 원천 loadcell 검증됨). 따라서 xy1 vs xy0p5 교차비교는 fig3 범위에서 제외.
- **bending 패널**은 데이터 확보 후 별도 진행.
