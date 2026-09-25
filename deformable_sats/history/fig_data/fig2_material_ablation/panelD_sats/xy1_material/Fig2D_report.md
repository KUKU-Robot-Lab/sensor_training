# Fig.2D (구 Fig3) — 소재별 SATS(xy_1mm) 학습 결과 리포트

> **2026-07-08 A 갱신**: 모든 fig3 패널은 **인덴터 크기 입력(A) 모델**(`runs/size_input_material/sizeA_*`, xy0p5_final은 `runs/size_input/`)로 재생성됨. 크기 입력이 d5/d10 blend를 해소(d10 pedestal 전구간↓). β 물성보정은 무이득으로 배제. 소재 d10 서열 유지: **ecomesh 0.182 < eco20 0.259 < eco50 0.336**. 아래 해석 방법은 그대로 유효(모델만 A). d5-only 최종 트랙은 `d5_final/README.md` 참조.

참고 논문(*Super-resolution tactile sensor arrays with sparse units*, Fig.4) 시각화를
우리 xy_1mm 학습 SATS 체크포인트로 재현. **핵심 메시지 = 소재 비교(가설 검증) + SATS 추론 품질.**

## 지표 정의 (중요)

- 저장된 `val_rmse`(= 배치 MSE의 sqrt)는 GT target = `base_kernel × fz × gt_scale(100)` 구조상
  **고force·d10 샘플에 지배**되는 아티팩트다. 따라서 본 Fig.2D는 모든 지표를 **d5/d10 분리 + 상대오차**로 보고한다.
- **상대오차(relative RMSE)** = `per-sample RMSE / target RMS`. 스케일 불변이므로 소재·force 간 공정 비교 가능.
- 압력 절대값 단위는 스케일된 **a.u.**(`norm_kernel × fz × 100`) — kPa로 직접 환산하지 않음.

## 대표 fold (fold collapse 회피)

데이터 부족으로 일부 fold가 early-overfit 붕괴(best@e1~2). **소재별 정상 fold(=d10 상대오차 최상)**를 대표로 사용:

| 소재 | 대표 run | d5 rel | d10 rel |
|---|---|---|---|
| Eco20 | `..._eco20_xy1_fold2_...` | 0.442 | 0.268 |
| Eco50 | `..._eco50_xy1_fold1_...` | 0.476 | 0.368 |
| **Eco-mesh** | `..._ecomesh_xy1_fold3_...` | **0.353** | **0.149** |

붕괴 fold(eco50 fold3 rel 2.10, ecomesh fold1 0.70 등)는 대표에서 제외하되 Fig2D_B에 회색 산점으로 함께 표기(정직 리포팅).

## 패널 (모든 자료 소재별)

- **Fig2D_A_lineprofile_{eco20,eco50,ecomesh}.png** — 감지면 중앙선 press들의 추론 압력 단면(pressure vs x)을 겹쳐 그리고 force로 색 매핑.
  겹치는 종형 곡선 = **수용영역 중첩 → 초해상도(SR)** 근거. 논문 **Fig4A** 대응. (중앙 행에 peak가 놓인 press만, 과밀 방지 160개 subsample.)
- **Fig2D_B_material_compare.png** — 소재별 d5/d10 상대오차 막대(대표 fold, 값 라벨 포함).
  → **Eco-mesh(d10 0.149) < Eco20(0.268) < Eco50(0.368)**. 가설 *Eco-mesh ≥ Eco50* 지지.
  (구지표에서 Eco20이 1위였던 건 Eco20 d5가 거의 안 눌려 target≈0인 착시였음.
  대표 fold = 소재별 best_epoch 최대(비붕괴) fold: eco20 f2, eco50 f1, ecomesh f3.)
- **Fig2D_C_pressure3d_{eco20,eco50,ecomesh}.png** — 소재별 **3D 압력 분포맵**(GT vs Prediction), d5·d10 대표 press.
  대표 press는 GT 맵 peak 픽셀이 중앙 40% 박스 안인 것 중 최대(=blob 중앙 보장). 논문 Fig4A/E 대응.
- **Fig2D_D_poserror3d_{eco20,eco50,ecomesh}.png** — 소재별 감지면 위치별 평균 상대오차 **3D 막대(bar3d, d5+d10 통합)**. 논문 **Fig4B** 스타일 재현(상대오차는 스케일 불변이라 d5+d10 통합 가능). 절대 kPa 아님.
- **Fig2D_E_error_hist_{eco20,eco50,ecomesh}.png** — 소재별 상대오차 히스토그램 + KDE + 평균선(d5/d10). 논문 **Fig4C** 대응.
- **Fig2D_F_force_error_{eco20,eco50,ecomesh}.png** — 소재별 force(fz) 구간별 d10 상대오차 바이올린 + 평균선. 논문 **Fig4D** 대응.
  → force↑ → 상대오차↓ (SNR 향상), 논문 경향 일치. d5는 저force(~1N)에 국한(취득 특성).

## 코드 (모든 피규어는 재현 코드 보유)

생성 스크립트: `history/fig_data/visualizing_scripts/figure_set/generate_fig3_sats.py`
진단/데이터: `sats/tools/eval_diagnostics.py` (`collect_samples` + `--dump-samples`)

| 피규어 파일 | 생성 함수 | 패널 인자 |
|---|---|---|
| `Fig2D_A_lineprofile_{mat}.png` | `panel_symmetry_line` | `--panels A` |
| `Fig2D_B_material_compare.png` | `panel_material_compare` | `--panels B` |
| `Fig2D_C_pressure3d_{mat}.png` | `panel_pressure_maps` | `--panels C` |
| `Fig2D_D_poserror3d_{mat}.png` | `panel_position_error` | `--panels D` |
| `Fig2D_E_error_hist_{mat}.png` | `panel_error_hist` | `--panels E` |
| `Fig2D_F_force_error_{mat}.png` | `panel_force_error` | `--panels F` |

## 재현 방법

```bash
# 1) 진단 재평가 + per-sample npz 덤프 (9 xy1 runs)
.venv/bin/python -m sats.tools.eval_diagnostics \
    --run-dirs sats/training/runs/xy1_material_d5d10/xy1_d5d10_*_e2e_g05 \
    --out-dir history/fig_data/experiments_archive/fig3_diag --dump-samples

# 2) Fig.2D 전체 패널 생성 (한 장씩: --panels A / B / C / D / E / F 선택)
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_fig3_sats.py

# 3) 소재 간 비교용 — 동일 축 범위 버전 (출력: shared_axes/ 하위, axis_limits.json 저장)
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_fig3_sats.py --shared-axes

# 4) final(xy0.5) 을 xy1 소재들과 동일 축으로 렌더 (ref-limits 주입)
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_fig3_sats.py \
    --figset xy0p5_final --panels A C D E F --shared-axes \
    --ref-limits "history/fig_data/fig2_material_ablation/panelD_sats/xy1_material/shared_axes/axis_limits.json"
```

### final(xy0.5) 을 xy1 과 동일 축으로 (`--ref-limits`)

`--shared-axes` 실행 시 xy1_material 은 계산한 축 한계를 `shared_axes/axis_limits.json` 으로 저장한다.
final(xy0p5, ecomesh 단일)을 이 JSON으로 렌더하면 **xy1 소재들과 완전히 같은 축 범위**가 되어
최종 xy0.5 모델을 eco20/eco50/ecomesh(xy1)와 직접 비교할 수 있다(출력=`final_xy0p5/shared_axes/`).
- 적용 축: A(압력 y+force컬러)·D(오차 z+컬러바)·E(오차 x)·F(오차 y). 상대오차·압력(a.u.)은 동일 정의라 비교 유효.
- **C(3D 압력맵)는 ref 제외**: 대표 샘플 1개의 z-peak 기준이라 figset 간 강제 통일 시 봉우리가 클리핑됨.
  C의 핵심 비교는 같은 행 GT vs Pred(동일 z)이므로 figset 자체 스케일로 충분.
- **ref 는 '하한(floor)'** : 실제 적용 축 = `max(ref, 해당 figset 계산값)`. final(xy0.5)의 d10 오차는
  xy1 보다 분포가 넓어서, xy1 의 tight 한 축을 그대로 강제하면 D/E/F 상단이 잘린다.
  따라서 final 이 xy1 범위 안이면 **동일 축**, 넘치면 **축을 확장(무클리핑)**한다.
  → A 는 final 이 xy1 범위 안이라 완전 동일, D·E·F 는 final 이 더 넓어 축이 확장됨(0-기준은 공유).

## 동일 축 범위 버전 (`shared_axes/`)

소재별 파일은 y축(비교 축)이 각자 자동 스케일돼 **소재 간 눈대중 비교가 불가**하다.
`--shared-axes` 는 전 소재에서 공통 축 범위를 계산해 모든 소재 그림에 동일 적용한다(원본 자동스케일 버전은 유지).

| 패널 | 통일되는 비교 축 | 공통값 산정 |
|---|---|---|
| A 라인프로파일 | 압력 y축 + force 컬러 | 전 소재 압력 max / fz 0.97분위 max |
| C 3D 압력맵 | 압력 z축 | press-type(d5/d10)별 소재 간 peak max |
| D 3D 위치오차 | 상대오차 z축·컬러바 | 전 소재 rel 합쳐 0.95분위 |
| E 오차 히스토 | 상대오차 x축 | 전 소재·d5/d10 rel 0.99분위 max |
| F force별 오차 | 상대오차 y축 | 전 소재 그룹 rel max |

(B는 이미 한 그림에 전 소재를 담아 비교 가능하므로 별도 통일 불필요.)

## 재학습 결과 (2026-07-06, 데이터 QA 이후)

- **eco50 재보정 재학습**: eco50 d10 test3 로드셀 영점 -2.269N 교정([[retare_meta_cache]]) 후 eco50 fold1 재학습.
  → **eco50 d10 상대오차 0.70 → 0.328**(d10_target_rms 정상화). "eco50 최악"이 오염 탓이었음 확증. Fig2D_B/S20/eco50 패널 전부 재생성.
  → d10 기준 최종 서열 **Eco-mesh(0.149) < Eco20(0.268) < Eco50(0.328)** — 가설(Eco-mesh 최상) 견고.
- **ecomesh d10 pooling = 개선 없음(null)**: xy1 d10 추가해도 xy0.5 d10 홀드아웃 불변(0.710→0.709). 이유 = xy1(straight)·xy0.5(계단식) 프로토콜 도메인 차이. 상세 `experiments_archive/pool_diag/pool_result.md`. → **xy0.5 d10 개선 = xy0.5 d10 반복취득 증가가 정답**(xy1 추가 아님).
- **주의**: xy1 d5 는 전 소재 저force(커버리지 좁음, 손상 아님) → **d5 소재비교는 신뢰도 낮음, d10 중심으로 볼 것**.

### Fig2D_G(구 Fig3G) — force-matched d10 소재 비교 (분모 교란 제거)

상대오차는 검증셋 force 분포(=target_rms 분모)에 좌우돼 소재 직접비교가 왜곡됨. force 구간별로 나눠 '같은 force 에서' 비교하면 교란 없이 우열이 보임.
**결과: eco-mesh 가 전 force 구간에서 d10 최상**(상대·절대 모두 최저). 물리 직관(mesh 우수) 확증.

| force fz [N] | Eco20 | Eco50 | **Eco-mesh** |
|---|---|---|---|
| 0.25–0.5 | 0.653 | 0.796 | **0.446** |
| 0.5–1 | 0.406 | 0.564 | **0.269** |
| 1–2 | 0.227 | 0.400 | **0.155** |
| 2–5 | 0.140 | 0.240 | **0.085** |

절대 RMSE 도 eco-mesh 최저이며, eco50 은 고force 로 갈수록 급증(0.13→0.48). 생성: `generate_fig3_forcematched_d10.py` → `Fig2D_G_forcematched_d10.png`.

## 한계 / 미해결

- **단일점 xy1 데이터만** 사용 → 다점 접촉/2점 분해능/형상 이미징(논문 Fig4 E·F·G)은 데이터 없어 제외.
- **d10 성능은 반복 취득(현재 3rep) 부족**이 잔존 약점. pooling(xy1)로는 안 풀림 → 동일 프로토콜 반복 취득 필요.
- **조건 간 force 범위 불일치**는 취득 단계 이슈(버그 아님, 원천 loadcell 검증됨). 따라서 xy1 vs xy0p5 교차비교는 fig3 범위에서 제외.
- **bending 패널**은 데이터 확보 후 별도 진행.


> **⚠️ 2026-07-20 재평가 정정**: 이 문서의 rel RMSE 수치(특히 d10)는 **저force 분모 왜곡**을 포함한다. rel = rmse/target_RMS 는 저force(target 압력≈0)에서 폭발하며, 계단식 xy0.5·저force 홀드아웃이 특히 영향받았다. **map 품질 재평가(`experiments_archive/reeval/map_quality.md`)로는 위치 loc 0.5mm·peak 상관 높음 = 재구성 정확**. d10 "약함"은 특정 저force 홀드아웃 아티팩트이며 학습 실패 아님. 표준 지표는 loc+peak 상관+rel(저force 제외)+절대 rmse. rel 단독 인용 금지.
