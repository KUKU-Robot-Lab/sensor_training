# d5-only 다해상도 SATS 결과 — 해석 가이드

**모델**: d5(5mm 인덴터) 단독 학습 + β 물성보정 GT. 크기입력 불필요(단일 인덴터라 애매성 없음) → 순수 SATS 구조. 재료 = ecomesh_xy0p5, 홀드아웃 = d5 test10.

**출력 grid(가상 taxel 해상도)를 4단계로 생성**: 1.0 / 0.5 / 0.25 / 0.1 mm. 논문 근거: 보정 EHS GT가 공간 연속이라 GT map을 임의 해상도로 생성 가능 → 가상 taxel 수는 자유 선택(더 조밀한 물리 취득 불필요).

| run | 출력 grid | 격자 | 가상 taxel | SR 배율 |
|---|---|---|---|---|
| d5only_beta_g1p0 | 1.0mm | 21×21 | 441 | 27× |
| d5only_beta_g0p5 | 0.5mm | 41×41 | 1,681 | 105× |
| d5only_beta_g0p25 | 0.25mm | 81×81 | 6,561 | 410× |
| d5only_beta_g0p1 | 0.1mm | 201×201 | 40,401 | 2,525× |

(참고: 논문은 2,700 가상 taxel / 117× SR)

---

## 파일별 해석 방법

### `diag_summary.csv` — 정량 지표
- **d5_rel_rmse**: 상대 RMSE = rmse / target_rms. **핵심 지표**(신호 크기로 정규화, 해상도·force 무관 비교 가능). 낮을수록 좋음.
- d5_rmse: 절대 RMSE(스케일 의존, 해상도 간 직접 비교 부적절 — grid_step에 따라 GT peak 스케일 다름).
- **해석**: d5_rel이 1.0mm 0.156 / 0.5mm 0.149 / 0.25mm 0.150 / 0.1mm 0.147 → **SR 27×~2525×(100배 범위)에 걸쳐 ~0.15로 안정**. 해상도를 올려도 상대 정확도 유지 = 논문의 "연속 GT → 임의 해상도 SR" 실증.

### `d5_resolution_compare.png` — 출력 grid 비교 (핵심)
- **좌 패널**: d5 rel-RMSE(중앙값) vs SR 배율(log축). 평탄한 곡선 = 해상도 무관 안정성.
- **우 패널**: force 구간별 rel-RMSE, 해상도별 곡선 오버레이. 저force에서 rel이 커지는 건 분모(작은 target) 효과.
- **읽는 법**: 곡선이 평평할수록 "해상도를 올려도 성능 유지". 특정 해상도가 튀면 그 grid에서 문제.

### `d5_SR_progression.png` — 같은 접촉을 4해상도로
- 동일 접촉(fz≈3N)을 1.0→0.1mm 출력으로 나란히. 위=GT, 아래=Pred.
- **읽는 법**: 오른쪽으로 갈수록 격자가 조밀(21²→201²)해지며 압력 분포가 매끄럽게 super-resolve. GT와 Pred가 시각적으로 일치하면 SR 성공.

### `d5_gtpred_gallery_{g0p5,g0p1}.png` — 표면 전반 위치별 GT vs Pred (핵심)
- 감지 표면을 **3×3 영역**으로 나눠 각 영역 대표 접촉 선택 → **중앙·상하좌우·모서리 9위치** 고루(가장자리 편중 해소).
- 각 행 = 한 위치, 열 = **GT | Pred | |Pred−GT|**. 왼쪽 라벨 = 접촉 (x,y)mm·fz. 청록 `+` = GT peak 위치.
- **읽는 법**:
  - GT와 Pred의 blob 위치·모양·크기가 일치 → 위치·형태 복원 성공.
  - |Pred−GT|(3열)이 어두울수록 오차 작음. 밝은 영역 = 오차 큰 곳.
  - 여러 위치에서 고르게 일치하면 표면 전체에서 동작(특정 위치 편향 없음).
  - g0p5(41²) vs g0p1(201²) 비교로 해상도별 세밀도 차이 확인.

### `samples_d5only_beta_*.npz` — per-sample 원자료
- 키: rmse, rel, dia, x, y, z, fz, is_d5. 재추론 없이 재분석/재플롯용. (resolution_compare가 이걸 사용)

---

## 재현 (코드)
```bash
# 학습(다해상도): 1.0/0.5/0.25
bash scripts/scratchpad_d5_multires.sh
# 0.1mm (AMP 필수): --grid-step-mm 0.1 --use-amp --num-workers 8 (README 상단 표 참조)
# 진단 재덤프
.venv/bin/python -m sats.tools.eval_diagnostics --run-dirs <runs> --out-dir <this dir> --dump-samples --no-fig
# 그림
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/viz_d5_resolution_compare.py
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/viz_d5_multires_SR.py
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/viz_d5_gtpred_gallery.py g0p5   # 또는 g0p1
```

## 주의/한계
- 상대 RMSE는 신호 크기 정규화라 해상도 비교엔 적절하나, **절대 정확도**는 센서 sparsity(16 taxel, 6.5mm pitch)+GT 모델(EHS+β) 충실도로 제한됨. 해상도를 올려도 이 물리 한계 이상의 실측 세부는 안 생김(논문 117×도 동일).
- β(물성보정)는 d5-only에선 이득(d10 없어 부작용 없음). 계수 = FE-Ogden 압축 유도(c1=0.00244, c2=1.7e-4).


## 해상도별 x·y·fz 추종 (2026-07-19 추가 — `scripts/analyze_loc_vs_resolution.py`)

동일 홀드아웃(d5 test10, 접촉 fz>0.3N)에서 4개 해상도 모델의 argmax 위치·맵적분 힘 추종:

| 출력 grid | loc median (mm) | loc mean | loc p90 | 양자화 하한 | fz rel median |
|---|---|---|---|---|---|
| 1.0mm | 0.0* | 0.32 | 1.00 | 0.41 | 0.39 |
| 0.5mm | 0.50 | 0.41 | 0.71 | 0.20 | **0.15** |
| 0.25mm | 0.35 | 0.41 | 0.75 | 0.10 | 0.17 |
| 0.1mm | **0.22** | 0.34 | 0.58 | 0.04 | 0.15 |

*1.0mm median 0은 양자화 아티팩트(취득 0.5mm 위치가 1.0mm 셀 중심과 일치하는 경우 다수).

해석: ①**위치 추종은 fine 출력에서 실질 개선**(median 0.50→0.22mm) ②0.1mm의 잔여 0.22mm ≫ 양자화 하한 0.04mm
→ **정보 한계(센서 sparsity)**이며 0.1mm 미만 출력은 무의미 ③**fz는 0.5mm 이하에서 해상도 불변(~15%)**,
1.0mm는 적분 부정확으로 악화. 권장: 실시간 x·y·fz 추종 = 0.25mm 균형점, 정밀 위치 figure = 0.1mm, fz만이면 0.5mm 충분.
산출: `loc_vs_resolution.{csv,png}`.
