# 소재 비교 — A(인덴터 크기 입력) 최종 모델 진단 · 해석 가이드

**모델**: eco20/eco50/ecomesh xy1 대표 fold + **인덴터 크기(지름) FiLM 입력(A)**. d5/d10 혼합 학습 시 모델이 크기를 몰라 blend하던 문제를 크기 입력으로 해소. β는 배제(무이득). 출력 grid 0.5mm.

- eco20 = fold2, eco50 = fold1, ecomesh = fold3 (fig3/summary가 쓰는 대표 fold)
- run: `sats/training/runs/size_input_material/sizeA_*`

## 파일별 해석

### `diag_summary.csv`
- **d10_rel_rmse**: d10(10mm 인덴터) 상대오차 — **소재 비교의 핵심 지표**(d5는 저force·커버리지 문제로 신뢰도 낮음, d10 중심).
- **소재 서열(낮을수록 우수)**: **ecomesh 0.182 < eco20 0.259 < eco50 0.336** → 가설(ecomesh 우수) 지지.
- d5_rel도 있으나 xy1 d5는 전 소재 저force라 참고용.

### `diag_<run>.png` — 위치별 RMSE 히트맵 + force별 오차 (소재 통일축)
- **좌**: 감지 표면 위치별 평균 RMSE 히트맵(magma). 전 run 공통 vmax → 소재 간 직접 비교 가능. 밝을수록 오차 큰 위치.
- **우**: RMSE vs fz 곡선(이동평균). 공통 y·x축.
- **읽는 법**: 히트맵이 전반적으로 어두우면 표면 전체 정확. 특정 영역만 밝으면 그 위치 취약. 소재끼리 나란히 보면 어느 소재가 더 균일·정확한지 판단.

### `samples_<run>.npz`
- per-sample rmse/rel/x/y/fz/is_d5. 재플롯용(generate_diag_unified가 사용).

## 재현
```bash
.venv/bin/python -m sats.tools.eval_diagnostics --run-dirs sats/training/runs/size_input_material/sizeA_* --out-dir <this> --dump-samples --no-fig
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_diag_unified.py   # 통일축 히트맵
```

## 주의
- **d5 비교는 신뢰도 낮음**(xy1 d5 전 소재 저force) → **d10·force-matched 중심** 해석.
- 상대오차는 신호 크기 정규화. eco50가 높은 건 소재 특성(취득 힘 분포 포함)이 섞인 결과 — 절대 압력장 ground-truth 없음.
- 크기 입력은 알려진 인덴터(교정·특성화)엔 정당. 미지 접촉 실사용엔 크기를 모르는 한계 있음(센서 sparsity 근본).
