# Fig.2 패널 D — 소재별 SATS 학습 성능 비교 (논문 §6 Fig.2D)

> eco20 / eco50 / ecomesh 를 **동일 조건(xy1, 크기입력 A 모델, 대표 healthy fold)** 으로 학습해
> SR 성능을 비교. 논문 C1("mesh 최종 선택")의 학습 성능 근거.
> 파일명 접두어 = `Fig2D_*` (2026-07-12 구명명 `Fig3*`에서 변경, figset `prefix="Fig2D_"`).

## 핵심 수치 (2026-07-07 A 모델 확정)

d10 상대오차(공정 지표): **ecomesh 0.182 < eco20 0.259 < eco50 0.336** — 가설(mesh 우세) 지지.
d5 비교는 xy1 전 소재 저force라 불신 → d10/force-matched 중심 해석.

## 파일 ↔ 생성 스크립트

| 파일 | 내용 | 생성 코드 |
|---|---|---|
| `xy1_material/Fig2D_B_material_compare.png` | 소재 비교 막대(대표 패널) | `visualizing_scripts/figure_set/generate_fig3_sats.py --figset xy1_material` |
| `xy1_material/Fig2D_{A,C,D,E,F}_*_{소재}.png` | 소재별 line profile·3D맵·오차 | 동일 스크립트 (패널 A~F) |
| `xy1_material/Fig2D_G_forcematched_d10.png` | force-matched d10 비교 | `generate_fig3_forcematched_d10.py` |
| `xy1_material/shared_axes/` | 소재 간 동일 축 버전 + `axis_limits.json` | 동일 스크립트 `--shared-axes` |
| `xy1_material/Fig2D_report.md` | 해석·재현 커맨드 상세 (구 Fig3_report) | — |

데이터 소스: `experiments_archive/sizeA_diag/` (진단 npz), 모델 `sats/training/runs/size_input_material/`.

## 2026-07-20 map 품질 재생성 (rel 왜곡 배제)

- 기존 `Fig2D_B_material_compare.png`(d5_rel/d10_rel)는 **d5 무접촉(eco20 d5 fz>0.3N 0%) + d10_rel 저force 왜곡**으로 소재 비교 부정확 → **`Fig2D_B_mapquality.png`(map 품질 = d10 loc·peak상관)** 추가.
- 서열 **Eco-mesh(loc 0.5·corr 0.976) > Eco20(0.71·0.944) > Eco50(1.0·0.901)** — rel 아닌 위치·형태라 신뢰. 원 가설(mesh 우수) 지지.
- 생성: `visualizing_scripts/figure_set/generate_material_mapquality.py` (데이터 `reeval/map_quality.csv`). rel 버전은 비교용 보존.
