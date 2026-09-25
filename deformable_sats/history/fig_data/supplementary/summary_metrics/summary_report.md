# 요약 지표 + SR scale factor (Note S1)

> **2026-07-08 A 갱신**: **A(크기 입력) 모델** 진단(`sizeA_diag`, `sizeA_final_xy0p5_diag`) 기반으로 재생성. 그림 주석도 "크기 입력 최종·β 배제"로 정정됨. d10 blend는 크기 입력이 해소, β 물성보정은 무이득(저force d10 악화)으로 배제. 해석 방법 동일. 상세는 `../../experiments_archive/sizeA_diag/README.md`.

모델별 핵심 지표 통합. 재학습·추가추론 없이 기존 CSV(diag_summary + loc_summary)만 사용.

## SR scale factor (Note S1)
- α = N_virtual / N_physical = 41² / 16 = **1681 / 16 ≈ 105**
- 논문(2700/23 ≈ 117)과 동급 수준. (우리 센서 16 physical taxel → 41×41 virtual map)

## 모델별 지표

| model | d5 rel | d10 rel | loc [mm] |
|---|---|---|---|
| eco20_xy1 | 0.442 | 0.268 | 0.95 |
| eco50_xy1 | 0.476 | 0.368 | 1.32 |
| **ecomesh_xy1** | 0.353 | **0.149** | **0.78** |
| ecomesh_xy0p5_final | **0.171** | 0.710 | 0.83 |

- **ecomesh_xy1**: d10 상대오차·위치오차 모두 최소 → 가설(ecomesh 최상) 재확인.
- **ecomesh_xy0p5_final**: d5 상대오차 최소(0.171, 최종 flat 데이터 성능). d10은 반복취득 부족으로 약함.

## 코드
```bash
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_supp_summary.py
```
입력: `fig3_diag/diag_summary.csv`, `final_xy0p5_diag/diag_summary.csv`, `S20_localization/loc_summary.csv`.

> **⚠️ 2026-07-20 재평가 정정**: 표의 **d10 rel 수치는 저force 분모 왜곡** 포함(rel=rmse/target_RMS, 저force에서 폭발). map 품질(`../../experiments_archive/reeval/map_quality.md`)로는 위치 loc 0.5mm·peak 상관 높음 = 재구성 정확. **d10 rel 단독 인용 금지**, loc+peak상관+rel(저force제외) 사용. d5 rel도 무접촉 구간 포함 주의.
