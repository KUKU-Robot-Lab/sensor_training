# experiments_archive — 과정 진단·실험 산출물 (논문 figure 아님)

> 논문 figure 폴더(fig1~4·supplementary)와 분리한 **실험 과정 기록**. 진단 npz(git-ignored)·
> 요약 CSV·비교 그림·null result 를 보존한다. 재현 스크립트는 `visualizing_scripts/figure_set/`.

| 폴더 | 내용 | 결론 요약 |
|---|---|---|
| `xy1_material_d5d10/` | Experiment A(구모델) 소재 비교 요약 CSV | A 모델로 대체됨(sizeA가 최신) |
| `ecomesh_resolution_controlled_d5d10/` | Experiment B xy1 vs xy0.5 controlled | 해상도 비교 요약 |
| `diagnostics*/`, `fig3_diag/` | 구모델 진단 덤프(소재·fold별) | sizeA_diag로 대체 |
| `sizeA_diag/`, `sizeA_final_xy0p5_diag/` | **★현행 A(크기입력) 모델 진단** — Fig.2D/Fig.3 데이터 소스 | ecomesh 0.182 < eco20 0.259 < eco50 0.336 (d10_rel) |
| `pool_diag/` | d10 pooling null result + d10 magnitude 검증 + A/β 비교 | `pool_result.md` — 크기입력(A) 단독 확정 |
| `d5_multires_diag/` | d5-only 다해상도(1.0/0.5/0.25/0.1mm) | d5_rel ~0.15 해상도 무관 안정 |
| `final_maps/`, `final_xy0p5_diag/`, `data_quality/` | 대표 맵·구 최종 진단·취득 QA | data_quality_report.md 참조 |

각 폴더 세부 해석은 폴더 내 README/report 참조.
