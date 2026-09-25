# supplementary — 보조 결과 모음

> ⚠️ **S번호(S19/S20/S29/S30)는 우리 논문의 supplementary 번호가 아니라
> SATS 원논문(*Super-resolution tactile sensor arrays...*)의 figure 번호**다.
> 즉 "선행 논문 분석 재현 세트" — 우리 논문 supplementary 구성 시 재번호 필요.
> 재현 가능성 판정표: `../SATS_paper_implementability.md`.

| 폴더 | 내용 (SATS 원논문 대응) | 생성 스크립트 |
|---|---|---|
| `S19_ablation/` | 모듈 ablation(noLSTM/noAttention/noCNN) — attention 최핵심 | `generate_supp_ablation.py` |
| `S20_localization/` | 위치추정 오차 — ecomesh_xy1 0.79mm ≈ 논문 0.73mm | `generate_supp_localization.py` |
| `S29_attention_tsne/` | attention 해석성 t-SNE | `generate_supp_attention_tsne.py` |
| `S30_regression/` | 좌표/힘 회귀(RegressionSATS, A/β 무관 독립 모델) | `generate_supp_regression.py` |
| `summary_metrics/` | Note S1 scale factor + 요약 카드 | `generate_supp_summary.py` |
| `d5_final/` | **d5-only 다해상도 최종**(β 물성보정, 0.5mm) — 본문 §6 패널 아님이라 여기 배치 | `generate_fig3_sats.py --figset d5_final` |
| `S8_sim_only/` | Note S8 — 순수 EHS 시뮬 학습: sim@sim 0.074 vs **sim@real 1.109**(전이 실패) → 실측 필수 근거 | `scripts/sim_only_s8.py` |

스크립트 위치: `../visualizing_scripts/figure_set/`. 모델 = 크기입력(A) 일관화 완료(2026-07-08, S30 제외=독립).
