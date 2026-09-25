# Fig.3 — SATS 최종(flat SR) + 밴딩 (논문 §6 Fig.3, C2)

> 논문 전략: ecomesh flat SR 최종 성능 + 밴딩 baseline → 곡률 자가 추정 → 잔차 보정으로
> 밴딩 상태 SR 유지 검증.

## 구조

| 폴더 | 내용 | 상태 |
|---|---|---|
| `algorithm/` | Fig.3 알고리즘 구조도(flat 학습→동결 + bending 경로) 초안들 + `generate_fig3_algorithm.py` | 초안 완료 |
| `flat_sr/final_xy0p5/` | **ecomesh xy0.5 최종 flat SR** (크기입력 A 모델) — 패널 Final A~F + report | 완료 |
| `bending/` | 밴딩 파트 (패널 B~E) — **데이터 취득 대기**, 취득 스펙은 폴더 README | 준비 |

## 재현

- flat SR 패널: `generate_fig3_sats.py --figset xy0p5_final` (진단 npz: `experiments_archive/`)
- 밴딩: `sats/bending/` (Phase 0 완료, 취득 후 P1→P4 — `bending/README.md`)

소재 비교(구 Fig3* → `Fig2D_*`)는 논문 Fig.2D로 이동 → `fig2_material_ablation/panelD_sats/`.
d5-only 다해상도(d5_final)는 본문 §6 패널이 아니라 → `supplementary/d5_final/` 이동(2026-07-12).
