# S8 — 순수 시뮬레이션 학습 (SATS 원논문 Note S8 대응)

> 생성: `scripts/sim_only_s8.py` (seed 11, 3000 step, batch 512).
> 시뮬 = EHS 커널로 GT·taxel 응답 동시 생성(taxel 좌표 bilinear 샘플 × 게인 11.677,
> 로딩 램프 + 5% 노이즈). **히스테리시스·점탄성 없음** — 하한 성격.
> 실측 사용은 게인 1스칼라·meta 분포 리샘플뿐(센서 신호는 미사용).

| setting | overall_rel | d5_rel | d10_rel |
|---|---|---|---|
| sim-trained @ sim val | 0.074 | — | — |
| **sim-trained @ real val** | 1.109 | 1.090 | 1.151 |
| real-trained(A) @ real val (참조) | 0.442 | 0.188 | 0.749 |

해석 가이드:
- sim@sim 이 낮으면 "구조는 시뮬 과제를 학습 가능".
- sim@real vs real@real 차 = **sim2real 갭** — 실센서의 히스테리시스·비선형·크로스토크가
  EHS 선형 시뮬에 없기 때문. 갭이 크면 "실측 데이터 필수" 근거, 작으면 "시뮬 사전학습 가치" 근거.
- 체크포인트: `sats/training/runs/sim_only_s8/sim_only_model.pt`.
