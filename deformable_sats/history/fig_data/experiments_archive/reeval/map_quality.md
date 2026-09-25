# SATS map 품질 재평가 — 위치·형태·magnitude (2026-07-20)

> `scripts/reeval_map_quality.py`. 접촉 fz>0.3N. loc=argmax 위치오차, peak_corr=GT/pred peak 상관,
> peak_ratio=pred/GT peak median(1=정확, >1=과대예측). 스케일-무관 map 품질.

| 모델 | 지름 | loc(mm) | peak 상관 | peak 비율(pred/GT) | fz中 |
|---|---|---|---|---|---|
| ecomesh xy1 | d5 | 0.5 | 0.781 | 0.749 | 0.49 |
| ecomesh xy1 | d10 | 0.5 | 0.976 | 1.025 | 1.95 |
| eco20 xy1 | d10 | 0.707 | 0.944 | 1.038 | 1.5 |
| eco50 xy1 | d5 | 0.5 | 0.423 | 0.905 | 0.49 |
| eco50 xy1 | d10 | 1.0 | 0.901 | 0.982 | 2.15 |
| mixed+A (섞음, size) | d5 | 0.5 | 0.962 | 0.949 | 1.01 |
| mixed+A (섞음, size) | d10 | 1.0 | 0.491 | 1.205 | 1.35 |
| mixed (섞음, no-size) | d5 | 0.5 | 0.968 | 0.976 | 1.01 |
| mixed (섞음, no-size) | d10 | 0.707 | 0.548 | 1.462 | 1.35 |
| d10-only (분리) | d10 | 1.581 | 0.323 | 1.231 | 1.35 |
| d5-only (분리, β) | d5 | 0.5 | 0.976 | 0.986 | 1.01 |

## 해석
- **loc 작고 peak_corr 높으면 = 위치·형태 정확** (rel 이 나빠도 map 은 좋음 = 지표 착시 확증).
- **peak_ratio > 1 = magnitude 과대예측** (d10 저force 에서 큼). 이건 실제 편향이나 위치와 분리됨.


## d5/d10 섞음 vs 분리 (2026-07-20 추가)

동일 xy0.5 홀드아웃(d5 test10 · d10 test3)에서:

| 학습 | d10 loc | d10 peak상관 | d5 loc | d5 peak상관 |
|---|---|---|---|---|
| mixed (섞음, no-size) | **0.71mm** | **0.548** | 0.5mm | 0.968 |
| mixed+A (섞음, 크기입력) | 1.0mm | 0.491 | 0.5mm | 0.962 |
| d10-only (분리) | **1.58mm** | **0.323** | — | — |
| d5-only (분리, β) | — | — | 0.5mm | 0.976 |

- **섞기가 d10에 명확히 유리**: d10-only(loc 1.58·corr 0.32) ≪ 섞음(0.71·0.55). d5 데이터가 부족한 d10(2 trial)을 보완 → 새 SOP(d5+d10 섞어 학습) 지지.
- d5는 섞든 분리든 동등.
- **크기입력(A) trade-off 관찰**: magnitude(peak비 1.46→1.21) 개선 / d10 위치·corr 미세 손해. rel(magnitude 지배)로만 보면 A 전면승리였으나 map 품질로는 위치 trade-off 존재. ⚠️ d10 홀드아웃 저force·3rep 한계라 방향으로만 해석.
