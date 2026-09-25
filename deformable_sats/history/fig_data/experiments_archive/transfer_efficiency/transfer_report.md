# 데이터 효율/전이 리허설 결과 (새 센서 취득량 결정 근거)

> 생성: `scripts/analyze_transfer_efficiency.py`. 러너: `scripts/rehearse_transfer_efficiency.sh`.
> 홀드아웃 = ecomesh_xy1 fold3(d5 test3 + d10 test3). zero-shot 은 rel RMSE, 학습은 best val RMSE — 단위 다름 주의.

| setting | metric | value |
|---|---|---|
| zero-shot: xy0.5 model (protocol shift) | zero-shot rel | 0.6593 |
| zero-shot: eco20 model (unit-variation proxy) | zero-shot rel | 0.8944 |
| scratch 1-pair (2 trials) | best val rmse | 0.4698 |
| warm(xy0.5) 1-pair | best val rmse | 0.3844 |
| scratch 2-pair (4 trials, ref fold3) | best val rmse | 0.1541 |
| warm(xy0.5) 2-pair | best val rmse | 0.1319 |
| cross-warm(eco20) 2-pair | best val rmse | 0.1072 |
| xy1 2-pair + 0.25mm output (81x81) | best val rmse | 0.1210 |

## 결론 (2026-07-19) — 새 센서 취득 계획 확정 근거

1. **Zero-shot 불가**: 프로토콜 전이 0.659·유닛편차 프록시 0.894 — 게인 보정 없는 그대로 적용은 무용. fine-tune 필수.
2. **최소 취득량 N = 2 pair (d5×2 + d10×2 = 4 trial)**: 1 pair는 warm으로도 부족(0.384), 2 pair warm 0.132 · cross-warm 0.107로 기존 최고(fold3 0.154, healthy 최고 0.129)와 동급 이상.
3. **Warm-start 확정 이득**: 최종 성능 14~30% 개선 + 2~5 epoch 만에 scratch 50ep 수준(학습시간 ~1/10).
4. **Cross-source 출발 성립**: 다른 소재 가중치(유닛편차 비관 프록시)에서 출발해도 4 trial로 완전 회복+개선(0.107, 전 조건 최고). **프로토콜 일치 > 소재 일치** 시사.
5. **Coarse 스캔 + fine 출력 성립**: xy1 취득 4 trial로 0.25mm(81²) 출력 학습 0.121 — "취득 간격과 출력 해상도 독립"의 학습 성능 직접 증거.

**새 센서 실행안**: xy1 프로토콜 d5×2+d10×2(+홀드아웃용 1 pair 권장 = 총 6 trial) 취득 → same-protocol 기존 가중치 warm-start(~30분) → 필요시 0.25mm 출력. 취득량 = 구 xy0.5 경로 대비 약 1/3, 프로토콜도 빠름.

**캐빗(정직)**: cross-warm 프록시는 '다른 소재·같은 취득계'이지 실제 다른 유닛이 아님 — 실제 새 센서에서 3단 평가(zero-shot→게인 보정→fine-tune)로 확증 필요. 홀드아웃도 동일 센서 내 trial 단위임.

## 2026-07-29 — 실측 새 센서(ecomesh v3)로 캐빗 해소

위 리허설의 프록시 한계(같은 취득계·다른 소재, 홀드아웃도 동일 센서 내 trial)를 **실제 새 센서 v3**로 확증했다. v3는 인덴터당 test1+test2 = 4 trial 취득 → **train=test1, holdout=test2 전체**(학습에 한 번도 미포함)로 진짜 일반화를 측정. 지표는 신뢰 가능한 **map 품질**(스케일-무관; rel RMSE 아님 — 저force 왜곡 아티팩트). 재현: `v3_realsensor/generate_v3_transfer.py`(→ `v3_realsensor_transfer.png`), 수치는 `scripts/reeval_map_quality.eval_model`.

| 인덴터 | 지표 | 전이 전(소스 모델) | 전이 후(v3 warm, 2 trial) |
|---|---|---|---|
| d5 | loc(mm) | 2.06 | **0.71** |
| d5 | peak_corr | 0.701 | **0.802** |
| d10 | loc(mm) | 2.06 | **1.41** |
| d10 | peak_corr | 0.735 | **0.789** |

- **결론**: 센서당 **2 trial warm-start만으로** 미관측 홀드아웃에서 위치·형태 재구성이 소스보다 확실히 개선(d5 loc −66%, d10 −32%). 리허설의 "N=2 pair 충분" 결론이 실측에서 성립 → SOP 확정.
- **⚠ 지표 주의**: 홀드아웃 val_rmse=0.78·epoch0(첫 에폭) best 이후 단조 상승은 map 저하가 아니라 저force RMSE 왜곡 + 소량 과적합이 rmse에만 나타난 것(map은 양호). 따라서 다음 센서는 epochs 20→~5로 축소해도 best_model이 동일 epoch을 선택 → 동일 성능·1/4 시간.
- run: `sats/training/runs/ecomesh_v3_calibtransfer_warm`(홀드아웃), 배포용 전량학습 = `ecomesh_v3_deploy_all4`. 인덱스 `learning_data/trial_indices/ecomesh_v3_warm`.
