# 취득 데이터 품질 QA (ecomesh/eco20/eco50, xy1·xy0.5, d5·d10)

meta cache 의 `target_ts`(실제 학습에 쓰는 프레임) 기준으로 31 trial 을 분석.
**health 는 물리 근거로 판정**(중앙값 아님 — 아래 "정정" 참조).

## 판정 방법 (물리 근거)

| 판정 | 기준 | 의미 |
|---|---|---|
| **tare_error** | 무접촉(센서<0.5)에서 force 가 0 이 아님 (\|offset\|>0.3N) | 로드셀 영점 오류. 대개 **상수 offset → 복구 가능** |
| **dead_sensor** | force 실렸는데(fz>0.5) 센서 무반응(<2) | 센서 고장 |
| **ok** | 무접촉 force≈0 + 힘 실릴 때 센서 반응 | 정상 |

force 범위(fz max)는 health 가 **아니라 '커버리지'** 로 별도 보고. 저force = 정상, 범위만 좁음.

## 결과: (교정 후) 31 ok / 0 tare_error / 0 dead_sensor

- **`eco50_xy1_d10_test3` — 재보정 완료.** tare offset **-2.269N** (무접촉 n=456 에서 매우 안정, 임계값 0.2~0.8 전부 동일 = 드리프트 없는 상수). `sats.tools.retare_meta_cache --apply` 로 `fz += 2.269` 교정.
  - **교정 후 검증**: 무접촉 force ≈ 0(전 임계값), 센서 구간별 force 가 정상 test1 과 일치(6~10: 0.44↔0.40, 20~40: 1.50↔1.49), 음수 비율 13%(정상 test1 과 동일). → **완전 복구**. 원본은 `.pt.bak` 백업, 이력 `corrections.json`.
- **저force 커버리지(정상, 범위만 좁음) 9개** = 전 소재 xy1 d5. 특히 eco20 xy1 d5 는 fz max ~0.28N 로 매우 좁음(데이터 문제 아님, 커버리지만 좁음).

## eco50 test3 재보정 (재현)

```bash
# 추정·검증(dry-run) 후 적용
python -m sats.tools.retare_meta_cache \
    learning_data/gt_meta_cache_xy_d5d10_g05/eco50_xy1_d10_z3.5_test3_870bc4ac6f_meta_cache.pt --apply
```
- 근본 원인: force = (loadcell_kg − baseline) × g 에서 baseline(무부하 영점)이 -2.27N 어긋남.
  dataset 이 `fz ≤ 0` 을 무접촉(GT=0)으로 처리하므로, 음수 영점이 실접촉을 무접촉으로 오판해 GT 오염.
- 교정 = 무접촉 프레임 force 중앙값으로 offset 추정 후 fz 에 더함(상수 re-tare).

## ★ 초기 오판 정정 (중요)

처음엔 **중앙값 기반**으로 xy1 d5 9개를 "센서 고장"으로 판단 → **틀림**. 원인:
- **취득 프로토콜 차이**: xy0.5 = 계단식 느린 하강(깊이마다 dwell → 고force·프레임 많음),
  xy1 = straight press(눌렀다 복귀, 빠름 → 저force·0근처 프레임 많음). → 중앙값만 낮았을 뿐.
- **force-matched 검증**(같은 fz 에서 센서 반응)으로 xy1 d5 센서 정상 확인:
  fz 0.5~1.0N 에서 센서 xy1 d5 = 13.0, xy0.5 d5 = 13.0 (동일). fz 0.3~0.5N 에선 xy1 이 오히려 큼(8.96 vs 6.18) = 소재 **점탄성(rate-dependence)**.
- **fz–sensor 시간 정렬**: lag=0 상관 0.99(전 프로토콜). fast-press 의 5~11프레임 지연은 점탄성 물리이지 기록 오류 아님 → **라벨 신뢰 가능**.

## 학습 함의

- **프로토콜 다양성(계단식+straight)은 학습에 유리**: 로딩 속도·force 범위 커버 → 강건. SATS LSTM 이 시간 윈도우로 로딩 속도 추론. → d10 pooling(xy0.5 느림 + xy1 빠름)은 이득.
- **eco50 xy1 소재 비교가 test3 로 오염**됐었음 → 재보정 또는 제외 후 재비교 필요.
- xy1 d5 는 정상이나 force 범위가 좁아(≤0.7N) 상대오차가 부풀려 보임 — **데이터 문제 아님, 지표/커버리지 이슈**.

## 코드 (재현)

```bash
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/analyze_data_quality.py
```
산출물: `data_quality_overview.png`(좌=tare offset, 우=force 커버리지), `data_quality_summary.csv`.
