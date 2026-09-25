# ecomesh d10 pooling 실험 결과 (2026-07-06)

**가설**: xy0.5 d10 은 취득 3rep 뿐이라 약함 → xy1 d10 3rep 를 추가(pooling)하면 개선될 것.

**셋업**: train 14 (0p5 d5 test1–9 + 0p5 d10 test1–2 + **xy1 d10 test1–3**), holdout d5 test10 · **d10 test3**.
datarich(0p5-only, 11 trial) 와 동일 하이퍼파라미터 → 단일변수(xy1 d10 추가) 비교.
run: `sats/training/runs/pool_d10/ecomesh_pool_d10_val_d5t10_d10t3`.

## 결과: **개선 없음 (null result)**

| 홀드아웃 d10 test3 | datarich(0p5만) | pooled(+xy1 d10) |
|---|---|---|
| 상대오차 d10 | 0.710 | 0.709 |
| 절대 rmse d10 | 0.415 | 0.418 |

force 구간별로도 미세 등락뿐(0.25–0.5N 상대 1.74→1.57 소폭↓, 0.5–2N 소폭↑), **순변화 없음**.

## 해석

- **xy1(straight press) 과 xy0.5(계단식 하강)는 로딩 프로토콜이 달라 사실상 다른 도메인**.
  점탄성(rate-dependence) 확인됨: 같은 force 에서 센서 반응이 xy1(빠름) > xy0.5(느림).
  → xy1 d10 을 추가해도 xy0.5 저force 홀드아웃으로 **전이가 안 됨**.
- 홀드아웃 d10 test3 자체가 **저force**(fz_med 0.29) → 상대오차가 분모(작은 target)로 부풀려짐. 데이터량으로 안 풀림.

## 결론 / 다음

- **xy0.5 d10 개선의 진짜 해법 = xy1 추가가 아니라 xy0.5 d10(같은 프로토콜) 반복 취득 증가.**
- pooling 은 "더 많은 데이터가 항상 낫다"가 아니라 **도메인이 맞아야 이득**임을 보여준 유효한 음성 결과.
- eco50 재보정 재학습은 별개로 성공(d10 0.70→0.33, Fig3B 참조).

## ★ d10 "약함"의 정체 검증 (2026-07-06)

"d10 상대오차 높음"이 평가 버그(eco50 tare 같은)인지, 진짜 약한지, 아니면 다른 성격인지 직접 추론 검증
(`verify_d10_magnitude.py` → `verify_d10_final.png`):

- **위치·형태는 최상**: GT peak vs 예측 peak 상관 **0.960**. 세 force 레벨 모두 GT와 예측이 같은 위치·같은 blob.
- **문제 = 압력 크기 과대예측**, 특히 저force: peak 비율(예측/GT) 중앙 **2.61**.
  - 0.27N: GT 1.19 → Pred 3.78 (**3.2×**)  ·  0.71N: 2.83→5.01 (1.8×)  ·  1.59N: 5.34→6.44 (**1.2× 거의 정확**)
- **결론**: 평가/GT 정상. "d10 추론 실패"가 아니라 **저force d10 압력 magnitude 과대예측**(= force/크기 캘리브레이션 문제). 위치추정은 정확.
- **원인**: 저force d10 을 거의 못 봄(d5:d10=9:2 편중 + xy0.5 d10 3rep 중 홀드아웃이 저force). 인덴터 크기별 fz-magnitude 관계 미학습.
- **fix 방향**: force-aware/magnitude 캘리브레이션, 저force d10 데이터 보강, d5/d10 균형 학습.

## 코드 (재현)
```bash
bash scripts/scratchpad_rerun_all.sh   # eco50 재학습 + ecomesh pooled + diag + figure 재생성
.venv/bin/python history/fig_data/visualizing_scripts/figure_set/verify_d10_magnitude.py  # d10 magnitude 검증
```


> **⚠️ 2026-07-20 재평가 정정**: 이 문서의 rel RMSE 수치(특히 d10)는 **저force 분모 왜곡**을 포함한다. rel = rmse/target_RMS 는 저force(target 압력≈0)에서 폭발하며, 계단식 xy0.5·저force 홀드아웃이 특히 영향받았다. **map 품질 재평가(`experiments_archive/reeval/map_quality.md`)로는 위치 loc 0.5mm·peak 상관 높음 = 재구성 정확**. d10 "약함"은 특정 저force 홀드아웃 아티팩트이며 학습 실패 아님. 표준 지표는 loc+peak 상관+rel(저force 제외)+절대 rmse. rel 단독 인용 금지.
