# Current vs Previous Training Regression Analysis

Date: 2026-04-15  
Author: Codex + ECC agents (`explorer`, `reviewer`)

## Executive Summary

현재 학습이 이전 모델보다 나빠진 가장 큰 이유는 `z/fz` downstream이 아니라 upstream `XY` localization 성능 저하이다.  
정량적으로 현재 `multi_head_field` Stage3의 5-fold 평균 MAE는 `x=1.3308 mm`, `y=0.5442 mm`, `z=0.1612 mm`, `fz=0.4249`이며, 이전 `SATS` 기준은 `runs_comparison_1-9_ms8`에서 `x=0.5024 mm`, `y=0.1893 mm`, `z=0.0803 mm`, `runs_comparison_7-9_ms8`에서 `x=0.4662 mm`, `y=0.1743 mm`, `z=0.0509 mm`였다. 즉 현재 파이프라인은 이전 `SATS` 대비 `x/y`에서 약 `2.6x~3.1x`, `z`에서 약 `2.0x~3.2x` 더 나쁘다.

핵심 원인은 모델 교체 하나가 아니라 다음 변경이 한 번에 들어간 것이다.

1. `xy` 타깃이 직접 연속 회귀에서 `40x40` heatmap decode 문제로 변경됨
2. 시퀀스가 실제 시간 순서가 아니라 `(trial, x, y)`별 `depth` 오름차순 윈도우로 바뀜
3. `multi_head_field`가 mixed diameter 실험(`d5`, `d10`)을 학습하면서도 radius를 입력으로 쓰지 않음
4. Stage3 multitask가 `z/fz`는 개선했지만 `xy`를 추가로 악화시킴
5. `z_fz_regressor`는 학습 시 `GT xy + radius`, 평가 시 옵션으로 `predicted xy + GT radius`를 사용해 conditioning mismatch가 있음

결론적으로 현재 설계는 `z/fz`를 별도 경로로 안정화하려는 목적에는 일부 부합했지만, 전체 성능을 좌우하는 `XY`, 특히 `X` 축 일반화가 이전 `SATS/CNNLSTM`보다 크게 후퇴했다.

## Comparison Baseline and Comparability

이번 보고서의 공식 baseline은 다음과 같이 고정한다.

- `x/y/z`: 이전 `SATS`, 보조 비교로 `CNNLSTM`
- `fz`: 이전 JSON baseline 부재로 인해 과거 MLP SR/FF 문서만 참고

직접 비교 가능 범위는 제한이 있다.

- 현재 결과는 5-fold trial-aware CV이며 split metadata가 명시돼 있다. 참고: [cv_manifest_comparison.json](/home/user/sensor_training/training/runs_comparison/cv_manifest_comparison.json), [cv_manifest_z_fz_regressor.json](/home/user/sensor_training/training/runs_z_fz/cv_manifest_z_fz_regressor.json)
- 이전 결과 JSON은 split metadata 없는 aggregate 요약이다. 참고: [runs_comparison_1-9_ms8/comparison_results.json](</home/user/sensor_training/training/training/접촉점 기준 학습_0409/학습결과/runs_comparison_1-9_ms8/comparison_results.json>), [runs_comparison_7-9_ms8/comparison_results.json](</home/user/sensor_training/training/training/접촉점 기준 학습_0409/학습결과/runs_comparison_7-9_ms8/comparison_results.json>)
- 따라서 `x/y/z`는 실질 비교는 가능하지만 완전한 apples-to-apples 비교는 아니다.
- 현재 `z_fz_regressor`의 `predicted_xy` 평가는 완전한 end-to-end가 아니다. `predicted XY + GT radius`를 사용한다. 참고: [README.md](/home/user/sensor_training/training/README.md)
- 이전 `comparison_results.json`는 `x,y,z`만 노출하고 `fz` baseline을 담지 않는다. 참고: [cnnlstm_sr.py](/home/user/sensor_training/training/models/cnnlstm_sr.py), [sats_model.py](/home/user/sensor_training/training/models/sats_model.py)

## Metric Comparison

### 1. Current Stage3 vs Previous Best Baselines

| Model / Run | x MAE | y MAE | z MAE | Note |
| --- | ---: | ---: | ---: | --- |
| Current `multi_head_field` Stage3 | 1.3308 | 0.5442 | 0.1612 | 5-fold mean, current main run |
| Previous `SATS` (`1-9`) | 0.5024 | 0.1893 | 0.0803 | previous main aggregate |
| Previous `CNNLSTM` (`1-9`) | 0.7772 | 0.3527 | 0.1517 | previous main aggregate |
| Previous `SATS` (`7-9`) | 0.4662 | 0.1743 | 0.0509 | best old result among checked runs |
| Previous `CNNLSTM` (`7-9`) | 0.7348 | 0.3457 | 0.0961 | old strong baseline |

Source files:

- Current: [training/runs_comparison/comparison_results.json](/home/user/sensor_training/training/runs_comparison/comparison_results.json)
- Previous: [runs_comparison_1-9_ms8/comparison_results.json](</home/user/sensor_training/training/training/접촉점 기준 학습_0409/학습결과/runs_comparison_1-9_ms8/comparison_results.json>), [runs_comparison_7-9_ms8/comparison_results.json](</home/user/sensor_training/training/training/접촉점 기준 학습_0409/학습결과/runs_comparison_7-9_ms8/comparison_results.json>)

관찰:

- 현재 Stage3는 이전 `SATS`보다 전 축에서 열세다.
- `CNNLSTM`와 비교해도 현재는 `x`, `y`, `z`에서 우세하지 않다.
- 따라서 “이전 모델보다 더 좋은 xy/z/fz를 위해 현재 학습으로 변경했다”는 목적은 적어도 `xy/z` 관점에서는 달성되지 않았다.

### 2. Current Internal Stage Comparison

현재 파이프라인 내부에서도 Stage3는 `xy`를 회복시키지 못했다.

| Stage | x MAE | y MAE | z MAE | fz MAE |
| --- | ---: | ---: | ---: | ---: |
| Stage1 `point + xy only` | 1.3473 | 0.5067 | 0.6550 | 0.5709 |
| Stage2 `depth-aware label + xy only` | 1.2656 | 0.5239 | 0.6072 | 0.5250 |
| Stage3 `depth-aware + z/fz auxiliary` | 1.3308 | 0.5442 | 0.1612 | 0.4249 |

Source files:

- [fold_0 stage1 metrics](</home/user/sensor_training/training/runs_comparison/folds/fold_0/metrics_multi_head_field_stage1_point_xybce1_zoff_fzoff.json>)
- [fold_0 stage2 metrics](</home/user/sensor_training/training/runs_comparison/folds/fold_0/metrics_multi_head_field_stage2_dlabel-gaussian-hertz_xybce1_zoff_fzoff_decsoftargmax.json>)
- [fold_0 stage3 metrics](</home/user/sensor_training/training/runs_comparison/folds/fold_0/metrics_multi_head_field_stage3_dlabel-gaussian-hertz_xybce1_zhuber0p2_fzhuber0p2_decsoftargmax.json>)
- Full aggregate is derived from all fold metric JSONs under [training/runs_comparison/folds](/home/user/sensor_training/training/runs_comparison/folds)

관찰:

- Stage2는 Stage1보다 `xy`가 약간 좋아진다.
- Stage3는 `z/fz`는 크게 좋아지지만 `x/y`는 Stage2보다 오히려 나빠진다.
- 즉 shared backbone multitask가 현재 설정에서는 `z/fz`를 위해 `xy` capacity를 일부 희생하고 있다.

### 3. XY Failure Pattern

현재 문제는 전반적 실패보다 `X` 축 편향에 가깝다.

- Heatmap aggregate: `mae_x=1.5855`, `mae_y=0.5881`, `mae_z=0.5734`, `mae_fz=0.5261`
- `xy_err_mean=1.8364 mm`, `xy_err_p95=7.3397 mm`
- `r2_x=0.7769`, `r2_y=0.9438`, `r2_z=-1.8841`

Source: [training/runs_comparison/heatmaps/summary_heatmap.json](/home/user/sensor_training/training/runs_comparison/heatmaps/summary_heatmap.json)

해석:

- `X` 오차가 `Y`보다 약 `2.7x` 크다.
- heatmap grid 기반 평가에서는 `z`도 매우 불안정하며, `r2_z`가 음수다.
- 위치별 평균 관점에서도 current run은 `X` localization이 구조적으로 무너져 있다.

### 4. Depth-bin Pattern

현재 `multi_head_field`는 shallow contact에서 특히 약하다.

- `0.8~1.1 mm`: `xy_mae≈1.02`, `success<=1cell≈0.40`
- `1.1~1.4 mm`: `xy_mae≈0.58`, `success<=1cell≈0.63`
- `1.4~1.7 mm`: `xy_mae≈0.50`, `success<=1cell≈0.70`

Source: [training/runs_comparison/comparison_results.json](/home/user/sensor_training/training/runs_comparison/comparison_results.json)

해석:

- 얕은 접촉일수록 heatmap localization이 약하다.
- 현재 설계는 낮은 depth 구간의 spatial ambiguity를 잘 풀지 못하고 있다.

### 5. Z/Fz Regressor Stability

현재 `z_fz_regressor`는 상대적으로 안정적이며, `predicted_xy` 입력 때문에 크게 붕괴하지는 않는다.

- `predicted_xy` mean MAE: `z=0.1356`, `fz=0.3560`
- fold별 `predicted_xy - gt_xy` delta:
  - fold0: `z=-0.0000`, `fz=+0.0003`
  - fold1: `z=-0.0011`, `fz=+0.0036`
  - fold2: `z=-0.0007`, `fz=+0.0029`
  - fold3: `z=+0.0004`, `fz=+0.0085`
  - fold4: `z=-0.0009`, `fz=-0.0050`

Source: [training/runs_z_fz/cv_summary_z_fz_regressor.json](/home/user/sensor_training/training/runs_z_fz/cv_summary_z_fz_regressor.json)

해석:

- 현재 평가 기준 안에서는 `z_fz_regressor`는 XY checkpoint 오차를 거의 증폭시키지 않는다.
- 따라서 전체 성능 악화의 주병목은 downstream regressor가 아니라 upstream `multi_head_field`의 XY localization이다.

## Code-Level Root Cause Analysis

### 1. XY target definition changed from continuous regression to heatmap decode

현재 `xy`는 연속 좌표 직접 회귀가 아니라 `40x40` heatmap 복원 후 decode 방식이다.

- grid resolution is fixed by `GRID_STEP = 0.5`, `GRID_MIN = -9.75`: [train_comparison.py](/home/user/sensor_training/training/pipelines/train_comparison.py:41)
- point/soft heatmap target build: [train_comparison.py](/home/user/sensor_training/training/pipelines/train_comparison.py:320), [train_comparison.py](/home/user/sensor_training/training/pipelines/train_comparison.py:343)
- decode via `softargmax` / `argmax_refine`: [train_comparison.py](/home/user/sensor_training/training/pipelines/train_comparison.py:231)

반면 이전 baseline 모델은 직접 연속 출력을 전제로 했다.

- `CNNLSTMSR` outputs `out_dim=4` directly: [cnnlstm_sr.py](/home/user/sensor_training/training/models/cnnlstm_sr.py:9)
- `SATSModel` outputs direct `[x, y, z, Fz]`: [sats_model.py](/home/user/sensor_training/training/models/sats_model.py:18)

영향:

- 문제 정의 자체가 바뀌었기 때문에 이전의 sub-mm 직접 회귀 강점을 버리고, grid decode 오차와 soft target 설계 품질에 성능이 더 민감해졌다.

### 2. Sequence semantics changed from temporal hysteresis modeling to depth-sorted windows

현재 `ZarrSequenceDataset`는 실제 시간축을 유지하지 않고 `(trial_id, x_mm, y_mm)`별 row를 모은 뒤 depth 오름차순으로 정렬해 윈도우를 만든다.

- current sequence build: [runtime_common.py](/home/user/sensor_training/training/pipelines/runtime_common.py:227), [runtime_common.py](/home/user/sensor_training/training/pipelines/runtime_common.py:251)

이 설계는 LSTM이 “시간 히스테리시스”를 학습하는 대신 사실상 “같은 위치의 depth progression”을 본다는 뜻이다.

이와 동시에 `train_comparison.py` 기본 phase는 `all`이다.

- default `--phase all`: [train_comparison.py](/home/user/sensor_training/training/pipelines/train_comparison.py:1112)

반면 현재 README와 과거 문서는 hysteresis semantics 보존을 위해 loading-only, stationary filtering을 강조한다.

- loading-only restriction for z/fz: [train_z_fz_regressor.py](/home/user/sensor_training/training/pipelines/train_z_fz_regressor.py:367)
- stationary filtering / hysteresis removal rationale: [sensor_learning_report_final_20260406.md](</home/user/sensor_training/training/training/접촉점 기준 학습_0409/md/sensor_learning_report_final_20260406.md:32>)

영향:

- 현재 recurrent backbone은 설계 의도와 다르게 mixed branch signal을 보며, 특히 `all` phase에서 loading/unloading이 함께 들어가면 ambiguity가 더 커진다.

### 3. Radius is removed from the current XY model input

현재 `MultiHeadFieldModel`은 `grid_seq`만 입력받고 radius를 쓰지 않는다.

- current model input: [multi_head_field_model.py](/home/user/sensor_training/training/models/multi_head_field_model.py:10), [multi_head_field_model.py](/home/user/sensor_training/training/models/multi_head_field_model.py:35)

하지만 현재 CV는 `d5`와 `d10` trial을 함께 학습한다.

- mixed-diameter folds: [cv_manifest_comparison.json](/home/user/sensor_training/training/runs_comparison/cv_manifest_comparison.json)

반면 이전 `CNNLSTMSR`는 radius를 입력에 concat한다.

- previous radius-conditioned path: [cnnlstm_sr.py](/home/user/sensor_training/training/models/cnnlstm_sr.py:17), [cnnlstm_sr.py](/home/user/sensor_training/training/models/cnnlstm_sr.py:24)

영향:

- 현재 label 생성에는 radius가 들어가는데 모델 입력에는 빠져 있다.
- mixed-diameter에서 동일 tactile pattern이 다른 contact geometry를 가지는 경우 모델이 이를 명시적으로 분리할 수 없다.

### 4. Stage3 multitask improves z/fz but steals capacity from xy

현재 `multi_head_field`는 backbone 하나 뒤에 scalar `[z, fz]` head와 field head를 동시에 둔다.

- current architecture: [multi_head_field_model.py](/home/user/sensor_training/training/models/multi_head_field_model.py:17)
- current loss combination: [train_comparison.py](/home/user/sensor_training/training/pipelines/train_comparison.py:161)
- stage3 train path: [train_comparison.py](/home/user/sensor_training/training/pipelines/train_comparison.py:907)

정량 결과는 다음을 보여준다.

- Stage2 -> Stage3에서 `z/fz`는 개선
- Stage2 -> Stage3에서 `x/y`는 악화

영향:

- 현재 lambda 설정과 shared feature 구조는 `xy` 정확도보다 `z/fz` 안정화에 더 유리한 방향으로 작동하고 있다.

### 5. Z/Fz regressor conditioning differs between train and eval

현재 `z_fz_regressor`는 학습 시 GT `xy + radius`를 사용한다.

- train path with GT xy: [train_z_fz_regressor.py](/home/user/sensor_training/training/pipelines/train_z_fz_regressor.py:287)

평가 시에만 optional frozen XY checkpoint를 넣어 `predicted_xy`를 계산한다.

- conditional evaluation path: [train_z_fz_regressor.py](/home/user/sensor_training/training/pipelines/train_z_fz_regressor.py:152), [train_z_fz_regressor.py](/home/user/sensor_training/training/pipelines/train_z_fz_regressor.py:299)

README도 이를 명시한다.

- `predicted_xy` is frozen XY decode + GT radius: [README.md](/home/user/sensor_training/training/README.md)

영향:

- train/eval conditioning mismatch가 존재한다.
- 다만 현재 수치상 이 mismatch는 전체 악화의 주원인은 아니다.

### 6. Phase policy is inconsistent between upstream and downstream

- `xy` model: `phase=all` default in current comparison pipeline
- `z_fz_regressor`: `phase=loading` only

Source:

- [train_comparison.py](/home/user/sensor_training/training/pipelines/train_comparison.py:1112)
- [train_z_fz_regressor.py](/home/user/sensor_training/training/pipelines/train_z_fz_regressor.py:367)

영향:

- 현재 체인은 `all-phase XY` checkpoint와 `loading-only z/fz` regressor를 연결한다.
- 이는 과거 stationary / hysteresis-controlled 설계와 일관되지 않다.

## Historical Fz Context

이전 JSON baseline에는 `fz`가 없기 때문에, `fz`는 과거 문서와 현재 값만 제한적으로 비교할 수 있다.

- 과거 문서상 FF 모델 정밀도는 `Fz MAE: 0.602 N`: [sensor_learning_report_final_20260406.md](</home/user/sensor_training/training/training/접촉점 기준 학습_0409/md/sensor_learning_report_final_20260406.md:60>)
- 현재 `multi_head_field` Stage3 aggregate `mae_fz`는 heatmap 평가 기준 `0.5261`: [training/runs_comparison/heatmaps/summary_heatmap.json](/home/user/sensor_training/training/runs_comparison/heatmaps/summary_heatmap.json)
- 현재 separate `z_fz_regressor` `predicted_xy` mean `fz mae`는 `0.3560`: [training/runs_z_fz/cv_summary_z_fz_regressor.json](/home/user/sensor_training/training/runs_z_fz/cv_summary_z_fz_regressor.json)

주의:

- 이 값들은 동일 split, 동일 target definition, 동일 preprocessing context가 아니므로 공식 승패 비교로 쓰면 안 된다.
- 보고서상에서는 “현재 `fz` 개선 신호는 일부 있으나, old JSON baseline 부재로 직접 비교는 제한적”으로 해석하는 것이 맞다.

## Final Conclusion

현재 학습이 이전보다 나빠진 이유는 다음 순서로 정리할 수 있다.

1. 가장 큰 문제는 `multi_head_field`의 `XY`, 특히 `X` 축 localization 실패다.
2. 이 실패는 shallow depth 구간에서 더 심하다.
3. 직접 회귀에서 heatmap decode로 문제 정의가 바뀌면서 `xy`가 구조적으로 더 어려워졌다.
4. depth-sorted sequence + `phase=all` 조합이 LSTM backbone의 temporal 의미를 약화시켰다.
5. mixed diameter 학습인데 radius를 입력에서 제거해 geometry ambiguity가 커졌다.
6. Stage3 multitask는 `z/fz`를 개선했지만 `xy`는 더 나빠졌다.
7. `z_fz_regressor`는 현재 기준 안에서는 비교적 안정적이므로, 전체 regression의 주병목은 downstream이 아니라 upstream XY 모델이다.

한 줄 요약:

> 현재 파이프라인은 `z/fz`를 개선하려고 분리/다단계화했지만, 정작 전체 성능을 좌우하는 upstream `XY` localization을 이전 `SATS/CNNLSTM`보다 크게 악화시켜 최종 성능이 나빠졌다.

## Recommended Next Validation Targets

이번 작업은 분석 보고서 작성이 목적이므로 구현은 하지 않았지만, 다음 실험 우선순위는 명확하다.

1. `multi_head_field`에 radius 입력을 추가한 `xy` 재실험
2. `train_comparison.py`를 `phase=loading`으로 고정한 비교 실험
3. depth-sorted sequence 대신 true temporal ordering 또는 non-recurrent baseline 재비교
4. Stage2와 Stage3를 분리해 `xy` checkpoint 선택 기준을 `xy` 중심으로 유지하는 실험
5. old/new 공통 split로 `SATS`, `CNNLSTM`, `multi_head_field`를 재평가하는 controlled benchmark
