# Sensor Training Reconstruction Plan

작성일: 2026-04-15

## 1. 배경과 문제 요약

현재 학습 프레임워크의 핵심 병목은 downstream `z/fz` 회귀가 아니라 upstream `XY` localization 성능 저하다.  
`md/260415_current_vs_previous_training_analysis.md` 기준으로 현재 `multi_head_field` Stage3는 이전 `SATS` 대비 `x/y/z` 전 축에서 열세이며, 특히 `x` 축 오차가 구조적으로 크다.

현재 구조의 핵심 문제는 다음과 같다.

1. `xy` 타깃이 direct regression에서 `40x40 heatmap decode` 문제로 바뀌면서 기존 sub-mm direct regression 강점을 잃었다.
2. mixed diameter(`d5`, `d10`)를 함께 학습하면서도 현재 `XY` 모델 입력에서 radius conditioning이 빠져 있다.
3. `XY`와 `z/fz`를 shared backbone multitask로 같이 학습하면서 Stage3에서 `z/fz`는 좋아졌지만 `xy`는 다시 나빠졌다.
4. 시퀀스 의미가 실제 시간축이 아니라 `(trial, x, y)` 기준 depth progression으로 바뀌어 hysteresis 제어 의도와 어긋난다.
5. 현재 평가는 `eval-split all`, `predicted_xy + GT radius` 같은 upper-bound 성격이 섞여 있어 실제 배포 조건을 제대로 반영하지 못한다.

정리하면, 현재 프레임워크는 목표 출력 관점에서 구조를 단순화한 것이 아니라 오히려 `XY` 추정, diameter 일반화, 평가 일관성을 동시에 악화시켰다.

## 2. 목표 출력 정의

재구성 이후 최종적으로 만들고 싶은 센서 출력은 다음 3개다.

### 2.1 출력 A: XY 접촉 중심과 접촉 면적

- 입력: `d5/d10`, `s1~s16`
- 출력:
  - `x_mm`
  - `y_mm`
  - `effective_contact_area_mm2` 또는 이에 대응하는 `footprint`

의도:

- `d5`, `d10`처럼 인덴터 크기가 다르면 같은 센서 반응이라도 다른 접촉면적이 형성될 수 있다.
- 따라서 diameter/radius는 숨은 조건이 아니라 명시 입력으로 사용해야 한다.
- `XY`는 단순 접촉점 하나가 아니라 “접촉 중심 + 면적”으로 정의해야 이후 `z`와 `fz`를 안정적으로 연결할 수 있다.

### 2.2 출력 B: 접촉영역 조건에서의 Z 압입

- 입력:
  - `s1~s16`
  - Stage1에서 추정한 `x_mm`, `y_mm`
  - Stage1에서 추정한 `effective_contact_area_mm2` 또는 footprint 요약값
  - `d5/d10` 또는 radius
- 출력:
  - `z_contact_mm`

의도:

- `z`는 단독 scalar보다 “현재 어떤 접촉면적에서의 압입인가”로 해석해야 한다.
- 즉 `z`는 geometry-conditioned regression으로 다뤄야 한다.

### 2.3 출력 C: 같은 조건에서의 Normal Force

- 입력: Stage2와 동일
- 출력:
  - `fz_N`

의도:

- `fz` 역시 단순 센서 신호만으로 직접 추정하기보다, 접촉 위치/면적/압입 조건을 함께 보고 추정해야 한다.

### 2.4 파생값

다음 값은 모델 직접 출력이 아니라 후처리 파생값으로 정의한다.

- `contact_volume_mm3`
- `geometry_consistency`

즉 모델은 먼저 `xy center`, `contact area`, `z_contact`, `fz`를 예측하고, 이후 Hertz/geometry 식으로 부피나 일관성 지표를 계산한다.

## 3. 재구성 아키텍처

### 3.1 유지할 원칙

기존 구조에서 유지할 것은 다음이다.

1. `XY`가 전체 성능의 선행 병목이라는 관점
2. 레거시의 direct regression 성격
3. diameter/radius를 입력에 명시적으로 넣는 conditioning
4. `loading-only` 중심의 hysteresis 제어

### 3.2 제거할 구조

다음은 새 기본 구조에서 제거한다.

1. `heatmap-first`를 기본 학습 surface로 두는 방식
2. `XY`와 `z/fz`를 shared backbone multitask로 동시에 최적화하는 구조
3. `(trial, x, y)` 그룹을 depth 오름차순으로 잘라 pseudo-time sequence를 만드는 방식
4. `predicted_xy + GT radius` 같은 부분 end-to-end 평가를 메인 지표로 쓰는 방식
5. calibration으로 구조적 `X` 문제를 가리는 방식

### 3.3 Stage 1: XY + Contact Area 전용 모델

목적:

- `s1~s16`와 diameter/radius를 이용해 `x_mm`, `y_mm`, `effective_contact_area_mm2`를 추정

입력:

- `s1~s16`
- `diameter_mm` 또는 `radius_mm`

출력:

- `x_mm`
- `y_mm`
- `effective_contact_area_mm2` 또는 footprint summary

원칙:

- baseline은 direct regression으로 복원한다.
- depth-aware heatmap label은 비교군으로만 남긴다.
- mixed diameter 학습에서는 radius conditioning을 반드시 포함한다.
- 필요하면 footprint map을 보조 출력으로 둘 수 있지만, 메인 산출물은 `center + area`다.

### 3.4 Stage 2: Geometry-Conditioned Z/Fz 모델

목적:

- Stage1 출력과 센서 신호를 바탕으로 `z_contact_mm`, `fz_N`를 추정

입력:

- `s1~s16`
- `x_mm`
- `y_mm`
- `effective_contact_area_mm2` 또는 footprint summary
- `diameter_mm` 또는 `radius_mm`

출력:

- `z_contact_mm`
- `fz_N`

원칙:

- 학습과 평가의 conditioning을 일치시킨다.
- `GT xy + GT radius`는 upper-bound 참고값으로만 남기고, checkpoint selection은 실제 deployment condition 기준으로 한다.
- `loading-only`를 기본 phase로 고정한다.

### 3.5 Stage 3: 물리 일관성 후처리

목적:

- Stage1/2 출력으로부터 접촉 부피 및 물리 consistency 값을 계산

예시:

- `contact_volume_mm3 = f(area, z_contact)`
- `geometry_consistency = g(radius, area, z_contact, fz)`

원칙:

- 물리식을 label 생성의 숨은 전제로 쓰기보다, 예측 후 검증/파생 단계로 둔다.

## 4. 데이터, 라벨, 시퀀스 정책

### 4.1 타깃 정책

- `XY`는 direct coordinate regression을 주 baseline으로 둔다.
- `contact area`는 독립 supervision target으로 명시한다.
- depth-aware heatmap label은 ablation 비교군으로만 유지한다.
- `z_contact_mm`, `fz_N`는 Stage2 전용 타깃으로 둔다.

### 4.2 입력 정책

- diameter/radius는 label 생성에만 쓰지 않고 모델 입력에도 반드시 포함한다.
- mixed diameter(`d5`, `d10`)를 쓸 경우 diameter별 성능을 분리 집계한다.

### 4.3 시퀀스 정책

- `chronological`, `depth progression`, `single frame` 중 하나로 고정해야 한다.
- 현재 재구성안의 기본값은 `loading-only chronological` 또는 `loading-only single-frame baseline`이다.
- upstream `XY`와 downstream `z/fz`가 서로 다른 phase semantics를 가지지 않도록 통일한다.

### 4.4 Split 정책

- `train/val/test`를 모두 명시 저장한다.
- 메인 metric은 반드시 `val/test`로만 계산한다.
- `eval-split all`은 exploratory 결과로만 저장한다.
- CV는 diameter와 depth regime를 함께 stratify한다.

## 5. 평가 및 성공 기준

### 5.1 XY 게이트

최소 목표:

- `x MAE <= 0.50 mm`
- `y MAE <= 0.19 mm`

이는 이전 `SATS` 수준 회복을 의미한다.

추가 보고:

- `raw metric`
- `calibrated metric`

단, calibrated metric은 참고값이며 raw recovery 없이 합격으로 보지 않는다.

### 5.2 Contact Area 게이트

다음 지표를 필수 저장한다.

- `area MAE`
- `relative area error`
- `IoU` 또는 `Dice` (footprint map을 쓰는 경우)

### 5.3 Z/Fz 게이트

다음 3조건을 모두 별도 저장한다.

1. `GT xy + GT radius`
2. `pred xy + GT radius`
3. `pred xy + pred/provided diameter` 또는 실제 deployment condition

원칙:

- checkpoint selection은 3번, 즉 실제 배포 조건과 동일한 metric 기준으로 한다.
- 1번은 upper-bound reference로만 유지한다.

### 5.4 Regime별 게이트

다음을 별도 보고한다.

- shallow / mid / deep depth bin
- `d5` / `d10`
- `X/Y` 비대칭
- diameter generalization gap

특히 shallow 구간에서의 `xy_mae`와 `success threshold`는 필수 게이트로 둔다.

## 6. 구현 순서

### Step 1. Baseline 복원

- 레거시 direct regression baseline을 다시 공식 기준으로 복원
- `SATS`, `CNNLSTM` 스타일 입력/출력과 현재 데이터셋 split을 맞춰 재측정

### Step 2. Stage1 재구성

- `XY + area` 전용 모델 구현
- `radius-in` / `radius-out` ablation 수행
- `direct regression` / `depth-aware label` 비교

### Step 3. Stage2 재구성

- `z_contact + fz` 전용 geometry-conditioned regressor 구현
- `GT condition`, `predicted condition`, `deployment condition` 3종 평가 구성

### Step 4. Evaluation 정비

- `train/val/test` 완전 분리
- `all` split 결과는 exploratory로만 별도 저장
- raw/calibrated metric 분리 저장
- diameter/depth regime별 리포트 추가

### Step 5. 최종 acceptance 표 확정

- `xy`
- `area`
- `z`
- `fz`
- diameter generalization
- shallow/deep robustness

이 6개 축을 한 표에 정리해 최종 모델 선택 기준으로 사용한다.

## 7. 실행 체크리스트

### 7.1 문제 고정

- [x] 현재 `runs_comparison`, `runs_z_fz`, 레거시 `0409` 결과를 공식 baseline 표로 정리
- [x] `eval-split all` 결과는 exploratory로 재분류
- [x] `predicted_xy + GT radius`를 메인 성능 표에서 분리

### 7.2 데이터/평가 정비

- [x] `train/val/test` split 정책 확정 및 held-out test trial 지정
- [x] diameter와 depth regime를 함께 고려한 stratified CV 설계
- [x] shallow / deep 구간 샘플 수 최소 기준 정의
- [x] raw metric / calibrated metric 분리 저장

### 7.3 Stage1

- [ ] `XY + area` target schema 확정
- [ ] Stage1 입력에 `diameter/radius` 명시 포함
- [ ] legacy direct regression baseline 재현
- [ ] `radius-in` vs `radius-out` 비교
- [ ] `direct regression` vs `depth-aware label` 비교
- [ ] `d5 only`, `d10 only`, mixed diameter 비교

### 7.4 Stage2

- [ ] `z_contact + fz` 전용 입력 schema 확정
- [ ] `GT condition`, `predicted condition`, `deployment condition` 3종 평가 구현
- [ ] checkpoint selection metric을 deployment condition 기준으로 변경
- [ ] `loading-only` phase를 기본값으로 고정

### 7.5 진단 리포트

- [ ] `X/Y` 비대칭 진단 추가
- [ ] depth bin별 성능 표 추가
- [ ] diameter별 성능 표 추가
- [ ] area 예측 품질(`MAE`, `relative error`, `IoU/Dice`) 추가

### 7.6 최종 acceptance

- [ ] `x MAE <= 0.50 mm`
- [ ] `y MAE <= 0.19 mm`
- [ ] `area` metric 기준 충족
- [ ] `z/fz`가 deployment condition 기준으로 안정적
- [ ] shallow/deep, `d5/d10` 모두에서 허용 범위 만족

## 8. 최종 결정 사항

이번 재구성에서 기본 설계는 다음으로 고정한다.

1. `XY/면적`과 `Z/Fz`는 분리 학습한다.
2. diameter/radius는 항상 명시 입력으로 넣는다.
3. `loading-only`를 기본 phase로 둔다.
4. `contact volume`은 직접 예측하지 않고 후처리 파생값으로 둔다.
5. `all split`, `GT-conditioned pseudo end-to-end`, calibration-only 개선은 메인 선택 기준으로 쓰지 않는다.

이 문서는 이후 코드/실험 재구성의 기준 문서로 사용한다.

## 9. 진행 로그

### 2026-04-15 업데이트

진행 단위:

- `7.1 문제 고정` 완료

적용 내용:

- `runs_comparison`, `runs_z_fz`, 레거시 `0409` 결과를 한 표로 묶는 baseline 정리 유틸을 추가했다.
- `eval-split all` heatmap 결과는 공식 baseline 표에서 제외하고 exploratory로 재분류했다.
- `predicted_xy + GT radius`는 pseudo end-to-end upper-bound로 분리했고, `gt_xy + gt_radius` 역시 deployment condition이 아니므로 reference-only로 명시했다.
- 생성 보고서: [260415_reconstruction_baseline_report.md](/home/user/sensor_training/md/260415_reconstruction_baseline_report.md)

수정 파일:

- [reconstruction_baseline.py](/home/user/sensor_training/training/utils/reconstruction_baseline.py)
- [generate_reconstruction_baseline_report.py](/home/user/sensor_training/scripts/generate_reconstruction_baseline_report.py)
- [test_reconstruction_baseline.py](/home/user/sensor_training/tests/test_reconstruction_baseline.py)

검증:

- `pytest -q tests/test_reconstruction_baseline.py`

### 2026-04-15 업데이트 (7.2)

진행 단위:

- `7.2 데이터/평가 정비` 완료

적용 내용:

- loading phase 기준 trial counts를 읽어 held-out test trial을 `ecomesh_d10_z1.0_test3`, `ecomesh_d5_z1.0_test3`, `ecomesh_d5_z1.5_test9`로 고정한 split policy 보고서를 추가했다.
- diameter/depth regime를 함께 고려하는 stratified CV 설계 초안을 생성해 fold별 validation candidate를 문서화했다.
- shallow/deep 최소 샘플 기준을 각각 `200,000`으로 정의했고, 현재 held-out test는 shallow `478,989`, deep `239,446` samples로 기준을 만족한다.
- [train_comparison.py](/home/user/sensor_training/training/pipelines/train_comparison.py)와 [evaluate_comparison_heatmap.py](/home/user/sensor_training/training/pipelines/evaluate_comparison_heatmap.py)에서 raw metric을 primary로 유지하고 calibrated metric을 별도 variant로 저장하도록 변경했다.
- 생성 보고서: [260415_split_policy_report.md](/home/user/sensor_training/md/260415_split_policy_report.md)

수정 파일:

- [split_policy.py](/home/user/sensor_training/training/utils/split_policy.py)
- [generate_split_policy_report.py](/home/user/sensor_training/scripts/generate_split_policy_report.py)
- [train_comparison.py](/home/user/sensor_training/training/pipelines/train_comparison.py)
- [evaluate_comparison_heatmap.py](/home/user/sensor_training/training/pipelines/evaluate_comparison_heatmap.py)
- [test_split_policy.py](/home/user/sensor_training/tests/test_split_policy.py)
- [test_multi_head_fz_metrics.py](/home/user/sensor_training/tests/test_multi_head_fz_metrics.py)

검증:

- `pytest -q tests/test_split_policy.py tests/test_reconstruction_baseline.py`
- `python3 -m py_compile training/utils/split_policy.py scripts/generate_split_policy_report.py training/pipelines/train_comparison.py training/pipelines/evaluate_comparison_heatmap.py training/utils/reconstruction_baseline.py scripts/generate_reconstruction_baseline_report.py`

참고:

- `tests/test_multi_head_fz_metrics.py` 전체 실행은 현재 환경에 `torch`가 없어 수집 단계에서 막혀 실행하지 못했다.
