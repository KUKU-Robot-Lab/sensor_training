# MLPSRClass Model

## 모델 개요
분류(Classification)와 회귀(Regression)를 혼합하여 촉각 고해상도(SR)를 구현한 MLP 모델입니다. 위치 추정을 분류 문제로 접근하여 정확도를 높입니다.

## 네트워크 구조
1. **Shared Backbone**:
   - `Linear` -> `BatchNorm1d` -> `ReLU` 반복 (512, 512, 256, 128).
   - 모든 태스크에서 공통으로 사용되는 특징을 추출합니다.

2. **Multi-Head Structure**:
   - **X-Head**: `Linear(128, 40)` - X 좌표를 40개 구간 중 하나로 분류.
   - **Y-Head**: `Linear(128, 40)` - Y 좌표를 40개 구간 중 하나로 분류.
   - **Z-Head**: `Linear(128, 1)` - Z 깊이를 회귀 방식으로 추정.

## 입력 및 출력
- **입력**: (B, 17) - 16채널 센서 신호 + 1개 지름.
- **출력**: 
  - `x_logits`: X축 분류 점수.
  - `y_logits`: Y축 분류 점수.
  - `z_depth`: Z축 깊이 값.

## 학습 방식
- 위치(X, Y)는 Cross-Entropy Loss를 사용하여 분류 학습을 수행합니다.
- 깊이(Z)는 MSE 또는 L1 Loss를 사용하여 회귀 학습을 수행합니다.
- 위치를 불연속적인 클래스로 학습함으로써, 회귀 모델이 겪는 평균값 수렴 문제를 완화합니다.
