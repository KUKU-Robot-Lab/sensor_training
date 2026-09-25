# MLPFF (Force Field) Model

## 모델 개요
특정 위치에서의 힘의 분포(Force Field)를 예측하기 위한 MLP 모델입니다. 접촉 위치 정보가 주어졌을 때 해당 지점의 수직 항력(Fz)을 정교하게 추정합니다.

## 네트워크 구조
1. **Input Concatenation**:
   - 16채널 센서값 + 3축 접촉 위치(x, y, z) + 인덴터 반지름 = 20차원.

2. **Hidden Layers**:
   - `Linear` -> `BatchNorm1d` -> `ReLU` -> `Dropout`.
   - 은닉층 구성: [256, 256, 128, 64].
   - 드롭아웃(Dropout)을 통해 과적합을 방지합니다.

3. **Output Layer**:
   - `Linear(64, 1)`: 특정 지점의 Fz 값 추정.

## 입력 및 출력
- **입력**:
  - `tactile`: (B, 16) 센서 신호.
  - `radius`: (B, 1) 반지름.
  - `sr_pos`: (B, 3) Super-Resolution으로 추정된 [x, y, z] 좌표.
- **출력**: `(B, 1)` 해당 위치의 Fz 값.

## 학습 방식
- 위치 정보가 모델의 입력으로 포함되어, 센서 신호와 위치 간의 비선형적 관계를 학습합니다.
- 고해상도 힘 분포 맵(Heatmap)을 생성하기 위한 기본 단위 예측기로 사용될 수 있습니다.
