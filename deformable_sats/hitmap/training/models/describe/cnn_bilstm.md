# CNNBiLSTM Model

## 모델 개요
CNN과 Bidirectional LSTM을 결합한 하이브리드 모델로, 촉각 센서의 공간적인 패턴과 시간적인 의존성(히스테리시스)을 동시에 학습합니다.

## 네트워크 구조
1. **CNN (Feature Extractor)**:
   - `Conv2d(1, 32, 3, padding=1)`: 1채널 4x4 입력을 32채널로 확장.
   - `BatchNorm2d(32)` + `ReLU`: 정규화 및 비선형 활성화.
   - `AdaptiveAvgPool2d(2)`: 4x4 그리드를 2x2로 축소하여 특징 추출 (128차원).
   - `Flatten`: LSTM 입력을 위해 1차원으로 변환.

2. **Bi-LSTM (Temporal Processor)**:
   - `input_size=128`, `hidden_size=hidden_dim`, `bidirectional=True`.
   - 정방향과 역방향의 시계열 정보를 모두 활용하여 센서의 반응 지연 및 복원 특성을 모델링.

3. **FC Head (Regressor)**:
   - `Linear(hidden_dim * 2, 128)` + `ReLU`.
   - `Linear(128, 6)`: 최종 출력 [x, y, z, Fx, Fy, Fz] 추정.

## 입력 및 출력
- **입력**: `(B, T, 1, 4, 4)` - 시간(T)에 따른 4x4 그리드 시퀀스.
- **출력**: `(B, 6)` - 마지막 타임스텝의 3축 위치 및 3축 힘 벡터.

## 학습 방식
- CNN으로 각 프레임의 공간 특징을 추출한 뒤, LSTM을 통해 시계열 흐름을 파악합니다.
- 마지막 타임스텝의 Hidden State를 사용하여 최종 값을 회귀 학습합니다.
