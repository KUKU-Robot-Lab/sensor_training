# UnifiedSensorModel

## 모델 개요
CNN, LSTM, 그리고 Isoline MLP 브랜치를 결합한 최종 통합 아키텍처입니다. 물리적 보정 정보와 시공간 특징을 모두 활용하여 고해상도 위치 및 다축 힘을 정밀하게 추정합니다.

## 네트워크 구조
1. **CNN-LSTM Backbone**:
   - **CNN Encoder**: 4x4 그리드에서 공간적 패턴 추출.
   - **LSTM**: 시계열 특징을 통합하여 동적 특성(히스테리시스 등) 반영.

2. **Isoline Branch (Physics-informed)**:
   - 센서 드리프트, 온도, 메타 정보 등을 포함한 19차원 특징을 별도의 MLP로 처리.
   - 물리적인 오차 요인을 보정하는 보조 특징을 생성합니다.

3. **Multi-Head Output Branches**:
   - **Branch 1 (Position & Field)**: 접촉 위치(xyz)와 25x25 압력 분포 맵(fz_map)을 추정.
   - **Branch 2 (6-DoF Force)**: 중심점에서의 3축 위치 및 3축 힘 벡터(Fx, Fy, Fz)를 추정.

## 입력 및 출력
- **입력**:
  - `x_grid`: (B, T, 1, 4, 4) - 센서 그리드 시퀀스.
  - `x_iso`: (B, T, 19) - 보정용 메타 특징 시퀀스.
- **출력**:
  - `res1`: {xyz, fz_map}.
  - `res2`: [x, y, z, Fx, Fy, Fz].

## 학습 방식
- 공간(CNN), 시간(LSTM), 물리적 보정(Isoline MLP) 정보를 모두 융합하여 학습합니다.
- 여러 태스크를 동시에 학습(Multi-Task Learning)함으로써 단일 모델보다 더 강건하고 일관된 촉각 추정 성능을 제공합니다.
