# MultiHeadFieldModel

## 모델 개요
접촉점의 6자유도(Position + Force Vector) 정보와 연속적인 압력 분포 필드(Heatmap)를 동시에 학습하는 최종 형태의 통합 모델입니다.

## 네트워크 구조
1. **Backbone (Spatial-Temporal)**:
   - **CNN**: 4x4 그리드 시퀀스의 각 프레임에서 공간 특징 추출.
   - **LSTM**: 추출된 공간 특징의 시간적 흐름 학습 (Hidden Dim: 128).

2. **Head 1: Force Vector Head**:
   - `Linear(128, 128)` -> `Linear(128, 6)`.
   - 위치(x, y, z)와 힘 벡터(Fx, Fy, Fz)를 추정.

3. **Head 2: Force Field Head**:
   - `Linear(128, 256)` -> `Linear(256, 625)`.
   - 25x25 해상도의 연속적인 힘 분포 필드(Heatmap)를 생성.
   - `Sigmoid` 활성화를 통해 상대적 압력 강도 출력.

## 입력 및 출력
- **입력**: (B, T, 1, 4, 4) - 4x4 그리드 시퀀스.
- **출력**:
  - `force_vec`: (B, 6) [x, y, z, Fx, Fy, Fz].
  - `field_map`: (B, 1, 25, 25) 압력 분포 맵.

## 학습 방식
- Multi-Task Learning 기법을 사용합니다.
- 벡터 추정과 필드 생성을 동시에 수행함으로써, 모델이 촉각 현상의 물리적 일관성을 더 잘 이해하도록 유도합니다.
