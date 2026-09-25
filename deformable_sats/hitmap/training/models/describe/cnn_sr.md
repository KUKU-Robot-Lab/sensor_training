# CNNSR Model

## 모델 개요
4x4 센서 그리드의 공간적 특징을 추출하여 고해상도 정보(Super-Resolution)를 추정하는 CNN 기반 모델입니다.

## 네트워크 구조
1. **CNN (Spatial Encoder)**:
   - `Conv2d(in_channels, 32, 3, padding=1)` + `BatchNorm2d(32)` + `ReLU`.
   - `Conv2d(32, 64, 3, padding=1)` + `BatchNorm2d(64)` + `ReLU`.
   - `Flatten`: (B, 64, 4, 4) -> (B, 1024).

2. **FC Head (Regressor)**:
   - 입력: CNN 특징(1024) + 인덴터 반지름(1).
   - `Linear(1025, 256)` + `ReLU`.
   - `Linear(256, 128)` + `ReLU`.
   - `Linear(128, out_dim)`: 최종 출력 추정.

## 입력 및 출력
- **입력**: 
  - `grid`: (B, 1, 4, 4) - 4x4 센서 그리드 이미지.
  - `radius`: (B, 1) - 인덴터의 반지름 정보.
- **출력**: `(B, out_dim)` (일반적으로 4: [x, y, z, Fz]).

## 학습 방식
- 센서 그리드를 이미지로 간주하여 공간적인 압력 분포 패턴을 학습합니다.
- 물리적인 조건(반지름)을 특징 벡터와 결합(Concatenate)하여 추정 정확도를 높입니다.
