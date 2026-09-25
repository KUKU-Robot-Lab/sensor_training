# MainModel (Phase 2)

## 모델 개요
Phase 2의 주력 모델로, 1D CNN 인코더와 FiLM(Feature-wise Linear Modulation) 컨디셔닝, 그리고 경량 UNet 디코더를 결합하여 시계열 데이터로부터 고해상도 맵을 생성합니다.

## 네트워크 구조
1. **1D CNN Encoder**:
   - 시간에 따른 깊이 변화 시퀀스 (B, K, 14)를 입력받아 시간적 흐름이 반영된 256차원 특징 추출.

2. **FiLM Conditioning**:
   - 보조 정보(Fx, Fy, z, radius)로부터 아핀 변환 파라미터(γ, β)를 추출.
   - `out = γ * latent + β` 연산을 통해 물리적 조건에 맞춰 인코더 특징을 변조(Modulation).

3. **UNet-Lite Decoder**:
   - 변조된 특징을 (B, 16, 4, 4)로 Reshape.
   - 4단계의 Upsampling (`ConvTranspose2d`)을 거쳐 64x64 해상도로 복원.

## 입력 및 출력
- **입력**:
  - `tactile`: (B, K, 14) 시퀀스 데이터.
  - `aux`: (B, 4) 보조 물리 정보.
- **출력**: `hr_map`: (B, 1, 64, 64) 고해상도 촉각 이미지.

## 학습 방식
- FiLM 레이어를 통해 단순 결합(Concatenation)보다 더 정교하게 물리적 제약 조건을 주입합니다.
- 시계열 인코더를 통해 정적인 프레임 이상의 문맥(Context)을 이해하여 맵을 생성합니다.
