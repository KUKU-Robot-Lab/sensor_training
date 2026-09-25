# SuperTacVAE Model

## 모델 개요
Variational Autoencoder(VAE)와 잔차 업샘플링(Residual Upsampling)을 결합한 모델로, 촉각 신호를 확률적인 잠재 공간에 투영한 후 고해상도 맵으로 복원합니다.

## 네트워크 구조
1. **Encoder**:
   - 4x4 그리드를 CNN으로 처리하여 잠재 공간의 평균(mu)과 로그 분산(logvar)을 산출.
   - `latent_dim`: 64.

2. **Reparameterization**:
   - 학습 시에는 확률적 샘플링을 수행하고, 추론 시에는 평균값을 사용하여 잠재 벡터 생성.

3. **Decoder & Residual Upsampling**:
   - 잠재 벡터를 256x4x4 특징 맵으로 확장.
   - 3단계의 `ConvTranspose2d`를 통해 4x4 -> 8x8 -> 16x16 -> 32x32로 단계적 확대.
   - 각 단계마다 배치 정규화와 ReLU를 적용하고, 마지막에 `Sigmoid`로 압력 세기를 출력.

## 입력 및 출력
- **입력**: (B, 1, 4, 4) - 저해상도 센서 그리드.
- **출력**:
  - `hr_map`: (B, 1, 32, 32) 고해상도 이미지.
  - `mu`, `logvar`: VAE 학습을 위한 잠재 변수 파라미터.

## 학습 방식
- Reconstruction Loss (MSE)와 KL-Divergence Loss를 결합하여 학습합니다.
- 잠재 공간이 물리적으로 유의미한 특징을 압축하여 담을 수 있도록 유도하며, 이를 통해 노이즈에 강한 고해상도 복원이 가능합니다.
