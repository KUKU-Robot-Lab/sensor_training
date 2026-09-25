# TactileTransformer Model

## 모델 개요
자연어 처리에서 성능이 검증된 Transformer 구조를 촉각 데이터에 적용한 모델입니다. 센서 배열 전체의 전역적인 공간 상관관계를 Self-attention으로 모델링합니다.

## 네트워크 구조
1. **Input Projection**:
   - 1차원 센서 신호를 `d_model`(64) 차원의 임베딩으로 변환.

2. **Positional Encoding**:
   - 4x4 그리드 상의 센서 위치 정보를 주기 위해 학습 가능한 파라미터(Positional Embedding)를 더해줍니다.

3. **Transformer Encoder**:
   - 3층의 `TransformerEncoderLayer`를 사용.
   - 4개의 Head를 가진 Multi-head Attention을 통해 모든 센서 간의 전역적인 상호작용을 병렬적으로 처리합니다.

4. **Regressor Head**:
   - Transformer의 출력 시퀀스를 평탄화하여 6자유도 값을 추정.

## 입력 및 출력
- **입력**: (B, 16) - 16채널 센서 신호.
- **출력**: (B, 6) - [x, y, z, Fx, Fy, Fz].

## 학습 방식
- Attention 메커니즘을 통해 모델이 스스로 어느 센서의 조합이 추정에 중요한지를 판단하도록 학습합니다.
- CNN의 국소적인(Local) 특징 추출 한계를 넘어, 전체 센서 판의 정보를 동시에 고려한 추론을 수행합니다.
