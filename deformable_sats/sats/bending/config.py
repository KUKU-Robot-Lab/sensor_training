"""밴딩 보상 프론트엔드 설정.

flat 학습 SATS를 동결한 채, 밴딩 상태 신호에서 곡률(signed deg)을 추정하고
flat 등가 baseline을 복원하는 전처리 단계의 하이퍼파라미터.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BendingConfig:
    # 입력 형식 (SATS와 동일해야 pipeline 호환)
    n_sensors: int = 16
    window_size: int = 10

    # 곡률 estimator 구조: lstm(시계열) / cnn2d(4×4 공간) / mlp_frame(프레임 기준선)
    estimator_arch: str = "lstm"

    # LSTM 인코더 (시계열/이력 → 밴딩 상태)
    lstm_hidden: int = 64
    lstm_layers: int = 2
    dropout: float = 0.1

    # MLP head (곡률 회귀 / 오프셋 복원)
    mlp_hidden: int = 64

    # 동결 SATS 체크포인트 경로 (pipeline 실사용 시). None이면 외부 주입.
    sats_run_dir: str | None = None

    # 곡률은 부호 있음(양/음 방향). 정규화 스케일(deg) — 회귀 안정화용.
    deg_scale: float = 90.0

    # restorer 오프셋 예측 방식:
    #  deg_only = 오프셋을 곡률(deg)만의 함수로(§5.3 Δp_bend≈κ·k_i z_i). 입력 비의존 →
    #            접촉을 건드릴 수 없음(접촉 보존). ★기본.
    #  seq_deg  = 입력+deg MLP(표현력↑, 무접촉 학습 시 접촉 파괴 위험 — 준-합성 검증서 실증). 레거시.
    restorer_mode: str = "deg_only"
    latent_dim: int = 2          # [latent 모드] 잠재 변형 코드 차원(정보 병목; 작을수록 접촉 보존↑)

    device: str = "cuda"
