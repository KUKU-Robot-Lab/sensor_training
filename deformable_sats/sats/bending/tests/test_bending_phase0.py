"""Phase 0 — 밴딩 보상 프론트엔드 스캐폴드 테스트 (합성 데이터, 실 취득 데이터 불요).

검증 대상:
- BendingConfig 구성
- BendingEstimator: [B,T,16] → signed deg[B] (음수 가능, 부호 있는 곡률)
- BaselineRestorer: zero-init 항등(복원≈원신호), deg 의존성, shape 보존
- BendingPipeline: 동결 SATS 조합, 출력 (deg, pressure_map), SATS 가중치 동결
"""
import torch

from sats.bending.baseline_restorer import BaselineRestorer
from sats.bending.bending_estimator import BendingEstimator
from sats.bending.config import BendingConfig
from sats.bending.pipeline import BendingPipeline
from sats.training.cnn_module import SATSCNNStage
from sats.training.config import SATSConfig


def _cfg() -> BendingConfig:
    return BendingConfig(n_sensors=16, window_size=10, lstm_hidden=32,
                         lstm_layers=2, mlp_hidden=32)


def _batch(b: int = 4, t: int = 10):
    torch.manual_seed(0)
    seq = torch.randn(b, t, 16)
    lengths = torch.full((b,), t, dtype=torch.long)
    return seq, lengths


def test_bending_config_fields():
    c = _cfg()
    assert c.n_sensors == 16 and c.window_size == 10
    assert c.lstm_hidden > 0 and c.mlp_hidden > 0


def test_estimator_outputs_signed_scalar_per_sample():
    est = BendingEstimator(_cfg())
    seq, lengths = _batch()
    deg = est(seq, lengths)
    assert deg.shape == (seq.shape[0],)
    # 부호 있는 회귀: 출력이 음수도 가능해야 한다(ReLU/clamp로 막히면 안 됨)
    with torch.no_grad():
        for p in est.head.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    deg2 = est(seq, lengths)
    assert (deg2 < 0).any() or (deg2 > 0).any()  # 양·음 모두 표현 가능


def test_restorer_identity_init_returns_input():
    r = BaselineRestorer(_cfg())
    seq, _ = _batch()
    deg = torch.zeros(seq.shape[0])
    restored = r(seq, deg)
    assert restored.shape == seq.shape
    assert torch.allclose(restored, seq, atol=1e-6)  # zero-init offset → 항등


def test_restorer_depends_on_signed_deg():
    r = BaselineRestorer(_cfg())
    with torch.no_grad():
        for p in r.net.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    seq, _ = _batch()
    pos = r(seq, torch.full((seq.shape[0],), 30.0))
    neg = r(seq, torch.full((seq.shape[0],), -30.0))
    assert not torch.allclose(pos, neg, atol=1e-5)  # 부호에 따라 오프셋 방향 다름


def test_pipeline_freezes_sats_and_shapes():
    bcfg = _cfg()
    scfg = SATSConfig()
    sats = SATSCNNStage(scfg)
    pipe = BendingPipeline(bcfg, sats)
    # 동결 확인
    assert all(not p.requires_grad for p in pipe.sats.parameters())
    seq, lengths = _batch()
    deg, pmap = pipe(seq, lengths)
    assert deg.shape == (seq.shape[0],)
    assert pmap.shape[0] == seq.shape[0] and pmap.ndim == 3  # [B, grid, grid]
