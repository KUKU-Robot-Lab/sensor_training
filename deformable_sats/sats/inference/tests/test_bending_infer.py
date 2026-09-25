"""bending_infer 재앵커(기압 무관) 단위 테스트 — 실제 v6 estimator 사용.

핵심: theta_from_pct 는 학습 참조 baseline(ref)에 pct를 얹으므로,
데모 당일 기압(절대 baseline)이 달라도 동일 pct면 동일 theta 여야 한다.
(반면 theta_from_raw 는 절대 raw라 baseline shift에 민감 — 그게 버그였음.)
"""
import numpy as np
import pytest

from sats.bending.config import BendingConfig
from sats.inference.bending_infer import BendingInference, pct_to_raw

_CKPT = "sats/bending/runs/estimator_v6"


def _bi():
    try:
        return BendingInference(_CKPT, device="cpu", cfg=BendingConfig())
    except FileNotFoundError:
        pytest.skip("estimator_v6 ckpt 없음")


def test_ref_baseline_loaded():
    bi = _bi()
    assert bi.ref_baseline is not None and bi.ref_baseline.shape == (16,)


def test_theta_from_pct_invariant_to_atmospheric_shift():
    """같은 pct 패턴이면 데모 baseline이 ±2% 달라도 theta 동일(재앵커 효과)."""
    bi = _bi()
    rng = np.random.default_rng(0)
    pct = rng.normal(0, 3, size=(bi.W, 16)).astype(np.float32)   # 임의 밴딩 패턴
    t_ref = bi.theta_from_pct(pct)                                # ref 앵커(demo base 무관)
    # demo_baseline 을 줘도 ref 가 있으면 ref 를 쓰므로 동일해야 함
    for scale in (0.98, 1.0, 1.02):
        demo_base = bi.ref_baseline * scale
        assert abs(bi.theta_from_pct(pct, demo_baseline=demo_base) - t_ref) < 1e-3


def test_raw_absolute_is_baseline_sensitive():
    """대조군: theta_from_raw(절대)는 baseline 1% shift에 크게 흔들림(버그 재현)."""
    bi = _bi()
    flat_pct = np.zeros((bi.W, 16), np.float32)
    t0 = bi.theta_from_raw(pct_to_raw(flat_pct, bi.ref_baseline))
    t1 = bi.theta_from_raw(pct_to_raw(flat_pct, bi.ref_baseline * 1.01))
    assert abs(t1 - t0) > 10.0     # 절대 raw는 1% shift만으로 10°+ 변동
