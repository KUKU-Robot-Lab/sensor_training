import numpy as np
import pytest

from robot_skin.sim import TaxelDomainRandomizer
from robot_skin.sim.touch_grid_env import TouchGridEnv


def test_params_within_ranges_and_reproducible():
    a = TaxelDomainRandomizer(seed=3).sample(100)
    b = TaxelDomainRandomizer(seed=3).sample(100)
    np.testing.assert_array_equal(a.gain, b.gain)
    assert a.gain.min() >= 0.7 and a.gain.max() <= 1.3
    assert a.sat.min() >= 40.0 and a.sat.max() <= 90.0
    assert not a.dropout.any()


def test_apply_gain_clip_and_saturation():
    r = TaxelDomainRandomizer(noise_pct_std=0.0, offset_pct_range=(0.0, 0.0), seed=0)
    p = r.sample(4)
    clean = np.array([[-5.0, -5.0, -500.0, 0.0]])
    obs, sat = r.apply(clean, p)
    np.testing.assert_allclose(obs[0, :2], p.gain[:2] * -5.0, rtol=1e-6)
    assert obs[0, 2] == pytest.approx(-p.sat[2]) and sat[0, 2]
    assert not sat[0, [0, 1, 3]].any()


def test_dropout_reads_minus_100_and_is_saturated():
    r = TaxelDomainRandomizer(dropout_prob=1.0, seed=0)
    obs, sat = r.apply(np.zeros((3, 5)), r.sample(5))
    assert np.all(obs == -100.0) and sat.all()


def test_env_stub():
    with pytest.raises(NotImplementedError):
        TouchGridEnv()
