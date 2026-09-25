from pathlib import Path

import numpy as np
import pytest

from common.signal import (
    ADC_MAX, PRESS_SIGN, NormStats, estimate_baseline, press_intensity, relative_change,
    saturation_mask,
)

REPO = Path(__file__).resolve().parents[2]


def test_relative_change_formula_and_dtype():
    base = np.array([100.0, 200.0])
    raw = np.array([[110.0, 180.0], [100.0, 200.0]])
    d = relative_change(raw, base)
    assert d.dtype == np.float32
    np.testing.assert_allclose(d, [[10.0, -10.0], [0.0, 0.0]], atol=1e-6)


def test_relative_change_matches_sats_dataset_convention():
    """Guard against drift from deformable_sats/sats/training/dataset.py (text check, no import)."""
    src = (REPO / "deformable_sats/sats/training/dataset.py").read_text(encoding="utf-8")
    assert "((s_raw - baseline) / baseline) * 100.0" in src
    rng = np.random.default_rng(0)
    base = rng.uniform(6e6, 8e6, 16)
    s_raw = base * rng.uniform(0.9, 1.1, (5, 16))
    ref = (((s_raw - base) / base) * 100.0).astype(np.float32)
    np.testing.assert_array_equal(relative_change(s_raw, base), ref)


def test_press_is_negative_delta_and_positive_intensity():
    base = np.full(4, 7e6)
    pressed = base * 0.95  # barometric taxel: raw drops under press
    d = relative_change(pressed[None], base)
    assert PRESS_SIGN == -1.0
    assert np.all(d < 0) and np.all(press_intensity(d) > 0)


def test_relative_change_rejects_zero_baseline():
    with pytest.raises(ValueError):
        relative_change(np.ones((2, 2)), np.array([1.0, 0.0]))


def test_estimate_baseline_median_ignores_spike():
    raw = np.full((200, 3), 50.0)
    raw[5] = 1e9  # early transient
    raw[150:] = 10.0  # later contact, outside window
    t = np.arange(200) / 200.0
    np.testing.assert_allclose(estimate_baseline(raw, t=t, duration_s=0.5), 50.0)
    np.testing.assert_allclose(estimate_baseline(raw, n_samples=100), 50.0)
    with pytest.raises(ValueError):
        estimate_baseline(raw, duration_s=None)


@pytest.mark.parametrize("method", ["std", "robust", "none"])
def test_normstats_roundtrip(tmp_path, method):
    rng = np.random.default_rng(1)
    x = rng.normal(3.0, 2.0, (500, 16))
    st = NormStats.fit(x, method=method)
    np.testing.assert_allclose(st.invert(st.apply(x)), x, rtol=1e-4, atol=1e-4)
    if method != "none":
        assert abs(float(st.apply(x).mean())) < 0.1
    st.save(tmp_path / "n.json")
    st2 = NormStats.load(tmp_path / "n.json")
    np.testing.assert_array_equal(st.offset, st2.offset)
    np.testing.assert_array_equal(st.scale, st2.scale)
    assert NormStats.from_dict(st.to_dict()).scale.dtype == np.float32


def test_normstats_constant_channel_is_finite():
    st = NormStats.fit(np.ones((10, 2)))
    assert np.all(np.isfinite(st.apply(np.ones((3, 2)))))


def test_saturation_mask_rails_and_cap():
    raw = np.array([[0.0, 5e6, ADC_MAX]])
    np.testing.assert_array_equal(saturation_mask(raw), [[True, False, True]])
    d = np.array([[-95.0, 10.0, 89.9]])
    np.testing.assert_array_equal(saturation_mask(delta_pct=d), [[True, False, False]])
    np.testing.assert_array_equal(saturation_mask(raw, d), [[True, False, True]])
    assert not saturation_mask(delta_pct=d, max_abs_pct=None).any()
    with pytest.raises(ValueError):
        saturation_mask()
