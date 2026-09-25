"""밴딩 데이터 사양/윈도잉 테스트 (합성)."""
import numpy as np
import pytest

from sats.bending.dataset import BendingTrial, load_bending_trial, make_windows


def test_trial_validates_shape():
    with pytest.raises(ValueError):
        BendingTrial(sensor=np.zeros((5, 8), np.float32), bend_deg=np.zeros(5, np.float32))


def test_make_windows_signed_labels():
    n, w = 25, 10
    sensor = np.random.randn(n, 16).astype(np.float32)
    deg = np.linspace(-40, 40, n).astype(np.float32)   # 양·음 모두
    tr = BendingTrial(sensor=sensor, bend_deg=deg)
    windows, dl = make_windows(tr, window_size=w)
    assert windows.shape == (n - w + 1, w, 16)
    assert dl.shape == (n - w + 1,)
    # 라벨 = 윈도우 마지막 시점 deg, 부호 보존
    assert np.isclose(dl[0], deg[w - 1])
    assert (dl < 0).any() and (dl > 0).any()


def test_load_roundtrip(tmp_path):
    p = tmp_path / "trial.npz"
    np.savez(p, sensor=np.zeros((12, 16), np.float32), bend_deg=np.zeros(12, np.float32))
    tr = load_bending_trial(p)
    assert tr.sensor.shape == (12, 16) and tr.contact is None


def test_load_missing_key_raises(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, sensor=np.zeros((12, 16), np.float32))   # bend_deg 누락
    with pytest.raises(KeyError):
        load_bending_trial(p)
