import numpy as np
import pytest
import torch

from robot_skin.baseline import (
    BaselinePredictor, NoContactSession, NoContactWindowDataset, predict_session, train_baseline,
)


def _synthetic_session(T=600, N=6, D=3, seed=0):
    """No-contact ΔS = per-taxel gain × (joint-driven bend) + slow pose term."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 200.0
    q = np.stack([np.sin(2 * np.pi * (0.3 + 0.2 * d) * t + d) for d in range(D)], 1)
    qd = np.gradient(q, t, axis=0)
    pos = np.broadcast_to(rng.normal(0, 0.02, (1, N, 3)), (T, N, 3)).copy()
    nrm = np.broadcast_to(np.array([0, 0, 1.0]), (T, N, 3)).copy()
    gain = rng.uniform(1.0, 3.0, N)
    delta = gain[None] * q[:, [0]] + 0.5 * q[:, [1]] ** 2 + rng.normal(0, 0.05, (T, N))
    return NoContactSession(delta.astype(np.float32), pos, nrm, q, qd)


def test_zero_init_predicts_zero():
    m = BaselinePredictor(n_taxels=4, joint_dim=2)
    out = m(torch.randn(3, 4, 3), torch.randn(3, 4, 3), torch.randn(3, 2), torch.randn(3, 2))
    assert out.shape == (3, 4)
    assert torch.count_nonzero(out) == 0


def test_wrong_taxel_count_raises():
    m = BaselinePredictor(n_taxels=4, joint_dim=2)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 5, 3), torch.zeros(1, 5, 3), torch.zeros(1, 2), torch.zeros(1, 2))


def test_dataset_windows_and_saturation_filter():
    s = _synthetic_session(T=100)
    ds = NoContactWindowDataset([s], window=10, stride=5)
    assert len(ds) == len(range(9, 100, 5))
    item = ds[0]
    assert item["pos"].shape == (6, 3) and item["q"].shape == (3,) and item["y"].shape == (6,)
    np.testing.assert_allclose(item["y"].numpy(), s.delta_pct[0:10].mean(0), rtol=1e-5)
    sat = np.zeros_like(s.delta_pct, dtype=bool)
    sat[0:30] = True
    ds2 = NoContactWindowDataset([NoContactSession(s.delta_pct, s.pos, s.nrm, s.q, s.qd, sat)],
                                 window=10, stride=5)
    assert len(ds2) < len(ds)
    with pytest.raises(ValueError):
        NoContactSession(s.delta_pct, s.pos[:, :3], s.nrm, s.q, s.qd)


def test_training_reduces_loss_and_predicts():
    s = _synthetic_session()
    ds = NoContactWindowDataset([s], window=4, stride=2)
    m = BaselinePredictor(n_taxels=6, joint_dim=3, hidden=64)
    hist = train_baseline(m, ds, epochs=40, batch_size=64, lr=3e-3, val_frac=0.2, seed=0)
    assert hist["train"][-1] < 0.3 * hist["train"][0]
    assert len(hist["val"]) == 40
    pred = predict_session(m, s, ds.joint_norm)
    assert pred.shape == s.delta_pct.shape
    resid = s.delta_pct - pred
    assert np.abs(resid).mean() < 0.5 * np.abs(s.delta_pct).mean()
