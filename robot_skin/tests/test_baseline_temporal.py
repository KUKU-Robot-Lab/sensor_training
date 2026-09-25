"""baseline.temporal: TemporalBaselinePredictor, Gaussian NLL, causal offline/online inference."""
import inspect
import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from common.layouts import load_layout
from common.signal import NormStats
from robot_skin.baseline import (CausalBaselineStream, TemporalBaselinePredictor, baseline_loss, causal_windows,
                                 episode_joint_view, gaussian_nll, load_baseline_model, predict_episode, qd_settings,
                                 save_baseline_model)
from robot_skin.baseline.temporal import soft_clamp
from robot_skin.datasets.episode import D_HAND_POSE_IMU, Episode, EpisodeMeta
from robot_skin.datasets.motion import BaselineWindowDataset


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)          # tiny models: avoid thread contention on shared CI boxes
    yield
    torch.set_num_threads(n)


def _lag(x, tau, dt):
    y = np.zeros_like(x)
    a = dt / (tau + dt)
    for t in range(1, len(x)):
        y[t] = y[t - 1] + a * (x[t] - y[t - 1])
    return y


def make_episode(eid="e0", *, T=600, N=4, D=2, seed=0, freqs=(0.7, 1.3), noise=0.02):
    """Robot-like episode (every frame a measured q) whose no-contact ΔS is a lagged, taxel-specific
    function of the joint angles and velocities — the structure the temporal model must learn."""
    rng = np.random.default_rng(seed)
    hz = 200.0
    t = np.arange(T) / hz
    ph = rng.uniform(0, 2 * np.pi, D)
    q = np.stack([0.6 * np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs[:D], ph)], 1)
    qd = np.gradient(q, 1 / hz, axis=0)
    W = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, -0.7], [-0.5, 0.8]])[:N, :D]
    x = 2.0 * q @ W.T + 0.3 * qd @ W.T
    delta = np.stack([_lag(x[:, n], 0.08, 1 / hz) for n in range(N)], 1) + rng.normal(0, noise, (T, N))
    pos = np.broadcast_to(np.linspace(0, 0.05, N * 3).reshape(N, 3), (T, N, 3)).astype(np.float32)
    nrm = np.broadcast_to(np.eye(3)[[2] * N], (T, N, 3)).astype(np.float32)
    meta = EpisodeMeta(episode_id=eid, dataset="motion", kind="robot", layout="grid", n_taxels=N,
                       joint_names=[f"j{i}" for i in range(D)])
    arrays = {"t": t, "q": q.astype(np.float32), "qd": qd.astype(np.float32), "delta_pct": delta.astype(np.float32),
              "taxel_pos": pos, "taxel_nrm": nrm, "contact_label": np.zeros((T, N), np.int8),
              "saturated": np.zeros((T, N), bool)}
    return Episode(meta, arrays)


def small_model(N=4, D=2, **kw):
    kw = {"window": 16, "hidden": 16, "head_hidden": 16, "n_layers": 3, "taxel_emb_dim": 4, **kw}
    torch.manual_seed(0)
    return TemporalBaselinePredictor(N, D, **kw)


# ───────────────────────────────────────────────────────────── model

def test_shapes_and_initialisation():
    m = small_model(sigma0=0.5)
    B, W, N, D = 3, 16, 4, 2
    mean, lv = m(torch.randn(B, W, D), torch.randn(B, W, D), torch.rand(B, N, 3) * 0.05, torch.randn(B, N, 3))
    assert mean.shape == (B, N) and lv.shape == (B, N)
    assert torch.all(mean == 0)                                        # identity warm start
    assert torch.allclose(lv, torch.full_like(lv, 2 * math.log(0.5)), atol=1e-3)
    m.set_target_scale([1.0, 2.0, 0.5, 4.0])
    mean, lv = m(torch.randn(B, W, D), torch.randn(B, W, D), torch.rand(B, N, 3), torch.randn(B, N, 3))
    exp = 2 * np.log(0.5 * np.array([1.0, 2.0, 0.5, 4.0]))
    assert np.allclose(lv.detach().numpy(), exp[None], atol=1e-3)     # σ0 in target-scale units
    assert m.receptive_field == 1 + 2 * (1 + 2 + 4)
    g = small_model(arch="gru", n_layers=1)
    assert g(torch.randn(2, 16, 2), torch.randn(2, 16, 2), torch.rand(2, 4, 3), torch.rand(2, 4, 3))[0].shape == (2, 4)


def test_observed_delta_is_never_an_input():
    """sats/bending lesson: the baseline must not see ΔS (it would learn to erase contact)."""
    params = set(inspect.signature(TemporalBaselinePredictor.forward).parameters)
    assert params == {"self", "q_hist", "qd_hist", "pos", "nrm"}


def test_invalid_arguments():
    with pytest.raises(ValueError):
        TemporalBaselinePredictor(4, 2, arch="lstm")
    with pytest.raises(ValueError):
        TemporalBaselinePredictor(4, 2, use_pose=False, taxel_emb_dim=0)
    with pytest.raises(ValueError):
        TemporalBaselinePredictor(4, 2, logvar_min=1.0, logvar_max=0.0)
    m = small_model()
    with pytest.raises(ValueError):
        m(torch.randn(1, 16, 3), torch.randn(1, 16, 3), torch.rand(1, 4, 3), torch.rand(1, 4, 3))
    with pytest.raises(ValueError):
        m(torch.randn(1, 16, 2), torch.randn(1, 16, 2), torch.rand(1, 5, 3), torch.rand(1, 5, 3))
    with pytest.raises(ValueError):
        m.set_joint_stats(NormStats(np.zeros(3), np.ones(3)), NormStats(np.zeros(2), np.ones(2)))
    with pytest.raises(ValueError):
        m.set_target_scale([1.0, np.nan, 1.0, 1.0])


def test_joint_stats_normalise_inside_the_model():
    m = small_model()
    q, qd = torch.randn(2, 16, 2) * 3 + 1, torch.randn(2, 16, 2) * 5
    ref = m.encode_joints((q - 1) / 3, qd / 5)
    m.set_joint_stats(NormStats(np.ones(2, np.float32), np.full(2, 3.0, np.float32)),
                      {"offset": [0.0, 0.0], "scale": [5.0, 5.0]})
    assert torch.allclose(m.encode_joints(q, qd), ref, atol=1e-5)
    js = m.joint_stats_dict()
    assert np.allclose(js["q"].scale, 3.0) and np.allclose(js["qd"].scale, 5.0)


def test_soft_clamp():
    with torch.no_grad():
        y = soft_clamp(torch.linspace(-60, 60, 121), -10.0, 5.0)
    assert float(y.min()) >= -10.0 - 1e-6 and float(y.max()) <= 5.0 + 1e-6
    assert abs(float(soft_clamp(torch.tensor(-2.5), -10.0, 5.0)) + 2.5) < 1e-2    # ≈ identity inside
    x = torch.tensor([-12.0, -10.0, 5.0, 7.0], requires_grad=True)                 # at / just past the bounds
    soft_clamp(x, -10.0, 5.0).sum().backward()
    assert torch.all(x.grad > 0.05)                                                # no dead gradient there


# ───────────────────────────────────────────────────────────── losses

def test_gaussian_nll_matches_formula_and_torch():
    g = torch.Generator().manual_seed(0)
    mean, lv, y = torch.randn(5, 3, generator=g), torch.randn(5, 3, generator=g), torch.randn(5, 3, generator=g)
    ref = (0.5 * (lv + (y - mean) ** 2 / lv.exp())).mean()
    assert torch.allclose(gaussian_nll(mean, lv, y), ref)
    full = torch.nn.GaussianNLLLoss(full=True, eps=1e-12)(mean, y, lv.exp())
    assert torch.allclose(gaussian_nll(mean, lv, y, include_const=True), full, atol=1e-6)
    valid = torch.zeros(5, 3, dtype=torch.bool)
    valid[0, 1] = valid[3, 2] = True
    part = 0.5 * (lv + (y - mean) ** 2 / lv.exp())
    assert torch.allclose(gaussian_nll(mean, lv, y, valid), (part[0, 1] + part[3, 2]) / 2)
    # clamp guard: an extreme log-variance is limited
    big = gaussian_nll(torch.zeros(1), torch.tensor([100.0]), torch.zeros(1), logvar_max=8.0)
    assert torch.allclose(big, torch.tensor(4.0))
    assert gaussian_nll(mean, lv, y, torch.zeros(5, 3, dtype=torch.bool)) == 0


def test_baseline_loss_units_and_stop_gradient():
    g = torch.Generator().manual_seed(1)
    mean = torch.randn(4, 3, generator=g, requires_grad=True)
    lv = torch.randn(4, 3, generator=g, requires_grad=True)
    y = torch.randn(4, 3, generator=g)
    valid = torch.ones(4, 3, dtype=torch.bool)
    s = torch.tensor([1.0, 2.0, 0.5])
    out = baseline_loss(mean, lv, y, valid, s, mean_loss="none", detach_mean=False)
    # NLL in target-scale units = NLL in % minus mean(log s)
    assert torch.allclose(out["nll"], gaussian_nll(mean, lv, y) - torch.log(s).mean(), atol=1e-6)
    out = baseline_loss(mean, lv, y, valid, s, mean_loss="none", detach_mean=True)
    out["loss"].backward()
    assert mean.grad is None or torch.all(mean.grad == 0)                     # NLL does not move the mean
    assert lv.grad is not None and torch.any(lv.grad != 0)
    mean.grad = lv.grad = None
    out = baseline_loss(mean, lv, y, valid, s, mean_loss="mse", nll_weight=0.0)
    out["loss"].backward()
    assert torch.allclose(mean.grad, 2 * (mean - y).detach() / s ** 2 / 12, atol=1e-6)
    assert torch.allclose(out["mae"], (mean - y).abs().mean())
    with pytest.raises(ValueError):
        baseline_loss(mean, lv, y, valid, mean_loss="l7")


# ───────────────────────────────────────────────────────────── causal inference

def test_causal_windows_edge_padding():
    x = np.arange(5)[:, None].astype(float)
    w = causal_windows(x, 3)
    assert w.shape == (5, 3, 1)
    assert w[0, :, 0].tolist() == [0, 0, 0] and w[1, :, 0].tolist() == [0, 0, 1] and w[4, :, 0].tolist() == [2, 3, 4]
    assert np.array_equal(causal_windows(x, 3, np.array([1, 4])), w[[1, 4]])
    with pytest.raises(ValueError):
        causal_windows(x, 0)


def test_predict_episode_is_causal_and_matches_the_stream():
    ep = make_episode(T=80)
    m = small_model()
    with torch.no_grad():                           # non-trivial weights: perturb the zero-init head
        m.mean_head.weight.normal_(0, 0.5)
        m.logvar_head.weight.normal_(0, 0.5)
    mean, lv = predict_episode(m, ep, batch_size=17)
    assert mean.shape == (80, 4) and lv.shape == (80, 4) and mean.dtype == np.float32
    stream = CausalBaselineStream(m)
    for t in range(80):
        ms, ls = stream.push(ep["q"][t], ep["qd"][t], ep["taxel_pos"][t], ep["taxel_nrm"][t])
        assert np.allclose(ms, mean[t], atol=1e-5) and np.allclose(ls, lv[t], atol=1e-5)
    # future frames never change the past; frames older than the window never matter
    arr = dict(ep.arrays)
    arr["q"] = ep["q"].copy()
    arr["q"][50:] += 1.0
    arr["q"][:10] -= 1.0
    ep2 = Episode(ep.meta, arr)
    mean2, _ = predict_episode(m, ep2)
    assert np.allclose(mean2[26:50], mean[26:50], atol=1e-6)
    assert not np.allclose(mean2[50:], mean[50:])
    stream.reset()
    assert stream.n_pushed == 0


def test_learns_a_lagged_artefact_and_generalises():
    """Trained on one episode's no-contact frames (BaselineWindowDataset windows), the model must
    remove most of the artefact of another episode with the same taxel physics but other motion."""
    train, test = make_episode("a", seed=0, T=800), make_episode("b", seed=1, T=600, freqs=(0.9, 1.1))
    ds = BaselineWindowDataset([train], window=16, stride=1)
    m = small_model()
    m.set_joint_stats(NormStats(np.zeros(2, np.float32), np.full(2, 0.4, np.float32)),
                      NormStats(np.zeros(2, np.float32), np.full(2, 3.0, np.float32)))
    m.set_target_scale(np.sqrt((np.asarray(train["delta_pct"]) ** 2).mean(0)))
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    dl = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
    step = 0
    while step < 250:
        for b in dl:
            mean, lv = m(b["q_hist"], b["qd_hist"], b["pos"], b["nrm"])
            loss = baseline_loss(mean, lv, b["y"], b["valid"], m.y_scale)["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step >= 250:
                break
    mean, lv = predict_episode(m, test)
    d = np.asarray(test["delta_pct"])
    raw, res = np.abs(d).mean(), np.abs(d - mean).mean()
    assert res < 0.3 * raw, (raw, res)
    assert np.all(np.isfinite(lv))
    z = (d - mean) / np.exp(0.5 * lv)
    assert 0.2 < z.std() < 5.0                                                     # σ is in the right range


# ───────────────────────────────────────────────────────────── joint-state source & bundle

def _glove_episode(T=40, with_imu_pose=True):
    lay = load_layout("glove_template")
    rng = np.random.default_rng(3)
    fp = (rng.normal(0, 0.2, (T, 15, 3))).astype(np.float32)
    meta = EpisodeMeta(episode_id="g", dataset="motion", kind="glove", layout="glove_template", n_taxels=lay.n,
                       hz=200.0, preprocessing={"q_source": "hand_pose", "taxel_frame": "mano_wrist"})
    arrays = {"t": np.arange(T) / 200.0, "q": np.zeros((T, 45), np.float32), "qd": np.zeros((T, 45), np.float32),
              "hand_pose_valid": np.zeros(T, bool), "taxel_pos": np.zeros((T, lay.n, 3), np.float32),
              "taxel_nrm": np.zeros((T, lay.n, 3), np.float32)}
    ep = Episode(meta, arrays)
    if with_imu_pose:
        ep.set_derived(D_HAND_POSE_IMU, fp, save=False)
    return ep, fp, lay


def test_episode_joint_view_hand_pose_imu():
    from robot_skin.datasets.build import joint_velocity
    from robot_skin.datasets.stats import q_valid_mask
    from robot_skin.pose.mano import ManoSkeleton, taxel_poses_from_hand

    ep, fp, lay = _glove_episode()
    assert episode_joint_view(ep, "q") is ep
    v = episode_joint_view(ep, "hand_pose_imu")
    assert np.allclose(v["q"], fp.reshape(40, 45))
    assert np.allclose(v["qd"], joint_velocity(fp.reshape(40, 45), 200.0), atol=1e-4)
    pos, nrm = taxel_poses_from_hand(lay, ManoSkeleton(), np.zeros((40, 3)), fp.astype(np.float64), None)
    assert np.allclose(v["taxel_pos"], pos, atol=1e-6) and np.allclose(v["taxel_nrm"], nrm, atol=1e-6)
    assert q_valid_mask(ep).sum() == 0 and q_valid_mask(v).all()     # IMU pose: every frame measured
    assert v.meta.preprocessing["q_source"] == "hand_pose_imu" and ep.meta.preprocessing["q_source"] == "hand_pose"
    assert episode_joint_view(v, "hand_pose_imu") is v
    with pytest.raises(ValueError):
        episode_joint_view(ep, "vision")
    with pytest.raises(ValueError):
        episode_joint_view(_glove_episode(with_imu_pose=False)[0], "hand_pose_imu")


def test_save_load_roundtrip(tmp_path):
    ep = make_episode(T=50)
    m = small_model(arch="gru", n_layers=2)
    m.set_target_scale([0.5, 1.0, 1.5, 2.0])
    m.set_joint_stats(NormStats(np.ones(2), np.full(2, 2.0)), NormStats(np.zeros(2), np.full(2, 4.0)))
    with torch.no_grad():
        m.mean_head.weight.normal_()
    p = save_baseline_model(tmp_path / "run", m, {"q_source": "q", "window": m.window})
    assert p.name == "baseline_model.pt"
    m2 = load_baseline_model(tmp_path / "run")
    assert m2.bundle_meta["q_source"] == "q" and m2.config == m.config
    a, b = predict_episode(m, ep), predict_episode(m2, ep)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    torch.save({"format": "other"}, tmp_path / "x.pt")
    with pytest.raises(ValueError):
        load_baseline_model(tmp_path / "x.pt")


def test_qd_settings_are_joint_velocity_kwargs():
    """The online processor calls ``joint_velocity(q, hz, **qd_settings(pre))``: preprocessing's
    ``qd`` section also holds ``source`` (not a kwarg), which must be dropped; defaults fill gaps."""
    from robot_skin.datasets.build import joint_velocity

    assert qd_settings(None) == {"method": "savgol_causal", "window_s": 0.05, "polyorder": 2}
    pre = {"config": {"qd": {"method": "gradient", "window_s": 0.03, "polyorder": 2, "source": "derivative"}}}
    kw = qd_settings(pre)
    assert kw == {"method": "gradient", "window_s": 0.03, "polyorder": 2}
    q = np.cumsum(np.random.default_rng(0).normal(0, 0.01, (50, 3)), 0)
    assert np.allclose(joint_velocity(q, 200.0, **kw), joint_velocity(q, 200.0, method="gradient", window_s=0.03))
    ep, fp, _ = _glove_episode()
    ep.meta.preprocessing["config"] = pre["config"]
    v = episode_joint_view(ep, "hand_pose_imu")                        # the view uses the episode's settings
    assert np.allclose(v["qd"], joint_velocity(fp.reshape(40, 45), 200.0, **kw), atol=1e-5)
