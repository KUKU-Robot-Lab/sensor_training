"""contact.{calibration, hysteresis, detector, pseudo_label}: calibrated z, levels, detector, labels."""
import numpy as np
import pytest
import torch

from robot_skin.contact import (ContactDetector, ContactLevel, HysteresisFilter, ResidualCalibrator, SatState,
                                contact_loss, focal_loss, frame_expectation, load_detector, phase_expectation,
                                predict_contact_prob, pseudo_label_episode, pseudo_label_metrics, residual_levels,
                                robust_sigma, saturation_gate, save_detector, taxel_world_positions)
from robot_skin.contact.detector import CausalDetectorStream
from robot_skin.datasets.episode import Episode, EpisodeMeta


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ───────────────────────────────────────────────────────────── calibration

def test_robust_sigma():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, [0.5, 3.0], (20000, 2))
    x[:200] = 1e3                                                    # outliers barely move MAD
    s, med, n = robust_sigma(x)
    assert np.allclose(s, [0.5, 3.0], rtol=0.05) and np.allclose(med, 2.0, atol=0.1) and n.tolist() == [20000] * 2
    valid = np.zeros_like(x, bool)
    valid[:, 0] = True
    s, med, n = robust_sigma(x, valid)
    assert np.isnan(s[1]) and n[1] == 0


def test_calibrator_without_logvar_levels_and_floors():
    rng = np.random.default_rng(1)
    T, sig = 5000, np.array([0.1, 0.5, 0.01])
    r = rng.normal(0, sig, (T, 3)) + np.array([0.2, 0.0, 0.0])        # taxel 0 residual offset (+0.2 %)
    cal = ResidualCalibrator.fit(r, weak_z=3, strong_z=8, weak_floor_pct=0.5, strong_floor_pct=2.0,
                                 sigma_floor_pct=0.001)
    assert np.allclose(cal.sigma, sig, rtol=0.06) and np.allclose(cal.center, [-0.2, 0, 0], atol=0.02)
    z = cal.transform(r)
    assert np.allclose(1.4826 * np.median(np.abs(z), 0), 1.0, atol=0.06)   # unit robust spread
    # a press (ΔS negative) of 6 σ on taxel 1 → z ≈ 6 → WEAK; 20 σ → STRONG; saturated → SAT
    press = np.array([[0.0, -3.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, -0.1], [0.0, -10.0, -1.0]])
    z, lv = cal.levels_from_residual(press, saturated=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]], bool))
    assert np.isclose(z[0, 1], 6.0, rtol=0.07)
    assert lv[0, 1] == ContactLevel.WEAK and lv[1, 1] == ContactLevel.STRONG
    # taxel 2 has σ = 0.01 %: 0.1 % is z ≈ 10 but below the 0.5 % floor → NONE; 1 % → WEAK (floor 2 % for STRONG)
    assert z[2, 2] > 8 and lv[2, 2] == ContactLevel.NONE and lv[3, 2] == ContactLevel.WEAK
    assert lv[3, 0] == ContactLevel.SATURATED
    assert cal.levels(np.array([[np.nan, 0.0, 0.0]]))[0, 0] == ContactLevel.NONE
    with pytest.raises(ValueError):
        cal.transform(np.zeros((2, 4)))


def test_calibrator_with_predicted_variance():
    """σ_eff = g·sqrt(σ² + exp(logvar)): a model that over-states its variance 4× gets g ≈ 0.5,
    and time-varying variance is honoured (z of a high-variance frame is smaller)."""
    rng = np.random.default_rng(2)
    T = 20000
    true_sd = np.where(np.arange(T) % 2 == 0, 0.1, 0.4)[:, None] * np.ones((1, 2))
    r = rng.normal(0, true_sd)
    lv = np.log((2 * true_sd) ** 2)                                  # predicted σ = 2 × true σ
    cal = ResidualCalibrator.fit(r, None, lv, sigma_floor_pct=1e-3)
    assert cal.use_logvar
    assert np.allclose(cal.gain, 0.5, rtol=0.1)
    z = cal.transform(r, lv)
    for par in (0, 1):
        assert np.isclose(np.std(z[par::2]), 1.0, atol=0.1)          # calibrated in both variance regimes
    with pytest.raises(ValueError):
        cal.transform(r)                                             # logvar required


def test_calibrator_fallback_roundtrip_and_validation(tmp_path):
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.2, (400, 3))
    valid = np.ones_like(r, bool)
    valid[:, 2] = False
    valid[:10, 2] = True                                             # < min_samples
    cal = ResidualCalibrator.fit(r, valid, min_samples=50, fsm={"enabled": False, "ok_pct": 1.0})
    assert cal.info["fallback_taxels"] == [2] and cal.center[2] == 0.0
    assert np.isclose(cal.sigma[2], np.median(cal.sigma[:2]))
    p = cal.save(tmp_path / "cal.json")
    cal2 = ResidualCalibrator.load(p)
    assert np.allclose(cal2.sigma, cal.sigma) and cal2.fsm == cal.fsm and cal2.weak_z == cal.weak_z
    assert np.allclose(cal2.transform(r), cal.transform(r))
    with pytest.raises(ValueError):
        ResidualCalibrator.fit(r, np.zeros_like(r, bool))
    with pytest.raises(ValueError):
        ResidualCalibrator(np.ones(2), np.zeros(2), np.ones(2), weak_z=5, strong_z=4)
    with pytest.raises(ValueError):
        ResidualCalibrator(np.ones(2), np.zeros(2), np.ones(2), fsm={"bogus": 1})
    with pytest.raises(ValueError):
        ResidualCalibrator.from_dict({**cal.to_dict(), "format": "x"})


def test_saturation_gate_and_residual_levels():
    T, dt = 200, 0.01
    r = np.zeros((T, 2), np.float32)
    sat = np.zeros((T, 2), bool)
    sat[50:60, 0] = True
    r[50:60, 0] = -95.0
    r[60:, 0] = -5.0                                                  # settles 5 % away → never OK again ...
    corr, untrusted, states = saturation_gate(r, sat, dt, ok_pct=1.0, ok_sec=0.2, max_recover_s=0.5)
    assert untrusted[55, 0] and states[55, 0] == SatState.SATURATED
    assert untrusted[80, 0] and states[80, 0] == SatState.RECOVERING
    assert not untrusted[150, 0] and np.isclose(corr[150, 0], 0.0)    # ... until the re-zero timeout
    assert not untrusted[:, 1].any()
    cal = ResidualCalibrator(np.array([0.1, 0.1]), np.zeros(2), np.ones(2),
                             fsm={"enabled": True, "ok_pct": 1.0, "ok_sec": 0.2, "max_recover_s": 0.5})
    out = residual_levels(cal, r, sat, dt=dt)
    assert out["contact_level"][80, 0] == ContactLevel.SATURATED and out["contact_level"][150, 0] == ContactLevel.NONE
    with pytest.raises(ValueError):
        residual_levels(cal, r, sat)                                  # FSM needs dt
    off = residual_levels(ResidualCalibrator(np.array([0.1, 0.1]), np.zeros(2), np.ones(2)), r, sat)
    assert off["contact_level"][80, 0] == ContactLevel.STRONG         # no gate: the offset reads as a press
    c2, u2, _ = saturation_gate(r, sat, dt, enabled=False)
    assert np.array_equal(u2, sat) and np.array_equal(c2, r)


# ───────────────────────────────────────────────────────────── hysteresis

def test_hysteresis_semantics_and_streaming():
    p = np.array([0.1, 0.7, 0.2, 0.7, 0.8, 0.5, 0.3, 0.3, 0.5, 0.3, 0.3, 0.3, np.nan, 0.9, 0.9])
    hf = HysteresisFilter(on_thr=0.6, off_thr=0.4, min_on=2, min_off=3)
    out = hf.run(p)
    exp = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]   # on at the 2nd high tick, off at the 3rd low one
    assert out.astype(int).tolist() == exp
    hf2 = HysteresisFilter.from_config(hf)
    stream = np.array([hf2.step(np.array([v]))[0] for v in p])
    assert np.array_equal(stream, out)
    multi = hf.run(np.stack([p, 1 - np.nan_to_num(p)], 1))
    assert np.array_equal(multi[:, 0], out) and multi.shape == (15, 2)
    with pytest.raises(ValueError):
        HysteresisFilter(on_thr=0.3, off_thr=0.5)
    with pytest.raises(ValueError):
        HysteresisFilter(min_on=0)
    with pytest.raises(ValueError):
        hf.step(np.zeros(3))                                          # taxel count fixed after the first step


# ───────────────────────────────────────────────────────────── detector

def test_focal_and_bce_losses():
    g = torch.Generator().manual_seed(0)
    logits, y = torch.randn(6, 4, generator=g), (torch.rand(6, 4, generator=g) > 0.7).float()
    mask = torch.rand(6, 4, generator=g) > 0.2
    p = torch.sigmoid(logits)
    pt = torch.where(y > 0, p, 1 - p)
    at = torch.where(y > 0, torch.tensor(0.25), torch.tensor(0.75))
    ref = -(at * (1 - pt) ** 2 * torch.log(pt))
    assert torch.allclose(focal_loss(logits, y, mask), ref[mask].mean(), atol=1e-6)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    assert torch.allclose(focal_loss(logits, y, gamma=0.0, alpha=None), bce.mean(), atol=1e-6)
    pw = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                                              pos_weight=torch.tensor(3.0))
    assert torch.allclose(contact_loss(logits, y, mask, kind="bce", pos_weight=3.0), pw[mask].mean(), atol=1e-6)
    with pytest.raises(ValueError):
        contact_loss(logits, y, kind="hinge")


def _z_data(T=600, N=3, seed=0):
    """z traces: presses = sustained steps (contact), motion = brief spikes during fast joint motion."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (T, N)).astype(np.float32)
    lab = np.zeros((T, N), np.int8)
    qd = np.zeros((T, 2), np.float32)
    for k in range(6):
        a = 40 + k * 90
        n = k % N
        z[a:a + 40, n] += rng.uniform(8, 20)
        lab[a:a + 40, n] = 1
        b = a + 60                                                    # motion burst: spikes, no contact
        qd[b:b + 10] = 4.0
        z[b:b + 3, (n + 1) % N] += 9.0
    return z, lab, qd


def test_detector_init_causality_stream_and_learning(tmp_path):
    torch.manual_seed(0)
    det = ContactDetector(3, window=12, hidden=16, joint_dim=2, n_layers=2, prior=0.02)
    z, lab, qd = _z_data()
    prob0 = predict_contact_prob(det, z=z, qd=qd)
    assert prob0.shape == z.shape and np.allclose(prob0, 0.02, atol=1e-4)      # prior initialisation
    with pytest.raises(ValueError):
        ContactDetector(3, use_motion=True, joint_dim=None)
    with pytest.raises(ValueError):
        det(torch.zeros(1, 12, 3))                                              # needs qd
    # train on windows (a plain loop; the stage uses robot_skin.train.Trainer)
    from robot_skin.baseline.temporal import causal_windows

    zw = torch.as_tensor(np.ascontiguousarray(causal_windows(z, 12)))
    qdt, y = torch.as_tensor(qd), torch.as_tensor(lab, dtype=torch.float32)
    opt = torch.optim.Adam(det.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    for _ in range(150):
        idx = torch.randint(0, len(zw), (64,), generator=gen)
        loss = focal_loss(det(zw[idx], None, qdt[idx]), y[idx], None, alpha=0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
    zt, labt, qdt2 = _z_data(seed=1)
    prob = predict_contact_prob(det, z=zt, qd=qdt2, batch_size=100)
    from robot_skin.eval.metrics import auroc

    assert auroc(prob[labt == 1], prob[labt == 0]) > 0.95
    # causal + online equivalence
    stream = CausalDetectorStream(det)
    for t in range(40):
        assert np.allclose(stream.push(zt[t], None, qdt2[t]), prob[t], atol=1e-5)
    z2 = zt.copy()
    z2[300:] += 50
    prob2 = predict_contact_prob(det, z=z2, qd=qdt2)
    assert np.allclose(prob2[:300], prob[:300], atol=1e-6)
    # bundle roundtrip
    save_detector(tmp_path / "d", det, {"window": det.window})
    det2 = load_detector(tmp_path / "d")
    assert det2.bundle_meta["window"] == 12
    assert np.allclose(predict_contact_prob(det2, z=zt, qd=qdt2), prob, atol=1e-6)


def test_detector_without_motion_or_embedding():
    det = ContactDetector(None, window=4, hidden=8, use_motion=False, use_sat=False)
    out = det(torch.randn(2, 4, 7))
    assert out.shape == (2, 7)                                                 # any taxel count
    assert det.taxel_emb is None and det.motion is None


# ───────────────────────────────────────────────────────────── pseudo labels

def _task_episode(T=100, N=3):
    t = np.arange(T) / 200.0
    phases = [{"name": "reach", "t0": 0.0, "t1": 0.1, "contact": "none"},
              {"name": "grasp", "t0": 0.1, "t1": 0.4, "contact": "object"},
              {"name": "retreat", "t0": 0.4, "t1": 0.45}]                     # no "contact" field → by name
    pid = np.full(T, -1, np.int16)
    pid[(t >= 0.0) & (t < 0.1)] = 0
    pid[(t >= 0.1) & (t < 0.4)] = 1
    pid[(t >= 0.4) & (t < 0.45)] = 2
    lab = np.full((T, N), -1, np.int8)
    lab[:5] = 0                                                               # preprocessing: no_contact segment
    sat = np.zeros((T, N), bool)
    sat[50:55, 2] = True
    meta = EpisodeMeta(episode_id="task", dataset="task", kind="robot", layout="grid", n_taxels=N,
                       phases=phases, phase_names=["reach", "grasp", "retreat"],
                       preprocessing={"taxel_frame": "urdf_root"})
    pos = np.zeros((T, N, 3), np.float32)
    pos[:, 1, 0] = 0.5                                                        # taxel 1 is far from the object
    arrays = {"t": t, "phase_id": pid, "contact_label": lab, "saturated": sat, "taxel_pos": pos,
              "object_pos": np.zeros((T, 3), np.float32)}
    return Episode(meta, arrays)


def test_phase_expectation_inference():
    assert phase_expectation({"name": "whatever", "contact": "self"}) == "self"
    assert phase_expectation({"name": "baseline_start", "labels": ["no_contact"]}) == "none"
    assert phase_expectation("manipulate") == "object" and phase_expectation("pinch_index") == "self"
    assert phase_expectation("retreat") == "none" and phase_expectation("sync_start") == "any"
    assert phase_expectation("mystery") is None
    ep = _task_episode()
    exp = frame_expectation(ep)
    assert exp[0] == "none" and exp[30] == "object" and exp[85] == "none" and exp[95] == ""
    assert frame_expectation(ep, overrides={"grasp": "any"})[30] == "any"
    with pytest.raises(ValueError):
        frame_expectation(ep, overrides={"grasp": "maybe"})


def test_pseudo_label_rules():
    ep = _task_episode()
    T, N = 100, 3
    prob = np.full((T, N), 0.25)                                               # ambiguous everywhere
    prob[10:20, 0] = 0.9                                                       # detector fires in reach (none)
    prob[30:60, 0] = 0.9                                                       # contact during grasp
    prob[30:60, 1] = 0.9                                                       # ... on the far taxel too
    prob[60:80, 0] = 0.01                                                      # confidently free during grasp
    hcfg = {"on_thr": 0.6, "off_thr": 0.4, "min_on": 1, "min_off": 1}
    res = pseudo_label_episode(ep, prob, hysteresis=hcfg, neg_thr=0.1)
    lab = res["label"]
    assert lab.dtype == np.int8 and set(np.unique(lab)) <= {-1, 0, 1}
    assert np.all(lab[:5] == 0)                                                # existing labels kept
    assert np.all(lab[10:20, 0] == -1) and np.all(lab[10:20, 1] == 0)         # conflict in a "none" phase
    assert np.all(lab[30:60, 0] == 1) and np.all(lab[62:80, 0] == 0)
    assert np.all(lab[25:30, 2] == -1)                                         # ambiguous in grasp → unknown
    assert np.all(lab[50:55, 2] == 1)                                          # saturated during grasp → contact
    assert np.all(lab[90:, :] == -1)                                           # outside any phase
    assert res["counts"]["n_conflict"] == 10
    zero = pseudo_label_episode(ep, prob, hysteresis=hcfg, none_conflict="zero")["label"]
    assert np.all(zero[10:20, 0] == 0)
    prox = pseudo_label_episode(ep, prob, hysteresis=hcfg, proximity={"max_dist_m": 0.1})
    assert np.all(prox["label"][30:60, 1] == -1) and np.all(prox["label"][30:60, 0] == 1)
    assert prox["counts"]["n_vetoed"] == 30
    anyp = pseudo_label_episode(ep, prob, hysteresis=hcfg, unknown_phase="none")["label"]
    assert np.all(anyp[90:, :] == 0)
    with pytest.raises(ValueError):
        pseudo_label_episode(ep, prob[:10], hysteresis=hcfg)
    m = pseudo_label_metrics(np.array([1, 1, 0, 0, -1]), np.array([1, 0, 0, 1, 1], bool))
    assert m["precision"] == 0.5 and np.isclose(m["recall"], 1 / 3) and m["neg_precision"] == 0.5
    assert m["coverage"] == 0.8


def test_taxel_world_positions_glove_frame():
    from robot_skin.geometry.rotations import aa_to_matrix

    T, N = 4, 2
    rng = np.random.default_rng(0)
    go, wp = rng.normal(0, 0.5, (T, 3)), rng.normal(0, 0.1, (T, 3))
    pos = rng.normal(0, 0.05, (T, N, 3)).astype(np.float32)
    meta = EpisodeMeta(episode_id="g", dataset="task", kind="glove", layout="glove_template", n_taxels=N,
                       preprocessing={"taxel_frame": "mano_wrist"})
    ep = Episode(meta, {"t": np.arange(T) / 200.0, "taxel_pos": pos, "hand_global_orient": go.astype(np.float32),
                        "hand_wrist_pos": wp.astype(np.float32), "hand_pose_valid": np.array([1, 1, 0, 1], bool)})
    w, valid = taxel_world_positions(ep)
    R = aa_to_matrix(torch.as_tensor(go)).numpy()
    assert np.allclose(w, np.einsum("tij,tnj->tni", R, pos) + wp[:, None], atol=1e-5)
    assert valid.tolist() == [True, True, False, True]


def test_proximity_veto_needs_a_hand_pose_for_gloves():
    """A camera-free glove episode has hand-frame taxel poses and no hand pose: its taxels are not in
    the object's frame, so no frame may be vetoed (distances would be meaningless)."""
    T, N = 40, 2
    t = np.arange(T) / 200.0
    meta = EpisodeMeta(episode_id="g", dataset="task", kind="glove", layout="glove_template", n_taxels=N,
                       phases=[{"name": "grasp", "t0": 0.0, "t1": 1.0, "contact": "object"}], phase_names=["grasp"],
                       preprocessing={"taxel_frame": "mano_wrist"})
    ep = Episode(meta, {"t": t, "phase_id": np.zeros(T, np.int16), "taxel_pos": np.zeros((T, N, 3), np.float32),
                        "object_pos": np.full((T, 3), 5.0, np.float32), "contact_label": np.full((T, N), -1, np.int8),
                        "saturated": np.zeros((T, N), bool)})
    w, valid = taxel_world_positions(ep)
    assert w.shape == (T, N, 3) and not valid.any()
    res = pseudo_label_episode(ep, np.full((T, N), 0.9), hysteresis={"min_on": 1}, proximity={"max_dist_m": 0.1})
    assert np.all(res["label"] == 1) and res["counts"]["n_vetoed"] == 0
