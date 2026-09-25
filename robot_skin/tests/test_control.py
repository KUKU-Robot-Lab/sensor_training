"""robot_skin.control: online tactile processor ≡ offline stages, safety filter, fake hand, runner,
deployment-log re-ingestion, bundle loading, latency, and the ``deploy`` stage."""
import warnings

import numpy as np
import pytest
import torch
import yaml

from common.layouts import load_layout
from common.signal import ADC_MIN, relative_change
from robot_skin.acquisition.sources import SimClock
from robot_skin.contact.ordinal import ContactLevel


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ─────────────────────────────────────────────────────────────── helpers

def make_bundle(path, kind="robot_joint", *, cams=("ego",), head="chunk", action_offset=None, calibrator=True,
                source="derived", obs_history=1, fsm=None, layouts=None, rel_mode="delta", cal_logvar=False,
                baseline_ref=None):
    """A small untrained VTLA bundle written with the vtla APIs (the way stages/vtla.py does).
    ``action_offset``: robot_joint actions ≈ this constant (normalizer offset, tiny scale);
    ``cal_logvar``: the embedded calibrator used the baseline log-variance (contact-stage default);
    ``baseline_ref``: the bundle's ``tactile.baseline_model`` reference."""
    from common.signal import NormStats
    from robot_skin.action import ActionNormalizer, ActionSpec
    from robot_skin.action.space import hand_action_from_arrays
    from robot_skin.contact.calibration import ResidualCalibrator
    from robot_skin.datasets.synthetic import ROBOT_JOINT_NAMES
    from robot_skin.vision import EvalTransform
    from robot_skin.vtla import VTLAConfig, VTLAPolicy, eval_transform_to_dict, save_policy_bundle

    A = 16 if kind == "robot_joint" else 54
    torch.manual_seed(0)
    cfg = VTLAConfig(action_dim=A, horizon=8, proprio_dim=A, obs_history=obs_history, d_model=32,
                     fusion_depth=1, fusion_heads=4, head_depth=1, cameras=cams, head=head, flow_steps=3,
                     vision={"type": "tiny", "out_dim": 16, "grid": [2, 2], "channels": [8, 16]},
                     text={"type": "hashing", "dim": 16, "max_len": 8},
                     tactile_encoder={"d_model": 16, "depth": 1, "heads": 2, "n_fourier": 2}, tactile_heads=2)
    m = VTLAPolicy(cfg).eval()
    rng = np.random.default_rng(0)
    if kind == "robot_joint":
        spec = ActionSpec.robot_joint(list(ROBOT_JOINT_NAMES))
        X = rng.uniform(0.0, 0.8, size=(200, A))
        an = ActionNormalizer.fit(0.05 * X, spec=spec)
        if action_offset is not None:
            an = ActionNormalizer(NormStats(np.full(A, action_offset, np.float32), np.full(A, 1e-4, np.float32)), spec)
        pn = ActionNormalizer.fit(X, spec=spec)
    else:
        spec = ActionSpec.hand_mano()
        X = hand_action_from_arrays(rng.normal(scale=0.1, size=(200, 3)), rng.normal(scale=0.2, size=(200, 15, 3)),
                                    rng.normal(scale=0.05, size=(200, 3)) + [0.0, 0.0, 0.4])
        an, pn = ActionNormalizer.fit(0.1 * X, spec=spec), ActionNormalizer.fit(X, spec=spec)
    cal = ResidualCalibrator(sigma=np.full(9, 0.2), center=np.zeros(9), gain=np.ones(9), fsm=fsm,
                             use_logvar=cal_logvar)
    tac = {"feature_spec": m.feature_spec.to_dict(), "contact_rule": "level_ge_weak", "source": source,
           "calibrator": None, "calibrator_state": cal.to_dict() if calibrator else None,
           "baseline_model": baseline_ref}
    if layouts is not None:
        tac["layouts"] = list(layouts)
    return save_policy_bundle(
        path, m, action={"spec": spec.to_dict(), "normalizer": an.to_dict(), "rel_mode": rel_mode, "chunk_offset": 1},
        proprio={"normalizer": pn.to_dict(), "history": obs_history, "source": "action_state"}, tactile=tac,
        vision={"cameras": list(m.cameras), "encoder": m.cfg.vision,
                "eval_transform": eval_transform_to_dict(EvalTransform((24, 32)))},
        language={"encoder": m.cfg.text},
        timing={"policy_hz": 20.0, "source_hz": 200.0, "stride": 10, "horizon": 8, "obs_history": obs_history},
        meta={"ensemble_k": 0.01})


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    d = tmp_path_factory.mktemp("bundles")
    return {"joint": make_bundle(d / "joint"), "hand": make_bundle(d / "hand", "hand_mano"),
            "squeeze": make_bundle(d / "squeeze", action_offset=1.2, cams=(), rel_mode="abs"),
            "boot": make_bundle(d / "boot", source="bootstrap", cams=())}


@pytest.fixture(scope="module")
def episodes(tmp_path_factory):
    """One processed synthetic motion episode per kind (in memory) + its layout / URDF."""
    from robot_skin.acquisition.manifest import SessionManifest
    from robot_skin.datasets.build import preprocess_session, resolve_layout
    from robot_skin.datasets.synthetic import generate_session

    root = tmp_path_factory.mktemp("ctrl_eps")
    out = {}
    for kind in ("robot", "glove"):
        sd = root / kind
        generate_session(sd, kind=kind, dataset="motion", duration_s=2.5, seed=3, cameras=())
        ep = preprocess_session(sd, None, {"baseline": {"duration_s": 0.3}})
        layout, _ = resolve_layout(sd, SessionManifest.load(sd))
        out[kind] = {"ep": ep, "layout": layout, "urdf": sd / "robot.urdf" if kind == "robot" else None}
    return out


def _with_dropout(ep, taxel=2, frames=slice(150, 175)):
    """Episode copy with a front-end dropout (raw on the lower rail) processed like preprocessing."""
    from robot_skin.datasets.episode import Episode

    arr = {k: np.array(v) for k, v in ep.arrays.items()}
    arr["pressure_raw"][frames, taxel] = ADC_MIN
    base = ep.static["baseline_raw"]
    arr["delta_pct"] = relative_change(arr["pressure_raw"], base)
    arr["saturated"][frames, taxel] = True
    return Episode(ep.meta, arr, ep.static)


def _random_baseline(ep):
    from robot_skin.baseline.temporal import TemporalBaselinePredictor
    from robot_skin.datasets.stats import compute_stats

    N, D = ep.meta.n_taxels, ep["q"].shape[1]
    torch.manual_seed(0)
    m = TemporalBaselinePredictor(N, D, window=16, hidden=16, head_hidden=16, n_layers=3)
    torch.nn.init.normal_(m.mean_head.weight, std=0.5)                  # non-trivial predictions
    torch.nn.init.normal_(m.logvar_head.weight, std=0.3)
    st = compute_stats([ep], keys=("q", "qd"), masks=None)
    m.set_joint_stats(st["q"], st["qd"])
    m.bundle_meta = {"hz": float(ep.meta.hz), "qd": {"method": "savgol_causal", "window_s": 0.05, "polyorder": 2},
                     "qd_source": "derivative", "joint_names": list(ep.meta.joint_names)}
    return m.eval()


# ─────────────────────────────────────────────────────────────── joint velocity

def test_causal_joint_velocity_matches_offline():
    from robot_skin.control.online import CausalJointVelocity
    from robot_skin.datasets.build import joint_velocity

    rng = np.random.default_rng(0)
    q = np.cumsum(rng.normal(scale=0.01, size=(120, 4)), axis=0).astype(np.float32)
    off = joint_velocity(q, 200.0, method="savgol_causal", window_s=0.05, polyorder=2)
    f = CausalJointVelocity(200.0, window_s=0.05, polyorder=2)
    on = np.stack([f.push(r) for r in q])
    np.testing.assert_allclose(on, off, rtol=1e-6, atol=1e-6)
    f.reset()
    np.testing.assert_allclose(f.push(q[5]), 0.0, atol=1e-6)            # fresh stream: edge-padded
    with pytest.raises(ValueError, match="not causal"):
        CausalJointVelocity(200.0, method="savgol")
    with pytest.raises(ValueError):
        f.push(np.zeros(3))


# ─────────────────────────────────────────────────────────────── online ≡ offline

@pytest.mark.parametrize("kind", ["robot", "glove"])
def test_online_processor_reproduces_offline_stage_outputs(episodes, kind):
    """Stream a processed episode through the processor: every intermediate equals the offline stage
    functions (baseline.predict_episode → residual → contact.residual_levels with the FSM gate →
    TactileFeatureSpec.from_arrays; detector + hysteresis)."""
    from robot_skin.baseline.temporal import predict_episode
    from robot_skin.contact.calibration import ResidualCalibrator, residual_levels
    from robot_skin.contact.detector import ContactDetector, predict_contact_prob
    from robot_skin.contact.hysteresis import HysteresisFilter
    from robot_skin.control import OnlineTactileProcessor, replay_episode
    from robot_skin.representation import TactileFeatureSpec

    E = episodes[kind]
    ep = _with_dropout(E["ep"])
    model = _random_baseline(ep)
    mean, lv = predict_episode(model, ep)
    res = (np.asarray(ep["delta_pct"]) - mean).astype(np.float32)
    sat = np.asarray(ep["saturated"])
    fsm = {"enabled": True, "ok_pct": 1.5, "ok_sec": 0.1, "max_recover_s": 0.4}
    cal = ResidualCalibrator.fit(res, (np.asarray(ep["contact_label"]) == 0) & ~sat, lv, fsm=fsm)
    off = residual_levels(cal, res, sat, lv, dt=1.0 / ep.meta.hz)
    assert (off["contact_level"] == ContactLevel.SATURATED).any() and (off["contact_level"] == ContactLevel.WEAK).any()
    spec = TactileFeatureSpec(history=3, stride=2)
    feats = spec.from_arrays(off["residual_z"], off["contact_level"], sat, np.arange(ep.T))
    torch.manual_seed(1)
    det = ContactDetector(ep.meta.n_taxels, joint_dim=ep["q"].shape[1], window=8, hidden=8).eval()
    prob = predict_contact_prob(det, z=off["residual_z"], saturated=sat, qd=np.asarray(ep["qd"]),
                                q_valid=np.ones(ep.T))
    hcfg = {"on_thr": 0.5, "off_thr": 0.4, "min_on": 2, "min_off": 2}
    hyst = HysteresisFilter(**hcfg).run(prob)

    proc = OnlineTactileProcessor(E["layout"], model, cal, feature_spec=spec, raw_order="layout", urdf=E["urdf"],
                                  detector=det, hysteresis=hcfg)
    assert proc.hz == ep.meta.hz and proc.fsm_config["enabled"]
    on = replay_episode(proc, ep)
    np.testing.assert_array_equal(on["delta"], ep["delta_pct"])
    np.testing.assert_array_equal(on["saturated"], sat)
    np.testing.assert_allclose(on["qd"], ep["qd"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(on["baseline_mean"], mean, atol=1e-5)
    np.testing.assert_allclose(on["baseline_logvar"], lv, atol=1e-5)
    np.testing.assert_allclose(on["residual_z"], off["residual_z"], rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(on["level"], off["contact_level"])
    np.testing.assert_array_equal(on["untrusted"], off["untrusted"])
    np.testing.assert_allclose(on["features"], feats, atol=1e-5)
    np.testing.assert_allclose(on["prob"], prob, atol=1e-6)
    np.testing.assert_array_equal(on["contact_on"], hyst)

    # the processor's own pose function (hand / URDF frame) and saturation rule agree with preprocessing
    proc2 = OnlineTactileProcessor(E["layout"], model, cal, raw_order="layout", urdf=E["urdf"])
    on2 = replay_episode(proc2, ep, poses="model", extra_saturated=False, frames=slice(0, 40))
    np.testing.assert_allclose(on2["pos"], np.asarray(ep["taxel_pos"])[:40], atol=1e-5)
    np.testing.assert_allclose(on2["nrm"], np.asarray(ep["taxel_nrm"])[:40], atol=1e-5)
    np.testing.assert_allclose(on2["baseline_mean"], mean[:40], atol=1e-4)
    on3 = replay_episode(proc2, ep, extra_saturated=False)
    assert not (on3["saturated"] & ~sat).any()                        # online flags ⊆ offline flags
    assert on3["saturated"][150:175, 2].all()

    # start-up baseline capture = preprocessing's baseline over the same no-contact window
    b = ep.meta.preprocessing["baseline"]
    t = np.asarray(ep.t)
    rows = (t >= b["segment"][0]) & (t < b["segment"][1])
    proc3 = OnlineTactileProcessor(E["layout"], None, None, raw_order="layout", baseline_s=0.3, urdf=E["urdf"])
    base = proc3.capture_baseline(np.asarray(ep["pressure_raw"])[rows], t=t[rows])
    np.testing.assert_array_equal(base, ep.static["baseline_raw"])
    # a calibrator fitted with the baseline log-variance cannot run without that baseline model
    assert cal.use_logvar
    with pytest.raises(ValueError, match="log-variance"):
        OnlineTactileProcessor(E["layout"], None, cal, raw_order="layout", urdf=E["urdf"])
    with pytest.raises(ValueError, match="log-variance"):
        proc3.set_calibrator(cal)


def test_processor_saturation_dead_taxels_and_uncalibrated():
    from robot_skin.control import OnlineTactileProcessor

    layout = load_layout("robot_hand_template")
    proc = OnlineTactileProcessor(layout, None, None, raw_order="channel")
    raw = np.full(10, 4.0e6)
    with pytest.raises(RuntimeError, match="baseline"):
        proc.step(raw, pos=layout.positions, nrm=layout.normals)
    with pytest.raises(RuntimeError, match="usable baseline"):
        proc.finish_baseline()
    proc.begin_baseline()
    assert proc.add_baseline_sample(np.where(np.arange(10) == 3, 0.0, raw)) == 0     # rail sample skipped
    for _ in range(5):
        proc.add_baseline_sample(raw)
    base = proc.finish_baseline()
    np.testing.assert_array_equal(base, 4.0e6)
    b = base.copy()
    b[4] = 0.0                                                          # dead channel
    proc.set_baseline(b)
    r = raw * 0.97
    r[1] = np.nan
    r[2] = ADC_MIN
    fr = proc.step(r, pos=layout.positions, nrm=layout.normals)
    assert fr.saturated[[1, 2, 4]].all() and not fr.saturated[[0, 3, 5]].any()
    assert fr.delta[4] == 0.0 and np.isfinite(fr.delta).all()
    np.testing.assert_allclose(fr.delta[0], -3.0, rtol=1e-5)
    assert np.isnan(fr.residual_z).all()                                # no calibrator yet
    assert (fr.level[[1, 2, 4]] == ContactLevel.SATURATED).all() and (fr.level[[0, 3]] == ContactLevel.NONE).all()
    assert fr.contact[[1, 2, 4]].all() and fr.features is None and fr.prob is None
    with pytest.raises(ValueError):
        proc.step(np.ones(5), pos=layout.positions, nrm=layout.normals)       # too few channels
    with pytest.raises(ValueError):
        OnlineTactileProcessor(layout, None, None, contact_rule="nope")
    with pytest.raises(ValueError):
        OnlineTactileProcessor(layout, None, None, pressure={"adc_mn": 0})


def test_processor_holds_the_last_joint_reading_on_a_driver_glitch():
    """A NaN joint reading must not poison the causal baseline / qd windows (NaN mean → NaN z → level
    NONE for W ticks): the processor holds the last finite reading, exactly as if it had been sent."""
    from robot_skin.baseline.temporal import TemporalBaselinePredictor
    from robot_skin.control import OnlineTactileProcessor
    from robot_skin.control.interfaces import load_urdf_model

    model, _ = load_urdf_model(None)
    torch.manual_seed(0)
    net = TemporalBaselinePredictor(9, 16, window=4, hidden=8, head_hidden=8, n_layers=2)
    torch.nn.init.normal_(net.mean_head.weight, std=0.5)
    net.eval()
    kw = dict(raw_order="layout", baseline_raw=np.full(9, 1.0e6), urdf=model)
    a, b = (OnlineTactileProcessor("robot_hand_template", net, None, **kw) for _ in range(2))
    assert a.joint_names == list(model.joint_names)                     # URDF order by default
    raw, q0 = np.full(9, 1.0e6), np.linspace(0.0, 0.6, 16)
    q1 = q0.copy()
    q1[4] = np.nan
    a.step(raw, q0), b.step(raw, q0)
    fa, fb = a.step(raw, q1), b.step(raw, q0)
    assert np.isfinite(fa.baseline_mean).all() and np.isfinite(fa.qd).all()
    np.testing.assert_allclose(fa.baseline_mean, fb.baseline_mean, atol=1e-6)
    np.testing.assert_allclose(fa.pos, fb.pos, atol=1e-6)
    with pytest.raises(ValueError, match="non-finite"):
        OnlineTactileProcessor("robot_hand_template", net, None, **kw).step(raw, q1)


def test_baseline_capture_window_is_real_time_when_rail_samples_are_skipped():
    """Without timestamps a sample's time is its tick index · dt — skipped (rail) samples still take
    time, so the window covers the same span as preprocessing's master-clock window."""
    from robot_skin.control import OnlineTactileProcessor

    proc = OnlineTactileProcessor("robot_hand_template", None, None, raw_order="layout", baseline_s=0.05)
    proc.begin_baseline()
    kept = []
    for i in range(40):                                                 # 200 Hz: 0.05 s = 10 ticks
        r = np.full(9, 1.0e6 + i)                                       # drifting raw
        if 5 <= i < 25:
            r[0] = ADC_MIN                                              # rail → skipped
        elif i * 0.005 < 0.05:
            kept.append(1.0e6 + i)
        proc.add_baseline_sample(r)
    np.testing.assert_array_equal(proc.finish_baseline(), np.median(kept))   # ticks 0-4 only


def test_startup_calibrator_matches_noise_level():
    from robot_skin.control import OnlineTactileProcessor, startup_calibrator
    from robot_skin.representation import TactileFeatureSpec

    rng = np.random.default_rng(0)
    r = rng.normal(scale=0.3, size=(400, 9)).astype(np.float32)
    cal = startup_calibrator(r, np.zeros_like(r, bool))
    np.testing.assert_allclose(cal.sigma, 0.3, rtol=0.2)
    assert cal.info["source"] == "startup_hold"
    layout = load_layout("robot_hand_template")
    proc = OnlineTactileProcessor(layout, None, None, feature_spec=TactileFeatureSpec(), raw_order="layout",
                                  baseline_raw=np.full(9, 1e6))
    proc.set_calibrator(cal)
    fr = proc.step(np.full(9, 1e6) * (1 - 0.02), pos=layout.positions, nrm=layout.normals)   # 2 % press
    assert (fr.level == ContactLevel.WEAK).all() and fr.features.shape == (9, 6) and fr.any_contact
    with pytest.raises(ValueError):
        proc.set_calibrator(startup_calibrator(r[:, :4]))


# ─────────────────────────────────────────────────────────────── safety

def test_safety_limits_rates_and_nonfinite():
    from robot_skin.control import SafetyFilter

    s = SafetyFilter([-1, -1], [1, 1], dt=0.01, max_vel=10.0, max_acc=2000.0, margin=0.05)
    s.reset([0.0, 0.0])
    q = s.filter([5.0, -0.05], t=0.0)
    np.testing.assert_allclose(q, [0.1, -0.05])                         # |Δq| ≤ max_vel·dt
    for i in range(20):
        q = s.filter([5.0, -5.0], t=0.01 * (i + 1))
    np.testing.assert_allclose(q, [0.95, -0.95])                        # limits − margin
    q2 = s.filter([np.nan, -0.9], t=1.0)
    assert q2[0] == q[0] and s.counts["nonfinite_target"] == 1
    assert s.counts["vel_clamp"] > 0 and s.counts["limit_clamp"] > 0
    a = SafetyFilter([-10], [10], dt=0.01, max_vel=100.0, max_acc=100.0)
    a.reset([0.0])
    v = np.diff([0.0] + [float(a.filter([10.0], t=i * 0.01)[0]) for i in range(5)]) / 0.01
    assert np.all(np.diff(np.r_[0.0, v]) <= 100.0 * 0.01 + 1e-9)       # |Δv| ≤ max_acc·dt
    with pytest.raises(ValueError):
        SafetyFilter([0, 0], [1, -1], dt=0.01)
    with pytest.raises(ValueError):
        SafetyFilter([0], [1], dt=0.01, tactile_stop={"mode": "panic"})
    with pytest.raises(ValueError):
        s.filter([0.0], t=0.0)


def test_safety_tactile_stop_freezes_closing_only_on_the_triggering_chain():
    from robot_skin.control import SafetyFilter, taxel_joint_mask
    from robot_skin.control.interfaces import load_urdf_model

    model, _ = load_urdf_model(None)
    layout = load_layout("robot_hand_template")
    mask = taxel_joint_mask(layout, model)
    names = list(model.joint_names)
    assert mask[1, names.index("index_dip")] and not mask[1, names.index("middle_dip")]
    assert mask[5].all()                                                 # palm taxel → every joint
    D = len(names)
    s = SafetyFilter(model.lower, model.upper, dt=0.005, tactile_stop={"min_ticks": 3, "release_ticks": 2},
                     taxel_joints=mask, joint_names=names, closing_sign={"thumb_cmc_abd": 0})
    q0 = np.full(D, 0.5)
    s.reset(q0)
    hot = np.zeros(9, np.int8)
    hot[1] = ContactLevel.STRONG                                         # index fingertip
    q = q0
    for i in range(3):
        q = s.filter(q + 0.1, t=i * 0.005, level=hot)
    assert s.stop_active and s.counts["tactile_stop_on"] == 1
    idx = [names.index(n) for n in ("index_mcp", "index_pip", "index_dip")]
    frozen = q[idx].copy()
    q_close = s.filter(q + 0.1, t=0.02, level=hot)                       # closing: index frozen, others move
    np.testing.assert_array_equal(q_close[idx], frozen)
    assert q_close[names.index("middle_mcp")] > q[names.index("middle_mcp")]
    q_open = s.filter(q - 0.1, t=0.025, level=hot)                       # opening is allowed
    assert np.all(q_open[idx] < frozen)
    calm = np.zeros(9, np.int8)
    s.filter(q_open, t=0.03, level=calm)
    s.filter(q_open, t=0.035, level=calm)
    assert not s.stop_active and s.counts["tactile_stop_off"] == 1
    types = [e.type for e in s.events]
    assert types == ["tactile_stop_on", "tactile_stop_off"]
    hold = SafetyFilter(model.lower, model.upper, dt=0.005, tactile_stop={"min_ticks": 1, "mode": "hold"})
    hold.reset(q0)
    np.testing.assert_array_equal(hold.filter(q0 - 0.2, t=0.0, level=hot), q0)   # hold: nothing moves
    called = []
    est = SafetyFilter(model.lower, model.upper, dt=0.005, tactile_stop={"min_ticks": 1, "mode": "estop"},
                       estop_callback=lambda r, t: called.append((r, t)))
    est.reset(q0)
    np.testing.assert_array_equal(est.filter(q0 + 0.2, t=0.0, level=hot), q0)
    assert est.estopped and called == [("tactile stop", 0.0)]
    np.testing.assert_array_equal(est.filter(q0 - 0.2, t=1.0), q0)      # latched until reset
    est.reset(q0)
    assert not est.estopped
    # a tactile e-stop holds the *measured* position: the servo lags its last command (q0), holding
    # that command would keep squeezing by the lag
    q_meas = q0 - 0.05
    np.testing.assert_allclose(est.filter(q0 + 0.2, q_meas, t=2.0, level=hot), q_meas)
    np.testing.assert_allclose(est.filter(q0 + 0.2, q0, t=2.1), q_meas)


def test_safety_watchdog_holds_then_estops():
    from robot_skin.control import SafetyFilter

    s = SafetyFilter([-1], [1], dt=0.01, watchdog={"max_age_s": {"pressure": 0.05, "camera_*": 0.5},
                                                   "estop_after_s": 0.1})
    s.reset([0.0])
    assert s.filter([0.5], t=1.0, stamps={"pressure": 0.99, "camera_ego": 0.8})[0] == 0.5
    q = s.filter([0.9], t=1.1, stamps={"pressure": 1.0, "camera_ego": 1.05})   # pressure 100 ms old → hold
    assert q[0] == 0.5 and s.stale and s.events[-1].type == "stale_on"
    s.filter([0.9], t=1.12, stamps={"pressure": 1.11})
    assert not s.stale and s.events[-1].type == "stale_off"
    s.filter([0.9], t=2.0, stamps={"pressure": 1.0})
    s.filter([0.9], t=2.2, stamps={"pressure": 1.0})
    assert s.estopped and "stale" in s.estop_reason
    assert s.summary()["estop"] is True


# ─────────────────────────────────────────────────────────────── fake hardware

def test_fake_robot_hand_tracks_touches_and_drops_out():
    from robot_skin.control import FakeCamera, FakeRobotHand, check_robot
    from robot_skin.control.interfaces import RobotHandInterface

    clock = SimClock(0.0)
    hand = FakeRobotHand(clock=clock, obj={"angle": 0.6, "compliance_rad": 0.2}, seed=1)
    check_robot(hand)
    assert isinstance(hand, RobotHandInterface) and hand.n_channels == 9 and len(hand.joint_names) == 16
    t0, raw0 = hand.read_pressure()
    t, q, qd = hand.read_state()
    assert t == t0 == 0.0 and np.allclose(q, 0.0)
    target = np.clip(np.full(16, 1.2), hand.lower, hand.upper)
    hand.send_joint_targets(target)
    for _ in range(100):                                                 # 0.5 s at 200 Hz
        clock.advance(0.005)
    tr = hand.truth()
    cl = hand.closure()
    assert tr["t"] == pytest.approx(0.5)
    for f in ("index", "middle", "ring", "pinky"):
        assert cl[f] == pytest.approx(0.8, abs=1e-6)                     # stopped by the object (0.6 + 0.2)
        assert tr["penetration_rad"][f] == pytest.approx(0.2, abs=1e-6)
    assert tr["contact"][[1, 2, 3, 4]].all() and tr["contact"][5:].all()  # fingertips + palm (power grasp)
    _, raw = hand.read_pressure()
    base = hand.baseline_raw[hand.layout.channels]
    d = relative_change(hand.layout.by_channel(raw), base)
    assert (d[1:5] < -5.0).all()                                         # press → negative ΔS
    hand.inject_dropout(2, 0.05)
    _, raw = hand.read_pressure()
    assert raw[hand.layout.channels[2]] == ADC_MIN
    clock.advance(0.1)
    assert hand.read_pressure()[1][hand.layout.channels[2]] > ADC_MIN
    with pytest.raises(ValueError):
        hand.send_joint_targets(np.full(16, np.nan))
    # free motion: artefact only, reproducible for a seed
    runs = []
    for _ in range(2):
        c2 = SimClock(0.0)
        h2 = FakeRobotHand(clock=c2, obj=None, seed=4)
        h2.send_joint_targets(np.clip(np.full(16, 1.0), h2.lower, h2.upper))
        c2.advance(0.3)
        runs.append((h2.read_pressure()[1], h2.truth()))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    assert not runs[0][1]["contact"].any() and np.abs(runs[0][1]["artefact_pct"]).max() > 0.2
    cam = FakeCamera("ego", hand, hw=(12, 16), rate_hz=10.0)
    tc, f1 = cam.read()
    assert f1.shape == (12, 16, 3) and f1.dtype == np.uint8 and tc == pytest.approx(0.6)
    with pytest.raises(TypeError):
        check_robot(object())
    # the simulation keeps up with the caller's clock exactly, also at a host-like clock origin where an
    # accumulated float time falls one integration step behind within a few ticks
    c3 = SimClock(12345.678)
    h3 = FakeRobotHand(clock=c3, obj=None, seed=0)
    for _ in range(50):
        c3.advance(0.005)
        assert h3.read_state()[0] == pytest.approx(c3(), abs=1e-9)


# ─────────────────────────────────────────────────────────────── bundle & latency

def test_load_policy_bundle_and_deployability(bundles):
    from robot_skin.control import PolicyBundle, load_policy_bundle

    b = load_policy_bundle(bundles["joint"])
    assert isinstance(b, PolicyBundle) and b.action_kind == "robot_joint" and b.action_dim == 16
    assert b.stride == 10 and b.cameras == ("ego",) and b.ensemble_k == 0.01 and b.uses_tactile
    assert b.calibrator().n_taxels == 9 and b.baseline_model_path() is None
    assert b.summary()["obs_mode"] == "full"
    b.check_deployable()
    with pytest.warns(UserWarning, match="bootstrap"):
        boot = load_policy_bundle(bundles["boot"])
    with pytest.raises(ValueError, match="bootstrap"):
        boot.check_deployable()
    with pytest.raises(ValueError, match="bootstrap"):
        load_policy_bundle(bundles["boot"], allow_bootstrap=False)


def test_latency_tools(bundles, tmp_path):
    from robot_skin.baseline.temporal import TemporalBaselinePredictor
    from robot_skin.control import LatencyMeter, benchmark_policy, export_torchscript, load_policy_bundle
    from robot_skin.control.latency import percentile_summary

    s = percentile_summary([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["p50_ms"] == 2.5 and s["max_ms"] == 4.0
    assert np.isnan(percentile_summary([])["p50_ms"])
    m = LatencyMeter()
    with m.time():
        sum(range(100))
    assert len(m) == 1 and m.summary()["p50_ms"] >= 0
    res = benchmark_policy(load_policy_bundle(bundles["joint"]), n=3, warmup=1)
    assert res["n"] == 3 and res["p95_ms"] >= res["p50_ms"] > 0 and res["batch_size"] == 1
    net = TemporalBaselinePredictor(9, 4, window=8, hidden=8, head_hidden=8, n_layers=2).eval()
    ex = (torch.randn(1, 8, 4), torch.randn(1, 8, 4), torch.randn(1, 9, 3), torch.randn(1, 9, 3))
    p = export_torchscript(net, ex, tmp_path / "baseline.ts")
    assert p.is_file()
    with pytest.raises(RuntimeError, match="TorchScript"):
        export_torchscript(load_policy_bundle(bundles["joint"]).policy, ({"x": 1},), tmp_path / "policy.ts")


# ─────────────────────────────────────────────────────────────── runner

def _fake_setup(bundle_path, *, obj=None, logger_dir=None, max_vel=3.0, tactile_stop=None, hand_state_fn=None):
    from robot_skin.control import (DeploymentLogger, FakeCamera, FakeRobotHand, OnlineTactileProcessor,
                                    PolicyRunner, SafetyFilter, load_policy_bundle, taxel_joint_mask)
    from robot_skin.stages.deploy import build_retargeter

    b = load_policy_bundle(bundle_path)
    clock = SimClock(0.0)
    hand = FakeRobotHand(clock=clock, obj=obj, seed=0)
    cams = {c: FakeCamera(c, hand) for c in b.cameras}
    proc = OnlineTactileProcessor.from_policy_bundle(b, hand.layout, urdf=hand.model, raw_order="channel")
    safety = SafetyFilter(hand.lower, hand.upper, dt=0.005, max_vel=max_vel, joint_names=hand.joint_names,
                          tactile_stop=tactile_stop, taxel_joints=taxel_joint_mask(hand.layout, hand.model),
                          watchdog={"max_age_s": {"pressure": 0.05, "joint_state": 0.05}})
    rt = build_retargeter(hand.model, hand.layout, {"iters": 5}, synthetic=True) \
        if b.action_kind == "hand_mano" else None
    logger = None
    if logger_dir is not None:
        logger = DeploymentLogger(logger_dir, layout=hand.layout, joint_names=hand.joint_names, clock=clock,
                                  cameras=list(cams), urdf_xml=hand.urdf_xml, instruction="grasp the cup")
    runner = PolicyRunner(hand, b, proc, cameras=cams, retargeter=rt, safety=safety, instruction="grasp the cup",
                          logger=logger, hand_state_fn=hand_state_fn)
    return runner, hand


def test_runner_robot_joint_loop_logs_a_reingestible_session(bundles, tmp_path):
    from robot_skin.control.runner import DEPLOY_LOG_NAME
    from robot_skin.datasets.build import preprocess_session

    sdir = tmp_path / "session"
    runner, hand = _fake_setup(bundles["joint"], logger_dir=sdir)
    m = runner.run(1.0, baseline_s=0.3)
    assert m["n_ticks"] == 200 and m["n_policy_ticks"] == 20 and m["policy_every"] == 10
    assert m["loop_hz"] == pytest.approx(200.0) and m["latency_p95_ms"] >= m["latency_p50_ms"] > 0
    assert not m["estop"] and m["startup"]["baseline_samples"] == 60 and m["startup"]["calibrator"] == "given"
    assert runner.ensembler.t == 20                                     # one ensembler step per policy tick
    log = np.load(sdir / DEPLOY_LOG_NAME)
    qc = log["tick_q_cmd"]
    assert np.all(np.abs(np.diff(qc, axis=0)) <= 3.0 * 0.005 + 1e-5)   # velocity limit respected
    assert np.all(qc >= hand.lower - 1e-6) and np.all(qc <= hand.upper + 1e-6)
    assert log["policy_action"].shape == (20, 16) and log["tick_level"].shape == (200, 9)
    ep = preprocess_session(sdir, None, {"cameras": {"copy_frames": "none"}, "baseline": {"duration_s": 0.2}})
    assert ep.meta.kind == "robot" and ep.meta.dataset == "other" and ep.meta.cameras == ["ego"]
    assert list(ep.meta.joint_names) == list(hand.joint_names) and ep.meta.phase_names == ["baseline", "rollout"]
    assert ep.meta.preprocessing["baseline"]["source"] == "no_contact_segment"
    assert ep.meta.task["instruction"] == "grasp the cup" and ep.meta.n_taxels == 9
    assert np.abs(np.asarray(ep["delta_pct"])[: 40]).max() < 1.0      # still, no contact at start-up
    assert ep["taxel_pos"].shape == (ep.T, 9, 3)
    # a second rollout restarts every stream
    runner.begin_rollout()
    assert runner.ensembler.t == 0 and runner.processor.n_steps == 0


def test_runner_metrics_describe_each_rollout(bundles):
    """run(startup=False) from a simulated clock at 0.0 (baseline given) and a second rollout: the
    loop counters restart with every rollout, the rate is measured from the rollout's own start."""
    from robot_skin.control import FakeRobotHand, OnlineTactileProcessor, PolicyRunner, load_policy_bundle

    b = load_policy_bundle(bundles["squeeze"])
    hand = FakeRobotHand(clock=SimClock(0.0), obj=None, seed=0)
    proc = OnlineTactileProcessor.from_policy_bundle(b, hand.layout, urdf=hand.model, raw_order="channel",
                                                     baseline_raw=hand.baseline_raw[hand.layout.channels])
    runner = PolicyRunner(hand, b, proc, instruction="x")
    m = runner.run(0.05, startup=False)
    assert m["n_ticks"] == 10 and m["n_policy_ticks"] == 1 and m["loop_hz"] == pytest.approx(200.0)
    m2 = runner.run(0.1, startup=False)
    assert m2["n_ticks"] == 20 and m2["n_policy_ticks"] == 2 and m2["loop_hz"] == pytest.approx(200.0)
    assert m2["inference_ms"]["n"] == 2


def test_runner_tactile_stop_on_a_squeezing_policy(bundles):
    """A policy that commands a closed hand (absolute targets 1.2 rad) hits the virtual object: the
    skin reports STRONG contact and the safety filter freezes further closing."""
    runner, hand = _fake_setup(bundles["squeeze"], obj={"angle": 0.5, "compliance_rad": 0.5}, max_vel=5.0,
                               tactile_stop={"min_ticks": 5, "release_ticks": 50})
    m = runner.run(1.0, baseline_s=0.2)
    assert m["safety_counts"].get("tactile_stop_on", 0) >= 1 and m["contact_frac"] > 0.2
    first = next(e for e in m["safety"]["events"] if e["type"] == "tactile_stop_on")
    assert first["t"] < 0.2 + 1.0
    cl = hand.closure()
    assert max(cl.values()) < 0.5 + 0.25                                # frozen soon after touching (0.5)
    assert m["safety_counts"].get("tactile_frozen_ticks", 0) > 0


def test_runner_hand_mano_retargets_and_bundle_checks(bundles):
    from robot_skin.control import PolicyRunner, load_policy_bundle

    runner, hand = _fake_setup(bundles["hand"])
    m = runner.run(0.5, baseline_s=0.2)
    assert m["action_kind"] == "hand_mano" and m["n_policy_ticks"] == 10 and "retarget_ms" in m
    assert runner._hand_state.shape == (54,)
    q = hand.truth()["target"]
    assert np.all(q >= hand.lower - 1e-9) and np.all(q <= hand.upper + 1e-9)
    with pytest.raises(ValueError, match="Retargeter"):
        PolicyRunner(hand, load_policy_bundle(bundles["hand"]), runner.processor, cameras=runner.cameras)
    with pytest.raises(ValueError, match="cameras"):
        PolicyRunner(hand, load_policy_bundle(bundles["joint"]), runner.processor)


class _DriverOrderHand:
    """A robot driver that reports joints in reverse URDF order and measures no velocities."""

    def __init__(self, hand):
        self.hand, self.clock, self.layout = hand, hand.clock, hand.layout
        self.joint_names = tuple(reversed(hand.joint_names))
        self.lower, self.upper = hand.lower[::-1].copy(), hand.upper[::-1].copy()

    def read_state(self):
        t, q, _ = self.hand.read_state()
        return t, q[::-1].copy(), None

    def read_pressure(self):
        return self.hand.read_pressure()

    def send_joint_targets(self, q):
        self.hand.send_joint_targets(np.asarray(q)[::-1].copy())


def test_runner_maps_driver_joint_order_and_logs_only_measured_qd(bundles, tmp_path):
    """Without a baseline model the URDF pose function still gets URDF-ordered q (the driver order
    differs), and a driver without velocities logs no fabricated qd; the session re-ingests."""
    from robot_skin.control import (DeploymentLogger, FakeRobotHand, OnlineTactileProcessor, PolicyRunner,
                                    SafetyFilter, load_policy_bundle)
    from robot_skin.datasets.build import preprocess_session
    from robot_skin.pose.robot_fk import taxel_poses_from_joints

    b = load_policy_bundle(bundles["squeeze"])                               # no cameras, no baseline model
    clock = SimClock(0.0)
    hand = FakeRobotHand(clock=clock, obj=None, seed=0, q0=np.linspace(0.0, 0.8, 16))
    rob = _DriverOrderHand(hand)
    proc = OnlineTactileProcessor.from_policy_bundle(b, hand.layout, urdf=hand.model, raw_order="channel")
    assert proc.joint_names == list(hand.model.joint_names)
    sdir = tmp_path / "s"
    logger = DeploymentLogger(sdir, layout=hand.layout, joint_names=rob.joint_names, clock=clock,
                              urdf_xml=hand.urdf_xml)
    runner = PolicyRunner(rob, b, proc, safety=SafetyFilter(rob.lower, rob.upper, dt=0.005, max_vel=3.0,
                                                            joint_names=rob.joint_names),
                          instruction="x", logger=logger)
    m = runner.run(0.1, baseline_s=0.1)                                     # closes the log
    assert m["n_ticks"] == 20
    runner.begin_rollout()
    frame = runner.step()
    _, q, _ = hand.read_state()
    p_true, _ = taxel_poses_from_joints(hand.layout, hand.model, q[None])
    np.testing.assert_allclose(frame.pos, p_true[0], atol=1e-5)
    js = np.load(sdir / "joint_state.npz")
    assert "qd" not in js.files and list(js["names"]) == list(rob.joint_names)
    assert np.all(np.diff(js["t"]) > 0) and np.all(np.diff(np.load(sdir / "pressure.npz")["t"]) > 0)
    ep = preprocess_session(sdir, None, {"baseline": {"duration_s": 0.05}})
    assert list(ep.meta.joint_names) == list(hand.model.joint_names)          # PRE reorders by name (URDF)


def test_runner_honours_chunk_offset_and_blanks_cameras_before_their_first_frame(bundles):
    """chunk[i] is the action at policy tick t + offset + i: the ensembler must yield the next tick's
    target (offset 0 → chunk[1]). A camera without a frame yet gets the dataset's stand-in (eval
    transform of a zero image, vision_valid False) instead of stopping the loop."""
    from common.signal import NormStats
    from robot_skin.action import ActionNormalizer
    from robot_skin.control import FakeCamera, FakeRobotHand, OnlineTactileProcessor, PolicyRunner, load_policy_bundle

    class LateCamera:
        def __init__(self, name, hand, n_none):
            self.name, self.cam, self.n_none, self.n = name, FakeCamera(name, hand), n_none, 0

        def read(self):
            self.n += 1
            t, f = self.cam.read()
            return (t, None) if self.n <= self.n_none else (t, f)

    steps = {}
    for offset in (1, 0):
        b = load_policy_bundle(bundles["joint"])                             # robot_joint, delta, camera "ego"
        b.chunk_offset = offset
        b.action_normalizer = ActionNormalizer(NormStats(np.zeros(16, np.float32), np.ones(16, np.float32)),
                                               b.action_spec)
        batches = []

        def predict(batch, n_steps=None, *, generator=None, noise=None, _b=b):
            batches.append(batch)
            return 0.01 * torch.arange(1, _b.horizon + 1, dtype=torch.float32)[None, :, None].expand(1, _b.horizon, 16)

        b.policy.predict = predict
        clock = SimClock(0.0)
        hand = FakeRobotHand(clock=clock, obj=None, seed=0)
        proc = OnlineTactileProcessor.from_policy_bundle(b, hand.layout, urdf=hand.model, raw_order="channel")
        runner = PolicyRunner(hand, b, proc, cameras={"ego": LateCamera("ego", hand, 1)}, instruction="x")
        runner.startup(0.05)
        runner.begin_rollout()
        batches.clear()
        _, q, _ = hand.read_state()
        runner.step()
        steps[offset] = runner._target_next - q                              # delta actions → next target − q
        vv = batches[0]["vision_valid"]["ego"]
        assert not bool(vv.reshape(-1)[0])
        blank = runner.eval_transform(np.zeros((1, 24, 32, 3), np.uint8))[0]
        torch.testing.assert_close(batches[0]["images"]["ego"][0], blank)
    np.testing.assert_allclose(steps[1], 0.01, atol=1e-6)                     # chunk[0]
    np.testing.assert_allclose(steps[0], 0.02, atol=1e-6)                     # chunk[1]: chunk[0] is "now"


# ─────────────────────────────────────────────────────────────── deploy stage

def test_deploy_stage_yaml_mirrors_defaults_and_validates():
    from robot_skin.stages import deploy

    assert yaml.safe_load(deploy.CONFIG_PATH.read_text()) == deploy.DEFAULTS
    cfg = deploy.load_stage_config(overrides={"duration_s": 2.0, "safety": {"max_vel": 1.0}})
    assert cfg["duration_s"] == 2.0 and cfg["safety"]["max_vel"] == 1.0 and cfg["safety"]["margin"] == 0.0
    with pytest.raises(ValueError, match="unknown"):
        deploy.load_stage_config(overrides={"safety": {"max_velocity": 1.0}})
    with pytest.raises(ValueError, match="unknown"):
        deploy.resolve_config({"robots": "fake"})
    with pytest.raises(NotImplementedError, match="RobotHandInterface"):
        deploy.run({"robot": "allegro", "bundle": "x"})
    with pytest.raises(ValueError, match="bundle"):
        deploy.run({"robot": "fake"})


def test_deploy_stage_fake_robot_end_to_end(bundles, tmp_path):
    from robot_skin.stages import deploy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = deploy.run({"bundle": str(bundles["joint"]), "duration_s": 0.5, "out_dir": str(tmp_path / "dep"),
                        "instruction": "pick up the cup", "startup": {"baseline_s": 0.2, "calib_s": 0.0},
                        "latency": {"n": 2, "warmup": 1}})
    assert m["stage"] == "deploy" and m["robot"] == "fake" and m["n_ticks"] == 100
    assert m["loop_hz"] == pytest.approx(200.0) and m["latency_p50_ms"] > 0 and m["benchmark"]["n"] == 2
    assert (tmp_path / "dep" / "metrics.json").is_file()
    assert (tmp_path / "dep" / "sessions" / "deploy_0" / "session.json").is_file()
    assert isinstance(m["safety_counts"], dict) and m["estop"] is False
    # a bundle trained on another skin: its calibrator is not used; bring-up calibration instead
    other = make_bundle(tmp_path / "other", cams=(), layouts=["glove_template"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2 = deploy.run({"bundle": str(other), "duration_s": 0.2, "out_dir": str(tmp_path / "dep2"),
                         "startup": {"baseline_s": 0.2, "calib_s": 0.3}, "log": {"enabled": False},
                         "latency": {"benchmark": False}})
    assert m2["startup"]["calibrator"] == "startup" and any("glove_template" in n for n in m2["notes"])
    assert "session_dir" not in m2
    # the same guard in OnlineTactileProcessor.from_policy_bundle
    from robot_skin.control import FakeRobotHand, OnlineTactileProcessor, load_policy_bundle

    hand = FakeRobotHand(clock=SimClock(0.0), obj=None)
    with pytest.warns(UserWarning, match="stage-1 references"):
        p_other = OnlineTactileProcessor.from_policy_bundle(load_policy_bundle(other), hand.layout, urdf=hand.model)
    assert p_other.calibrator is None
    # this skin's bundle, but its calibrator was fitted with the baseline log-variance and the baseline
    # model file is missing (bundle copied to the robot PC alone): not used — a start-up calibrator
    # stands in instead of the rollout failing on its first tick
    lv = make_bundle(tmp_path / "lv", cams=(), layouts=["robot_hand_template"], cal_logvar=True,
                     baseline_ref=str(tmp_path / "missing_baseline"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m3 = deploy.run({"bundle": str(lv), "duration_s": 0.1, "out_dir": str(tmp_path / "dep3"),
                         "startup": {"baseline_s": 0.1, "calib_s": 0.2}, "log": {"enabled": False},
                         "latency": {"benchmark": False}})
        p_lv = OnlineTactileProcessor.from_policy_bundle(load_policy_bundle(lv), hand.layout, urdf=hand.model)
    assert m3["startup"]["calibrator"] == "startup" and m3["n_ticks"] == 20
    assert any("log-variance" in n for n in m3["notes"]) and any("not found" in n for n in m3["notes"])
    assert p_lv.calibrator is None
    mine = make_bundle(tmp_path / "mine", cams=(), layouts=["robot_hand_template"])
    assert OnlineTactileProcessor.from_policy_bundle(load_policy_bundle(mine), hand.layout,
                                                     urdf=hand.model).calibrator is not None
