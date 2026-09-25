"""Stage-1 runners end to end on a tiny synthetic glove dataset: imu_pose → baseline → contact.

Data: three 6 s synthetic glove sessions (``datasets.synthetic``) → ``datasets.build`` episodes.
All three use the **same generator seed** (two motion sessions of different subjects + one task
session): the generator draws the per-taxel artefact physics (gains, signs, lags) from the session
seed, so a shared seed plays "one physical glove" and the baseline stage can be checked on held-out
data (train on subject s0, validate on s1, predict the D2 task). ``generate_dataset`` gives every
session its own seed, i.e. a different glove per session, which no baseline model can generalise
across. Caveat: the motion RNG is seeded by the session seed too, so s1 replays s0's motion plan with
the subject's amplitude / speed style (time-warped, rescaled) — the D2 task episode (reach / grasp /
manipulate, different joint trajectories) is the genuinely novel-motion generalisation check.
"""
import json
import shutil

import numpy as np
import pytest
import torch
import yaml

from robot_skin.datasets.episode import (D_BASELINE_LOGVAR, D_BASELINE_PRED, D_CONTACT_PROB, D_HAND_POSE_IMU,
                                         D_LEVEL, D_RESIDUAL, D_RESIDUAL_Z, Episode, list_episodes)
from robot_skin.stages import baseline as st_baseline
from robot_skin.stages import check_stage_keys, finite_json, parse_overrides
from robot_skin.stages import contact as st_contact
from robot_skin.stages import imu_pose as st_imu

TRAIN = {"batch_size": 64, "lr": 3e-3, "warmup_steps": 10, "device": "cpu", "precision": "fp32",
         "log_every": 1000, "seed": 0}


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    from robot_skin.datasets.build import build_all
    from robot_skin.datasets.synthetic import generate_session

    n = torch.get_num_threads()
    torch.set_num_threads(1)                   # tiny models: avoid thread contention on shared CI boxes
    root = tmp_path_factory.mktemp("stage1")
    for ds, subj in (("motion", "s0"), ("motion", "s1"), ("task", "s0")):
        generate_session(root / "raw" / ds / subj / f"{ds}_{subj}", kind="glove", dataset=ds, duration_s=6.0,
                         seed=7, subject=subj, cameras=("ego",))
    rep = build_all(root / "raw", root / "processed", {"cameras": {"copy_frames": "none"}})
    assert [r["status"] for r in rep] == ["built"] * 3
    labels0 = {d.name: np.array(Episode.load(d)["contact_label"]) for d in list_episodes(root / "processed")}
    yield root, labels0
    torch.set_num_threads(n)
    shutil.rmtree(root, ignore_errors=True)


def _data(root, **kw):
    return {"processed_root": str(root / "processed"), "val_frac": 0.5, "split_seed": 0, **kw}


@pytest.fixture(scope="module")
def pipeline(processed):
    root, _ = processed
    m_imu = st_imu.run({"data": _data(root), "model": {"window": 16, "hidden": 32, "n_layers": 1},
                        "train": {**TRAIN, "max_steps": 60}, "out_dir": str(root / "runs" / "imu_pose")})
    m_base = st_baseline.run({"data": _data(root),
                              "model": {"window": 32, "hidden": 32, "head_hidden": 32, "n_layers": 4},
                              "train": {**TRAIN, "max_steps": 120}, "out_dir": str(root / "runs" / "baseline")})
    m_con = st_contact.run({"data": _data(root, stride=2), "detector": {"window": 16, "hidden": 16},
                            "train": {**TRAIN, "max_steps": 120}, "out_dir": str(root / "runs" / "contact")})
    return {"imu_pose": m_imu, "baseline": m_base, "contact": m_con}


def _episodes(root):
    return {d.name: Episode.load(d) for d in list_episodes(root / "processed")}


# ───────────────────────────────────────────────────────────── configs

@pytest.mark.parametrize("stage", [st_imu, st_baseline, st_contact])
def test_stage_yaml_mirrors_defaults(stage):
    assert yaml.safe_load(stage.CONFIG_PATH.read_text()) == stage.DEFAULTS
    assert stage.DEFAULTS["stage"] == stage.STAGE


def test_stage_config_loading_hardware_and_validation(tmp_path):
    cfg = st_baseline.load_stage_config(overrides={"hardware": "cpu", "train": {"lr": 1e-4}})
    assert cfg["hardware_applied"] and cfg["train"]["device"] == "cpu"
    assert cfg["train"]["batch_size"] == 256 and cfg["train"]["lr"] == 1e-4     # cpu profile suggest.baseline
    ov = st_contact.load_stage_config(overrides=parse_overrides(["train.batch_size=32", "hardware=cpu"]))
    assert ov["train"]["batch_size"] == 32                                     # overrides beat the profile
    with pytest.raises(ValueError, match="unknown"):
        st_imu.load_stage_config(overrides={"modle": {}})
    with pytest.raises(ValueError, match="unknown keys"):
        st_contact.resolve_config({"detector": {"windw": 3}})
    check_stage_keys({"train": {"anything": 1}}, st_contact.DEFAULTS, "contact")   # train is open
    with pytest.raises(FileNotFoundError):
        st_baseline.load_stage_config(tmp_path / "missing.yaml")
    assert finite_json({"a": float("nan"), "b": np.float32(2.0), "c": (np.int64(3),)}) == {"a": None, "b": 2.0,
                                                                                          "c": [3]}


# ───────────────────────────────────────────────────────────── pipeline

def test_imu_pose_stage(processed, pipeline):
    root, _ = processed
    m = pipeline["imu_pose"]
    for k in ("val/rot_deg", "val/tip_mm", "val/rot_deg_flat", "n_samples", "best", "model_path"):
        assert k in m
    assert m["val/rot_deg"] < m["val/rot_deg_flat"]                           # learned something
    for ep in _episodes(root).values():
        fp = ep.derived(D_HAND_POSE_IMU)
        assert fp.shape == (ep.T, 15, 3) and fp.dtype == np.float32 and np.isfinite(fp).all()
    model = st_imu.load_imu_pose_model(m["model_path"])
    assert model.bundle_meta["window"] == 16 and model.bundle_meta["features"]["gyro"]
    assert json.loads((root / "runs" / "imu_pose" / "metrics.json").read_text())["stage"] == "imu_pose"


def test_baseline_stage_removes_the_motion_artefact(processed, pipeline):
    from robot_skin.baseline import CausalBaselineStream, load_baseline_model, predict_episode

    root, _ = processed
    m = pipeline["baseline"]
    for k in ("val/nll", "val/mae_raw", "val/mae_resid", "val/resid_reduction", "val/coverage_2sigma",
              "val/sep_auroc_before", "val/sep_auroc_after", "val/gt_artefact_mae", "task/resid_reduction"):
        assert k in m, k
    assert m["n_episodes"]["train"] == 1 and m["n_episodes"]["val"] == 1 and m["n_episodes"]["predicted"] == 3
    # held-out subject, same glove: most of the no-contact artefact is predicted ...
    assert m["val/resid_reduction"] > 0.4, m["val/resid_reduction"]
    # ... and it is the true artefact (synthetic ground truth), not noise fitting
    assert m["val/gt_artefact_mae"] < 0.6 * m["val/gt_artefact_abs"]
    assert m["task/gt_artefact_mae"] < 0.75 * m["task/gt_artefact_abs"]       # transfers to the D2 task (novel motion)
    assert m["val/sep_auroc_after"] >= m["val/sep_auroc_before"] - 0.02
    eps = _episodes(root)
    for ep in eps.values():
        pred, lv, res = ep.derived(D_BASELINE_PRED), ep.derived(D_BASELINE_LOGVAR), ep.derived(D_RESIDUAL)
        assert pred.shape == lv.shape == res.shape == (ep.T, ep.meta.n_taxels)
        assert np.allclose(res, np.asarray(ep["delta_pct"]) - pred, atol=1e-5)
        assert np.isfinite(lv).all()
    # the saved bundle reproduces the derived arrays offline and online (causal stream) with the
    # CTRL recipe: qd = joint_velocity(q ring buffer, hz, **bundle_meta["qd"])
    from robot_skin.datasets.build import joint_velocity

    model = load_baseline_model(m["model_path"])
    meta = model.bundle_meta
    assert meta["q_source"] == "q" and meta["qd_source"] == "derivative"
    assert meta["qd"] == {"method": "savgol_causal", "window_s": 0.05, "polyorder": 2}     # joint_velocity kwargs only
    ep = eps["syn_glove_motion_s1_7"]
    q = np.asarray(ep["q"])
    assert np.allclose(joint_velocity(q, meta["hz"], **meta["qd"]), ep["qd"], atol=1e-4)
    mean, lv = predict_episode(model, ep)
    assert np.allclose(mean, ep.derived(D_BASELINE_PRED), atol=1e-5)
    stream = CausalBaselineStream(model)
    for t in range(40):
        qd_t = joint_velocity(q[max(0, t - 15):t + 1], meta["hz"], **meta["qd"])[-1]         # 16-frame ring
        ms, _ = stream.push(q[t], qd_t, ep["taxel_pos"][t], ep["taxel_nrm"][t])
        assert np.allclose(ms, mean[t], atol=1e-4)


def test_contact_stage_outputs_and_metrics(processed, pipeline):
    from robot_skin.contact import D_CONTACT_LABEL_PSEUDO, ResidualCalibrator, load_detector
    from robot_skin.contact.calibration import residual_levels

    root, labels0 = processed
    m = pipeline["contact"]
    for k in ("val/z_hallucination_taxel", "val/z_auroc", "val/prob_auroc", "val/prob_hallucination_taxel",
              "val/hyst_recall", "val/prob_gt_auroc", "pseudo/n_contact", "pseudo/gt_recall", "calibrator"):
        assert k in m, k
    assert m["calibration_split"] == "val" and m["contact_prob_source"] == "detector"
    assert m["val/z_auroc"] > 0.9 and m["val/prob_auroc"] > 0.9
    assert m["val/prob_hallucination_taxel"] < 0.1
    assert m["pseudo/n_episodes"] == 1 and m["pseudo/n_contact"] > 0
    cal = ResidualCalibrator.load(root / "runs" / "contact" / "calibrator.json")
    det = load_detector(root / "runs" / "contact" / "contact_detector.pt")
    assert det.bundle_meta["hysteresis"]["min_off"] == 4
    for name, ep in _episodes(root).items():
        z, lv, p = ep.derived(D_RESIDUAL_Z), ep.derived(D_LEVEL), ep.derived(D_CONTACT_PROB)
        assert z.shape == lv.shape == p.shape == (ep.T, ep.meta.n_taxels)
        assert z.dtype == np.float32 and lv.dtype == np.int8 and set(np.unique(lv)) <= {0, 1, 2, 3}
        assert np.all((p >= 0) & (p <= 1))
        assert np.array_equal(np.asarray(ep["contact_label"]), labels0[name])       # array never modified
        ref = residual_levels(cal, ep.derived(D_RESIDUAL), np.asarray(ep["saturated"]),
                              ep.derived(D_BASELINE_LOGVAR), dt=1 / ep.meta.hz)
        assert np.allclose(ref["residual_z"], z, equal_nan=True) and np.array_equal(ref["contact_level"], lv)
        if ep.meta.dataset == "task":
            pl = ep.derived(D_CONTACT_LABEL_PSEUDO)
            assert pl.dtype == np.int8 and set(np.unique(pl)) <= {-1, 0, 1} and (pl == 1).any()
            known = labels0[name] >= 0
            assert np.array_equal(pl[known], labels0[name][known])                   # certain labels kept
        else:
            assert not ep.has_derived(D_CONTACT_LABEL_PSEUDO)
    txt = (root / "runs" / "contact" / "metrics.json").read_text()
    json.loads(txt, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))


def test_baseline_from_imu_pose_and_no_detector(processed, pipeline):
    """Camera-free variant: the baseline reads the IMU pose model's finger pose; the contact stage
    without a detector falls back to sigmoid(z − weak_z)."""
    root, _ = processed
    m = st_baseline.run({"data": _data(root, q_source="hand_pose_imu"),
                         "model": {"window": 16, "hidden": 16, "head_hidden": 16, "n_layers": 3},
                         "predict": {"write_derived": False},
                         "train": {**TRAIN, "max_steps": 20}, "out_dir": str(root / "runs" / "baseline_imu")})
    assert m["q_source"] == "hand_pose_imu" and np.isfinite(m["val/mae_resid"])
    assert m["n_episodes"]["predicted"] == 0
    c = st_contact.run({"data": _data(root, q_source="hand_pose_imu"), "detector": {"enabled": False},
                        "fsm": {"enabled": True}, "pseudo_label": {"enabled": False},
                        "predict": {"write_derived": False}, "out_dir": str(root / "runs" / "contact_z")})
    assert c["contact_prob_source"] == "z_logistic" and "val/z_auroc" in c and "val/prob_auroc" not in c
    assert c["q_source"] == "hand_pose_imu" and c["n_episodes"]["predicted"] == 0 and c["val/z_auroc"] > 0.9
    from robot_skin.contact import ResidualCalibrator

    cal = ResidualCalibrator.load(root / "runs" / "contact_z" / "calibrator.json")
    assert cal.fsm["enabled"] and cal.info["q_source"] == "hand_pose_imu"


def test_detector_label_bootstrap_for_episodes_without_self_touch():
    """Robot D1 has no geometric self-touch labels: STRONG calibrated levels inside contact-allowed
    phases become positives of in-memory training views (the episode arrays stay untouched)."""
    from robot_skin.datasets.episode import EpisodeMeta
    from robot_skin.stages.contact import LABEL_TRAIN, _bootstrap_views

    T, N = 60, 2
    t = np.arange(T) / 200.0
    pid = np.where(t < 0.1, 0, 1).astype(np.int16)
    meta = EpisodeMeta(episode_id="r", dataset="motion", kind="robot", layout="robot_hand_template", n_taxels=N,
                       phases=[{"name": "baseline_start", "t0": 0.0, "t1": 0.1, "contact": "none"},
                               {"name": "pinch_index", "t0": 0.1, "t1": 0.3, "contact": "self"}],
                       phase_names=["baseline_start", "pinch_index"])
    lab = np.full((T, N), -1, np.int8)
    lab[:20] = 0
    ep = Episode(meta, {"t": t, "phase_id": pid, "contact_label": lab})
    lv = np.zeros((T, N), np.int8)
    lv[10:15, 0] = 2                                                  # STRONG inside the no-contact phase
    lv[30:40, 0] = 2                                                  # STRONG inside the pinch
    lv[30:40, 1] = 1                                                  # WEAK only: not bootstrapped
    lv[45:50, 1] = 3                                                  # SATURATED / FSM-untrusted: not either
    ep.set_derived(D_LEVEL, lv, save=False)
    ep.set_derived(D_RESIDUAL_Z, np.zeros((T, N), np.float32), save=False)
    (v,), n = _bootstrap_views([ep])
    assert n == 10
    assert np.all(v[LABEL_TRAIN][30:40, 0] == 1) and np.all(v[LABEL_TRAIN][30:50, 1] == -1)
    assert np.all(v[LABEL_TRAIN][10:15, 0] == 0)                      # labelled no-contact stays 0
    assert np.array_equal(v["contact_label"], lab) and v.has_derived(D_RESIDUAL_Z)


def test_contact_usable_follows_q_source():
    """Camera-free glove episodes have no vision q/qd arrays: with ``q_source: hand_pose_imu`` the
    contact stage takes qd from the IMU pose model's derived finger pose instead of skipping them."""
    from robot_skin.datasets.episode import EpisodeMeta
    from robot_skin.stages.contact import _usable

    T, N = 20, 2
    meta = EpisodeMeta(episode_id="imu_only", dataset="task", kind="glove", layout="glove_template", n_taxels=N)
    ep = Episode(meta, {"t": np.arange(T) / 200.0, "contact_label": np.full((T, N), -1, np.int8),
                        "saturated": np.zeros((T, N), bool)})
    for k in (D_RESIDUAL, D_BASELINE_LOGVAR):
        ep.set_derived(k, np.zeros((T, N), np.float32), save=False)
    assert "qd" in _usable(ep, True, "q", need_qd=True)                    # vision q_source needs the arrays
    assert _usable(ep, True, "q", need_qd=False) is None                     # z rule only: no qd needed
    assert D_HAND_POSE_IMU in _usable(ep, True, "hand_pose_imu")             # IMU pose not predicted yet
    ep.set_derived(D_HAND_POSE_IMU, np.zeros((T, 15, 3), np.float32), save=False)
    assert _usable(ep, True, "hand_pose_imu") is None
    assert "baseline_logvar" in _usable(Episode(meta, dict(ep.arrays)), True, "q", need_qd=False)


def test_baseline_rejects_a_loss_that_never_trains_the_mean(tmp_path):
    """mean_loss none + detach_mean: the NLL is the only term and it is stop-gradient on the mean."""
    with pytest.raises(ValueError, match="mean head never learns"):
        st_baseline.run({"data": {"processed_root": str(tmp_path)}, "loss": {"mean_loss": "none"},
                         "out_dir": str(tmp_path / "run")})
    with pytest.raises(ValueError, match="mean_loss"):
        st_baseline.run({"data": {"processed_root": str(tmp_path)}, "loss": {"mean_loss": "l1"},
                         "out_dir": str(tmp_path / "run")})


def test_stage_runs_are_reproducible(processed, pipeline):
    """torch's initial seed is random per process and the Trainer seeds only after the model is
    built: stages seed model construction from ``train.seed`` (``seed_model_init``), so the same
    config gives the same result whatever the global RNG state was."""
    root, _ = processed
    cfg = {"data": _data(root, stride=8), "model": {"window": 8, "hidden": 8, "head_hidden": 8, "n_layers": 2},
           "predict": {"write_derived": False}, "train": {**TRAIN, "max_steps": 3, "warmup_steps": 0}}
    out = []
    for i in range(2):
        torch.rand(5 + i)                                                   # a different global RNG state
        m = st_baseline.run({**cfg, "out_dir": str(root / "runs" / f"repro{i}")})
        out.append((m["final_train_loss"], m["val/mae_resid"]))
    assert out[0] == out[1]


def test_detection_metrics_hand_values():
    """Values derived by hand: 2 positives / 4 labelled negatives after the ignore mask."""
    from robot_skin.stages.contact import detection_metrics

    lab = np.array([[1, 0], [0, 0], [-1, 0], [1, 1]], np.int8)
    pred = np.array([[1, 1], [0, 0], [1, 0], [0, 1]], bool)
    score = np.array([[0.9, 0.8], [0.1, 0.2], [0.7, 0.3], [0.4, 0.95]])
    ign = np.array([[0, 0], [0, 0], [0, 0], [0, 1]], bool)
    gt = np.array([[1, 0], [0, 0], [1, 0], [1, 1]], bool)
    m = detection_metrics(pred, score, lab, ign, gt)
    assert (m["n_pos"], m["n_neg"]) == (2, 4)
    assert m["hallucination_taxel"] == 0.25                  # (0,1) of the 4 labelled negatives
    assert m["hallucination_frame"] == 0.0                   # only row 1 is fully labelled no-contact
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["f1"] == 0.5
    assert m["auroc"] == 0.875                               # 7 of 8 (pos, neg) pairs ordered
    assert np.isclose(m["gt_auroc"], 10 / 12) and m["gt_hallucination"] == 0.25 and np.isclose(m["gt_recall"], 2 / 3)
