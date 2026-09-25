"""End to end through ``python -m robot_skin``: synthetic glove D1 + D2 → preprocessing → the training
pipeline (splits → imu_pose → baseline → contact → pretrain → vtla) → deployment on a fake robot hand
→ the deployment session back through preprocessing.

Data: ``synth`` (``datasets.synthetic.generate_dataset``: 4 motion + 4 task sessions of 8 s, subjects
s0–s3, one shared glove) plus one D1 session recorded with the **same glove** through a
channel-permuted Layout object (default layouts map taxel i → channel i and would hide a missing
``layout.by_channel``), then ``datasets.build.build_all``. The pipeline runs on the ``cpu`` hardware
profile with tiny models / few steps (``--set``); it creates one subject-grouped ``splits.json`` that
every stage must use. The VTLA policy acts in the MANO hand space (``hand_mano``), so deployment
retargets its actions onto the synthetic 16-DoF robot hand (``FakeRobotHand``) — the glove-trained
stage-1 models do not fit the robot skin, whose start-up calibration stands in.

Also covered: resuming (finished stages are skipped; a re-run stage re-runs everything downstream)
and ``train`` / ``pipeline`` under ``torchrun`` (2 CPU ranks, gloo). Runtime ≈ 30 s on a 4-CPU box.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from common.layouts import layout_from_dict, load_layout
from robot_skin.__main__ import STAGE_ARTEFACTS, STAGES
from robot_skin.__main__ import main as cli
from robot_skin.datasets.episode import (D_BASELINE_LOGVAR, D_BASELINE_PRED, D_CONTACT_LABEL_PSEUDO, D_CONTACT_PROB,
                                         D_HAND_POSE_IMU, D_LEVEL, D_RESIDUAL, D_RESIDUAL_Z, S_CHANNELS, Episode,
                                         list_episodes)

REPO = Path(__file__).resolve().parents[2]
SEED = 11
PERM = [7, 3, 8, 0, 5, 1, 6, 2, 4]              # layout taxel i → raw channel PERM[i]
PERM_ID = "syn_glove_motion_s1_perm"

#: tiny models / few steps for every stage (global keys reach every stage that has them;
#: ``<stage>.<key>`` only that stage and wins over a global key)
TINY = [
    "train.max_steps=30", "train.warmup_steps=5", "train.log_every=1000",
    "imu_pose.model.window=16", "imu_pose.model.hidden=32", "imu_pose.model.n_layers=1",
    "baseline.model.hidden=32", "baseline.model.head_hidden=32", "baseline.train.max_steps=60",
    "contact.data.stride=2", "contact.detector.hidden=16",
    "pretrain.data.frame_stride=8", "pretrain.model.d_model=32", "pretrain.model.depth=1", "pretrain.model.heads=2",
    "vtla.policy.horizon=8", "vtla.model.d_model=32", "vtla.model.fusion_depth=1", "vtla.model.head_depth=1",
    "vtla.model.n_tactile_tokens=2", "vtla.model.tactile_heads=2",
    "vtla.vision.encoder={type: tiny, out_dim: 16, grid: [2, 2], channels: [8, 16]}",
    "vtla.image.image_size=[24, 32]", "vtla.language.encoder={type: hashing, dim: 16, max_len: 8}",
]
SETS = [x for kv in TINY for x in ("--set", kv)]


def _permuted_glove_layout():
    d = load_layout("glove_template").to_dict(units="mm")
    d["name"] = "glove_permuted"
    for tx, ch in zip(d["taxels"], PERM):
        tx["channel"] = ch
    return layout_from_dict(d)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    from robot_skin.datasets.build import build_all
    from robot_skin.datasets.synthetic import generate_session

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)                   # tiny models: thread contention costs more than it gains
    lg = logging.getLogger("robot_skin")       # the CLI configures the robot_skin logger: restore it after
    lg_state = (lg.level, list(lg.handlers))
    root = tmp_path_factory.mktemp("e2e")
    raw, proc, runs = root / "raw", root / "processed", root / "runs"
    t0 = time.perf_counter()
    assert cli(["synth", "--out", str(raw), "--n-motion", "4", "--n-task", "4", "--subjects", "s0,s1,s2,s3",
                "--seed", str(SEED), "--duration", "8", "--cameras", "ego"]) == 0
    generate_session(raw / "motion" / "s1" / PERM_ID, kind="glove", dataset="motion", duration_s=8.0, seed=99,
                     glove_seed=SEED, subject="s1", layout=_permuted_glove_layout(), cameras=("ego",),
                     session_id=PERM_ID)
    rep = build_all(raw, proc)
    assert [r["status"] for r in rep] == ["built"] * 9, rep
    t_data = time.perf_counter()
    assert _quiet(cli, ["pipeline", "--processed", str(proc), "--out", str(runs), "--hardware", "cpu", "-q",
                        *SETS]) == 0
    t_pipe = time.perf_counter()
    assert _quiet(cli, ["deploy", "--hardware", "cpu", "--set", f"bundle={runs / 'vtla'}", "--set", "duration_s=0.5",
                        "--set", f"out_dir={runs / 'deploy'}", "--set", "instruction=pick up the cup",
                        "--set", "startup.baseline_s=0.2", "--set", "startup.calib_s=0.3",
                        "--set", "latency.n=3", "--set", "latency.warmup=1"]) == 0
    t_dep = time.perf_counter()
    yield {"root": root, "raw": raw, "proc": proc, "runs": runs,
           "times": {"data": t_data - t0, "pipeline": t_pipe - t_data, "deploy": t_dep - t_pipe}}
    torch.set_num_threads(n_threads)
    lg.setLevel(lg_state[0])
    for h in list(lg.handlers):
        if h not in lg_state[1]:
            lg.removeHandler(h)


def _strict_json(p: Path) -> dict:
    def bad(x):
        raise ValueError(f"non-finite JSON constant {x} in {p}")
    return json.loads(p.read_text(), parse_constant=bad)


def _record(runs: Path) -> dict:
    return json.loads((runs / "pipeline.json").read_text())


def _episodes(proc: Path) -> dict[str, Episode]:
    return {d.name: Episode.load(d) for d in list_episodes(proc)}


# ───────────────────────────────────────────────────────────── pipeline outputs

def test_every_stage_wrote_metrics_and_artefacts(e2e):
    runs = e2e["runs"]
    rec = _record(runs)
    assert list(rec["stages"]) == list(STAGES)
    extra = {"imu_pose": ["imu_stats.json"], "baseline": ["joint_stats.json"], "contact": ["contact_detector.pt"],
             "pretrain": [], "vtla": []}
    for s in STAGES:
        r = rec["stages"][s]
        assert r["status"] == "ran" and r["seconds"] > 0 and r["out_dir"] == str(runs / s), (s, r)
        for f in ["metrics.json", STAGE_ARTEFACTS[s], "ckpt_best.pt", "pipeline_config.yaml", *extra[s]]:
            assert (runs / s / f).is_file(), (s, f)
        m = _strict_json(runs / s / "metrics.json")
        assert m["stage"] == s and m["steps"] == (60 if s == "baseline" else 30), (s, m.get("steps"))
        cfg = yaml.safe_load((runs / s / "pipeline_config.yaml").read_text())
        assert cfg["hardware"] == "cpu" and cfg["train"]["device"] == "cpu" and cfg["out_dir"] == str(runs / s)
    m = {s: _strict_json(runs / s / "metrics.json") for s in STAGES}
    # every stage learned something on held-out subjects (tiny models, few steps: loose bounds)
    assert m["imu_pose"]["val/rot_deg"] < m["imu_pose"]["val/rot_deg_flat"]
    assert m["baseline"]["val/resid_reduction"] > 0.2 and m["baseline"]["test/resid_reduction"] > 0.2
    assert m["contact"]["val/z_auroc"] > 0.9 and m["contact"]["val/prob_auroc"] > 0.9
    assert m["contact"]["pseudo/n_episodes"] == 4
    assert np.isfinite(m["pretrain"]["val/loss"]) and m["pretrain"]["n_episodes"]["skipped"] == 0
    v = m["vtla"]
    assert v["tactile_source"] == "derived" and v["n_samples"]["train"] > 0 and np.isfinite(v["val/l1"])
    assert v["n_episodes"]["skipped"] == 0 and np.isfinite(v["test/l1_raw/finger_aa"])
    # the cpu profile's per-stage batch suggestion reached the stage (no batch_size override given)
    assert yaml.safe_load((runs / "vtla" / "pipeline_config.yaml").read_text())["train"]["batch_size"] == 8


def test_derived_arrays_on_every_episode(e2e):
    eps = _episodes(e2e["proc"])
    assert len(eps) == 9 and sorted({e.meta.dataset for e in eps.values()}) == ["motion", "task"]
    for name, ep in eps.items():
        T, N = ep.T, ep.meta.n_taxels
        for k in (D_BASELINE_PRED, D_BASELINE_LOGVAR, D_RESIDUAL, D_RESIDUAL_Z, D_CONTACT_PROB):
            a = np.asarray(ep.derived(k))
            assert a.shape == (T, N) and a.dtype == np.float32 and np.isfinite(a).all(), (name, k)
        lv = np.asarray(ep.derived(D_LEVEL))
        assert lv.shape == (T, N) and lv.dtype == np.int8 and set(np.unique(lv)) <= {0, 1, 2, 3}
        fp = np.asarray(ep.derived(D_HAND_POSE_IMU))
        assert fp.shape == (T, 15, 3) and np.isfinite(fp).all()
        np.testing.assert_allclose(np.asarray(ep.derived(D_RESIDUAL)),
                                   np.asarray(ep["delta_pct"]) - np.asarray(ep.derived(D_BASELINE_PRED)), atol=1e-5)
        assert ep.has_derived(D_CONTACT_LABEL_PSEUDO) == (ep.meta.dataset == "task"), name
        if ep.meta.dataset == "motion":                    # 8 s D1 sessions include the no-contact grasp shape
            assert "air_grasp_slow_power" in ep.meta.phase_names, name
    # the channel-permuted session: layout order restored, and it went through every stage like the others
    perm = eps[PERM_ID]
    np.testing.assert_array_equal(perm.static[S_CHANNELS], PERM)
    assert perm.meta.layout.endswith(".yaml") and perm.meta.preprocessing["taxel_frame"] == "mano_wrist"


def test_one_splits_file_shared_by_every_stage(e2e):
    from robot_skin.baseline.temporal import load_baseline_model
    from robot_skin.datasets.splits import load_splits
    from robot_skin.representation.encoder import read_encoder_state
    from robot_skin.stages.imu_pose import load_imu_pose_model
    from robot_skin.vtla.model import read_policy_bundle

    proc, runs, sp_path = e2e["proc"], e2e["runs"], e2e["runs"] / "splits.json"
    splits = {k: {str(p.resolve()) for p in v} for k, v in load_splits(sp_path, root=proc).items()}
    assert all(splits.values()) and sum(map(len, splits.values())) == 9
    eps = _episodes(proc)
    subj = {k: {eps[Path(p).name].meta.subject for p in v} for k, v in splits.items()}
    assert not (subj["train"] & subj["val"] or subj["train"] & subj["test"] or subj["val"] & subj["test"])
    rec = _record(runs)
    sha = hashlib.sha256(sp_path.read_bytes()).hexdigest()
    assert rec["splits"] == {"path": str(sp_path), "sha256": sha, "created": True}
    for s in STAGES:
        cfg = yaml.safe_load((runs / s / "pipeline_config.yaml").read_text())
        assert cfg["data"]["splits"] == str(sp_path) and cfg["data"]["processed_root"] == str(proc), s
        assert rec["stages"][s]["splits_sha256"] == sha
    # the episodes each stage actually trained / validated / tested on sit in the same split of splits.json
    used = {"imu_pose": load_imu_pose_model(runs / "imu_pose").bundle_meta["episodes"],
            "baseline": load_baseline_model(runs / "baseline").bundle_meta["episodes"],
            "pretrain": read_encoder_state(runs / "pretrain")["meta"]["episodes"],
            "vtla": read_policy_bundle(runs / "vtla")["meta"]["episodes"]}
    for s, parts in used.items():
        assert parts["train"], s
        for k, v in parts.items():
            assert {str(Path(p).resolve()) for p in v} <= splits[k], (s, k)
    motion = {k: sum(eps[Path(p).name].meta.dataset == "motion" for p in v) for k, v in splits.items()}
    n = _strict_json(runs / "contact" / "metrics.json")["n_episodes"]
    assert {k: n[k] for k in ("train", "val", "test")} == motion
    # `python -m robot_skin splits` with the same defaults reproduces the pipeline's split exactly
    again = e2e["root"] / "splits_again.json"
    assert cli(["splits", "--processed", str(proc), "--out", str(again)]) == 0
    assert {k: {str(p.resolve()) for p in v} for k, v in load_splits(again, root=proc).items()} == splits
    with pytest.raises(SystemExit, match="exists"):
        cli(["splits", "--processed", str(proc), "--out", str(again)])


def test_shared_splits_file_warns_only_about_real_problems(e2e, caplog):
    """The pipeline's splits.json lists D1 and D2 episodes; a stage that uses one dataset sees the
    other's entries as existing episodes outside its pool — logged, not warned. Unresolvable entries
    and usable episodes the file does not list still warn."""
    import warnings as w

    from robot_skin.stages import split_stage_episodes
    from robot_skin.stages import vtla as st_vtla

    proc, sp_path = e2e["proc"], e2e["runs"] / "splits.json"
    eps = _episodes(proc)
    motion = [e for e in eps.values() if e.meta.dataset == "motion"]
    task = [e for e in eps.values() if e.meta.dataset == "task"]
    d_cfg = {"splits": str(sp_path), "processed_root": str(proc)}
    with w.catch_warnings():
        w.simplefilter("error")                                         # any warning fails
        with caplog.at_level(logging.INFO, logger="robot_skin.stages"):
            parts = split_stage_episodes(motion, d_cfg, stage="baseline")
            vparts = st_vtla.split_episodes(task, d_cfg)
    assert sum(map(len, parts.values())) == len(motion) and sum(map(len, vparts.values())) == len(task)
    msgs = [r.getMessage() for r in caplog.records if "outside this stage's pool" in r.getMessage()]
    assert len(msgs) == 2 and msgs[0].startswith("baseline:") and msgs[1].startswith("vtla:")
    bad = e2e["root"] / "splits_bad.json"
    spec = json.loads(sp_path.read_text())
    bad.write_text(json.dumps({**spec, "test": [*spec["test"], "motion/no_such_episode"]}))
    with pytest.warns(UserWarning, match="1 entries without a usable episode, 0 usable"):
        split_stage_episodes(motion, {**d_cfg, "splits": str(bad)}, stage="baseline")
    left_out = next(x for x in spec["train"] if x.startswith("task/"))
    short = e2e["root"] / "splits_short.json"
    short.write_text(json.dumps({**spec, "train": [x for x in spec["train"] if x != left_out]}))
    with pytest.warns(UserWarning, match="0 entries without a usable episode, 1 usable episodes not listed"):
        st_vtla.split_episodes(task, {**d_cfg, "splits": str(short)})


def test_vtla_bundle_is_wired_to_the_earlier_stages(e2e):
    from robot_skin.representation.encoder import read_encoder_state
    from robot_skin.vtla.model import build_policy_from_bundle, read_policy_bundle

    runs = e2e["runs"]
    b = read_policy_bundle(runs / "vtla")
    tac = b["tactile"]
    assert tac["source"] == "derived" and tac["pretrained_encoder"] == str(runs / "pretrain" / "encoder_state.pt")
    assert tac["calibrator"] == str(runs / "contact" / "calibrator.json")
    assert tac["calibrator_state"] == json.loads((runs / "contact" / "calibrator.json").read_text())
    assert tac["baseline_model"] == str(runs / "baseline" / "baseline_model.pt")
    assert b["action"]["spec"]["kind"] == "hand_mano"
    # the policy's tactile branch is the pretrained encoder (architecture + feature spec)
    enc = read_encoder_state(runs / "pretrain")
    pol = _quiet(build_policy_from_bundle, b)
    assert pol.tactile_encoder.config == enc["config"] and pol.feature_spec.to_dict() == tac["feature_spec"]
    rec = _record(runs)["stages"]
    assert set(rec["vtla"]["upstream"]) == {"baseline", "contact", "pretrain"}
    assert rec["contact"]["upstream"] == {"baseline": str(runs / "baseline")}
    assert rec["vtla"]["wired"]["data.tactile_source"] == "derived"


# ───────────────────────────────────────────────────────────── deployment

def test_deploy_on_the_fake_robot_hand_reports_latency_and_safety(e2e):
    m = _strict_json(e2e["runs"] / "deploy" / "metrics.json")
    assert m["stage"] == "deploy" and m["robot"] == "fake" and m["action_kind"] == "hand_mano"
    assert m["n_ticks"] == 100 and m["n_policy_ticks"] == 10 and m["loop_hz"] == pytest.approx(200.0)
    assert m["latency_p95_ms"] >= m["latency_p50_ms"] > 0 and m["retarget_ms"] is not None
    assert m["benchmark"]["n"] == 3 and m["benchmark"]["p50_ms"] > 0
    assert isinstance(m["safety_counts"], dict) and isinstance(m["safety_events"], list)
    assert m["estop"] is False and m["startup"]["calibrator"] == "startup"
    assert any("retarget scale" in n for n in m["notes"])
    assert any("stage-1 references" in n for n in m["notes"])   # glove stage-1 models not used on the robot
    assert Path(m["session_dir"]).is_dir()


def test_deployment_session_is_reingestible(e2e):
    from robot_skin.datasets.synthetic import ROBOT_JOINT_NAMES

    runs, out = e2e["runs"], e2e["root"] / "reingest"
    assert cli(["preprocess", "--raw", str(runs / "deploy" / "sessions"), "--out", str(out), "-q",
                "--set", "baseline.duration_s=0.2", "--set", "cameras.copy_frames=none"]) == 0
    (d,) = list_episodes(out)
    ep = Episode.load(d)
    assert ep.meta.kind == "robot" and ep.meta.dataset == "other" and ep.meta.n_taxels == 9
    assert list(ep.meta.joint_names) == list(ROBOT_JOINT_NAMES) and "rollout" in ep.meta.phase_names
    assert ep.meta.task["instruction"] == "pick up the cup" and ep.meta.cameras == ["ego"]
    assert ep.T > 150 and ep["taxel_pos"].shape == (ep.T, 9, 3) and np.isfinite(ep["delta_pct"]).all()


# ───────────────────────────────────────────────────────────── CLI plumbing

def test_cli_help_config_errors_and_torchrun(e2e):
    """``python -m robot_skin`` as a module, a typo in ``--set`` / the command, and both ``train`` and
    ``pipeline`` under ``torchrun`` (2 CPU ranks each, gloo; the two jobs run concurrently): rank 0
    alone writes and prints, the pipeline's splits.json is created once and shared."""
    import shutil

    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]),
           "OMP_NUM_THREADS": "1"}
    h = subprocess.run([sys.executable, "-m", "robot_skin", "--help"], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=60)
    assert h.returncode == 0 and "pipeline" in h.stdout and "train <stage>" in h.stdout
    with pytest.raises(SystemExit, match="no selected stage"):
        cli(["pipeline", "--processed", str(e2e["proc"]), "--out", str(e2e["runs"]), "--set", "trian.lr=1"])
    with pytest.raises(SystemExit, match="unknown command"):
        cli(["trian"])
    root = e2e["root"]
    torchrun = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "-m",
                "robot_skin"]
    tiny = ["--set", "train.max_steps=4", "--set", "train.warmup_steps=1"]
    # train: the shared episodes are only read (predict.write_derived false)
    t_out = root / "ddp_train_imu_pose"
    train = [*torchrun, "train", "imu_pose", "--hardware", "cpu", "-q", *tiny,
             "--set", f"data.processed_root={e2e['proc']}", "--set", f"data.splits={e2e['runs'] / 'splits.json'}",
             "--set", f"out_dir={t_out}", "--set", "predict.write_derived=false",
             "--set", "model.window=16", "--set", "model.hidden=16", "--set", "model.n_layers=1"]
    # pipeline: on a copy of the processed root (it writes derived arrays)
    proc2, runs2 = root / "ddp_processed", root / "ddp_runs"
    shutil.copytree(e2e["proc"], proc2, symlinks=True)
    pipe = [*torchrun, "pipeline", "--processed", str(proc2), "--out", str(runs2), "--hardware", "cpu", "-q",
            "--stages", "imu_pose,baseline", *tiny, "--set", "imu_pose.model.window=16",
            "--set", "imu_pose.model.hidden=16", "--set", "imu_pose.model.n_layers=1",
            "--set", "baseline.model.hidden=16", "--set", "baseline.model.head_hidden=16"]
    jobs = [subprocess.Popen(c, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for c in (train, pipe)]
    (t_so, t_se), (p_so, p_se) = (j.communicate(timeout=300) for j in jobs)
    assert jobs[0].returncode == 0, t_se[-3000:]
    assert jobs[1].returncode == 0, p_se[-3000:]
    m = _strict_json(t_out / "metrics.json")
    assert m["steps"] == 4 and json.loads((t_out / "env.json").read_text())["env"]["WORLD_SIZE"] == "2"
    assert t_so.count('"stage": "imu_pose"') == 1                         # only rank 0 prints
    rec = _record(runs2)
    assert rec["splits"]["created"] and list(rec["stages"]) == ["imu_pose", "baseline"]
    for s in ("imu_pose", "baseline"):
        assert rec["stages"][s]["status"] == "ran" and _strict_json(runs2 / s / "metrics.json")["steps"] == 4
        assert json.loads((runs2 / s / "env.json").read_text())["env"]["WORLD_SIZE"] == "2"
    assert p_so.count("pipeline ") == 1 and "baseline  ran" in p_so


def test_pipeline_resumes_and_reruns_everything_downstream(e2e):
    """Last (it retrains pretrain + vtla): a second run skips every finished stage; a stage without
    results re-runs, and so does every stage after it."""
    proc, runs = e2e["proc"], e2e["runs"]
    stamp = {s: (runs / s / "metrics.json").stat().st_mtime_ns for s in STAGES}
    args = ["pipeline", "--processed", str(proc), "--out", str(runs), "--hardware", "cpu", "-q", *SETS]
    assert _quiet(cli, args) == 0
    rec = _record(runs)["stages"]
    assert all(rec[s]["status"] == "skipped" for s in STAGES)
    assert stamp == {s: (runs / s / "metrics.json").stat().st_mtime_ns for s in STAGES}
    (runs / "pretrain" / "metrics.json").unlink()
    assert _quiet(cli, args) == 0
    rec = _record(runs)["stages"]
    assert [rec[s]["status"] for s in STAGES] == ["skipped", "skipped", "skipped", "ran", "ran"]
    assert all((runs / s / "metrics.json").stat().st_mtime_ns == stamp[s] for s in STAGES[:3])
    assert (runs / "vtla" / "metrics.json").stat().st_mtime_ns > stamp["vtla"]
    # derived arrays wiped from an episode (e.g. `preprocess --force`) make the stage that wrote them stale
    from robot_skin.__main__ import _derived_intact

    f = next(iter(list_episodes(proc))) / "derived" / f"{D_RESIDUAL_Z}.npy"
    assert _derived_intact("contact", runs / "contact", proc) and _derived_intact("pretrain", runs / "pretrain", proc)
    f.rename(f.with_suffix(".moved"))
    try:
        assert not _derived_intact("contact", runs / "contact", proc)
        assert _derived_intact("baseline", runs / "baseline", proc)
    finally:
        f.with_suffix(".moved").rename(f)
