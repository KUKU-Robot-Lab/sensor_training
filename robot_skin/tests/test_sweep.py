"""Tests for robot_skin.train.sweep (grid / random spaces, run_sweep bookkeeping, CLI sharding)."""
from __future__ import annotations

import importlib.util
import json
import math

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import TensorDataset

from robot_skin.train import DistInfo, TrainConfig, Trainer
from robot_skin.train.sweep import (expand_grid, flatten, get_by_path, load_results, main,
                                    run_optuna, run_sweep, sample_random, set_by_path, shard,
                                    suggest_from_space, trial_config, unflatten)


def toy_train(cfg: dict) -> dict:
    """Deterministic objective used by the CLI test (importable as module:function)."""
    lr = cfg["train"]["lr"]
    hidden = cfg["model"]["hidden"]
    return {"val/loss": (math.log10(lr) + 3.4) ** 2 + hidden / 1000, "hidden": hidden}


def hw_train(cfg: dict) -> dict:
    """CLI objective that reports what the hardware profile put into the trial config."""
    return {"metric": cfg["train"]["lr"], "hardware": cfg.get("hardware"),
            "device": cfg["train"].get("device"), "batch_size": cfg["train"].get("batch_size")}


def imu_pose_train_keys(cfg: dict) -> dict:
    """CLI objective: the train keys the real imu_pose stage trains with — its ``resolve_config``
    is the first thing ``stages.imu_pose.run`` does (and where a profile could be re-applied)."""
    from robot_skin.stages import imu_pose

    t = imu_pose.resolve_config(cfg)["train"]
    return {"metric": float(t["batch_size"]),
            **{k: t.get(k) for k in ("batch_size", "grad_accum", "precision", "compile", "device")}}


# ───────────────────────────────────────────────────────────────────────────── paths

def test_dotted_path_helpers():
    d: dict = {}
    set_by_path(d, "train.lr", 1e-3)
    set_by_path(d, "train.betas", [0.9, 0.95])
    set_by_path(d, "model.enc.depth", 4)
    assert d == {"train": {"lr": 1e-3, "betas": [0.9, 0.95]}, "model": {"enc": {"depth": 4}}}
    assert get_by_path(d, "model.enc.depth") == 4 and get_by_path(d, "x.y", None) is None
    assert get_by_path({"val/loss": 1.0}, "val/loss") == 1.0
    with pytest.raises(KeyError):
        get_by_path(d, "train.nope")
    assert unflatten(flatten(d)) == d
    with pytest.raises(ValueError):
        set_by_path({}, "a..b", 1)


# ───────────────────────────────────────────────────────────────────────────── spaces

def test_expand_grid():
    space = {"train.lr": [1e-3, 1e-4], "model.hidden": [64, 128, 256], "train.seed": 7,
             "train.betas": [[0.9, 0.95]]}
    grid = expand_grid(space)
    assert len(grid) == 6
    assert grid[0] == {"train": {"lr": 1e-3, "seed": 7, "betas": [0.9, 0.95]},
                       "model": {"hidden": 64}}
    assert [g["model"]["hidden"] for g in grid[:3]] == [64, 128, 256]      # last axis fastest
    assert {(g["train"]["lr"], g["model"]["hidden"]) for g in grid} == {
        (lr, h) for lr in (1e-3, 1e-4) for h in (64, 128, 256)}
    assert expand_grid({}) == [{}]
    assert len(expand_grid({"a": {"choice": [1, 2, 3]}})) == 3
    with pytest.raises(ValueError, match="random"):
        expand_grid({"train.lr": {"log_uniform": [1e-4, 1e-2]}})
    with pytest.raises(ValueError, match="empty"):
        expand_grid({"a": []})


def test_sample_random_distributions():
    space = {"train.lr": {"log_uniform": [1e-5, 1e-1]}, "train.wd": {"uniform": [0.0, 0.2]},
             "model.depth": {"int": [2, 4]}, "model.act": ["relu", "gelu"], "train.seed": 3}
    a = sample_random(space, 200, seed=1)
    assert a == sample_random(space, 200, seed=1)
    assert a != sample_random(space, 200, seed=2)
    lrs = [s["train"]["lr"] for s in a]
    assert all(1e-5 <= x <= 1e-1 for x in lrs)
    logs = [math.log10(x) for x in lrs]
    assert sum(x < -3 for x in logs) > 60 and sum(x > -3 for x in logs) > 60   # log-spread
    assert all(0.0 <= s["train"]["wd"] <= 0.2 for s in a)
    assert {s["model"]["depth"] for s in a} == {2, 3, 4}                        # inclusive
    assert {s["model"]["act"] for s in a} == {"relu", "gelu"}
    assert all(s["train"]["seed"] == 3 for s in a)
    assert isinstance(a[0]["train"]["lr"], float) and isinstance(a[0]["model"]["depth"], int)
    assert sample_random(space, 0) == []
    with pytest.raises(ValueError, match="positive"):
        sample_random({"x": {"log_uniform": [0, 1]}}, 1)
    with pytest.raises(ValueError, match="integers"):
        sample_random({"x": {"int": [0.5, 2]}}, 1)
    with pytest.raises(ValueError, match="low > high"):
        sample_random({"x": {"uniform": [2, 1]}}, 1)


def test_shard_partitions():
    items = list(range(7))
    parts = [shard(items, i, 3) for i in range(3)]
    assert sorted(x for p in parts for x in p) == items
    assert parts[0] == [0, 3, 6]
    with pytest.raises(ValueError):
        shard(items, 3, 3)


# ───────────────────────────────────────────────────────────────────────────── run_sweep

def test_run_sweep_sorting_files_and_resume(tmp_path):
    calls: list[dict] = []

    def fn(cfg):
        calls.append(cfg)
        if cfg["model"]["hidden"] == 999:
            raise RuntimeError("boom")
        return toy_train(cfg)["val/loss"]

    base = {"train": {"lr": 1.0, "epochs": 3}, "model": {"hidden": 1}}
    ovs = expand_grid({"train.lr": [1e-2, 10 ** -3.5, 1e-5], "model.hidden": [10, 999]})
    res = run_sweep(fn, base, ovs, tmp_path / "sw")
    assert len(calls) == 6 and len(res) == 6
    ok = [r for r in res if r["status"] == "ok"]
    assert len(ok) == 3 and all(r["status"] == "failed" for r in res[3:])
    assert res[0]["overrides"] == {"train.lr": 10 ** -3.5, "model.hidden": 10}
    assert [r["metric"] for r in ok] == sorted(r["metric"] for r in ok)
    assert "boom" in res[-1]["error"]
    assert calls[0]["train"]["epochs"] == 3                               # base merged
    assert calls[0]["train"]["out_dir"] == str(tmp_path / "sw" / "trial_000")
    lines = (tmp_path / "sw" / "results.jsonl").read_text().splitlines()
    assert len(lines) == 6 and json.loads(lines[0])["trial"] == 0

    calls.clear()
    res2 = run_sweep(fn, base, ovs, tmp_path / "sw")                    # resume
    assert len(calls) == 3 and all(c["model"]["hidden"] == 999 for c in calls)
    assert [r["metric"] for r in res2 if r["status"] == "ok"] == [r["metric"] for r in ok]

    best_max = run_sweep(fn, base, ovs[:1] + ovs[2:3], None, mode="max")
    assert best_max[0]["metric"] >= best_max[1]["metric"]
    with pytest.raises(RuntimeError):
        run_sweep(fn, base, [ovs[1]], None, catch_errors=False)


def test_run_sweep_dict_results_and_metric_key(tmp_path):
    ovs = [{"train": {"lr": 1e-3}, "model": {"hidden": 8}}]
    res = run_sweep(toy_train, {}, ovs, None, metric_key="val/loss")
    assert res[0]["status"] == "ok" and res[0]["result"]["hidden"] == 8
    res = run_sweep(lambda c: {"val": {"loss": 0.5}}, {}, [{}], None, metric_key="val.loss")
    assert res[0]["metric"] == 0.5
    res = run_sweep(lambda c: {"acc": 1.0}, {}, [{}], None)
    assert res[0]["status"] == "failed" and "metric" in res[0]["error"]
    res = run_sweep(lambda c: float("nan"), {}, [{}], None)
    assert res[0]["status"] == "failed"


def test_load_results_merges_files(tmp_path):
    run_sweep(lambda c: c["x"], {}, [{"x": 3.0}, {"x": 1.0}], tmp_path / "m1", trial_dir_key=None)
    run_sweep(lambda c: c["x"], {}, [{"x": 2.0}], tmp_path / "m2", trial_dir_key=None,
              indices=[2])
    merged = load_results(tmp_path / "m1", tmp_path / "m2" / "results.jsonl",
                          tmp_path / "missing")
    assert [r["metric"] for r in merged] == [1.0, 2.0, 3.0]
    assert load_results(tmp_path / "m1", mode="max")[0]["metric"] == 3.0


def test_cli_shards_cover_all_trials(tmp_path):
    (tmp_path / "space.yaml").write_text(yaml.safe_dump(
        {"mode": "grid", "space": {"train.lr": [1e-2, 1e-3, 1e-4], "model.hidden": [4, 8]}}))
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"train": {"lr": 1.0},
                                                        "model": {"hidden": 1}}))
    common = ["--fn", "robot_skin.tests.test_sweep:toy_train", "--space",
              str(tmp_path / "space.yaml"), "--base", str(tmp_path / "base.yaml"),
              "--metric", "val/loss"]
    r0 = main(common + ["--out", str(tmp_path / "box_a"), "--shard", "0/2"])
    r1 = main(common + ["--out", str(tmp_path / "box_b"), "--shard", "1/2"])
    assert len(r0) == 3 and len(r1) == 3
    merged = load_results(tmp_path / "box_a", tmp_path / "box_b")
    assert sorted(r["trial"] for r in merged) == list(range(6))
    assert merged[0]["overrides"] == {"train.lr": 1e-3, "model.hidden": 4}
    assert {r["trial_dir"].rsplit("trial_", 1)[1] for r in merged} == {f"{i:03d}" for i in range(6)}


def test_cli_hardware_profile_applied_under_trial_overrides(tmp_path):
    (tmp_path / "space.yaml").write_text(yaml.safe_dump({"space": {"train.lr": [0.1, 0.01]}}))
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"stage": "vtla",
                                                        "train": {"lr": 1.0, "device": "cuda"}}))
    res = main(["--fn", "robot_skin.tests.test_sweep:hw_train", "--space",
                str(tmp_path / "space.yaml"), "--base", str(tmp_path / "base.yaml"),
                "--out", str(tmp_path / "o"), "--hardware", "cpu"])
    assert [r["metric"] for r in res] == [0.01, 0.1]               # trial override beats base
    for r in res:
        assert r["result"]["hardware"] == "cpu" and r["result"]["device"] == "cpu"  # profile
        assert r["result"]["batch_size"] == 8                                  # suggest.vtla


@pytest.mark.parametrize("how", ["--hardware", "base-yaml"])
def test_cli_swept_profile_keys_survive_the_stage_resolve(tmp_path, how):
    """Regression: the profile used to be applied again inside the stage's run() (hardware set,
    hardware_applied false), silently replacing every swept batch_size / precision / compile with
    the profile's values — all trials trained identically."""
    from robot_skin.stages import imu_pose

    base = yaml.safe_load(imu_pose.CONFIG_PATH.read_text())
    if how == "base-yaml":
        base["hardware"] = "cpu"
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / "space.yaml").write_text(yaml.safe_dump({"space": {
        "train.batch_size": [16, 32], "train.precision": ["bf16"], "train.compile": [True]}}))
    argv = ["--fn", "robot_skin.tests.test_sweep:imu_pose_train_keys", "--space", str(tmp_path / "space.yaml"),
            "--base", str(tmp_path / "base.yaml"), "--out", str(tmp_path / "o")]
    res = main(argv + (["--hardware", "cpu"] if how == "--hardware" else []))
    assert sorted(r["result"]["batch_size"] for r in res) == [16, 32]      # not cpu suggest (64)
    for r in res:
        assert r["status"] == "ok"
        assert r["result"]["precision"] == "bf16" and r["result"]["compile"] is True   # not fp32/False
        assert r["result"]["device"] == "cpu" and r["result"]["grad_accum"] == 1       # profile keys
    # an axis the trial does not sweep keeps the profile's suggest.imu_pose value
    (tmp_path / "space2.yaml").write_text(yaml.safe_dump({"space": {"train.lr": [1e-3]}}))
    res = main(["--fn", "robot_skin.tests.test_sweep:imu_pose_train_keys", "--space", str(tmp_path / "space2.yaml"),
                "--base", str(tmp_path / "base.yaml"), "--out", str(tmp_path / "o2"), "--hardware", "cpu"])
    assert res[0]["result"]["batch_size"] == 64 and res[0]["result"]["precision"] == "fp32"


def test_trial_config_applies_a_swept_profile_before_the_trial_overrides(tmp_path):
    from robot_skin.stages import vtla

    prof = tmp_path / "big.yaml"
    prof.write_text(yaml.safe_dump({"name": "big", "train": {"device": "cpu", "precision": "bf16"},
                                    "suggest": {"vtla": {"batch_size": 99, "grad_accum": 3}}, "env": {}}))
    base = {"stage": "vtla", "hardware": "cpu", "train": {"lr": 1.0, "batch_size": 5}}
    cfg = trial_config(base, {"train": {"batch_size": 7}})
    assert cfg["hardware_applied"] is True and cfg["train"]["batch_size"] == 7   # trial beats profile
    assert cfg["train"]["device"] == "cpu" and cfg["train"]["grad_accum"] == 1
    assert vtla.resolve_config(cfg)["train"]["batch_size"] == 7                  # not re-applied
    cfg = trial_config(base, {"hardware": str(prof), "train": {"lr": 0.1}})      # swept profile
    assert cfg["hardware"] == "big" and cfg["train"]["batch_size"] == 99 and cfg["train"]["grad_accum"] == 3
    assert cfg["train"]["precision"] == "bf16" and cfg["train"]["lr"] == 0.1
    assert trial_config({"train": {"lr": 1.0}}, {"train": {"lr": 2.0}}) == {"train": {"lr": 2.0}}  # no profile
    with pytest.warns(UserWarning, match="without a stage"):
        trial_config({"hardware": "cpu", "train": {}}, {})


# ───────────────────────────────────────────────────────────────────────────── optuna glue

class FakeTrial:
    def __init__(self):
        self.calls: list[tuple] = []

    def suggest_float(self, name, low, high, log=False):
        self.calls.append(("float", name, low, high, log))
        return low

    def suggest_int(self, name, low, high):
        self.calls.append(("int", name, low, high))
        return high

    def suggest_categorical(self, name, choices):
        self.calls.append(("cat", name, tuple(choices)))
        return choices[-1]


def test_suggest_from_space_with_fake_trial():
    t = FakeTrial()
    ov = suggest_from_space(t, {"train.lr": {"log_uniform": [1e-4, 1e-2]},
                                "train.wd": {"uniform": [0, 0.1]}, "model.depth": {"int": [2, 6]},
                                "model.betas": [[0.9, 0.99], [0.9, 0.95]], "train.seed": 1})
    assert ov == {"train": {"lr": 1e-4, "wd": 0.0, "seed": 1},
                  "model": {"depth": 6, "betas": [0.9, 0.95]}}
    assert ("float", "train.lr", 1e-4, 1e-2, True) in t.calls
    assert ("cat", "model.betas", (0, 1)) in t.calls


def test_trials_get_their_own_dir_even_when_the_base_sets_a_top_level_out_dir(tmp_path):
    """Regression: trials wrote only ``train.out_dir`` while the stages prefer a top-level ``out_dir`` — with a
    base that sets it (e.g. a pipeline's ``<runs>/<stage>/pipeline_config.yaml``) every trial trained into that
    one directory, overwriting the pipeline's run and each other, and no trial_XXX dir was created."""
    from robot_skin.stages import imu_pose

    seen = []

    def fn(cfg):
        seen.append(imu_pose.resolve_config(cfg)["out_dir"])    # the directory the stage really writes to
        return 0.0

    base = yaml.safe_load(imu_pose.CONFIG_PATH.read_text())
    base["out_dir"] = str(tmp_path / "pipeline_runs" / "imu_pose")
    res = run_sweep(fn, base, [{"train": {"lr": 1e-3}}, {"train": {"lr": 2e-3}}], tmp_path / "sw")
    assert sorted(seen) == [str(tmp_path / "sw" / "trial_000"), str(tmp_path / "sw" / "trial_001")]
    assert sorted(r["trial_dir"] for r in res) == sorted(seen)
    # a base without a top-level out_dir (the documented stage YAML) and a custom trial_dir_key are unchanged
    seen.clear()
    run_sweep(lambda c: (seen.append((c.get("out_dir"), c["train"]["out_dir"])), 0.0)[1],
              {"train": {"lr": 1.0}}, [{}], tmp_path / "sw2")
    assert seen == [(None, str(tmp_path / "sw2" / "trial_000"))]
    seen.clear()
    run_sweep(lambda c: (seen.append((c["out_dir"], c["log_dir"])), 0.0)[1], {"out_dir": "keep"}, [{}],
              tmp_path / "sw3", trial_dir_key="log_dir")
    assert seen == [("keep", str(tmp_path / "sw3" / "trial_000"))]


def test_optuna_glue_is_exported_from_robot_skin_train():
    """TRAINING.md names ``robot_skin.train.run_optuna``; it was only importable from ``.sweep``."""
    import robot_skin.train as T

    from robot_skin.train import run_optuna as ro, suggest_from_space as sfs, trial_config as tc  # noqa: F401

    assert ro is run_optuna and sfs is suggest_from_space and tc is trial_config
    assert {"run_optuna", "suggest_from_space", "trial_config"} <= set(T.__all__)


@pytest.mark.skipif(importlib.util.find_spec("optuna") is not None, reason="optuna installed")
def test_run_optuna_requires_optuna():
    with pytest.raises(ImportError, match="pip install optuna"):
        run_optuna(lambda c: 0.0, {}, {"x": [1, 2]}, n_trials=1)


# ───────────────────────────────────────────────────────────────────────────── integration

def test_sweep_over_real_trainer(tmp_path):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 3, generator=g)
    ds = TensorDataset(x, x @ torch.tensor([1.0, -1.0, 2.0]))

    def loss_fn(model, batch):
        xb, yb = batch
        return {"loss": ((model(xb).squeeze(-1) - yb) ** 2).mean()}

    def train_fn(cfg):
        torch.manual_seed(0)
        tr = Trainer(nn.Linear(3, 1), loss_fn, TrainConfig.from_dict(cfg["train"]), ds, ds,
                     dist_info=DistInfo())
        tr.fit()
        return {"best": tr.best_value}

    base = {"train": {"max_epochs": 3, "batch_size": 16, "device": "cpu", "precision": "fp32",
                      "warmup_steps": 0, "schedule": "constant"}}
    res = run_sweep(train_fn, base, expand_grid({"train.lr": [1e-4, 5e-2]}), tmp_path / "sw",
                    metric_key="best")
    assert res[0]["overrides"]["train.lr"] == 5e-2 and res[0]["metric"] < res[1]["metric"]
    for r in res:
        assert (tmp_path / "sw" / f"trial_{r['trial']:03d}" / "ckpt_best.pt").exists()
