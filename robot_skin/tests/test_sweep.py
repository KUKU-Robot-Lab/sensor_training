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
                                    suggest_from_space, unflatten)


def toy_train(cfg: dict) -> dict:
    """Deterministic objective used by the CLI test (importable as module:function)."""
    lr = cfg["train"]["lr"]
    hidden = cfg["model"]["hidden"]
    return {"val/loss": (math.log10(lr) + 3.4) ** 2 + hidden / 1000, "hidden": hidden}


def hw_train(cfg: dict) -> dict:
    """CLI objective that reports what the hardware profile put into the trial config."""
    return {"metric": cfg["train"]["lr"], "hardware": cfg.get("hardware"),
            "device": cfg["train"].get("device"), "batch_size": cfg["train"].get("batch_size")}


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
