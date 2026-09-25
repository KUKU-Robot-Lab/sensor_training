"""Tests for robot_skin.train: engine (Trainer / TrainConfig), optim, checkpoint, logging.

CPU-only, seeded, single process (no torchrun / process groups)."""
from __future__ import annotations

import collections
import copy
import dataclasses
import importlib.util
import json
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from robot_skin.train import (EMA, BEST_NAME, LAST_NAME, DistInfo, JsonlLogger, TrainConfig,
                              Trainer, build_optimizer, build_scheduler, find_best, find_last,
                              load_checkpoint, lr_factor, move_to_device, param_groups,
                              read_jsonl, save_checkpoint, seed_everything,
                              strip_state_dict_prefixes)
from robot_skin.train.optim import split_decay_params

SINGLE = DistInfo()


# ───────────────────────────────────────────────────────────────────────────── fixtures

def _data(n: int = 64, d: int = 4, seed: int = 0, noise: float = 0.0) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    w = torch.arange(1, d + 1, dtype=torch.float32)
    y = x @ w + 0.5 + noise * torch.randn(n, generator=g)
    return TensorDataset(x, y)


def mse_loss(model, batch):
    x, y = batch
    pred = model(x).squeeze(-1)
    return {"loss": ((pred - y) ** 2).mean(), "mae": (pred - y).abs().mean()}


def _model(seed: int = 0, d: int = 4) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Linear(d, 1)


def _cfg(tmp_path, name: str = "run", **kw) -> TrainConfig:
    base = dict(max_epochs=4, batch_size=16, lr=0.05, weight_decay=0.0, warmup_steps=0,
                schedule="constant", device="cpu", precision="fp32", log_every=1000,
                out_dir=str(tmp_path / name), seed=0, monitor="train/loss")
    base.update(kw)
    return TrainConfig(**base)


def _params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


# ───────────────────────────────────────────────────────────────────────────── Trainer

def test_trainer_fits_tiny_regression(tmp_path):
    cfg = _cfg(tmp_path, max_epochs=25, schedule="cosine", warmup_steps=3, min_lr_ratio=0.05,
               log_every=5, lr=0.1, monitor="val/loss")
    tr = Trainer(_model(), mse_loss, cfg, _data(128, noise=0.01), _data(64, seed=1),
                 dist_info=SINGLE)
    hist = tr.fit()
    assert len(hist) == 25
    assert hist[-1]["train/loss"] < 0.01 * hist[0]["train/loss"]
    assert hist[-1]["val/loss"] < 0.05
    assert {"epoch", "step", "lr", "train/loss", "train/mae", "val/loss", "val/mae"} <= set(hist[-1])
    assert tr.step == 25 * 8 and tr.epoch == 25
    run = tmp_path / "run"
    for f in ("config.json", "env.json", "metrics.jsonl", "history.json", LAST_NAME, BEST_NAME,
              "summary.json"):
        assert (run / f).exists(), f
    summary = json.loads((run / "summary.json").read_text())
    assert summary["finished"] and not summary["stopped_early"] and summary["step"] == tr.step
    assert json.loads((run / "config.json").read_text())["effective_batch_size"] == 16
    steps = read_jsonl(run / "metrics.jsonl", kind="step")
    assert [r["step"] for r in steps] == list(range(5, 201, 5))
    assert len(read_jsonl(run / "metrics.jsonl", kind="epoch")) == 25
    # lr follows warm-up + cosine: last epoch LR ≈ min_lr_ratio · lr
    assert hist[-1]["lr"] == pytest.approx(0.1 * 0.05, rel=1e-6)


def test_trainer_accepts_dict_cfg_dataloader_and_tensor_loss(tmp_path):
    ds = _data(32)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    def loss_tensor(model, batch):
        return mse_loss(model, batch)["loss"]

    cfg = {"max_epochs": 2, "batch_size": 8, "lr": "1e-2", "device": "cpu", "precision": "fp32",
           "schedule": "constant", "warmup_steps": 0, "out_dir": str(tmp_path / "r"),
           "monitor": "train/loss"}
    tr = Trainer(_model(), loss_tensor, cfg, loader, dist_info=SINGLE)
    hist = tr.fit()
    assert tr.cfg.lr == pytest.approx(1e-2) and len(hist) == 2 and tr.step == 8


def test_loss_fn_contract_errors(tmp_path):
    tr = Trainer(_model(), lambda m, b: {"mse": mse_loss(m, b)["loss"]}, _cfg(tmp_path),
                 _data(32), dist_info=SINGLE)
    with pytest.raises(KeyError, match="loss"):
        tr.fit()
    tr = Trainer(_model(), lambda m, b: {"loss": mse_loss(m, b)["loss"].detach()},
                 _cfg(tmp_path, name="r2"), _data(32), dist_info=SINGLE)
    with pytest.raises(RuntimeError, match="require grad"):
        tr.fit()


# 56: last accumulation group has a single micro-batch; 60 / 62: the last micro-batch is short
# (4 / 6 samples) so equal per-micro-batch weights would NOT match the large batch
@pytest.mark.parametrize("n", [64, 56, 60, 62])
@pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
def test_grad_accumulation_equivalence(tmp_path, n, optimizer):
    ds = _data(n, noise=0.1)
    kw = dict(max_epochs=3, schedule="cosine", warmup_steps=2, lr=0.02, weight_decay=0.05,
              grad_clip=1.0, optimizer=optimizer)
    a = Trainer(_model(), mse_loss, _cfg(tmp_path, "a", batch_size=8, grad_accum=2, **kw), ds,
                dist_info=SINGLE)
    b = Trainer(_model(), mse_loss, _cfg(tmp_path, "b", batch_size=16, grad_accum=1, **kw), ds,
                dist_info=SINGLE)
    a.fit()
    b.fit()
    assert a.step == b.step == 3 * math.ceil(n / 16)
    torch.testing.assert_close(_params(a.model), _params(b.model), atol=1e-5, rtol=1e-5)
    # sanity: accumulation is not a no-op (differs from plain batch-8 training)
    c = Trainer(_model(), mse_loss, _cfg(tmp_path, "c", batch_size=8, grad_accum=1, **kw), ds,
                dist_info=SINGLE)
    c.fit()
    assert not torch.allclose(_params(a.model), _params(c.model), atol=1e-4)


def _stop_after(epoch: int):
    def cb(trainer, row):
        if row["epoch"] == epoch:
            trainer.should_stop = True
    return cb


def test_checkpoint_resume_reproduces_uninterrupted_run(tmp_path):
    ds, val = _data(64, noise=0.1), _data(32, seed=1)
    kw = dict(max_epochs=4, schedule="cosine", warmup_steps=2, lr=0.05, weight_decay=0.01,
              ema_decay=0.9, batch_size=8, grad_accum=2, monitor="val/loss")
    full = Trainer(_model(), mse_loss, _cfg(tmp_path, "full", **kw), ds, val, dist_info=SINGLE)
    full.fit()

    part = Trainer(_model(), mse_loss, _cfg(tmp_path, "part", **kw), ds, val,
                   callbacks=[_stop_after(2)], dist_info=SINGLE)
    part.fit()
    assert part.step == 2 * 4 and part.epoch == 2 and len(part.history) == 2
    ck = load_checkpoint(tmp_path / "part" / LAST_NAME)
    assert {"model", "optimizer", "scheduler", "scaler", "ema", "step", "epoch", "config",
            "extra"} <= set(ck)
    assert ck["step"] == 8 and ck["epoch"] == 2

    resumed = Trainer(_model(seed=123), mse_loss, _cfg(tmp_path, "part", resume="auto", **kw),
                      ds, val, dist_info=SINGLE)
    resumed.fit()
    assert resumed.resumed_from == tmp_path / "part" / LAST_NAME
    assert resumed.step == full.step == 16 and resumed.epoch == 4
    assert len(resumed.history) == 4
    assert resumed.history[:2] == part.history
    torch.testing.assert_close(_params(resumed.model), _params(full.model), atol=1e-6, rtol=1e-6)
    for k, v in full.ema.shadow.items():
        torch.testing.assert_close(resumed.ema.shadow[k], v, atol=1e-6, rtol=1e-6)
    assert resumed.history[-1]["val/loss"] == pytest.approx(full.history[-1]["val/loss"], rel=1e-5)


def test_mid_epoch_resume_via_max_steps(tmp_path):
    ds = _data(64, noise=0.1)
    kw = dict(batch_size=8, grad_accum=2, lr=0.05, max_epochs=1)   # 4 optimizer steps / epoch
    full = Trainer(_model(), mse_loss, _cfg(tmp_path, "full", max_steps=14, **kw), ds,
                   dist_info=SINGLE)
    full.fit()
    assert full.step == 14 and full.history[-1].get("partial") is True

    first = Trainer(_model(), mse_loss, _cfg(tmp_path, "p", max_steps=10, **kw), ds,
                    dist_info=SINGLE)
    first.fit()
    assert first.step == 10 and first.epoch == 2 and first.batch_in_epoch == 4
    second = Trainer(_model(seed=7), mse_loss, _cfg(tmp_path, "p", max_steps=14, **kw), ds,
                     dist_info=SINGLE)
    assert second.resume("auto") is not None
    assert second.batch_in_epoch == 4
    second.fit()
    assert second.step == 14
    torch.testing.assert_close(_params(second.model), _params(full.model), atol=1e-6, rtol=1e-6)


def test_keyboard_interrupt_saves_resumable_checkpoint(tmp_path):
    """Ctrl-C / preemption in the middle of an accumulation group: the saved state points at the
    last optimizer step, and resuming reproduces the uninterrupted run."""
    ds = _data(64, noise=0.1)
    kw = dict(batch_size=8, grad_accum=2, lr=0.05, max_epochs=2, schedule="cosine",
              warmup_steps=1)
    full = Trainer(_model(), mse_loss, _cfg(tmp_path, "full", **kw), ds, dist_info=SINGLE)
    full.fit()

    calls = {"n": 0}

    def flaky(model, batch):
        calls["n"] += 1
        if calls["n"] == 11:          # epoch 2, first micro-batch of its 2nd group
            raise KeyboardInterrupt
        return mse_loss(model, batch)

    tr = Trainer(_model(), flaky, _cfg(tmp_path, "p", **kw), ds, dist_info=SINGLE)
    with pytest.raises(KeyboardInterrupt):
        tr.fit()
    ck = load_checkpoint(tmp_path / "p" / LAST_NAME)
    assert (ck["step"], ck["epoch"], ck["batch_in_epoch"]) == (5, 1, 2)
    assert not (tmp_path / "p" / "summary.json").exists()

    def no_summary_during_run(trainer, row):
        assert not (tmp_path / "p" / "summary.json").exists()

    res = Trainer(_model(seed=3), mse_loss, _cfg(tmp_path, "p", resume="auto", **kw), ds,
                  callbacks=[no_summary_during_run], dist_info=SINGLE)
    res.fit()
    assert res.step == full.step == 8
    torch.testing.assert_close(_params(res.model), _params(full.model), atol=1e-6, rtol=1e-6)
    assert json.loads((tmp_path / "p" / "summary.json").read_text())["finished"] is True


def test_refit_after_interrupt_in_same_process_drops_stale_grads(tmp_path):
    """fit() re-entered on the same Trainer after Ctrl-C must not reuse the half-accumulated
    gradients of the interrupted group."""
    ds = _data(64, noise=0.1)
    kw = dict(batch_size=8, grad_accum=2, lr=0.05, max_epochs=2, optimizer="sgd")
    full = Trainer(_model(), mse_loss, _cfg(tmp_path, "full", **kw), ds, dist_info=SINGLE)
    full.fit()
    calls = {"n": 0}

    def flaky(model, batch):
        calls["n"] += 1
        if calls["n"] == 4:           # group 2: micro-batch 3 already backpropagated
            raise KeyboardInterrupt
        return mse_loss(model, batch)

    tr = Trainer(_model(), flaky, _cfg(tmp_path, "p", **kw), ds, dist_info=SINGLE)
    with pytest.raises(KeyboardInterrupt):
        tr.fit()
    assert tr.step == 1 and tr.batch_in_epoch == 2
    tr.fit()                           # continue in-process (same RNG-free data order)
    assert tr.step == full.step
    torch.testing.assert_close(_params(tr.model), _params(full.model), atol=1e-6, rtol=1e-6)


class _Stream(torch.utils.data.IterableDataset):
    def __init__(self, ds: TensorDataset):
        self.ds = ds

    def __iter__(self):
        return iter(zip(*self.ds.tensors))


def test_iterable_dataset_partial_last_group_is_renormalised(tmp_path):
    """Unknown-length loader: 3 micro-batches with grad_accum=2 → groups (0,1) and (2); the last
    optimizer step must use the plain mean gradient of micro-batch 2 (not half of it)."""
    ds = _data(24, noise=0.1)
    cfg = _cfg(tmp_path, max_steps=2, batch_size=8, grad_accum=2, optimizer="sgd", grad_clip=None)
    tr = Trainer(_model(), mse_loss, cfg, _Stream(ds), dist_info=SINGLE)
    tr.fit()
    assert tr.step == 2
    ref = _model()
    opt = torch.optim.SGD(ref.parameters(), lr=0.05, momentum=0.9, nesterov=True)
    x, y = ds.tensors
    for sl in (slice(0, 16), slice(16, 24)):
        opt.zero_grad()
        mse_loss(ref, (x[sl], y[sl]))["loss"].backward()
        opt.step()
    torch.testing.assert_close(_params(tr.model), _params(ref), atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="max_steps"):
        Trainer(_model(), mse_loss, _cfg(tmp_path, "c", schedule="cosine"), _Stream(ds),
                dist_info=SINGLE)


def test_lr_mult_zero_freezes_params_without_grad_buildup(tmp_path):
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(4, 4)
            self.head = nn.Linear(4, 1)

        def forward(self, x):
            return self.head(self.enc(x))

    torch.manual_seed(0)
    m = Two()
    enc0 = copy.deepcopy(m.enc.state_dict())
    tr = Trainer(m, mse_loss, _cfg(tmp_path, max_epochs=2, lr_mult={"enc": 0.0}), _data(32),
                 dist_info=SINGLE)
    tr.fit()
    for k, v in enc0.items():
        torch.testing.assert_close(m.enc.state_dict()[k], v)             # frozen
    assert all(p.grad is None for p in m.parameters())                   # no stale grads
    assert not torch.allclose(m.head.weight, Two().head.weight)


def test_eval_every_zero_falls_back_to_train_monitor(tmp_path):
    tr = Trainer(_model(), mse_loss, _cfg(tmp_path, max_epochs=2, monitor="val/loss",
                                          eval_every_epochs=0),
                 _data(32), _data(16, seed=1), dist_info=SINGLE)
    with pytest.warns(UserWarning, match="eval_every_epochs"):
        tr.fit()
    assert all("val/loss" not in r for r in tr.history)
    assert tr.best_value == min(r["train/loss"] for r in tr.history)
    assert (tmp_path / "run" / BEST_NAME).exists()


def test_resume_auto_without_checkpoint_starts_fresh(tmp_path):
    tr = Trainer(_model(), mse_loss, _cfg(tmp_path, resume="auto", max_epochs=1), _data(32),
                 dist_info=SINGLE)
    tr.fit()
    assert tr.resumed_from is None and tr.step == 2


@pytest.mark.parametrize("mode,best_epoch", [("min", 4), ("max", 5)])
def test_best_checkpoint_by_monitor(tmp_path, mode, best_epoch):
    seq = [3.0, 1.0, 2.0, 0.5, 4.0]

    def cb(trainer, row):
        return {"val/custom": seq[row["epoch"] - 1]}

    cfg = _cfg(tmp_path, max_epochs=5, monitor="val/custom", monitor_mode=mode)
    tr = Trainer(_model(), mse_loss, cfg, _data(32), callbacks=[cb], extra_state={"note": "x"},
                 dist_info=SINGLE)
    tr.fit()
    assert tr.best_epoch == best_epoch and tr.best_value == seq[best_epoch - 1]
    best = load_checkpoint(find_best(tmp_path / "run"))
    assert best["epoch"] == best_epoch and best["step"] == best_epoch * 2
    assert best["best"]["value"] == seq[best_epoch - 1] and best["best"]["monitor"] == "val/custom"
    assert best["extra"] == {"note": "x"}
    assert load_checkpoint(find_last(tmp_path / "run"))["epoch"] == 5


def test_monitor_falls_back_to_train_loss_without_val(tmp_path):
    tr = Trainer(_model(), mse_loss, _cfg(tmp_path, max_epochs=3, monitor="val/loss"),
                 _data(32), dist_info=SINGLE)
    with pytest.warns(UserWarning, match="no validation data"):
        tr.fit()
    losses = [r["train/loss"] for r in tr.history]
    assert tr.best_value == min(losses) and (tmp_path / "run" / BEST_NAME).exists()


def test_early_stopping(tmp_path):
    seq = [1.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    cfg = _cfg(tmp_path, max_epochs=10, monitor="val/custom", early_stop_patience=2)
    tr = Trainer(_model(), mse_loss, cfg, _data(32),
                 callbacks=[lambda t, row: {"val/custom": seq[row["epoch"] - 1]}],
                 dist_info=SINGLE)
    hist = tr.fit()
    assert len(hist) == 4 and tr.stopped_early and tr.best_epoch == 2
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["stopped_early"] is True and summary["best"]["epoch"] == 2
    assert load_checkpoint(tmp_path / "run" / LAST_NAME)["epoch"] == 4


def test_min_delta_counts_small_improvements_as_bad(tmp_path):
    seq = [1.0, 0.99, 0.985, 0.98]
    cfg = _cfg(tmp_path, max_epochs=4, monitor="val/custom", early_stop_patience=2,
               early_stop_min_delta=0.05)
    tr = Trainer(_model(), mse_loss, cfg, _data(32),
                 callbacks=[lambda t, row: {"val/custom": seq[row["epoch"] - 1]}],
                 dist_info=SINGLE)
    assert len(tr.fit()) == 3 and tr.best_epoch == 1


def test_ema_weights_differ_and_evaluate_uses_them(tmp_path):
    val = _data(32, seed=1)
    cfg = _cfg(tmp_path, max_epochs=3, ema_decay=0.8, ema_warmup=False, lr=0.05)
    tr = Trainer(_model(), mse_loss, cfg, _data(64, noise=0.1), val, dist_info=SINGLE)
    tr.fit()
    assert not torch.allclose(tr.ema.shadow["weight"], tr.model.weight.detach())

    def manual(state):
        m = _model()
        m.load_state_dict(state)
        with torch.no_grad():
            return mse_loss(m, val.tensors)["loss"].item()

    ev_ema = tr.evaluate()["val/loss"]
    ev_raw = tr.evaluate(use_ema=False)["val/loss"]
    assert ev_ema == pytest.approx(manual(tr.ema.shadow), rel=1e-5)
    assert ev_raw == pytest.approx(manual(tr.model.state_dict()), rel=1e-5)
    assert ev_ema != pytest.approx(ev_raw, rel=1e-4)
    # evaluate restored the raw weights and training mode
    assert tr.model.training
    assert tr.evaluate(use_ema=False)["val/loss"] == pytest.approx(ev_raw)
    # inference helper
    m = Trainer.load_model_weights(_model(seed=9), tmp_path / "run" / LAST_NAME, use_ema=True)
    torch.testing.assert_close(m.weight, tr.ema.shadow["weight"])
    m = Trainer.load_model_weights(_model(seed=9), tmp_path / "run" / LAST_NAME, use_ema=False)
    torch.testing.assert_close(m.weight, tr.model.weight)


def test_max_steps_takes_precedence(tmp_path):
    tr = Trainer(_model(), mse_loss, _cfg(tmp_path, max_steps=5, max_epochs=1), _data(64),
                 dist_info=SINGLE)                                # 4 steps / epoch
    hist = tr.fit()
    assert tr.step == 5 and len(hist) == 2 and hist[-1]["partial"] is True


def test_nonfinite_loss_raises(tmp_path):
    def bad_loss(model, batch):
        return {"loss": mse_loss(model, batch)["loss"] * float("nan")}

    tr = Trainer(_model(), bad_loss, _cfg(tmp_path, max_epochs=1), _data(32), dist_info=SINGLE)
    with pytest.raises(FloatingPointError):
        tr.fit()


def test_bf16_autocast_on_cpu_trains(tmp_path):
    cfg = _cfg(tmp_path, precision="bf16", max_epochs=2)
    tr = Trainer(_model(), mse_loss, cfg, _data(32), dist_info=SINGLE)
    assert tr.precision.name == "bf16" and tr.scaler is None
    hist = tr.fit()
    assert all(math.isfinite(r["train/loss"]) for r in hist)


def _cpu_fp16_autocast_ok() -> bool:
    try:
        with torch.autocast("cpu", dtype=torch.float16):
            return (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.float16
    except Exception:
        return False


@pytest.mark.skipif(not _cpu_fp16_autocast_ok(), reason="no CPU fp16 autocast in this torch")
def test_fp16_grad_scaler_path_and_resume(tmp_path, monkeypatch):
    """The fp16 + GradScaler branch (Turing/Volta GPUs) exercised on CPU: unscale → clip → step
    → update, scaler state checkpointed and restored exactly on resume."""
    from robot_skin.train import engine
    from robot_skin.train.hardware import PrecisionPlan

    monkeypatch.setattr(engine, "resolve_precision",
                        lambda pref, dev: PrecisionPlan("fp16", torch.float16, True, "cpu"))
    ds = _data(64, noise=0.1)

    def loss16(model, batch):
        x, y = batch
        return {"loss": ((model(x).squeeze(-1).float() - y) ** 2).mean()}

    kw = dict(max_epochs=4, batch_size=8, grad_accum=2, lr=0.05, grad_clip=1.0)
    full = Trainer(_model(), loss16, _cfg(tmp_path, "full", **kw), ds, dist_info=SINGLE)
    assert full.scaler is not None and full.precision.name == "fp16"
    hist = full.fit()
    assert hist[-1]["train/loss"] < hist[0]["train/loss"]
    ck = load_checkpoint(tmp_path / "full" / LAST_NAME)
    assert ck["scaler"]["scale"] == full.scaler.get_scale()

    part = Trainer(_model(), loss16, _cfg(tmp_path, "p", **kw), ds,
                   callbacks=[_stop_after(2)], dist_info=SINGLE)
    part.fit()
    res = Trainer(_model(seed=5), loss16, _cfg(tmp_path, "p", resume="auto", **kw), ds,
                  dist_info=SINGLE)
    res.fit()
    assert res.scaler.get_scale() == full.scaler.get_scale()
    torch.testing.assert_close(_params(res.model), _params(full.model), atol=1e-6, rtol=1e-6)


# ───────────────────────────────────────────────────────────────────────────── helpers

NT = collections.namedtuple("NT", ["x", "tag"])


@dataclasses.dataclass
class DC:
    t: torch.Tensor
    name: str


def test_move_to_device_nested_batch():
    batch = collections.OrderedDict(
        a=torch.ones(2, 3),
        b=[torch.zeros(2), "text", 3],
        c=(torch.ones(2), {"d": torch.arange(2)}),
        nt=NT(torch.ones(2, 1), "tag"),
        dc=DC(torch.ones(2), "n"),
        instr=["pick up the cup", "pour water"],
        arr=np.zeros(2),
    )
    out = move_to_device(batch, "meta")
    assert isinstance(out, collections.OrderedDict)
    assert out["a"].device.type == "meta"
    assert isinstance(out["b"], list) and out["b"][0].is_meta and out["b"][1:] == ["text", 3]
    assert isinstance(out["c"], tuple) and out["c"][0].is_meta and out["c"][1]["d"].is_meta
    assert isinstance(out["nt"], NT) and out["nt"].x.is_meta and out["nt"].tag == "tag"
    assert isinstance(out["dc"], DC) and out["dc"].t.is_meta and out["dc"].name == "n"
    assert out["instr"] == batch["instr"] and out["arr"] is batch["arr"]
    assert batch["a"].device.type == "cpu"  # input untouched
    dd = collections.defaultdict(list, {"x": torch.ones(1)})
    assert move_to_device(dd, "meta")["x"].is_meta


def test_trainconfig_from_dict():
    with pytest.warns(UserWarning, match="unknown keys.*bogus"):
        cfg = TrainConfig.from_dict({"lr": "3e-4", "betas": [0.9, 0.95], "max_steps": "none",
                                     "batch_size": 32.0, "compile": "true", "resume": True,
                                     "grad_clip": None, "lr_mult": {"vision": 0.1},
                                     "bogus": 1})
    assert cfg.lr == pytest.approx(3e-4) and isinstance(cfg.lr, float)
    assert cfg.betas == (0.9, 0.95) and cfg.max_steps is None and cfg.batch_size == 32
    assert cfg.compile is True and cfg.resume == "auto" and cfg.grad_clip is None
    assert cfg.lr_mult == {"vision": 0.1}
    assert TrainConfig.from_dict(None) == TrainConfig()
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    json.dumps(cfg.to_dict())
    with pytest.raises(KeyError):
        TrainConfig.from_dict({"bogus": 1}, strict=True)
    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig.from_dict({"batch_size": "abc"})
    with pytest.raises(ValueError, match="grad_accum"):
        TrainConfig.from_dict({"grad_accum": 0})
    with pytest.raises(ValueError, match="precision"):
        TrainConfig.from_dict({"precision": "int8"})
    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig.from_dict({"batch_size": 2.5})


def test_seed_everything_is_deterministic():
    seed_everything(5)
    a = (torch.randn(3), np.random.rand(3))
    seed_everything(5)
    b = (torch.randn(3), np.random.rand(3))
    torch.testing.assert_close(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


# ───────────────────────────────────────────────────────────────────────────── optim

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 8)
        self.lin = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.bn = nn.BatchNorm1d(8)
        self.cls_token = nn.Parameter(torch.zeros(1, 8))
        self.query = nn.Linear(8, 8)                      # a projection named "query": decayed
        self.readout_queries = nn.Parameter(torch.zeros(4, 8))
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, 8))
        self.head = nn.Linear(8, 2, bias=False)
        self.frozen = nn.Linear(8, 8)
        self.frozen.requires_grad_(False)


def test_param_groups_excludes_bias_norm_embedding():
    net = _Net()
    decay, no_decay = split_decay_params(net)
    assert {n for n, _ in decay} == {"lin.weight", "query.weight", "head.weight"}
    assert {n for n, _ in no_decay} == {
        "emb.weight", "lin.bias", "norm.weight", "norm.bias", "bn.weight", "bn.bias",
        "cls_token", "query.bias", "readout_queries", "pos_embed"}
    groups = param_groups(net, weight_decay=0.05)
    assert [g["weight_decay"] for g in groups] == [0.05, 0.0]
    assert sum(len(g["params"]) for g in groups) == 13  # frozen params excluded

    groups = param_groups(net, 0.05, lr=1e-3, lr_mult={"head": 0.1, "emb": 0.0})
    by_name = {g["group_name"]: g for g in groups}
    assert by_name["decay@0.1"]["lr"] == pytest.approx(1e-4)
    assert by_name["decay@0.1"]["params"][0] is net.head.weight
    assert all(p is not net.emb.weight for g in groups for p in g["params"])

    opt = build_optimizer(net, "adamw", lr=1e-3, weight_decay=0.05, lr_mult={"head": 0.1})
    assert isinstance(opt, torch.optim.AdamW)
    assert sorted(g["lr"] for g in opt.param_groups) == pytest.approx([1e-4, 1e-3, 1e-3])
    assert isinstance(build_optimizer(net, "sgd", lr=0.1), torch.optim.SGD)
    with pytest.raises(ValueError):
        build_optimizer(net, "lion")
    with pytest.raises(ValueError):
        param_groups(nn.Linear(2, 2).requires_grad_(False))


def test_param_groups_norm_by_class_name_and_lr_mult_typo():
    class NormalEncoder(nn.Module):          # encodes taxel surface normals — NOT a norm layer
        def __init__(self):
            super().__init__()
            self.proj = nn.Parameter(torch.zeros(3, 8))

    class ChannelLayerNorm2d(nn.Module):     # custom norm layer (by class-name convention)
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1, 8))

    net = nn.ModuleDict({"nrm": NormalEncoder(), "ln": ChannelLayerNorm2d()})
    decay, no_decay = split_decay_params(net)
    assert [n for n, _ in decay] == ["nrm.proj"] and [n for n, _ in no_decay] == ["ln.scale"]
    with pytest.warns(UserWarning, match="vison"):
        param_groups(net, 0.05, lr_mult={"vison": 0.1, "nrm": 0.5})


def test_scheduler_warmup_cosine_values():
    f = lambda s: lr_factor(s, "cosine", warmup_steps=10, total_steps=110, min_lr_ratio=0.1)  # noqa: E731
    assert f(0) == pytest.approx(0.1) and f(4) == pytest.approx(0.5) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.1 + 0.9 * 0.5)
    assert f(35) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.25)))
    assert f(110) == pytest.approx(0.1) and f(500) == pytest.approx(0.1)
    assert lr_factor(35, "linear", 10, 110, 0.1) == pytest.approx(0.1 + 0.9 * 0.75)
    assert lr_factor(3, "constant", 10) == pytest.approx(0.4)
    assert lr_factor(50, "constant", 10) == 1.0

    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=2.0)
    sched = build_scheduler(opt, "cosine", warmup_steps=10, total_steps=110, min_lr_ratio=0.1)
    seen = []
    for _ in range(120):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert seen == pytest.approx([2.0 * f(s) for s in range(120)])
    with pytest.raises(ValueError):
        build_scheduler(opt, "cosine", total_steps=None)
    with pytest.raises(ValueError):
        build_scheduler(opt, "step", total_steps=10)


def test_ema_update_math_and_swap():
    m = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(0.0)
    ema = EMA(m, decay=0.5, warmup=False)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m)
    torch.testing.assert_close(ema.shadow["weight"], torch.full((1, 2), 0.5))
    ema.update(m)
    torch.testing.assert_close(ema.shadow["weight"], torch.full((1, 2), 0.75))
    with ema.swap(m):
        torch.testing.assert_close(m.weight, torch.full((1, 2), 0.75))
    torch.testing.assert_close(m.weight, torch.full((1, 2), 1.0))
    st = ema.state_dict()
    ema2 = EMA(m, decay=0.5)
    ema2.load_state_dict(st)
    assert ema2.num_updates == 2
    torch.testing.assert_close(ema2.shadow["weight"], ema.shadow["weight"])
    assert EMA(m, 0.999).current_decay() < 0.2  # warm-up keeps early EMA close to the model
    with pytest.raises(ValueError):
        EMA(m, decay=1.0)


# ───────────────────────────────────────────────────────────────────────────── checkpoint / log

def test_checkpoint_atomic_save_load_find(tmp_path, monkeypatch):
    p = save_checkpoint(tmp_path / "a" / LAST_NAME, step=3, t=torch.arange(3))
    ck = load_checkpoint(p)
    assert ck["step"] == 3 and torch.equal(ck["t"], torch.arange(3))
    assert load_checkpoint(tmp_path / "a")["step"] == 3  # directory → find_last

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError):
        save_checkpoint(p, step=4)
    monkeypatch.undo()
    assert load_checkpoint(p)["step"] == 3                     # old file intact
    assert sorted(x.name for x in (tmp_path / "a").iterdir()) == [LAST_NAME]  # no tmp left

    d = tmp_path / "b"
    assert find_last(d) is None and find_last(tmp_path / "missing") is None
    save_checkpoint(d / "ckpt_epoch0002.pt", step=2)
    save_checkpoint(d / "ckpt_epoch0010.pt", step=10)
    save_checkpoint(d / BEST_NAME, step=5)
    assert find_last(d).name == "ckpt_epoch0010.pt" and find_best(d).name == BEST_NAME
    save_checkpoint(d / LAST_NAME, step=11)
    assert find_last(d).name == LAST_NAME
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")
    assert strip_state_dict_prefixes({"module._orig_mod.w": 1, "b": 2}) == {"w": 1, "b": 2}


def test_jsonl_logger(tmp_path):
    with JsonlLogger(tmp_path / "log") as lg:
        lg.log(1, {"loss": torch.tensor(0.5), "n": np.int64(3), "name": "x", "bad": float("nan")},
               kind="step")
        lg.log(2, {"loss": 0.25})
        lg.log(3, {"arr": np.array([1.0, np.nan]), "tnan": torch.tensor(float("nan")),
                   "vec": torch.tensor([1.0, float("inf")])})
    recs = read_jsonl(tmp_path / "log" / "metrics.jsonl")
    assert [r["step"] for r in recs] == [1, 2, 3] and recs[0]["loss"] == 0.5 and recs[0]["n"] == 3
    assert recs[0]["bad"] == "nan" and read_jsonl(tmp_path / "log" / "metrics.jsonl", kind="step")
    assert recs[2]["arr"] == [1.0, "nan"] and recs[2]["tnan"] == "nan"
    assert recs[2]["vec"] == [1.0, "inf"]
    for line in (tmp_path / "log" / "metrics.jsonl").read_text().splitlines():
        json.loads(line, parse_constant=lambda c: pytest.fail(f"invalid JSON constant {c}"))
    off = JsonlLogger(tmp_path / "off", enabled=False)
    off.log(1, {"x": 1})
    assert not (tmp_path / "off").exists()
    if importlib.util.find_spec("tensorboard") is None:
        with pytest.warns(UserWarning, match="TensorBoard"):
            JsonlLogger(tmp_path / "tb", tensorboard=True).close()


def test_trainer_copy_independence(tmp_path):
    """Two trainers on deep copies do not share parameters (guards against aliasing bugs)."""
    base = _model()
    a = Trainer(copy.deepcopy(base), mse_loss, _cfg(tmp_path, "a", max_epochs=1), _data(32),
                dist_info=SINGLE)
    a.fit()
    torch.testing.assert_close(_params(base), _params(_model()))
