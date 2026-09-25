"""Stage ``baseline``: temporal no-contact baseline (motion artefact) predictor.

Pipeline position: after preprocessing (and ``imu_pose`` when ``data.q_source: hand_pose_imu``),
before ``contact``. Trains :class:`~robot_skin.baseline.temporal.TemporalBaselinePredictor` on the
**no-contact** frames of D1 (``motion``) episodes — ``contact_label`` ∈ ``only_labels`` (0), not
saturated, joint state measured over the whole window — then predicts every episode (D1 + D2):

    derived/baseline_pred   [T,N] float32  predicted no-contact ΔS %
    derived/baseline_logvar [T,N] float32  predicted log-variance (%²)
    derived/residual        [T,N] float32  delta_pct − baseline_pred (SATS sign: press → negative)

``run(cfg) -> metrics`` (config ``robot_skin/configs/stages/baseline.yaml``; missing keys fall back
to :data:`DEFAULTS`, unknown keys raise):

1. episodes: ``data.processed_root`` × (``data.datasets`` ∪ ``data.predict_datasets``) or
   ``data.episodes``; training pool = ``datasets`` (default D1), split with ``data.splits`` /
   ``val_frac`` (:func:`robot_skin.stages.split_stage_episodes`, same semantics as ``pretrain``);
2. ``q``/``qd`` normalisation (``datasets.stats.compute_stats`` on the train split, hand-valid
   frames) and the per-taxel target scale (RMS no-contact ΔS) → model buffers (the checkpoint is
   self-contained; ``forward`` takes raw units);
3. :class:`~robot_skin.datasets.motion.BaselineWindowDataset` (causal windows ``W = model.window``,
   ``joint_stats=None`` — the model normalises) → :class:`robot_skin.train.Trainer` with
   :func:`~robot_skin.baseline.temporal.baseline_loss` (MSE mean + Gaussian NLL variance, Kendall &
   Gal 2017); the best checkpoint (``train.monitor``) is restored;
4. evaluation per split (``val/``, ``test/``, and ``<dataset>/`` for prediction-only datasets such
   as ``task/``) on no-contact frames: ``mae_raw`` (|ΔS|, i.e. no baseline) vs ``mae_resid``
   (|ΔS − pred|) and ``resid_reduction = 1 − mae_resid / mae_raw``, NLL, calibration (``z_std``,
   ``coverage_2sigma``), motion-vs-contact separability of press intensity before/after
   subtraction (``eval.motion_contact_separability``: positives = self-touch labels, negatives =
   no-contact while the joints move), and — synthetic data (``gt_artefact_pct``) — the error to
   the true artefact on no-contact / contact frames;
5. writes the derived arrays, ``<out_dir>/baseline_model.pt`` (:func:`load_baseline_model`, meta:
   ``q_source``, ``window``, ``qd`` — the ``joint_velocity`` kwargs the online processor must reuse,
   ``joint_velocity(q_buffer, hz, **meta["qd"])`` — and ``qd_source`` (``derivative`` | ``file``),
   joint names, layout), ``joint_stats.json`` and ``metrics.json``.

The observed ΔS is never a model input (``deformable_sats/sats/bending`` lesson). The prediction of
frame t is causal and reproducible online with :class:`~robot_skin.baseline.temporal.CausalBaselineStream`.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import (finite_json, fit_and_restore, load_stage_yaml, parse_overrides, resolve_stage_config,
               seed_model_init, split_stage_episodes, stage_episodes, write_json_atomic)

log = logging.getLogger("robot_skin.stages.baseline")

STAGE = "baseline"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "stages" / "baseline.yaml"
METRICS_NAME = "metrics.json"
STATS_NAME = "joint_stats.json"

#: built-in defaults (mirrored by configs/stages/baseline.yaml; a test keeps them in sync)
DEFAULTS: dict[str, Any] = {
    "stage": STAGE,
    "hardware": None,
    "hardware_applied": False,
    "out_dir": None,
    "data": {
        "processed_root": "robot_skin/data/processed",
        "datasets": ["motion"],
        "predict_datasets": ["motion", "task"],
        "episodes": None,
        "splits": None,
        "use_test": False,
        "split_by": "episode",
        "val_frac": 0.2,
        "split_seed": 0,
        "kind": None,
        "stride": 2,
        "q_source": "q",
        "only_labels": [0],
        "exclude_saturated": True,
        "min_valid": 1,
        "require_q_valid": True,
    },
    "stats": {"method": "std"},
    "target_scale": {"enabled": True, "floor_pct": 0.05},
    "model": {"window": 32, "hidden": 64, "taxel_emb_dim": 8, "kernel": 3, "n_layers": 4, "arch": "tcn",
              "head_hidden": 64, "dropout": 0.0, "pos_scale": 10.0, "use_pose": True, "sigma0": 1.0,
              "logvar_min": -12.0, "logvar_max": 8.0, "var_detach": True},
    "loss": {"mean_loss": "mse", "mean_weight": 1.0, "nll_weight": 1.0, "detach_mean": True, "huber_delta": 1.0},
    "eval": {"moving_qd_rms": 0.2},
    "predict": {"write_derived": True, "batch_size": 4096},
    "train": {
        "max_epochs": 30,
        "batch_size": 256,
        "lr": 3.0e-3,
        "weight_decay": 1.0e-4,
        "warmup_steps": 100,
        "schedule": "cosine",
        "min_lr_ratio": 0.05,
        "grad_clip": 1.0,
        "precision": "auto",
        "ema_decay": None,
        "monitor": "val/loss",
        "log_every": 50,
        "seed": 0,
        "out_dir": "robot_skin/runs/baseline",
    },
}


def load_stage_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict:
    """:data:`DEFAULTS` ⊕ YAML (default ``configs/stages/baseline.yaml``) ⊕ hardware ⊕ overrides."""
    return load_stage_yaml(DEFAULTS, STAGE, CONFIG_PATH, path, overrides)


def resolve_config(cfg: Mapping[str, Any] | None) -> dict:
    return resolve_stage_config(cfg, DEFAULTS, STAGE)


# ─────────────────────────────────────────────────────────────── helpers

def _usable(ep, q_source: str) -> str | None:
    """Reason the episode cannot be used by this stage (None if fine)."""
    from ..datasets.episode import D_HAND_POSE_IMU, K_CONTACT_LABEL, K_DELTA, K_Q, K_QD, K_TAXEL_NRM, K_TAXEL_POS

    need = [K_DELTA, K_TAXEL_POS, K_TAXEL_NRM, K_CONTACT_LABEL]
    if q_source == "q":
        need += [K_Q, K_QD]
    missing = [k for k in need if not ep.has(k)]
    if q_source == "hand_pose_imu" and not ep.has_derived(D_HAND_POSE_IMU):
        missing.append(f"derived/{D_HAND_POSE_IMU}")
    return f"missing {missing}" if missing else None


def target_scale(views: Sequence, only_labels: Sequence[int], floor_pct: float) -> np.ndarray:
    """Per-taxel RMS of ΔS over the no-contact, unsaturated frames of ``views`` (floored)."""
    from ..datasets.episode import K_CONTACT_LABEL, K_DELTA, K_SATURATED

    ss = cnt = None
    for ep in views:
        d = np.asarray(ep[K_DELTA], dtype=np.float64)
        m = np.isin(np.asarray(ep[K_CONTACT_LABEL]), list(only_labels)) & np.isfinite(d)
        if ep.has(K_SATURATED):
            m &= ~np.asarray(ep[K_SATURATED], dtype=bool)
        a, b = (np.where(m, d, 0.0) ** 2).sum(0), m.sum(0)
        ss, cnt = (a, b) if ss is None else (ss + a, cnt + b)
    rms = np.sqrt(ss / np.maximum(cnt, 1))
    rms = np.where(cnt > 0, rms, np.nanmedian(np.where(cnt > 0, rms, np.nan)) if (cnt > 0).any() else 1.0)
    return np.maximum(rms, float(floor_pct)).astype(np.float32)


def _loss(model, batch, *, loss_cfg: Mapping[str, Any]):
    from ..baseline.temporal import baseline_loss
    from ..train import unwrap_model

    mean, lv = model(batch["q_hist"], batch["qd_hist"], batch["pos"], batch["nrm"])
    return baseline_loss(mean, lv, batch["y"], batch["valid"], unwrap_model(model).y_scale, **loss_cfg)


def _sep(delta: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> dict[str, float]:
    """``eval.motion_contact_separability`` on pre-selected samples (ΔS-space values ``delta``;
    positives ``pos``, negatives ``neg`` = no-contact while moving)."""
    from ..eval.metrics import motion_contact_separability

    sel = pos | neg
    if pos.sum() < 2 or neg.sum() < 2:
        return {"auroc": float("nan"), "dprime": float("nan")}
    return motion_contact_separability(delta[sel][:, None], pos[sel][:, None], np.ones(int(sel.sum()), bool))


def evaluate_baseline(model, episodes: Sequence, *, q_source: str = "q", only_labels: Sequence[int] = (0,),
                      moving_qd_rms: float = 0.2, batch_size: int = 4096,
                      predictions: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
                      views: Mapping[int, Any] | None = None) -> dict[str, float]:
    """No-contact residual metrics over ``episodes`` (pooled over all samples; stage docstring).
    ``predictions`` (``{id(ep): (mean, logvar)}``) and ``views`` (``{id(ep): joint-state view}``)
    avoid recomputing."""
    from ..baseline.temporal import episode_joint_view, predict_episode
    from ..datasets.episode import K_CONTACT_LABEL, K_DELTA, K_QD, K_SATURATED
    from ..datasets.stats import qd_valid_mask

    acc: dict[str, float] = {}

    def add(k: str, v: float, n: float = 1.0) -> None:
        acc[k] = acc.get(k, 0.0) + float(v)
        acc["_n_" + k] = acc.get("_n_" + k, 0.0) + float(n)

    scores: dict[str, list] = {"raw": [], "res": [], "pos": [], "neg": []}
    zs = []
    for ep in episodes:
        view = views[id(ep)] if views is not None and id(ep) in views else episode_joint_view(ep, q_source)
        if predictions is not None and id(ep) in predictions:
            mean, lv = predictions[id(ep)]
        else:
            mean, lv = predict_episode(model, view, batch_size=batch_size)
        d = np.asarray(ep[K_DELTA], dtype=np.float64)
        lab = np.asarray(ep[K_CONTACT_LABEL])
        sat = np.asarray(ep[K_SATURATED], dtype=bool) if ep.has(K_SATURATED) else np.zeros(d.shape, bool)
        qv = qd_valid_mask(view)[:, None]
        ok = np.isfinite(d) & ~sat & qv
        nc = np.isin(lab, list(only_labels)) & ok
        r = d - mean
        var = np.exp(lv.astype(np.float64))
        if nc.any():
            add("mae_raw", np.abs(d[nc]).sum(), nc.sum())
            add("mae_resid", np.abs(r[nc]).sum(), nc.sum())
            add("mse_raw", (d[nc] ** 2).sum(), nc.sum())
            add("mse_resid", (r[nc] ** 2).sum(), nc.sum())
            add("nll", (0.5 * (np.log(2 * np.pi * var[nc]) + r[nc] ** 2 / var[nc])).sum(), nc.sum())
            add("coverage_2sigma", (np.abs(r[nc]) <= 2 * np.sqrt(var[nc])).sum(), nc.sum())
            zs.append(r[nc] / np.sqrt(var[nc]))
        if ep.has("gt_artefact_pct"):
            g = np.asarray(ep["gt_artefact_pct"], dtype=np.float64)
            add("gt_artefact_abs", np.abs(g[nc]).sum(), nc.sum())
            add("gt_artefact_mae", np.abs(mean - g)[nc].sum(), nc.sum())
            cm = (np.asarray(ep["gt_contact"], dtype=bool) if ep.has("gt_contact") else lab == 1) & ok
            if cm.any():
                add("gt_artefact_abs_contact", np.abs(g[cm]).sum(), cm.sum())
                add("gt_artefact_mae_contact", np.abs(mean - g)[cm].sum(), cm.sum())
        speed = np.sqrt(np.mean(np.asarray(view[K_QD], dtype=np.float64) ** 2, axis=1))
        moving = (speed > float(moving_qd_rms))[:, None]
        pos = (lab == 1) & ok
        neg = np.isin(lab, list(only_labels)) & ok & moving
        scores["raw"].append(d[pos | neg])                  # ΔS-space: the metric takes press_intensity
        scores["res"].append(r[pos | neg])
        scores["pos"].append(pos[pos | neg])
        scores["neg"].append(neg[pos | neg])
    out: dict[str, float] = {}
    for k in [k for k in acc if not k.startswith("_n_")]:
        out[k] = acc[k] / max(acc["_n_" + k], 1.0)
    for k in ("mse_raw", "mse_resid"):
        if k in out:
            out["rmse" + k[3:]] = math.sqrt(out.pop(k))
    if "mae_raw" in out and out["mae_raw"] > 0:
        out["resid_reduction"] = 1.0 - out["mae_resid"] / out["mae_raw"]
    if "gt_artefact_abs" in out and out["gt_artefact_abs"] > 0:
        out["gt_artefact_reduction"] = 1.0 - out["gt_artefact_mae"] / out["gt_artefact_abs"]
    if zs:
        z = np.concatenate(zs)
        out["z_std"] = float(np.std(z))
        out["z_robust_std"] = float(1.4826 * np.median(np.abs(z - np.median(z))))
    out["n_no_contact"] = int(acc.get("_n_mae_raw", 0))
    if scores["pos"]:
        pos, neg = np.concatenate(scores["pos"]), np.concatenate(scores["neg"])
        out["n_sep_pos"], out["n_sep_neg"] = int(pos.sum()), int(neg.sum())
        for tag, key in (("before", "raw"), ("after", "res")):
            s = _sep(np.concatenate(scores[key]), pos, neg)
            out[f"sep_auroc_{tag}"], out[f"sep_dprime_{tag}"] = s["auroc"], s["dprime"]
    return out


# ─────────────────────────────────────────────────────────────── run

def run(cfg: Mapping[str, Any] | None = None) -> dict:
    """Train the temporal baseline, write derived baseline_pred/logvar/residual; returns metrics."""
    from ..baseline.temporal import (BASELINE_MODEL_NAME, MEAN_LOSSES, Q_SOURCES, TemporalBaselinePredictor,
                                     episode_joint_view, qd_settings)
    from ..datasets.episode import K_Q, K_QD
    from ..datasets.motion import BaselineWindowDataset
    from ..datasets.stats import compute_stats

    cfg = resolve_config(cfg)
    out_dir = Path(cfg["out_dir"])
    d_cfg, m_cfg = cfg["data"], dict(cfg["model"])
    q_source = d_cfg["q_source"]
    if q_source not in Q_SOURCES:
        raise ValueError(f"data.q_source must be one of {Q_SOURCES}, got {q_source!r}")
    l_cfg = cfg["loss"]
    if l_cfg["mean_loss"] not in MEAN_LOSSES:
        raise ValueError(f"loss.mean_loss must be one of {MEAN_LOSSES}, got {l_cfg['mean_loss']!r}")
    if l_cfg["mean_loss"] == "none" and (l_cfg["detach_mean"] or not float(l_cfg["nll_weight"]) > 0):
        raise ValueError("loss.mean_loss: none trains the mean through the NLL only — it needs "
                         "detach_mean: false and nll_weight > 0 (else the mean head never learns)")
    only = [int(v) for v in d_cfg["only_labels"]]
    train_sets, pred_sets = set(d_cfg["datasets"] or []), set(d_cfg["predict_datasets"] or [])

    eps, skipped = stage_episodes(d_cfg)
    usable = []
    for ep in eps:
        why = _usable(ep, q_source)
        if not why and d_cfg["kind"] and ep.meta.kind != d_cfg["kind"]:
            why = f"kind {ep.meta.kind!r} (data.kind = {d_cfg['kind']!r})"
        if why:
            skipped.append({"episode": str(ep.root), "reason": why})
        else:
            usable.append(ep)
    pool = [e for e in usable if e.meta.dataset in train_sets]
    if len({(e.meta.kind, e.meta.n_taxels) for e in pool}) > 1:
        raise ValueError("baseline: training episodes of different kinds / taxel counts "
                         f"{sorted({(e.meta.kind, e.meta.n_taxels) for e in pool})} — one model per layout: "
                         "set data.kind (glove | robot) or select episodes with data.episodes")
    split = split_stage_episodes(pool, d_cfg)
    if not split["train"]:
        raise ValueError(f"baseline: no usable training episodes in datasets {sorted(train_sets)} under "
                         f"{d_cfg.get('processed_root')!r} ({len(eps)} loaded, {len(skipped)} skipped)")
    views = {id(e): episode_joint_view(e, q_source) for e in pool}
    tr_v = [views[id(e)] for e in split["train"]]
    va_v = [views[id(e)] for e in split["val"]]
    # one model = one time base: the window is in frames and the online processor reproduces
    # exactly one qd definition (bundle_meta["hz"] / ["qd"])
    if len({float(v.meta.hz) for v in views.values()}) > 1:
        raise ValueError(f"baseline: training episodes have different master rates "
                         f"{sorted({float(v.meta.hz) for v in views.values()})} Hz — preprocess with one master_hz")
    if len({tuple(sorted(qd_settings(v.meta.preprocessing).items())) for v in views.values()}) > 1:
        warnings.warn("baseline: training episodes were preprocessed with different qd settings; the bundle "
                      "records those of the first train episode (the online processor can reproduce only one)",
                      stacklevel=2)

    stats = compute_stats(tr_v, keys=(K_Q, K_QD), method=cfg["stats"]["method"])
    N, D = tr_v[0].meta.n_taxels, tr_v[0][K_Q].shape[1]
    seed_model_init(cfg["train"])                 # reproducible initial weights (train.seed)
    model = TemporalBaselinePredictor(N, D, **m_cfg)
    model.set_joint_stats(stats[K_Q], stats[K_QD])
    ts_cfg = cfg["target_scale"]
    if ts_cfg["enabled"]:
        model.set_target_scale(target_scale(tr_v, only, float(ts_cfg["floor_pct"])))
    ds_kw = dict(window=model.window, stride=int(d_cfg["stride"]), only_labels=only, joint_stats=None,
                 exclude_saturated=bool(d_cfg["exclude_saturated"]), min_valid=int(d_cfg["min_valid"]),
                 require_q_valid=bool(d_cfg["require_q_valid"]))
    train_ds = BaselineWindowDataset(tr_v, **ds_kw)
    val_ds = None
    if va_v:
        try:
            val_ds = BaselineWindowDataset(va_v, **ds_kw)
        except ValueError as e:                  # e.g. a val episode without eligible frames
            log.warning("baseline: no validation windows (%s)", e)
    joint_names = list(tr_v[0].meta.joint_names)
    pre0 = tr_v[0].meta.preprocessing or {}
    # CTRL: qd = joint_velocity(q_buffer, hz, **bundle_meta["qd"]) — only the joint_velocity kwargs
    # (the preprocessing section also has a non-kwarg "source"); "file" = the driver's velocities
    bundle_meta = {"stage": STAGE, "q_source": q_source, "window": model.window, "joint_names": joint_names,
                   "n_taxels": N, "layout": tr_v[0].meta.layout, "kind": tr_v[0].meta.kind,
                   "hz": float(tr_v[0].meta.hz), "qd": qd_settings(pre0),
                   "qd_source": "file" if pre0.get("qd_source") == "file" else "derivative",
                   "only_labels": only, "loss": dict(cfg["loss"]),
                   "joint_stats": {k: v.to_dict() for k, v in stats.items()},
                   "target_scale": model.y_scale.tolist()}
    trainer = fit_and_restore(model, functools.partial(_loss, loss_cfg=dict(cfg["loss"])), cfg["train"],
                              train_ds, val_ds, extra_state={"stage": STAGE, "model_config": model.config,
                                                             "bundle_meta": bundle_meta})
    metrics: dict[str, Any] = {
        "stage": STAGE, "out_dir": str(out_dir), "model_path": str(out_dir / BASELINE_MODEL_NAME),
        "q_source": q_source, "window": model.window, "receptive_field": int(model.receptive_field),
        "n_episodes": {"loaded": len(eps), "train": len(split["train"]), "val": len(split["val"]),
                       "test": len(split["test"]), "skipped": len(skipped)},
        "n_samples": {"train": len(train_ds), "val": len(val_ds) if val_ds is not None else 0},
        "epochs": trainer.epoch, "steps": trainer.step,
        "best": {"monitor": trainer.cfg.monitor, "value": trainer.best_value, "epoch": trainer.best_epoch},
        "final_train_loss": trainer.history[-1].get("train/loss") if trainer.history else None,
    }
    if trainer.dist.is_main:
        metrics = _predict_evaluate_save(cfg, model, trainer, metrics, usable, views, split, skipped, stats,
                                         bundle_meta, q_source, only, train_sets, pred_sets, N, D, joint_names)
    from ..train import barrier

    barrier(trainer.dist)
    return finite_json(metrics)


def _predict_evaluate_save(cfg, model, trainer, metrics, usable, views, split, skipped, stats, bundle_meta,
                           q_source, only, train_sets, pred_sets, N, D, joint_names) -> dict:
    """Rank-0 part of :func:`run`: derived arrays for every episode, evaluation, bundle + metrics."""
    import torch

    from ..baseline.temporal import BASELINE_MODEL_NAME, episode_joint_view, predict_episode, save_baseline_model
    from ..datasets.episode import D_BASELINE_LOGVAR, D_BASELINE_PRED, D_RESIDUAL, K_DELTA, K_Q
    from ..datasets.stats import save_stats

    out_dir = Path(cfg["out_dir"])

    # ── predict every episode (derived arrays) ─────────────────────────────
    p_cfg, e_cfg = cfg["predict"], cfg["eval"]
    preds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    predicted = []
    for ep in usable:
        if ep.meta.dataset not in pred_sets and ep.meta.dataset not in train_sets:
            continue
        if id(ep) not in views:
            views[id(ep)] = episode_joint_view(ep, q_source)
        view = views[id(ep)]
        if view.meta.n_taxels != N or view[K_Q].shape[1] != D or list(view.meta.joint_names) != joint_names:
            skipped.append({"episode": str(ep.root), "reason": "layout / joint space differs from the training "
                            f"episodes (N={view.meta.n_taxels}, D={view[K_Q].shape[1]})"})
            continue
        mean, lv = predict_episode(model, view, batch_size=int(p_cfg["batch_size"]))
        preds[id(ep)] = (mean, lv)
        if ep.meta.dataset in pred_sets and p_cfg["write_derived"]:
            ep.set_derived(D_BASELINE_PRED, mean)
            ep.set_derived(D_BASELINE_LOGVAR, lv)
            ep.set_derived(D_RESIDUAL, (np.asarray(ep[K_DELTA], dtype=np.float32) - mean).astype(np.float32))
            predicted.append(str(ep.root))
    metrics["n_episodes"]["predicted"] = len(predicted)
    metrics["n_episodes"]["skipped"] = len(skipped)
    metrics["skipped"] = skipped

    # ── evaluation ───────────────────────────────────────────────────────────
    ev_kw = dict(q_source=q_source, only_labels=only, moving_qd_rms=float(e_cfg["moving_qd_rms"]),
                 batch_size=int(p_cfg["batch_size"]), predictions=preds, views=views)
    groups: dict[str, list] = {}
    if split["val"]:
        groups["val"] = split["val"]
    else:
        groups["train_eval"] = split["train"]
    if split["test"]:
        groups["test"] = split["test"]
    for ds in sorted(pred_sets - train_sets):
        g = [e for e in usable if e.meta.dataset == ds and id(e) in preds]
        if g:
            groups[ds] = g
    for name, g in groups.items():
        with torch.no_grad():
            res = evaluate_baseline(model, [e for e in g if id(e) in preds], **ev_kw)
        metrics.update({f"{name}/{k}": v for k, v in res.items()})

    bundle_meta["metrics"] = {k: v for k, v in metrics.items() if "/" in k}
    bundle_meta["episodes"] = {k: [str(e.root) for e in v] for k, v in split.items()}
    save_baseline_model(out_dir / BASELINE_MODEL_NAME, model, bundle_meta)
    save_stats(stats, out_dir / STATS_NAME, meta={"stage": STAGE, "q_source": q_source, "joint_names": joint_names})
    write_json_atomic(out_dir / METRICS_NAME, metrics)
    log.info("baseline done: %s", {k: v for k, v in metrics.items() if k.startswith(("val/", "train_eval/"))})
    return metrics


# ─────────────────────────────────────────────────────────────── CLI

def main(argv: Sequence[str] | None = None) -> int:
    """``python -m robot_skin.stages.baseline [--config yaml] [--set train.lr=1e-3 ...]``."""
    ap = argparse.ArgumentParser(prog="python -m robot_skin.stages.baseline",
                                 description="Temporal no-contact baseline (motion artefact) training.")
    ap.add_argument("--config", default=None, help=f"stage YAML (default {CONFIG_PATH})")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted override, value parsed as YAML (repeatable)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    metrics = run(load_stage_config(args.config, parse_overrides(args.set)))
    json.dump({k: v for k, v in metrics.items() if k != "skipped"}, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
