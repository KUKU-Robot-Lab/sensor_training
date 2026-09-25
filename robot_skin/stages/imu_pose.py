"""Stage ``imu_pose``: glove IMUs → MANO finger pose (in-house VIFNet-S-role baseline).

Pipeline position: first stage-1 runner, after preprocessing. Trains
:class:`robot_skin.pose.imu_model.ImuHandPoseNet` on episodes with vision / mocap hand labels
(``hand_finger_pose`` where ``hand_pose_valid``) and calibrated IMUs, then predicts the finger pose
of **every** glove episode with IMUs — the camera-free joint state for later stages and deployment::

    derived/hand_finger_pose_imu [T,15,3] float32  MANO axis-angle, MANO joint order

``run(cfg) -> metrics`` (config ``robot_skin/configs/stages/imu_pose.yaml``; missing keys fall back
to :data:`DEFAULTS`, unknown keys raise):

1. episodes: ``data.processed_root`` × (``data.datasets`` ∪ ``data.predict_datasets``) or
   ``data.episodes``; training pool = ``datasets`` episodes with hand labels and IMU arrays, split
   like every stage (``data.splits`` / ``val_frac``, :func:`robot_skin.stages.split_stage_episodes`);
2. feature statistics of ``pose.imu_model.imu_features`` on the train split
   (``datasets.stats.compute_stats`` key ``imu_features``) stored *inside* the model
   (``set_feature_stats``), so the checkpoint is self-contained;
3. :class:`~robot_skin.datasets.motion.ImuPoseWindowDataset` (causal windows, labelled frames) →
   :class:`robot_skin.train.Trainer` with ``pose.imu_model.hand_pose_loss`` (geodesic rotation +
   fingertip distance through ``pose.mano.ManoSkeleton``; 6D rotations, Zhou et al., CVPR 2019);
4. evaluation (``val/`` or ``train_eval/``, ``test/``): mean joint geodesic error (deg) and
   fingertip error (mm) vs the flat-hand prediction;
5. writes ``derived/hand_finger_pose_imu`` with ``predict_finger_pose_sequence`` (causal, same
   window), ``<out_dir>/imu_pose_model.pt`` (:func:`load_imu_pose_model`), ``imu_stats.json`` and
   ``metrics.json``.

The VIHand VIFNet-S backbone (Wang et al., ACM MM 2025) is the intended pretrained alternative
(``pose.glove_imu2mano.load_vifnet_s`` stub); this runner trains the in-house model with the same
role. Note: predictions on the model's own training episodes are optimistic; a baseline trained on
``q_source: hand_pose_imu`` should prefer episodes the IMU model did not see.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import (finite_json, fit_and_restore, load_stage_yaml, parse_overrides, resolve_stage_config,
               seed_model_init, split_stage_episodes, stage_episodes, write_json_atomic)

log = logging.getLogger("robot_skin.stages.imu_pose")

STAGE = "imu_pose"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "stages" / "imu_pose.yaml"
METRICS_NAME = "metrics.json"
STATS_NAME = "imu_stats.json"
MODEL_NAME = "imu_pose_model.pt"
MODEL_FORMAT = "robot_skin.stages.imu_pose/1"

#: built-in defaults (mirrored by configs/stages/imu_pose.yaml; a test keeps them in sync)
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
        "stride": 2,
    },
    "features": {"gyro": True, "acc": True, "vec_frame": "sensor"},
    "stats": {"method": "std"},
    "model": {"window": 32, "hidden": 128, "n_layers": 2, "arch": "gru", "dropout": 0.1, "kernel": 3},
    "loss": {"tip_weight": 10.0},
    "predict": {"write_derived": True, "batch_size": 2048},
    "train": {
        "max_epochs": 30,
        "batch_size": 128,
        "lr": 1.0e-3,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "schedule": "cosine",
        "min_lr_ratio": 0.05,
        "grad_clip": 1.0,
        "precision": "auto",
        "ema_decay": None,
        "monitor": "val/loss",
        "log_every": 50,
        "seed": 0,
        "out_dir": "robot_skin/runs/imu_pose",
    },
}


def load_stage_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict:
    """:data:`DEFAULTS` ⊕ YAML (default ``configs/stages/imu_pose.yaml``) ⊕ hardware ⊕ overrides."""
    return load_stage_yaml(DEFAULTS, STAGE, CONFIG_PATH, path, overrides)


def resolve_config(cfg: Mapping[str, Any] | None) -> dict:
    return resolve_stage_config(cfg, DEFAULTS, STAGE)


# ─────────────────────────────────────────────────────────────── model bundle

def save_imu_pose_model(path: str | Path, model, meta: Mapping[str, Any] | None = None) -> Path:
    """Atomic ``{format, config, state_dict, meta}`` (meta: window, feature flags, IMU sites …)."""
    from ..train.checkpoint import save_checkpoint

    p = Path(path)
    if p.suffix != ".pt":
        p = p / MODEL_NAME
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return save_checkpoint(p, format=MODEL_FORMAT, config=dict(model.config), state_dict=sd,
                           meta=finite_json(dict(meta or {})))


def load_imu_pose_model(path: str | Path, *, map_location: str = "cpu"):
    """Rebuild the ``ImuHandPoseNet`` (feature normalisation included); ``model.bundle_meta`` holds
    ``window``, ``features`` (gyro/acc/vec_frame), ``imu_sites``, ``wrist_index``."""
    import torch

    from ..pose.imu_model import ImuHandPoseNet

    p = Path(path)
    if p.is_dir():
        p = p / MODEL_NAME
    ck = torch.load(p, map_location=map_location, weights_only=True)
    if not isinstance(ck, Mapping) or ck.get("format") != MODEL_FORMAT:
        raise ValueError(f"{p} is not a {MODEL_FORMAT} bundle")
    model = ImuHandPoseNet.from_config(ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.bundle_meta = dict(ck.get("meta") or {})
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────── helpers

def _has_imu(ep, feat: Mapping[str, Any]) -> str | None:
    from ..datasets.episode import K_IMU_ACC, K_IMU_GYRO, K_IMU_QUAT

    need = [K_IMU_QUAT] + ([K_IMU_GYRO] if feat["gyro"] else []) + ([K_IMU_ACC] if feat["acc"] else [])
    missing = [k for k in need if not ep.has(k)]
    return f"missing {missing}" if missing else None


def _loss(model, batch, *, skeleton, tip_weight: float):
    from ..pose.imu_model import hand_pose_loss

    return hand_pose_loss(model(batch["feat"]), batch["finger_pose"], skeleton, tip_weight)


def evaluate_pose(model, dataset, skeleton, *, batch_size: int = 1024) -> dict[str, float]:
    """Sample-weighted mean joint geodesic error (deg) and fingertip error (mm) over a
    :class:`ImuPoseWindowDataset`, with the flat-hand (identity) prediction as reference."""
    import torch
    from torch.utils.data import DataLoader

    from ..pose.imu_model import IDENTITY_6D, hand_pose_loss

    dev = next(model.parameters()).device
    tot = {"rot": 0.0, "tip": 0.0, "rot_flat": 0.0, "tip_flat": 0.0}
    n = 0
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            for b in DataLoader(dataset, batch_size=batch_size):
                feat, gt = b["feat"].to(dev), b["finger_pose"].to(dev)
                res = hand_pose_loss(model(feat), gt, skeleton, 1.0)
                flat = torch.tensor(IDENTITY_6D, device=dev).expand(gt.shape[0], 15, 6)
                res0 = hand_pose_loss(flat, gt, skeleton, 1.0)
                k = gt.shape[0]
                tot["rot"] += float(res["rot"]) * k
                tot["tip"] += float(res["tip"]) * k
                tot["rot_flat"] += float(res0["rot"]) * k
                tot["tip_flat"] += float(res0["tip"]) * k
                n += k
    finally:
        model.train(was)
    n = max(n, 1)
    return {"rot_deg": float(np.degrees(tot["rot"] / n)), "tip_mm": 1e3 * tot["tip"] / n,
            "rot_deg_flat": float(np.degrees(tot["rot_flat"] / n)), "tip_mm_flat": 1e3 * tot["tip_flat"] / n,
            "n_frames": int(n)}


# ─────────────────────────────────────────────────────────────── run

def run(cfg: Mapping[str, Any] | None = None) -> dict:
    """Train the IMU → finger-pose net, write derived ``hand_finger_pose_imu``; returns metrics."""
    from ..datasets.episode import K_HAND_FINGERS, K_HAND_VALID
    from ..datasets.motion import ImuPoseWindowDataset, imu_wrist_index
    from ..datasets.stats import IMU_FEATURES, compute_stats
    from ..pose.imu_model import ImuHandPoseNet
    from ..pose.mano import ManoSkeleton

    cfg = resolve_config(cfg)
    out_dir = Path(cfg["out_dir"])
    d_cfg, f_cfg, m_cfg = cfg["data"], dict(cfg["features"]), dict(cfg["model"])
    window = int(m_cfg.pop("window"))
    train_sets, pred_sets = set(d_cfg["datasets"] or []), set(d_cfg["predict_datasets"] or [])

    eps, skipped = stage_episodes(d_cfg)
    with_imu = []
    for ep in eps:
        why = _has_imu(ep, f_cfg)
        if why:
            skipped.append({"episode": str(ep.root), "reason": why})
        else:
            with_imu.append(ep)
    pool = [e for e in with_imu if e.meta.dataset in train_sets and e.has(K_HAND_FINGERS) and e.has(K_HAND_VALID)
            and bool(np.asarray(e[K_HAND_VALID]).any())]
    split = split_stage_episodes(pool, d_cfg, stage=STAGE)
    if not split["train"]:
        raise ValueError(f"imu_pose: no training episodes with IMUs and hand labels in {sorted(train_sets)} under "
                         f"{d_cfg.get('processed_root')!r} ({len(eps)} loaded, {len(skipped)} skipped)")
    imu_kw = {"gyro": bool(f_cfg["gyro"]), "acc": bool(f_cfg["acc"]), "vec_frame": str(f_cfg["vec_frame"])}
    stats = compute_stats(split["train"], keys=(IMU_FEATURES,), method=cfg["stats"]["method"], imu_kw=imu_kw)
    fs = stats[IMU_FEATURES]
    seed_model_init(cfg["train"])                 # reproducible initial weights (train.seed)
    model = ImuHandPoseNet(int(fs.offset.shape[0]), **m_cfg)
    model.set_feature_stats(fs.offset, fs.scale)
    ds_kw = dict(window=window, stride=int(d_cfg["stride"]), imu_stats=None, **imu_kw)
    train_ds = ImuPoseWindowDataset(split["train"], **ds_kw)
    val_ds = None
    if split["val"]:
        try:
            val_ds = ImuPoseWindowDataset(split["val"], **ds_kw)
        except ValueError as e:                  # e.g. a val episode without labelled frames
            log.warning("imu_pose: no validation windows (%s)", e)
    skeleton = ManoSkeleton()
    sites = list(split["train"][0].meta.imu_sites)
    bundle_meta = {"stage": STAGE, "window": window, "features": imu_kw, "imu_sites": sites,
                   "wrist_index": imu_wrist_index(split["train"][0]), "stats": fs.to_dict()}
    trainer = fit_and_restore(model, functools.partial(_loss, skeleton=skeleton,
                                                       tip_weight=float(cfg["loss"]["tip_weight"])),
                              cfg["train"], train_ds, val_ds,
                              extra_state={"stage": STAGE, "model_config": model.config, "bundle_meta": bundle_meta})
    metrics: dict[str, Any] = {
        "stage": STAGE, "out_dir": str(out_dir), "model_path": str(out_dir / MODEL_NAME), "window": window,
        "n_episodes": {"loaded": len(eps), "train": len(split["train"]), "val": len(split["val"]),
                       "test": len(split["test"])},
        "n_samples": {"train": len(train_ds), "val": len(val_ds) if val_ds is not None else 0},
        "epochs": trainer.epoch, "steps": trainer.step,
        "best": {"monitor": trainer.cfg.monitor, "value": trainer.best_value, "epoch": trainer.best_epoch},
        "final_train_loss": trainer.history[-1].get("train/loss") if trainer.history else None,
    }
    if trainer.dist.is_main:
        metrics = _predict_evaluate_save(cfg, model, metrics, with_imu, split, skipped, stats, bundle_meta,
                                         window, imu_kw, sites, pred_sets, train_ds, val_ds, skeleton)
    from ..train import barrier

    barrier(trainer.dist)
    return finite_json(metrics)


def _predict_evaluate_save(cfg, model, metrics, with_imu, split, skipped, stats, bundle_meta, window, imu_kw,
                           sites, pred_sets, train_ds, val_ds, skeleton) -> dict:
    from ..datasets.episode import D_HAND_POSE_IMU, K_IMU_ACC, K_IMU_GYRO, K_IMU_QUAT
    from ..datasets.motion import ImuPoseWindowDataset, imu_wrist_index
    from ..datasets.stats import save_stats
    from ..pose.imu_model import predict_finger_pose_sequence

    out_dir = Path(cfg["out_dir"])
    p_cfg = cfg["predict"]
    groups = {"val": val_ds} if val_ds is not None else {"train_eval": train_ds}
    if split["test"]:
        try:
            groups["test"] = ImuPoseWindowDataset(split["test"], window=window, stride=1, imu_stats=None, **imu_kw)
        except ValueError as e:
            log.warning("imu_pose: no test windows (%s)", e)
    for name, ds in groups.items():
        metrics.update({f"{name}/{k}": v for k, v in evaluate_pose(model, ds, skeleton).items()})

    predicted = []
    for ep in with_imu:
        if ep.meta.dataset not in pred_sets:
            continue
        if list(ep.meta.imu_sites) != sites:
            skipped.append({"episode": str(ep.root), "reason": f"IMU sites {ep.meta.imu_sites} differ from {sites}"})
            continue
        # np.array copies: the episode arrays are read-only memmaps (torch warns on those)
        res = predict_finger_pose_sequence(
            model, np.array(ep[K_IMU_QUAT]), np.array(ep[K_IMU_GYRO]) if imu_kw["gyro"] else None,
            np.array(ep[K_IMU_ACC]) if imu_kw["acc"] else None, window=window, wrist_index=imu_wrist_index(ep),
            vec_frame=imu_kw["vec_frame"], batch_size=int(p_cfg["batch_size"]))
        if p_cfg["write_derived"]:
            ep.set_derived(D_HAND_POSE_IMU, res["finger_pose"].astype(np.float32))
            predicted.append(str(ep.root))
    metrics["n_episodes"]["predicted"] = len(predicted)
    metrics["n_episodes"]["skipped"] = len(skipped)
    metrics["skipped"] = skipped
    bundle_meta["metrics"] = {k: v for k, v in metrics.items() if "/" in k}
    bundle_meta["episodes"] = {k: [str(e.root) for e in v] for k, v in split.items()}
    save_imu_pose_model(out_dir / MODEL_NAME, model, bundle_meta)
    save_stats(stats, out_dir / STATS_NAME, meta={"stage": STAGE, "features": imu_kw, "imu_sites": sites})
    write_json_atomic(out_dir / METRICS_NAME, metrics)
    log.info("imu_pose done: %s", {k: v for k, v in metrics.items() if k.startswith(("val/", "train_eval/"))})
    return metrics


# ─────────────────────────────────────────────────────────────── CLI

def main(argv: Sequence[str] | None = None) -> int:
    """``python -m robot_skin.stages.imu_pose [--config yaml] [--set train.lr=1e-3 ...]``."""
    ap = argparse.ArgumentParser(prog="python -m robot_skin.stages.imu_pose",
                                 description="IMU → MANO finger pose training (ImuHandPoseNet).")
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
