"""Stage ``contact``: residual calibration → z / levels, learned detector, D2 pseudo labels.

Pipeline position: after ``baseline`` (needs ``derived/residual`` and ``baseline_logvar`` in every
episode), before ``pretrain`` / ``vtla`` (which read the ``residual_z`` / ``contact_level`` written
here). Writes, for every episode of ``data.predict_datasets``::

    derived/residual_z            [T,N] float32  calibrated press-positive z (contact.calibration)
    derived/contact_level         [T,N] int8     ContactLevel NONE/WEAK/STRONG/SATURATED
    derived/contact_prob          [T,N] float32  detector probability (causal)
    derived/contact_label_pseudo  [T,N] int8     D2 pseudo labels (−1/0/1; ``pseudo_label.datasets``)

``residual_z`` / ``contact_level`` are also written for the training (``data.datasets``) episodes —
the detector trains on them. The episode ``contact_label`` *array* (preprocessing output) is never
modified; pseudo labels are a derived array (:data:`robot_skin.contact.D_CONTACT_LABEL_PSEUDO`).

``run(cfg) -> metrics`` (config ``robot_skin/configs/stages/contact.yaml``; unknown keys raise):

1. episodes and the D1 split exactly as the other stages (``data.splits`` / ``val_frac``);
2. :class:`~robot_skin.contact.calibration.ResidualCalibrator` fitted on the **validation** split's
   no-contact frames (``calibration.split``; the baseline did not train on them — falls back to
   train with a note), with the baseline's predicted variance (``use_logvar``), thresholds in z
   with % floors, the optional :class:`~robot_skin.contact.SaturationFSM` gate (``fsm``; when
   enabled the fit uses the FSM-corrected residual of trusted samples, exactly what the levels see);
3. :class:`~robot_skin.contact.detector.ContactDetector` on D1 labels (``contact_label``:
   self-touch 1, no-contact 0) from :class:`~robot_skin.datasets.motion.ContactWindowDataset`
   windows of ``residual_z`` + saturation + joint speed, focal loss (Lin et al., ICCV 2017);
   ``detector.enabled: false`` uses ``sigmoid(z − weak_z)`` as the probability instead;
   ``detector.bootstrap`` (``auto``: when the train split has no self-touch label, e.g. robot D1
   without geometric self-touch) adds positives where the calibrated level is STRONG inside
   contact-allowed phases (pinch / grasp …) — self-training from the z rule;
4. metrics on the evaluation split (``val/`` or ``train_eval/``; plus ``test/``). ``val/`` is also the
   calibration split and the detector's early-stopping set, so its numbers are optimistic — pass a
   ``data.splits`` file with a test split for unbiased ``test/`` metrics. Reported: hallucination
   rate on labelled no-contact samples (taxel and frame level) and recall / precision / F1 /
   AUROC on self-touch, for the z-level rule (``level ≥ WEAK``), the detector at
   ``eval.prob_threshold`` and the hysteresis-filtered detector; with synthetic ground truth
   (``gt_contact``) also AUROC against it; pseudo-label counts and (synthetic) agreement;
5. ``<out_dir>/calibrator.json`` (everything the online processor needs for z / levels / FSM),
   ``contact_detector.pt`` (:func:`robot_skin.contact.load_detector`; meta: window, q_source,
   hysteresis), ``metrics.json``.
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
               seed_model_init, split_stage_episodes, stage_episodes, warn_legacy_taxel_frame,
               write_json_atomic)

log = logging.getLogger("robot_skin.stages.contact")

STAGE = "contact"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "stages" / "contact.yaml"
METRICS_NAME = "metrics.json"
CALIBRATOR_NAME = "calibrator.json"
#: in-memory label array of bootstrapped detector training views (never written to disk)
LABEL_TRAIN = "contact_label_train"

#: built-in defaults (mirrored by configs/stages/contact.yaml; a test keeps them in sync)
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
        "stride": 1,
        "q_source": "q",
    },
    "calibration": {"split": "val", "labels": [0], "use_logvar": True, "center": True, "weak_z": 3.0,
                    "strong_z": 8.0, "weak_floor_pct": 0.5, "strong_floor_pct": 3.0, "sigma_floor_pct": 0.02,
                    "min_samples": 50},
    "fsm": {"enabled": False, "ok_pct": 1.5, "ok_sec": 2.0, "max_recover_s": 30.0},
    "detector": {"enabled": True, "bootstrap": "auto", "window": 16, "hidden": 32, "n_layers": 3, "kernel": 3,
                 "taxel_emb_dim": 4, "use_motion": True, "use_sat": True, "z_scale": 2.0, "z_clip": 100.0,
                 "prior": 0.01, "dropout": 0.0},
    "loss": {"kind": "focal", "gamma": 2.0, "alpha": 0.25, "pos_weight": None},
    "eval": {"prob_threshold": 0.5},
    "hysteresis": {"on_thr": 0.6, "off_thr": 0.4, "min_on": 2, "min_off": 4},
    "pseudo_label": {"enabled": True, "datasets": ["task"], "neg_thr": 0.1, "keep_existing": True,
                     "none_conflict": "unknown", "unknown_phase": "unknown", "saturated_as_contact": True,
                     "proximity_m": None, "phase_overrides": {}},
    "predict": {"write_derived": True, "batch_size": 4096},
    "train": {
        "max_epochs": 20,
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
        "out_dir": "robot_skin/runs/contact",
    },
}


def load_stage_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict:
    """:data:`DEFAULTS` ⊕ YAML (default ``configs/stages/contact.yaml``) ⊕ hardware ⊕ overrides."""
    return load_stage_yaml(DEFAULTS, STAGE, CONFIG_PATH, path, overrides)


def resolve_config(cfg: Mapping[str, Any] | None) -> dict:
    return resolve_stage_config(cfg, DEFAULTS, STAGE)


# ─────────────────────────────────────────────────────────────── helpers

def _usable(ep, use_logvar: bool, q_source: str = "q", need_qd: bool = True) -> str | None:
    """Reason the episode cannot be used (None if fine). ``qd`` comes from the episode arrays
    (``q_source: q``, needed by a motion-aware detector) or from the IMU pose model's derived finger
    pose (``hand_pose_imu``: camera-free gloves have no vision ``q``/``qd`` arrays)."""
    from ..datasets.episode import (D_BASELINE_LOGVAR, D_HAND_POSE_IMU, D_RESIDUAL, K_CONTACT_LABEL, K_QD,
                                    K_SATURATED)

    missing = [k for k in (K_CONTACT_LABEL, K_SATURATED) if not ep.has(k)]
    if q_source == "q" and need_qd and not ep.has(K_QD):
        missing.append(K_QD)
    if q_source == "hand_pose_imu" and not ep.has_derived(D_HAND_POSE_IMU):
        missing.append(f"derived/{D_HAND_POSE_IMU}")
    missing += [f"derived/{k}" for k in ((D_RESIDUAL, D_BASELINE_LOGVAR) if use_logvar else (D_RESIDUAL,))
                if not ep.has_derived(k)]
    return f"missing {missing} (run the imu_pose / baseline stages first)" if missing else None


def _loss(model, batch, *, loss_cfg: Mapping[str, Any]):
    from ..contact.detector import contact_loss

    logits = model(batch["z_hist"], batch["sat_hist"], batch["qd"], batch["q_valid"])
    loss = contact_loss(logits, batch["label"], batch["label_mask"], **loss_cfg)
    with_lab = batch["label_mask"].float()
    acc = (((logits > 0).float() == batch["label"]).float() * with_lab).sum() / with_lab.sum().clamp_min(1.0)
    return {"loss": loss, "acc": acc.detach()}


def _auroc(score: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    from ..eval.metrics import auroc

    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return auroc(score[pos], score[neg])


def detection_metrics(pred: np.ndarray, score: np.ndarray | None, label: np.ndarray, ignore: np.ndarray,
                      gt: np.ndarray | None = None) -> dict[str, float]:
    """Contact detection quality of a boolean prediction ``[T,N]`` (+ a score for AUROC) against
    ``label`` (−1 ignored; 0 no-contact, 1 contact) on non-``ignore`` samples:
    ``hallucination_taxel`` (fraction of labelled no-contact samples predicted contact),
    ``hallucination_frame`` (``eval.hallucination_rate`` over frames whose labelled taxels are all
    no-contact), recall / precision / F1 on label 1, AUROC of the score; with ``gt`` (synthetic
    contact mask) ``gt_auroc`` / ``gt_hallucination`` over all non-ignored samples."""
    from ..eval.metrics import hallucination_rate

    keep = ~np.asarray(ignore, dtype=bool)
    lab = np.asarray(label)
    p = np.asarray(pred, dtype=bool)
    pos, neg = (lab == 1) & keep, (lab == 0) & keep
    tp, fp = float((p & pos).sum()), float((p & neg).sum())
    out: dict[str, float] = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
    out["hallucination_taxel"] = fp / neg.sum() if neg.sum() else float("nan")
    frames = neg.any(1) & ~pos.any(1) & ~((lab == -1) & keep).any(1)
    out["hallucination_frame"] = hallucination_rate(p[frames] & neg[frames], np.zeros_like(neg[frames])) \
        if frames.any() else float("nan")
    out["recall"] = tp / pos.sum() if pos.sum() else float("nan")
    out["precision"] = tp / (tp + fp) if tp + fp else float("nan")
    r, pr = out["recall"], out["precision"]
    out["f1"] = 2 * r * pr / (r + pr) if (r == r and pr == pr and r + pr > 0) else float("nan")
    if score is not None:
        out["auroc"] = _auroc(np.asarray(score, dtype=np.float64), pos, neg)
    if gt is not None:
        g = np.asarray(gt, dtype=bool)
        if score is not None:
            out["gt_auroc"] = _auroc(np.asarray(score, dtype=np.float64), g & keep, ~g & keep)
        out["gt_hallucination"] = float((p & ~g & keep).sum() / max((~g & keep).sum(), 1))
        out["gt_recall"] = float((p & g & keep).sum() / max((g & keep).sum(), 1)) if (g & keep).any() else float("nan")
    return out


# ─────────────────────────────────────────────────────────────── run

def run(cfg: Mapping[str, Any] | None = None) -> dict:
    """Calibrate residuals, train the detector, write z / levels / probabilities / pseudo labels."""
    from ..baseline.temporal import Q_SOURCES, episode_joint_view
    from ..contact.calibration import ResidualCalibrator, saturation_gate
    from ..contact.detector import ContactDetector
    from ..datasets.episode import (D_BASELINE_LOGVAR, D_RESIDUAL, D_RESIDUAL_Z, K_CONTACT_LABEL, K_QD,
                                    K_SATURATED)
    from ..datasets.motion import ContactWindowDataset
    from ..datasets.stats import compute_stats, qd_valid_mask

    cfg = resolve_config(cfg)
    out_dir = Path(cfg["out_dir"])
    d_cfg, c_cfg, det_cfg = cfg["data"], cfg["calibration"], dict(cfg["detector"])
    q_source = d_cfg["q_source"]
    if q_source not in Q_SOURCES:
        raise ValueError(f"data.q_source must be one of {Q_SOURCES}, got {q_source!r}")
    use_logvar = bool(c_cfg["use_logvar"])
    train_sets, pred_sets = set(d_cfg["datasets"] or []), set(d_cfg["predict_datasets"] or [])

    need_qd = bool(det_cfg["enabled"]) and bool(det_cfg["use_motion"])
    eps, skipped = stage_episodes(d_cfg)
    warn_legacy_taxel_frame(eps, STAGE)
    usable = []
    for ep in eps:
        why = _usable(ep, use_logvar, q_source, need_qd)
        if not why and d_cfg["kind"] and ep.meta.kind != d_cfg["kind"]:
            why = f"kind {ep.meta.kind!r} (data.kind = {d_cfg['kind']!r})"
        if why:
            skipped.append({"episode": str(ep.root), "reason": why})
        else:
            usable.append(ep)
    pool = [e for e in usable if e.meta.dataset in train_sets]
    if len({(e.meta.kind, e.meta.n_taxels) for e in pool}) > 1:
        raise ValueError("contact: training episodes of different kinds / taxel counts "
                         f"{sorted({(e.meta.kind, e.meta.n_taxels) for e in pool})} — one calibrator / detector "
                         "per layout: set data.kind (glove | robot) or select episodes with data.episodes")
    split = split_stage_episodes(pool, d_cfg, stage=STAGE)
    if not split["train"]:
        raise ValueError(f"contact: no usable D1 episodes in {sorted(train_sets)} under "
                         f"{d_cfg.get('processed_root')!r} ({len(eps)} loaded, {len(skipped)} skipped)")

    # joint-state views (q_source) — created once; they read derived arrays lazily from disk
    views = {id(e): episode_joint_view(e, q_source)
             for e in usable if e.meta.dataset in pred_sets or e.meta.dataset in train_sets}

    # ── 1. residual calibration on held-out no-contact frames ───────────────
    cal_split = c_cfg["split"]
    cal_eps = split.get(cal_split) or []
    notes = []
    if not cal_eps:
        notes.append(f"calibration split {cal_split!r} is empty: calibrated on the train split")
        cal_split, cal_eps = "train", split["train"]
    labels = [int(v) for v in c_cfg["labels"]]
    fsm_cfg = dict(cfg["fsm"])
    res, lvs, valid = [], [], []
    for ep in cal_eps:
        r = np.asarray(ep.derived(D_RESIDUAL), dtype=np.float32)
        sat = np.asarray(ep[K_SATURATED], dtype=bool)
        m = np.isin(np.asarray(ep[K_CONTACT_LABEL]), labels) & ~sat
        if fsm_cfg.get("enabled"):
            # fit on what the levels will see: the FSM-corrected residual, trusted samples only
            # (a taxel recovering from a dropout is untrusted → SATURATED, not calibration data)
            r, untrusted, _ = saturation_gate(r, sat, 1.0 / float(ep.meta.hz), **fsm_cfg)
            m &= ~untrusted
        m &= qd_valid_mask(views[id(ep)])[:, None]
        res.append(r)
        valid.append(m)
        if use_logvar:
            lvs.append(np.asarray(ep.derived(D_BASELINE_LOGVAR), dtype=np.float32))
    cal = ResidualCalibrator.fit(
        np.concatenate(res), np.concatenate(valid), np.concatenate(lvs) if use_logvar else None,
        weak_z=float(c_cfg["weak_z"]), strong_z=float(c_cfg["strong_z"]),
        weak_floor_pct=float(c_cfg["weak_floor_pct"]), strong_floor_pct=float(c_cfg["strong_floor_pct"]),
        sigma_floor_pct=float(c_cfg["sigma_floor_pct"]), center=bool(c_cfg["center"]),
        min_samples=int(c_cfg["min_samples"]), fsm=fsm_cfg)
    cal.info.update(split=cal_split, episodes=[str(e.root) for e in cal_eps], q_source=q_source)

    # ── 2. z / levels for every episode (rank 0 writes; all ranks need them for the detector) ──
    from ..train.distributed import barrier, init_distributed

    dist = init_distributed()
    targets = [e for e in usable if e.meta.dataset in pred_sets or e.meta.dataset in train_sets]
    for ep in targets:
        if ep.meta.n_taxels != cal.n_taxels:
            skipped.append({"episode": str(ep.root),
                            "reason": f"{ep.meta.n_taxels} taxels, calibrator {cal.n_taxels}"})
            continue
        # the detector needs residual_z of the D1 episodes: with predict.write_derived false they
        # are kept in memory only (on this episode object and its joint-state view)
        save = bool(cfg["predict"]["write_derived"]) and dist.is_main
        if dist.is_main or not cfg["predict"]["write_derived"]:
            _write_levels(ep, cal, save=save, view=views.get(id(ep)))
    barrier(dist)
    targets = [e for e in targets if e.meta.n_taxels == cal.n_taxels]

    # ── 3. detector ─────────────────────────────────────────────────────────
    det = None
    det_enabled = bool(det_cfg.pop("enabled"))
    metrics: dict[str, Any] = {"stage": STAGE, "out_dir": str(out_dir), "q_source": q_source,
                               "calibrator_path": str(out_dir / CALIBRATOR_NAME), "calibration_split": cal_split,
                               "notes": notes}
    if det_enabled:
        boot = det_cfg.pop("bootstrap")
        tr_v = [views[id(e)] for e in split["train"]]
        va_v = [views[id(e)] for e in split["val"]]
        n_pos = sum(int((np.asarray(v[K_CONTACT_LABEL]) == 1).sum()) for v in tr_v)
        if boot not in ("auto", True, False):
            raise ValueError(f"detector.bootstrap must be auto | true | false, got {boot!r}")
        if boot is True or (boot == "auto" and n_pos == 0):
            tr_v, n_boot = _bootstrap_views(tr_v)
            va_v, _ = _bootstrap_views(va_v)
            notes.append(f"detector labels bootstrapped: {n_boot} STRONG-level samples in contact-allowed phases "
                         f"labelled 1 (train split had {n_pos} self-touch labels)")
            metrics_boot = {"detector_bootstrap": True, "n_bootstrap_positives": n_boot}
        else:
            metrics_boot = {"detector_bootstrap": False}
        label_key = LABEL_TRAIN if metrics_boot["detector_bootstrap"] else K_CONTACT_LABEL
        N = tr_v[0].meta.n_taxels
        D = int(tr_v[0][K_QD].shape[1])
        seed_model_init(cfg["train"])             # reproducible initial weights (train.seed)
        det = ContactDetector(N, joint_dim=D, **det_cfg)
        if det.use_motion:
            det.set_joint_stats(compute_stats(tr_v, keys=(K_QD,))[K_QD])
        ds_kw = dict(window=det.window, stride=int(d_cfg["stride"]), joint_stats=None, z_key=D_RESIDUAL_Z,
                     label_key=label_key)
        train_ds = ContactWindowDataset(tr_v, **ds_kw)
        val_ds = None
        if va_v:
            try:
                val_ds = ContactWindowDataset(va_v, **ds_kw)
            except ValueError as e:
                log.warning("contact: no validation windows (%s)", e)
        extra = {"stage": STAGE, "model_config": det.config, "calibrator": cal.to_dict()}
        trainer = fit_and_restore(det, functools.partial(_loss, loss_cfg=dict(cfg["loss"])), cfg["train"],
                                  train_ds, val_ds, extra_state=extra)
        metrics.update({
            "detector_path": str(out_dir / "contact_detector.pt"),
            "n_samples": {"train": len(train_ds), "val": len(val_ds) if val_ds is not None else 0},
            "epochs": trainer.epoch, "steps": trainer.step,
            "best": {"monitor": trainer.cfg.monitor, "value": trainer.best_value, "epoch": trainer.best_epoch},
            "final_train_loss": trainer.history[-1].get("train/loss") if trainer.history else None,
            **metrics_boot})
    metrics["n_episodes"] = {"loaded": len(eps), "train": len(split["train"]), "val": len(split["val"]),
                             "test": len(split["test"]), "calibration": len(cal_eps)}
    if dist.is_main:
        metrics = _predict_evaluate_save(cfg, cal, det, metrics, targets, views, split, skipped, pred_sets,
                                         q_source)
    barrier(dist)
    return finite_json(metrics)


def _bootstrap_views(views: Sequence) -> tuple[list, int]:
    """Detector training labels for episodes without (enough) self-touch labels (e.g. robot D1:
    no geometric self-touch): in-memory views with an extra ``contact_label_train`` array = the
    episode ``contact_label`` plus 1 where the calibrated level is STRONG inside a contact-allowed
    phase (``pseudo_label.frame_expectation`` self / object / any) and the label is unknown.
    SATURATED levels are not used: they include FSM-untrusted (recovering) samples, which the
    dataset's saturation mask does not hide. Returns ``(views, n_added)``."""
    from ..contact.ordinal import ContactLevel
    from ..contact.pseudo_label import frame_expectation
    from ..datasets.episode import D_LEVEL, D_RESIDUAL_Z, K_CONTACT_LABEL, Episode

    out, n = [], 0
    for v in views:
        lab = np.array(v[K_CONTACT_LABEL], dtype=np.int8)
        allowed = np.isin(frame_expectation(v), ("self", "object", "any"))[:, None]
        add = (lab == -1) & allowed & (np.asarray(v.derived(D_LEVEL)) == ContactLevel.STRONG)
        lab[add] = 1
        n += int(add.sum())
        nv = Episode(v.meta, {**v.arrays, LABEL_TRAIN: lab}, v.static, root=v.root)
        for k in (D_RESIDUAL_Z, D_LEVEL):                   # keep in-memory derived arrays (write_derived false)
            nv.set_derived(k, np.asarray(v.derived(k)), save=False)
        out.append(nv)
    return out, n


def _write_levels(ep, cal, *, save: bool = True, view=None) -> None:
    """residual_z / contact_level of one episode (:func:`robot_skin.contact.residual_levels`)."""
    from ..contact.calibration import residual_levels
    from ..datasets.episode import D_BASELINE_LOGVAR, D_LEVEL, D_RESIDUAL, D_RESIDUAL_Z, K_SATURATED

    out = residual_levels(cal, np.asarray(ep.derived(D_RESIDUAL), dtype=np.float32),
                          np.asarray(ep[K_SATURATED], dtype=bool),
                          np.asarray(ep.derived(D_BASELINE_LOGVAR), dtype=np.float32) if cal.use_logvar else None,
                          dt=1.0 / float(ep.meta.hz))
    z, lv = out["residual_z"].astype(np.float32), out["contact_level"].astype(np.int8)
    ep.set_derived(D_RESIDUAL_Z, z, save=save)
    ep.set_derived(D_LEVEL, lv, save=save)
    if view is not None and view is not ep:
        view.set_derived(D_RESIDUAL_Z, z, save=False)
        view.set_derived(D_LEVEL, lv, save=False)


def _probability(det, view, cal, batch_size: int) -> np.ndarray:
    """Detector probability, or ``sigmoid(z − weak_z)`` without a detector."""
    from ..contact.detector import predict_contact_prob
    from ..datasets.episode import D_RESIDUAL_Z

    if det is not None:
        return predict_contact_prob(det, view, batch_size=batch_size)
    z = np.nan_to_num(np.asarray(view.derived(D_RESIDUAL_Z), dtype=np.float64), nan=0.0)
    return (1.0 / (1.0 + np.exp(-np.clip(z - cal.weak_z, -50, 50)))).astype(np.float32)


def _predict_evaluate_save(cfg, cal, det, metrics, targets, views, split, skipped, pred_sets, q_source) -> dict:
    from ..contact.detector import save_detector
    from ..contact.hysteresis import HysteresisFilter
    from ..contact.ordinal import ContactLevel
    from ..contact.pseudo_label import D_CONTACT_LABEL_PSEUDO, pseudo_label_episode, pseudo_label_metrics
    from ..datasets.episode import D_CONTACT_PROB, D_LEVEL, D_RESIDUAL_Z, K_CONTACT_LABEL, K_SATURATED

    out_dir = Path(cfg["out_dir"])
    p_cfg, pl_cfg = cfg["predict"], dict(cfg["pseudo_label"])
    thr = float(cfg["eval"]["prob_threshold"])
    hcfg = dict(cfg["hysteresis"])
    probs: dict[int, np.ndarray] = {}
    for ep in targets:
        pr = _probability(det, views[id(ep)], cal, int(p_cfg["batch_size"]))
        probs[id(ep)] = pr
        if ep.meta.dataset in pred_sets and p_cfg["write_derived"]:
            ep.set_derived(D_CONTACT_PROB, pr)
    metrics["contact_prob_source"] = "detector" if det is not None else "z_logistic"

    # ── evaluation on labelled D1 frames ─────────────────────────────────────
    groups = {"val": split["val"]} if split["val"] else {"train_eval": split["train"]}
    if split["test"]:
        groups["test"] = split["test"]
    for name, g in groups.items():
        g = [e for e in g if id(e) in probs]
        if not g:
            continue
        cat = {k: [] for k in ("lab", "ign", "z", "lv", "p", "h", "gt")}
        for ep in g:
            lv = np.asarray(ep.derived(D_LEVEL))
            cat["lab"].append(np.asarray(ep[K_CONTACT_LABEL]))
            cat["ign"].append(np.asarray(ep[K_SATURATED], dtype=bool) | (lv == ContactLevel.SATURATED))
            cat["z"].append(np.nan_to_num(np.asarray(ep.derived(D_RESIDUAL_Z), dtype=np.float64), nan=0.0))
            cat["lv"].append(lv)
            cat["p"].append(probs[id(ep)])
            cat["h"].append(HysteresisFilter.from_config(hcfg).run(probs[id(ep)]))
            cat["gt"].append(np.asarray(ep["gt_contact"], dtype=bool) if ep.has("gt_contact") else None)
        c = {k: (np.concatenate(v) if all(x is not None for x in v) else None) for k, v in cat.items()}
        rules = {"z": (c["lv"] >= ContactLevel.WEAK, c["z"]), "prob": (c["p"] >= thr, c["p"]),
                 "hyst": (c["h"], c["p"])}
        for rule, (pred, score) in rules.items():
            if det is None and rule != "z":
                continue
            res = detection_metrics(pred, score, c["lab"], c["ign"], c["gt"])
            metrics.update({f"{name}/{rule}_{k}": v for k, v in res.items()})

    # ── pseudo labels for D2 ─────────────────────────────────────────────────
    if pl_cfg.pop("enabled"):
        pl_sets = set(pl_cfg.pop("datasets") or [])
        prox = pl_cfg.pop("proximity_m")
        counts: dict[str, int] = {}
        labs, gts, ign = [], [], []
        n_eps = 0
        for ep in targets:
            if ep.meta.dataset not in pl_sets:
                continue
            res = pseudo_label_episode(ep, probs[id(ep)], hysteresis=hcfg,
                                       proximity=None if prox is None else {"max_dist_m": float(prox)}, **pl_cfg)
            ep.set_derived(D_CONTACT_LABEL_PSEUDO, res["label"], save=bool(p_cfg["write_derived"]))
            n_eps += 1
            for k, v in res["counts"].items():
                counts[k] = counts.get(k, 0) + int(v)
            if ep.has("gt_contact"):
                labs.append(res["label"])
                gts.append(np.asarray(ep["gt_contact"], dtype=bool))
                ign.append(np.asarray(ep[K_SATURATED], dtype=bool))
        metrics["pseudo/n_episodes"] = n_eps
        metrics.update({f"pseudo/{k}": v for k, v in counts.items()})
        if labs:
            pm = pseudo_label_metrics(np.concatenate(labs), np.concatenate(gts), ignore=np.concatenate(ign))
            metrics.update({f"pseudo/gt_{k}": v for k, v in pm.items()})

    metrics["n_episodes"]["predicted"] = (sum(1 for e in targets if e.meta.dataset in pred_sets)
                                          if p_cfg["write_derived"] else 0)      # written, as in baseline
    metrics["n_episodes"]["skipped"] = len(skipped)
    metrics["skipped"] = skipped
    metrics["calibrator"] = {"sigma_pct": [float(x) for x in cal.sigma], "gain": [float(x) for x in cal.gain],
                             "center_pct": [float(x) for x in cal.center], "use_logvar": cal.use_logvar}
    cal.save(out_dir / CALIBRATOR_NAME)
    if det is not None:
        save_detector(out_dir / "contact_detector.pt", det, {
            "stage": STAGE, "window": det.window, "q_source": q_source, "hysteresis": hcfg,
            "prob_threshold": thr, "calibrator": cal.to_dict(),
            "metrics": {k: v for k, v in metrics.items() if "/" in k}})
    write_json_atomic(out_dir / METRICS_NAME, metrics)
    log.info("contact done: %s", {k: v for k, v in metrics.items() if k.startswith(("val/", "train_eval/"))})
    return metrics


# ─────────────────────────────────────────────────────────────── CLI

def main(argv: Sequence[str] | None = None) -> int:
    """``python -m robot_skin.stages.contact [--config yaml] [--set detector.window=32 ...]``."""
    ap = argparse.ArgumentParser(prog="python -m robot_skin.stages.contact",
                                 description="Residual calibration, contact detector and D2 pseudo labels.")
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
