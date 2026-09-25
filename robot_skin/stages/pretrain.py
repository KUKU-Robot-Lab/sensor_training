"""Stage ``pretrain``: MAE-style masked-taxel pretraining of the tactile :class:`TaxelEncoder`.

Pipeline position: after ``contact`` (which writes the derived ``residual_z`` / ``contact_level``
of every processed episode), before ``vtla`` (which may initialise — and optionally freeze — its
tactile branch from the encoder saved here). Self-supervised: uses every D1 motion + D2 task
episode with those derived arrays, no labels. Method: MAE (He et al., CVPR 2022,
arXiv:2111.06377); see :mod:`robot_skin.representation.pretrain`.

``run(cfg) -> metrics`` (config: ``robot_skin/configs/stages/pretrain.yaml``; missing keys fall
back to :data:`DEFAULTS`):

1. find episodes (``data.processed_root`` × ``data.datasets``, or an explicit ``data.episodes``
   list); episodes without derived ``residual_z``/``contact_level`` or taxel poses are skipped
   (listed in the metrics);
2. split **by episode** (or subject): ``data.splits`` = a ``splits.json`` (``datasets.splits``
   format ``{"train": [...], "val": [...], "test": [...]}``; entries are episode dirs or ids;
   test episodes are excluded unless ``data.use_test``), otherwise a seeded ``val_frac`` split;
3. :class:`TaxelPretrainDataset` (``features`` = :class:`TactileFeatureSpec`), level class
   weights (``balanced`` = inverse-sqrt frequency on the train split);
4. :class:`robot_skin.train.Trainer` (``train`` = :class:`TrainConfig`), best checkpoint by
   ``train.monitor`` (EMA weights when enabled);
5. writes ``<out_dir>/encoder_state.pt`` ({format, version, config (incl. feature spec),
   state_dict, meta}) — reload with
   :func:`robot_skin.representation.load_pretrained_encoder` — and ``<out_dir>/metrics.json``
   (masked reconstruction metrics on val with trivial baselines, split sizes, skipped
   episodes). ``out_dir`` = ``cfg.out_dir`` or ``train.out_dir``.

Hardware profiles (:func:`apply_hardware`): precedence is YAML ``train`` < profile
(``suggest.pretrain``, then ``train``) < explicit overrides. :func:`load_stage_config` applies
the profile between the YAML and the overrides and sets ``hardware_applied: true``; :func:`run`
applies ``cfg.hardware`` itself only when ``hardware_applied`` is not true (a raw dict) — then
the profile wins over the dict's ``train`` keys. The profile's ``env`` is exported
(``setdefault``) before any CUDA call.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ..config import deep_merge
from ..datasets.episode import (D_LEVEL, D_RESIDUAL_Z, K_TAXEL_NRM, K_TAXEL_POS, Episode,
                                list_episodes)
from ..representation.encoder import (ENCODER_STATE_NAME, TactileFeatureSpec, TaxelEncoder,
                                      save_pretrained_encoder)
from ..representation.pretrain import (MaskedTaxelPretrainer, TaxelPretrainDataset,
                                       collate_pretrain, evaluate_reconstruction,
                                       level_class_weights, pretrain_loss)

log = logging.getLogger("robot_skin.stages.pretrain")

STAGE = "pretrain"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "stages" / "pretrain.yaml"
METRICS_NAME = "metrics.json"

#: built-in defaults (mirrored by configs/stages/pretrain.yaml; a test keeps them in sync)
DEFAULTS: dict[str, Any] = {
    "stage": STAGE,
    "hardware": None,
    "hardware_applied": False,
    "out_dir": None,
    "data": {
        "processed_root": "robot_skin/data/processed",
        "datasets": ["motion", "task"],
        "episodes": None,
        "splits": None,
        "use_test": False,
        "split_by": "episode",
        "val_frac": 0.2,
        "split_seed": 0,
        "frame_stride": 4,
        "contact_repeat": 1,
        "use_groups": True,
        "max_group_frac": 0.5,
        "layout": None,
    },
    "features": {"obs_mode": "full", "history": 1, "stride": 1, "z_clip": 100.0, "z_scale": 2.0},
    "model": {"d_model": 64, "depth": 2, "heads": 4, "ff_mult": 4, "dropout": 0.0,
              "n_fourier": 8, "fourier_scale": 0.05, "n_taxels": None},
    "pretrain": {
        "mask_ratio": 0.3,
        "mask_mode": "mixed",
        "group_prob": 0.5,
        "decoder_dim": None,
        "decoder_depth": 1,
        "decoder_heads": None,
        "huber_delta": 1.0,
        "z_weight": 1.0,
        "level_weight": 1.0,
        "level_class_weights": "balanced",
        "class_weight_power": 0.5,
        "class_weight_max": 10.0,
        "eval_seed": 0,
    },
    "eval": {"batch_size": 512, "seed": 0},
    "train": {
        "max_epochs": 20,
        "batch_size": 256,
        "lr": 1.0e-3,
        "weight_decay": 0.05,
        "warmup_steps": 200,
        "schedule": "cosine",
        "min_lr_ratio": 0.05,
        "grad_clip": 1.0,
        "precision": "auto",
        "ema_decay": None,
        "monitor": "val/loss",
        "log_every": 50,
        "seed": 0,
        "out_dir": "robot_skin/runs/pretrain",
    },
}

_SPLITS = ("train", "val", "test")


# ─────────────────────────────────────────────────────────────── config

def apply_hardware(cfg: Mapping[str, Any]) -> dict:
    """Apply ``cfg["hardware"]`` (profile name / YAML path / mapping / ``"auto"``) once.

    Exports the profile's ``env`` (``setdefault``; before the first CUDA call), applies its
    ``train`` / ``suggest.pretrain`` keys (:func:`robot_skin.train.apply_hw_profile`) and sets
    ``hardware_applied``. No-op without ``hardware`` or when already applied. The profile *wins*
    over the ``train`` keys already in ``cfg`` — merge explicit overrides afterwards
    (:func:`load_stage_config` does).
    """
    out = copy.deepcopy(dict(cfg))
    hw = out.get("hardware")
    if not hw or out.get("hardware_applied"):
        return out
    from ..train.hardware import (apply_hw_profile, apply_profile_env, detect_hw_profile,
                                  load_hw_profile)

    if hw == "auto" and detect_hw_profile() is None:
        warnings.warn("hardware: auto — no built-in profile matches this machine; using the "
                      "stage defaults", stacklevel=2)
    else:
        prof = dict(hw) if isinstance(hw, Mapping) else load_hw_profile(hw)
        apply_profile_env(prof)
        out = apply_hw_profile(out, prof, stage=STAGE)
    out["hardware_applied"] = True
    return out


def load_stage_config(path: str | Path | None = None,
                      overrides: Mapping[str, Any] | None = None) -> dict:
    """:data:`DEFAULTS` ⊕ YAML (default ``configs/stages/pretrain.yaml``) ⊕ hardware profile ⊕
    ``overrides``.

    The profile (``hardware`` from the YAML or the overrides) is applied *before* the other
    overrides, so e.g. ``--set train.batch_size=32`` beats the profile's ``suggest.pretrain``;
    the result has ``hardware_applied: true`` and :func:`run` does not re-apply it.
    """
    p = Path(path) if path is not None else CONFIG_PATH
    if p.is_file():
        cfg = deep_merge(DEFAULTS, yaml.safe_load(p.read_text()) or {})
    elif path is not None:
        raise FileNotFoundError(f"stage config not found: {p}")
    else:
        cfg = copy.deepcopy(DEFAULTS)
    ov = dict(overrides or {})
    hw_keys = {k: ov.pop(k) for k in ("hardware", "hardware_applied") if k in ov}
    cfg = apply_hardware(deep_merge(cfg, hw_keys))
    return deep_merge(cfg, ov)


def resolve_config(cfg: Mapping[str, Any] | None) -> dict:
    """Merge ``cfg`` over :data:`DEFAULTS`, apply the hardware profile once, fix ``out_dir``."""
    out = apply_hardware(deep_merge(DEFAULTS, cfg or {}))
    out_dir = out.get("out_dir") or out["train"].get("out_dir") or DEFAULTS["train"]["out_dir"]
    out["out_dir"] = str(out_dir)
    out["train"]["out_dir"] = str(out_dir)
    return out


# ─────────────────────────────────────────────────────────────── episodes & splits

def discover_episodes(data_cfg: Mapping[str, Any]) -> list[Path]:
    """Episode directories from ``data.episodes`` or ``processed_root`` × ``datasets``."""
    root = Path(data_cfg.get("processed_root") or ".")
    explicit = data_cfg.get("episodes")
    if explicit:
        dirs = []
        for e in explicit:
            p = Path(e)
            if not p.is_dir() and not p.is_absolute():
                p = root / e
            if not p.is_dir():
                raise FileNotFoundError(f"episode directory not found: {e}")
            dirs.append(p)
        return sorted(set(dirs))
    datasets = data_cfg.get("datasets") or [None]
    if isinstance(datasets, str):
        datasets = [datasets]
    found: set[Path] = set()
    for ds in datasets:
        found.update(list_episodes(root, ds))
    return sorted(found)


def load_usable_episodes(dirs: Sequence[Path]) -> tuple[list[Episode], list[dict]]:
    """Load (mmap) and keep episodes with derived residual_z/contact_level + taxel poses."""
    usable, skipped = [], []
    for d in dirs:
        try:
            ep = Episode.load(d, mmap=True)
        except Exception as e:  # corrupt / foreign directory: report, never abort the stage
            skipped.append({"episode": str(d), "reason": f"load failed: {e}"})
            continue
        missing = [k for k in (D_RESIDUAL_Z, D_LEVEL) if not ep.has_derived(k)]
        missing += [k for k in (K_TAXEL_POS, K_TAXEL_NRM) if not ep.has(k)]
        if missing:
            skipped.append({"episode": str(d), "reason": f"missing {missing}"})
            continue
        usable.append(ep)
    return usable, skipped


def _match(entry: str, eps: Sequence[Episode], bases: Sequence[Path]) -> Episode | None:
    """Episode for a splits entry: a path (absolute, or relative to one of ``bases``), an
    episode id or an episode directory name."""
    cand = [Path(entry)]
    if not Path(entry).is_absolute():
        cand += [b / entry for b in bases]
    resolved = {c.resolve() for c in cand if c.exists()}
    for ep in eps:
        r = ep.root.resolve() if ep.root is not None else None
        if (r is not None and r in resolved) or ep.meta.episode_id == entry or (
                ep.root is not None and ep.root.name == entry):
            return ep
    return None


def split_episodes(eps: Sequence[Episode], data_cfg: Mapping[str, Any]) -> dict[str, list[Episode]]:
    """Leakage-safe split (whole episodes, or whole subjects with ``split_by: subject``)."""
    root = Path(data_cfg.get("processed_root") or ".")
    splits_path = data_cfg.get("splits")
    if splits_path:
        spec = json.loads(Path(splits_path).read_text())
        # relative entries: processed root (datasets.splits.save_splits(root=...)) or the
        # directory of splits.json (datasets.splits.load_splits default)
        bases = [root, Path(splits_path).parent]
        out: dict[str, list[Episode]] = {k: [] for k in _SPLITS}
        unmatched = 0
        for name in _SPLITS:
            for entry in spec.get(name, []) or []:
                ep = _match(str(entry), eps, bases)
                if ep is None:
                    unmatched += 1
                elif all(ep is not e for part in out.values() for e in part):
                    out[name].append(ep)
        listed = {id(e) for part in out.values() for e in part}
        n_unlisted = sum(id(e) not in listed for e in eps)
        if unmatched or n_unlisted:
            warnings.warn(f"splits {splits_path}: {unmatched} entries without a usable episode, "
                          f"{n_unlisted} usable episodes not listed (ignored)", stacklevel=2)
        if data_cfg.get("use_test"):
            out["train"] = out["train"] + out["test"]
            out["test"] = []
        return out
    by = data_cfg.get("split_by", "episode")
    if by not in ("episode", "subject"):
        raise ValueError(f"data.split_by must be episode|subject, got {by!r}")
    keys = sorted({(ep.meta.subject or "") if by == "subject" else ep.meta.episode_id for ep in eps})
    rng = np.random.default_rng(int(data_cfg.get("split_seed", 0)))
    order = [keys[i] for i in rng.permutation(len(keys))]
    val_frac = float(data_cfg.get("val_frac", 0.2))
    n_val = 0
    if len(keys) >= 2 and val_frac > 0:
        n_val = min(max(1, int(round(val_frac * len(keys)))), len(keys) - 1)
    val_keys = set(order[:n_val])
    key = (lambda e: e.meta.subject or "") if by == "subject" else (lambda e: e.meta.episode_id)
    return {"train": [e for e in eps if key(e) not in val_keys],
            "val": [e for e in eps if key(e) in val_keys], "test": []}


# ─────────────────────────────────────────────────────────────── run

def _class_weights(spec: Any, ds: TaxelPretrainDataset, p_cfg: Mapping[str, Any]) -> np.ndarray | None:
    if spec is None or spec is False or spec == "none":
        return None
    if spec == "balanced":
        return level_class_weights(ds.level_counts(), power=float(p_cfg.get("class_weight_power", 0.5)),
                                   max_weight=float(p_cfg.get("class_weight_max", 10.0)))
    return np.asarray(spec, dtype=np.float32)


def _finite(obj: Any) -> Any:
    """Recursively replace non-finite floats by ``None`` (strict JSON: no NaN / Infinity)."""
    if isinstance(obj, Mapping):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run(cfg: Mapping[str, Any] | None = None) -> dict:
    """Train the masked-taxel pretrainer; returns (and writes) the metrics dict."""
    from ..train import TrainConfig, Trainer, barrier
    from ..train.checkpoint import BEST_NAME

    cfg = resolve_config(cfg)
    out_dir = Path(cfg["out_dir"])
    d_cfg, p_cfg = cfg["data"], dict(cfg["pretrain"])
    spec = TactileFeatureSpec.from_dict(cfg["features"])
    if spec.dim < 1:
        raise ValueError("features.obs_mode 'none' has no tactile values to pretrain on")

    dirs = discover_episodes(d_cfg)
    eps, skipped = load_usable_episodes(dirs)
    for s in skipped:
        log.warning("pretrain: skipping %s (%s)", s["episode"], s["reason"])
    split = split_episodes(eps, d_cfg)
    if not split["train"]:
        raise ValueError(f"no usable training episodes ({len(dirs)} found under "
                         f"{d_cfg.get('processed_root')!r}, {len(skipped)} skipped: need derived "
                         "residual_z/contact_level from the contact stage)")
    ds_kw = dict(frame_stride=int(d_cfg.get("frame_stride", 1)),
                 use_groups=bool(d_cfg.get("use_groups", True)),
                 max_group_frac=float(d_cfg.get("max_group_frac", 0.5)), layout=d_cfg.get("layout"))
    train_ds = TaxelPretrainDataset(split["train"], spec,
                                    contact_repeat=int(d_cfg.get("contact_repeat", 1)), **ds_kw)
    if len(train_ds) == 0:
        raise ValueError(f"{len(split['train'])} training episodes but no usable frames "
                         "(non-finite taxel poses everywhere?)")
    val_ds = TaxelPretrainDataset(split["val"], spec, **ds_kw) if split["val"] else None
    if val_ds is not None and len(val_ds) == 0:
        val_ds = None

    weights = _class_weights(p_cfg.pop("level_class_weights", "balanced"), train_ds, p_cfg)
    p_cfg.pop("class_weight_power", None)
    p_cfg.pop("class_weight_max", None)
    encoder = TaxelEncoder(spec.dim, **cfg["model"], feature_spec=spec)
    model = MaskedTaxelPretrainer(encoder, spec.dim, level_class_weights=weights, **p_cfg)

    t_cfg = dict(cfg["train"])
    if val_ds is None and str(t_cfg.get("monitor", "val/loss")).startswith("val/"):
        t_cfg["monitor"] = "train/" + str(t_cfg.get("monitor", "val/loss"))[4:]
    tcfg = TrainConfig.from_dict(t_cfg)
    extra = {"stage": STAGE, "encoder_config": encoder.config, "pretrainer_config": model.config,
             "feature_spec": spec.to_dict()}
    trainer = Trainer(model, pretrain_loss, tcfg, train_ds, val_ds, collate_fn=collate_pretrain,
                      extra_state=extra)
    history = trainer.fit()
    barrier(trainer.dist)      # rank 0 has finished writing ckpt_best.pt

    metrics: dict[str, Any] = {
        "stage": STAGE,
        "out_dir": str(out_dir),
        "encoder_path": str(out_dir / ENCODER_STATE_NAME),
        "n_episodes": {"found": len(dirs), "train": len(split["train"]), "val": len(split["val"]),
                       "skipped": len(skipped)},
        "n_samples": {"train": len(train_ds), "val": len(val_ds) if val_ds is not None else 0},
        "skipped": skipped,
        "feature_spec": spec.to_dict(),
        "level_class_weights": None if weights is None else [float(w) for w in weights],
        "epochs": trainer.epoch,
        "steps": trainer.step,
        "best": {"monitor": tcfg.monitor, "value": trainer.best_value, "epoch": trainer.best_epoch},
        "final_train_loss": history[-1].get("train/loss") if history else None,
    }
    if trainer.dist.is_main:
        # export the best checkpoint's weights (EMA when enabled), not the last step's
        best = out_dir / BEST_NAME
        if best.is_file():
            Trainer.load_model_weights(model, best, use_ema=True)
        elif trainer.ema is not None:
            trainer.ema.apply_to(model)
        e_cfg = cfg.get("eval") or {}
        eval_ds, prefix = (val_ds, "val/") if val_ds is not None else (train_ds, "train_eval/")
        res = evaluate_reconstruction(model, eval_ds, batch_size=int(e_cfg.get("batch_size", 512)),
                                      device=trainer.device, seed=int(e_cfg.get("seed", 0)))
        metrics.update({f"{prefix}{k}": v for k, v in res.items()})
        metrics = _finite(metrics)
        save_pretrained_encoder(out_dir / ENCODER_STATE_NAME, model.encoder, meta={
            "stage": STAGE, "pretrainer_config": model.config, "feature_spec": spec.to_dict(),
            "episodes": {k: [str(e.root) for e in v] for k, v in split.items()},
            "best": metrics["best"], "metrics": {k: v for k, v in metrics.items()
                                                 if k.startswith(prefix)}})
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f".{METRICS_NAME}.tmp"
        tmp.write_text(json.dumps(metrics, indent=2, default=str, allow_nan=False))
        os.replace(tmp, out_dir / METRICS_NAME)
        log.info("pretrain done: %s", {k: v for k, v in metrics.items() if k.startswith(prefix)})
    barrier(trainer.dist)
    return _finite(metrics)


# ─────────────────────────────────────────────────────────────── CLI

def _parse_set(items: Sequence[str]) -> dict:
    from ..train.sweep import set_by_path

    out: dict = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        set_by_path(out, k.strip(), yaml.safe_load(v))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m robot_skin.stages.pretrain [--config yaml] [--set train.lr=1e-3 ...]``."""
    ap = argparse.ArgumentParser(prog="python -m robot_skin.stages.pretrain",
                                 description="Masked-taxel (MAE-style) encoder pretraining.")
    ap.add_argument("--config", default=None, help=f"stage YAML (default {CONFIG_PATH})")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted override, value parsed as YAML (repeatable)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_stage_config(args.config, _parse_set(args.set))
    metrics = run(cfg)
    json.dump({k: v for k, v in metrics.items() if k != "skipped"}, sys.stdout, indent=2,
              default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
