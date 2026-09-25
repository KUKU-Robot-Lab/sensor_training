"""Stage ``vtla``: train the vision–tactile–language–action policy on D2 task episodes.

Pipeline position: last training stage — after ``contact`` (derived ``residual_z`` /
``contact_level`` of every episode) and optionally ``pretrain`` (a pretrained tactile encoder,
``tactile.pretrained``). Its output ``<out_dir>/policy_bundle.pt`` is what ``control`` deploys.
Model: :class:`robot_skin.vtla.VTLAPolicy` (ACT chunk regression or flow-matching head, see
``vtla/README.md``); data: :class:`robot_skin.vtla.VTLADataset`.

``run(cfg) -> metrics`` (config ``robot_skin/configs/stages/vtla.yaml``; missing keys fall back to
:data:`DEFAULTS`):

1. find episodes (``data.processed_root`` × ``data.datasets`` or ``data.episodes``); episodes that
   cannot provide the configured actions / cameras / tactile inputs are skipped (listed);
2. split: ``data.splits`` (a ``datasets.splits`` ``splits.json``: dirs or ids) or
   :func:`robot_skin.datasets.splits.make_splits` grouped by ``data.split_by`` (leakage-safe);
3. datasets at ``policy.policy_hz`` ticks inside ``data.phases``; the action normalizer (on the
   relative chunks) and the proprio normalizer are fit on the **train** split only;
4. model (tactile branch optionally initialised from ``tactile.pretrained`` = the pretrain stage's
   ``encoder_state.pt``, optionally frozen); image transforms from ``image`` (train augmentation /
   matching eval transform); optional frozen-vision feature cache (``vision.cache_features``: the
   cache key includes a hash of the encoder weights, so a re-initialised encoder never reuses a
   stale cache — note that cached training sees the *eval* transform, no augmentation);
5. :class:`robot_skin.train.Trainer` (``train`` = TrainConfig; best checkpoint by
   ``train.monitor``, EMA weights when enabled);
6. evaluation of the best weights (:func:`evaluate_policy`: masked L1 of the normalized chunk per
   horizon step, by task, and per action group in raw units) on ``eval.splits``;
7. writes ``<out_dir>/policy_bundle.pt`` (:func:`robot_skin.vtla.save_policy_bundle`: model config +
   weights, action spec + normalizer + relative mode, proprio normalizer, tactile feature spec and
   stage-1 references, vision/text encoder configs + eval-transform parameters, cameras, policy
   rate, horizon) and ``<out_dir>/metrics.json``.

Hardware profiles: identical rules to ``stages/pretrain.py`` (YAML ``train`` < profile
``suggest.vtla`` / ``train`` < explicit overrides; :func:`load_stage_config` applies the profile
once and sets ``hardware_applied``).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from ..config import deep_merge
from ..datasets.episode import (D_LEVEL, D_RESIDUAL_Z, K_HAND_FINGERS, K_HAND_GLOBAL, K_HAND_WRIST,
                                K_Q, K_TAXEL_NRM, K_TAXEL_POS, Episode, cam_idx_key, list_episodes)
from ..representation.encoder import TactileFeatureSpec

log = logging.getLogger("robot_skin.stages.vtla")

STAGE = "vtla"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "stages" / "vtla.yaml"
METRICS_NAME = "metrics.json"

#: built-in defaults (mirrored by configs/stages/vtla.yaml; a test keeps them in sync)
DEFAULTS: dict[str, Any] = {
    "stage": STAGE,
    "hardware": None,
    "hardware_applied": False,
    "out_dir": None,
    "data": {
        "processed_root": "robot_skin/data/processed",
        "datasets": ["task"],
        "episodes": None,
        "splits": None,
        "split_by": "subject",
        "val_frac": 0.15,
        "test_frac": 0.15,
        "split_seed": 0,
        "phases": "task",
        "sample_stride": None,
        "min_valid_steps": 1,
        "require_valid_state": True,
        "require_success": False,
        "tactile_source": "auto",
        "bootstrap": {},
        "contact_rule": "level_ge_weak",
        "aux_target": "label",
    },
    "policy": {"policy_hz": 20.0, "horizon": 16, "obs_history": 1, "chunk_offset": 1,
               "cameras": ["ego"]},
    "action": {"kind": "hand_mano", "rel_mode": "delta", "norm_method": "std", "min_scale": 0.01},
    "features": {"obs_mode": "full", "history": 1, "stride": 1, "z_clip": 100.0, "z_scale": 2.0},
    "tactile": {"pretrained": None, "freeze": False, "calibrator": None, "baseline_model": None},
    "vision": {"encoder": {"type": "tiny", "out_dim": 128, "grid": [4, 4]}, "frozen": False,
               "cache_features": False, "cache_batch_size": 64},
    "image": {"image_size": [96, 128], "scale": [0.8, 1.0], "brightness": 0.2, "contrast": 0.2,
              "seed": None, "eval_crop_scale": None},
    "language": {"encoder": {"type": "hashing", "dim": 128, "max_len": 32}, "frozen": False},
    "model": {
        "d_model": 128, "fusion_depth": 2, "fusion_heads": 4, "ff_mult": 4, "dropout": 0.1,
        "n_readout": 1,
        "tactile_encoder": {"d_model": 64, "depth": 2, "heads": 4, "n_fourier": 6,
                            "fourier_scale": 0.3, "ff_mult": 4, "dropout": 0.0},
        "n_tactile_tokens": 4, "tactile_heads": 4, "tactile_gate": "hard",
        "head": "chunk", "head_depth": 2, "head_heads": None,
        "flow_steps": 10, "flow_tau": "uniform", "flow_tau_beta_b": 1.5,
        "p_drop_tactile": 0.1, "p_drop_vision": 0.0, "p_drop_language": 0.0,
        "aux_contact_weight": 0.0, "contact_pos_weight": None,
    },
    "eval": {"batch_size": 64, "seed": 0, "flow_steps": None, "splits": ["val", "test"]},
    "train": {
        "max_epochs": 50,
        "batch_size": 64,
        "lr": 3.0e-4,
        "weight_decay": 0.05,
        "warmup_steps": 200,
        "schedule": "cosine",
        "min_lr_ratio": 0.1,
        "grad_clip": 1.0,
        "precision": "auto",
        "ema_decay": None,
        "monitor": "val/loss",
        "log_every": 50,
        "seed": 0,
        "out_dir": "robot_skin/runs/vtla",
        "find_unused_parameters": False,
        "lr_mult": None,
    },
}

_SPLITS = ("train", "val", "test")
#: sections whose keys are validated elsewhere (TrainConfig warns; build_transforms raises)
_OPEN_SECTIONS = ("train", "image")


# ─────────────────────────────────────────────────────────────── config

def check_config(cfg: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` for keys unknown to :data:`DEFAULTS` (top level and one level into
    every section except ``train`` / ``image``) and for non-mapping sections — a typo in the YAML or
    a ``--set`` must not be silently ignored. Architecture keys the stage derives itself (e.g.
    ``model.horizon`` ← ``policy.horizon``) are therefore rejected in ``model`` too."""
    bad = sorted(set(cfg) - set(DEFAULTS))
    if bad:
        raise ValueError(f"{STAGE}: unknown config keys {bad}; valid: {sorted(DEFAULTS)}")
    for sec, dv in DEFAULTS.items():
        if not isinstance(dv, Mapping):
            continue
        v = cfg.get(sec)
        if v is None and sec == "image":
            continue
        if not isinstance(v, Mapping):
            hint = f" (disable with `{sec}.encoder: null`)" if sec in ("vision", "language") else ""
            raise ValueError(f"{STAGE}: section {sec!r} must be a mapping, got {v!r}{hint}")
        if sec in _OPEN_SECTIONS:
            continue
        bad = sorted(set(v) - set(dv))
        if bad:
            raise ValueError(f"{STAGE}: unknown keys {bad} in section {sec!r}; valid: {sorted(dv)}")

def apply_hardware(cfg: Mapping[str, Any]) -> dict:
    """Apply ``cfg["hardware"]`` (profile name / path / mapping / ``"auto"``) once: export its
    ``env`` (before any CUDA call), apply ``train`` / ``suggest.vtla`` and set
    ``hardware_applied``. The profile wins over ``train`` keys already in ``cfg``."""
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
    """:data:`DEFAULTS` ⊕ YAML (default ``configs/stages/vtla.yaml``) ⊕ hardware profile ⊕
    ``overrides`` (so ``--set train.batch_size=8`` beats the profile's ``suggest.vtla``)."""
    p = Path(path) if path is not None else CONFIG_PATH
    if p.is_file():
        y = yaml.safe_load(p.read_text()) or {}
        cfg = _replace_encoders(deep_merge(DEFAULTS, y), y)
    elif path is not None:
        raise FileNotFoundError(f"stage config not found: {p}")
    else:
        cfg = copy.deepcopy(DEFAULTS)
    ov = dict(overrides or {})
    hw_keys = {k: ov.pop(k) for k in ("hardware", "hardware_applied") if k in ov}
    cfg = apply_hardware(deep_merge(cfg, hw_keys))
    cfg = _replace_encoders(deep_merge(cfg, ov), ov)
    check_config(cfg)
    return cfg


def _replace_encoders(merged: dict, override: Mapping[str, Any]) -> dict:
    """After a deep merge: a *complete* encoder block (one with a ``type`` key) given in
    ``override`` (``vision.encoder`` / ``language.encoder``) replaces the base block instead of
    merging into it, so e.g. a ``resnet18`` block never inherits the ``tiny`` defaults' keys;
    partial blocks (``--set vision.encoder.out_dim=64``) still merge. ``null`` disables."""
    for sec, key in (("vision", "encoder"), ("language", "encoder")):
        blk = override.get(sec)
        if isinstance(blk, Mapping) and isinstance(blk.get(key), Mapping) and "type" in blk[key]:
            merged[sec][key] = copy.deepcopy(dict(blk[key]))
    return merged


def resolve_config(cfg: Mapping[str, Any] | None) -> dict:
    """Merge ``cfg`` over :data:`DEFAULTS` (complete encoder blocks replace, see
    :func:`_replace_encoders`), apply the hardware profile once, fix ``out_dir``."""
    cfg = dict(cfg or {})
    out = apply_hardware(_replace_encoders(deep_merge(DEFAULTS, cfg), cfg))
    check_config(out)
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


def _missing_inputs(ep: Episode, cfg: Mapping[str, Any], spec: TactileFeatureSpec) -> list[str]:
    miss: list[str] = []
    kind = cfg["action"]["kind"]
    if kind == "hand_mano":
        miss += [k for k in (K_HAND_GLOBAL, K_HAND_FINGERS, K_HAND_WRIST) if not ep.has(k)]
    elif not ep.has(K_Q):
        miss.append(K_Q)
    if spec.dim > 0:
        miss += [k for k in (K_TAXEL_POS, K_TAXEL_NRM) if not ep.has(k)]
        if cfg["data"].get("tactile_source") == "derived":
            miss += [k for k in (D_RESIDUAL_Z, D_LEVEL) if not ep.has_derived(k)]
    for cam in _cameras(cfg):
        if not ep.has(cam_idx_key(cam)):
            miss.append(f"camera_{cam}")
    return miss


def load_usable_episodes(dirs: Sequence[Path], cfg: Mapping[str, Any],
                         spec: TactileFeatureSpec) -> tuple[list[Episode], list[dict]]:
    """Load (mmap) the episodes that can provide every configured input; the others are listed
    with the reason (never abort the stage for one bad episode)."""
    usable, skipped = [], []
    for d in dirs:
        try:
            ep = Episode.load(d, mmap=True)
        except Exception as e:  # corrupt / foreign directory
            skipped.append({"episode": str(d), "reason": f"load failed: {e}"})
            continue
        miss = _missing_inputs(ep, cfg, spec)
        if miss:
            skipped.append({"episode": str(d), "reason": f"missing {miss}"})
            continue
        if cfg["data"].get("require_success") and not (ep.meta.task or {}).get("success"):
            skipped.append({"episode": str(d), "reason": "task not successful (require_success)"})
            continue
        usable.append(ep)
    return usable, skipped


def _match(entry: str, eps: Sequence[Episode], bases: Sequence[Path]) -> Episode | None:
    cand = [Path(entry)] + ([b / entry for b in bases] if not Path(entry).is_absolute() else [])
    resolved = {c.resolve() for c in cand if c.exists()}
    for ep in eps:
        r = ep.root.resolve() if ep.root is not None else None
        if (r is not None and r in resolved) or ep.meta.episode_id == entry or (
                ep.root is not None and ep.root.name == entry):
            return ep
    return None


def _is_episode_dir(entry: str, bases: Sequence[Path]) -> bool:
    """A splits entry that names an episode directory on disk (absolute or relative to one of
    ``bases``) — an episode outside this stage's pool, not a broken entry."""
    from ..datasets.episode import EPISODE_JSON

    cand = [Path(entry)] + ([b / entry for b in bases] if not Path(entry).is_absolute() else [])
    return any((c / EPISODE_JSON).is_file() for c in cand)


def split_episodes(eps: Sequence[Episode], data_cfg: Mapping[str, Any], *,
                   stage: str | None = STAGE) -> dict[str, list[Episode]]:
    """``{"train", "val", "test"}`` lists of episodes: from ``data.splits`` (splits.json) or
    :func:`robot_skin.datasets.splits.make_splits` (``split_by``: a ``GROUP_KEYS`` field;
    ``episode`` = ``episode_id``) — the latter logged as a warning
    (:func:`robot_skin.stages.warn_unshared_split`): it is not shared with the other stages. With a
    splits file, unresolvable entries and unlisted usable episodes are warned about; entries naming
    an existing episode directory outside this pool (e.g. D1 motion) are only logged."""
    splits_path = data_cfg.get("splits")
    if splits_path:
        spec = json.loads(Path(splits_path).read_text())
        spec = spec.get("splits", spec) if isinstance(spec, Mapping) else spec
        bases = [Path(data_cfg.get("processed_root") or "."), Path(splits_path).parent]
        out: dict[str, list[Episode]] = {k: [] for k in _SPLITS}
        unmatched = outside = 0
        for name in _SPLITS:
            for entry in spec.get(name, []) or []:
                ep = _match(str(entry), eps, bases)
                if ep is None:
                    # a shared splits.json lists every dataset's episodes (e.g. D1 motion): an existing
                    # episode dir outside this stage's pool is expected, an unresolvable entry is not
                    if _is_episode_dir(str(entry), bases):
                        outside += 1
                    else:
                        unmatched += 1
                elif all(ep is not e for part in out.values() for e in part):
                    out[name].append(ep)
        listed = {id(e) for part in out.values() for e in part}
        n_unlisted = sum(id(e) not in listed for e in eps)
        if unmatched or n_unlisted:
            warnings.warn(f"splits {splits_path}: {unmatched} entries without a usable episode, "
                          f"{n_unlisted} usable episodes not listed (ignored)", stacklevel=2)
        if outside:
            log.info("%s: splits %s lists %d episodes outside this stage's pool (other datasets / not usable here)",
                     stage or STAGE, splits_path, outside)
        return out
    from ..datasets.splits import make_splits

    by = data_cfg.get("split_by", "subject")
    by = "episode_id" if by == "episode" else by
    if any(e.root is None for e in eps):
        raise ValueError("automatic splits need on-disk episodes")
    from . import warn_unshared_split

    warn_unshared_split(stage, data_cfg, f"make_splits by {by}, val_frac={data_cfg.get('val_frac', 0.15)}, "
                        f"test_frac={data_cfg.get('test_frac', 0.15)}, split_seed={data_cfg.get('split_seed', 0)}, "
                        f"{len(eps)} episodes")
    parts = make_splits([e.root for e in eps], by=by, val_frac=float(data_cfg.get("val_frac", 0.15)),
                        test_frac=float(data_cfg.get("test_frac", 0.15)),
                        seed=int(data_cfg.get("split_seed", 0)))
    by_root = {str(e.root): e for e in eps}
    return {k: [by_root[p] for p in parts[k]] for k in _SPLITS}


# ─────────────────────────────────────────────────────────────── evaluation

def _group_slices(spec: Any) -> dict[str, slice]:
    return dict(spec.slices) if spec is not None else {}


@torch.no_grad()
def evaluate_policy(model: torch.nn.Module, dataset: Any, *, batch_size: int = 64,
                    device: torch.device | str | None = None, seed: int = 0,
                    n_steps: int | None = None, collate_fn: Any = None) -> dict[str, Any]:
    """Exact (count-weighted) chunk metrics of ``model.predict`` on ``dataset``.

    - ``l1``: mean |pred − target| over valid (step, dim) entries, normalized units;
    - ``l1_per_step``: the same per horizon step ``[H]`` (list; NaN → None where no step is valid);
    - ``l1_by_task``: ``{task_id: l1}``;
    - ``l1_raw/<group>``: mean abs error in raw (unnormalized, relative) units per action group
      (hand: ``wrist_pos`` m, ``wrist_rot6d``, ``finger_aa`` rad; robot: ``q``);
    - ``n_samples``, ``n_valid_steps``.

    The flow head samples with a CPU generator seeded by ``seed`` (deterministic given the
    dataset order and ``batch_size``).
    """
    from torch.utils.data import DataLoader

    from ..train import move_to_device
    from ..vtla.dataset import collate_vtla
    from ..vtla.losses import masked_step_sums

    device = torch.device(device) if device is not None else next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False,
                        collate_fn=collate_fn or collate_vtla)
    gen = torch.Generator().manual_seed(int(seed))
    norm = getattr(dataset, "action_normalizer", None)
    groups = _group_slices(getattr(dataset, "action_spec", None))
    H = int(getattr(dataset, "horizon", 0) or 0)
    step_sum = torch.zeros(H, dtype=torch.float64)
    step_cnt = torch.zeros(H, dtype=torch.float64)
    task_sum: dict[str, float] = {}
    task_cnt: dict[str, float] = {}
    raw_sum = {g: 0.0 for g in groups}
    raw_cnt = {g: 0.0 for g in groups}
    n = n_valid = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        pred = model.predict(batch, n_steps=n_steps, generator=gen).float()
        tgt = batch["actions"].float()
        valid = batch["action_valid"].bool()
        err = (pred - tgt).abs()
        s, c = masked_step_sums(err, valid)
        step_sum += s.cpu()
        step_cnt += c.cpu()
        per_sample = (err * valid[..., None]).sum(dim=(1, 2)).double().cpu()
        per_count = (valid.sum(1) * err.shape[-1]).double().cpu()
        for b, task in enumerate(batch.get("task_id") or [""] * err.shape[0]):
            task_sum[task] = task_sum.get(task, 0.0) + float(per_sample[b])
            task_cnt[task] = task_cnt.get(task, 0.0) + float(per_count[b])
        if norm is not None and groups:
            pr, tr = norm.unnormalize(pred), norm.unnormalize(tgt)
            rerr = (pr - tr).abs()
            for g, sl in groups.items():
                e = rerr[..., sl]
                raw_sum[g] += float((e * valid[..., None]).sum())
                raw_cnt[g] += float(valid.sum()) * e.shape[-1]
        n += err.shape[0]
        n_valid += int(valid.sum())
    tot_c = float(step_cnt.sum())
    means = (step_sum / step_cnt.clamp_min(1)).tolist()
    per_step = [float(v) if c > 0 else None for v, c in zip(means, step_cnt.tolist(), strict=True)]
    out: dict[str, Any] = {
        "l1": float(step_sum.sum()) / tot_c if tot_c > 0 else None,
        "l1_per_step": per_step,
        "l1_by_task": {k: (task_sum[k] / task_cnt[k] if task_cnt[k] > 0 else None) for k in sorted(task_sum)},
        "n_samples": n,
        "n_valid_steps": n_valid,
    }
    for g in groups:
        out[f"l1_raw/{g}"] = raw_sum[g] / raw_cnt[g] if raw_cnt[g] > 0 else None
    return out


# ─────────────────────────────────────────────────────────────── run

def _cameras(cfg: Mapping[str, Any]) -> list[str]:
    if not (cfg.get("vision") or {}).get("encoder"):
        return []
    return [str(c) for c in (cfg["policy"].get("cameras") or [])]


def _finite(obj: Any) -> Any:
    """Recursively replace non-finite floats by ``None`` (strict JSON)."""
    if isinstance(obj, Mapping):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _calibrator_state(ref: Any) -> dict | None:
    """The contact stage's ``calibrator.json`` content (``ref`` = the file or its run directory) —
    embedded in the bundle so control does not depend on the training box's paths."""
    if not ref:
        return None
    p = Path(ref)
    if p.is_dir():
        p = p / "calibrator.json"
    if not p.is_file():
        warnings.warn(f"tactile.calibrator {ref!r} not found: the bundle keeps only the reference",
                      stacklevel=2)
        return None
    return json.loads(p.read_text())


def _weights_hash(module: torch.nn.Module) -> str:
    """Short hash of a module's weights (raw bytes, so bf16 / fp16 / bool tensors work too)."""
    h = hashlib.sha1()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:10]


def _model_config(cfg: Mapping[str, Any], spec: TactileFeatureSpec, action_dim: int,
                  cameras: Sequence[str]) -> dict:
    m = dict(cfg["model"])
    p = cfg["policy"]
    return {
        **m,
        "action_dim": int(action_dim), "proprio_dim": int(action_dim),
        "horizon": int(p["horizon"]), "obs_history": int(p["obs_history"]),
        "cameras": list(cameras),
        "vision": copy.deepcopy(cfg["vision"].get("encoder")) if cameras else None,
        "vision_frozen": bool(cfg["vision"].get("frozen") or cfg["vision"].get("cache_features")),
        "text": copy.deepcopy((cfg.get("language") or {}).get("encoder")),
        "text_frozen": bool((cfg.get("language") or {}).get("frozen")),
        "feature_spec": spec.to_dict(),
        "tactile_frozen": bool(cfg["tactile"].get("freeze")),
    }


def run(cfg: Mapping[str, Any] | None = None) -> dict:
    """Train the VTLA policy; returns (and writes) the metrics dict; writes ``policy_bundle.pt``."""
    from ..representation.encoder import load_pretrained_encoder
    from ..train import TrainConfig, Trainer, barrier, seed_everything
    from ..train.checkpoint import BEST_NAME
    from ..train.distributed import init_distributed
    from ..vtla.dataset import VTLACollator, VTLADataset, eval_transform_to_dict
    from ..vtla.losses import vtla_loss
    from ..vtla.model import POLICY_BUNDLE_NAME, VTLAPolicy, save_policy_bundle

    cfg = resolve_config(cfg)
    out_dir = Path(cfg["out_dir"])
    d_cfg, p_cfg, a_cfg, t_cfg = cfg["data"], cfg["policy"], cfg["action"], cfg["tactile"]
    spec = TactileFeatureSpec.from_dict(cfg["features"])

    # pretrained tactile encoder (its feature spec is authoritative)
    pre_enc = None
    if t_cfg.get("pretrained") and spec.dim > 0:
        pre_enc = load_pretrained_encoder(t_cfg["pretrained"], freeze=bool(t_cfg.get("freeze")))
        if pre_enc.feature_spec is not None and pre_enc.feature_spec != spec:
            warnings.warn(f"features {spec} differ from the pretrained encoder's {pre_enc.feature_spec}; "
                          "using the pretrained encoder's feature spec", stacklevel=2)
            spec = pre_enc.feature_spec

    dirs = discover_episodes(d_cfg)
    eps, skipped = load_usable_episodes(dirs, cfg, spec)
    for s in skipped:
        log.warning("vtla: skipping %s (%s)", s["episode"], s["reason"])
    from . import warn_legacy_taxel_frame

    warn_legacy_taxel_frame(eps, STAGE)
    if not eps:
        raise ValueError(f"no usable episodes ({len(dirs)} found under {d_cfg.get('processed_root')!r}, "
                         f"{len(skipped)} skipped: {skipped[:3]})")
    split = split_episodes(eps, d_cfg)
    if not split["train"]:
        raise ValueError("the training split is empty")

    cameras = _cameras(cfg)
    ds_kw = dict(cameras=cameras, policy_hz=float(p_cfg["policy_hz"]), horizon=int(p_cfg["horizon"]),
                 obs_history=int(p_cfg["obs_history"]), chunk_offset=int(p_cfg["chunk_offset"]),
                 action_spec=a_cfg["kind"], rel_mode=a_cfg["rel_mode"], feature_spec=spec,
                 phases=d_cfg.get("phases", "task"), sample_stride=d_cfg.get("sample_stride"),
                 min_valid_steps=int(d_cfg.get("min_valid_steps", 1)),
                 require_valid_state=bool(d_cfg.get("require_valid_state", True)),
                 contact_rule=d_cfg.get("contact_rule", "level_ge_weak"),
                 aux_target=d_cfg.get("aux_target", "label"),
                 tactile_source=d_cfg.get("tactile_source", "auto"),
                 bootstrap=d_cfg.get("bootstrap") or {})
    train_ds = VTLADataset(split["train"], **ds_kw)
    if len(train_ds) == 0:
        raise ValueError(f"{len(split['train'])} training episodes but no sample ticks (phases "
                         f"{d_cfg.get('phases')!r}, policy_hz {p_cfg['policy_hz']})")
    a_norm, p_norm = train_ds.fit_normalizers(a_cfg.get("norm_method", "std"),
                                              min_scale=float(a_cfg.get("min_scale", 0.01)))
    ds_kw["action_spec"] = train_ds.action_spec
    others = {}
    for name in ("val", "test"):
        if split[name]:
            ds = VTLADataset(split[name], action_normalizer=a_norm, proprio_normalizer=p_norm, **ds_kw)
            others[name] = ds if len(ds) else None
        else:
            others[name] = None

    # model
    t_train = dict(cfg["train"])
    seed_everything(int(t_train.get("seed", 0)))
    mcfg = _model_config(cfg, spec, train_ds.action_dim, cameras)
    model = VTLAPolicy(mcfg, tactile_encoder=pre_enc)

    # images: transforms (+ optional frozen-feature cache)
    train_tf = eval_tf = None
    cache_key = None
    if model.vision_encoder is not None:
        from ..vision.transforms import build_transforms
        train_tf, eval_tf = build_transforms(cfg.get("image") or {}, encoder=model.vision_encoder)
        if cfg["vision"].get("cache_features"):
            from ..vision.feature_cache import cache_features
            dist = init_distributed()
            cache_key = f"{model.vision_encoder.cache_key}_{_weights_hash(model.vision_encoder)}"
            if dist.is_main:
                want_cpu = str(t_train.get("device", "auto")) == "cpu"
                dev = "cuda" if torch.cuda.is_available() and not want_cpu else "cpu"
                model.vision_encoder.to(dev)
                cache_features([e for part in split.values() for e in part], cameras, model.vision_encoder,
                               eval_tf, key=cache_key, device=dev,
                               batch_size=int(cfg["vision"].get("cache_batch_size", 64)))
                model.vision_encoder.to("cpu")
            barrier(dist)
    for ds, tf in ((train_ds, train_tf), (others["val"], eval_tf), (others["test"], eval_tf)):
        if ds is not None:
            ds.image_transform = tf
            ds.use_cached_vision = cache_key

    # train
    val_ds = others["val"]
    if val_ds is None and str(t_train.get("monitor", "val/loss")).startswith("val/"):
        t_train["monitor"] = "train/" + str(t_train.get("monitor", "val/loss"))[4:]
    tcfg = TrainConfig.from_dict(t_train)
    tokenizer = model.text_encoder.get_tokenizer() if model.text_encoder is not None else None
    collate = VTLACollator(tokenizer)
    extra = {"stage": STAGE, "model_config": model.config, "action_spec": train_ds.action_spec.to_dict(),
             "action_normalizer": a_norm.to_dict(), "proprio_normalizer": p_norm.to_dict(),
             "rel_mode": a_cfg["rel_mode"], "feature_spec": spec.to_dict()}
    trainer = Trainer(model, vtla_loss, tcfg, train_ds, val_ds, collate_fn=collate, extra_state=extra)
    history = trainer.fit()
    barrier(trainer.dist)

    metrics: dict[str, Any] = {
        "stage": STAGE,
        "out_dir": str(out_dir),
        "bundle_path": str(out_dir / POLICY_BUNDLE_NAME),
        "n_episodes": {"found": len(dirs), "skipped": len(skipped),
                       **{k: len(v) for k, v in split.items()}},
        "n_samples": {"train": len(train_ds),
                      **{k: (len(v) if v is not None else 0) for k, v in others.items()}},
        "skipped": skipped,
        "tactile_source": train_ds.tactile_source,
        "feature_spec": spec.to_dict(),
        "head": model.cfg.head,
        "epochs": trainer.epoch,
        "steps": trainer.step,
        "best": {"monitor": tcfg.monitor, "value": trainer.best_value, "epoch": trainer.best_epoch},
        "final_train_loss": history[-1].get("train/loss") if history else None,
    }
    if trainer.dist.is_main:
        best = out_dir / BEST_NAME
        if best.is_file():
            Trainer.load_model_weights(model, best, use_ema=True)
        elif trainer.ema is not None:
            trainer.ema.apply_to(model)
        model.eval()
        e_cfg = cfg.get("eval") or {}
        eval_sets = {k: others.get(k) for k in (e_cfg.get("splits") or ["val", "test"])
                     if others.get(k) is not None}
        if not eval_sets:
            train_eval = VTLADataset(split["train"], action_normalizer=a_norm, proprio_normalizer=p_norm,
                                     image_transform=eval_tf, use_cached_vision=cache_key, **ds_kw)
            eval_sets = {"train_eval": train_eval}
        for name, ds in eval_sets.items():
            res = evaluate_policy(model, ds, batch_size=int(e_cfg.get("batch_size", 64)),
                                  device=trainer.device, seed=int(e_cfg.get("seed", 0)),
                                  n_steps=e_cfg.get("flow_steps"), collate_fn=collate)
            metrics.update({f"{name}/{k}": v for k, v in res.items()})
        metrics = _finite(metrics)
        eval_metrics = {k: v for k, v in metrics.items() if "/" in k and not k.startswith("best")}
        save_policy_bundle(
            out_dir / POLICY_BUNDLE_NAME, model,
            action={"spec": train_ds.action_spec.to_dict(), "normalizer": a_norm.to_dict(),
                    "rel_mode": a_cfg["rel_mode"], "chunk_offset": int(p_cfg["chunk_offset"])},
            proprio={"source": "action_state", "normalizer": p_norm.to_dict(),
                     "history": int(p_cfg["obs_history"])},
            tactile={"feature_spec": spec.to_dict(), "obs_mode": spec.obs_mode,
                     "contact_rule": ds_kw["contact_rule"], "source": train_ds.tactile_source,
                     "bootstrap": (dict(d_cfg.get("bootstrap") or {})
                                   if train_ds.tactile_source in ("bootstrap", "mixed") else None),
                     "calibrator": t_cfg.get("calibrator"),
                     "calibrator_state": _calibrator_state(t_cfg.get("calibrator")),
                     "baseline_model": t_cfg.get("baseline_model"),
                     "pretrained_encoder": t_cfg.get("pretrained"), "frozen": bool(t_cfg.get("freeze")),
                     "layouts": sorted({e.meta.layout for e in eps})},
            vision={"cameras": list(cameras), "encoder": model.cfg.vision, "frozen": model.cfg.vision_frozen,
                    "cached_features_key": cache_key, "eval_transform": eval_transform_to_dict(eval_tf),
                    "transform_config": cfg.get("image")},
            language={"encoder": model.cfg.text, "frozen": model.cfg.text_frozen},
            timing={"policy_hz": float(p_cfg["policy_hz"]), "source_hz": train_ds.source_hz,
                    "stride": train_ds.stride, "horizon": int(p_cfg["horizon"]),
                    "obs_history": int(p_cfg["obs_history"]), "sample_stride": train_ds.sample_stride},
            meta={"stage": STAGE, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "episodes": {k: [str(e.root) for e in v] for k, v in split.items()},
                  "phases": d_cfg.get("phases"), "best": metrics["best"], "metrics": eval_metrics,
                  "ensemble_k": 0.01})
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f".{METRICS_NAME}.tmp"
        tmp.write_text(json.dumps(metrics, indent=2, default=str, allow_nan=False))
        os.replace(tmp, out_dir / METRICS_NAME)
        log.info("vtla done: %s", eval_metrics)
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
    """``python -m robot_skin.stages.vtla [--config yaml] [--set model.head=flow ...]``."""
    ap = argparse.ArgumentParser(prog="python -m robot_skin.stages.vtla",
                                 description="Train the VTLA policy and write policy_bundle.pt.")
    ap.add_argument("--config", default=None, help=f"stage YAML (default {CONFIG_PATH})")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted override, value parsed as YAML (repeatable)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_stage_config(args.config, _parse_set(args.set))
    metrics = run(cfg)
    json.dump({k: v for k, v in metrics.items() if k != "skipped"}, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
