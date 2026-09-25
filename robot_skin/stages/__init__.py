"""Stage runners: each ``robot_skin.stages.<stage>.run(cfg) -> metrics`` trains one pipeline stage.

Stage modules are imported explicitly (``from robot_skin.stages import baseline``); this package
only holds stage-agnostic helpers shared by the stage-1 runners (``imu_pose``, ``baseline``,
``contact``): defaults ⊕ YAML ⊕ hardware profile ⊕ overrides, key validation, episode pools and
splits (same ``data.*`` semantics as ``stages.pretrain``), the Trainer wrapper that restores the best
checkpoint, strict-JSON metrics and the ``--set`` CLI parser. Importing the package does not import
torch (heavy imports happen inside the functions).
"""
from __future__ import annotations

import copy
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["apply_stage_hardware", "load_stage_yaml", "resolve_stage_config", "check_stage_keys",
           "finite_json", "write_json_atomic", "parse_overrides", "stage_episodes", "split_stage_episodes",
           "seed_model_init", "fit_and_restore"]


def _deep_merge(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict:
    from ..config import deep_merge

    return deep_merge(a, b)


def apply_stage_hardware(cfg: Mapping[str, Any], stage: str) -> dict:
    """Apply ``cfg["hardware"]`` (profile name / YAML path / mapping / ``"auto"``) once: export the
    profile ``env`` (before any CUDA call), merge its ``suggest.<stage>`` and ``train`` keys over
    ``cfg["train"]`` (:func:`robot_skin.train.apply_hw_profile`) and set ``hardware_applied``.
    No-op without ``hardware`` or when already applied."""
    out = copy.deepcopy(dict(cfg))
    hw = out.get("hardware")
    if not hw or out.get("hardware_applied"):
        return out
    from ..train.hardware import apply_hw_profile, apply_profile_env, detect_hw_profile, load_hw_profile

    if hw == "auto" and detect_hw_profile() is None:
        warnings.warn("hardware: auto — no built-in profile matches this machine; using the stage defaults",
                      stacklevel=2)
    else:
        prof = dict(hw) if isinstance(hw, Mapping) else load_hw_profile(hw)
        apply_profile_env(prof)
        out = apply_hw_profile(out, prof, stage=stage)
    out["hardware_applied"] = True
    return out


def check_stage_keys(cfg: Mapping[str, Any], defaults: Mapping[str, Any], stage: str,
                     *, open_sections: Sequence[str] = ("train",)) -> None:
    """Raise ``ValueError`` for keys unknown to ``defaults`` (top level and one level into every
    mapping section) — a typo in a stage YAML must not be silently ignored. ``open_sections``
    (``train``: :class:`TrainConfig` warns about its own unknown keys) are not checked."""
    bad = sorted(set(cfg) - set(defaults))
    if bad:
        raise ValueError(f"{stage}: unknown config keys {bad}; valid: {sorted(defaults)}")
    for sec, dv in defaults.items():
        if sec in open_sections or not isinstance(dv, Mapping) or not isinstance(cfg.get(sec), Mapping):
            continue
        bad = sorted(set(cfg[sec]) - set(dv))
        if bad:
            raise ValueError(f"{stage}: unknown keys {bad} in section {sec!r}; valid: {sorted(dv)}")


def load_stage_yaml(defaults: Mapping[str, Any], stage: str, config_path: str | Path,
                    path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict:
    """``defaults`` ⊕ YAML (``path`` or the stage's ``config_path`` if it exists) ⊕ hardware profile
    ⊕ ``overrides``. The profile (``hardware`` from the YAML or the overrides) is applied before the
    other overrides, so ``--set train.batch_size=32`` beats the profile's suggestion."""
    import yaml

    p = Path(path) if path is not None else Path(config_path)
    if p.is_file():
        cfg = _deep_merge(defaults, yaml.safe_load(p.read_text()) or {})
    elif path is not None:
        raise FileNotFoundError(f"stage config not found: {p}")
    else:
        cfg = copy.deepcopy(dict(defaults))
    ov = dict(overrides or {})
    hw = {k: ov.pop(k) for k in ("hardware", "hardware_applied") if k in ov}
    cfg = apply_stage_hardware(_deep_merge(cfg, hw), stage)
    cfg = _deep_merge(cfg, ov)
    check_stage_keys(cfg, defaults, stage)
    return cfg


def resolve_stage_config(cfg: Mapping[str, Any] | None, defaults: Mapping[str, Any], stage: str) -> dict:
    """Merge ``cfg`` over ``defaults``, validate keys, apply the hardware profile once and set
    ``out_dir`` (``cfg.out_dir`` or ``train.out_dir``; both end up equal)."""
    out = _deep_merge(defaults, cfg or {})
    check_stage_keys(out, defaults, stage)
    out = apply_stage_hardware(out, stage)
    out_dir = out.get("out_dir") or (out.get("train") or {}).get("out_dir") or f"robot_skin/runs/{stage}"
    out["out_dir"] = str(out_dir)
    out.setdefault("train", {})["out_dir"] = str(out_dir)
    return out


def finite_json(obj: Any) -> Any:
    """Recursively make ``obj`` strict-JSON safe: non-finite floats → None, numpy scalars / arrays →
    Python, tuples → lists."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(obj, Mapping):
        return {str(k): finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.ndarray):
            return finite_json(obj.tolist())
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_json_atomic(path: str | Path, obj: Any) -> Path:
    """Write strict JSON (:func:`finite_json`) via a temp file + ``os.replace``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(finite_json(obj), indent=2, default=str, allow_nan=False))
    os.replace(tmp, p)
    return p


def parse_overrides(items: Sequence[str]) -> dict:
    """``["train.lr=1e-3", "data.datasets=[motion]"]`` → nested dict (values parsed as YAML)."""
    import yaml

    from ..train.sweep import set_by_path

    out: dict = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        set_by_path(out, k.strip(), yaml.safe_load(v))
    return out


def stage_episodes(data_cfg: Mapping[str, Any], dataset_keys: Sequence[str] = ("datasets", "predict_datasets")
                   ) -> tuple[list, list[dict]]:
    """Load (mmap) every episode the stage may touch: ``data.episodes`` (explicit dirs, relative ones
    resolved against ``processed_root``) or ``processed_root`` × the union of the ``dataset_keys``
    lists. Returns ``(episodes, skipped)`` — unreadable directories are reported, never fatal."""
    from ..datasets.episode import Episode, list_episodes

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
    else:
        names: list[str] = []
        for k in dataset_keys:
            v = data_cfg.get(k) or []
            names += [v] if isinstance(v, str) else list(v)
        dirs = [d for ds in sorted(set(names)) for d in list_episodes(root, ds)]
    eps, skipped = [], []
    for d in sorted(set(dirs)):
        try:
            eps.append(Episode.load(d, mmap=True))
        except Exception as e:  # noqa: BLE001 - corrupt / foreign dir: report, do not abort
            skipped.append({"episode": str(d), "reason": f"load failed: {type(e).__name__}: {e}"})
    return eps, skipped


def split_stage_episodes(eps: Sequence, data_cfg: Mapping[str, Any]) -> dict[str, list]:
    """Leakage-safe train / val / test split of loaded episodes — exactly
    :func:`robot_skin.stages.pretrain.split_episodes` (``data.splits`` = a ``splits.json`` from
    ``datasets.splits``, test excluded from training unless ``use_test``; else a seeded ``val_frac``
    split by episode or subject), so every stage sees the same held-out episodes."""
    from .pretrain import split_episodes

    return split_episodes(list(eps), data_cfg)


def seed_model_init(train_cfg: Mapping[str, Any]) -> int:
    """Seed torch's global RNG with ``train.seed`` right before a stage builds its model. The
    :class:`robot_skin.train.Trainer` seeds only *after* the model exists, and torch's initial seed
    is random per process, so without this the initial weights — and every stage result — would
    differ from run to run. The same seed on every rank (DDP broadcasts rank 0's weights anyway)."""
    import torch

    seed = int(train_cfg.get("seed", 0) or 0)
    torch.manual_seed(seed)
    return seed


def fit_and_restore(model, loss_fn, train_cfg: Mapping[str, Any], train_ds, val_ds=None, *,
                    collate_fn=None, extra_state: Mapping[str, Any] | None = None):
    """Build a :class:`robot_skin.train.Trainer` (``val/*`` monitor → ``train/*`` without a
    validation set), ``fit()``, then load the best checkpoint's weights (EMA when enabled) into
    ``model`` on every rank. Returns the trainer (``history``, ``step``, ``best_value`` …)."""
    from ..train import TrainConfig, Trainer, barrier
    from ..train.checkpoint import BEST_NAME

    t_cfg = dict(train_cfg)
    mon = str(t_cfg.get("monitor", "val/loss"))
    if val_ds is None and mon.startswith("val/"):
        t_cfg["monitor"] = "train/" + mon[4:]
    trainer = Trainer(model, loss_fn, TrainConfig.from_dict(t_cfg), train_ds, val_ds, collate_fn=collate_fn,
                      extra_state=dict(extra_state or {}))
    trainer.fit()
    barrier(trainer.dist)                                  # rank 0 has written ckpt_best.pt
    best = Path(trainer.out_dir) / BEST_NAME
    if best.is_file():
        Trainer.load_model_weights(model, best, use_ema=True)
    elif trainer.ema is not None:
        trainer.ema.apply_to(model)
    model.eval()
    return trainer
