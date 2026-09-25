"""Load ``robot_skin/configs/default.yaml`` and deep-merge overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"

#: the keys ``configs/default.yaml`` (and a config / overrides merged over it) may hold; ``None`` = any
#: value. A misspelled key (``pipeline.splits.val_fraction``) is an error, never silently ignored.
SCHEMA: dict[str, Any] = {
    "paths": {"raw_root": None, "processed_root": None, "runs_root": None, "synthetic_root": None},
    "hardware": None,
    "stages": {k: None for k in ("preprocess", "imu_pose", "baseline", "contact", "pretrain", "vtla", "deploy")},
    "pipeline": {"stages": None, "splits": {k: None for k in ("file", "by", "val_frac", "test_frac", "seed",
                                                                "datasets", "holdout")}},
    "synthetic": {k: None for k in ("kind", "n_motion", "n_task", "subjects", "seed", "duration_s", "cameras",
                                    "image_hw")},
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(dict(base))
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def check_config(cfg: Mapping[str, Any], schema: Mapping[str, Any] = SCHEMA, where: str = "") -> None:
    """Raise ``ValueError`` for a key of ``cfg`` that :data:`SCHEMA` does not know (recursively)."""
    for k, v in cfg.items():
        if k not in schema:
            raise ValueError(f"configs/default.yaml: unknown key {where + str(k)!r}; valid: {sorted(schema)}")
        sub = schema[k]
        if isinstance(sub, Mapping) and isinstance(v, Mapping):
            check_config(v, sub, f"{where}{k}.")


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict:
    """``configs/default.yaml`` ⊕ the YAML ``path`` ⊕ ``overrides`` (deep merge); unknown keys raise
    (:func:`check_config`)."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path is not None:
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    check_config(cfg)
    return cfg
