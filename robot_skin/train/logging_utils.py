"""Metric logging: always a JSONL file, optionally TensorBoard / Weights & Biases.

``metrics.jsonl`` (one JSON object per line: ``{"step", "time", "kind", <metrics>}``) is the
source of truth — it is plain text, appends safely, survives crashes and is trivially
``rsync``-ed back from a remote GPU box over Tailscale. TensorBoard (``tensorboard`` package)
and W&B (``wandb``) are optional sinks; if the package is missing a warning is issued and the
sink is skipped, never an error. Only the main rank should log (``enabled=is_main``).
"""
from __future__ import annotations

import json
import logging
import math
import numbers
import time
import warnings
from pathlib import Path
from typing import Any, Mapping

import torch

__all__ = ["JsonlLogger", "read_jsonl", "to_float_dict", "setup_logging"]


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return _to_jsonable(v.detach().item() if v.numel() == 1 else v.detach().cpu().tolist())
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        return f if math.isfinite(f) else str(f)  # JSON has no NaN/inf
    if hasattr(v, "item") and callable(v.item):  # numpy scalar / size-1 array
        try:
            return _to_jsonable(v.item())
        except (ValueError, TypeError):
            pass
    if hasattr(v, "tolist") and callable(v.tolist):  # numpy array (NaN handled per element)
        try:
            return _to_jsonable(v.tolist())
        except (ValueError, TypeError):
            pass
    if isinstance(v, Mapping):
        return {str(k): _to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return str(v)


def to_float_dict(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep only scalar-valued entries, as python floats (for TensorBoard / reductions)."""
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            if v.numel() != 1:
                continue
            v = v.detach().item()
        if isinstance(v, bool):
            v = float(v)
        if isinstance(v, numbers.Real):
            out[k] = float(v)
        elif hasattr(v, "item") and not isinstance(v, str):
            try:
                out[k] = float(v.item())
            except (TypeError, ValueError):
                pass
    return out


class JsonlLogger:
    """Append-only JSONL metric logger with optional TensorBoard / W&B mirrors.

    Args:
        out_dir: run directory (created).
        filename: JSONL file name inside ``out_dir``.
        enabled: ``False`` makes every call a no-op (non-main DDP ranks).
        tensorboard: also write ``out_dir/tb`` event files (needs ``tensorboard``).
        wandb: ``None`` or kwargs for ``wandb.init`` (e.g. ``{"project": "robot_skin"}``).
    """

    def __init__(self, out_dir: str | Path, filename: str = "metrics.jsonl", *,
                 enabled: bool = True, tensorboard: bool = False,
                 wandb: Mapping[str, Any] | None = None):
        self.enabled = bool(enabled)
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / filename
        self._tb = None
        self._wandb = None
        if not self.enabled:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # needs `tensorboard`

                self._tb = SummaryWriter(log_dir=str(self.out_dir / "tb"))
            except Exception as e:  # ImportError or tensorboard version issues
                warnings.warn(f"TensorBoard logging disabled ({e}); pip install tensorboard",
                              stacklevel=2)
        if wandb is not None:
            try:
                import wandb as _wandb  # type: ignore[import-not-found]

                kw = dict(wandb)
                kw.setdefault("dir", str(self.out_dir))
                self._wandb = _wandb.init(**kw)
            except Exception as e:
                warnings.warn(f"W&B logging disabled ({e}); pip install wandb && wandb login",
                              stacklevel=2)

    def log(self, step: int, metrics: Mapping[str, Any], *, kind: str | None = None) -> None:
        """Append one record ``{"step", "time", ["kind"], **metrics}``."""
        if not self.enabled:
            return
        rec: dict[str, Any] = {"step": int(step), "time": round(time.time(), 3)}
        if kind is not None:
            rec["kind"] = kind
        rec.update({str(k): _to_jsonable(v) for k, v in metrics.items()})
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        scalars = to_float_dict(metrics)
        if self._tb is not None:
            for k, v in scalars.items():
                self._tb.add_scalar(k, v, global_step=int(step))
        if self._wandb is not None:
            self._wandb.log(scalars, step=int(step))

    def log_config(self, config: Mapping[str, Any]) -> None:
        """Record the run config (W&B config / TensorBoard text)."""
        if not self.enabled:
            return
        if self._wandb is not None:
            self._wandb.config.update(_to_jsonable(dict(config)), allow_val_change=True)
        if self._tb is not None:
            self._tb.add_text("config", "```\n" + json.dumps(_to_jsonable(dict(config)),
                                                               indent=2) + "\n```")

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
            self._tb = None
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str | Path, kind: str | None = None) -> list[dict]:
    """Read a JSONL metrics file (optionally only records of one ``kind``); skips a torn last
    line left by a crash."""
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if kind is None or rec.get("kind") == kind:
            out.append(rec)
    return out


def setup_logging(level: int = logging.INFO, rank: int = 0,
                  name: str = "robot_skin") -> logging.Logger:
    """Configure the ``robot_skin`` logger: INFO on rank 0, WARNING on other ranks, rank-tagged
    format. Idempotent."""
    logger = logging.getLogger(name)
    logger.setLevel(level if rank == 0 else max(level, logging.WARNING))
    if not any(getattr(h, "_robot_skin", False) for h in logger.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(f"%(asctime)s [r{rank}] %(name)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        h._robot_skin = True  # type: ignore[attr-defined]
        logger.addHandler(h)
    return logger
