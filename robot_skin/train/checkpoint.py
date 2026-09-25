"""Crash-safe checkpoint IO for :mod:`robot_skin.train`.

Long runs on remote GPU boxes (RTX 5090 / 4090 reached over Tailscale) get interrupted: SSH
drops, preemption, OOM kills. A checkpoint that is half-written when that happens must never
replace the previous good one, so :func:`save_checkpoint` writes to a temporary file in the
same directory, ``fsync``-s it and atomically ``os.replace``-s it into place (POSIX rename
semantics; also atomic on Windows/NTFS for same-volume replaces).

File naming inside a run directory (``TrainConfig.out_dir``)::

    ckpt_last.pt   latest state (resume point)       — find_last() (else the newest other ckpt*.pt)
    ckpt_best.pt   best state by TrainConfig.monitor — find_best()

Checkpoint dict keys written by :class:`robot_skin.train.engine.Trainer`: ``model, optimizer,
scheduler, scaler, ema, step, epoch, config, extra`` (+ ``best, bad_epochs, history, rng,
batch_in_epoch``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

LAST_NAME = "ckpt_last.pt"
BEST_NAME = "ckpt_best.pt"

__all__ = ["LAST_NAME", "BEST_NAME", "save_checkpoint", "load_checkpoint", "find_last",
           "find_best", "strip_state_dict_prefixes"]


def save_checkpoint(path: str | Path, **state: Any) -> Path:
    """Atomically ``torch.save(state, path)`` (temp file + fsync + ``os.replace``).

    Parent directories are created. Returns the final path. A crash mid-write leaves the
    previous file at ``path`` untouched (only a hidden ``.<name>.tmp-<pid>`` may remain, which
    :func:`find_last` ignores).
    """
    path = Path(path)
    if not state:
        raise ValueError("save_checkpoint: nothing to save (pass state as keyword arguments)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    _fsync_dir(path.parent)
    return path


def _fsync_dir(d: Path) -> None:
    """Best-effort directory fsync so the rename itself is durable (no-op where unsupported)."""
    if os.name != "posix":
        return
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def load_checkpoint(path: str | Path, map_location: Any = "cpu", *,
                    weights_only: bool = False) -> dict:
    """Load a checkpoint dict written by :func:`save_checkpoint`.

    ``weights_only=False`` (default) because our checkpoints carry plain-python config/history
    and possibly numpy arrays in ``extra``; only load files you produced yourself — unpickling
    untrusted files can execute code. Pass ``weights_only=True`` for third-party weights.
    """
    path = Path(path)
    if path.is_dir():
        found = find_last(path)
        if found is None:
            raise FileNotFoundError(f"no checkpoint found in directory {path}")
        path = found
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    obj = torch.load(path, map_location=map_location, weights_only=weights_only)
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected a checkpoint dict, got {type(obj).__name__}")
    return obj


_STEP_RE = re.compile(r"(\d+)")


def _candidates(out_dir: Path) -> Iterable[Path]:
    """Trainer checkpoints only (``ckpt*.pt``): a run directory also holds the stage artefacts
    (``policy_bundle.pt``, ``encoder_state.pt``, ``*_model.pt`` …), which are not resumable."""
    for p in out_dir.glob("ckpt*.pt"):
        if p.name.startswith(".") or p.name == BEST_NAME:
            continue
        yield p


def find_last(out_dir: str | Path) -> Path | None:
    """Resume point of a run directory: ``ckpt_last.pt`` if present, otherwise the most recent
    other ``ckpt*.pt`` (by the largest number in its name, then mtime; ``ckpt_best.pt`` and stage
    artefacts such as ``policy_bundle.pt`` never count); ``None`` if nothing."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    last = out_dir / LAST_NAME
    if last.is_file():
        return last
    cands = list(_candidates(out_dir))
    if not cands:
        return None

    def key(p: Path) -> tuple[int, float]:
        nums = _STEP_RE.findall(p.stem)
        return (int(nums[-1]) if nums else -1, p.stat().st_mtime)

    return max(cands, key=key)


def find_best(out_dir: str | Path) -> Path | None:
    """``out_dir/ckpt_best.pt`` if it exists."""
    p = Path(out_dir) / BEST_NAME
    return p if p.is_file() else None


def strip_state_dict_prefixes(state: Mapping[str, Any],
                              prefixes: tuple[str, ...] = ("module.", "_orig_mod.")) -> dict:
    """Remove wrapper prefixes left by DDP (``module.``) / ``torch.compile`` (``_orig_mod.``)
    so weights load into a bare module. Repeated / nested prefixes are all stripped."""
    out = {}
    for k, v in state.items():
        changed = True
        while changed:
            changed = False
            for pre in prefixes:
                if k.startswith(pre):
                    k = k[len(pre):]
                    changed = True
        out[k] = v
    return out
