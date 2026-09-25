"""Generic training engine shared by every ``robot_skin`` stage (imu_pose, baseline, contact,
pretrain, vtla).

Keeps the run-directory conventions of the SATS trainer (``deformable_sats/sats/training/
train_e2e.py``: ``config.json``, ``history.json``, best/last checkpoints) and adds what
multi-GPU / multi-machine training needs:

* device + precision resolution (:mod:`.hardware`): bf16 autocast on Ampere+/Blackwell (RTX 5090),
  fp16 + ``GradScaler`` on older GPUs, fp32 on CPU; TF32 matmuls; ``torch.compile`` opt-in;
* ``torchrun`` DDP (:mod:`.distributed`): ``DistributedSampler`` + ``set_epoch``, ``no_sync`` during
  gradient accumulation, rank-0-only IO, exact cross-rank metric reduction;
* gradient accumulation (each micro-batch loss is weighted by its share of the group's samples,
  so ``accum × B`` ≡ one ``accum·B`` batch for per-sample-mean losses, also when the last
  micro-batch of an epoch is short), gradient clipping, per-optimizer-step warm-up + cosine LR
  (:mod:`.optim`), EMA weights;
* crash-safe checkpoints (:mod:`.checkpoint`) with full state (model/optimizer/scheduler/scaler/
  EMA/RNG/data position) so a run interrupted on a remote box resumes exactly (``resume="auto"``);
* best-by-monitor checkpoint, early stopping, JSONL/TensorBoard/W&B logging.

Contract for stage code::

    def loss_fn(model, batch) -> dict[str, Tensor]   # must contain scalar "loss"; other scalar
                                                     # entries are averaged + logged
    trainer = Trainer(model, loss_fn, TrainConfig.from_dict(cfg["train"]), train_ds, val_ds)
    history = trainer.fit()

During training ``loss_fn`` receives the (DDP / compiled) wrapper — call ``model(...)``, not
custom methods (use :func:`~robot_skin.train.distributed.unwrap_model` for those); during
evaluation it receives the bare module with EMA weights swapped in.

Run directory (``cfg.out_dir``): ``config.json, env.json, metrics.jsonl, history.json,
ckpt_last.pt, ckpt_best.pt, summary.json`` (``summary.json`` appears only when ``fit`` finished —
use it as the "run complete" marker when syncing results between machines).
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import itertools
import json
import logging
import math
import os
import random
import shutil
import time
import types
import typing
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .checkpoint import (BEST_NAME, LAST_NAME, find_last, load_checkpoint, save_checkpoint,
                         strip_state_dict_prefixes)
from .distributed import (DistInfo, all_ranks_equal, all_reduce_sum, init_distributed,
                          make_eval_sampler, make_sampler, unwrap_model, wrap_ddp)
from .hardware import (_PRECISION_ALIASES, describe_environment, enable_tf32, resolve_device,
                       resolve_precision)
from .logging_utils import JsonlLogger
from .optim import EMA, OPTIMIZERS, SCHEDULES, build_optimizer, build_scheduler

log = logging.getLogger("robot_skin.train")

#: sub-directory of ``out_dir`` that receives an earlier run's files when a fresh run starts there
PREVIOUS_RUN_DIR = "previous"

__all__ =["TrainConfig", "Trainer", "seed_everything", "seed_worker", "move_to_device",
           "maybe_compile"]


# ───────────────────────────────────────────────────────────────────────────── config

@dataclass
class TrainConfig:
    """All knobs of :class:`Trainer`; build from YAML with :meth:`from_dict`.

    Budget: ``max_steps`` (optimizer steps), if set, takes precedence over ``max_epochs``.
    Effective batch = ``batch_size × grad_accum × world_size``. ``warmup_steps`` counts
    optimizer steps. ``monitor`` is a history-row key (``val/<loss key>``, ``train/loss`` or a
    callback metric); ``monitor_mode`` min|max.
    """
    # ── contract fields
    max_epochs: int = 10
    max_steps: int | None = None
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    schedule: str = "cosine"
    min_lr_ratio: float = 0.1
    grad_clip: float | None = 1.0
    grad_accum: int = 1
    precision: str = "auto"
    compile: bool = False
    ema_decay: float | None = None
    num_workers: int = 0
    pin_memory: bool = True
    log_every: int = 50
    eval_every_epochs: int = 1
    ckpt_every_epochs: int = 1
    out_dir: str = "robot_skin/runs/default"
    seed: int = 0
    device: str = "auto"
    deterministic: bool = False
    early_stop_patience: int | None = None
    monitor: str = "val/loss"
    # ── extensions (optional)
    monitor_mode: str = "min"
    early_stop_min_delta: float = 0.0
    optimizer: str = "adamw"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    lr_mult: dict[str, float] | None = None      # {"vision": 0.1} → slower pretrained encoder
    drop_last: bool = False
    eval_batch_size: int | None = None
    compile_mode: str | None = None               # None | "reduce-overhead" | "max-autotune"
    tf32: bool = True
    ema_warmup: bool = True
    ckpt_every_steps: int | None = None           # extra mid-epoch ckpt_last (long remote runs)
    resume: str | None = None                     # None | "auto" | path
    find_unused_parameters: bool = False          # DDP; needed with modality dropout branches
    tensorboard: bool = False
    wandb_project: str | None = None

    # -- construction -------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None, *, strict: bool = False) -> "TrainConfig":
        """Build from a plain dict (YAML ``train:`` section). Unknown keys are ignored with a
        warning (``strict=True`` raises). Values are coerced to the field types, so YAML quirks
        such as ``lr: 3e-4`` (parsed as a *string* by YAML 1.1) or ``betas: [0.9, 0.95]`` work."""
        d = dict(d or {})
        if isinstance(d.get("resume"), bool):  # YAML `resume: true` → "auto"
            d["resume"] = "auto" if d["resume"] else None
        hints = typing.get_type_hints(cls)
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            msg = f"TrainConfig: ignoring unknown keys {unknown}"
            if strict:
                raise KeyError(msg)
            warnings.warn(msg, stacklevel=2)
        kw = {k: _coerce(k, v, hints[k]) for k, v in d.items() if k in names}
        cfg = cls(**kw)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["betas"] = list(self.betas)
        return out

    def validate(self) -> "TrainConfig":
        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"TrainConfig: {msg}")

        need(self.batch_size >= 1, f"batch_size must be >= 1 (got {self.batch_size})")
        need(self.grad_accum >= 1, f"grad_accum must be >= 1 (got {self.grad_accum})")
        need(self.max_steps is None or self.max_steps >= 1, "max_steps must be >= 1 or None")
        need(self.max_steps is not None or self.max_epochs >= 1, "max_epochs must be >= 1")
        need(self.lr > 0, f"lr must be > 0 (got {self.lr})")
        need(self.weight_decay >= 0, "weight_decay must be >= 0")
        need(self.warmup_steps >= 0, "warmup_steps must be >= 0")
        need(self.schedule in SCHEDULES, f"schedule must be one of {SCHEDULES}")
        need(0.0 <= self.min_lr_ratio <= 1.0, "min_lr_ratio must be in [0, 1]")
        need(self.grad_clip is None or self.grad_clip >= 0, "grad_clip must be >= 0 or None")
        need(str(self.precision).lower() in _PRECISION_ALIASES,
             f"precision must be one of auto|bf16|fp16|fp32 (got {self.precision!r})")
        need(self.ema_decay is None or 0.0 < self.ema_decay < 1.0, "ema_decay must be in (0, 1)")
        need(self.num_workers >= 0, "num_workers must be >= 0")
        need(self.log_every >= 1, "log_every must be >= 1")
        need(self.eval_every_epochs >= 0, "eval_every_epochs must be >= 0 (0 = never)")
        need(self.ckpt_every_epochs >= 0, "ckpt_every_epochs must be >= 0 (0 = only at end)")
        need(self.ckpt_every_steps is None or self.ckpt_every_steps >= 1,
             "ckpt_every_steps must be >= 1 or None")
        need(self.early_stop_patience is None or self.early_stop_patience >= 1,
             "early_stop_patience must be >= 1 or None")
        need(self.monitor_mode in ("min", "max"), "monitor_mode must be 'min' or 'max'")
        need(self.optimizer.lower() in OPTIMIZERS, f"optimizer must be one of {OPTIMIZERS}")
        need(len(self.betas) == 2, "betas must have two values")
        need(self.eval_batch_size is None or self.eval_batch_size >= 1, "eval_batch_size >= 1")
        return self


_TRUE = {"true", "yes", "on", "1", "y"}
_FALSE = {"false", "no", "off", "0", "n"}


def _coerce(name: str, value: Any, hint: Any) -> Any:
    """Coerce a config value to its dataclass type hint (float/int/bool/str/tuple/dict/None)."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        allows_none = type(None) in args
        cands = [a for a in args if a is not type(None)]
        if value is None or (allows_none and isinstance(value, str)
                             and value.strip().lower() in ("none", "null", "")):
            if allows_none:
                return None
            raise ValueError(f"TrainConfig.{name} may not be None")
        err: Exception | None = None
        for c in cands:
            try:
                return _coerce(name, value, c)
            except (TypeError, ValueError) as e:
                err = e
        raise ValueError(f"TrainConfig.{name}: cannot interpret {value!r} as {hint}") from err
    if value is None:
        raise ValueError(f"TrainConfig.{name} may not be None")
    try:
        if hint is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
                return value.strip().lower() in _TRUE
            raise ValueError
        if hint is int:
            if isinstance(value, bool):
                raise ValueError
            if isinstance(value, int):
                return value
            f = float(value)
            if not f.is_integer():
                raise ValueError
            return int(f)
        if hint is float:
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        if hint is str:
            if isinstance(value, (str, int, float, os.PathLike)) and not isinstance(value, bool):
                return os.fspath(value) if isinstance(value, os.PathLike) else str(value)
            raise ValueError
        if origin is tuple:
            if not isinstance(value, (list, tuple)):
                raise ValueError
            elem = args[0] if args else (lambda x: x)
            return tuple(_coerce(name, v, elem) for v in value)
        if origin is dict or hint is dict:
            if not isinstance(value, Mapping):
                raise ValueError
            return {str(k): (float(v) if args and args[1] is float else v) for k, v in value.items()}
    except (TypeError, ValueError) as e:
        raise ValueError(f"TrainConfig.{name}: cannot interpret {value!r} as {hint}") from e
    return value


# ───────────────────────────────────────────────────────────────────────────── helpers

def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python / numpy / torch (all devices). ``deterministic=True`` also requests
    deterministic kernels (``warn_only``), disables cuDNN autotuning and sets
    ``CUBLAS_WORKSPACE_CONFIG`` — slower, for debugging / exact reproduction."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)  # also seeds every CUDA device
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.are_deterministic_algorithms_enabled():
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - DataLoader signature
    """``DataLoader(worker_init_fn=...)``: derive numpy / python seeds from the torch worker seed
    (which the loader's seeded generator makes reproducible)."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def move_to_device(batch: Any, device: torch.device | str, non_blocking: bool = False) -> Any:
    """Recursively move tensors in dicts / lists / tuples / namedtuples / dataclasses to
    ``device``; other leaves (strings, numbers, numpy arrays) are returned unchanged."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        moved = {k: move_to_device(v, device, non_blocking) for k, v in batch.items()}
        try:
            return type(batch)(moved)  # type: ignore[call-arg]
        except TypeError:  # e.g. defaultdict, MappingProxyType
            return moved
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):  # namedtuple
        return type(batch)(*(move_to_device(v, device, non_blocking) for v in batch))
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device, non_blocking) for v in batch)
    if dataclasses.is_dataclass(batch) and not isinstance(batch, type):
        return dataclasses.replace(batch, **{
            f.name: move_to_device(getattr(batch, f.name), device, non_blocking)
            for f in fields(batch) if f.init})
    return batch


def _batch_size(batch: Any) -> int:
    """Samples in a (nested) batch: leading dim of the first tensor found depth-first, or the
    length of a list of non-container leaves (e.g. instruction strings); 1 if unknown."""
    n = _find_batch_size(batch)
    return n if n is not None else 1


def _find_batch_size(batch: Any) -> int | None:
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0]) if batch.ndim >= 1 else None
    if isinstance(batch, Mapping):
        items: Sequence[Any] = list(batch.values())
    elif isinstance(batch, (list, tuple)):
        if batch and not any(isinstance(x, (torch.Tensor, Mapping, list, tuple)) for x in batch):
            return len(batch)
        items = batch
    elif dataclasses.is_dataclass(batch) and not isinstance(batch, type):
        items = [getattr(batch, f.name) for f in fields(batch)]
    else:
        return None
    for v in items:
        n = _find_batch_size(v)
        if n is not None:
            return n
    return None


def maybe_compile(module: nn.Module, device: torch.device | str,
                  mode: str | None = None) -> tuple[nn.Module, bool]:
    """``torch.compile`` when the toolchain is present, else return ``module`` unchanged
    (warning). CUDA needs Triton (absent on Windows); CPU needs a C++ compiler; MPS is skipped.
    Returns ``(module, compiled?)``."""
    dt = torch.device(device).type
    reason = None
    if not hasattr(torch, "compile"):
        reason = "torch.compile unavailable (torch < 2.0)"
    elif dt == "cuda" and importlib.util.find_spec("triton") is None:
        reason = "Triton is not installed (e.g. Windows) — required by the inductor backend"
    elif dt == "cpu":
        cxx = os.environ.get("CXX")
        if not ((cxx and shutil.which(cxx)) or any(shutil.which(c) for c in ("c++", "g++", "clang++"))):
            reason = "no C++ compiler found for the CPU inductor backend"
    elif dt not in ("cuda", "cpu"):
        reason = f"torch.compile not enabled for device type {dt!r}"
    if reason is not None:
        warnings.warn(f"compile=True ignored: {reason}", stacklevel=2)
        return module, False
    try:
        return torch.compile(module, mode=mode), True
    except Exception as e:  # pragma: no cover - toolchain specific
        warnings.warn(f"torch.compile failed ({e}); running eagerly", stacklevel=2)
        return module, False


class _ResumableSampler(Sampler[int]):
    """Wraps the train sampler; ``skip`` drops the first N indices of the *next* iteration only
    (exact mid-epoch resume without loading the skipped batches)."""

    def __init__(self, base: Sampler):
        self.base = base
        self.skip = 0

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __iter__(self) -> Iterator[int]:
        skip, self.skip = self.skip, 0
        return itertools.islice(iter(self.base), skip, None)

    def __len__(self) -> int:
        return max(0, len(self.base) - self.skip)  # type: ignore[arg-type]


def _safe_len(obj: Any) -> int | None:
    try:
        return len(obj)
    except TypeError:
        return None


def _accumulate(sums: dict[str, Any], out: Mapping[str, Any], n: int) -> None:
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            if v.numel() != 1:
                continue
            v = v.detach().float().reshape(())
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            v = float(v)
        else:
            continue
        sums[k] = sums.get(k, 0.0) + v * n


def _rng_state(gen: torch.Generator) -> dict:
    st: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch": torch.get_rng_state(), "loader": gen.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: Mapping[str, Any], gen: torch.Generator) -> None:
    try:
        random.setstate(st["python"])
        np.random.set_state(st["numpy"])
        torch.set_rng_state(st["torch"].cpu())
        gen.set_state(st["loader"].cpu())
        if "cuda" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])
    except Exception as e:  # different device count etc. — not fatal
        warnings.warn(f"could not restore RNG state ({e})", stacklevel=2)


# ───────────────────────────────────────────────────────────────────────────── trainer

LossFn = Callable[[nn.Module, Any], "Mapping[str, torch.Tensor] | torch.Tensor"]
Callback = Callable[["Trainer", dict], "Mapping[str, Any] | None"]


class Trainer:
    """Epoch/step training loop (see module docstring).

    Args:
        model: bare ``nn.Module`` (moved to the resolved device; wrapped for DDP/compile here).
        loss_fn: ``loss_fn(model, batch) -> {"loss": scalar, **scalars}`` (a bare tensor is taken
            as the loss).
        cfg: :class:`TrainConfig` or dict (``TrainConfig.from_dict``).
        train_data / val_data: map-style ``Dataset`` (loader + sampler built here), an
            ``IterableDataset`` (needs ``max_steps`` for decaying schedules) or a ready
            ``DataLoader`` (used as-is).
        collate_fn: passed to the DataLoaders built here.
        extra_state: dict (or zero-arg callable returning one) stored as ``ckpt["extra"]`` —
            normalisers, model config, anything inference needs. Restored on resume only when
            none is given (the caller's freshly computed state wins).
        optimizer: optional ``Optimizer`` or ``callable(model) -> Optimizer`` replacing
            :func:`~robot_skin.train.optim.build_optimizer`.
        callbacks: ``cb(trainer, row) -> dict | None`` called at each epoch end (after
            evaluation); returned metrics are merged into the history row (usable as
            ``monitor``). Set ``trainer.should_stop = True`` to stop after this epoch.
        dist_info: override :func:`init_distributed` (tests).
    """

    def __init__(self, model: nn.Module, loss_fn: LossFn,
                 cfg: TrainConfig | Mapping[str, Any] | None = None,
                 train_data: Dataset | DataLoader | None = None,
                 val_data: Dataset | DataLoader | None = None,
                 collate_fn: Callable | None = None,
                 extra_state: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
                 *, optimizer: torch.optim.Optimizer | Callable[[nn.Module], torch.optim.Optimizer] | None = None,
                 callbacks: Sequence[Callback] = (),
                 dist_info: DistInfo | None = None):
        c = cfg if isinstance(cfg, TrainConfig) else TrainConfig.from_dict(cfg or {})
        c.validate()
        self.cfg = c
        self.loss_fn = loss_fn
        self.collate_fn = collate_fn
        self.extra_state = extra_state
        self.callbacks = list(callbacks)
        self.dist = dist_info if dist_info is not None else init_distributed()
        seed_everything(c.seed + self.dist.rank, c.deterministic)

        self.device = resolve_device(c.device,
                                     local_rank=self.dist.local_rank if self.dist.distributed else None)
        # process-global: set it for *this* run either way — an earlier Trainer in the same process
        # (pipeline, sweep) may have enabled TF32, and `tf32: false` / a CPU run must not inherit it
        enable_tf32(bool(c.tf32 and self.device.type == "cuda"))
        self.precision = resolve_precision(c.precision, self.device)
        self._non_blocking = self.device.type == "cuda" and c.pin_memory
        self.out_dir = Path(c.out_dir)
        self.model = unwrap_model(model).to(self.device)

        # data
        self._loader_gen = torch.Generator()
        self._loader_gen.manual_seed(c.seed + self.dist.rank)
        self._train_sampler: _ResumableSampler | None = None
        self._own_train_loader = False
        self.train_loader = self._build_train_loader(train_data) if train_data is not None else None
        self.val_loader = self._build_eval_loader(val_data) if val_data is not None else None
        self.batches_per_epoch = _safe_len(self.train_loader) if self.train_loader is not None else None
        if self.batches_per_epoch == 0:
            raise ValueError("training data yields no batches (dataset smaller than batch_size "
                             "with drop_last=True, or empty dataset)")
        self.steps_per_epoch = (math.ceil(self.batches_per_epoch / c.grad_accum)
                                if self.batches_per_epoch else None)
        if c.max_steps is not None:
            self.total_steps: int | None = c.max_steps
        elif self.steps_per_epoch is not None:
            self.total_steps = c.max_epochs * self.steps_per_epoch
        else:
            self.total_steps = None

        # optimisation
        self._own_optimizer = optimizer is None   # resume re-applies the config's lr / weight decay
        if optimizer is None:
            self.optimizer = build_optimizer(self.model, c.optimizer, c.lr, c.weight_decay,
                                             c.betas, eps=c.eps, lr_mult=c.lr_mult)
        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer(self.model)
        self.scheduler = None
        if self.train_loader is not None or c.schedule == "constant":
            if c.schedule != "constant" and self.total_steps is None:
                raise ValueError(f"schedule {c.schedule!r} needs a known length: set max_steps "
                                 "(iterable dataset) or use schedule: constant")
            if self.total_steps is not None and c.warmup_steps >= self.total_steps:
                warnings.warn(f"warmup_steps={c.warmup_steps} >= total optimizer steps "
                              f"{self.total_steps}; the LR never leaves warm-up", stacklevel=2)
            self.scheduler = build_scheduler(self.optimizer, c.schedule, c.warmup_steps,
                                             self.total_steps, c.min_lr_ratio)
        self.scaler = self.precision.make_scaler()

        # wrappers: DDP first, then compile (recommended order for DDP + torch.compile)
        self._ddp = (wrap_ddp(self.model, self.dist, find_unused_parameters=c.find_unused_parameters)
                     if self.dist.distributed else None)
        m: nn.Module = self._ddp if self._ddp is not None else self.model
        self.compiled = False
        if c.compile:
            m, self.compiled = maybe_compile(m, self.device, c.compile_mode)
        self.train_model = m
        # EMA only after DDP has broadcast rank 0's weights — otherwise every rank's shadow keeps
        # a trace of its own (different) initial weights
        self.ema = EMA(self.model, c.ema_decay, warmup=c.ema_warmup) if c.ema_decay else None

        # state
        self.step = 0                 # optimizer steps taken
        self.epoch = 0                # fully completed epochs
        self.batch_in_epoch = 0       # micro-batches consumed in the current (partial) epoch
        self.history: list[dict] = []
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.best_step: int | None = None
        self.bad_epochs = 0
        self.should_stop = False
        self.stopped_early = False
        self._finished_early = False  # resumed a run that had already stopped early → nothing to do
        self.resumed_from: Path | None = None
        self.logger: JsonlLogger | None = None
        self._monitor_used: str | None = None
        self._warned: set[str] = set()

    # ── data ──────────────────────────────────────────────────────────────────────────────
    def _loader_kwargs(self) -> dict:
        c = self.cfg
        kw: dict[str, Any] = dict(collate_fn=self.collate_fn, num_workers=c.num_workers,
                                  pin_memory=bool(c.pin_memory and self.device.type == "cuda"),
                                  generator=self._loader_gen)
        if c.num_workers > 0:
            kw.update(worker_init_fn=seed_worker, persistent_workers=True)
        return kw

    def _build_train_loader(self, data: Dataset | DataLoader) -> DataLoader:
        if isinstance(data, DataLoader):
            if self.dist.distributed and not hasattr(data.sampler, "set_epoch"):
                warnings.warn("DDP with a user DataLoader without DistributedSampler: every rank "
                              "sees the full dataset", stacklevel=3)
            return data
        base = make_sampler(data, self.dist, shuffle=True, seed=self.cfg.seed)
        self._train_sampler = _ResumableSampler(base) if base is not None else None
        self._own_train_loader = True
        return DataLoader(data, batch_size=self.cfg.batch_size, sampler=self._train_sampler,
                          drop_last=self.cfg.drop_last, **self._loader_kwargs())

    def _build_eval_loader(self, data: Dataset | DataLoader) -> DataLoader:
        if isinstance(data, DataLoader):
            return data
        n = _safe_len(data)
        if self.dist.distributed and n is not None and n < self.dist.world_size:
            raise ValueError(f"validation set ({n}) smaller than world_size "
                             f"({self.dist.world_size})")
        return DataLoader(data, batch_size=self.cfg.eval_batch_size or self.cfg.batch_size,
                          sampler=make_eval_sampler(data, self.dist), drop_last=False,
                          **self._loader_kwargs())

    def _micro_batch_sizes(self) -> list[int] | None:
        """Per-rank sample count of every micro-batch of a full epoch — known exactly for loaders
        built here (map-style dataset + our sampler); ``None`` otherwise (user ``DataLoader`` /
        ``IterableDataset``: micro-batches are then weighted equally within a group)."""
        if not self._own_train_loader or self._train_sampler is None:
            return None
        n = _safe_len(self._train_sampler.base)
        if n is None:
            return None
        b = self.cfg.batch_size
        if self.cfg.drop_last:
            return [b] * (n // b)
        return [min(b, n - i * b) for i in range(math.ceil(n / b))]

    @torch.no_grad()
    def _scale_grads(self, factor: float) -> None:
        for p in self._params():
            if p.grad is not None:
                p.grad.mul_(factor)

    @torch.no_grad()
    def _all_reduce_grads(self) -> None:
        """Average the gradients over the DDP ranks outside DDP's hooks (what its backward
        all-reduce does): params without a gradient on some rank contribute zeros; params without
        one on every rank keep ``grad=None``. Same parameter order on every rank."""
        import torch.distributed as tdist

        group = getattr(self._ddp, "process_group", None)
        world = tdist.get_world_size(group)
        params = [p for p in self._params() if p.requires_grad]
        has = torch.tensor([float(p.grad is not None) for p in params], device=self.device)
        tdist.all_reduce(has, group=group)
        for p, n in zip(params, has.tolist()):
            if n == 0:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            tdist.all_reduce(p.grad, group=group)
            p.grad.div_(world)

    def _zero_grad(self) -> None:
        """Clear optimizer *and* model grads (params excluded via ``lr_mult: 0`` still receive
        gradients; without this they would accumulate forever)."""
        self.optimizer.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)

    def _set_epoch(self, epoch: int) -> None:
        sampler = self._train_sampler if self._own_train_loader else getattr(
            self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    # ── core steps ────────────────────────────────────────────────────────────────────────
    def _call_loss(self, model: nn.Module, batch: Any, train: bool) -> Mapping[str, Any]:
        out = self.loss_fn(model, batch)
        if isinstance(out, torch.Tensor):
            out = {"loss": out}
        if not isinstance(out, Mapping) or "loss" not in out:
            raise KeyError("loss_fn must return a Tensor or a dict with key 'loss' "
                           f"(got {type(out).__name__} with keys "
                           f"{list(out) if isinstance(out, Mapping) else None})")
        loss = out["loss"]
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError("loss_fn()['loss'] must be a scalar tensor")
        if train and not loss.requires_grad:
            raise RuntimeError("loss does not require grad — was it detached or computed under "
                               "torch.no_grad()?")
        return out

    def _params(self) -> list[torch.Tensor]:
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    def _optimizer_step(self) -> torch.Tensor | None:
        c = self.cfg
        grad_norm = None
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if c.grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self._params(), c.grad_clip)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self._zero_grad()
        if self.scheduler is not None:
            with warnings.catch_warnings():  # GradScaler may skip optimizer.step() on inf grads
                warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
                self.scheduler.step()
        self.step += 1
        if self.ema is not None:
            self.ema.update(self.model)
        return grad_norm

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _train_epoch(self, epoch: int) -> tuple[dict[str, float], bool]:
        """One (possibly resumed / truncated) epoch. Returns (mean train metrics, completed)."""
        c = self.cfg
        accum = c.grad_accum
        skip = self.batch_in_epoch
        offset = 0
        if skip and self._train_sampler is not None:
            self._train_sampler.skip = skip * c.batch_size  # skip at index level
            offset = skip
        n_rem = _safe_len(self.train_loader)
        n_total = (offset + n_rem) if n_rem is not None else None
        sizes = self._micro_batch_sizes()
        if sizes is not None and len(sizes) != n_total:  # defensive: fall back to equal weights
            sizes = None

        self.train_model.train()
        sums: dict[str, Any] = {}
        count = 0
        win: dict[str, Any] = {}
        win_n, win_t0 = 0, time.perf_counter()
        grad_norm = None
        completed = True
        last_abs = offset - 1
        pending = 0  # micro-batches backpropagated since the last optimizer step
        for i, batch in enumerate(self.train_loader):
            abs_i = offset + i
            if abs_i < skip:
                continue  # user-provided loader: skip by iteration
            last_abs = abs_i
            g0 = (abs_i // accum) * accum
            gsize = min(accum, n_total - g0) if n_total is not None else accum
            is_last = abs_i + 1 - g0 == gsize
            # weight = this micro-batch's share of the group's samples (exact large-batch mean);
            # equal shares when sizes are unknown
            weight = (sizes[abs_i] / sum(sizes[g0:g0 + gsize]) if sizes is not None
                      else 1.0 / gsize)
            batch = move_to_device(batch, self.device, non_blocking=self._non_blocking)
            bs = _batch_size(batch)
            sync = (self._ddp.no_sync() if (self._ddp is not None and not is_last)
                    else contextlib.nullcontext())
            with sync:
                with self.precision.autocast():
                    out = self._call_loss(self.train_model, batch, train=True)
                scaled = out["loss"].float() * weight
                if self.scaler is not None:
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()
            pending += 1
            _accumulate(sums, out, bs)
            _accumulate(win, out, bs)
            count += bs
            win_n += bs
            if not is_last:
                continue
            grad_norm = self._optimizer_step()
            pending = 0
            self.batch_in_epoch = abs_i + 1
            if self.step % c.log_every == 0:
                self._log_window(win, win_n, time.perf_counter() - win_t0, grad_norm, epoch)
                win, win_n, win_t0 = {}, 0, time.perf_counter()
            if c.ckpt_every_steps and self.step % c.ckpt_every_steps == 0 and self.dist.is_main:
                self.save_checkpoint(self.out_dir / LAST_NAME)
            if (c.max_steps is not None and self.step >= c.max_steps) or self.should_stop:
                completed = n_total is not None and abs_i + 1 >= n_total
                break
        else:
            if pending:  # unknown-length loader ended mid accumulation group
                if self._ddp is not None:
                    # every micro-batch of this group ran under no_sync (gsize = accum) → no
                    # all-reduce happened; average by hand or each replica steps with its own
                    # local gradient and the replicas drift apart (an IterableDataset must still
                    # yield the same batch count on every rank)
                    self._all_reduce_grads()
                # each micro-batch was weighted 1/accum → renormalise to the group actually seen
                if pending < accum:
                    self._scale_grads(accum / pending)
                grad_norm = self._optimizer_step()
                self.batch_in_epoch = last_abs + 1
        if completed:
            self.epoch += 1
            self.batch_in_epoch = 0
        tot = {k: float(v) for k, v in sums.items()}
        tot["__count__"] = float(count)
        tot = all_reduce_sum(tot)
        n = tot.pop("__count__")
        metrics = {k: v / n for k, v in tot.items()} if n > 0 else {}
        if "loss" in metrics and not math.isfinite(metrics["loss"]):
            raise FloatingPointError(
                f"non-finite train loss at epoch {epoch + 1} (step {self.step}); try a lower lr, "
                "precision bf16/fp32 instead of fp16, grad_clip, or check input normalisation")
        return metrics, completed

    def _log_window(self, win: Mapping[str, Any], n: int, dt: float, grad_norm: Any,
                    epoch: int) -> None:
        vals = {f"train/{k}": float(v) / max(n, 1) for k, v in win.items()}
        finite = all(math.isfinite(v) for v in vals.values())
        if self.dist.distributed:
            finite = all_reduce_sum({"bad": 0.0 if finite else 1.0})["bad"] == 0.0
        if not finite:
            raise FloatingPointError(
                f"non-finite training metrics at step {self.step}: {vals}; try a lower lr, bf16 "
                "instead of fp16, grad_clip, or check input normalisation")
        vals["lr"] = self.current_lr()
        vals["epoch"] = epoch + 1
        if grad_norm is not None:
            vals["train/grad_norm"] = float(grad_norm)
        vals["sys/samples_per_s"] = n * self.dist.world_size / max(dt, 1e-9)
        if self.device.type == "cuda":
            vals["sys/gpu_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
        if self.scaler is not None:
            vals["sys/grad_scale"] = float(self.scaler.get_scale())
        self._log(self.step, vals, kind="step")
        if self.dist.is_main:
            log.info("step %d  loss %.5g  lr %.3g", self.step, vals.get("train/loss", float("nan")),
                     vals["lr"])

    # ── evaluation ────────────────────────────────────────────────────────────────────────
    @contextlib.contextmanager
    def eval_weights(self, use_ema: bool = True) -> Iterator[nn.Module]:
        """Bare model in eval mode, with EMA weights swapped in if enabled (restored after)."""
        was_training = self.model.training
        self.model.eval()
        try:
            if use_ema and self.ema is not None:
                with self.ema.swap(self.model):
                    yield self.model
            else:
                yield self.model
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def evaluate(self, data: Dataset | DataLoader | None = None, *, use_ema: bool = True,
                 prefix: str = "val/") -> dict[str, float]:
        """Sample-weighted mean of every scalar returned by ``loss_fn`` over ``data`` (default:
        the validation set), exact across DDP ranks; keys are prefixed (``val/loss``). Uses EMA
        weights when EMA is enabled and ``use_ema``."""
        loader = self.val_loader if data is None else self._build_eval_loader(data)
        if loader is None:
            return {}
        sums: dict[str, Any] = {}
        count = 0
        with self.eval_weights(use_ema) as model:
            for batch in loader:
                batch = move_to_device(batch, self.device, non_blocking=self._non_blocking)
                bs = _batch_size(batch)
                with self.precision.autocast():
                    out = self._call_loss(model, batch, train=False)
                _accumulate(sums, out, bs)
                count += bs
        tot = {k: float(v) for k, v in sums.items()}
        tot["__count__"] = float(count)
        tot = all_reduce_sum(tot)
        n = tot.pop("__count__")
        if n <= 0:
            return {}
        return {f"{prefix}{k}": v / n for k, v in tot.items()}

    # ── monitor / early stopping ──────────────────────────────────────────────────────────
    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(msg, stacklevel=3)

    def _monitor_key(self, row: Mapping[str, Any] | None = None) -> str:
        """``cfg.monitor``; falls back to ``train/<name>`` when a ``val/<name>`` monitor has no
        validation behind it (no val data, or ``eval_every_epochs: 0``; warning once)."""
        m = self.cfg.monitor
        no_val = self.val_loader is None or self.cfg.eval_every_epochs == 0
        if (m.startswith("val/") and no_val
                and (row is None or m not in row)):
            alt = "train/" + m[len("val/"):]
            if row is None or alt in row:
                self._warn_once("monitor", f"monitor {m!r} but no validation data (or "
                                f"eval_every_epochs: 0); using {alt!r}")
                return alt
        return m

    def _update_monitor(self, row: dict) -> bool:
        key = self._monitor_key(row)
        if key not in row:
            if any(k.startswith("val/") for k in row) or not key.startswith("val/"):
                self._warn_once("monitor-missing", f"monitor {key!r} missing from epoch "
                                f"{row.get('epoch')} metrics {sorted(row)}; epochs without it are "
                                "skipped for best-checkpoint / early-stopping")
            return False
        v = float(row[key])
        c = self.cfg
        if math.isfinite(v) and (
                self.best_value is None
                or (c.monitor_mode == "min" and v < self.best_value - c.early_stop_min_delta)
                or (c.monitor_mode == "max" and v > self.best_value + c.early_stop_min_delta)):
            self.best_value, self.best_epoch, self.best_step = v, int(row["epoch"]), self.step
            self.bad_epochs = 0
            self._monitor_used = key
            return True
        self._monitor_used = key
        self.bad_epochs += 1
        if c.early_stop_patience is not None and self.bad_epochs >= c.early_stop_patience:
            self.should_stop = True
            self.stopped_early = True
            if self.dist.is_main:
                log.info("early stopping at epoch %d (best %s=%.6g at epoch %s)", row["epoch"],
                         key, self.best_value if self.best_value is not None else float("nan"),
                         self.best_epoch)
        return False

    def _budget_done(self) -> bool:
        c = self.cfg
        if self._finished_early:
            return True
        if c.max_steps is not None:
            return self.step >= c.max_steps
        return self.epoch >= c.max_epochs

    # ── fit ───────────────────────────────────────────────────────────────────────────────
    def _log(self, step: int, metrics: Mapping[str, Any], kind: str) -> None:
        if self.logger is not None:
            self.logger.log(step, metrics, kind=kind)

    def _write_json(self, name: str, obj: Any) -> None:
        if not self.dist.is_main:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.out_dir / f".{name}.tmp"
        tmp.write_text(json.dumps(obj, indent=2, default=str))
        os.replace(tmp, self.out_dir / name)

    def _set_aside_previous_run(self) -> None:
        """A fresh (not resumed) run into an ``out_dir`` holding another run's files: move that
        run's checkpoints, ``metrics.jsonl`` and ``history.json`` to ``out_dir/previous/``
        (replacing an older set). Otherwise a stale ``ckpt_best.pt`` could be exported as this
        run's model (when this run never records a finite monitor value) and ``metrics.jsonl``
        would interleave two runs."""
        names = (LAST_NAME, BEST_NAME, "metrics.jsonl", "history.json")
        stale = [self.out_dir / n for n in names if (self.out_dir / n).is_file()]
        if not stale:
            return
        prev = self.out_dir / PREVIOUS_RUN_DIR
        shutil.rmtree(prev, ignore_errors=True)
        prev.mkdir(parents=True, exist_ok=True)
        for f in stale:
            os.replace(f, prev / f.name)
        log.warning("fresh run (no resume) in %s: moved the previous run's %s to %s/", self.out_dir,
                    ", ".join(f.name for f in stale), PREVIOUS_RUN_DIR)

    def _open_run(self) -> None:
        c = self.cfg
        if (self.dist.is_main and self.resumed_from is None and self.step == 0
                and not self.history):
            self._set_aside_previous_run()
        if self.logger is None:
            self.logger = JsonlLogger(
                self.out_dir, enabled=self.dist.is_main, tensorboard=c.tensorboard,
                wandb=({"project": c.wandb_project, "config": c.to_dict()}
                       if c.wandb_project else None))
        if self.dist.is_main:
            # summary.json marks a *finished* run (used when syncing results between machines);
            # a resumed / restarted run is unfinished until fit() returns again
            (self.out_dir / "summary.json").unlink(missing_ok=True)
            self._write_json("config.json", {**c.to_dict(), "world_size": self.dist.world_size,
                                             "effective_batch_size": c.batch_size * c.grad_accum
                                             * self.dist.world_size})
            try:
                env = describe_environment()
            except Exception as e:  # pragma: no cover - diagnostics must never kill training
                env = {"error": str(e)}
            env.update(device=str(self.device), precision=self.precision.name,
                       compiled=self.compiled)
            self._write_json("env.json", env)
            for w in env.get("warnings", []) or []:
                warnings.warn(w, stacklevel=3)

    def fit(self) -> list[dict]:
        """Train until the budget (``max_steps`` or ``max_epochs``), early stopping or
        ``should_stop``; returns the per-epoch history (list of dicts)."""
        if self.train_loader is None:
            raise ValueError("Trainer.fit() needs train_data")
        c = self.cfg
        if c.resume and self.resumed_from is None:
            self.resume(c.resume)
        self._open_run()
        self.should_stop = False
        # drop stale grads of a partial accumulation group (fit() re-entered after an interrupt)
        self._zero_grad()
        t_fit = time.perf_counter()
        try:
            while not self._budget_done():
                e = self.epoch
                self._set_epoch(e)
                t0 = time.perf_counter()
                train_m, completed = self._train_epoch(e)
                row: dict[str, Any] = {"epoch": e + 1, "step": self.step, "lr": self.current_lr()}
                if not completed:
                    row["partial"] = True
                row.update({f"train/{k}": v for k, v in train_m.items()})
                last_epoch = self._budget_done() or self.should_stop
                if self.val_loader is not None and c.eval_every_epochs > 0 and (
                        (e + 1) % c.eval_every_epochs == 0 or last_epoch):
                    row.update(self.evaluate())
                for cb in self.callbacks:
                    extra = cb(self, row)
                    if extra:
                        row.update(extra)
                row["time_s"] = time.perf_counter() - t0
                improved = self._update_monitor(row)
                self.history.append(row)
                self._log(self.step, row, kind="epoch")
                if self.dist.is_main:
                    log.info("epoch %d  %s", row["epoch"], "  ".join(
                        f"{k} {v:.5g}" for k, v in row.items()
                        if k.startswith(("train/", "val/")) and isinstance(v, float)))
                    if improved:
                        self.save_checkpoint(self.out_dir / BEST_NAME)
                    final = self._budget_done() or self.should_stop
                    if final or (c.ckpt_every_epochs and (e + 1) % c.ckpt_every_epochs == 0):
                        self.save_checkpoint(self.out_dir / LAST_NAME)
                    self._write_json("history.json", self.history)
                if self.should_stop:
                    break
        except KeyboardInterrupt:
            if self.dist.is_main:
                p = self.save_checkpoint(self.out_dir / LAST_NAME)
                log.warning("interrupted at step %d — saved %s (resume with resume='auto')",
                            self.step, p)
            raise
        finally:
            if self.logger is not None:
                self.logger.close()
                self.logger = None
        self._write_json("summary.json", {
            "finished": True, "stopped_early": self.stopped_early, "step": self.step,
            "epoch": self.epoch, "monitor": self._monitor_used or self.cfg.monitor,
            "best": {"value": self.best_value, "epoch": self.best_epoch, "step": self.best_step},
            "time_s": time.perf_counter() - t_fit, "world_size": self.dist.world_size,
            "device": str(self.device), "precision": self.precision.name})
        return self.history

    # ── checkpointing ─────────────────────────────────────────────────────────────────────
    def _extra(self) -> Any:
        return self.extra_state() if callable(self.extra_state) else self.extra_state

    def state_dict(self) -> dict:
        """Full training state (the checkpoint payload)."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "step": self.step,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "config": self.cfg.to_dict(),
            "extra": self._extra(),
            "best": {"value": self.best_value, "epoch": self.best_epoch, "step": self.best_step,
                     "monitor": self._monitor_used or self.cfg.monitor},
            "bad_epochs": self.bad_epochs,
            "stopped_early": self.stopped_early,
            "history": [dict(r) for r in self.history],
            "data": self._data_fingerprint(),
            "rng": _rng_state(self._loader_gen),
        }

    def _data_fingerprint(self) -> dict:
        """What the run trains on, stored in the checkpoint: a resume onto another training set
        (other split / processed root) warns instead of silently continuing the old model."""
        return {"train_samples": _safe_len(getattr(self.train_loader, "dataset", None)),
                "batches_per_epoch": self.batches_per_epoch, "world_size": self.dist.world_size}

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Atomically write :meth:`state_dict` (default ``out_dir/ckpt_last.pt``)."""
        return save_checkpoint(path if path is not None else self.out_dir / LAST_NAME,
                               **self.state_dict())

    def load_state_dict(self, ckpt: Mapping[str, Any]) -> None:
        """Restore everything saved by :meth:`state_dict`.

        With the optimizer built here (no ``optimizer=`` argument) the *current* config's ``lr``
        (× ``lr_mult``), ``weight_decay``, ``betas`` / ``eps`` are re-applied over the restored
        optimizer state, and the LR at the restored step is recomputed from them (a run can be
        continued with a lower ``lr``); the moments / momentum buffers are kept. A user-supplied
        optimizer keeps the checkpoint's hyper-parameters (plain PyTorch semantics)."""
        self.model.load_state_dict(strip_state_dict_prefixes(ckpt["model"]))
        fresh = ([{k: v for k, v in g.items() if k != "params"} for g in self.optimizer.param_groups]
                 if self._own_optimizer and ckpt.get("optimizer") is not None else None)
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if fresh is not None:
            self._reapply_optimizer_config(fresh)
        if ckpt.get("scaler") is not None:
            if self.scaler is not None:
                self.scaler.load_state_dict(ckpt["scaler"])
            else:
                warnings.warn("checkpoint has GradScaler state but current precision "
                              f"{self.precision.name} uses none; ignored", stacklevel=2)
        if ckpt.get("ema") is not None and self.ema is not None:
            self.ema.load_state_dict(ckpt["ema"])
        elif self.ema is not None:
            warnings.warn("EMA enabled but checkpoint has no EMA state; EMA restarts from the "
                          "loaded weights", stacklevel=2)
            self.ema = EMA(self.model, self.cfg.ema_decay, warmup=self.cfg.ema_warmup)
        self.step = int(ckpt.get("step", 0))
        self.epoch = int(ckpt.get("epoch", 0))
        self.batch_in_epoch = int(ckpt.get("batch_in_epoch", 0))
        self.history = [dict(r) for r in ckpt.get("history", [])]
        best = ckpt.get("best") or {}
        self.best_value, self.best_epoch, self.best_step = (best.get("value"), best.get("epoch"),
                                                            best.get("step"))
        self.bad_epochs = int(ckpt.get("bad_epochs", 0))
        if ckpt.get("extra") is not None and self.extra_state is None:
            self.extra_state = ckpt["extra"]           # the caller's own extra_state wins
        old = ckpt.get("config") or {}
        changed = [k for k in ("batch_size", "grad_accum", "seed")
                   if k in old and old[k] != getattr(self.cfg, k)]
        if changed:
            warnings.warn(f"resuming with changed {changed}: data order / effective batch differ "
                          "from the original run", stacklevel=2)
        old_n = (ckpt.get("data") or {}).get("train_samples")
        new_n = self._data_fingerprint()["train_samples"]
        if old_n is not None and new_n is not None and old_n != new_n:
            warnings.warn(f"resuming on a different training set ({old_n} → {new_n} samples): the "
                          "checkpoint was trained on other data (another split / processed root?) "
                          "— set train.resume: null to train from scratch", stacklevel=2)
        self.stopped_early = bool(ckpt.get("stopped_early", False))
        self._finished_early = False
        if self.stopped_early:
            if self._early_stop_relaxed(old):
                self.stopped_early = False             # patience / budget raised: keep training
            else:
                self._finished_early = True            # already done: fit() takes no step
        if ckpt.get("rng") is not None and not self.dist.distributed:
            _set_rng_state(ckpt["rng"], self._loader_gen)
        elif self.dist.distributed:  # keep per-rank streams distinct after resume
            seed_everything(self.cfg.seed + self.dist.rank + 1000003 * self.step,
                            self.cfg.deterministic)

    #: optimizer-group keys reported when a resume re-applies a changed config value
    _HPARAM_KEYS = ("initial_lr", "weight_decay", "betas", "eps", "momentum", "lr_mult")

    def _reapply_optimizer_config(self, fresh: Sequence[Mapping[str, Any]]) -> None:
        """After loading optimizer / scheduler state: put back the hyper-parameters the current
        config built (``fresh``: the groups before loading) and recompute the LR at the restored
        step from them — the checkpoint's lr / weight decay would otherwise silently win."""
        groups = self.optimizer.param_groups
        if len(groups) != len(fresh):  # pragma: no cover - load_state_dict already checks this
            return
        changed = sorted({("lr" if k == "initial_lr" else k) for g, f in zip(groups, fresh)
                          for k in self._HPARAM_KEYS if k in f and k in g and g[k] != f[k]})
        for g, f in zip(groups, fresh):
            g.update({k: v for k, v in f.items() if k != "lr"})
        if self.scheduler is not None:
            sch = self.scheduler
            sch.base_lrs = [g.get("initial_lr", f["lr"]) for g, f in zip(groups, fresh)]
            lams = getattr(sch, "lr_lambdas", None)
            if lams is not None and len(lams) == len(groups):
                lrs = [b * lam(sch.last_epoch) for b, lam in zip(sch.base_lrs, lams)]
            else:  # pragma: no cover - build_scheduler always returns a LambdaLR
                lrs = list(sch.base_lrs)
            for g, lr in zip(groups, lrs):
                g["lr"] = lr
            sch._last_lr = list(lrs)
        else:
            for g, f in zip(groups, fresh):
                g["lr"] = f["lr"]
        if changed:
            warnings.warn(f"resuming with changed optimizer settings {changed}: the current config's "
                          "values are used (optimizer moments are kept)", stacklevel=3)

    def _early_stop_relaxed(self, old: Mapping[str, Any]) -> bool:
        """Whether the current config lets a run that stopped early keep training: early stopping
        disabled or its patience raised, or a larger budget than the checkpoint's."""
        c = self.cfg
        op = old.get("early_stop_patience")
        if c.early_stop_patience is None or (op is not None and c.early_stop_patience > op):
            return True
        if c.max_steps is not None:
            om = old.get("max_steps")
            return om is None or c.max_steps > int(om)
        if old.get("max_steps") is not None:
            return True
        return c.max_epochs > int(old.get("max_epochs", c.max_epochs))

    def resume(self, path: str | Path = "auto") -> Path | None:
        """Load a checkpoint (``"auto"`` → :func:`find_last` in ``out_dir``; returns ``None`` if
        there is nothing to resume). Returns the path loaded.

        Under DDP this is collective (``fit()`` calls it on every rank): the ranks must agree on
        whether a checkpoint exists and on its step / epoch — otherwise (e.g. node-local
        ``out_dir``s where only rank 0's node holds the checkpoint) the replicas would resume from
        different states, so a ``RuntimeError`` is raised on every rank. Multi-node resume needs a
        shared ``out_dir`` or the checkpoint copied to every node."""
        if str(path) == "auto":
            p = find_last(self.out_dir)
        else:
            p = Path(path)
        found = p is not None and p.exists()
        self._check_ranks_agree("whether a checkpoint to resume exists", [float(found)], p)
        if not found:
            if str(path) != "auto":
                raise FileNotFoundError(f"checkpoint not found: {p}")
            log.info("resume='auto': no checkpoint in %s — starting fresh", self.out_dir)
            return None
        ckpt = load_checkpoint(p, map_location="cpu")
        self.load_state_dict(ckpt)
        self._check_ranks_agree("the resume point (step, epoch, batch_in_epoch)",
                                [self.step, self.epoch, self.batch_in_epoch], p)
        self.resumed_from = p
        if self.dist.is_main:
            log.info("resumed from %s (epoch %d, step %d)", p, self.epoch, self.step)
        return p

    def _check_ranks_agree(self, what: str, values: Sequence[float], path: Path | None) -> None:
        if not self.dist.distributed or all_ranks_equal(values):
            return
        raise RuntimeError(
            f"DDP resume: the ranks disagree on {what} (rank {self.dist.rank}: {list(values)}, "
            f"checkpoint {path}); every rank must see the same checkpoint — use an out_dir shared by "
            "all nodes or copy the checkpoint to every node's out_dir (only rank 0 writes checkpoints)")

    @staticmethod
    def load_model_weights(model: nn.Module, path: str | Path, use_ema: bool = True,
                           strict: bool = True, map_location: Any = "cpu") -> nn.Module:
        """Load weights for inference: the EMA weights when present and ``use_ema`` (else the raw
        weights) of a Trainer checkpoint, or a plain ``state_dict`` file."""
        ckpt = load_checkpoint(path, map_location=map_location)
        if use_ema and ckpt.get("ema"):
            state = ckpt["ema"]["shadow"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
        unwrap_model(model).load_state_dict(strip_state_dict_prefixes(state), strict=strict)
        return model
