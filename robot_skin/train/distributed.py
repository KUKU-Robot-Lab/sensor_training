"""Thin DDP helpers around ``torch.distributed`` for ``torchrun`` launches.

Launch model (see ``robot_skin/train/README.md``): one process per GPU started by ``torchrun``,
which exports ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR`` and ``MASTER_PORT``.
Without those variables every helper here is a no-op, so the same training code runs as a
plain single process on CPU (tests) or one GPU.

Backend: NCCL on CUDA, gloo otherwise. For multi-machine runs over Tailscale set
``NCCL_SOCKET_IFNAME=tailscale0`` / ``GLOO_SOCKET_IFNAME=tailscale0`` — but prefer one
single-node job per machine (see README): the WireGuard tunnel is far slower than
PCIe/NVLink and the slowest GPU sets the pace of a synchronous all-reduce.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DistributedSampler, IterableDataset, Sampler

__all__ = ["DistInfo", "init_distributed", "is_dist_initialized", "wrap_ddp", "unwrap_model",
           "make_sampler", "make_eval_sampler", "ShardSampler", "barrier", "cleanup",
           "all_reduce_sum", "all_reduce_mean", "all_ranks_equal"]


@dataclass(frozen=True)
class DistInfo:
    """Process-group coordinates. ``is_main`` gates file IO / logging; it defaults to
    ``rank == 0`` when not given (so ``DistInfo(rank=1, world_size=2)`` is not a main rank)."""
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    is_main: bool | None = None
    backend: str | None = None

    def __post_init__(self) -> None:
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"DistInfo: invalid rank {self.rank} for world_size {self.world_size}")
        if self.is_main is None:
            object.__setattr__(self, "is_main", self.rank == 0)

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @classmethod
    def single(cls) -> "DistInfo":
        return cls()


def _env_int(name: str, default: int | None = None) -> int | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"environment variable {name}={v!r} is not an integer") from e


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_distributed(backend: str | None = None, timeout_s: float = 1800.0) -> DistInfo:
    """Initialise the default process group from ``torchrun`` env vars.

    No-op (single-process :class:`DistInfo`) when ``WORLD_SIZE`` is unset or 1. Idempotent: if
    a group already exists its coordinates are returned. ``backend`` defaults to ``nccl`` when
    CUDA is available, else ``gloo``; with NCCL the current CUDA device is set to ``LOCAL_RANK``.
    """
    world = _env_int("WORLD_SIZE", 1) or 1
    local_rank = _env_int("LOCAL_RANK", 0) or 0
    if world <= 1:
        return DistInfo(rank=0, world_size=1, local_rank=local_rank, is_main=True, backend=None)
    if not dist.is_available():
        raise RuntimeError("WORLD_SIZE > 1 but torch.distributed is not available in this build")
    if dist.is_initialized():
        rank = dist.get_rank()
        return DistInfo(rank, dist.get_world_size(), local_rank, rank == 0, dist.get_backend())
    missing = [k for k in ("RANK", "MASTER_ADDR", "MASTER_PORT") if k not in os.environ]
    if missing:
        raise RuntimeError(f"WORLD_SIZE={world} but {missing} not set — launch with torchrun, e.g. "
                           "`torchrun --standalone --nproc_per_node=2 -m robot_skin train vtla`")
    rank = _env_int("RANK")
    assert rank is not None
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("backend 'nccl' requires CUDA; use backend='gloo' on CPU")
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world,
                            timeout=timedelta(seconds=timeout_s))
    return DistInfo(rank, world, local_rank, rank == 0, backend)


def wrap_ddp(model: nn.Module, info: DistInfo, *, find_unused_parameters: bool = False,
             **ddp_kwargs) -> nn.Module:
    """Wrap in ``DistributedDataParallel`` when ``info.distributed``; otherwise return ``model``.

    Move the model to its device first. ``find_unused_parameters=True`` is needed when some
    parameters get no gradient in a step (e.g. modality dropout skipping the tactile branch).
    """
    if not info.distributed:
        return model
    if not is_dist_initialized():
        raise RuntimeError("wrap_ddp: process group not initialised (call init_distributed())")
    from torch.nn.parallel import DistributedDataParallel as DDP

    try:
        p = next(model.parameters())
    except StopIteration:
        p = None
    if p is not None and p.is_cuda:
        # the device the model actually lives on (normally cuda:<local_rank>, but an explicit
        # `device: cuda:N` in the config must not be contradicted here)
        idx = p.device.index if p.device.index is not None else torch.cuda.current_device()
        return DDP(model, device_ids=[idx], output_device=idx,
                   find_unused_parameters=find_unused_parameters, **ddp_kwargs)
    return DDP(model, find_unused_parameters=find_unused_parameters, **ddp_kwargs)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip ``torch.compile`` (``_orig_mod``) and DDP / DataParallel (``module``) wrappers."""
    from torch.nn.parallel import DataParallel, DistributedDataParallel

    while True:
        if isinstance(getattr(model, "_orig_mod", None), nn.Module):
            model = model._orig_mod
        elif isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        else:
            return model


def make_sampler(dataset: Dataset, info: DistInfo | None = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> Sampler | None:
    """Training sampler with epoch-seeded order (call ``set_epoch(e)`` each epoch).

    Always a :class:`DistributedSampler` — with ``num_replicas=1`` in single-process runs, which
    gives a deterministic, resumable ``(seed, epoch)``-keyed permutation without a process group.
    ``None`` for ``IterableDataset`` (the dataset must shard itself across ranks).
    """
    info = info or DistInfo.single()
    if isinstance(dataset, IterableDataset):
        if info.distributed:
            warnings.warn("IterableDataset under DDP: shard the stream by rank yourself",
                          stacklevel=2)
        return None
    return DistributedSampler(dataset, num_replicas=info.world_size, rank=info.rank,
                              shuffle=shuffle, seed=seed, drop_last=drop_last)


class ShardSampler(Sampler[int]):
    """Deterministic, non-padding shard ``rank::world_size`` for evaluation.

    Unlike :class:`DistributedSampler` it never duplicates samples, so metrics reduced with
    :func:`all_reduce_sum` (sum, count) are exact. Ranks may get different batch counts, so the
    evaluated model must not run collectives in ``forward`` (use the unwrapped module).
    """

    def __init__(self, dataset: Dataset, info: DistInfo | None = None):
        info = info or DistInfo.single()
        self.n = len(dataset)  # type: ignore[arg-type]
        self.rank, self.world = info.rank, info.world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.n, self.world))

    def __len__(self) -> int:
        return max(0, math.ceil((self.n - self.rank) / self.world))


def make_eval_sampler(dataset: Dataset, info: DistInfo | None = None) -> Sampler | None:
    """Sequential (single process) / :class:`ShardSampler` (DDP) sampler for validation."""
    if isinstance(dataset, IterableDataset):
        return None
    return ShardSampler(dataset, info)


def barrier(info: DistInfo | None = None) -> None:
    """Synchronise all ranks (no-op without a process group)."""
    if not is_dist_initialized():
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def cleanup() -> None:
    """Destroy the default process group if one exists."""
    if is_dist_initialized():
        dist.destroy_process_group()


def _reduce_device() -> torch.device:
    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def all_reduce_sum(values: Mapping[str, float]) -> dict[str, float]:
    """Sum scalar metrics over ranks (keys must match on every rank). Identity when not
    distributed."""
    out = {k: float(v) for k, v in values.items()}
    if not is_dist_initialized() or dist.get_world_size() == 1 or not out:
        return out
    keys = sorted(out)
    t = torch.tensor([out[k] for k in keys], dtype=torch.float64, device=_reduce_device())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {k: float(x) for k, x in zip(keys, t.tolist())}


def all_ranks_equal(values: Sequence[float]) -> bool:
    """Collective: ``True`` iff every rank passed the same numbers (MIN == MAX all-reduce).
    ``True`` without a process group. Every rank must call it with the same length."""
    vals = [float(v) for v in values]
    if not is_dist_initialized() or dist.get_world_size() == 1 or not vals:
        return True
    t = torch.tensor(vals, dtype=torch.float64, device=_reduce_device())
    lo, hi = t.clone(), t.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    return bool(torch.equal(lo, hi))


def all_reduce_mean(values: Mapping[str, float]) -> dict[str, float]:
    """Unweighted mean of scalar metrics over ranks (use :func:`all_reduce_sum` with explicit
    counts when ranks saw different numbers of samples)."""
    out = all_reduce_sum(values)
    if is_dist_initialized():
        w = dist.get_world_size()
        out = {k: v / w for k, v in out.items()}
    return out
