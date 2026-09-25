"""robot_skin.train — shared training infrastructure for every stage.

Device/precision resolution and GPU hardware profiles (:mod:`.hardware`), ``torchrun`` DDP
helpers (:mod:`.distributed`), optimiser / LR schedule / EMA (:mod:`.optim`), crash-safe
checkpoints (:mod:`.checkpoint`), JSONL + optional TensorBoard/W&B logging
(:mod:`.logging_utils`), the generic :class:`Trainer` (:mod:`.engine`) and hyper-parameter
sweeps (:mod:`.sweep`). See ``robot_skin/train/README.md`` for RTX 5090 / multi-GPU /
Tailscale multi-machine usage.

Exports are resolved lazily (PEP 562) so ``python -m robot_skin.train.hardware`` and
``python -m robot_skin.train.sweep`` run without importing their own module twice.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, str] = {
    # engine
    "TrainConfig": "engine", "Trainer": "engine", "seed_everything": "engine",
    "seed_worker": "engine", "move_to_device": "engine", "maybe_compile": "engine",
    # hardware
    "PrecisionPlan": "hardware", "resolve_device": "hardware", "resolve_precision": "hardware",
    "enable_tf32": "hardware", "list_hw_profiles": "hardware", "load_hw_profile": "hardware",
    "apply_hw_profile": "hardware", "maybe_apply_hw_profile": "hardware",
    "apply_profile_env": "hardware", "detect_hw_profile": "hardware",
    "describe_environment": "hardware", "check_arch_support": "hardware",
    # distributed
    "DistInfo": "distributed", "init_distributed": "distributed", "wrap_ddp": "distributed",
    "unwrap_model": "distributed", "make_sampler": "distributed",
    "make_eval_sampler": "distributed", "barrier": "distributed", "cleanup": "distributed",
    "all_reduce_sum": "distributed", "all_reduce_mean": "distributed",
    # optim
    "param_groups": "optim", "build_optimizer": "optim", "build_scheduler": "optim",
    "lr_factor": "optim", "EMA": "optim",
    # checkpoint / logging
    "save_checkpoint": "checkpoint", "load_checkpoint": "checkpoint", "find_last": "checkpoint",
    "find_best": "checkpoint", "strip_state_dict_prefixes": "checkpoint",
    "LAST_NAME": "checkpoint", "BEST_NAME": "checkpoint",
    "JsonlLogger": "logging_utils", "read_jsonl": "logging_utils",
    "setup_logging": "logging_utils",
    # sweep
    "expand_grid": "sweep", "sample_random": "sweep", "shard": "sweep", "run_sweep": "sweep",
    "load_results": "sweep",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'robot_skin.train' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value  # cache
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # static analysers / IDEs see the real symbols
    from .checkpoint import (BEST_NAME, LAST_NAME, find_best, find_last, load_checkpoint,
                             save_checkpoint, strip_state_dict_prefixes)
    from .distributed import (DistInfo, all_reduce_mean, all_reduce_sum, barrier, cleanup,
                              init_distributed, make_eval_sampler, make_sampler, unwrap_model,
                              wrap_ddp)
    from .engine import (TrainConfig, Trainer, maybe_compile, move_to_device, seed_everything,
                         seed_worker)
    from .hardware import (PrecisionPlan, apply_hw_profile, apply_profile_env,
                           check_arch_support, describe_environment, detect_hw_profile,
                           enable_tf32, list_hw_profiles, load_hw_profile,
                           maybe_apply_hw_profile, resolve_device, resolve_precision)
    from .logging_utils import JsonlLogger, read_jsonl, setup_logging
    from .optim import EMA, build_optimizer, build_scheduler, lr_factor, param_groups
    from .sweep import expand_grid, load_results, run_sweep, sample_random, shard
