"""Optimiser, LR schedule and EMA utilities shared by every training stage.

* :func:`param_groups` — AdamW with *decoupled* weight decay (Loshchilov & Hutter) should not
  decay biases, normalisation gains, embeddings and learned query/token parameters (standard
  practice in transformer training, e.g. MAE / ViT recipes); optional per-module LR multipliers
  (``lr_mult``) let a pretrained encoder fine-tune slower than a fresh head (VTLA).
* :func:`build_scheduler` — linear warm-up followed by cosine / linear decay to
  ``min_lr_ratio`` (or constant), as a ``LambdaLR`` stepped once per *optimizer* step.
* :class:`EMA` — exponential moving average of weights (with the usual
  ``min(decay, (1+n)/(10+n))`` warm-up) used for evaluation / export.
"""
from __future__ import annotations

import contextlib
import inspect
import math
import re
import warnings
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from .distributed import unwrap_model

#: substrings of a parameter's *own* attribute name that mark it as non-decayed
DEFAULT_NO_DECAY_KEYWORDS: tuple[str, ...] = ("token", "query", "queries", "embed")

_NORM_TYPES: tuple[type, ...] = (nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm,
                                 nn.LocalResponseNorm)
if hasattr(nn, "RMSNorm"):
    _NORM_TYPES = _NORM_TYPES + (nn.RMSNorm,)
_EMBED_TYPES: tuple[type, ...] = (nn.Embedding, nn.EmbeddingBag)

SCHEDULES = ("cosine", "linear", "constant")
OPTIMIZERS = ("adamw", "adam", "sgd")

__all__ = ["DEFAULT_NO_DECAY_KEYWORDS", "SCHEDULES", "OPTIMIZERS", "split_decay_params",
           "param_groups", "build_optimizer", "lr_factor", "build_scheduler", "EMA"]


# ───────────────────────────────────────────────────────────────────────────── param groups

# custom normalisation layers by class name: LayerNorm2d, FusedRMSNorm, L2Norm … but NOT e.g.
# "NormalEncoder" / "Normalizer" (taxel surface normals are everywhere in this code base)
_NORM_NAME_RE = re.compile(r"norm(\d+d)?$")


def _is_norm(module: nn.Module) -> bool:
    return (isinstance(module, _NORM_TYPES)
            or _NORM_NAME_RE.search(type(module).__name__.lower()) is not None)


def _no_decay(module: nn.Module, local_name: str, p: torch.Tensor,
              keywords: Sequence[str]) -> bool:
    if p.ndim <= 1 or local_name.endswith("bias"):
        return True
    if _is_norm(module) or isinstance(module, _EMBED_TYPES):
        return True
    low = local_name.lower()
    return any(k in low for k in keywords)


def split_decay_params(model: nn.Module,
                       no_decay_keywords: Sequence[str] = DEFAULT_NO_DECAY_KEYWORDS,
                       ) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    """Split trainable parameters into ``(decay, no_decay)`` lists of ``(name, param)``.

    No decay for: 0/1-D tensors (biases, norm gains, scalars), anything named ``*bias``, all
    parameters of normalisation layers and ``nn.Embedding``, and free parameters whose own
    attribute name contains a keyword (``cls_token``, ``readout_tokens``, ``query``,
    ``pos_embed``, ``camera_embedding`` …). Keywords match the *local* name, so a ``Linear``
    stored as ``self.query`` (params ``weight``/``bias``) keeps weight decay on its weight.
    Frozen (``requires_grad=False``) and shared/tied parameters (after the first) are skipped.
    """
    model = unwrap_model(model)
    seen: set[int] = set()
    decay: list[tuple[str, nn.Parameter]] = []
    no_decay: list[tuple[str, nn.Parameter]] = []
    for mod_name, module in model.named_modules():
        for local, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            full = f"{mod_name}.{local}" if mod_name else local
            (no_decay if _no_decay(module, local, p, no_decay_keywords) else decay).append((full, p))
    return decay, no_decay


def _lr_mult_for(name: str, lr_mult: Mapping[str, float] | None) -> float:
    if not lr_mult:
        return 1.0
    best, best_len = 1.0, -1
    for prefix, m in lr_mult.items():
        if (name == prefix or name.startswith(prefix + ".")) and len(prefix) > best_len:
            best, best_len = float(m), len(prefix)
    return best


def param_groups(model: nn.Module, weight_decay: float = 0.05, *, lr: float | None = None,
                 lr_mult: Mapping[str, float] | None = None,
                 no_decay_keywords: Sequence[str] = DEFAULT_NO_DECAY_KEYWORDS) -> list[dict]:
    """Optimizer param groups: decayed vs non-decayed, further split by ``lr_mult``.

    ``lr_mult`` maps a dotted module prefix (``"vision"``, ``"tactile.encoder"``) to an LR
    multiplier (longest prefix wins; ``0`` excludes those params → frozen). Each group carries
    ``weight_decay``, ``lr_mult``, ``group_name`` and, if ``lr`` is given, ``lr = lr·mult``.
    Empty groups are omitted.
    """
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    decay, no_decay = split_decay_params(model, no_decay_keywords)
    if lr_mult:
        names = [n for n, _ in decay + no_decay]
        unused = [k for k in lr_mult
                  if not any(n == k or n.startswith(k + ".") for n in names)]
        if unused:  # most likely a typo in the config — it would silently do nothing
            warnings.warn(f"lr_mult prefixes {unused} match no trainable parameter "
                          f"(top-level modules: {sorted({n.split('.')[0] for n in names})})",
                          stacklevel=2)
    buckets: dict[tuple[bool, float], list[nn.Parameter]] = {}
    for is_decay, items in ((True, decay), (False, no_decay)):
        for name, p in items:
            m = _lr_mult_for(name, lr_mult)
            if m == 0.0:
                continue
            if m < 0:
                raise ValueError(f"lr_mult for {name!r} must be >= 0, got {m}")
            buckets.setdefault((is_decay, m), []).append(p)
    groups = []
    for (is_decay, m), params in sorted(buckets.items(), key=lambda kv: (not kv[0][0], -kv[0][1])):
        g: dict[str, Any] = {
            "params": params,
            "weight_decay": float(weight_decay) if is_decay else 0.0,
            "lr_mult": m,
            "group_name": ("decay" if is_decay else "no_decay") + ("" if m == 1.0 else f"@{m:g}"),
        }
        if lr is not None:
            g["lr"] = float(lr) * m
        groups.append(g)
    if not groups:
        raise ValueError("param_groups: model has no trainable parameters")
    return groups


def build_optimizer(model_or_params: nn.Module | Iterable[Any], name: str = "adamw",
                    lr: float = 3e-4, weight_decay: float = 0.05,
                    betas: Sequence[float] = (0.9, 0.999), *, eps: float = 1e-8,
                    momentum: float = 0.9, fused: bool | str = "auto",
                    lr_mult: Mapping[str, float] | None = None) -> torch.optim.Optimizer:
    """Build ``adamw`` (default) / ``adam`` / ``sgd`` (Nesterov).

    Given an ``nn.Module`` the groups come from :func:`param_groups`; given params / group dicts
    they are used as-is (group ``lr`` defaults to ``lr·lr_mult``). ``fused="auto"`` enables the
    fused CUDA Adam(W) kernel when every parameter is a floating CUDA tensor — build the
    optimizer *after* moving the model to its device.
    """
    key = name.strip().lower()
    if key not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; use one of {OPTIMIZERS}")
    if not lr > 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if isinstance(model_or_params, nn.Module):
        groups: list[Any] = param_groups(model_or_params, weight_decay, lr=lr, lr_mult=lr_mult)
    else:
        groups = list(model_or_params)
        if not groups:
            raise ValueError("build_optimizer: empty parameter list")
        for g in groups:
            if isinstance(g, dict) and "lr" not in g:
                g["lr"] = lr * float(g.get("lr_mult", 1.0))
    all_params = [p for g in groups for p in (g["params"] if isinstance(g, dict) else [g])]

    if key in ("adamw", "adam"):
        cls = torch.optim.AdamW if key == "adamw" else torch.optim.Adam
        kw: dict[str, Any] = dict(lr=lr, betas=tuple(float(b) for b in betas), eps=eps,
                                  weight_decay=weight_decay)
        use_fused = (all(p.is_cuda and p.is_floating_point() for p in all_params)
                     if fused == "auto" else bool(fused))
        if use_fused and "fused" in inspect.signature(cls.__init__).parameters:
            kw["fused"] = True
        return cls(groups, **kw)
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=momentum > 0,
                           weight_decay=weight_decay)


# ───────────────────────────────────────────────────────────────────────────── schedule

def lr_factor(step: int, schedule: str = "cosine", warmup_steps: int = 0,
              total_steps: int | None = None, min_lr_ratio: float = 0.0) -> float:
    """LR multiplier at optimizer step ``step`` (0-based: the value used by the first update).

    Warm-up: ``(step+1)/warmup_steps`` for ``step < warmup_steps`` (reaches 1 on the last
    warm-up step, never 0). Then with ``p = (step − warmup)/(total − warmup)`` clipped to [0, 1]:
    cosine ``r + (1−r)·½(1+cos πp)``, linear ``r + (1−r)(1−p)``, constant ``1``;
    ``r = min_lr_ratio``.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    if total_steps is None:
        raise ValueError(f"schedule {schedule!r} needs total_steps")
    span = max(1, total_steps - warmup_steps)
    p = min(1.0, max(0.0, (step - warmup_steps) / span))
    r = min_lr_ratio
    if schedule == "cosine":
        return r + (1.0 - r) * 0.5 * (1.0 + math.cos(math.pi * p))
    if schedule == "linear":
        return r + (1.0 - r) * (1.0 - p)
    raise ValueError(f"unknown schedule {schedule!r}; use one of {SCHEDULES}")


def build_scheduler(optimizer: torch.optim.Optimizer, schedule: str = "cosine",
                    warmup_steps: int = 0, total_steps: int | None = None,
                    min_lr_ratio: float = 0.1) -> LambdaLR:
    """``LambdaLR`` implementing :func:`lr_factor`; call ``.step()`` after every optimizer step.

    The factor is a plain closure (not serialised by ``LambdaLR.state_dict``), so a resumed run
    follows the *current* config's schedule at the restored step.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; use one of {SCHEDULES}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
    if schedule != "constant":
        if total_steps is None or total_steps <= 0:
            raise ValueError(f"schedule {schedule!r} needs total_steps > 0 (got {total_steps})")
    w, t, r = int(warmup_steps), total_steps, float(min_lr_ratio)

    def fn(step: int) -> float:
        return lr_factor(step, schedule, w, t, r)

    return LambdaLR(optimizer, fn)


# ───────────────────────────────────────────────────────────────────────────── EMA

class EMA:
    """Exponential moving average of a model's ``state_dict`` (params *and* buffers).

    ``shadow ← d·shadow + (1−d)·θ`` after each optimizer step, with
    ``d = min(decay, (1+n)/(10+n))`` when ``warmup`` (``n`` = number of updates) so short runs are
    not dominated by the initial weights. Integer buffers (e.g. BatchNorm counters) are copied.
    Keys follow the *unwrapped* model (no ``module.`` / ``_orig_mod.`` prefixes).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, *, warmup: bool = True,
                 device: torch.device | str | None = None):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.device = torch.device(device) if device is not None else None
        sd = unwrap_model(model).state_dict()
        self.shadow: dict[str, torch.Tensor] = {
            k: (v.detach().clone() if self.device is None else v.detach().to(self.device).clone())
            for k, v in sd.items()}

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        n = self.num_updates
        return min(self.decay, (1.0 + n) / (10.0 + n))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = self.current_decay()
        msd = unwrap_model(model).state_dict()
        for k, s in self.shadow.items():
            v = msd[k].detach()
            if s.is_floating_point():
                s.lerp_(v.to(device=s.device, dtype=s.dtype), 1.0 - d)
            else:
                s.copy_(v.to(s.device))

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        """Overwrite ``model``'s weights with the EMA weights (permanent)."""
        unwrap_model(model).load_state_dict(self.shadow, strict=True)

    @contextlib.contextmanager
    def swap(self, model: nn.Module) -> Iterator[nn.Module]:
        """Temporarily load EMA weights into ``model`` (restored on exit)."""
        m = unwrap_model(model)
        backup = {k: v.detach().clone() for k, v in m.state_dict().items()}
        try:
            with torch.no_grad():
                m.load_state_dict(self.shadow, strict=True)
            yield m
        finally:
            with torch.no_grad():
                m.load_state_dict(backup, strict=True)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "warmup": self.warmup, "num_updates": self.num_updates,
                "shadow": {k: v.detach().clone() for k, v in self.shadow.items()}}

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore ``num_updates`` and the shadow weights (the configured ``decay`` is kept)."""
        shadow = state["shadow"]
        missing = set(self.shadow) - set(shadow)
        if missing:
            raise KeyError(f"EMA state is missing keys: {sorted(missing)[:5]}...")
        for k, s in self.shadow.items():
            s.copy_(shadow[k].to(device=s.device, dtype=s.dtype))
        self.num_updates = int(state.get("num_updates", 0))
