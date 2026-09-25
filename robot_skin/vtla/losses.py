"""Losses of the VTLA policy: masked action-chunk losses, auxiliary per-taxel contact BCE and the
:class:`~robot_skin.train.Trainer` loss function.

Action chunks are padded at the episode end (and where the source label is invalid) with
``action_valid = False`` (ACT's ``is_pad``, :func:`robot_skin.action.chunking.action_chunk`), so
every action loss here is a **masked mean over valid (step, dim) entries**:

- :func:`masked_l1` — ACT's chunk regression loss (Zhao et al., RSS 2023, arXiv:2304.13705 use L1
  because it gives more precise chunks than L2);
- :func:`masked_mse` — the conditional flow-matching regression of the velocity field
  (Lipman et al., ICLR 2023, arXiv:2210.02747), see :mod:`robot_skin.vtla.heads`;
- :func:`masked_step_sums` — per-horizon-step error sums and counts for exact evaluation
  (``Σ err / Σ count`` over a whole split, independent of the batching).

With no valid entry in a batch the losses return a zero that keeps the graph (a DDP step never sees
a parameter without gradient). Masked entries are selected out with ``torch.where`` before the
reduction, so even non-finite padding cannot leak into the loss, and no host sync is needed.

:func:`contact_bce` is the optional auxiliary per-taxel contact loss on the tactile encoder tokens
(targets with ``mask = False`` — unknown labels, padded taxels — are ignored).

:func:`vtla_loss` is the Trainer ``loss_fn``: it calls ``model(batch)`` (the DDP / compiled wrapper
during training) and keeps the scalar entries of the returned dict.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

__all__ = ["masked_l1", "masked_mse", "masked_error", "masked_step_sums", "contact_bce",
           "weighted_total", "vtla_loss"]


def _mask_weights(valid: torch.Tensor | None, err: torch.Tensor) -> torch.Tensor:
    """``valid`` ``[B,H]`` (or ``[B,H,A]``, or None) → float weights broadcastable to ``err``."""
    if valid is None:
        return torch.ones_like(err)
    v = valid.to(device=err.device)
    if v.ndim == err.ndim - 1:
        v = v.unsqueeze(-1)
    if v.ndim != err.ndim:
        raise ValueError(f"valid {tuple(valid.shape)} does not match error {tuple(err.shape)}")
    return torch.broadcast_to(v.to(err.dtype), err.shape)


def masked_error(err: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Mean of ``err`` ``[B,H,A]`` over entries with ``valid`` ``[B,H]`` True (scalar).

    A graph-preserving zero when nothing is valid; invalid entries never contribute (not even NaN).
    """
    w = _mask_weights(valid, err)
    e = torch.where(w > 0, err, torch.zeros((), dtype=err.dtype, device=err.device))
    return (e * w).sum() / w.sum().clamp_min(1.0)


def _safe_target(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    """Target with invalid entries replaced by ``pred.detach()`` (zero error *and* zero, finite
    gradient there, even for NaN padding)."""
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    if valid is None:
        return target
    w = _mask_weights(valid, pred) > 0
    return torch.where(w, target.to(pred.dtype), pred.detach())


def masked_l1(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """ACT chunk loss: mean ``|pred − target|`` over valid steps and all action dims."""
    return masked_error((pred - _safe_target(pred, target, valid)).abs(), valid)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Mean ``(pred − target)²`` over valid steps and all dims (flow-matching velocity loss)."""
    return masked_error((pred - _safe_target(pred, target, valid)).square(), valid)


def masked_step_sums(err: torch.Tensor, valid: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-horizon-step sums for exact aggregation: ``err`` ``[B,H,A]`` →
    ``(Σ_b,a err·valid  [H], Σ_b,a valid  [H])`` (float64). ``sum / count`` is the per-step mean."""
    if err.ndim != 3:
        raise ValueError(f"err must be [B,H,A], got {tuple(err.shape)}")
    w = _mask_weights(valid, err).to(torch.float64)
    e = torch.where(w > 0, err.detach().to(torch.float64), 0.0)
    return (e * w).sum(dim=(0, 2)), w.sum(dim=(0, 2))


def contact_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, *,
                pos_weight: float | None = None) -> torch.Tensor:
    """Masked per-taxel BCE-with-logits (``logits``/``target``/``mask`` ``[B,N]``).

    ``pos_weight`` (> 0) up-weights positive (contact) taxels — contact is rare in most frames.
    Mean over ``mask`` True entries; graph-preserving zero when none.
    """
    if logits.shape != target.shape:
        raise ValueError(f"logits {tuple(logits.shape)} != target {tuple(target.shape)}")
    pw = None if pos_weight is None else torch.as_tensor(float(pos_weight), dtype=logits.dtype,
                                                         device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none",
                                              pos_weight=pw)
    if mask is None:
        return loss.mean()
    m = mask.to(device=logits.device).bool()
    kept = torch.where(m, loss, torch.zeros((), dtype=loss.dtype, device=loss.device))
    return kept.sum() / m.sum().clamp_min(1).to(loss.dtype)


def weighted_total(terms: Mapping[str, torch.Tensor], weights: Mapping[str, float]) -> torch.Tensor:
    """``Σ_k weights[k] · terms[k]`` over the keys of ``weights`` present in ``terms`` (weight 0
    terms are skipped)."""
    total = None
    for k, w in weights.items():
        if k in terms and float(w) != 0.0:
            t = terms[k] * float(w)
            total = t if total is None else total + t
    if total is None:
        raise ValueError(f"no weighted loss term among {sorted(terms)} (weights {dict(weights)})")
    return total


def vtla_loss(model: Any, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """:class:`robot_skin.train.Trainer` loss function: ``model(batch)`` (``VTLAPolicy.forward``
    computes the head loss itself, so it also works through DDP / torch.compile) → the scalar
    entries (``loss``, ``action_loss``, ``contact_loss`` …)."""
    out = model(batch)
    return {k: v for k, v in out.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
