"""Masked-taxel pretraining (stub) + masking helper (implemented).

Plan: tokens from :class:`TaxelTokenizer` with ``mask`` → small transformer encoder →
reconstruct masked taxels' residual ΔS (+ ordinal level) from neighbours and pose. Data: all
glove/robot sessions (no labels needed). Masking whole groups (a fingertip, the palm) forces
cross-region reasoning.
"""
from __future__ import annotations

import torch


def random_taxel_mask(batch: int, n_taxels: int, ratio: float, *,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Bool ``[B,N]`` with exactly ``round(ratio*N)`` (≥1 if ratio>0) masked taxels per row."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio must be in [0, 1)")
    k = max(1, round(ratio * n_taxels)) if ratio > 0 else 0
    idx = torch.rand(batch, n_taxels, generator=generator).argsort(dim=1)[:, :k]
    m = torch.zeros(batch, n_taxels, dtype=torch.bool)
    m.scatter_(1, idx, True)
    return m


class MaskedTaxelPretrainer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("masked taxel pretraining not implemented yet (see module docstring)")
