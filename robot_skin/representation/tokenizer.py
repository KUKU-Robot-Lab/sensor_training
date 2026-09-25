"""TaxelTokenizer: per-taxel value features + pose embedding → one token per taxel.

token_i = LN( value_mlp(v_i) + pose_mlp([fourier(pos_i), nrm_i]) (+ id_emb_i) )

Pose (not channel index) carries the geometry, so the same tokenizer can serve a glove and a
robot hand with different taxel counts; the optional id embedding is for single-layout models.

Position features are Fourier features (Tancik et al., NeurIPS 2020, arXiv:2006.10739) at octave
frequencies: the lowest octave has period ``fourier_scale`` and the finest ``fourier_scale /
2**(n_fourier−1)``. Defaults ``fourier_scale = 0.3`` m, ``n_fourier = 6`` (finest period ≈ 9 mm):
the lowest octave spans a whole hand (≈ 0.2 m in the hand frame) without wrapping, and no octave is
finer than the mm-level error of vision / IMU pose labels (with 0.05 m × 8 octaves the finest
period was 0.4 mm, so the top octaves were pure pose noise).
``mask[B,N]`` swaps the value part for a learned ``[MASK]`` vector (masked pretraining) while
keeping the pose part, so the model knows *where* it has to fill in.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def fourier_features(pos: torch.Tensor, n_freq: int, scale: float) -> torch.Tensor:
    """``[...,3]`` → ``[..., 3*2*n_freq]`` sin/cos at octave frequencies (lowest period ``scale``,
    halving per octave; Tancik et al., arXiv:2006.10739)."""
    freqs = (2.0 ** torch.arange(n_freq, device=pos.device, dtype=pos.dtype)) * (2 * math.pi / scale)
    x = pos.unsqueeze(-1) * freqs  # [...,3,F]
    return torch.cat([x.sin(), x.cos()], dim=-1).flatten(-2)


class TaxelTokenizer(nn.Module):
    def __init__(self, value_dim: int, d_model: int = 64, *, n_fourier: int = 6,
                 fourier_scale: float = 0.3, n_taxels: int | None = None) -> None:
        super().__init__()
        self.n_fourier, self.fourier_scale = n_fourier, fourier_scale
        self.value_mlp = nn.Sequential(nn.Linear(value_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pose_mlp = nn.Sequential(nn.Linear(6 * n_fourier + 3, d_model), nn.GELU(),
                                      nn.Linear(d_model, d_model))
        self.id_emb = nn.Embedding(n_taxels, d_model) if n_taxels else None
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values: torch.Tensor, pos: torch.Tensor, nrm: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """values ``[B,N,F]``, pos/nrm ``[B,N,3]``, mask ``[B,N]`` bool → tokens ``[B,N,D]``."""
        v = self.value_mlp(values)
        if mask is not None:
            v = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(v), v)
        p = self.pose_mlp(torch.cat([fourier_features(pos, self.n_fourier, self.fourier_scale), nrm], -1))
        tok = v + p
        if self.id_emb is not None:
            tok = tok + self.id_emb(torch.arange(values.shape[1], device=values.device))
        return self.norm(tok)
