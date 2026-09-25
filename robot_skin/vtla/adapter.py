"""Tactile tokens → a VLA/VTLA backbone.

:class:`TactileTokenAdapter` compresses ``N`` taxel tokens into ``K`` query tokens with
cross-attention (Perceiver-style, taxel count agnostic) and projects to the backbone width.
:class:`ContactGate` scales them by contact evidence so that *without contact the tactile
stream contributes nothing* — the backbone cannot hallucinate touch from sensor drift.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ContactGate(nn.Module):
    """``hard``: tokens × 1[any contact]. ``soft``: tokens × σ(w·contact_frac + b) × 1[any contact].

    The soft variant keeps the hard zero at no-contact but learns how much a light/sparse
    contact should be trusted.
    """

    def __init__(self, mode: str = "hard"):
        super().__init__()
        if mode not in ("hard", "soft"):
            raise ValueError("mode must be hard|soft")
        self.mode = mode
        if mode == "soft":
            self.w = nn.Parameter(torch.tensor(8.0))
            self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, tokens: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
        """tokens ``[B,K,D]``, contact ``[B,N]`` bool/float → gated tokens."""
        c = contact.float()
        any_c = (c.sum(-1) > 0).float()[:, None, None]
        if self.mode == "hard":
            return tokens * any_c
        frac = c.mean(-1)[:, None, None]
        return tokens * any_c * torch.sigmoid(self.w * frac + self.b)


class TactileTokenAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, *, n_query: int = 4, n_heads: int = 4,
                 gate: str = "hard"):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_query, d_in) * 0.02)
        self.attn = nn.MultiheadAttention(d_in, n_heads, batch_first=True)
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out))
        self.gate = ContactGate(gate)

    def forward(self, taxel_tokens: torch.Tensor, contact: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """taxel_tokens ``[B,N,d_in]``, contact ``[B,N]`` → ``[B,K,d_out]``."""
        q = self.queries.unsqueeze(0).expand(taxel_tokens.shape[0], -1, -1)
        out, _ = self.attn(q, taxel_tokens, taxel_tokens, key_padding_mask=key_padding_mask)
        return self.gate(self.proj(out), contact)
