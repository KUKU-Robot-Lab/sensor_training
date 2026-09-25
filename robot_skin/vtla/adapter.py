"""Tactile tokens → a VLA/VTLA backbone.

:class:`TactileTokenAdapter` compresses ``N`` taxel tokens into ``K`` query tokens with
cross-attention (Perceiver-style latent queries, Jaegle et al., ICML 2021, arXiv:2103.03206 —
taxel count agnostic) and projects to the backbone width. :class:`ContactGate` scales them by
contact evidence so that *without contact the tactile stream contributes nothing* — the backbone
cannot hallucinate touch from sensor drift.

Padding (mixed-layout batches): pass ``key_padding_mask`` (True = padded taxel); padded taxels are
ignored as attention keys and never count as contact (nor, for the soft gate, in the contact
fraction's denominator).
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

    def forward(self, tokens: torch.Tensor, contact: torch.Tensor,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        """tokens ``[B,K,D]``, contact ``[B,N]`` bool/float → gated tokens.

        ``valid`` ``[B,N]`` bool (optional): real (non-padded) taxels; invalid ones are neither
        contact nor part of the contact fraction.
        """
        c = contact.to(tokens.dtype)
        if valid is not None:
            v = valid.to(tokens.dtype)
            c = c * v
            n = v.sum(-1).clamp_min(1.0)
        else:
            n = torch.full(c.shape[:-1], float(c.shape[-1]), dtype=tokens.dtype, device=c.device)
        any_c = (c.sum(-1) > 0).to(tokens.dtype)[:, None, None]
        if self.mode == "hard":
            return tokens * any_c
        frac = (c.sum(-1) / n)[:, None, None]
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
        q = self.queries.unsqueeze(0).expand(taxel_tokens.shape[0], -1, -1).to(taxel_tokens.dtype)
        kpm = None if key_padding_mask is None else key_padding_mask.bool()
        out, _ = self.attn(q, taxel_tokens, taxel_tokens, key_padding_mask=kpm, need_weights=False)
        return self.gate(self.proj(out), contact, None if kpm is None else ~kpm)
