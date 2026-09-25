"""BaselinePredictor — no-contact ΔS% from ``[taxel pose, q, q̇]``.

Generalises ``deformable_sats/sats/bending/baseline_restorer.py`` (deg → offset, 16 fixed
taxels, one bending DoF) to arbitrary taxels on a moving hand: the "condition" is no longer a
single bend angle but each taxel's pose plus the joint state, and the output is the ΔS the
taxel shows *without contact* (stretch, bending, thermal/air-pressure coupling …).

One MLP is shared across taxels; a learned per-taxel embedding absorbs per-taxel gain/offset.
As in the restorer the last layer is zero-initialised → prediction starts at 0 (identity
warm start: residual == observation).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BaselinePredictor(nn.Module):
    def __init__(self, n_taxels: int, joint_dim: int, *, hidden: int = 128, depth: int = 3,
                 taxel_embedding_dim: int = 8) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.n_taxels = n_taxels
        self.joint_dim = joint_dim
        self.taxel_emb = nn.Embedding(n_taxels, taxel_embedding_dim) if taxel_embedding_dim else None
        in_dim = 6 + 2 * joint_dim + (taxel_embedding_dim or 0)
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        last = nn.Linear(d, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(*layers, last)

    def forward(self, pos: torch.Tensor, nrm: torch.Tensor, q: torch.Tensor,
                qd: torch.Tensor) -> torch.Tensor:
        """pos/nrm ``[B,N,3]``, q/qd ``[B,D]`` → predicted no-contact ΔS% ``[B,N]``."""
        B, N, _ = pos.shape
        if N != self.n_taxels:
            raise ValueError(f"expected {self.n_taxels} taxels, got {N}")
        joint = torch.cat([q, qd], dim=-1).unsqueeze(1).expand(B, N, 2 * self.joint_dim)
        parts = [pos, nrm, joint]
        if self.taxel_emb is not None:
            ids = torch.arange(N, device=pos.device)
            parts.append(self.taxel_emb(ids).unsqueeze(0).expand(B, N, -1))
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)
