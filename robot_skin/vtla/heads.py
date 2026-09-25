"""Action heads of the VTLA policy: fused observation tokens → a chunk of ``H`` future actions.

Both heads read the fused token sequence ``memory [B,S,d]`` (language, vision, tactile, proprio and
readout tokens; ``memory_mask [B,S]`` True = ignore) through a small pre-LN transformer decoder
(self-attention over the ``H`` action slots + cross-attention to the memory) and emit
**normalized** actions ``[B,H,A]`` (``action.space.ActionNormalizer`` units).

:class:`ChunkRegressionHead` — ACT (Zhao et al., RSS 2023, arXiv:2304.13705): ``H`` learned query
slots cross-attend the observation tokens and regress the chunk directly; trained with the masked
L1 loss (:func:`robot_skin.vtla.losses.masked_l1`). Deterministic, one pass at inference.

:class:`FlowMatchingHead` — conditional flow matching (Lipman et al., ICLR 2023,
arXiv:2210.02747) on the straight noise→data path of rectified flow (Liu et al., ICLR 2023,
arXiv:2209.03003), the family of action experts used by π0 (Black et al., arXiv:2410.24164). It
models the full (possibly multi-modal) distribution of chunks instead of their mean.

**τ convention used here** (conventions differ between papers — π0's text and code, for
instance, name the ends differently; only this one applies to robot_skin):

- ``τ ∈ [0, 1]``; **τ = 0 is pure noise, τ = 1 is data**;
- path ``x_τ = τ · a + (1 − τ) · ε`` with ``ε ~ N(0, I)`` and ``a`` the normalized action chunk;
- target velocity ``u = d x_τ / dτ = a − ε`` (constant along the path);
- loss ``‖v_θ(x_τ, τ, obs) − u‖²`` averaged over valid (step, dim) entries (padded chunk steps
  are excluded; their ``x_τ`` still enters self-attention as context);
- training τ: ``uniform`` ``U[0, 1)`` (default), or ``beta`` = ``Beta(1, b)`` (inverse-CDF
  ``τ = 1 − (1 − u)^{1/b}``; ``b > 1`` puts more mass near τ = 0, i.e. on noisier inputs, where
  the conditional velocity is hardest to predict);
- inference: Euler integration of ``dx/dτ = v_θ`` from ``x_0 = ε`` (τ = 0) to τ = 1 in ``K``
  steps, ``x_{k+1} = x_k + (1/K) · v_θ(x_k, k/K)``.

τ enters through a sinusoidal embedding (transformer / DDPM style, scaled by 1000) → MLP, added to
every action token. The output layer is zero-initialised, so an untrained head predicts ``v = 0``
(samples = the input noise) and an untrained chunk head predicts the normalized mean (0).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .losses import masked_l1, masked_mse

__all__ = ["HEAD_TYPES", "TAU_DISTS", "sinusoidal_embedding", "ChunkRegressionHead",
           "FlowMatchingHead", "build_head"]

HEAD_TYPES = ("chunk", "flow")
TAU_DISTS = ("uniform", "beta")


def sinusoidal_embedding(x: torch.Tensor, dim: int, *, max_period: float = 10000.0,
                         scale: float = 1000.0) -> torch.Tensor:
    """``x`` ``[B]`` (e.g. τ ∈ [0,1]) → ``[B, dim]`` ``[cos | sin]`` features of ``scale·x`` at
    geometric frequencies ``max_period^{-i/half}`` (the transformer position encoding applied to a
    continuous scalar). Odd ``dim`` gets a zero last column."""
    if dim < 1:
        raise ValueError("dim must be >= 1")
    half = dim // 2
    x = x.reshape(-1).to(torch.float32)
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=x.device,
                                                           dtype=torch.float32) / max(half, 1))
    ang = (scale * x)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, emb.new_zeros(emb.shape[0], 1)], dim=-1)
    return emb


class _CrossDecoder(nn.Module):
    """Pre-LN ``nn.TransformerDecoder`` (self-attn over targets + cross-attn to memory) + LN."""

    def __init__(self, d_model: int, depth: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        if depth < 1:
            raise ValueError("head depth must be >= 1")
        if d_model % heads:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
        layer = nn.TransformerDecoderLayer(d_model, heads, dim_feedforward=ff_mult * d_model,
                                           dropout=dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, depth, norm=nn.LayerNorm(d_model))

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                memory_mask: torch.Tensor | None) -> torch.Tensor:
        kpm = None if memory_mask is None else memory_mask.bool()
        return self.decoder(tgt, memory, memory_key_padding_mask=kpm)


def _check_memory(memory: torch.Tensor, d_model: int) -> None:
    if memory.ndim != 3 or memory.shape[-1] != d_model:
        raise ValueError(f"memory must be [B,S,{d_model}], got {tuple(memory.shape)}")


class ChunkRegressionHead(nn.Module):
    """ACT-style chunk regression: ``H`` learned queries → decoder → ``Linear`` → ``[B,H,A]``.

    Args:
        d_model: width of the fused tokens.
        action_dim, horizon: ``A`` and ``H``.
        depth, heads, ff_mult, dropout: decoder size.
    """

    kind = "chunk"

    def __init__(self, d_model: int, action_dim: int, horizon: int, *, depth: int = 2,
                 heads: int = 4, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        if action_dim < 1 or horizon < 1:
            raise ValueError("action_dim and horizon must be >= 1")
        self.d_model, self.action_dim, self.horizon = int(d_model), int(action_dim), int(horizon)
        self.queries = nn.Parameter(torch.randn(self.horizon, self.d_model) * 0.02)
        self.decoder = _CrossDecoder(self.d_model, depth, heads, ff_mult, dropout)
        self.out = nn.Linear(self.d_model, self.action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``memory [B,S,d]`` → normalized action chunk ``[B,H,A]``."""
        _check_memory(memory, self.d_model)
        q = self.queries.to(memory.dtype).unsqueeze(0).expand(memory.shape[0], -1, -1)
        return self.out(self.decoder(q, memory, memory_mask))

    def loss(self, memory: torch.Tensor, memory_mask: torch.Tensor | None, actions: torch.Tensor,
             valid: torch.Tensor | None = None, **_: object) -> dict[str, torch.Tensor]:
        """Masked L1 against normalized ``actions [B,H,A]`` (``valid [B,H]``)."""
        pred = self(memory, memory_mask)
        l1 = masked_l1(pred.float(), actions.float(), valid)
        return {"action_loss": l1, "action_l1": l1.detach()}

    def per_sample_error(self, memory: torch.Tensor, memory_mask: torch.Tensor | None,
                         actions: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """``[B]`` mean squared error of the regressed chunk (for the DPO Gaussian surrogate)."""
        pred = self(memory, memory_mask).float()
        return _per_sample_mean((pred - actions.float()).square(), valid)

    @torch.no_grad()
    def sample(self, memory: torch.Tensor, memory_mask: torch.Tensor | None = None,
               **_: object) -> torch.Tensor:
        """Deterministic prediction (= :meth:`forward`), always float32 (also under autocast —
        like :meth:`FlowMatchingHead.sample`, so unnormalized control targets keep fp32 precision)."""
        return self(memory, memory_mask).float()


def _per_sample_mean(err: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    """``[B,H,A]`` → ``[B]`` mean over valid (step, dim) entries (0 where none)."""
    if valid is None:
        return err.mean(dim=(1, 2))
    w = valid.to(device=err.device).bool().unsqueeze(-1).expand_as(err)
    kept = torch.where(w, err, torch.zeros((), dtype=err.dtype, device=err.device))
    return kept.sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1).to(err.dtype)


class FlowMatchingHead(nn.Module):
    """Flow-matching action head (see the module docstring for the τ convention).

    Args:
        d_model, action_dim, horizon: as :class:`ChunkRegressionHead`.
        depth, heads, ff_mult, dropout: denoiser decoder size.
        n_steps: default Euler steps ``K`` at inference.
        tau_dist: ``uniform`` | ``beta`` (``Beta(1, tau_beta_b)``) training-time τ distribution.
        tau_beta_b: ``b ≥ 1`` of ``Beta(1, b)``.
        eval_seed: in eval mode, :meth:`loss` without explicit ``noise`` / ``tau`` / ``generator``
            draws them from a generator re-seeded with this value on every call, so validation
            losses (``Trainer.evaluate`` → best-checkpoint selection) compare epochs on identical
            ``(ε, τ)`` draws instead of fresh sampling noise. ``None`` → global RNG. Training mode
            always samples fresh draws.
    """

    kind = "flow"

    def __init__(self, d_model: int, action_dim: int, horizon: int, *, depth: int = 2,
                 heads: int = 4, ff_mult: int = 4, dropout: float = 0.0, n_steps: int = 10,
                 tau_dist: str = "uniform", tau_beta_b: float = 1.5, eval_seed: int | None = 0):
        super().__init__()
        if action_dim < 1 or horizon < 1:
            raise ValueError("action_dim and horizon must be >= 1")
        if int(n_steps) < 1:
            raise ValueError("n_steps must be >= 1")
        if tau_dist not in TAU_DISTS:
            raise ValueError(f"tau_dist must be one of {TAU_DISTS}, got {tau_dist!r}")
        if float(tau_beta_b) < 1.0:
            raise ValueError("tau_beta_b must be >= 1 (Beta(1, b) emphasising noisy τ)")
        self.d_model, self.action_dim, self.horizon = int(d_model), int(action_dim), int(horizon)
        self.n_steps, self.tau_dist, self.tau_beta_b = int(n_steps), tau_dist, float(tau_beta_b)
        self.eval_seed = None if eval_seed is None else int(eval_seed)
        self.in_proj = nn.Linear(self.action_dim, self.d_model)
        self.pos = nn.Parameter(torch.randn(self.horizon, self.d_model) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.SiLU(),
                                      nn.Linear(self.d_model, self.d_model))
        self.decoder = _CrossDecoder(self.d_model, depth, heads, ff_mult, dropout)
        self.out = nn.Linear(self.d_model, self.action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # ── the velocity field ────────────────────────────────────────────────────────────
    def velocity(self, x: torch.Tensor, tau: torch.Tensor, memory: torch.Tensor,
                 memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``v_θ(x_τ, τ, obs)``: ``x [B,H,A]``, ``tau [B]`` (or scalar) → ``[B,H,A]``."""
        _check_memory(memory, self.d_model)
        B = x.shape[0]
        if x.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"x must be [B,{self.horizon},{self.action_dim}], got {tuple(x.shape)}")
        tau = torch.as_tensor(tau, device=x.device).reshape(-1)
        if tau.numel() == 1:
            tau = tau.expand(B)
        temb = self.time_mlp(sinusoidal_embedding(tau, self.d_model).to(memory.dtype))
        h = self.in_proj(x.to(memory.dtype)) + self.pos.to(memory.dtype).unsqueeze(0) + temb[:, None]
        return self.out(self.decoder(h, memory, memory_mask))

    forward = velocity

    # ── training ──────────────────────────────────────────────────────────────────────
    def sample_tau(self, n: int, *, generator: torch.Generator | None = None,
                   device: torch.device | str | None = None) -> torch.Tensor:
        """``n`` training times τ from :attr:`tau_dist` (drawn on the CPU generator, then moved)."""
        u = torch.rand(n, generator=generator, dtype=torch.float32)
        if self.tau_dist == "beta":
            u = 1.0 - (1.0 - u).pow(1.0 / self.tau_beta_b)      # Beta(1, b) inverse CDF
        return u.to(device) if device is not None else u

    def _path(self, actions: torch.Tensor, noise: torch.Tensor | None, tau: torch.Tensor | None,
              generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = actions.float()
        if noise is None:
            noise = torch.randn(a.shape, generator=generator, dtype=torch.float32).to(a.device)
        if tau is None:
            tau = self.sample_tau(a.shape[0], generator=generator, device=a.device)
        tau = torch.as_tensor(tau, dtype=torch.float32, device=a.device).reshape(-1)
        if tau.numel() == 1:
            tau = tau.expand(a.shape[0])
        t3 = tau[:, None, None]
        x_tau = t3 * a + (1.0 - t3) * noise.to(a)
        return x_tau, tau, a - noise.to(a)

    def loss(self, memory: torch.Tensor, memory_mask: torch.Tensor | None, actions: torch.Tensor,
             valid: torch.Tensor | None = None, *, noise: torch.Tensor | None = None,
             tau: torch.Tensor | None = None, generator: torch.Generator | None = None,
             **_: object) -> dict[str, torch.Tensor]:
        """Conditional flow-matching loss ``masked_mse(v_θ(x_τ, τ), a − ε)``; ``noise`` / ``tau``
        may be given (tests, DPO surrogate), else drawn (``generator``; in eval mode a generator
        seeded with :attr:`eval_seed`; otherwise the global RNG)."""
        if (generator is None and not self.training and self.eval_seed is not None
                and (noise is None or tau is None)):
            generator = torch.Generator().manual_seed(self.eval_seed)
        x_tau, tau, u = self._path(actions, noise, tau, generator)
        v = self.velocity(x_tau, tau, memory, memory_mask).float()
        mse = masked_mse(v, u, valid)
        return {"action_loss": mse, "flow_mse": mse.detach()}

    def per_sample_error(self, memory: torch.Tensor, memory_mask: torch.Tensor | None,
                         actions: torch.Tensor, valid: torch.Tensor | None = None, *,
                         noise: torch.Tensor | None = None, tau: torch.Tensor | None = None,
                         generator: torch.Generator | None = None) -> torch.Tensor:
        """``[B]`` flow-matching error per sample (the DPO likelihood surrogate; pass the same
        ``noise``/``tau`` to every model being compared)."""
        x_tau, tau, u = self._path(actions, noise, tau, generator)
        v = self.velocity(x_tau, tau, memory, memory_mask).float()
        return _per_sample_mean((v - u).square(), valid)

    # ── inference ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, memory: torch.Tensor, memory_mask: torch.Tensor | None = None, *,
               n_steps: int | None = None, noise: torch.Tensor | None = None,
               generator: torch.Generator | None = None, **_: object) -> torch.Tensor:
        """Euler-integrate from ``x_0 = noise`` (τ = 0) to τ = 1 in ``n_steps`` → ``[B,H,A]``."""
        K = self.n_steps if n_steps is None else int(n_steps)
        if K < 1:
            raise ValueError("n_steps must be >= 1")
        B = memory.shape[0]
        shape = (B, self.horizon, self.action_dim)
        if noise is None:
            noise = torch.randn(shape, generator=generator, dtype=torch.float32)
        if tuple(noise.shape) != shape:
            raise ValueError(f"noise must be {shape}, got {tuple(noise.shape)}")
        x = noise.to(device=memory.device, dtype=torch.float32)
        dt = 1.0 / K
        for k in range(K):
            tau = torch.full((B,), k * dt, device=memory.device, dtype=torch.float32)
            x = x + dt * self.velocity(x, tau, memory, memory_mask).float()
        return x


def build_head(kind: str, d_model: int, action_dim: int, horizon: int, **kw) -> nn.Module:
    """``chunk`` → :class:`ChunkRegressionHead`, ``flow`` → :class:`FlowMatchingHead` (flow-only
    keys ``n_steps``/``tau_dist``/``tau_beta_b``/``eval_seed`` are dropped for the chunk head)."""
    if kind == "chunk":
        for k in ("n_steps", "tau_dist", "tau_beta_b", "eval_seed"):
            kw.pop(k, None)
        return ChunkRegressionHead(d_model, action_dim, horizon, **kw)
    if kind == "flow":
        return FlowMatchingHead(d_model, action_dim, horizon, **kw)
    raise ValueError(f"head must be one of {HEAD_TYPES}, got {kind!r}")

