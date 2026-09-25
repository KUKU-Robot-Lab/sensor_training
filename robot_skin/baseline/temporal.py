"""Temporal no-contact baseline: joint-state *history* → per-taxel ΔS mean and variance.

The motion artefact of a skin taxel (stretch / bending / air-pressure coupling) depends on the
joint angles **and** their recent history: barometric cavities and elastomer relax with a lag, so
the same pose reached quickly or slowly shows a different ΔS. :class:`TemporalBaselinePredictor`
therefore reads a causal window of the joint state ``q_hist, qd_hist [W,D]`` (a causal dilated
TCN or a GRU → one context vector) and, per taxel, the taxel pose at ``t`` plus a learned taxel
embedding, and returns the no-contact ΔS as a Gaussian ``(mean, logvar)`` per taxel.

Design rules
- **Never feed the observed ΔS** (or anything derived from it) into the model. This is the lesson
  of ``deformable_sats/sats/bending/baseline_restorer.py``: a baseline model that sees ΔS learns
  to explain contact as "offset" and erases it (the ``seq_deg`` collapse). Inputs are only the
  joint state and the taxel geometry; ``forward`` has no argument for ΔS.
- Mean head zero-initialised (identity warm start, as in the restorer: residual == observation
  before training); log-variance head bias initialised to ``log σ0²``.
- Heteroscedastic aleatoric uncertainty trained with the Gaussian negative log-likelihood
  (Kendall & Gal, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*,
  NeurIPS 2017, arXiv:1703.04977): ``½ (log σ² + (y − μ)² / σ²)``; the log-variance is soft-
  clamped to ``[logvar_min, logvar_max]`` so a few perfectly fitted samples cannot drive σ → 0.
  The contact stage turns the residual into a z-score with this σ (``contact.calibration``).
  :func:`baseline_loss` default: the **mean** is fitted with a scaled MSE and the NLL trains only
  the variance (stop-gradient on the mean, and — ``var_detach`` — on the features the log-variance
  head reads). Joint NLL training was measured to fit the mean much more slowly (the ``1/σ²``
  weighting and its large gradients fight the shared features, and gradient clipping then slows
  everything); ``mean_loss="none", detach_mean=False, var_detach=False`` restores the plain joint
  Kendall & Gal objective.
- Per-taxel heads are FiLM-style: taxel features (pose at ``t`` + embedding) scale and shift the
  shared joint context, so each taxel can read its own combination of joints (the artefact of a
  fingertip taxel is driven by its own finger's joints).
- Normalisation travels with the model: ``q``/``qd`` statistics and the per-taxel target scale are
  buffers (:meth:`set_joint_stats`, :meth:`set_target_scale`), so ``forward`` takes **raw units**
  (rad, rad/s, m) and returns ΔS in % — a checkpoint is self-contained for the online processor.

Causal windowing contract (offline :func:`predict_episode` ≡ online :class:`CausalBaselineStream`)
- The prediction for master-clock frame ``t`` uses ``q[t−W+1 … t]`` and ``qd[t−W+1 … t]`` (oldest
  first); indices < 0 are replaced by frame 0 (edge padding — the stream repeats its first sample),
  and ``pos[t]`` / ``nrm[t]`` (taxel poses in the hand / robot base frame at ``t``).
- ``q``/``qd`` are raw episode units, exactly as in the episode arrays: glove ``q`` = MANO finger
  pose flattened (45, ``datasets.build.HAND_Q_NAMES``), robot ``q`` = URDF joint order; ``qd`` =
  ``datasets.build.joint_velocity(q, hz, **qd_settings(meta.preprocessing))`` (default causal
  Savitzky–Golay, 50 ms; the stage stores these kwargs as ``bundle_meta["qd"]``) — online, compute
  ``qd`` with the same function over the ring buffer of ``q`` (unless ``bundle_meta["qd_source"]``
  is ``"file"``: robot sessions preprocessed with the driver's joint velocities).
- No state beyond the window: the network is a pure function of the window (TCN convolutions are
  zero-padded *inside* the window, normalisation is per time step; with the default depth the
  receptive field 1 + (k−1)(2^L − 1) = 31 ≤ W, so the output never sees that padding).
- ``W`` = ``model.window`` (stored in the checkpoint).

Related work: Yu et al., *Pose-Aware Modeling to Mitigate Pose-Related Artifacts in Tactile
Gloves* (arXiv:2607.22964) addresses the same no-contact artefact problem (see docs/REFERENCES.md).
The v1 :class:`~robot_skin.baseline.model.BaselinePredictor` (instantaneous MLP) is unchanged.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.signal import NormStats

__all__ = [
    "BASELINE_FORMAT", "BASELINE_MODEL_NAME", "Q_SOURCES", "MEAN_LOSSES", "TemporalBaselinePredictor",
    "gaussian_nll", "baseline_loss", "soft_clamp", "causal_windows", "predict_episode", "CausalBaselineStream",
    "qd_settings", "episode_joint_view", "save_baseline_model", "load_baseline_model",
]
#: the ``joint_velocity`` keyword arguments of the preprocessing ``qd`` config section
QD_KWARGS = ("method", "window_s", "polyorder")

BASELINE_FORMAT = "robot_skin.baseline.temporal/1"
BASELINE_MODEL_NAME = "baseline_model.pt"
#: joint-state sources: episode ``q``/``qd`` (vision hand pose / robot joints) or the IMU pose
#: model's derived ``hand_finger_pose_imu`` (what a camera-free glove deployment has)
Q_SOURCES = ("q", "hand_pose_imu")
MEAN_LOSSES = ("mse", "huber", "none")
_ARCHES = ("tcn", "gru")


# ─────────────────────────────────────────────────────────────── helpers

def soft_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Smooth clamp: ≈ identity inside ``[lo, hi]``, saturating softly outside — the gradient
    decays smoothly past the bounds instead of being cut to zero there like ``torch.clamp``'s."""
    return lo + F.softplus(x - lo) - F.softplus(x - hi)


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor,
                 valid: torch.Tensor | None = None, *, logvar_min: float = -12.0,
                 logvar_max: float = 8.0, include_const: bool = False) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL (Kendall & Gal 2017): ``½ (logvar + (y − mean)² e^{−logvar})``
    averaged over ``valid`` entries (bool / float weights, broadcastable to ``mean``).

    ``logvar`` is hard-clamped to ``[logvar_min, logvar_max]`` as a numerical guard (the model
    already soft-clamps its output). ``include_const`` adds ``½ log 2π``. Returns 0 (with a
    gradient path) when nothing is valid.
    """
    lv = logvar.clamp(logvar_min, logvar_max)
    nll = 0.5 * (lv + (y - mean) ** 2 * torch.exp(-lv))
    if include_const:
        nll = nll + 0.5 * math.log(2.0 * math.pi)
    if valid is None:
        return nll.mean()
    w = valid.to(nll.dtype).expand_as(nll)
    return (nll * w).sum() / w.sum().clamp_min(1.0)


def baseline_loss(mean: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor, valid: torch.Tensor,
                  y_scale: torch.Tensor | None = None, *, mean_loss: str = "mse", mean_weight: float = 1.0,
                  nll_weight: float = 1.0, detach_mean: bool = True,
                  huber_delta: float = 1.0) -> dict[str, torch.Tensor]:
    """Training loss of :class:`TemporalBaselinePredictor` on no-contact targets.

    All terms are computed in per-taxel **target-scale units** (``y / y_scale``, ``logvar − 2 log
    y_scale``; default scale 1), so taxels with large and small artefacts weigh alike; the NLL then
    differs from the %-unit NLL only by a constant. ``loss = mean_weight · {mse | huber}(mean, y) +
    nll_weight · gaussian_nll(mean[.detach()], logvar, y)`` over ``valid`` entries. Also returns
    ``nll``, ``mean_loss`` and ``mae`` (in %, for logging)."""
    if mean_loss not in MEAN_LOSSES:
        raise ValueError(f"mean_loss must be one of {MEAN_LOSSES}, got {mean_loss!r}")
    v = valid.to(mean.dtype)
    n = v.sum().clamp_min(1.0)
    s = torch.ones_like(mean[..., :1]) if y_scale is None else y_scale.to(mean.dtype)
    ms, ys = mean / s, y / s
    lvs = logvar - 2.0 * torch.log(s)
    nll = gaussian_nll(ms.detach() if detach_mean else ms, lvs, ys, valid, logvar_min=-30.0, logvar_max=30.0)
    err = ms - ys
    if mean_loss == "mse":
        ml = (err ** 2 * v).sum() / n
    elif mean_loss == "huber":
        ml = (F.huber_loss(ms, ys, reduction="none", delta=float(huber_delta)) * v).sum() / n
    else:
        ml = torch.zeros((), dtype=mean.dtype, device=mean.device)
    loss = float(nll_weight) * nll + float(mean_weight) * ml
    mae = (((mean - y).abs() * v).sum() / n).detach()
    return {"loss": loss, "nll": nll.detach(), "mean_loss": ml.detach(), "mae": mae}


def causal_windows(x: np.ndarray, window: int, index: np.ndarray | None = None) -> np.ndarray:
    """Causal edge-padded windows of ``x[T, ...]`` → ``[len(index) or T, W, ...]`` (row t = frames
    ``t−W+1 … t``, indices < 0 replaced by frame 0). A strided view when ``index`` is None."""
    x = np.asarray(x)
    if x.ndim < 1 or x.shape[0] == 0:
        raise ValueError(f"x must be non-empty [T, ...], got {x.shape}")
    if int(window) < 1:
        raise ValueError("window must be ≥ 1")
    W = int(window)
    if index is None:
        pad = np.concatenate([np.repeat(x[:1], W - 1, axis=0), x], axis=0)
        v = np.lib.stride_tricks.sliding_window_view(pad, W, axis=0)          # [T, ..., W]
        return np.moveaxis(v, -1, 1)
    idx = np.asarray(index, dtype=np.int64)
    rows = np.clip(idx[:, None] + np.arange(-W + 1, 1)[None, :], 0, None)     # [B, W]
    return x[rows]


def _as_stats(stats: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if stats is None:
        return None
    if isinstance(stats, NormStats):
        return np.asarray(stats.offset, np.float32), np.asarray(stats.scale, np.float32)
    if isinstance(stats, Mapping):
        return np.asarray(stats["offset"], np.float32), np.asarray(stats["scale"], np.float32)
    off, sc = stats
    return np.asarray(off, np.float32), np.asarray(sc, np.float32)


class _CausalConvBlock(nn.Module):
    """Residual causal dilated conv: ``x + Conv₁(GELU(Conv_k(GELU(LN x))))`` over ``[B, C, W]``;
    the LayerNorm is per time step, so step k depends on steps ≤ k only."""

    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = F.pad(F.gelu(h), (self.pad, 0))                                     # left pad = causal
        h = self.conv2(self.drop(F.gelu(self.conv1(h))))
        return x + h


class CausalTCN(nn.Module):
    """Stack of residual causal dilated convolutions (dilations 1, 2, 4, …) over ``[B, W, C]``.
    Receptive field ``1 + (kernel − 1)(2^L − 1)`` frames."""

    def __init__(self, dim: int, n_layers: int, kernel: int, dropout: float = 0.0):
        super().__init__()
        if kernel < 1 or n_layers < 1:
            raise ValueError("kernel and n_layers must be ≥ 1")
        self.blocks = nn.ModuleList(_CausalConvBlock(dim, kernel, 2 ** i, dropout) for i in range(n_layers))
        self.receptive_field = 1 + (kernel - 1) * (2 ** n_layers - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                        # [B, W, C]
        h = x.transpose(1, 2)
        for b in self.blocks:
            h = b(h)
        return h.transpose(1, 2)


# ─────────────────────────────────────────────────────────────── model

class TemporalBaselinePredictor(nn.Module):
    """No-contact ΔS ``(mean, logvar) [B,N]`` from a causal joint-state window (module docstring).

    Args:
        n_taxels: N (taxel embedding table; the model is tied to one layout's taxel order).
        joint_dim: D (``q`` columns).
        window: W frames of history (200 Hz master clock: 32 → 160 ms).
        hidden: width of the temporal encoder.
        taxel_emb_dim: learned per-taxel embedding size (0 disables it).
        kernel / n_layers / arch: TCN kernel and depth (``arch="tcn"``) or GRU layers (``"gru"``).
        head_hidden: width of the per-taxel FiLM head.
        pos_scale: taxel positions (m) are multiplied by this before the head (10 → dm ≈ O(1)).
        use_pose: feed ``pos``/``nrm`` to the head (False: embedding only).
        sigma0: initial σ in target-scale units (the log-variance bias is ``log σ0²``).
        logvar_min / logvar_max: soft clamp of the log-variance (target-scale units).
        var_detach: the log-variance head reads stop-gradient features (the variance is fitted
            on top of the mean's representation without disturbing it; see module docstring).
    """

    def __init__(self, n_taxels: int, joint_dim: int, *, window: int = 32, hidden: int = 128,
                 taxel_emb_dim: int = 8, kernel: int = 3, n_layers: int = 4, arch: str = "tcn",
                 head_hidden: int = 64, dropout: float = 0.0, pos_scale: float = 10.0,
                 use_pose: bool = True, sigma0: float = 1.0, logvar_min: float = -12.0,
                 logvar_max: float = 8.0, var_detach: bool = True) -> None:
        super().__init__()
        if arch not in _ARCHES:
            raise ValueError(f"arch must be one of {_ARCHES}, got {arch!r}")
        if min(n_taxels, joint_dim, window, hidden, head_hidden, n_layers) < 1:
            raise ValueError("n_taxels, joint_dim, window, hidden, head_hidden, n_layers must be ≥ 1")
        if not logvar_min < logvar_max or sigma0 <= 0:
            raise ValueError("need logvar_min < logvar_max and sigma0 > 0")
        self.config = dict(n_taxels=int(n_taxels), joint_dim=int(joint_dim), window=int(window),
                           hidden=int(hidden), taxel_emb_dim=int(taxel_emb_dim), kernel=int(kernel),
                           n_layers=int(n_layers), arch=arch, head_hidden=int(head_hidden),
                           dropout=float(dropout), pos_scale=float(pos_scale), use_pose=bool(use_pose),
                           sigma0=float(sigma0), logvar_min=float(logvar_min), logvar_max=float(logvar_max),
                           var_detach=bool(var_detach))
        self.n_taxels, self.joint_dim, self.window = int(n_taxels), int(joint_dim), int(window)
        self.arch, self.use_pose, self.pos_scale = arch, bool(use_pose), float(pos_scale)
        self.logvar_min, self.logvar_max = float(logvar_min), float(logvar_max)
        self.var_detach = bool(var_detach)
        D = self.joint_dim
        self.register_buffer("q_mean", torch.zeros(D))
        self.register_buffer("q_std", torch.ones(D))
        self.register_buffer("qd_mean", torch.zeros(D))
        self.register_buffer("qd_std", torch.ones(D))
        self.register_buffer("y_scale", torch.ones(self.n_taxels))

        self.inp = nn.Linear(2 * D, hidden)
        if arch == "tcn":
            self.core: nn.Module = CausalTCN(hidden, n_layers, kernel, dropout)
            self.receptive_field = int(self.core.receptive_field)
        else:
            self.core = nn.GRU(hidden, hidden, n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
            self.receptive_field = self.window
        self.ctx_norm = nn.LayerNorm(hidden)
        self.ctx = nn.Linear(hidden, head_hidden)
        self.taxel_emb = nn.Embedding(self.n_taxels, taxel_emb_dim) if taxel_emb_dim > 0 else None
        t_in = (6 if self.use_pose else 0) + max(0, int(taxel_emb_dim))
        if t_in == 0:
            raise ValueError("the per-taxel head needs use_pose=True or taxel_emb_dim > 0")
        # default (random) init: taxels differ from the first step — with a zero-init FiLM *and* a
        # zero-init mean head every taxel would get the same gradient until γ/β move (slow start)
        self.taxel_mlp = nn.Sequential(nn.Linear(t_in, head_hidden), nn.SiLU(),
                                       nn.Linear(head_hidden, 2 * head_hidden))
        self.trunk = nn.Sequential(nn.Linear(head_hidden, head_hidden), nn.SiLU(), nn.Dropout(dropout))
        self.mean_head = nn.Linear(head_hidden, 1)
        nn.init.zeros_(self.mean_head.weight)               # identity warm start: prediction 0
        nn.init.zeros_(self.mean_head.bias)
        self.logvar_head = nn.Linear(head_hidden, 1)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, 2.0 * math.log(sigma0))

    # ── normalisation buffers ─────────────────────────────────────────────
    def set_joint_stats(self, q_stats: Any, qd_stats: Any, eps: float = 1e-6) -> None:
        """Store ``q``/``qd`` normalisation (``NormStats``, ``{"offset","scale"}`` or a tuple)."""
        for (m, s), (bm, bs) in ((_as_stats(q_stats), (self.q_mean, self.q_std)),
                                 (_as_stats(qd_stats), (self.qd_mean, self.qd_std))):
            if m.shape != (self.joint_dim,) or s.shape != (self.joint_dim,):
                raise ValueError(f"joint stats must have {self.joint_dim} entries, got {m.shape}/{s.shape}")
            bm.copy_(torch.as_tensor(m))
            bs.copy_(torch.as_tensor(s).clamp_min(eps))

    def set_target_scale(self, scale: Any, floor: float = 1e-3) -> None:
        """Per-taxel ΔS scale (%): the heads predict ``mean / scale`` and ``logvar − 2 log scale``,
        which balances taxels with very different artefact gains."""
        s = torch.as_tensor(np.asarray(scale, dtype=np.float32)).reshape(-1)
        if s.numel() != self.n_taxels or not torch.isfinite(s).all():
            raise ValueError(f"target scale must be {self.n_taxels} finite values")
        self.y_scale.copy_(s.clamp_min(floor))

    def joint_stats_dict(self) -> dict[str, NormStats]:
        return {"q": NormStats(self.q_mean.cpu().numpy(), self.q_std.cpu().numpy()),
                "qd": NormStats(self.qd_mean.cpu().numpy(), self.qd_std.cpu().numpy())}

    # ── forward ───────────────────────────────────────────────────────────
    def encode_joints(self, q_hist: torch.Tensor, qd_hist: torch.Tensor) -> torch.Tensor:
        """Joint-history context ``[B, hidden]`` (last step of the causal encoder)."""
        if q_hist.ndim != 3 or q_hist.shape[-1] != self.joint_dim or qd_hist.shape != q_hist.shape:
            raise ValueError(f"q_hist/qd_hist must both be [B, W, {self.joint_dim}], got "
                             f"{tuple(q_hist.shape)} / {tuple(qd_hist.shape)}")
        dt = self.q_mean.dtype
        x = torch.cat([(q_hist.to(dt) - self.q_mean) / self.q_std,
                       (qd_hist.to(dt) - self.qd_mean) / self.qd_std], dim=-1)
        h = self.inp(x)
        h = self.core(h) if self.arch == "tcn" else self.core(h)[0]
        return self.ctx_norm(h[:, -1])

    def forward(self, q_hist: torch.Tensor, qd_hist: torch.Tensor, pos: torch.Tensor,
                nrm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``q_hist, qd_hist [B,W,D]`` (raw units, oldest first), ``pos, nrm [B,N,3]`` (at the last
        frame) → ``(mean [B,N] ΔS %, logvar [B,N] of ΔS in %²)``. No ΔS input by design."""
        B, N = pos.shape[0], pos.shape[1]
        if N != self.n_taxels or pos.shape != nrm.shape or pos.shape[-1] != 3:
            raise ValueError(f"pos/nrm must be [B, {self.n_taxels}, 3], got {tuple(pos.shape)}/{tuple(nrm.shape)}")
        c = self.ctx(self.encode_joints(q_hist, qd_hist))                    # [B, Hh]
        dt = c.dtype
        parts = []
        if self.use_pose:
            parts += [pos.to(dt) * self.pos_scale, nrm.to(dt)]
        if self.taxel_emb is not None:
            parts.append(self.taxel_emb.weight.to(dt).unsqueeze(0).expand(B, N, -1))
        gamma, beta = self.taxel_mlp(torch.cat(parts, dim=-1)).chunk(2, dim=-1)  # [B, N, Hh]
        h = self.trunk(F.silu(c.unsqueeze(1) * (1.0 + gamma) + beta))
        scale = self.y_scale.to(dt)
        mean = self.mean_head(h).squeeze(-1) * scale
        hv = h.detach() if self.var_detach else h
        lv = soft_clamp(self.logvar_head(hv).squeeze(-1), self.logvar_min, self.logvar_max)
        return mean, lv + 2.0 * torch.log(scale)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "TemporalBaselinePredictor":
        cfg = dict(cfg)
        return cls(cfg.pop("n_taxels"), cfg.pop("joint_dim"), **cfg)


# ─────────────────────────────────────────────────────────────── offline / online inference

def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def predict_episode(model: TemporalBaselinePredictor, episode, joint_stats: Any = None, window: int | None = None,
                    batch_size: int = 2048, *, device: str | torch.device | None = None,
                    q_source: str = "q") -> tuple[np.ndarray, np.ndarray]:
    """Causal per-frame prediction over a whole episode → ``(mean [T,N], logvar [T,N])`` float32.

    Frame ``t`` sees the window ``t−W+1 … t`` of ``q``/``qd`` (edge-padded at the start) and the
    taxel pose at ``t`` (module docstring). ``window`` defaults to ``model.window``. ``joint_stats``
    (``{"q","qd"}`` NormStats) is only for models trained on *externally* normalised inputs — the
    stage-trained models normalise internally, pass None. ``q_source``: see
    :func:`episode_joint_view` (``"q"`` or ``"hand_pose_imu"``). Every frame gets a prediction,
    contact or not (glove frames without a valid hand label use the held pose)."""
    ep = episode_joint_view(episode, q_source)
    W = int(window or model.window)
    q = np.asarray(ep["q"], dtype=np.float32)
    qd = np.asarray(ep["qd"], dtype=np.float32)
    pos = np.asarray(ep["taxel_pos"], dtype=np.float32)
    nrm = np.asarray(ep["taxel_nrm"], dtype=np.float32)
    js = joint_stats if joint_stats is None or not isinstance(joint_stats, tuple) else \
        {"q": joint_stats[0], "qd": joint_stats[1]}
    if js is not None:
        qs, qds = _as_stats(js["q"]), _as_stats(js["qd"])
        q, qd = (q - qs[0]) / qs[1], (qd - qds[0]) / qds[1]
    qw, qdw = causal_windows(q, W), causal_windows(qd, W)
    dev = torch.device(device) if device is not None else _model_device(model)
    was = model.training
    model.eval()
    means, lvs = [], []
    try:
        for s in range(0, q.shape[0], max(1, int(batch_size))):
            e = s + max(1, int(batch_size))
            args = [torch.from_numpy(np.array(a[s:e], dtype=np.float32)).to(dev)     # copy: memmaps are read-only
                    for a in (qw, qdw, pos, nrm)]
            m, lv = model(*args)
            means.append(m.float().cpu().numpy())
            lvs.append(lv.float().cpu().numpy())
    finally:
        model.train(was)
    return np.concatenate(means).astype(np.float32), np.concatenate(lvs).astype(np.float32)


class CausalBaselineStream:
    """Online counterpart of :func:`predict_episode`: push one frame per master-clock tick.

    Keeps a ring buffer of the last ``W`` ``q``/``qd`` rows; before ``W`` frames were pushed the
    buffer is edge-padded with the first frame, exactly like the offline windows, so
    ``push`` at tick ``t`` returns the same ``(mean[N], logvar[N])`` as row ``t`` of
    :func:`predict_episode` (tested). Call :meth:`reset` at every episode / session start."""

    def __init__(self, model: TemporalBaselinePredictor, *, window: int | None = None,
                 device: str | torch.device | None = None):
        self.model = model
        self.window = int(window or model.window)
        self.device = torch.device(device) if device is not None else _model_device(model)
        self.reset()

    def reset(self) -> None:
        self._q: np.ndarray | None = None
        self._qd: np.ndarray | None = None
        self.n_pushed = 0

    @torch.no_grad()
    def push(self, q: np.ndarray, qd: np.ndarray, pos: np.ndarray, nrm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        qd = np.asarray(qd, dtype=np.float32).reshape(-1)
        if self._q is None:
            self._q = np.repeat(q[None], self.window, axis=0)
            self._qd = np.repeat(qd[None], self.window, axis=0)
        else:
            self._q = np.concatenate([self._q[1:], q[None]], axis=0)
            self._qd = np.concatenate([self._qd[1:], qd[None]], axis=0)
        self.n_pushed += 1
        t = lambda a: torch.from_numpy(np.array(a, dtype=np.float32)[None]).to(self.device)  # noqa: E731 (copy)
        was = self.model.training
        self.model.eval()
        try:
            m, lv = self.model(t(self._q), t(self._qd), t(pos), t(nrm))
        finally:
            self.model.train(was)
        return m[0].float().cpu().numpy(), lv[0].float().cpu().numpy()


# ─────────────────────────────────────────────────────────────── joint-state source

def qd_settings(preprocessing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``joint_velocity`` keyword arguments (``method``, ``window_s``, ``polyorder``) of an episode's
    preprocessing: ``datasets.build.DEFAULTS["qd"]`` ⊕ ``meta.preprocessing["config"]["qd"]``,
    without the non-kwarg ``source`` key — ``joint_velocity(q, hz, **qd_settings(pre))`` reproduces
    the episode ``qd`` (when it was differentiated, i.e. ``qd_source`` ≠ ``"file"``)."""
    from ..datasets.build import DEFAULTS as PRE_DEFAULTS

    cfg = ((preprocessing or {}).get("config") or {}).get("qd") or {}
    qc = {**PRE_DEFAULTS["qd"], **cfg}
    return {k: qc[k] for k in QD_KWARGS}


def _preprocessing_skeleton(preprocessing: Mapping[str, Any] | None):
    """The MANO skeleton preprocessing used for taxel poses (``hand_pose.mano_model`` or the default)."""
    from ..pose.mano import ManoSkeleton

    path = (((preprocessing or {}).get("config") or {}).get("hand_pose") or {}).get("mano_model")
    return ManoSkeleton.from_mano_pkl(path) if path else ManoSkeleton()


def episode_joint_view(episode, q_source: str = "q", *, skeleton=None):
    """The episode with ``q``/``qd``/``taxel_pos``/``taxel_nrm`` taken from ``q_source``.

    ``"q"``: the episode itself (vision finger pose for gloves, joint state for robots).
    ``"hand_pose_imu"``: an in-memory view whose ``q`` is the IMU pose model's derived
    ``hand_finger_pose_imu`` (flattened to 45, ``HAND_Q_NAMES`` order), ``qd`` =
    ``datasets.build.joint_velocity`` with the episode's preprocessing ``qd`` settings
    (:func:`qd_settings`), and taxel poses recomputed from that pose in the hand frame
    (``pose.mano.taxel_poses_from_hand`` with ``global_orient = 0``, wrist at the origin, and the
    preprocessing skeleton unless ``skeleton`` is given — the preprocessing convention). This is what a
    camera-free glove deployment sees; every frame is a measurement (``q_source`` is recorded as
    ``hand_pose_imu`` in ``meta.preprocessing``, so ``hand_pose_valid`` no longer gates ``q``).
    The view shares the episode's ``root`` (derived arrays are readable) — write stage outputs
    through the original episode."""
    from ..datasets.episode import D_HAND_POSE_IMU, K_Q, K_QD, K_TAXEL_NRM, K_TAXEL_POS, Episode

    if q_source not in Q_SOURCES:
        raise ValueError(f"q_source must be one of {Q_SOURCES}, got {q_source!r}")
    if q_source == "q":
        return episode
    ep = episode
    if (ep.meta.preprocessing or {}).get("q_source") == "hand_pose_imu":
        return ep                                                          # already a view
    if not ep.has_derived(D_HAND_POSE_IMU):
        raise ValueError(f"episode {ep.meta.episode_id!r} has no derived {D_HAND_POSE_IMU!r} "
                         "(run the imu_pose stage first)")
    from ..datasets.build import HAND_Q_NAMES, joint_velocity, load_episode_layout
    from ..pose.mano import taxel_poses_from_hand

    fp = np.asarray(ep.derived(D_HAND_POSE_IMU), dtype=np.float64)
    if fp.shape != (ep.T, 15, 3):
        raise ValueError(f"derived {D_HAND_POSE_IMU!r} must be [T,15,3], got {fp.shape}")
    pre = dict(ep.meta.preprocessing or {})
    q = fp.reshape(ep.T, 45).astype(np.float32)
    qd = joint_velocity(q, float(ep.meta.hz), **qd_settings(pre))
    layout = load_episode_layout(ep)
    sk = skeleton if skeleton is not None else _preprocessing_skeleton(pre)
    pos, nrm = taxel_poses_from_hand(layout, sk, np.zeros((ep.T, 3)), fp, None)
    arrays = dict(ep.arrays)
    arrays.update({K_Q: q, K_QD: qd, K_TAXEL_POS: pos.astype(np.float32), K_TAXEL_NRM: nrm.astype(np.float32)})
    pre.update(q_source="hand_pose_imu", qd_source="derivative", taxel_pose_source="hand_pose_imu")
    meta = dataclasses.replace(ep.meta, joint_names=list(HAND_Q_NAMES), preprocessing=pre)
    return Episode(meta, arrays, ep.static, root=ep.root)


# ─────────────────────────────────────────────────────────────── checkpoint bundle


def _plain(obj: Any) -> Any:
    """JSON-plain copy of bundle meta (numpy scalars / arrays → Python), loadable with
    ``torch.load(weights_only=True)``."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_baseline_model(path: str | Path, model: TemporalBaselinePredictor,
                        meta: Mapping[str, Any] | None = None) -> Path:
    """Atomic ``{format, config, state_dict, meta}`` bundle (``meta``: q_source, qd settings,
    joint names, layout, metrics …). Reload with :func:`load_baseline_model`."""
    from ..train.checkpoint import save_checkpoint

    p = Path(path)
    if p.suffix != ".pt":
        p = p / BASELINE_MODEL_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    save_checkpoint(p, format=BASELINE_FORMAT, config=dict(model.config), state_dict=sd,
                    meta=_plain(dict(meta or {})))
    return p


def load_baseline_model(path: str | Path, *, map_location: str | torch.device = "cpu",
                        strict: bool = True) -> TemporalBaselinePredictor:
    """Rebuild a :class:`TemporalBaselinePredictor` from :func:`save_baseline_model` output (a
    ``.pt`` file or its directory); the bundle ``meta`` is attached as ``model.bundle_meta``."""
    p = Path(path)
    if p.is_dir():
        p = p / BASELINE_MODEL_NAME
    ck = torch.load(p, map_location=map_location, weights_only=True)
    if not isinstance(ck, Mapping) or ck.get("format") != BASELINE_FORMAT:
        raise ValueError(f"{p} is not a {BASELINE_FORMAT} bundle")
    model = TemporalBaselinePredictor.from_config(ck["config"])
    model.load_state_dict(ck["state_dict"], strict=strict)
    model.bundle_meta = dict(ck.get("meta") or {})
    model.eval()
    return model
