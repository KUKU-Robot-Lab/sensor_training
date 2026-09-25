"""ContactDetector — learned per-taxel contact probability from the calibrated residual history.

Thresholding the calibrated z (``contact.calibration``) already separates presses from motion
artefacts, but a press has a *shape* in time (a fast drop that holds) while residual model errors
during fast motion are brief and correlated with joint speed. The detector looks at each taxel's
causal history ``z[t−W+1 … t]`` (+ its saturation flags), a joint-speed summary and an optional
taxel embedding and outputs a contact logit::

    per taxel: [asinh(clip(z)/z_scale), sat] (W steps) ─ causal dilated conv (shared across taxels) ─┐
    |q̇| summary (RMS, max of the normalised joint speed; 0 when q is not a measurement) ─ Linear ─┤─ MLP → logit
    taxel embedding (optional; ties the model to one layout) ─ Linear ────────────────────────────────┘

Trained on D1 labels (``contact_label``: self-touch = 1, no-contact = 0, −1 ignored) with the
focal loss (Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017, arXiv:1708.02002;
``FL = −α_t (1 − p_t)^γ log p_t``) or BCE with ``pos_weight``; the output bias is initialised to
the prior ``π`` (``b = −log((1 − π)/π)``, the same paper) so early training is not swamped by the
many easy negatives.

Causal windowing (offline :func:`predict_contact_prob` ≡ online :class:`CausalDetectorStream`):
frame ``t`` sees ``z[t−W+1 … t]`` and ``saturated[t−W+1 … t]`` (indices < 0 → frame 0, edge
padding), and ``qd[t]`` (raw units; the model normalises with its ``qd`` buffers) with the
``q_valid[t]`` flag. Non-finite z are fed as 0.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.signal import NormStats

__all__ = ["DETECTOR_FORMAT", "DETECTOR_MODEL_NAME", "LOSS_KINDS", "ContactDetector", "focal_loss",
           "contact_loss", "predict_contact_prob", "predict_episode", "CausalDetectorStream",
           "save_detector", "load_detector"]

DETECTOR_FORMAT = "robot_skin.contact.detector/1"
DETECTOR_MODEL_NAME = "contact_detector.pt"
LOSS_KINDS = ("focal", "bce")


def _causal_windows(x: np.ndarray, window: int) -> np.ndarray:
    pad = np.concatenate([np.repeat(x[:1], window - 1, axis=0), x], axis=0)
    return np.moveaxis(np.lib.stride_tricks.sliding_window_view(pad, window, axis=0), -1, 1)


class _TemporalConv(nn.Module):
    """Residual causal dilated 1-D convs over ``[B, C, W]`` (dilations 1, 2, 4, …)."""

    def __init__(self, c_in: int, hidden: int, n_layers: int, kernel: int, dropout: float):
        super().__init__()
        self.inp = nn.Conv1d(c_in, hidden, 1)
        self.convs = nn.ModuleList(nn.Conv1d(hidden, hidden, kernel, dilation=2 ** i) for i in range(n_layers))
        self.pads = [(kernel - 1) * 2 ** i for i in range(n_layers)]
        self.drop = nn.Dropout(dropout)
        self.receptive_field = 1 + (kernel - 1) * (2 ** n_layers - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for conv, pad in zip(self.convs, self.pads):
            h = h + self.drop(F.gelu(conv(F.pad(h, (pad, 0)))))
        return h


class ContactDetector(nn.Module):
    """Per-taxel temporal contact classifier (module docstring).

    Args:
        n_taxels: enables a per-taxel embedding when given (layout-specific); None → shared model
            usable on any layout.
        window: W frames of residual history.
        hidden: conv / head width.
        joint_dim: D of ``qd`` (needed for the speed summary; its normalisation buffers).
        use_motion / use_sat: include the joint-speed summary / the saturation flags.
        z_scale / z_clip: input compression ``asinh(clip(z, ±z_clip) / z_scale)``.
        prior: initial contact probability of the output bias.
    """

    def __init__(self, n_taxels: int | None = None, window: int = 16, hidden: int = 32, *,
                 joint_dim: int | None = None, n_layers: int = 3, kernel: int = 3, taxel_emb_dim: int = 4,
                 use_motion: bool = True, use_sat: bool = True, z_scale: float = 2.0, z_clip: float = 100.0,
                 prior: float = 0.01, dropout: float = 0.0) -> None:
        super().__init__()
        if int(window) < 1 or int(hidden) < 1 or int(n_layers) < 1 or int(kernel) < 1:
            raise ValueError("window, hidden, n_layers and kernel must be ≥ 1")
        if use_motion and not joint_dim:
            raise ValueError("use_motion needs joint_dim")
        if not 0.0 < prior < 1.0 or z_scale <= 0 or z_clip <= 0:
            raise ValueError("need 0 < prior < 1, z_scale > 0, z_clip > 0")
        emb = int(taxel_emb_dim) if n_taxels else 0
        self.config = dict(n_taxels=None if not n_taxels else int(n_taxels), window=int(window), hidden=int(hidden),
                           joint_dim=None if not joint_dim else int(joint_dim), n_layers=int(n_layers),
                           kernel=int(kernel), taxel_emb_dim=emb, use_motion=bool(use_motion),
                           use_sat=bool(use_sat), z_scale=float(z_scale), z_clip=float(z_clip),
                           prior=float(prior), dropout=float(dropout))
        self.n_taxels = self.config["n_taxels"]
        self.window, self.use_motion, self.use_sat = int(window), bool(use_motion), bool(use_sat)
        self.z_scale, self.z_clip = float(z_scale), float(z_clip)
        D = int(joint_dim or 1)
        self.joint_dim = self.config["joint_dim"]
        self.register_buffer("qd_mean", torch.zeros(D))
        self.register_buffer("qd_std", torch.ones(D))
        self.conv = _TemporalConv(2 if use_sat else 1, hidden, n_layers, kernel, dropout)
        self.receptive_field = self.conv.receptive_field
        self.motion = nn.Linear(3, hidden) if use_motion else None
        self.taxel_emb = nn.Embedding(int(n_taxels), emb) if emb > 0 else None
        self.emb_proj = nn.Linear(emb, hidden) if emb > 0 else None
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, -math.log((1.0 - prior) / prior))   # Lin et al. 2017 prior init

    def set_joint_stats(self, qd_stats: Any, eps: float = 1e-6) -> None:
        """``qd`` normalisation (``NormStats`` / ``{"offset","scale"}`` / tuple) for the speed summary."""
        if isinstance(qd_stats, NormStats):
            m, s = qd_stats.offset, qd_stats.scale
        elif isinstance(qd_stats, Mapping):
            m, s = qd_stats["offset"], qd_stats["scale"]
        else:
            m, s = qd_stats
        m = torch.as_tensor(np.asarray(m, dtype=np.float32)).reshape(-1)
        s = torch.as_tensor(np.asarray(s, dtype=np.float32)).reshape(-1)
        if m.numel() != self.qd_mean.numel() or s.numel() != self.qd_std.numel():
            raise ValueError(f"qd stats must have {self.qd_mean.numel()} entries")
        # only the scale matters for a speed summary; the offset is kept for completeness
        self.qd_mean.copy_(m)
        self.qd_std.copy_(s.clamp_min(eps))

    def z_feature(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.asinh(z.clamp(-self.z_clip, self.z_clip) / self.z_scale)

    def forward(self, z_hist: torch.Tensor, sat_hist: torch.Tensor | None = None, qd: torch.Tensor | None = None,
                q_valid: torch.Tensor | None = None) -> torch.Tensor:
        """``z_hist [B,W,N]`` (press-positive calibrated z, oldest first), ``sat_hist [B,W,N]``,
        ``qd [B,D]`` (raw units at the last frame), ``q_valid [B]`` → logits ``[B,N]``."""
        if z_hist.ndim != 3:
            raise ValueError(f"z_hist must be [B, W, N], got {tuple(z_hist.shape)}")
        B, W, N = z_hist.shape
        if self.n_taxels is not None and N != self.n_taxels:
            raise ValueError(f"expected {self.n_taxels} taxels, got {N}")
        dt = self.head[-1].weight.dtype
        feats = [self.z_feature(z_hist.to(dt))]
        if self.use_sat:
            s = torch.zeros_like(feats[0]) if sat_hist is None else sat_hist.to(dt)
            feats.append(s)
        x = torch.stack(feats, dim=-1)                                       # [B, W, N, C]
        x = x.permute(0, 2, 3, 1).reshape(B * N, x.shape[-1], W)             # [B·N, C, W]
        h = self.conv(x)[..., -1].reshape(B, N, -1)                          # last step
        if self.motion is not None:
            if qd is None:
                raise ValueError("this detector uses the joint speed: pass qd")
            qn = (qd.to(dt) - self.qd_mean) / self.qd_std
            valid = torch.ones(B, dtype=dt, device=qd.device) if q_valid is None else q_valid.to(dt).reshape(B)
            summ = torch.stack([qn.pow(2).mean(-1).sqrt(), qn.abs().amax(-1), torch.ones_like(valid)], -1)
            summ = summ * valid[:, None]                                    # invalid q → no motion evidence
            h = h + self.motion(torch.log1p(summ)).unsqueeze(1)
        if self.taxel_emb is not None:
            h = h + self.emb_proj(self.taxel_emb.weight.to(dt)).unsqueeze(0)
        return self.head(h).squeeze(-1)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ContactDetector":
        return cls(**dict(cfg))


# ─────────────────────────────────────────────────────────────── losses

def focal_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, *,
               gamma: float = 2.0, alpha: float | None = 0.25) -> torch.Tensor:
    """Binary focal loss (Lin et al. 2017), numerically via BCE-with-logits:
    ``α_t (1 − p_t)^γ · CE``, averaged over ``mask`` entries (masked mean; the paper normalises
    by the number of positives — for dense per-taxel labels the masked mean is better behaved).
    ``alpha=None`` disables the class weight; ``gamma=0`` gives (α-weighted) BCE."""
    y = target.to(logits.dtype)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p_t = torch.exp(-ce)
    loss = ce * (1.0 - p_t).clamp_min(0.0) ** float(gamma)
    if alpha is not None:
        loss = loss * (float(alpha) * y + (1.0 - float(alpha)) * (1.0 - y))
    if mask is None:
        return loss.mean()
    w = mask.to(loss.dtype)
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def contact_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, *,
                 kind: str = "focal", gamma: float = 2.0, alpha: float | None = 0.25,
                 pos_weight: float | None = None) -> torch.Tensor:
    """``kind="focal"`` (:func:`focal_loss`) or ``"bce"`` (masked BCE with optional ``pos_weight``)."""
    if kind not in LOSS_KINDS:
        raise ValueError(f"kind must be one of {LOSS_KINDS}, got {kind!r}")
    if kind == "focal":
        return focal_loss(logits, target, mask, gamma=gamma, alpha=alpha)
    pw = None if pos_weight is None else torch.as_tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none", pos_weight=pw)
    if mask is None:
        return loss.mean()
    w = mask.to(loss.dtype)
    return (loss * w).sum() / w.sum().clamp_min(1.0)


# ─────────────────────────────────────────────────────────────── inference

def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def predict_contact_prob(detector: ContactDetector, episode=None, *, z: np.ndarray | None = None,
                         saturated: np.ndarray | None = None, qd: np.ndarray | None = None,
                         q_valid: np.ndarray | None = None, window: int | None = None, batch_size: int = 1024,
                         device: str | torch.device | None = None, z_key: str | None = None) -> np.ndarray:
    """Causal per-frame contact probability ``[T,N]`` float32 (module docstring).

    From an ``episode`` (derived ``residual_z`` or ``z_key``, ``saturated``, ``qd`` and
    :func:`datasets.stats.qd_valid_mask`) and/or explicit arrays (which take precedence)."""
    if episode is not None:
        from ..datasets.episode import D_RESIDUAL_Z, K_QD, K_SATURATED
        from ..datasets.stats import qd_valid_mask

        if z is None:
            z = np.asarray(episode.derived(z_key or D_RESIDUAL_Z), dtype=np.float32)
        if saturated is None and episode.has(K_SATURATED):
            saturated = np.asarray(episode[K_SATURATED], dtype=bool)
        if qd is None and detector.use_motion:
            qd = np.asarray(episode[K_QD], dtype=np.float32)
            if q_valid is None:
                q_valid = qd_valid_mask(episode)
    if z is None:
        raise ValueError("need an episode or z")
    z = np.nan_to_num(np.asarray(z, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    T, N = z.shape
    sat = np.zeros((T, N), dtype=np.float32) if saturated is None else np.asarray(saturated, dtype=np.float32)
    W = int(window or detector.window)
    zw, sw = _causal_windows(z, W), _causal_windows(sat, W)
    qv = np.ones(T, dtype=np.float32) if q_valid is None else np.asarray(q_valid, dtype=np.float32)
    dev = torch.device(device) if device is not None else _device(detector)
    was = detector.training
    detector.eval()
    out = []
    try:
        for s in range(0, T, max(1, int(batch_size))):
            e = s + max(1, int(batch_size))
            f32 = lambda a: torch.from_numpy(np.array(a, dtype=np.float32)).to(dev)  # noqa: E731 (copy: memmaps)
            tz, ts, tv = f32(zw[s:e]), f32(sw[s:e]), f32(qv[s:e])
            tq = None if qd is None else f32(qd[s:e])
            out.append(torch.sigmoid(detector(tz, ts, tq, tv)).float().cpu().numpy())
    finally:
        detector.train(was)
    return np.concatenate(out).astype(np.float32)


#: spec name (``contact.detector.predict_episode``)
predict_episode = predict_contact_prob


class CausalDetectorStream:
    """Online counterpart of :func:`predict_contact_prob`: :meth:`push` one tick of ``z[N]``,
    ``saturated[N]``, ``qd[D]`` → ``prob[N]``; the history is edge-padded with the first tick like
    the offline windows (tested equal). :meth:`reset` at each episode start."""

    def __init__(self, detector: ContactDetector, *, window: int | None = None,
                 device: str | torch.device | None = None):
        self.detector = detector
        self.window = int(window or detector.window)
        self.device = torch.device(device) if device is not None else _device(detector)
        self.reset()

    def reset(self) -> None:
        self._z: np.ndarray | None = None
        self._s: np.ndarray | None = None

    @torch.no_grad()
    def push(self, z: np.ndarray, saturated: np.ndarray | None = None, qd: np.ndarray | None = None,
             q_valid: bool = True) -> np.ndarray:
        z = np.nan_to_num(np.asarray(z, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        s = np.zeros_like(z) if saturated is None else np.asarray(saturated, dtype=np.float32).reshape(-1)
        if self._z is None:
            self._z, self._s = np.repeat(z[None], self.window, 0), np.repeat(s[None], self.window, 0)
        else:
            self._z = np.concatenate([self._z[1:], z[None]], 0)
            self._s = np.concatenate([self._s[1:], s[None]], 0)
        d = self.device
        tq = None if qd is None else torch.from_numpy(np.array(qd, dtype=np.float32)[None]).to(d)
        was = self.detector.training
        self.detector.eval()
        try:
            logit = self.detector(torch.as_tensor(self._z[None], device=d), torch.as_tensor(self._s[None], device=d),
                                  tq, torch.as_tensor([float(q_valid)], device=d))
        finally:
            self.detector.train(was)
        return torch.sigmoid(logit)[0].float().cpu().numpy()


# ─────────────────────────────────────────────────────────────── bundle


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


def save_detector(path: str | Path, detector: ContactDetector, meta: Mapping[str, Any] | None = None) -> Path:
    """Atomic ``{format, config, state_dict, meta}`` bundle; reload with :func:`load_detector`."""
    from ..train.checkpoint import save_checkpoint

    p = Path(path)
    if p.suffix != ".pt":
        p = p / DETECTOR_MODEL_NAME
    sd = {k: v.detach().cpu() for k, v in detector.state_dict().items()}
    save_checkpoint(p, format=DETECTOR_FORMAT, config=dict(detector.config), state_dict=sd,
                    meta=_plain(dict(meta or {})))
    return p


def load_detector(path: str | Path, *, map_location: str | torch.device = "cpu",
                  strict: bool = True) -> ContactDetector:
    """Rebuild a :class:`ContactDetector` (``meta`` → ``detector.bundle_meta``)."""
    p = Path(path)
    if p.is_dir():
        p = p / DETECTOR_MODEL_NAME
    ck = torch.load(p, map_location=map_location, weights_only=True)
    if not isinstance(ck, Mapping) or ck.get("format") != DETECTOR_FORMAT:
        raise ValueError(f"{p} is not a {DETECTOR_FORMAT} bundle")
    det = ContactDetector.from_config(ck["config"])
    det.load_state_dict(ck["state_dict"], strict=strict)
    det.bundle_meta = dict(ck.get("meta") or {})
    det.eval()
    return det
