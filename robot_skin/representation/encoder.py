"""Per-taxel tactile value features (single source of truth) + :class:`TaxelEncoder`.

Tactile values
--------------
:func:`tactile_value_features` turns the stage-1 outputs of one frame into the per-taxel value
vector that **every** tactile consumer uses — masked pretraining (:mod:`.pretrain`), the VTLA
dataset/model and the online controller. Keeping one function (and one
:class:`TactileFeatureSpec` stored inside every pretrained encoder) guarantees that training and
deployment see bit-identical inputs.

Inputs (``datasets.episode`` keys):

- ``residual_z`` ``[...,N]`` — :data:`~robot_skin.datasets.episode.D_RESIDUAL_Z`, calibrated
  residual z-score, **press-positive** (contact → large positive z);
- ``level`` ``[...,N]`` int — :data:`~robot_skin.datasets.episode.D_LEVEL`,
  :class:`~robot_skin.contact.ordinal.ContactLevel` (NONE/WEAK/STRONG/SATURATED);
- ``saturated`` ``[...,N]`` bool — :data:`~robot_skin.datasets.episode.K_SATURATED` (optional).

Per-frame features for the ablation modes of :data:`robot_skin.policy.OBS_MODES`:

======== ===== ==========================================================================
mode      dim  features per taxel
======== ===== ==========================================================================
full        6  ``[zf, 1[NONE], 1[WEAK], 1[STRONG], 1[SATURATED], sat]``
ordinal     4  ``[1[NONE], 1[WEAK], 1[STRONG], 1[SATURATED]]`` (= ``OrdinalQuantizer.one_hot``)
binary      1  ``1[WEAK or STRONG]`` (contact, saturated excluded — as ``policy`` binary)
none        0  nothing: ``[...,N,0]``; consumers must **disable the tactile branch** when
               :func:`tactile_value_dim` is 0 (a zero-width value MLP is not built)
======== ===== ==========================================================================

``zf = asinh(clip(z, ±z_clip) / z_scale)`` — linear (``≈ z/z_scale``) in the noise band and
logarithmic for strong presses, so the feature neither saturates at contact nor explodes on a
taxel with tiny calibrated σ; :func:`z_from_feature` inverts it. A saturated taxel (``saturated``
or ``level == SATURATED``) has an untrustworthy residual: ``zf = 0``, its level is forced to
SATURATED and ``sat = 1`` (the same rule as :func:`robot_skin.policy.tactile_features`).
Non-finite z gives ``zf = 0``; a level outside 0..3 (e.g. -1 = unknown) gives an all-zero one-hot.
``full`` here is richer than the RL observation's ``full`` (2/taxel, residual % + sat flag); the
mode *names* and the ordinal/binary/none subsets are identical.

Temporal stacking: with ``history = k`` the value vector of taxel n is the concatenation of k
per-frame vectors ``[f(t-(k-1)s) | … | f(t)]`` (oldest first, ``s = stride`` master-clock ticks;
indices before the first frame repeat frame 0 — causal edge padding). Offline use
:meth:`TactileFeatureSpec.from_arrays` / :meth:`~TactileFeatureSpec.from_episode`; online use
:class:`TactileHistory`, which reproduces them frame by frame.

Encoder
-------
:class:`TaxelEncoder` = :class:`~robot_skin.representation.tokenizer.TaxelTokenizer` (value MLP
+ pose MLP on Fourier features of the taxel position (Tancik et al., NeurIPS 2020,
arXiv:2006.10739) ⊕ normal) → pre-LN ``nn.TransformerEncoder`` (batch_first) → final LayerNorm →
tokens ``[B,N,D]``. Taxels are a *set*: without the optional id embedding the encoder is
permutation-equivariant and taxel-count agnostic (glove and robot hand share weights; 3D-ViTac,
arXiv:2410.24091, likewise keeps tactile points in 3D). ``key_padding_mask`` (True = padding)
supports batches of layouts with different N; padded outputs are zeroed.

:func:`save_pretrained_encoder` / :func:`load_pretrained_encoder` store ``{config, state_dict}``
(``encoder_state.pt``, written by ``stages/pretrain.py``); the config includes the
:class:`TactileFeatureSpec`, so a consumer rebuilds the exact input features with
``encoder.feature_spec``.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from ..contact.ordinal import N_LEVELS, ContactLevel
from ..datasets.episode import D_LEVEL, D_RESIDUAL_Z, K_SATURATED
from ..policy.observation import OBS_MODES
from .tokenizer import TaxelTokenizer

__all__ = [
    "OBS_MODES", "TACTILE_FRAME_DIMS", "Z_CLIP", "Z_SCALE", "ENCODER_STATE_NAME",
    "ENCODER_FORMAT", "tactile_value_dim", "tactile_value_features", "z_feature",
    "z_from_feature", "history_indices", "stack_history", "episode_tactile_arrays",
    "TactileFeatureSpec", "TactileHistory", "TaxelEncoder", "save_pretrained_encoder",
    "read_encoder_state", "load_pretrained_encoder",
]

#: per-frame feature width for each :data:`OBS_MODES` entry
TACTILE_FRAME_DIMS: dict[str, int] = {"full": 2 + N_LEVELS, "ordinal": N_LEVELS, "binary": 1,
                                      "none": 0}
Z_CLIP = 100.0          # |z| clip before compression (guards against a taxel with tiny σ)
Z_SCALE = 2.0           # zf ≈ z / Z_SCALE in the noise band
ENCODER_STATE_NAME = "encoder_state.pt"
ENCODER_FORMAT = "robot_skin/taxel_encoder"
_ENCODER_FORMAT_VERSION = 1

assert set(TACTILE_FRAME_DIMS) == set(OBS_MODES), "keep TACTILE_FRAME_DIMS in sync with OBS_MODES"


def _check_mode(obs_mode: str) -> None:
    if obs_mode not in OBS_MODES:
        raise ValueError(f"obs_mode must be one of {OBS_MODES}, got {obs_mode!r}")


def tactile_value_dim(obs_mode: str = "full", history: int = 1) -> int:
    """Width of the per-taxel value vector: ``TACTILE_FRAME_DIMS[obs_mode] × history``."""
    _check_mode(obs_mode)
    if int(history) < 1:
        raise ValueError(f"history must be >= 1, got {history}")
    return TACTILE_FRAME_DIMS[obs_mode] * int(history)


# ─────────────────────────────────────────────────────────────── per-frame features

def z_feature(z: Any, z_clip: float = Z_CLIP, z_scale: float = Z_SCALE) -> Any:
    """``asinh(clip(z, ±z_clip) / z_scale)`` with non-finite → 0 (numpy or torch)."""
    if isinstance(z, torch.Tensor):
        zz = torch.nan_to_num(z, nan=0.0, posinf=z_clip, neginf=-z_clip).clamp(-z_clip, z_clip)
        return torch.asinh(zz / z_scale)
    zz = np.clip(np.nan_to_num(np.asarray(z, dtype=np.float32), nan=0.0, posinf=z_clip,
                               neginf=-z_clip), -z_clip, z_clip)
    return np.arcsinh(zz / np.float32(z_scale)).astype(np.float32)


def z_from_feature(f: Any, z_scale: float = Z_SCALE) -> Any:
    """Inverse of :func:`z_feature` inside the clip range: ``z = z_scale · sinh(f)``."""
    if isinstance(f, torch.Tensor):
        return z_scale * torch.sinh(f)
    return (np.float32(z_scale) * np.sinh(np.asarray(f, dtype=np.float32))).astype(np.float32)


def tactile_value_features(residual_z: Any, level: Any, saturated: Any = None, *,
                           obs_mode: str = "full", z_clip: float = Z_CLIP,
                           z_scale: float = Z_SCALE) -> Any:
    """Per-taxel tactile value features ``[...,N] → [...,N,F]`` (see module docstring).

    Args:
        residual_z: calibrated residual z-score ``[...,N]``, press-positive.
        level: :class:`ContactLevel` ints ``[...,N]`` (same shape).
        saturated: optional bool ``[...,N]``; ``None`` → only ``level == SATURATED`` counts.
        obs_mode: one of :data:`OBS_MODES`; ``F = TACTILE_FRAME_DIMS[obs_mode]`` (0 for none).

    numpy inputs give float32 numpy; torch inputs (any of them) give a torch tensor on the
    device of ``residual_z``/``level`` (dtype of a floating ``residual_z``, else float32).
    """
    _check_mode(obs_mode)
    if z_clip <= 0 or z_scale <= 0:
        raise ValueError("z_clip and z_scale must be > 0")
    use_torch = any(isinstance(a, torch.Tensor) for a in (residual_z, level, saturated))
    if use_torch:
        return _features_torch(residual_z, level, saturated, obs_mode, z_clip, z_scale)
    return _features_numpy(residual_z, level, saturated, obs_mode, z_clip, z_scale)


def _check_shapes(z_shape: tuple, lv_shape: tuple, sat_shape: tuple | None) -> None:
    if len(z_shape) < 1:
        raise ValueError("residual_z must have a taxel axis [...,N]")
    if tuple(lv_shape) != tuple(z_shape):
        raise ValueError(f"level shape {tuple(lv_shape)} != residual_z shape {tuple(z_shape)}")
    if sat_shape is not None and tuple(sat_shape) != tuple(z_shape):
        raise ValueError(f"saturated shape {tuple(sat_shape)} != residual_z shape {tuple(z_shape)}")


def _features_numpy(residual_z, level, saturated, obs_mode, z_clip, z_scale) -> np.ndarray:
    z = np.asarray(residual_z, dtype=np.float32)
    lv = np.asarray(level)
    sat_in = None if saturated is None else np.asarray(saturated, dtype=bool)
    _check_shapes(z.shape, lv.shape, None if sat_in is None else sat_in.shape)
    if obs_mode == "none":
        return np.zeros(z.shape + (0,), dtype=np.float32)
    lv = lv.astype(np.int64)
    sat = lv == int(ContactLevel.SATURATED)
    if sat_in is not None:
        sat = sat | sat_in
    lv = np.where(sat, int(ContactLevel.SATURATED), lv)
    if obs_mode == "binary":
        c = (lv == int(ContactLevel.WEAK)) | (lv == int(ContactLevel.STRONG))
        return c.astype(np.float32)[..., None]
    onehot = np.stack([lv == c for c in range(N_LEVELS)], axis=-1).astype(np.float32)
    if obs_mode == "ordinal":
        return onehot
    zf = np.where(sat, np.float32(0.0), z_feature(z, z_clip, z_scale)).astype(np.float32)
    return np.concatenate([zf[..., None], onehot, sat.astype(np.float32)[..., None]], axis=-1)


def _features_torch(residual_z, level, saturated, obs_mode, z_clip, z_scale) -> torch.Tensor:
    ref = residual_z if isinstance(residual_z, torch.Tensor) else (
        level if isinstance(level, torch.Tensor) else saturated)
    dev = ref.device
    z = torch.as_tensor(residual_z, device=dev)
    dtype = z.dtype if z.is_floating_point() else torch.float32
    z = z.to(dtype)
    lv = torch.as_tensor(level, device=dev)
    sat_in = None if saturated is None else torch.as_tensor(saturated, device=dev).bool()
    _check_shapes(tuple(z.shape), tuple(lv.shape), None if sat_in is None else tuple(sat_in.shape))
    if obs_mode == "none":
        return z.new_zeros(tuple(z.shape) + (0,))
    lv = lv.long()
    sat = lv == int(ContactLevel.SATURATED)
    if sat_in is not None:
        sat = sat | sat_in
    lv = torch.where(sat, torch.full_like(lv, int(ContactLevel.SATURATED)), lv)
    if obs_mode == "binary":
        c = (lv == int(ContactLevel.WEAK)) | (lv == int(ContactLevel.STRONG))
        return c.to(dtype).unsqueeze(-1)
    onehot = torch.stack([lv == c for c in range(N_LEVELS)], dim=-1).to(dtype)
    if obs_mode == "ordinal":
        return onehot
    zf = torch.where(sat, torch.zeros_like(z), z_feature(z, z_clip, z_scale))
    return torch.cat([zf.unsqueeze(-1), onehot, sat.to(dtype).unsqueeze(-1)], dim=-1)


# ─────────────────────────────────────────────────────────────── temporal stacking

def history_indices(t_index: Any, history: int, stride: int = 1) -> np.ndarray:
    """Frame indices ``[..., history]`` (oldest first) ending at ``t_index``; indices < 0 are
    clipped to 0 (causal edge padding)."""
    if history < 1 or stride < 1:
        raise ValueError("history and stride must be >= 1")
    t = np.asarray(t_index, dtype=np.int64)
    if np.any(t < 0):
        raise ValueError("t_index must be >= 0")
    offs = (np.arange(history, dtype=np.int64) - (history - 1)) * int(stride)
    return np.maximum(t[..., None] + offs, 0)


def stack_history(frames: Any) -> Any:
    """``[..., K, N, F]`` per-frame features (oldest first) → ``[..., N, K·F]``."""
    if frames.ndim < 3:
        raise ValueError("frames must be [..., K, N, F]")
    if isinstance(frames, torch.Tensor):
        x = frames.movedim(-3, -2)
        return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
    x = np.moveaxis(np.asarray(frames), -3, -2)
    return x.reshape(x.shape[:-2] + (x.shape[-2] * x.shape[-1],))


def episode_tactile_arrays(episode: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``(residual_z[T,N], level[T,N], saturated[T,N] | None)`` of a processed Episode
    (derived keys written by the contact stage)."""
    missing = [k for k in (D_RESIDUAL_Z, D_LEVEL) if not episode.has_derived(k)]
    if missing:
        eid = getattr(getattr(episode, "meta", None), "episode_id", "?")
        raise KeyError(f"episode {eid!r} lacks derived {missing}; run the contact stage first")
    sat = episode[K_SATURATED] if episode.has(K_SATURATED) else None
    return episode.derived(D_RESIDUAL_Z), episode.derived(D_LEVEL), sat


@dataclass(frozen=True)
class TactileFeatureSpec:
    """How tactile values are built: mode, temporal stacking and z compression.

    ``history`` frames spaced ``stride`` master-clock ticks (200 Hz) are stacked (``history=1`` →
    current frame only). Stored in every pretrained encoder's config so VTLA/control rebuild
    exactly the same inputs.
    """
    obs_mode: str = "full"
    history: int = 1
    stride: int = 1
    z_clip: float = Z_CLIP
    z_scale: float = Z_SCALE

    def __post_init__(self) -> None:
        _check_mode(self.obs_mode)
        if int(self.history) < 1 or int(self.stride) < 1:
            raise ValueError("history and stride must be >= 1")
        if not (self.z_clip > 0 and self.z_scale > 0):
            raise ValueError("z_clip and z_scale must be > 0")
        object.__setattr__(self, "history", int(self.history))
        object.__setattr__(self, "stride", int(self.stride))
        object.__setattr__(self, "z_clip", float(self.z_clip))
        object.__setattr__(self, "z_scale", float(self.z_scale))

    @property
    def frame_dim(self) -> int:
        return TACTILE_FRAME_DIMS[self.obs_mode]

    @property
    def dim(self) -> int:
        """Per-taxel value width (``frame_dim × history``); 0 for ``obs_mode="none"``."""
        return self.frame_dim * self.history

    @property
    def span(self) -> int:
        """Master-clock ticks covered by the stacked history (``(history-1)·stride + 1``)."""
        return (self.history - 1) * self.stride + 1

    def features(self, residual_z: Any, level: Any, saturated: Any = None) -> Any:
        """Per-frame features ``[...,N,frame_dim]`` (:func:`tactile_value_features`)."""
        return tactile_value_features(residual_z, level, saturated, obs_mode=self.obs_mode,
                                      z_clip=self.z_clip, z_scale=self.z_scale)

    def from_arrays(self, residual_z: Any, level: Any, saturated: Any, t_index: Any) -> np.ndarray:
        """Stacked values ``[..., N, dim]`` at frame(s) ``t_index`` of ``[T,N]`` arrays."""
        idx = history_indices(t_index, self.history, self.stride)
        T = np.shape(residual_z)[0]
        if idx.size and int(idx.max()) >= T:
            raise IndexError(f"t_index out of range for T={T}")
        z = np.asarray(residual_z)[idx]
        lv = np.asarray(level)[idx]
        sat = None if saturated is None else np.asarray(saturated)[idx]
        return stack_history(self.features(z, lv, sat))

    def from_episode(self, episode: Any, t_index: Any) -> np.ndarray:
        """Stacked values ``[..., N, dim]`` of a processed Episode at ``t_index``."""
        return self.from_arrays(*episode_tactile_arrays(episode), t_index)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "TactileFeatureSpec":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown TactileFeatureSpec keys {unknown} (known: {sorted(known)})")
        return cls(**d)


class TactileHistory:
    """Streaming counterpart of :meth:`TactileFeatureSpec.from_arrays` for online control.

    ``push`` one frame per master-clock tick; it returns the stacked values ``[N, dim]`` equal to
    the offline result at the same tick (the first frame is repeated until the history fills).
    """

    def __init__(self, spec: TactileFeatureSpec | Mapping[str, Any] | None = None) -> None:
        self.spec = spec if isinstance(spec, TactileFeatureSpec) else TactileFeatureSpec.from_dict(spec)
        self._buf: deque = deque(maxlen=self.spec.span)
        self._first: Any = None

    def reset(self) -> None:
        self._buf.clear()
        self._first = None

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, residual_z: Any, level: Any, saturated: Any = None) -> Any:
        """Add frame ``[N]`` (or ``[...,N]``) and return stacked values ``[..., N, dim]``."""
        f = self.spec.features(residual_z, level, saturated)
        if self._first is None:
            self._first = f
        self._buf.append(f)
        L, s = len(self._buf), self.spec.stride
        frames = []
        for j in range(self.spec.history):
            back = (self.spec.history - 1 - j) * s     # ticks before now
            frames.append(self._buf[L - 1 - back] if back < L else self._first)
        if isinstance(f, torch.Tensor):
            return stack_history(torch.stack(frames, dim=-3))
        return stack_history(np.stack(frames, axis=-3))


# ─────────────────────────────────────────────────────────────── encoder

class TaxelEncoder(nn.Module):
    """TaxelTokenizer + pre-LN transformer over the taxel set → tokens ``[B,N,d_model]``.

    Args:
        value_dim: per-taxel value width (``TactileFeatureSpec.dim``); must be ≥ 1 — for
            ``obs_mode="none"`` build no tactile encoder at all.
        d_model, depth, heads: transformer width / layers (≥ 1) / attention heads.
        n_fourier, fourier_scale: Fourier position features (octaves, base period in metres;
            defaults 6 × 0.3 m: finest period ≈ 9 mm, above mm-level pose-label noise — see
            ``representation.tokenizer``).
        n_taxels: optional per-index id embedding (single-layout models only; breaks the
            permutation equivariance and taxel-count independence).
        ff_mult, dropout: feed-forward width multiplier and dropout of the transformer.
        feature_spec: the :class:`TactileFeatureSpec` producing ``values`` (stored in
            :attr:`config`; its ``dim`` must equal ``value_dim``).
    """

    def __init__(self, value_dim: int, d_model: int = 64, depth: int = 2, heads: int = 4, *,
                 n_fourier: int = 6, fourier_scale: float = 0.3, n_taxels: int | None = None,
                 ff_mult: int = 4, dropout: float = 0.0,
                 feature_spec: TactileFeatureSpec | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        value_dim, d_model, depth, heads = int(value_dim), int(d_model), int(depth), int(heads)
        if value_dim < 1:
            raise ValueError("value_dim must be >= 1 (obs_mode 'none' has no tactile values: "
                             "disable the tactile branch instead of building an encoder)")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if d_model % heads:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
        if feature_spec is not None and not isinstance(feature_spec, TactileFeatureSpec):
            feature_spec = TactileFeatureSpec.from_dict(feature_spec)
        if feature_spec is not None and feature_spec.dim != value_dim:
            raise ValueError(f"feature_spec.dim={feature_spec.dim} != value_dim={value_dim}")
        self.feature_spec: TactileFeatureSpec | None = feature_spec
        self.pretrain_meta: dict = {}      # filled by load_pretrained_encoder
        self._cfg = dict(value_dim=value_dim, d_model=d_model, depth=depth, heads=heads,
                         n_fourier=int(n_fourier), fourier_scale=float(fourier_scale),
                         n_taxels=None if n_taxels is None else int(n_taxels),
                         ff_mult=int(ff_mult), dropout=float(dropout))
        self.tokenizer = TaxelTokenizer(value_dim, d_model, n_fourier=int(n_fourier),
                                        fourier_scale=float(fourier_scale), n_taxels=n_taxels)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=int(ff_mult) * d_model,
                                           dropout=float(dropout), activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(d_model),
                                            enable_nested_tensor=False)

    @property
    def value_dim(self) -> int:
        return self._cfg["value_dim"]

    @property
    def d_model(self) -> int:
        return self._cfg["d_model"]

    @property
    def out_dim(self) -> int:
        return self._cfg["d_model"]

    @property
    def config(self) -> dict:
        """JSON-able constructor kwargs (incl. ``feature_spec``) — :meth:`from_config` inverse."""
        return {**self._cfg,
                "feature_spec": None if self.feature_spec is None else self.feature_spec.to_dict()}

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "TaxelEncoder":
        cfg = dict(cfg)
        return cls(cfg.pop("value_dim"), **cfg)

    def forward(self, values: torch.Tensor, pos: torch.Tensor, nrm: torch.Tensor,
                mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """values ``[B,N,F]``, pos/nrm ``[B,N,3]`` (m, hand/robot base frame) → ``[B,N,D]``.

        ``mask`` ``[B,N]`` bool swaps a taxel's value for the tokenizer's ``[MASK]`` vector (pose
        kept). Note: MAE pretraining (:mod:`.pretrain`) never lets such tokens act as attention
        keys, so the tokenizer ``[MASK]`` receives zero gradient there and stays at its random
        init — to drop dead/quarantined taxels from a pretrained encoder use
        ``key_padding_mask`` (or fine-tune the ``[MASK]`` downstream). ``key_padding_mask``
        ``[B,N]`` bool (True = ignore) removes taxels as attention keys and zeroes their
        outputs — padding in mixed-layout batches, or the hidden taxels of MAE pretraining.
        """
        if values.shape[-1] != self.value_dim:
            raise ValueError(f"values last dim {values.shape[-1]} != value_dim {self.value_dim}")
        kpm = None if key_padding_mask is None else key_padding_mask.bool()
        if kpm is not None and bool(kpm.all(dim=-1).any()):
            raise ValueError("key_padding_mask hides every taxel of a sample (≥1 must be visible)")
        # parameter dtype (fp32 under autocast): Fourier features of positions need fp32 phases
        pdt = self.tokenizer.pose_mlp[0].weight.dtype
        tok = self.tokenizer(values.to(pdt), pos.to(pdt), nrm.to(pdt), mask)
        x = self.blocks(tok, src_key_padding_mask=kpm)
        if kpm is not None:
            x = x.masked_fill(kpm.unsqueeze(-1), 0.0)
        return x

    def encode(self, residual_z: Any, level: Any, saturated: Any, pos: Any, nrm: Any,
               key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Single-frame convenience (``history == 1``): build values with :attr:`feature_spec`
        and encode. Inputs ``[B,N]`` / ``[B,N,3]`` (numpy or torch)."""
        spec = self.feature_spec
        if spec is None or spec.history != 1:
            raise ValueError("encode() needs a feature_spec with history == 1; build stacked "
                             "values with TactileFeatureSpec.from_arrays / TactileHistory")
        p = next(self.parameters())
        as_t = lambda a: torch.as_tensor(a, dtype=p.dtype, device=p.device)  # noqa: E731
        v = as_t(spec.features(residual_z, level, saturated))
        return self(v, as_t(pos), as_t(nrm), key_padding_mask=key_padding_mask)


# ─────────────────────────────────────────────────────────────── persistence

def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars/arrays and tuples so ``torch.load(weights_only=True)`` accepts it."""
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_pretrained_encoder(path: str | Path, encoder: TaxelEncoder,
                            meta: Mapping[str, Any] | None = None) -> Path:
    """Atomically write ``{format, version, config, state_dict, meta}`` (``encoder_state.pt``).

    ``path`` may be a directory (→ ``<dir>/encoder_state.pt``). Tensors are saved on CPU.
    """
    from ..train.checkpoint import save_checkpoint  # atomic tmp + fsync + rename

    p = Path(path)
    if p.suffix != ".pt":
        p = p / ENCODER_STATE_NAME
    state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
    return save_checkpoint(p, format=ENCODER_FORMAT, version=_ENCODER_FORMAT_VERSION,
                           config=_json_safe(encoder.config), state_dict=state,
                           meta=_json_safe(dict(meta or {})))


def read_encoder_state(path: str | Path, map_location: Any = "cpu") -> dict:
    """Load the raw ``encoder_state.pt`` dict (``path`` may be the run directory)."""
    p = Path(path)
    if p.is_dir():
        p = p / ENCODER_STATE_NAME
    if not p.is_file():
        raise FileNotFoundError(f"no encoder state at {p}")
    state = torch.load(p, map_location=map_location, weights_only=True)
    if not isinstance(state, dict) or state.get("format") != ENCODER_FORMAT:
        raise ValueError(f"{p} is not a {ENCODER_FORMAT} file (got format "
                         f"{state.get('format') if isinstance(state, dict) else type(state)!r})")
    if int(state.get("version", 0)) > _ENCODER_FORMAT_VERSION:
        raise ValueError(f"{p}: encoder format v{state['version']} is newer than supported "
                         f"v{_ENCODER_FORMAT_VERSION}")
    return state


def load_pretrained_encoder(path: str | Path, *, map_location: Any = "cpu", strict: bool = True,
                            freeze: bool = False) -> TaxelEncoder:
    """Rebuild a :class:`TaxelEncoder` from ``encoder_state.pt`` (file or run directory).

    ``map_location`` (a device, or anything :func:`torch.load` accepts) — a device also places
    the returned module there. ``freeze=True`` disables gradients and puts it in eval mode
    (frozen tactile branch). ``encoder.feature_spec`` holds the tactile feature recipe used in
    pretraining; ``encoder.pretrain_meta`` the stored metadata. Parameters are float32.
    """
    state = read_encoder_state(path, map_location=map_location)
    enc = TaxelEncoder.from_config(state["config"])
    if isinstance(map_location, (str, torch.device)):
        enc = enc.to(map_location)
    enc.load_state_dict(state["state_dict"], strict=strict)
    enc.pretrain_meta = dict(state.get("meta") or {})
    if freeze:
        enc.requires_grad_(False)
        enc.eval()
    return enc
