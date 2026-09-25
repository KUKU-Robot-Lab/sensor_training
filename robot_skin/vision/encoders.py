"""Vision encoders: normalized images ``float [B,3,H,W]`` → visual tokens ``[B,P,D]``.

All encoders share one contract (:class:`VisionEncoder`) so the VTLA model, the feature cache
and deployment never care which backbone is inside:

- ``forward(images)`` with images normalized by ``encoder.mean`` / ``encoder.std`` (use
  ``vision.transforms`` with those values) → tokens ``[B,P,D]``;
- ``out_dim`` = ``D``; ``n_tokens(h, w)`` = ``P`` for an ``h×w`` input;
- ``cache_key`` names cached features (``vision.feature_cache``); ``is_frozen`` says whether the
  output can be cached at all (no trainable parameters).

Backbones:

- :class:`TinyConvEncoder` — small GroupNorm CNN trained from scratch; no extra dependencies
  (default for tests, smoke runs and low-data ablations).
- :class:`ResNetEncoder` — torchvision ResNet-18/34/50 (optional ``torchvision``). Diffusion Policy
  (Chi et al., arXiv:2303.04137) uses ResNet-18 with GroupNorm instead of BatchNorm and
  spatial-softmax pooling; both are options here (``group_norm``, ``pool="spatial_softmax"``).
- :class:`HFVisionEncoder` — any HuggingFace vision transformer, e.g. ``facebook/dinov2-small``,
  ``google/siglip-base-patch16-224`` or CLIP (optional ``transformers``); the SigLIP/DINOv2
  families are the vision towers used by recent VLAs (OpenVLA arXiv:2406.09246).

Token pooling (``pool``): ``none`` (every feature-map cell is a token), ``grid`` (adaptive
average pool to a fixed ``gh×gw`` grid → token count independent of image size), ``avg`` (one
token), ``spatial_softmax`` (``K`` keypoint tokens: expected ``(x, y)`` + attended feature). Grid
tokens get fixed 2-D sin-cos position embeddings as in MAE (He et al., arXiv:2111.06377).

Pretrained backbones are frozen by default (``frozen=True`` freezes the backbone only; an
``out_dim`` projection stays trainable). For feature caching use ``frozen=True, out_dim=None`` so
the whole encoder is frozen and cached features never go stale.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import math
import re
import warnings
from typing import Any, Mapping

import torch
import torch.nn as nn

from .transforms import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, SIGLIP_MEAN, SIGLIP_STD

__all__ = [
    "VisionEncoder", "TinyConvEncoder", "ResNetEncoder", "HFVisionEncoder", "SpatialSoftmax",
    "TokenPool", "sincos_pos_embed_2d", "build_vision_encoder", "POOL_MODES", "sanitize_key",
]

POOL_MODES = ("none", "grid", "avg", "spatial_softmax")


def sanitize_key(s: str) -> str:
    """Filesystem-safe cache key: ``[A-Za-z0-9_.-]``, everything else → ``_``."""
    k = re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(s)).strip("_.")
    if not k:
        raise ValueError(f"cannot build a cache key from {s!r}")
    return k


def _require(module: str, what: str, hint: str):
    """Import an optional dependency or raise an ImportError that says what to do."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"{what} needs the optional package `{module}`, which is not installed ({e}). {hint} "
            f"Without it, use the dependency-free encoder: build_vision_encoder({{'type': 'tiny'}}).") from e


def _pair(v) -> tuple[int, int]:
    if isinstance(v, int):
        return v, v
    if len(v) != 2:
        raise ValueError(f"expected an int or a pair, got {v!r}")
    return int(v[0]), int(v[1])


def _conv_out(n: int, k: int, s: int, p: int) -> int:
    return (n + 2 * p - k) // s + 1


# ── building blocks ───────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=64)
def _sincos_cpu(h: int, w: int, dim: int, temperature: float) -> torch.Tensor:
    d4 = dim // 4
    out = torch.zeros(h * w, dim, dtype=torch.float64)
    if d4 > 0:
        omega = 1.0 / temperature ** (torch.arange(d4, dtype=torch.float64) / d4)
        ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                                torch.arange(w, dtype=torch.float64), indexing="ij")
        ax = xs.reshape(-1, 1) * omega
        ay = ys.reshape(-1, 1) * omega
        out[:, :4 * d4] = torch.cat([ax.sin(), ax.cos(), ay.sin(), ay.cos()], dim=1)
    return out


def sincos_pos_embed_2d(h: int, w: int, dim: int, *, temperature: float = 10000.0,
                        device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Fixed 2-D sin-cos position embedding ``[h*w, dim]`` (row-major grid, MAE-style).

    ``dim/4`` frequencies ``ω_i = temperature^(−i/(dim/4))`` each for ``sin(x·ω)``, ``cos(x·ω)``,
    ``sin(y·ω)``, ``cos(y·ω)`` (``x`` = column, ``y`` = row); if ``dim`` is not a multiple of 4 the
    remainder is zero-padded. Computed once per size in float64 on the CPU (cached; MPS has no
    float64) and returned as a fresh tensor on ``device`` in ``dtype``.
    """
    if h < 1 or w < 1 or dim < 1:
        raise ValueError(f"h, w, dim must be positive, got {(h, w, dim)}")
    pe = _sincos_cpu(int(h), int(w), int(dim), float(temperature))
    return pe.to(device=device if device is not None else "cpu", dtype=dtype, copy=True)


class SpatialSoftmax(nn.Module):
    """Feature map ``[B,C,H,W]`` → keypoints ``[B,K,2]`` = soft-argmax ``(x, y)`` in ``[-1,1]``.

    An optional 1×1 conv maps ``C → K`` keypoint channels; each channel is soft-maxed over the
    ``H·W`` cells (learnable temperature) and the expected pixel-centre coordinate is returned
    (``x`` = column, ``y`` = row, ``align_corners=False`` convention). ``forward(...,
    return_attention=True)`` also returns the attention maps ``[B,K,H*W]``.
    """

    def __init__(self, in_channels: int, n_keypoints: int | None = None, *,
                 temperature: float = 1.0, learnable_temperature: bool = True):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.n_keypoints = n_keypoints or in_channels
        self.proj = nn.Conv2d(in_channels, n_keypoints, 1) if n_keypoints else nn.Identity()
        log_t = torch.tensor(math.log(temperature))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_t)
        else:
            self.register_buffer("log_temperature", log_t)

    def forward(self, feat: torch.Tensor, return_attention: bool = False):
        x = self.proj(feat)
        B, K, H, W = x.shape
        attn = torch.softmax(x.reshape(B, K, H * W) / self.log_temperature.exp(), dim=-1)
        xs = (2 * torch.arange(W, device=x.device, dtype=attn.dtype) + 1) / W - 1
        ys = (2 * torch.arange(H, device=x.device, dtype=attn.dtype) + 1) / H - 1
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # [HW,2]
        kp = attn @ coords                                               # [B,K,2]
        return (kp, attn) if return_attention else kp


class TokenPool(nn.Module):
    """Feature map ``[B,C,h,w]`` → tokens ``[B,P,D]`` (see module docstring for ``pool`` modes).

    ``out_dim=None`` keeps ``D = C`` (no trainable projection; not allowed for spatial_softmax,
    which always projects ``[feature, x, y]``).
    """

    def __init__(self, in_channels: int, out_dim: int | None = None, pool: str = "none", *,
                 grid=(4, 4), n_keypoints: int = 32, pos_embed: bool = True):
        super().__init__()
        if pool not in POOL_MODES:
            raise ValueError(f"pool must be one of {POOL_MODES}, got {pool!r}")
        self.pool, self.pos_embed = pool, bool(pos_embed)
        self.grid = _pair(grid) if pool == "grid" else None
        if self.grid is not None and min(self.grid) < 1:
            raise ValueError(f"grid must be positive, got {grid}")
        if pool == "spatial_softmax":
            if n_keypoints < 1:
                raise ValueError("n_keypoints must be >= 1")
            self.keypoints = SpatialSoftmax(in_channels, n_keypoints)
            self.out_dim = int(out_dim or in_channels)
            self.proj = nn.Linear(in_channels + 2, self.out_dim)
        else:
            self.out_dim = int(out_dim or in_channels)
            self.proj = nn.Linear(in_channels, out_dim) if out_dim else nn.Identity()

    def n_tokens(self, fh: int, fw: int) -> int:
        if self.pool == "none":
            return fh * fw
        if self.pool == "grid":
            return self.grid[0] * self.grid[1]
        if self.pool == "avg":
            return 1
        return self.keypoints.n_keypoints

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.pool == "spatial_softmax":
            kp, attn = self.keypoints(feat, return_attention=True)
            f = attn @ feat.flatten(2).transpose(1, 2)                   # [B,K,C]
            return self.proj(torch.cat([f, kp.to(f)], dim=-1))
        if self.pool == "avg":
            return self.proj(feat.mean(dim=(-2, -1)).unsqueeze(1))
        if self.pool == "grid":
            feat = nn.functional.adaptive_avg_pool2d(feat, self.grid)
        h, w = feat.shape[-2:]
        tok = self.proj(feat.flatten(2).transpose(1, 2))                 # [B,h*w,D]
        if self.pos_embed:
            tok = tok + sincos_pos_embed_2d(h, w, tok.shape[-1], device=tok.device, dtype=tok.dtype)
        return tok


# ── base ──────────────────────────────────────────────────────────────────────
class VisionEncoder(nn.Module):
    """Base class. Subclasses set ``out_dim`` and implement ``forward`` (and ideally ``n_tokens``)."""

    out_dim: int
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __init__(self) -> None:
        super().__init__()
        self._n_tokens_cache: dict[tuple[int, int], int] = {}

    def n_tokens(self, h: int, w: int) -> int:
        """Token count ``P`` for an ``h×w`` input. Fallback: one dummy forward (cached)."""
        key = (int(h), int(w))
        if key not in self._n_tokens_cache:
            ref = next(iter(self.parameters()), None)
            dev = ref.device if ref is not None else torch.device("cpu")
            dt = ref.dtype if ref is not None and ref.is_floating_point() else torch.float32
            was = self.training
            self.eval()
            try:
                with torch.no_grad():
                    x = torch.zeros(1, 3, *key, device=dev, dtype=dt)
                    self._n_tokens_cache[key] = int(self(x).shape[1])
            finally:
                self.train(was)
        return self._n_tokens_cache[key]

    @property
    def cache_key(self) -> str:
        return sanitize_key(type(self).__name__.lower())

    @property
    def is_frozen(self) -> bool:
        """True if no parameter requires grad (output is a fixed function → cacheable)."""
        return not any(p.requires_grad for p in self.parameters())

    @property
    def frozen_backbone(self) -> bool:
        """True if ``self.backbone`` exists and none of its parameters requires grad.

        Read from the parameters (not stored at construction), so unfreezing a pretrained trunk
        for fine-tuning (``for p in enc.backbone.parameters(): p.requires_grad_(True)``) takes
        effect: the no-grad fast path and the BatchNorm/dropout eval pinning then switch off
        (the latter at the next ``.train()`` call).
        """
        bb = self._modules.get("backbone")
        return bb is not None and not any(p.requires_grad for p in bb.parameters())

    def freeze(self) -> "VisionEncoder":
        """Freeze every parameter and switch to eval mode (in place)."""
        for p in self.parameters():
            p.requires_grad_(False)
        return self.eval()

    @staticmethod
    def _check(images: torch.Tensor) -> None:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"vision encoders take normalized float images [B,3,H,W], got shape "
                             f"{tuple(images.shape)} (use vision.transforms to convert uint8 frames)")
        if not images.is_floating_point():
            raise TypeError(f"images must be float, got {images.dtype} (use vision.transforms.to_float_tensor)")


# ── tiny CNN ──────────────────────────────────────────────────────────────────
def _gn(c: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(groups, c), c)


class TinyConvEncoder(VisionEncoder):
    """Small from-scratch CNN: ``len(channels)`` stride-2 stages (3×3 conv, GroupNorm, GELU ×2).

    Default ``pool="grid"`` with ``grid=(4, 4)`` → 16 tokens for *any* image size; ``pool="none"``
    → ``ceil(h/2^S)·ceil(w/2^S)`` tokens (``S`` = number of stages). GroupNorm (not BatchNorm)
    so behaviour does not depend on per-GPU batch size or EMA (Diffusion Policy practice).

    Args:
        out_dim: token width ``D`` (``None`` = last stage width, no projection).
        grid: token grid ``(gh, gw)`` for ``pool="grid"``.
        pool: one of :data:`POOL_MODES`.
        channels: stage widths.
        n_keypoints: keypoints for ``pool="spatial_softmax"``.
        groups: GroupNorm groups (reduced to a divisor of each width).
        pos_embed: add fixed 2-D sin-cos position embeddings to grid/none tokens.
    """

    def __init__(self, out_dim: int | None = 128, grid=(4, 4), *, pool: str = "grid",
                 channels=(32, 64, 128), n_keypoints: int = 16, groups: int = 8,
                 pos_embed: bool = True):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if not channels or min(channels) < 1:
            raise ValueError(f"channels must be a non-empty list of positive ints, got {channels}")
        if out_dim is not None and out_dim < 1:
            raise ValueError("out_dim must be >= 1 (or None = last stage width)")
        layers: list[nn.Module] = []
        c_in = 3
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), _gn(c, groups), nn.GELU(),
                       nn.Conv2d(c, c, 3, stride=1, padding=1), _gn(c, groups), nn.GELU()]
            c_in = c
        self.backbone = nn.Sequential(*layers)
        self.n_stages = len(channels)
        self.head = TokenPool(channels[-1], out_dim, pool, grid=grid, n_keypoints=n_keypoints,
                              pos_embed=pos_embed)
        self.out_dim = self.head.out_dim
        self.pool = pool

    def feature_hw(self, h: int, w: int) -> tuple[int, int]:
        if h < 1 or w < 1:
            raise ValueError(f"image size must be positive, got {(h, w)}")
        for _ in range(self.n_stages):
            h, w = _conv_out(h, 3, 2, 1), _conv_out(w, 3, 2, 1)
        return h, w

    def n_tokens(self, h: int, w: int) -> int:
        return self.head.n_tokens(*self.feature_hw(h, w))

    @property
    def cache_key(self) -> str:
        return f"tiny_{self.pool}"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self._check(images)
        return self.head(self.backbone(images))


# ── torchvision ResNet ────────────────────────────────────────────────────────
_RESNETS = ("resnet18", "resnet34", "resnet50")


def _bn_to_gn(module: nn.Module, groups: int) -> nn.Module:
    """Recursively replace BatchNorm2d with GroupNorm (affine params copied)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = _gn(child.num_features, groups)
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            _bn_to_gn(child, groups)
    return module


class ResNetEncoder(VisionEncoder):
    """torchvision ResNet trunk (up to ``layer4``, stride 32) + :class:`TokenPool`.

    ``pool="none"`` → ``ceil(h/32)·ceil(w/32)`` tokens of width 512 (2048 for resnet50), or
    ``out_dim``. ``frozen=True`` freezes the trunk and keeps its BatchNorm in eval mode (as long
    as every trunk parameter stays frozen, see :attr:`VisionEncoder.frozen_backbone`).
    ``group_norm=True`` swaps BatchNorm → GroupNorm (Diffusion Policy; meant for training the
    trunk, it discards pretrained BN statistics). Pretrained weights come from torchvision
    (downloaded once into ``$TORCH_HOME``).
    """

    def __init__(self, name: str = "resnet18", *, pretrained: bool = True, frozen: bool = True,
                 out_dim: int | None = None, pool: str = "none", grid=(4, 4),
                 n_keypoints: int = 32, group_norm: bool = False, pos_embed: bool = True):
        super().__init__()
        if name not in _RESNETS:
            raise ValueError(f"name must be one of {_RESNETS}, got {name!r}")
        tv = _require("torchvision", f"ResNetEncoder({name!r})",
                      "Install a torchvision build matching your torch/CUDA (e.g. the cu128 wheels for "
                      "RTX 5090 / Blackwell): pip install torchvision --index-url "
                      "https://download.pytorch.org/whl/cu128 .")
        ctor = getattr(tv.models, name)
        try:
            net = ctor(weights="DEFAULT" if pretrained else None)
        except TypeError:  # torchvision < 0.13
            net = ctor(pretrained=pretrained)
        feat_ch = int(net.fc.in_features)
        trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                              net.layer1, net.layer2, net.layer3, net.layer4)
        if group_norm:
            if pretrained and frozen:
                warnings.warn("ResNetEncoder(group_norm=True, pretrained=True, frozen=True): the "
                              "BatchNorm→GroupNorm swap discards the pretrained BN statistics, so a "
                              "frozen trunk gives degraded features; use group_norm with frozen=False",
                              stacklevel=2)
            _bn_to_gn(trunk, 32)
        self.backbone = trunk
        self.name, self.pretrained, self.pool = name, bool(pretrained), pool
        self.group_norm = bool(group_norm)
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.head = TokenPool(feat_ch, out_dim, pool, grid=grid, n_keypoints=n_keypoints,
                              pos_embed=pos_embed)
        self.out_dim = self.head.out_dim
        self.train(self.training)

    def train(self, mode: bool = True) -> "ResNetEncoder":
        super().train(mode)
        if self.frozen_backbone:
            self.backbone.eval()   # frozen trunk: BN running stats must not drift
        return self

    def feature_hw(self, h: int, w: int) -> tuple[int, int]:
        if h < 1 or w < 1:
            raise ValueError(f"image size must be positive, got {(h, w)}")
        h, w = _conv_out(h, 7, 2, 3), _conv_out(w, 7, 2, 3)          # conv1
        h, w = _conv_out(h, 3, 2, 1), _conv_out(w, 3, 2, 1)          # maxpool
        for _ in range(3):                                            # layer2..4
            h, w = _conv_out(h, 3, 2, 1), _conv_out(w, 3, 2, 1)
        return h, w

    def n_tokens(self, h: int, w: int) -> int:
        return self.head.n_tokens(*self.feature_hw(h, w))

    @property
    def cache_key(self) -> str:
        gn = "_gn" if self.group_norm else ""
        return sanitize_key(f"{self.name}{'' if self.pretrained else '_scratch'}{gn}_{self.pool}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self._check(images)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_backbone):
            feat = self.backbone(images)
        return self.head(feat)


# ── HuggingFace transformers ──────────────────────────────────────────────────
_HF_TOKEN_MODES = ("all", "pooled")


def _hf_norm_stats(tf, model_id: str, kw: dict) -> tuple[tuple, tuple]:
    try:
        proc = tf.AutoImageProcessor.from_pretrained(model_id, **kw)
        mean, std = tuple(float(v) for v in proc.image_mean), tuple(float(v) for v in proc.image_std)
        if len(mean) == 3 and len(std) == 3:
            return mean, std
    except Exception:  # noqa: BLE001 — processor is optional; fall back on the model family
        pass
    mid = model_id.lower()
    if "siglip" in mid:
        return SIGLIP_MEAN, SIGLIP_STD
    if "clip" in mid:
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


class HFVisionEncoder(VisionEncoder):
    """HuggingFace vision backbone → ``last_hidden_state`` tokens (``tokens="all"``) or one pooled token.

    Works with ``AutoModel`` vision models (DINOv2, ViT, SigLIP/CLIP vision towers); for dual
    encoders (``CLIPModel``, ``SiglipModel``) the ``vision_model`` tower is used. ``mean``/``std``
    come from the checkpoint's image processor (fallback: family defaults). Non-native input sizes
    use ``interpolate_pos_encoding=True`` when the model supports it (DINOv2 always does).
    """

    def __init__(self, model_id: str = "facebook/dinov2-small", *, pretrained: bool = True,
                 frozen: bool = True, out_dim: int | None = None, tokens: str = "all",
                 revision: str | None = None, local_files_only: bool = False):
        super().__init__()
        if tokens not in _HF_TOKEN_MODES:
            raise ValueError(f"tokens must be one of {_HF_TOKEN_MODES}, got {tokens!r}")
        tf = _require("transformers", f"HFVisionEncoder({model_id!r})",
                      "Install it with: pip install transformers (weights download from the "
                      "HuggingFace hub on first use; set HF_HOME to a shared cache on GPU boxes).")
        kw: dict[str, Any] = {"local_files_only": local_files_only}
        if revision is not None:
            kw["revision"] = revision
        if pretrained:
            model = tf.AutoModel.from_pretrained(model_id, **kw)
        else:
            model = tf.AutoModel.from_config(tf.AutoConfig.from_pretrained(model_id, **kw))
        if hasattr(model, "vision_model") and hasattr(model, "text_model"):
            model = model.vision_model
        cfg = model.config
        hidden = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "vision_config", None),
                                                               "hidden_size", None)
        if hidden is None:
            raise ValueError(f"cannot find hidden_size in the config of {model_id!r}")
        self.backbone = model
        self.model_id, self.tokens = model_id, tokens
        self.pretrained, self.revision = bool(pretrained), revision
        self.mean, self.std = _hf_norm_stats(tf, model_id, kw)
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.proj = nn.Linear(int(hidden), int(out_dim)) if out_dim else nn.Identity()
        self.out_dim = int(out_dim or hidden)
        try:
            params = inspect.signature(model.forward).parameters
            self._interp = "interpolate_pos_encoding" in params
        except (TypeError, ValueError):
            self._interp = False
        self.train(self.training)

    def train(self, mode: bool = True) -> "HFVisionEncoder":
        super().train(mode)
        if self.frozen_backbone:
            self.backbone.eval()   # no dropout / drop-path in a frozen tower
        return self

    @property
    def cache_key(self) -> str:
        key = self.model_id.rstrip("/").rsplit("/", 1)[-1]
        if self.revision:
            key += f"_rev-{str(self.revision)[:12]}"
        if not self.pretrained:
            key += "_scratch"
        if self.tokens != "all":
            key += f"_{self.tokens}"
        return sanitize_key(key)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self._check(images)
        kw = {"interpolate_pos_encoding": True} if self._interp else {}
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_backbone):
            out = self.backbone(pixel_values=images, **kw)
        if self.tokens == "pooled":
            pooled = getattr(out, "pooler_output", None)
            x = pooled.unsqueeze(1) if pooled is not None else out.last_hidden_state.mean(1, keepdim=True)
        else:
            x = out.last_hidden_state
        return self.proj(x)


# ── factory ───────────────────────────────────────────────────────────────────
_HF_FAMILIES = {
    "dinov2": "facebook/dinov2-small",
    "siglip": "google/siglip-base-patch16-224",
    "clip": "openai/clip-vit-base-patch32",
}
_TYPES = {"tiny": TinyConvEncoder, "tiny_conv": TinyConvEncoder, "conv": TinyConvEncoder,
          "resnet": ResNetEncoder, "torchvision": ResNetEncoder,
          "hf": HFVisionEncoder, "transformers": HFVisionEncoder, "huggingface": HFVisionEncoder}


def _construct(cls, kwargs: dict, what: str):
    ok = [n for n in inspect.signature(cls.__init__).parameters if n != "self"]
    bad = sorted(set(kwargs) - set(ok))
    if bad:
        raise ValueError(f"unknown {what} option(s) {bad} for {cls.__name__}; accepted: {ok}")
    return cls(**kwargs)


def build_vision_encoder(cfg: Mapping[str, Any] | None = None, **overrides) -> VisionEncoder:
    """Vision encoder from a config dict (YAML ``vision:`` block).

    ``type``: ``tiny`` (default) | ``resnet`` / ``resnet18`` / ``resnet34`` / ``resnet50`` /
    ``torchvision`` | ``hf`` / ``transformers`` (needs ``model_id``) | family shortcuts ``dinov2``,
    ``siglip``, ``clip`` (default checkpoints). Every other key is passed to the class constructor
    and unknown keys raise ``ValueError``. Missing optional packages raise ``ImportError`` with an
    install hint.

    Example::

        build_vision_encoder({"type": "resnet18", "pretrained": True, "frozen": True})
        build_vision_encoder({"type": "hf", "model_id": "facebook/dinov2-small", "out_dim": 256})
    """
    c = dict(cfg or {})
    c.update(overrides)
    typ = str(c.pop("type", "tiny")).lower()
    if typ in _RESNETS:
        c.setdefault("name", typ)
        typ = "resnet"
    elif typ in _HF_FAMILIES:
        c.setdefault("model_id", _HF_FAMILIES[typ])
        typ = "hf"
    if typ not in _TYPES:
        raise ValueError(f"unknown vision encoder type {typ!r}; choose from "
                         f"{sorted(set(_TYPES) | set(_RESNETS) | set(_HF_FAMILIES))}")
    for k in ("grid", "channels"):
        if isinstance(c.get(k), list):
            c[k] = tuple(c[k])
    return _construct(_TYPES[typ], c, "vision encoder")
