"""Image transforms for camera frames (pure torch, batched, CPU or GPU, seeded).

Frames are stored ``uint8 [F,H,W,3]`` (``camera_<name>/frames.npy`` or jpg); encoders take
``float [B,3,H,W]`` normalized with the encoder's ``mean``/``std``. Everything here keeps any
leading dims (``[B,3,H,W]``, ``[B,T,3,H,W]`` for an observation history …) and works on the
device the input lives on, so augmentation can run in DataLoader workers *or* on the GPU.

Geometry is one affine crop → resize (``F.affine_grid`` + ``F.grid_sample``), used both by
:class:`TrainAugment` (random) and :class:`EvalTransform` (fixed centre crop), so train and eval
see the same pipeline. Small random-crop scale ranges + brightness/contrast jitter follow the
visuomotor-policy practice of Diffusion Policy (Chi et al., arXiv:2303.04137); horizontal flips
are **off** by default because they mirror hand/scene geometry that the action refers to.

Randomness: ``seed=None`` (default) draws from torch's global RNG — DataLoader already seeds it
per worker and per epoch, and ``train.seed_everything`` makes it reproducible. ``seed=int`` uses a
private ``torch.Generator``; inside a DataLoader worker it is re-derived once per worker/epoch
from ``(seed, worker seed)`` so workers never replay identical augmentations.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "IMAGENET_MEAN", "IMAGENET_STD", "CLIP_MEAN", "CLIP_STD", "SIGLIP_MEAN", "SIGLIP_STD",
    "to_float_tensor", "Normalize", "resize", "resize_short", "center_crop",
    "crop_box", "crop_resize", "TrainAugment", "EvalTransform", "build_transforms",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


# ── conversion ────────────────────────────────────────────────────────────────
def to_float_tensor(images, *, channels_last: bool | None = None,
                    device: torch.device | str | None = None) -> torch.Tensor:
    """``uint8 [...,H,W,3]`` (numpy or torch) → ``float32 [...,3,H,W]`` in ``[0,1]``.

    ``channels_last=None`` infers the layout: uint8 input is channel-last (the storage format);
    float input is channel-last only if its last dim is 3 and dim ``-3`` is not. Float input is
    assumed to already be in ``[0,1]``. Leading dims are preserved (a single ``[H,W,3]`` image
    becomes ``[3,H,W]``).
    """
    if isinstance(images, torch.Tensor):
        x = images
    else:
        arr = np.asarray(images)
        if not (arr.flags.writeable and arr.flags.c_contiguous):
            arr = np.array(arr, copy=True)  # memmap slices are read-only; torch wants writable
        x = torch.from_numpy(arr)
    if device is not None:
        x = x.to(device)
    if x.ndim < 3:
        raise ValueError(f"expected an image [...,H,W,3] or [...,3,H,W], got shape {tuple(x.shape)}")
    is_uint8 = x.dtype == torch.uint8
    if not is_uint8 and not x.is_floating_point():
        raise TypeError(f"images must be uint8 (0..255) or float (0..1), got {x.dtype}")
    if channels_last is None:
        channels_last = True if is_uint8 else (x.shape[-1] == 3 and x.shape[-3] != 3)
    if channels_last:
        if x.shape[-1] != 3:
            raise ValueError(f"channel-last images need 3 channels in the last dim, got shape {tuple(x.shape)}")
        x = x.movedim(-1, -3)
    elif x.shape[-3] != 3:
        raise ValueError(f"channel-first images need 3 channels at dim -3, got shape {tuple(x.shape)}")
    x = x.to(torch.float32)
    if is_uint8:
        x = x / 255.0
    return x.contiguous()


class Normalize(nn.Module):
    """``(x − mean) / std`` over the channel dim ``-3`` (ImageNet defaults). ``inverse`` undoes it."""

    def __init__(self, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD):
        super().__init__()
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std_t = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean_t.numel() != 3 or std_t.numel() != 3:
            raise ValueError("mean and std need 3 values (RGB)")
        if bool((std_t <= 0).any()):
            raise ValueError(f"std must be positive, got {std_t.tolist()}")
        self.register_buffer("mean", mean_t.reshape(3, 1, 1), persistent=False)
        self.register_buffer("std", std_t.reshape(3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x)) / self.std.to(x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x) + self.mean.to(x)

    def extra_repr(self) -> str:
        return f"mean={self.mean.flatten().tolist()}, std={self.std.flatten().tolist()}"


# ── deterministic geometry ────────────────────────────────────────────────────
def _as_hw(size) -> tuple[int, int]:
    if isinstance(size, (int, np.integer)):
        h = w = int(size)
    else:
        if len(size) != 2:
            raise ValueError(f"size must be an int or (h, w), got {size!r}")
        h, w = int(size[0]), int(size[1])
    if h < 1 or w < 1:
        raise ValueError(f"size must be positive, got {(h, w)}")
    return h, w


def _flat(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if x.ndim < 3 or x.shape[-3] != 3:
        raise ValueError(f"expected float images [...,3,H,W], got shape {tuple(x.shape)}")
    lead = tuple(x.shape[:-3])
    return x.reshape(math.prod(lead), *x.shape[-3:]), lead   # explicit N: works for empty batches


def resize(x: torch.Tensor, size, *, mode: str = "bilinear", antialias: bool = True) -> torch.Tensor:
    """Resize float ``[...,3,H,W]`` to ``size`` (int → square, or ``(h, w)``)."""
    h, w = _as_hw(size)
    xf, lead = _flat(x)
    if tuple(xf.shape[-2:]) == (h, w):
        return x
    kw = {"antialias": antialias} if mode in ("bilinear", "bicubic") else {}
    ac = False if mode in ("bilinear", "bicubic") else None
    out = F.interpolate(xf, size=(h, w), mode=mode, align_corners=ac, **kw)
    return out.reshape(*lead, *out.shape[-3:])


def resize_short(x: torch.Tensor, size: int, **kw) -> torch.Tensor:
    """Resize so that the short side equals ``size`` (aspect preserved)."""
    H, W = x.shape[-2:]
    s = int(size) / min(H, W)
    h, w = (int(size), max(1, round(W * s))) if H <= W else (max(1, round(H * s)), int(size))
    return resize(x, (h, w), **kw)


def center_crop(x: torch.Tensor, size) -> torch.Tensor:
    """Centre crop ``[...,3,H,W]`` to ``size``; the crop must fit inside the image."""
    h, w = _as_hw(size)
    H, W = x.shape[-2:]
    if h > H or w > W:
        raise ValueError(f"crop {(h, w)} larger than image {(H, W)}; resize_short first")
    top, left = (H - h) // 2, (W - w) // 2
    return x[..., top:top + h, left:left + w]


def crop_box(in_hw: tuple[int, int], out_hw: tuple[int, int], scale: torch.Tensor,
             log_ratio: torch.Tensor, *, stretch: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop extent fractions ``(fw, fh)`` (1 = full image width / height) of a crop box.

    The crop aspect (pixels) is the *output* aspect × ``exp(log_ratio)`` (``stretch=True``: the
    *input* aspect, i.e. the whole image is squashed into the output). ``scale`` is the area
    fraction of the **largest crop with that aspect that fits in the image**, so ``scale=1`` with a
    different output aspect is the classic resize-short-side + centre-crop, and ``scale=0.8`` is
    always a 20 % smaller area than that, whatever the camera/output aspects are. (Measuring
    ``scale`` against the full image area instead would clamp every crop of a 4:3 camera with a
    square output to the same box and silently disable the zoom augmentation.)
    """
    H, W = in_hw
    oh, ow = out_hw
    base = (W / H) if stretch else (ow / oh)
    aspect = base * torch.exp(log_ratio)                     # crop width / height in pixels
    w_max = torch.clamp(H * aspect, max=float(W))            # largest crop with this aspect
    lin = torch.sqrt(scale)
    return lin * w_max / W, lin * (w_max / aspect) / H


def crop_resize(x: torch.Tensor, fw: torch.Tensor, fh: torch.Tensor, cx: torch.Tensor,
                cy: torch.Tensor, out_hw: tuple[int, int], *, antialias: bool = True) -> torch.Tensor:
    """Per-sample affine crop + bilinear resize. ``x [N,3,H,W]``; all box params ``[N]``.

    Box = centre ``(cx, cy)`` in normalized ``[-1,1]`` image coords, extent fractions ``(fw, fh)``.
    With ``fw=fh=1, cx=cy=0`` and ``out_hw == (H, W)`` this is the identity (pixel centres,
    ``align_corners=False``). If the crop would be minified by more than 2× along an axis, the
    whole image is first downsampled along that axis with an anti-aliased resize (geometry
    unchanged: boxes are relative), so bilinear sampling does not alias.
    """
    N, _, H, W = x.shape
    oh, ow = out_hw
    if antialias and N > 0:
        gh = oh / (H * float(fh.min()))      # output px per crop px (largest crop in the batch)
        gw = ow / (W * float(fw.min()))
        if min(gh, gw) < 0.5:
            nh = H if gh >= 0.5 else max(oh, math.ceil(H * gh))
            nw = W if gw >= 0.5 else max(ow, math.ceil(W * gw))
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    theta = torch.zeros(N, 2, 3, dtype=x.dtype, device=x.device)
    theta[:, 0, 0] = fw.to(x)
    theta[:, 0, 2] = cx.to(x)
    theta[:, 1, 1] = fh.to(x)
    theta[:, 1, 2] = cy.to(x)
    grid = F.affine_grid(theta, [N, x.shape[1], oh, ow], align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _prepare(images) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Any accepted input → float ``[N,3,H,W]`` + leading dims (≥1 dim: a batch dim is added)."""
    x = to_float_tensor(images)
    if x.ndim == 3:
        x = x.unsqueeze(0)
        lead: tuple[int, ...] = ()
    else:
        lead = tuple(x.shape[:-3])
    return x.reshape(math.prod(lead) if lead else 1, *x.shape[-3:]), lead   # explicit N: empty batches


def _unflatten(x: torch.Tensor, lead: tuple[int, ...]) -> torch.Tensor:
    return x.reshape(*lead, *x.shape[-3:]) if lead else x[0]


def _gray(x: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


# ── transforms ────────────────────────────────────────────────────────────────
class EvalTransform:
    """Deterministic: → float → centre crop of area fraction ``crop_scale`` → resize → normalize.

    Args:
        out_size: output ``(h, w)`` or int; ``None`` keeps the input size.
        crop_scale: area fraction of the centre crop (``1`` = largest crop with the output aspect).
        stretch: squash the whole image to ``out_size`` instead of cropping to its aspect.
        mean, std: normalization (``None`` → return ``[0,1]`` images).
    """

    def __init__(self, out_size=None, *, crop_scale: float = 1.0, stretch: bool = False,
                 mean: Sequence[float] | None = IMAGENET_MEAN,
                 std: Sequence[float] | None = IMAGENET_STD, antialias: bool = True):
        if not 0.0 < crop_scale <= 1.0:
            raise ValueError(f"crop_scale must be in (0, 1], got {crop_scale}")
        self.out_size = None if out_size is None else _as_hw(out_size)
        self.crop_scale, self.stretch, self.antialias = float(crop_scale), bool(stretch), antialias
        self.normalize = None if mean is None or std is None else Normalize(mean, std)

    def __call__(self, images) -> torch.Tensor:
        x, lead = _prepare(images)
        N, _, H, W = x.shape
        out = self.out_size or (H, W)
        if N == 0:
            x = x.new_zeros(0, 3, *out)
        elif out != (H, W) or self.crop_scale < 1.0 or self.stretch:
            one = torch.ones(N)
            fw, fh = crop_box((H, W), out, one * self.crop_scale, one * 0.0, stretch=self.stretch)
            zero = torch.zeros(N)
            x = crop_resize(x, fw, fh, zero, zero, out, antialias=self.antialias)
        if self.normalize is not None:
            x = self.normalize(x)
        return _unflatten(x, lead)

    def __repr__(self) -> str:
        # deterministic (no object ids): the feature cache stores it to detect stale caches
        norm = None if self.normalize is None else (
            tuple(round(v, 6) for v in self.normalize.mean.flatten().tolist()),
            tuple(round(v, 6) for v in self.normalize.std.flatten().tolist()))
        return (f"EvalTransform(out_size={self.out_size}, crop_scale={self.crop_scale}, "
                f"stretch={self.stretch}, antialias={bool(self.antialias)}, normalize={norm})")


class TrainAugment:
    """Random resized crop (small scale range) + colour jitter + normalize, all torch ops.

    Parameters are sampled per item of the **first** dim and shared across the remaining leading
    dims, so ``[B,T,3,H,W]`` (an observation history) gets one consistent crop per sample.

    Args:
        out_size: output ``(h, w)`` / int, ``None`` keeps the input size.
        scale: crop area range as a fraction of the largest crop with the output aspect
            (default 0.8–1.0: mild, keeps the hand in view; see :func:`crop_box`).
        ratio: multiplicative aspect jitter range around the output aspect (log-uniform).
        brightness, contrast, saturation: jitter strengths ``β`` → factor ``U[1−β, 1+β]``.
        stretch: see :func:`crop_box`.
        mean, std: normalization (``None`` → ``[0,1]`` output).
        seed: ``None`` → torch global RNG; int → private generator (see module docstring).
    """

    def __init__(self, out_size=None, *, scale: tuple[float, float] = (0.8, 1.0),
                 ratio: tuple[float, float] = (1.0, 1.0), brightness: float = 0.2,
                 contrast: float = 0.2, saturation: float = 0.0, stretch: bool = False,
                 mean: Sequence[float] | None = IMAGENET_MEAN,
                 std: Sequence[float] | None = IMAGENET_STD, seed: int | None = None,
                 antialias: bool = True):
        lo, hi = float(scale[0]), float(scale[1])
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError(f"scale must satisfy 0 < lo <= hi <= 1, got {scale}")
        rlo, rhi = float(ratio[0]), float(ratio[1])
        if not 0.0 < rlo <= rhi:
            raise ValueError(f"ratio must satisfy 0 < lo <= hi, got {ratio}")
        for name, v in (("brightness", brightness), ("contrast", contrast), ("saturation", saturation)):
            if not 0.0 <= v < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {v}")
        self.out_size = None if out_size is None else _as_hw(out_size)
        self.scale, self.ratio = (lo, hi), (rlo, rhi)
        self.brightness, self.contrast, self.saturation = float(brightness), float(contrast), float(saturation)
        self.stretch, self.antialias = bool(stretch), antialias
        self.mean = None if mean is None else tuple(float(m) for m in mean)
        self.std = None if std is None else tuple(float(s) for s in std)
        self.normalize = None if mean is None or std is None else Normalize(mean, std)
        self.seed = seed
        self.generator: torch.Generator | None = None
        self._worker_seed: int | None = None
        if seed is not None:
            self.reseed(seed)

    # ── randomness ────────────────────────────────────────────────────────
    def reseed(self, seed: int) -> None:
        """(Re)create the private generator from ``seed``."""
        self.generator = torch.Generator().manual_seed(int(seed) % (2 ** 63))

    def _gen(self, generator: torch.Generator | None) -> torch.Generator | None:
        if generator is not None:
            return generator
        if self.seed is None:
            return None
        info = torch.utils.data.get_worker_info()
        if info is not None and info.seed != self._worker_seed:
            # per worker & per epoch (info.seed changes each epoch), still reproducible
            self._worker_seed = info.seed
            self.reseed(self.seed * 1_000_003 + info.seed)
        return self.generator

    def sample_params(self, n: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Draw ``n`` parameter sets (CPU tensors): scale, log_ratio, cx/cy in [-1,1], b/c/s factors."""
        u = torch.rand(n, 7, generator=generator, dtype=torch.float64)
        lr = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        return {
            "scale": self.scale[0] + (self.scale[1] - self.scale[0]) * u[:, 0],
            "log_ratio": lr[0] + (lr[1] - lr[0]) * u[:, 1],
            "tx": 2 * u[:, 2] - 1, "ty": 2 * u[:, 3] - 1,
            "brightness": 1 + self.brightness * (2 * u[:, 4] - 1),
            "contrast": 1 + self.contrast * (2 * u[:, 5] - 1),
            "saturation": 1 + self.saturation * (2 * u[:, 6] - 1),
        }

    # ── apply ─────────────────────────────────────────────────────────────
    def __call__(self, images, generator: torch.Generator | None = None) -> torch.Tensor:
        x, lead = _prepare(images)
        N, _, H, W = x.shape
        out = self.out_size or (H, W)
        if N == 0:   # empty batch: nothing to draw (keeps the generator state unchanged)
            return x.new_zeros(*lead, 3, *out)
        B = lead[0] if lead else 1
        rep = N // B
        p = self.sample_params(B, self._gen(generator))
        p = {k: v.repeat_interleave(rep) for k, v in p.items()}
        fw, fh = crop_box((H, W), out, p["scale"], p["log_ratio"], stretch=self.stretch)
        cx, cy = p["tx"] * (1 - fw), p["ty"] * (1 - fh)   # keep the crop inside the image
        x = crop_resize(x, fw, fh, cx, cy, out, antialias=self.antialias)
        x = self._color(x, p)
        if self.normalize is not None:
            x = self.normalize(x)
        return _unflatten(x, lead)

    def _color(self, x: torch.Tensor, p: dict[str, torch.Tensor]) -> torch.Tensor:
        def f(k):
            return p[k].to(x).reshape(-1, 1, 1, 1)
        if self.brightness > 0:
            x = (x * f("brightness")).clamp(0, 1)
        if self.contrast > 0:
            m = _gray(x).mean(dim=(-2, -1), keepdim=True)
            x = ((x - m) * f("contrast") + m).clamp(0, 1)
        if self.saturation > 0:
            g = _gray(x)
            x = ((x - g) * f("saturation") + g).clamp(0, 1)
        return x

    def eval_transform(self, crop_scale: float | None = None) -> EvalTransform:
        """Matching deterministic transform; default crop = mean train crop area (same zoom)."""
        cs = 0.5 * (self.scale[0] + self.scale[1]) if crop_scale is None else crop_scale
        return EvalTransform(self.out_size, crop_scale=cs, stretch=self.stretch, mean=self.mean,
                             std=self.std, antialias=self.antialias)

    # torch.Generator is not reliably picklable (spawned DataLoader workers) → store its state
    def __getstate__(self):
        d = dict(self.__dict__)
        d["generator"] = None if self.generator is None else self.generator.get_state()
        return d

    def __setstate__(self, d):
        state = d.pop("generator")
        self.__dict__.update(d)
        self.generator = None
        if state is not None:
            self.generator = torch.Generator()
            self.generator.set_state(state)

    def __repr__(self) -> str:
        return (f"TrainAugment(out_size={self.out_size}, scale={self.scale}, ratio={self.ratio}, "
                f"brightness={self.brightness}, contrast={self.contrast}, "
                f"saturation={self.saturation}, seed={self.seed})")


def build_transforms(cfg: dict | None = None,
                     encoder=None) -> tuple[TrainAugment | EvalTransform, EvalTransform]:
    """``(train, eval)`` transforms from a config dict.

    Keys (all optional): ``image_size`` (int | [h, w] | null = keep), ``scale``, ``ratio``,
    ``brightness``, ``contrast``, ``saturation``, ``stretch``, ``seed``, ``eval_crop_scale``
    (default: mean train crop area), ``mean``/``std`` (default: ``encoder.mean``/``std`` if an
    encoder is given, else ImageNet), ``augment`` (false → the train transform *is* the eval
    transform: no randomness at all).
    """
    c = dict(cfg or {})
    known = {"image_size", "scale", "ratio", "brightness", "contrast", "saturation", "stretch",
             "seed", "eval_crop_scale", "mean", "std", "augment", "antialias"}
    bad = set(c) - known
    if bad:
        raise ValueError(f"unknown transform option(s) {sorted(bad)}; accepted: {sorted(known)}")
    mean = c.get("mean", getattr(encoder, "mean", IMAGENET_MEAN))
    std = c.get("std", getattr(encoder, "std", IMAGENET_STD))
    size = c.get("image_size")
    common = dict(stretch=bool(c.get("stretch", False)), mean=mean, std=std,
                  antialias=bool(c.get("antialias", True)))
    if not c.get("augment", True):
        eval_tf = EvalTransform(size, crop_scale=float(c.get("eval_crop_scale") or 1.0), **common)
        return eval_tf, eval_tf
    train = TrainAugment(size, scale=tuple(c.get("scale", (0.8, 1.0))),
                         ratio=tuple(c.get("ratio", (1.0, 1.0))),
                         brightness=float(c.get("brightness", 0.2)),
                         contrast=float(c.get("contrast", 0.2)),
                         saturation=float(c.get("saturation", 0.0)), seed=c.get("seed"), **common)
    eval_tf = train.eval_transform(c.get("eval_crop_scale"))
    return train, eval_tf
