"""Per-frame visual feature cache for frozen encoders (training speed-up).

With a frozen image backbone (ResNet / DINOv2 / SigLIP) the visual tokens of a frame never
change, so they are computed once per camera frame and stored next to the episode::

    <episode>/derived/vision_<key>_<camera>.npy    float16 [F,P,D]   (F = camera frames)
    <episode>/derived/vision_<key>_<camera>.json   sidecar: encoder, transform, shapes, time

Features are **per camera frame**, not per master-clock tick (cameras run at ~30 Hz, the master
clock at 200 Hz). Map a master index ``i`` to its frame with ``episode[cam_idx_key(camera)][i]``
(``-1`` before the first frame) and use :func:`gather_frame_features`. The VTLA dataset reads
these files instead of decoding + encoding images (``use_cached_vision=<key>``); augmentation is
then impossible, which is the usual trade-off for frozen features.

These files deliberately bypass ``Episode.set_derived`` (which requires ``T`` rows); load them
with :func:`load_cached`, not ``Episode.derived``.

Staleness: an existing cache is reused only if its row count matches the camera and its sidecar
records the same encoder ``cache_key``, ``out_dim`` and eval transform; otherwise
:func:`cache_episode_features` raises (pass ``overwrite=True`` or use another ``key``). Only the
sidecar and ``camera_<name>/timestamps.npy`` are needed to validate a cache, so a training box
can receive ``derived/`` + ``timestamps.npy`` without the (large) frames.

CLI (e.g. on the GPU box, once per encoder)::

    python -m robot_skin.vision.feature_cache --root robot_skin/data/processed --dataset task \
        --encoder '{"type": "resnet18"}' --image-size 224 224 --device cuda --batch-size 128
"""
from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch

from robot_skin.datasets.episode import Episode, list_episodes

from .encoders import VisionEncoder
from .transforms import EvalTransform

__all__ = [
    "feature_path", "has_cached", "cache_episode_features", "cache_features", "load_cached",
    "load_cache_info", "gather_frame_features", "main",
]

_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
_F16_MAX = float(np.finfo(np.float16).max)


def _check_name(name: str, what: str) -> str:
    if not isinstance(name, str) or not _KEY_RE.match(name):
        raise ValueError(f"{what} {name!r} must match [A-Za-z0-9][A-Za-z0-9_.-]* "
                         f"(use vision.encoders.sanitize_key)")
    return name


def _root(episode: Episode | str | Path) -> Path:
    if isinstance(episode, Episode):
        if episode.root is None:
            raise ValueError("feature caching needs an on-disk episode (Episode.save it first)")
        return episode.root
    return Path(episode)


def feature_path(episode: Episode | str | Path, camera: str, key: str) -> Path:
    """``<episode>/derived/vision_<key>_<camera>.npy``."""
    _check_name(key, "feature key")
    _check_name(camera, "camera name")
    return _root(episode) / "derived" / f"vision_{key}_{camera}.npy"


def _info_path(path: Path) -> Path:
    return path.with_suffix(".json")


def load_cache_info(episode: Episode | str | Path, camera: str, key: str) -> dict | None:
    p = _info_path(feature_path(episode, camera, key))
    return json.loads(p.read_text()) if p.is_file() else None


def _n_frames(root: Path, camera: str) -> int:
    ts = root / f"camera_{camera}" / "timestamps.npy"
    if not ts.is_file():
        raise FileNotFoundError(f"no camera {camera!r} in episode {root} (missing {ts})")
    return int(np.load(ts, mmap_mode="r").shape[0])


def _expected_frames(root: Path, camera: str, path: Path) -> int | None:
    """Camera frame count from ``timestamps.npy``, else from the cache sidecar, else ``None``."""
    try:
        return _n_frames(root, camera)
    except FileNotFoundError:
        pass
    try:
        return int(json.loads(_info_path(path).read_text())["n_frames"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def has_cached(episode: Episode | str | Path, camera: str, key: str) -> bool:
    """True if a complete cache file exists and matches the camera's frame count (when known)."""
    path = feature_path(episode, camera, key)
    if not path.is_file():
        return False
    try:
        n = _expected_frames(_root(episode), camera, path)
        return n is None or int(np.load(path, mmap_mode="r").shape[0]) == n
    except (OSError, ValueError):
        return False


def _describe(transform) -> str | None:
    """Deterministic description of a transform (``None`` if its repr is an object address)."""
    r = repr(transform)
    return None if " at 0x" in r else r


def _signature(encoder: VisionEncoder, transform) -> dict:
    return {"encoder_cache_key": encoder.cache_key, "out_dim": int(encoder.out_dim),
            "transform": _describe(transform)}


def _signature_mismatch(info: dict | None, sig: dict) -> list[str]:
    if info is None:        # no sidecar (e.g. written by hand): nothing to compare
        return []
    return [f"{k}: cached {info.get(k)!r} vs now {v!r}" for k, v in sig.items()
            if v is not None and info.get(k) is not None and info.get(k) != v]


def _frame_batches(root: Path, camera: str, n: int, batch_size: int) -> Iterator[np.ndarray]:
    cam = root / f"camera_{camera}"
    npy = cam / "frames.npy"
    if npy.is_file():
        frames = np.load(npy, mmap_mode="r")
        if frames.shape[0] != n:
            raise ValueError(f"{npy} has {frames.shape[0]} frames but timestamps.npy has {n}")
        for s in range(0, n, batch_size):
            yield np.array(frames[s:s + batch_size])   # copy: writable, contiguous
        return
    from PIL import Image  # jpg storage only
    for s in range(0, n, batch_size):
        yield np.stack([np.asarray(Image.open(cam / f"{i:06d}.jpg").convert("RGB"))
                        for i in range(s, min(n, s + batch_size))])


def cache_episode_features(episode: Episode | str | Path, camera: str, encoder: VisionEncoder,
                           transform: Callable[[torch.Tensor], torch.Tensor] | None = None, *,
                           batch_size: int = 64, device: str | torch.device = "cpu",
                           key: str | None = None, overwrite: bool = False,
                           autocast_dtype: torch.dtype | None = None) -> Path:
    """Encode every frame of ``camera`` and write ``derived/vision_<key>_<camera>.npy`` ``[F,P,D]`` f16.

    Args:
        episode: an on-disk :class:`Episode` or its directory.
        camera: camera name (frames in ``camera_<name>/``).
        encoder: a :class:`VisionEncoder`; should be frozen (``is_frozen``) — a warning is issued
            otherwise because the cache would go stale as the encoder trains.
        transform: ``uint8 [b,H,W,3]`` tensor (on ``device``) → normalized float ``[b,3,h,w]``;
            default :class:`EvalTransform` at native size with the encoder's mean/std.
        batch_size, device: encoding batch size / device (the transform runs there too).
        key: cache name (default ``encoder.cache_key``).
        overwrite: recompute even if a cache exists. Without it, an existing cache with the right
            frame count is reused, unless its sidecar shows a different encoder/transform, which
            raises ``ValueError`` (never silently serves features of another pipeline).
        autocast_dtype: e.g. ``torch.bfloat16`` for faster encoding on CUDA.

    Returns the ``.npy`` path. The file is written atomically (temp file + rename). The encoder
    is moved to ``device`` (and left there); its train/eval mode is restored.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    ep_root = _root(episode)
    key = encoder.cache_key if key is None else key
    path = feature_path(ep_root, camera, key)   # validates key / camera
    n = _n_frames(ep_root, camera)
    tf = transform or EvalTransform(None, mean=encoder.mean, std=encoder.std)
    sig = _signature(encoder, tf)
    if not overwrite and has_cached(ep_root, camera, key):
        bad = _signature_mismatch(load_cache_info(ep_root, camera, key), sig)
        if bad:
            raise ValueError(f"feature cache {path} exists but was built differently ({'; '.join(bad)}); "
                             "pass overwrite=True to recompute it, or use another key")
        return path
    if not encoder.is_frozen:
        warnings.warn(f"caching features of an encoder with trainable parameters ({type(encoder).__name__}); "
                      "the cache will not follow training — freeze it (encoder.freeze()) or use "
                      "out_dim=None, frozen=True", stacklevel=2)
    device = torch.device(device)
    was_training = encoder.training
    encoder.eval().to(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    out = None
    in_hw = out_hw = None
    n_clipped = 0
    try:
        with torch.inference_mode():
            s = 0
            for frames in _frame_batches(ep_root, camera, n, batch_size):
                x = tf(torch.from_numpy(frames).to(device)).to(device)
                if in_hw is None:
                    in_hw, out_hw = list(frames.shape[1:3]), list(x.shape[-2:])
                if autocast_dtype is not None:
                    with torch.autocast(device.type, dtype=autocast_dtype):
                        z = encoder(x)
                else:
                    z = encoder(x)
                z = z.float().cpu().numpy()
                if out is None:
                    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                                    shape=(n, *z.shape[1:]))
                big = np.abs(z) > _F16_MAX
                if big.any():
                    n_clipped += int(big.sum())
                    z = np.clip(z, -_F16_MAX, _F16_MAX)
                out[s:s + len(z)] = z.astype(np.float16)
                s += len(z)
        if out is None:  # zero frames
            d = int(encoder.out_dim)
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(0, 0, d))
        out.flush()
        shape = list(out.shape)
        del out
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
        encoder.train(was_training)
    if n_clipped:
        warnings.warn(f"{n_clipped} feature values exceeded the float16 range and were clipped "
                      f"({path.name})", stacklevel=2)
    info = {"key": key, "camera": camera, "n_frames": n, "shape": shape, "dtype": "float16",
            "encoder": type(encoder).__name__, **sig, "frozen": bool(encoder.is_frozen),
            "mean": list(encoder.mean), "std": list(encoder.std),
            "image_hw": in_hw, "encoded_hw": out_hw,
            "created_utc": datetime.now(timezone.utc).isoformat()}
    if info["transform"] is None:
        info["transform_repr"] = repr(tf)   # informative only (not compared)
    info_path = _info_path(path)
    tmp_info = info_path.with_name(info_path.name + ".tmp")
    tmp_info.write_text(json.dumps(info, indent=2))
    os.replace(tmp_info, info_path)
    return path


def cache_features(episodes: Iterable[Episode | str | Path], cameras: Sequence[str] | None,
                   encoder: VisionEncoder, transform=None, **kw) -> list[Path]:
    """:func:`cache_episode_features` for many episodes. ``cameras=None`` → each episode's
    ``meta.cameras``."""
    paths = []
    for ep in episodes:
        e = ep if isinstance(ep, Episode) else Episode.load(ep, keys=())
        for cam in (cameras if cameras is not None else e.meta.cameras):
            paths.append(cache_episode_features(e, cam, encoder, transform, **kw))
    return paths


def load_cached(episode: Episode | str | Path, camera: str, key: str, *, mmap: bool = True,
                validate: bool = True) -> np.ndarray:
    """Cached features ``float16 [F,P,D]`` (memory-mapped by default).

    ``validate`` checks the row count against the camera's ``timestamps.npy`` (a stale cache
    after re-preprocessing raises instead of silently misaligning frames); if the camera
    directory was not copied, the sidecar's frame count is used, and without either the check
    is skipped.
    """
    path = feature_path(episode, camera, key)
    if not path.is_file():
        raise FileNotFoundError(f"no cached features {path.name} in {path.parent} — run "
                                f"vision.feature_cache.cache_episode_features(episode, {camera!r}, encoder)")
    feats = np.load(path, mmap_mode="r" if mmap else None)
    if validate:
        n = _expected_frames(_root(episode), camera, path)
        if n is not None and feats.shape[0] != n:
            raise ValueError(f"stale feature cache {path}: {feats.shape[0]} rows, camera has {n} frames "
                             "(re-run with overwrite=True)")
    return feats


def gather_frame_features(features: np.ndarray, frame_idx) -> tuple[np.ndarray, np.ndarray]:
    """Features for frame indices (``-1`` = no frame yet) → ``(float32 [...,P,D], valid [...] bool)``.

    ``frame_idx`` is an int or int array (e.g. ``episode[cam_idx_key(cam)][ticks]``); invalid
    entries are zero-filled.
    """
    idx = np.asarray(frame_idx, dtype=np.int64)
    if idx.size and int(idx.max(initial=-1)) >= features.shape[0]:
        raise IndexError(f"frame index {int(idx.max())} out of range for {features.shape[0]} frames")
    valid = idx >= 0
    safe = np.where(valid, idx, 0)
    if features.shape[0] == 0:
        out = np.zeros((*idx.shape, *features.shape[1:]), np.float32)
    else:
        out = np.asarray(features[safe.reshape(-1)], dtype=np.float32).reshape(*idx.shape, *features.shape[1:])
        out[~valid] = 0.0
    return out, valid


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_encoder_cfg(s: str) -> dict:
    p = Path(s)
    if p.is_file():
        import yaml
        d = yaml.safe_load(p.read_text()) or {}
        return dict(d.get("vision", d)) if isinstance(d, dict) else {}
    try:
        import yaml
        d = yaml.safe_load(s)  # JSON is valid YAML
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"--encoder: not a file and not JSON/YAML: {e}")
    if not isinstance(d, dict):
        raise SystemExit("--encoder must be a mapping, e.g. '{\"type\": \"resnet18\"}'")
    return d


def main(argv: Sequence[str] | None = None) -> int:
    from .encoders import build_vision_encoder

    ap = argparse.ArgumentParser(prog="python -m robot_skin.vision.feature_cache",
                                 description="Cache frozen-encoder features for processed episodes.")
    ap.add_argument("--root", required=True, help="processed root (robot_skin/data/processed) or one episode dir")
    ap.add_argument("--dataset", default=None, help="motion | task | … (subdir of --root)")
    ap.add_argument("--cameras", nargs="*", default=None, help="default: every camera in episode.json")
    ap.add_argument("--encoder", default='{"type": "resnet18"}', help="JSON/YAML mapping or YAML file")
    ap.add_argument("--image-size", nargs=2, type=int, default=None, metavar=("H", "W"))
    ap.add_argument("--crop-scale", type=float, default=1.0)
    ap.add_argument("--key", default=None, help="cache key (default: encoder.cache_key)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--bf16", action="store_true", help="autocast bfloat16 (CUDA)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    enc = build_vision_encoder(_parse_encoder_cfg(a.encoder)).freeze()
    tf = EvalTransform(a.image_size, crop_scale=a.crop_scale, mean=enc.mean, std=enc.std)
    root = Path(a.root)
    eps = [root] if (root / "episode.json").is_file() else list_episodes(root, a.dataset)
    paths = cache_features(eps, a.cameras, enc, tf, batch_size=a.batch_size, device=a.device,
                           key=a.key, overwrite=a.overwrite,
                           autocast_dtype=torch.bfloat16 if a.bf16 else None)
    for p in paths:
        print(p)
    print(f"cached {len(paths)} camera streams from {len(eps)} episodes (key={a.key or enc.cache_key})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
