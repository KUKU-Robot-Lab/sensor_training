"""Masked-taxel pretraining of :class:`~robot_skin.representation.encoder.TaxelEncoder` (MAE-style).

Grounding: MAE (He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022,
arXiv:2111.06377) — hide a large fraction of the input units, encode only the **visible** ones,
and let a lightweight decoder that sees learned mask tokens + positional embeddings reconstruct
the hidden units; the loss is computed on hidden units only. Here the units are taxels of one
frame (a set with 3D poses instead of an image grid):

- **mask** ``round(mask_ratio·N)`` taxels per sample (:func:`sample_taxel_mask`) — uniformly at
  random, by whole layout **group** (a fingertip, a finger, the palm: forces cross-region
  reasoning instead of copying the nearest neighbour), or ``mixed`` (group with probability
  ``group_prob``); ≥ 1 taxel always stays visible;
- **encoder** — the downstream :class:`TaxelEncoder`, run with the hidden taxels removed as
  attention keys (``key_padding_mask``), i.e. exactly the encoder applied to the visible subset
  (MAE: no mask tokens in the encoder). Their values are additionally replaced by the tokenizer's
  ``[MASK]`` vector so they never enter the network;
- **decoder** — ``decoder_depth`` pre-LN transformer layers over all taxels: visible tokens
  (projected) + a learned decoder mask token at hidden taxels, both plus a decoder pose embedding
  (Fourier(pos) ⊕ normal);
- **targets** (current frame of the hidden taxels): the compressed residual z
  ``zf = z_feature(residual_z)`` (Huber; saturated / non-finite z excluded) and the contact level
  (cross-entropy over :class:`~robot_skin.contact.ordinal.ContactLevel`, optional class weights
  since NONE dominates).

Data: every processed episode (D1 motion + D2 task, glove or robot) that has the contact stage's
derived ``residual_z`` / ``contact_level`` — no labels needed. :class:`TaxelPretrainDataset`
samples frames, :func:`collate_pretrain` pads mixed-layout batches (``pad_mask`` True =
padding). The Trainer contract is ``loss_fn = pretrain_loss`` (calls ``model(batch)`` only).
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.layouts import Layout, load_layout

from ..contact.ordinal import N_LEVELS, ContactLevel
from ..datasets.episode import (D_LEVEL, D_RESIDUAL_Z, K_SATURATED, K_TAXEL_NRM, K_TAXEL_POS,
                                Episode)
from .encoder import TactileFeatureSpec, TaxelEncoder, z_feature
from .tokenizer import fourier_features

__all__ = [
    "MASK_MODES", "random_taxel_mask", "sample_taxel_mask", "layout_group_matrix",
    "TaxelPretrainDataset", "collate_pretrain", "level_class_weights", "MaskedTaxelPretrainer",
    "pretrain_loss", "evaluate_reconstruction",
]

MASK_MODES = ("random", "group", "mixed")
_SAT = int(ContactLevel.SATURATED)


# ─────────────────────────────────────────────────────────────── masking

def random_taxel_mask(batch: int, n_taxels: int, ratio: float, *,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Bool ``[B,N]`` with exactly ``round(ratio*N)`` (≥1 if ratio>0) masked taxels per row."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio must be in [0, 1)")
    k = max(1, round(ratio * n_taxels)) if ratio > 0 else 0
    idx = torch.rand(batch, n_taxels, generator=generator).argsort(dim=1)[:, :k]
    m = torch.zeros(batch, n_taxels, dtype=torch.bool)
    m.scatter_(1, idx, True)
    return m


def sample_taxel_mask(batch: int, n_taxels: int, ratio: float, *,
                      pad_mask: torch.Tensor | None = None, groups: torch.Tensor | None = None,
                      mode: str = "random", group_prob: float = 0.5,
                      generator: torch.Generator | None = None,
                      device: torch.device | str | None = None) -> torch.Tensor:
    """Bool ``[B,N]`` pretraining mask (True = hidden) for padded, grouped batches.

    Per sample with ``n`` real taxels (``pad_mask`` True = padding): ``k = round(ratio·n)``
    (≥ 1 if ratio > 0) random taxels, capped at ``n − 1`` so one stays visible (``n = 1`` → none).

    ``mode="group"``: one eligible group of ``groups[b]`` (``[B,G,N]`` bool membership; eligible
    = non-empty and ≤ ``n − 1`` real taxels) is chosen uniformly and hidden entirely, topped up
    with random taxels to ``k``; samples without an eligible group fall back to random.
    ``mode="mixed"``: group masking with probability ``group_prob`` per sample.
    Deterministic for a given ``generator`` (drawn on its device, then moved to ``device``).
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio must be in [0, 1)")
    if mode not in MASK_MODES:
        raise ValueError(f"mode must be one of {MASK_MODES}, got {mode!r}")
    if not 0.0 <= group_prob <= 1.0:
        raise ValueError("group_prob must be in [0, 1]")
    if device is None:
        device = pad_mask.device if pad_mask is not None else (
            groups.device if groups is not None else "cpu")
    device = torch.device(device)
    gdev = generator.device if generator is not None else device

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=generator, device=gdev).to(device)

    pad = (torch.zeros(batch, n_taxels, dtype=torch.bool, device=device) if pad_mask is None
           else pad_mask.to(device=device, dtype=torch.bool))
    if pad.shape != (batch, n_taxels):
        raise ValueError(f"pad_mask shape {tuple(pad.shape)} != {(batch, n_taxels)}")
    n_valid = (~pad).sum(1)
    k = torch.round(ratio * n_valid.to(torch.float64)).long()
    if ratio > 0:
        k = k.clamp_min(1)
    k = torch.minimum(k, (n_valid - 1).clamp_min(0))
    scores = rand(batch, n_taxels)
    if mode != "random" and groups is not None and groups.shape[1] > 0:
        g = groups.to(device=device, dtype=torch.bool)
        if g.shape[0] != batch or g.shape[2] != n_taxels:
            raise ValueError(f"groups shape {tuple(g.shape)} incompatible with {(batch, n_taxels)}")
        g = g & ~pad[:, None, :]
        gsize = g.sum(-1)                                              # [B,G]
        eligible = (gsize > 0) & (gsize <= (n_valid - 1)[:, None])
        use = eligible.any(1)
        if mode == "mixed":
            use = use & (rand(batch) < group_prob)
        gscore = rand(batch, g.shape[1]).masked_fill(~eligible, -1.0)
        pick = gscore.argmax(1)                                        # [B]
        chosen = g[torch.arange(batch, device=device), pick] & use[:, None]
        scores = torch.where(chosen, torch.full_like(scores, -1.0), scores)
        k = torch.minimum(torch.maximum(k, chosen.sum(1)), (n_valid - 1).clamp_min(0))
    scores = scores.masked_fill(pad, 2.0)                              # padding ranks last
    ranks = scores.argsort(1).argsort(1)
    return (ranks < k[:, None]) & ~pad


def layout_group_matrix(layout: Layout, *, max_group_frac: float = 0.5,
                        n_taxels: int | None = None) -> tuple[list[str], np.ndarray]:
    """Maskable layout groups as ``(names, bool[G,N])``.

    Groups covering more than ``max_group_frac`` of the taxels (e.g. ``all``) and duplicate
    memberships are dropped.
    """
    if not 0.0 < max_group_frac <= 1.0:
        raise ValueError("max_group_frac must be in (0, 1]")
    n = layout.n if n_taxels is None else int(n_taxels)
    if layout.n != n:
        raise ValueError(f"layout {layout.name!r} has {layout.n} taxels, expected {n}")
    names, rows, seen = [], [], set()
    for name, idx in sorted(layout.groups.items()):
        if not idx or len(idx) > max_group_frac * n:
            continue
        key = tuple(sorted(idx))
        if key in seen:
            continue
        seen.add(key)
        row = np.zeros(n, dtype=bool)
        row[list(idx)] = True
        names.append(name)
        rows.append(row)
    mat = np.stack(rows) if rows else np.zeros((0, n), dtype=bool)
    return names, mat


# ─────────────────────────────────────────────────────────────── data

#: per-episode layout copy written by ``datasets.build`` (``build.LAYOUT_FILE``)
EPISODE_LAYOUT_FILE = "layout.yaml"

#: taxel-centroid travel (m) above which ``taxel_pos`` looks world-framed rather than in the
#: hand / robot-base frame the Episode contract specifies (finger motion moves it a few cm).
#: ``datasets.build`` ≥ ``/2`` writes hand-frame poses (``meta.preprocessing.taxel_frame``), so the
#: check only guards episodes built by older preprocessing (world-frame glove poses) or other tools.
POSE_FRAME_TRAVEL_M = 0.15


def _centroid_travel(pos: np.ndarray, frames: np.ndarray, n_probe: int = 64) -> float:
    """Largest per-axis range (m) of the taxel centroid over ≤ ``n_probe`` of ``frames``: a hand
    moving in the world frame travels tens of cm, finger motion in the hand frame a few cm."""
    if frames.size < 2:
        return 0.0
    idx = frames[np.unique(np.linspace(0, frames.size - 1, min(n_probe, frames.size)).astype(np.int64))]
    c = np.asarray(pos[idx], dtype=np.float64).mean(axis=1)            # [k,3]
    return float(np.max(c.max(0) - c.min(0)))


def _finite_pose_frames(pos: np.ndarray, nrm: np.ndarray, frames: np.ndarray,
                        chunk: int = 65536) -> np.ndarray:
    """Subset of ``frames`` whose taxel poses are all finite (a NaN position would turn every
    Fourier feature — and the whole training run — into NaN)."""
    keep = np.ones(frames.size, dtype=bool)
    for s in range(0, frames.size, chunk):
        f = frames[s:s + chunk]
        keep[s:s + chunk] = (np.isfinite(np.asarray(pos[f])).all(axis=(1, 2))
                             & np.isfinite(np.asarray(nrm[f])).all(axis=(1, 2)))
    return frames[keep]


class TaxelPretrainDataset(Dataset):
    """Frames of processed episodes → masked-pretraining samples (numpy dicts).

    Args:
        episodes: :class:`Episode` objects or episode directories (loaded with mmap).
        spec: :class:`TactileFeatureSpec` for the input values (``dim ≥ 1``).
        frame_stride: use every ``frame_stride``-th master-clock frame.
        contact_repeat: frames with any WEAK/STRONG/SATURATED taxel appear this many times
            (≥ 1; oversamples the rare contact frames).
        use_groups: attach layout groups for group masking (``layout``, else the episode's
            ``layout.yaml`` copy, else ``meta.layout``; unresolvable → random masking, 1 warning).
        max_group_frac: see :func:`layout_group_matrix`.
        layout: override layout (object, built-in name or YAML path) for all episodes.

    Sample keys: ``values [N,dim]``, ``pos/nrm [N,3]``, ``target_z [N]`` (compressed z of the
    current frame), ``z_valid [N]`` (finite and not saturated), ``target_level [N]`` int64
    (saturation forced to SATURATED; -1 = unknown), ``groups [G,N]``, ``episode``, ``t``.

    Frames whose taxel poses are not finite are skipped (one warning). One warning also lists
    episodes whose taxel centroid travels more than :data:`POSE_FRAME_TRAVEL_M` (world-framed
    ``taxel_pos`` instead of the hand/robot-base frame of the Episode contract — episodes of
    ``datasets.build`` < ``/2``; rebuild them with ``--force``).
    """

    def __init__(self, episodes: Sequence[Episode | str | Path],
                 spec: TactileFeatureSpec | Mapping[str, Any] | None = None, *,
                 frame_stride: int = 1, contact_repeat: int = 1, use_groups: bool = True,
                 max_group_frac: float = 0.5, layout: Layout | str | Path | None = None) -> None:
        self.spec = spec if isinstance(spec, TactileFeatureSpec) else TactileFeatureSpec.from_dict(spec)
        if self.spec.dim < 1:
            raise ValueError("pretraining needs tactile values (obs_mode 'none' has none)")
        if int(frame_stride) < 1 or int(contact_repeat) < 1:
            raise ValueError("frame_stride and contact_repeat must be >= 1")
        self.frame_stride, self.contact_repeat = int(frame_stride), int(contact_repeat)
        self.episodes = [ep if isinstance(ep, Episode) else Episode.load(ep, mmap=True)
                         for ep in episodes]
        self._arrays: list[tuple] = []
        self._groups: list[np.ndarray] = []
        self._frames: list[np.ndarray] = []             # per episode, incl. contact repeats
        self.group_names: list[list[str]] = []
        layout_cache: dict[str, Layout | Exception] = {}
        ep_idx, t_idx = [], []
        world_framed: list[tuple[str, float]] = []
        dropped: list[tuple[str, int]] = []
        no_groups: list[str] = []
        for e, ep in enumerate(self.episodes):
            eid = ep.meta.episode_id
            missing = [k for k in (D_RESIDUAL_Z, D_LEVEL) if not ep.has_derived(k)]
            missing += [k for k in (K_TAXEL_POS, K_TAXEL_NRM) if not ep.has(k)]
            if missing:
                raise KeyError(f"episode {eid!r} lacks {missing} (derived residual_z/contact_level "
                               "come from the contact stage; taxel poses from preprocessing)")
            z, lv = ep.derived(D_RESIDUAL_Z), ep.derived(D_LEVEL)
            sat = ep[K_SATURATED] if ep.has(K_SATURATED) else None
            pos, nrm = ep[K_TAXEL_POS], ep[K_TAXEL_NRM]
            T, N = ep.T, ep.meta.n_taxels
            for key, a in (("residual_z", z), ("contact_level", lv), ("taxel_pos", pos),
                           ("taxel_nrm", nrm)) + ((("saturated", sat),) if sat is not None else ()):
                if a.shape[:2] != (T, N):
                    raise ValueError(f"episode {eid!r}: {key} shape {a.shape} != (T={T}, N={N}, ...)")
            self._arrays.append((z, lv, sat, pos, nrm))
            names, gm = ([], np.zeros((0, N), dtype=bool))
            if use_groups:
                lay, why = self._resolve_layout(ep, layout, layout_cache)
                if lay is not None and lay.n != N:
                    lay, why = None, f"layout {lay.name!r} has {lay.n} taxels, episode {N}"
                if lay is not None:
                    names, gm = layout_group_matrix(lay, max_group_frac=max_group_frac)
                elif why:
                    no_groups.append(f"{eid!r} ({why})")
            self.group_names.append(names)
            self._groups.append(gm)
            frames = np.arange(0, T, self.frame_stride, dtype=np.int64)
            n_all = frames.size
            frames = _finite_pose_frames(pos, nrm, frames)
            if frames.size < n_all:
                dropped.append((eid, n_all - frames.size))
            travel = _centroid_travel(pos, frames)
            if travel > POSE_FRAME_TRAVEL_M:
                world_framed.append((eid, travel))
            if self.contact_repeat > 1 and frames.size:
                lvf = np.asarray(lv[frames]).astype(np.int64)
                if sat is not None:
                    lvf = np.where(np.asarray(sat[frames], dtype=bool), _SAT, lvf)
                touch = ((lvf >= int(ContactLevel.WEAK)) & (lvf <= _SAT)).any(1)
                frames = np.concatenate([frames] + [frames[touch]] * (self.contact_repeat - 1))
            self._frames.append(frames)
            ep_idx.append(np.full(frames.size, e, dtype=np.int64))
            t_idx.append(frames)
        self._ep_idx = np.concatenate(ep_idx) if ep_idx else np.zeros(0, dtype=np.int64)
        self._t_idx = np.concatenate(t_idx) if t_idx else np.zeros(0, dtype=np.int64)
        if no_groups:
            warnings.warn(f"{len(no_groups)}/{len(self.episodes)} episode(s) without a usable layout, "
                          f"group masking disabled for them (random masking only): "
                          f"{'; '.join(no_groups[:3])}", stacklevel=2)
        if dropped:
            warnings.warn(f"{sum(n for _, n in dropped)} frames with non-finite taxel_pos/nrm "
                          f"skipped in {len(dropped)} episode(s), e.g. {dropped[:3]}", stacklevel=2)
        if world_framed:
            ex = ", ".join(f"{i!r} ({d:.2f} m)" for i, d in world_framed[:3])
            warnings.warn(f"{len(world_framed)}/{len(self.episodes)} episode(s) have a taxel centroid "
                          f"travelling > {POSE_FRAME_TRAVEL_M} m (e.g. {ex}) — taxel_pos seems to be "
                          "in a world/camera frame, not the hand/robot-base frame the encoder's "
                          "position features assume (episodes preprocessed before "
                          "robot_skin.datasets.build/2 stored world-frame glove poses: rebuild them "
                          "with `python -m robot_skin.datasets.build --force`)", stacklevel=2)

    @staticmethod
    def _resolve_layout(ep: Episode, override: Layout | str | Path | None,
                        cache: dict[str, Layout | Exception]) -> tuple[Layout | None, str]:
        """``(layout, reason_if_none)`` for group masking. Order: ``override`` → the episode's
        own ``layout.yaml`` copy (written by ``datasets.build``; survives moving the dataset) →
        ``meta.layout`` (built-in name or YAML path) → ``meta.layout`` relative to the episode
        dir. Loads are cached by *resolved file path* (or built-in name), never by a relative
        reference (two episodes may carry different ``layout.yaml`` files)."""
        if isinstance(override, Layout):
            return override, ""
        refs: list[str] = []
        if override is not None:
            refs.append(str(override))
        else:
            if ep.root is not None and (ep.root / EPISODE_LAYOUT_FILE).is_file():
                refs.append(str(ep.root / EPISODE_LAYOUT_FILE))
            ref = str(ep.meta.layout or "")
            if ref:
                refs.append(ref)
                if ep.root is not None and not Path(ref).is_absolute():
                    refs.append(str(ep.root / ref))
        errors: list[str] = []
        for r in refs:
            p = Path(r)
            key = str(p.resolve()) if p.suffix in (".yaml", ".yml") else r
            if key not in cache:
                try:
                    cache[key] = load_layout(r)
                except Exception as e:  # missing / malformed layout: best effort, random masking
                    cache[key] = e
            hit = cache[key]
            if isinstance(hit, Layout):
                return hit, ""
            errors.append(f"{r!r}: {type(hit).__name__}")
        return None, ", ".join(errors)

    def __len__(self) -> int:
        return int(self._t_idx.size)

    def level_counts(self) -> np.ndarray:
        """Histogram ``[N_LEVELS]`` of target levels over all samples (incl. repeats)."""
        counts = np.zeros(N_LEVELS, dtype=np.int64)
        for (_, lv, sat, _, _), t in zip(self._arrays, self._frames):
            if not t.size:
                continue
            ut, rep = np.unique(t, return_counts=True)
            lvf = np.asarray(lv[ut]).astype(np.int64)
            if sat is not None:
                lvf = np.where(np.asarray(sat[ut], dtype=bool), _SAT, lvf)
            for c in range(N_LEVELS):
                counts[c] += int(((lvf == c).sum(1) * rep).sum())
        return counts

    def __getitem__(self, i: int) -> dict[str, Any]:
        e, t = int(self._ep_idx[i]), int(self._t_idx[i])
        z, lv, sat, pos, nrm = self._arrays[e]
        values = self.spec.from_arrays(z, lv, sat, t).astype(np.float32)
        z_t = np.asarray(z[t], dtype=np.float32)
        lv_t = np.asarray(lv[t]).astype(np.int64)
        sat_t = lv_t == _SAT
        if sat is not None:
            sat_t |= np.asarray(sat[t], dtype=bool)
        level = np.where(sat_t, _SAT, lv_t)
        level = np.where((level >= 0) & (level < N_LEVELS), level, -1)
        return {
            "values": values,
            "pos": np.array(pos[t], dtype=np.float32),       # copy: mmap rows are read-only
            "nrm": np.array(nrm[t], dtype=np.float32),
            "target_z": z_feature(z_t, self.spec.z_clip, self.spec.z_scale),
            "z_valid": np.isfinite(z_t) & ~sat_t,
            "target_level": level,
            "groups": self._groups[e],
            "episode": e,
            "t": t,
        }


def collate_pretrain(samples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack samples, padding the taxel axis to the batch max (``pad_mask`` True = padding) and
    the group axis to the max group count (padded groups are empty)."""
    if not samples:
        raise ValueError("empty batch")
    B = len(samples)
    Fdim = {int(s["values"].shape[-1]) for s in samples}
    if len(Fdim) != 1:
        raise ValueError(f"mixed value widths in one batch: {sorted(Fdim)}")
    Fd = Fdim.pop()
    N = max(int(s["values"].shape[0]) for s in samples)
    G = max(int(np.shape(s["groups"])[0]) for s in samples)
    out = {
        "values": torch.zeros(B, N, Fd), "pos": torch.zeros(B, N, 3), "nrm": torch.zeros(B, N, 3),
        "target_z": torch.zeros(B, N), "z_valid": torch.zeros(B, N, dtype=torch.bool),
        "target_level": torch.full((B, N), -1, dtype=torch.long),
        "pad_mask": torch.ones(B, N, dtype=torch.bool),
        "groups": torch.zeros(B, G, N, dtype=torch.bool),
        "episode": torch.tensor([int(s.get("episode", -1)) for s in samples]),
        "t": torch.tensor([int(s.get("t", -1)) for s in samples]),
    }
    for b, s in enumerate(samples):
        n = int(s["values"].shape[0])
        out["values"][b, :n] = torch.as_tensor(s["values"], dtype=torch.float32)
        out["pos"][b, :n] = torch.as_tensor(s["pos"], dtype=torch.float32)
        out["nrm"][b, :n] = torch.as_tensor(s["nrm"], dtype=torch.float32)
        out["target_z"][b, :n] = torch.as_tensor(s["target_z"], dtype=torch.float32)
        out["z_valid"][b, :n] = torch.as_tensor(s["z_valid"], dtype=torch.bool)
        out["target_level"][b, :n] = torch.as_tensor(s["target_level"], dtype=torch.long)
        out["pad_mask"][b, :n] = False
        g = np.asarray(s["groups"], dtype=bool)
        if g.size:
            out["groups"][b, :g.shape[0], :n] = torch.from_numpy(g)
    return out


def level_class_weights(counts: Sequence[float] | np.ndarray, *, power: float = 0.5,
                        max_weight: float = 10.0) -> np.ndarray:
    """Class weights ``∝ freq^-power`` normalised to an expected weight of 1 under the observed
    distribution (``Σ freq·w = 1``), clipped to ``max_weight``; absent classes get 1."""
    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    if c.size != N_LEVELS or np.any(c < 0):
        raise ValueError(f"counts must be {N_LEVELS} non-negative numbers")
    if c.sum() <= 0:
        return np.ones(N_LEVELS, dtype=np.float32)
    freq = c / c.sum()
    w = np.ones(N_LEVELS)
    present = freq > 0
    w[present] = freq[present] ** (-float(power))
    w[present] /= float((freq[present] * w[present]).sum())
    return np.clip(w, 0.0, float(max_weight)).astype(np.float32)


# ─────────────────────────────────────────────────────────────── model

class MaskedTaxelPretrainer(nn.Module):
    """MAE-style masked-taxel autoencoder around a :class:`TaxelEncoder` (see module docstring).

    Args:
        encoder: the encoder to pretrain (kept as ``.encoder``; save it with
            :func:`~robot_skin.representation.encoder.save_pretrained_encoder`).
        value_dim: optional check that ``encoder.value_dim`` matches the data.
        mask_ratio, mask_mode, group_prob: see :func:`sample_taxel_mask`.
        decoder_dim / decoder_depth / decoder_heads: decoder width (default = encoder width),
            layers (≥ 1) and heads (default = encoder heads).
        huber_delta: Huber δ on the compressed z (``zf`` units).
        z_weight, level_weight: loss weights.
        level_class_weights: optional ``[N_LEVELS]`` CE weights (:func:`level_class_weights`).
        eval_seed: masks in eval mode come from a generator re-seeded with this value on every
            call, so validation losses are comparable across epochs.

    ``forward(batch, mask=None)`` returns the loss dict ``{loss, z_huber, level_ce, z_mae,
    level_acc, mask_frac}``; :meth:`predict` returns the raw predictions.
    """

    def __init__(self, encoder: TaxelEncoder, value_dim: int | None = None, *,
                 mask_ratio: float = 0.3, mask_mode: str = "random", group_prob: float = 0.5,
                 decoder_dim: int | None = None, decoder_depth: int = 1,
                 decoder_heads: int | None = None, ff_mult: int = 4, dropout: float = 0.0,
                 huber_delta: float = 1.0, z_weight: float = 1.0, level_weight: float = 1.0,
                 level_class_weights: Sequence[float] | np.ndarray | torch.Tensor | None = None,
                 eval_seed: int = 0) -> None:
        super().__init__()
        if not isinstance(encoder, TaxelEncoder):
            raise TypeError("encoder must be a TaxelEncoder")
        if value_dim is not None and int(value_dim) != encoder.value_dim:
            raise ValueError(f"value_dim={value_dim} != encoder.value_dim={encoder.value_dim}")
        mask_ratio, group_prob = float(mask_ratio), float(group_prob)
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if mask_mode not in MASK_MODES:
            raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
        if not 0.0 <= group_prob <= 1.0:
            raise ValueError("group_prob must be in [0, 1]")
        if float(huber_delta) <= 0 or float(z_weight) < 0 or float(level_weight) < 0:
            raise ValueError("huber_delta must be > 0 and loss weights >= 0")
        if float(z_weight) + float(level_weight) <= 0:
            raise ValueError("at least one of z_weight / level_weight must be > 0")
        self.encoder = encoder
        self.mask_ratio, self.mask_mode, self.group_prob = mask_ratio, mask_mode, group_prob
        self.huber_delta = float(huber_delta)
        self.z_weight, self.level_weight = float(z_weight), float(level_weight)
        self.eval_seed = int(eval_seed)
        D = encoder.d_model
        Dd = int(decoder_dim) if decoder_dim else D
        heads = int(decoder_heads) if decoder_heads else encoder.config["heads"]
        if Dd % heads:
            raise ValueError(f"decoder_dim={Dd} must be divisible by decoder_heads={heads}")
        if int(decoder_depth) < 1:
            raise ValueError("decoder_depth must be >= 1")
        self._n_fourier = encoder.config["n_fourier"]
        self._fourier_scale = encoder.config["fourier_scale"]
        self.enc_to_dec = nn.Linear(D, Dd)
        self.dec_mask_token = nn.Parameter(torch.zeros(Dd))
        nn.init.normal_(self.dec_mask_token, std=0.02)
        self.dec_pose = nn.Sequential(nn.Linear(6 * self._n_fourier + 3, Dd), nn.GELU(),
                                      nn.Linear(Dd, Dd))
        layer = nn.TransformerEncoderLayer(Dd, heads, dim_feedforward=int(ff_mult) * Dd,
                                           dropout=float(dropout), activation="gelu",
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, int(decoder_depth), norm=nn.LayerNorm(Dd),
                                             enable_nested_tensor=False)
        self.z_head = nn.Linear(Dd, 1)
        self.level_head = nn.Linear(Dd, N_LEVELS)
        w = torch.ones(N_LEVELS) if level_class_weights is None else torch.as_tensor(
            np.asarray(level_class_weights, dtype=np.float32)).reshape(-1)
        if w.numel() != N_LEVELS or bool((w < 0).any()) or not bool(torch.isfinite(w).all()):
            raise ValueError(f"level_class_weights must be {N_LEVELS} finite non-negative numbers")
        self.register_buffer("level_weights", w.float())
        self.decoder_config = dict(decoder_dim=Dd, decoder_depth=int(decoder_depth),
                                   decoder_heads=heads, ff_mult=int(ff_mult), dropout=float(dropout))

    # -- masking ---------------------------------------------------------------------------
    def sample_mask(self, batch: Mapping[str, torch.Tensor],
                    generator: torch.Generator | None = None) -> torch.Tensor:
        """Mask for ``batch`` with this model's settings; eval mode without a generator uses a
        fresh ``eval_seed`` generator (deterministic validation)."""
        values = batch["values"]
        B, N = values.shape[:2]
        if generator is None and not self.training:
            generator = torch.Generator().manual_seed(self.eval_seed)
        return sample_taxel_mask(B, N, self.mask_ratio, pad_mask=batch.get("pad_mask"),
                                 groups=batch.get("groups"), mode=self.mask_mode,
                                 group_prob=self.group_prob, generator=generator,
                                 device=values.device)

    # -- forward ---------------------------------------------------------------------------
    def predict(self, batch: Mapping[str, torch.Tensor], mask: torch.Tensor | None = None, *,
                generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """``{z_pred [B,N] (zf units), level_logits [B,N,C], mask [B,N], pad [B,N]}``."""
        values, pos, nrm = batch["values"], batch["pos"], batch["nrm"]
        B, N = values.shape[:2]
        pad = batch.get("pad_mask")
        pad = (torch.zeros(B, N, dtype=torch.bool, device=values.device) if pad is None
               else pad.bool())
        if mask is None:
            mask = batch.get("mask")
        if mask is None:
            mask = self.sample_mask(batch, generator)
        mask = mask.to(device=values.device, dtype=torch.bool) & ~pad
        hidden = mask | pad
        # MAE encoder: hidden taxels are removed as attention keys (their [MASK]-valued tokens
        # only act as queries whose outputs are discarded) → visible tokens == encoder(visible)
        tokens = self.encoder(values, pos, nrm, mask=mask, key_padding_mask=hidden)
        x = self.enc_to_dec(tokens)
        x = torch.where(mask.unsqueeze(-1), self.dec_mask_token.to(x.dtype).expand_as(x), x)
        pdt = self.dec_pose[0].weight.dtype          # fp32 phases even under bf16 autocast
        pe = torch.cat([fourier_features(pos.to(pdt), self._n_fourier, self._fourier_scale),
                        nrm.to(pdt)], dim=-1)
        x = x + self.dec_pose(pe)
        x = self.decoder(x, src_key_padding_mask=pad if bool(pad.any()) else None)
        return {"z_pred": self.z_head(x).squeeze(-1), "level_logits": self.level_head(x),
                "mask": mask, "pad": pad}

    def forward(self, batch: Mapping[str, torch.Tensor],
                mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = self.predict(batch, mask)
        return self.loss_from_predictions(out, batch)

    def loss(self, batch: Mapping[str, torch.Tensor],
             mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Loss dict of ``batch`` (the spec's ``loss(batch)``; same as ``self(batch, mask)``).
        Inside a DDP/compiled Trainer call the wrapper (``model(batch)``, :func:`pretrain_loss`)."""
        return self(batch, mask)

    def loss_from_predictions(self, out: Mapping[str, torch.Tensor],
                              batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Masked Huber (z) + weighted CE (level) over hidden, valid taxels."""
        m = out["mask"] & ~out["pad"]
        z_pred = out["z_pred"].float()
        logits = out["level_logits"].float()
        mz = m & batch["z_valid"].to(m.device).bool()
        # unused targets (visible / invalid, possibly NaN) must not reach the arithmetic: NaN·0 = NaN
        tz = _masked_target(batch["target_z"].to(z_pred.device).float(), mz)
        hub = F.huber_loss(z_pred, tz, reduction="none", delta=self.huber_delta)
        nz = mz.sum()
        z_loss = (hub * mz).sum() / nz.clamp_min(1)
        tl = batch["target_level"].to(m.device).long()
        ml = m & (tl >= 0) & (tl < N_LEVELS)
        tlc = tl.clamp(0, N_LEVELS - 1)
        ce = F.cross_entropy(logits.reshape(-1, N_LEVELS), tlc.reshape(-1),
                             reduction="none").reshape(tl.shape)
        w = self.level_weights.to(logits.device)[tlc] * ml
        wsum = w.sum()
        level_loss = (ce * w).sum() / torch.where(wsum > 0, wsum, torch.ones_like(wsum))
        loss = self.z_weight * z_loss + self.level_weight * level_loss
        with torch.no_grad():
            z_mae = ((z_pred - tz).abs() * mz).sum() / nz.clamp_min(1)
            nl = ml.sum()
            acc = ((logits.argmax(-1) == tl) & ml).sum() / nl.clamp_min(1)
            frac = m.sum() / (~out["pad"]).sum().clamp_min(1)
        return {"loss": loss, "z_huber": z_loss.detach(), "level_ce": level_loss.detach(),
                "z_mae": z_mae, "level_acc": acc.float(), "mask_frac": frac.float()}

    @property
    def config(self) -> dict:
        return {"mask_ratio": self.mask_ratio, "mask_mode": self.mask_mode,
                "group_prob": self.group_prob, **self.decoder_config,
                "huber_delta": self.huber_delta, "z_weight": self.z_weight,
                "level_weight": self.level_weight,
                "level_class_weights": [float(v) for v in self.level_weights.cpu()],
                "eval_seed": self.eval_seed}


def _masked_target(target: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """``target`` where ``keep`` else 0 — so non-finite unused targets cannot poison sums/grads."""
    return torch.where(keep, target, torch.zeros_like(target))


def pretrain_loss(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``loss_fn`` for :class:`robot_skin.train.Trainer` (forward only → DDP/compile safe)."""
    return model(batch)


# ─────────────────────────────────────────────────────────────── evaluation

@torch.no_grad()
def evaluate_reconstruction(model: MaskedTaxelPretrainer, data: Dataset | DataLoader, *,
                            batch_size: int = 256, device: torch.device | str | None = None,
                            seed: int = 0) -> dict[str, float]:
    """Exact (count-weighted) reconstruction metrics on hidden taxels with seeded masks.

    Returns ``loss, z_huber, z_mae, level_ce, level_acc, level_bal_acc, recall_<level>``,
    ``contact_precision/recall/f1`` (contact = level ≥ WEAK, incl. SATURATED) and trivial
    baselines — ``z_huber_zero`` / ``z_mae_zero`` (predict zf = 0) and ``level_acc_majority``
    (always the most frequent hidden level) — plus the counts ``n_z`` / ``n_level``.
    """
    from ..train.distributed import unwrap_model

    model = unwrap_model(model)
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    loader = data if isinstance(data, DataLoader) else DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=collate_pretrain)
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(int(seed))
    s = dict(hub=0.0, hub0=0.0, mae=0.0, mae0=0.0, nz=0, ce=0.0, wsum=0.0, nl=0, correct=0,
             tp=0, fp=0, fn=0)
    conf = np.zeros((N_LEVELS, N_LEVELS), dtype=np.int64)           # [true, pred]
    try:
        for batch in loader:
            batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model.predict(batch, generator=gen)
            m = out["mask"] & ~out["pad"]
            mz = m & batch["z_valid"].bool()
            zp, tz = out["z_pred"].float(), _masked_target(batch["target_z"].float(), mz)
            s["hub"] += float((F.huber_loss(zp, tz, reduction="none", delta=model.huber_delta)
                               * mz).sum())
            s["hub0"] += float((F.huber_loss(torch.zeros_like(tz), tz, reduction="none",
                                             delta=model.huber_delta) * mz).sum())
            s["mae"] += float(((zp - tz).abs() * mz).sum())
            s["mae0"] += float((tz.abs() * mz).sum())
            s["nz"] += int(mz.sum())
            tl = batch["target_level"].long()
            ml = m & (tl >= 0) & (tl < N_LEVELS)
            logits = out["level_logits"].float()
            tlc = tl.clamp(0, N_LEVELS - 1)
            ce = F.cross_entropy(logits.reshape(-1, N_LEVELS), tlc.reshape(-1),
                                 reduction="none").reshape(tl.shape)
            w = model.level_weights.to(dev)[tlc] * ml
            s["ce"] += float((ce * w).sum())
            s["wsum"] += float(w.sum())
            pred = logits.argmax(-1)
            s["nl"] += int(ml.sum())
            s["correct"] += int(((pred == tl) & ml).sum())
            t_np, p_np = tl[ml].cpu().numpy(), pred[ml].cpu().numpy()
            np.add.at(conf, (t_np, p_np), 1)
            tc, pc = t_np >= int(ContactLevel.WEAK), p_np >= int(ContactLevel.WEAK)
            s["tp"] += int((tc & pc).sum())
            s["fp"] += int((~tc & pc).sum())
            s["fn"] += int((tc & ~pc).sum())
    finally:
        model.train(was_training)

    def ratio(a: float, b: float) -> float:
        return float(a / b) if b > 0 else float("nan")

    z_h, lv_ce = ratio(s["hub"], s["nz"]), ratio(s["ce"], s["wsum"])
    res = {
        "z_huber": z_h, "z_huber_zero": ratio(s["hub0"], s["nz"]),
        "z_mae": ratio(s["mae"], s["nz"]), "z_mae_zero": ratio(s["mae0"], s["nz"]),
        "level_ce": lv_ce, "level_acc": ratio(s["correct"], s["nl"]),
        "level_acc_majority": ratio(conf.sum(1).max() if s["nl"] else 0, s["nl"]),
        "n_z": float(s["nz"]), "n_level": float(s["nl"]),
    }
    # same composition as the training loss: a term without any target contributes 0 — but with
    # no target at all the loss is undefined (NaN), never a spuriously perfect 0.0
    loss_terms = []
    if model.z_weight > 0:
        loss_terms.append(model.z_weight * (z_h if math.isfinite(z_h) else 0.0))
    if model.level_weight > 0:
        loss_terms.append(model.level_weight * (lv_ce if math.isfinite(lv_ce) else 0.0))
    res["loss"] = float(sum(loss_terms)) if (s["nz"] or s["wsum"] > 0) else float("nan")
    recalls = []
    for c in range(N_LEVELS):
        n_c = conf[c].sum()
        r = ratio(conf[c, c], n_c)
        res[f"recall_{ContactLevel(c).name.lower()}"] = r
        if n_c > 0:
            recalls.append(r)
    res["level_bal_acc"] = float(np.mean(recalls)) if recalls else float("nan")
    prec, rec = ratio(s["tp"], s["tp"] + s["fp"]), ratio(s["tp"], s["tp"] + s["fn"])
    res["contact_precision"], res["contact_recall"] = prec, rec
    res["contact_f1"] = (2 * prec * rec / (prec + rec)
                         if math.isfinite(prec) and math.isfinite(rec) and prec + rec > 0
                         else float("nan"))
    return res
