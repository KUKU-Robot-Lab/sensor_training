"""Taxels on a capsule skeleton: embodiment-independent "where on the hand" coordinates.

A hand — the human MANO hand of the glove or a robot hand from its URDF — is approximated by
capsules (bone axis ``p0 → p1`` + radius; ``pose.mano.ManoSkeleton.capsules`` for MANO, links for
URDFs). :func:`project_to_skeleton` maps each taxel to its nearest capsule and returns

- ``segment`` / ``name`` — the capsule (bone) it sits on, and its ``finger`` (thumb … pinky, palm);
- ``t ∈ [0,1]`` — position along that bone (``p0`` → ``p1``), ``closest`` / ``offset`` — the axis
  point and the vector from it to the taxel, ``distance`` / ``surface_distance`` (``− radius``);
- ``u ∈ [0,1]`` — position along the whole finger, base → tip (bone lengths accumulated over the
  finger's capsules in chain order): the fingertip pad of a 3-phalanx human finger and of a
  2-link robot finger both have ``u ≈ 1``;
- ``side = cos∠(offset, palmar direction)`` — +1 on the palmar (pad) side, −1 dorsal (NaN if the
  skeleton has no palmar directions);
- ``v`` — for palm taxels, the lateral position of their palm capsule (the wrist → knuckle ray it
  lies on: thumb −1/3, index 0, middle 1/3, ring 2/3, pinky 1; NaN on fingers).

``(finger, u, side, v)`` are the canonical coordinates used by :func:`robot_skin.transfer.align_layouts`
to put glove and robot taxels on one hand (MANO: Romero et al., SIGGRAPH Asia 2017; cross-embodiment
transfer of tactile gloves to robot hands as in OSMO, arXiv:2512.08920, and DexUMI,
arXiv:2505.21864 — they share the hand model, not the sensor layout).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["HAND_GROUPS", "CapsuleSkeleton", "SkeletonProjection", "project_to_skeleton", "project_to_mano",
           "finger_group"]

HAND_GROUPS = ("thumb", "index", "middle", "ring", "pinky", "palm")
#: lateral coordinate of a palm capsule by the finger it runs to (radial → ulnar)
PALM_LATERAL = {"thumb": -1.0 / 3.0, "index": 0.0, "middle": 1.0 / 3.0, "ring": 2.0 / 3.0, "pinky": 1.0}


def finger_group(name: str) -> str:
    """Finger group of a MANO segment / capsule / URDF link name (``index3``, ``palm_index``,
    ``index_distal_link`` → index / palm …; ``little`` → pinky); unknown names → ``palm`` if they
    look like a palm / base / wrist / hand link, else ``other``."""
    s = str(name).lower()
    if s.startswith("palm") or s.startswith("wrist"):
        return "palm"
    for f in HAND_GROUPS[:5]:
        if s.startswith(f):
            return f
    if s.startswith("little"):
        return "pinky"
    for f in HAND_GROUPS[:5]:
        if f in s:
            return f
    if any(k in s for k in ("palm", "base", "wrist", "hand", "root")):
        return "palm"
    return "other"


@dataclass
class CapsuleSkeleton:
    """Capsules ``p0[S,3] → p1[S,3]`` with ``radius[S]``, ``names[S]``, per-capsule chain ``rank``
    (0 = most proximal within its finger) and optional unit ``palmar[S,3]`` directions."""

    p0: np.ndarray
    p1: np.ndarray
    radius: np.ndarray
    names: list[str]
    rank: np.ndarray | None = None
    palmar: np.ndarray | None = None
    groups: list[str] | None = None

    def __post_init__(self) -> None:
        self.p0 = np.asarray(self.p0, dtype=np.float64).reshape(-1, 3)
        self.p1 = np.asarray(self.p1, dtype=np.float64).reshape(-1, 3)
        S = self.p0.shape[0]
        self.radius = np.broadcast_to(np.asarray(self.radius, dtype=np.float64), (S,)).copy()
        self.names = [str(n) for n in self.names]
        if self.p1.shape != (S, 3) or len(self.names) != S:
            raise ValueError("p0, p1, names must describe the same number of capsules")
        self.groups = [finger_group(n) for n in self.names] if self.groups is None else list(self.groups)
        self.rank = np.zeros(S, dtype=np.int64) if self.rank is None else np.asarray(self.rank, np.int64).reshape(S)
        if self.palmar is not None:
            p = np.asarray(self.palmar, dtype=np.float64).reshape(S, 3)
            self.palmar = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)

    @property
    def n(self) -> int:
        return int(self.p0.shape[0])

    @property
    def lateral(self) -> np.ndarray:
        """``[S]`` palm-capsule lateral coordinate (:data:`PALM_LATERAL` of the finger named after
        ``palm_`` / ``>``); NaN for finger capsules."""
        out = np.full(self.n, np.nan)
        for i, (n, g) in enumerate(zip(self.names, self.groups)):
            if g != "palm":
                continue
            tail = n.split(">", 1)[1] if ">" in n else (n[5:] if n.lower().startswith("palm_") else "")
            f = finger_group(tail) if tail else "other"
            if f in PALM_LATERAL:
                out[i] = PALM_LATERAL[f]
        return out

    @property
    def lengths(self) -> np.ndarray:
        return np.linalg.norm(self.p1 - self.p0, axis=1)

    def finger_u_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        """``(start[S], total[S])``: accumulated bone length before each capsule within its finger
        (chain order by ``rank``) and the finger's total length — ``u = (start + t·len) / total``."""
        L = self.lengths
        start, total = np.zeros(self.n), np.ones(self.n)
        for g in set(self.groups):
            idx = [i for i in range(self.n) if self.groups[i] == g]
            if g == "palm":                       # palm capsules run side by side: u = t
                start[idx], total[idx] = 0.0, L[idx]
                continue
            order = sorted(idx, key=lambda i: (self.rank[i], i))
            acc = 0.0
            tot = float(sum(L[i] for i in order)) or 1.0
            for i in order:
                start[i], total[i] = acc, tot
                acc += L[i]
        return start, np.maximum(total, 1e-12)

    # ── constructors ──────────────────────────────────────────────────────
    @classmethod
    def from_mano(cls, skeleton: Any = None, finger_pose: Any = None) -> "CapsuleSkeleton":
        """The MANO capsules of ``ManoSkeleton.capsules`` in the **hand frame** (global orientation
        0, wrist at the origin — the frame of the processed episodes' glove ``taxel_pos``) for
        ``finger_pose[15,3]`` (default flat hand); palmar = −y of the bone-aligned segment frames."""
        import torch

        from ..pose.mano import CAPSULE_NAMES, ManoSkeleton

        sk = skeleton if skeleton is not None else ManoSkeleton()
        fp = torch.zeros(15, 3, dtype=torch.float64) if finger_pose is None else \
            torch.as_tensor(np.asarray(finger_pose, dtype=np.float64).reshape(15, 3))
        with torch.no_grad():
            fk = sk.forward(finger_pose=fp)
            p0, p1, rad, names = sk.capsules(fk)
            seg = sk.segment_transforms(fk)
        palmar, rank = [], []
        for n in names:
            frame = seg["palm" if n.startswith("palm") else n]
            palmar.append(-frame[:3, 1].numpy())
            rank.append(int(n[-1]) - 1 if n[-1].isdigit() else 0)
        assert list(names) == list(CAPSULE_NAMES)
        return cls(p0.numpy(), p1.numpy(), rad, list(names), np.asarray(rank), np.stack(palmar))

    @classmethod
    def from_urdf(cls, model: Any, q: Sequence[float] | None = None, *, radius: float = 0.008,
                  tip_length: float | Mapping[str, float] | None = None,
                  palmar_axis: Sequence[float] | None = None, min_length: float = 1e-4) -> "CapsuleSkeleton":
        """Capsules of a URDF hand at ``q`` (default zeros, clipped to limits), root-link frame: one per
        parent → child link pair (``link`` origin → child origin; ``palm_link>index_proximal_link``
        style names when a link has several children) and one for every leaf link, extended along its
        incoming bone by ``tip_length`` (m; per-link dict, default 0.8 × the parent bone). Zero-length
        pairs (co-located frames) are skipped. ``palmar_axis`` (link frame, e.g. ``(1, 0, 0)`` for
        :func:`robot_skin.datasets.synthetic.robot_hand_urdf`, whose pads face +x) gives ``side``."""
        D = model.n_dof
        qv = np.zeros(D) if q is None else np.asarray(q, dtype=np.float64).reshape(D)
        qv = np.clip(qv, model.lower, model.upper)
        fk = model.fk_numpy(qv)
        children: dict[str, list[str]] = {}
        for j in model.joints:
            children.setdefault(j.parent, []).append(j.child)
        depth = {ln: len(model.chain(ln)) for ln in model.link_names}
        p0, p1, names, rank, palm = [], [], [], [], []
        ax = None if palmar_axis is None else np.asarray(palmar_axis, dtype=np.float64)

        def add(link: str, a: np.ndarray, b: np.ndarray, name: str) -> None:
            if np.linalg.norm(b - a) < min_length:
                return
            p0.append(a)
            p1.append(b)
            names.append(name)
            rank.append(depth[link])
            palm.append(None if ax is None else fk[link][:3, :3] @ ax)

        for ln in model.link_names:
            o = fk[ln][:3, 3]
            kids = children.get(ln, [])
            for c in kids:
                add(ln, o, fk[c][:3, 3], ln if len(kids) == 1 else f"{ln}>{c}")
            if not kids:
                par, d = ln, np.zeros(3)
                while model.parent_joint(par) is not None and np.linalg.norm(d) < min_length:
                    par = model.parent_joint(par).parent
                    d = o - fk[par][:3, 3]
                if np.linalg.norm(d) < min_length:
                    continue
                L = (tip_length.get(ln) if isinstance(tip_length, Mapping) else tip_length)
                L = 0.8 * float(np.linalg.norm(d)) if L is None else float(L)
                add(ln, o, o + d / np.linalg.norm(d) * L, ln)
        if not names:
            raise ValueError("the URDF gives no capsules (all links co-located?)")
        pal = None if ax is None else np.stack(palm)
        return cls(np.stack(p0), np.stack(p1), radius, names, np.asarray(rank), pal)


@dataclass
class SkeletonProjection:
    """Result of :func:`project_to_skeleton` (arrays over the taxel axis, see module docstring)."""

    segment: np.ndarray
    names: list[str]
    finger: np.ndarray            # object array of group names, shape of ``segment``
    t: np.ndarray
    u: np.ndarray
    side: np.ndarray
    v: np.ndarray
    closest: np.ndarray
    offset: np.ndarray
    distance: np.ndarray
    surface_distance: np.ndarray

    @property
    def segment_names(self) -> list[str]:
        return [self.names[int(i)] for i in np.asarray(self.segment).reshape(-1)]


def _as_skeleton(sk: Any) -> CapsuleSkeleton:
    if isinstance(sk, CapsuleSkeleton):
        return sk
    if isinstance(sk, (tuple, list)) and len(sk) in (3, 4):
        p0, p1, rad = sk[0], sk[1], sk[2]
        names = list(sk[3]) if len(sk) == 4 else [f"seg{i}" for i in range(np.shape(p0)[-2])]
        p0 = p0.detach().cpu().numpy() if hasattr(p0, "detach") else p0
        p1 = p1.detach().cpu().numpy() if hasattr(p1, "detach") else p1
        return CapsuleSkeleton(p0, p1, rad, names)
    raise TypeError("skeleton must be a CapsuleSkeleton or a (p0, p1, radius[, names]) tuple")


def project_to_skeleton(taxel_pos: np.ndarray, skeleton: Any, *, taxel_groups: Sequence[str] | None = None,
                        allowed: np.ndarray | None = None) -> SkeletonProjection:
    """Project taxel positions ``[..., N, 3]`` onto the nearest capsule axis of ``skeleton``
    (:class:`CapsuleSkeleton` or ``ManoSkeleton.capsules`` output), in the skeleton's frame.

    ``taxel_groups`` (finger group per taxel, e.g. from :func:`robot_skin.transfer.taxel_groups`)
    restricts each taxel to capsules of its own group (falling back to all capsules when its group
    has none); ``allowed`` ``bool[N, S]`` restricts explicitly."""
    sk = _as_skeleton(skeleton)
    p = np.asarray(taxel_pos, dtype=np.float64)
    if p.shape[-1] != 3:
        raise ValueError(f"taxel_pos must be [..., N, 3], got {p.shape}")
    N, S = p.shape[-2], sk.n
    a, b = sk.p0, sk.p1
    ab = b - a
    den = np.maximum(np.sum(ab * ab, axis=-1), 1e-18)                                  # [S]
    tt = np.clip(np.einsum("...nsk,sk->...ns", p[..., :, None, :] - a, ab) / den, 0.0, 1.0)  # [...,N,S]
    cl = a + tt[..., None] * ab                                                        # [...,N,S,3]
    d = np.linalg.norm(p[..., :, None, :] - cl, axis=-1)                               # [...,N,S]
    mask = np.ones((N, S), dtype=bool)
    if taxel_groups is not None:
        if len(taxel_groups) != N:
            raise ValueError(f"{len(taxel_groups)} taxel groups for {N} taxels")
        for i, g in enumerate(taxel_groups):
            m = np.array([sg == g for sg in sk.groups])
            if m.any():
                mask[i] = m
    if allowed is not None:
        al = np.asarray(allowed, dtype=bool)
        if al.shape != (N, S):
            raise ValueError(f"allowed must be [{N}, {S}], got {al.shape}")
        mask &= al
        if not mask.any(1).all():
            raise ValueError("some taxels have no allowed capsule")
    d = np.where(mask, d, np.inf)
    seg = np.argmin(d, axis=-1)                                                        # [...,N]
    take = lambda x: np.take_along_axis(x, seg[..., None], axis=-1)[..., 0]           # noqa: E731
    t = take(tt)
    dist = take(d)
    closest = np.take_along_axis(cl, seg[..., None, None].repeat(3, -1), axis=-2)[..., 0, :]
    offset = p - closest
    start, total = sk.finger_u_offsets()
    L = sk.lengths
    u = (start[seg] + t * L[seg]) / total[seg]
    if sk.palmar is not None:
        pal = sk.palmar[seg]
        on = np.linalg.norm(offset, axis=-1)
        side = np.where(on > 1e-9, np.sum(offset * pal, axis=-1) / np.maximum(on, 1e-12), np.nan)
    else:
        side = np.full(t.shape, np.nan)
    finger = np.asarray(sk.groups, dtype=object)[seg]
    return SkeletonProjection(segment=seg, names=list(sk.names), finger=finger, t=t, u=u, side=side,
                              v=sk.lateral[seg],
                              closest=closest, offset=offset, distance=dist, surface_distance=dist - sk.radius[seg])


def project_to_mano(taxel_pos: np.ndarray, skeleton: Any = None, finger_pose: Any = None, **kw: Any
                    ) -> SkeletonProjection:
    """:func:`project_to_skeleton` onto the MANO capsules (:meth:`CapsuleSkeleton.from_mano`, hand
    frame, ``finger_pose`` default flat) — e.g. robot taxels already expressed in the MANO hand frame."""
    return project_to_skeleton(taxel_pos, CapsuleSkeleton.from_mano(skeleton, finger_pose), **kw)
