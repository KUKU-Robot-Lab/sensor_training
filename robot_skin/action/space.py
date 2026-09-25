"""Action spaces: the canonical 54-D MANO hand action and robot joint actions.

Canonical action ``hand_mano`` — what the VTLA policy predicts from human demonstrations (D2),
independent of any particular robot::

    a = [ wrist_pos (3) | wrist_rot 6D (6) | finger_pose axis-angle (15 × 3 = 45) ]   → 54-D
          0:3             3:9                 9:54

- ``wrist_pos`` — MANO wrist position (m) in the frame of ``hand_pose.npz`` (world / fixed third
  camera recommended; with only an egocentric camera use relative actions, see below).
- ``wrist_rot`` — MANO ``global_orient`` as the continuous 6D representation (first two columns
  of R, Zhou et al., CVPR 2019, arXiv:1812.07035). Axis-angle / quaternions are discontinuous
  and hurt regression of a free-floating wrist.
- ``finger_pose`` — 15 MANO finger joints (MANO order index1..3, middle1..3, pinky1..3,
  ring1..3, thumb1..3; Romero et al., SIGGRAPH Asia 2017) kept as axis-angle: anatomical finger
  rotations stay well below π, where axis-angle is smooth, so 45-D is compact and continuous in
  practice. Decoding is lossless (``hand_action_to_arrays``).

``robot_joint`` = raw robot joint targets ``q[D]`` (URDF actuated order) for when robot teleop
data exists. Human hand actions reach a robot through :mod:`robot_skin.action.retarget`.

Relative actions (:func:`make_relative` / :func:`make_absolute`): chunks can be expressed
relative to the current state (the proprio at the chunk's start), which removes the dependence
on where the episode happened in the world frame — useful with a moving egocentric camera and
common practice for chunked policies. Normalisation (:class:`ActionNormalizer`) wraps
``common.signal.NormStats`` and is fit on the *same* representation the policy is trained on.

All array functions accept numpy or torch; numpy in → float32 numpy out, torch in → torch out
(same float dtype / device).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import torch

from common.signal import NormStats
from robot_skin.geometry.rotations import (aa_to_matrix, as_tensor, matrix_to_6d, matrix_to_aa,
                                           sixd_to_matrix)

# ── layout of the canonical hand action ─────────────────────────────────────
HAND_MANO_DIM = 54
N_FINGER_JOINTS = 15
#: MANO finger joints in MANO order (= ``robot_skin.pose.mano.MANO_JOINTS[1:]``).
MANO_FINGER_JOINTS: tuple[str, ...] = (
    "index1", "index2", "index3",
    "middle1", "middle2", "middle3",
    "pinky1", "pinky2", "pinky3",
    "ring1", "ring2", "ring3",
    "thumb1", "thumb2", "thumb3",
)
WRIST_POS = slice(0, 3)
WRIST_ROT6D = slice(3, 9)
FINGERS_AA = slice(9, 54)
HAND_SLICES: dict[str, slice] = {"wrist_pos": WRIST_POS, "wrist_rot6d": WRIST_ROT6D,
                                 "finger_aa": FINGERS_AA}
HAND_MANO_NAMES: tuple[str, ...] = (
    ("wrist_x", "wrist_y", "wrist_z")
    + tuple(f"wrist_r6d_{i}" for i in range(6))
    + tuple(f"{j}_{ax}" for j in MANO_FINGER_JOINTS for ax in "xyz"))
assert len(HAND_MANO_NAMES) == HAND_MANO_DIM

ACTION_KINDS = ("hand_mano", "robot_joint")
#: ``abs``: absolute; ``delta``: hand → wrist position minus current (world frame), robot → q − q_cur;
#: ``delta_pose`` (hand only): wrist position *and* rotation in the current wrist frame.
REL_MODES = ("abs", "delta", "delta_pose")


# ── spec ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ActionSpec:
    """Describes an action vector: ``kind`` (``hand_mano`` | ``robot_joint``), ``dim``, per-dim
    ``names`` and the wrist rotation representation ``rot_repr`` (``6d`` for hand_mano, ``none``
    for robot_joint). Serialisable (``to_dict``/``from_dict``) and shipped with the policy.

    Omitted ``names`` / ``rot_repr`` take the kind's defaults (``HAND_MANO_NAMES`` + ``6d`` for
    hand_mano, ``a0 … a{D-1}`` + ``none`` for robot_joint), so ``ActionSpec("hand_mano", 54) ==
    ActionSpec.hand_mano()``."""

    kind: str
    dim: int
    names: tuple[str, ...] = field(default_factory=tuple)
    rot_repr: str | None = None

    def __post_init__(self):
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind {self.kind!r}; expected one of {ACTION_KINDS}")
        if int(self.dim) < 1:
            raise ValueError(f"action dim must be >= 1, got {self.dim}")
        object.__setattr__(self, "dim", int(self.dim))
        if self.rot_repr is None:
            object.__setattr__(self, "rot_repr", "6d" if self.kind == "hand_mano" else "none")
        if self.names:
            names = tuple(str(n) for n in self.names)
        elif self.kind == "hand_mano" and self.dim == HAND_MANO_DIM:
            names = HAND_MANO_NAMES
        else:
            names = tuple(f"a{i}" for i in range(self.dim))
        object.__setattr__(self, "names", names)
        if len(names) != self.dim:
            raise ValueError(f"{len(names)} names for a {self.dim}-D action")
        if self.kind == "hand_mano":
            if self.dim != HAND_MANO_DIM:
                raise ValueError(f"hand_mano actions are {HAND_MANO_DIM}-D, got dim={self.dim}")
            if self.rot_repr != "6d":
                raise ValueError("hand_mano supports rot_repr='6d' only")
        elif self.rot_repr != "none":
            raise ValueError("robot_joint actions have rot_repr='none'")

    @classmethod
    def hand_mano(cls) -> "ActionSpec":
        return cls("hand_mano", HAND_MANO_DIM, HAND_MANO_NAMES, "6d")

    @classmethod
    def robot_joint(cls, names: Sequence[str] | int) -> "ActionSpec":
        """From URDF actuated joint names (or just a dimension)."""
        if isinstance(names, (int, np.integer)):
            return cls("robot_joint", int(names), (), "none")
        names = tuple(str(n) for n in names)
        return cls("robot_joint", len(names), names, "none")

    @property
    def slices(self) -> dict[str, slice]:
        return dict(HAND_SLICES) if self.kind == "hand_mano" else {"q": slice(0, self.dim)}

    def to_dict(self) -> dict:
        return {"kind": self.kind, "dim": self.dim, "names": list(self.names), "rot_repr": self.rot_repr}

    @classmethod
    def from_dict(cls, d: dict) -> "ActionSpec":
        return cls(d["kind"], int(d["dim"]), tuple(d.get("names") or ()), d.get("rot_repr"))


def _as_spec(spec: ActionSpec | str | dict, dim: int | None = None) -> ActionSpec:
    """ActionSpec | its dict | ``"hand_mano"`` | ``"robot_joint"`` (needs ``dim``) → ActionSpec."""
    if isinstance(spec, ActionSpec):
        return spec
    if isinstance(spec, dict):
        return ActionSpec.from_dict(spec)
    if spec == "hand_mano":
        return ActionSpec.hand_mano()
    if spec == "robot_joint" and dim is not None:
        return ActionSpec.robot_joint(int(dim))
    raise ValueError(f"cannot build an ActionSpec from {spec!r}: pass an ActionSpec, its dict, 'hand_mano' "
                     "or 'robot_joint' (where the dim can be inferred)")


# ── numpy / torch plumbing ──────────────────────────────────────────────────
def _any_torch(*xs) -> bool:
    return any(isinstance(x, torch.Tensor) for x in xs)


def _out_like(y: torch.Tensor, torch_out: bool, ref: torch.Tensor | None = None):
    """float64 work tensor → torch (ref dtype/device) or float32 numpy."""
    if torch_out:
        if ref is not None and ref.is_floating_point():
            return y.to(dtype=ref.dtype, device=ref.device)
        return y.to(torch.float32)
    return y.detach().cpu().numpy().astype(np.float32)


def _f64(x) -> torch.Tensor:
    t = as_tensor(x)
    return t.to(torch.float64) if t.dtype != torch.float64 else t


def _check_last(x: torch.Tensor, n: int, what: str) -> None:
    if x.ndim < 1 or x.shape[-1] != n:
        raise ValueError(f"{what} must be [..., {n}], got {tuple(x.shape)}")


# ── hand action ⇄ MANO arrays ───────────────────────────────────────────────
def hand_action_from_arrays(global_orient, finger_pose, wrist_pos):
    """MANO arrays → canonical hand action ``[..., 54]``.

    ``global_orient[...,3]`` (axis-angle), ``finger_pose[...,15,3]`` (axis-angle, MANO order),
    ``wrist_pos[...,3]`` (m). Leading dims must match (typically ``[T]``).
    """
    torch_out = _any_torch(global_orient, finger_pose, wrist_pos)
    ref = next((x for x in (global_orient, finger_pose, wrist_pos) if isinstance(x, torch.Tensor)), None)
    go, fp, wp = _f64(global_orient), _f64(finger_pose), _f64(wrist_pos)
    if ref is not None:
        go, fp, wp = go.to(ref.device), fp.to(ref.device), wp.to(ref.device)
    _check_last(go, 3, "global_orient")
    _check_last(wp, 3, "wrist_pos")
    if fp.ndim < 2 or tuple(fp.shape[-2:]) != (N_FINGER_JOINTS, 3):
        raise ValueError(f"finger_pose must be [..., 15, 3], got {tuple(fp.shape)}")
    lead = go.shape[:-1]
    if wp.shape[:-1] != lead or fp.shape[:-2] != lead:
        raise ValueError(f"leading dims differ: global_orient {tuple(go.shape)}, finger_pose "
                         f"{tuple(fp.shape)}, wrist_pos {tuple(wp.shape)}")
    r6 = matrix_to_6d(aa_to_matrix(go))
    a = torch.cat([wp, r6, fp.reshape(*lead, N_FINGER_JOINTS * 3)], dim=-1)
    return _out_like(a, torch_out, ref)


class HandArrays(NamedTuple):
    """MANO arrays decoded from a hand action. A tuple in the argument order of
    :func:`hand_action_from_arrays` (``go, fp, wp = hand_action_to_arrays(a)``) that also behaves
    like a read-only mapping (``h["finger_pose"]``, ``h.keys()``, ``skel.forward(**h)``)."""

    global_orient: np.ndarray | torch.Tensor
    finger_pose: np.ndarray | torch.Tensor
    wrist_pos: np.ndarray | torch.Tensor

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._fields:
                raise KeyError(key)
            return getattr(self, key)
        return tuple.__getitem__(self, key)

    def __contains__(self, key) -> bool:
        return isinstance(key, str) and key in self._fields

    def keys(self) -> tuple[str, ...]:
        return self._fields

    def items(self):
        return tuple(zip(self._fields, self))


def hand_action_to_arrays(action) -> HandArrays:
    """Inverse of :func:`hand_action_from_arrays`: ``[...,54]`` → :class:`HandArrays`
    ``(global_orient[...,3], finger_pose[...,15,3], wrist_pos[...,3])`` — unpack it as a tuple or
    use it as a mapping (keys match ``ManoSkeleton.forward``, so ``skel.forward(**h)`` works).

    The 6D part goes through Gram–Schmidt (``sixd_to_matrix``), so any network output decodes to
    a valid rotation; axis-angle angles come back in ``[0, π]``. Differentiable for torch input.
    """
    torch_out = isinstance(action, torch.Tensor)
    a = _f64(action)
    _check_last(a, HAND_MANO_DIM, "hand action")
    lead = a.shape[:-1]
    go = matrix_to_aa(sixd_to_matrix(a[..., WRIST_ROT6D]))
    ref = action if torch_out else None
    return HandArrays(global_orient=_out_like(go, torch_out, ref),
                      finger_pose=_out_like(a[..., FINGERS_AA].reshape(*lead, N_FINGER_JOINTS, 3), torch_out, ref),
                      wrist_pos=_out_like(a[..., WRIST_POS], torch_out, ref))


def wrist_rotation(action):
    """Wrist rotation matrices ``[...,3,3]`` of hand actions ``[...,54]`` (Gram–Schmidt on 6D)."""
    torch_out = isinstance(action, torch.Tensor)
    a = _f64(action)
    _check_last(a, HAND_MANO_DIM, "hand action")
    R = sixd_to_matrix(a[..., WRIST_ROT6D])
    return _out_like(R, torch_out, action if torch_out else None)


def robot_action_from_q(q):
    """Robot joint action = joint position targets; ``q[T,D]`` → float32 ``[T,D]`` copy."""
    if isinstance(q, torch.Tensor):
        if q.ndim != 2:
            raise ValueError(f"q must be [T, D], got {tuple(q.shape)}")
        return q.detach().clone().float()
    q = np.asarray(q)
    if q.ndim != 2:
        raise ValueError(f"q must be [T, D], got {q.shape}")
    return np.array(q, dtype=np.float32, copy=True)


def hand_action_from_episode(episode) -> tuple[np.ndarray, np.ndarray]:
    """``(actions[T,54] float32, valid[T] bool)`` from a processed Episode's MANO hand arrays.

    ``valid`` = ``hand_pose_valid`` when present (confidence-gated labels), else all True.
    """
    from robot_skin.datasets.episode import K_HAND_FINGERS, K_HAND_GLOBAL, K_HAND_VALID, K_HAND_WRIST
    missing = [k for k in (K_HAND_GLOBAL, K_HAND_FINGERS, K_HAND_WRIST) if not episode.has(k)]
    if missing:
        raise KeyError(f"episode {episode.meta.episode_id!r} lacks hand arrays {missing}; "
                       "hand_mano actions need hand_pose.npz labels (see pose.vision_hand)")
    a = hand_action_from_arrays(np.asarray(episode[K_HAND_GLOBAL]), np.asarray(episode[K_HAND_FINGERS]),
                                np.asarray(episode[K_HAND_WRIST]))
    valid = np.asarray(episode[K_HAND_VALID], dtype=bool) if episode.has(K_HAND_VALID) \
        else np.ones(a.shape[0], dtype=bool)
    return a, valid & np.isfinite(a).all(-1)


def actions_from_episode(episode, spec: ActionSpec | str | dict) -> tuple[np.ndarray, np.ndarray]:
    """Episode → ``(actions[T,A] float32, valid[T] bool)`` for ``spec.kind``.

    ``"robot_joint"`` (string) takes its names from ``episode.meta.joint_names`` (or just the dim
    of ``q``). Note that for glove episodes ``q`` is the flattened MANO finger pose (45-D).
    """
    from robot_skin.datasets.episode import K_Q
    if isinstance(spec, str) and spec == "robot_joint":
        if not episode.has(K_Q):
            raise KeyError(f"episode {episode.meta.episode_id!r} has no joint state {K_Q!r}")
        spec = ActionSpec.robot_joint(list(episode.meta.joint_names) or episode[K_Q].shape[1])
    spec = _as_spec(spec)
    if spec.kind == "hand_mano":
        return hand_action_from_episode(episode)
    if not episode.has(K_Q):
        raise KeyError(f"episode {episode.meta.episode_id!r} has no joint state {K_Q!r}")
    a = robot_action_from_q(np.asarray(episode[K_Q]))
    if a.shape[1] != spec.dim:
        raise ValueError(f"episode q is {a.shape[1]}-D but the action spec is {spec.dim}-D")
    return a, np.isfinite(a).all(-1)


# ── relative actions ────────────────────────────────────────────────────────
def _broadcast_state(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    if s.ndim == a.ndim - 1:          # chunk [...,H,A] with state [...,A]
        s = s.unsqueeze(-2)
    elif s.ndim != a.ndim:
        raise ValueError(f"state {tuple(s.shape)} does not match actions {tuple(a.shape)}")
    return s


def _relative(actions, state, spec, mode: str, inverse: bool):
    spec = _as_spec(spec, dim=np.shape(actions)[-1] if np.ndim(actions) else None)
    if mode not in REL_MODES:
        raise ValueError(f"mode must be one of {REL_MODES}, got {mode!r}")
    torch_out = _any_torch(actions, state)
    ref = actions if isinstance(actions, torch.Tensor) else (state if isinstance(state, torch.Tensor) else None)
    a = _f64(actions)
    s = _f64(state).to(a.device)
    _check_last(a, spec.dim, "actions")
    _check_last(s, spec.dim, "state")
    s = _broadcast_state(a, s)
    if mode == "abs":
        return _out_like(a.clone(), torch_out, ref)
    if spec.kind == "robot_joint":
        if mode != "delta":
            raise ValueError("robot_joint actions support mode 'abs' | 'delta'")
        return _out_like(a + s if inverse else a - s, torch_out, ref)
    out = a.clone()
    sp = s[..., WRIST_POS]
    if mode == "delta":
        out[..., WRIST_POS] = a[..., WRIST_POS] + sp if inverse else a[..., WRIST_POS] - sp
        return _out_like(out, torch_out, ref)
    # delta_pose: wrist pose in the current wrist frame. rel: p' = Rsᵀ(p − ps), R' = RsᵀR.
    Rs = sixd_to_matrix(s[..., WRIST_ROT6D])
    Ra = sixd_to_matrix(a[..., WRIST_ROT6D])
    Rs, sp = torch.broadcast_to(Rs, Ra.shape), torch.broadcast_to(sp, a[..., WRIST_POS].shape)
    if inverse:
        p = (Rs @ a[..., WRIST_POS, None])[..., 0] + sp
        R = Rs @ Ra
    else:
        p = (Rs.transpose(-1, -2) @ (a[..., WRIST_POS] - sp)[..., None])[..., 0]
        R = Rs.transpose(-1, -2) @ Ra
    out[..., WRIST_POS] = p
    out[..., WRIST_ROT6D] = matrix_to_6d(R)
    return _out_like(out, torch_out, ref)


def make_relative(actions, state, spec: ActionSpec | str | dict = "hand_mano", mode: str = "delta"):
    """Express ``actions`` (``[...,A]`` or a chunk ``[...,H,A]``) relative to the current
    ``state`` (``[...,A]``, same action space — e.g. the current hand action / robot q).

    ``spec``: ActionSpec (or its dict), ``"hand_mano"`` (default) or ``"robot_joint"``.
    ``delta``: hand → wrist position minus current wrist position (rotation & fingers absolute);
    robot → ``q − q_cur``. ``delta_pose`` (hand): wrist position and rotation in the current wrist
    frame. ``abs``: unchanged copy. Inverse: :func:`make_absolute`.
    """
    return _relative(actions, state, spec, mode, inverse=False)


def make_absolute(rel_actions, state, spec: ActionSpec | str | dict = "hand_mano", mode: str = "delta"):
    """Inverse of :func:`make_relative` (same ``state`` and ``mode``)."""
    return _relative(rel_actions, state, spec, mode, inverse=True)


# ── normalisation ───────────────────────────────────────────────────────────
NORM_METHODS = ("std", "robust", "minmax", "none")


def _stack_rows(actions, valid) -> np.ndarray:
    """list/array of ``[...,A]`` (+ optional bool masks over the leading dims) → rows ``[M,A]``."""
    xs = list(actions) if isinstance(actions, (list, tuple)) else [actions]
    vs = [None] * len(xs) if valid is None else (list(valid) if isinstance(valid, (list, tuple)) else [valid])
    if len(vs) != len(xs):
        raise ValueError(f"{len(xs)} action arrays but {len(vs)} valid masks")
    rows = []
    for x, v in zip(xs, vs):
        x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        x = x.astype(np.float64).reshape(-1, x.shape[-1]) if x.ndim > 1 else x.astype(np.float64)[None]
        if v is not None:
            v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            v = v.astype(bool).reshape(-1)
            if v.shape[0] != x.shape[0]:
                raise ValueError(f"valid mask has {v.shape[0]} entries for {x.shape[0]} action rows")
            x = x[v]
        rows.append(x[np.isfinite(x).all(-1)])
    dims = {r.shape[1] for r in rows}
    if len(dims) != 1:
        raise ValueError(f"action arrays have different dims {sorted(dims)}")
    out = np.concatenate(rows, 0)
    if out.shape[0] == 0:
        raise ValueError("no valid action rows to fit the normalizer on")
    return out


class ActionNormalizer:
    """Per-dimension affine action normalisation ``(a − offset) / scale`` (wraps ``NormStats``).

    Fit on the training split only, on the representation the policy is trained on (absolute or
    relative). ``method``: ``std`` (default, all 54 dims incl. 6D), ``robust`` (median/IQR),
    ``minmax`` (→ [−1, 1], the bounded range Diffusion-Policy/flow heads like), ``none``.
    ``min_scale`` floors the scale so a near-constant dim (e.g. a joint the demonstrator never
    moved) cannot amplify noise at test time; ``eps`` is added as in ``NormStats``.
    """

    def __init__(self, stats: NormStats, spec: ActionSpec | str | dict | None = None, method: str = "std"):
        if stats.offset.shape != stats.scale.shape or stats.offset.ndim != 1:
            raise ValueError("stats offset/scale must be 1-D of equal length")
        if spec is not None:
            spec = _as_spec(spec, dim=stats.offset.shape[0])
        if spec is not None and spec.dim != stats.offset.shape[0]:
            raise ValueError(f"stats are {stats.offset.shape[0]}-D but spec is {spec.dim}-D")
        if not (np.isfinite(stats.offset).all() and np.isfinite(stats.scale).all() and (stats.scale > 0).all()):
            raise ValueError("stats offset/scale must be finite and scale > 0")
        if method not in NORM_METHODS:
            raise ValueError(f"method must be one of {NORM_METHODS}, got {method!r}")
        self.stats, self.spec, self.method = stats, spec, method

    @property
    def dim(self) -> int:
        return int(self.stats.offset.shape[0])

    @classmethod
    def fit(cls, actions, valid=None, *, spec: ActionSpec | str | dict | None = None, method: str = "std",
            eps: float = 1e-6, min_scale: float = 1e-2) -> "ActionNormalizer":
        """``actions``: array or list of arrays ``[...,A]`` (e.g. per-episode ``[T,A]`` or chunks
        ``[N,H,A]``); ``valid``: matching bool masks over the leading dims (or None)."""
        if method not in NORM_METHODS:
            raise ValueError(f"method must be one of {NORM_METHODS}")
        x = _stack_rows(actions, valid)
        if method == "minmax":
            lo, hi = x.min(0), x.max(0)
            st = NormStats(offset=((hi + lo) / 2).astype(np.float32),
                           scale=((hi - lo) / 2 + eps).astype(np.float32))
        else:
            st = NormStats.fit(x, method=method, eps=eps)
        if method != "none":
            st = NormStats(offset=st.offset, scale=np.maximum(st.scale, np.float32(min_scale)).astype(np.float32))
        return cls(st, spec, method)

    def _params(self, x):
        if isinstance(x, torch.Tensor):
            dt = x.dtype if x.is_floating_point() else torch.float32
            return (torch.as_tensor(self.stats.offset, dtype=dt, device=x.device),
                    torch.as_tensor(self.stats.scale, dtype=dt, device=x.device))
        return self.stats.offset, self.stats.scale

    def _check(self, x):
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected [..., {self.dim}] actions, got {tuple(x.shape)}")

    def normalize(self, a):
        """numpy → float32 numpy; torch → torch (dtype/device kept)."""
        a = a if isinstance(a, torch.Tensor) else np.asarray(a, dtype=np.float32)
        self._check(a)
        off, sc = self._params(a)
        return (a - off) / sc

    def unnormalize(self, a_norm):
        a_norm = a_norm if isinstance(a_norm, torch.Tensor) else np.asarray(a_norm, dtype=np.float32)
        self._check(a_norm)
        off, sc = self._params(a_norm)
        return a_norm * sc + off

    __call__ = normalize
    apply = normalize            # NormStats-style names
    invert = unnormalize

    def to_dict(self) -> dict:
        return {"method": self.method, "stats": self.stats.to_dict(),
                "spec": None if self.spec is None else self.spec.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "ActionNormalizer":
        """From :meth:`to_dict` output (or a bare ``NormStats`` dict ``{offset, scale}``)."""
        if "stats" not in d and {"offset", "scale"} <= set(d):
            return cls(NormStats.from_dict(d))
        spec = None if d.get("spec") is None else ActionSpec.from_dict(d["spec"])
        return cls(NormStats.from_dict(d["stats"]), spec, d.get("method", "std"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "ActionNormalizer":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:
        kind = None if self.spec is None else self.spec.kind
        return f"ActionNormalizer(dim={self.dim}, method={self.method!r}, kind={kind!r})"


__all__ = [
    "HAND_MANO_DIM", "HAND_MANO_NAMES", "HAND_SLICES", "MANO_FINGER_JOINTS", "WRIST_POS", "WRIST_ROT6D",
    "FINGERS_AA", "ACTION_KINDS", "REL_MODES", "NORM_METHODS", "ActionSpec", "hand_action_from_arrays",
    "hand_action_to_arrays", "wrist_rotation", "robot_action_from_q", "hand_action_from_episode",
    "actions_from_episode", "make_relative", "make_absolute", "ActionNormalizer", "HandArrays",
]
