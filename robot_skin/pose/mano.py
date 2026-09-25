"""MANO hand skeleton — kinematic tree, forward kinematics, segment frames, capsules.

Skeleton-only re-implementation of the MANO kinematic chain (Romero et al., *Embodied Hands:
Modeling and Capturing Hands and Bodies Together*, SIGGRAPH Asia 2017): 16 joints, one
axis-angle rotation per joint, applied in the **rest-aligned** joint frame exactly like MANO/SMPL
linear-blend skinning::

    G_0 = [R(global_orient) | wrist_pos]
    G_k = G_parent(k) · [R(θ_k) | J_k − J_parent(k)]          k = 1 … 15

so at the zero pose every joint frame is axis-aligned with the canonical (MANO) frame and sits at
its rest position ``J_k``. No mesh, no blend shapes and no MANO licence files are needed; if you
have them, :meth:`ManoSkeleton.from_mano_pkl` reads subject rest joints from the official model.

Conventions (all of ``robot_skin``)
- right hand; metres; axis-angle as in MANO; ``finger_pose`` is the full rotation relative to the
  **flat** template (MANO ``flat_hand_mean=True``). Labels that are relative to MANO's
  ``hands_mean`` must have it added first; PCA pose coefficients must be expanded first.
- default canonical axes follow the MANO right-hand template as far as we know it: fingers point
  towards **−x** from the wrist, the radial (index/thumb) side is **+z**, the palm faces **−y**
  (dorsal = +y). With these axes finger flexion is (approximately) a positive rotation about +z.
  The default rest joints are an *approximate adult right hand* hand-made in these axes — not
  MANO's exact template. Load real rest joints (``from_mano_pkl``) or measured ones for accuracy.
- ``wrist_pos`` is the world position of joint 0. From MANO's ``transl``:
  ``wrist_pos = transl + J_0(β)`` (MANO rotates about ``J_0``).
- MANO joint order (:data:`MANO_JOINTS`) is index, middle, pinky, ring, thumb; everything that is
  *per finger* here (tips, :data:`FINGERS`) uses the anatomical order thumb, index, middle, ring,
  pinky, matching the glove IMU sites and ``contact.self_touch.FINGERS``.

Segment frames (``parent`` of glove taxels / IMU sites, names = ``common.layouts.MANO_SEGMENTS``)
- ``wrist``  → joint 0 frame itself (MANO axes, = global orientation).
- ``palm``   → joint 0 frame · fixed palm-aligned rotation (+ optional ``palm_offset``). Origin at
  the wrist joint by default because ``glove_template`` palm taxels are measured from there
  (z = 30–55 mm along the palm).
- ``<finger><1|2|3>`` → frame of the joint that moves that phalanx (index1 = MCP → proximal
  phalanx, …, thumb1 = CMC → metacarpal) · fixed *bone-aligned* rotation: local **+z along the
  bone** (joint → child joint / fingertip), local **−y palmar** (pad side), x = y × z.
  This matches ``glove_template`` (fingertip pads at local ``[0, −6, 10–12] mm``). With
  ``bone_aligned=False`` segment frames are the raw (rest-aligned) MANO joint frames.
  In the bone-aligned frames **flexion is a positive rotation about local +x** and abduction a
  rotation about local +y (see :meth:`ManoSkeleton.flexion_pose`).

Capsules (self-touch geometry): 15 phalanges (``p0`` = joint, ``p1`` = child joint or tip) + 4
palm capsules (wrist → each finger MCP), radii in :data:`DEFAULT_SEG_RADIUS` /
``DEFAULT_PALM_RADIUS``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from common.layouts import MANO_SEGMENTS, Layout
from robot_skin.geometry.rotations import aa_to_matrix, as_tensor, make_transform, matrix_to_aa

# ── kinematic tree ─────────────────────────────────────────────────────────
MANO_JOINTS: tuple[str, ...] = (
    "wrist",
    "index1", "index2", "index3",
    "middle1", "middle2", "middle3",
    "pinky1", "pinky2", "pinky3",
    "ring1", "ring2", "ring3",
    "thumb1", "thumb2", "thumb3",
)
MANO_PARENTS: tuple[int, ...] = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)
N_JOINTS = 16
N_FINGER_JOINTS = 15

#: Anatomical finger order used for fingertips and per-finger arrays.
FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
#: finger → MANO joint indices (proximal → distal).
FINGER_JOINTS: dict[str, tuple[int, int, int]] = {
    "thumb": (13, 14, 15), "index": (1, 2, 3), "middle": (4, 5, 6),
    "ring": (10, 11, 12), "pinky": (7, 8, 9)}
TIP_NAMES: tuple[str, ...] = tuple(f"{f}_tip" for f in FINGERS)

#: Every ``common.layouts.MANO_SEGMENTS`` name → the MANO joint whose frame carries it.
SEGMENT_TO_JOINT: dict[str, int] = {"wrist": 0, "palm": 0,
                                    **{n: i for i, n in enumerate(MANO_JOINTS) if i > 0}}
assert set(SEGMENT_TO_JOINT) == set(MANO_SEGMENTS)
_SEG_JOINT_IDX = np.array([SEGMENT_TO_JOINT[s] for s in MANO_SEGMENTS], dtype=np.int64)
_SEG_INDEX = {s: i for i, s in enumerate(MANO_SEGMENTS)}

#: child joint of each joint 1..15 (``None`` → the phalanx ends at the fingertip).
_CHILD: dict[int, int | None] = {}
for _f, (_a, _b, _c) in FINGER_JOINTS.items():
    _CHILD.update({_a: _b, _b: _c, _c: None})
_FINGER_OF_JOINT = {j: f for f, js in FINGER_JOINTS.items() for j in js}

PALM_CAPSULES: tuple[str, ...] = ("palm_index", "palm_middle", "palm_ring", "palm_pinky")
_PALM_CAPSULE_MCP = (1, 4, 10, 7)
#: capsule names returned by :meth:`ManoSkeleton.capsules` (15 phalanges + palm).
CAPSULE_NAMES: tuple[str, ...] = MANO_JOINTS[1:] + PALM_CAPSULES

# ── default geometry (approximate adult right hand, metres, wrist at origin) ─
DEFAULT_REST_JOINTS = np.array([
    [0.0, 0.0, 0.0],              # wrist
    [-0.0881, -0.0051, 0.0207],   # index1  (MCP)
    [-0.1208, -0.0027, 0.0225],   # index2  (PIP)
    [-0.1430, -0.0031, 0.0227],   # index3  (DIP)
    [-0.0919, -0.0026, -0.0035],  # middle1
    [-0.1249, -0.0022, -0.0086],  # middle2
    [-0.1486, -0.0020, -0.0122],  # middle3
    [-0.0758, -0.0094, -0.0453],  # pinky1
    [-0.0983, -0.0092, -0.0567],  # pinky2
    [-0.1133, -0.0090, -0.0648],  # pinky3
    [-0.0833, -0.0073, -0.0251],  # ring1
    [-0.1139, -0.0061, -0.0320],  # ring2
    [-0.1371, -0.0055, -0.0368],  # ring3
    [-0.0235, -0.0176, 0.0193],   # thumb1  (CMC)
    [-0.0448, -0.0201, 0.0374],   # thumb2  (MCP)
    [-0.0663, -0.0230, 0.0491],   # thumb3  (IP)
], dtype=np.float64)
#: distal phalanx length (DIP/IP joint → fingertip), FINGERS order.
DEFAULT_TIP_LENGTHS = {"thumb": 0.024, "index": 0.021, "middle": 0.023, "ring": 0.022, "pinky": 0.019}
#: capsule radius per phalanx, MANO joint order 1..15 (≈ half finger thickness incl. glove).
DEFAULT_SEG_RADIUS = np.array([
    0.0095, 0.0085, 0.0075,   # index
    0.0095, 0.0085, 0.0075,   # middle
    0.0080, 0.0072, 0.0065,   # pinky
    0.0090, 0.0080, 0.0072,   # ring
    0.0110, 0.0095, 0.0085,   # thumb
], dtype=np.float64)
DEFAULT_PALM_RADIUS = 0.012
#: capsules ignored for self-touch in addition to the taxel's own finger, keyed by the taxel's
#: finger group (``finger_of(parent)``: thumb…pinky | palm) **or** its segment name (``index1``…).
#: - palm taxels: the thumb metacarpal (thumb1) lies inside the thenar eminence, under the palm skin;
#: - proximal phalanges: the palm capsule of the same ray ends at that finger's MCP, so taxels near
#:   the base of ``<finger>1`` always lie inside it (flat hand → false self-touch without this).
DEFAULT_SELF_TOUCH_EXCLUDE: dict[str, tuple[str, ...]] = {
    "palm": ("thumb1",),
    **{f"{f}1": (f"palm_{f}",) for f in ("index", "middle", "ring", "pinky")},
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length direction in skeleton geometry")
    return v / n


def bone_frame(direction: np.ndarray, dorsal_hint: np.ndarray) -> np.ndarray:
    """Rotation whose columns are ``(x, y, z)`` with z ∥ ``direction`` and y the dorsal hint
    orthogonalised against z (x = y × z)."""
    z = _unit(np.asarray(direction, dtype=np.float64))
    h = np.asarray(dorsal_hint, dtype=np.float64)
    y = h - np.dot(h, z) * z
    if np.linalg.norm(y) < 1e-6:  # hint parallel to the bone → any perpendicular
        alt = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        y = alt - np.dot(alt, z) * z
    y = _unit(y)
    return np.stack([np.cross(y, z), y, z], axis=1)


def _to_np(x) -> np.ndarray:
    """chumpy / scipy.sparse / array → dense float64 ndarray."""
    if hasattr(x, "r"):            # chumpy.Ch
        x = x.r
    if hasattr(x, "toarray"):      # scipy.sparse
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


class ManoSkeleton:
    """MANO kinematic skeleton with forward kinematics, segment frames and capsules.

    Parameters
    ----------
    rest_joints : [16, 3] rest joint positions (MANO order, canonical frame). Translated so the
        wrist is at the origin. Default :data:`DEFAULT_REST_JOINTS`.
    tip_offsets : [5, 3] fingertip offset from the last joint of each finger (FINGERS order,
        canonical rest frame). Default: :data:`DEFAULT_TIP_LENGTHS` along the distal bone.
    seg_radius : [15] capsule radii (MANO joint order 1..15). palm_radius: palm capsule radius.
    palm_offset : [3] palm-frame origin expressed in the palm frame (default wrist joint).
    dorsal / thumb_dorsal : dorsal direction hints (canonical) for the bone-aligned frames.
    bone_aligned : bone-aligned segment frames (default) or raw MANO joint frames.

    Instances are meant to be immutable: torch copies of the geometry are cached per
    dtype/device, so build a new skeleton instead of editing ``rest_joints`` etc. in place.
    """

    def __init__(self, rest_joints=None, tip_offsets=None, *, seg_radius=None,
                 palm_radius: float = DEFAULT_PALM_RADIUS, palm_offset=(0.0, 0.0, 0.0),
                 dorsal=(0.0, 1.0, 0.0), thumb_dorsal=(0.0, 0.5, 1.0), bone_aligned: bool = True):
        J = np.array(DEFAULT_REST_JOINTS if rest_joints is None else rest_joints, dtype=np.float64)
        if J.shape != (N_JOINTS, 3):
            raise ValueError(f"rest_joints must be [16, 3] (MANO order), got {J.shape}")
        J = J - J[0]
        self.rest_joints = J
        if tip_offsets is None:
            tips = np.stack([_unit(J[FINGER_JOINTS[f][2]] - J[FINGER_JOINTS[f][1]]) * DEFAULT_TIP_LENGTHS[f]
                             for f in FINGERS])
        else:
            tips = np.array(tip_offsets, dtype=np.float64)
            if tips.shape != (5, 3):
                raise ValueError(f"tip_offsets must be [5, 3] (FINGERS order), got {tips.shape}")
        self.tip_offsets = tips
        r = DEFAULT_SEG_RADIUS if seg_radius is None else np.asarray(seg_radius, dtype=np.float64)
        self.seg_radius = np.broadcast_to(r, (N_FINGER_JOINTS,)).astype(np.float64).copy()
        if palm_radius <= 0 or np.any(self.seg_radius <= 0):
            raise ValueError("capsule radii must be positive")
        self.palm_radius = float(palm_radius)
        self.palm_offset = np.asarray(palm_offset, dtype=np.float64).reshape(3)
        self.bone_aligned = bool(bone_aligned)
        self._dorsal = np.asarray(dorsal, dtype=np.float64).reshape(3)
        self._thumb_dorsal = np.asarray(thumb_dorsal, dtype=np.float64).reshape(3)

        # bone frames (always computed; used for flexion axes) in MANO_SEGMENTS order
        frames = np.zeros((len(MANO_SEGMENTS), 3, 3))
        for s, name in enumerate(MANO_SEGMENTS):
            if name == "wrist":
                frames[s] = np.eye(3)
            elif name == "palm":
                frames[s] = bone_frame(J[FINGER_JOINTS["middle"][0]] - J[0], self._dorsal)
            else:
                k = SEGMENT_TO_JOINT[name]
                fi = FINGERS.index(_FINGER_OF_JOINT[k])
                end = J[_CHILD[k]] if _CHILD[k] is not None else J[k] + tips[fi]
                hint = self._thumb_dorsal if name.startswith("thumb") else self._dorsal
                frames[s] = bone_frame(end - J[k], hint)
        self.bone_frames = frames
        local = np.tile(np.eye(4), (len(MANO_SEGMENTS), 1, 1))
        for s, name in enumerate(MANO_SEGMENTS):
            if name == "wrist":
                continue
            R = frames[s] if self.bone_aligned else np.eye(3)
            local[s, :3, :3] = R
            if name == "palm":
                local[s, :3, 3] = R @ self.palm_offset
        #: [17, 4, 4] fixed segment-frame → joint-frame transforms (MANO_SEGMENTS order).
        self.segment_local = local
        self._cache: dict = {}

    # ── construction from the official MANO model ─────────────────────────
    @classmethod
    def from_mano_pkl(cls, path: str | Path, betas=None, *,
                      tip_vertex_ids: Mapping[str, int] | None = None, **kw) -> "ManoSkeleton":
        """Rest joints from an official MANO model file (``MANO_RIGHT.pkl`` or a converted ``.npz``
        with ``v_template [778,3]``, ``J_regressor [16,778]``, ``shapedirs [778,3,10]``).

        ``J = J_regressor · (v_template + shapedirs · β)``. ``tip_vertex_ids`` (finger → vertex
        index) places fingertips on mesh vertices; otherwise default tip lengths are scaled by
        hand size. The original ``.pkl`` stores ``chumpy`` objects: install ``chumpy`` or convert
        the file to ``.npz`` once. MANO files are licence-restricted and are not shipped here
        (https://mano.is.tue.mpg.de).
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"MANO model file not found: {p} (download from mano.is.tue.mpg.de)")
        if p.suffix == ".npz":
            data = dict(np.load(p, allow_pickle=False))
        else:
            try:
                with p.open("rb") as fh:
                    data = pickle.load(fh, encoding="latin1")
            except ModuleNotFoundError as e:  # pragma: no cover - depends on user env
                raise ImportError(
                    f"reading {p.name} needs {e.name!r} (the official MANO pkl stores chumpy objects). "
                    "`pip install chumpy`, or convert once to .npz with keys v_template, J_regressor, "
                    "shapedirs.") from e
        for key in ("v_template", "J_regressor"):
            if key not in data:
                raise KeyError(f"MANO file lacks {key!r}; keys: {sorted(data)}")
        v = _to_np(data["v_template"])
        if betas is not None:
            if "shapedirs" not in data:
                raise KeyError("betas given but the MANO file has no 'shapedirs'")
            sd = _to_np(data["shapedirs"])
            b = np.asarray(betas, dtype=np.float64).reshape(-1)
            v = v + sd[..., : b.shape[0]] @ b
        Jr = _to_np(data["J_regressor"])
        J = Jr @ v
        if J.shape != (N_JOINTS, 3):
            raise ValueError(f"unexpected J_regressor output {J.shape}; expected 16 MANO joints")
        if tip_vertex_ids is not None:
            tips = np.stack([v[int(tip_vertex_ids[f])] - J[FINGER_JOINTS[f][2]] for f in FINGERS])
        else:
            scale = np.linalg.norm(J[4] - J[0]) / np.linalg.norm(DEFAULT_REST_JOINTS[4])
            tips = np.stack([_unit(J[FINGER_JOINTS[f][2]] - J[FINGER_JOINTS[f][1]]) * DEFAULT_TIP_LENGTHS[f]
                             * scale for f in FINGERS])
        return cls(rest_joints=J, tip_offsets=tips, **kw)

    # ── torch constants ───────────────────────────────────────────────────
    def _consts(self, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
        key = (dtype, str(device))
        if key not in self._cache:
            J = self.rest_joints
            off = np.zeros_like(J)
            for k in range(1, N_JOINTS):
                off[k] = J[k] - J[MANO_PARENTS[k]]
            self._cache[key] = {
                "offsets": torch.as_tensor(off, dtype=dtype, device=device),
                "tips": torch.as_tensor(self.tip_offsets, dtype=dtype, device=device),
                "seg_local": torch.as_tensor(self.segment_local, dtype=dtype, device=device),
                "flex_axes": torch.as_tensor(self.bone_frames[[_SEG_INDEX[n] for n in MANO_JOINTS[1:]], :, 0],
                                             dtype=dtype, device=device),
                "abd_axes": torch.as_tensor(self.bone_frames[[_SEG_INDEX[n] for n in MANO_JOINTS[1:]], :, 1],
                                            dtype=dtype, device=device),
            }
        return self._cache[key]

    # ── forward kinematics ────────────────────────────────────────────────
    def forward(self, global_orient=None, finger_pose=None, wrist_pos=None) -> dict[str, torch.Tensor]:
        """MANO FK. ``global_orient[...,3]``, ``finger_pose[...,15,3]`` (axis-angle, MANO order),
        ``wrist_pos[...,3]`` (any may be None → zeros; leading dims broadcast; numpy accepted).

        Returns ``joint_T[...,16,4,4]``, ``joint_rot[...,16,3,3]``, ``joint_pos[...,16,3]``,
        ``tip_pos[...,5,3]`` (FINGERS order). Differentiable.
        """
        go, fp, wp = self._prep(global_orient, finger_pose, wrist_pos, rot_tail=(3,), fin_tail=(15, 3))
        R_loc = aa_to_matrix(torch.cat([go[..., None, :], fp], dim=-2))    # [...,16,3,3]
        return self._chain(R_loc, wp)

    def forward_matrices(self, global_rot=None, finger_rot=None, wrist_pos=None) -> dict[str, torch.Tensor]:
        """FK from rotation matrices: ``global_rot[...,3,3]``, ``finger_rot[...,15,3,3]`` (e.g. a
        6D network output through ``sixd_to_matrix`` — no lossy matrix→axis-angle step)."""
        go, fp, wp = self._prep(global_rot, finger_rot, wrist_pos, rot_tail=(3, 3), fin_tail=(15, 3, 3))
        return self._chain(torch.cat([go[..., None, :, :], fp], dim=-3), wp)

    def _prep(self, g, f, w, *, rot_tail: tuple, fin_tail: tuple):
        given = [x for x in (g, f, w) if isinstance(x, torch.Tensor)]
        dtype = given[0].dtype if given and given[0].is_floating_point() else torch.float64
        device = given[0].device if given else torch.device("cpu")

        def prep(x, tail):
            if x is None:
                return None
            x = as_tensor(x).to(dtype=dtype, device=device)
            if x.ndim < len(tail) or tuple(x.shape[x.ndim - len(tail):]) != tail:
                raise ValueError(f"expected trailing shape {tail}, got {tuple(x.shape)}")
            return x

        g, f, w = prep(g, rot_tail), prep(f, fin_tail), prep(w, (3,))
        leads = ([g.shape[:g.ndim - len(rot_tail)]] if g is not None else []) + \
                ([f.shape[:f.ndim - len(fin_tail)]] if f is not None else []) + \
                ([w.shape[:-1]] if w is not None else [])
        lead = torch.broadcast_shapes(*leads) if leads else torch.Size([])
        if g is None:
            g = (torch.zeros(*lead, 3, dtype=dtype, device=device) if rot_tail == (3,)
                 else torch.eye(3, dtype=dtype, device=device).expand(*lead, 3, 3))
        if f is None:
            f = (torch.zeros(*lead, *fin_tail, dtype=dtype, device=device) if fin_tail == (15, 3)
                 else torch.eye(3, dtype=dtype, device=device).expand(*lead, 15, 3, 3))
        w = torch.zeros(*lead, 3, dtype=dtype, device=device) if w is None else w
        return g.expand(*lead, *rot_tail), f.expand(*lead, *fin_tail), w.expand(*lead, 3)

    def _chain(self, R_loc: torch.Tensor, wp: torch.Tensor) -> dict[str, torch.Tensor]:
        c = self._consts(R_loc.dtype, R_loc.device)
        rot: list[torch.Tensor] = [R_loc[..., 0, :, :]]
        pos: list[torch.Tensor] = [wp]
        for k in range(1, N_JOINTS):
            p = MANO_PARENTS[k]
            rot.append(rot[p] @ R_loc[..., k, :, :])
            pos.append(pos[p] + torch.matmul(rot[p], c["offsets"][k]))
        tips = [pos[FINGER_JOINTS[f][2]] + torch.matmul(rot[FINGER_JOINTS[f][2]], c["tips"][i])
                for i, f in enumerate(FINGERS)]
        joint_rot = torch.stack(rot, dim=-3)
        joint_pos = torch.stack(pos, dim=-2)
        return {"joint_T": make_transform(joint_rot, joint_pos), "joint_rot": joint_rot,
                "joint_pos": joint_pos, "tip_pos": torch.stack(tips, dim=-2)}

    __call__ = forward

    def rest_fk(self, dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
        """FK at the zero (flat) pose."""
        return self.forward(torch.zeros(3, dtype=dtype))

    # ── segment frames ────────────────────────────────────────────────────
    def segment_transform_tensor(self, fk: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``[...,17,4,4]`` segment frames in ``MANO_SEGMENTS`` order."""
        JT = fk["joint_T"]
        c = self._consts(JT.dtype, JT.device)
        return JT[..., _SEG_JOINT_IDX, :, :] @ c["seg_local"]

    def segment_transforms(self, fk: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """``{segment_name: [...,4,4]}`` for every ``common.layouts.MANO_SEGMENTS`` name."""
        T = self.segment_transform_tensor(fk)
        return {name: T[..., i, :, :] for i, name in enumerate(MANO_SEGMENTS)}

    # ── capsules ──────────────────────────────────────────────────────────
    def capsules(self, fk: Mapping[str, torch.Tensor]):
        """``(p0[...,S,3], p1[...,S,3], radius[S] np, names list[S])`` with S = 15 phalanges +
        4 palm capsules (:data:`CAPSULE_NAMES`)."""
        jp, tp = fk["joint_pos"], fk["tip_pos"]
        pts = torch.cat([jp, tp], dim=-2)                                   # [...,21,3]
        i0 = list(range(1, N_JOINTS)) + [0] * len(PALM_CAPSULES)
        i1 = [(_CHILD[k] if _CHILD[k] is not None else N_JOINTS + FINGERS.index(_FINGER_OF_JOINT[k]))
              for k in range(1, N_JOINTS)] + list(_PALM_CAPSULE_MCP)
        radius = np.concatenate([self.seg_radius, np.full(len(PALM_CAPSULES), self.palm_radius)])
        return pts[..., i0, :], pts[..., i1, :], radius, list(CAPSULE_NAMES)

    # ── pose helpers ──────────────────────────────────────────────────────
    def flexion_pose(self, flex, abduction=None) -> torch.Tensor:
        """Finger pose from anatomical angles (rad) using the bone-aligned axes.

        ``flex``: ``[...,15]`` (MANO joint order 1..15) or ``[...,5]`` (FINGERS order, same angle
        on all three joints). Positive = flexion (curl towards the palm). ``abduction``:
        ``[...,5]`` (FINGERS order) rotation about the dorsal axis at the first joint of each finger.
        Returns ``finger_pose[...,15,3]`` axis-angle.
        """
        f = as_tensor(flex)
        dtype = f.dtype if f.is_floating_point() else torch.float64
        f = f.to(dtype)
        c = self._consts(dtype, f.device)
        if f.shape[-1] == 5:
            per_joint = torch.zeros(*f.shape[:-1], 15, dtype=dtype, device=f.device)
            for i, fn in enumerate(FINGERS):
                for j in FINGER_JOINTS[fn]:
                    per_joint[..., j - 1] = f[..., i]
            f = per_joint
        elif f.shape[-1] != 15:
            raise ValueError(f"flex must be [...,15] or [...,5], got {tuple(f.shape)}")
        R = aa_to_matrix(f[..., None] * c["flex_axes"])                   # [...,15,3,3]
        if abduction is not None:
            a = as_tensor(abduction).to(dtype=dtype, device=f.device)
            if a.shape[-1] != 5:
                raise ValueError(f"abduction must be [...,5], got {tuple(a.shape)}")
            a15 = torch.zeros(*a.shape[:-1], 15, dtype=dtype, device=f.device)
            for i, fn in enumerate(FINGERS):
                a15[..., FINGER_JOINTS[fn][0] - 1] = a[..., i]
            R = aa_to_matrix(a15[..., None] * c["abd_axes"]) @ R
        return matrix_to_aa(R)

    def segment_local_rotation(self, segment: str) -> np.ndarray:
        """Fixed rotation from a segment frame to its joint frame ([3,3])."""
        return self.segment_local[_SEG_INDEX[segment], :3, :3].copy()


# ── batched numpy helpers for preprocessing ────────────────────────────────
def _hand_arrays(global_orient, finger_pose, wrist_pos):
    fp = np.asarray(finger_pose, dtype=np.float64)
    single = fp.ndim == 2
    if single:
        fp = fp[None]
    if fp.ndim != 3 or fp.shape[1:] != (15, 3):
        raise ValueError(f"finger_pose must be [T,15,3] (or [15,3]), got {np.shape(finger_pose)}")
    T = fp.shape[0]

    def vec(x, name):
        if x is None:
            return np.zeros((T, 3))
        a = np.asarray(x, dtype=np.float64)
        a = a[None] if a.ndim == 1 else a
        if a.shape != (T, 3):
            raise ValueError(f"{name} must be [T,3] with T={T}, got {np.shape(x)}")
        return a

    return vec(global_orient, "global_orient"), fp, vec(wrist_pos, "wrist_pos"), single


def _parent_index(layout: Layout) -> np.ndarray:
    if layout.parent_frame != "mano":
        raise ValueError(f"layout {layout.name!r} has parent_frame {layout.parent_frame!r}; "
                         "MANO-based taxel poses need parent_frame: mano")
    return np.array([_SEG_INDEX[p] for p in layout.parents], dtype=np.int64)


def _taxels_from_fk(skeleton: ManoSkeleton, fk, idx: np.ndarray, pl: torch.Tensor, nl: torch.Tensor):
    ST = skeleton.segment_transform_tensor(fk)[:, idx]                        # [t,N,4,4]
    R, p = ST[..., :3, :3], ST[..., :3, 3]
    n = torch.matmul(R, nl[..., None])[..., 0]
    return (torch.matmul(R, pl[..., None])[..., 0] + p).numpy(), (n / n.norm(dim=-1, keepdim=True)).numpy()


def _fk_chunks(skeleton: ManoSkeleton, go, fp, wp, chunk: int):
    """Yield ``(start, end, fk)`` over the time axis (no autograd)."""
    T = fp.shape[0]
    with torch.no_grad():
        for s in range(0, T, max(1, int(chunk))):
            e = min(T, s + max(1, int(chunk)))
            yield s, e, skeleton.forward(torch.as_tensor(go[s:e]), torch.as_tensor(fp[s:e]),
                                         torch.as_tensor(wp[s:e]))


def taxel_poses_from_hand(layout: Layout, skeleton: ManoSkeleton, global_orient, finger_pose,
                          wrist_pos=None, *, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """World taxel poses for a glove layout from MANO hand pose sequences.

    ``global_orient[T,3]``, ``finger_pose[T,15,3]``, ``wrist_pos[T,3]`` (None → origin) →
    ``(pos[T,N,3], nrm[T,N,3])`` float64 (single-frame inputs → ``[N,3]``). Equivalent to
    ``provider.transform_taxels(layout, segment_transforms)`` per frame, batched.
    """
    idx = _parent_index(layout)
    go, fp, wp, single = _hand_arrays(global_orient, finger_pose, wrist_pos)
    pl = torch.as_tensor(layout.positions, dtype=torch.float64)
    nl = torch.as_tensor(layout.normals, dtype=torch.float64)
    pos = np.empty((fp.shape[0], layout.n, 3))
    nrm = np.empty_like(pos)
    for s, e, fk in _fk_chunks(skeleton, go, fp, wp, chunk):
        pos[s:e], nrm[s:e] = _taxels_from_fk(skeleton, fk, idx, pl, nl)
    return (pos[0], nrm[0]) if single else (pos, nrm)


def self_touch_from_hand(layout: Layout, skeleton: ManoSkeleton, global_orient, finger_pose,
                         wrist_pos=None, *, margin: float = 0.004,
                         exclude: Mapping[str, Sequence[str]] | None = None,
                         chunk: int = 4096) -> np.ndarray:
    """Geometric self-touch labels ``bool[T,N]`` (single frame → ``[N]``).

    Wraps ``contact.self_touch_labels`` with the skeleton capsules: a taxel is self-touched when
    within ``radius + margin`` of a capsule not on its own finger. ``exclude`` maps a taxel finger
    group (``finger_of(parent)``: thumb…pinky | palm) and/or a segment name (``index1``, …) to
    extra capsule names to ignore; both entries apply (default
    :data:`DEFAULT_SELF_TOUCH_EXCLUDE`: palm taxels ignore the thumb metacarpal, proximal-phalanx
    taxels ignore the palm capsule ending at their own MCP).
    """
    from robot_skin.contact.self_touch import finger_of, self_touch_labels  # lazy: avoid cycles

    idx = _parent_index(layout)
    excl = DEFAULT_SELF_TOUCH_EXCLUDE if exclude is None else exclude
    unknown = {c for v in excl.values() for c in v} - set(CAPSULE_NAMES)
    if unknown:
        raise ValueError(f"unknown capsule names in exclude: {sorted(unknown)}")
    bad_keys = set(excl) - set(MANO_SEGMENTS) - {"palm", "thumb", "index", "middle", "ring", "pinky"}
    if bad_keys:
        raise ValueError(f"exclude keys must be finger groups or MANO segment names, got {sorted(bad_keys)}")
    go, fp, wp, single = _hand_arrays(global_orient, finger_pose, wrist_pos)
    parents = layout.parents
    # taxels sharing a parent segment share the same capsule subset → one call per segment
    groups: dict[str, list[int]] = {}
    for i, par in enumerate(parents):
        groups.setdefault(par, []).append(i)
    keep = {}
    for seg in groups:
        drop = set(excl.get(finger_of(seg), ())) | set(excl.get(seg, ()))
        keep[seg] = [j for j, n in enumerate(CAPSULE_NAMES) if n not in drop]
    pl = torch.as_tensor(layout.positions, dtype=torch.float64)
    nl = torch.as_tensor(layout.normals, dtype=torch.float64)
    out = np.zeros((fp.shape[0], layout.n), dtype=bool)
    for s, e, fk in _fk_chunks(skeleton, go, fp, wp, chunk):
        pos, _ = _taxels_from_fk(skeleton, fk, idx, pl, nl)
        p0, p1, rad, names = skeleton.capsules(fk)
        p0, p1 = p0.numpy(), p1.numpy()
        for g, tax in groups.items():
            k = keep[g]
            out[s:e, tax] = self_touch_labels(pos[:, tax], [parents[i] for i in tax], p0[:, k], p1[:, k],
                                              rad[k], [names[j] for j in k], margin=margin)
    return out[0] if single else out


class ManoPoseProvider:
    """``TaxelPoseProvider`` for a glove: ``pose_at_fn(t)`` → hand pose → taxel poses.

    ``pose_at_fn(t)`` returns ``(global_orient[3], finger_pose[15,3], wrist_pos[3])`` or a dict
    with those keys (``wrist_pos`` optional).
    """

    def __init__(self, layout: Layout, skeleton: ManoSkeleton | None,
                 pose_at_fn: Callable[[float], object]):
        _parent_index(layout)
        self.layout = layout
        self.skeleton = skeleton if skeleton is not None else ManoSkeleton()
        self._fn = pose_at_fn

    @property
    def n_taxels(self) -> int:
        return self.layout.n

    def hand_pose_at(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        out = self._fn(float(t))
        if isinstance(out, Mapping):
            return out["global_orient"], out["finger_pose"], out.get("wrist_pos")
        if len(out) == 2:
            return out[0], out[1], None
        return out[0], out[1], out[2]

    def pose_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        go, fp, wp = self.hand_pose_at(t)
        return taxel_poses_from_hand(self.layout, self.skeleton, np.asarray(go).reshape(3),
                                     np.asarray(fp).reshape(15, 3),
                                     None if wp is None else np.asarray(wp).reshape(3))
