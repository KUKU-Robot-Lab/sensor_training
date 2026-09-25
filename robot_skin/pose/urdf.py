"""Minimal URDF parser + differentiable forward kinematics (torch, batched).

Only what a robot *hand* needs, with no dependency beyond the standard library and torch:
links, and joints of type ``revolute`` / ``continuous`` / ``prismatic`` / ``fixed`` with
``<origin xyz rpy>``, ``<axis xyz>``, ``<limit lower upper effort velocity>`` and ``<mimic joint
multiplier offset>``. ``floating`` / ``planar`` joints are treated as fixed (with a warning);
visuals, collisions, inertials, transmissions and gazebo tags are ignored. xacro must be expanded
beforehand (``xacro model.urdf.xacro > model.urdf``).

Kinematics follow the URDF spec: the joint frame is the parent link frame moved by ``origin``
(``R = Rz(yaw)·Ry(pitch)·Rx(roll)``, see ``geometry.rotations.rpy_to_matrix``); the child link
frame is the joint frame moved by the joint motion (rotation by ``q`` about ``axis`` for
revolute/continuous, translation ``q·axis`` for prismatic). ``T_child = T_parent · T_origin ·
T_motion(q)``.

Actuated joints (``URDFModel.joint_names``) are the movable, non-mimic joints in file order —
this is the order of ``q[..., D]`` everywhere (``joint_state.npz`` columns must be reordered with
:meth:`URDFModel.reorder_q` when the driver publishes another order). Mimic joints follow
``q_mimic = multiplier · q_source + offset`` (``mimic="ignore"`` keeps them at 0 instead).
"""
from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from robot_skin.geometry.rotations import as_tensor, rpy_to_matrix

JOINT_TYPES = ("revolute", "continuous", "prismatic", "fixed")
_MOVABLE = ("revolute", "continuous", "prismatic")


@dataclass(frozen=True)
class URDFJoint:
    name: str
    type: str                       # revolute | continuous | prismatic | fixed
    parent: str
    child: str
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)   # unit
    lower: float = -math.inf
    upper: float = math.inf
    effort: float = math.inf
    velocity: float = math.inf
    mimic: tuple[str, float, float] | None = None       # (source joint, multiplier, offset)

    @property
    def movable(self) -> bool:
        return self.type in _MOVABLE

    def origin_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(torch.tensor(self.origin_rpy, dtype=torch.float64)).numpy()
        T[:3, 3] = self.origin_xyz
        return T


def _floats(s: str | None, n: int, default: Sequence[float], what: str) -> tuple[float, ...]:
    if s is None:
        return tuple(float(v) for v in default)
    vals = s.split()
    if len(vals) != n:
        raise ValueError(f"{what}: expected {n} numbers, got {s!r}")
    return tuple(float(v) for v in vals)


def _parse_joint(el: ET.Element, mimic_mode: str) -> URDFJoint:
    name = el.get("name")
    jtype = el.get("type")
    if not name or not jtype:
        raise ValueError("every <joint> needs name and type attributes")
    p, c = el.find("parent"), el.find("child")
    if p is None or c is None or not p.get("link") or not c.get("link"):
        raise ValueError(f"joint {name!r}: missing <parent link=…> or <child link=…>")
    if jtype in ("floating", "planar"):
        warnings.warn(f"joint {name!r}: type {jtype!r} unsupported → treated as fixed", stacklevel=3)
        jtype = "fixed"
    if jtype not in JOINT_TYPES:
        raise ValueError(f"joint {name!r}: unknown type {jtype!r}")
    o = el.find("origin")
    xyz = _floats(o.get("xyz") if o is not None else None, 3, (0, 0, 0), f"joint {name!r} origin xyz")
    rpy = _floats(o.get("rpy") if o is not None else None, 3, (0, 0, 0), f"joint {name!r} origin rpy")
    a = el.find("axis")
    axis = np.asarray(_floats(a.get("xyz") if a is not None else None, 3, (1, 0, 0), f"joint {name!r} axis"))
    if jtype in _MOVABLE:
        n = np.linalg.norm(axis)
        if n < 1e-12:
            raise ValueError(f"joint {name!r}: zero axis")
        axis = axis / n
    lower, upper, effort, velocity = -math.inf, math.inf, math.inf, math.inf
    lim = el.find("limit")
    if lim is not None:
        effort = float(lim.get("effort", math.inf))
        velocity = float(lim.get("velocity", math.inf))
        if jtype in ("revolute", "prismatic"):
            lower, upper = float(lim.get("lower", 0.0)), float(lim.get("upper", 0.0))
            if upper < lower:
                raise ValueError(f"joint {name!r}: upper limit {upper} < lower limit {lower}")
    elif jtype in ("revolute", "prismatic"):
        warnings.warn(f"joint {name!r}: {jtype} joint without <limit> → unbounded", stacklevel=3)
    mimic = None
    m = el.find("mimic")
    if m is not None and jtype in _MOVABLE:
        if not m.get("joint"):
            raise ValueError(f"joint {name!r}: <mimic> needs a joint attribute")
        if mimic_mode == "ignore":
            warnings.warn(f"joint {name!r}: mimic ignored (mimic='ignore') → held at 0", stacklevel=3)
        mimic = (m.get("joint"), float(m.get("multiplier", 1.0)), float(m.get("offset", 0.0)))
    return URDFJoint(name=name, type=jtype, parent=p.get("link"), child=c.get("link"),
                     origin_xyz=xyz, origin_rpy=rpy, axis=tuple(float(v) for v in axis),
                     lower=lower, upper=upper, effort=effort, velocity=velocity, mimic=mimic)


class URDFModel:
    """Kinematic tree of a URDF. Build with :meth:`from_string` / :meth:`from_file`.

    Attributes: ``name``, ``link_names`` (file order), ``joints`` (all, file order),
    ``joint_names`` (actuated), ``root_link``, ``lower`` / ``upper`` / ``velocity_limits`` ([D] np).
    """

    def __init__(self, name: str, links: Sequence[str], joints: Sequence[URDFJoint], *,
                 mimic: str = "follow"):
        if mimic not in ("follow", "ignore"):
            raise ValueError("mimic must be 'follow' or 'ignore'")
        self.name = name
        self.mimic_mode = mimic
        self.joints: tuple[URDFJoint, ...] = tuple(joints)
        names = [j.name for j in self.joints]
        if len(set(names)) != len(names):
            raise ValueError("duplicate joint names")
        link_list = list(dict.fromkeys(links))
        for j in self.joints:                      # links referenced only by joints
            for ln in (j.parent, j.child):
                if ln not in link_list:
                    link_list.append(ln)
        self.link_names: tuple[str, ...] = tuple(link_list)
        self._joint_by_name = {j.name: j for j in self.joints}
        self._parent_joint: dict[str, URDFJoint] = {}
        for j in self.joints:
            if j.child in self._parent_joint:
                raise ValueError(f"link {j.child!r} has two parent joints")
            self._parent_joint[j.child] = j
        roots = [ln for ln in self.link_names if ln not in self._parent_joint]
        if len(roots) != 1:
            raise ValueError(f"URDF must have exactly one root link, found {roots}")
        self.root_link: str = roots[0]
        # topological (BFS) order of joints from the root; detects cycles / disconnected parts
        children: dict[str, list[URDFJoint]] = {}
        for j in self.joints:
            children.setdefault(j.parent, []).append(j)
        order: list[URDFJoint] = []
        queue = deque([self.root_link])
        seen = {self.root_link}
        while queue:
            ln = queue.popleft()
            for j in children.get(ln, []):
                if j.child in seen:
                    raise ValueError(f"kinematic loop at link {j.child!r}")
                seen.add(j.child)
                order.append(j)
                queue.append(j.child)
        if len(seen) != len(self.link_names):
            raise ValueError(f"links not connected to root {self.root_link!r}: "
                             f"{sorted(set(self.link_names) - seen)}")
        self._order: tuple[URDFJoint, ...] = tuple(order)
        self._children = children
        # actuated joints and mimic resolution → each movable joint = (actuated index, mult, offset)
        self.joint_names: tuple[str, ...] = tuple(
            j.name for j in self.joints if j.movable and j.mimic is None)
        self._act_index = {n: i for i, n in enumerate(self.joint_names)}
        self._drive: dict[str, tuple[int, float, float] | None] = {}
        for j in self.joints:
            if j.movable:
                self._drive[j.name] = self._resolve_drive(j.name, set())
        act = [self._joint_by_name[n] for n in self.joint_names]
        self.lower = np.array([j.lower for j in act], dtype=np.float64)
        self.upper = np.array([j.upper for j in act], dtype=np.float64)
        self.velocity_limits = np.array([j.velocity for j in act], dtype=np.float64)
        self._cache: dict = {}

    def _resolve_drive(self, name: str, visiting: set) -> tuple[int, float, float] | None:
        j = self._joint_by_name[name]
        if j.mimic is None:
            return (self._act_index[name], 1.0, 0.0)
        if self.mimic_mode == "ignore":
            return None                                   # held at 0
        if name in visiting:
            raise ValueError(f"mimic cycle through joint {name!r}")
        src, mult, off = j.mimic
        if src not in self._joint_by_name or not self._joint_by_name[src].movable:
            raise ValueError(f"joint {name!r} mimics unknown/fixed joint {src!r}")
        base = self._resolve_drive(src, visiting | {name})
        if base is None:
            return None
        i, m0, o0 = base
        return (i, mult * m0, mult * o0 + off)

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def from_string(cls, xml: str, *, mimic: str = "follow") -> "URDFModel":
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as e:
            raise ValueError(f"invalid URDF XML: {e}") from e
        if root.tag != "robot":
            raise ValueError(f"URDF root element must be <robot>, got <{root.tag}> (expand xacro first?)")
        links = [el.get("name") for el in root.findall("link")]
        if any(not ln for ln in links):
            raise ValueError("every <link> needs a name")
        joints = [_parse_joint(el, mimic) for el in root.findall("joint")]
        return cls(root.get("name", ""), links, joints, mimic=mimic)

    @classmethod
    def from_file(cls, path: str | Path, *, mimic: str = "follow") -> "URDFModel":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"URDF not found: {p}")
        return cls.from_string(p.read_text(), mimic=mimic)

    # ── views ─────────────────────────────────────────────────────────────
    @property
    def n_dof(self) -> int:
        return len(self.joint_names)

    def joint(self, name: str) -> URDFJoint:
        return self._joint_by_name[name]

    def parent_joint(self, link: str) -> URDFJoint | None:
        return self._parent_joint.get(link)

    def chain(self, link: str) -> list[URDFJoint]:
        """Joints from the root to ``link`` (root first)."""
        if link not in self.link_names:
            raise KeyError(f"unknown link {link!r}")
        out = []
        while link in self._parent_joint:
            j = self._parent_joint[link]
            out.append(j)
            link = j.parent
        return out[::-1]

    def clamp(self, q):
        """Clip ``q[...,D]`` to joint limits (numpy or torch)."""
        if isinstance(q, torch.Tensor):
            lo = torch.as_tensor(self.lower, dtype=q.dtype, device=q.device)
            hi = torch.as_tensor(self.upper, dtype=q.dtype, device=q.device)
            return torch.maximum(torch.minimum(q, hi), lo)
        return np.clip(q, self.lower, self.upper)

    def reorder_q(self, q, names: Sequence[str], *, fill: float = 0.0) -> np.ndarray:
        """Map ``q[...,K]`` given in ``names`` order to model order ``[...,D]``.

        Names that are not actuated joints (e.g. mimic joints some drivers publish) are dropped;
        actuated joints missing from ``names`` get ``fill`` (a warning is issued).
        """
        q = np.asarray(q, dtype=np.float64)
        names = [str(n) for n in names]
        if q.shape[-1] != len(names):
            raise ValueError(f"q has {q.shape[-1]} columns but {len(names)} names")
        pos = {n: i for i, n in enumerate(names)}
        missing = [n for n in self.joint_names if n not in pos]
        if missing:
            warnings.warn(f"joints missing from joint_state, filled with {fill}: {missing}", stacklevel=2)
        out = np.full(q.shape[:-1] + (self.n_dof,), fill, dtype=np.float64)
        for i, n in enumerate(self.joint_names):
            if n in pos:
                out[..., i] = q[..., pos[n]]
        return out

    # ── forward kinematics ────────────────────────────────────────────────
    def _consts(self, dtype: torch.dtype, device: torch.device) -> dict:
        key = (dtype, str(device))
        if key not in self._cache:
            self._cache[key] = {
                j.name: (torch.as_tensor(j.origin_matrix(), dtype=dtype, device=device),
                         torch.as_tensor(j.axis, dtype=dtype, device=device))
                for j in self._order}
        return self._cache[key]

    def _needed_joints(self, links: Iterable[str] | None) -> tuple[URDFJoint, ...]:
        if links is None:
            return self._order
        need: set[str] = set()
        for ln in links:
            need.update(j.name for j in self.chain(ln))
        return tuple(j for j in self._order if j.name in need)

    def fk(self, q, *, base_T=None, links: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
        """Link poses ``{link: [...,4,4]}`` in the root-link frame (or ``base_T · root``).

        ``q[...,D]`` in :attr:`joint_names` order (numpy → float64 tensor). ``base_T[...,4,4]``
        (optional) broadcasts against the leading dims of ``q``. Differentiable w.r.t. ``q`` (and
        ``base_T``). ``links`` restricts the output (and the work) to those links.
        """
        q = as_tensor(q)
        if not q.is_floating_point():
            q = q.to(torch.float64)
        if q.shape[-1] != self.n_dof:
            raise ValueError(f"q must be [..., {self.n_dof}] ({list(self.joint_names)}), got {tuple(q.shape)}")
        lead = q.shape[:-1]
        dtype, device = q.dtype, q.device
        c = self._consts(dtype, device)
        eye = torch.eye(4, dtype=dtype, device=device)
        if base_T is None:
            root = eye.expand(*lead, 4, 4)
        else:
            base = as_tensor(base_T).to(dtype=dtype, device=device)
            if base.ndim < 2 or tuple(base.shape[-2:]) != (4, 4):
                raise ValueError(f"base_T must be [..., 4, 4], got {tuple(base.shape)}")
            lead = torch.broadcast_shapes(lead, base.shape[:-2])
            root = base.expand(*lead, 4, 4)
            q = q.expand(*lead, self.n_dof)
        out: dict[str, torch.Tensor] = {self.root_link: root}
        for j in self._needed_joints(links):
            T_o, axis = c[j.name]
            T = out[j.parent] @ T_o
            drive = self._drive.get(j.name)
            if j.movable and drive is not None:
                i, mult, off = drive
                qj = q[..., i] * mult + off
                T = T @ self._motion(j.type, axis, qj)
            out[j.child] = T
        if links is not None:
            return {ln: out[ln] for ln in links}
        return out

    @staticmethod
    def _motion(jtype: str, axis: torch.Tensor, qj: torch.Tensor) -> torch.Tensor:
        """``[...,4,4]`` joint motion transform (closed-form Rodrigues; exact gradients at q=0)."""
        M = torch.zeros(*qj.shape, 4, 4, dtype=qj.dtype, device=qj.device)
        M[..., 3, 3] = 1.0
        if jtype == "prismatic":
            M[..., 0, 0] = M[..., 1, 1] = M[..., 2, 2] = 1.0
            M[..., :3, 3] = qj[..., None] * axis
            return M
        x, y, z = axis[0], axis[1], axis[2]
        K = torch.stack([torch.stack([0 * x, -z, y]), torch.stack([z, 0 * x, -x]),
                         torch.stack([-y, x, 0 * x])])
        s, c1 = torch.sin(qj)[..., None, None], (1 - torch.cos(qj))[..., None, None]
        M[..., :3, :3] = torch.eye(3, dtype=qj.dtype, device=qj.device) + s * K + c1 * (K @ K)
        return M

    def fk_numpy(self, q, *, base_T=None, links: Iterable[str] | None = None) -> dict[str, np.ndarray]:
        """:meth:`fk` without autograd, returning float64 numpy arrays."""
        with torch.no_grad():
            out = self.fk(torch.as_tensor(np.asarray(q, dtype=np.float64)),
                          base_T=None if base_T is None else torch.as_tensor(np.asarray(base_T, dtype=np.float64)),
                          links=links)
        return {k: v.numpy() for k, v in out.items()}

    def __repr__(self) -> str:
        return (f"URDFModel({self.name!r}, links={len(self.link_names)}, joints={len(self.joints)}, "
                f"dof={self.n_dof}, root={self.root_link!r})")
