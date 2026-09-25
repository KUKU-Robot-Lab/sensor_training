"""Human hand → robot hand retargeting by fingertip-vector matching.

The VTLA policy predicts the canonical human hand action (``action.space``, MANO). A robot hand
has other kinematics, so per control tick we solve a small inverse problem (optimisation-based
retargeting as in DexPilot, Handa et al., ICRA 2020, arXiv:1910.03135, and AnyTeleop, Qin et al.,
arXiv:2307.04577)::

    q* = argmin_{lower ≤ q ≤ upper}  Σ_v w_v ‖ v_r(q) − s · R_hr · v_h ‖²
                                    + λ_reg ‖q − q_nominal‖² + λ_smooth ‖q − q_prev‖²

- ``v_h`` — human keypoint vectors in the human wrist frame: wrist → fingertip (``tips``) and
  thumb → finger (``pairs``, DexPilot's thumb–finger vectors, which decide whether a pinch
  closes). ``v_r(q)`` — the same vectors between robot links from forward kinematics, in the
  robot base (palm) link frame.
- ``s`` (``scale``) — robot/human size ratio applied to the *human* vectors (AnyTeleop
  convention; the ``s·robot`` form in some write-ups is the same with ``s → 1/s``).
  :meth:`FingertipRetargeter.estimate_scale` computes it from a reference pose.
- ``R_hr`` (``human_to_robot``) — rotation from the human wrist frame (MANO canonical frame when
  tips come from ``pose.mano.ManoSkeleton``) to the robot base-link frame.
- Optional pinch handling (DexPilot-inspired): when a human thumb–finger distance drops below
  ``pinch_threshold`` the robot target for that pair becomes ``pinch_distance`` along the same
  direction with weight ``pinch_weight``, so a human pinch produces a robot pinch despite
  differing finger sizes.

Solvers (all torch autograd, batched over leading dims): ``lm`` (default) — projected
Levenberg–Marquardt on the stacked residual with an active-set for joint limits (few
iterations, exact on reachable targets); ``adam`` — projected Adam; ``lbfgs`` — unconstrained
L-BFGS followed by a projection onto the limits (prefer ``lm``/``adam`` when limits are active).
Sequences are solved frame by frame, warm-started from the previous solution with the
``λ_smooth`` term anchoring it (:meth:`retarget_sequence`, :meth:`step`).

The kinematics are abstract: ``fk`` is any callable ``q[B,D] → {name: pos[B,3] | T[B,4,4]}``,
or a URDF-model-like object with ``fk(q, links=...) → {link: [B,4,4]}`` and ``lower``/``upper``
(``robot_skin.pose.urdf.URDFModel`` fits; it is duck-typed, not imported).
"""
from __future__ import annotations

import inspect
import math
import warnings
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from robot_skin.geometry.rotations import as_tensor

#: Human fingertip order (= ``robot_skin.pose.mano.FINGERS``, the ``tip_pos`` order of ManoSkeleton).
HAND_FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
WRIST = "wrist"
RETARGET_METHODS = ("lm", "adam", "lbfgs")
JACOBIAN_MODES = ("autograd", "fd")
VECTOR_SETS = ("tips", "pairs", "tips+pairs")
_DEFAULT_ITERS = {"lm": 50, "adam": 300, "lbfgs": 100}
_LAM_MIN, _LAM_MAX = 1e-12, 1e10


def _accepts_kw(fn: Callable, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class FingertipRetargeter:
    """Optimisation-based fingertip-vector retargeting (see module docstring).

    Parameters
    ----------
    fk : callable ``q[B,D] → {name: pos[B,3] or T[B,4,4]}`` (torch, differentiable, batched), or a
        URDF-model-like object exposing ``fk(q, links=None) → {link: [B,4,4]}`` and ``lower`` /
        ``upper`` (and optionally ``n_dof`` / ``joint_names``).
    tip_links : ``{human finger: fk output name}``, e.g. ``{"thumb": "thumb_tip_link", …}``. Robots
        with fewer fingers map a subset. ``None`` (callables only) → every finger in
        ``human_tip_names`` that is also a key of the FK output.
    base_link : FK output name of the robot frame matching the human wrist (palm/base link). If it
        is a 4×4 transform the robot vectors are expressed in its frame. ``None`` → FK origin.
    lower, upper : joint limits ``[D]`` (±inf allowed). Default: from the model, else unbounded.
    dof : number of joints; only needed when neither limits, a model nor ``q_nominal`` give it.
    human_tip_names : row order of ``human_tip_pos`` (default MANO/``HAND_FINGERS`` order).
    scale : robot/human size ratio multiplying the *human* vectors (AnyTeleop convention; < 1 for a
        robot hand smaller than the human's). A cost written as ``‖s·v_r − v_h‖²`` has the same
        minimiser with ``scale = 1/s``. Calibrate with :meth:`estimate_scale`.
    human_to_robot : ``[3,3]`` rotation human wrist frame → robot base frame (default identity).
    vectors : ``tips`` | ``pairs`` | ``tips+pairs`` or explicit ``[(origin, target), …]`` with names
        from the mapped fingers and ``"wrist"``.
    tip_weight, pair_weight : weights of wrist→tip and finger→finger vectors.
    reg_weight : pull toward ``q_nominal`` — resolves redundancy (null space); units: 1 rad² costs
        as much as ``reg_weight`` m² of vector error.
    q_nominal : regularisation target and default initial guess. Default: mid-range for bounded
        joints (0 for unbounded) — starting a flexion joint at its limit, where the finger is
        straight and the Jacobian singular, can trap the solver on the boundary.
    smooth_weight : pull toward the previous solution (temporal smoothness; sequences/streaming).
    pinch_threshold, pinch_distance, pinch_weight : optional pinch handling for pair vectors
        (human metres / robot metres / weight, default ``10·pair_weight``).
    method : ``lm`` | ``adam`` | ``lbfgs``; ``iters`` (default per method); ``lr`` (Adam);
        ``tol`` step tolerance (rad) and ``ftol`` relative cost-decrease tolerance for ``lm``.
        For a tight control loop cap ``iters`` (warm-started frames need only a few).
    jacobian : ``autograd`` (vectorised reverse mode) | ``fd`` (batched central differences) for ``lm``.
    """

    def __init__(self, fk, tip_links: Mapping[str, str] | None = None, lower=None, upper=None,
                 human_tip_names: Sequence[str] = HAND_FINGERS, scale: float = 1.0,
                 reg_weight: float = 1e-7, smooth_weight: float = 1e-6, iters: int | None = None,
                 lr: float = 0.02, *, base_link: str | None = None, dof: int | None = None,
                 human_to_robot=None, vectors: str | Sequence[tuple[str, str]] = "tips+pairs",
                 tip_weight: float = 1.0, pair_weight: float = 1.0, q_nominal=None,
                 pinch_threshold: float | None = None, pinch_distance: float = 0.0,
                 pinch_weight: float | None = None, method: str = "lm", tol: float = 1e-8,
                 ftol: float = 1e-10, jacobian: str = "autograd", dtype: torch.dtype = torch.float64,
                 device: str | torch.device = "cpu"):
        if method not in RETARGET_METHODS:
            raise ValueError(f"method must be one of {RETARGET_METHODS}, got {method!r}")
        if jacobian not in JACOBIAN_MODES:
            raise ValueError(f"jacobian must be one of {JACOBIAN_MODES}, got {jacobian!r}")
        for nm, v in (("tip_weight", tip_weight), ("pair_weight", pair_weight), ("reg_weight", reg_weight),
                      ("smooth_weight", smooth_weight), ("lr", lr), ("tol", tol), ("ftol", ftol)):
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"{nm} must be finite and >= 0, got {v}")
        if not (math.isfinite(scale) and scale > 0):
            raise ValueError(f"scale must be > 0, got {scale}")
        self.dtype, self.device = dtype, torch.device(device)
        self.method = method
        self.iters = int(iters if iters is not None else _DEFAULT_ITERS[method])
        if self.iters < 1:
            raise ValueError("iters must be >= 1")
        self.lr, self.tol, self.ftol = float(lr), float(tol), float(ftol)
        self.jacobian = jacobian
        self.scale = float(scale)
        self.tip_weight, self.pair_weight = float(tip_weight), float(pair_weight)
        self.reg_weight, self.smooth_weight = float(reg_weight), float(smooth_weight)
        self.pinch_threshold = None if pinch_threshold is None else float(pinch_threshold)
        self.pinch_distance = float(pinch_distance)
        self.pinch_weight = float(10.0 * pair_weight if pinch_weight is None else pinch_weight)
        self.base_link = base_link
        self.human_tip_names = tuple(human_tip_names)
        if len(set(self.human_tip_names)) != len(self.human_tip_names) or WRIST in self.human_tip_names:
            raise ValueError(f"human_tip_names must be unique and not contain {WRIST!r}")

        # ── kinematics: callable or URDF-model-like ──────────────────────────
        self.model = None
        self.joint_names: tuple[str, ...] | None = None
        if hasattr(fk, "fk") and callable(getattr(fk, "fk")):
            self.model = fk
            lower = getattr(fk, "lower", None) if lower is None else lower
            upper = getattr(fk, "upper", None) if upper is None else upper
            jn = getattr(fk, "joint_names", None)
            self.joint_names = None if jn is None else tuple(jn)
            if dof is None:
                dof = getattr(fk, "n_dof", None) or (None if jn is None else len(jn))
            if tip_links is None:
                raise ValueError("tip_links {finger: link name} is required with a URDF model")
        elif not callable(fk):
            raise TypeError("fk must be a callable q -> {name: pos|T} or a model with .fk(q)")

        D = self._infer_dof(dof, lower, upper, q_nominal)
        self.dof = D
        lo = np.full(D, -np.inf) if lower is None else np.asarray(lower, dtype=np.float64).reshape(-1)
        hi = np.full(D, np.inf) if upper is None else np.asarray(upper, dtype=np.float64).reshape(-1)
        if lo.shape != (D,) or hi.shape != (D,):
            raise ValueError(f"lower/upper must have {D} entries, got {lo.shape}, {hi.shape}")
        if np.any(hi < lo):
            raise ValueError("upper < lower for some joints")
        self.lower, self.upper = lo, hi
        self._lo = torch.as_tensor(lo, dtype=dtype, device=self.device)
        self._hi = torch.as_tensor(hi, dtype=dtype, device=self.device)
        if q_nominal is None:        # mid-range: away from limits and the straight-finger singularity
            fin = np.isfinite(lo) & np.isfinite(hi)
            qn = np.zeros(D)
            qn[fin] = 0.5 * (lo[fin] + hi[fin])
        else:
            qn = np.asarray(q_nominal, dtype=np.float64).reshape(-1)
        if qn.shape != (D,):
            raise ValueError(f"q_nominal must have {D} entries, got {qn.shape}")
        self.q_nominal = np.clip(qn, lo, hi)
        self._qn = torch.as_tensor(self.q_nominal, dtype=dtype, device=self.device)

        R = np.eye(3) if human_to_robot is None else np.asarray(human_to_robot, dtype=np.float64)
        if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or np.linalg.det(R) <= 0:
            raise ValueError("human_to_robot must be a 3x3 rotation matrix")
        self.human_to_robot = R
        self._R_hr = torch.as_tensor(R, dtype=dtype, device=self.device)

        # ── FK wrapper + finger mapping (probe once to fail fast) ───────────
        if self.model is not None:
            self.tip_links = dict(tip_links)
            needed = list(dict.fromkeys(list(self.tip_links.values()) + ([base_link] if base_link else [])))
            mfk = self.model.fk
            self._fk = (lambda q: mfk(q, links=needed)) if _accepts_kw(mfk, "links") else mfk
        else:
            self._fk = fk
        with torch.no_grad():
            probe = self._fk(self._qn[None])
        if not isinstance(probe, Mapping):
            raise TypeError(f"fk must return a mapping name -> tensor, got {type(probe).__name__}")
        if tip_links is None:
            tip_links = {f: f for f in self.human_tip_names if f in probe}
            if not tip_links:
                raise ValueError(f"fk output keys {sorted(probe)} match none of {self.human_tip_names}; "
                                 "pass tip_links={finger: name}")
        self.tip_links = dict(tip_links)
        unknown = [f for f in self.tip_links if f not in self.human_tip_names]
        if unknown:
            raise ValueError(f"tip_links fingers {unknown} are not in human_tip_names {self.human_tip_names}")
        self.fingers: tuple[str, ...] = tuple(f for f in self.human_tip_names if f in self.tip_links)
        self._human_idx = [self.human_tip_names.index(f) for f in self.fingers]
        for name in list(self.tip_links.values()) + ([base_link] if base_link else []):
            if name not in probe:
                raise KeyError(f"fk output has no {name!r}; available: {sorted(probe)}")
        self._point_shapes_ok(probe)

        self.vectors = self._build_vectors(vectors)
        self._is_pair = torch.as_tensor([o != WRIST for o, _ in self.vectors], device=self.device)
        self._w = torch.as_tensor([self.tip_weight if o == WRIST else self.pair_weight for o, _ in self.vectors],
                                  dtype=dtype, device=self.device)
        self._vec_jac_ok: bool | None = None
        self._last: torch.Tensor | None = None
        self.last_info: dict = {}

    @classmethod
    def from_config(cls, fk, cfg: Mapping | None = None) -> "FingertipRetargeter":
        """Build from a plain dict (YAML ``retarget:`` block): keys are the constructor arguments
        (``tip_links``, ``base_link``, ``scale``, ``human_to_robot`` as nested lists, ``vectors``,
        weights, ``method``, ``iters``, ``jacobian`` …). Unknown keys raise (typos must not pass
        silently into a robot controller)."""
        cfg = dict(cfg or {})
        allowed = set(inspect.signature(cls.__init__).parameters) - {"self", "fk", "dtype", "device"}
        unknown = sorted(set(cfg) - allowed)
        if unknown:
            raise ValueError(f"unknown retarget config keys {unknown}; allowed: {sorted(allowed)}")
        if isinstance(cfg.get("vectors"), list):
            cfg["vectors"] = [tuple(v) for v in cfg["vectors"]]
        return cls(fk, **cfg)

    # ── setup helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _infer_dof(dof, lower, upper, q_nominal) -> int:
        for src in (dof, None if lower is None else np.size(lower), None if upper is None else np.size(upper),
                    None if q_nominal is None else np.size(q_nominal)):
            if src is not None:
                if int(src) < 1:
                    raise ValueError("robot needs at least one joint")
                return int(src)
        raise ValueError("cannot infer the number of joints: pass dof=, lower/upper or q_nominal")

    def _point_shapes_ok(self, probe: Mapping) -> None:
        for name in list(self.tip_links.values()) + ([self.base_link] if self.base_link else []):
            v = probe[name]
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"fk output {name!r} must be a torch tensor (differentiable), got {type(v).__name__}")
            if not (tuple(v.shape[-2:]) == (4, 4) or v.shape[-1] == 3):
                raise ValueError(f"fk output {name!r} must be [...,3] positions or [...,4,4] transforms, "
                                 f"got {tuple(v.shape)}")

    def _build_vectors(self, vectors) -> tuple[tuple[str, str], ...]:
        tips = [(WRIST, f) for f in self.fingers]
        pairs = [("thumb", f) for f in self.fingers if f != "thumb"] if "thumb" in self.fingers else []
        if isinstance(vectors, str):
            if vectors not in VECTOR_SETS:
                raise ValueError(f"vectors must be one of {VECTOR_SETS} or a list of (origin, target)")
            out = {"tips": tips, "pairs": pairs, "tips+pairs": tips + pairs}[vectors]
        else:
            out = [(str(o), str(t)) for o, t in vectors]
            allowed = set(self.fingers) | {WRIST}
            bad = [v for v in out if v[0] not in allowed or v[1] not in allowed or v[0] == v[1]]
            if bad:
                raise ValueError(f"invalid vectors {bad}; names must be distinct and in {sorted(allowed)}")
        if not out:
            raise ValueError(f"no retargeting vectors for fingers {self.fingers} with vectors={vectors!r}")
        return tuple(out)

    # ── kinematics ─────────────────────────────────────────────────────────
    def robot_points(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """Robot keypoints ``{finger | "wrist": [B,3]}`` in the base-link frame for ``q[B,D]``."""
        out = self._fk(q)
        base_p, base_R = None, None
        if self.base_link:
            b = out[self.base_link].to(self.dtype)
            if tuple(b.shape[-2:]) == (4, 4):
                base_p, base_R = b[..., :3, 3], b[..., :3, :3]
            else:
                base_p = b
        pts = {WRIST: torch.zeros(q.shape[0], 3, dtype=self.dtype, device=q.device)}
        for f in self.fingers:
            v = out[self.tip_links[f]].to(self.dtype)
            p = v[..., :3, 3] if tuple(v.shape[-2:]) == (4, 4) else v
            p = p.expand(q.shape[0], 3) if p.ndim == 1 else p
            if base_p is not None:
                p = p - base_p
                if base_R is not None:                 # express in the base frame: Rᵀ(p − p_b)
                    p = (p[..., None, :] @ base_R)[..., 0, :]
            pts[f] = p
        return pts

    def robot_vectors(self, q: torch.Tensor) -> torch.Tensor:
        """``[B,V,3]`` robot keypoint vectors (``self.vectors`` order)."""
        pts = self.robot_points(q)
        return torch.stack([pts[t] - pts[o] for o, t in self.vectors], dim=1)

    # ── human side ─────────────────────────────────────────────────────────
    def _human_local(self, human_tip_pos, human_wrist_T) -> torch.Tensor:
        p = as_tensor(human_tip_pos).to(dtype=self.dtype, device=self.device)
        F = len(self.human_tip_names)
        if p.ndim < 2 or tuple(p.shape[-2:]) != (F, 3):
            raise ValueError(f"human_tip_pos must be [..., {F}, 3] ({self.human_tip_names}), got {tuple(p.shape)}")
        # a NaN target would silently turn into NaN joint commands (adam / lbfgs) → fail loudly
        if not bool(torch.isfinite(p).all()):
            raise ValueError("human_tip_pos contains non-finite values")
        if human_wrist_T is not None:
            T = as_tensor(human_wrist_T).to(dtype=self.dtype, device=self.device)
            if T.ndim < 2 or tuple(T.shape[-2:]) != (4, 4):
                raise ValueError(f"human_wrist_T must be [..., 4, 4], got {tuple(T.shape)}")
            if not bool(torch.isfinite(T).all()):
                raise ValueError("human_wrist_T contains non-finite values")
            R, t = T[..., :3, :3], T[..., :3, 3]
            p = (p - t[..., None, :]) @ R                # row form of Rᵀ(p − t)
        return p

    def human_vectors(self, human_tip_pos, human_wrist_T=None) -> torch.Tensor:
        """``[...,V,3]`` human keypoint vectors in the human wrist frame (unscaled, unrotated)."""
        p = self._human_local(human_tip_pos, human_wrist_T)
        pts = {WRIST: torch.zeros_like(p[..., 0, :])}
        pts.update({f: p[..., i, :] for f, i in zip(self.fingers, self._human_idx)})
        return torch.stack([pts[t] - pts[o] for o, t in self.vectors], dim=-2)

    def _targets(self, v_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Human vectors ``[B,V,3]`` → robot-frame targets ``[B,V,3]`` and weights ``[B,V]``."""
        v = v_h @ self._R_hr.T
        target = self.scale * v
        w = self._w.expand(v.shape[0], -1)
        if self.pinch_threshold is not None:
            d = v.norm(dim=-1)
            pinch = (d < self.pinch_threshold) & self._is_pair[None]
            unit = v / d.clamp_min(1e-9)[..., None]
            target = torch.where(pinch[..., None], self.pinch_distance * unit, target)
            w = torch.where(pinch, torch.full_like(w, self.pinch_weight), w)
        return target, w

    # ── objective ──────────────────────────────────────────────────────────
    def _residual(self, q, target, sw, q_prev) -> torch.Tensor:
        """Stacked weighted residual ``[B,R]`` (cost = Σ residual²)."""
        r = [(sw[..., None] * (self.robot_vectors(q) - target)).flatten(1)]
        if self.reg_weight > 0:
            r.append(math.sqrt(self.reg_weight) * (q - self._qn))
        if q_prev is not None and self.smooth_weight > 0:
            r.append(math.sqrt(self.smooth_weight) * (q - q_prev))
        return torch.cat(r, dim=-1)

    def _clamp(self, q: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(q, self._hi), self._lo)

    def _jacobian(self, q, target, sw, q_prev) -> torch.Tensor:
        """``[B,R,D]`` Jacobian of the stacked residual.

        ``autograd``: rows are independent across the batch, so d(Σ_b r_b)/dq gives every J_b in
        one vectorised reverse pass. ``fd``: central differences evaluated in a single batched FK
        call (``2·D`` extra rows) — faster for Python-heavy FK and usable when the FK is not
        differentiable; float64 keeps its error ~h² ≈ 1e-12.
        """
        if self.jacobian == "fd":
            B, D = q.shape
            h = 1e-6 if self.dtype == torch.float64 else 1e-3
            E = torch.eye(D, dtype=self.dtype, device=q.device) * h
            Q = torch.cat([q[:, None, :] + E, q[:, None, :] - E], dim=1).reshape(B * 2 * D, D)

            def rep(x):
                return None if x is None else x.repeat_interleave(2 * D, dim=0)
            with torch.no_grad():
                r = self._residual(Q, rep(target), rep(sw), rep(q_prev)).reshape(B, 2, D, -1)
            return ((r[:, 0] - r[:, 1]) / (2 * h)).transpose(1, 2)

        def f(qq):
            return self._residual(qq, target, sw, q_prev).sum(0)
        if self._vec_jac_ok is not False:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")      # vmap "batching rule not implemented" notes
                    J = torch.autograd.functional.jacobian(f, q, vectorize=True)
                self._vec_jac_ok = True
                return J.permute(1, 0, 2)
            except Exception:                           # fk not vmap-friendly → per-row backward
                if self._vec_jac_ok:
                    raise
                self._vec_jac_ok = False
        return torch.autograd.functional.jacobian(f, q, vectorize=False).permute(1, 0, 2)

    # ── solvers ────────────────────────────────────────────────────────────
    def _solve(self, target, sw, q0, q_prev) -> torch.Tensor:
        if self.method == "lm":
            return self._solve_lm(target, sw, q0, q_prev)
        if self.method == "adam":
            return self._solve_adam(target, sw, q0, q_prev)
        return self._solve_lbfgs(target, sw, q0, q_prev)

    # Levenberg–Marquardt damping schedule: A = JᵀJ + λ·mean(diag JᵀJ)·I
    _LM_LAMBDA0, _LM_DOWN, _LM_UP = 1e-4, 0.2, 5.0

    def _solve_lm(self, target, sw, q0, q_prev) -> torch.Tensor:
        """Projected Levenberg–Marquardt with an active set at the joint limits.

        Stops per sample when an accepted step is below ``tol`` (rad), the relative cost decrease
        is below ``ftol``, the damping saturates (no descent possible) or the cost is ~0.
        """
        q = self._clamp(q0.detach())
        B, D = q.shape
        eye = torch.eye(D, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            r = self._residual(q, target, sw, q_prev)
        cost = (r * r).sum(-1)
        lam = torch.full((B,), self._LM_LAMBDA0, dtype=self.dtype, device=self.device)
        done = torch.zeros(B, dtype=torch.bool, device=self.device)
        J = g = H = None
        it = 0
        for it in range(1, self.iters + 1):
            if J is None:
                J = self._jacobian(q, target, sw, q_prev).detach()
                g = (J.transpose(1, 2) @ r[..., None])[..., 0]           # ∇(½cost) = Jᵀr
                H = J.transpose(1, 2) @ J
            # joints on a limit whose gradient pushes outward stay fixed this iteration
            active = ((q <= self._lo) & (g > 0)) | ((q >= self._hi) & (g < 0))
            free = ~active
            ff = free[:, :, None] & free[:, None, :]
            mu = H.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)
            A = torch.where(ff, H, torch.zeros_like(H)) + (lam * mu)[:, None, None] * eye \
                + torch.diag_embed(active.to(self.dtype))
            dq = -torch.linalg.solve(A, torch.where(free, g, torch.zeros_like(g))[..., None])[..., 0]
            q_new = self._clamp(q + dq)
            with torch.no_grad():
                r_new = self._residual(q_new, target, sw, q_prev)
            cost_new = (r_new * r_new).sum(-1)
            acc = (cost_new < cost) & ~done
            step = (q_new - q).abs().amax(-1)
            small_gain = (cost - cost_new) <= self.ftol * cost
            q = torch.where(acc[:, None], q_new, q)
            r = torch.where(acc[:, None], r_new, r)
            cost = torch.where(acc, cost_new, cost)
            lam = torch.where(acc, lam * self._LM_DOWN, lam * self._LM_UP).clamp(_LAM_MIN, _LAM_MAX)
            done = done | (acc & ((step < self.tol) | small_gain)) | (~acc & (lam >= _LAM_MAX)) \
                | (cost < 1e-24)
            if bool(done.all()):
                break
            if bool(acc.any()):
                J = None                                 # re-linearise at the new iterate
        self.last_info = {"method": "lm", "iters": it, "cost": cost.detach().cpu().numpy()}
        return q.detach()

    def _solve_adam(self, target, sw, q0, q_prev) -> torch.Tensor:
        q = self._clamp(q0.detach()).clone().requires_grad_(True)
        opt = torch.optim.Adam([q], lr=self.lr)
        for _ in range(self.iters):
            opt.zero_grad(set_to_none=True)
            loss = self._residual(q, target, sw, q_prev).square().sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                q.copy_(self._clamp(q))
        with torch.no_grad():
            cost = self._residual(q, target, sw, q_prev).square().sum(-1)
        self.last_info = {"method": "adam", "iters": self.iters, "cost": cost.cpu().numpy()}
        return q.detach()

    def _solve_lbfgs(self, target, sw, q0, q_prev) -> torch.Tensor:
        q = self._clamp(q0.detach()).clone().requires_grad_(True)
        opt = torch.optim.LBFGS([q], lr=1.0, max_iter=self.iters, history_size=20,
                                tolerance_grad=1e-14, tolerance_change=1e-18, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = self._residual(q, target, sw, q_prev).square().sum()
            loss.backward()
            return loss

        opt.step(closure)
        qc = self._clamp(q.detach())
        with torch.no_grad():
            cost = self._residual(qc, target, sw, q_prev).square().sum(-1)
        self.last_info = {"method": "lbfgs", "iters": self.iters, "cost": cost.cpu().numpy()}
        return qc

    # ── public API ─────────────────────────────────────────────────────────
    def _q_batch(self, q, B: int, what: str) -> torch.Tensor:
        t = as_tensor(q).to(dtype=self.dtype, device=self.device)
        if t.ndim < 1 or t.shape[-1] != self.dof:
            raise ValueError(f"{what} must be [..., {self.dof}], got {tuple(t.shape)}")
        t = t.reshape(-1, self.dof)                  # leading dims flattened like the targets'
        if t.shape[0] not in (1, B):
            raise ValueError(f"{what} batch {t.shape[0]} does not match {B}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{what} contains non-finite values")
        return t.expand(B, self.dof).clone()

    def retarget(self, human_tip_pos, human_wrist_T=None, q_init=None, q_prev=None):
        """Robot joints ``q[..., D]`` for human fingertips ``[..., F, 3]`` (``human_tip_names`` order).

        Fingertips are wrist-relative in the human wrist frame, or in any frame together with the
        wrist pose ``human_wrist_T[..., 4, 4]`` in that frame. ``q_init`` warm-starts (default
        ``q_nominal``); ``q_prev`` enables the smoothness term. Leading dims are solved as one
        batch (independently). numpy in → numpy out, torch in → torch out.
        """
        is_np = not isinstance(human_tip_pos, torch.Tensor)
        v_h = self.human_vectors(human_tip_pos, human_wrist_T)          # [...,V,3]
        lead = v_h.shape[:-2]
        v_h = v_h.reshape(-1, *v_h.shape[-2:])
        B = v_h.shape[0]
        target, w = self._targets(v_h)
        q0 = self._q_batch(self._qn if q_init is None else q_init, B, "q_init")
        qp = None if q_prev is None else self._q_batch(q_prev, B, "q_prev")
        q = self._solve(target, w.sqrt(), q0, qp).reshape(*lead, self.dof)
        return q.cpu().numpy() if is_np else q

    def retarget_sequence(self, human_tip_pos, human_wrist_T=None, q_init=None, *, warm_start: bool = True,
                          smooth: bool = True):
        """``[T, F, 3]`` → ``q[T, D]``. With ``warm_start`` each frame starts from the previous
        solution and (``smooth``) is anchored to it by ``smooth_weight``; otherwise all frames are
        solved as one independent batch from ``q_init``."""
        is_np = not isinstance(human_tip_pos, torch.Tensor)
        p = as_tensor(human_tip_pos).to(dtype=self.dtype, device=self.device)
        if p.ndim != 3:
            raise ValueError(f"human_tip_pos must be [T, F, 3], got {tuple(p.shape)}")
        T = p.shape[0]
        W = None
        if human_wrist_T is not None:
            W = as_tensor(human_wrist_T).to(dtype=self.dtype, device=self.device)
            W = W.expand(T, 4, 4) if W.ndim == 2 else W
            if tuple(W.shape) != (T, 4, 4):
                raise ValueError(f"human_wrist_T must be [4,4] or [{T},4,4], got {tuple(W.shape)}")
        if not warm_start:
            q = self.retarget(p, W, q_init=q_init)
            return q.cpu().numpy() if is_np else q
        out, q_last = [], None
        costs, iters = [], []
        for t in range(T):
            init = q_last if q_last is not None else q_init
            q_t = self.retarget(p[t], None if W is None else W[t], q_init=init,
                                q_prev=q_last if smooth else None)
            costs.append(float(self.last_info["cost"][0]))
            iters.append(int(self.last_info["iters"]))
            out.append(q_t)
            q_last = q_t
        q = torch.stack(out)
        self.last_info = {"method": self.method, "iters": np.asarray(iters), "cost": np.asarray(costs)}
        return q.cpu().numpy() if is_np else q

    def step(self, human_tip_pos, human_wrist_T=None):
        """Streaming retarget for a control loop: warm start + smoothness w.r.t. the last output."""
        q = self.retarget(human_tip_pos, human_wrist_T, q_init=self._last, q_prev=self._last)
        self._last = as_tensor(q).to(dtype=self.dtype, device=self.device).detach().reshape(-1, self.dof)[-1]
        return q

    def reset(self, q=None) -> None:
        """Forget the streaming state (or set it to the robot's current ``q``)."""
        self._last = None if q is None else self._q_batch(q, 1, "q")[0]

    # ── diagnostics / calibration ──────────────────────────────────────────
    def vector_error(self, q, human_tip_pos, human_wrist_T=None) -> np.ndarray:
        """Per-vector error ``‖v_r(q) − target‖`` ``[..., V]`` (robot metres, pinch targets applied)."""
        v_h = self.human_vectors(human_tip_pos, human_wrist_T)
        lead = v_h.shape[:-2]
        v_h = v_h.reshape(-1, *v_h.shape[-2:])
        target, _ = self._targets(v_h)
        qb = self._q_batch(q, v_h.shape[0], "q")
        with torch.no_grad():
            err = (self.robot_vectors(qb) - target).norm(dim=-1)
        return err.reshape(*lead, len(self.vectors)).cpu().numpy()

    def estimate_scale(self, human_tip_pos, q_ref=None, human_wrist_T=None) -> float:
        """Robot/human size ratio from a matching reference pose (e.g. both hands flat):
        mean ‖wrist→tip‖ of the robot at ``q_ref`` (default ``q_nominal``) over the human's."""
        p = self._human_local(human_tip_pos, human_wrist_T)
        h = torch.stack([p[..., i, :] for i in self._human_idx], dim=-2).norm(dim=-1).mean()
        q = self._q_batch(self._qn if q_ref is None else q_ref, 1, "q_ref")
        with torch.no_grad():
            pts = self.robot_points(q)
        r = torch.stack([pts[f][0] for f in self.fingers]).norm(dim=-1).mean()
        if float(h) <= 0:
            raise ValueError("human fingertips coincide with the wrist")
        return float(r / h)

    def __repr__(self) -> str:
        return (f"FingertipRetargeter(dof={self.dof}, fingers={self.fingers}, vectors={len(self.vectors)}, "
                f"method={self.method!r}, scale={self.scale:g})")


# ── human fingertips from MANO ─────────────────────────────────────────────
def human_fingertips(finger_pose, skeleton=None):
    """Wrist-frame fingertip positions ``[..., 5, 3]`` (``HAND_FINGERS`` order) from MANO
    ``finger_pose[..., 15, 3]`` via ``robot_skin.pose.mano.ManoSkeleton`` (default skeleton).
    Global orientation and wrist position are zeroed, so the result is in the MANO canonical
    wrist frame — what :meth:`FingertipRetargeter.retarget` expects."""
    from robot_skin.pose.mano import ManoSkeleton  # lazy: keeps retargeting URDF/MANO-agnostic
    skel = skeleton if skeleton is not None else ManoSkeleton()
    is_np = not isinstance(finger_pose, torch.Tensor)
    fp = as_tensor(finger_pose)
    with torch.no_grad():
        tips = skel.forward(finger_pose=fp)["tip_pos"]
    return tips.cpu().numpy() if is_np else tips


def hand_action_fingertips(action, skeleton=None):
    """Canonical hand actions ``[..., 54]`` → wrist-frame fingertips ``[..., 5, 3]``."""
    from robot_skin.action.space import FINGERS_AA, HAND_MANO_DIM
    a = action if isinstance(action, torch.Tensor) else np.asarray(action, dtype=np.float64)
    if a.shape[-1] != HAND_MANO_DIM:
        raise ValueError(f"hand action must be [..., {HAND_MANO_DIM}], got {tuple(a.shape)}")
    fp = a[..., FINGERS_AA].reshape(*a.shape[:-1], 15, 3)
    return human_fingertips(fp, skeleton)


__all__ = ["HAND_FINGERS", "WRIST", "RETARGET_METHODS", "JACOBIAN_MODES", "VECTOR_SETS", "FingertipRetargeter",
           "human_fingertips", "hand_action_fingertips"]
