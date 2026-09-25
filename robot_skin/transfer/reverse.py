"""Robot hand → human (MANO) hand: the reverse of fingertip retargeting.

A policy trained on the human hand action (``action.space`` ``hand_mano``) needs a *human* hand
state as proprioception, which a robot does not have. :class:`RobotToManoEstimator` estimates the
MANO finger pose that best reproduces the robot's current fingertip geometry — the same
fingertip-vector objective as :class:`robot_skin.action.FingertipRetargeter` (DexPilot, Handa et
al., arXiv:1910.03135; AnyTeleop, Qin et al., arXiv:2307.04577) with the roles swapped: the MANO
hand (``pose.mano.ManoSkeleton``, parametrised by 15 flexion + 5 abduction angles through
``ManoSkeleton.flexion_pose`` so the solution stays anatomical) is the "robot" and the robot's
fingertip vectors, rotated into the MANO frame and divided by the size ratio, are the targets.

Use it as ``PolicyRunner(hand_state_fn=RobotToManoEstimator(forward_retargeter))``: the wrist part
of the state is taken from the commanded hand action, the finger part from the estimate.

:class:`RobotToManoTaxelFrame` applies the same inverse map to the robot's **taxel poses**: a policy
whose tactile encoder was trained on glove episodes (``taxel_frame = mano_wrist``: fingers −x, palm
−y) must see the robot skin's URDF-root-frame poses re-expressed in that MANO wrist frame
(``PolicyRunner`` does it automatically for such bundles).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["RobotToManoEstimator", "RobotToManoTaxelFrame", "mano_tip_fk", "FLEX_LIMITS", "ABD_LIMITS"]

FLEX_LIMITS = (-0.3, 1.8)       # rad, per MANO finger joint (curl toward the palm positive)
ABD_LIMITS = (-0.5, 0.5)        # rad, per finger (dorsal-axis rotation at the first joint)


def mano_tip_fk(skeleton: Any = None):
    """``x[B,20]`` (15 flexions in MANO joint order + 5 abductions in FINGERS order) →
    ``{finger: tip[B,3]}`` in the MANO wrist frame (differentiable)."""
    from ..pose.mano import FINGERS, ManoSkeleton

    sk = skeleton if skeleton is not None else ManoSkeleton()

    def fk(x):
        fp = sk.flexion_pose(x[..., :15], x[..., 15:20])
        tips = sk.forward(finger_pose=fp)["tip_pos"]
        return {f: tips[..., i, :] for i, f in enumerate(FINGERS)}

    return fk


class RobotToManoEstimator:
    """Estimate the MANO finger pose ``[15,3]`` matching a robot configuration (module docstring).

    Args:
        retargeter: the forward :class:`~robot_skin.action.FingertipRetargeter` (human → robot) of
            the deployment — its robot FK, finger mapping, ``scale`` and ``human_to_robot`` define
            the correspondence that is inverted here.
        skeleton: MANO skeleton (default ``ManoSkeleton()``).
        iters / method / jacobian: solver settings of the inner retargeter (warm-started per call;
            ``fd`` = one batched MANO FK per LM iteration, ≈ 30 ms per call on a CPU core).
    """

    def __init__(self, retargeter: Any, skeleton: Any = None, *, iters: int = 10, method: str = "lm",
                 jacobian: str = "fd", reg_weight: float = 1e-5, smooth_weight: float = 1e-5):
        from ..action.retarget import FingertipRetargeter
        from ..pose.mano import FINGERS, ManoSkeleton

        self.fwd = retargeter
        self.skeleton = skeleton if skeleton is not None else ManoSkeleton()
        fingers = list(retargeter.fingers)
        lo = np.array([FLEX_LIMITS[0]] * 15 + [ABD_LIMITS[0]] * 5)
        hi = np.array([FLEX_LIMITS[1]] * 15 + [ABD_LIMITS[1]] * 5)
        self.inv = FingertipRetargeter(mano_tip_fk(self.skeleton), {f: f for f in fingers}, lo, hi,
                                       human_tip_names=tuple(FINGERS), scale=1.0, reg_weight=reg_weight,
                                       smooth_weight=smooth_weight, iters=iters, method=method, jacobian=jacobian,
                                       vectors=retargeter.vectors, q_nominal=np.zeros(20))
        self._fingers = fingers
        self._all = tuple(FINGERS)
        self.last_params: np.ndarray | None = None

    def robot_tips_in_mano(self, q_robot: Sequence[float]) -> np.ndarray:
        """Robot fingertips (wrist-relative, base-link frame) → MANO frame / scale ``[5,3]`` (fingers
        the robot lacks get the flat-hand MANO tip, which the vectors never use)."""
        import torch

        q = torch.as_tensor(np.asarray(q_robot, np.float64).reshape(1, -1), dtype=self.fwd.dtype)
        with torch.no_grad():
            pts = self.fwd.robot_points(q)
        R = self.fwd.human_to_robot
        out = self.skeleton.forward()["tip_pos"].detach().cpu().numpy().astype(np.float64).copy()
        for i, f in enumerate(self._all):
            if f in pts and f in self._fingers:
                out[i] = (R.T @ pts[f][0].cpu().numpy()) / self.fwd.scale
        return out

    def estimate(self, q_robot: Sequence[float]) -> np.ndarray:
        """MANO ``finger_pose [15,3]`` for robot joints ``q_robot`` (retargeter joint order)."""
        import torch

        tips = self.robot_tips_in_mano(q_robot)
        x = np.asarray(self.inv.step(tips), np.float64).reshape(-1)
        self.last_params = x
        with torch.no_grad():
            fp = self.skeleton.flexion_pose(torch.as_tensor(x[:15]), torch.as_tensor(x[15:20]))
        return fp.cpu().numpy().astype(np.float32)

    def reset(self) -> None:
        self.inv.reset()

    def __call__(self, q_robot: Sequence[float], commanded: Sequence[float] | None = None) -> np.ndarray:
        """``hand_state_fn`` for :class:`~robot_skin.control.runner.PolicyRunner`: the commanded hand
        action ``[54]`` with its finger part replaced by the estimate (the wrist is not observable
        from a hand-only robot)."""
        from ..action.space import FINGERS_AA, HAND_MANO_DIM

        a = np.zeros(HAND_MANO_DIM, np.float32) if commanded is None else np.asarray(commanded, np.float32).copy()
        a[FINGERS_AA] = self.estimate(q_robot).reshape(-1)
        return a


class RobotToManoTaxelFrame:
    """Robot taxel poses (URDF root frame, :func:`robot_skin.control.online.robot_pose_fn`) → the MANO
    wrist frame of glove training data, by inverting the forward retargeter's human → robot map
    ``v_r = s · R_hr · v_h`` (``v`` relative to the human wrist / the robot ``base_link``)::

        pos_h = R_hrᵀ · R_bᵀ (pos − t_b) / s,     nrm_h = R_hrᵀ · R_bᵀ · nrm

    with ``(R_b, t_b)`` the base link pose at ``q`` (:meth:`FingertipRetargeter.base_transform`;
    identity without a base link), ``R_hr = human_to_robot`` and ``s = scale`` — the same
    correspondence :meth:`RobotToManoEstimator.robot_tips_in_mano` uses for the fingertips, so a
    robot fingertip pad lands near the human fingertip pad the policy was trained on. The quality
    of the map is that of ``human_to_robot`` / ``scale`` (and of the two hands' similarity).

    ``__call__(pos[N,3], nrm[N,3], q)`` with ``q`` in the retargeter's joint order (only needed
    with a moving base link) → ``(pos, nrm)`` float32.
    """

    frame = "mano_wrist"

    def __init__(self, retargeter: Any):
        self.fwd = retargeter

    def __call__(self, pos: np.ndarray, nrm: np.ndarray, q: Sequence[float] | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(pos, np.float64).reshape(-1, 3)
        n = np.asarray(nrm, np.float64).reshape(-1, 3)
        if getattr(self.fwd, "base_link", None):
            if q is None:
                raise ValueError("the retargeter has a base_link: pass q (retargeter joint order)")
            T = self.fwd.base_transform(q)
            p, n = (p - T[:3, 3]) @ T[:3, :3], n @ T[:3, :3]            # row form of R_bᵀ(p − t_b), R_bᵀ n
        R = np.asarray(self.fwd.human_to_robot, np.float64)
        return (p @ R / float(self.fwd.scale)).astype(np.float32), (n @ R).astype(np.float32)

    def __repr__(self) -> str:
        return (f"RobotToManoTaxelFrame(base_link={getattr(self.fwd, 'base_link', None)!r}, "
                f"scale={float(self.fwd.scale):.3f})")
