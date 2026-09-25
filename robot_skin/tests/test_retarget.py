import math

import numpy as np
import pytest
import torch

from robot_skin.action import (FingertipRetargeter, hand_action_fingertips, hand_action_from_arrays,
                               human_fingertips)
from robot_skin.geometry import rotations as rot

# ── toy robot: two planar 2-link fingers in the xy-plane of the palm frame ──────
L1, L2 = 0.04, 0.03
BASES = {"thumb": (0.0, -0.02), "index": (0.0, 0.02)}
LO = np.array([-1.0, 0.0, -1.0, 0.0])
HI = np.array([1.0, 1.5, 1.0, 1.5])
TIPS = {"thumb": "thumb_tip", "index": "index_tip"}
HUMAN = ("thumb", "index")


def toy_fk(q):
    """q[B,4] → {"thumb_tip": [B,3], "index_tip": [B,3]} (differentiable, batched)."""
    out = {}
    for i, f in enumerate(("thumb", "index")):
        a, b = q[..., 2 * i], q[..., 2 * i + 1]
        bx, by = BASES[f]
        x = bx + L1 * torch.cos(a) + L2 * torch.cos(a + b)
        y = by + L1 * torch.sin(a) + L2 * torch.sin(a + b)
        out[f"{f}_tip"] = torch.stack([x, y, torch.zeros_like(x)], -1)
    return out


def toy_tips(q) -> np.ndarray:
    o = toy_fk(torch.as_tensor(np.asarray(q, dtype=np.float64)))
    return torch.stack([o["thumb_tip"], o["index_tip"]], -2).numpy()


def make(**kw):
    kw.setdefault("reg_weight", 1e-12)
    return FingertipRetargeter(toy_fk, TIPS, lower=LO, upper=HI, human_tip_names=HUMAN, **kw)


def _rand_q(n, seed=0, margin=0.1):
    rng = np.random.default_rng(seed)
    return rng.uniform(LO + margin, HI - margin, size=(n, 4))


# ── recovery / limits ─────────────────────────────────────────────────────────
def test_recovers_joint_angles_from_fingertip_vectors():
    rt = make()
    assert rt.dof == 4 and rt.fingers == HUMAN
    assert rt.vectors == (("wrist", "thumb"), ("wrist", "index"), ("thumb", "index"))
    for q_true in _rand_q(5, seed=1):
        q = rt.retarget(toy_tips(q_true))
        assert q.shape == (4,) and isinstance(q, np.ndarray)
        np.testing.assert_allclose(q, q_true, atol=1e-5)
        assert rt.vector_error(q, toy_tips(q_true)).max() < 1e-7


def test_batched_retarget_matches_single_and_torch_io():
    rt = make()
    qs = _rand_q(6, seed=2)
    tips = toy_tips(qs).reshape(2, 3, 2, 3)                      # leading dims [2,3]
    qb = rt.retarget(tips)
    assert qb.shape == (2, 3, 4)
    np.testing.assert_allclose(qb.reshape(6, 4), qs, atol=1e-5)
    qt = rt.retarget(torch.as_tensor(toy_tips(qs[0])))
    assert isinstance(qt, torch.Tensor)
    np.testing.assert_allclose(qt.numpy(), qs[0], atol=1e-5)


def test_respects_joint_limits():
    rt = make()
    # thumb PIP beyond its limit → tip closer to the base than any feasible pose; index MCP beyond its
    # limit with a small PIP (the mirrored IK branch would need PIP < 0) → both end on the bound
    q_out = np.array([0.3, 2.2, 1.4, 0.2])
    q = rt.retarget(toy_tips(q_out))
    assert np.all(q >= LO - 1e-12) and np.all(q <= HI + 1e-12)
    assert q[1] == pytest.approx(HI[1]) and q[2] == pytest.approx(HI[2])
    # the best feasible fit beats the naive clamp of the true configuration
    tips = toy_tips(q_out)
    naive = np.clip(q_out, LO, HI)
    assert rt.vector_error(q, tips).sum() <= rt.vector_error(naive, tips).sum() + 1e-9
    # every method keeps the solution inside the limits
    for m in ("adam", "lbfgs"):
        qm = make(method=m).retarget(tips)
        assert np.all(qm >= LO - 1e-12) and np.all(qm <= HI + 1e-12)


def test_other_solvers_converge():
    q_true = np.array([0.3, 0.8, -0.2, 0.5])
    tips = toy_tips(q_true)
    np.testing.assert_allclose(make(method="lbfgs").retarget(tips), q_true, atol=1e-4)
    np.testing.assert_allclose(make(method="adam", iters=400, lr=0.02).retarget(tips), q_true, atol=3e-2)


def test_regularisation_resolves_redundancy():
    # only the index fingertip is a target: the thumb joints do not affect the cost (null space) and
    # stay at q_nominal thanks to the regulariser, while the index joints are recovered
    rt = FingertipRetargeter(toy_fk, {"index": "index_tip"}, lower=LO, upper=HI, human_tip_names=("index",),
                             vectors="tips", reg_weight=1e-9, q_nominal=[0.2, 0.3, 0.0, 0.75])
    q_true = np.array([0.9, 1.1, -0.3, 0.7])
    q = rt.retarget(toy_tips(q_true)[1:])
    np.testing.assert_allclose(q[:2], [0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(q[2:], q_true[2:], atol=1e-3)


# ── sequences / streaming ─────────────────────────────────────────────────────
def _trajectory(T=30):
    s = np.linspace(0, 1, T)[:, None]
    q0, q1 = np.array([-0.5, 0.2, 0.6, 0.3]), np.array([0.6, 1.2, -0.4, 1.1])
    w = 0.5 - 0.5 * np.cos(math.pi * s)                          # smooth ease-in/out
    return q0 + w * (q1 - q0)


def test_retarget_sequence_warm_start_is_accurate_and_smooth():
    q_true = _trajectory()
    tips = toy_tips(q_true)
    rt = make(reg_weight=1e-9, smooth_weight=1e-8)
    q = rt.retarget_sequence(tips)
    assert q.shape == q_true.shape
    np.testing.assert_allclose(q, q_true, atol=2e-3)
    dq, dq_true = np.abs(np.diff(q, axis=0)).max(), np.abs(np.diff(q_true, axis=0)).max()
    assert dq <= 1.2 * dq_true + 1e-3                            # no jumps between frames
    warm_iters = rt.last_info["iters"][1:].mean()
    cold = make(reg_weight=1e-9)
    q_cold = cold.retarget_sequence(tips, warm_start=False)      # one batch from q_nominal
    np.testing.assert_allclose(q_cold, q_true, atol=2e-3)
    cold_iters = []
    for t in range(1, len(tips)):
        cold.retarget(tips[t])
        cold_iters.append(cold.last_info["iters"])
    assert warm_iters < np.mean(cold_iters)                      # warm start converges faster


def test_smoothing_weight_damps_jitter():
    q_true = _trajectory(40)
    rng = np.random.default_rng(5)
    tips = toy_tips(q_true) + rng.normal(scale=1.5e-3, size=(40, 2, 3)) * np.array([1, 1, 0])
    rough = make(reg_weight=1e-9, smooth_weight=0.0).retarget_sequence(tips)
    smooth = make(reg_weight=1e-9, smooth_weight=3e-5).retarget_sequence(tips)
    jitter = lambda q: np.abs(np.diff(q, n=2, axis=0)).mean()
    assert jitter(smooth) < 0.7 * jitter(rough)


def test_step_streaming_equals_warm_sequence():
    tips = toy_tips(_trajectory(10))
    rt = make(smooth_weight=1e-6)
    seq = rt.retarget_sequence(tips)
    rt.reset()
    stream = np.stack([rt.step(tp) for tp in tips])
    np.testing.assert_allclose(stream, seq, atol=1e-10)
    rt.reset(q=seq[-2])                    # resume from a known robot state
    np.testing.assert_allclose(rt.step(tips[-1]), seq[-1], atol=1e-10)


# ── frames / scale / pinch ────────────────────────────────────────────────────
def test_wrist_transform_frame_rotation_and_scale():
    q_true = np.array([0.1, 0.9, -0.5, 0.4])
    local = toy_tips(q_true)                                     # robot palm frame
    R = rot.aa_to_matrix(torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64)).numpy()
    p = np.array([0.4, -0.2, 1.0])
    # human tips given in a world frame together with the wrist pose
    world = local @ R.T + p
    W = np.eye(4)
    W[:3, :3], W[:3, 3] = R, p
    np.testing.assert_allclose(make().retarget(world, human_wrist_T=W), q_true, atol=1e-5)
    # human wrist frame rotated w.r.t. the robot base frame
    Rhr = rot.aa_to_matrix(torch.tensor([0.0, 0.0, 0.8], dtype=torch.float64)).numpy()
    human_local = local @ Rhr                                    # = Rhrᵀ · robot vectors
    np.testing.assert_allclose(make(human_to_robot=Rhr).retarget(human_local), q_true, atol=1e-5)
    # human hand twice as large → scale 0.5
    big = 2.0 * local
    rt = make()
    s = rt.estimate_scale(big, q_ref=q_true)
    assert s == pytest.approx(0.5)
    np.testing.assert_allclose(make(scale=s).retarget(big), q_true, atol=1e-5)


def test_pinch_targets_close_the_gap():
    # a human pinch (thumb–index 8 mm apart); the robot is told it is 1.6× larger, which would open
    # the pinch to ~13 mm — pinch handling pulls the robot tips together instead
    tips_h = np.array([[0.050, -0.004, 0.0], [0.050, 0.004, 0.0]])
    plain = make(scale=1.6, reg_weight=1e-9)
    pinch = make(scale=1.6, reg_weight=1e-9, pinch_threshold=0.015, pinch_distance=0.002)

    def gap(rt):
        q = rt.retarget(tips_h)
        o = toy_fk(torch.as_tensor(q)[None])
        return float((o["thumb_tip"] - o["index_tip"]).norm())

    g_plain, g_pinch = gap(plain), gap(pinch)
    assert g_plain > 0.010
    assert g_pinch < 0.004 and g_pinch < g_plain


# ── model / duck typing ───────────────────────────────────────────────────────
class ToyModel:
    """URDF-model-like: fk(q, links=None) → {link: T[...,4,4]}, lower/upper, joint_names.
    Its palm link sits at an offset/rotation from the model root."""

    joint_names = ("t0", "t1", "i0", "i1")
    lower, upper = LO, HI

    def __init__(self):
        self.base_R = rot.aa_to_matrix(torch.tensor([0.2, 0.1, -0.4], dtype=torch.float64))
        self.base_p = torch.tensor([0.1, 0.05, -0.02], dtype=torch.float64)
        self.calls = []

    def fk(self, q, links=None):
        self.calls.append(None if links is None else tuple(links))
        q = torch.as_tensor(q)
        loc = toy_fk(q)
        B = q.shape[:-1]
        out = {"palm_link": rot.make_transform(self.base_R.expand(*B, 3, 3), self.base_p.expand(*B, 3))}
        for f, a in (("thumb", q[..., 0] + q[..., 1]), ("index", q[..., 2] + q[..., 3])):
            Rz = rot.aa_to_matrix(torch.stack([torch.zeros_like(a), torch.zeros_like(a), a], -1))
            p_world = (self.base_R @ loc[f"{f}_tip"][..., None])[..., 0] + self.base_p
            out[f"{f}_distal"] = rot.make_transform(self.base_R @ Rz, p_world)
        out["unused_link"] = out["palm_link"]
        return out if links is None else {k: out[k] for k in links}


def test_urdf_like_model_duck_typed_with_base_link():
    m = ToyModel()
    rt = FingertipRetargeter(m, {"thumb": "thumb_distal", "index": "index_distal"}, base_link="palm_link",
                             human_tip_names=HUMAN, reg_weight=1e-12)
    assert rt.dof == 4 and rt.joint_names == ToyModel.joint_names
    np.testing.assert_array_equal(rt.lower, LO)
    assert set(m.calls[-1]) == {"thumb_distal", "index_distal", "palm_link"}   # only needed links
    q_true = np.array([0.2, 0.7, -0.1, 1.0])
    # human tips are in the wrist frame ≙ robot palm frame → the base transform must be undone
    np.testing.assert_allclose(rt.retarget(toy_tips(q_true)), q_true, atol=1e-5)
    with pytest.raises(ValueError):
        FingertipRetargeter(m, None, human_tip_names=HUMAN)


TWO_FINGER_URDF = """<?xml version="1.0"?>
<robot name="two_finger">
  <link name="palm"/>
  <link name="th1"/><link name="th2"/><link name="th_tip"/>
  <link name="ix1"/><link name="ix2"/><link name="ix_tip"/>
  <joint name="th_mcp" type="revolute"><parent link="palm"/><child link="th1"/>
    <origin xyz="0 -0.02 0"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="th_pip" type="revolute"><parent link="th1"/><child link="th2"/>
    <origin xyz="0.04 0 0"/><axis xyz="0 0 1"/><limit lower="0" upper="1.5" effort="1" velocity="1"/></joint>
  <joint name="th_fix" type="fixed"><parent link="th2"/><child link="th_tip"/><origin xyz="0.03 0 0"/></joint>
  <joint name="ix_mcp" type="revolute"><parent link="palm"/><child link="ix1"/>
    <origin xyz="0 0.02 0"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="ix_pip" type="revolute"><parent link="ix1"/><child link="ix2"/>
    <origin xyz="0.04 0 0"/><axis xyz="0 0 1"/><limit lower="0" upper="1.5" effort="1" velocity="1"/></joint>
  <joint name="ix_fix" type="fixed"><parent link="ix2"/><child link="ix_tip"/><origin xyz="0.03 0 0"/></joint>
</robot>
"""


def test_real_urdf_model_if_available():
    urdf = pytest.importorskip("robot_skin.pose.urdf")
    model = urdf.URDFModel.from_string(TWO_FINGER_URDF)
    rt = FingertipRetargeter(model, {"thumb": "th_tip", "index": "ix_tip"}, base_link="palm",
                             human_tip_names=HUMAN, reg_weight=1e-12)
    np.testing.assert_allclose(rt.lower, LO)
    q_true = np.array([0.3, 0.8, -0.2, 0.5])                    # same kinematics as toy_fk
    np.testing.assert_allclose(rt.retarget(toy_tips(q_true)), q_true, atol=1e-5)


# ── validation ────────────────────────────────────────────────────────────────
def test_constructor_validation():
    with pytest.raises(KeyError):
        FingertipRetargeter(toy_fk, {"thumb": "nope"}, lower=LO, upper=HI, human_tip_names=HUMAN)
    with pytest.raises(ValueError):
        FingertipRetargeter(toy_fk, {"pinky": "thumb_tip"}, lower=LO, upper=HI, human_tip_names=HUMAN)
    with pytest.raises(ValueError):
        FingertipRetargeter(toy_fk, TIPS, human_tip_names=HUMAN)            # dof unknown
    with pytest.raises(ValueError):
        FingertipRetargeter(toy_fk, TIPS, lower=LO, upper=HI, human_tip_names=HUMAN, vectors=[("thumb", "thumb")])
    with pytest.raises(ValueError):
        FingertipRetargeter(toy_fk, TIPS, lower=LO, upper=HI, human_tip_names=HUMAN, method="sqp")
    with pytest.raises(ValueError):
        FingertipRetargeter(toy_fk, TIPS, lower=LO, upper=HI, human_tip_names=HUMAN, human_to_robot=2 * np.eye(3))
    # tip_links=None → fingers found among the FK output keys (callable fk only)
    rt = FingertipRetargeter(lambda q: {"thumb": toy_fk(q)["thumb_tip"], "index": toy_fk(q)["index_tip"]},
                             None, dof=4, human_tip_names=HUMAN)
    assert rt.fingers == HUMAN and np.isinf(rt.lower).all()
    with pytest.raises(ValueError):
        make().retarget(np.zeros((5, 3)))                                    # 5 tips for 2 human names


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "big"])
def test_scale_is_validated_on_assignment_too(bad):
    """``scale`` must be finite and > 0 whether passed to the constructor or assigned later (the
    deploy stage sets it after building the retargeter): 0 collapses every target, < 0 mirrors it."""
    with pytest.raises(ValueError, match="scale"):
        make(scale=bad)
    rt = make(scale=1.5)
    with pytest.raises(ValueError, match="scale"):
        rt.scale = bad
    assert rt.scale == 1.5                                                   # unchanged after the rejection
    rt.scale = 2                                                             # ints are fine
    assert rt.scale == 2.0 and isinstance(rt.scale, float)


def test_spec_positional_order_and_input_validation():
    # spec order: fk, tip_links, lower, upper, human_tip_names, scale, reg_weight, smooth_weight, iters, lr
    rt = FingertipRetargeter(toy_fk, TIPS, LO, HI, HUMAN, 1.0, 1e-12, 1e-6, 40, 0.02)
    assert rt.dof == 4 and rt.iters == 40 and rt.scale == 1.0 and rt.reg_weight == 1e-12
    q_true = np.array([0.3, 0.8, -0.2, 0.5])
    np.testing.assert_allclose(rt.retarget(toy_tips(q_true)), q_true, atol=1e-5)
    # non-finite targets must fail loudly for every solver (adam / lbfgs would return NaN joints)
    bad = toy_tips(q_true)
    bad[0, 1] = np.nan
    for m in ("lm", "adam", "lbfgs"):
        with pytest.raises(ValueError):
            make(method=m).retarget(bad)
    W = np.eye(4)
    W[0, 3] = np.inf
    with pytest.raises(ValueError):
        rt.retarget(toy_tips(q_true), human_wrist_T=W)
    with pytest.raises(ValueError):
        rt.retarget(toy_tips(q_true), q_init=np.full(4, np.nan))
    # q_init may carry the same leading dims as the targets
    qs = _rand_q(6, seed=3)
    q = rt.retarget(toy_tips(qs).reshape(2, 3, 2, 3), q_init=qs.reshape(2, 3, 4))
    assert q.shape == (2, 3, 4)
    np.testing.assert_allclose(q.reshape(6, 4), qs, atol=1e-6)
    with pytest.raises(ValueError):
        rt.retarget(toy_tips(qs).reshape(2, 3, 2, 3), q_init=qs[:4])


def test_hand_derived_planar_fingertip():
    # thumb MCP 0, PIP 1.2 rad: tip = base + (L1 + L2·cos 1.2, L2·sin 1.2); index straight at q = 0
    q = np.array([0.0, 1.2, 0.0, 0.0])
    tips = toy_tips(q)
    np.testing.assert_allclose(tips[0], [L1 + L2 * math.cos(1.2), -0.02 + L2 * math.sin(1.2), 0.0], atol=1e-12)
    np.testing.assert_allclose(tips[1], [L1 + L2, 0.02, 0.0], atol=1e-12)
    rt = make()
    q_hat = rt.retarget(tips)
    np.testing.assert_allclose(q_hat[:2], q[:2], atol=1e-5)
    # the straight index finger sits on a singular, limit-touching pose: the angles are only
    # determined to second order, but the fingertip vectors are matched to the micrometre
    assert rt.vector_error(q_hat, tips).max() < 1e-5
    assert np.all(q_hat >= LO) and np.all(q_hat <= HI)


# ── MANO helpers ──────────────────────────────────────────────────────────────
def test_human_fingertips_from_mano():
    mano = pytest.importorskip("robot_skin.pose.mano")
    skel = mano.ManoSkeleton()
    flat = human_fingertips(np.zeros((15, 3)), skel)
    assert flat.shape == (5, 3)
    np.testing.assert_allclose(flat, skel.forward()["tip_pos"].numpy(), atol=1e-9)
    rng = np.random.default_rng(0)
    fp = rng.normal(scale=0.3, size=(4, 15, 3))
    go = rng.normal(size=(4, 3))
    wp = rng.normal(size=(4, 3))
    a = hand_action_from_arrays(go, fp, wp)
    # wrist pose does not matter: tips are in the wrist frame
    np.testing.assert_allclose(hand_action_fingertips(a, skel), human_fingertips(fp, skel), atol=1e-5)
    # default human_tip_names order is MANO FINGERS order
    rt = FingertipRetargeter(toy_fk, TIPS, lower=LO, upper=HI)
    assert rt.human_tip_names == tuple(mano.FINGERS)
    q = rt.retarget(human_fingertips(fp[0], skel))
    assert q.shape == (4,) and np.all(q >= LO) and np.all(q <= HI)


# ── Jacobian modes / config ───────────────────────────────────────────────────
def test_fd_jacobian_matches_autograd():
    a, f = make(smooth_weight=1e-4), make(jacobian="fd", smooth_weight=1e-4)
    q = torch.as_tensor(_rand_q(3, seed=7))
    q_prev = torch.as_tensor(_rand_q(3, seed=9))
    target, w = a._targets(a.human_vectors(toy_tips(_rand_q(3, seed=8))))
    Ja = a._jacobian(q, target, w.sqrt(), q_prev)
    Jf = f._jacobian(q, target, w.sqrt(), q_prev)
    assert Ja.shape == (3, 3 * 3 + 4 + 4, 4)                     # V·3 vector rows + reg + smooth
    torch.testing.assert_close(Jf, Ja, atol=1e-8, rtol=0)
    q_true = np.array([0.3, 0.8, -0.2, 0.5])
    np.testing.assert_allclose(f.retarget(toy_tips(q_true)), q_true, atol=1e-5)
    with pytest.raises(ValueError):
        make(jacobian="numeric")


def test_from_config():
    cfg = {"tip_links": TIPS, "lower": LO.tolist(), "upper": HI.tolist(), "human_tip_names": list(HUMAN),
           "vectors": [["wrist", "thumb"], ["wrist", "index"]], "scale": 1.0, "reg_weight": 1e-12,
           "method": "lm", "iters": 60}
    rt = FingertipRetargeter.from_config(toy_fk, cfg)
    assert rt.vectors == (("wrist", "thumb"), ("wrist", "index")) and rt.iters == 60
    q_true = np.array([0.3, 0.8, -0.2, 0.5])
    np.testing.assert_allclose(rt.retarget(toy_tips(q_true)), q_true, atol=1e-5)
    with pytest.raises(ValueError):
        FingertipRetargeter.from_config(toy_fk, {**cfg, "scael": 1.2})
