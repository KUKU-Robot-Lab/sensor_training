import math
import warnings

import numpy as np
import pytest
import torch

from common.layouts import layout_from_dict, load_layout
from robot_skin.pose import TaxelPoseProvider, transform_taxels
from robot_skin.pose.robot_fk import RobotFKPoseProvider, taxel_poses_from_joints
from robot_skin.pose.urdf import URDFModel

TOY_URDF = """<?xml version="1.0"?>
<robot name="toy_hand">
  <link name="base"/>
  <link name="palm_link"><visual><geometry><box size="0.1 0.1 0.02"/></geometry></visual></link>
  <link name="finger_prox"/>
  <link name="finger_dist"/>
  <link name="slider"/>
  <link name="tip"/>
  <link name="mimic_link"/>
  <joint name="base_to_palm" type="fixed">
    <parent link="base"/><child link="palm_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 1.5707963267948966"/>
  </joint>
  <joint name="mcp" type="revolute">
    <parent link="palm_link"/><child link="finger_prox"/>
    <origin xyz="0.01 0 0.08" rpy="0.1 0.2 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="1.6" effort="1" velocity="2"/>
  </joint>
  <joint name="pip" type="continuous">
    <parent link="finger_prox"/><child link="finger_dist"/>
    <origin xyz="0 0 0.04"/>
    <axis xyz="1 0 0"/>
  </joint>
  <joint name="slide" type="prismatic">
    <parent link="palm_link"/><child link="slider"/>
    <origin xyz="0 0.02 0" rpy="0 1.0 0"/>
    <axis xyz="0 0 2"/>
    <limit lower="0" upper="0.03" effort="1" velocity="0.1"/>
  </joint>
  <joint name="dist_to_tip" type="fixed">
    <parent link="finger_dist"/><child link="tip"/>
    <origin xyz="0 0 0.03"/>
  </joint>
  <joint name="dip_mimic" type="revolute">
    <parent link="tip"/><child link="mimic_link"/>
    <origin xyz="0 0 0.01"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
    <mimic joint="pip" multiplier="0.5" offset="0.1"/>
  </joint>
</robot>
"""


# independent numpy reference ------------------------------------------------
def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _T(R=None, p=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = (np.eye(3) if R is None else R), p
    return T


def _ref_fk(q):
    rpy = _rz(0.3) @ _ry(0.2) @ _rx(0.1)
    palm = _T(_rz(math.pi / 2), (0, 0, 0.05))
    prox = palm @ _T(rpy, (0.01, 0, 0.08)) @ _T(_ry(q[0]))
    dist = prox @ _T(p=(0, 0, 0.04)) @ _T(_rx(q[1]))
    slider = palm @ _T(_ry(1.0), (0, 0.02, 0)) @ _T(p=(0, 0, q[2]))
    tip = dist @ _T(p=(0, 0, 0.03))
    mimic = tip @ _T(p=(0, 0, 0.01)) @ _T(_rx(0.5 * q[1] + 0.1))
    return {"base": np.eye(4), "palm_link": palm, "finger_prox": prox, "finger_dist": dist,
            "slider": slider, "tip": tip, "mimic_link": mimic}


def test_parse_structure_and_limits():
    m = URDFModel.from_string(TOY_URDF)
    assert m.name == "toy_hand" and m.root_link == "base"
    assert m.joint_names == ("mcp", "pip", "slide") and m.n_dof == 3
    assert set(m.link_names) == {"base", "palm_link", "finger_prox", "finger_dist", "slider", "tip", "mimic_link"}
    np.testing.assert_allclose(m.lower, [-0.5, -np.inf, 0.0])
    np.testing.assert_allclose(m.upper, [1.6, np.inf, 0.03])
    np.testing.assert_allclose(m.velocity_limits, [2.0, np.inf, 0.1])
    np.testing.assert_allclose(m.joint("slide").axis, [0, 0, 1])          # normalised
    assert m.joint("dip_mimic").mimic == ("pip", 0.5, 0.1)
    assert [j.name for j in m.chain("mimic_link")] == ["base_to_palm", "mcp", "pip", "dist_to_tip", "dip_mimic"]
    np.testing.assert_allclose(m.clamp(np.array([2.0, 5.0, -1.0])), [1.6, 5.0, 0.0])


def test_fk_matches_hand_computed_transforms():
    m = URDFModel.from_string(TOY_URDF)
    rng = np.random.default_rng(0)
    Q = rng.uniform(-1, 1, size=(6, 3)) * [1.0, 2.0, 0.03]
    out = m.fk_numpy(Q)
    assert set(out) == set(m.link_names)
    for i, q in enumerate(Q):
        ref = _ref_fk(q)
        for link, T in ref.items():
            np.testing.assert_allclose(out[link][i], T, atol=1e-12, err_msg=link)
    # literal check: joint origin rotated by the fixed palm yaw → prox origin at (0, 0.01, 0.13)
    np.testing.assert_allclose(out["finger_prox"][:, :3, 3], np.tile([0.0, 0.01, 0.13], (6, 1)), atol=1e-12)
    # prismatic: slider moves by q along its (rotated) axis
    d = out["slider"][:, :3, 3] - _ref_fk(np.zeros(3))["slider"][:3, 3]
    np.testing.assert_allclose(np.linalg.norm(d, axis=-1), np.abs(Q[:, 2]), atol=1e-12)


def test_fk_batch_shapes_links_subset_and_base():
    m = URDFModel.from_string(TOY_URDF)
    q = torch.zeros(2, 5, 3)
    out = m.fk(q, links=["tip"])
    assert list(out) == ["tip"] and out["tip"].shape == (2, 5, 4, 4) and out["tip"].dtype == torch.float32
    base = _T(_rz(0.4), (1.0, 2.0, 3.0))
    o1 = m.fk_numpy(np.zeros(3), base_T=base)
    np.testing.assert_allclose(o1["tip"], base @ _ref_fk(np.zeros(3))["tip"], atol=1e-12)
    with pytest.raises(ValueError):
        m.fk(torch.zeros(4))


def test_fk_gradients_flow():
    m = URDFModel.from_string(TOY_URDF)
    q = torch.tensor([0.0, 0.0, 0.01], dtype=torch.float64, requires_grad=True)
    p = m.fk(q)["mimic_link"][:3, 3]
    p.sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad[0].abs() > 0 and q.grad[1].abs() > 0
    assert q.grad[2] == 0                                     # slider is not on the mimic chain
    q2 = torch.tensor([0.3, -0.7, 0.01], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: m.fk(x)["mimic_link"][:3, 3], (q2,))


def test_mimic_ignore_and_parse_errors(tmp_path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = URDFModel.from_string(TOY_URDF, mimic="ignore")
    assert any("mimic" in str(x.message) for x in w)
    q = np.array([0.2, 0.4, 0.0])
    T = m.fk_numpy(q)["mimic_link"]
    ref = _ref_fk(q)["tip"] @ _T(p=(0, 0, 0.01))                    # held at 0
    np.testing.assert_allclose(T, ref, atol=1e-12)

    p = tmp_path / "toy.urdf"
    p.write_text(TOY_URDF)
    assert URDFModel.from_file(p).n_dof == 3
    with pytest.raises(FileNotFoundError):
        URDFModel.from_file(tmp_path / "nope.urdf")
    with pytest.raises(ValueError):
        URDFModel.from_string("<robot><link name='a'/><link name='b'/></robot>")      # two roots
    with pytest.raises(ValueError):
        URDFModel.from_string("<xacro/>")
    with pytest.raises(ValueError):
        URDFModel.from_string("<robot>")
    bad = TOY_URDF.replace('type="continuous"', 'type="spherical"')
    with pytest.raises(ValueError):
        URDFModel.from_string(bad)
    zero_axis = TOY_URDF.replace('<axis xyz="1 0 0"/>\n  </joint>\n  <joint name="slide"',
                                 '<axis xyz="0 0 0"/>\n  </joint>\n  <joint name="slide"')
    assert zero_axis != TOY_URDF
    with pytest.raises(ValueError, match="zero axis"):
        URDFModel.from_string(zero_axis)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mf = URDFModel.from_string(TOY_URDF.replace('type="prismatic"', 'type="floating"'))
    assert mf.joint_names == ("mcp", "pip") and any("floating" in str(x.message) for x in w)


def test_reorder_q():
    m = URDFModel.from_string(TOY_URDF)
    q = np.array([[3.0, 1.0, 9.0, 2.0]])
    out = m.reorder_q(q, ["slide", "mcp", "dip_mimic", "pip"])
    np.testing.assert_allclose(out, [[1.0, 2.0, 3.0]])
    with pytest.warns(UserWarning):
        out = m.reorder_q(np.array([1.0, 2.0]), ["mcp", "pip"])
    np.testing.assert_allclose(out, [1.0, 2.0, 0.0])


def _toy_layout():
    return layout_from_dict({
        "name": "toy", "units": "mm", "parent_frame": "urdf",
        "taxels": [
            {"id": "tip", "channel": 0, "parent": "tip", "position": [0, 5, 3], "normal": [0, 1, 0]},
            {"id": "prox", "channel": 1, "parent": "finger_prox", "position": [0, 5, 20], "normal": [0, 1, 0.2]},
            {"id": "palm", "channel": 2, "parent": "palm_link", "position": [10, 0, 40], "normal": [1, 0, 0]},
        ]})


def test_robot_fk_provider_and_batched_poses(tmp_path):
    m = URDFModel.from_string(TOY_URDF)
    L = _toy_layout()
    Q = np.random.default_rng(1).uniform(-1, 1, size=(4, 3)) * [1.0, 2.0, 0.03]
    prov = RobotFKPoseProvider(L, m, lambda t: Q[int(t)])
    assert isinstance(prov, TaxelPoseProvider) and prov.n_taxels == 3
    P, N = taxel_poses_from_joints(L, m, Q, chunk=3)
    assert P.shape == N.shape == (4, 3, 3)
    for i in range(4):
        pos, nrm = prov.pose_at(float(i))
        ref = transform_taxels(L, _ref_fk(Q[i]))
        np.testing.assert_allclose(pos, ref[0], atol=1e-12)
        np.testing.assert_allclose(nrm, ref[1], atol=1e-12)
        np.testing.assert_allclose(P[i], pos, atol=1e-12)
        np.testing.assert_allclose(N[i], nrm, atol=1e-12)
    # path input + validation
    p = tmp_path / "toy.urdf"
    p.write_text(TOY_URDF)
    assert RobotFKPoseProvider(L, p, lambda t: Q[0]).pose_at(0.0)[0].shape == (3, 3)
    with pytest.raises(ValueError):
        RobotFKPoseProvider(load_layout("robot_hand_template"), m, lambda t: Q[0])   # unknown links
    with pytest.raises(ValueError):
        RobotFKPoseProvider(L, m, lambda t: np.zeros(2)).pose_at(0.0)
    with pytest.raises(ValueError):
        taxel_poses_from_joints(L, m, np.zeros((4, 2)))


PLANAR = """<robot name="planar"><link name="b"/><link name="l1"/><link name="l2"/><link name="tip"/>
<joint name="j1" type="revolute"><parent link="b"/><child link="l1"/><axis xyz="0 0 1"/>
  <limit lower="-3" upper="3"/></joint>
<joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.05 0 0"/><axis xyz="0 0 1"/>
  <limit lower="-3" upper="3"/></joint>
<joint name="jt" type="fixed"><parent link="l2"/><child link="tip"/><origin xyz="0.03 0 0"/></joint></robot>"""


def test_planar_two_link_closed_form():
    """tip = (l1·cos q1 + l2·cos(q1+q2), l1·sin q1 + l2·sin(q1+q2), 0), l1 = 0.05, l2 = 0.03."""
    m = URDFModel.from_string(PLANAR)
    Q = np.array([[0.4, 0.9], [-1.2, 2.5], [0.0, 0.0]])
    p = m.fk_numpy(Q, links=["tip"])["tip"][:, :3, 3]
    ref = np.stack([0.05 * np.cos(Q[:, 0]) + 0.03 * np.cos(Q.sum(1)),
                    0.05 * np.sin(Q[:, 0]) + 0.03 * np.sin(Q.sum(1)), np.zeros(3)], -1)
    np.testing.assert_allclose(p, ref, atol=1e-12)


def test_base_T_broadcasts_and_chunk_edge_cases():
    m = URDFModel.from_string(TOY_URDF)
    bases = np.stack([_T(_rz(a), (a, 0.0, 0.0)) for a in (0.0, 0.5, 1.0)])
    q = np.array([0.2, -0.3, 0.01])
    out = m.fk_numpy(q, base_T=bases, links=["tip"])["tip"]                  # [3] bases × one q
    assert out.shape == (3, 4, 4)
    for i in range(3):
        np.testing.assert_allclose(out[i], bases[i] @ _ref_fk(q)["tip"], atol=1e-12)
    with pytest.raises(ValueError):
        m.fk(q, base_T=np.eye(3))
    # chunk ≤ 0 must not leave the output uninitialised
    L = _toy_layout()
    Q = np.random.default_rng(3).uniform(-1, 1, size=(5, 3)) * [1.0, 2.0, 0.03]
    P0, N0 = taxel_poses_from_joints(L, m, Q, chunk=0)
    P, N = taxel_poses_from_joints(L, m, Q)
    np.testing.assert_allclose(P0, P, atol=1e-12)
    np.testing.assert_allclose(N0, N, atol=1e-12)
    with pytest.raises(ValueError):
        taxel_poses_from_joints(L, m, Q, base_T=np.eye(3))
