import numpy as np
import pytest
import torch

from common.layouts import MANO_SEGMENTS, grid_layout, load_layout
from robot_skin.geometry.rotations import aa_to_matrix
from robot_skin.pose import TaxelPoseProvider, transform_taxels
from robot_skin.pose.mano import (
    CAPSULE_NAMES, FINGER_JOINTS, FINGERS, MANO_JOINTS, MANO_PARENTS, SEGMENT_TO_JOINT,
    ManoPoseProvider, ManoSkeleton, self_touch_from_hand, taxel_poses_from_hand,
)


def _rand_pose(T, seed, scale=0.4):
    g = np.random.default_rng(seed)
    return g.normal(size=(T, 3)), g.normal(size=(T, 15, 3)) * scale, g.normal(size=(T, 3)) * 0.1


def _descendants(k):
    out, frontier = set(), {k}
    while frontier:
        frontier = {j for j, p in enumerate(MANO_PARENTS) if p in frontier}
        out |= frontier
    return out


def test_tree_constants():
    assert len(MANO_JOINTS) == 16 and MANO_PARENTS[0] == -1
    assert all(MANO_PARENTS[k] < k for k in range(1, 16))
    assert set(SEGMENT_TO_JOINT) == set(MANO_SEGMENTS)
    assert SEGMENT_TO_JOINT["wrist"] == SEGMENT_TO_JOINT["palm"] == 0
    for f, js in FINGER_JOINTS.items():
        assert [MANO_JOINTS[j] for j in js] == [f"{f}1", f"{f}2", f"{f}3"]


def test_zero_pose_reproduces_rest_joints():
    sk = ManoSkeleton()
    p = np.array([0.1, -0.2, 0.3])
    fk = sk.forward(np.zeros(3), np.zeros((15, 3)), p)
    np.testing.assert_allclose(fk["joint_pos"].numpy(), sk.rest_joints + p, atol=1e-12)
    np.testing.assert_allclose(fk["joint_rot"].numpy(), np.broadcast_to(np.eye(3), (16, 3, 3)), atol=1e-12)
    tips = np.stack([sk.rest_joints[FINGER_JOINTS[f][2]] for f in FINGERS]) + sk.tip_offsets + p
    np.testing.assert_allclose(fk["tip_pos"].numpy(), tips, atol=1e-12)
    # rest_joints given with the wrist elsewhere are re-centred
    sk2 = ManoSkeleton(rest_joints=sk.rest_joints + 5.0)
    np.testing.assert_allclose(sk2.rest_joints, sk.rest_joints, atol=1e-12)


def test_global_orient_rotates_all_joints_about_wrist():
    sk = ManoSkeleton()
    go, fp, wp = _rand_pose(4, 0)
    local = sk.forward(None, fp, None)
    fk = sk.forward(go, fp, wp)
    R = aa_to_matrix(torch.as_tensor(go)).numpy()
    exp = wp[:, None] + np.einsum("tij,tkj->tki", R, local["joint_pos"].numpy())
    np.testing.assert_allclose(fk["joint_pos"].numpy(), exp, atol=1e-12)
    exp_tip = wp[:, None] + np.einsum("tij,tkj->tki", R, local["tip_pos"].numpy())
    np.testing.assert_allclose(fk["tip_pos"].numpy(), exp_tip, atol=1e-12)
    np.testing.assert_allclose(fk["joint_pos"][:, 0].numpy(), wp, atol=1e-12)


@pytest.mark.parametrize("joint", [2, 13, 7])
def test_finger_joint_rotation_moves_only_descendants(joint):
    sk = ManoSkeleton()
    fp = np.zeros((15, 3))
    fp[joint - 1] = [0.3, -0.5, 0.7]
    base, moved = sk.forward(None, np.zeros((15, 3))), sk.forward(None, fp)
    changed = np.linalg.norm(moved["joint_pos"].numpy() - base["joint_pos"].numpy(), axis=-1) > 1e-9
    assert set(np.flatnonzero(changed)) == _descendants(joint)
    tip_changed = np.linalg.norm(moved["tip_pos"].numpy() - base["tip_pos"].numpy(), axis=-1) > 1e-9
    finger = next(f for f, js in FINGER_JOINTS.items() if joint in js)
    assert list(np.flatnonzero(tip_changed)) == [FINGERS.index(finger)]


def test_batching_dtype_and_broadcast():
    sk = ManoSkeleton()
    fp = torch.zeros(2, 3, 15, 3, dtype=torch.float32)
    fk = sk.forward(torch.zeros(3, dtype=torch.float32), fp, None)
    assert fk["joint_T"].shape == (2, 3, 16, 4, 4) and fk["joint_T"].dtype == torch.float32
    assert fk["tip_pos"].shape == (2, 3, 5, 3)
    with pytest.raises(ValueError):
        sk.forward(np.zeros(3), np.zeros((14, 3)))


def test_fk_is_differentiable():
    sk = ManoSkeleton()
    fp = torch.zeros(15, 3, dtype=torch.float64, requires_grad=True)
    sk.forward(None, fp)["tip_pos"][1].sum().backward()
    assert torch.isfinite(fp.grad).all() and fp.grad[0:3].abs().sum() > 0   # index joints
    assert fp.grad[3:12].abs().sum() == 0                                     # middle/pinky/ring untouched


def test_segment_transforms_cover_all_segments_and_are_bone_aligned():
    sk = ManoSkeleton()
    go, fp, wp = _rand_pose(3, 1)
    fk = sk.forward(go, fp, wp)
    seg = sk.segment_transforms(fk)
    assert set(seg) == set(MANO_SEGMENTS)
    for name, T in seg.items():
        assert T.shape == (3, 4, 4)
        R = T[..., :3, :3].numpy()
        np.testing.assert_allclose(R @ np.swapaxes(R, -1, -2), np.broadcast_to(np.eye(3), R.shape), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
        np.testing.assert_allclose(T[..., :3, 3].numpy(), fk["joint_pos"][:, SEGMENT_TO_JOINT[name]].numpy(),
                                   atol=1e-12)   # default palm_offset = 0
    torch.testing.assert_close(seg["wrist"], fk["joint_T"][:, 0])
    # rest pose: local +z of a phalanx points along the bone, -y is palmar (world -y)
    rest = sk.segment_transforms(sk.rest_fk())
    J = sk.rest_joints
    z = rest["index1"][:3, 2].numpy()
    np.testing.assert_allclose(z, (J[2] - J[1]) / np.linalg.norm(J[2] - J[1]), atol=1e-12)
    assert rest["index3"][:3, 1].numpy() @ np.array([0, 1.0, 0]) > 0.9
    raw = ManoSkeleton(bone_aligned=False).segment_transforms(sk.rest_fk())
    np.testing.assert_allclose(raw["index1"][:3, :3].numpy(), np.eye(3), atol=1e-12)


def test_taxel_poses_zero_pose_equal_transform_taxels():
    L, sk = load_layout("glove_template"), ManoSkeleton()
    pos, nrm = taxel_poses_from_hand(L, sk, np.zeros((1, 3)), np.zeros((1, 15, 3)), np.zeros((1, 3)))
    seg = {k: v.numpy() for k, v in sk.segment_transforms(sk.rest_fk()).items()}
    p_ref, n_ref = transform_taxels(L, seg)
    np.testing.assert_allclose(pos[0], p_ref, atol=1e-12)
    np.testing.assert_allclose(nrm[0], n_ref, atol=1e-12)
    # batched == per frame at random poses (chunking included)
    go, fp, wp = _rand_pose(5, 2)
    P, N = taxel_poses_from_hand(L, sk, go, fp, wp, chunk=2)
    for t in range(5):
        seg = {k: v.numpy() for k, v in sk.segment_transforms(sk.forward(go[t], fp[t], wp[t])).items()}
        p_ref, n_ref = transform_taxels(L, seg)
        np.testing.assert_allclose(P[t], p_ref, atol=1e-12)
        np.testing.assert_allclose(N[t], n_ref, atol=1e-12)
    single = taxel_poses_from_hand(L, sk, go[0], fp[0], wp[0])
    np.testing.assert_allclose(single[0], P[0], atol=1e-12)
    with pytest.raises(ValueError):
        taxel_poses_from_hand(grid_layout(2, 2, 5.0), sk, go, fp, wp)


def test_template_taxels_sit_on_the_palmar_side():
    L, sk = load_layout("glove_template"), ManoSkeleton()
    _, nrm = taxel_poses_from_hand(L, sk, np.zeros(3), np.zeros((15, 3)))
    ids = [t.id for t in L.taxels]
    for k in ("index_tip", "middle_tip", "palm_00"):
        assert nrm[ids.index(k)][1] < -0.9          # normals face the palm side (−y)


def _pinch_pose(L, sk, steps=200):
    """Optimise thumb + index joints so the thumb and index pads meet."""
    ids = [t.id for t in L.taxels]
    it, ii = ids.index("thumb_tip"), ids.index("index_tip")
    seg_idx = [MANO_SEGMENTS.index(p) for p in L.parents]
    pl = torch.as_tensor(L.positions)
    mask = torch.zeros(15, 1, dtype=torch.float64)
    mask[[0, 1, 2, 12, 13, 14]] = 1.0
    fp = torch.zeros(15, 3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([fp], lr=0.05)
    for _ in range(steps):
        ST = sk.segment_transform_tensor(sk.forward(None, fp * mask))
        pt = ST[seg_idx[it], :3, :3] @ pl[it] + ST[seg_idx[it], :3, 3]
        pi = ST[seg_idx[ii], :3, :3] @ pl[ii] + ST[seg_idx[ii], :3, 3]
        d = (pt - pi).norm()
        opt.zero_grad()
        (d + 1e-3 * (fp * mask).pow(2).sum()).backward()
        opt.step()
    return (fp * mask).detach().numpy(), float(d.detach())


def test_self_touch_pinch_but_not_open_hand():
    L, sk = load_layout("glove_template"), ManoSkeleton()
    ids = [t.id for t in L.taxels]
    spread = sk.flexion_pose(np.zeros(5), abduction=np.array([0.0, 0.1, 0.0, -0.1, -0.15])).numpy()
    open_hand = np.stack([np.zeros((15, 3)), spread])
    st = self_touch_from_hand(L, sk, np.zeros((2, 3)), open_hand, np.zeros((2, 3)))
    assert st.shape == (2, L.n) and not st.any()
    pinch, d = _pinch_pose(L, sk)
    assert d < 0.003
    st = self_touch_from_hand(L, sk, np.array([0.3, -1.0, 0.2]), pinch, np.array([0.5, 0.0, 0.1]))
    assert st.shape == (L.n,)
    assert st[ids.index("thumb_tip")] and st[ids.index("index_tip")]
    assert not st[[ids.index(k) for k in ("ring_tip", "pinky_tip", "palm_00", "palm_10")]].any()


def test_self_touch_exclude_validation():
    L, sk = load_layout("glove_template"), ManoSkeleton()
    with pytest.raises(ValueError):
        self_touch_from_hand(L, sk, np.zeros(3), np.zeros((15, 3)), exclude={"palm": ("nope",)})


def test_capsules():
    sk = ManoSkeleton()
    fk = sk.forward(None, np.zeros((4, 15, 3)))
    p0, p1, r, names = sk.capsules(fk)
    assert p0.shape == p1.shape == (4, len(CAPSULE_NAMES), 3) and r.shape == (len(CAPSULE_NAMES),)
    assert names == list(CAPSULE_NAMES) and np.all(r > 0)
    i3 = names.index("index3")
    np.testing.assert_allclose(p1[:, i3].numpy(), fk["tip_pos"][:, FINGERS.index("index")].numpy())
    i1 = names.index("index1")
    np.testing.assert_allclose(p1[:, i1].numpy(), fk["joint_pos"][:, 2].numpy())


def test_flexion_pose_curls_towards_palm():
    sk = ManoSkeleton()
    base = sk.forward(None, np.zeros((15, 3)))["tip_pos"].numpy()
    fp = sk.flexion_pose(np.array([0.0, 0.6, 0.0, 0.0, 0.0]))
    assert fp.shape == (15, 3)
    tips = sk.forward(None, fp)["tip_pos"].numpy()
    i = FINGERS.index("index")
    assert tips[i, 1] < base[i, 1] - 0.02                 # moved palmar (−y)
    np.testing.assert_allclose(np.delete(tips, i, 0), np.delete(base, i, 0), atol=1e-12)
    fp15 = sk.flexion_pose(np.full(15, 0.2))
    assert fp15.shape == (15, 3)
    with pytest.raises(ValueError):
        sk.flexion_pose(np.zeros(4))


def test_mano_pose_provider():
    L, sk = load_layout("glove_template"), ManoSkeleton()
    go, fp, wp = _rand_pose(1, 3)
    prov = ManoPoseProvider(L, sk, lambda t: {"global_orient": go[0] * t, "finger_pose": fp[0], "wrist_pos": wp[0]})
    assert isinstance(prov, TaxelPoseProvider) and prov.n_taxels == L.n
    pos, nrm = prov.pose_at(1.0)
    P, N = taxel_poses_from_hand(L, sk, go, fp, wp)
    np.testing.assert_allclose(pos, P[0], atol=1e-12)
    prov2 = ManoPoseProvider(L, None, lambda t: (go[0], fp[0]))
    assert prov2.pose_at(0.0)[0].shape == (L.n, 3)


def test_from_mano_npz(tmp_path):
    sk = ManoSkeleton()
    V = 778
    g = np.random.default_rng(0)
    v = g.normal(size=(V, 3)) * 0.05
    v[:16] = sk.rest_joints + np.array([0.09, 0.006, 0.006])
    Jr = np.zeros((16, V))
    Jr[np.arange(16), np.arange(16)] = 1.0
    shapedirs = np.zeros((V, 3, 10))
    shapedirs[:16, 0, 0] = 0.01                               # β0 shifts all joints along x
    p = tmp_path / "mano.npz"
    np.savez(p, v_template=v, J_regressor=Jr, shapedirs=shapedirs)
    sk2 = ManoSkeleton.from_mano_pkl(p)
    np.testing.assert_allclose(sk2.rest_joints, sk.rest_joints, atol=1e-12)
    sk3 = ManoSkeleton.from_mano_pkl(p, betas=[1.0], tip_vertex_ids={f: 100 + i for i, f in enumerate(FINGERS)})
    np.testing.assert_allclose(sk3.rest_joints, sk.rest_joints, atol=1e-12)   # uniform shift re-centred
    np.testing.assert_allclose(sk3.tip_offsets[0], v[100] - (v[15] + [0.01, 0.0, 0.0]), atol=1e-12)
    with pytest.raises(FileNotFoundError):
        ManoSkeleton.from_mano_pkl(tmp_path / "missing.pkl")


def test_fk_matches_hand_derived_mcp_rotation():
    """index MCP (joint 1) rotated θ about +z, PIP (joint 2) φ about local x:
    J2' = J1 + Rz(θ)(J2−J1), J3' = J2' + Rz(θ)Rx(φ)(J3−J2); positive z-rotation curls palmar (−y)."""
    sk = ManoSkeleton()
    J = sk.rest_joints
    th, ph = 0.7, -0.4
    fp = np.zeros((15, 3))
    fp[0], fp[1] = [0.0, 0.0, th], [ph, 0.0, 0.0]
    fk = sk.forward(None, fp)
    c, s = np.cos(th), np.sin(th)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
    j2 = J[1] + Rz @ (J[2] - J[1])
    j3 = j2 + Rz @ Rx @ (J[3] - J[2])
    np.testing.assert_allclose(fk["joint_pos"][2].numpy(), j2, atol=1e-12)
    np.testing.assert_allclose(fk["joint_pos"][3].numpy(), j3, atol=1e-12)
    np.testing.assert_allclose(fk["joint_rot"][3].numpy(), Rz @ Rx, atol=1e-12)
    np.testing.assert_allclose(fk["tip_pos"][1].numpy(), j3 + Rz @ Rx @ sk.tip_offsets[1], atol=1e-12)
    mcp_only = np.zeros((15, 3))
    mcp_only[0] = [0.0, 0.0, th]
    assert sk.forward(None, mcp_only)["tip_pos"][1, 1] < sk.rest_fk()["tip_pos"][1, 1] - 0.03


def _proximal_layout():
    from common.layouts import layout_from_dict
    tx = [{"id": f"{f}1_base", "channel": i, "parent": f"{f}1", "position": [0, -8, 5], "normal": [0, -1, 0]}
          for i, f in enumerate(("index", "middle", "ring", "pinky"))]
    tx.append({"id": "index1_mid", "channel": 4, "parent": "index1", "position": [0, -8, 18], "normal": [0, -1, 0]})
    tx.append({"id": "palm_c", "channel": 5, "parent": "palm", "position": [0, -12, 45], "normal": [0, -1, 0]})
    return layout_from_dict({"name": "prox", "units": "mm", "parent_frame": "mano", "taxels": tx})


def test_self_touch_no_false_positive_on_proximal_phalanges():
    """Taxels at the base of a proximal phalanx lie inside the palm capsule ending at their own MCP;
    the default exclusion must keep a flat / spread hand free of self-touch."""
    L, sk = _proximal_layout(), ManoSkeleton()
    spread = sk.flexion_pose(np.zeros(5), abduction=np.array([0.0, 0.1, 0.0, -0.1, -0.15])).numpy()
    st = self_touch_from_hand(L, sk, np.zeros((2, 3)), np.stack([np.zeros((15, 3)), spread]))
    assert not st.any()
    # without the per-segment exclusion the base taxels are (wrongly) flagged
    st_raw = self_touch_from_hand(L, sk, np.zeros(3), np.zeros((15, 3)), exclude={"palm": ("thumb1",)})
    assert st_raw[:4].all() and not st_raw[4:].any()
    # segment-name keys and finger-group keys combine
    st_seg = self_touch_from_hand(L, sk, np.zeros(3), np.zeros((15, 3)),
                                  exclude={"index1": ("palm_index",), "middle": ("palm_middle",)})
    assert list(st_seg[:4]) == [False, False, True, True]
    with pytest.raises(ValueError):
        self_touch_from_hand(L, sk, np.zeros(3), np.zeros((15, 3)), exclude={"index4": ("palm_index",)})
