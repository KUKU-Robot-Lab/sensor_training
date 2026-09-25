"""robot_skin.transfer: capsule skeletons, taxel projection, layout alignment, value mapping and the
robot → MANO hand-state estimate."""
import numpy as np
import pytest
import torch

from common.layouts import load_layout
from robot_skin.transfer import (CapsuleSkeleton, LayoutAlignment, align_layouts, finger_group, layout_rest_poses,
                                 map_taxel_values, project_to_mano, project_to_skeleton, taxel_groups)


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def hands():
    from robot_skin.datasets.synthetic import robot_hand_urdf
    from robot_skin.pose.urdf import URDFModel

    model = URDFModel.from_string(robot_hand_urdf())
    g, r = load_layout("glove_template"), load_layout("robot_hand_template")
    gp, gn = layout_rest_poses(g)
    rp, rn = layout_rest_poses(r, urdf=model)
    return {"model": model, "glove": g, "robot": r, "gp": gp, "gn": gn, "rp": rp, "rn": rn,
            "mano": CapsuleSkeleton.from_mano(), "rsk": CapsuleSkeleton.from_urdf(model, palmar_axis=(1, 0, 0))}


def test_finger_groups_from_names_and_layout_groups():
    assert [finger_group(n) for n in ("index3", "palm_index", "thumb_distal_link", "little_tip", "wrist",
                                      "base_link", "r_middle_link", "camera")] == \
        ["index", "palm", "thumb", "pinky", "palm", "palm", "middle", "other"]
    assert taxel_groups("glove_template") == ["thumb", "index", "middle", "ring", "pinky"] + ["palm"] * 4
    assert taxel_groups("robot_hand_template") == taxel_groups("glove_template")


def test_projection_matches_hand_derived_coordinates():
    """Two-capsule finger (lengths 1 and 0.5) + one palm capsule: t, u (accumulated along the finger),
    side, distances and the palm lateral coordinate derived by hand."""
    sk = CapsuleSkeleton(p0=[[0, 0, 0], [0, 0, 1.0], [1, 0, 0]], p1=[[0, 0, 1.0], [0, 0, 1.5], [2, 0, 0]],
                         radius=[0.1, 0.1, 0.2], names=["index_a", "index_b", "palm_index"], rank=[0, 1, 0],
                         palmar=[[1, 0, 0], [1, 0, 0], [0, 0, 1]])
    pts = np.array([[0.2, 0.0, 0.25], [-0.05, 0.0, 1.25], [1.5, 0.0, 0.3]])
    pr = project_to_skeleton(pts, sk, taxel_groups=["index", "index", "palm"])
    np.testing.assert_array_equal(pr.segment, [0, 1, 2])
    np.testing.assert_allclose(pr.t, [0.25, 0.5, 0.5])
    np.testing.assert_allclose(pr.u, [0.25 / 1.5, (1.0 + 0.25) / 1.5, 0.5])     # palm: u = t
    np.testing.assert_allclose(pr.side, [1.0, -1.0, 1.0])                        # palmar / dorsal
    np.testing.assert_allclose(pr.distance, [0.2, 0.05, 0.3])
    np.testing.assert_allclose(pr.surface_distance, [0.1, -0.05, 0.1])
    assert np.isnan(pr.v[:2]).all() and pr.v[2] == 0.0                           # index-side palm capsule
    np.testing.assert_allclose(pr.closest, [[0, 0, 0.25], [0, 0, 1.25], [1.5, 0, 0]])


def test_palm_lateral_coordinate_interpolates_between_rays():
    """``v`` of a palm taxel is continuous: linear between the nearest wrist → knuckle ray and its
    neighbour on the taxel's side, from in-plane distances (the palmar offset of the skin is ignored),
    clamped beyond the outermost rays. Three parallel rays (index 0, middle 1/3, ring 2/3) at y = 0, 1, 2."""
    sk = CapsuleSkeleton(p0=[[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 0]],
                         p1=[[1, 0, 0], [1, 1, 0], [1, 2, 0], [0, 0, 1]],
                         radius=0.1, names=["palm_index", "palm_middle", "palm_ring", "index1"])
    pts = np.array([[0.5, 0.25, 0.3],      # 1/4 of the way index → middle, 0.3 palmar: 1/12
                    [0.5, 1.5, -0.2],      # halfway middle → ring: 1/2
                    [0.5, -0.5, 0.0],      # radial of the index ray: 0
                    [0.5, 2.4, 0.1],       # ulnar of the ring ray: 2/3
                    [0.0, 0.0, 0.9]])      # on the finger: NaN
    groups = ["palm"] * 4 + ["index"]
    pr = project_to_skeleton(pts, sk, taxel_groups=groups)
    np.testing.assert_allclose(pr.v[:4], [1 / 12, 0.5, 0.0, 2 / 3], atol=1e-12)
    assert np.isnan(pr.v[4])
    pb = project_to_skeleton(np.stack([pts, pts[::-1]]), sk, taxel_groups=None)       # batched
    np.testing.assert_allclose(pb.v[0, :4], pr.v[:4], atol=1e-12)
    np.testing.assert_allclose(pb.v[1, 1:], pr.v[:4][::-1], atol=1e-12)


def test_template_palm_pads_map_one_to_one_with_sides_kept(hands):
    """glove ⇄ robot templates in skeleton space: each of the four palm pads maps to its counterpart
    (radial/ulnar × proximal/distal) in both directions — none crosses the hand's midline."""
    g, r = hands["glove"], hands["robot"]
    g2r = align_layouts(g, r, src_pos=hands["gp"], dst_pos=hands["rp"], src_skeleton=hands["mano"],
                        dst_skeleton=hands["rsk"])
    r2g = align_layouts(r, g, src_pos=hands["rp"], dst_pos=hands["gp"], src_skeleton=hands["rsk"],
                        dst_skeleton=hands["mano"])
    assert [g.taxels[i].id for i in g2r.index[5:, 0]] == [t.id for t in r.taxels[5:]]
    assert [r.taxels[i].id for i in r2g.index[5:, 0]] == [t.id for t in g.taxels[5:]]
    # the ids name the same pads: radial side is glove palm-frame +x / robot palm-frame +y
    assert all((g.taxels[i].position[0] > 0) == (r.taxels[i].position[1] > 0) for i in range(5, 9))


def test_mano_skeleton_projection_of_glove_taxels(hands):
    sk = hands["mano"]
    assert sk.n == 19 and sk.names[0] == "index1" and np.allclose(np.linalg.norm(sk.palmar, axis=1), 1.0)
    assert list(sk.rank[:3]) == [0, 1, 2] and np.all(sk.lengths > 0.015)
    pr = project_to_skeleton(hands["gp"], sk, taxel_groups=taxel_groups(hands["glove"]))
    assert pr.segment_names[:5] == ["thumb3", "index3", "middle3", "ring3", "pinky3"]
    assert all(n.startswith("palm_") for n in pr.segment_names[5:])
    assert np.all((pr.t >= 0) & (pr.t <= 1)) and np.all(pr.u[:5] > 0.75)       # fingertip pads: distal
    assert np.all(pr.side > 0.9)                                                # every glove pad is palmar
    assert np.all(np.abs(pr.surface_distance) < 0.005)                          # on the skin surface
    assert np.isnan(pr.v[:5]).all() and pr.v[6] == 0.0                         # palm_01 radial of the index ray
    assert 2 / 3 < pr.v[5] < 1.0 and 1 / 3 < pr.v[7] < 2 / 3 and 0.0 < pr.v[8] < 1 / 3   # palm_00/10 ulnar, 11 radial
    np.testing.assert_allclose(pr.closest + pr.offset, hands["gp"], atol=1e-12)
    # batched input + unrestricted projection; alias with the default flat MANO hand
    T = np.stack([hands["gp"], hands["gp"] + [0.0, 0.0, 0.001]])
    pb = project_to_skeleton(T, sk)
    assert pb.segment.shape == (2, 9) and pb.u.shape == (2, 9) and pb.finger.shape == (2, 9)
    pm = project_to_mano(hands["gp"])
    np.testing.assert_array_equal(pm.segment, pb.segment[0])
    # a curled finger moves the capsules with it
    fp = np.zeros((15, 3))
    fp[1, 2] = 1.0
    assert not np.allclose(CapsuleSkeleton.from_mano(finger_pose=fp).p1[1], sk.p1[1])
    with pytest.raises(ValueError):
        project_to_skeleton(hands["gp"], sk, allowed=np.zeros((9, 19), bool))
    with pytest.raises(TypeError):
        project_to_skeleton(hands["gp"], "mano")


def test_urdf_skeleton_projection_of_robot_taxels(hands):
    rsk = hands["rsk"]
    assert "index_distal_link" in rsk.names and "palm_link>index_proximal_link" in rsk.names
    assert all(L > 1e-4 for L in rsk.lengths)                                   # co-located frames skipped
    pr = project_to_skeleton(hands["rp"], rsk, taxel_groups=taxel_groups(hands["robot"]))
    assert pr.segment_names[:5] == [f"{f}_distal_link" for f in ("thumb", "index", "middle", "ring", "pinky")]
    assert np.all(pr.u[:5] > 0.8) and np.all(pr.u[5:] < 0.8)
    # (p0, p1, radius, names) tuples work too, e.g. ManoSkeleton.capsules output
    tup = (rsk.p0, rsk.p1, rsk.radius, rsk.names)
    np.testing.assert_array_equal(project_to_skeleton(hands["rp"], tup).segment, project_to_skeleton(hands["rp"],
                                                                                                    rsk).segment)


def test_align_layouts_same_and_cross_embodiment(hands):
    g, r = hands["glove"], hands["robot"]
    same = align_layouts(g, g)
    np.testing.assert_array_equal(same.index[:, 0], np.arange(9))
    assert same.valid.all() and same.space == "euclidean" and np.allclose(same.distance[:, 0], 0.0)
    r2g = align_layouts(g, r, src_pos=hands["gp"], dst_pos=hands["rp"], src_skeleton=hands["mano"],
                        dst_skeleton=hands["rsk"])
    assert r2g.space == "skeleton" and r2g.valid.all()
    np.testing.assert_array_equal(r2g.index[:5, 0], np.arange(5))              # fingertip → same fingertip
    assert set(r2g.index[5:, 0]) <= {5, 6, 7, 8}                               # palm → palm
    assert r2g.index[5, 0] == 5 and r2g.index[6, 0] == 6                       # ulnar / radial side kept
    g2r = align_layouts(r, g, src_pos=hands["rp"], dst_pos=hands["gp"], src_skeleton=hands["rsk"],
                        dst_skeleton=hands["mano"], k=2)
    assert g2r.k == 2 and np.allclose(g2r.weight.sum(1), 1.0)
    np.testing.assert_allclose(g2r.weight[:5, 0], 1.0)                         # one thumb taxel per hand
    M = g2r.matrix()
    assert M.shape == (9, 9) and np.allclose(M.sum(1), 1.0)
    rt = LayoutAlignment.from_dict(g2r.to_dict())
    np.testing.assert_array_equal(rt.index, g2r.index)
    # euclidean with a URDF layout needs the model; max_dist invalidates far matches
    with pytest.raises(ValueError, match="urdf"):
        align_layouts(g, r)
    far = align_layouts(g, r, dst_urdf=hands["model"], max_dist=0.01)
    assert not far.valid.all()
    with pytest.raises(ValueError, match="both"):
        align_layouts(g, r, src_skeleton=hands["mano"], dst_urdf=hands["model"])


def test_map_taxel_values_reduce_modes_and_fill(tmp_path):
    al = LayoutAlignment(index=np.array([[0, 1], [2, 0], [1, 2]]), weight=np.array([[0.75, 0.25], [1.0, 0.0],
                                                                                    [0.5, 0.5]]),
                         distance=np.zeros((3, 2)), valid=np.array([True, True, False]), n_src=3)
    v = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])                           # [T=2, Ns=3]
    np.testing.assert_allclose(map_taxel_values(v, al), [[1.5, 5.0, np.nan], [2.5, 6.0, np.nan]])
    np.testing.assert_array_equal(map_taxel_values(v, al, reduce="nearest", fill=0.0), [[1, 5, 0], [2, 6, 0]])
    lv = np.array([0, 2, 3], np.int8)
    out = map_taxel_values(lv, al, reduce="max")
    assert out.dtype == np.int8 and list(out) == [2, 3, -1]
    b = map_taxel_values(np.array([True, False, False]), al, reduce="max")
    assert list(b) == [True, True, False]
    feats = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)                  # [Ns, F]
    f2 = map_taxel_values(feats, al, taxel_axis=-2, reduce="nearest")
    assert f2.shape == (3, 4) and np.array_equal(f2[1], feats[2]) and np.isnan(f2[2]).all()
    with pytest.raises(ValueError):
        map_taxel_values(np.zeros(4), al)
    with pytest.raises(ValueError):
        map_taxel_values(v, al, reduce="median")
    al.save(tmp_path / "al.json")
    assert LayoutAlignment.load(tmp_path / "al.json").n_src == 3


def test_k_beyond_group_size_and_max_dist_never_leak_across_matches(hands):
    """Each fingertip group has one taxel per hand: with k = 2 the second column has no same-group
    source. It must not carry another finger's flag / level (reduce max) or NaN (weighted), and
    max_dist must drop every far match, not only the nearest."""
    g, r = hands["glove"], hands["robot"]
    al = align_layouts(g, r, src_pos=hands["gp"], dst_pos=hands["rp"], src_skeleton=hands["mano"],
                       dst_skeleton=hands["rsk"], k=2)
    tips = slice(0, 5)
    assert np.all(al.index[tips, 1] == al.index[tips, 0]) and np.isinf(al.distance[tips, 1]).all()
    for i in range(r.n):                                                     # no column names another group
        assert all(al.src_groups[j] == al.dst_groups[i] for j in al.index[i])
    flags = np.zeros(9, bool)
    flags[0] = True                                                          # only the glove thumb tip touched
    out = map_taxel_values(flags, al, reduce="max")
    assert [t.id for t, f in zip(r.taxels, out) if f] == ["thumb_tip"]
    lv = np.zeros(9, np.int64)
    lv[0] = 3
    np.testing.assert_array_equal(map_taxel_values(lv, al, reduce="max"), [3, 0, 0, 0, 0, 0, 0, 0, 0])
    vals = np.arange(9, dtype=float)
    vals[0] = np.nan                                                         # invalid thumb channel
    w = map_taxel_values(vals, al)
    assert np.isnan(w[0]) and np.isfinite(w[1:]).all()
    np.testing.assert_allclose(w[1:5], [1.0, 2.0, 3.0, 4.0])
    # an alignment saved before the padding (cross-group column, weight 0, distance inf) is masked too
    old = LayoutAlignment(index=np.array([[0, 1], [1, 0]]), weight=np.array([[1.0, 0.0], [1.0, 0.0]]),
                          distance=np.array([[0.1, np.inf], [0.1, np.inf]]), valid=np.array([True, True]), n_src=2)
    assert list(map_taxel_values(np.array([True, False]), old, reduce="max")) == [True, False]
    np.testing.assert_allclose(map_taxel_values(np.array([np.nan, 2.0]), old), [np.nan, 2.0])
    # max_dist applies to every match: glove → glove, palm_00 ← palm_00 (0) but not palm_01 (24 mm away)
    same = align_layouts(g, g, k=2, max_dist=1e-3)
    assert same.valid.all() and np.all(same.index[:, 1] == np.arange(9)) and np.all(same.weight[:, 1] == 0.0)
    f2 = np.zeros(9, bool)
    f2[6] = True                                                             # palm_01 only
    assert list(np.flatnonzero(map_taxel_values(f2, same, reduce="max"))) == [6]


def test_robot_to_mano_estimator_inverts_the_retargeting(hands):
    """human flexion pose → forward retarget → robot q → reverse estimate: the estimate reproduces
    the human fingertips much better than the flat hand does (the robot cannot reach every human pose,
    so exactness is not expected)."""
    from robot_skin.action.retarget import human_fingertips
    from robot_skin.pose.mano import ManoSkeleton
    from robot_skin.stages.deploy import build_retargeter
    from robot_skin.transfer import RobotToManoEstimator

    rt = build_retargeter(hands["model"], hands["robot"], {"iters": 30}, synthetic=True)
    sk = ManoSkeleton()
    x = np.r_[np.full(15, 0.5), np.zeros(5)]
    fp = sk.flexion_pose(torch.as_tensor(x[:15]), torch.as_tensor(x[15:])).numpy()
    tips = human_fingertips(fp)
    q = rt.retarget(tips)
    est = RobotToManoEstimator(rt, iters=30)
    fp_hat = est.estimate(q)
    err = np.linalg.norm(human_fingertips(fp_hat) - tips, axis=1).mean()
    err_flat = np.linalg.norm(human_fingertips(np.zeros((15, 3))) - tips, axis=1).mean()
    assert fp_hat.shape == (15, 3) and err < 0.5 * err_flat
    a = est(q, np.zeros(54, np.float32))                                      # warm-started second solve
    assert a.shape == (54,) and np.allclose(a[9:], fp_hat.reshape(-1), atol=1e-3) and not a[:9].any()


def test_compat_module_path():
    from robot_skin.transfer.mano_projection import align_layouts as a2, project_to_mano as p2

    assert a2 is align_layouts and p2 is project_to_mano
