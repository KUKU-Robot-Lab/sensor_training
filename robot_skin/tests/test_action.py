import json
import math

import numpy as np
import pytest
import torch

from robot_skin.action import (HAND_MANO_DIM, HAND_MANO_NAMES, MANO_FINGER_JOINTS, ActionNormalizer, ActionSpec,
                               TemporalEnsembler, action_chunk, action_chunks, actions_from_episode,
                               hand_action_from_arrays, hand_action_to_arrays, make_absolute, make_relative,
                               policy_stride, policy_tick_indices, robot_action_from_q, wrist_rotation)
from robot_skin.datasets import episode as E
from robot_skin.geometry import rotations as rot


def _hand_arrays(T=32, seed=0):
    rng = np.random.default_rng(seed)
    go = rng.normal(size=(T, 3))
    go = go / np.linalg.norm(go, axis=-1, keepdims=True) * rng.uniform(0.0, 3.0, (T, 1))
    go[0] = [0.0, 0.0, 0.0]                  # identity
    go[1] = [0.0, 0.0, 4.0]                  # angle > π: raw aa is not unique, rotation is
    fp = rng.normal(scale=0.4, size=(T, 15, 3))
    wp = rng.normal(scale=0.2, size=(T, 3))
    return go, fp, wp


def _R(aa):
    return rot.aa_to_matrix(torch.as_tensor(np.asarray(aa, dtype=np.float64))).numpy()


# ── space ─────────────────────────────────────────────────────────────────────
def test_hand_action_layout_is_54d():
    go, fp, wp = _hand_arrays()
    a = hand_action_from_arrays(go, fp, wp)
    assert a.shape == (32, HAND_MANO_DIM) == (32, 54) and a.dtype == np.float32
    np.testing.assert_allclose(a[:, 0:3], wp, atol=1e-6)
    R = _R(go)
    np.testing.assert_allclose(a[:, 3:6], R[:, :, 0], atol=1e-6)      # 6D = first two columns
    np.testing.assert_allclose(a[:, 6:9], R[:, :, 1], atol=1e-6)
    np.testing.assert_allclose(a[:, 9:], fp.reshape(32, 45), atol=1e-6)
    spec = ActionSpec.hand_mano()
    assert spec.dim == 54 and spec.names == HAND_MANO_NAMES and spec.rot_repr == "6d"
    assert spec.names[:3] == ("wrist_x", "wrist_y", "wrist_z") and spec.names[9] == "index1_x"
    assert spec.names[-1] == "thumb3_z"
    assert spec.slices["finger_aa"] == slice(9, 54)


def test_mano_joint_order_matches_pose_module():
    mano = pytest.importorskip("robot_skin.pose.mano")
    assert MANO_FINGER_JOINTS == tuple(mano.MANO_JOINTS[1:])
    # decoded arrays splat straight into the MANO FK
    go, fp, wp = _hand_arrays(T=3)
    skel = mano.ManoSkeleton()
    fk = skel.forward(**hand_action_to_arrays(hand_action_from_arrays(go, fp, wp)))
    np.testing.assert_allclose(fk["joint_pos"].numpy(), skel.forward(go, fp, wp)["joint_pos"].numpy(), atol=1e-5)


def test_hand_action_decode_is_differentiable():
    go, fp, wp = (torch.as_tensor(x) for x in _hand_arrays(T=4))
    a = hand_action_from_arrays(go, fp, wp).requires_grad_(True)
    out = hand_action_to_arrays(a)
    (out.global_orient.sum() + out.finger_pose.sum() + out.wrist_pos.sum()).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad[:, 3:9].abs().sum() > 0


def test_hand_action_roundtrip_rotation_equality():
    go, fp, wp = _hand_arrays()
    back = hand_action_to_arrays(hand_action_from_arrays(go, fp, wp))
    assert set(back.keys()) == {"global_orient", "finger_pose", "wrist_pos"}
    assert back["finger_pose"].shape == (32, 15, 3)
    # tuple semantics in the argument order of hand_action_from_arrays (spec: "inverse") …
    go2, fp2, wp2 = back
    assert go2 is back["global_orient"] and fp2 is back.finger_pose and wp2 is back[2]
    # … and mapping semantics (splat into ManoSkeleton.forward / hand_action_from_arrays)
    assert "finger_pose" in back and "nope" not in back and dict(back.items()).keys() == set(back.keys())
    np.testing.assert_allclose(hand_action_from_arrays(**back), hand_action_from_arrays(go, fp, wp), atol=2e-6)
    with pytest.raises(KeyError):
        back["fingers"]
    # compare rotations, not raw axis-angle (row 1 has |θ| > π and comes back as 2π − θ)
    np.testing.assert_allclose(_R(back["global_orient"]), _R(go), atol=2e-5)
    assert np.linalg.norm(back["global_orient"], axis=-1).max() <= math.pi + 1e-5
    assert not np.allclose(back["global_orient"][1], go[1], atol=1e-3)
    np.testing.assert_allclose(back["finger_pose"], fp, atol=1e-6)
    np.testing.assert_allclose(back["wrist_pos"], wp, atol=1e-6)
    np.testing.assert_allclose(wrist_rotation(hand_action_from_arrays(go, fp, wp)), _R(go), atol=2e-5)


def test_hand_action_torch_and_batch_dims():
    go, fp, wp = (torch.as_tensor(x) for x in _hand_arrays(T=12))
    a = hand_action_from_arrays(go.reshape(3, 4, 3), fp.reshape(3, 4, 15, 3), wp.reshape(3, 4, 3))
    assert isinstance(a, torch.Tensor) and a.dtype == torch.float64 and a.shape == (3, 4, 54)
    back = hand_action_to_arrays(a)
    torch.testing.assert_close(rot.aa_to_matrix(back["global_orient"]), rot.aa_to_matrix(go.reshape(3, 4, 3)))
    # non-orthonormal 6D (network output) still decodes to a rotation
    noisy = a + 0.05 * torch.randn(a.shape, generator=torch.Generator().manual_seed(0), dtype=a.dtype)
    Rn = rot.aa_to_matrix(hand_action_to_arrays(noisy)["global_orient"])
    torch.testing.assert_close(torch.linalg.det(Rn), torch.ones(3, 4, dtype=torch.float64))


def test_hand_action_shape_errors():
    go, fp, wp = _hand_arrays(T=4)
    with pytest.raises(ValueError):
        hand_action_from_arrays(go, fp[:, :14], wp)
    with pytest.raises(ValueError):
        hand_action_from_arrays(go[:3], fp, wp)
    with pytest.raises(ValueError):
        hand_action_to_arrays(np.zeros((4, 53)))


def test_action_spec_validation_and_serialization():
    s = ActionSpec.robot_joint(["j0", "j1", "j2"])
    assert s.kind == "robot_joint" and s.dim == 3 and s.slices == {"q": slice(0, 3)}
    assert ActionSpec.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    h = ActionSpec.hand_mano()
    assert ActionSpec.from_dict(json.loads(json.dumps(h.to_dict()))) == h
    assert ActionSpec.robot_joint(5).names == ("a0", "a1", "a2", "a3", "a4")
    with pytest.raises(ValueError):
        ActionSpec("hand_mano", 48, (), "6d")
    with pytest.raises(ValueError):
        ActionSpec("gripper", 1)
    with pytest.raises(ValueError):
        ActionSpec("robot_joint", 2, ("a",))
    q = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = robot_action_from_q(q)
    assert out.dtype == np.float32 and np.array_equal(out, q) and out is not q
    # kind defaults: names / rot_repr may be omitted
    assert ActionSpec("hand_mano", 54) == h
    assert ActionSpec("robot_joint", 2).rot_repr == "none"
    assert ActionSpec.from_dict({"kind": "hand_mano", "dim": 54}) == h
    with pytest.raises(ValueError):
        ActionSpec("robot_joint", 0)
    with pytest.raises(ValueError):
        ActionSpec("robot_joint", 2, rot_repr="6d")


def test_actions_from_episode_hand_and_robot():
    go, fp, wp = _hand_arrays(T=20)
    valid = np.ones(20, bool)
    valid[5] = False
    meta = E.EpisodeMeta(episode_id="e", dataset="task", kind="glove", layout="glove_template", n_taxels=2,
                         joint_names=["a", "b"])
    arrays = {E.K_T: np.arange(20) / 200.0, E.K_HAND_GLOBAL: go.astype(np.float32),
              E.K_HAND_FINGERS: fp.astype(np.float32), E.K_HAND_WRIST: wp.astype(np.float32),
              E.K_HAND_VALID: valid, E.K_Q: np.ones((20, 2), np.float32)}
    ep = E.Episode(meta, arrays)
    a, v = actions_from_episode(ep, "hand_mano")
    assert a.shape == (20, 54) and v.dtype == bool and not v[5] and v.sum() == 19
    ar, vr = actions_from_episode(ep, "robot_joint")
    assert ar.shape == (20, 2) and vr.all()
    with pytest.raises(ValueError):
        actions_from_episode(ep, ActionSpec.robot_joint(3))
    del arrays[E.K_HAND_GLOBAL]
    with pytest.raises(KeyError):
        actions_from_episode(E.Episode(meta, arrays), ActionSpec.hand_mano())


# ── normaliser ────────────────────────────────────────────────────────────────
def test_normalizer_fit_roundtrip_and_serialization(tmp_path):
    rng = np.random.default_rng(1)
    eps_a = [rng.normal(loc=3.0, scale=2.0, size=(50, 4)).astype(np.float32) for _ in range(3)]
    masks = [np.ones(50, bool) for _ in range(3)]
    eps_a[0][:10] = 1e6                     # invalid rows must not leak into the stats
    masks[0][:10] = False
    eps_a[1][:, 3] = 7.0                    # constant dim → min_scale floor
    eps_a[2][:, 3] = 7.0
    eps_a[0][:, 3] = 7.0
    spec = ActionSpec.robot_joint(4)
    norm = ActionNormalizer.fit(eps_a, masks, spec=spec, method="std", min_scale=0.05)
    assert norm.dim == 4
    rows = np.concatenate([eps_a[0][10:], eps_a[1], eps_a[2]])
    z = norm.normalize(rows)
    assert z.dtype == np.float32
    np.testing.assert_allclose(z[:, :3].mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(z[:, :3].std(0), 1, atol=1e-3)
    assert norm.stats.scale[3] == pytest.approx(0.05)
    np.testing.assert_allclose(norm.unnormalize(z), rows, rtol=1e-5, atol=1e-4)
    # torch path == numpy path, keeps dtype
    zt = norm.normalize(torch.as_tensor(rows, dtype=torch.float64))
    assert zt.dtype == torch.float64
    np.testing.assert_allclose(zt.numpy(), z, atol=1e-5)
    # serialization
    d = json.loads(json.dumps(norm.to_dict()))
    n2 = ActionNormalizer.from_dict(d)
    assert n2.spec == spec and n2.method == "std"
    np.testing.assert_array_equal(n2.normalize(rows), z)
    norm.save(tmp_path / "act_norm.json")
    n3 = ActionNormalizer.load(tmp_path / "act_norm.json")
    np.testing.assert_array_equal(n3.unnormalize(z), norm.unnormalize(z))
    with pytest.raises(ValueError):
        norm.normalize(np.zeros((2, 5)))
    # NormStats-style aliases and a bare NormStats dict
    np.testing.assert_array_equal(norm.apply(rows), z)
    np.testing.assert_array_equal(norm.invert(z), norm.unnormalize(z))
    bare = ActionNormalizer.from_dict(norm.stats.to_dict())
    np.testing.assert_array_equal(bare.normalize(rows), z)
    with pytest.raises(ValueError):
        ActionNormalizer(type(norm.stats)(offset=np.zeros(2, np.float32), scale=np.array([1, np.nan], np.float32)))


def test_normalizer_hand_computed():
    # std: mean 2, population std sqrt(8/3); constant dim floored at min_scale
    n = ActionNormalizer.fit(np.array([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]]), method="std", eps=0.0)
    np.testing.assert_allclose(n.stats.offset, [2.0, 1.0])
    np.testing.assert_allclose(n.stats.scale, [math.sqrt(8 / 3), 0.01], rtol=1e-6)
    # minmax: [0, 4] → offset 2, scale 2 → endpoints map to ±1
    m = ActionNormalizer.fit(np.array([[0.0], [2.0], [4.0]]), method="minmax", eps=0.0)
    np.testing.assert_allclose(m.normalize(np.array([[0.0], [4.0], [3.0]]))[:, 0], [-1.0, 1.0, 0.5])


def test_normalizer_minmax_and_chunks():
    rng = np.random.default_rng(2)
    chunks = rng.uniform(-3, 5, size=(40, 8, 54))
    valid = rng.random((40, 8)) > 0.2
    norm = ActionNormalizer.fit(chunks, valid, spec="hand_mano", method="minmax")   # str spec accepted
    assert norm.spec == ActionSpec.hand_mano()
    z = norm.normalize(chunks[valid])
    assert z.min() >= -1 - 1e-5 and z.max() <= 1 + 1e-5
    np.testing.assert_allclose(norm.unnormalize(z), chunks[valid], atol=1e-4)
    with pytest.raises(ValueError):
        ActionNormalizer.fit(chunks, np.zeros((40, 8), bool))
    with pytest.raises(ValueError):
        ActionNormalizer.fit(chunks, method="zscore")


# ── relative actions ──────────────────────────────────────────────────────────
def _hand_chunk(B=3, H=5, seed=3):
    go, fp, wp = _hand_arrays(T=B * (H + 1), seed=seed)
    a = hand_action_from_arrays(go, fp, wp).astype(np.float64).reshape(B, H + 1, 54)
    return a[:, 1:], a[:, 0]                 # chunk [B,H,54], state [B,54]


def test_make_relative_absolute_roundtrip_hand():
    chunk, state = _hand_chunk()
    spec = ActionSpec.hand_mano()
    rel = make_relative(chunk, state, spec)                      # default: wrist position delta
    np.testing.assert_allclose(rel[..., :3], chunk[..., :3] - state[:, None, :3], atol=1e-6)
    np.testing.assert_allclose(rel[..., 3:], chunk[..., 3:], atol=1e-6)
    np.testing.assert_allclose(make_absolute(rel, state, spec), chunk, atol=1e-6)

    relp = make_relative(chunk, state, "hand_mano", mode="delta_pose")
    np.testing.assert_allclose(make_absolute(relp, state, "hand_mano", mode="delta_pose"), chunk, atol=1e-5)
    np.testing.assert_allclose(relp[..., 9:], chunk[..., 9:], atol=1e-6)  # fingers stay absolute
    # the state relative to itself is the identity pose
    self_rel = make_relative(state, state, spec, mode="delta_pose")
    np.testing.assert_allclose(self_rel[:, :3], 0, atol=1e-6)
    np.testing.assert_allclose(self_rel[:, 3:9], np.tile([1, 0, 0, 0, 1, 0], (3, 1)), atol=1e-6)
    # position is expressed in the current wrist frame: Rsᵀ (p − ps)
    Rs = wrist_rotation(state).astype(np.float64)
    expect = np.einsum("bji,bhj->bhi", Rs, chunk[..., :3] - state[:, None, :3])
    np.testing.assert_allclose(relp[..., :3], expect, atol=1e-5)
    # abs mode is an identity copy
    np.testing.assert_allclose(make_relative(chunk, state, spec, mode="abs"), chunk, atol=1e-6)
    with pytest.raises(ValueError):
        make_relative(chunk, state, spec, mode="velocity")


def test_make_relative_hand_computed_and_default_spec():
    # state: wrist at (1,0,0) rotated +90° about z; action: wrist at (1,1,0) rotated 180° about z.
    # delta_pose: p' = Rz(−90°)·(0,1,0) = (1,0,0); R' = Rz(90°) → 6D (0,1,0, −1,0,0)
    z = np.zeros((15, 3))
    s = hand_action_from_arrays([0, 0, math.pi / 2], z, [1, 0, 0])
    x = hand_action_from_arrays([0, 0, math.pi], z, [1, 1, 0])
    np.testing.assert_allclose(s[3:9], [0, 1, 0, -1, 0, 0], atol=1e-6)
    rel = make_relative(x, s, mode="delta_pose")                 # spec defaults to hand_mano
    np.testing.assert_allclose(rel[:3], [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(rel[3:9], [0, 1, 0, -1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(make_relative(x, s)[:3], [0, 1, 0], atol=1e-6)   # delta: world frame
    # "robot_joint" string: dim inferred from the actions
    q = np.arange(8.0).reshape(2, 4)
    np.testing.assert_allclose(make_relative(q, q[0], "robot_joint"), q - q[0])
    with pytest.raises(ValueError):                              # 4-D robot actions are not hand actions
        make_relative(q, q[0])


def test_make_relative_robot_and_torch():
    spec = ActionSpec.robot_joint(4)
    g = torch.Generator().manual_seed(0)
    chunk = torch.randn(2, 6, 4, generator=g, dtype=torch.float64)
    state = torch.randn(2, 4, generator=g, dtype=torch.float64)
    rel = make_relative(chunk, state, spec)
    assert isinstance(rel, torch.Tensor) and rel.dtype == torch.float64
    torch.testing.assert_close(rel, chunk - state[:, None])
    torch.testing.assert_close(make_absolute(rel, state, spec), chunk)
    with pytest.raises(ValueError):
        make_relative(chunk, state, spec, mode="delta_pose")
    with pytest.raises(ValueError):
        make_relative(chunk, state[:, :3], spec)


# ── chunking ──────────────────────────────────────────────────────────────────
def _ramp(T=10):
    return np.stack([np.arange(T), 10 * np.arange(T)], -1).astype(np.float32)


def test_action_chunk_future_padding_and_mask():
    a = _ramp(10)
    c, v = action_chunk(a, 6, horizon=4)                         # t+1 … t+4 → 7, 8, 9, pad
    np.testing.assert_array_equal(c[:, 0], [7, 8, 9, 9])
    np.testing.assert_array_equal(v, [True, True, True, False])
    c, v = action_chunk(a, 2, horizon=4, stride=2)               # 4, 6, 8, 10→pad(9)
    np.testing.assert_array_equal(c[:, 0], [4, 6, 8, 9])
    np.testing.assert_array_equal(c[:, 1], [40, 60, 80, 90])
    np.testing.assert_array_equal(v, [True, True, True, False])
    c, v = action_chunk(a, 2, horizon=3, stride=2, offset=0)     # includes the current step
    np.testing.assert_array_equal(c[:, 0], [2, 4, 6])
    c, v = action_chunk(a, 9, horizon=3)                         # last frame: nothing in the future
    np.testing.assert_array_equal(c[:, 0], [9, 9, 9])
    assert not v.any()
    src_valid = np.ones(10, bool)
    src_valid[8] = False                                         # e.g. hand_pose_valid gap
    c, v = action_chunk(a, 6, horizon=3, valid=src_valid)
    np.testing.assert_array_equal(v, [True, False, True])
    with pytest.raises(IndexError):
        action_chunk(a, 10, horizon=2)
    with pytest.raises(ValueError):
        action_chunk(a, 0, horizon=0)


def test_action_chunk_pads_with_last_valid_frame():
    # invalid source frames hold NaN (e.g. missing hand labels); masked steps must never carry them
    a = _ramp(10).astype(np.float64)
    src_valid = np.ones(10, bool)
    src_valid[[7, 9]] = False
    a[~src_valid] = np.nan
    c, v = action_chunk(a, 3, horizon=4, stride=2, valid=src_valid)  # 5, 7(✗), 9(✗), 11→pad
    np.testing.assert_array_equal(v, [True, False, False, False])
    np.testing.assert_array_equal(c[:, 0], [5, 6, 8, 8])            # last valid frame at/before each step
    assert np.isfinite(c).all()
    lead_bad = np.ones(10, bool)
    lead_bad[:3] = False                                            # nothing valid before → first valid
    a2 = _ramp(10).astype(np.float64)
    a2[:3] = np.nan
    c, v = action_chunk(a2, 0, horizon=3, offset=0, valid=lead_bad)
    np.testing.assert_array_equal(c[:, 0], [3, 3, 3])
    np.testing.assert_array_equal(v, [False, False, False])
    C, V = action_chunks(torch.as_tensor(a), np.arange(10), horizon=5, stride=2, valid=torch.as_tensor(src_valid))
    assert torch.isfinite(C).all() and V.dtype == torch.bool
    # plain lists are accepted
    c, v = action_chunk(_ramp(4).tolist(), 1, horizon=2)
    np.testing.assert_array_equal(np.asarray(c)[:, 0], [2, 3])


def test_action_chunks_vectorised_matches_loop_and_torch():
    a = np.random.default_rng(0).normal(size=(37, 5)).astype(np.float32)
    ts = np.array([0, 5, 20, 30, 36])
    C, V = action_chunks(a, ts, horizon=6, stride=3)
    assert C.shape == (5, 6, 5) and V.shape == (5, 6)
    for i, t in enumerate(ts):
        c, v = action_chunk(a, t, horizon=6, stride=3)
        np.testing.assert_array_equal(C[i], c)
        np.testing.assert_array_equal(V[i], v)
    Ct, Vt = action_chunks(torch.as_tensor(a), ts, horizon=6, stride=3)
    assert isinstance(Ct, torch.Tensor) and Vt.dtype == torch.bool
    np.testing.assert_array_equal(Ct.numpy(), C)
    np.testing.assert_array_equal(Vt.numpy(), V)


def test_policy_stride_and_ticks():
    assert policy_stride(200, 20) == 10
    assert policy_stride(200, 50) == 4
    with pytest.warns(UserWarning):
        assert policy_stride(200, 30) == 7
    with pytest.raises(ValueError):
        policy_stride(200, 400)
    np.testing.assert_array_equal(policy_tick_indices(25, 10), [0, 10, 20])
    np.testing.assert_array_equal(policy_tick_indices(25, 10, start=5), [5, 15])
    np.testing.assert_array_equal(policy_tick_indices(25, 10, min_future=10), [0, 10])
    mask = np.zeros(25, bool)
    mask[8:] = True
    np.testing.assert_array_equal(policy_tick_indices(25, 5, mask=mask), [10, 15, 20])


def _act_reference(chunks, k):
    """ACT's temporal aggregation (all_time_actions buffer), for one chunk per step."""
    n, H, A = chunks.shape
    buf = np.zeros((n, n + H, A))
    filled = np.zeros((n, n + H), bool)
    out = []
    for t in range(n):
        buf[t, t:t + H] = chunks[t]
        filled[t, t:t + H] = True
        acts = buf[:, t][filled[:, t]]                # rows ordered by production time: oldest first
        w = np.exp(-k * np.arange(len(acts)))
        w = w / w.sum()
        out.append((acts * w[:, None]).sum(0))
    return np.array(out)


def test_temporal_ensembler_hand_computed_and_reset():
    k = 0.5
    ens = TemporalEnsembler(horizon=3, action_dim=1, k=k)
    ens.add([[1.0], [2.0], [3.0]])
    assert ens.step()[0] == pytest.approx(1.0)
    ens.add([[10.0], [20.0], [30.0]])
    w = np.exp(-k * np.arange(2))
    assert ens.step()[0] == pytest.approx((2.0 * w[0] + 10.0 * w[1]) / w.sum())    # oldest weighs most
    ens.add([[100.0], [200.0], [300.0]])
    w = np.exp(-k * np.arange(3))
    assert ens.step()[0] == pytest.approx((3.0 * w[0] + 20.0 * w[1] + 100.0 * w[2]) / w.sum())
    # no new chunk: the first chunk has expired, the other two still cover step 3
    assert ens.n_chunks == 2
    w = np.exp(-k * np.arange(2))
    assert ens.step()[0] == pytest.approx((30.0 * w[0] + 200.0 * w[1]) / w.sum())
    assert ens.step()[0] == pytest.approx(300.0)
    assert not ens.ready
    with pytest.raises(RuntimeError):
        ens.step()
    ens.reset()
    assert ens.t == 0 and ens.n_chunks == 0
    ens.add([[5.0], [6.0], [7.0]])
    assert ens.step()[0] == pytest.approx(5.0)


def test_temporal_ensembler_matches_act_reference():
    rng = np.random.default_rng(4)
    chunks = rng.normal(size=(12, 5, 3))
    ref = _act_reference(chunks, k=0.01)
    ens = TemporalEnsembler(horizon=5, action_dim=3, k=0.01)
    got = []
    for c in chunks:
        ens.add(torch.as_tensor(c))
        got.append(ens.step())
    np.testing.assert_allclose(np.array(got), ref, atol=1e-12)
    # adding twice at the same step replaces the earlier chunk
    e2 = TemporalEnsembler(horizon=2, action_dim=1, k=0.0)
    e2.add([[1.0], [1.0]])
    e2.add([[3.0], [3.0]])
    assert e2.n_chunks == 1 and e2.step()[0] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        e2.add(np.zeros((3, 1)))
    with pytest.raises(ValueError):
        e2.add(np.zeros((2, 2)))
