import numpy as np
import pytest
import torch

from common.layouts import load_layout
from robot_skin.geometry.rotations import (
    aa_to_matrix, aa_to_quat, matrix_to_6d, quat_conj, quat_mul, quat_normalize, quat_to_matrix,
)
from robot_skin.pose import TaxelPoseProvider
from robot_skin.pose.glove_imu2mano import GloveImu2ManoPoseProvider, finetune_vifnet_s, load_vifnet_s
from robot_skin.pose.imu_model import (
    ImuHandPoseNet, apply_imu_offsets, apply_imu_offsets_to_vectors, calibrate_imu_offsets,
    estimate_world_alignment, hand_pose_loss, imu_calibration_from_dict, imu_calibration_to_dict,
    imu_feature_dim, imu_features, imu_reference_rotations, imu_site_quats, imu_windows, mean_quaternion,
    predict_finger_pose_sequence, rotation_geodesic, synthesize_imu, to_axis_angle,
)
from robot_skin.pose.mano import ManoSkeleton, taxel_poses_from_hand


def _rq(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return quat_normalize(torch.randn(*shape, 4, generator=g, dtype=torch.float64))


def _qang(a, b):
    """Angle between quaternions (sign-invariant)."""
    return rotation_geodesic(quat_to_matrix(torch.as_tensor(a)), quat_to_matrix(torch.as_tensor(b)))


def _motion(T=64, hz=100.0, seed=0):
    """Smooth synthetic hand motion: finger flexion sinusoids + slow global rotation."""
    sk = ManoSkeleton()
    t = np.arange(T) / hz
    g = np.random.default_rng(seed)
    ph = g.uniform(0, 2 * np.pi, 5)
    flex = 0.6 + 0.5 * np.sin(2 * np.pi * 1.3 * t[:, None] + ph)
    fp = sk.flexion_pose(flex).numpy()
    go = np.stack([0.3 * np.sin(2 * np.pi * 0.4 * t), 0.5 * t, 0.2 * np.ones_like(t)], -1)
    wp = np.stack([0.05 * np.sin(2 * np.pi * 0.5 * t), np.zeros_like(t), 0.02 * t], -1)
    return sk, t, go, fp, wp


# ── features ──────────────────────────────────────────────────────────────
def test_imu_features_shape_and_relative_orientation():
    q = _rq(5, 7, seed=1)
    gyro, acc = torch.randn(5, 7, 3, dtype=torch.float64), torch.randn(5, 7, 3, dtype=torch.float64)
    f = imu_features(q, gyro, acc)
    assert f.shape == (5, 7 * 12) and isinstance(f, torch.Tensor)
    assert imu_feature_dim(7) == 84 and imu_feature_dim(7, gyro=False) == 63
    per = f.reshape(5, 7, 12)
    Rw, Rs = quat_to_matrix(q[:, :1]), quat_to_matrix(q)
    torch.testing.assert_close(per[..., :6], matrix_to_6d(Rw.transpose(-1, -2) @ Rs))
    torch.testing.assert_close(per[:, 0, :6], torch.tensor([1.0, 0, 0, 0, 1.0, 0], dtype=torch.float64).expand(5, 6))
    torch.testing.assert_close(per[..., 6:9], (Rw.transpose(-1, -2) @ Rs @ gyro[..., None])[..., 0])
    fn = imu_features(q.numpy(), None, acc.numpy(), wrist_index=2)
    assert isinstance(fn, np.ndarray) and fn.dtype == np.float32 and fn.shape == (5, 63)
    with pytest.raises(ValueError):
        imu_features(q, gyro[:, :6])


def test_imu_features_invariant_to_common_global_rotation():
    q = _rq(3, 10, 7, seed=2)                                   # [B, W, S, 4]
    gyro = torch.randn(3, 10, 7, 3, dtype=torch.float64)
    acc = torch.randn(3, 10, 7, 3, dtype=torch.float64)
    G = _rq(seed=3)
    qg = quat_mul(G.expand_as(q), q)
    torch.testing.assert_close(imu_features(qg, gyro, acc), imu_features(q, gyro, acc))
    # world-frame vectors rotate with the frame too
    RG = quat_to_matrix(G)
    rot = lambda v: (RG @ v[..., None])[..., 0]  # noqa: E731
    torch.testing.assert_close(imu_features(qg, rot(gyro), rot(acc), vec_frame="world"),
                               imu_features(q, gyro, acc, vec_frame="world"))
    # but not to a rotation of a single site
    q2 = q.clone()
    q2[..., 3, :] = quat_mul(G.expand_as(q2[..., 3, :]), q2[..., 3, :])
    assert not torch.allclose(imu_features(q2, gyro, acc), imu_features(q, gyro, acc))


def test_imu_windows_are_causal():
    feat = np.arange(20, dtype=np.float32).reshape(10, 2)
    w = imu_windows(feat, 4)
    assert w.shape == (10, 4, 2)
    np.testing.assert_array_equal(w[:, -1], feat)
    np.testing.assert_array_equal(w[0], np.repeat(feat[:1], 4, 0))
    np.testing.assert_array_equal(w[6], feat[3:7])
    with pytest.raises(ValueError):
        imu_windows(feat, 0)


# ── calibration ───────────────────────────────────────────────────────────
def test_mean_quaternion_sign_invariant():
    q = _rq(seed=4).numpy()
    noisy = np.stack([q, -q, q, -q])
    m = mean_quaternion(noisy)
    assert float(_qang(m, q)) < 1e-9 and m[0] >= 0


def test_calibration_recovers_known_mounting_offset():
    S, T = 7, 50
    q_ref = _rq(S, seed=5)                                     # segment orientations at the calib pose
    M = _rq(S, seed=6)                                         # sensor mounting
    g = torch.Generator().manual_seed(7)
    noise = quat_normalize(torch.cat([torch.ones(T, S, 1, dtype=torch.float64),
                                      0.002 * torch.randn(T, S, 3, generator=g, dtype=torch.float64)], -1))
    q_meas = quat_mul(quat_mul(q_ref.expand(T, S, 4), M.expand(T, S, 4)), noise)
    q_meas[::2] = -q_meas[::2]                                 # sign flips must not matter
    off = calibrate_imu_offsets(q_meas.numpy(), quat_to_matrix(q_ref).numpy())
    assert off.shape == (S, 4)
    assert float(_qang(off, quat_conj(M).numpy()).max()) < 5e-3   # q_off = M⁻¹
    # a new pose is recovered exactly (up to the calibration noise)
    q_seg = _rq(S, seed=8)
    q_new = quat_mul(q_seg, M)
    rec = apply_imu_offsets(q_new.numpy(), off)
    assert float(_qang(rec, q_seg.numpy()).max()) < 5e-3
    # vectors: sensor frame → segment frame
    v_seg = np.random.default_rng(0).normal(size=(S, 3))
    RM = quat_to_matrix(M).numpy()
    v_sensor = np.einsum("sji,sj->si", RM, v_seg)               # R_Mᵀ v
    np.testing.assert_allclose(apply_imu_offsets_to_vectors(v_sensor, off), v_seg, atol=5e-2)
    # identity reference: calibrated frames read identity at the calibration pose
    off_id = calibrate_imu_offsets(q_meas.numpy())
    q_mean = mean_quaternion(q_meas.numpy(), axis=0)
    assert float(_qang(apply_imu_offsets(q_mean, off_id), np.tile([1.0, 0, 0, 0], (S, 1))).max()) < 1e-7
    with pytest.raises(ValueError):
        calibrate_imu_offsets(q_meas.numpy(), np.zeros((3, 3, 3)))


def test_world_frame_vectors_are_calibrated_with_the_world_alignment():
    """Devices reporting WORLD-frame gyro/acc (``vec_frame="world"``): calibration must re-express
    them in model world (``G⁻¹ v``) like the quaternions, not rotate them by the per-site mounting
    offsets (the sensor-frame rule) — otherwise ``imu_features(vec_frame="world")`` mixes frames and
    depends on the session's arbitrary IMU heading. Reference: ideal segment quaternions + ideal
    world-frame vectors."""
    lay = load_layout("glove_template")
    sk, t, go, fp, wp = _motion(T=64)
    ideal = synthesize_imu(lay, sk, t, go, fp, wp)
    T, S = ideal["quat"].shape[:2]
    R_seg = quat_to_matrix(torch.as_tensor(ideal["quat"])).numpy()
    gw = np.einsum("tsij,tsj->tsi", R_seg, ideal["gyro"])                       # model-world vectors
    aw = np.einsum("tsij,tsj->tsi", R_seg, ideal["acc"])
    f_ref = imu_features(ideal["quat"], gw, aw, 0, vec_frame="world")
    M = _rq(S, seed=11)
    M[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)            # wrist mount = I (as G's estimate needs)
    R_ref = imu_reference_rotations(lay, sk)
    q_cal = torch.as_tensor(imu_site_quats(lay, sk, np.zeros((10, 3)), np.zeros((10, 15, 3))))
    for heading in (0.0, 1.0, 2.5):
        G = aa_to_quat(torch.tensor([0.2, -0.1, heading], dtype=torch.float64))
        RG = quat_to_matrix(G).numpy()
        q_meas = quat_mul(quat_mul(G.expand(T, S, 4), torch.as_tensor(ideal["quat"])), M.expand(T, S, 4)).numpy()
        qc = quat_mul(quat_mul(G.expand(10, S, 4), q_cal), M.expand(10, S, 4)).numpy()
        Gh = estimate_world_alignment(qc, R_ref, index=0)
        off = calibrate_imu_offsets(qc, R_ref, world=Gh)
        kw = {"vec_frame": "world", "world": Gh}
        f = imu_features(apply_imu_offsets(q_meas, off, world=Gh), apply_imu_offsets_to_vectors(gw @ RG.T, off, **kw),
                         apply_imu_offsets_to_vectors(aw @ RG.T, off, **kw), 0, vec_frame="world")
        np.testing.assert_allclose(f, f_ref, atol=1e-4)                        # heading-independent, exact
    # without a world alignment world-frame vectors stay as they are; sensor-frame ones keep R_offᵀ v
    v = np.random.default_rng(1).normal(size=(5, S, 3))
    np.testing.assert_array_equal(apply_imu_offsets_to_vectors(v, off, vec_frame="world"), v)
    np.testing.assert_allclose(apply_imu_offsets_to_vectors(v, off, world=Gh),
                               np.einsum("sji,tsj->tsi", quat_to_matrix(torch.as_tensor(off)).numpy(), v))
    with pytest.raises(ValueError, match="vec_frame"):
        apply_imu_offsets_to_vectors(v, off, vec_frame="body")


def test_world_alignment_two_sided_calibration():
    S, T = 7, 20
    G = _rq(seed=9)
    q_ref = _rq(S, seed=10)
    M = _rq(S, seed=11)
    M[0] = torch.tensor([1.0, 0, 0, 0])                        # wrist sensor mounted aligned
    meas = lambda qs: quat_mul(quat_mul(G.expand_as(qs), qs), M)  # noqa: E731
    q_cal = meas(q_ref).expand(T, S, 4).numpy()
    Ghat = estimate_world_alignment(q_cal, quat_to_matrix(q_ref).numpy(), index=0)
    assert float(_qang(Ghat, G.numpy())) < 1e-7
    off = calibrate_imu_offsets(q_cal, quat_to_matrix(q_ref).numpy(), world=Ghat)
    q_seg = _rq(S, seed=12)
    rec = apply_imu_offsets(meas(q_seg).numpy(), off, world=Ghat)
    assert float(_qang(rec, q_seg.numpy()).max()) < 1e-7


def test_calibration_dict_roundtrip():
    import json
    off = _rq(3, seed=14).numpy()
    d = json.loads(json.dumps(imu_calibration_to_dict(off, world=[1.0, 0, 0, 0], sites=["wrist", "palm", "thumb"])))
    o2, w2 = imu_calibration_from_dict(d, sites=["thumb", "wrist", "palm"])
    np.testing.assert_allclose(o2, off[[2, 0, 1]])
    np.testing.assert_allclose(w2, [1.0, 0, 0, 0])
    assert imu_calibration_from_dict({}) == (None, None)
    with pytest.raises(ValueError):
        imu_calibration_from_dict(d, sites=["wrist", "index"])
    with pytest.raises(ValueError):
        imu_calibration_from_dict(imu_calibration_to_dict(off), sites=["a", "b"])


# ── synthetic IMU ─────────────────────────────────────────────────────────
def test_synthesize_imu_static_and_rotating():
    L, sk = load_layout("glove_template"), ManoSkeleton()
    T, hz = 50, 100.0
    t = np.arange(T) / hz
    fp = np.zeros((T, 15, 3))
    go = np.tile([0.2, -0.4, 0.1], (T, 1))
    imu = synthesize_imu(L, sk, t, go, fp)
    assert imu["quat"].shape == (T, 7, 4) and list(imu["sites"]) == [s.name for s in L.imu_sites]
    np.testing.assert_allclose(imu["gyro"], 0.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(imu["acc"], axis=-1), 9.81, atol=1e-9)
    Rw = aa_to_matrix(torch.as_tensor(go[0])).numpy()
    np.testing.assert_allclose(imu["acc"][0, 0], Rw.T @ [0, 0, 9.81], atol=1e-9)   # wrist site
    np.testing.assert_allclose(imu["quat"], imu_site_quats(L, sk, go, fp), atol=1e-12)
    # constant yaw rate about world z → wrist body rate = R0ᵀ (0, 0, w)
    w = 1.5
    R0 = aa_to_matrix(torch.tensor([0.3, 0.2, -0.1], dtype=torch.float64))
    Rz = aa_to_matrix(torch.as_tensor(np.stack([np.zeros(T), np.zeros(T), w * t], -1)))
    from robot_skin.geometry.rotations import matrix_to_aa
    go_rot = matrix_to_aa(Rz @ R0).numpy()
    imu = synthesize_imu(L, sk, t, go_rot, fp)
    np.testing.assert_allclose(imu["gyro"][5:-5, 0], np.tile(R0.numpy().T @ [0, 0, w], (T - 10, 1)), atol=1e-6)


# ── network + loss ────────────────────────────────────────────────────────
def test_hand_pose_loss_zero_for_identical_and_positive_otherwise():
    sk = ManoSkeleton()
    gt = torch.randn(4, 15, 3, dtype=torch.float64) * 0.5
    pred6d = matrix_to_6d(aa_to_matrix(gt)).clone().requires_grad_(True)
    out = hand_pose_loss(pred6d, gt, skeleton=sk, tip_weight=10.0)
    assert set(out) >= {"loss", "rot", "tip"}
    assert float(out["loss"].detach()) < 1e-6
    out["loss"].backward()
    assert torch.isfinite(pred6d.grad).all()
    other = matrix_to_6d(aa_to_matrix(gt + 0.3))
    o2 = hand_pose_loss(other, gt, skeleton=sk)
    assert float(o2["rot"]) > 0.1 and float(o2["tip"]) > 1e-3
    # valid mask: only invalid samples wrong → zero loss
    mixed = matrix_to_6d(aa_to_matrix(gt)).detach().clone()
    mixed[1] = other[1]
    o3 = hand_pose_loss(mixed, gt, valid=torch.tensor([True, False, True, True]))
    assert float(o3["loss"]) < 1e-6
    go = torch.randn(4, 3, dtype=torch.float64)
    o4 = hand_pose_loss(pred6d.detach(), gt, pred_global6d=matrix_to_6d(aa_to_matrix(go)), gt_global=go)
    assert float(o4["global"]) < 1e-6


def test_rotation_geodesic_and_to_axis_angle():
    aa = torch.randn(32, 3, dtype=torch.float64)
    aa = aa / aa.norm(dim=-1, keepdim=True) * torch.linspace(0.01, 3.0, 32, dtype=torch.float64)[:, None]
    R = aa_to_matrix(aa)
    I = torch.eye(3, dtype=torch.float64).expand_as(R)
    torch.testing.assert_close(rotation_geodesic(I, R), aa.norm(dim=-1))
    torch.testing.assert_close(aa_to_matrix(to_axis_angle(matrix_to_6d(R))), R)


def _dataset(window=8):
    L, (sk, t, go, fp, wp) = load_layout("glove_template"), _motion()
    imu = synthesize_imu(L, sk, t, go, fp, wp)
    feat = imu_features(imu["quat"], imu["gyro"], imu["acc"])
    X = torch.as_tensor(np.ascontiguousarray(imu_windows(feat, window)))
    return X, torch.as_tensor(fp), sk, imu


@pytest.fixture
def one_thread():
    """Small RNN training is fastest (and robust to CPU contention) single-threaded."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.mark.parametrize("arch", ["gru", "tcn"])
def test_imu_hand_pose_net_overfits_tiny_dataset(arch, one_thread):
    torch.manual_seed(0)
    X, Y, sk, _ = _dataset()
    net = ImuHandPoseNet(X.shape[-1], hidden=48, n_layers=2, arch=arch, dropout=0.0)
    net.set_feature_stats(X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0))
    out0 = net(X)
    assert out0.shape == (X.shape[0], 15, 6)
    torch.testing.assert_close(out0, net.identity6d.expand_as(out0))       # zero-init → flat hand
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    losses = []
    for _ in range(80):
        loss = hand_pose_loss(net(X), Y, skeleton=sk, tip_weight=10.0)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.3 * losses[0]
    pred = net.predict(X[:3])
    assert pred["finger_pose"].shape == (3, 15, 3)
    assert net.forward_all(X[:2], all_steps=True)["finger_rot6d"].shape == (2, X.shape[1], 15, 6)


def test_net_config_roundtrip_and_global_head():
    net = ImuHandPoseNet(84, hidden=32, n_layers=1, predict_global=True)
    net2 = ImuHandPoseNet.from_config(net.config)
    net2.load_state_dict(net.state_dict())
    out = net2.predict(torch.zeros(2, 5, 84))
    assert out["global_rot6d"].shape == (2, 6) and out["global_orient"].shape == (2, 3)
    with pytest.raises(ValueError):
        net(torch.zeros(2, 5, 80))
    with pytest.raises(ValueError):
        ImuHandPoseNet(84, arch="lstm")


def test_predict_sequence_and_glove_provider():
    L = load_layout("glove_template")
    sk, t, go, fp, wp = _motion(T=40)
    imu = synthesize_imu(L, sk, t, go, fp, wp)
    net = ImuHandPoseNet(imu_feature_dim(7), hidden=16, n_layers=1)          # untrained: flat hand
    seq = predict_finger_pose_sequence(net, imu["quat"], imu["gyro"], imu["acc"], window=6, batch_size=7)
    assert seq["finger_pose"].shape == (40, 15, 3)
    np.testing.assert_allclose(seq["finger_pose"], 0.0, atol=1e-6)
    with pytest.raises(ValueError):
        predict_finger_pose_sequence(net, imu["quat"], None, imu["acc"], window=6)

    # mounting offsets on the raw IMU, recovered by calibration at frame 0
    M = _rq(7, seed=13).numpy()
    q_raw = quat_mul(torch.as_tensor(imu["quat"]), torch.as_tensor(M).expand(40, 7, 4)).numpy()
    RM = quat_to_matrix(torch.as_tensor(M)).numpy()
    to_sensor = lambda v: np.einsum("sji,tsj->tsi", RM, v)  # noqa: E731
    ref = quat_to_matrix(torch.as_tensor(imu["quat"][0])).numpy()
    off = calibrate_imu_offsets(q_raw[:1], ref)
    prov = GloveImu2ManoPoseProvider(L, net, sk, t, q_raw, to_sensor(imu["gyro"]), to_sensor(imu["acc"]),
                                     window=6, offsets=off, wrist_pos=wp)
    assert isinstance(prov, TaxelPoseProvider) and prov.n_taxels == L.n
    R_est = aa_to_matrix(torch.as_tensor(prov.global_orient))
    assert float(rotation_geodesic(R_est, aa_to_matrix(torch.as_tensor(go))).max()) < 1e-6
    P, N = taxel_poses_from_hand(L, sk, go, np.zeros_like(fp), wp)
    np.testing.assert_allclose(prov.poses[0], P, atol=1e-6)
    pos, nrm = prov.pose_at(float(t[3]))
    np.testing.assert_allclose(pos, P[3], atol=1e-6)
    mid, _ = prov.pose_at(float(0.5 * (t[3] + t[4])))
    np.testing.assert_allclose(mid, 0.5 * (P[3] + P[4]), atol=1e-6)
    np.testing.assert_allclose(prov.pose_at(-1.0)[0], P[0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(nrm, axis=-1), 1.0, atol=1e-9)

    with pytest.raises(ValueError):
        GloveImu2ManoPoseProvider(load_layout("robot_hand_template"), net, sk, t, q_raw)
    with pytest.raises(ValueError):
        GloveImu2ManoPoseProvider(L, net, sk, t, q_raw[:, :5])


def test_vifnet_stubs():
    with pytest.raises(NotImplementedError):
        load_vifnet_s("vifnet_s.pt")
    with pytest.raises(NotImplementedError):
        finetune_vifnet_s([], "out")


def test_flat_hand_calibration_pipeline_with_reference_rotations():
    """Raw IMU = G ⊗ q_segment ⊗ M (unknown IMU world G, mounting M, wrist M = I), calibration
    recorded in a flat-hand pose held at an unknown global orientation: calibrated features equal
    the ideal ones, and calibrated orientations equal the true ones re-anchored at the calibration
    heading."""
    L = load_layout("glove_template")
    sk, t, go, fp, wp = _motion(T=30)
    ideal = synthesize_imu(L, sk, t, go, fp, wp)
    S = ideal["quat"].shape[1]
    G = _rq(seed=20)
    M = _rq(S, seed=21)
    M[0] = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)                 # wrist strap aligned
    raw = lambda q: quat_mul(quat_mul(G.expand_as(q), q), M.expand_as(q))  # noqa: E731
    go_cal = np.array([0.4, -1.1, 0.3])                                        # unknown hand heading
    q_cal_seg = torch.as_tensor(imu_site_quats(L, sk, go_cal, np.zeros((15, 3))))
    q_cal = raw(q_cal_seg.expand(10, S, 4)).numpy()
    R_ref = imu_reference_rotations(L, sk)
    assert R_ref.shape == (S, 3, 3)
    np.testing.assert_allclose(R_ref @ np.swapaxes(R_ref, -1, -2), np.broadcast_to(np.eye(3), (S, 3, 3)), atol=1e-12)
    Ghat = estimate_world_alignment(q_cal, R_ref, index=0)
    off = calibrate_imu_offsets(q_cal, R_ref, world=Ghat)
    q_rec = apply_imu_offsets(raw(torch.as_tensor(ideal["quat"])).numpy(), off, world=Ghat)
    np.testing.assert_allclose(imu_features(q_rec), imu_features(ideal["quat"]), atol=1e-5)
    q_go_cal = aa_to_quat(torch.as_tensor(go_cal))
    q_anchor = quat_mul(quat_conj(q_go_cal).expand(len(t), S, 4), torch.as_tensor(ideal["quat"]))
    assert float(_qang(q_rec, q_anchor.numpy()).max()) < 1e-7
    # negative index = counted from the end; bad index / ref shape rejected
    np.testing.assert_allclose(estimate_world_alignment(q_cal[:, ::-1], R_ref[::-1], index=-1), Ghat, atol=1e-12)
    with pytest.raises(ValueError):
        estimate_world_alignment(q_cal, R_ref, index=S)
    with pytest.raises(ValueError):
        estimate_world_alignment(q_cal, R_ref[:3])


def test_hand_pose_loss_broadcasts_valid_over_steps_and_checks_shapes():
    g = torch.Generator().manual_seed(0)
    gt = torch.randn(3, 4, 15, 3, generator=g, dtype=torch.float64) * 0.4    # [B, W, 15, 3]
    pred = matrix_to_6d(aa_to_matrix(gt)).clone()
    pred[1] = matrix_to_6d(aa_to_matrix(gt[1] + 0.5))                        # only sample 1 wrong
    out = hand_pose_loss(pred, gt, skeleton=ManoSkeleton(), valid=torch.tensor([True, False, True]))
    assert float(out["loss"]) < 1e-6
    assert float(hand_pose_loss(pred, gt, valid=torch.tensor([0.0, 1.0, 0.0]))["rot"]) > 0.1
    with pytest.raises(ValueError):
        hand_pose_loss(pred, gt[..., :14, :])
    with pytest.raises(ValueError):
        hand_pose_loss(pred, gt, valid=torch.ones(4))


def test_net_accepts_float64_features():
    net = ImuHandPoseNet(imu_feature_dim(2), hidden=8, n_layers=1)
    out = net(torch.zeros(2, 3, imu_feature_dim(2), dtype=torch.float64))
    assert out.dtype == torch.float32 and out.shape == (2, 15, 6)


def test_glove_provider_world_alignment_without_offsets():
    L = load_layout("glove_template")
    sk, t, go, fp, wp = _motion(T=12)
    imu = synthesize_imu(L, sk, t, go, fp, wp)
    G = _rq(seed=30)
    q_raw = quat_mul(G.expand(12, 7, 4), torch.as_tensor(imu["quat"])).numpy()
    net = ImuHandPoseNet(imu_feature_dim(7, gyro=False, acc=False), hidden=8, n_layers=1)
    prov = GloveImu2ManoPoseProvider(L, net, sk, t, q_raw, window=4, world=G.numpy())
    R_est = aa_to_matrix(torch.as_tensor(prov.global_orient))
    assert float(rotation_geodesic(R_est, aa_to_matrix(torch.as_tensor(go))).max()) < 1e-6
    with pytest.raises(ValueError, match="argument order"):
        GloveImu2ManoPoseProvider(L, net, sk, q_raw, t, window=4)
