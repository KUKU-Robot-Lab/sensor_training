"""datasets.splits (leakage-safe grouping) and datasets.stats (masked train-split NormStats)."""
import json

import numpy as np
import pytest

from common.signal import NormStats
from robot_skin.datasets import episode as E
from robot_skin.datasets import splits as SP
from robot_skin.datasets import stats as ST
from robot_skin.pose.imu_model import imu_features

N, D, S = 4, 6, 3


def _ep(root, eid, subject="s0", task=None, obj=None, dataset="task", T=60, seed=0, kind="glove", save=True):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 200.0
    delta = rng.normal(0.0, 1.0, (T, N)).astype(np.float32)
    sat = np.zeros((T, N), bool)
    sat[5:8, 1] = True
    delta[sat] = -100.0                                           # dropouts must not reach the stats
    valid = np.ones(T, bool)
    valid[10:15] = False
    q = rng.normal(0.5, 2.0, (T, D)).astype(np.float32)
    q[~valid] = 1e3                                               # held garbage on invalid hand frames
    quat = rng.normal(size=(T, S, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    arrays = {E.K_T: t, E.K_DELTA: delta, E.K_SATURATED: sat, E.K_Q: q, E.K_QD: q * 0.1,
              E.K_HAND_VALID: valid, E.K_HAND_FINGERS: rng.normal(size=(T, 15, 3)).astype(np.float32),
              E.K_IMU_QUAT: quat.astype(np.float32), E.K_IMU_GYRO: rng.normal(size=(T, S, 3)).astype(np.float32),
              E.K_IMU_ACC: rng.normal(size=(T, S, 3)).astype(np.float32)}
    task_d = None if task is None and obj is None else {"task_id": task, "object": obj, "instruction": "x"}
    meta = E.EpisodeMeta(episode_id=eid, dataset=dataset, kind=kind, layout="glove_template", n_taxels=N,
                         subject=subject, task=task_d, imu_sites=["wrist", "a", "b"],
                         source_session=f"/raw/{eid}",
                         preprocessing={"q_source": "hand_pose", "imu": {"wrist_index": 0}})
    ep = E.Episode(meta, arrays)
    if save:
        ep.save(root / dataset / eid)
    return ep


@pytest.fixture()
def corpus(tmp_path):
    """12 D2 episodes: 4 subjects × 3 (tasks/objects cycle) + 3 D1 episodes without a task."""
    tasks, objs = ["pour", "wipe", "grasp"], ["cup", "sponge", "ball", "block"]
    for i in range(12):
        _ep(tmp_path, f"t{i:02d}", subject=f"s{i // 3}", task=tasks[i % 3], obj=objs[i % 4], seed=i)
    for i in range(3):
        _ep(tmp_path, f"m{i}", subject=f"s{i}", dataset="motion", seed=100 + i)
    return tmp_path, E.list_episodes(tmp_path)


def _group_of(p, by):
    return SP.episode_group(p, by)


def test_splits_by_subject_are_disjoint_deterministic_and_sized(corpus):
    root, eps = corpus
    s = SP.make_splits(eps, by="subject", val_frac=0.2, test_frac=0.2, seed=0)
    assert sorted(sum(s.values(), [])) == sorted(str(p) for p in eps)
    groups = {k: {_group_of(p, "subject") for p in v} for k, v in s.items()}
    assert all(groups[a].isdisjoint(groups[b]) for a in SP.SPLITS for b in SP.SPLITS if a < b)
    assert all(len(groups[k]) >= 1 for k in SP.SPLITS)
    SP.check_splits(s, "subject")
    # order-independent and seeded
    assert SP.make_splits(list(reversed(eps)), by="subject", val_frac=0.2, test_frac=0.2, seed=0) == s
    others = [SP.make_splits(eps, by="subject", val_frac=0.2, test_frac=0.2, seed=k) for k in range(1, 6)]
    assert any(o != s for o in others)
    # fractions steer the episode counts (groups are atomic)
    s2 = SP.make_splits(eps, by="session", val_frac=0.2, test_frac=0.2, seed=0)
    assert [len(s2[k]) for k in SP.SPLITS] == [9, 3, 3]
    no_val = SP.make_splits(eps, by="subject", val_frac=0.0, test_frac=0.3, seed=0)
    assert no_val["val"] == [] and no_val["test"] and no_val["train"]


def test_splits_keep_a_dominant_group_in_train(tmp_path):
    """One subject with 12 of 18 episodes: the greedy assignment must not put it into val/test
    (it did for 2 of 6 seeds before the first-group size cap)."""
    for i in range(12):
        _ep(tmp_path, f"big{i:02d}", subject="BIG", T=8, seed=i)
    for i in range(6):
        _ep(tmp_path, f"s{i}", subject=f"S{i}", T=8, seed=50 + i)
    eps = E.list_episodes(tmp_path)
    for seed in range(8):
        s = SP.make_splits(eps, by="subject", val_frac=0.15, test_frac=0.15, seed=seed)
        SP.check_splits(s, "subject")
        assert {SP.episode_group(p, "subject") for p in s["train"]} >= {"BIG"}
        assert len(s["val"]) == len(s["test"]) == 3                   # 3 singles ≈ 0.15 · 18 = 2.7


def test_splits_by_object_task_and_missing_keys(corpus):
    root, eps = corpus
    for by in ("object", "task", ("subject", "task")):
        s = SP.make_splits(eps, by=by, val_frac=0.25, test_frac=0.25, seed=3)
        SP.check_splits(s, by)
    # D1 episodes have no object → one group per episode, never merged with each other
    assert SP.episode_group(root / "motion" / "m0", "object") == "episode:m0"
    assert SP.episode_group(root / "task" / "t04", ("subject", "task")) == "s1|wipe"
    with pytest.raises(ValueError, match="unknown group key"):
        SP.episode_group(root / "task" / "t00", "colour")


def test_holdout_forms_and_leak_detection(corpus):
    root, eps = corpus
    s = SP.make_splits(eps, by="subject", val_frac=0.2, test_frac=0.0, seed=0, holdout={"subject": ["s3"]})
    assert {SP.episode_group(p, "subject") for p in s["test"]} == {"s3"} and len(s["test"]) == 3
    s = SP.make_splits(eps, by="subject", val_frac=0.0, test_frac=0.0, seed=0,
                       holdout={"test": {"task": "pour"}, "val": {"object": ["ball"]}})
    assert all(json.loads((E.Path(p) / E.EPISODE_JSON).read_text())["task"]["task_id"] == "pour" for p in s["test"])
    assert all(json.loads((E.Path(p) / E.EPISODE_JSON).read_text())["task"]["object"] == "ball" for p in s["val"])
    assert len(s["test"]) == 4 and s["train"]
    with pytest.raises(ValueError):                              # subject crosses train / test by design here
        SP.check_splits(s, "subject")
    SP.check_splits(s, "subject", ignore=s["test"] + s["val"])
    with pytest.raises(ValueError, match="unknown holdout key"):
        SP.make_splits(eps, holdout={"colour": ["red"]})
    with pytest.raises(ValueError):
        SP.make_splits(eps, val_frac=0.6, test_frac=0.5)
    with pytest.raises(ValueError, match="both"):
        SP.check_splits({"train": [str(eps[0])], "test": [str(eps[0])]}, "subject")


def test_save_load_splits_relative_roundtrip(corpus, tmp_path_factory):
    root, eps = corpus
    s = SP.make_splits(eps, by="subject", seed=1)
    p = SP.save_splits(s, root / "splits.json", root=root, meta={"by": "subject", "seed": 1})
    d = json.loads(p.read_text())
    assert d["relative"] and all(not x.startswith("/") for x in d["train"]) and d["meta"]["seed"] == 1
    loaded = SP.load_splits(p)
    assert {k: [str(x) for x in v] for k, v in loaded.items()} == s
    other = tmp_path_factory.mktemp("elsewhere")
    assert SP.load_splits(p, root=other)["train"][0].parent.parent == other
    SP.save_splits(s, root / "abs.json")
    assert {k: [str(x) for x in v] for k, v in SP.load_splits(root / "abs.json").items()} == s


def test_stats_std_masks_match_manual_fit(corpus):
    root, eps = corpus
    train = eps[:5]
    st = ST.compute_stats(train, keys=(E.K_DELTA, E.K_Q, E.K_QD, E.K_HAND_FINGERS))
    loaded = ST.as_episodes(train)
    d = np.concatenate([np.asarray(e[E.K_DELTA], np.float64) for e in loaded])
    sat = np.concatenate([np.asarray(e[E.K_SATURATED]) for e in loaded])
    for n in range(N):
        x = d[~sat[:, n], n]
        np.testing.assert_allclose(st[E.K_DELTA].offset[n], x.mean(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(st[E.K_DELTA].scale[n], x.std() + 1e-6, rtol=1e-5)
    assert st[E.K_DELTA].scale.max() < 2.0                         # the −100 % dropouts are excluded
    valid = np.concatenate([np.asarray(e[E.K_HAND_VALID]) for e in loaded])
    q = np.concatenate([np.asarray(e[E.K_Q]) for e in loaded])[valid]
    ref = NormStats.fit(q)
    np.testing.assert_allclose(st[E.K_Q].offset, ref.offset, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(st[E.K_Q].scale, ref.scale, rtol=1e-5)
    assert st[E.K_HAND_FINGERS].offset.shape == (45,)
    # glove qd: hand_pose_valid eroded by the causal SG footprint (11 frames at 200 Hz) — frames
    # 10..14 invalid → qd invalid on 10..25 (the derivative of a held label's jump)
    qdv = ST.qd_valid_mask(loaded[0])
    np.testing.assert_array_equal(np.flatnonzero(~qdv), np.arange(10, 26))
    qd = np.concatenate([np.asarray(e[E.K_QD]) for e in loaded])[np.concatenate([ST.qd_valid_mask(e) for e in loaded])]
    np.testing.assert_allclose(st[E.K_QD].offset, NormStats.fit(qd).offset, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(st[E.K_QD].scale, NormStats.fit(qd).scale, rtol=1e-5)
    # masks=None → plain NormStats.fit over everything (q includes the held garbage)
    raw = ST.compute_stats(train, keys=(E.K_Q,), masks=None)[E.K_Q]
    allq = np.concatenate([np.asarray(e[E.K_Q]) for e in loaded])
    np.testing.assert_allclose(raw.offset, NormStats.fit(allq).offset, rtol=1e-5)
    # robot episodes: q is a measurement on every frame (no hand mask)
    robot = _ep(root, "r0", kind="robot", save=False)
    robot.meta.preprocessing = {"q_source": "joint_state"}
    assert ST.default_mask(robot, E.K_Q) is None and ST.default_mask(robot, E.K_DELTA).shape == (60, N)


def test_stats_robust_none_imu_features_derived_and_io(corpus, tmp_path):
    root, eps = corpus
    train = ST.as_episodes(eps[:4])
    rb = ST.compute_stats(train, keys=(E.K_DELTA,), method="robust")[E.K_DELTA]
    d = np.concatenate([np.asarray(e[E.K_DELTA], np.float64) for e in train])
    sat = np.concatenate([np.asarray(e[E.K_SATURATED]) for e in train])
    x = d[~sat[:, 1], 1]
    q75, q25 = np.percentile(x, [75, 25])
    np.testing.assert_allclose(rb.offset[1], np.median(x), rtol=1e-5)
    np.testing.assert_allclose(rb.scale[1], (q75 - q25) / 1.349 + 1e-6, rtol=1e-5)
    nn = ST.compute_stats(train, keys=(E.K_Q,), method="none")[E.K_Q]
    assert (nn.offset == 0).all() and np.allclose(nn.scale, 1.0)
    # IMU features: the same function the datasets use
    fi = ST.compute_stats(train, keys=(ST.IMU_FEATURES,))[ST.IMU_FEATURES]
    feats = np.concatenate([imu_features(np.array(e[E.K_IMU_QUAT]), np.array(e[E.K_IMU_GYRO]),
                                         np.array(e[E.K_IMU_ACC]), 0) for e in train])
    ref = NormStats.fit(feats)
    np.testing.assert_allclose(fi.offset, ref.offset, atol=1e-5)
    np.testing.assert_allclose(fi.scale, ref.scale, rtol=1e-4, atol=1e-6)
    fi_q = ST.compute_stats(train, keys=(ST.IMU_FEATURES,), imu_kw={"gyro": False, "acc": False})
    assert fi_q[ST.IMU_FEATURES].offset.shape == (S * 6,)
    # derived keys + callable masks + empty features
    for e in train:
        e.set_derived(E.D_RESIDUAL_Z, np.full((e.T, N), 2.0, np.float32))
    with pytest.warns(UserWarning, match="no valid samples"):
        rz = ST.compute_stats(train, keys=(E.D_RESIDUAL_Z,), masks={E.D_RESIDUAL_Z: lambda ep: np.zeros(ep.T, bool)})
    assert np.allclose(rz[E.D_RESIDUAL_Z].scale, 1.0 + 1e-6) and (rz[E.D_RESIDUAL_Z].offset == 0).all()
    with pytest.raises(KeyError):
        ST.compute_stats(train, keys=("nope",))
    with pytest.raises(ValueError):
        ST.compute_stats(train, method="minmax")
    # apply / invert over any leading shape; save / load
    st = ST.compute_stats(train, keys=(E.K_HAND_FINGERS, E.K_Q))
    fp = np.asarray(train[0][E.K_HAND_FINGERS])
    z = ST.apply_stats(st[E.K_HAND_FINGERS], fp)
    assert z.shape == fp.shape and abs(float(z.mean())) < 0.5
    np.testing.assert_allclose(ST.invert_stats(st[E.K_HAND_FINGERS], z), fp, atol=1e-4)
    qh = np.asarray(train[0][E.K_Q])[:8].reshape(2, 4, D)
    np.testing.assert_allclose(ST.apply_stats(st[E.K_Q], qh), st[E.K_Q].apply(qh.reshape(8, D)).reshape(2, 4, D))
    with pytest.raises(ValueError):
        ST.apply_stats(st[E.K_Q], np.zeros((3, 5)))
    p = ST.save_stats(st, tmp_path / "stats" / "joint.json", meta={"split": "train", "n_episodes": 4})
    back, meta = ST.load_stats(p, return_meta=True)
    assert meta["n_episodes"] == 4 and set(back) == set(st)
    np.testing.assert_array_equal(back[E.K_Q].scale, st[E.K_Q].scale)
