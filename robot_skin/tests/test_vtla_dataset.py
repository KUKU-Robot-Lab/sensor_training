"""VTLA dataset on tiny synthetic D2 episodes (datasets.synthetic → datasets.build) and the
``vtla`` stage end-to-end (policy_bundle.pt reproduces the trained policy)."""
import json
import logging
import warnings

import numpy as np
import pytest
import torch
import yaml

from robot_skin.action import ActionSpec, hand_action_from_episode, make_absolute
from robot_skin.contact.ordinal import ContactLevel
from robot_skin.datasets.episode import (D_LEVEL, D_RESIDUAL_Z, K_CONTACT_LABEL, K_HAND_VALID,
                                         Episode, cam_idx_key)
from robot_skin.representation import TactileFeatureSpec
from robot_skin.vision import EvalTransform, TinyConvEncoder
from robot_skin.vision.feature_cache import cache_episode_features, gather_frame_features, load_cached
from robot_skin.vtla import (PSEUDO_LABEL_KEY, TASK_PHASES, VTLACollator, VTLAConfig, VTLADataset, VTLAPolicy,
                             bootstrap_tactile_arrays, build_policy_from_bundle, bundle_components,
                             collate_vtla, contact_from_level, eval_transform_from_dict,
                             eval_transform_to_dict, history_ticks, read_policy_bundle,
                             sample_phase_mask)

HW = (24, 32)


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """3 glove D2 episodes (cameras ego + third) + 1 robot D2 episode, preprocessed."""
    from robot_skin.datasets.build import build_all
    from robot_skin.datasets.synthetic import generate_dataset, generate_session

    root = tmp_path_factory.mktemp("vtla_data")
    generate_dataset(root / "raw", n_motion=0, n_task=3, kind="glove", duration_s=3.0,
                     cameras=("ego", "third"))
    generate_session(root / "raw_robot" / "task" / "s0" / "robot_task", kind="robot", dataset="task",
                     duration_s=3.0, seed=7, cameras=("ego",))
    res = build_all([root / "raw", root / "raw_robot"], root / "proc", {"baseline": {"duration_s": 0.3}})
    assert all(r["status"] == "built" for r in res), res
    eps = [Episode.load(r["episode"]) for r in res]
    glove = sorted([e for e in eps if e.meta.kind == "glove"], key=lambda e: e.meta.episode_id)
    robot = [e for e in eps if e.meta.kind == "robot"]
    return {"root": root, "proc": root / "proc", "glove": glove, "robot": robot}


def _ds(eps, **kw):
    base = dict(cameras=("ego", "third"), horizon=8, image_transform=EvalTransform(HW))
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VTLADataset(eps, **base)


def _mem_copy(ep):
    """Writable in-memory copy (no root → no camera frames)."""
    return Episode(ep.meta, {k: np.array(v) for k, v in ep.arrays.items()}, dict(ep.static))


# ─────────────────────────────────────────────────────────── basics

def test_task_phases_match_protocol():
    from robot_skin.acquisition.protocol import TASK_PHASES as ACQ_TASK_PHASES
    assert tuple(TASK_PHASES) == tuple(ACQ_TASK_PHASES)


def test_pseudo_label_key_matches_contact_stage():
    try:
        from robot_skin.contact.pseudo_label import D_CONTACT_LABEL_PSEUDO
    except ImportError:          # contact stage not installed in this checkout
        pytest.skip("robot_skin.contact.pseudo_label unavailable")
    assert PSEUDO_LABEL_KEY == D_CONTACT_LABEL_PSEUDO


def test_aux_targets(data):
    ep = _mem_copy(data["glove"][0])
    ds = _ds([ep], cameras=(), aux_target="label")
    s = ds[5]
    lab = np.asarray(ep[K_CONTACT_LABEL][s["t_index"]])
    assert torch.equal(s["contact_target_mask"], torch.from_numpy(lab >= 0))
    pseudo = np.ones((ep.T, ep.meta.n_taxels), np.int8)            # the contact stage's D2 labels win
    pseudo[:, 0] = -1
    ep.set_derived(PSEUDO_LABEL_KEY, pseudo, save=False)
    s = _ds([ep], cameras=(), aux_target="label")[5]
    assert s["contact_target"][1:].eq(1).all() and not s["contact_target_mask"][0]
    g = _ds([ep], cameras=(), aux_target="gt")[5]
    assert torch.equal(g["contact_target"].bool(), torch.from_numpy(np.asarray(ep["gt_contact"][g["t_index"]])))
    lv = _ds([ep], cameras=(), aux_target="level")[5]
    assert torch.equal(lv["contact_target"].bool() & lv["contact_target_mask"],
                       lv["contact"] & lv["contact_target_mask"])


def test_history_ticks_and_transform_dict():
    assert history_ticks(25, 3, 10).tolist() == [5, 15, 25]
    assert history_ticks(5, 3, 10).tolist() == [0, 0, 5]
    tf = EvalTransform((12, 16), crop_scale=0.9, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
    d = eval_transform_to_dict(tf)
    tf2 = eval_transform_from_dict(d)
    x = np.random.default_rng(0).integers(0, 255, (2, 24, 32, 3), dtype=np.uint8)
    assert torch.allclose(tf(x), tf2(x))
    assert eval_transform_from_dict(None) is None


def test_dataset_shapes_ticks_and_bootstrap_warning(data):
    eps = data["glove"]
    with pytest.warns(UserWarning, match="BOOTSTRAP"):
        ds = VTLADataset(eps, cameras=("ego", "third"), horizon=8, image_transform=EvalTransform(HW))
    assert ds.stride == 10 and ds.tactile_source == "bootstrap" and len(ds) > 0
    for e, t in ds.index:                                      # ticks: policy grid ∩ task phases
        ep = eps[e]
        assert t % ds.stride == 0
        assert ep.meta.phase_names[int(ep["phase_id"][t])] in TASK_PHASES
    s = ds[3]
    N = eps[0].meta.n_taxels
    assert s["proprio"].shape == (54,) and s["actions"].shape == (8, 54)
    assert s["action_valid"].dtype == torch.bool and s["action_valid"].shape == (8,)
    assert s["tactile_values"].shape == (N, 6) and s["taxel_pos"].shape == (N, 3)
    assert s["contact"].dtype == torch.bool and s["contact"].shape == (N,)
    assert set(s["images"]) == {"ego", "third"} and s["images"]["ego"].shape == (3, *HW)
    assert s["vision_valid"]["ego"].shape == (1,)
    assert isinstance(s["instruction"], str) and s["instruction"]
    assert s["task_id"] == eps[int(ds.index[3, 0])].meta.task["task_id"]
    # baseline frames are never sampled with the default phases; "all" adds them
    all_ds = _ds(eps, phases="all")
    assert len(all_ds) > len(ds)
    m = sample_phase_mask(eps[0], ["grasp"])
    assert m.sum() == eps[0].phase_mask("grasp").sum()


def test_chunk_alignment_normalization_and_end_padding(data):
    ep = data["glove"][0]
    ds = _ds([ep], cameras=())
    a, v = hand_action_from_episode(ep)
    T, st = ep.T, ds.stride
    for i in range(len(ds)):
        s = ds[i]
        t = s["t_index"]
        rel = s["actions"].numpy()
        absolute = make_absolute(rel, a[t], "hand_mano", "delta")
        for j in range(ds.horizon):
            src = t + (1 + j) * st                            # chunk[0] ↔ t + stride
            assert bool(s["action_valid"][j]) == (src <= T - 1 and bool(v[src]))
            if s["action_valid"][j]:
                np.testing.assert_allclose(absolute[j], a[src], atol=1e-5)
        np.testing.assert_allclose(s["proprio"].numpy(), a[t], atol=1e-6)
    # the last ticks run past the episode end: trailing steps invalid, leading valid
    last = ds[len(ds) - 1]
    n_in = sum(last["t_index"] + (1 + j) * st <= T - 1 for j in range(ds.horizon))
    assert n_in < ds.horizon and last["action_valid"][:n_in].all() and not last["action_valid"][n_in:].any()
    # normalizers: fit on train, round trip, proprio normalized
    an, pn = ds.fit_normalizers()
    s = ds[5]
    rel = an.unnormalize(s["actions"].numpy())
    t = s["t_index"]
    raw = _ds([ep], cameras=())[5]
    np.testing.assert_allclose(rel, raw["actions"].numpy(), atol=1e-4)
    np.testing.assert_allclose(pn.unnormalize(s["proprio"].numpy()), a[t], atol=1e-4)
    chunks, valid, _ = ds.raw_targets()
    rows = chunks[valid]
    z = an.normalize(rows)
    moving = rows.std(0) > 0.02                                # dims above the min_scale floor
    np.testing.assert_allclose(z.mean(0), 0.0, atol=1e-3)
    np.testing.assert_allclose(z[:, moving].std(0), 1.0, atol=1e-2)


def test_invalid_hand_frames_are_masked(data):
    ep = _mem_copy(data["glove"][1])
    ds0 = _ds([ep], cameras=())
    t0 = int(ds0.index[len(ds0) // 2, 1])
    hv = ep.arrays[K_HAND_VALID]
    v_orig = hv.copy()
    hv[t0 + 2 * ds0.stride] = False                            # chunk step j = 1 of tick t0
    hv[t0 + ds0.stride] = False                                # state of the next tick
    ds = _ds([ep], cameras=())
    ticks = ds.index[:, 1].tolist()
    assert t0 + ds0.stride not in ticks                        # invalid current state → skipped
    s = ds[ticks.index(t0)]
    assert not s["action_valid"][0] and not s["action_valid"][1]
    assert bool(s["action_valid"][2]) == bool(v_orig[t0 + 3 * ds0.stride])
    assert torch.isfinite(s["actions"]).all()


# ─────────────────────────────────────────────────────────── tactile

def test_tactile_values_single_source_and_derived_preferred(data):
    ep = data["glove"][0]
    spec = TactileFeatureSpec(obs_mode="full", history=2, stride=3)
    ds = _ds([ep], cameras=(), feature_spec=spec)
    z, lv, sat = bootstrap_tactile_arrays(ep)
    s = ds[4]
    t = s["t_index"]
    np.testing.assert_allclose(s["tactile_values"].numpy(), spec.from_arrays(z, lv, sat, t))
    assert s["tactile_values"].shape[-1] == spec.dim == 12
    assert torch.equal(s["contact"], torch.from_numpy(contact_from_level(lv[t], sat[t])))
    # derived stage-1 outputs win over the bootstrap
    mem = _mem_copy(ep)
    zz = np.full((mem.T, mem.meta.n_taxels), 4.0, np.float32)
    ll = np.full((mem.T, mem.meta.n_taxels), int(ContactLevel.WEAK), np.int8)
    mem.set_derived(D_RESIDUAL_Z, zz, save=False)
    mem.set_derived(D_LEVEL, ll, save=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                         # no bootstrap warning
        dsd = VTLADataset([mem], cameras=(), horizon=8, tactile_source="derived")
    assert dsd.tactile_source == "derived"
    sd = dsd[0]
    sat_t = np.asarray(mem["saturated"][sd["t_index"]])
    exp = TactileFeatureSpec().from_arrays(zz, ll, mem["saturated"], sd["t_index"])
    np.testing.assert_allclose(sd["tactile_values"].numpy(), exp)
    assert torch.equal(sd["contact"], torch.ones_like(sd["contact"]) | torch.from_numpy(sat_t))
    with pytest.raises(KeyError, match="contact stage"):
        VTLADataset([ep], cameras=(), tactile_source="derived")
    # obs_mode none: zero-width values, no tactile arrays needed
    dn = VTLADataset([ep], cameras=(), feature_spec={"obs_mode": "none"})
    assert dn[0]["tactile_values"].shape == (ep.meta.n_taxels, 0) and dn.tactile_source == "none"


def test_bootstrap_levels_follow_ground_truth(data):
    ep = data["glove"][0]
    z, lv, sat = bootstrap_tactile_arrays(ep)
    assert z.dtype == np.float32 and lv.dtype == np.int8 and z.shape == lv.shape == sat.shape
    gt_press = np.asarray(ep["gt_press_pct"])                  # SATS sign: press → negative
    strong = (gt_press < -10.0) & ~sat
    assert strong.sum() > 20
    assert np.median(z[strong]) > 5.0                          # press-positive z
    assert (lv[strong] >= ContactLevel.WEAK).mean() > 0.9
    ref = (np.asarray(ep[K_CONTACT_LABEL]) == 0) & ~sat
    assert abs(float(np.median(z[ref]))) < 0.5
    assert (lv[ref] == ContactLevel.NONE).mean() > 0.95
    assert (lv[sat] == ContactLevel.SATURATED).all()
    with pytest.raises(ValueError):
        bootstrap_tactile_arrays(ep, weak=1.0)
    c_all = contact_from_level(np.array([0, 1, 2, 3]))
    c_ws = contact_from_level(np.array([0, 1, 2, 3]), np.array([False, True, False, False]), "weak_or_strong")
    assert c_all.tolist() == [False, True, True, True] and c_ws.tolist() == [False, False, True, False]


# ─────────────────────────────────────────────────────────── vision

def test_camera_validity_and_cached_features(data):
    ep = data["glove"][0]
    ds = _ds([ep], phases="all", sample_stride=1, require_valid_state=False)
    first = {c: int(np.argmax(np.asarray(ep[cam_idx_key(c)]) >= 0)) for c in ("ego", "third")}
    seen = set()
    for i in range(6):                                          # the first ticks (from t = 0)
        si = ds[i]
        for c in ("ego", "third"):
            ok = int(ep[cam_idx_key(c)][si["t_index"]]) >= 0
            assert bool(si["vision_valid"][c][0]) == ok
            seen.add(ok)
    assert seen == {False, True}                                # before and after the first frame
    t_late = int(ds.index[-1, 1])
    assert t_late > max(first.values()) and ds[len(ds) - 1]["vision_valid"]["ego"].all()
    # frozen-encoder feature cache → vision_feats
    enc = TinyConvEncoder(8, grid=(2, 2), channels=(4, 8)).freeze()
    tf = EvalTransform(HW)
    for c in ("ego", "third"):
        cache_episode_features(ep, c, enc, tf, key="t0", overwrite=True)
    dsc = _ds([ep], use_cached_vision="t0")
    s = dsc[2]
    assert "images" not in s and s["vision_feats"]["ego"].shape == (4, 8)
    idx = int(ep[cam_idx_key("ego")][s["t_index"]])
    f, ok = gather_frame_features(load_cached(ep, "ego", "t0"), idx)
    np.testing.assert_allclose(s["vision_feats"]["ego"].numpy(), f)
    # cached features = the encoder run online on the same tick's image (up to the fp16 cache)
    online_ds = _ds([ep])
    assert online_ds[2]["t_index"] == s["t_index"]
    online = enc(online_ds[2]["images"]["ego"][None])[0]
    assert torch.allclose(online, s["vision_feats"]["ego"], atol=2e-2)


def test_obs_history_collate_and_model(data):
    eps = data["glove"][:2]
    ds = _ds(eps, obs_history=2)
    an, pn = ds.fit_normalizers()
    s = ds[1]
    assert s["images"]["ego"].shape == (2, 3, *HW) and s["proprio"].shape == (2 * 54,)
    assert s["vision_valid"]["third"].shape == (2,)
    text_cfg = {"type": "hashing", "dim": 16, "max_len": 12}
    cfg = VTLAConfig(horizon=8, obs_history=2, d_model=32, cameras=("ego", "third"), fusion_depth=1,
                     head_depth=1,
                     vision={"type": "tiny", "out_dim": 16, "grid": [2, 2], "channels": [8, 16]},
                     text=text_cfg, tactile_encoder={"d_model": 16, "depth": 1, "heads": 2, "n_fourier": 2},
                     tactile_heads=2)
    torch.manual_seed(0)
    model = VTLAPolicy(cfg)
    batch = VTLACollator(model.text_encoder.get_tokenizer())([ds[i] for i in range(5)])
    assert batch["images"]["ego"].shape == (5, 2, 3, *HW) and batch["input_ids"].shape == (5, 12)
    assert batch["taxel_pad"].shape == (5, eps[0].meta.n_taxels) and not batch["taxel_pad"].any()
    out = model(batch)
    assert torch.isfinite(out["loss"])
    assert model.predict(batch).shape == (5, 8, 54)
    with pytest.raises(ValueError):
        collate_vtla([])


def _with_dead_channel(ep, taxel=0):
    """In-memory copy whose ``taxel`` is a dead channel exactly as ``datasets.build`` marks one
    (baseline ≤ 0: ΔS 0, saturated in every frame, listed in ``meta.preprocessing.dead_taxels``)."""
    import copy

    from robot_skin.datasets.episode import K_DELTA, K_SATURATED

    meta = copy.deepcopy(ep.meta)
    meta.preprocessing = {**(meta.preprocessing or {}), "dead_taxels": [int(taxel)]}
    arrays = {k: np.array(v) for k, v in ep.arrays.items()}
    arrays[K_DELTA][:, taxel] = 0.0
    arrays[K_SATURATED][:, taxel] = True
    return Episode(meta, arrays, dict(ep.static))


def test_dead_channel_is_masked_so_the_contact_gate_can_close(data):
    """VTLA-2: a dead channel is SATURATED in every frame, i.e. contact under the default
    ``level_ge_weak`` rule, so it held the hard ContactGate open on every sample and tactile drift of
    the other taxels reached the fusion on real no-contact frames. ``mask_dead_taxels`` (default)
    hides ``meta.preprocessing.dead_taxels`` via ``taxel_pad``: the gate closes again exactly where
    the live taxels see no contact."""
    kw = dict(cameras=(), tactile_source="bootstrap")
    ref = _ds([_mem_copy(data["glove"][0])], **kw)
    ep = _with_dead_channel(data["glove"][0], 0)
    raw = _ds([ep], mask_dead_taxels=False, **kw)
    masked = _ds([ep], **kw)
    assert raw.dead_taxels == masked.dead_taxels == {ep.meta.episode_id: [0]} and ref.dead_taxels == {}
    n = len(masked)
    assert len(raw) == len(ref) == n > 0

    def gate(ds, live=slice(None)):
        return np.array([bool(ds[i]["contact"][live].any()) for i in range(n)])

    assert gate(raw).all()                                             # the old behaviour: always open
    assert all(bool(masked[i]["taxel_pad"][0]) and not masked[i]["taxel_pad"][1:].any()
               and not masked[i]["contact"][0] for i in range(n))
    np.testing.assert_array_equal(gate(masked), gate(ref, slice(1, None)))
    assert not gate(masked).all()
    # through the model: on a no-contact sample the tactile drift of the live taxels no longer matters
    i = int(np.flatnonzero(~gate(masked))[0])
    cfg = VTLAConfig(horizon=8, d_model=32, cameras=(), vision=None, text=None, fusion_depth=1, head_depth=1,
                     tactile_encoder={"d_model": 16, "depth": 1, "heads": 2, "n_fourier": 2}, tactile_heads=2)
    torch.manual_seed(0)
    model = VTLAPolicy(cfg)
    torch.nn.init.normal_(model.head.out.weight, std=0.5)

    def drift_changes_output(ds):
        batch = collate_vtla([ds[i]])
        drifted = {**batch, "tactile_values": batch["tactile_values"].clone()}
        drifted["tactile_values"][:, 1:] += 3.0
        return not torch.allclose(model.predict(batch), model.predict(drifted))

    assert drift_changes_output(raw) and not drift_changes_output(masked)


def test_robot_joint_actions(data):
    ep = data["robot"][0]
    ds = _ds([ep], cameras=("ego",), action_spec="robot_joint", rel_mode="delta")
    D = len(ep.meta.joint_names)
    assert ds.action_spec == ActionSpec.robot_joint(ep.meta.joint_names) and ds.action_dim == D
    s = ds[2]
    assert s["actions"].shape == (8, D) and s["proprio"].shape == (D,)
    q = np.asarray(ep["q"])
    t = s["t_index"]
    np.testing.assert_allclose(s["actions"][0].numpy(), q[t + ds.stride] - q[t], atol=1e-5)
    wrong = ActionSpec.robot_joint(list(reversed(ep.meta.joint_names)))
    with pytest.raises(ValueError, match="joint order"):
        _ds([ep], cameras=("ego",), action_spec=wrong)
    with pytest.raises(ValueError, match="delta_pose"):
        _ds([ep], cameras=("ego",), action_spec="robot_joint", rel_mode="delta_pose")


# ─────────────────────────────────────────────────────────── stage

def test_stage_yaml_mirrors_defaults_and_encoder_blocks_replace(tmp_path):
    from robot_skin.stages import vtla as stage

    y = yaml.safe_load(stage.CONFIG_PATH.read_text())
    assert y == stage.DEFAULTS
    cfg = stage.load_stage_config(None, {"vision": {"encoder": {"type": "resnet18", "pretrained": False}},
                                         "language": {"encoder": {"dim": 64}}})
    assert cfg["vision"]["encoder"] == {"type": "resnet18", "pretrained": False}   # replaced
    assert cfg["language"]["encoder"] == {"type": "hashing", "dim": 64, "max_len": 32}  # merged
    hw = stage.load_stage_config(None, {"hardware": "cpu", "train": {"lr": 1e-3}})
    assert hw["hardware_applied"] and hw["train"]["device"] == "cpu"
    assert hw["train"]["batch_size"] == 8 and hw["train"]["lr"] == 1e-3        # suggest.vtla < --set
    assert stage.resolve_config(hw)["train"]["batch_size"] == 8                 # applied only once
    r = stage.resolve_config({"vision": {"encoder": None}, "out_dir": str(tmp_path)})
    assert r["vision"]["encoder"] is None and stage._cameras(r) == []
    assert r["train"]["out_dir"] == str(tmp_path)


def test_stage_config_rejects_unknown_keys():
    from robot_skin.stages import vtla as stage

    for bad in ({"action": {"rel_mod": "abs"}}, {"bogus": 1}, {"model": {"horizon": 32}},
                {"vision": None}, {"data": {"phase": "all"}}):
        with pytest.raises(ValueError):
            stage.load_stage_config(None, bad)
        with pytest.raises(ValueError):
            stage.resolve_config(bad)
    ok = stage.resolve_config({"image": None, "train": {"max_steps": 1},
                               "vision": {"encoder": {"type": "tiny", "anything": 1}}})
    assert ok["image"] is None                       # open sections are validated downstream
    for fn in (stage.resolve_config, lambda c: stage.load_stage_config(None, c)):
        with pytest.raises(ValueError, match="custom_key"):     # train: TrainConfig fields only (CLI-5)
            fn({"train": {"custom_key": 1}})
    lin = torch.nn.Linear(3, 2)
    h32 = stage._weights_hash(lin)
    assert h32 == stage._weights_hash(lin) and h32 != stage._weights_hash(lin.to(torch.bfloat16))


def test_stage_mask_dead_taxels_is_configurable_and_warns_when_off(data):
    """VTLA-2: ``data.mask_dead_taxels`` is a stage key (every key that could carry the documented
    remedy was rejected as unknown); with it off under ``level_ge_weak`` the stage warns about the
    episodes' dead channels instead of silently training with an always-open ContactGate."""
    from robot_skin.stages import vtla as stage

    assert stage.DEFAULTS["data"]["mask_dead_taxels"] is True
    assert stage.resolve_config({"data": {"mask_dead_taxels": False}})["data"]["mask_dead_taxels"] is False
    assert stage.load_stage_config(None, {"data": {"mask_dead_taxels": False}})["data"]["mask_dead_taxels"] is False
    eps = [_with_dead_channel(data["glove"][0], 2), _mem_copy(data["glove"][1])]
    with pytest.warns(UserWarning, match=r"1/2 episodes have dead tactile channels.*ContactGate never closes"):
        stage._report_dead_taxels(eps, {"mask_dead_taxels": False, "contact_rule": "level_ge_weak"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stage._report_dead_taxels(eps, {"mask_dead_taxels": True, "contact_rule": "level_ge_weak"})
        stage._report_dead_taxels(eps, {"mask_dead_taxels": False, "contact_rule": "weak_or_strong"})
        stage._report_dead_taxels(eps[1:], {"mask_dead_taxels": False, "contact_rule": "level_ge_weak"})


def test_stage_splits_json_shared_with_other_stages(data, tmp_path, caplog):
    """data.splits: a datasets.splits splits.json (paths relative to the processed root, or ids)
    assigns episodes exactly; unlisted / unknown entries are reported, never guessed."""
    from robot_skin.datasets.splits import save_splits
    from robot_skin.stages import vtla as stage

    g = data["glove"]
    path = save_splits({"train": [g[0].root, g[1].root], "val": [g[2].meta.episode_id], "test": ["nope"]},
                       tmp_path / "splits.json", root=data["proc"])
    assert json.loads(path.read_text())["relative"]
    d_cfg = {"splits": str(path), "processed_root": str(data["proc"])}
    with pytest.warns(UserWarning, match="1 entries without a usable episode, 1 usable episodes not listed"):
        parts = stage.split_episodes(g + data["robot"], d_cfg)
    assert [e.meta.episode_id for e in parts["train"]] == [g[0].meta.episode_id, g[1].meta.episode_id]
    assert parts["val"] == [g[2]] and parts["test"] == []
    assert not [r for r in caplog.records if "data.splits is not set" in r.getMessage()]
    # without data.splits: the stage's own make_splits split, logged as not shared with other stages
    with caplog.at_level(logging.WARNING, logger="robot_skin.stages"):
        auto = stage.split_episodes(g, {"split_by": "episode", "processed_root": str(data["proc"])})
    assert sum(len(v) for v in auto.values()) == len(g)
    msgs = [r.getMessage() for r in caplog.records if "data.splits is not set" in r.getMessage()]
    assert len(msgs) == 1 and msgs[0].startswith("vtla:") and "make_splits by episode_id" in msgs[0]


def test_stage_run_with_pretrained_frozen_tactile_encoder(data, tmp_path):
    """tactile.pretrained: the pretrained encoder's feature spec wins over ``features``, its frozen
    weights reach the bundle unchanged, and obs_history 2 (camera history) trains end to end."""
    from robot_skin.representation import TaxelEncoder, save_pretrained_encoder
    from robot_skin.stages import vtla as stage

    spec = TactileFeatureSpec(obs_mode="full", history=2, stride=2)
    torch.manual_seed(0)
    enc = TaxelEncoder(spec.dim, d_model=16, depth=1, heads=2, n_fourier=2, feature_spec=spec)
    save_pretrained_encoder(tmp_path / "pre", enc)
    cfg = _stage_cfg(data, tmp_path / "run", head="chunk")
    cfg["features"] = {"obs_mode": "binary"}
    cfg["tactile"] = {"pretrained": str(tmp_path / "pre"), "freeze": True}
    cfg["policy"]["obs_history"] = 2
    cfg["train"]["max_epochs"] = 1
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        metrics = stage.run(cfg)
    assert any("pretrained encoder's feature spec" in str(w.message) for w in rec)
    assert metrics["feature_spec"] == spec.to_dict() and np.isfinite(metrics["val/l1"])
    b = read_policy_bundle(tmp_path / "run")
    assert b["tactile"]["feature_spec"] == spec.to_dict() and b["model_config"]["tactile_frozen"]
    assert b["timing"]["obs_history"] == 2 and "history_emb" in b["state_dict"]
    for k, v in enc.state_dict().items():
        assert torch.equal(b["state_dict"]["tactile_encoder." + k], v), k
    policy = build_policy_from_bundle(b)
    assert policy.feature_spec == spec and policy.tactile_encoder.value_dim == spec.dim


def _stage_cfg(data, out_dir, **model):
    return {
        "out_dir": str(out_dir),
        "data": {"processed_root": str(data["proc"]), "datasets": ["task"], "split_by": "episode",
                 "val_frac": 0.34, "test_frac": 0.0, "split_seed": 1},
        "policy": {"horizon": 8, "cameras": ["ego"]},
        "vision": {"encoder": {"type": "tiny", "out_dim": 16, "grid": [2, 2], "channels": [8, 16]}},
        "image": {"image_size": list(HW), "seed": 0},
        "language": {"encoder": {"type": "hashing", "dim": 16, "max_len": 12}},
        "model": {"d_model": 32, "fusion_depth": 1, "head_depth": 1, "tactile_heads": 2, "flow_steps": 4,
                  "tactile_encoder": {"d_model": 16, "depth": 1, "heads": 2, "n_fourier": 2}, **model},
        "train": {"max_epochs": 2, "batch_size": 16, "lr": 1e-3, "warmup_steps": 1, "device": "cpu",
                  "log_every": 1000},
        "eval": {"batch_size": 16, "seed": 3},
    }


@pytest.mark.parametrize("variant", ["flow_full_aux", "chunk_none_cached"])
def test_stage_run_writes_bundle_that_reproduces_policy(data, tmp_path, variant):
    from robot_skin.stages import vtla as stage
    from robot_skin.train import Trainer
    from robot_skin.vtla import VTLACollator as Collator

    if variant == "flow_full_aux":
        cfg = _stage_cfg(data, tmp_path / "run", head="flow", aux_contact_weight=0.2)
        cfg["data"]["aux_target"] = "gt"
        (tmp_path / "contact").mkdir()
        (tmp_path / "contact" / "calibrator.json").write_text(json.dumps({"sigma_pct": [1.0], "weak_z": 3.0}))
        cfg["tactile"] = {"calibrator": str(tmp_path / "contact")}
    else:
        cfg = _stage_cfg(data, tmp_path / "run", head="chunk")
        cfg["features"] = {"obs_mode": "none"}
        cfg["vision"]["cache_features"] = True
        cfg["language"] = {"encoder": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = stage.run(cfg)
    out = tmp_path / "run"
    assert (out / "policy_bundle.pt").is_file() and (out / "metrics.json").is_file()
    assert json.loads((out / "metrics.json").read_text())["val/l1"] == pytest.approx(metrics["val/l1"])
    assert metrics["n_episodes"]["train"] == 2 and metrics["n_episodes"]["val"] == 1
    assert metrics["n_episodes"]["skipped"] == 1                    # the robot episode has no hand labels
    assert np.isfinite(metrics["val/l1"]) and len(metrics["val/l1_per_step"]) == 8
    assert set(metrics["val/l1_by_task"]) and "val/l1_raw/wrist_pos" in metrics

    b = read_policy_bundle(out)
    comp = bundle_components(b)
    policy = comp["policy"]
    assert not policy.training and comp["stride"] == 10 and comp["horizon"] == 8
    assert comp["feature_spec"].obs_mode == ("full" if variant == "flow_full_aux" else "none")
    assert b["tactile"]["source"] == ("bootstrap" if variant == "flow_full_aux" else "none")
    assert b["vision"]["cameras"] == ["ego"] and b["vision"]["eval_transform"]["out_size"] == list(HW)
    assert b["tactile"]["mask_dead_taxels"] is True                 # control masks dead channels likewise
    if variant == "flow_full_aux":
        assert b["tactile"]["calibrator_state"] == {"sigma_pct": [1.0], "weak_z": 3.0}
    if variant == "chunk_none_cached":
        assert b["vision"]["cached_features_key"] and policy.tactile_encoder is None
        assert policy.text_encoder is None and comp["tokenizer"] is None

    # rebuild the val dataset purely from the bundle → same metrics as the stage
    tim, tac = b["timing"], b["tactile"]
    val_ds = VTLADataset(b["meta"]["episodes"]["val"], cameras=comp["cameras"], policy_hz=tim["policy_hz"],
                         horizon=tim["horizon"], obs_history=tim["obs_history"],
                         chunk_offset=comp["chunk_offset"], action_spec=comp["action_spec"],
                         rel_mode=comp["rel_mode"], feature_spec=comp["feature_spec"],
                         image_transform=comp["eval_transform"],
                         use_cached_vision=b["vision"]["cached_features_key"],
                         action_normalizer=comp["action_normalizer"],
                         proprio_normalizer=comp["proprio_normalizer"], phases=b["meta"]["phases"],
                         contact_rule=tac["contact_rule"], mask_dead_taxels=tac["mask_dead_taxels"],
                         tactile_source="bootstrap" if tac["source"] == "bootstrap" else "auto")
    res = stage.evaluate_policy(policy, val_ds, batch_size=16, seed=3, collate_fn=Collator(comp["tokenizer"]))
    assert res["l1"] == pytest.approx(metrics["val/l1"], rel=1e-5, abs=1e-6)
    assert res["l1_per_step"] == pytest.approx(metrics["val/l1_per_step"], rel=1e-5, abs=1e-6)

    # the bundle weights are the best checkpoint's
    fresh = VTLAPolicy(b["model_config"])
    Trainer.load_model_weights(fresh, out / "ckpt_best.pt", use_ema=True)
    batch = Collator(comp["tokenizer"])([val_ds[i] for i in range(4)])
    noise = torch.randn(4, 8, 54, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(fresh.predict(batch, noise=noise), policy.predict(batch, noise=noise), atol=1e-6)
    same = build_policy_from_bundle(out / "policy_bundle.pt")
    assert torch.equal(same.predict(batch, noise=noise), policy.predict(batch, noise=noise))
