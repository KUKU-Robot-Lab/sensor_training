import json
import logging
import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from common.layouts import load_layout
from robot_skin.contact.ordinal import ContactLevel
from robot_skin.datasets.episode import D_LEVEL, D_RESIDUAL_Z, Episode, EpisodeMeta
from robot_skin.representation import (MaskedTaxelPretrainer, TactileFeatureSpec,
                                       TaxelEncoder, TaxelPretrainDataset, collate_pretrain,
                                       evaluate_reconstruction, layout_group_matrix,
                                       level_class_weights, load_pretrained_encoder,
                                       pretrain_loss, sample_taxel_mask, z_feature)
from robot_skin.stages import pretrain as stage


def make_episode(eid, *, T=300, seed=0, layout="glove_template", dataset="motion",
                 subject="s0", derived=True):
    """Tiny processed episode whose z is shared within layout groups (palm, fingertips), so a
    hidden taxel is predictable from visible taxels of its group."""
    rng = np.random.default_rng(seed)
    lay = load_layout(layout)
    N = lay.n
    t = np.arange(T) / 200.0
    z = rng.normal(0, 0.3, (T, N))
    shared = [lay.groups.get("palm", []), lay.groups.get("fingertip", [])] if "palm" in lay.groups \
        else [lay.groups["row0"] + lay.groups["row1"], lay.groups["row2"] + lay.groups["row3"]]
    for idx, f in zip(shared, (1.3, 0.7)):
        a = 10 * np.maximum(0, np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))) ** 2
        z[:, idx] += a[:, None]
    lv = np.zeros((T, N), np.int8)
    lv[z >= 3] = ContactLevel.WEAK
    lv[z >= 8] = ContactLevel.STRONG
    sat = z > 9.5
    pos = (np.broadcast_to(lay.positions, (T, N, 3)) + rng.normal(0, 1e-3, (T, N, 3))).astype(np.float32)
    nrm = np.broadcast_to(lay.normals, (T, N, 3)).astype(np.float32)
    meta = EpisodeMeta(episode_id=eid, dataset=dataset, kind="glove", layout=layout, n_taxels=N,
                       subject=subject)
    ep = Episode(meta, {"t": t, "taxel_pos": pos, "taxel_nrm": nrm, "saturated": sat})
    if derived:
        ep.set_derived(D_RESIDUAL_Z, z.astype(np.float32), save=False)
        ep.set_derived(D_LEVEL, lv, save=False)  # saturation lives in the `saturated` array
    return ep


def make_batch(B=6, N=5, F=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"values": torch.randn(B, N, F, generator=g), "pos": torch.randn(B, N, 3, generator=g) * 0.05,
            "nrm": torch.randn(B, N, 3, generator=g), "target_z": torch.randn(B, N, generator=g),
            "z_valid": torch.rand(B, N, generator=g) > 0.2,
            "target_level": torch.randint(-1, 4, (B, N), generator=g),
            "pad_mask": torch.zeros(B, N, dtype=torch.bool)}


# ─────────────────────────────────────────────────────────── masks

def test_sample_taxel_mask_counts_padding_and_visibility():
    pad = torch.zeros(4, 10, dtype=torch.bool)
    pad[1, 6:] = True          # 6 real taxels
    pad[2, 2:] = True          # 2 real taxels
    pad[3, 1:] = True          # 1 real taxel → nothing can be hidden
    g = torch.Generator().manual_seed(0)
    m = sample_taxel_mask(4, 10, 0.3, pad_mask=pad, generator=g)
    assert m.dtype == torch.bool and not (m & pad).any()
    assert m.sum(1).tolist() == [3, 2, 1, 0]            # round(0.3·n), ≥1, ≤ n−1
    assert ((~m & ~pad).sum(1) >= 1).all()
    m2 = sample_taxel_mask(4, 10, 0.3, pad_mask=pad, generator=torch.Generator().manual_seed(0))
    assert torch.equal(m, m2)
    with pytest.raises(ValueError):
        sample_taxel_mask(1, 5, 1.0)
    with pytest.raises(ValueError):
        sample_taxel_mask(1, 5, 0.3, mode="bogus")


def test_group_masking_hides_whole_group_and_tops_up():
    B, N = 64, 8
    groups = torch.zeros(B, 3, N, dtype=torch.bool)
    groups[:, 0, :2] = True          # size 2
    groups[:, 1, 2:5] = True         # size 3
    groups[:, 2, :] = True           # all taxels: never eligible (nothing would stay visible)
    g = torch.Generator().manual_seed(1)
    m = sample_taxel_mask(B, N, 0.25, groups=groups, mode="group", generator=g)
    k = 2                             # round(0.25·8)
    # group 0 has k taxels → exactly group 0; group 1 has 3 > k → exactly group 1 (no top-up)
    chosen = [torch.equal(m[b], groups[b, 0]) or torch.equal(m[b], groups[b, 1]) for b in range(B)]
    assert all(chosen)
    assert set(m.sum(1).tolist()) == {2, 3}
    # a group smaller than k is topped up with random taxels to k
    small = torch.zeros(B, 1, N, dtype=torch.bool)
    small[:, 0, 7] = True
    ms = sample_taxel_mask(B, N, 0.5, groups=small, mode="group",
                           generator=torch.Generator().manual_seed(3))
    assert ms[:, 7].all() and (ms.sum(1) == 4).all()
    # mixed with group_prob=0 is plain random masking (same draws are not required, only counts)
    mr = sample_taxel_mask(B, N, 0.25, groups=groups, mode="mixed", group_prob=0.0,
                           generator=torch.Generator().manual_seed(1))
    assert (mr.sum(1) == k).all()
    # no eligible group → falls back to random
    only_all = groups[:, 2:3]
    mf = sample_taxel_mask(B, N, 0.25, groups=only_all, mode="group",
                           generator=torch.Generator().manual_seed(2))
    assert (mf.sum(1) == k).all()


def test_layout_group_matrix():
    names, gm = layout_group_matrix(load_layout("glove_template"))
    assert "fingertip" not in names                     # 5/9 > max_group_frac 0.5
    assert "palm" in names and gm[names.index("palm")].sum() == 4
    assert gm.shape == (len(names), 9)
    names16, gm16 = layout_group_matrix(load_layout("sats_4x4"))
    assert "all" not in names16 and len(names16) == 8 and (gm16.sum(1) == 4).all()
    with pytest.raises(ValueError):
        layout_group_matrix(load_layout("sats_4x4"), max_group_frac=0.0)


def test_level_class_weights():
    w = level_class_weights([81, 9, 9, 1])
    freq = np.array([81, 9, 9, 1]) / 100
    raw = freq ** -0.5
    np.testing.assert_allclose(w, raw / (freq * raw).sum(), rtol=1e-6)
    assert (freq * w).sum() == pytest.approx(1.0)
    w2 = level_class_weights([100, 0, 0, 0])
    np.testing.assert_allclose(w2, [1, 1, 1, 1])
    assert level_class_weights([1e6, 1, 0, 0], max_weight=5.0).max() == pytest.approx(5.0)
    with pytest.raises(ValueError):
        level_class_weights([1, 2, 3])


# ─────────────────────────────────────────────────────────── data

def test_dataset_targets_groups_and_collate(tmp_path):
    ep9 = make_episode("g", T=40, seed=0)
    ep16 = make_episode("s", T=30, seed=1, layout="sats_4x4")
    ep9.save(tmp_path / "g")
    spec = TactileFeatureSpec(history=2, stride=3)
    ds = TaxelPretrainDataset([tmp_path / "g", ep16], spec, frame_stride=5)
    assert len(ds) == 8 + 6
    s = ds[3]                                              # episode g, t = 15
    assert s["episode"] == 0 and s["t"] == 15
    z, lv, sat = ep9.derived(D_RESIDUAL_Z), ep9.derived(D_LEVEL), ep9["saturated"]
    np.testing.assert_array_equal(s["values"], spec.from_arrays(z, lv, sat, 15))
    np.testing.assert_allclose(s["target_z"], z_feature(z[15]))
    np.testing.assert_array_equal(s["z_valid"], ~sat[15])
    np.testing.assert_array_equal(s["target_level"], np.where(sat[15], 3, lv[15]))
    assert s["groups"].shape[1] == 9 and len(ds.group_names[1]) == 8
    b = collate_pretrain([ds[0], ds[-1]])
    assert b["values"].shape == (2, 16, 12) and b["groups"].shape == (2, 8, 16)
    assert b["pad_mask"][0, 9:].all() and not b["pad_mask"][0, :9].any()
    assert not b["pad_mask"][1].any()
    assert (b["target_level"][0, 9:] == -1).all() and not b["z_valid"][0, 9:].any()
    assert not b["groups"][0, :, 9:].any()
    # contact oversampling + level histogram (counts every taxel of every sample, repeats incl.)
    ds_rep = TaxelPretrainDataset([ep9], spec, frame_stride=1, contact_repeat=3)
    lvl = np.where(sat, 3, lv)
    touch = (lvl >= 1).any(1)
    assert len(ds_rep) == 40 + 2 * int(touch.sum())
    exp = np.bincount(lvl.reshape(-1), minlength=4) + 2 * np.bincount(lvl[touch].reshape(-1), minlength=4)
    np.testing.assert_array_equal(ds_rep.level_counts(), exp)
    with pytest.raises(KeyError, match="contact stage"):
        TaxelPretrainDataset([make_episode("x", T=10, derived=False)], spec)
    with pytest.raises(ValueError):
        TaxelPretrainDataset([ep9], TactileFeatureSpec(obs_mode="none"))
    # unknown layout → warning, random masking only
    bad = make_episode("b", T=10)
    bad.meta.layout = "no_such_layout"
    with pytest.warns(UserWarning, match="group masking disabled"):
        ds_bad = TaxelPretrainDataset([bad], spec)
    assert ds_bad[0]["groups"].shape == (0, 9)
    # world-framed taxel poses (hand travelling 0.5 m) are flagged — once for all episodes
    moving = make_episode("w", T=50)
    drift = np.linspace(0, 0.5, 50, dtype=np.float32)[:, None, None]
    moving.arrays["taxel_pos"] = moving["taxel_pos"] + drift
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        TaxelPretrainDataset([moving, moving], spec)
    msgs = [str(w.message) for w in rec if "world/camera frame" in str(w.message)]
    assert len(msgs) == 1 and msgs[0].startswith("2/2 episode(s)")


def test_dataset_layout_resolution_per_episode(tmp_path):
    from common.layouts import LAYOUT_DIR
    base = yaml.safe_load((LAYOUT_DIR / "glove_template.yaml").read_text())

    def write(d, groups):
        d.mkdir(parents=True, exist_ok=True)
        lay = {**base, "taxels": [{**t, "groups": [g for g, idx in groups.items() if j in idx]}
                                  for j, t in enumerate(base["taxels"])]}
        (d / "layout.yaml").write_text(yaml.safe_dump(lay))

    # two episodes with the same *relative* meta.layout but different layout files
    for i, groups in enumerate([{"palm": [5, 6, 7, 8]}, {"a": [0, 1], "b": [2, 3]}]):
        ep = make_episode(f"e{i}", T=10, seed=i)
        ep.meta.layout = "layout.yaml"
        ep.save(tmp_path / f"e{i}")
        write(tmp_path / f"e{i}", groups)
    ds = TaxelPretrainDataset([tmp_path / "e0", tmp_path / "e1"], TactileFeatureSpec())
    assert ds.group_names == [["palm"], ["a", "b"]]
    # moved dataset: stale absolute meta.layout, but the episode keeps its own layout.yaml copy
    ep = make_episode("moved", T=10)
    ep.meta.layout = str(tmp_path / "old_root" / "moved" / "layout.yaml")
    ep.save(tmp_path / "moved")
    write(tmp_path / "moved", {"c": [4, 5]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds = TaxelPretrainDataset([tmp_path / "moved"], TactileFeatureSpec())
    assert ds.group_names == [["c"]] and ds[0]["groups"].sum() == 2


def test_dataset_skips_frames_with_non_finite_poses():
    ep = make_episode("nan", T=20)
    pos = np.array(ep["taxel_pos"])
    pos[3, 2] = np.nan
    nrm = np.array(ep["taxel_nrm"])
    nrm[7, 0, 1] = np.inf
    ep.arrays["taxel_pos"], ep.arrays["taxel_nrm"] = pos, nrm
    with pytest.warns(UserWarning, match="2 frames with non-finite"):
        ds = TaxelPretrainDataset([ep], TactileFeatureSpec(), contact_repeat=2)
    ts = {int(ds[i]["t"]) for i in range(len(ds))}
    assert ts == set(range(20)) - {3, 7}
    assert all(np.isfinite(ds[i]["pos"]).all() for i in range(len(ds)))
    # level histogram follows the kept frames (incl. contact repeats)
    lv = np.where(ep["saturated"], 3, ep.derived(D_LEVEL)).astype(np.int64)
    keep = np.array(sorted(ts))
    touch = keep[(lv[keep] >= 1).any(1)]
    exp = np.bincount(lv[keep].ravel(), minlength=4) + np.bincount(lv[touch].ravel(), minlength=4)
    np.testing.assert_array_equal(ds.level_counts(), exp)


# ─────────────────────────────────────────────────────────── model

def _model(F=6, seed=0, **kw):
    torch.manual_seed(seed)
    enc = TaxelEncoder(F, d_model=16, depth=2, heads=2, n_fourier=3)
    return MaskedTaxelPretrainer(enc, F, **kw)


def test_mae_encoder_only_sees_visible_taxels():
    m = _model().eval()
    batch = make_batch()
    mask = torch.zeros(6, 5, dtype=torch.bool)
    mask[:, [1, 3]] = True
    a = m.predict(batch, mask)
    b2 = dict(batch)
    b2["values"] = batch["values"].clone()
    b2["values"][:, [1, 3]] = 1e3                               # hidden values never leak
    b = m.predict(b2, mask)
    torch.testing.assert_close(a["z_pred"], b["z_pred"])
    torch.testing.assert_close(a["level_logits"], b["level_logits"])
    # visible tokens == the encoder applied to the visible subset alone (MAE encoder)
    vis = [0, 2, 4]
    full = m.encoder(batch["values"], batch["pos"], batch["nrm"], mask=mask, key_padding_mask=mask)
    sub = m.encoder(batch["values"][:, vis], batch["pos"][:, vis], batch["nrm"][:, vis])
    torch.testing.assert_close(full[:, vis], sub, rtol=1e-5, atol=1e-5)
    # eval + no_grad (PyTorch's fused transformer fast path, used by evaluation) agrees and
    # stays finite with padding
    batch["pad_mask"][0, 4] = True
    ref = m.predict(batch, mask)
    with torch.no_grad():
        fast = m.predict(batch, mask)
    for k in ("z_pred", "level_logits"):
        assert torch.isfinite(fast[k]).all()
        torch.testing.assert_close(fast[k], ref[k], rtol=1e-5, atol=1e-5)


def test_loss_matches_manual_and_every_parameter_gets_a_gradient():
    w = [0.5, 2.0, 3.0, 1.0]
    m = _model(level_class_weights=w, huber_delta=0.7, z_weight=2.0, level_weight=0.5).train()
    batch = make_batch()
    mask = torch.zeros(6, 5, dtype=torch.bool)
    mask[:, :2] = True
    batch["pad_mask"][0, 4] = True
    torch.manual_seed(3)
    pred = m.predict(batch, mask)
    torch.manual_seed(3)
    out = m(batch, mask)
    hidden = mask & ~batch["pad_mask"]
    mz = hidden & batch["z_valid"]
    hub = F.huber_loss(pred["z_pred"], batch["target_z"], reduction="none", delta=0.7)
    z_l = hub[mz].mean()
    tl = batch["target_level"]
    ml = hidden & (tl >= 0)
    ce = F.cross_entropy(pred["level_logits"][ml], tl[ml], weight=torch.tensor(w))
    torch.testing.assert_close(out["z_huber"], z_l.detach())
    torch.testing.assert_close(out["level_ce"], ce.detach())
    torch.testing.assert_close(out["loss"], 2.0 * z_l + 0.5 * ce)
    assert out["mask_frac"] == pytest.approx(hidden.sum().item() / 29)
    out["loss"].backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters without grad (DDP would fail): {missing}"
    assert pretrain_loss(m, batch)["loss"].ndim == 0
    torch.testing.assert_close(m.loss(batch, mask)["loss"], m(batch, mask)["loss"])   # spec alias


def test_unused_non_finite_targets_do_not_poison_loss_or_grads():
    m = _model().train()
    batch = make_batch()
    mask = torch.zeros(6, 5, dtype=torch.bool)
    mask[:, :2] = True
    batch["z_valid"][:] = True
    batch["target_z"][:, 3] = float("nan")          # visible taxel: never a target
    batch["target_z"][:, 0] = float("inf")          # hidden but flagged invalid
    batch["z_valid"][:, 0] = False
    out = m(batch, mask)
    out["loss"].backward()
    assert all(torch.isfinite(v).all() for v in out.values())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    batch["z_valid"][:, 3] = False                  # evaluation samples its own masks
    samples = [{**{k: v[i].numpy() for k, v in batch.items()}, "groups": np.zeros((0, 5), bool)}
               for i in range(6)]
    res = evaluate_reconstruction(m, samples)
    assert np.isfinite(res["loss"]) and np.isfinite(res["z_mae"])


def test_evaluate_reconstruction_empty_is_nan_not_zero():
    m = _model()
    res = evaluate_reconstruction(m, [])
    assert np.isnan(res["loss"]) and res["n_z"] == 0 and res["n_level"] == 0
    # only level targets (no valid z) → loss = level term only
    s = {"values": np.zeros((5, 6), np.float32), "pos": np.zeros((5, 3), np.float32),
         "nrm": np.zeros((5, 3), np.float32), "target_z": np.zeros(5, np.float32),
         "z_valid": np.zeros(5, bool), "target_level": np.zeros(5, np.int64),
         "groups": np.zeros((0, 5), bool)}
    res = evaluate_reconstruction(m, [s, s])
    assert res["n_z"] == 0 and res["n_level"] > 0
    assert res["loss"] == pytest.approx(m.level_weight * res["level_ce"])


def test_eval_masks_are_deterministic():
    m = _model(mask_mode="mixed")
    batch = make_batch(B=16, N=9, seed=4)
    batch["groups"] = torch.zeros(16, 2, 9, dtype=torch.bool)
    batch["groups"][:, 0, :3] = True
    m.eval()
    with torch.no_grad():
        l1, l2 = m(batch)["loss"], m(batch)["loss"]
    assert torch.equal(l1, l2)
    m.train()
    torch.manual_seed(0)
    m1 = m.sample_mask(batch)
    m2 = m.sample_mask(batch)
    assert not torch.equal(m1, m2)
    with pytest.raises(ValueError):
        _model(mask_ratio=0.0)
    with pytest.raises(ValueError):
        MaskedTaxelPretrainer(TaxelEncoder(6, d_model=16, heads=2), value_dim=7)
    with pytest.raises(ValueError):
        _model(level_class_weights=[1.0, 2.0])


def test_pretraining_learns_cross_taxel_structure():
    torch.manual_seed(0)
    spec = TactileFeatureSpec()
    train = TaxelPretrainDataset([make_episode(f"e{i}", T=400, seed=i) for i in range(3)], spec,
                                 frame_stride=2)
    val = TaxelPretrainDataset([make_episode("v", T=400, seed=9)], spec, frame_stride=2)
    enc = TaxelEncoder(spec.dim, d_model=32, depth=1, heads=4, feature_spec=spec)
    m = MaskedTaxelPretrainer(enc, mask_ratio=0.3, mask_mode="random",
                              level_class_weights=level_class_weights(train.level_counts()))
    before = evaluate_reconstruction(m, val)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    dl = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True,
                                     collate_fn=collate_pretrain,
                                     generator=torch.Generator().manual_seed(0))
    for _ in range(8):
        for b in dl:
            loss = m(b)["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
    after = evaluate_reconstruction(m, val)
    assert after["n_z"] > 100 and after["loss"] < 0.5 * before["loss"]
    assert after["z_mae"] < 0.5 * after["z_mae_zero"]          # beats predicting "no residual"
    assert after["level_acc"] > max(0.85, after["level_acc_majority"] + 0.1)
    assert after["contact_f1"] > 0.85
    again = evaluate_reconstruction(m, val)
    assert again == after                                    # seeded eval masks


# ─────────────────────────────────────────────────────────── stage runner

def _write_root(root):
    for i in range(3):
        make_episode(f"m{i}", T=120, seed=i, subject=f"s{i % 2}").save(root / "motion" / f"m{i}")
    make_episode("t0", T=120, seed=5, dataset="task").save(root / "task" / "t0")
    make_episode("raw", T=50, seed=6, derived=False).save(root / "motion" / "raw")


def _cfg(root, out, **train):
    return {"data": {"processed_root": str(root), "frame_stride": 3},
            "model": {"d_model": 16, "depth": 1, "heads": 2, "n_fourier": 3},
            "pretrain": {"decoder_depth": 1},
            "eval": {"batch_size": 64},
            "train": {"max_epochs": 2, "batch_size": 32, "lr": 2e-3, "warmup_steps": 2,
                      "device": "cpu", "precision": "fp32", "log_every": 1000,
                      "out_dir": str(out), **train}}


def test_stage_run_writes_encoder_and_metrics(tmp_path):
    root = tmp_path / "processed"
    _write_root(root)
    out = tmp_path / "run"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = stage.run(_cfg(root, out))
    assert metrics["n_episodes"] == {"found": 5, "train": 3, "val": 1, "skipped": 1}
    assert metrics["skipped"][0]["episode"].endswith("raw")
    assert (out / "encoder_state.pt").is_file() and (out / "metrics.json").is_file()
    saved = json.loads((out / "metrics.json").read_text())
    for k in ("val/loss", "val/z_mae", "val/z_mae_zero", "val/level_acc", "val/contact_f1"):
        assert k in metrics and k in saved
    assert metrics["steps"] > 0 and metrics["best"]["value"] is not None
    json.loads((out / "metrics.json").read_text(), parse_constant=lambda c: pytest.fail(c))  # strict
    assert len(metrics["level_class_weights"]) == 4
    enc = load_pretrained_encoder(out)
    assert isinstance(enc, TaxelEncoder) and enc.feature_spec == TactileFeatureSpec()
    assert enc.pretrain_meta["stage"] == "pretrain"
    assert len(enc.pretrain_meta["episodes"]["val"]) == 1
    # the encoder saved is the best checkpoint's
    from robot_skin.train import Trainer
    m2 = MaskedTaxelPretrainer(TaxelEncoder.from_config(enc.config))
    Trainer.load_model_weights(m2, out / "ckpt_best.pt")
    for k, v in enc.state_dict().items():
        torch.testing.assert_close(m2.encoder.state_dict()[k], v)


def test_stage_splits_file_and_subject_split(tmp_path):
    root = tmp_path / "processed"
    _write_root(root)
    splits = {"train": ["motion/m0", "m1"], "val": [str(root / "task" / "t0")], "test": ["m2"]}
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    cfg = _cfg(root, tmp_path / "run", max_epochs=1)
    cfg["data"]["splits"] = str(tmp_path / "splits.json")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = stage.run(cfg)
    assert metrics["n_episodes"]["train"] == 2 and metrics["n_episodes"]["val"] == 1
    eps, _ = stage.load_usable_episodes(stage.discover_episodes({"processed_root": str(root)}))
    sp = stage.split_episodes(eps, {"split_by": "subject", "val_frac": 0.5, "split_seed": 0})
    subj = {k: {e.meta.subject for e in v} for k, v in sp.items() if v}
    assert not (subj["train"] & subj["val"])                  # no subject in two splits
    with pytest.raises(ValueError, match="no usable training episodes"):
        stage.run(_cfg(tmp_path / "empty", tmp_path / "run2"))
    # relative entries may also be relative to the splits file's directory (load_splits default)
    (root / "splits.json").write_text(json.dumps({"train": ["motion/m0", "motion/m1"],
                                                  "val": ["task/t0"], "test": ["motion/m2"]}))
    sp = stage.split_episodes(eps, {"processed_root": str(tmp_path / "elsewhere"),
                                    "splits": str(root / "splits.json")})
    assert [e.meta.episode_id for e in sp["train"]] == ["m0", "m1"]
    assert [e.meta.episode_id for e in sp["val"]] == ["t0"] and sp["test"][0].meta.episode_id == "m2"


def test_stage_split_without_splits_json_warns_it_is_not_shared(tmp_path, caplog):
    """Every stage accepts ``data.splits``; without it the stage splits its own pool and logs that
    this split is not shared with the other stages (named per stage)."""
    from robot_skin.stages import split_stage_episodes

    root = tmp_path / "processed"
    _write_root(root)
    eps, _ = stage.load_usable_episodes(stage.discover_episodes({"processed_root": str(root)}))
    with caplog.at_level(logging.WARNING, logger="robot_skin.stages"):
        stage.split_episodes(eps, {"val_frac": 0.5})
        split_stage_episodes(eps, {"val_frac": 0.5}, stage="baseline")
    msgs = [r.getMessage() for r in caplog.records if "data.splits is not set" in r.getMessage()]
    assert len(msgs) == 2 and msgs[0].startswith("pretrain:") and msgs[1].startswith("baseline:")
    assert "NOT shared" in msgs[0] and "val_frac=0.5" in msgs[0]
    caplog.clear()
    (root / "splits.json").write_text(json.dumps({"train": ["motion/m0", "motion/m1"], "val": ["task/t0"],
                                                  "test": ["motion/m2"]}))
    with caplog.at_level(logging.WARNING, logger="robot_skin.stages"):
        sp = split_stage_episodes(eps, {"splits": str(root / "splits.json"), "processed_root": str(root)},
                                  stage="contact")
    assert not [r for r in caplog.records if "data.splits is not set" in r.getMessage()]
    assert [e.meta.episode_id for e in sp["val"]] == ["t0"]


def test_legacy_world_frame_glove_episodes_are_flagged(caplog):
    """Glove episodes without ``meta.preprocessing.taxel_frame`` (datasets.build < /2: world-frame
    taxel poses) are named in one warning by the stages that consume taxel poses."""
    from robot_skin.stages import warn_legacy_taxel_frame

    old, new = make_episode("old", T=10), make_episode("new", T=10)
    new.meta.preprocessing = {"taxel_frame": "mano_wrist"}
    robot = make_episode("robot", T=10)
    robot.meta.kind = "robot"
    with caplog.at_level(logging.WARNING, logger="robot_skin.stages"):
        assert warn_legacy_taxel_frame([old, new, robot], "baseline") == ["old"]
        assert warn_legacy_taxel_frame([new, robot], "vtla") == []
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1 and msgs[0].startswith("baseline: 1 glove episode(s)") and "--force" in msgs[0]


def test_stage_cli(tmp_path, capsys):
    root = tmp_path / "processed"
    _write_root(root)
    cfg = _cfg(root, tmp_path / "run", max_epochs=1)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = stage.main(["--config", str(tmp_path / "cfg.yaml"), "--set", "pretrain.mask_mode=group",
                         "--set", f"out_dir={tmp_path / 'cli'}"])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["out_dir"] == str(tmp_path / "cli") and "val/loss" in printed
    meta = load_pretrained_encoder(tmp_path / "cli").pretrain_meta
    assert meta["pretrainer_config"]["mask_mode"] == "group"
    with pytest.raises(FileNotFoundError):
        stage.load_stage_config(tmp_path / "missing.yaml")


def test_stage_yaml_matches_defaults():
    y = yaml.safe_load(stage.CONFIG_PATH.read_text())

    def keys(d, prefix=""):
        out = set()
        for k, v in d.items():
            out.add(prefix + k)
            if isinstance(v, dict):
                out |= keys(v, prefix + k + ".")
        return out

    assert keys(y) == keys(stage.DEFAULTS)
    assert y["stage"] == "pretrain"
    cfg = stage.load_stage_config(overrides={"train": {"lr": 5e-4}})
    assert cfg["train"]["lr"] == 5e-4 and cfg["pretrain"]["mask_mode"] == "mixed"
    r = stage.resolve_config({"out_dir": "x/y"})
    assert r["train"]["out_dir"] == "x/y"


def test_stage_hardware_profile_precedence():
    # CLI path: YAML < profile (suggest.pretrain batch 16, device cpu) < explicit overrides
    cfg = stage.load_stage_config(overrides={"hardware": "cpu", "train": {"batch_size": 4}})
    assert cfg["hardware"] == "cpu" and cfg["hardware_applied"] is True
    assert cfg["train"]["batch_size"] == 4 and cfg["train"]["device"] == "cpu"
    assert stage.resolve_config(cfg)["train"]["batch_size"] == 4            # not re-applied
    cfg = stage.load_stage_config(overrides={"hardware": "cpu"})
    assert cfg["train"]["batch_size"] == 16                                  # profile suggestion
    # raw dict given to run(): the profile is applied once and wins over the dict's train keys
    r = stage.resolve_config({"hardware": "cpu", "train": {"batch_size": 4}})
    assert r["hardware_applied"] is True and r["train"]["batch_size"] == 16
    assert stage.resolve_config(r)["train"] == r["train"]
    assert "hardware_applied" not in stage.load_stage_config()["train"]
    assert stage.load_stage_config()["hardware_applied"] is False            # no profile → untouched
