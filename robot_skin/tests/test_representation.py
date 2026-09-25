import numpy as np
import pytest
import torch

from robot_skin.contact.ordinal import ContactLevel, OrdinalQuantizer
from robot_skin.datasets.episode import D_LEVEL, D_RESIDUAL_Z, Episode, EpisodeMeta
from robot_skin.policy import OBS_MODES, tactile_features
from robot_skin.representation import (ENCODER_STATE_NAME, TACTILE_FRAME_DIMS,
                                       MaskedTaxelPretrainer, TactileFeatureSpec, TactileHistory,
                                       TaxelEncoder, TaxelTokenizer, history_indices,
                                       load_pretrained_encoder, random_taxel_mask,
                                       read_encoder_state, save_pretrained_encoder, stack_history,
                                       tactile_value_dim, tactile_value_features, z_feature,
                                       z_from_feature)


def test_tokenizer_shapes_and_pose_sensitivity():
    tok = TaxelTokenizer(value_dim=4, d_model=32, n_fourier=4)
    v = torch.randn(2, 9, 4)
    pos, nrm = torch.randn(2, 9, 3) * 0.05, torch.nn.functional.normalize(torch.randn(2, 9, 3), dim=-1)
    out = tok(v, pos, nrm)
    assert out.shape == (2, 9, 32)
    assert not torch.allclose(out, tok(v, pos + 0.01, nrm))
    # taxel-count agnostic without id embedding
    assert tok(torch.randn(1, 16, 4), torch.zeros(1, 16, 3), torch.zeros(1, 16, 3)).shape == (1, 16, 32)


def test_mask_replaces_value_only():
    torch.manual_seed(0)
    tok = TaxelTokenizer(value_dim=2, d_model=16, n_taxels=5)
    pos, nrm = torch.randn(1, 5, 3), torch.randn(1, 5, 3)
    m = torch.tensor([[True, False, False, False, False]])
    a = tok(torch.randn(1, 5, 2), pos, nrm, mask=m)
    b = tok(torch.randn(1, 5, 2), pos, nrm, mask=m)
    assert torch.allclose(a[:, 0], b[:, 0])          # masked taxel ignores its value
    assert not torch.allclose(a[:, 1], b[:, 1])


def test_random_taxel_mask():
    g = torch.Generator().manual_seed(0)
    m = random_taxel_mask(4, 10, 0.3, generator=g)
    assert m.shape == (4, 10) and (m.sum(1) == 3).all()
    assert random_taxel_mask(2, 10, 0.0).sum() == 0
    with pytest.raises(ValueError):
        random_taxel_mask(1, 10, 1.0)
    # the pretrainer is implemented now: it wraps a TaxelEncoder (no longer a stub)
    with pytest.raises(TypeError):
        MaskedTaxelPretrainer(torch.nn.Linear(2, 2))


# ─────────────────────────────────────────────────────────── tactile value features

Z = np.array([0.0, 2.0, 200.0, np.nan, 5.0, -4.0], dtype=np.float32)
LV = np.array([0, 1, 2, 3, -1, 0], dtype=np.int8)
SAT = np.array([0, 0, 0, 0, 1, 0], dtype=bool)


def test_full_features_hand_values():
    f = tactile_value_features(Z, LV, SAT)
    assert f.shape == (6, 6) and f.dtype == np.float32
    exp = np.array([
        [0.0, 1, 0, 0, 0, 0],                                  # z 0, NONE
        [np.arcsinh(1.0), 0, 1, 0, 0, 0],                      # z 2 → asinh(2/2), WEAK
        [np.arcsinh(50.0), 0, 0, 1, 0, 0],                     # z 200 clipped to 100 → asinh(50)
        [0.0, 0, 0, 0, 1, 1],                                  # level SATURATED → zf 0, sat 1
        [0.0, 0, 0, 0, 1, 1],                                  # saturated flag overrides level -1
        [np.arcsinh(-2.0), 1, 0, 0, 0, 0],                     # negative z kept (sign-preserving)
    ], dtype=np.float32)
    np.testing.assert_allclose(f, exp, rtol=1e-6, atol=1e-7)
    # saturation may come from the level alone
    g = tactile_value_features(Z, LV, None)
    np.testing.assert_allclose(g[3], exp[3])
    assert g[4, 1:5].sum() == 0 and g[4, 5] == 0             # unknown level -1 → all-zero one-hot


def test_modes_dims_and_policy_consistency():
    assert set(TACTILE_FRAME_DIMS) == set(OBS_MODES)
    for mode in OBS_MODES:
        f = tactile_value_features(np.zeros((3, 5)), np.zeros((3, 5), np.int8), obs_mode=mode)
        assert f.shape == (3, 5, tactile_value_dim(mode))
    assert tactile_value_dim("full", 3) == 18 and tactile_value_dim("none", 4) == 0
    # ordinal / binary are exactly the policy observation's features for the same levels
    rng = np.random.default_rng(0)
    resid = rng.normal(0, 10, (7, 9)).astype(np.float32)       # ΔS% residual (press negative)
    sat = rng.random((7, 9)) < 0.1
    q = OrdinalQuantizer(3.0, 15.0)
    levels = q(resid, sat)
    z = -resid                                                  # any press-positive z
    for mode in ("ordinal", "binary"):
        ours = tactile_value_features(z, levels, sat, obs_mode=mode).reshape(7, -1)
        np.testing.assert_array_equal(ours, tactile_features(mode, resid, sat, q))
    np.testing.assert_array_equal(tactile_value_features(z, levels, sat, obs_mode="ordinal"),
                                  OrdinalQuantizer.one_hot(levels))


def test_torch_matches_numpy():
    for mode in OBS_MODES:
        a = tactile_value_features(Z, LV, SAT, obs_mode=mode)
        b = tactile_value_features(torch.tensor(Z), torch.tensor(LV), torch.tensor(SAT), obs_mode=mode)
        assert isinstance(b, torch.Tensor) and b.dtype == torch.float32
        np.testing.assert_allclose(b.numpy(), a, rtol=1e-6, atol=1e-7)
    d = tactile_value_features(torch.tensor(Z, dtype=torch.float64), LV, None)
    assert d.dtype == torch.float64


def test_feature_validation():
    with pytest.raises(ValueError):
        tactile_value_features(Z, LV, obs_mode="bogus")
    with pytest.raises(ValueError):
        tactile_value_features(Z, LV[:3])
    with pytest.raises(ValueError):
        tactile_value_features(Z, LV, SAT[:2])
    with pytest.raises(ValueError):
        tactile_value_features(Z, LV, z_scale=0.0)
    with pytest.raises(ValueError):
        TactileFeatureSpec(history=0)
    with pytest.raises(ValueError):
        TactileFeatureSpec.from_dict({"obs_mode": "full", "typo": 1})


def test_z_feature_inverse():
    z = np.array([-80.0, -3.0, 0.0, 0.5, 7.0, 99.0], np.float32)
    np.testing.assert_allclose(z_from_feature(z_feature(z)), z, rtol=1e-5, atol=1e-5)
    zt = torch.tensor(z)
    torch.testing.assert_close(z_from_feature(z_feature(zt)), zt, rtol=1e-5, atol=1e-5)
    assert z_feature(np.array([np.inf]))[0] == pytest.approx(np.arcsinh(50.0))


def test_history_stacking_offline_matches_online():
    np.testing.assert_array_equal(history_indices(5, 3, 2), [1, 3, 5])
    np.testing.assert_array_equal(history_indices(np.array([0, 1]), 3, 2), [[0, 0, 0], [0, 0, 1]])
    rng = np.random.default_rng(1)
    T, N = 12, 4
    z = rng.normal(0, 5, (T, N)).astype(np.float32)
    lv = rng.integers(0, 4, (T, N)).astype(np.int8)
    sat = rng.random((T, N)) < 0.2
    spec = TactileFeatureSpec(obs_mode="full", history=3, stride=2)
    assert spec.dim == 18 and spec.span == 5
    off = spec.from_arrays(z, lv, sat, np.arange(T))
    assert off.shape == (T, N, 18)
    # hand check: t=5, taxel 2 → frames 1, 3, 5 (oldest first), 6 features each
    per = tactile_value_features(z, lv, sat)
    np.testing.assert_array_equal(off[5, 2], np.concatenate([per[1, 2], per[3, 2], per[5, 2]]))
    np.testing.assert_array_equal(off[0, 1], np.tile(per[0, 1], 3))    # causal edge padding
    for backend in ("numpy", "torch"):
        hist = TactileHistory(spec)
        for t in range(T):
            args = (z[t], lv[t], sat[t]) if backend == "numpy" else (
                torch.tensor(z[t]), torch.tensor(lv[t]), torch.tensor(sat[t]))
            on = hist.push(*args)
            on = on.numpy() if isinstance(on, torch.Tensor) else on
            np.testing.assert_allclose(on, off[t], rtol=1e-6, atol=1e-7)
    x = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    np.testing.assert_array_equal(stack_history(x)[1, 2], x[1, :, 2].reshape(-1))
    np.testing.assert_array_equal(stack_history(torch.tensor(x)).numpy(), stack_history(x))


def _episode(T=20, N=9, seed=0, derived=True):
    rng = np.random.default_rng(seed)
    meta = EpisodeMeta(episode_id=f"ep{seed}", dataset="motion", kind="glove",
                       layout="glove_template", n_taxels=N)
    arrays = {"t": np.arange(T) / 200.0, "saturated": rng.random((T, N)) < 0.1,
              "taxel_pos": rng.normal(0, 0.05, (T, N, 3)).astype(np.float32),
              "taxel_nrm": np.tile([0, -1.0, 0], (T, N, 1)).astype(np.float32)}
    ep = Episode(meta, arrays)
    if derived:
        ep.set_derived(D_RESIDUAL_Z, rng.normal(0, 4, (T, N)).astype(np.float32), save=False)
        ep.set_derived(D_LEVEL, rng.integers(0, 4, (T, N)).astype(np.int8), save=False)
    return ep


def test_spec_from_episode():
    ep = _episode()
    spec = TactileFeatureSpec(history=2, stride=3)
    v = spec.from_episode(ep, np.array([0, 7, 19]))
    exp = spec.from_arrays(ep.derived(D_RESIDUAL_Z), ep.derived(D_LEVEL), ep["saturated"],
                           np.array([0, 7, 19]))
    np.testing.assert_array_equal(v, exp)
    with pytest.raises(KeyError, match="contact stage"):
        spec.from_episode(_episode(derived=False), 0)
    with pytest.raises(IndexError):
        spec.from_episode(ep, 20)


# ─────────────────────────────────────────────────────────── encoder

def test_encoder_shapes_padding_and_equivariance():
    torch.manual_seed(0)
    enc = TaxelEncoder(6, d_model=32, depth=2, heads=4, n_fourier=4).eval()
    v, pos, nrm = torch.randn(2, 7, 6), torch.randn(2, 7, 3) * 0.05, torch.randn(2, 7, 3)
    out = enc(v, pos, nrm)
    assert out.shape == (2, 7, 32) and enc.out_dim == 32
    # permutation equivariant without id embedding
    perm = torch.randperm(7)
    torch.testing.assert_close(enc(v[:, perm], pos[:, perm], nrm[:, perm]), out[:, perm],
                               rtol=1e-5, atol=1e-5)
    # padded taxels: zero output, and they do not influence the real ones
    kpm = torch.zeros(2, 7, dtype=torch.bool)
    kpm[0, 5:] = True
    a = enc(v, pos, nrm, key_padding_mask=kpm)
    assert torch.all(a[0, 5:] == 0)
    v2 = v.clone()
    v2[0, 5:] += 100.0
    torch.testing.assert_close(enc(v2, pos, nrm, key_padding_mask=kpm), a)
    torch.testing.assert_close(a[0, :5], enc(v[:1, :5], pos[:1, :5], nrm[:1, :5])[0],
                               rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="every taxel"):
        enc(v, pos, nrm, key_padding_mask=torch.ones(2, 7, dtype=torch.bool))
    with pytest.raises(ValueError):
        enc(torch.randn(2, 7, 5), pos, nrm)


def test_encoder_validation():
    with pytest.raises(ValueError, match="none"):
        TaxelEncoder(tactile_value_dim("none"))
    with pytest.raises(ValueError):
        TaxelEncoder(6, feature_spec=TactileFeatureSpec(obs_mode="ordinal"))
    with pytest.raises(ValueError):
        TaxelEncoder(6, d_model=30, heads=4)
    with pytest.raises(ValueError):
        TaxelEncoder(6, depth=0)


def test_encoder_config_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    spec = TactileFeatureSpec(obs_mode="ordinal", history=2, stride=4)
    enc = TaxelEncoder(spec.dim, d_model=16, depth=1, heads=2, n_fourier=3, n_taxels=9,
                       feature_spec=spec).eval()
    cfg = enc.config
    assert cfg["value_dim"] == 8 and cfg["feature_spec"]["history"] == 2
    p = save_pretrained_encoder(tmp_path / "run", enc, meta={"best": {"value": np.float32(0.5)}})
    assert p == tmp_path / "run" / ENCODER_STATE_NAME and p.is_file()
    st = read_encoder_state(tmp_path / "run")
    assert set(st) >= {"format", "version", "config", "state_dict", "meta"}
    enc2 = load_pretrained_encoder(tmp_path / "run", freeze=True)
    assert enc2.feature_spec == spec and enc2.pretrain_meta["best"]["value"] == 0.5
    assert not any(p.requires_grad for p in enc2.parameters()) and not enc2.training
    v, pos, nrm = torch.randn(3, 9, 8), torch.randn(3, 9, 3) * 0.05, torch.randn(3, 9, 3)
    torch.testing.assert_close(enc2(v, pos, nrm), enc(v, pos, nrm))
    torch.save({"model": {}}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="taxel_encoder"):
        read_encoder_state(tmp_path / "bad.pt")


def test_encoder_encode_convenience():
    torch.manual_seed(0)
    spec = TactileFeatureSpec()
    enc = TaxelEncoder(spec.dim, d_model=16, depth=1, heads=2, feature_spec=spec).eval()
    ep = _episode(T=5)
    z, lv, sat = ep.derived(D_RESIDUAL_Z)[:2], ep.derived(D_LEVEL)[:2], ep["saturated"][:2]
    pos, nrm = ep["taxel_pos"][:2], ep["taxel_nrm"][:2]
    out = enc.encode(z, lv, sat, pos, nrm)
    ref = enc(torch.as_tensor(tactile_value_features(z, lv, sat)), torch.as_tensor(pos),
              torch.as_tensor(nrm))
    torch.testing.assert_close(out, ref)
    with pytest.raises(ValueError):
        TaxelEncoder(12, feature_spec=TactileFeatureSpec(history=2)).encode(z, lv, sat, pos, nrm)
    assert ContactLevel.SATURATED == 3
