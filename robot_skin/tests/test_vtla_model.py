"""VTLA model, heads, losses, bundle and DPO hook (tiny encoders, CPU, seeded)."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from robot_skin.vtla import (MODALITIES, ChunkRegressionHead, ContactGate, FlowMatchingHead,
                             TactileTokenAdapter, VTLAConfig, VTLAPolicy, build_policy_from_bundle,
                             bundle_components, collate_vtla, contact_bce, dpo_loss,
                             make_observation, make_reference_policy, masked_l1, masked_mse,
                             masked_step_sums, preference_loss, read_policy_bundle,
                             save_policy_bundle, sinusoidal_embedding, vtla_loss)
from robot_skin.vtla.dpo import build_preference_pairs

TACTILE, VISION, LANG = MODALITIES.index("tactile"), MODALITIES.index("vision"), MODALITIES.index("language")
B, N, H, A = 4, 9, 5, 12


@pytest.fixture(autouse=True)
def _one_thread():
    """Tiny models: one intra-op thread is faster and immune to CPU oversubscription."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def tiny_cfg(**kw):
    base = dict(action_dim=A, horizon=H, proprio_dim=A, d_model=32, fusion_depth=1, fusion_heads=4,
                head_depth=1, cameras=("ego", "third"),
                vision={"type": "tiny", "out_dim": 16, "grid": [2, 2], "channels": [8, 16]},
                text={"type": "hashing", "dim": 16, "max_len": 8},
                tactile_encoder={"d_model": 16, "depth": 1, "heads": 2, "n_fourier": 2},
                tactile_heads=2, flow_steps=3)
    base.update(kw)
    return VTLAConfig(**base)


def make_batch(seed=0, *, F_=6, n=N, actions=True, cams=("ego", "third")):
    g = torch.Generator().manual_seed(seed)
    b = {"proprio": torch.randn(B, A, generator=g),
         "images": {c: torch.randn(B, 3, 24, 32, generator=g) for c in cams},
         "tactile_values": torch.randn(B, n, F_, generator=g),
         "taxel_pos": torch.randn(B, n, 3, generator=g) * 0.05,
         "taxel_nrm": F.normalize(torch.randn(B, n, 3, generator=g), dim=-1),
         "contact": torch.rand(B, n, generator=g) > 0.6,
         "instruction": ["pick up the cup", "pour the water", "wipe", "open the jar"][:B]}
    if actions:
        b["actions"] = torch.randn(B, H, A, generator=g)
        b["action_valid"] = torch.ones(B, H, dtype=torch.bool)
        b["action_valid"][1, 3:] = False
    return b


# ─────────────────────────────────────────────────────────── losses

def test_masked_losses_ignore_padded_steps():
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(3, H, A, generator=g, requires_grad=True)
    tgt = torch.randn(3, H, A, generator=g)
    valid = torch.ones(3, H, dtype=torch.bool)
    valid[0, 2:] = False
    valid[2, :] = False
    garbage = tgt.clone()
    garbage[~valid] = 1e6                              # padded targets must not matter
    l1 = masked_l1(pred, tgt, valid)
    assert torch.allclose(l1, masked_l1(pred, garbage, valid))
    ref = (pred - tgt).abs()[valid].mean()             # hand-computed masked mean
    assert torch.allclose(l1, ref)
    assert torch.allclose(masked_mse(pred, garbage, valid), (pred - tgt).square()[valid].mean())
    l1.backward()
    assert torch.count_nonzero(pred.grad[~valid]) == 0 and torch.count_nonzero(pred.grad[valid]) > 0
    # nothing valid → graph-preserving zero; NaN padding never leaks into the loss
    z = masked_l1(pred, tgt, torch.zeros(3, H, dtype=torch.bool))
    assert z.item() == 0.0 and z.requires_grad
    nan_pad = tgt.clone()
    nan_pad[~valid] = float("nan")
    p2 = pred.detach().clone().requires_grad_(True)
    l_nan = masked_l1(p2, nan_pad, valid)
    assert torch.allclose(l_nan, l1)
    l_nan.backward()
    assert torch.isfinite(p2.grad).all() and torch.count_nonzero(p2.grad[~valid]) == 0
    s, c = masked_step_sums((pred - tgt).abs(), valid)
    assert c.tolist() == [2 * A, 2 * A, A, A, A]
    assert torch.allclose(s.sum() / c.sum(), l1.double())
    with pytest.raises(ValueError):
        masked_l1(pred, tgt[:, :2], valid)


def test_contact_bce_masks_and_pos_weight():
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    tgt = torch.tensor([[1.0, 0.0, 1.0]])
    mask = torch.tensor([[True, True, False]])
    ref = F.binary_cross_entropy_with_logits(logits[:, :2], tgt[:, :2])
    assert torch.allclose(contact_bce(logits, tgt, mask), ref)
    pw = contact_bce(logits, tgt, mask, pos_weight=3.0)
    ref_pw = (3 * F.softplus(-logits[0, 0]) + F.softplus(logits[0, 1])) / 2
    assert torch.allclose(pw, ref_pw)
    assert contact_bce(logits, tgt, torch.zeros_like(mask)).item() == 0.0


# ─────────────────────────────────────────────────────────── heads

def test_sinusoidal_embedding():
    e = sinusoidal_embedding(torch.tensor([0.0, 0.5]), 8)
    assert e.shape == (2, 8)
    assert torch.allclose(e[0], torch.tensor([1.0] * 4 + [0.0] * 4))    # cos 0 | sin 0
    assert sinusoidal_embedding(torch.tensor([0.3]), 7).shape == (1, 7)


def test_chunk_head_shapes_and_zero_init():
    head = ChunkRegressionHead(32, A, H, depth=1, heads=4)
    mem = torch.randn(B, 7, 32)
    mask = torch.zeros(B, 7, dtype=torch.bool)
    mask[:, 5:] = True
    out = head(mem, mask)
    assert out.shape == (B, H, A) and torch.count_nonzero(out) == 0      # zero-init output
    assert torch.equal(head.sample(mem, mask), out)


class _OracleFlow(FlowMatchingHead):
    """Velocity of the straight path to a single known target: v(x, τ) = (a* − x) / (1 − τ)."""

    def __init__(self, target):
        super().__init__(8, target.shape[-1], target.shape[-2], depth=1, heads=2)
        self.target = target

    def velocity(self, x, tau, memory, memory_mask=None):
        return (self.target - x) / (1.0 - tau.reshape(-1, 1, 1))


def test_flow_tau_convention_noise_at_0_data_at_1():
    torch.manual_seed(0)
    target = torch.randn(2, 4, 3)
    head = _OracleFlow(target)
    mem = torch.zeros(2, 1, 8)
    for K in (1, 3, 10):
        # Euler from x_0 = ε (τ = 0) reaches the data a* exactly at τ = 1
        assert torch.allclose(head.sample(mem, n_steps=K), target, atol=1e-5)
    # training path: x_τ = τ·a + (1 − τ)·ε, target velocity u = a − ε
    a, eps = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    tau = torch.tensor([0.25, 0.75])
    x, t, u = head._path(a, eps, tau, None)
    assert torch.allclose(x, tau[:, None, None] * a + (1 - tau[:, None, None]) * eps)
    assert torch.allclose(u, a - eps)
    # a velocity equal to u gives zero error
    head2 = FlowMatchingHead(8, 3, 4, depth=1, heads=2)
    head2.velocity = lambda x_, t_, m_, mm_=None: a - eps               # noqa: E731
    assert torch.allclose(head2.per_sample_error(mem, None, a, noise=eps, tau=tau), torch.zeros(2))


def test_flow_eval_loss_is_deterministic_train_loss_is_not():
    """Eval-mode losses re-seed (ε, τ) from ``eval_seed`` → comparable val/loss across epochs."""
    torch.manual_seed(0)
    head = FlowMatchingHead(32, A, H, depth=1, heads=4)
    torch.nn.init.normal_(head.out.weight, std=0.1)                 # a non-trivial velocity field
    mem = torch.randn(B, 6, 32)
    a = torch.randn(B, H, A)
    head.eval()
    l1, l2 = head.loss(mem, None, a)["action_loss"], head.loss(mem, None, a)["action_loss"]
    assert torch.equal(l1, l2)
    g = torch.Generator().manual_seed(0)
    x_tau, tau, u = head._path(a, None, None, g)                     # = the eval_seed draws
    ref = (head.velocity(x_tau, tau, mem) - u).square().mean()
    assert torch.allclose(l1, ref, atol=1e-6)
    head.train()
    assert not torch.equal(head.loss(mem, None, a)["action_loss"], head.loss(mem, None, a)["action_loss"])
    with pytest.raises(ValueError):
        head.sample(mem, n_steps=0)                                  # 0 steps is an error, not "default"


def test_flow_head_sampling_finite_and_tau_dists():
    head = FlowMatchingHead(32, A, H, depth=1, heads=4, n_steps=4, tau_dist="beta", tau_beta_b=2.0)
    mem = torch.randn(B, 6, 32)
    noise = torch.randn(B, H, A)
    s = head.sample(mem, noise=noise)
    assert s.shape == (B, H, A) and torch.isfinite(s).all()
    assert torch.allclose(s, noise)                     # zero-init output: untrained v = 0
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.equal(head.sample(mem, generator=g1), head.sample(mem, generator=g2))
    taus = head.sample_tau(20000, generator=torch.Generator().manual_seed(0))
    assert 0 <= taus.min() and taus.max() < 1
    assert abs(taus.mean().item() - 1 / 3) < 0.01      # Beta(1, 2) mean = 1/3 → more noisy τ
    uni = FlowMatchingHead(32, A, H, depth=1, heads=4).sample_tau(
        20000, generator=torch.Generator().manual_seed(0))
    assert abs(uni.mean().item() - 0.5) < 0.01
    with pytest.raises(ValueError):
        FlowMatchingHead(32, A, H, tau_dist="bogus")


# ─────────────────────────────────────────────────────────── model

@pytest.mark.parametrize("head", ["chunk", "flow"])
def test_model_forward_backward_shapes(head):
    torch.manual_seed(0)
    m = VTLAPolicy(tiny_cfg(head=head, aux_contact_weight=0.5))
    batch = make_batch()
    batch["contact_target"] = torch.rand(B, N).round()
    batch["contact_target_mask"] = torch.ones(B, N, dtype=torch.bool)
    out = m(batch, return_encoding=True)
    L = 8 + 2 * 4 + 4 + 1 + 1                           # lang + 2 cams × 2×2 + K + proprio + readout
    assert out["memory"].shape == (B, L, 32) and out["memory_mask"].shape == (B, L)
    assert out["token_types"].tolist().count(TACTILE) == 4
    assert out["contact_logits"].shape == (B, N)
    losses = vtla_loss(m, batch)
    assert {"loss", "action_loss", "contact_loss"} <= set(losses)
    assert all(v.numel() == 1 for v in losses.values())
    losses["loss"].backward()
    no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert no_grad == []                                # DDP-safe: every trainable param is used
    pred = m.predict(batch, generator=torch.Generator().manual_seed(0))
    assert pred.shape == (B, H, A) and torch.isfinite(pred).all()
    # tokenized instructions give the same result as raw strings
    tok = m.text_encoder.get_tokenizer()(batch["instruction"])
    b2 = {**batch, "input_ids": tok["input_ids"], "text_pad_mask": tok["pad_mask"]}
    z = torch.zeros(B, H, A)
    assert torch.allclose(m.predict(b2, noise=z), m.predict(batch, noise=z))


def test_obs_history_params_all_used_with_and_without_vision():
    """history_emb exists only for vision history; every trainable param gets a gradient (DDP)."""
    for vision in (None, {"type": "tiny", "out_dim": 16, "grid": [2, 2], "channels": [8, 16]}):
        torch.manual_seed(0)
        m = VTLAPolicy(tiny_cfg(obs_history=2, vision=vision, cameras=("ego",), text=None))
        assert (m.history_emb is None) == (vision is None)
        b = make_batch(cams=("ego",) if vision else ())
        b["proprio"] = torch.randn(B, 2 * A)
        if vision:
            b["images"] = {"ego": torch.randn(B, 2, 3, 24, 32)}
        m(b)["loss"].backward()
        assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == []


def test_chunk_predict_is_float32_under_autocast():
    torch.manual_seed(0)
    m = VTLAPolicy(tiny_cfg(text=None, vision=None))
    torch.nn.init.normal_(m.head.out.weight, std=0.1)
    b = make_batch(cams=(), actions=False)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        p = m.predict(b)
    assert p.dtype == torch.float32 and p.shape == (B, H, A)


def test_modality_dropout_masks_tactile_tokens_and_obs_mode_none():
    torch.manual_seed(0)
    m = VTLAPolicy(tiny_cfg(p_drop_tactile=0.5, p_drop_vision=0.5))
    batch = make_batch(actions=False)
    m.train()
    torch.manual_seed(1)
    enc = m.encode(batch)
    types = enc["token_types"]
    tac, vis = types == TACTILE, types == VISION
    drop = enc["tactile_dropped"]
    assert drop is not None and 0 < int(drop.sum()) < B
    assert enc["memory_mask"][drop][:, tac].all() and not enc["memory_mask"][~drop][:, tac].any()
    assert enc["memory_mask"][enc["vision_dropped"]][:, vis].all()
    # a dropped sample's fused tokens do not depend on its tactile input
    other = {**batch, "tactile_values": batch["tactile_values"] * 5 + 1}
    torch.manual_seed(1)
    enc2 = m.encode(other)
    keep = ~tac
    assert torch.allclose(enc["memory"][drop][:, keep], enc2["memory"][drop][:, keep], atol=1e-5)
    assert not torch.allclose(enc["memory"][~drop][:, keep], enc2["memory"][~drop][:, keep], atol=1e-5)
    # eval: no dropout
    m.eval()
    e = m.encode(batch)
    assert e["tactile_dropped"] is None and not e["memory_mask"][:, e["token_types"] == TACTILE].any()

    # obs_mode none: no tactile branch, no tactile tokens, output independent of tactile inputs
    m0 = VTLAPolicy(tiny_cfg(feature_spec={"obs_mode": "none"}))
    assert m0.tactile_encoder is None and m0.adapter is None
    assert not any("tactile" in n for n, _ in m0.named_parameters())
    b0 = make_batch(F_=0)
    e0 = m0.encode(b0)
    assert (e0["token_types"] == TACTILE).sum() == 0
    p1 = m0.predict(b0)
    p2 = m0.predict({**b0, "contact": ~b0["contact"], "taxel_pos": b0["taxel_pos"] + 1})
    assert torch.equal(p1, p2)


@pytest.mark.parametrize("mode,F_", [("full", 6), ("ordinal", 4), ("binary", 1)])
def test_obs_mode_sets_tactile_value_width(mode, F_):
    m = VTLAPolicy(tiny_cfg(feature_spec={"obs_mode": mode}, text=None, vision=None))
    assert m.tactile_encoder.value_dim == F_ and m.cameras == ()
    out = m(make_batch(F_=F_, cams=()))
    assert torch.isfinite(out["loss"])
    with pytest.raises(ValueError):
        m(make_batch(F_=F_ + 1, cams=()))


def test_contact_gate_no_contact_zeroes_tactile_tokens_through_model():
    torch.manual_seed(0)
    m = VTLAPolicy(tiny_cfg(text=None, vision=None)).eval()
    b = make_batch(cams=(), actions=False)
    b["contact"][:] = False
    b2 = {**b, "tactile_values": torch.randn(B, N, 6) * 10}        # drift without contact
    assert torch.allclose(m.predict(b), m.predict(b2), atol=1e-6)
    b3 = {**b2, "contact": torch.ones(B, N, dtype=torch.bool)}
    assert not torch.allclose(m.encode(b)["memory"], m.encode(b3)["memory"])


def test_padded_taxels_and_gate_valid():
    torch.manual_seed(0)
    m = VTLAPolicy(tiny_cfg(text=None, vision=None, tactile_gate="soft")).eval()
    obs = []
    for n in (9, 5):
        o = make_observation(proprio_states=np.zeros((1, A)), tactile_values=np.random.randn(n, 6),
                             taxel_pos=np.random.randn(n, 3) * 0.05, taxel_nrm=np.eye(3)[np.arange(n) % 3],
                             contact=np.ones(n, bool), instruction="x")
        obs.append(o)
    batch = collate_vtla(obs)
    assert batch["taxel_pad"].tolist()[1] == [False] * 5 + [True] * 4
    assert not batch["contact"][1, 5:].any()
    single = collate_vtla([obs[1]])
    p_pad = m.predict(batch)[1]
    p_one = m.predict(single)[0]
    assert torch.allclose(p_pad, p_one, atol=1e-5)                  # padding changes nothing
    # soft gate: padded taxels are not part of the contact fraction
    g = ContactGate("soft")
    tok = torch.ones(1, 1, 1)
    c = torch.tensor([[True, False, False, False]])
    v = torch.tensor([[True, True, False, False]])
    assert torch.allclose(g(tok, c, v), torch.sigmoid(g.w * 0.5 + g.b).reshape(1, 1, 1))
    with pytest.raises(ValueError, match="taxels"):          # N mismatch (e.g. online wiring bug)
        make_observation(proprio_states=np.zeros((1, A)), tactile_values=np.zeros((8, 6)),
                         taxel_pos=np.zeros((9, 3)), taxel_nrm=np.zeros((9, 3)), contact=np.zeros(9, bool))
    with pytest.raises(ValueError, match=r"\[N,3\]"):
        make_observation(proprio_states=np.zeros((1, A)), tactile_values=np.zeros((9, 6)),
                         taxel_pos=np.zeros((9, 3)), taxel_nrm=np.zeros((9, 2)), contact=np.zeros(9, bool))
    ad = TactileTokenAdapter(16, 8, n_query=2, n_heads=2)
    x = torch.randn(1, 4, 16)
    kpm = torch.tensor([[False, False, True, True]])
    y1 = ad(x, torch.ones(1, 4), kpm)
    x2 = x.clone()
    x2[0, 2:] = 99.0
    assert torch.allclose(y1, ad(x2, torch.ones(1, 4), kpm), atol=1e-5)


@pytest.mark.parametrize("head,steps", [("chunk", 60), ("flow", 200)])
def test_overfit_tiny_batch(head, steps):
    """A tiny fixed batch is memorised: the training loss falls and the predicted chunks (flow:
    Euler samples from fixed noise) approach the targets."""
    torch.manual_seed(0)
    cfg = tiny_cfg(head=head, cameras=("ego",), p_drop_tactile=0.0) if head == "chunk" else \
        tiny_cfg(head=head, vision=None, text=None, p_drop_tactile=0.0)
    m = VTLAPolicy(cfg)
    batch = make_batch(cams=("ego",) if head == "chunk" else ())
    noise = torch.randn(B, H, A, generator=torch.Generator().manual_seed(5))
    l1_before = masked_l1(m.predict(batch, noise=noise), batch["actions"], batch["action_valid"]).item()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(steps):
        loss = m(batch)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first, last = np.mean(losses[:10]), np.mean(losses[-10:])
    assert last < 0.6 * first, (first, last)
    l1_after = masked_l1(m.predict(batch, noise=noise), batch["actions"], batch["action_valid"]).item()
    assert l1_after < 0.7 * l1_before, (l1_before, l1_after)


def test_config_roundtrip_and_validation():
    cfg = tiny_cfg(head="flow")
    d = cfg.to_dict()
    assert VTLAConfig.from_dict(d) == cfg
    assert VTLAPolicy.from_config(d).config == d
    with pytest.raises(ValueError):
        VTLAConfig.from_dict({**d, "bogus": 1})
    with pytest.raises(ValueError):
        VTLAConfig(head="diffusion")
    with pytest.raises(ValueError):
        VTLAConfig(p_drop_tactile=1.0)
    with pytest.raises(ValueError):
        VTLAConfig(feature_spec={"obs_mode": "rgb"})


def test_frozen_encoders_stay_in_eval_and_get_no_grad():
    m = VTLAPolicy(tiny_cfg(vision_frozen=True, text_frozen=True, tactile_frozen=True))
    m.train()
    assert not m.vision_encoder.training and not m.tactile_encoder.training and not m.text_encoder.training
    assert not any(p.requires_grad for p in m.vision_encoder.parameters())
    out = m(make_batch())
    out["loss"].backward()
    assert all(p.grad is None for p in m.tactile_encoder.parameters())
    assert m.adapter.queries.grad is not None


# ─────────────────────────────────────────────────────────── bundle

@pytest.mark.parametrize("head", ["chunk", "flow"])
def test_bundle_roundtrip_reproduces_predictions(tmp_path, head):
    from robot_skin.action import ActionNormalizer, ActionSpec
    from robot_skin.vision import EvalTransform
    from robot_skin.vtla import eval_transform_to_dict

    torch.manual_seed(0)
    m = VTLAPolicy(tiny_cfg(head=head)).eval()
    spec = ActionSpec.robot_joint(A)
    an = ActionNormalizer.fit(np.random.default_rng(0).normal(size=(50, A)), spec=spec)
    path = save_policy_bundle(
        tmp_path, m, action={"spec": spec.to_dict(), "normalizer": an.to_dict(), "rel_mode": "delta",
                             "chunk_offset": 1},
        proprio={"normalizer": an.to_dict(), "history": 1, "source": "action_state"},
        tactile={"feature_spec": m.feature_spec.to_dict(), "contact_rule": "level_ge_weak",
                 "source": "derived", "calibrator": None},
        vision={"cameras": list(m.cameras), "encoder": m.cfg.vision,
                "eval_transform": eval_transform_to_dict(EvalTransform((24, 32)))},
        language={"encoder": m.cfg.text},
        timing={"policy_hz": 20.0, "source_hz": 200.0, "stride": 10, "horizon": H, "obs_history": 1},
        meta={"note": float("nan"), "arr": np.arange(3)})
    assert path.name == "policy_bundle.pt"
    raw = torch.load(path, weights_only=True)            # plain containers only
    assert raw["meta"]["note"] == "nan" and raw["meta"]["arr"] == [0, 1, 2]
    b = read_policy_bundle(tmp_path)
    p2 = build_policy_from_bundle(b)
    batch = make_batch(actions=False)
    noise = torch.randn(B, H, A, generator=torch.Generator().manual_seed(1))
    assert torch.equal(m.predict(batch, noise=noise), p2.predict(batch, noise=noise))
    comp = bundle_components(path)
    assert comp["stride"] == 10 and comp["rel_mode"] == "delta" and comp["cameras"] == ("ego", "third")
    assert comp["eval_transform"].out_size == (24, 32) and comp["tokenizer"] is not None
    assert comp["action_normalizer"].dim == A and comp["head"] == head
    with pytest.raises(ValueError):
        build_policy_from_bundle({"format": "other"})
    with pytest.raises(FileNotFoundError):
        read_policy_bundle(tmp_path / "missing")


# ─────────────────────────────────────────────────────────── DPO

def test_dpo_loss_hand_values():
    pc, pr = torch.tensor([-1.0, -2.0]), torch.tensor([-3.0, -1.0])
    rc, rr = torch.tensor([-1.5, -2.0]), torch.tensor([-2.5, -1.5])
    out = dpo_loss(pc, pr, rc, rr, beta=0.5)
    logits = 0.5 * ((pc - rc) - (pr - rr))                # 0.5·([0.5, 0] − [−0.5, 0.5]) = [0.5, −0.25]
    assert torch.allclose(logits, torch.tensor([0.5, -0.25]))
    assert torch.allclose(out["loss"], -F.logsigmoid(logits).mean())
    assert torch.allclose(out["reward_accuracy"], torch.tensor(0.5))
    same = dpo_loss(pc, pr, pc, pr, beta=0.1)
    assert math.isclose(float(same["loss"]), math.log(2), rel_tol=1e-6)
    with pytest.raises(ValueError):
        dpo_loss(pc, pr, rc, rr[:1])


@pytest.mark.parametrize("head", ["chunk", "flow"])
def test_preference_loss_trains_policy_only(head):
    torch.manual_seed(0)
    policy = VTLAPolicy(tiny_cfg(head=head, vision=None, text=None))
    ref = make_reference_policy(policy)
    batch = make_batch(cams=(), actions=False)
    batch["actions_chosen"] = torch.randn(B, H, A)
    batch["actions_rejected"] = torch.randn(B, H, A)
    out = preference_loss(policy, ref, batch, beta=0.5, n_draws=2, generator=torch.Generator().manual_seed(0))
    assert math.isclose(out["loss"].item(), math.log(2), rel_tol=1e-5)   # policy == reference
    out["loss"].backward()
    assert any(p.grad is not None for p in policy.parameters())
    assert all(p.grad is None for p in ref.parameters())
    with pytest.raises(NotImplementedError):
        build_preference_pairs([])


@pytest.mark.parametrize("head", ["chunk", "flow"])
def test_preference_loss_disables_dropout_and_shares_encoding(head):
    """With dropout / modality dropout configured, the default DPO loss is exactly log 2 at
    π_θ = π_ref (one deterministic encoding per model), grads flow and the train mode is restored;
    the implicit rewards match a hand computation from one shared encoding."""
    torch.manual_seed(0)
    policy = VTLAPolicy(tiny_cfg(head=head, vision=None, text=None, dropout=0.3, p_drop_tactile=0.5))
    torch.nn.init.normal_(policy.head.out.weight, std=0.5)          # memory-dependent predictions
    ref = make_reference_policy(policy)
    batch = make_batch(cams=(), actions=False)
    batch["actions_chosen"] = torch.randn(B, H, A)
    batch["actions_rejected"] = torch.randn(B, H, A)
    policy.train()
    for s in range(3):
        out = preference_loss(policy, ref, batch, beta=1.0, generator=torch.Generator().manual_seed(s))
        assert math.isclose(out["loss"].item(), math.log(2), rel_tol=1e-6)
        assert policy.training and not ref.training
    out["loss"].backward()
    assert policy.head.out.weight.grad is not None
    # a policy that moved: implicit reward = β·(log π_θ − log π_ref) with the surrogate log π
    with torch.no_grad():
        policy.head.out.bias.add_(0.3)
    g = torch.Generator().manual_seed(7)
    out = preference_loss(policy, ref, batch, beta=0.5, sigma=2.0, generator=g)
    g = torch.Generator().manual_seed(7)
    kw = {}
    if head == "flow":
        kw = {"noise": torch.randn(B, H, A, generator=g), "tau": policy.head.sample_tau(B, generator=g)}
    policy.eval()
    with torch.no_grad():
        ep, er = policy.encode(batch), ref.encode(batch)
        lp = {k: -policy.head.per_sample_error(ep["memory"], ep["memory_mask"], batch[k], **kw) / 8.0
              for k in ("actions_chosen", "actions_rejected")}
        lr = {k: -ref.head.per_sample_error(er["memory"], er["memory_mask"], batch[k], **kw) / 8.0
              for k in ("actions_chosen", "actions_rejected")}
    rc = 0.5 * (lp["actions_chosen"] - lr["actions_chosen"])
    rr = 0.5 * (lp["actions_rejected"] - lr["actions_rejected"])
    assert torch.allclose(out["reward_chosen"], rc.mean(), atol=1e-5)
    assert torch.allclose(out["loss"], -F.logsigmoid(rc - rr).mean(), atol=1e-5)
