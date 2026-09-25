import pickle
import sys
import types
import zlib

import pytest
import torch
import torch.nn as nn

from robot_skin.language import (
    BOS_ID, N_SPECIAL, PAD_ID, HashingTextEncoder, HashingTokenizer, HFTextEncoder, InstructionCache,
    TextEncoder, build_text_encoder, hash_token, pool_tokens, simple_word_tokenize,
)

SENTS = ["Pick up the red cup and place it on the tray.", "open the jar", "", "컵을 들어 올려"]


def test_word_tokenizer_and_hash():
    assert simple_word_tokenize("Pick up the RED cup!") == ["pick", "up", "the", "red", "cup"]
    assert simple_word_tokenize("컵을  들어, 올려") == ["컵을", "들어", "올려"]
    assert simple_word_tokenize("Ｃｕｐ") == ["cup"]                       # NFKC full-width
    assert simple_word_tokenize("Cup", lowercase=False) == ["Cup"]
    with pytest.raises(TypeError):
        simple_word_tokenize(None)
    ids = {hash_token(w, 1000) for w in ["cup", "jar", "tray", "pick", "place"]}
    assert all(N_SPECIAL <= i < 1000 for i in ids) and len(ids) == 5
    assert hash_token("cup", 1000, seed=0) == hash_token("cup", 1000, seed=0)
    assert hash_token("cup", 2 ** 16, seed=0) != hash_token("cup", 2 ** 16, seed=1)


def test_ids_deterministic_across_instances_and_seeds():
    torch.manual_seed(0)
    a = HashingTextEncoder(dim=16, max_len=12, hash_seed=7)
    torch.manual_seed(123)                                      # different weights, same hashing
    b = HashingTextEncoder(dim=32, max_len=12, hash_seed=7)
    ta, tb = a.tokenize(SENTS), b.tokenize(SENTS)
    assert torch.equal(ta["input_ids"], tb["input_ids"]) and torch.equal(ta["pad_mask"], tb["pad_mask"])
    c = HashingTextEncoder(dim=16, max_len=12, hash_seed=8)
    assert not torch.equal(c.tokenize(SENTS)["input_ids"], ta["input_ids"])
    # case / punctuation insensitive
    assert torch.equal(a.tokenize("OPEN the jar!!")["input_ids"], a.tokenize("open the jar")["input_ids"])


def test_padding_mask_bos_and_truncation():
    enc = HashingTextEncoder(dim=8, max_len=6)
    t = enc.tokenize(["open the jar", "", "a b c d e f g h i"])
    ids, pm = t["input_ids"], t["pad_mask"]
    assert ids.shape == pm.shape == (3, 6) and pm.dtype == torch.bool
    assert torch.all(ids[:, 0] == BOS_ID) and not pm[:, 0].any()           # BOS always valid
    assert pm[0].tolist() == [False] * 4 + [True] * 2                      # BOS + 3 words
    assert pm[1].tolist() == [False] + [True] * 5                          # empty → BOS only
    assert not pm[2].any()                                                 # truncated to max_len
    assert torch.equal(pm, ids == PAD_ID)
    assert enc.tokenize("a b c d e f g h i")["input_ids"][0, 1:].tolist() == \
        enc.tokenize("a b c d e")["input_ids"][0, 1:].tolist()             # keeps the first max_len-1 words
    dyn = HashingTextEncoder(dim=8, max_len=6, pad_to_max=False)
    assert dyn.tokenize(["open the jar", "pick"])["input_ids"].shape == (2, 4)
    assert dyn.fixed_len is None and enc.fixed_len == 6


@pytest.mark.parametrize("n_layers", [0, 1])
def test_encode_shapes_zero_padding_and_distinct_sentences(n_layers):
    torch.manual_seed(0)
    enc = HashingTextEncoder(dim=16, max_len=12, n_layers=n_layers, n_heads=2).eval()
    tok, pm = enc.encode(SENTS)
    assert tok.shape == (4, 12, 16) and pm.shape == (4, 12)
    assert torch.all(tok[pm] == 0) and torch.isfinite(tok).all()
    assert torch.all(tok[~pm].abs().sum(-1) > 0)
    pooled = pool_tokens(tok, pm)
    assert pooled.shape == (4, 16)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(pooled[i], pooled[j], atol=1e-4)
    a, _ = enc.encode(["pick up the cup"])
    b, _ = enc.encode(["pick up the jar"])
    assert not torch.allclose(a, b)
    # trainable, gradients reach embeddings
    enc.train()
    t, p = enc.encode(["pick up the cup"])
    pool_tokens(t, p).sum().backward()
    assert enc.tok_emb.weight.grad is not None and enc.pos_emb.weight.grad is not None
    assert enc.tok_emb.weight.grad[PAD_ID].abs().sum() == 0


def test_contextual_encoder_is_padding_invariant():
    torch.manual_seed(0)
    enc = HashingTextEncoder(dim=16, max_len=16, n_layers=2, n_heads=4, pad_to_max=False).eval()
    alone, _ = enc.encode(["pick up"])
    batched, pm = enc.encode(["pick up", "pick up the red cup and place it on the tray"])
    assert batched.shape[1] > alone.shape[1]
    assert torch.allclose(alone[0], batched[0, : alone.shape[1]], atol=1e-5)
    assert torch.all(batched[0, alone.shape[1]:] == 0)


def test_standalone_tokenizer_is_light_and_picklable():
    enc = HashingTextEncoder(dim=8, max_len=10, hash_seed=3)
    tok = enc.get_tokenizer()
    assert isinstance(tok, HashingTokenizer)
    clone = pickle.loads(pickle.dumps(tok))                                # for DataLoader workers
    standalone = HashingTokenizer(max_len=10, hash_seed=3)
    for t in (tok, clone, standalone):
        out = t(SENTS)
        assert torch.equal(out["input_ids"], enc.tokenize(SENTS)["input_ids"])
    ids = standalone(SENTS)
    tokens = enc(ids["input_ids"], ids["pad_mask"])                        # model side: tensors only
    assert torch.allclose(tokens, enc.encode(SENTS)[0])


def test_forward_validation_and_single_string():
    enc = HashingTextEncoder(dim=8, max_len=4)
    assert enc.encode("open jar")[0].shape == (1, 4, 8)                    # str → batch of one
    with pytest.raises(ValueError):
        enc(torch.ones(1, 5, dtype=torch.long))
    with pytest.raises(ValueError):
        HashingTextEncoder(max_len=1)
    with pytest.raises(TypeError):
        enc.tokenize(["ok", 3])


def test_build_text_encoder():
    enc = build_text_encoder({"type": "hashing", "dim": 24, "max_len": 10})
    assert isinstance(enc, HashingTextEncoder) and isinstance(enc, TextEncoder)
    assert enc.out_dim == 24 and enc.max_len == 10
    assert isinstance(build_text_encoder(), HashingTextEncoder)
    with pytest.raises(ValueError, match="unknown text encoder option"):
        build_text_encoder({"dim": 8, "vocab": 3})
    with pytest.raises(ValueError, match="unknown text encoder type"):
        build_text_encoder({"type": "bert"})


@pytest.mark.parametrize("cfg", [{"type": "hf", "model_id": "t5-small"}, {"type": "clip"},
                                 {"type": "siglip"}])
def test_hf_text_encoder_missing_transformers(monkeypatch, cfg):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError) as ei:
        build_text_encoder(cfg)
    assert "`transformers`" in str(ei.value) and "pip install transformers" in str(ei.value)


class _Counting(HashingTextEncoder):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = 0

    def forward(self, input_ids, pad_mask=None):
        self.calls += input_ids.shape[0]
        return super().forward(input_ids, pad_mask)


@pytest.mark.parametrize("pad_to_max", [True, False])
def test_instruction_cache_matches_encoder_and_memoizes(tmp_path, pad_to_max):
    torch.manual_seed(0)
    enc = _Counting(dim=16, max_len=12, n_layers=1, n_heads=2, pad_to_max=pad_to_max)
    with pytest.raises(ValueError, match="frozen"):
        InstructionCache(enc)
    enc.freeze()
    cache = InstructionCache(enc, batch_size=2)
    texts = ["open the jar", "pick up the cup", "open the jar", ""]
    tok, pm = cache.encode(texts)
    ref_tok, ref_pm = enc.encode(texts)
    assert torch.equal(pm, ref_pm) and torch.allclose(tok, ref_tok, atol=1e-5)
    assert len(cache) == 3 and "open the jar" in cache and cache.n_encoded == 3
    calls = enc.calls
    tok2, _ = cache.encode(["pick up the cup", "open the jar"])
    assert enc.calls == calls                                              # served from cache
    assert torch.allclose(tok2[1], tok[0, : tok2.shape[1]])
    assert cache.get("open the jar").shape == (4, 16)                      # valid tokens only
    # persistence
    path = cache.save(tmp_path / "instr.pt")
    loaded = InstructionCache.load(path, enc)
    assert len(loaded) == 3 and loaded.n_encoded == 0
    t3, p3 = loaded.encode(texts)
    assert torch.equal(t3, tok) and torch.equal(p3, pm)
    other = HashingTextEncoder(dim=16, max_len=12, hash_seed=1).freeze()
    with pytest.raises(ValueError, match="built with encoder"):
        InstructionCache.load(path, other)
    half = InstructionCache(enc, dtype=torch.float16)
    assert half.get("open the jar").dtype == torch.float16
    assert InstructionCache(HashingTextEncoder(dim=4), allow_trainable=True).encode("x")[0].shape[-1] == 4


# ── hashing is a fixed, documented function (ids must match across machines) ──
def test_hash_token_hand_values():
    # 2 + crc32("0:cup") % 998; crc32("0:cup") = 0xb049dea8 (IEEE CRC-32, re-derived bitwise)
    assert zlib.crc32(b"0:cup") == 0xB049DEA8 and 0xB049DEA8 % 998 == 260
    assert hash_token("cup", 1000) == 262
    assert hash_token("cup", 2 ** 16, seed=0) == 2 + zlib.crc32(b"0:cup") % (2 ** 16 - 2)
    ids = HashingTokenizer(max_len=4, vocab_size=1000)("Cup!")["input_ids"][0].tolist()
    assert ids == [BOS_ID, 262, PAD_ID, PAD_ID]


def test_pool_tokens_hand_values():
    tok = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    pm = torch.tensor([[False, False, True]])
    assert torch.allclose(pool_tokens(tok, pm), torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(pool_tokens(tok, torch.ones(1, 3, dtype=torch.bool)), torch.zeros(1, 2))


def test_hashing_cache_key_tracks_tokenization():
    a = HashingTextEncoder(dim=8, max_len=6)
    assert a.cache_key != HashingTextEncoder(dim=8, max_len=7).cache_key
    assert a.cache_key != HashingTextEncoder(dim=8, max_len=6, lowercase=False).cache_key


# ── HFTextEncoder wrapper logic with a fake `transformers` (no optional deps) ─
class _FakeTok:
    """Word-level tokenizer: id = 3 + (sum of code points % 50), no special tokens."""

    model_max_length = 16

    def __init__(self, with_mask=True):
        self.model_input_names = ["input_ids", "attention_mask"] if with_mask else ["input_ids"]
        self.padding_side = "left"          # the encoder must switch this to "right"

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        rows = [[3 + sum(map(ord, w)) % 50 for w in t.split()][:max_length] for t in texts]
        L = max_length if padding == "max_length" else max([len(r) for r in rows] + [1])
        ids = torch.zeros(len(rows), L, dtype=torch.long)
        am = torch.zeros_like(ids)
        for i, r in enumerate(rows):
            sl = slice(0, len(r)) if self.padding_side == "right" else slice(L - len(r), L)
            ids[i, sl], am[i, sl] = torch.tensor(r, dtype=torch.long), 1
        out = {"input_ids": ids}
        if "attention_mask" in self.model_input_names:
            out["attention_mask"] = am
        return out


class _FakeTextModel(nn.Module):
    """Embedding + masked-mean context: outputs depend on padding iff the mask is ignored."""

    def __init__(self, hidden=8, key="hidden_size"):
        super().__init__()
        self.config = types.SimpleNamespace(**{key: hidden})
        self.emb, self.lin, self.drop = nn.Embedding(64, hidden), nn.Linear(hidden, hidden), nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)
        m = torch.ones_like(h[..., :1]) if attention_mask is None else attention_mask[..., None].to(h)
        ctx = (h * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
        return types.SimpleNamespace(last_hidden_state=self.drop(self.lin(h + ctx)))


class _FakeDual(nn.Module):          # CLIPModel / SiglipModel layout
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace()
        self.text_model, self.vision_model = _FakeTextModel(), nn.Linear(2, 2)


class _FakeT5(nn.Module):            # encoder–decoder layout (d_model, get_encoder)
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(is_encoder_decoder=True)
        self.encoder, self.decoder = _FakeTextModel(6, key="d_model"), nn.Linear(2, 2)

    def get_encoder(self):
        return self.encoder


@pytest.fixture
def fake_transformers(monkeypatch):
    def model_for(mid):
        return _FakeT5() if "t5" in mid else _FakeDual()
    mod = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda mid, **kw: _FakeTok(with_mask="siglip" not in mid)),
        AutoModel=types.SimpleNamespace(from_pretrained=lambda mid, **kw: model_for(mid)),
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda mid, **kw: mid))
    mod.AutoModel.from_config = model_for
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return mod


def test_hf_text_wrapper_masks_padding_and_matches_cache(fake_transformers):
    torch.manual_seed(0)
    enc = build_text_encoder({"type": "clip"})
    assert isinstance(enc, HFTextEncoder) and isinstance(enc.backbone, _FakeTextModel)
    assert enc.tokenizer.padding_side == "right" and enc.max_len == 16 and enc.fixed_len is None
    assert enc.is_frozen and enc.frozen_backbone and enc.cache_key == "clip-vit-base-patch32"
    texts = ["pick up the cup", "open jar", ""]
    t = enc.tokenize(texts)
    assert t["pad_mask"].tolist() == [[False] * 4, [False, False, True, True], [False, True, True, True]]
    enc.train()                                                           # frozen: dropout stays off
    tok, pm = enc.encode(texts)
    assert torch.equal(tok, enc.encode(texts)[0]) and not tok.requires_grad
    assert torch.all(tok[pm] == 0)
    alone, _ = enc.encode(["open jar"])                                   # mask honoured → no padding leak
    assert torch.allclose(alone[0], tok[1, :2], atol=1e-6)
    cache = InstructionCache(enc)
    ct, cp = cache.encode(texts)
    assert torch.equal(cp, pm) and torch.allclose(ct, tok, atol=1e-6)
    pickle.loads(pickle.dumps(enc.get_tokenizer()))                       # collate-side tokenizer


def test_hf_text_wrapper_families_and_trainable_path(fake_transformers):
    torch.manual_seed(0)
    sig = build_text_encoder({"type": "siglip"})                         # no attention mask, max_length
    assert sig.fixed_len == 16
    tok, pm = sig.encode(["open jar"])
    assert tok.shape == (1, 16, 8) and not pm.any()
    t5 = build_text_encoder({"type": "t5", "out_dim": 4, "frozen": False, "pretrained": False})
    assert isinstance(t5.backbone, _FakeTextModel) and t5.out_dim == 4 and not t5.frozen_backbone
    assert t5.cache_key == "t5-small_scratch"
    out, pm = t5.encode(["pick up the cup", "open jar"])
    out.sum().backward()
    assert t5.backbone.emb.weight.grad is not None and pm.tolist()[1] == [False, False, True, True]
    frozen_t5 = build_text_encoder({"type": "t5"})
    for p in frozen_t5.backbone.parameters():                             # manual unfreeze is honoured
        p.requires_grad_(True)
    frozen_t5.encode(["open jar"])[0].sum().backward()
    assert frozen_t5.backbone.emb.weight.grad is not None
