"""Instruction text → language tokens for the VTLA policy.

Contract (:class:`TextEncoder`)::

    batch = enc.tokenize(["pick up the cup"])      # {"input_ids": [B,L] long, "pad_mask": [B,L] bool}
    tokens = enc(batch["input_ids"], batch["pad_mask"])   # [B,L,D] float, padding rows zeroed
    tokens, pad_mask = enc.encode(texts)            # both steps; pad_mask True = padding

``tokenize`` is pure CPU/python (can run in DataLoader workers / collate); ``forward`` is a tensor
op (DDP / torch.compile friendly). Encoders:

- :class:`HashingTextEncoder` — dependency-free, trainable: Unicode word tokenizer (lower-cased,
  NFKC; works for English and Korean) + stable ``zlib.crc32`` feature hashing into
  ``vocab_size`` buckets (no vocabulary file, deterministic across processes and machines) +
  learned token and position embeddings (+ optional small transformer). Position 0 is always a
  ``[BOS]`` summary token, so even an empty instruction has one valid token (no all-masked rows
  in attention). Instructions come from a small template catalog (``acquisition/protocols``), for
  which this is sufficient and fast.
- :class:`HFTextEncoder` — frozen pretrained text tower from HuggingFace (CLIP, SigLIP, T5 …;
  optional ``transformers``), with an optional trainable projection.

:class:`InstructionCache` memoizes a *frozen* encoder's output per distinct instruction string
(an episode has one instruction; a dataset has few distinct ones), so the text tower runs once
per string instead of once per sample.
"""
from __future__ import annotations

import importlib
import inspect
import re
import unicodedata
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

__all__ = [
    "PAD_ID", "BOS_ID", "N_SPECIAL", "simple_word_tokenize", "hash_token", "pool_tokens",
    "TextEncoder", "HashingTokenizer", "HashingTextEncoder", "HFTokenizerFn", "HFTextEncoder",
    "InstructionCache", "build_text_encoder",
]

PAD_ID = 0
BOS_ID = 1
N_SPECIAL = 2

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def simple_word_tokenize(text: str, *, lowercase: bool = True) -> list[str]:
    """NFKC-normalize, case-fold, split into Unicode word runs (punctuation dropped).

    ``"Pick up the RED cup!"`` → ``["pick", "up", "the", "red", "cup"]``; Hangul words are kept
    whole (``"컵을 들어"`` → ``["컵을", "들어"]``).
    """
    if not isinstance(text, str):
        raise TypeError(f"instruction must be a str, got {type(text).__name__}")
    text = unicodedata.normalize("NFKC", text)
    if lowercase:
        text = text.casefold()
    return _WORD_RE.findall(text)


def hash_token(token: str, vocab_size: int, seed: int = 0) -> int:
    """Stable id in ``[N_SPECIAL, vocab_size)``: ``crc32(f"{seed}:{token}") mod (vocab − N_SPECIAL)``."""
    if vocab_size <= N_SPECIAL:
        raise ValueError(f"vocab_size must be > {N_SPECIAL}")
    h = zlib.crc32(f"{int(seed)}:{token}".encode("utf-8"))
    return N_SPECIAL + h % (vocab_size - N_SPECIAL)


def pool_tokens(tokens: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over valid (non-padding) tokens: ``[B,L,D]`` → ``[B,D]`` (all-padding rows → 0)."""
    if pad_mask is None:
        return tokens.mean(1)
    w = (~pad_mask).to(tokens.dtype).unsqueeze(-1)
    return (tokens * w).sum(1) / w.sum(1).clamp_min(1.0)


def _as_list(texts: str | Iterable[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    out = list(texts)
    for t in out:
        if not isinstance(t, str):
            raise TypeError(f"instructions must be str, got {type(t).__name__}")
    return out


def _require(module: str, what: str, hint: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"{what} needs the optional package `{module}`, which is not installed ({e}). {hint} "
            f"Without it, use the dependency-free encoder: build_text_encoder({{'type': 'hashing'}}).") from e


# ── base ──────────────────────────────────────────────────────────────────────
class TextEncoder(nn.Module):
    """Base class: ``tokenize`` (python) + ``forward`` (tensors) + ``encode`` (both)."""

    out_dim: int
    max_len: int

    def get_tokenizer(self) -> Callable[[str | Sequence[str]], dict[str, torch.Tensor]]:
        """A light, picklable ``texts → {"input_ids", "pad_mask"}`` callable (no weights), for
        DataLoader workers / collate functions."""
        raise NotImplementedError

    def tokenize(self, texts: str | Sequence[str]) -> dict[str, torch.Tensor]:
        return self.get_tokenizer()(texts)

    def forward(self, input_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, texts: str | Sequence[str], device: torch.device | str | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """``texts`` → ``(tokens [B,L,D], pad_mask [B,L] bool, True = padding)``."""
        batch = self.tokenize(texts)
        dev = torch.device(device) if device is not None else self.device
        ids, pm = batch["input_ids"].to(dev), batch["pad_mask"].to(dev)
        return self(ids, pm), pm

    @property
    def device(self) -> torch.device:
        p = next(iter(self.parameters()), None)
        return p.device if p is not None else torch.device("cpu")

    @property
    def fixed_len(self) -> int | None:
        """Sequence length if every batch is padded to the same ``L`` (static shapes), else None."""
        return None

    @property
    def cache_key(self) -> str:
        return type(self).__name__.lower()

    @property
    def is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.parameters())

    def freeze(self) -> "TextEncoder":
        for p in self.parameters():
            p.requires_grad_(False)
        return self.eval()


# ── hashing encoder ───────────────────────────────────────────────────────────
class HashingTokenizer:
    """``texts → {"input_ids": long[B,L], "pad_mask": bool[B,L]}`` with ``[BOS]`` + hashed words.

    Pure python + torch tensors, picklable, no weights — what :class:`HashingTextEncoder` uses and
    what a dataset/collate function can hold instead of the encoder module.
    """

    def __init__(self, max_len: int = 32, vocab_size: int = 2 ** 16, *, hash_seed: int = 0,
                 lowercase: bool = True, pad_to_max: bool = True):
        if max_len < 2:
            raise ValueError("max_len must be >= 2 ([BOS] + at least one word)")
        if vocab_size <= N_SPECIAL:
            raise ValueError(f"vocab_size must be > {N_SPECIAL}")
        self.max_len, self.vocab_size, self.hash_seed = int(max_len), int(vocab_size), int(hash_seed)
        self.lowercase, self.pad_to_max = bool(lowercase), bool(pad_to_max)
        self._memo: dict[str, int] = {}

    def token_ids(self, text: str) -> list[int]:
        """``[BOS, id(w1), …]`` truncated to ``max_len`` (no padding)."""
        ids = [BOS_ID]
        for w in simple_word_tokenize(text, lowercase=self.lowercase)[: self.max_len - 1]:
            i = self._memo.get(w)
            if i is None:
                if len(self._memo) >= 100_000:
                    self._memo.clear()
                i = self._memo[w] = hash_token(w, self.vocab_size, self.hash_seed)
            ids.append(i)
        return ids

    def __call__(self, texts: str | Sequence[str]) -> dict[str, torch.Tensor]:
        rows = [self.token_ids(t) for t in _as_list(texts)]
        L = self.max_len if self.pad_to_max else max((len(r) for r in rows), default=1)
        ids = torch.full((len(rows), L), PAD_ID, dtype=torch.long)
        for b, r in enumerate(rows):
            ids[b, : len(r)] = torch.tensor(r, dtype=torch.long)
        return {"input_ids": ids, "pad_mask": ids == PAD_ID}

    def __repr__(self) -> str:
        return (f"HashingTokenizer(max_len={self.max_len}, vocab_size={self.vocab_size}, "
                f"hash_seed={self.hash_seed}, pad_to_max={self.pad_to_max})")


class HashingTextEncoder(TextEncoder):
    """Dependency-free trainable text encoder (feature hashing + learned embeddings).

    Args:
        dim: token width ``D``.
        max_len: sequence length including the ``[BOS]`` token (≤ ``max_len − 1`` words kept;
            longer instructions are truncated).
        vocab_size: hash buckets (incl. ``PAD_ID=0``, ``BOS_ID=1``).
        hash_seed: salt of the hash function (same seed ⇒ same ids everywhere).
        n_layers, n_heads, dropout: optional pre-LN transformer over the tokens (0 = none).
        lowercase: case-insensitive tokenization.
        pad_to_max: pad every batch to ``max_len`` (static shapes for torch.compile); False pads
            to the longest instruction in the batch.
    """

    def __init__(self, dim: int = 256, max_len: int = 32, vocab_size: int = 2 ** 16, *,
                 hash_seed: int = 0, n_layers: int = 0, n_heads: int = 4, dropout: float = 0.0,
                 lowercase: bool = True, pad_to_max: bool = True):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.tokenizer = HashingTokenizer(max_len, vocab_size, hash_seed=hash_seed,
                                          lowercase=lowercase, pad_to_max=pad_to_max)
        if n_layers and dim % n_heads:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.out_dim, self.max_len, self.vocab_size = int(dim), int(max_len), int(vocab_size)
        self.hash_seed, self.lowercase, self.pad_to_max = int(hash_seed), bool(lowercase), bool(pad_to_max)
        self.tok_emb = nn.Embedding(self.vocab_size, dim, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(self.max_len, dim)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[PAD_ID].zero_()
        self.norm = nn.LayerNorm(dim)
        self.context = None
        if n_layers:
            layer = nn.TransformerEncoderLayer(dim, n_heads, dim_feedforward=4 * dim, dropout=dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(dim),
                                                 enable_nested_tensor=False)

    @property
    def fixed_len(self) -> int | None:
        return self.max_len if self.pad_to_max else None

    @property
    def cache_key(self) -> str:
        # identifies the tokenization + shapes, not the trained weights (InstructionCache users:
        # a trained-then-frozen hashing encoder needs its own cache file per checkpoint)
        case = "" if self.lowercase else "_cased"
        return f"hash_v{self.vocab_size}_s{self.hash_seed}_d{self.out_dim}_L{self.max_len}{case}"

    def get_tokenizer(self) -> HashingTokenizer:
        return self.tokenizer

    def token_ids(self, text: str) -> list[int]:
        return self.tokenizer.token_ids(text)

    def forward(self, input_ids: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B,L], got {tuple(input_ids.shape)}")
        L = input_ids.shape[1]
        if L > self.max_len:
            raise ValueError(f"sequence length {L} exceeds max_len {self.max_len}")
        if pad_mask is None:
            pad_mask = input_ids == PAD_ID
        x = self.tok_emb(input_ids) + self.pos_emb(torch.arange(L, device=input_ids.device))
        x = self.norm(x)
        if self.context is not None:
            x = self.context(x, src_key_padding_mask=pad_mask)
        return x.masked_fill(pad_mask.unsqueeze(-1), 0.0)


# ── HuggingFace encoder ───────────────────────────────────────────────────────
class HFTokenizerFn:
    """Picklable wrapper: HF tokenizer → ``{"input_ids", "pad_mask"}`` (see :class:`HFTextEncoder`)."""

    def __init__(self, tokenizer, max_len: int, padding: str, use_mask: bool):
        self.tokenizer, self.max_len, self.padding, self.use_mask = tokenizer, int(max_len), padding, use_mask

    def __call__(self, texts: str | Sequence[str]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(_as_list(texts), padding=self.padding, truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        ids = enc["input_ids"]
        if self.use_mask and "attention_mask" in enc:
            pad_mask = enc["attention_mask"] == 0
        else:
            pad_mask = torch.zeros_like(ids, dtype=torch.bool)
        # never emit an all-padding row (empty text with a tokenizer that adds no special
        # tokens): a fully masked key set makes downstream attention NaN
        pad_mask[pad_mask.all(dim=1), 0] = False
        return {"input_ids": ids, "pad_mask": pad_mask}


class HFTextEncoder(TextEncoder):
    """Pretrained HuggingFace text tower (frozen by default) → ``last_hidden_state`` tokens.

    For dual encoders (``CLIPModel``, ``SiglipModel``) the ``text_model`` tower is used; for
    encoder–decoder models (T5) the encoder. ``padding``: ``"longest"`` (default) or
    ``"max_length"`` (SigLIP was trained with max-length padding to 64 tokens; the family
    shortcut in :func:`build_text_encoder` sets that). If the tokenizer produces no attention mask
    (SigLIP) every position is treated as valid.
    """

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", *, frozen: bool = True,
                 max_len: int = 77, out_dim: int | None = None, padding: str = "longest",
                 pretrained: bool = True, revision: str | None = None,
                 local_files_only: bool = False):
        super().__init__()
        if padding not in ("longest", "max_length"):
            raise ValueError("padding must be 'longest' or 'max_length'")
        tf = _require("transformers", f"HFTextEncoder({model_id!r})",
                      "Install it with: pip install transformers (plus sentencepiece for T5/SigLIP "
                      "tokenizers).")
        kw: dict[str, Any] = {"local_files_only": local_files_only}
        if revision is not None:
            kw["revision"] = revision
        self.tokenizer = tf.AutoTokenizer.from_pretrained(model_id, **kw)
        # right padding: valid tokens form a prefix (InstructionCache re-pads on the right, and the
        # pad-mask contract assumes it); decoder-style tokenizers may default to left padding
        if getattr(self.tokenizer, "padding_side", "right") != "right":
            self.tokenizer.padding_side = "right"
        if pretrained:
            model = tf.AutoModel.from_pretrained(model_id, **kw)
        else:
            model = tf.AutoModel.from_config(tf.AutoConfig.from_pretrained(model_id, **kw))
        if hasattr(model, "text_model") and hasattr(model, "vision_model"):
            model = model.text_model
        elif getattr(model.config, "is_encoder_decoder", False) and hasattr(model, "get_encoder"):
            model = model.get_encoder()
        cfg = model.config
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        if hidden is None:
            raise ValueError(f"cannot find the hidden size in the config of {model_id!r}")
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and 0 < tok_max < 100_000:
            max_len = min(int(max_len), tok_max)
        self.backbone = model
        self.model_id, self.padding, self.max_len = model_id, padding, int(max_len)
        self.pretrained, self.revision = bool(pretrained), revision
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.proj = nn.Linear(int(hidden), int(out_dim)) if out_dim else nn.Identity()
        self.out_dim = int(out_dim or hidden)
        self._use_mask = "attention_mask" in getattr(self.tokenizer, "model_input_names",
                                                     ["input_ids", "attention_mask"])
        try:
            self._accepts_mask = "attention_mask" in inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            self._accepts_mask = True
        self.train(self.training)

    @property
    def frozen_backbone(self) -> bool:
        """True while no backbone parameter requires grad (read live, so manual unfreezing for
        fine-tuning re-enables gradients and, at the next ``.train()``, dropout)."""
        return not any(p.requires_grad for p in self.backbone.parameters())

    def train(self, mode: bool = True) -> "HFTextEncoder":
        super().train(mode)
        if "backbone" in self._modules and self.frozen_backbone:
            self.backbone.eval()
        return self

    @property
    def fixed_len(self) -> int | None:
        return self.max_len if self.padding == "max_length" else None

    @property
    def cache_key(self) -> str:
        key = self.model_id.rstrip("/").rsplit("/", 1)[-1]
        if self.revision:
            key += f"_rev-{str(self.revision)[:12]}"
        if not self.pretrained:
            key += "_scratch"
        return re.sub(r"[^A-Za-z0-9_.\-]+", "_", key)

    def get_tokenizer(self) -> HFTokenizerFn:
        return HFTokenizerFn(self.tokenizer, self.max_len, self.padding, self._use_mask)

    def forward(self, input_ids: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        if pad_mask is None:
            pad_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        kw = {}
        if self._use_mask and self._accepts_mask:
            kw["attention_mask"] = (~pad_mask).long()
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_backbone):
            h = self.backbone(input_ids=input_ids, **kw).last_hidden_state
        x = self.proj(h)
        return x.masked_fill(pad_mask.unsqueeze(-1), 0.0)


# ── factory ───────────────────────────────────────────────────────────────────
_HF_FAMILIES: dict[str, dict[str, Any]] = {
    "clip": {"model_id": "openai/clip-vit-base-patch32", "max_len": 77},
    "siglip": {"model_id": "google/siglip-base-patch16-224", "max_len": 64, "padding": "max_length"},
    "t5": {"model_id": "t5-small", "max_len": 64},
}
_TYPES = {"hashing": HashingTextEncoder, "hash": HashingTextEncoder,
          "hf": HFTextEncoder, "transformers": HFTextEncoder, "huggingface": HFTextEncoder}


def build_text_encoder(cfg: Mapping[str, Any] | None = None, **overrides) -> TextEncoder:
    """Text encoder from a config dict (YAML ``language:`` block).

    ``type``: ``hashing`` (default, no deps) | ``hf`` / ``transformers`` (needs ``model_id``) |
    family shortcuts ``clip``, ``siglip``, ``t5`` (default checkpoints and padding). Other keys go
    to the constructor; unknown keys raise ``ValueError``; a missing ``transformers`` raises
    ``ImportError`` with an install hint.
    """
    c = dict(cfg or {})
    c.update(overrides)
    typ = str(c.pop("type", "hashing")).lower()
    if typ in _HF_FAMILIES:
        for k, v in _HF_FAMILIES[typ].items():
            c.setdefault(k, v)
        typ = "hf"
    if typ not in _TYPES:
        raise ValueError(f"unknown text encoder type {typ!r}; choose from "
                         f"{sorted(set(_TYPES) | set(_HF_FAMILIES))}")
    cls = _TYPES[typ]
    ok = [n for n in inspect.signature(cls.__init__).parameters if n != "self"]
    bad = sorted(set(c) - set(ok))
    if bad:
        raise ValueError(f"unknown text encoder option(s) {bad} for {cls.__name__}; accepted: {ok}")
    return cls(**c)


# ── instruction cache ─────────────────────────────────────────────────────────
class InstructionCache:
    """Memoized ``instruction → tokens`` for a **frozen** :class:`TextEncoder`.

    Stores only the valid (non-padding) tokens of each string (CPU, ``dtype``) and re-pads on
    :meth:`encode` to ``encoder.fixed_len`` or the longest entry in the batch, so the output has
    the same contract as ``encoder.encode``. Save/load lets a GPU box precompute large frozen
    towers once (``cache.save(run_dir / "instructions.pt")``).

    Args:
        encoder: the text encoder (must be frozen unless ``allow_trainable``: a trainable
            encoder's cache would be stale after the first optimizer step).
        batch_size: strings per encoder call when filling misses.
        dtype: storage dtype (e.g. ``torch.float16`` to halve memory).
    """

    def __init__(self, encoder: TextEncoder, *, batch_size: int = 64, dtype: torch.dtype = torch.float32,
                 allow_trainable: bool = False):
        if not allow_trainable and not encoder.is_frozen:
            raise ValueError("InstructionCache needs a frozen text encoder (call encoder.freeze(), or use "
                             "frozen=True for HFTextEncoder); pass allow_trainable=True to override")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.encoder, self.batch_size, self.dtype = encoder, int(batch_size), dtype
        self._store: dict[str, torch.Tensor] = {}
        self.n_encoded = 0   # strings actually run through the encoder (diagnostics)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, text: str) -> bool:
        return text in self._store

    @property
    def key(self) -> str:
        return self.encoder.cache_key

    def warm(self, texts: Iterable[str]) -> int:
        """Encode every not-yet-cached string (deduplicated); returns how many were added."""
        todo = list(dict.fromkeys(t for t in _as_list(texts) if t not in self._store))
        if not todo:
            return 0
        enc = self.encoder
        was = enc.training
        enc.eval()
        try:
            with torch.no_grad():
                for s in range(0, len(todo), self.batch_size):
                    chunk = todo[s:s + self.batch_size]
                    tok, pm = enc.encode(chunk)
                    for i, t in enumerate(chunk):
                        self._store[t] = tok[i][~pm[i]].detach().to("cpu", self.dtype).clone()
        finally:
            enc.train(was)
        self.n_encoded += len(todo)
        return len(todo)

    def get(self, text: str) -> torch.Tensor:
        """Valid tokens ``[L_i, D]`` of one instruction (encodes it on a miss)."""
        self.warm([text])
        return self._store[text]

    def encode(self, texts: str | Sequence[str], device: torch.device | str | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same output as :meth:`TextEncoder.encode`: ``(tokens [B,L,D], pad_mask [B,L])``.

        Returned on CPU unless ``device`` is given (so it can run inside a collate function)."""
        texts = _as_list(texts)
        self.warm(texts)
        rows = [self._store[t] for t in texts]
        D = self.encoder.out_dim
        longest = max((r.shape[0] for r in rows), default=1)
        L = max(self.encoder.fixed_len or 0, longest, 1)
        tok = torch.zeros(len(rows), L, D, dtype=torch.float32)
        pm = torch.ones(len(rows), L, dtype=torch.bool)
        for b, r in enumerate(rows):
            tok[b, : r.shape[0]] = r.float()
            pm[b, : r.shape[0]] = False
        if device is not None:
            tok, pm = tok.to(device), pm.to(device)
        return tok, pm

    # ── persistence ───────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {"key": self.key, "out_dim": int(self.encoder.out_dim), "entries": dict(self._store)}

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        if strict and state.get("key") != self.key:
            raise ValueError(f"instruction cache was built with encoder {state.get('key')!r}, "
                             f"this encoder is {self.key!r}")
        if int(state.get("out_dim", self.encoder.out_dim)) != int(self.encoder.out_dim):
            raise ValueError("instruction cache token width does not match the encoder")
        self._store.update({str(k): v.to("cpu", self.dtype) for k, v in state["entries"].items()})

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(self.state_dict(), tmp)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path, encoder: TextEncoder, *, strict: bool = True, **kw) -> "InstructionCache":
        cache = cls(encoder, **kw)
        cache.load_state_dict(torch.load(Path(path), map_location="cpu", weights_only=True), strict=strict)
        return cache
