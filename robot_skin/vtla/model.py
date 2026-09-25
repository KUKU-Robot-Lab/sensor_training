"""VTLA policy: vision + tactile + language (+ proprio) → action chunk, and its deployable bundle.

Architecture (one fusion transformer over typed tokens, Octo-style — Octo Model Team,
arXiv:2405.12213 — with the tactile branch of this repo)::

    instruction ─ TextEncoder (language.build_text_encoder) ─ Linear ────────────────┐ lang tokens
    camera c    ─ VisionEncoder (vision.build_vision_encoder) ─ Linear + cam_emb[c] ─┤ P per camera
                   (or cached frozen features [P,Dv] from vision.feature_cache)       │ (× obs_history)
    taxels      ─ TaxelEncoder (representation, optionally stage-2 pretrained/frozen)│
                  → TactileTokenAdapter (K Perceiver queries) → ContactGate ─────────┤ K tokens
    proprio     ─ MLP (current hand action / robot q, normalized, × obs_history) ────┤ 1 token
    readout     ─ learned ─────────────────────────────────────────────────────────┘ R tokens
          + modality type embedding → pre-LN TransformerEncoder (key padding = absent tokens)
          → action head (heads.ChunkRegressionHead | heads.FlowMatchingHead) → [B,H,A] normalized

Design points
- **Tactile values** come from :func:`robot_skin.representation.tactile_value_features` via the
  :class:`~robot_skin.representation.TactileFeatureSpec` stored in :class:`VTLAConfig` (the same
  function online control uses). Taxels are a pose-tokenized set (3D-ViTac, arXiv:2410.24091), so
  glove and robot layouts share weights; ``taxel_pad`` supports mixed-layout batches.
- **No hallucinated touch**: the :class:`~robot_skin.vtla.adapter.ContactGate` zeroes the K tactile
  tokens of a sample with no contact (``contact = level ≥ WEAK``), so drift without contact cannot
  reach the fusion (only the constant modality embedding does).
- **obs_mode ablation** (``feature_spec.obs_mode`` ∈ full / ordinal / binary / none): the feature
  subset changes the value width; ``none`` builds **no** tactile branch at all (no tokens).
- **Modality dropout** (training only, per sample): ``p_drop_tactile`` / ``p_drop_vision`` /
  ``p_drop_language`` hide the whole modality's tokens through the fusion key-padding mask — the
  policy learns to act when a sensor is missing and cannot lean on one modality. Masked tokens stay
  in the autograd graph (zero gradient), so DDP needs no ``find_unused_parameters``.
- **Aux contact head** (``aux_contact_weight > 0``): per-taxel logit from the tactile encoder tokens,
  BCE on the dataset's ``contact_target`` (shapes the tactile representation around contact).
- Actions/proprio are **normalized** (``action.space.ActionNormalizer``); the head predicts
  normalized chunks and :meth:`VTLAPolicy.predict` returns normalized actions — undo with the
  bundle's normalizer (and ``make_absolute`` for relative actions).

Module attribute names (for ``train.lr_mult`` prefixes): ``text_encoder``, ``text_proj``,
``vision_encoder``, ``vision_proj``, ``camera_emb``, ``history_emb`` (vision with
``obs_history > 1`` only), ``tactile_encoder``, ``adapter``, ``contact_head``, ``proprio_mlp``,
``readout``, ``type_emb``, ``fusion``, ``head``.

ContactGate caveat: with the default ``contact_rule: level_ge_weak`` a SATURATED taxel counts as
contact (a rail-clipped press is contact). A *permanently* saturated taxel — a dead channel, which
preprocessing marks saturated in every frame — therefore keeps the gate open for every sample; mask
such taxels (``taxel_pad``) or use ``weak_or_strong`` on hardware with dead channels.

Bundle (``policy_bundle.pt``, written by ``stages/vtla.py``): everything control needs to rebuild
the policy — :func:`save_policy_bundle`, :func:`read_policy_bundle`,
:func:`build_policy_from_bundle`, :func:`bundle_components`. Loadable with
``torch.load(weights_only=True)`` (plain containers + tensors only).
"""
from __future__ import annotations

import copy
import math
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from ..representation.encoder import TactileFeatureSpec, TaxelEncoder
from .adapter import TactileTokenAdapter
from .heads import HEAD_TYPES, TAU_DISTS, build_head
from .losses import contact_bce

__all__ = [
    "MODALITIES", "LANG", "VISION", "TACTILE", "PROPRIO", "READOUT", "VTLAConfig", "VTLAPolicy",
    "POLICY_BUNDLE_NAME", "BUNDLE_FORMAT", "BUNDLE_VERSION", "save_policy_bundle",
    "read_policy_bundle", "build_policy_from_bundle", "bundle_components",
]

MODALITIES = ("language", "vision", "tactile", "proprio", "readout")
LANG, VISION, TACTILE, PROPRIO, READOUT = range(len(MODALITIES))

_TACTILE_ENCODER_DEFAULTS = dict(d_model=64, depth=2, heads=4, n_fourier=8, fourier_scale=0.05,
                                 ff_mult=4, dropout=0.0)


def _default_vision() -> dict:
    return {"type": "tiny", "out_dim": 128}


def _default_text() -> dict:
    return {"type": "hashing", "dim": 128, "max_len": 32}


def _default_spec() -> dict:
    return TactileFeatureSpec().to_dict()


def _default_tactile_encoder() -> dict:
    return dict(_TACTILE_ENCODER_DEFAULTS)


@dataclass
class VTLAConfig:
    """Architecture of :class:`VTLAPolicy` (JSON-able; stored in ``policy_bundle.pt``).

    Sizes: ``action_dim`` A, ``horizon`` H, ``proprio_dim`` P per history step, ``obs_history`` k
    (policy ticks of proprio / camera frames stacked), ``d_model`` fusion width.
    Branches: ``vision`` (``build_vision_encoder`` cfg; ``None`` or no ``cameras`` → no vision),
    ``text`` (``build_text_encoder`` cfg; ``None`` → no language), tactile = ``feature_spec``
    (:class:`TactileFeatureSpec` dict; ``obs_mode: none`` → no tactile branch) +
    ``tactile_encoder`` (:class:`TaxelEncoder` kwargs without ``value_dim``/``feature_spec``).
    ``*_frozen`` freeze the corresponding encoder. ``head``: ``chunk`` | ``flow``.
    """
    action_dim: int = 54
    horizon: int = 16
    proprio_dim: int = 54
    obs_history: int = 1
    d_model: int = 128
    fusion_depth: int = 2
    fusion_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.0
    n_readout: int = 1
    cameras: tuple[str, ...] = ("ego",)
    vision: dict | None = field(default_factory=_default_vision)
    vision_frozen: bool = False
    text: dict | None = field(default_factory=_default_text)
    text_frozen: bool = False
    feature_spec: dict = field(default_factory=_default_spec)
    tactile_encoder: dict | None = field(default_factory=_default_tactile_encoder)
    tactile_frozen: bool = False
    n_tactile_tokens: int = 4
    tactile_heads: int = 4
    tactile_gate: str = "hard"
    head: str = "chunk"
    head_depth: int = 2
    head_heads: int | None = None
    flow_steps: int = 10
    flow_tau: str = "uniform"
    flow_tau_beta_b: float = 1.5
    p_drop_tactile: float = 0.0
    p_drop_vision: float = 0.0
    p_drop_language: float = 0.0
    aux_contact_weight: float = 0.0
    contact_pos_weight: float | None = None

    def __post_init__(self) -> None:
        self.cameras = tuple(str(c) for c in (self.cameras or ()))
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError(f"duplicate cameras {self.cameras}")
        spec = TactileFeatureSpec.from_dict(self.feature_spec)      # validates
        self.feature_spec = spec.to_dict()
        if self.tactile_encoder is not None:
            te = {k: v for k, v in dict(self.tactile_encoder).items()
                  if k not in ("value_dim", "feature_spec")}
            self.tactile_encoder = {**_TACTILE_ENCODER_DEFAULTS, **te}
        for name in ("action_dim", "horizon", "proprio_dim", "obs_history", "d_model",
                     "fusion_depth", "fusion_heads", "n_readout", "n_tactile_tokens", "flow_steps"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
            setattr(self, name, int(getattr(self, name)))
        if self.d_model % self.fusion_heads:
            raise ValueError(f"d_model={self.d_model} not divisible by fusion_heads={self.fusion_heads}")
        if self.head not in HEAD_TYPES:
            raise ValueError(f"head must be one of {HEAD_TYPES}, got {self.head!r}")
        if self.flow_tau not in TAU_DISTS:
            raise ValueError(f"flow_tau must be one of {TAU_DISTS}, got {self.flow_tau!r}")
        if self.tactile_gate not in ("hard", "soft"):
            raise ValueError("tactile_gate must be hard|soft")
        for name in ("p_drop_tactile", "p_drop_vision", "p_drop_language"):
            p = float(getattr(self, name))
            if not 0.0 <= p < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {p}")
            setattr(self, name, p)
        if float(self.aux_contact_weight) < 0:
            raise ValueError("aux_contact_weight must be >= 0")

    @property
    def spec(self) -> TactileFeatureSpec:
        return TactileFeatureSpec.from_dict(self.feature_spec)

    @property
    def obs_mode(self) -> str:
        return self.feature_spec["obs_mode"]

    @property
    def use_tactile(self) -> bool:
        return self.tactile_encoder is not None and self.spec.dim > 0

    @property
    def use_vision(self) -> bool:
        return self.vision is not None and len(self.cameras) > 0

    @property
    def use_language(self) -> bool:
        return self.text is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cameras"] = list(self.cameras)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "VTLAConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown VTLAConfig keys {unknown} (known: {sorted(known)})")
        return cls(**copy.deepcopy(d))


def _set_frozen(module: nn.Module | None) -> None:
    if module is None:
        return
    if hasattr(module, "freeze"):
        module.freeze()
    else:
        module.requires_grad_(False)
        module.eval()


class VTLAPolicy(nn.Module):
    """Vision–tactile–language–action policy (see the module docstring).

    Args:
        cfg: :class:`VTLAConfig` or its dict.
        tactile_encoder: optional pre-built :class:`TaxelEncoder` (e.g. from
            :func:`robot_skin.representation.load_pretrained_encoder`); its config replaces
            ``cfg.tactile_encoder`` and its ``feature_spec`` (if set) must equal ``cfg.feature_spec``.

    Batch keys (from :func:`robot_skin.vtla.dataset.collate_vtla`; ``B`` samples):
    ``proprio [B, k·P]``; ``images {cam: [B,(k,)3,H,W]}`` *or* ``vision_feats {cam: [B,(k,)P,Dv]}``
    (+ optional ``vision_valid {cam: [B(,k)] bool}``); ``tactile_values [B,N,F]``, ``taxel_pos``,
    ``taxel_nrm [B,N,3]``, ``contact [B,N] bool`` (+ ``taxel_pad [B,N]``); ``input_ids`` /
    ``text_pad_mask [B,L]`` *or* ``instruction`` (list of str); for training ``actions [B,H,A]``,
    ``action_valid [B,H]`` and optionally ``contact_target`` / ``contact_target_mask [B,N]``.
    """

    def __init__(self, cfg: VTLAConfig | Mapping[str, Any] | None = None, *,
                 tactile_encoder: TaxelEncoder | None = None) -> None:
        super().__init__()
        cfg = cfg if isinstance(cfg, VTLAConfig) else VTLAConfig.from_dict(cfg)
        cfg = copy.deepcopy(cfg)
        d = cfg.d_model

        # ── language
        self.text_encoder = self.text_proj = None
        if cfg.use_language:
            from ..language import build_text_encoder
            self.text_encoder = build_text_encoder(cfg.text)
            self.text_proj = nn.Linear(self.text_encoder.out_dim, d)
            if cfg.text_frozen:
                _set_frozen(self.text_encoder)

        # ── vision
        self.vision_encoder = self.vision_proj = None
        self.camera_emb = self.history_emb = None
        if cfg.use_vision:
            from ..vision import build_vision_encoder
            self.vision_encoder = build_vision_encoder(cfg.vision)
            self.vision_proj = nn.Linear(self.vision_encoder.out_dim, d)
            self.camera_emb = nn.Parameter(torch.randn(len(cfg.cameras), d) * 0.02)
            # per-history-step embedding of the camera tokens (proprio history is flattened into
            # the proprio MLP input instead); only built when it is used, so every trainable
            # parameter gets a gradient (DDP without find_unused_parameters)
            if cfg.obs_history > 1:
                self.history_emb = nn.Parameter(torch.randn(cfg.obs_history, d) * 0.02)
            if cfg.vision_frozen:
                _set_frozen(self.vision_encoder)
        else:
            cfg.cameras = ()

        # ── tactile
        self.tactile_encoder = self.adapter = self.contact_head = None
        spec = cfg.spec
        if tactile_encoder is not None and spec.dim > 0:
            if tactile_encoder.value_dim != spec.dim:
                raise ValueError(f"tactile encoder value_dim {tactile_encoder.value_dim} != "
                                 f"feature spec dim {spec.dim} ({spec})")
            if tactile_encoder.feature_spec is not None and tactile_encoder.feature_spec != spec:
                raise ValueError(f"tactile encoder feature_spec {tactile_encoder.feature_spec} != "
                                 f"cfg.feature_spec {spec}")
            ecfg = dict(tactile_encoder.config)
            ecfg.pop("value_dim", None)
            ecfg.pop("feature_spec", None)
            cfg.tactile_encoder = ecfg
        if cfg.use_tactile:
            if tactile_encoder is None:
                tactile_encoder = TaxelEncoder(spec.dim, **cfg.tactile_encoder, feature_spec=spec)
            elif tactile_encoder.feature_spec is None:
                tactile_encoder.feature_spec = spec
            self.tactile_encoder = tactile_encoder
            # the [MASK] vector is only used by masked pretraining → never gets a gradient here
            self.tactile_encoder.tokenizer.mask_token.requires_grad_(False)
            dt = self.tactile_encoder.d_model
            self.adapter = TactileTokenAdapter(dt, d, n_query=cfg.n_tactile_tokens,
                                               n_heads=cfg.tactile_heads, gate=cfg.tactile_gate)
            if cfg.aux_contact_weight > 0:
                self.contact_head = nn.Linear(dt, 1)
            if cfg.tactile_frozen:
                _set_frozen(self.tactile_encoder)
        elif cfg.aux_contact_weight > 0:
            warnings.warn("aux_contact_weight > 0 without a tactile branch (obs_mode none): "
                          "no auxiliary contact loss", stacklevel=2)

        # ── proprio, readout, types, fusion, head
        self.proprio_mlp = nn.Sequential(nn.Linear(cfg.proprio_dim * cfg.obs_history, d), nn.GELU(),
                                         nn.Linear(d, d))
        self.readout = nn.Parameter(torch.randn(cfg.n_readout, d) * 0.02)
        self.type_emb = nn.Embedding(len(MODALITIES), d)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.fusion_heads, dim_feedforward=cfg.ff_mult * d,
                                           dropout=cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, cfg.fusion_depth, norm=nn.LayerNorm(d),
                                            enable_nested_tensor=False)
        self.head = build_head(cfg.head, d, cfg.action_dim, cfg.horizon, depth=cfg.head_depth,
                               heads=cfg.head_heads or cfg.fusion_heads, ff_mult=cfg.ff_mult,
                               dropout=cfg.dropout, n_steps=cfg.flow_steps, tau_dist=cfg.flow_tau,
                               tau_beta_b=cfg.flow_tau_beta_b)
        self.cfg = cfg

    # ── properties ────────────────────────────────────────────────────────────────────
    @property
    def config(self) -> dict:
        """JSON-able :class:`VTLAConfig` dict (:meth:`from_config` inverse)."""
        return self.cfg.to_dict()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "VTLAPolicy":
        return cls(VTLAConfig.from_dict(cfg))

    @property
    def cameras(self) -> tuple[str, ...]:
        return self.cfg.cameras

    @property
    def feature_spec(self) -> TactileFeatureSpec:
        return self.cfg.spec

    @property
    def obs_mode(self) -> str:
        return self.cfg.obs_mode

    @property
    def horizon(self) -> int:
        return self.cfg.horizon

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    def train(self, mode: bool = True) -> "VTLAPolicy":
        """Keep frozen encoders in eval mode (no dropout / BN updates) while training the rest."""
        super().train(mode)
        for frozen, m in ((self.cfg.vision_frozen, self.vision_encoder),
                          (self.cfg.text_frozen, self.text_encoder),
                          (self.cfg.tactile_frozen, self.tactile_encoder)):
            if frozen and m is not None:
                m.eval()
        return self

    # ── observation encoding ──────────────────────────────────────────────────────────
    def _drop(self, p: float, B: int, device: torch.device) -> torch.Tensor | None:
        if not self.training or p <= 0:
            return None
        return torch.rand(B, device=device) < p

    def _language(self, batch: Mapping[str, Any], B: int, device: torch.device):
        if "input_ids" in batch:
            ids = batch["input_ids"].to(device)
            pm = batch.get("text_pad_mask")
            pm = (ids == 0) if pm is None else pm.to(device).bool()
        elif "instruction" in batch:
            texts = batch["instruction"]
            texts = [texts] if isinstance(texts, str) else [str(t) for t in texts]
            if len(texts) != B:
                raise ValueError(f"{len(texts)} instructions for a batch of {B}")
            tok = self.text_encoder.tokenize(texts)
            ids, pm = tok["input_ids"].to(device), tok["pad_mask"].to(device).bool()
        else:
            raise KeyError("batch needs 'input_ids' (collate with a tokenizer) or 'instruction'")
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cfg.text_frozen):
            tokens = self.text_encoder(ids, pm)
        return self.text_proj(tokens.to(self.text_proj.weight.dtype)), pm.clone()

    def _vision(self, batch: Mapping[str, Any], B: int, device: torch.device):
        k = self.cfg.obs_history
        feats_in = batch.get("vision_feats") or {}
        imgs = batch.get("images") or {}
        valid_in = batch.get("vision_valid") or {}
        toks, masks = [], []
        for ci, cam in enumerate(self.cameras):
            if cam in feats_in:
                f = feats_in[cam].to(device)
                if f.ndim == 3:
                    f = f.unsqueeze(1)
                if f.ndim != 4 or f.shape[0] != B or f.shape[1] != k:
                    raise ValueError(f"vision_feats[{cam!r}] must be [B,{k},P,D] (or [B,P,D] for k=1), "
                                     f"got {tuple(feats_in[cam].shape)}")
            elif cam in imgs:
                x = imgs[cam].to(device)
                if x.ndim == 4:
                    x = x.unsqueeze(1)
                if x.ndim != 5 or x.shape[0] != B or x.shape[1] != k:
                    raise ValueError(f"images[{cam!r}] must be [B,{k},3,H,W] (or [B,3,H,W] for k=1), "
                                     f"got {tuple(imgs[cam].shape)}")
                with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cfg.vision_frozen):
                    f = self.vision_encoder(x.reshape(B * k, *x.shape[2:]))
                f = f.reshape(B, k, *f.shape[1:])
            else:
                raise KeyError(f"batch has neither images nor vision_feats for camera {cam!r}")
            t = self.vision_proj(f.to(self.vision_proj.weight.dtype)) + self.camera_emb[ci]
            if self.history_emb is not None:
                t = t + self.history_emb[:, None, :]
            P = t.shape[2]
            toks.append(t.reshape(B, k * P, t.shape[-1]))
            v = valid_in.get(cam)
            if v is None:
                v = torch.ones(B, k, dtype=torch.bool, device=device)
            else:
                v = v.to(device).bool().reshape(B, -1)
                if v.shape[1] == 1 and k > 1:
                    v = v.expand(B, k)
            masks.append((~v)[:, :, None].expand(B, k, P).reshape(B, k * P))
        return torch.cat(toks, 1), torch.cat(masks, 1)

    def _tactile(self, batch: Mapping[str, Any], B: int, device: torch.device):
        values = batch["tactile_values"].to(device)
        pos, nrm = batch["taxel_pos"].to(device), batch["taxel_nrm"].to(device)
        tp = batch.get("taxel_pad")
        tp = None if tp is None else tp.to(device).bool()
        contact = batch["contact"].to(device).bool()
        if tp is not None:
            contact = contact & ~tp
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cfg.tactile_frozen):
            tt = self.tactile_encoder(values, pos, nrm, key_padding_mask=tp)
        tok = self.adapter(tt, contact, key_padding_mask=tp)
        logits = self.contact_head(tt).squeeze(-1) if self.contact_head is not None else None
        return tok, logits

    def encode(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Observation tokens → fused ``memory [B,S,d]`` with ``memory_mask [B,S]`` (True =
        absent / dropped), ``token_types [S]`` (:data:`MODALITIES` ids), the dropout draws and
        the aux ``contact_logits [B,N]`` (when the aux head exists)."""
        proprio = batch["proprio"]
        device = self.readout.device
        proprio = proprio.to(device=device, dtype=self.readout.dtype)
        if proprio.ndim != 2:
            raise ValueError(f"proprio must be [B, k·P], got {tuple(proprio.shape)}")
        B = proprio.shape[0]
        parts: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        out: dict[str, Any] = {}

        if self.text_encoder is not None:
            lt, lm = self._language(batch, B, device)
            drop = self._drop(self.cfg.p_drop_language, B, device)
            if drop is not None:
                lm = lm | drop[:, None]
            out["language_dropped"] = drop
            parts.append((lt, lm, LANG))
        if self.vision_encoder is not None:
            vt, vm = self._vision(batch, B, device)
            drop = self._drop(self.cfg.p_drop_vision, B, device)
            if drop is not None:
                vm = vm | drop[:, None]
            out["vision_dropped"] = drop
            parts.append((vt, vm, VISION))
        if self.tactile_encoder is not None:
            tt, logits = self._tactile(batch, B, device)
            tm = torch.zeros(B, tt.shape[1], dtype=torch.bool, device=device)
            drop = self._drop(self.cfg.p_drop_tactile, B, device)
            if drop is not None:
                tm = tm | drop[:, None]
            out["tactile_dropped"] = drop
            out["contact_logits"] = logits
            parts.append((tt, tm, TACTILE))
        pt = self.proprio_mlp(proprio).unsqueeze(1)
        parts.append((pt, torch.zeros(B, 1, dtype=torch.bool, device=device), PROPRIO))
        rt = self.readout.unsqueeze(0).expand(B, -1, -1)
        parts.append((rt, torch.zeros(B, rt.shape[1], dtype=torch.bool, device=device), READOUT))

        types = torch.cat([torch.full((p[0].shape[1],), p[2], dtype=torch.long, device=device)
                           for p in parts])
        x = torch.cat([p[0].to(rt.dtype) for p in parts], 1) + self.type_emb(types).unsqueeze(0)
        mask = torch.cat([p[1] for p in parts], 1)
        out["memory"] = self.fusion(x, src_key_padding_mask=mask)
        out["memory_mask"] = mask
        out["token_types"] = types
        return out

    # ── training / inference ──────────────────────────────────────────────────────────
    def forward(self, batch: Mapping[str, Any], *, return_encoding: bool = False) -> dict[str, Any]:
        """Training forward: head loss (+ aux contact loss) → ``{"loss", "action_loss", …}``.

        Without ``actions`` in the batch it returns ``{"actions": head.sample(...)}`` instead (the
        flow head then draws noise from the global RNG — use :meth:`predict` for control).
        """
        enc = self.encode(batch)
        mem, mm = enc["memory"], enc["memory_mask"]
        out: dict[str, Any] = {}
        if "actions" in batch:
            actions = batch["actions"].to(mem.device)
            valid = batch.get("action_valid")
            valid = None if valid is None else valid.to(mem.device).bool()
            terms = self.head.loss(mem, mm, actions, valid)
            out.update(terms)
            total = terms["action_loss"]
            logits = enc.get("contact_logits")
            if logits is not None and self.cfg.aux_contact_weight > 0 and "contact_target" in batch:
                cm = batch.get("contact_target_mask")
                cm = torch.ones_like(logits, dtype=torch.bool) if cm is None else cm.to(mem.device).bool()
                tp = batch.get("taxel_pad")
                if tp is not None:
                    cm = cm & ~tp.to(mem.device).bool()
                cl = contact_bce(logits.float(), batch["contact_target"].to(mem.device).float(), cm,
                                 pos_weight=self.cfg.contact_pos_weight)
                out["contact_loss"] = cl
                total = total + self.cfg.aux_contact_weight * cl
            out["loss"] = total
        else:
            out["actions"] = self.head.sample(mem, mm)
        if return_encoding:
            out.update({k: v for k, v in enc.items() if k not in out})
        return out

    @torch.no_grad()
    def predict(self, batch: Mapping[str, Any], n_steps: int | None = None, *,
                generator: torch.Generator | None = None,
                noise: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized action chunk ``[B,H,A]`` in eval mode (no modality dropout). ``n_steps``,
        ``generator`` / ``noise`` apply to the flow head (Euler steps / initial noise)."""
        was = self.training
        self.eval()
        try:
            enc = self.encode(batch)
            return self.head.sample(enc["memory"], enc["memory_mask"], n_steps=n_steps,
                                    generator=generator, noise=noise)
        finally:
            self.train(was)

    def per_sample_action_error(self, batch: Mapping[str, Any], actions: torch.Tensor,
                                valid: torch.Tensor | None = None, **kw: Any) -> torch.Tensor:
        """``[B]`` per-sample head error for ``actions [B,H,A]`` (chunk: squared error of the
        regression; flow: flow-matching error with the given ``noise``/``tau``) — the likelihood
        surrogate used by :mod:`robot_skin.vtla.dpo`."""
        enc = self.encode(batch)
        mem = enc["memory"]
        v = None if valid is None else valid.to(mem.device).bool()
        return self.head.per_sample_error(mem, enc["memory_mask"], actions.to(mem.device), v, **kw)


# ─────────────────────────────────────────────────────────────── policy bundle

POLICY_BUNDLE_NAME = "policy_bundle.pt"
BUNDLE_FORMAT = "robot_skin/vtla_policy_bundle"
BUNDLE_VERSION = 1


def _plain(obj: Any) -> Any:
    """Recursively convert to ``torch.load(weights_only=True)``-safe containers (dict/list/str/
    int/float/bool/None; numpy → lists; non-finite floats → str; tensors kept)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_policy_bundle(path: str | Path, policy: VTLAPolicy, *, action: Mapping[str, Any],
                       proprio: Mapping[str, Any], tactile: Mapping[str, Any],
                       vision: Mapping[str, Any], language: Mapping[str, Any],
                       timing: Mapping[str, Any], meta: Mapping[str, Any] | None = None,
                       state_dict: Mapping[str, torch.Tensor] | None = None) -> Path:
    """Atomically write ``policy_bundle.pt`` (``path`` may be a directory).

    Sections (all plain containers):
    - ``model_config`` (:class:`VTLAConfig`), ``state_dict`` (CPU tensors; default: the policy's
      current weights — pass the EMA / best weights explicitly if they are not loaded);
    - ``action``: ``spec`` (ActionSpec dict), ``normalizer`` (ActionNormalizer dict), ``rel_mode``,
      ``chunk_offset``; ``proprio``: ``normalizer``, ``history``, ``source``;
    - ``tactile``: ``feature_spec``, ``contact_rule``, ``source`` (derived | bootstrap), references
      to the stage-1 ``calibrator`` / ``baseline_model`` and the ``pretrained_encoder``;
    - ``vision``: ``cameras``, ``encoder`` cfg, ``eval_transform`` params, ``cached_features_key``;
      ``language``: ``encoder`` cfg; ``timing``: ``policy_hz``, ``source_hz``, ``stride``,
      ``horizon``, ``obs_history``; ``head``; ``meta``.
    """
    from ..train.checkpoint import save_checkpoint  # atomic tmp + fsync + rename

    p = Path(path)
    if p.suffix != ".pt":
        p = p / POLICY_BUNDLE_NAME
    sd = policy.state_dict() if state_dict is None else state_dict
    sd = {k: v.detach().cpu() for k, v in sd.items()}
    cfg = policy.cfg
    return save_checkpoint(
        p, format=BUNDLE_FORMAT, version=BUNDLE_VERSION, model_config=_plain(cfg.to_dict()),
        state_dict=sd, action=_plain(action), proprio=_plain(proprio), tactile=_plain(tactile),
        vision=_plain(vision), language=_plain(language), timing=_plain(timing),
        head=_plain({"kind": cfg.head, "flow_steps": cfg.flow_steps}),
        meta=_plain(dict(meta or {})))


def read_policy_bundle(path: str | Path, map_location: Any = "cpu") -> dict:
    """Load and validate ``policy_bundle.pt`` (``path`` may be the run directory)."""
    p = Path(path)
    if p.is_dir():
        p = p / POLICY_BUNDLE_NAME
    if not p.is_file():
        raise FileNotFoundError(f"no policy bundle at {p}")
    b = torch.load(p, map_location=map_location, weights_only=True)
    _check_bundle(b, p)
    return b


def _check_bundle(b: Any, where: Any = "bundle") -> None:
    if not isinstance(b, Mapping) or b.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{where} is not a {BUNDLE_FORMAT} (format "
                         f"{b.get('format') if isinstance(b, Mapping) else type(b)!r})")
    if int(b.get("version", 0)) > BUNDLE_VERSION:
        raise ValueError(f"{where}: bundle v{b['version']} newer than supported v{BUNDLE_VERSION}")
    missing = [k for k in ("model_config", "state_dict", "action", "proprio", "tactile", "vision",
                           "timing") if k not in b]
    if missing:
        raise ValueError(f"{where}: bundle lacks {missing}")


def _offline_encoder_cfg(cfg: Mapping[str, Any] | None) -> dict | None:
    """Encoder cfg for rebuilding from a bundle: never download pretrained weights (the bundle's
    state_dict holds them)."""
    if cfg is None:
        return None
    c = dict(cfg)
    if "pretrained" in c:
        c["pretrained"] = False
    return c


def build_policy_from_bundle(bundle: Mapping[str, Any] | str | Path, *, map_location: Any = "cpu",
                             device: torch.device | str | None = None,
                             strict: bool = True) -> VTLAPolicy:
    """Rebuild the trained :class:`VTLAPolicy` (eval mode) from a bundle dict or path.

    The architecture comes from ``model_config`` (encoders are built with ``pretrained: false`` —
    their weights are in ``state_dict``), the weights from ``state_dict``. Normalizers, transforms
    and timing are read with :func:`bundle_components`.
    """
    b = read_policy_bundle(bundle, map_location) if isinstance(bundle, (str, Path)) else bundle
    _check_bundle(b)
    cfg = dict(b["model_config"])
    cfg["vision"] = _offline_encoder_cfg(cfg.get("vision"))
    cfg["text"] = _offline_encoder_cfg(cfg.get("text"))
    policy = VTLAPolicy(VTLAConfig.from_dict(cfg))
    sd = {k: torch.as_tensor(v) for k, v in b["state_dict"].items()}
    policy.load_state_dict(sd, strict=strict)
    if device is not None:
        policy = policy.to(device)
    return policy.eval()


def bundle_components(bundle: Mapping[str, Any] | str | Path, *, device: torch.device | str | None = None
                      ) -> dict[str, Any]:
    """Everything an online controller needs, rebuilt from a bundle:

    ``policy`` (eval, on ``device``), ``action_spec``, ``action_normalizer``, ``rel_mode``,
    ``chunk_offset``, ``proprio_normalizer``, ``feature_spec`` (:class:`TactileFeatureSpec` —
    feed a :class:`~robot_skin.representation.TactileHistory`), ``contact_rule``,
    ``eval_transform`` (:class:`~robot_skin.vision.EvalTransform` or None), ``tokenizer``
    (picklable text tokenizer or None), ``cameras``, ``policy_hz``, ``source_hz``, ``stride``,
    ``horizon``, ``obs_history``, ``head``, ``flow_steps``, plus the raw ``tactile`` / ``vision`` /
    ``meta`` sections.
    """
    from ..action.space import ActionNormalizer, ActionSpec
    from .dataset import eval_transform_from_dict

    b = read_policy_bundle(bundle) if isinstance(bundle, (str, Path)) else bundle
    _check_bundle(b)
    policy = build_policy_from_bundle(b, device=device)
    act, pro, tac, vis, tim = b["action"], b["proprio"], b["tactile"], b["vision"], b["timing"]
    return {
        "policy": policy,
        "action_spec": ActionSpec.from_dict(act["spec"]),
        "action_normalizer": ActionNormalizer.from_dict(act["normalizer"]),
        "rel_mode": act.get("rel_mode", "abs"),
        "chunk_offset": int(act.get("chunk_offset", 1)),
        "proprio_normalizer": (None if pro.get("normalizer") is None
                               else ActionNormalizer.from_dict(pro["normalizer"])),
        "feature_spec": TactileFeatureSpec.from_dict(tac["feature_spec"]),
        "contact_rule": tac.get("contact_rule", "level_ge_weak"),
        "eval_transform": eval_transform_from_dict(vis.get("eval_transform")),
        "tokenizer": policy.text_encoder.get_tokenizer() if policy.text_encoder is not None else None,
        "cameras": tuple(vis.get("cameras") or ()),
        "policy_hz": float(tim["policy_hz"]),
        "source_hz": float(tim.get("source_hz", 200.0)),
        "stride": int(tim["stride"]),
        "horizon": int(tim["horizon"]),
        "obs_history": int(tim.get("obs_history", 1)),
        "head": policy.cfg.head,
        "flow_steps": policy.cfg.flow_steps,
        "tactile": tac, "vision": vis, "meta": b.get("meta", {}),
    }

