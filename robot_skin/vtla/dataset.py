"""VTLA training samples from processed D2 task episodes (policy-rate observation → action chunk).

One sample = one **policy tick** ``t`` (a master-clock index; ticks every
``stride = round(hz / policy_hz)`` frames, :func:`robot_skin.action.chunking.policy_stride`)
inside the selected phases:

- ``actions [H,A]`` — the ACT chunk ``a[t + (offset + i)·stride]``, ``i = 0 … H−1``
  (``offset = 1``: ``actions[0]`` is the state one policy tick after the observation), expressed
  relative to the current state (``action.rel_mode``, :func:`robot_skin.action.make_relative`) and
  normalized (:class:`~robot_skin.action.ActionNormalizer`); ``action_valid [H]`` is False past the
  episode end and where the source label is invalid (``hand_pose_valid``);
- ``proprio [k·A]`` — the current action-space state (hand action / robot q) at the last ``k =
  obs_history`` policy ticks (oldest first, causal edge padding), normalized;
- ``tactile_values [N,F]`` — :meth:`TactileFeatureSpec.from_arrays` of the stage-1 outputs
  (derived ``residual_z`` / ``contact_level`` + ``saturated``) — the *same* function
  (:func:`~robot_skin.representation.tactile_value_features`) the pretrainer and online control
  use; ``taxel_pos`` / ``taxel_nrm [N,3]`` (hand / robot base frame); ``contact [N]`` for the
  :class:`~robot_skin.vtla.adapter.ContactGate` (:func:`contact_from_level`);
- ``images {cam: [(k,)3,h,w]}`` through the image transform (train augmentation or eval), or cached
  frozen-encoder features ``vision_feats {cam: [(k,)P,D]}`` (:mod:`robot_skin.vision.feature_cache`);
  ``vision_valid {cam: [k]}`` is False before a camera's first frame;
- ``instruction`` (``meta.task.instruction``), ``task_id``, ``episode`` / ``t_index`` ids and the
  aux-contact targets ``contact_target`` / ``contact_target_mask [N]``.

:func:`make_observation` packs the observation part; online control builds its single-sample
observation with the same function so training and deployment inputs cannot drift apart.

Stage-1 fallback (**bootstrap only**): when an episode lacks the derived ``residual_z`` /
``contact_level`` (the contact stage has not run yet), ``tactile_source="auto"`` computes them on
the fly with :func:`bootstrap_tactile_arrays` and warns. That is a *static* per-taxel baseline —
residual = ΔS − median ΔS over the episode's ``no_contact`` frames, σ = 1.4826·MAD there (floored),
z = press intensity / σ, levels from z thresholds with %-floors — with **no motion-artefact model**,
so hand motion alone can raise false WEAK levels. Use it for tests and pipeline bring-up only; real
training needs ``tactile_source="derived"``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from common.signal import press_intensity

from ..action.chunking import action_chunk, action_chunks, policy_stride, policy_tick_indices
from ..action.space import REL_MODES, ActionNormalizer, ActionSpec, actions_from_episode, make_relative
from ..contact.ordinal import ContactLevel
from ..datasets.episode import (D_LEVEL, D_RESIDUAL_Z, K_CONTACT_LABEL, K_DELTA, K_PHASE,
                                K_SATURATED, K_TAXEL_NRM, K_TAXEL_POS, Episode, cam_idx_key)
from ..representation.encoder import TactileFeatureSpec

__all__ = [
    "TASK_PHASES", "CONTACT_RULES", "AUX_TARGETS", "TACTILE_SOURCES", "BOOTSTRAP_DEFAULTS",
    "PSEUDO_LABEL_KEY",
    "bootstrap_tactile_arrays", "episode_tactile", "contact_from_level", "sample_phase_mask",
    "history_ticks", "eval_transform_to_dict", "eval_transform_from_dict", "make_observation",
    "VTLADataset", "VTLACollator", "collate_vtla",
]

#: D2 task phases (= ``robot_skin.acquisition.protocol.TASK_PHASES``; a test keeps them equal)
TASK_PHASES = ("reach", "grasp", "manipulate", "release", "retreat")
#: ``level_ge_weak``: WEAK / STRONG / SATURATED count as contact; ``weak_or_strong``: saturated
#: taxels (rail / dropout, untrustworthy) excluded
CONTACT_RULES = ("level_ge_weak", "weak_or_strong")
#: aux contact targets: ``label`` = the contact stage's D2 pseudo labels (derived
#: ``contact_label_pseudo``) if present, else the episode ``contact_label`` (−1 → ignored);
#: ``level`` = the tactile contact rule (saturated ignored); ``gt`` = synthetic ``gt_contact``
#: (else ``label``)
AUX_TARGETS = ("label", "level", "gt")
TACTILE_SOURCES = ("auto", "derived", "bootstrap")
BOOTSTRAP_DEFAULTS: dict[str, float] = {
    "weak_z": 3.0, "strong_z": 8.0,         # z thresholds (press-positive)
    "weak_pct": 3.0, "strong_pct": 15.0,    # % press floors (= default OrdinalQuantizer levels)
    "min_sigma_pct": 1.0,                   # σ floor (a static reference underestimates motion)
    "min_ref_frames": 20,                   # fewer no_contact frames → first fallback_s seconds
    "fallback_s": 0.5,
}
_GT_CONTACT = "gt_contact"                   # optional synthetic ground truth (datasets.build GT_KEYS)
#: = ``robot_skin.contact.pseudo_label.D_CONTACT_LABEL_PSEUDO`` (derived, written by the contact stage)
PSEUDO_LABEL_KEY = "contact_label_pseudo"


# ─────────────────────────────────────────────────────────────── tactile arrays

def bootstrap_tactile_arrays(ep: Episode, **kw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """**Bootstrap-only** stand-in for the contact stage: ``(z[T,N] float32, level[T,N] int8,
    saturated[T,N] bool)`` from ``delta_pct`` with a static per-taxel reference.

    Reference frames per taxel: ``contact_label == 0`` and not saturated (else the first
    ``fallback_s`` seconds). ``r = ΔS − median_ref(ΔS)``; ``σ = max(1.4826·MAD_ref(r),
    min_sigma_pct)``; ``z = press_intensity(r) / σ`` (press-positive); level WEAK where
    ``z ≥ weak_z`` and press ≥ ``weak_pct``, STRONG where ``z ≥ strong_z`` and press ≥
    ``strong_pct``, SATURATED where saturated. Keys of :data:`BOOTSTRAP_DEFAULTS` override.
    """
    unknown = sorted(set(kw) - set(BOOTSTRAP_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown bootstrap options {unknown} (known: {sorted(BOOTSTRAP_DEFAULTS)})")
    c = {**BOOTSTRAP_DEFAULTS, **kw}
    if not ep.has(K_DELTA):
        raise KeyError(f"episode {ep.meta.episode_id!r} has no {K_DELTA!r} to bootstrap tactile levels")
    d = np.asarray(ep[K_DELTA], dtype=np.float32)
    T, N = d.shape
    finite = np.isfinite(d)
    sat = np.asarray(ep[K_SATURATED], dtype=bool) if ep.has(K_SATURATED) else np.zeros_like(finite)
    sat = sat | ~finite
    ref = (np.asarray(ep[K_CONTACT_LABEL]) == 0) if ep.has(K_CONTACT_LABEL) else np.zeros((T, N), bool)
    ref = ref & ~sat
    n_fb = min(T, max(int(c["min_ref_frames"]), int(round(float(c["fallback_s"]) * float(ep.meta.hz)))))
    z = np.zeros((T, N), np.float32)
    press = np.zeros((T, N), np.float32)
    for n in range(N):
        rows = ref[:, n]
        if rows.sum() < int(c["min_ref_frames"]):
            rows = np.zeros(T, bool)
            rows[:n_fb] = True
            rows &= ~sat[:, n]
        x = d[rows, n]
        off = float(np.median(x)) if x.size else 0.0
        r = np.where(finite[:, n], d[:, n] - off, 0.0)
        mad = float(np.median(np.abs(r[rows] - np.median(r[rows])))) if x.size else 0.0
        sigma = max(1.4826 * mad, float(c["min_sigma_pct"]))
        press[:, n] = press_intensity(r)
        z[:, n] = press[:, n] / sigma
    lv = np.full((T, N), int(ContactLevel.NONE), np.int8)
    lv[(z >= c["weak_z"]) & (press >= c["weak_pct"])] = int(ContactLevel.WEAK)
    lv[(z >= c["strong_z"]) & (press >= c["strong_pct"])] = int(ContactLevel.STRONG)
    lv[sat] = int(ContactLevel.SATURATED)
    z[~finite] = 0.0
    return z.astype(np.float32), lv, sat


def episode_tactile(ep: Episode, source: str = "auto", bootstrap: Mapping[str, float] | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    """``(residual_z, level, saturated | None, used)`` of an episode; ``used`` = ``derived`` (the
    contact stage's ``residual_z``/``contact_level``) or ``bootstrap``
    (:func:`bootstrap_tactile_arrays`). ``source``: ``auto`` (derived if present) | ``derived``
    (raise if missing) | ``bootstrap``."""
    if source not in TACTILE_SOURCES:
        raise ValueError(f"tactile source must be one of {TACTILE_SOURCES}, got {source!r}")
    has = ep.has_derived(D_RESIDUAL_Z) and ep.has_derived(D_LEVEL)
    if source == "derived" and not has:
        raise KeyError(f"episode {ep.meta.episode_id!r} lacks derived {D_RESIDUAL_Z!r}/{D_LEVEL!r}: "
                       "run the contact stage (or use tactile_source 'auto'/'bootstrap' for tests)")
    if has and source != "bootstrap":
        sat = np.asarray(ep[K_SATURATED], dtype=bool) if ep.has(K_SATURATED) else None
        return ep.derived(D_RESIDUAL_Z), ep.derived(D_LEVEL), sat, "derived"
    z, lv, sat = bootstrap_tactile_arrays(ep, **dict(bootstrap or {}))
    return z, lv, sat, "bootstrap"


def contact_from_level(level: Any, saturated: Any = None, rule: str = "level_ge_weak") -> np.ndarray:
    """Per-taxel contact flag for the :class:`~robot_skin.vtla.adapter.ContactGate` (bool, same
    shape as ``level``). ``level_ge_weak``: ``level ≥ WEAK`` (incl. SATURATED — the rule of the
    spec: a rail-clipped press is still contact); ``weak_or_strong``: WEAK / STRONG only
    (saturated or ``saturated=True`` taxels excluded, like the ``binary`` features)."""
    if rule not in CONTACT_RULES:
        raise ValueError(f"contact rule must be one of {CONTACT_RULES}, got {rule!r}")
    lv = np.asarray(level).astype(np.int64)
    if rule == "level_ge_weak":
        c = lv >= int(ContactLevel.WEAK)
        if saturated is not None:
            c = c | np.asarray(saturated, dtype=bool)
        return c
    c = (lv == int(ContactLevel.WEAK)) | (lv == int(ContactLevel.STRONG))
    if saturated is not None:
        c = c & ~np.asarray(saturated, dtype=bool)
    return c


# ─────────────────────────────────────────────────────────────── ticks & phases

def sample_phase_mask(ep: Episode, phases: str | Sequence[str] | None = "task") -> np.ndarray:
    """``[T]`` frames eligible as observation ticks.

    ``"task"`` (default): the D2 task phases :data:`TASK_PHASES` present in the episode (if none is
    present: every labelled frame, with a warning); ``"all"`` / ``None``: frames inside any phase
    (``phase_id ≥ 0``; all frames without phase labels); a list: frames inside those phases —
    calibration / sync / baseline blocks are thereby never imitated.
    """
    T = ep.T
    pid = np.asarray(ep[K_PHASE]) if ep.has(K_PHASE) else None
    labelled = np.ones(T, bool) if pid is None or not (pid >= 0).any() else pid >= 0
    if phases is None or phases == "all":
        return labelled
    names = list(TASK_PHASES) if phases == "task" else ([phases] if isinstance(phases, str) else list(phases))
    present = [n for n in names if n in ep.meta.phase_names]
    if not present:
        if phases == "task":
            warnings.warn(f"episode {ep.meta.episode_id!r} has none of the task phases {TASK_PHASES}; "
                          "sampling every labelled frame", stacklevel=2)
            return labelled
        return np.zeros(T, bool)
    m = np.zeros(T, bool)
    for n in present:
        m |= ep.phase_mask(n)
    return m


def history_ticks(t: int, k: int, stride: int) -> np.ndarray:
    """``[k]`` master indices of the last ``k`` policy ticks ending at ``t`` (oldest first,
    clipped at 0 — causal edge padding, the convention of
    :func:`robot_skin.representation.history_indices`)."""
    if k < 1 or stride < 1:
        raise ValueError("k and stride must be >= 1")
    return np.maximum(int(t) - (np.arange(k, dtype=np.int64)[::-1]) * int(stride), 0)


# ─────────────────────────────────────────────────────────────── image transform params

def eval_transform_to_dict(tf: Any) -> dict | None:
    """Parameters of a deterministic :class:`~robot_skin.vision.EvalTransform` (for the bundle);
    None for None. Train augmentations are converted with their ``eval_transform()`` first."""
    if tf is None:
        return None
    if not hasattr(tf, "crop_scale") and hasattr(tf, "eval_transform"):
        tf = tf.eval_transform()
    norm = getattr(tf, "normalize", None)
    return {
        "out_size": None if tf.out_size is None else [int(v) for v in tf.out_size],
        "crop_scale": float(tf.crop_scale), "stretch": bool(tf.stretch),
        "antialias": bool(tf.antialias),
        "mean": None if norm is None else [float(v) for v in norm.mean.flatten().tolist()],
        "std": None if norm is None else [float(v) for v in norm.std.flatten().tolist()],
    }


def eval_transform_from_dict(d: Mapping[str, Any] | None) -> Any:
    """Inverse of :func:`eval_transform_to_dict` → :class:`~robot_skin.vision.EvalTransform`."""
    if d is None:
        return None
    from ..vision.transforms import EvalTransform
    out = d.get("out_size")
    return EvalTransform(None if out is None else tuple(int(v) for v in out),
                         crop_scale=float(d.get("crop_scale", 1.0)), stretch=bool(d.get("stretch", False)),
                         mean=d.get("mean"), std=d.get("std"), antialias=bool(d.get("antialias", True)))


# ─────────────────────────────────────────────────────────────── observation

def make_observation(*, proprio_states: Any, tactile_values: Any, taxel_pos: Any, taxel_nrm: Any,
                     contact: Any, instruction: str = "",
                     proprio_normalizer: ActionNormalizer | None = None,
                     images: Mapping[str, Any] | None = None,
                     vision_feats: Mapping[str, Any] | None = None,
                     vision_valid: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One observation as the model consumes it (unbatched torch tensors; batch with
    :func:`collate_vtla`). Shared by :class:`VTLADataset` and online control.

    Args:
        proprio_states: ``[k, A]`` absolute action-space states of the last ``k`` policy ticks
            (oldest first) — normalized with ``proprio_normalizer`` and flattened to ``[k·A]``.
        tactile_values: ``[N, F]`` (:meth:`TactileFeatureSpec.from_arrays` /
            :class:`~robot_skin.representation.TactileHistory`); ``taxel_pos``/``taxel_nrm``
            ``[N,3]`` (non-finite poses are zeroed and the taxel flagged in ``taxel_pad``);
            ``contact`` ``[N]`` bool (:func:`contact_from_level`).
        images: ``{cam: float [(k,)3,h,w]}`` already transformed (eval transform online), or
        vision_feats: ``{cam: [(k,)P,D]}``; ``vision_valid``: ``{cam: bool [k]}`` (default True).
    """
    st = np.asarray(proprio_states, dtype=np.float32)
    if st.ndim == 1:
        st = st[None]
    if proprio_normalizer is not None:
        st = np.asarray(proprio_normalizer.normalize(st), dtype=np.float32)
    pos = np.asarray(taxel_pos, dtype=np.float32)
    nrm = np.asarray(taxel_nrm, dtype=np.float32)
    vals = np.asarray(tactile_values, dtype=np.float32)
    con = np.asarray(contact, dtype=bool)
    if pos.ndim != 2 or pos.shape[-1] != 3 or nrm.shape != pos.shape:
        raise ValueError(f"taxel_pos / taxel_nrm must both be [N,3], got {pos.shape} / {nrm.shape}")
    if vals.ndim != 2 or vals.shape[0] != pos.shape[0] or con.shape != (pos.shape[0],):
        raise ValueError(f"tactile_values [N,F] / contact [N] do not match {pos.shape[0]} taxels "
                         f"(got {vals.shape} / {con.shape})")
    bad = ~(np.isfinite(pos).all(-1) & np.isfinite(nrm).all(-1))
    obs: dict[str, Any] = {
        "proprio": torch.from_numpy(np.ascontiguousarray(st.reshape(-1))),
        "tactile_values": torch.as_tensor(vals),
        "taxel_pos": torch.from_numpy(np.where(bad[:, None], 0.0, pos).astype(np.float32)),
        "taxel_nrm": torch.from_numpy(np.where(bad[:, None], 0.0, nrm).astype(np.float32)),
        "contact": torch.from_numpy(con & ~bad),
        "taxel_pad": torch.from_numpy(bad),
        "instruction": str(instruction or ""),
    }
    k = st.shape[0]
    if images is not None:
        obs["images"] = {c: torch.as_tensor(v, dtype=torch.float32) for c, v in images.items()}
    if vision_feats is not None:
        obs["vision_feats"] = {c: torch.as_tensor(np.asarray(v, dtype=np.float32))
                               for c, v in vision_feats.items()}
    cams = list((images or {}).keys()) + list((vision_feats or {}).keys())
    if cams:
        vv = dict(vision_valid or {})
        obs["vision_valid"] = {c: torch.as_tensor(np.asarray(vv.get(c, np.ones(k, bool)), dtype=bool)
                                                  .reshape(-1)) for c in cams}
    return obs


# ─────────────────────────────────────────────────────────────── dataset

class VTLADataset(Dataset):
    """Policy-tick samples of processed (D2) episodes — see the module docstring.

    Args:
        episodes: :class:`Episode` objects or episode directories.
        cameras: camera names (``camera_<name>`` in the episode); ``()`` → no vision inputs.
        policy_hz: policy rate; ``stride = round(meta.hz / policy_hz)`` master frames per tick.
        horizon: chunk length ``H``; ``chunk_offset``: ``actions[0] = a[t + offset·stride]``.
        obs_history: ``k`` policy ticks of proprio / camera frames.
        action_spec: :class:`ActionSpec` | ``"hand_mano"`` | ``"robot_joint"`` | dict.
        rel_mode: :data:`robot_skin.action.REL_MODES` (relative to the state at ``t``).
        feature_spec: :class:`TactileFeatureSpec` (or dict) of the tactile values
            (``obs_mode: none`` → ``tactile_values [N,0]``, no tactile arrays needed).
        image_transform: callable on uint8 ``[...,H,W,3]`` (train: ``TrainAugment``, eval:
            ``EvalTransform``); None → ``EvalTransform()`` (ImageNet normalization, native size).
        use_cached_vision: feature-cache key (``vision.feature_cache``) → ``vision_feats``.
        action_normalizer / proprio_normalizer: fitted normalizers (None → raw units; fit with
            :meth:`fit_normalizers` on the training split).
        phases: :func:`sample_phase_mask` selector. ``sample_stride``: tick spacing of the samples
            (default: the policy stride; 1 = every master frame, a 10× denser augmentation).
        min_valid_steps: minimum valid chunk steps for a tick to be sampled.
        require_valid_state: skip ticks whose current state is invalid (``hand_pose_valid``).
        contact_rule: :data:`CONTACT_RULES`; aux_target: :data:`AUX_TARGETS`.
        tactile_source: :data:`TACTILE_SOURCES`; bootstrap: :data:`BOOTSTRAP_DEFAULTS` overrides.
    """

    def __init__(self, episodes: Sequence[Episode | str | Path], *, cameras: Sequence[str] = ("ego",),
                 policy_hz: float = 20.0, horizon: int = 16, obs_history: int = 1,
                 chunk_offset: int = 1, action_spec: ActionSpec | str | Mapping = "hand_mano",
                 rel_mode: str = "delta",
                 feature_spec: TactileFeatureSpec | Mapping[str, Any] | None = None,
                 image_transform: Callable | None = None, use_cached_vision: str | None = None,
                 action_normalizer: ActionNormalizer | None = None,
                 proprio_normalizer: ActionNormalizer | None = None,
                 phases: str | Sequence[str] | None = "task", sample_stride: int | None = None,
                 min_valid_steps: int = 1, require_valid_state: bool = True,
                 contact_rule: str = "level_ge_weak", aux_target: str = "label",
                 tactile_source: str = "auto", bootstrap: Mapping[str, float] | None = None) -> None:
        if horizon < 1 or obs_history < 1 or chunk_offset < 0 or min_valid_steps < 0:
            raise ValueError("need horizon >= 1, obs_history >= 1, chunk_offset >= 0, min_valid_steps >= 0")
        if rel_mode not in REL_MODES:
            raise ValueError(f"rel_mode must be one of {REL_MODES}, got {rel_mode!r}")
        if contact_rule not in CONTACT_RULES:
            raise ValueError(f"contact_rule must be one of {CONTACT_RULES}, got {contact_rule!r}")
        if aux_target not in AUX_TARGETS:
            raise ValueError(f"aux_target must be one of {AUX_TARGETS}, got {aux_target!r}")
        if tactile_source not in TACTILE_SOURCES:
            raise ValueError(f"tactile_source must be one of {TACTILE_SOURCES}, got {tactile_source!r}")
        self.episodes = [e if isinstance(e, Episode) else Episode.load(e, mmap=True) for e in episodes]
        if not self.episodes:
            raise ValueError("VTLADataset needs at least one episode")
        self.cameras = tuple(str(c) for c in cameras)
        self.policy_hz, self.horizon, self.obs_history = float(policy_hz), int(horizon), int(obs_history)
        self.chunk_offset, self.rel_mode = int(chunk_offset), rel_mode
        self.feature_spec = (feature_spec if isinstance(feature_spec, TactileFeatureSpec)
                             else TactileFeatureSpec.from_dict(feature_spec))
        if image_transform is None and not use_cached_vision and self.cameras:
            from ..vision.transforms import EvalTransform
            image_transform = EvalTransform()
        self.image_transform = image_transform
        self.use_cached_vision = use_cached_vision or None
        self.action_normalizer, self.proprio_normalizer = action_normalizer, proprio_normalizer
        self.contact_rule, self.aux_target = contact_rule, aux_target
        self.phases, self.min_valid_steps = phases, int(min_valid_steps)

        hz = {float(ep.meta.hz) for ep in self.episodes}
        if len(hz) != 1:
            raise ValueError(f"episodes have different master rates {sorted(hz)}; one policy stride "
                             "per dataset is required")
        self.source_hz = hz.pop()
        self.stride = policy_stride(self.source_hz, self.policy_hz)
        every = int(sample_stride) if sample_stride else self.stride
        if every < 1:
            raise ValueError("sample_stride must be >= 1")
        self.sample_stride = every

        # actions (+ spec from the first episode for "robot_joint")
        self._actions: list[tuple[np.ndarray, np.ndarray]] = []
        spec = action_spec
        for ep in self.episodes:
            with warnings.catch_warnings():   # read-only memmaps → torch (copies; harmless)
                warnings.filterwarnings("ignore", message=".*NumPy array is not writable.*")
                a, v = actions_from_episode(ep, spec)
            if not isinstance(spec, ActionSpec):
                spec = (ActionSpec.robot_joint(list(ep.meta.joint_names) or a.shape[1])
                        if spec == "robot_joint" else
                        (ActionSpec.from_dict(spec) if isinstance(spec, Mapping) else ActionSpec.hand_mano()))
            if a.shape[1] != spec.dim:
                raise ValueError(f"episode {ep.meta.episode_id!r}: {a.shape[1]}-D actions, "
                                 f"spec is {spec.dim}-D")
            generic = tuple(f"a{i}" for i in range(spec.dim))
            if (spec.kind == "robot_joint" and ep.meta.joint_names and spec.names != generic
                    and tuple(ep.meta.joint_names) != spec.names):
                raise ValueError(f"episode {ep.meta.episode_id!r}: joint order {ep.meta.joint_names} "
                                 f"differs from the action spec {list(spec.names)}")
            self._actions.append((np.asarray(a, np.float32), np.asarray(v, bool)))
        self.action_spec: ActionSpec = spec
        if spec.kind == "robot_joint" and rel_mode == "delta_pose":
            raise ValueError("rel_mode 'delta_pose' is hand_mano only; robot_joint supports abs | delta")
        if action_normalizer is not None and action_normalizer.dim != spec.dim:
            raise ValueError("action_normalizer dim does not match the action spec")

        # cameras
        for ep in self.episodes:
            missing = [c for c in self.cameras if not ep.has(cam_idx_key(c))]
            if missing:
                raise ValueError(f"episode {ep.meta.episode_id!r} has no camera(s) {missing} "
                                 f"(episode cameras: {ep.meta.cameras})")
        self._frames: dict[tuple[int, str], Any] = {}

        # tactile arrays
        self.tactile_sources: dict[str, str] = {}
        self._tactile: list[tuple[Any, Any, Any] | None] = []
        for ep in self.episodes:
            if self.feature_spec.dim == 0:
                self._tactile.append(None)
                self.tactile_sources[ep.meta.episode_id] = "none"
                continue
            z, lv, sat, used = episode_tactile(ep, tactile_source, bootstrap)
            self._tactile.append((z, lv, sat))
            self.tactile_sources[ep.meta.episode_id] = used
        n_boot = sum(s == "bootstrap" for s in self.tactile_sources.values())
        if n_boot and tactile_source == "auto":
            warnings.warn(f"{n_boot}/{len(self.episodes)} episodes lack the contact stage's derived "
                          f"{D_RESIDUAL_Z}/{D_LEVEL}: using the BOOTSTRAP static-baseline tactile "
                          "levels (no motion-artefact model — tests / bring-up only; run the "
                          "contact stage before real training)", stacklevel=2)

        # sample index
        rows = []
        self.n_ticks_dropped = 0
        for e, ep in enumerate(self.episodes):
            a, v = self._actions[e]
            mask = sample_phase_mask(ep, phases)
            if require_valid_state:
                mask = mask & v
            ticks = policy_tick_indices(ep.T, every, mask=mask)
            if ticks.size and self.min_valid_steps > 0:
                _, cv = action_chunks(a, ticks, self.horizon, self.stride, offset=self.chunk_offset, valid=v)
                keep = cv.sum(1) >= self.min_valid_steps
                self.n_ticks_dropped += int((~keep).sum())
                ticks = ticks[keep]
            rows.append(np.stack([np.full(ticks.shape, e, np.int64), ticks], 1))
        self.index = np.concatenate(rows, 0) if rows else np.zeros((0, 2), np.int64)

    # ── sizes ─────────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return int(self.index.shape[0])

    @property
    def action_dim(self) -> int:
        return self.action_spec.dim

    @property
    def proprio_dim(self) -> int:
        """Per-history-step proprio width (= action dim: proprio is the action-space state)."""
        return self.action_spec.dim

    @property
    def n_taxels(self) -> int:
        return max(int(ep.meta.n_taxels) for ep in self.episodes)

    @property
    def tactile_source(self) -> str:
        """``derived`` | ``bootstrap`` | ``mixed`` | ``none`` (obs_mode none)."""
        used = set(self.tactile_sources.values())
        return used.pop() if len(used) == 1 else "mixed"

    def task_id(self, e: int) -> str:
        return str((self.episodes[e].meta.task or {}).get("task_id") or "")

    # ── normalizers ───────────────────────────────────────────────────────────────────
    def raw_targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unnormalized ``(relative chunks [M,H,A], chunk valid [M,H], current states [M,A])`` of
        every sample — what :meth:`fit_normalizers` fits on."""
        chunks, valids, states = [], [], []
        for e in range(len(self.episodes)):
            ticks = self.index[self.index[:, 0] == e, 1]
            if not ticks.size:
                continue
            a, v = self._actions[e]
            c, cv = action_chunks(a, ticks, self.horizon, self.stride, offset=self.chunk_offset, valid=v)
            s = a[ticks]
            chunks.append(np.asarray(make_relative(c, s, self.action_spec, self.rel_mode), np.float32))
            valids.append(cv)
            states.append(s)
        A = self.action_dim
        if not chunks:
            return (np.zeros((0, self.horizon, A), np.float32), np.zeros((0, self.horizon), bool),
                    np.zeros((0, A), np.float32))
        return np.concatenate(chunks), np.concatenate(valids), np.concatenate(states)

    def fit_normalizers(self, method: str = "std", *, min_scale: float = 1e-2,
                        proprio_method: str | None = None) -> tuple[ActionNormalizer, ActionNormalizer]:
        """Fit (and install) the action normalizer on the valid relative chunk steps and the
        proprio normalizer on the current states of all samples. Call on the **training** split
        and pass the results to the val/test datasets."""
        chunks, valid, states = self.raw_targets()
        if chunks.shape[0] == 0:
            raise ValueError("no samples to fit normalizers on")
        an = ActionNormalizer.fit(chunks, valid, spec=self.action_spec, method=method, min_scale=min_scale)
        pn = ActionNormalizer.fit(states, spec=self.action_spec, method=proprio_method or method,
                                  min_scale=min_scale)
        self.action_normalizer, self.proprio_normalizer = an, pn
        return an, pn

    # ── items ─────────────────────────────────────────────────────────────────────────
    def _camera_frames(self, e: int, cam: str) -> Any:
        key = (e, cam)
        if key not in self._frames:
            ep = self.episodes[e]
            if self.use_cached_vision:
                from ..vision.feature_cache import load_cached
                self._frames[key] = load_cached(ep, cam, self.use_cached_vision)
            else:
                npy = None if ep.root is None else ep.root / f"camera_{cam}" / "frames.npy"
                self._frames[key] = (np.load(npy, mmap_mode="r") if npy is not None and npy.is_file()
                                     else None)
        return self._frames[key]

    def _images(self, e: int, idx: np.ndarray, cam: str) -> tuple[torch.Tensor, np.ndarray]:
        ep = self.episodes[e]
        frames = self._camera_frames(e, cam)
        valid = idx >= 0
        if frames is not None:
            shape = frames.shape[1:]
            imgs = np.stack([np.asarray(frames[i]) if i >= 0 else np.zeros(shape, np.uint8) for i in idx])
        else:
            first = next((int(i) for i in idx if i >= 0), None)
            ref = ep.get_frame(cam, 0 if first is None else first)
            imgs = np.stack([ep.get_frame(cam, int(i)) if i >= 0 else np.zeros_like(ref) for i in idx])
        x = self.image_transform(imgs[None])[0]         # one crop shared by the history frames
        return (x[0] if self.obs_history == 1 else x), valid

    def __getitem__(self, i: int) -> dict[str, Any]:
        e, t = (int(v) for v in self.index[int(i)])
        ep = self.episodes[e]
        a, v = self._actions[e]
        chunk, cvalid = action_chunk(a, t, self.horizon, self.stride, offset=self.chunk_offset, valid=v)
        state = a[t]
        rel = np.asarray(make_relative(chunk, state, self.action_spec, self.rel_mode), np.float32)
        act = rel if self.action_normalizer is None else np.asarray(self.action_normalizer.normalize(rel))
        hist = history_ticks(t, self.obs_history, self.stride)

        N = int(ep.meta.n_taxels)
        tac = self._tactile[e]
        if tac is None:
            values = np.zeros((N, 0), np.float32)
            contact = np.zeros(N, bool)
            lv_t = sat_t = None
        else:
            z, lv, sat = tac
            values = self.feature_spec.from_arrays(z, lv, sat, t)
            lv_t = np.asarray(lv[t])
            sat_t = None if sat is None else np.asarray(sat[t], bool)
            contact = contact_from_level(lv_t, sat_t, self.contact_rule)
        pos = ep[K_TAXEL_POS][t] if ep.has(K_TAXEL_POS) else np.full((N, 3), np.nan, np.float32)
        nrm = ep[K_TAXEL_NRM][t] if ep.has(K_TAXEL_NRM) else np.full((N, 3), np.nan, np.float32)

        images = feats = valid = None
        if self.cameras:
            valid = {}
            if self.use_cached_vision:
                from ..vision.feature_cache import gather_frame_features
                feats = {}
                for cam in self.cameras:
                    idx = np.asarray(ep[cam_idx_key(cam)][hist], np.int64)
                    f, ok = gather_frame_features(self._camera_frames(e, cam), idx)
                    feats[cam] = f[0] if self.obs_history == 1 else f
                    valid[cam] = ok
            else:
                images = {}
                for cam in self.cameras:
                    idx = np.asarray(ep[cam_idx_key(cam)][hist], np.int64)
                    images[cam], valid[cam] = self._images(e, idx, cam)

        obs = make_observation(proprio_states=a[hist], tactile_values=values, taxel_pos=pos,
                               taxel_nrm=nrm, contact=contact, instruction=ep.meta.instruction or "",
                               proprio_normalizer=self.proprio_normalizer, images=images,
                               vision_feats=feats, vision_valid=valid)
        tgt, tmask = self._aux_target(ep, t, N, lv_t, sat_t)
        obs.update({
            "actions": torch.from_numpy(np.ascontiguousarray(act, dtype=np.float32)),
            "action_valid": torch.from_numpy(np.asarray(cvalid, bool)),
            "contact_target": torch.from_numpy(tgt), "contact_target_mask": torch.from_numpy(tmask),
            "episode": e, "t_index": t, "task_id": self.task_id(e),
        })
        return obs

    def _aux_target(self, ep: Episode, t: int, N: int, lv_t, sat_t) -> tuple[np.ndarray, np.ndarray]:
        mode = self.aux_target
        if mode == "gt" and ep.has(_GT_CONTACT):
            g = np.asarray(ep[_GT_CONTACT][t], bool)
            return g.astype(np.float32), np.ones(N, bool)
        if mode == "level":
            if lv_t is None:
                return np.zeros(N, np.float32), np.zeros(N, bool)
            c = contact_from_level(lv_t, sat_t, "weak_or_strong")
            ok = np.asarray(lv_t) != int(ContactLevel.SATURATED)
            if sat_t is not None:
                ok &= ~sat_t
            return c.astype(np.float32), ok
        if ep.has_derived(PSEUDO_LABEL_KEY) or ep.has(K_CONTACT_LABEL):
            src = ep.derived(PSEUDO_LABEL_KEY) if ep.has_derived(PSEUDO_LABEL_KEY) else ep[K_CONTACT_LABEL]
            lab = np.asarray(src[t])
            return (lab == 1).astype(np.float32), lab >= 0
        return np.zeros(N, np.float32), np.zeros(N, bool)


# ─────────────────────────────────────────────────────────────── collate

_TAXEL_KEYS = {"tactile_values": 0.0, "taxel_pos": 0.0, "taxel_nrm": 0.0, "contact": False,
               "contact_target": 0.0, "contact_target_mask": False}


def collate_vtla(samples: Sequence[Mapping[str, Any]],
                 tokenizer: Callable[[Sequence[str]], Mapping[str, torch.Tensor]] | None = None
                 ) -> dict[str, Any]:
    """Stack samples: tensors → ``[B, ...]``; camera dicts per camera; strings (``instruction``,
    ``task_id``) → lists; ints → long tensors. The taxel axis is padded to the batch maximum
    (``taxel_pad [B,N]`` True = padding) so glove and robot layouts can share a batch. With a
    ``tokenizer`` (``text_encoder.get_tokenizer()``) the instructions are also tokenized into
    ``input_ids`` / ``text_pad_mask`` (True = padding)."""
    if not samples:
        raise ValueError("empty batch")
    B = len(samples)
    out: dict[str, Any] = {}
    N = max(int(s["taxel_pos"].shape[0]) for s in samples) if "taxel_pos" in samples[0] else 0
    for key in samples[0]:
        vals = [s[key] for s in samples]
        v0 = vals[0]
        if key in _TAXEL_KEYS or key == "taxel_pad":
            continue
        if isinstance(v0, Mapping):
            cams = list(v0.keys())
            if any(list(v.keys()) != cams for v in vals):
                raise ValueError(f"samples disagree on the cameras of {key!r}")
            out[key] = {c: torch.stack([torch.as_tensor(v[c]) for v in vals]) for c in cams}
        elif isinstance(v0, torch.Tensor):
            out[key] = torch.stack(vals)
        elif isinstance(v0, str):
            out[key] = [str(v) for v in vals]
        elif isinstance(v0, (bool, np.bool_)):
            out[key] = torch.tensor([bool(v) for v in vals])
        elif isinstance(v0, (int, np.integer)):
            out[key] = torch.tensor([int(v) for v in vals], dtype=torch.long)
        elif isinstance(v0, (float, np.floating)):
            out[key] = torch.tensor([float(v) for v in vals], dtype=torch.float32)
        else:
            out[key] = vals
    if N:
        pad = torch.ones(B, N, dtype=torch.bool)
        for b, s in enumerate(samples):
            n = int(s["taxel_pos"].shape[0])
            pad[b, :n] = torch.as_tensor(s["taxel_pad"], dtype=torch.bool) if "taxel_pad" in s else False
        out["taxel_pad"] = pad
        for key, fill in _TAXEL_KEYS.items():
            if key not in samples[0]:
                continue
            t0 = torch.as_tensor(samples[0][key])
            buf = torch.full((B, N, *t0.shape[1:]), fill, dtype=t0.dtype)
            for b, s in enumerate(samples):
                x = torch.as_tensor(s[key])
                buf[b, :x.shape[0]] = x
            out[key] = buf
    if tokenizer is not None and "instruction" in out:
        tok = tokenizer(out["instruction"])
        out["input_ids"] = tok["input_ids"]
        out["text_pad_mask"] = tok["pad_mask"].bool()
    return out


class VTLACollator:
    """Picklable ``collate_fn`` (DataLoader workers) holding the text tokenizer (or None)."""

    def __init__(self, tokenizer: Callable[[Sequence[str]], Mapping[str, torch.Tensor]] | None = None):
        self.tokenizer = tokenizer

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return collate_vtla(samples, self.tokenizer)

    def __repr__(self) -> str:
        return f"VTLACollator(tokenizer={self.tokenizer!r})"

