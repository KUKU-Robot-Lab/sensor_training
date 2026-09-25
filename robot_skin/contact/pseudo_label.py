"""Pseudo contact labels for D2 (task) episodes: detector + hysteresis + phase expectation + extras.

Preprocessing labels D2 frames only where it is certain (``contact_label``: 0 in ``no_contact``
segments, 1 where the geometric self-touch fires, −1 elsewhere — object contact is never inferred
there). The contact stage fills the unknown frames by fusing, per (frame, taxel):

1. **existing labels** (kept as they are when ``keep_existing``);
2. the **detector** probability (``contact.detector``), debounced by a :class:`HysteresisFilter`;
3. the **phase expectation** of the protocol (``meta.phases[].contact`` — ``none`` / ``self`` /
   ``object`` / ``any`` from the recorder / synthetic phase events; if absent, inferred from the
   phase name: reach / retreat / baseline / calibration / air_grasp → none; grasp / manipulate / release →
   object; pinch / fist → self):

   - ``none`` phases → 0; a detector firing there is a conflict → −1 (``none_conflict="unknown"``)
     or 0 (``"zero"``);
   - ``object`` / ``self`` / ``any`` phases → 1 where the filter is on, 0 where the probability
     is below ``neg_thr`` and the filter is off, −1 in between;
   - frames outside any phase → −1 (``unknown_phase="unknown"``) or treated as ``any`` / ``none``;
4. **saturation**: a saturated taxel in a contact-allowed phase is labelled 1 (a dropout during a
   grasp is a press; ``saturated_as_contact``), −1 elsewhere;
5. optional **hand–object proximity** veto (``proximity={"max_dist_m": …}``): a "contact" taxel
   farther than ``max_dist_m`` from ``object_pos`` becomes −1 (it cannot be touching the object; a
   genuine self-touch is already covered by step 1). Distances use world taxel positions
   ``R(hand_global_orient) · taxel_pos + hand_wrist_pos`` for gloves (``taxel_pos`` is in the hand
   frame) and the stored frame otherwise (synthetic robot sessions record ``object_pos`` in the hand
   base frame, ``meta.preprocessing.object_frame``). Frames without a valid hand pose are not vetoed.

Output is the int8 ``[T,N]`` label in {−1, 0, 1}; the stage writes it as the derived array
:data:`D_CONTACT_LABEL_PSEUDO` (the episode ``contact_label`` array is preprocessing output and is
never modified).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# derived key written by the contact stage: defined in the Episode contract
# (``datasets.episode.D_CONTACT_LABEL_PSEUDO``) and re-exported here under the same name for existing
# imports (labels of a later stage, so they live under ``derived/`` next to ``contact_prob``)
from ..datasets.episode import D_CONTACT_LABEL_PSEUDO
from .hysteresis import HysteresisFilter

__all__ = ["D_CONTACT_LABEL_PSEUDO", "EXPECTATIONS", "phase_expectation", "frame_expectation",
           "taxel_world_positions", "pseudo_label_episode", "pseudo_label_metrics"]

EXPECTATIONS = ("none", "self", "object", "any")
_NONE_NAMES = ("reach", "retreat", "baseline", "rest", "imu_calibration", "calibration", "open_close",
               "wrist_rotation", "free_motion", "finger_flex", "air_grasp", "idle")
_OBJECT_NAMES = ("grasp", "manipulate", "release", "task", "lift", "place", "pour", "wipe", "insert")
_SELF_NAMES = ("pinch", "fist", "self_touch", "finger_cross", "palm_touch")
_ALLOWED = ("self", "object", "any")


def phase_expectation(phase: Mapping[str, Any] | str) -> str | None:
    """Contact expectation of one phase: its ``contact`` field (recorder / synthetic phase events)
    if valid, else inferred from the name (module docstring); None when unknown. ``sync*`` phases
    (index-fingertip taps) → ``any``."""
    if isinstance(phase, Mapping):
        c = phase.get("contact")
        if c in EXPECTATIONS:
            return str(c)
        labels = phase.get("labels") or ()
        if "no_contact" in labels:
            return "none"
        if "self_touch" in labels:
            return "self"
        name = str(phase.get("name", ""))
    else:
        name = str(phase)
    n = name.lower()
    if n.startswith("sync"):
        return "any"
    for prefixes, exp in ((_SELF_NAMES, "self"), (_OBJECT_NAMES, "object"), (_NONE_NAMES, "none")):
        if any(n.startswith(p) for p in prefixes):
            return exp
    return None


def frame_expectation(episode, *, overrides: Mapping[str, str] | None = None) -> np.ndarray:
    """Per-frame expectation codes ``[T]`` (str objects: none|self|object|any, or "" outside any
    phase / unknown) from ``phase_id`` + ``meta.phases``; ``overrides`` maps phase names to an
    expectation."""
    from ..datasets.episode import K_PHASE

    T = episode.T
    out = np.full(T, "", dtype=object)
    if not episode.has(K_PHASE):
        return out
    names = list(episode.meta.phase_names)
    by_name: dict[str, str | None] = {}
    for p in episode.meta.phases or []:
        by_name.setdefault(str(p.get("name")), phase_expectation(p))
    ov = dict(overrides or {})
    for k, v in ov.items():
        if v not in EXPECTATIONS:
            raise ValueError(f"phase override {k!r}: expectation must be one of {EXPECTATIONS}, got {v!r}")
    pid = np.asarray(episode[K_PHASE])
    for i, nm in enumerate(names):
        exp = ov.get(nm, by_name.get(nm) if nm in by_name else phase_expectation(nm))
        if exp:
            out[pid == i] = exp
    return out


def taxel_world_positions(episode) -> tuple[np.ndarray, np.ndarray]:
    """``(pos[T,N,3], valid[T])`` taxel positions in the frame of ``object_pos``.

    Glove episodes (``taxel_frame`` ``mano_wrist``): ``R(hand_global_orient) · taxel_pos +
    hand_wrist_pos``, valid where ``hand_pose_valid``; without a hand pose (camera-free session)
    the hand is nowhere in the object frame → no frame is valid. Otherwise the stored ``taxel_pos``
    (robot: URDF root = hand base, the frame of synthetic robot object poses), all frames valid."""
    from ..datasets.episode import K_HAND_GLOBAL, K_HAND_VALID, K_HAND_WRIST, K_TAXEL_POS
    from ..geometry.rotations import aa_to_matrix, as_tensor

    pos = np.asarray(episode[K_TAXEL_POS], dtype=np.float64)
    pre = episode.meta.preprocessing or {}
    if pre.get("taxel_frame") == "mano_wrist":
        if not (episode.has(K_HAND_GLOBAL) and episode.has(K_HAND_WRIST)):
            return pos, np.zeros(episode.T, dtype=bool)
        R = aa_to_matrix(as_tensor(np.asarray(episode[K_HAND_GLOBAL], dtype=np.float64))).numpy()
        wp = np.asarray(episode[K_HAND_WRIST], dtype=np.float64)
        world = np.einsum("tij,tnj->tni", R, pos) + wp[:, None, :]
        valid = (np.asarray(episode[K_HAND_VALID], dtype=bool) if episode.has(K_HAND_VALID)
                 else np.ones(episode.T, bool))
        return world, valid
    return pos, np.ones(episode.T, dtype=bool)


def pseudo_label_episode(episode, prob: np.ndarray, *, hysteresis: HysteresisFilter | Mapping | None = None,
                         neg_thr: float = 0.1, keep_existing: bool = True, none_conflict: str = "unknown",
                         unknown_phase: str = "unknown", saturated_as_contact: bool = True,
                         proximity: Mapping[str, Any] | None = None,
                         phase_overrides: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Pseudo labels of one episode (module docstring) → ``{"label" int8[T,N], "on" bool[T,N],
    "expectation" [T], "counts" {...}}``. ``prob[T,N]``: detector probability (or any score in
    [0,1]); ``hysteresis``: a :class:`HysteresisFilter` or its config (default on 0.6 / off 0.4,
    dwell 2 / 4 ticks)."""
    from ..datasets.episode import K_CONTACT_LABEL, K_OBJECT_POS, K_SATURATED

    if none_conflict not in ("unknown", "zero"):
        raise ValueError("none_conflict must be 'unknown' or 'zero'")
    if unknown_phase not in ("unknown", "any", "none"):
        raise ValueError("unknown_phase must be 'unknown', 'any' or 'none'")
    p = np.asarray(prob, dtype=np.float64)
    T, N = episode.T, episode.meta.n_taxels
    if p.shape != (T, N):
        raise ValueError(f"prob must be {(T, N)}, got {p.shape}")
    hf = HysteresisFilter.from_config(hysteresis if hysteresis is not None
                                      else {"on_thr": 0.6, "off_thr": 0.4, "min_on": 2, "min_off": 4})
    on = hf.run(p)
    exp = frame_expectation(episode, overrides=phase_overrides)
    if unknown_phase != "unknown":
        exp = np.where(exp == "", unknown_phase, exp)
    allowed = np.isin(exp, _ALLOWED)[:, None]
    none = (exp == "none")[:, None]
    sat = np.asarray(episode[K_SATURATED], dtype=bool) if episode.has(K_SATURATED) else np.zeros((T, N), bool)
    fin = np.isfinite(p)
    low = fin & (p < float(neg_thr)) & ~on

    lab = np.full((T, N), -1, dtype=np.int8)
    # "none" phases: 0, conflicts (detector on) → −1 / 0
    lab[none & ~on] = 0
    lab[none & on] = 0 if none_conflict == "zero" else -1
    # contact-allowed phases
    lab[allowed & low] = 0
    lab[allowed & on] = 1
    sat_b = np.broadcast_to(allowed, (T, N)) & sat
    lab[sat_b] = 1 if saturated_as_contact else -1
    lab[sat & ~np.broadcast_to(allowed, (T, N))] = -1
    n_vetoed = 0
    if proximity:
        if not episode.has(K_OBJECT_POS):
            raise ValueError("proximity veto needs object_pos")
        world, pvalid = taxel_world_positions(episode)
        d = np.linalg.norm(world - np.asarray(episode[K_OBJECT_POS], dtype=np.float64)[:, None, :], axis=-1)
        far = (d > float(proximity.get("max_dist_m", 0.12))) & pvalid[:, None]
        veto = (lab == 1) & far
        n_vetoed = int(veto.sum())
        lab[veto] = -1
    n_conflict = int((none & on).sum())
    if keep_existing and episode.has(K_CONTACT_LABEL):
        ex = np.asarray(episode[K_CONTACT_LABEL])
        known = ex >= 0
        lab[known] = ex[known]
    counts = {"n_contact": int((lab == 1).sum()), "n_no_contact": int((lab == 0).sum()),
              "n_unknown": int((lab == -1).sum()), "n_conflict": n_conflict, "n_vetoed": n_vetoed,
              "n_on": int(on.sum())}
    return {"label": lab, "on": on, "expectation": exp, "counts": counts}


def pseudo_label_metrics(label: np.ndarray, truth: np.ndarray, *,
                         ignore: np.ndarray | None = None) -> dict[str, float]:
    """Agreement of pseudo labels with a ground-truth contact mask (synthetic data): precision and
    recall of label 1, negative precision of label 0 and label coverage (fraction of non-ignored
    samples that got 0/1)."""
    lab = np.asarray(label)
    g = np.asarray(truth, dtype=bool)
    keep = np.ones(lab.shape, bool) if ignore is None else ~np.asarray(ignore, dtype=bool)
    pos, neg = (lab == 1) & keep, (lab == 0) & keep

    def ratio(a: np.ndarray, b: np.ndarray) -> float:
        return float(a.sum() / b.sum()) if b.sum() else float("nan")

    return {"precision": ratio(pos & g, pos), "recall": ratio(pos & g, g & keep),
            "neg_precision": ratio(neg & ~g, neg), "coverage": ratio((lab >= 0) & keep, keep)}
