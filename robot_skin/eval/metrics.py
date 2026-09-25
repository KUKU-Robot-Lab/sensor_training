"""Evaluation metrics: hallucination rate, motion–contact separability, saturation recovery."""
from __future__ import annotations

import numpy as np

from common.signal import press_intensity

from ..contact.saturation_fsm import SatState


def hallucination_rate(pred_contact: np.ndarray, true_contact: np.ndarray, *,
                       level: str = "frame") -> float:
    """Fraction of truly contact-free units where contact is predicted.

    ``level="frame"``: unit = frame ``[T]`` (no true contact on any taxel; any predicted →
    hallucination). ``level="taxel"``: unit = (frame, taxel). NaN if there are no free units.
    """
    p = np.asarray(pred_contact, dtype=bool)
    g = np.asarray(true_contact, dtype=bool)
    if p.shape != g.shape:
        raise ValueError("shape mismatch")
    if level == "frame":
        free = ~g.reshape(g.shape[0], -1).any(-1)
        hit = p.reshape(p.shape[0], -1).any(-1)
    elif level == "taxel":
        free, hit = ~g, p
    else:
        raise ValueError("level must be frame|taxel")
    return float(hit[free].mean()) if free.any() else float("nan")


def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mann–Whitney AUROC (ties count ½). Positives should score higher."""
    pos = np.asarray(pos_scores, dtype=np.float64).ravel()
    neg = np.asarray(neg_scores, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(allv)
    sv = allv[order]
    i = 0
    while i < sv.size:  # average ranks over ties
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def motion_contact_separability(resid_pct: np.ndarray, contact: np.ndarray,
                                moving: np.ndarray) -> dict[str, float]:
    """How well press intensity separates real contact from motion-only (no contact, moving).

    ``resid_pct[T,N]``, ``contact[T,N]`` bool, ``moving[T]`` bool. Positives = contact samples;
    negatives = non-contact samples during motion. Returns ``{"auroc", "dprime"}`` — before vs
    after baseline subtraction is the headline comparison.
    """
    s = press_intensity(resid_pct).astype(np.float64)
    c = np.asarray(contact, dtype=bool)
    mv = np.broadcast_to(np.asarray(moving, dtype=bool)[:, None], c.shape)
    pos, neg = s[c], s[~c & mv]
    if pos.size < 2 or neg.size < 2:
        return {"auroc": float("nan"), "dprime": float("nan")}
    dprime = (pos.mean() - neg.mean()) / np.sqrt(0.5 * (pos.var() + neg.var()) + 1e-12)
    return {"auroc": auroc(pos, neg), "dprime": float(dprime)}


def saturation_recovery_times(states: np.ndarray, dt: float) -> np.ndarray:
    """Seconds from leaving SATURATED to reaching OK, per completed episode (all taxels).

    ``states[T,N]`` from :class:`~robot_skin.contact.SaturationFSM`. Episodes still recovering
    at the end are not counted.
    """
    st = np.asarray(states)
    out = []
    for n in range(st.shape[1]):
        start = None
        for t in range(1, st.shape[0]):
            prev, cur = st[t - 1, n], st[t, n]
            if prev == SatState.SATURATED and cur != SatState.SATURATED:
                start = t
            if start is not None:
                if cur == SatState.SATURATED:
                    start = None
                elif cur == SatState.OK:
                    out.append((t - start) * dt)
                    start = None
    return np.asarray(out, dtype=np.float64)
