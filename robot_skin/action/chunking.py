"""Action chunking (training targets) and temporal ensembling (execution), following ACT.

ACT (Zhao et al., arXiv:2304.13705) predicts a *chunk* of the next ``H`` actions per
observation, which reduces the effective horizon of imitation and absorbs non-Markovian pauses
in human demonstrations; at execution it queries the policy every step and averages all
predictions that overlap the current step with exponential weights ``w_i = exp(−k·i)``, ``i = 0``
the *oldest* prediction (:class:`TemporalEnsembler`).

Episodes live on the 200 Hz master clock, policies run slower (default 20 Hz). A policy tick
every ``stride = round(200 / policy_hz)`` master frames (:func:`policy_stride`) and a chunk of
``H`` future actions spaced by the same stride (:func:`action_chunk`)::

    chunk[i] = actions[t + (offset + i) · stride],  i = 0 … H−1   (offset = 1 → strictly future)

With state-as-action data (human hand poses / robot q) ``actions[t]`` equals the current state,
so the default ``offset=1`` makes the chunk start at the *next* policy tick; use ``offset=0`` when
``actions[t]`` is already the command issued at ``t``. Steps past the episode end are padded with
the last frame's action (the last *valid* frame when a source mask such as ``hand_pose_valid`` is
given) and masked out (``valid=False``), as in ACT's ``is_pad``.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import torch


# ── rates ───────────────────────────────────────────────────────────────────
def policy_stride(source_hz: float = 200.0, policy_hz: float = 20.0, *, tol: float = 0.01) -> int:
    """Master frames per policy tick, ``round(source_hz / policy_hz)`` (≥ 1).

    Warns when the effective policy rate ``source_hz / stride`` deviates from ``policy_hz`` by
    more than ``tol`` (relative), i.e. the ratio is not (close to) an integer.
    """
    if source_hz <= 0 or policy_hz <= 0:
        raise ValueError("source_hz and policy_hz must be > 0")
    if policy_hz > source_hz:
        raise ValueError(f"policy_hz {policy_hz} exceeds the data rate {source_hz}")
    ratio = source_hz / policy_hz
    stride = max(1, int(round(ratio)))
    if abs(source_hz / stride - policy_hz) > tol * policy_hz:
        warnings.warn(f"{source_hz}/{policy_hz} Hz is not an integer ratio; using stride {stride} "
                      f"(effective policy rate {source_hz / stride:.3g} Hz)")
    return stride


def policy_tick_indices(T: int, stride: int, *, start: int = 0, mask: np.ndarray | None = None,
                        min_future: int = 0) -> np.ndarray:
    """Master-clock indices of policy ticks ``start, start+stride, …`` (< ``T``).

    ``mask`` [T] bool keeps only ticks where it is True (e.g. ``phase_id >= 0``);
    ``min_future`` drops ticks with fewer than that many frames after them (e.g. ``stride`` to
    guarantee at least one valid chunk step).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if start < 0:
        raise ValueError("start must be >= 0")
    idx = np.arange(int(start), int(T), int(stride), dtype=np.int64)
    if min_future > 0:
        idx = idx[idx + int(min_future) <= T - 1]
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (T,):
            raise ValueError(f"mask must be [{T}], got {mask.shape}")
        idx = idx[mask[idx]]
    return idx


# ── chunk extraction ────────────────────────────────────────────────────────
def _check_chunk_args(T: int, horizon: int, stride: int, offset: int) -> None:
    if T < 1:
        raise ValueError("actions must have at least one row")
    if horizon < 1 or stride < 1 or offset < 0:
        raise ValueError(f"need horizon >= 1, stride >= 1, offset >= 0 (got {horizon}, {stride}, {offset})")


def _last_valid_index(v: np.ndarray) -> np.ndarray:
    """``[T]`` bool → for each frame the index of the last valid frame at or before it (the first
    valid frame if none precedes; the frame itself if nothing is valid)."""
    T = v.shape[0]
    if not v.any():
        return np.arange(T, dtype=np.int64)
    ar = np.where(v, np.arange(T, dtype=np.int64), -1)
    ff = np.maximum.accumulate(ar)
    return np.where(ff < 0, int(np.argmax(v)), ff)


def action_chunks(actions, t_indices, horizon: int, stride: int = 1, *, offset: int = 1,
                  valid=None):
    """Vectorised :func:`action_chunk`: ``t_indices[B]`` → ``(chunks[B,H,A], valid[B,H])``.

    Works on numpy (incl. memmaps — only the needed rows are read) and torch; the chunk keeps the
    input type, ``valid`` is a bool array of the same library.
    """
    is_t = isinstance(actions, torch.Tensor)
    if not is_t and not hasattr(actions, "shape"):
        actions = np.asarray(actions)
    if actions.ndim < 1:
        raise ValueError("actions must be [T, ...]")
    T = int(actions.shape[0])
    _check_chunk_args(T, horizon, stride, offset)
    t = np.asarray(t_indices, dtype=np.int64).reshape(-1)
    if t.size and (t.min() < 0 or t.max() >= T):
        raise IndexError(f"t_index out of range [0, {T})")
    idx = t[:, None] + (int(offset) + np.arange(int(horizon), dtype=np.int64))[None, :] * int(stride)  # [B,H]
    ok = idx <= T - 1
    idx_c = np.minimum(idx, T - 1)                 # past the end → last frame (padding)
    if valid is not None:
        v = valid.detach().cpu().numpy() if isinstance(valid, torch.Tensor) else np.asarray(valid)
        if v.shape != (T,):
            raise ValueError(f"valid must be [{T}], got {v.shape}")
        v = v.astype(bool)
        ok = ok & v[idx_c]
        # masked steps repeat the last *valid* source frame, so padding never carries the
        # (possibly NaN / garbage) values of invalid frames into a masked loss (NaN·0 = NaN)
        idx_c = np.where(ok, idx_c, _last_valid_index(v)[idx_c])
    if is_t:
        chunk = actions[torch.as_tensor(idx_c, device=actions.device)]
        return chunk, torch.as_tensor(ok, device=actions.device)
    arr = np.asarray(actions) if not isinstance(actions, np.ndarray) else actions
    flat = np.asarray(arr[idx_c.reshape(-1)])
    return flat.reshape(*idx_c.shape, *arr.shape[1:]), ok


def action_chunk(actions, t_index: int, horizon: int, stride: int = 1, *, offset: int = 1, valid=None):
    """Chunk of ``horizon`` future actions for the observation at master index ``t_index``.

    ``chunk[i] = actions[t + (offset+i)·stride]``; indices past the end are padded with the last
    frame and flagged ``valid[i] = False``. ``valid`` (optional [T] bool, e.g. ``hand_pose_valid``)
    additionally masks steps whose source frame is invalid; with it, every masked step (past the
    end or invalid source) holds the last *valid* source frame at or before it, so padded values
    are always finite when the valid frames are. Returns ``(chunk[H,A], valid[H])``.
    """
    c, v = action_chunks(actions, [int(t_index)], horizon, stride, offset=offset, valid=valid)
    return c[0], v[0]


# ── temporal ensembling (ACT) ───────────────────────────────────────────────
class TemporalEnsembler:
    """ACT temporal aggregation over overlapping action chunks.

    Call :meth:`add` with each new chunk (``[h, A]``, ``h ≤ horizon``; entry ``j`` is the action
    for ``j`` steps after the step it was added at), then :meth:`step` once per executed step. The
    action for the current step is ``Σ_i w_i a_i / Σ_i w_i`` over all stored chunks that cover it,
    ordered oldest → newest with ``w_i = exp(−k·i)`` — ``k > 0`` trusts older predictions more
    (ACT's default ``k = 0.01``), ``k = 0`` is a plain mean, ``k < 0`` favours the newest.
    Chunks may be added every step (ACT) or every few steps; adding twice at the same step
    replaces the earlier chunk of that step.
    """

    def __init__(self, horizon: int, action_dim: int, k: float = 0.01):
        if horizon < 1 or action_dim < 1:
            raise ValueError("horizon and action_dim must be >= 1")
        if not math.isfinite(k):
            raise ValueError("k must be finite")
        self.horizon, self.action_dim, self.k = int(horizon), int(action_dim), float(k)
        self.reset()

    def reset(self) -> None:
        """Forget all chunks and restart the step counter (new episode)."""
        self._chunks: list[tuple[int, np.ndarray]] = []    # (step added, chunk [h, A]) oldest first
        self._t = 0

    @property
    def t(self) -> int:
        """Number of executed steps since the last reset."""
        return self._t

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    @property
    def ready(self) -> bool:
        """True when some stored chunk covers the current step."""
        return any(0 <= self._t - t0 < len(c) for t0, c in self._chunks)

    def add(self, chunk) -> None:
        c = chunk.detach().cpu().numpy() if isinstance(chunk, torch.Tensor) else np.asarray(chunk)
        c = np.asarray(c, dtype=np.float64)
        if c.ndim == 3 and c.shape[0] == 1:        # tolerate a batch of one
            c = c[0]
        if c.ndim != 2 or c.shape[1] != self.action_dim or not 1 <= c.shape[0] <= self.horizon:
            raise ValueError(f"chunk must be [h<= {self.horizon}, {self.action_dim}], got {c.shape}")
        if not np.isfinite(c).all():
            raise ValueError("chunk contains non-finite values")
        if self._chunks and self._chunks[-1][0] == self._t:
            self._chunks.pop()
        self._chunks.append((self._t, c.copy()))

    def weights(self, n: int) -> np.ndarray:
        """Normalised ACT weights for ``n`` contributing predictions (oldest first)."""
        w = np.exp(-self.k * np.arange(n, dtype=np.float64))
        return w / w.sum()

    def current_predictions(self) -> np.ndarray:
        """``[n, A]`` predictions for the current step, oldest first (n may be 0)."""
        rows = [c[self._t - t0] for t0, c in self._chunks if 0 <= self._t - t0 < len(c)]
        return np.stack(rows) if rows else np.zeros((0, self.action_dim))

    def step(self) -> np.ndarray:
        """Ensembled action ``[A]`` (float64) for the current step, then advance one step."""
        preds = self.current_predictions()
        if preds.shape[0] == 0:
            raise RuntimeError(f"no chunk covers step {self._t}: call add() with a new chunk first "
                               f"(chunks expire {self.horizon} steps after they are added)")
        action = (self.weights(preds.shape[0])[:, None] * preds).sum(0)
        self._t += 1
        self._chunks = [(t0, c) for t0, c in self._chunks if self._t - t0 < len(c)]
        return action


__all__ = ["policy_stride", "policy_tick_indices", "action_chunk", "action_chunks", "TemporalEnsembler"]
