"""Preference optimisation hook for the VTLA policy (DPO on action chunks).

Direct Preference Optimization (Rafailov et al., NeurIPS 2023, arXiv:2305.18290) fine-tunes a
policy ``π_θ`` from preference pairs ``(o, a_w ≻ a_l)`` against a frozen reference ``π_ref`` (the
imitation-trained policy) without an explicit reward model::

    L = −log σ( β · [ (log π_θ(a_w|o) − log π_ref(a_w|o)) − (log π_θ(a_l|o) − log π_ref(a_l|o)) ] )

The VTLA paper (Zhang et al., arXiv:2505.09577) applies DPO-style preference learning to its
vision–tactile–language–action model for contact-rich insertion. Here :func:`dpo_loss` is the
generic loss on log-likelihoods, and :func:`preference_loss` plugs a :class:`VTLAPolicy` in.

Log-likelihood surrogates (continuous chunks have no cheap exact likelihood; constants cancel in
the policy − reference differences as long as both use the same surrogate):

- **chunk head** — isotropic Gaussian around the regressed chunk with fixed ``σ`` (normalized
  units): ``log π(a|o) ≈ −‖a − μ_θ(o)‖² / (2σ²)`` with ‖·‖² the mean over valid (step, dim) entries;
- **flow head** — ``−(flow-matching error) / (2σ²)`` at shared noise / time draws ``(ε, τ)``
  (the *same* draws for chosen / rejected and policy / reference, :func:`preference_loss`). This is
  a heuristic surrogate, not an exact likelihood (which would need the divergence integral of the
  probability-flow ODE).

Variance control in :func:`preference_loss`: each model encodes the observation **once** and the
chosen and rejected chunks are scored on that same encoding, and by default the policy runs with
dropout / modality dropout disabled (eval mode, gradients kept — as ``disable_dropout`` in common
DPO trainers). Otherwise independent dropout draws for the four log-likelihoods would make the
implicit rewards noisy and the loss ≠ log 2 even when ``π_θ = π_ref``.

Pipeline status — **not implemented** (:func:`build_preference_pairs` raises): preference pairs
must come from robot/teleop **rollouts** of the same instruction and a comparable initial state,
labelled success ≻ failure (``events.jsonl`` ``success`` event / ``manifest.task.success``), or
ranked by a task metric (e.g. insertion depth / force). Each pair holds the observation at the
divergence tick and the two executed action chunks (normalized with the bundle's normalizer, in
the same relative frame). D2 human demonstrations are (almost) all successes, so they provide the
reference policy, not the pairs.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import draw_on_generator

__all__ = ["dpo_loss", "chunk_log_likelihood", "preference_loss", "make_reference_policy",
           "PreferencePair", "build_preference_pairs"]


def dpo_loss(policy_logp_chosen: torch.Tensor, policy_logp_rejected: torch.Tensor,
             ref_logp_chosen: torch.Tensor, ref_logp_rejected: torch.Tensor, beta: float = 0.1,
             *, reduction: str = "mean") -> dict[str, torch.Tensor]:
    """DPO loss on per-pair log-likelihoods ``[B]`` (Rafailov et al., 2023, eq. 7).

    Returns ``{"loss", "reward_chosen", "reward_rejected", "reward_margin", "reward_accuracy"}``
    where the implicit rewards are ``β·(log π_θ − log π_ref)`` (detached, batch means) and
    ``reward_accuracy`` is the fraction of pairs with chosen reward > rejected reward. ``reduction``:
    ``mean`` | ``sum`` | ``none`` (per-pair loss ``[B]``).
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")
    shapes = {tuple(x.shape) for x in (policy_logp_chosen, policy_logp_rejected, ref_logp_chosen,
                                        ref_logp_rejected)}
    if len(shapes) != 1:
        raise ValueError(f"log-likelihoods must share one shape, got {sorted(shapes)}")
    chosen = policy_logp_chosen - ref_logp_chosen.detach()
    rejected = policy_logp_rejected - ref_logp_rejected.detach()
    logits = beta * (chosen - rejected)
    per_pair = -F.logsigmoid(logits)
    if reduction == "mean":
        loss = per_pair.mean()
    elif reduction == "sum":
        loss = per_pair.sum()
    elif reduction == "none":
        loss = per_pair
    else:
        raise ValueError("reduction must be mean|sum|none")
    rc, rr = (beta * chosen).detach(), (beta * rejected).detach()
    return {"loss": loss, "reward_chosen": rc.mean(), "reward_rejected": rr.mean(),
            "reward_margin": (rc - rr).mean(), "reward_accuracy": (rc > rr).float().mean()}


def chunk_log_likelihood(policy: nn.Module, batch: Mapping[str, Any], actions: torch.Tensor,
                         valid: torch.Tensor | None = None, *, sigma: float = 1.0,
                         noise: torch.Tensor | None = None,
                         tau: torch.Tensor | None = None,
                         encoding: Mapping[str, Any] | None = None) -> torch.Tensor:
    """Surrogate ``log π(actions | obs)`` ``[B]`` (up to a constant) of a :class:`VTLAPolicy`
    (bare module — under DDP pass ``robot_skin.train.unwrap_model(model)``): ``−err / (2σ²)``
    with ``err`` the head's per-sample error (see the module docstring). ``noise`` / ``tau`` are
    used by the flow head. ``encoding`` (``policy.encode(batch)`` output) reuses one observation
    encoding for several action sets instead of re-encoding ``batch``."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    kw = {"noise": noise, "tau": tau} if getattr(policy.head, "kind", "") == "flow" else {}
    if encoding is None:
        err = policy.per_sample_action_error(batch, actions, valid, **kw)
    else:
        mem = encoding["memory"]
        v = None if valid is None else valid.to(mem.device).bool()
        err = policy.head.per_sample_error(mem, encoding["memory_mask"], actions.to(mem.device), v, **kw)
    return -err / (2.0 * sigma * sigma)


@contextlib.contextmanager
def _eval_mode(*modules: nn.Module) -> Iterator[None]:
    """Temporarily switch modules to eval mode (dropout off; gradients unaffected)."""
    states = [m.training for m in modules]
    for m in modules:
        m.eval()
    try:
        yield
    finally:
        for m, s in zip(modules, states, strict=True):
            m.train(s)


def preference_loss(policy: nn.Module, reference: nn.Module, batch: Mapping[str, Any], *,
                    beta: float = 0.1, sigma: float = 1.0, n_draws: int = 1,
                    generator: torch.Generator | None = None,
                    disable_dropout: bool = True) -> dict[str, torch.Tensor]:
    """DPO loss of ``policy`` vs the frozen ``reference`` on a batch of preference pairs.

    ``batch`` = an observation batch (:func:`robot_skin.vtla.dataset.collate_vtla`) plus
    ``actions_chosen`` / ``actions_rejected [B,H,A]`` (normalized, relative like the training
    targets) and optional ``valid_chosen`` / ``valid_rejected [B,H]``. Each model encodes the
    observation once (shared by chosen / rejected and all draws). For the flow head the surrogate
    is averaged over ``n_draws`` shared ``(ε, τ)`` draws (``generator`` for reproducibility — drawn on
    the generator's device, so a CUDA generator works for a CUDA policy). The
    reference runs without gradients. ``disable_dropout`` (default) evaluates the policy in eval
    mode — no dropout / modality dropout, gradients still flow — so that ``loss = log 2`` exactly
    when ``policy`` equals ``reference``; the policy's previous mode is restored afterwards.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be >= 1")
    ac, ar = batch["actions_chosen"], batch["actions_rejected"]
    if ac.shape != ar.shape or ac.ndim != 3:
        raise ValueError(f"actions_chosen {tuple(ac.shape)} / actions_rejected {tuple(ar.shape)} "
                         "must both be [B,H,A]")
    vc, vr = batch.get("valid_chosen"), batch.get("valid_rejected")
    is_flow = getattr(policy.head, "kind", "") == "flow"
    draws = n_draws if is_flow else 1
    sums = [torch.zeros(ac.shape[0], device=ac.device) for _ in range(4)]
    mode = _eval_mode(policy, reference) if disable_dropout else _eval_mode(reference)
    with mode:
        enc_p = policy.encode(batch)
        with torch.no_grad():
            enc_r = reference.encode(batch)
        for _ in range(draws):
            kw: dict[str, Any] = {}
            if is_flow:
                kw["noise"] = draw_on_generator(torch.randn, ac.shape, generator=generator, device=ac.device)
                kw["tau"] = policy.head.sample_tau(ac.shape[0], generator=generator, device=ac.device)
            pc = chunk_log_likelihood(policy, batch, ac, vc, sigma=sigma, encoding=enc_p, **kw)
            pr = chunk_log_likelihood(policy, batch, ar, vr, sigma=sigma, encoding=enc_p, **kw)
            with torch.no_grad():
                rc = chunk_log_likelihood(reference, batch, ac, vc, sigma=sigma, encoding=enc_r, **kw)
                rr = chunk_log_likelihood(reference, batch, ar, vr, sigma=sigma, encoding=enc_r, **kw)
            for i, x in enumerate((pc, pr, rc, rr)):
                sums[i] = sums[i] + x.to(sums[i].device)
    pc, pr, rc, rr = (s / draws for s in sums)
    return dpo_loss(pc, pr, rc, rr, beta)


def make_reference_policy(policy: nn.Module) -> nn.Module:
    """Frozen deep copy (eval mode, no gradients) of ``policy`` — the DPO reference ``π_ref``."""
    ref = copy.deepcopy(policy)
    ref.requires_grad_(False)
    return ref.eval()


@dataclass
class PreferencePair:
    """One preference pair: the observation tick shared by both rollouts and the two executed
    chunks (``chosen`` ≻ ``rejected``). Paths / ids point at processed deployment episodes."""
    instruction: str
    episode_chosen: str
    episode_rejected: str
    t_chosen: int
    t_rejected: int
    meta: dict = field(default_factory=dict)


def build_preference_pairs(rollouts: Sequence[Any], **kwargs: Any) -> list[PreferencePair]:
    """**Not implemented** — documented pipeline stub.

    Intended pipeline (see the module docstring): (1) deploy the imitation policy
    (``control.runner`` logs every tick as a deployment session, reusable as data); (2) label each
    rollout success / failure (``success`` event) or score it with a task metric; (3) pair rollouts
    with the same instruction and a comparable initial state (object pose / hand state within a
    tolerance), preferring success over failure; (4) pick the divergence tick and store both
    executed chunks → :class:`PreferencePair`; (5) fine-tune with :func:`preference_loss` against
    :func:`make_reference_policy` of the starting policy.
    """
    raise NotImplementedError(
        "preference pairs need robot / teleop rollouts labelled success vs failure (VTLA, "
        "arXiv:2505.09577, uses preference learning on insertion rollouts); no rollout data exists "
        "yet. Collect deployment sessions with control.runner, label success, pair rollouts of the "
        "same instruction and initial state, then train with vtla.dpo.preference_loss.")
