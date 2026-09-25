"""Latency measurement for the control loop and the policy (p50 / p95), optional TorchScript export.

Budget at the default rates (``docs/DEPLOYMENT.md``): the control loop ticks at the tactile master
rate (200 Hz → 5 ms per tick: read sensors, tactile processor incl. the baseline network, safety,
send); the policy runs every ``stride`` ticks (20 Hz → one inference per 50 ms). A synchronous
inference longer than one control period delays that tick (an *overrun*, counted by the runner),
so on a CPU the policy should stay well under ~5 ms or the loop falls behind; with ACT-style
chunking + temporal ensembling (Zhao et al., arXiv:2304.13705) the robot keeps executing the
previous chunk, so occasional overruns degrade smoothness rather than safety.

- :class:`LatencyMeter` — accumulate per-call wall times (``time.perf_counter``) and summarise.
- :func:`percentile_summary` — ``{n, mean_ms, p50_ms, p95_ms, p99_ms, max_ms}``.
- :func:`example_batch` — a single-sample observation batch matching a bundle (dummy values).
- :func:`benchmark_policy` — warm-up + ``n`` timed ``predict`` calls (CUDA-synchronised on GPU).
- :func:`export_torchscript` — trace / script a module for C++ / low-overhead deployment
  (guarded: VTLA policies with string instructions / dict batches are generally not traceable; the
  baseline predictor and tactile encoder are).
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

__all__ = ["percentile_summary", "LatencyMeter", "example_batch", "benchmark_policy", "export_torchscript"]


def percentile_summary(ms: Sequence[float]) -> dict[str, float]:
    """``{n, mean_ms, p50_ms, p95_ms, p99_ms, max_ms}`` of latencies in milliseconds (NaN when empty)."""
    a = np.asarray(list(ms), dtype=np.float64)
    if a.size == 0:
        nan = float("nan")
        return {"n": 0, "mean_ms": nan, "p50_ms": nan, "p95_ms": nan, "p99_ms": nan, "max_ms": nan}
    return {"n": int(a.size), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())}


class LatencyMeter:
    """Collect call durations (ms): ``with meter.time(): ...`` or ``meter.record(ms)``."""

    def __init__(self, name: str = "", max_samples: int | None = 100000):
        self.name = name
        self.max_samples = max_samples
        self.samples: list[float] = []

    def record(self, ms: float) -> None:
        if self.max_samples is None or len(self.samples) < self.max_samples:
            self.samples.append(float(ms))

    @contextlib.contextmanager
    def time(self, sync: Callable[[], Any] | None = None) -> Iterator[None]:
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync is not None:
                sync()
            self.record((time.perf_counter() - t0) * 1e3)

    def reset(self) -> None:
        self.samples = []

    def summary(self) -> dict[str, float]:
        return percentile_summary(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


def _cuda_sync(device: Any) -> Callable[[], Any] | None:
    import torch

    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        return lambda: torch.cuda.synchronize(dev)
    return None


def example_batch(bundle: Any, *, n_taxels: int = 9, image_hw: Sequence[int] | None = None,
                  instruction: str = "pick up the cup", seed: int = 0) -> dict[str, Any]:
    """A collated single-sample batch shaped like the bundle's training observations (random
    tactile values / poses, grey images at ``image_hw`` — default the eval transform's output
    size or 96×128 — and the bundle's proprio history). For latency benchmarks, not inference."""
    import torch

    from ..vtla.dataset import collate_vtla, make_observation

    rng = np.random.default_rng(seed)
    k, A = int(bundle.obs_history), int(bundle.action_dim)
    F = int(bundle.feature_spec.dim) if bundle.feature_spec is not None else 0
    pos = rng.normal(scale=0.05, size=(n_taxels, 3)).astype(np.float32)
    nrm = rng.normal(size=(n_taxels, 3)).astype(np.float32)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    images = None
    if bundle.cameras:
        tf = bundle.eval_transform
        hw = tuple(image_hw) if image_hw is not None else (tuple(tf.out_size) if tf is not None and tf.out_size
                                                           else (96, 128))
        frame = np.full((k, *hw, 3), 128, np.uint8)
        x = tf(frame) if tf is not None else torch.from_numpy(frame).permute(0, 3, 1, 2).float() / 255.0
        images = {c: (x[0] if k == 1 else x) for c in bundle.cameras}
    obs = make_observation(proprio_states=np.zeros((k, A), np.float32),
                           tactile_values=rng.normal(size=(n_taxels, F)).astype(np.float32),
                           taxel_pos=pos, taxel_nrm=nrm, contact=rng.random(n_taxels) > 0.7,
                           instruction=instruction, proprio_normalizer=bundle.proprio_normalizer, images=images)
    return collate_vtla([obs], bundle.tokenizer)


def _to_device(batch: Any, device: Any) -> Any:
    import torch

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list) and batch and isinstance(batch[0], torch.Tensor):
        return [b.to(device) for b in batch]
    return batch


def benchmark_policy(bundle_or_policy: Any, batch: Mapping[str, Any] | None = None, *, device: Any = None,
                     n: int = 50, warmup: int = 5, n_steps: int | None = None, seed: int = 0,
                     **batch_kw: Any) -> dict[str, Any]:
    """Time ``policy.predict(batch)`` (``n`` calls after ``warmup``) → ``{n, mean_ms, p50_ms, p95_ms,
    p99_ms, max_ms, device, head, batch_size}``. ``bundle_or_policy``: a
    :class:`~robot_skin.control.bundle.PolicyBundle`, a bundle path, or a policy with ``predict``
    (then ``batch`` is required); ``batch`` defaults to :func:`example_batch` (``batch_kw``)."""
    import torch

    from .bundle import PolicyBundle, load_policy_bundle

    if isinstance(bundle_or_policy, (str, Path)):
        bundle_or_policy = load_policy_bundle(bundle_or_policy, device=device or "cpu")
    if isinstance(bundle_or_policy, PolicyBundle):
        policy = bundle_or_policy.policy
        if batch is None:
            batch = example_batch(bundle_or_policy, seed=seed, **batch_kw)
    else:
        policy = bundle_or_policy
        if batch is None:
            raise ValueError("pass a batch when benchmarking a bare policy")
    dev = torch.device(device) if device is not None else next(policy.parameters()).device
    policy = policy.to(dev).eval()
    batch = _to_device(batch, dev)
    sync = _cuda_sync(dev)
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    meter = LatencyMeter("policy")
    with torch.inference_mode():
        for i in range(int(warmup) + int(n)):
            if i < warmup:
                policy.predict(batch, n_steps, generator=gen)
                continue
            with meter.time(sync):
                policy.predict(batch, n_steps, generator=gen)
    B = next((int(v.shape[0]) for v in batch.values() if isinstance(v, torch.Tensor)), 1)
    return {**meter.summary(), "device": str(dev), "head": getattr(getattr(policy, "cfg", None), "head", None),
            "batch_size": B}


class _ExportMismatch(RuntimeError):
    pass


def export_torchscript(module: Any, example_inputs: Any, path: str | Path, *, method: str = "trace",
                       check: bool = True, rtol: float = 1e-4, atol: float = 1e-5) -> Path:
    """Export ``module`` with ``torch.jit.trace`` (``example_inputs``: a tensor or tuple) or
    ``torch.jit.script`` and save it to ``path``; with ``check`` the reloaded module must reproduce
    the eager outputs. Raises ``RuntimeError`` (with the reason) when the module cannot be exported —
    e.g. a VTLA policy fed dicts of strings; export its tensor-only parts instead. (Recent torch
    versions deprecate TorchScript in favour of ``torch.export``; the artefact still loads with
    ``torch.jit.load`` / libtorch.)"""
    import warnings

    import torch

    if method not in ("trace", "script"):
        raise ValueError("method must be 'trace' or 'script'")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    was = module.training
    module.eval()
    inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            ts = torch.jit.trace(module, inputs) if method == "trace" else torch.jit.script(module)
            ts.save(str(p))
            if check:
                ref = module(*inputs)
                got = torch.jit.load(str(p))(*inputs)
                refs = ref if isinstance(ref, (tuple, list)) else (ref,)
                gots = got if isinstance(got, (tuple, list)) else (got,)
                for a, b in zip(refs, gots):
                    if not torch.allclose(a, b, rtol=rtol, atol=atol):
                        raise _ExportMismatch("TorchScript output differs from the eager module")
    except _ExportMismatch:
        raise
    except Exception as e:  # noqa: BLE001 - tracing errors come in many types
        raise RuntimeError(f"TorchScript export ({method}) failed: {type(e).__name__}: {e}") from e
    finally:
        module.train(was)
    return p
