"""Hyper-parameter sweeps: grid / random search over dotted config paths.

A *space* maps dotted config paths to candidates::

    {"train.lr": {"log_uniform": [1e-4, 1e-3]},   # random: log-uniform float
     "train.weight_decay": {"uniform": [0.0, 0.1]},
     "model.depth": {"int": [2, 6]},               # inclusive integer range
     "model.hidden": [128, 256]}                   # list = choices (grid axis / random choice)

(a list value that should be one *fixed* value, e.g. betas, is written ``[[0.9, 0.95]]``).
Each trial is an override dict (nested) that is deep-merged onto the base stage config and
passed to ``train_fn(cfg) -> float | {"metric": float, ...}``; results are appended to
``out_dir/results.jsonl`` as trials finish (crash-safe, and already-finished trials are skipped
when a sweep is restarted).

Multi-machine (e.g. an RTX 5090 box and an RTX 4090 box on a Tailscale tailnet): generate the
same trial list on every machine (same space + seed) and run a disjoint slice with
:func:`shard` (``--shard 0/2`` and ``--shard 1/2``), then ``rsync`` the ``results.jsonl`` files
back and merge with :func:`load_results`. Optuna (TPE, pruning) is supported through
:func:`run_optuna` when ``optuna`` is installed.

CLI::

    python -m robot_skin.train.sweep --fn robot_skin.stages.vtla:run \\
        --base robot_skin/configs/stages/vtla.yaml --space sweep.yaml \\
        --metric val/loss --out robot_skin/runs/sweeps/vtla_lr --shard 0/2
"""
from __future__ import annotations

import copy
import importlib
import itertools
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from robot_skin.config import deep_merge

__all__ = ["set_by_path", "get_by_path", "unflatten", "flatten", "expand_grid", "sample_random",
           "shard", "run_sweep", "load_results", "suggest_from_space", "run_optuna", "main"]

_DIST_KEYS = ("log_uniform", "uniform", "int", "choice")


# ───────────────────────────────────────────────────────────────────────────── dotted paths

def set_by_path(d: dict, path: str, value: Any) -> dict:
    """``set_by_path(d, "train.lr", 1e-3)`` → ``d["train"]["lr"] = 1e-3`` (creates levels)."""
    keys = path.split(".")
    if not all(keys):
        raise ValueError(f"invalid dotted path {path!r}")
    cur = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value
    return d


_MISSING = object()


def get_by_path(d: Mapping[str, Any], path: str, default: Any = _MISSING) -> Any:
    """Inverse of :func:`set_by_path`; a flat key containing dots is tried first."""
    if path in d:
        return d[path]
    cur: Any = d
    for k in path.split("."):
        if isinstance(cur, Mapping) and k in cur:
            cur = cur[k]
        elif default is not _MISSING:
            return default
        else:
            raise KeyError(path)
    return cur


def unflatten(flat: Mapping[str, Any]) -> dict:
    """``{"a.b": 1, "a.c": 2}`` → ``{"a": {"b": 1, "c": 2}}``."""
    out: dict = {}
    for k, v in flat.items():
        set_by_path(out, k, copy.deepcopy(v))
    return out


def flatten(nested: Mapping[str, Any], prefix: str = "") -> dict:
    """``{"a": {"b": 1}}`` → ``{"a.b": 1}`` (inverse of :func:`unflatten`)."""
    out: dict = {}
    for k, v in nested.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


# ───────────────────────────────────────────────────────────────────────────── spaces

def _is_dist(spec: Any) -> bool:
    return isinstance(spec, Mapping) and len(spec) == 1 and next(iter(spec)) in _DIST_KEYS


def _choices(path: str, spec: Any) -> list:
    if isinstance(spec, Mapping) and "choice" in spec and len(spec) == 1:
        spec = spec["choice"]
    if _is_dist(spec):
        raise ValueError(f"{path}: {next(iter(spec))!r} distributions are only valid for "
                         "random search; use a list of values for grid search")
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError(f"{path}: empty candidate list")
        return list(spec)
    return [spec]  # scalar → fixed value


def expand_grid(space: Mapping[str, Any]) -> list[dict]:
    """Cartesian product of the list-valued axes (scalars are fixed) → nested override dicts,
    in insertion order (last key varies fastest)."""
    if not space:
        return [{}]
    paths = list(space)
    axes = [_choices(p, space[p]) for p in paths]
    return [unflatten(dict(zip(paths, combo))) for combo in itertools.product(*axes)]


def _range(path: str, spec: Any, name: str) -> tuple[float, float]:
    try:
        a, b = (float(x) for x in spec)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: {name} needs [low, high], got {spec!r}") from e
    if a > b:
        raise ValueError(f"{path}: {name} low > high ({a} > {b})")
    return a, b


def _sample_one(path: str, spec: Any, rng: np.random.Generator) -> Any:
    if _is_dist(spec):
        kind, arg = next(iter(spec.items()))
        if kind == "choice":
            return _sample_one(path, list(arg), rng)
        if kind == "int":
            a, b = _range(path, arg, kind)
            if not (a.is_integer() and b.is_integer()):
                raise ValueError(f"{path}: int bounds must be integers, got {arg!r}")
            return int(rng.integers(int(a), int(b) + 1))
        a, b = _range(path, arg, kind)
        if kind == "uniform":
            return float(rng.uniform(a, b))
        if a <= 0:
            raise ValueError(f"{path}: log_uniform needs positive bounds, got {arg!r}")
        return float(math.exp(rng.uniform(math.log(a), math.log(b))))
    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError(f"{path}: empty candidate list")
        return copy.deepcopy(spec[int(rng.integers(len(spec)))])
    return copy.deepcopy(spec)


def sample_random(space: Mapping[str, Any], n: int, seed: int = 0) -> list[dict]:
    """``n`` random configurations (nested override dicts), deterministic for a given seed.
    Lists = uniform choice; ``{"log_uniform"|"uniform"|"int"|"choice": ...}`` distributions."""
    if n < 0:
        raise ValueError("n must be >= 0")
    rng = np.random.default_rng(seed)
    return [unflatten({p: _sample_one(p, spec, rng) for p, spec in space.items()})
            for _ in range(n)]


def shard(items: Sequence[Any], index: int, count: int) -> list:
    """Disjoint slice ``items[index::count]`` for machine ``index`` of ``count``."""
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"invalid shard {index}/{count}")
    return list(items[index::count])


# ───────────────────────────────────────────────────────────────────────────── running

def _key(overrides: Mapping[str, Any]) -> str:
    return json.dumps(flatten(overrides), sort_keys=True, default=str)


def _sort_results(results: Iterable[dict], mode: str) -> list[dict]:
    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'")
    results = list(results)
    ok = [r for r in results if r.get("status") == "ok" and r.get("metric") is not None
          and math.isfinite(float(r["metric"]))]
    ok_ids = {id(r) for r in ok}
    bad = [r for r in results if id(r) not in ok_ids]
    ok.sort(key=lambda r: float(r["metric"]), reverse=(mode == "max"))
    return ok + bad


def load_results(*paths: str | Path, mode: str = "min") -> list[dict]:
    """Read and merge one or more ``results.jsonl`` files (or sweep directories), sorted best
    first (failed trials last)."""
    rows: list[dict] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            p = p / "results.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return _sort_results(rows, mode)


def _extract_metric(result: Any, metric_key: str | None) -> tuple[float, dict]:
    if isinstance(result, Mapping):
        key = metric_key or "metric"
        try:
            value = get_by_path(result, key)
        except KeyError as e:
            raise KeyError(f"train_fn result has no metric {key!r}; keys: {sorted(result)}") from e
        extra = {k: v for k, v in flatten(result).items()
                 if isinstance(v, (int, float, str, bool)) or v is None}
        return float(value), extra
    return float(result), {}


def run_sweep(train_fn: Callable[[dict], Any], base_cfg: Mapping[str, Any],
              overrides: Sequence[Mapping[str, Any]], out_dir: str | Path | None = None, *,
              mode: str = "min", metric_key: str | None = None,
              trial_dir_key: str | None = "train.out_dir", resume: bool = True,
              catch_errors: bool = True, indices: Sequence[int] | None = None) -> list[dict]:
    """Run ``train_fn(deep_merge(base_cfg, override))`` for every override.

    ``train_fn`` returns a float or a mapping holding ``metric_key`` (dotted path allowed,
    default ``"metric"``). When ``out_dir`` is set each trial gets ``out_dir/trial_XXX`` written
    at ``trial_dir_key``, results are appended to ``out_dir/results.jsonl`` and (``resume``)
    overrides already recorded as ``ok`` are not re-run. Failed trials are recorded
    (``status: failed`` + error) unless ``catch_errors=False``. ``indices`` gives the global
    trial numbers (default ``0..len-1``; used with :func:`shard` so trial directories stay unique
    across machines). Returns results sorted best first.
    """
    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'")
    indices = list(indices) if indices is not None else list(range(len(overrides)))
    if len(indices) != len(overrides):
        raise ValueError("indices must match overrides in length")
    out = Path(out_dir) if out_dir is not None else None
    done: dict[str, dict] = {}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if resume:
            for r in load_results(out, mode=mode):
                if r.get("status") == "ok":
                    done[_key(unflatten(r.get("overrides") or {}))] = r
    results: list[dict] = []
    for i, ov in zip(indices, overrides):
        k = _key(ov)
        if k in done:
            results.append(done[k])
            continue
        cfg = deep_merge(base_cfg, ov)
        trial_dir = None
        if out is not None:
            trial_dir = out / f"trial_{i:03d}"
            if trial_dir_key:
                set_by_path(cfg, trial_dir_key, str(trial_dir))
        rec: dict[str, Any] = {"trial": i, "overrides": flatten(ov), "status": "ok",
                               "metric": None}
        if trial_dir is not None:
            rec["trial_dir"] = str(trial_dir)
        t0 = time.perf_counter()
        try:
            value, extra = _extract_metric(train_fn(cfg), metric_key)
            rec["metric"] = value
            if extra:
                rec["result"] = extra
            if not math.isfinite(value):
                rec["status"] = "failed"
                rec["error"] = f"non-finite metric {value}"
        except Exception as e:
            if not catch_errors:
                raise
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc(limit=5)
        rec["time_s"] = round(time.perf_counter() - t0, 3)
        results.append(rec)
        if out is not None:
            with open(out / "results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
    return _sort_results(results, mode)


# ───────────────────────────────────────────────────────────────────────────── optuna

def suggest_from_space(trial: Any, space: Mapping[str, Any]) -> dict:
    """Translate a space into ``trial.suggest_*`` calls (duck-typed Optuna ``Trial``) → nested
    override dict. Lists become ``suggest_categorical`` over indices (Optuna only accepts
    primitive categorical values)."""
    flat: dict[str, Any] = {}
    for path, spec in space.items():
        if _is_dist(spec):
            kind, arg = next(iter(spec.items()))
            if kind == "choice":
                spec = list(arg)
            elif kind == "int":
                a, b = _range(path, arg, kind)
                flat[path] = int(trial.suggest_int(path, int(a), int(b)))
                continue
            else:
                a, b = _range(path, arg, kind)
                flat[path] = float(trial.suggest_float(path, a, b, log=(kind == "log_uniform")))
                continue
        if isinstance(spec, (list, tuple)):
            if not spec:
                raise ValueError(f"{path}: empty candidate list")
            idx = trial.suggest_categorical(path, list(range(len(spec))))
            flat[path] = copy.deepcopy(spec[int(idx)])
        else:
            flat[path] = copy.deepcopy(spec)
    return unflatten(flat)


def run_optuna(train_fn: Callable[[dict], Any], base_cfg: Mapping[str, Any],
               space: Mapping[str, Any], n_trials: int, out_dir: str | Path | None = None, *,
               mode: str = "min", metric_key: str | None = None, seed: int = 0,
               study_name: str | None = None, storage: str | None = None,
               trial_dir_key: str | None = "train.out_dir") -> Any:
    """Optuna TPE search (requires ``pip install optuna``). ``storage`` (e.g. an SQLite or
    PostgreSQL URL reachable over Tailscale) lets several machines share one study.
    Returns the ``optuna.Study``."""
    try:
        import optuna  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("run_optuna needs optuna: pip install optuna "
                          "(or use expand_grid / sample_random + run_sweep)") from e
    out = Path(out_dir) if out_dir is not None else None
    study = optuna.create_study(direction="minimize" if mode == "min" else "maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                study_name=study_name, storage=storage,
                                load_if_exists=storage is not None)

    def objective(trial: Any) -> float:
        ov = suggest_from_space(trial, space)
        cfg = deep_merge(base_cfg, ov)
        if out is not None and trial_dir_key:
            set_by_path(cfg, trial_dir_key, str(out / f"optuna_{trial.number:04d}"))
        value, extra = _extract_metric(train_fn(cfg), metric_key)
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            rec = {"trial": trial.number, "overrides": flatten(ov), "status": "ok",
                   "metric": value, "result": extra}
            with open(out / "results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        return value

    study.optimize(objective, n_trials=n_trials)
    return study


# ───────────────────────────────────────────────────────────────────────────── CLI

def _load_callable(spec: str) -> Callable[[dict], Any]:
    if ":" not in spec:
        raise ValueError(f"--fn must be 'package.module:function', got {spec!r}")
    mod, fn = spec.split(":", 1)
    obj = getattr(importlib.import_module(mod), fn)
    if not callable(obj):
        raise TypeError(f"{spec} is not callable")
    return obj


def main(argv: Sequence[str] | None = None) -> list[dict]:
    """CLI entry point (see module docstring). The space YAML holds ``space:`` plus optional
    ``mode: grid|random``, ``n`` and ``seed``."""
    import argparse

    import yaml

    ap = argparse.ArgumentParser(description="robot_skin hyper-parameter sweep")
    ap.add_argument("--fn", required=True, help="train function 'module:callable' (cfg -> metric)")
    ap.add_argument("--base", help="base config YAML (stage config)")
    ap.add_argument("--space", required=True, help="YAML with `space:` (+ mode, n, seed)")
    ap.add_argument("--out", required=True, help="sweep output directory")
    ap.add_argument("--metric", default=None, help="metric key in the result dict (dotted ok)")
    ap.add_argument("--direction", choices=("min", "max"), default="min")
    ap.add_argument("--mode", choices=("grid", "random"), default=None)
    ap.add_argument("--n", type=int, default=None, help="random trials")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th trial starting at i")
    ap.add_argument("--hardware", default=None, help="hardware profile applied to every trial")
    args = ap.parse_args(argv)

    spec = yaml.safe_load(Path(args.space).read_text()) or {}
    space = spec.get("space", spec)
    mode = args.mode or spec.get("mode", "grid")
    seed = args.seed if args.seed is not None else int(spec.get("seed", 0))
    base = yaml.safe_load(Path(args.base).read_text()) if args.base else {}
    base = base or {}
    if args.hardware:
        from .hardware import apply_hw_profile, apply_profile_env, load_hw_profile

        profile = load_hw_profile(args.hardware)
        apply_profile_env(profile)  # e.g. PYTORCH_CUDA_ALLOC_CONF — before any CUDA init
        base = apply_hw_profile(base, profile)
    if mode == "grid":
        trials = expand_grid(space)
    else:
        n = args.n if args.n is not None else int(spec.get("n", 10))
        trials = sample_random(space, n, seed)
    try:
        i, n_sh = (int(x) for x in args.shard.split("/"))
    except ValueError as e:
        raise SystemExit(f"--shard must look like 0/2, got {args.shard!r}") from e
    picked = shard(list(enumerate(trials)), i, n_sh)
    fn = _load_callable(args.fn)
    ranked = run_sweep(fn, base, [ov for _, ov in picked], args.out, mode=args.direction,
                       metric_key=args.metric, indices=[k for k, _ in picked])
    for r in ranked[:10]:
        print(json.dumps({k: r.get(k) for k in ("trial", "metric", "status", "overrides")},
                         default=str))
    return ranked


if __name__ == "__main__":  # pragma: no cover
    main()
