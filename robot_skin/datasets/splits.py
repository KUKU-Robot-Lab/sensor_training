"""Leakage-safe train / val / test splits of processed episodes.

Episodes are grouped by ``by`` and whole groups are assigned to splits, so a group never crosses
splits (e.g. ``by="subject"``: no subject appears in both train and test — the generalisation
question for a glove worn by new people). Group keys, read from ``episode.json`` only:

    subject → meta.subject          session → meta.source_session (else episode_id)
    object  → meta.task.object      task    → meta.task.task_id
    dataset / kind / episode_id     (also allowed; a tuple ``by`` combines keys)

Episodes without the key (e.g. D1 ``motion`` episodes under ``by="object"``) form one group per
episode — they cannot leak an object. ``holdout`` forces matching episodes into a split before the
random assignment: ``{"subject": ["S07"]}`` (→ test) or ``{"test": {"task": ["pour"]}, "val":
{"subject": ["S03"]}}``. A holdout overrides the grouping by design: with ``by="subject"`` and a
held-out task, that subject's other episodes may still land in train.

The assignment is deterministic in ``seed`` and independent of the input order: groups are sorted,
shuffled with ``numpy.random.default_rng(seed)`` and added greedily to test, then val, whenever that
brings the split's episode count closer to ``frac · n`` (each requested split gets ≥ 1 group while
train keeps ≥ 1). A split's first group is the first shuffled group with ≤ ``2 · frac · n`` episodes
(else the smallest), so one dominant group (e.g. a subject with most of the recordings) stays in
train instead of swallowing val/test.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .episode import EPISODE_JSON, Episode

__all__ = ["SPLITS", "GROUP_KEYS", "SPLITS_FORMAT", "episode_group", "make_splits", "check_splits",
           "save_splits", "load_splits"]

SPLITS = ("train", "val", "test")
GROUP_KEYS = ("subject", "session", "object", "task", "dataset", "kind", "episode_id")
SPLITS_FORMAT = 1


def _meta(ep: Episode | str | Path | Mapping) -> dict:
    if isinstance(ep, Episode):
        from dataclasses import asdict

        return asdict(ep.meta)
    if isinstance(ep, Mapping):
        return dict(ep)
    return json.loads((Path(ep) / EPISODE_JSON).read_text())


def _field(meta: Mapping, key: str) -> str | None:
    task = meta.get("task") or {}
    if key == "subject":
        v = meta.get("subject")
    elif key == "session":
        v = meta.get("source_session") or meta.get("episode_id")
    elif key == "object":
        v = task.get("object")
    elif key == "task":
        v = task.get("task_id")
    elif key in ("dataset", "kind", "episode_id"):
        v = meta.get(key)
    else:
        raise ValueError(f"unknown group key {key!r}; valid: {GROUP_KEYS}")
    return None if v in (None, "") else str(v)


def episode_group(ep: Episode | str | Path | Mapping, by: str | Sequence[str] = "subject") -> str:
    """Group key of one episode (see module docstring); missing → ``episode:<episode_id>``."""
    meta = _meta(ep)
    keys = (by,) if isinstance(by, str) else tuple(by)
    vals = [_field(meta, k) for k in keys]
    if any(v is None for v in vals):
        return f"episode:{meta.get('episode_id')}"
    return "|".join(vals)


def _holdout_rules(holdout: Mapping | None) -> list[tuple[str, dict]]:
    if not holdout:
        return []
    if set(holdout) <= set(SPLITS):
        rules = [(s, dict(r)) for s, r in holdout.items()]
    else:
        rules = [("test", dict(holdout))]
    for _, r in rules:
        for k, v in r.items():
            if k not in GROUP_KEYS:
                raise ValueError(f"unknown holdout key {k!r}; valid: {GROUP_KEYS}")
            if isinstance(v, (str, bytes)):
                r[k] = [v]
    return rules


def _matches(meta: Mapping, rule: Mapping[str, Sequence]) -> bool:
    return any(_field(meta, k) in {str(x) for x in vals} for k, vals in rule.items())


def make_splits(episode_dirs: Iterable[str | Path | Episode], by: str | Sequence[str] = "subject",
                val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0,
                holdout: Mapping | None = None) -> dict[str, list[str]]:
    """``{"train": [...], "val": [...], "test": [...]}`` of episode dir strings (sorted), with no
    ``by``-group in more than one split (holdout aside)."""
    if not (0.0 <= val_frac < 1.0 and 0.0 <= test_frac < 1.0 and val_frac + test_frac < 1.0):
        raise ValueError(f"need 0 ≤ val_frac, test_frac and val_frac + test_frac < 1, got {val_frac}, {test_frac}")
    items = []
    for e in episode_dirs:
        if isinstance(e, Episode):
            if e.root is None:
                raise ValueError("in-memory episodes have no directory; save them first")
            items.append((str(e.root), _meta(e)))
        else:
            items.append((str(Path(e)), _meta(e)))
    items.sort(key=lambda x: x[0])
    if len({p for p, _ in items}) != len(items):
        raise ValueError("duplicate episode directories")
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    rest = []
    for p, m in items:
        for split, rule in _holdout_rules(holdout):
            if _matches(m, rule):
                out[split].append(p)
                break
        else:
            rest.append((p, m))
    groups: dict[str, list[str]] = {}
    for p, m in rest:
        groups.setdefault(episode_group(m, by), []).append(p)
    names = sorted(groups)
    order = [names[i] for i in np.random.default_rng(int(seed)).permutation(len(names))]
    n = len(rest)
    remaining = list(order)
    for split, frac in (("test", test_frac), ("val", val_frac)):
        if frac <= 0 or len(remaining) <= 1:
            continue
        target = frac * n
        # first group: the first (shuffled) one not larger than 2·target — a dominant group (one
        # subject with most of the episodes) must not be dumped into val/test — else the smallest
        first = next((g for g in remaining if len(groups[g]) <= max(2.0 * target, 1.0)),
                     min(remaining, key=lambda g: len(groups[g])))
        out[split] += groups[first]
        count = len(groups[first])
        remaining.remove(first)
        for g in list(remaining):
            if len(remaining) <= 1:
                break
            size = len(groups[g])
            if abs(count + size - target) < abs(count - target):
                out[split] += groups[g]
                count += size
                remaining.remove(g)
    for g in remaining:
        out["train"] += groups[g]
    return {s: sorted(v) for s, v in out.items()}


def check_splits(splits: Mapping[str, Sequence], by: str | Sequence[str] = "subject",
                 *, ignore: Iterable[str | Path] = ()) -> None:
    """Raise ``ValueError`` if an episode or a ``by``-group occurs in more than one split
    (``ignore``: episodes allowed to cross, e.g. deliberate holdouts)."""
    skip = {str(Path(p)) for p in ignore}
    seen_ep: dict[str, str] = {}
    seen_g: dict[str, str] = {}
    for split, eps in splits.items():
        for p in eps:
            p = str(Path(p))
            if p in seen_ep:
                raise ValueError(f"episode {p} is in both {seen_ep[p]!r} and {split!r}")
            seen_ep[p] = split
            if p in skip:
                continue
            g = episode_group(p, by)
            if seen_g.setdefault(g, split) != split:
                raise ValueError(f"group {g!r} ({by}) is in both {seen_g[g]!r} and {split!r}")


def save_splits(splits: Mapping[str, Sequence], path: str | Path, *, root: str | Path | None = None,
                meta: Mapping[str, Any] | None = None) -> Path:
    """Write ``splits.json``. With ``root`` (e.g. the processed root) episode paths under it are
    stored relative to it, so the split file survives moving the dataset."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    base = None if root is None else Path(root).resolve()

    def rel(x) -> str:
        x = Path(x)
        if base is not None:
            try:
                return Path(x).resolve().relative_to(base).as_posix()
            except ValueError:
                pass
        return str(x)

    d = {"format": SPLITS_FORMAT, "relative": base is not None,
         **{s: [rel(x) for x in splits.get(s, [])] for s in SPLITS}, "meta": dict(meta or {})}
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    return p


def load_splits(path: str | Path, *, root: str | Path | None = None) -> dict[str, list[Path]]:
    """Read ``splits.json`` → ``{split: [Path]}``. Relative entries resolve against ``root``
    (default: the directory of ``splits.json``)."""
    p = Path(path)
    d = json.loads(p.read_text())
    if int(d.get("format", SPLITS_FORMAT)) > SPLITS_FORMAT:
        raise ValueError(f"splits format {d['format']} is newer than supported {SPLITS_FORMAT}")
    base = Path(root) if root is not None else p.parent
    out = {}
    for s in SPLITS:
        out[s] = [Path(x) if Path(x).is_absolute() else base / x for x in d.get(s, [])]
    return out
