"""Language instructions for D2 task episodes: template slots, rendering, seeded sampling.

Every D2 episode (one task repetition) carries one natural-language instruction. Instructions are
rendered from the task catalog (``protocols/d2_task.yaml``, ``tasks[*].templates``) with
``str.format``-style slots (``"pick up the {object} and place it on the {target}"``); the
operator may override the rendered text per episode (``--instruction`` →
``events.jsonl`` ``instruction`` event and ``manifest.task.instruction``).

Several paraphrase templates per task are sampled at random (seeded) so the language encoder does
not overfit to one phrasing; the template index is stored with the episode so evaluation can be
split by phrasing. Slot values written with underscores in YAML (``red_cup``) are rendered as
words (``red cup``). This module has no dependencies beyond the standard library.
"""
from __future__ import annotations

import random
import re
import string
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "check_template", "humanize", "normalize_instruction", "render_instruction", "sample_instruction",
    "template_slots",
]

_WS = re.compile(r"\s+")


def template_slots(template: str) -> tuple[str, ...]:
    """Slot names used by ``template`` in order of first appearance (``"{object}"`` → ``object``).

    Raises ``ValueError`` for positional (``{}``/``{0}``) or attribute/index slots, which the
    catalog does not allow (every slot must be a plain name filled from the episode).
    """
    names: list[str] = []
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as e:  # unbalanced braces
        raise ValueError(f"bad instruction template {template!r}: {e}") from None
    for _, field, _, _ in parsed:
        if field is None:
            continue
        if not field or not field.isidentifier() or field.isdigit():
            raise ValueError(f"template {template!r}: slot {{{field}}} must be a plain name")
        if field not in names:
            names.append(field)
    return tuple(names)


def check_template(template: str, available: Iterable[str]) -> None:
    """Raise ``ValueError`` if ``template`` uses a slot that is not in ``available``."""
    avail = set(available)
    missing = [s for s in template_slots(template) if s not in avail]
    if missing:
        raise ValueError(f"template {template!r} uses unknown slots {missing} (available: {sorted(avail)})")


def humanize(value: Any) -> str:
    """Catalog slot value → words: ``"red_cup"`` → ``"red cup"`` (numbers are kept as is)."""
    return _WS.sub(" ", str(value).replace("_", " ")).strip()


def normalize_instruction(text: str) -> str:
    """Collapse whitespace and strip; keeps case and punctuation (encoders lower-case themselves)."""
    out = _WS.sub(" ", str(text)).strip()
    if not out:
        raise ValueError("instruction is empty")
    return out


def render_instruction(template: str, slots: Mapping[str, Any]) -> str:
    """Fill ``template`` from ``slots`` (values humanized); missing slots raise ``KeyError``."""
    names = template_slots(template)
    missing = [s for s in names if s not in slots or slots[s] is None]
    if missing:
        raise KeyError(f"template {template!r} needs slots {missing}")
    return normalize_instruction(template.format(**{s: humanize(slots[s]) for s in names}))


def sample_instruction(templates: Sequence[str], slots: Mapping[str, Any],
                       rng: random.Random | None = None) -> tuple[int, str]:
    """Pick one template (uniformly, ``rng`` for reproducibility) and render it.

    Returns ``(template_index, text)``.
    """
    if not templates:
        raise ValueError("no instruction templates")
    i = (rng or random.Random(0)).randrange(len(templates))
    return i, render_instruction(templates[i], slots)
