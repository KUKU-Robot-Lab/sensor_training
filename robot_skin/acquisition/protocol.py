"""Recording protocols (D1 motion / D2 task): YAML schema, validation, session planning, scripts.

A protocol (``protocols/<name>.yaml``) describes WHAT the subject does while the recorder runs:

- **D1 ``d1_motion``** — an ordered list of *blocks* (static baseline, IMU calibration pose,
  per-finger flexion, open/close slow/fast, wrist rotations, free motion, self-touch set, final
  baseline, 3-tap sync at start/end). Each block has a duration (or ``cycle_s × repetitions``),
  an expected contact (``none`` / ``self`` / ``object`` / ``any``), segment labels, a speed tag
  and an operator prompt. ``for_each`` expands a block into one step per finger / axis.
- **D2 ``d2_task``** — a task *catalog*: tasks with objects, targets, instruction templates,
  phases from a shared vocabulary (reach → grasp → manipulate → release → retreat, each with a
  contact expectation), repetitions and success criteria. **One episode = one task repetition =
  one session directory**; every episode starts with its own static baseline and 3-tap sync.

:func:`plan_session` turns a protocol into a :class:`SessionPlan` (seeded: object/target/template
choice and episode order), :func:`format_script` prints the operator script, and
:meth:`EpisodePlan.timeline` gives nominal step times (used by ``--dry-run`` manifests and by the
synthetic ``--fake`` scene). Steps become ``phase_start``/``phase_end`` events whose ``value``
carries the contact expectation and segment labels; the recorder turns them into manifest
segments (``no_contact`` / ``self_touch`` / ``calibration`` / ``sync``; D2 episodes add ``task``).

Grounding: the session layout (per-stream files + manifest + event log, ego + third-person
cameras) follows ActionSense (DelPreto et al., NeurIPS 2022 Datasets & Benchmarks); the D2 task
set includes insertion as in VTLA (arXiv:2505.09577) and wiping as in OSMO (arXiv:2512.08920).
See ``docs/DATA_ACQUISITION.md`` for the operator procedure and ``docs/REFERENCES.md``.
"""
from __future__ import annotations

import dataclasses
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .instructions import check_template, normalize_instruction, render_instruction

__all__ = [
    "ADVANCE_MODES", "CONTACT_EXPECTATIONS", "DEFAULT_CONTACT_LABELS", "EpisodePlan", "KO_NAMES",
    "PROTOCOL_DIR", "Protocol", "ProtocolError", "SPEEDS", "STEP_KINDS", "SessionPlan", "Step",
    "SyncSpec", "TASK_PHASES", "TaskSpec", "TimedStep", "Timing", "format_script", "list_protocols",
    "load_protocol", "make_session_id", "plan_session", "protocol_from_dict", "scale_steps", "scale_timing",
]

PROTOCOL_DIR = Path(__file__).resolve().parent / "protocols"

#: expected contact during a step
CONTACT_EXPECTATIONS = ("none", "self", "object", "any")
#: default manifest segment labels per contact expectation (override with ``labels:``)
DEFAULT_CONTACT_LABELS: dict[str, tuple[str, ...]] = {
    "none": ("no_contact",), "self": ("self_touch",), "object": (), "any": (),
}
STEP_KINDS = ("static", "calibration", "sync", "motion", "self_touch", "task_phase")
SPEEDS = ("static", "slow", "medium", "fast", "mixed")
ADVANCE_MODES = ("timed", "manual")
#: canonical D2 phase vocabulary (a protocol may define more)
TASK_PHASES = ("reach", "grasp", "manipulate", "release", "retreat")
DATASETS = ("motion", "task", "other")
KINDS = ("glove", "robot", "bench", "any")

#: Korean display names for prompt placeholders ``{<key>_ko}``
KO_NAMES = {
    "thumb": "엄지", "index": "검지", "middle": "중지", "ring": "약지", "pinky": "새끼",
    "pronation_supination": "회내/회외(손바닥 뒤집기)", "flexion_extension": "굴곡/신전(손목 위아래)",
    "radial_ulnar": "요측/척측 편위(손목 좌우)",
    # D2 grasp types (air_grasp blocks)
    "power": "파워 그립(원통형 물체를 손 전체로 감싸 쥐는)",
    "precision": "정밀 집기(엄지·검지 끝으로 작은 물체를 집는)",
    "lateral": "옆 집기(열쇠를 쥐듯 엄지 끝을 검지 옆면 쪽으로 — 닿기 전까지만)",
    "tripod": "세 손가락 집기(엄지·검지·중지 끝으로 공을 집는)",
}

_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_BLOCK_KEYS = {"id", "kind", "duration_s", "cycle_s", "repetitions", "contact", "labels", "speed",
               "advance", "motion", "pose", "for_each", "prompt", "prompt_en", "scalable", "amplitude"}
_PHASE_KEYS = {"contact", "labels", "duration_s", "advance", "prompt", "prompt_en", "boundary"}
_TASK_KEYS = {"id", "description", "objects", "targets", "templates", "phases", "durations", "repetitions",
              "success", "slots", "grasp", "manipulate"}
_TOP_KEYS = {"name", "version", "dataset", "kind", "description", "cameras", "rates", "timing", "sync", "language",
             "blocks", "phases", "tasks", "episode"}
_EPISODE_KEYS = {"pre", "post"}


class ProtocolError(ValueError):
    """Invalid protocol YAML / planning request (message names the offending path)."""


# ── small value types ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SyncSpec:
    """3-tap sync block: ``taps`` fingertip taps with unequal ``intervals_s`` (short–long rhythm,
    so cross-correlation has a unique peak), ``lead_s``/``tail_s`` of stillness around them."""

    taps: int = 3
    intervals_s: tuple[float, ...] = (0.6, 1.2)
    lead_s: float = 1.0
    tail_s: float = 1.5
    finger: str = "index"

    def __post_init__(self) -> None:
        if self.taps < 1:
            raise ProtocolError("sync.taps must be ≥ 1")
        if len(self.intervals_s) != self.taps - 1 or any(i <= 0 for i in self.intervals_s):
            raise ProtocolError(f"sync.intervals_s needs {self.taps - 1} positive values, got {self.intervals_s}")
        if self.lead_s < 0 or self.tail_s < 0:
            raise ProtocolError("sync.lead_s / tail_s must be ≥ 0")

    @property
    def duration_s(self) -> float:
        return float(self.lead_s + sum(self.intervals_s) + self.tail_s)

    def tap_offsets(self) -> list[float]:
        """Tap times relative to the sync step start."""
        out, t = [], float(self.lead_s)
        for k in range(self.taps):
            out.append(t)
            if k < self.taps - 1:
                t += float(self.intervals_s[k])
        return out

    def to_dict(self) -> dict:
        return {"taps": self.taps, "intervals_s": list(self.intervals_s), "lead_s": self.lead_s,
                "tail_s": self.tail_s, "finger": self.finger}


@dataclass(frozen=True)
class Timing:
    lead_in_s: float = 1.0      # recording before the first step
    transition_s: float = 1.0   # nominal gap between steps (real runs: operator-paced)
    lead_out_s: float = 1.0     # recording after the last step
    min_step_s: float = 0.5     # floor for time-scaled steps

    def __post_init__(self) -> None:
        for k, v in dataclasses.asdict(self).items():
            if v < 0:
                raise ProtocolError(f"timing.{k} must be ≥ 0")


@dataclass(frozen=True)
class Step:
    """One phase of a session (becomes ``phase_start``/``phase_end`` events named ``id``)."""

    id: str
    block: str
    kind: str
    duration_s: float
    contact: str = "none"
    labels: tuple[str, ...] = ()
    advance: str = "timed"
    speed: str | None = None
    prompt: str = ""
    prompt_en: str = ""
    motion: Mapping[str, Any] = field(default_factory=dict)
    scalable: bool = True
    boundary: str = ""          # manual phases: the physical moment that starts the phase (operator cue)

    def event_value(self) -> dict:
        """``value`` of the ``phase_start`` event (what the recorder / preprocessing read)."""
        v = {"block": self.block, "kind": self.kind, "contact": self.contact, "labels": list(self.labels),
             "advance": self.advance, "nominal_s": round(float(self.duration_s), 6)}
        if self.speed:
            v["speed"] = self.speed
        if self.motion.get("type"):
            v["motion"] = str(self.motion["type"])
        for k in ("finger", "axis", "grasp"):
            if k in self.motion:
                v[k] = self.motion[k]
        return v

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["labels"] = list(self.labels)
        d["motion"] = dict(self.motion)
        return d


@dataclass(frozen=True)
class TimedStep:
    step: Step
    t0: float
    t1: float


@dataclass(frozen=True)
class TaskSpec:
    """One D2 catalog task."""

    id: str
    objects: tuple[str, ...]
    templates: tuple[str, ...]
    phases: tuple[str, ...] = TASK_PHASES
    targets: tuple[str, ...] = ()
    durations: Mapping[str, float] = field(default_factory=dict)
    repetitions: int = 5
    success: str = ""
    description: str = ""
    slots: Mapping[str, tuple] = field(default_factory=dict)
    grasp: str = "power"
    manipulate: str = "transport"


@dataclass
class EpisodePlan:
    """Steps of one session directory (D1: the whole protocol; D2: one task repetition)."""

    index: int
    steps: list[Step]
    task: dict | None = None      # manifest.task (D2): task_id, instruction, object, target, ...

    def timeline(self, timing: Timing) -> list[TimedStep]:
        """Nominal step times on the session clock (t = 0 at recorder start)."""
        out, t = [], float(timing.lead_in_s)
        for i, s in enumerate(self.steps):
            if i:
                t += float(timing.transition_s)
            out.append(TimedStep(s, t, t + float(s.duration_s)))
            t += float(s.duration_s)
        return out

    def duration_s(self, timing: Timing) -> float:
        tl = self.timeline(timing)
        return (tl[-1].t1 if tl else float(timing.lead_in_s)) + float(timing.lead_out_s)

    def to_dict(self, timing: Timing | None = None) -> dict:
        d = {"index": self.index, "task": self.task, "steps": [s.to_dict() for s in self.steps]}
        if timing is not None:
            d["timeline"] = [{"id": ts.step.id, "t0": round(ts.t0, 6), "t1": round(ts.t1, 6)}
                             for ts in self.timeline(timing)]
            d["duration_s"] = round(self.duration_s(timing), 6)
        return d


@dataclass
class Protocol:
    name: str
    dataset: str
    kind: str
    version: int = 1
    description: str = ""
    cameras: tuple[str, ...] = ("ego", "third")
    rates: dict[str, float] = field(default_factory=dict)
    timing: Timing = field(default_factory=Timing)
    sync: SyncSpec = field(default_factory=SyncSpec)
    blocks: list[dict] = field(default_factory=list)            # D1 (validated raw block dicts)
    phases: dict[str, dict] = field(default_factory=dict)       # D2 phase vocabulary
    tasks: dict[str, TaskSpec] = field(default_factory=dict)    # D2 catalog
    episode_pre: list[dict] = field(default_factory=list)
    episode_post: list[dict] = field(default_factory=list)
    language: str = "en"
    source: str = ""

    @property
    def is_task(self) -> bool:
        return bool(self.tasks)

    def expand_blocks(self, blocks: Sequence[dict] | None = None) -> list[Step]:
        """Expand (``for_each``) block dicts into steps (default: the D1 ``blocks``)."""
        steps: list[Step] = []
        for i, b in enumerate(self.blocks if blocks is None else blocks):
            steps.extend(_expand_block(b, self, where=f"blocks[{i}]"))
        _check_unique(steps, self.name)
        return steps

    def task_steps(self, task: TaskSpec, slots: Mapping[str, Any]) -> list[Step]:
        """Task-phase steps of one D2 episode (contact / labels from the phase vocabulary)."""
        out = []
        for ph in task.phases:
            p = self.phases[ph]
            contact = p.get("contact", "any")
            labels = tuple(p["labels"]) if "labels" in p else DEFAULT_CONTACT_LABELS[contact]
            out.append(Step(
                id=ph, block=task.id, kind="task_phase",
                duration_s=float(task.durations.get(ph, p.get("duration_s", 1.0))),
                contact=contact, labels=labels, advance=p.get("advance", "manual"),
                prompt=str(p.get("prompt", "")), prompt_en=str(p.get("prompt_en", "")),
                motion={"type": ph, "task": task.id, "grasp": task.grasp, "manipulate": task.manipulate,
                        **{k: v for k, v in slots.items() if v is not None}},
                boundary=str(p.get("boundary", ""))))
        return out


# ── loading & validation ─────────────────────────────────────────────────────
def list_protocols() -> list[str]:
    return sorted(p.stem for p in PROTOCOL_DIR.glob("*.yaml"))


def load_protocol(name_or_path: str | Path) -> Protocol:
    """Load a built-in protocol by name (``"d1_motion"``) or any YAML path; validates the schema."""
    p = Path(name_or_path)
    if p.suffix not in (".yaml", ".yml"):
        p = PROTOCOL_DIR / f"{name_or_path}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"protocol not found: {p} (built-ins: {list_protocols()})")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    return protocol_from_dict(d, source=str(p))


def _req(d: Mapping, key: str, where: str):
    if key not in d:
        raise ProtocolError(f"{where}: missing required key {key!r}")
    return d[key]


def _pos_float(v, where: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ProtocolError(f"{where}: expected a number, got {v!r}") from None
    if not f > 0:
        raise ProtocolError(f"{where}: must be > 0, got {v!r}")
    return f


def _pos_int(v, where: str) -> int:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ProtocolError(f"{where}: expected an integer, got {v!r}") from None
    if not f.is_integer() or f < 1:
        raise ProtocolError(f"{where}: must be an integer ≥ 1, got {v!r}")
    return int(f)


def _str_list(v, where: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if v is None:
        v = []
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, (str, int, float)) for x in v):
        raise ProtocolError(f"{where}: expected a list of names, got {v!r}")
    out = tuple(str(x) for x in v)
    if not allow_empty and not out:
        raise ProtocolError(f"{where}: must not be empty")
    if len(set(out)) != len(out):
        raise ProtocolError(f"{where}: duplicate entries in {list(out)}")
    return out


def _validate_block(b: Any, where: str, *, allow_task_phase: bool = False) -> dict:
    if not isinstance(b, Mapping):
        raise ProtocolError(f"{where}: block must be a mapping, got {type(b).__name__}")
    unknown = set(b) - _BLOCK_KEYS
    if unknown:
        raise ProtocolError(f"{where}: unknown keys {sorted(unknown)} (allowed: {sorted(_BLOCK_KEYS)})")
    bid = str(_req(b, "id", where))
    if not _ID.match(bid):
        raise ProtocolError(f"{where}: id {bid!r} must be lower_snake_case")
    kind = str(_req(b, "kind", where))
    kinds = STEP_KINDS if allow_task_phase else tuple(k for k in STEP_KINDS if k != "task_phase")
    if kind not in kinds:
        raise ProtocolError(f"{where}: kind {kind!r} not in {kinds}")
    if kind != "sync":
        if "duration_s" in b:
            _pos_float(b["duration_s"], f"{where}.duration_s")
        elif "cycle_s" in b and "repetitions" in b:
            _pos_float(b["cycle_s"], f"{where}.cycle_s")
            _pos_int(b["repetitions"], f"{where}.repetitions")
        else:
            raise ProtocolError(f"{where}: needs duration_s or cycle_s + repetitions")
    contact = b.get("contact", "any" if kind == "sync" else "none")
    if contact not in CONTACT_EXPECTATIONS:
        raise ProtocolError(f"{where}.contact {contact!r} not in {CONTACT_EXPECTATIONS}")
    if "labels" in b:
        _str_list(b["labels"], f"{where}.labels")
    if b.get("speed") is not None and b["speed"] not in SPEEDS:
        raise ProtocolError(f"{where}.speed {b['speed']!r} not in {SPEEDS}")
    if b.get("advance", "timed") not in ADVANCE_MODES:
        raise ProtocolError(f"{where}.advance {b['advance']!r} not in {ADVANCE_MODES}")
    fe = b.get("for_each")
    if fe is not None:
        if not isinstance(fe, Mapping) or len(fe) != 1:
            raise ProtocolError(f"{where}.for_each must be a one-key mapping {{key: [values]}}")
        (k, vals), = fe.items()
        if not str(k).isidentifier():
            raise ProtocolError(f"{where}.for_each key {k!r} must be an identifier")
        _str_list(vals, f"{where}.for_each.{k}", allow_empty=False)
    return dict(b)


def _validate_task(t: Any, where: str, phases: Mapping[str, dict]) -> TaskSpec:
    if not isinstance(t, Mapping):
        raise ProtocolError(f"{where}: task must be a mapping")
    unknown = set(t) - _TASK_KEYS
    if unknown:
        raise ProtocolError(f"{where}: unknown keys {sorted(unknown)} (allowed: {sorted(_TASK_KEYS)})")
    tid = str(_req(t, "id", where))
    if not _ID.match(tid):
        raise ProtocolError(f"{where}: id {tid!r} must be lower_snake_case")
    objects = _str_list(_req(t, "objects", where), f"{where}.objects", allow_empty=False)
    targets = _str_list(t.get("targets"), f"{where}.targets")
    templates = _str_list(_req(t, "templates", where), f"{where}.templates", allow_empty=False)
    slots_raw = t.get("slots") or {}
    if not isinstance(slots_raw, Mapping):
        raise ProtocolError(f"{where}.slots must be a mapping {{slot: [values]}}")
    slots = {str(k): _str_list(v, f"{where}.slots.{k}", allow_empty=False) for k, v in slots_raw.items()}
    for k in slots:
        if k in ("object", "target"):
            raise ProtocolError(f"{where}.slots: {k!r} is reserved")
    avail = {"object", *slots} | ({"target"} if targets else set())
    for i, tpl in enumerate(templates):
        try:
            check_template(tpl, avail)
        except ValueError as e:
            raise ProtocolError(f"{where}.templates[{i}]: {e}") from None
    ph = _str_list(t.get("phases", list(phases)), f"{where}.phases", allow_empty=False)
    bad = [p for p in ph if p not in phases]
    if bad:
        raise ProtocolError(f"{where}.phases: unknown phases {bad} (vocabulary: {list(phases)})")
    durations = t.get("durations") or {}
    if not isinstance(durations, Mapping):
        raise ProtocolError(f"{where}.durations must be a mapping phase → seconds")
    for k, v in durations.items():
        if k not in ph:
            raise ProtocolError(f"{where}.durations: {k!r} is not one of the task phases {list(ph)}")
        _pos_float(v, f"{where}.durations.{k}")
    reps = _pos_int(t.get("repetitions", 5), f"{where}.repetitions")
    return TaskSpec(id=tid, objects=objects, templates=templates, phases=ph, targets=targets,
                    durations={k: float(v) for k, v in durations.items()}, repetitions=reps,
                    success=str(t.get("success", "")), description=str(t.get("description", "")),
                    slots=slots, grasp=str(t.get("grasp", "power")),
                    manipulate=str(t.get("manipulate", "transport")))


def protocol_from_dict(d: Mapping, *, source: str = "<dict>") -> Protocol:
    """Validate a parsed protocol mapping and build a :class:`Protocol`."""
    if not isinstance(d, Mapping):
        raise ProtocolError(f"{source}: protocol must be a mapping")
    unknown = set(d) - _TOP_KEYS
    if unknown:
        raise ProtocolError(f"{source}: unknown keys {sorted(unknown)} (allowed: {sorted(_TOP_KEYS)})")
    name = str(_req(d, "name", source))
    dataset = str(_req(d, "dataset", source))
    if dataset not in DATASETS:
        raise ProtocolError(f"{source}: dataset {dataset!r} not in {DATASETS}")
    kind = str(d.get("kind", "any"))
    if kind not in KINDS:
        raise ProtocolError(f"{source}: kind {kind!r} not in {KINDS}")
    cams = _str_list(d.get("cameras", ["ego", "third"]), f"{source}.cameras")
    for c in cams:
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_-]*$", c):
            raise ProtocolError(f"{source}.cameras: bad camera name {c!r}")
    rates = {str(k): _pos_float(v, f"{source}.rates.{k}") for k, v in (d.get("rates") or {}).items()}
    try:
        timing = Timing(**{k: float(v) for k, v in (d.get("timing") or {}).items()})
    except TypeError as e:
        raise ProtocolError(f"{source}.timing: {e}") from None
    sd = dict(d.get("sync") or {})
    if "intervals_s" in sd:
        sd["intervals_s"] = tuple(float(x) for x in sd["intervals_s"])
    try:
        sync = SyncSpec(**sd)
    except TypeError as e:
        raise ProtocolError(f"{source}.sync: {e}") from None
    proto = Protocol(name=name, dataset=dataset, kind=kind, version=int(d.get("version", 1)),
                     description=str(d.get("description", "")).strip(), cameras=cams, rates=rates,
                     timing=timing, sync=sync, language=str(d.get("language", "en")), source=source)

    has_blocks, has_tasks = "blocks" in d, "tasks" in d
    if has_blocks == has_tasks:
        raise ProtocolError(f"{source}: define exactly one of `blocks` (motion protocol) or `tasks` (task catalog)")
    if has_blocks:
        blocks = d["blocks"]
        if not isinstance(blocks, list) or not blocks:
            raise ProtocolError(f"{source}.blocks must be a non-empty list")
        proto.blocks = [_validate_block(b, f"{source}.blocks[{i}]") for i, b in enumerate(blocks)]
        proto.expand_blocks()   # checks prompts / unique step ids
        return proto

    phases_raw = d.get("phases") or {}
    if not isinstance(phases_raw, Mapping) or not phases_raw:
        raise ProtocolError(f"{source}: a task catalog needs a `phases` vocabulary")
    for ph, p in phases_raw.items():
        w = f"{source}.phases.{ph}"
        if not _ID.match(str(ph)):
            raise ProtocolError(f"{w}: phase name must be lower_snake_case")
        if p is not None and not isinstance(p, Mapping):
            raise ProtocolError(f"{w}: phase must be a mapping")
        p = dict(p or {})
        unknown = set(p) - _PHASE_KEYS
        if unknown:
            raise ProtocolError(f"{w}: unknown keys {sorted(unknown)}")
        if p.get("contact", "any") not in CONTACT_EXPECTATIONS:
            raise ProtocolError(f"{w}.contact {p.get('contact')!r} not in {CONTACT_EXPECTATIONS}")
        if "labels" in p:
            p["labels"] = list(_str_list(p["labels"], f"{w}.labels"))
        if "duration_s" in p:
            _pos_float(p["duration_s"], f"{w}.duration_s")
        if p.get("advance", "manual") not in ADVANCE_MODES:
            raise ProtocolError(f"{w}.advance not in {ADVANCE_MODES}")
        proto.phases[str(ph)] = p
    tasks = d["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ProtocolError(f"{source}.tasks must be a non-empty list")
    for i, t in enumerate(tasks):
        spec = _validate_task(t, f"{source}.tasks[{i}]", proto.phases)
        if spec.id in proto.tasks:
            raise ProtocolError(f"{source}.tasks[{i}]: duplicate task id {spec.id!r}")
        proto.tasks[spec.id] = spec
    ep = d.get("episode") or {}
    if not isinstance(ep, Mapping) or set(ep) - _EPISODE_KEYS:
        raise ProtocolError(f"{source}.episode must be a mapping with keys {sorted(_EPISODE_KEYS)}")
    proto.episode_pre = [_validate_block(b, f"{source}.episode.pre[{i}]") for i, b in enumerate(ep.get("pre") or [])]
    proto.episode_post = [_validate_block(b, f"{source}.episode.post[{i}]") for i, b in enumerate(ep.get("post") or [])]
    pre_post = proto.expand_blocks(proto.episode_pre + proto.episode_post)
    clash = {s.id for s in pre_post} & set(proto.phases)
    if clash:
        raise ProtocolError(f"{source}.episode: step ids {sorted(clash)} clash with phase names")
    return proto


def _format_prompt(text: str, ctx: Mapping[str, Any], where: str) -> str:
    try:
        return str(text).format(**ctx)
    except (KeyError, IndexError, ValueError) as e:
        raise ProtocolError(f"{where}: prompt {text!r} uses an unknown placeholder ({e})") from None


def _fmt_num(x: float) -> str:
    return f"{x:g}"


def _expand_block(b: Mapping, proto: Protocol, *, where: str) -> list[Step]:
    kind = b["kind"]
    if kind == "sync":
        dur = proto.sync.duration_s
        motion = {"type": "sync_taps", "tap_times": proto.sync.tap_offsets(), "finger": proto.sync.finger}
    else:
        if "duration_s" in b:
            dur = float(b["duration_s"])
        else:
            dur = float(b["cycle_s"]) * int(b["repetitions"])
        mtype = b.get("motion") or b.get("pose") or {"static": "rest", "calibration": "flat_hand"}.get(kind, kind)
        motion = {"type": str(mtype)}
        if "pose" in b:
            motion["pose"] = str(b["pose"])
        for k in ("cycle_s", "repetitions", "amplitude"):
            if k in b:
                motion[k] = b[k]
    contact = b.get("contact", "any" if kind == "sync" else "none")
    if "labels" in b:
        labels = tuple(str(x) for x in b["labels"])
    else:
        labels = ("sync",) if kind == "sync" else DEFAULT_CONTACT_LABELS[contact]
    items: list[tuple[str | None, str | None]] = [(None, None)]
    if b.get("for_each"):
        (key, vals), = b["for_each"].items()
        items = [(str(key), str(v)) for v in vals]
    steps = []
    for key, val in items:
        ctx = {"duration_s": _fmt_num(dur), "cycle_s": _fmt_num(float(b.get("cycle_s", dur))),
               "repetitions": int(b.get("repetitions", 1))}
        m = dict(motion)
        sid = b["id"]
        if key is not None:
            ctx[key] = val
            ctx[f"{key}_ko"] = KO_NAMES.get(val, val)
            m[key] = val
            sid = f"{b['id']}_{val}"
        steps.append(Step(
            id=sid, block=b["id"], kind=kind, duration_s=dur, contact=contact, labels=labels,
            advance=b.get("advance", "timed"), speed=b.get("speed", "static" if kind in ("static", "calibration") else None),
            prompt=_format_prompt(b.get("prompt", ""), ctx, f"{where}.prompt"),
            prompt_en=_format_prompt(b.get("prompt_en", ""), ctx, f"{where}.prompt_en"),
            motion=m, scalable=bool(b.get("scalable", kind != "sync"))))
    return steps


def _check_unique(steps: Sequence[Step], name: str) -> None:
    seen: set[str] = set()
    for s in steps:
        if s.id in seen:
            raise ProtocolError(f"{name}: duplicate step id {s.id!r} (rename the block or its for_each values)")
        seen.add(s.id)


# ── planning ─────────────────────────────────────────────────────────────────
def scale_steps(steps: Sequence[Step], scale: float, min_step_s: float = 0.5) -> list[Step]:
    """Shrink/stretch step durations by ``scale`` (quick synthetic runs, rehearsals).

    Sync steps (``scalable=False``) keep their duration; others become
    ``min(d, max(d·scale, floor))`` when shrinking, with ``floor = max(min_step_s, cycle_s)`` so at
    least one motion cycle survives, and ``d·scale`` when stretching.
    """
    if not scale > 0:
        raise ProtocolError(f"time scale must be > 0, got {scale}")
    out = []
    for s in steps:
        if not s.scalable or scale == 1.0:
            out.append(s)
            continue
        d = float(s.duration_s)
        if scale < 1.0:
            floor = max(float(min_step_s), float(s.motion.get("cycle_s", 0.0) or 0.0))
            nd = min(d, max(d * scale, floor))
        else:
            nd = d * scale
        out.append(dataclasses.replace(s, duration_s=nd))
    return out


def scale_timing(timing: Timing, scale: float, floor_s: float = 0.25) -> Timing:
    """Shrink lead-in / transitions / lead-out with ``scale < 1`` (each floored at
    ``min(original, floor_s)``); ``scale ≥ 1`` keeps the protocol's pauses."""
    if not scale > 0:
        raise ProtocolError(f"time scale must be > 0, got {scale}")
    if scale >= 1.0:
        return timing
    f = lambda v: max(v * scale, min(v, floor_s))  # noqa: E731
    return dataclasses.replace(timing, lead_in_s=f(timing.lead_in_s), transition_s=f(timing.transition_s),
                               lead_out_s=f(timing.lead_out_s))


@dataclass
class SessionPlan:
    """A planned recording: one or more episodes (= session directories)."""

    protocol: str
    version: int
    dataset: str
    kind: str
    seed: int
    timing: Timing
    episodes: list[EpisodePlan]
    cameras: tuple[str, ...] = ()
    rates: dict[str, float] = field(default_factory=dict)
    sync: SyncSpec = field(default_factory=SyncSpec)
    time_scale: float = 1.0

    @property
    def total_duration_s(self) -> float:
        return float(sum(e.duration_s(self.timing) for e in self.episodes))

    def to_dict(self) -> dict:
        return {"protocol": self.protocol, "version": self.version, "dataset": self.dataset, "kind": self.kind,
                "seed": self.seed, "time_scale": self.time_scale, "cameras": list(self.cameras),
                "rates": dict(self.rates), "timing": dataclasses.asdict(self.timing), "sync": self.sync.to_dict(),
                "total_duration_s": round(self.total_duration_s, 3),
                "episodes": [e.to_dict(self.timing) for e in self.episodes]}


def plan_session(protocol: Protocol | str | Path, *, seed: int = 0, tasks: Iterable[str] | None = None,
                 objects: Iterable[str] | None = None, n_episodes: int | None = None,
                 repetitions: int | None = None, time_scale: float = 1.0, instruction: str | None = None,
                 shuffle: bool = True) -> SessionPlan:
    """Plan a recording.

    Motion protocols → one episode with all blocks (expanded, optionally time-scaled). Task
    catalogs → one episode per (task, object, repetition): target, extra slots and instruction
    template are drawn with ``random.Random(seed)``; the order is shuffled (seeded) so tasks and
    objects interleave (less fatigue/order confound). ``tasks`` / ``objects`` restrict the
    catalog, ``repetitions`` overrides the per-object count, ``n_episodes`` truncates (or cycles
    through fresh draws), ``instruction`` overrides every rendered instruction (operator text).
    """
    proto = protocol if isinstance(protocol, Protocol) else load_protocol(protocol)
    rng = random.Random(seed)
    if instruction is not None:
        try:
            instruction = normalize_instruction(instruction)
        except ValueError as e:
            raise ProtocolError(f"operator instruction: {e}") from None
    common = dict(protocol=proto.name, version=proto.version, dataset=proto.dataset, kind=proto.kind,
                  seed=seed, timing=scale_timing(proto.timing, time_scale), cameras=proto.cameras, rates=dict(proto.rates),
                  sync=proto.sync, time_scale=float(time_scale))
    if not proto.is_task:
        if tasks or objects:
            raise ProtocolError(f"{proto.name}: --task/--object only apply to task catalogs")
        steps = scale_steps(proto.expand_blocks(), time_scale, proto.timing.min_step_s)
        return SessionPlan(episodes=[EpisodePlan(0, steps, None)], **common)

    sel = list(proto.tasks) if tasks is None else list(dict.fromkeys(str(t) for t in tasks))   # dedupe, keep order
    unknown = [t for t in sel if t not in proto.tasks]
    if unknown:
        raise ProtocolError(f"unknown tasks {unknown} (catalog: {list(proto.tasks)})")
    obj_filter = None if objects is None else {str(o) for o in objects}
    items: list[tuple[TaskSpec, str, int]] = []
    for tid in sel:
        spec = proto.tasks[tid]
        objs = [o for o in spec.objects if obj_filter is None or o in obj_filter]
        reps = spec.repetitions if repetitions is None else int(repetitions)
        items.extend((spec, o, r) for o in objs for r in range(reps))
    if not items:
        raise ProtocolError("nothing to record: the task/object selection is empty")
    if shuffle:
        rng.shuffle(items)
    if n_episodes is not None:
        if n_episodes < 1:
            raise ProtocolError("n_episodes must be ≥ 1")
        items = [items[i % len(items)] for i in range(int(n_episodes))]
    pre = scale_steps(proto.expand_blocks(proto.episode_pre), time_scale, proto.timing.min_step_s)
    post = scale_steps(proto.expand_blocks(proto.episode_post), time_scale, proto.timing.min_step_s)
    episodes = []
    for i, (spec, obj, rep) in enumerate(items):
        slots: dict[str, Any] = {"object": obj, "target": rng.choice(spec.targets) if spec.targets else None}
        for k, vals in spec.slots.items():
            slots[k] = rng.choice(vals)
        ti = rng.randrange(len(spec.templates))
        tpl = spec.templates[ti]
        text = instruction if instruction else render_instruction(tpl, slots)
        task = {"task_id": spec.id, "instruction": text, "object": obj, "target": slots["target"],
                "success": None, "repetition": rep, "template_index": ti if not instruction else None,
                "template": tpl if not instruction else None, "instruction_source": "operator" if instruction else "template",
                "slots": {k: v for k, v in slots.items() if k not in ("object", "target")},
                "success_criteria": spec.success, "grasp": spec.grasp, "manipulate": spec.manipulate}
        steps = pre + scale_steps(proto.task_steps(spec, slots), time_scale, proto.timing.min_step_s) + post
        _check_unique(steps, proto.name)
        episodes.append(EpisodePlan(i, steps, task))
    return SessionPlan(episodes=episodes, **common)


# ── operator script ──────────────────────────────────────────────────────────
_CONTACT_KO = {"none": "무접촉", "self": "자기접촉", "object": "물체접촉", "any": "-"}


def format_script(plan: SessionPlan, *, lang: str = "ko", max_episodes: int | None = None) -> str:
    """Human-readable operator script (Korean by default, ``lang="en"`` for English prompts)."""
    lines = [f"# {plan.protocol} v{plan.version} — dataset={plan.dataset} kind={plan.kind} seed={plan.seed}"
             + (f" time_scale={plan.time_scale:g}" if plan.time_scale != 1.0 else ""),
             f"# episodes: {len(plan.episodes)}, nominal total {plan.total_duration_s / 60:.1f} min "
             f"(+ operator pauses); cameras: {', '.join(plan.cameras) or '-'}"]
    eps = plan.episodes if max_episodes is None else plan.episodes[:max_episodes]
    for ep in eps:
        lines.append("")
        if ep.task:
            lines.append(f"## episode {ep.index:03d}: {ep.task['task_id']} / {ep.task['object']}"
                         + (f" → {ep.task['target']}" if ep.task.get("target") else "")
                         + f" (rep {ep.task['repetition'] + 1})")
            lines.append(f'   instruction: "{ep.task["instruction"]}"')
            if ep.task.get("success_criteria"):
                lines.append(f"   success: {ep.task['success_criteria']}")
        else:
            lines.append(f"## session ({len(ep.steps)} steps, {ep.duration_s(plan.timing) / 60:.1f} min)")
        for k, ts in enumerate(ep.timeline(plan.timing), 1):
            s = ts.step
            prompt = (s.prompt_en or s.prompt) if lang == "en" else (s.prompt or s.prompt_en)
            tag = "manual" if s.advance == "manual" else f"{s.duration_s:g}s"
            lines.append(f"{k:3d}. [{tag:>6}] {s.id:<28} {_CONTACT_KO.get(s.contact, s.contact):<5} {prompt}")
    if max_episodes is not None and len(plan.episodes) > max_episodes:
        lines.append(f"... ({len(plan.episodes) - max_episodes} more episodes)")
    return "\n".join(lines)


# ── naming ───────────────────────────────────────────────────────────────────
_SLUG = re.compile(r"[^A-Za-z0-9]+")


def _slug(x: Any) -> str:
    s = _SLUG.sub("-", str(x)).strip("-")
    return s or "x"


def make_session_id(dataset: str, subject: str, when: datetime | None = None, *, task_id: str | None = None,
                    obj: str | None = None, rep: int | None = None, index: int | None = None) -> str:
    """Session directory name: ``<dataset>-<subject>-<YYYYMMDD>-<HHMMSS>[-<task>-<object>-r<rep>]``.

    ASCII only (safe on every filesystem); ``index`` (episode number) disambiguates episodes that
    start within the same second.
    """
    when = when or datetime.now()
    parts = [_slug(dataset), _slug(subject or "S00"), when.strftime("%Y%m%d"), when.strftime("%H%M%S")]
    if index is not None:
        parts.append(f"e{int(index):03d}")
    if task_id:
        parts.append(_slug(task_id).replace("-", "_"))
    if obj:
        parts.append(_slug(obj).replace("-", "_"))
    if rep is not None:
        parts.append(f"r{int(rep) + 1:02d}")
    return "-".join(parts)
