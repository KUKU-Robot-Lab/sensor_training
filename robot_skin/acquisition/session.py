"""Run a protocol plan through the recorder: operators, one directory per episode, post-processing.

::

    plan = plan_session("d1_motion", seed=0)
    results = run_plan(plan, out="robot_skin/data/raw/motion/S01/<id>", kind="glove", subject="S01",
                       source_factory=fake_source_factory(plan, kind="glove"), clock_factory=SimClock,
                       operator=AutoOperator())

For every :class:`~robot_skin.acquisition.protocol.EpisodePlan` (D1: the whole session; D2: one
task repetition) :func:`record_episode` creates a :class:`~robot_skin.acquisition.recorder.Recorder`,
walks the steps (``phase_start`` → operator performs → ``phase_end``; the calibration block is
checked live and can be repeated), logs the instruction and the success verdict (D2), adds the
``task`` segment, stops, and post-processes the directory (:func:`postprocess_session`):

1. **sync** — 3-tap cross-correlation, timestamps corrected in place (``sync.sync_session``);
2. **IMU calibration** — flat-hand block → ``manifest.calibration`` (``calibration.calibrate_session_imu``);
3. **QC** — ``qc.session_qc`` → ``qc.json``.

Operators: :class:`AutoOperator` follows the nominal timeline exactly (synthetic ``--fake`` runs
under a :class:`~robot_skin.acquisition.sources.SimClock`); :class:`ConsoleOperator` prompts a
human (Korean script, Enter to start timed blocks, Enter to mark manual phase transitions,
y/n success) while the recorder threads keep polling.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .calibration import calibrate_session_imu, compute_imu_calibration
from .manifest import SessionManifest, StreamInfo
from .protocol import EpisodePlan, SessionPlan, Step, SyncSpec, TimedStep, Timing, make_session_id
from .qc import format_report, session_qc
from .recorder import TIME_EPS, Recorder, segments_from_events
from .sources import Clock, MonotonicClock
from .sync import ClockModel, apply_clock_models, sync_session

__all__ = [
    "AutoOperator", "ConsoleOperator", "Operator", "adhoc_plan", "copy_imu_calibration", "fake_source_factory",
    "planned_streams",
    "postprocess_session", "record_episode", "run_plan", "write_dry_run",
]

log = logging.getLogger(__name__)
IMU_SITES = ("wrist", "palm", "thumb", "index", "middle", "ring", "pinky")


# ── operators ────────────────────────────────────────────────────────────────
class Operator(Protocol):
    def begin_episode(self, rec: Recorder, episode: EpisodePlan, plan: SessionPlan) -> None: ...
    def before_step(self, rec: Recorder, ts: TimedStep, k: int, n: int) -> None: ...
    def perform(self, rec: Recorder, ts: TimedStep) -> None: ...
    def after_step(self, rec: Recorder, ts: TimedStep, info: Mapping) -> bool: ...
    def end_episode(self, rec: Recorder, episode: EpisodePlan, plan: SessionPlan) -> None: ...
    def success(self, rec: Recorder, episode: EpisodePlan) -> bool | None: ...


class AutoOperator:
    """Follows the nominal timeline exactly (no human; used with a SimClock for ``--fake``)."""

    def __init__(self, success: bool | None = True):
        self._success = success
        self._timing: Timing | None = None
        self._end = 0.0

    def begin_episode(self, rec, episode, plan) -> None:
        self._timing = plan.timing
        tl = episode.timeline(plan.timing)
        self._end = (tl[-1].t1 if tl else plan.timing.lead_in_s) + plan.timing.lead_out_s
        rec.run_until(plan.timing.lead_in_s)

    def before_step(self, rec, ts, k, n) -> None:
        rec.run_until(ts.t0)

    def perform(self, rec, ts) -> None:
        rec.run_until(ts.t1)

    def after_step(self, rec, ts, info) -> bool:
        return False

    def end_episode(self, rec, episode, plan) -> None:
        rec.run_until(self._end)

    def success(self, rec, episode) -> bool | None:
        return self._success


class ConsoleOperator:
    """Interactive operator on the terminal (``input_fn``/``print_fn`` injectable for tests).

    Timed steps: Enter starts the block, then it runs for its duration with a per-second countdown
    (and a bell per motion cycle as metronome). Manual steps (D2 phases, ``advance: manual``):
    every Enter marks a phase **start** boundary as defined in the protocol (``boundary``) — the
    first Enter starts the first phase of a manual chain (``reach``: the hand leaves the start
    pose), each further Enter ends the running phase and starts the next one, and one last Enter
    ends the chain (``retreat``: the hand is back at the start pose). A chain of n phases therefore
    takes n + 1 key presses. After the calibration block the live stillness check is shown and the
    block can be repeated.
    """

    def __init__(self, *, input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
                 lang: str = "ko", metronome: bool = True, sleep=time.sleep):
        self.input, self.print, self.lang, self.metronome, self.sleep = input_fn, print_fn, lang, metronome, sleep
        self._steps: list[Step] = []
        self._k = 0

    def _prompt(self, s: Step) -> str:
        return (s.prompt_en or s.prompt) if self.lang == "en" else (s.prompt or s.prompt_en)

    def _manual(self, k: int) -> bool:
        return 0 <= k < len(self._steps) and self._steps[k].advance == "manual"

    @staticmethod
    def _boundary(s: Step) -> str:
        return f" ({s.boundary})" if s.boundary else ""

    @staticmethod
    def _end_cue(s: Step) -> str:
        """The ``(끝: …)`` / ``(end: …)`` part of a boundary text, if the protocol gives one."""
        m = re.search(r"\((?:끝|end):\s*([^)]*)\)", s.boundary or "")
        return f" ({m.group(1).strip()})" if m else ""

    def begin_episode(self, rec, episode, plan) -> None:
        self._steps = [ts.step for ts in episode.timeline(plan.timing)]
        if episode.task:
            self.print(f"\n=== episode {episode.index:03d}: {episode.task['task_id']} / {episode.task['object']} ===")
            self.print(f'    지시문: "{episode.task["instruction"]}"')
            if episode.task.get("success_criteria"):
                self.print(f"    성공 기준: {episode.task['success_criteria']}")
        self.input("준비되면 Enter (기록은 이미 시작됨) ")

    def before_step(self, rec, ts, k, n) -> None:
        s = ts.step
        self._k = k
        tag = "수동 전환" if s.advance == "manual" else f"{s.duration_s:g}초"
        self.print(f"\n[{k + 1}/{n}] {s.id} ({tag}) — {self._prompt(s)}")
        if s.advance == "timed":
            self.input("  시작하려면 Enter ")
        elif not self._manual(k - 1):          # first phase of a manual chain: mark its start too
            self.input(f"  {s.id} 시작 순간에 Enter{self._boundary(s)} ")

    def perform(self, rec, ts) -> None:
        s = ts.step
        if s.advance == "manual":
            if self._manual(self._k + 1):      # this Enter = start boundary of the next phase
                nxt = self._steps[self._k + 1]
                self.input(f"  {nxt.id} 시작 순간에 Enter{self._boundary(nxt)} ")
            else:                              # last phase of the chain: mark its end
                self.input(f"  {s.id} 끝 순간에 Enter{self._end_cue(s)} ")
            return
        cycle = float(s.motion.get("cycle_s", 0) or 0)
        t_end = rec.now() + s.duration_s
        next_beat = rec.now() + cycle if cycle else None
        last_sec = None
        while (rem := t_end - rec.now()) > TIME_EPS:    # same tolerance as Recorder.run_until
            sec = int(rem) + 1
            if sec != last_sec:
                self.print(f"  … {sec}s")
                last_sec = sec
            if self.metronome and next_beat is not None and rec.now() >= next_beat:
                self.print("\a")
                next_beat += cycle
            rec.run_until(min(t_end, rec.now() + 0.1), sleep=self.sleep)

    def after_step(self, rec, ts, info) -> bool:
        q = info.get("calibration")
        if not q:
            return False
        self.print(f"  보정 품질: spread={q['quat_spread_deg']:.2f}° gyro_rms={q.get('gyro_rms', float('nan')):.3f} "
                   f"→ {'OK' if q['ok'] else '움직임 과다'}")
        if not q["ok"]:
            return self.input("  보정 블록을 다시 할까요? [Y/n] ").strip().lower() not in ("n", "no")
        return False

    def end_episode(self, rec, episode, plan) -> None:
        rec.run_for(plan.timing.lead_out_s, sleep=self.sleep)

    def success(self, rec, episode) -> bool | None:
        a = self.input("  성공? [y/n, Enter=판정 보류] ").strip().lower()
        return True if a in ("y", "yes") else False if a in ("n", "no") else None


# ── planning helpers ─────────────────────────────────────────────────────────
def adhoc_plan(duration_s: float | None, *, no_contact: bool = False, kind: str = "glove", dataset: str = "other",
               cameras: Sequence[str] = (), rates: Mapping[str, float] | None = None) -> SessionPlan:
    """Protocol-free session: one step ``recording`` starting at t = 0 (labelled ``no_contact``
    when ``no_contact``) — the legacy ``--duration/--no-contact`` mode. ``duration_s=None`` → a
    manual step that runs until the operator stops it (nominal 60 s for planning)."""
    if duration_s is not None and not duration_s > 0:
        raise ValueError("duration must be > 0")
    step = Step(id="recording", block="recording", kind="motion",
                duration_s=60.0 if duration_s is None else float(duration_s),
                contact="none" if no_contact else "any", labels=("no_contact",) if no_contact else (),
                advance="manual" if duration_s is None else "timed",
                motion={"type": "free"},
                prompt="자유 기록 (Enter = 시작, 다시 Enter = 종료)" if duration_s is None else "자유 기록",
                prompt_en="free recording")
    return SessionPlan(protocol="adhoc", version=1, dataset=dataset, kind=kind, seed=0,
                       timing=Timing(0.0, 0.0, 0.0, 0.1), episodes=[EpisodePlan(0, [step])],
                       cameras=tuple(cameras), rates=dict(rates or {}), sync=SyncSpec())


def planned_streams(kind: str, cameras: Sequence[str] = (), rates: Mapping[str, float] | None = None, *,
                    imu_sites: Sequence[str] = IMU_SITES, imu: bool = True) -> dict[str, StreamInfo]:
    """Streams a session of ``kind`` will contain (for dry-run manifests)."""
    r = {"pressure": 200.0, "imu": 100.0, "camera": 30.0, "joint_state": 100.0, **(rates or {})}
    out = {"pressure": StreamInfo(file="pressure.npz", rate_hz=r["pressure"], fields=["raw[C]"])}
    if kind == "glove":
        if imu:
            out["imu"] = StreamInfo(file="imu.npz", rate_hz=r["imu"],
                                    fields=[f"{s}.{k}" for s in imu_sites for k in ("quat", "gyro", "acc")])
        for c in cameras:
            out[f"camera_{c}"] = StreamInfo(file=f"camera_{c}", rate_hz=r["camera"], fields=["frame"], method="zoh")
    elif kind == "robot":
        out["joint_state"] = StreamInfo(file="joint_state.npz", rate_hz=r["joint_state"], fields=["q[D]", "qd[D]", "tau[D]"])
    return out


def _base_meta(plan: SessionPlan, episode: EpisodePlan, extra: Mapping | None) -> dict:
    meta = {"protocol": plan.protocol, "protocol_version": plan.version, "seed": plan.seed,
            "episode_index": episode.index, "n_episodes": len(plan.episodes), "time_scale": plan.time_scale,
            "cameras": list(plan.cameras), "sync_spec": plan.sync.to_dict(), "plan": episode.to_dict(plan.timing)}
    meta.update(extra or {})
    return meta


def write_dry_run(plan: SessionPlan, out: str | Path, *, kind: str, layout: str, subject: str = "",
                  streams: Mapping[str, StreamInfo] | None = None, notes: str = "",
                  meta: Mapping | None = None) -> Path:
    """Plan only. Single-episode plans → ``<out>/session.json`` with the planned streams and the
    nominal segments; multi-episode plans → ``<out>/plan.json``. Returns the written file."""
    out = Path(out)
    if len(plan.episodes) == 1:
        ep = plan.episodes[0]
        m = SessionManifest(kind=kind, layout=layout, streams=dict(streams or {}), notes=notes,
                            dataset=plan.dataset if plan.dataset in ("motion", "task") else "other",
                            subject=subject, task=ep.task,
                            meta={"dry_run": True, "duration_s": ep.duration_s(plan.timing),
                                  **_base_meta(plan, ep, meta)})
        evs = []
        for ts in ep.timeline(plan.timing):
            evs.append({"t": ts.t0, "type": "phase_start", "name": ts.step.id, "value": ts.step.event_value()})
            evs.append({"t": ts.t1, "type": "phase_end", "name": ts.step.id, "value": None})
        for s in segments_from_events(evs):
            m.add_segment(s["t0"], s["t1"], s["label"])
        return m.save(out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "plan.json"
    p.write_text(json.dumps({"dry_run": True, "kind": kind, "layout": layout, "subject": subject,
                             "notes": notes, **plan.to_dict()}, indent=2, ensure_ascii=False))
    return p


def fake_source_factory(plan: SessionPlan, *, kind: str = "glove", layout: str = "glove_template",
                        cameras: Sequence[str] | None = None, seed: int | None = None, hand_pose: bool = True,
                        config=None, imu: bool = True) -> Callable[[EpisodePlan], list]:
    """``episode → sources`` for synthetic sessions (one :class:`FakeScene` per episode)."""
    from .fake import FakeScene
    from .sources import fake_sources

    cams = tuple(plan.cameras if cameras is None else cameras)
    base = plan.seed if seed is None else int(seed)

    def factory(episode: EpisodePlan) -> list:
        scene = FakeScene.from_episode(episode, plan.timing, kind=kind, layout=layout, cameras=cams,
                                       seed=base * 1000 + episode.index, config=config)
        srcs = fake_sources(scene, hand_pose=hand_pose)
        if not imu:
            srcs = [s for s in srcs if s.kind != "imu"]
        factory.scenes.append(scene)                       # ground truth for tests
        return srcs

    factory.scenes = []                                    # type: ignore[attr-defined]
    return factory


# ── recording ────────────────────────────────────────────────────────────────
def _live_calibration(rec: Recorder, layout, t0: float, t1: float, trim_s: float = 0.5) -> dict | None:
    imu = next((s for s in rec.sources if s.kind == "imu"), None)
    if imu is None or not getattr(layout, "imu_sites", ()):
        return None
    a, b = (t0 + trim_s, t1 - trim_s) if t1 - t0 > 2 * trim_s + 0.1 else (t0, t1)
    snap = rec.snapshot(imu.name, a, b)
    if snap["t"].size < 3:
        return None
    sites = rec.source_info(imu.name).get("sites") or [s.name for s in layout.imu_sites]
    try:
        _, q = compute_imu_calibration(snap["quat"], sites, layout, gyro=snap.get("gyro"))
    except ValueError as e:
        log.warning("live IMU calibration check failed: %s", e)
        return None
    return q


def record_episode(plan: SessionPlan, episode: EpisodePlan, session_dir: str | Path, sources: Sequence[Any], *,
                   kind: str, layout: str = "glove_template", subject: str = "", operator: Operator | None = None,
                   clock: Clock | None = None, notes: str = "", meta: Mapping | None = None,
                   camera_format: str = "auto", overwrite: bool = False, postprocess: bool = True,
                   sync: bool = True, calibrate: bool = True, qc: bool = True,
                   sync_from: str | Path | None = None, calibration_from: str | Path | None = None) -> dict:
    """Record one episode into ``session_dir`` and post-process it. Returns
    ``{"session_dir", "manifest", "qc", "sync", "calibration"}``."""
    from common.layouts import load_layout

    lay = load_layout(layout)
    operator = operator or AutoOperator()
    task = dict(episode.task) if episode.task else None
    m = SessionManifest(kind=kind, layout=layout, dataset=plan.dataset if plan.dataset in ("motion", "task") else "other",
                        subject=subject, task=task, notes=notes, meta=_base_meta(plan, episode, meta))
    rec = Recorder(sources, session_dir, m, clock=clock, camera_format=camera_format, overwrite=overwrite)
    timeline = episode.timeline(plan.timing)
    rec.start()
    task_span: list[float] = []
    try:
        operator.begin_episode(rec, episode, plan)
        if task:
            rec.instruction(task["instruction"])
        for k, ts in enumerate(timeline):
            s = ts.step
            while True:
                operator.before_step(rec, ts, k, len(timeline))
                rec.phase_start(s.id, s.event_value())
                t0 = rec.now()
                operator.perform(rec, ts)
                rec.phase_end(s.id)
                t1 = rec.now()
                info: dict[str, Any] = {}
                if s.kind == "calibration":
                    q = _live_calibration(rec, lay, t0, t1)
                    if q is not None:
                        info["calibration"] = q
                if not operator.after_step(rec, ts, info):
                    break
            if s.kind == "task_phase":
                task_span = [task_span[0] if task_span else t0, t1]
        operator.end_episode(rec, episode, plan)
        if task:
            ok = operator.success(rec, episode)
            rec.success(ok)
            m.task["success"] = ok
        if task_span:
            rec.add_segment(task_span[0], task_span[1], "task")
    except BaseException as e:
        if rec.running:
            rec.marker("aborted", f"{type(e).__name__}: {e}")
            rec.stop()
        raise
    m = rec.stop()
    result: dict[str, Any] = {"session_dir": str(session_dir), "manifest": m}
    if postprocess:
        result.update(postprocess_session(session_dir, sync=sync, calibrate=calibrate and kind == "glove", qc=qc,
                                          sync_from=sync_from, calibration_from=calibration_from))
    return result


def copy_imu_calibration(session_dir: str | Path, source_session: str | Path) -> dict:
    """Copy the IMU calibration (``imu_offsets`` / ``imu_world`` / ``imu_sites`` + quality) of
    another session of the same sitting (glove not re-donned) into this one."""
    src = SessionManifest.load(source_session).calibration
    keys = [k for k in ("imu_offsets", "imu_world", "imu_sites", "imu_calibration_quality") if k in src]
    if "imu_offsets" not in keys:
        raise ValueError(f"{source_session} has no IMU calibration")
    m = SessionManifest.load(session_dir)
    m.calibration.update({k: src[k] for k in keys})
    m.calibration["imu_calibration_copied_from"] = str(source_session)
    m.save(session_dir)
    return {k: src[k] for k in keys}


def postprocess_session(session_dir: str | Path, *, sync: bool = True, calibrate: bool = True, qc: bool = True,
                        sync_from: str | Path | None = None, calibration_from: str | Path | None = None) -> dict:
    """Sync (or apply another session's clock models), IMU calibration (or copy it from
    ``calibration_from`` when this session has no calibration block), QC (``qc.json``)."""
    d = Path(session_dir)
    out: dict[str, Any] = {"sync": None, "calibration": None, "qc": None}
    if sync_from is not None:
        src = SessionManifest.load(sync_from).calibration.get("sync") or {}
        models = {n: ClockModel.from_dict(e) for n, e in src.get("streams", {}).items() if e.get("offset_s") is not None}
        m = SessionManifest.load(d)
        done = apply_clock_models(d, {n: c for n, c in models.items() if n in m.streams}, manifest=m)
        m.calibration["sync"] = {**src, "applied": bool(done), "copied_from": str(sync_from),
                                 "streams": {n: src["streams"][n] for n in done}}
        m.save(d)
        out["sync"] = m.calibration["sync"]
    elif sync:
        out["sync"] = sync_session(d)
    if calibrate:
        out["calibration"] = calibrate_session_imu(d)
        if out["calibration"] is None and calibration_from is not None:
            out["calibration"] = copy_imu_calibration(d, calibration_from)
    if qc:
        out["qc"] = session_qc(d, write=True)
    out["manifest"] = SessionManifest.load(d)
    return out


def run_plan(plan: SessionPlan, *, kind: str, source_factory: Callable[[EpisodePlan], Sequence[Any]],
             out: str | Path | None = None, root: str | Path = "robot_skin/data/raw", subject: str = "S00",
             layout: str = "glove_template", operator: Operator | None = None,
             clock_factory: Callable[[], Clock] = MonotonicClock, notes: str = "", meta: Mapping | None = None,
             camera_format: str = "auto", postprocess: bool = True, sync: bool = True, calibrate: bool = True,
             qc: bool = True, sync_from: str | Path | None = None, calibration_from: str | Path | None = None,
             when: datetime | None = None,
             on_result: Callable[[dict], None] | None = None) -> list[dict]:
    """Record every episode of ``plan``. Directory rule: single-episode plans record into ``out``
    (or ``<root>/<dataset>/<subject>/<session_id>``); multi-episode plans (D2) record each episode
    into ``<out or root/dataset/subject>/<session_id>`` (one task repetition = one directory).
    A ``KeyboardInterrupt`` finishes the current episode's files and stops the plan."""
    results = []
    multi = len(plan.episodes) > 1
    parent = Path(out) if out is not None else Path(root) / plan.dataset / (subject or "S00")
    for ep in plan.episodes:
        stamp = when or datetime.now()
        t = ep.task or {}
        sid = make_session_id(plan.dataset, subject, stamp, task_id=t.get("task_id"), obj=t.get("object"),
                              rep=t.get("repetition"), index=ep.index if multi else None)
        d = parent / sid if (multi or out is None) else Path(out)
        try:
            res = record_episode(plan, ep, d, source_factory(ep), kind=kind, layout=layout, subject=subject,
                                 operator=operator, clock=clock_factory(), notes=notes, meta=meta,
                                 camera_format=camera_format, postprocess=postprocess, sync=sync,
                                 calibrate=calibrate, qc=qc, sync_from=sync_from, calibration_from=calibration_from)
        except KeyboardInterrupt:
            log.warning("interrupted — episode %d saved as far as recorded in %s", ep.index, d)
            if postprocess and (d / "session.json").exists():
                try:                                     # qc.json flags it (auto-closed phases)
                    postprocess_session(d, sync=sync, calibrate=calibrate and kind == "glove", qc=qc,
                                        sync_from=sync_from, calibration_from=calibration_from)
                except Exception:                        # pragma: no cover - best effort
                    log.exception("post-processing of the interrupted episode failed")
            break
        results.append(res)
        if on_result is not None:
            on_result(res)
    return results


# ── CLI: re-run post-processing on recorded sessions ─────────────────────────
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m robot_skin.acquisition.session",
                                description="Re-run sync / IMU calibration / QC on recorded session dirs.")
    p.add_argument("sessions", nargs="+", type=Path)
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--sync-from", type=Path, default=None, help="apply the clock models of this session instead")
    p.add_argument("--calibration-from", type=Path, default=None,
                   help="copy the IMU calibration of this session when a session has no calibration block")
    a = p.parse_args(argv)
    ok = True
    for s in a.sessions:
        r = postprocess_session(s, sync=not a.no_sync, calibrate=not a.no_calibrate, sync_from=a.sync_from,
                                calibration_from=a.calibration_from)
        print(format_report(r["qc"]))
        ok &= r["qc"]["passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
