import json

import numpy as np
import pytest

import robot_skin.acquisition as acq
from robot_skin.acquisition import MANIFEST_NAME, SessionManifest, StreamInfo
from robot_skin.acquisition import glove_logger, robot_logger


def test_manifest_roundtrip(tmp_path):
    m = SessionManifest(kind="glove", layout="glove_template",
                        streams={"pressure": StreamInfo(file="pressure.bin", rate_hz=200.0)})
    m.add_segment(0.0, 5.0, "no_contact")
    p = m.save(tmp_path / "S1")
    assert p.name == MANIFEST_NAME
    m2 = SessionManifest.load(tmp_path / "S1")
    assert m2 == m
    assert m2.spans("no_contact") == [(0.0, 5.0)]
    assert m2.stream_path(tmp_path / "S1", "pressure") == tmp_path / "S1" / "pressure.bin"


@pytest.mark.parametrize("bad", [
    dict(kind="car"),
    dict(streams={"x": StreamInfo(file="/abs/path.bin")}),
    dict(streams={"x": StreamInfo(file="a.bin", method="cubic")}),
    dict(segments=[{"t0": 2.0, "t1": 1.0, "label": "x"}]),
])
def test_manifest_validation(bad):
    kw = dict(kind="robot", layout="robot_hand_template") | bad
    with pytest.raises(ValueError):
        SessionManifest(**kw)


def test_manifest_rejects_newer_schema(tmp_path):
    d = SessionManifest(kind="bench", layout="sats_4x4").to_dict()
    d["schema_version"] = 999
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(d))
    with pytest.raises(ValueError):
        SessionManifest.load(tmp_path)


def test_glove_logger_dry_run(tmp_path):
    assert glove_logger.main(["--out", str(tmp_path / "g"), "--dry-run", "--no-contact",
                              "--duration", "10"]) == 0
    m = SessionManifest.load(tmp_path / "g")
    assert m.kind == "glove" and m.layout == "glove_template" and m.meta["dry_run"]
    assert set(m.streams) == {"pressure", "imu", "camera_ego", "camera_third"}
    assert m.cameras == ["ego", "third"]
    assert len([f for f in m.streams["imu"].fields if f.endswith(".quat")]) == 7
    assert m.spans("no_contact") == [(0.0, 10.0)]
    assert glove_logger.main(["--out", str(tmp_path / "g1"), "--dry-run", "--duration", "3",
                              "--cameras", "none", "--no-imu"]) == 0
    assert set(SessionManifest.load(tmp_path / "g1").streams) == {"pressure"}


def test_protocol_dry_runs(tmp_path, capsys):
    assert glove_logger.main(["--protocol", "d1_motion", "--subject", "S03", "--dry-run", "--out",
                              str(tmp_path / "d1")]) == 0
    m = SessionManifest.load(tmp_path / "d1")
    assert m.dataset == "motion" and m.subject == "S03" and m.meta["protocol"] == "d1_motion"
    assert m.spans("calibration") and len(m.spans("no_contact")) >= 2 and m.spans("self_touch")
    assert len(m.spans("sync")) == 2
    assert "finger_flex_index" in capsys.readouterr().out
    assert glove_logger.main(["--protocol", "d2_task", "--task", "pour,wipe", "--episodes", "4", "--dry-run",
                              "--out", str(tmp_path / "d2")]) == 0
    plan = json.loads((tmp_path / "d2" / "plan.json").read_text())
    assert len(plan["episodes"]) == 4 and {e["task"]["task_id"] for e in plan["episodes"]} <= {"pour", "wipe"}
    with pytest.raises(SystemExit):                     # robot protocol on the glove logger
        glove_logger.main(["--protocol", "robot_sweep", "--dry-run", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit):                     # fake/dry-run without protocol need a duration
        glove_logger.main(["--dry-run", "--out", str(tmp_path / "y")])


def test_robot_logger_dry_run_and_stub(tmp_path):
    assert robot_logger.main(["--out", str(tmp_path / "r"), "--dry-run"]) == 0
    m = SessionManifest.load(tmp_path / "r")
    assert set(m.streams) == {"pressure", "joint_state"} and m.meta["protocol"] == "robot_sweep"
    with pytest.raises(NotImplementedError):
        robot_logger.main(["--out", str(tmp_path / "r2")])
    with pytest.raises(NotImplementedError):
        glove_logger.main(["--out", str(tmp_path / "g2")])
    with pytest.raises(NotImplementedError):
        glove_logger.main(["--protocol", "d1_motion", "--out", str(tmp_path / "g3")])


def _sessions(root):
    return sorted(p.parent for p in root.rglob(MANIFEST_NAME))


def test_glove_logger_fake_d1_end_to_end(tmp_path):
    out = tmp_path / "raw" / "motion" / "S01" / "sess"
    assert glove_logger.main(["--protocol", "d1_motion", "--subject", "S01", "--fake", "--time-scale", "0.05",
                              "--cameras", "ego", "--out", str(out), "--seed", "2"]) == 0
    m = SessionManifest.load(out)
    assert m.dataset == "motion" and m.kind == "glove" and m.subject == "S01" and not m.meta.get("dry_run")
    assert set(m.streams) == {"pressure", "imu", "camera_ego", "hand_pose"}
    assert {"imu_offsets", "imu_world", "imu_sites", "sync"} <= set(m.calibration)
    assert m.calibration["sync"]["applied"]
    qc = json.loads((out / "qc.json").read_text())
    assert qc["passed"] and qc["n_warnings"] == 0
    phases = [e["name"] for e in (json.loads(x) for x in (out / "events.jsonl").read_text().splitlines())
              if e["type"] == "phase_start"]
    assert phases[0] == "baseline_start" and phases[-1] == "baseline_end" and "pinch_index" in phases
    with np.load(out / "pressure.npz") as z:
        assert z["raw"].shape[1] == 9 and np.all(np.diff(z["t"]) > 0)


def test_glove_logger_fake_d2_one_dir_per_episode(tmp_path):
    root = tmp_path / "d2"
    assert glove_logger.main(["--protocol", "d2_task", "--task", "grasp_lift_place", "--task", "press_button",
                              "--episodes", "2", "--fake", "--time-scale", "0.4", "--cameras", "third",
                              "--subject", "S04", "--out", str(root),
                              "--instruction", "do the thing"]) == 0
    dirs = _sessions(root)
    assert len(dirs) == 2
    for d in dirs:
        m = SessionManifest.load(d)
        assert m.dataset == "task" and m.task["task_id"] in ("grasp_lift_place", "press_button")
        assert m.task["instruction"] == "do the thing" and m.task["success"] is True
        assert "object_pose" in m.streams and m.spans("task")
        assert d.name.startswith("task-S04-") and m.task["task_id"] in d.name
        ev = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()]
        assert {"instruction", "success"} <= {e["type"] for e in ev}
        assert json.loads((d / "qc.json").read_text())["passed"]


def test_robot_logger_fake(tmp_path):
    assert robot_logger.main(["--fake", "--time-scale", "0.05", "--subject", "R01", "--root", str(tmp_path)]) == 0
    (d,) = _sessions(tmp_path)
    assert d.parent.name == "R01" and d.parent.parent.name == "motion"
    m = SessionManifest.load(d)
    assert set(m.streams) == {"pressure", "joint_state"} and m.kind == "robot"
    with np.load(d / "joint_state.npz") as z:
        assert z["q"].shape[1] == 15 and z["names"][0] == "thumb_j1"
    assert json.loads((d / "qc.json").read_text())["passed"]


def test_legacy_write_plan_and_lazy_exports(tmp_path):
    import argparse
    from robot_skin.acquisition._cli import write_plan
    args = argparse.Namespace(duration=4.0, no_contact=True, out=tmp_path / "w", layout="glove_template", notes="n")
    m = write_plan(args, "glove", glove_logger.planned_streams_default())
    assert m.spans("no_contact") == [(0.0, 4.0)] and "camera_ego" in m.streams
    for name in acq.__all__:
        assert getattr(acq, name) is not None
    with pytest.raises(AttributeError):
        getattr(acq, "definitely_not_there")


def test_console_operator_marks_manual_phase_starts(tmp_path):
    """Interactive D2 run under a SimClock (scripted key presses, 0.7 s each): the timed-step
    countdown terminates, and every Enter of a manual chain is a phase START boundary — the first
    one starts ``reach`` (hand leaves the start pose, idle time before it stays outside the phase),
    the last one ends ``retreat``: n phases → n + 1 presses, contiguous phases."""
    from robot_skin.acquisition.protocol import plan_session
    from robot_skin.acquisition.recorder import load_events, phases_from_events
    from robot_skin.acquisition.session import ConsoleOperator, fake_source_factory, record_episode
    from robot_skin.acquisition.sources import SimClock
    plan = plan_session("d2_task", seed=0, tasks=["press_button"], n_episodes=1, time_scale=0.3)
    ep = plan.episodes[0]
    prompts = []

    class Op(ConsoleOperator):
        def begin_episode(self, rec, episode, plan):
            self.rec = rec
            super().begin_episode(rec, episode, plan)

    def human(prompt):
        op.rec.run_for(0.7)
        prompts.append((op.rec.now(), prompt.strip()))
        return "y" if "성공" in prompt else ("n" if "다시" in prompt else "")

    op = Op(input_fn=human, print_fn=lambda s: None, metronome=False)
    res = record_episode(plan, ep, tmp_path / "S", fake_source_factory(plan, cameras=())(ep), kind="glove",
                         operator=op, clock=SimClock(), sync=False, calibrate=False, qc=False)
    ph = {p["name"]: p for p in phases_from_events(load_events(tmp_path / "S"))}
    manual = [s.id for s in ep.steps if s.advance == "manual"]
    assert manual == ["reach", "manipulate", "release", "retreat"]
    marks = [t for t, p in prompts if "순간에 Enter" in p]
    assert len(marks) == len(manual) + 1
    np.testing.assert_allclose([ph[n]["t0"] for n in manual] + [ph["retreat"]["t1"]], marks, atol=1e-9)
    for a, b in zip(manual, manual[1:]):
        assert ph[a]["t1"] == ph[b]["t0"]
    assert ph["reach"]["t0"] - ph["sync_start"]["t1"] >= 0.7 - 1e-9
    reach_prompt = next(p for _, p in prompts if p.startswith("reach"))
    assert "시작 자세를 떠나는" in reach_prompt
    assert prompts[-2][1].startswith("retreat 끝") and "시작 자세 도착" in prompts[-2][1]
    m = res["manifest"]
    assert m.task["success"] is True and m.spans("task") == [(ph["reach"]["t0"], ph["retreat"]["t1"])]
    # timed steps ran for their planned durations
    base = next(s for s in ep.steps if s.id == "baseline")
    assert ph["baseline"]["t1"] - ph["baseline"]["t0"] == pytest.approx(base.duration_s, abs=0.02)
