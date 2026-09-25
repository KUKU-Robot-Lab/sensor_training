"""Protocol YAML schema, planning and instruction rendering (robot_skin.acquisition.protocol / .instructions)."""
import copy
import random

import pytest
import yaml

from robot_skin.acquisition.instructions import (
    check_template, humanize, normalize_instruction, render_instruction, sample_instruction, template_slots,
)
from robot_skin.acquisition.protocol import (
    PROTOCOL_DIR, TASK_PHASES, EpisodePlan, ProtocolError, Step, SyncSpec, Timing, format_script, list_protocols,
    load_protocol, make_session_id, plan_session, protocol_from_dict, scale_steps,
)


def _raw(name):
    return yaml.safe_load((PROTOCOL_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


# ── instructions ─────────────────────────────────────────────────────────────
def test_template_slots_and_rendering():
    tpl = "pick up the {object} and place it on the {target}"
    assert template_slots(tpl) == ("object", "target")
    assert template_slots("rotate {object} {angle} then {object}") == ("object", "angle")
    assert render_instruction(tpl, {"object": "red_cup", "target": "box_lid"}) == \
        "pick up the red cup and place it on the box lid"
    assert humanize("cup_of__beads ") == "cup of beads"
    assert normalize_instruction("  press   the  button ") == "press the button"
    with pytest.raises(KeyError):
        render_instruction(tpl, {"object": "cup"})
    with pytest.raises(ValueError):
        template_slots("bad {0} slot")
    with pytest.raises(ValueError):
        template_slots("unbalanced {object")
    with pytest.raises(ValueError):
        check_template(tpl, {"object"})
    with pytest.raises(ValueError):
        normalize_instruction("   ")
    i, text = sample_instruction(["a {object}", "b {object}"], {"object": "x"}, random.Random(3))
    assert text == ("a x", "b x")[i]


# ── loading / validation ─────────────────────────────────────────────────────
def test_builtin_protocols_load():
    assert {"d1_motion", "d2_task", "robot_sweep"} <= set(list_protocols())
    d1 = load_protocol("d1_motion")
    assert d1.dataset == "motion" and d1.kind == "glove" and not d1.is_task
    assert d1.cameras == ("ego", "third")
    d2 = load_protocol("d2_task")
    assert d2.is_task and d2.dataset == "task"
    assert {"grasp_lift_place", "pour", "peg_insert", "open_jar", "wipe", "handover", "press_button",
            "in_hand_rotate"} <= set(d2.tasks)
    assert set(TASK_PHASES) <= set(d2.phases)
    assert d2.phases["reach"]["contact"] == "none" and d2.phases["grasp"]["contact"] == "object"
    assert load_protocol(PROTOCOL_DIR / "robot_sweep.yaml").kind == "robot"
    with pytest.raises(FileNotFoundError):
        load_protocol("nope")


def test_d1_blocks_cover_the_required_motions():
    steps = load_protocol("d1_motion").expand_blocks()
    ids = [s.id for s in steps]
    assert len(ids) == len(set(ids))
    assert ids[0] == "baseline_start" and ids[-1] == "baseline_end"
    assert ids.index("sync_start") < ids.index("imu_calibration") < ids.index("sync_end")
    for f in ("thumb", "index", "middle", "ring", "pinky"):
        assert f"finger_flex_{f}" in ids and f"palm_touch_{f}" in ids
    for f in ("index", "middle", "ring", "pinky"):
        assert f"pinch_{f}" in ids
    assert {"open_close_slow", "open_close_fast", "free_motion", "fist", "finger_crossing"} <= set(ids)
    assert {"wrist_rotation_pronation_supination", "wrist_rotation_flexion_extension",
            "wrist_rotation_radial_ulnar"} <= set(ids)
    by = {s.id: s for s in steps}
    # contact expectation → segment labels
    assert by["baseline_start"].labels == ("no_contact",)
    assert by["imu_calibration"].labels == ("calibration", "no_contact")
    assert by["pinch_index"].contact == "self" and by["pinch_index"].labels == ("self_touch",)
    assert by["sync_start"].labels == ("sync",) and not by["sync_start"].scalable
    assert all(s.labels == ("no_contact",) for s in steps if s.kind == "motion")
    # cycles × repetitions, speed tags, per-item motion params, formatted prompts (Korean names)
    assert by["finger_flex_index"].duration_s == pytest.approx(10.0)
    assert by["open_close_fast"].speed == "fast" and by["open_close_slow"].speed == "slow"
    assert by["finger_flex_ring"].motion == {"type": "finger_flex", "cycle_s": 2.0, "repetitions": 5, "finger": "ring"}
    assert "약지" in by["finger_flex_ring"].prompt and "{" not in by["finger_flex_ring"].prompt
    assert by["sync_start"].duration_s == pytest.approx(SyncSpec(3, (0.6, 1.2), 1.0, 1.5).duration_s)
    assert by["sync_start"].motion["tap_times"] == pytest.approx([1.0, 1.6, 2.8])
    ev = by["pinch_index"].event_value()
    assert ev["contact"] == "self" and ev["labels"] == ["self_touch"] and ev["finger"] == "index"
    # contact-free "air grasps": the D2 grasp shapes held without an object, slow and fast
    grasps = ("power", "precision", "lateral", "tripod")
    for speed in ("slow", "fast"):
        for g in grasps:
            st = by[f"air_grasp_{speed}_{g}"]
            assert st.kind == "motion" and st.speed == speed and st.contact == "none"
            assert st.labels == ("no_contact",) and st.motion["type"] == "air_grasp" and st.motion["grasp"] == g
            assert "{" not in st.prompt and "{" not in st.prompt_en and g in st.prompt_en
            assert st.event_value()["grasp"] == g and st.event_value()["labels"] == ["no_contact"]
    assert "파워 그립" in by["air_grasp_slow_power"].prompt and "세 손가락" in by["air_grasp_fast_tripod"].prompt
    assert by["air_grasp_slow_power"].duration_s == pytest.approx(12.0)
    assert by["air_grasp_fast_lateral"].duration_s == pytest.approx(6.0)
    assert ids.index("free_motion") < ids.index("air_grasp_slow_power") < ids.index("pinch_index")


def _broken(mutate, name="d1_motion"):
    d = copy.deepcopy(_raw(name))
    mutate(d)
    return d


@pytest.mark.parametrize("mutate", [
    lambda d: d["blocks"][0].update(contact="maybe"),
    lambda d: d["blocks"][0].update(speed="warp"),
    lambda d: d["blocks"][0].update(typo_key=1),
    lambda d: d["blocks"][0].update(id="Bad-Id"),
    lambda d: d["blocks"][0].pop("duration_s"),
    lambda d: d["blocks"][0].update(duration_s=-1),
    lambda d: d["blocks"][0].update(prompt="{unknown_placeholder}"),
    lambda d: d["blocks"].append(dict(d["blocks"][0])),                     # duplicate step id
    lambda d: d["blocks"][3].update(for_each={"finger": []}),
    lambda d: d.update(dataset="games"),
    lambda d: d.update(sync={"taps": 3, "intervals_s": [0.5]}),
    lambda d: d.update(tasks=[]),                                           # both blocks and tasks
    lambda d: d.update(blcoks=[]),                                          # top-level typo
    lambda d: d["blocks"][3].update(repetitions="five"),
    lambda d: d["blocks"][3].update(repetitions=2.5),
])
def test_d1_schema_errors(mutate):
    with pytest.raises(ProtocolError):
        protocol_from_dict(_broken(mutate))


@pytest.mark.parametrize("mutate", [
    lambda d: d["tasks"][0]["templates"].append("put the {object} near the {thing}"),   # unknown slot
    lambda d: d["tasks"][3]["templates"].append("open the {object} on the {target}"),   # no targets
    lambda d: d["tasks"][0].update(phases=["reach", "teleport"]),
    lambda d: d["tasks"][0].update(durations={"retreat_fast": 1.0}),
    lambda d: d["tasks"].append(dict(d["tasks"][0])),                                   # duplicate id
    lambda d: d["phases"]["grasp"].update(contact="sometimes"),
    lambda d: d["tasks"][0].update(objects=[]),
    lambda d: d["episode"]["pre"].append({"id": "reach", "kind": "static", "duration_s": 1}),  # clash
    lambda d: d["episode"].update(prelude=[]),                                         # unknown episode key
    lambda d: d["phases"].update(hover=["contact", "none"]),                           # phase not a mapping
    lambda d: d["tasks"][0].update(repetitions=0),
])
def test_d2_schema_errors(mutate):
    with pytest.raises(ProtocolError):
        protocol_from_dict(_broken(mutate, "d2_task"))


# ── planning ─────────────────────────────────────────────────────────────────
def test_plan_d1_and_time_scale():
    plan = plan_session("d1_motion", seed=0)
    assert len(plan.episodes) == 1 and plan.episodes[0].task is None
    full = plan.total_duration_s
    assert 4 * 60 < full < 7 * 60
    quick = plan_session("d1_motion", time_scale=0.05)
    steps = {s.id: s for s in quick.episodes[0].steps}
    assert steps["sync_start"].duration_s == pytest.approx(plan.sync.duration_s)       # not scaled
    assert steps["finger_flex_index"].duration_s == pytest.approx(2.0)                 # ≥ one cycle
    assert steps["baseline_start"].duration_s == pytest.approx(0.5)                    # min_step_s
    assert quick.total_duration_s < full / 3
    assert quick.timing.transition_s == pytest.approx(0.25) and quick.timing.lead_in_s == pytest.approx(0.25)
    assert plan_session("d1_motion", time_scale=2.0).timing == plan.timing          # pauses never stretched
    # timeline: lead-in, transitions, lead-out
    tl = plan.episodes[0].timeline(plan.timing)
    assert tl[0].t0 == pytest.approx(plan.timing.lead_in_s)
    assert tl[1].t0 == pytest.approx(tl[0].t1 + plan.timing.transition_s)
    assert plan.episodes[0].duration_s(plan.timing) == pytest.approx(tl[-1].t1 + plan.timing.lead_out_s)
    with pytest.raises(ProtocolError):
        plan_session("d1_motion", tasks=["pour"])
    with pytest.raises(ProtocolError):
        scale_steps(plan.episodes[0].steps, 0.0)
    stretched = scale_steps([Step("a", "a", "motion", 2.0)], 2.0)
    assert stretched[0].duration_s == 4.0


def test_plan_d2_episodes_are_seeded_rendered_and_filtered():
    proto = load_protocol("d2_task")
    plan = plan_session(proto, seed=7)
    n_expected = sum(len(t.objects) * t.repetitions for t in proto.tasks.values())
    assert len(plan.episodes) == n_expected
    again = plan_session(proto, seed=7)
    assert [e.task for e in plan.episodes] == [e.task for e in again.episodes]
    other = plan_session(proto, seed=8)
    assert [e.task["task_id"] for e in plan.episodes] != [e.task["task_id"] for e in other.episodes]
    ep = plan.episodes[0]
    t = ep.task
    spec = proto.tasks[t["task_id"]]
    assert t["object"] in spec.objects and t["instruction"] and "{" not in t["instruction"]
    assert t["template"] == spec.templates[t["template_index"]]
    assert (t["target"] in spec.targets) if spec.targets else t["target"] is None
    assert t["success"] is None and t["success_criteria"] == spec.success
    ids = [s.id for s in ep.steps]
    assert ids[:3] == ["baseline", "imu_calibration", "sync_start"]
    assert ids[3:] == list(spec.phases)
    phase = {s.id: s for s in ep.steps}
    assert phase["reach"].contact == "none" and phase["reach"].labels == ()
    assert phase["manipulate"].contact == "object" and phase["manipulate"].advance == "manual"
    assert phase["manipulate"].motion["task"] == t["task_id"]
    assert phase["reach"].boundary == proto.phases["reach"]["boundary"] and phase["reach"].boundary
    assert ep.steps[0].boundary == ""                                   # timed pre-blocks have none

    sub = plan_session(proto, seed=1, tasks=["pour", "press_button"], objects=["bowl_nope", "red_button", "cup_of_beads"],
                       repetitions=2)
    assert {e.task["task_id"] for e in sub.episodes} == {"pour", "press_button"}
    assert {e.task["object"] for e in sub.episodes} == {"red_button", "cup_of_beads"}
    assert len(sub.episodes) == 4
    press = next(e for e in sub.episodes if e.task["task_id"] == "press_button")
    assert "grasp" not in [s.id for s in press.steps]
    cyc = plan_session(proto, tasks=["open_jar"], n_episodes=13)
    assert len(cyc.episodes) == 13 and [e.index for e in cyc.episodes] == list(range(13))
    rot = plan_session(proto, tasks=["in_hand_rotate"], repetitions=1, seed=2)
    assert all(e.task["slots"]["angle"] in ("90", "180") and e.task["slots"]["angle"] in e.task["instruction"]
               for e in rot.episodes)
    op = plan_session(proto, tasks=["wipe"], n_episodes=2, instruction="wipe the table please")
    assert all(e.task["instruction"] == "wipe the table please" and e.task["instruction_source"] == "operator"
               for e in op.episodes)
    assert len(plan_session(proto, tasks=["wipe", "wipe"], repetitions=1).episodes) == 2   # 2 objects, deduped
    spaced = plan_session(proto, tasks=["wipe"], n_episodes=1, instruction="  wipe   the table ")
    assert spaced.episodes[0].task["instruction"] == "wipe the table"
    with pytest.raises(ProtocolError):
        plan_session(proto, tasks=["wipe"], instruction="   ")
    with pytest.raises(ProtocolError):
        plan_session(proto, tasks=["fly"])
    with pytest.raises(ProtocolError):
        plan_session(proto, objects=["nothing_here"])


def test_format_script_and_session_id():
    from datetime import datetime
    plan = plan_session("d2_task", seed=0, tasks=["grasp_lift_place"], n_episodes=2)
    s = format_script(plan)
    assert "grasp_lift_place" in s and "instruction:" in s and "manual" in s and "물체접촉" in s
    assert plan.episodes[0].task["instruction"] in s
    s_en = format_script(plan_session("d1_motion"), lang="en")
    assert "Slowly flex" in s_en and "baseline_end" in s_en
    assert "more episodes" in format_script(plan_session("d2_task"), max_episodes=1)
    when = datetime(2026, 9, 25, 14, 30, 12)
    assert make_session_id("motion", "S01", when) == "motion-S01-20260925-143012"
    sid = make_session_id("task", "S 2", when, task_id="grasp_lift_place", obj="cup of/beads", rep=0, index=7)
    assert sid == "task-S-2-20260925-143012-e007-grasp_lift_place-cup_of_beads-r01"


def test_plan_to_dict_is_json_serialisable():
    import json
    plan = plan_session("d2_task", seed=0, n_episodes=2)
    d = json.loads(json.dumps(plan.to_dict()))
    assert d["protocol"] == "d2_task" and len(d["episodes"]) == 2
    assert d["episodes"][0]["timeline"][0]["id"] == "baseline"
    ep = EpisodePlan(0, [Step("x", "x", "static", 1.0)])
    assert ep.duration_s(Timing(0.5, 0.0, 0.25)) == pytest.approx(1.75)
