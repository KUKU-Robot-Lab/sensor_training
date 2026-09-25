# Wave 2 AS-BUILT digest (interface requests + open concerns)


## SYNTH
### public_api
Module `robot_skin/datasets/synthetic.py` (`__all__` has 20 names).

**Main functions**
- `generate_session(out_dir, *, kind="glove"|"robot", dataset="motion"|"task", duration_s=6.0, seed=0, cameras=("ego",), image_hw=(24,32), layout=None, subject="s0", task_id=None, session_id=None, n_channels=None, params=None, overwrite=False) -> SessionManifest`
  - `out_dir` is the session directory itself.
  - Minimum duration is 2.5 s for motion sessions and 3.0 s for task sessions.
  - `layout` may be `None` (the default template for the kind), a built-in name or YAML path (stored as given), or a `Layout` object. A `Layout` object is written to `<session>/layout.yaml`, and `manifest.layout` holds that file's absolute path.
  - `session_id` defaults to `syn_<kind>_<dataset>_<subject>_<seed>`.
  - An existing session raises `FileExistsError`; with `overwrite=True` the old files are removed first.
- `generate_dataset(root, *, n_motion=2, n_task=2, kind="glove", subjects=("s0","s1"), seed=0, **kw) -> list[Path]`
  - Writes to `root/<dataset>/<subject>/<session_id>`, motion sessions first.
  - Subjects are assigned round-robin; each session gets a distinct seed `seed*10007+k`.
  - Tasks cycle through `sorted(SYNTH_TASKS)` unless `task_id` is passed.
- `load_ground_truth(session_dir) -> dict` reads `gt_synthetic.npz`. All arrays are at the pressure timestamps, in layout order:
  - Tactile components (ΔS %): `artefact_pct` (the true no-contact motion artefact, relative to the true baseline), `press_pct`, `drift_pct`, `noise_pct`, `delta_true_pct`.
  - Contact: bool masks `contact`, `self_touch`, `object_contact`, `saturated` (raw on an ADC rail), plus `penetration_m`.
  - Baselines and geometry: `baseline_raw` [N], `baseline_raw_all` [C], `taxel_pos`, `channels`.
  - Artefact model input and parameters: `joint_angle` (θ) with `joint_angle_names`, `artefact_weights` [N,J], `angle_gain`, `vel_gain`, `quad_gain`, `tau_s`, `press_gain`, `press_max_pct`, `press_tau_s`, `noise_std_pct`.
  - True pose: glove `hand_global_orient`, `hand_finger_pose`, `hand_wrist_pos`; robot `q`; task sessions also `object_pos`.

**Helpers**
- `plan_motion_session(duration_s=6.0, kind="glove") -> list[Block]`
- `plan_task_session(duration_s=6.0) -> list[Block]`
- `Block(name, gen, t0, t1, labels: tuple, contact: "none"|"self"|"object", info)` with a `.dur` property.
- `robot_hand_urdf() -> str`: the synthetic 16-DoF hand. Links are `base_link`, `palm_link`, `thumb_base/metacarpal/proximal/distal_link` and `<f>_proximal/middle/distal_link`; `ROBOT_JOINT_NAMES` equals `URDFModel.joint_names`.
- `SynthParams` dataclass holds every knob (rates, jitter, camera offset range, noise, drift, artefact gains, press model, `sat_prob`, contact margins, IMU mounting/noise, hand-label noise and dropout rate, `success_prob`). Build it with `.from_dict`; unknown keys raise `ValueError`.
- Constants: `GENERATOR_VERSION`, `GT_FILE`, `URDF_FILE`, `EVENTS_FILE`, `LAYOUT_FILE`, `TASK_PHASES`, `CONTACT_PHASES`, `CONTACT_LABELS`, `MOTION_BLOCKS`, `SYNTH_TASKS` (6 tasks), `OBJECT_RADIUS`.

**Files written per session**
- `session.json`: manifest v2 with segments, `task`, `calibration`, and `meta` containing `generator`, `seed`, `duration_s`, `synthetic{...}` and, for robot sessions, `urdf="robot.urdf"`.
- `pressure.npz`: `t` float64 and `raw[T,C]` float64, channel-major (layout taxel i goes to column `layout.channels[i]`; unused channels carry baseline + noise).
- Glove: `imu.npz` (quat/gyro/acc float32 in the sensor frame, quat w ≥ 0, `sites` as a str array) and `hand_pose.npz` (written with `save_hand_labels`).
- Robot: `joint_state.npz` (`t`, `q`/`qd` float32, `names`) and `robot.urdf`.
- Task sessions: `object_pose.npz`.
- One `camera_<name>/` directory per camera with `frames.npy` (uint8) and `timestamps.npy`.
- `events.jsonl`: phase_start/phase_end with value `{contact, labels, generator, ...}`; task sessions also have an `instruction` event first and a `success` event last.
- `gt_synthetic.npz`.

**How the parts map to the brief**
- Hand motion uses `ManoSkeleton.flexion_pose`.
- IMU streams come from `synthesize_imu`, with a known mounting M (wrist = identity) and an IMU-world yaw G. `manifest.calibration` is `imu_calibration_to_dict(conj(M), G, sites)`.
- Glove self-touch is decided by `self_touch_from_hand` itself; its press magnitude is the penetration into the same margin-inflated capsules.
- Tactile model: ΔS = lagged (angle + angle² + velocity) artefact + soft-saturating press ∝ penetration (fast lag) + drift + noise; raw = b(1+ΔS/100), clipped and rounded; occasional 50–300 ms dropouts to ADC_MIN.
- Phase names follow `acquisition/protocols/d1_motion.yaml` and `d2_task.yaml` (`imu_calibration`, `baseline_start`, `open_close_slow/fast`, `wrist_rotation`, `free_motion`, `pinch_<finger>`, `fist`, `baseline_end`; D2: `baseline`, reach, grasp, manipulate, release, retreat, `baseline_end`).
### deviations
1. **Extra arguments on `generate_session`:** `session_id`, `n_channels` (extra unused ADC channels), `params` (a `SynthParams` or a dict) and `overwrite`. The default `session_id` is deterministic so repeated runs are reproducible.
2. **Phase names follow ACQ's protocol ids, not the ones I first planned.**
   - The glove D1 start is split into `imu_calibration` (flat hand, palm down; labels calibration and no_contact) and `baseline_start` (move to the relaxed pose).
   - Each phase value carries `{contact, labels}` like ACQ's recorder.
   - D2 static blocks are named `baseline` / `baseline_end`.
   - The D2 `task` segment spans reach through retreat and is added separately, as ACQ's `session.py` does.
3. **Robot sessions are narrower.** The synthetic thumb can only oppose index and middle, so robot D1 has no ring/pinky pinches and no wrist block. Robot grasps are thumb–index(–middle) pinch grasps for both the power and precision types.
4. **Object pose frames differ by kind.** Robot object poses are in the hand base (URDF root) frame, recorded as `meta.synthetic.object_frame = "hand_base"`, because joint_state carries no arm or base pose. Glove object poses are in the world frame.
5. **D2 glove sessions have no calibration segment.** `manifest.calibration` is still written; the flat-hand calibration hold exists only in D1.
6. **Label timestamps.** hand_pose.npz timestamps are on the host clock with no offset. Only camera timestamps carry the constant offset, recorded in `meta.synthetic.camera_offset_s`.
7. **Grasp and self-touch details.**
   - Glove power grasps can also bring the thumb pad within the 4 mm self-touch margin of the index finger. Those frames are labelled and pressed as self-touch, exactly as preprocessing would compute them.
   - The press is not purely coincident with contact: a fast press lag (10–30 ms) leaves a short tail after contact ends.
8. **Fixed skeleton.** Sessions always use the default `ManoSkeleton()`, so labels agree with preprocessing, which uses the same default. The pinch keys are hardcoded as verified defaults and re-optimised automatically if the geometry changes, which would cost about 7 s.
9. **Ground-truth artefact sign.** The artefact gain is signed at random per taxel. Each taxel's artefact is relative to the true baseline and is zero at the flat hand / q = 0.
### interface_requests
None are blocking. Notes for owners:
1. **[PRE] datasets/build.py and docs/DATA_FORMAT.md**
   - When a `Layout` object is passed, `manifest.layout` is an absolute path to `<session>/layout.yaml`, and `meta.synthetic.layout_file` gives the session-relative name. Please resolve `manifest.layout` relative to the session directory first, so datasets can be moved.
   - IMU quats use the device hemisphere convention (w ≥ 0), so a continuity fix is required.
   - Camera timestamps include a constant clock offset (±20–30 ms) that is not pre-corrected.
   - Robot D2 `object_pose` is in the hand base frame.
   - `gt_synthetic.npz` is a sidecar, not a stream, and should be ignored or copied as needed.
2. **[datasets/__init__.py owner]** Please do not import `synthetic` eagerly; it pulls in torch and the pose code. Users import `robot_skin.datasets.synthetic` directly.
3. **[ACQ]** For compatibility, my event values and phase ids mirror `recorder.py` and the protocol YAMLs. Pinches are named `pinch_<finger>` rather than `pinch`. Please keep `{contact, labels}` in phase_start values stable.
4. **[STAGE1 / INTEG]** For end-to-end tests, use `generate_dataset(root, n_motion, n_task, kind, duration_s=3..6, cameras=(...))`. `load_ground_truth(dir)["artefact_pct"]` is the target a trained baseline stage should recover on no_contact frames; `contact` / `self_touch` / `object_contact` are the ground-truth labels.
### open_todos
- The robot hand's thumb cannot oppose ring/pinky, so robot grasps are pinch grasps. A 5-DoF thumb or a relocated thumb base would allow true power grasps.
- The capsule self-touch model inherits POSE's crude geometry. Fist blocks produce sparse self-touch labels, mostly thumb/index tips and palm.
- The tactile model is phenomenological: signed per-taxel gains plus a first-order lag, with no hysteresis or viscoelastic tail. Parameters should be fitted to real mk555 glove recordings once D1 data exists.
- Camera frames are a procedural proxy for real frames (finger bars plus an object disk), not a rendered hand.
- The D2 glove sessions could also include ACQ's `imu_calibration` and `sync_start` pre-blocks, and synthetic 3-tap sync transients for testing `acquisition.sync`. Both are left out to keep sessions tiny.
### reviewer fixes
- Glove D2 'baseline' (_GloveMotion._g_task_start) now holds a flat hand for 65 % of the block, then moves to the relaxed start pose, the same as the robot's static_start. Every session (D1/D2, glove/robot) now begins its first no_contact segment with zero artefact. New parametrized test test_first_no_contact_segment_is_artefact_free covers all 4 session kinds: the artefact is 0 during the hold and the median of the first no_contact segment equals the true baseline (rtol 2e-3).
- no_contact spans are now trimmed around geometric contact and around any press tail with |press| > PRESS_EPS_PCT (0.02 %, below the noise floor; exported in __all__). The sweep shows nothing is trimmed at ≥ 6 s; 2.5 s sessions lose ≤ 0.11 s. New parametrized test test_short_session_no_contact_stays_truthful (glove and robot) checks the press bound, the trimmed amount and that trimmed spans stay inside their blocks.
- Glove reach (_g_reach) now travels 5 cm (REACH_LIFT_M) above the straight-line path and descends almost vertically onto the pre-grasp pose. New regression test test_reach_clears_the_object covers open_jar / in_hand_rotate / grasp_lift_place with seed 3: no object contact in baseline, reach, retreat or baseline_end.
- Robot self-contact geometry:
- New helper _robot_taxel_inset: finger-link taxels are on the link axis (robot_pad_inset), palm taxels on the surface (0).
- Self-touch labels and penetration use capsule radius robot_link_radius + inset + margin; _capsule_penetration gained a per-taxel inset argument.
- Object contact uses the same helper.
- SynthParams: robot_contact_radius is replaced by robot_link_radius = 8 mm, and robot_pinch_depth = 3 mm is added.
- The robot fist key was deepened to (1.6, 1.8, 1.4) so it still presses the distal palm taxels (penetration 3–4 mm).
- Robot pinches now close to a per-session contact pose, _RobotMotion._pinch_target: the first pose along q_n → key where the on-axis pads are 2r − robot_pinch_depth apart. Pinch penetration is now 7–12 mm, similar to the glove's. New test test_robot_self_contact_geometry recomputes capsule distances independently from joint_state FK and link lengths from the URDF. It checks that labels match exact thresholds (2r + margin on axis, r + margin on the palm), that pinch and fist produce contact, and that the thumb–index pad separation stays between 4 and 16 mm.
- Event names now match the Recorder ('instruction', 'success'), and the glove task test asserts this. Offline check: acquisition.recorder.segments_from_events reproduces manifest.segments, and acquisition.qc.session_qc passes with 0 errors / 0 warnings on all 4 session kinds.
- generate_session now runs with torch.set_num_threads(1), restored afterwards (decorator _single_torch_thread). A 6 s session takes 0.16–0.22 s even under load, and results no longer depend on the machine's thread count. The ridge regression in the camera test now solves in dual form (90×90 instead of 2305×2305), so it stays fast when the CPU is loaded.
- Validation:
- task_id and subject are checked before any file is written, and subject is coerced to str.
- Camera and subject names reject path separators, '.' and '..'.
- SynthParams.__post_init__ checks for finite real numbers (numpy scalars accepted), tuple lengths, lo ≤ hi, non-negativity, positive rates and time constants, probabilities in [0, 1] and rate_error < 0.5.
- generate_dataset rejects the per-session keys (dataset/seed/subject/session_id) and negative counts with ValueError.
- The tests assert that nothing is written on validation failure.
- A YAML layout path is now recorded as an absolute path; built-in names are still recorded as given. New test test_layout_yaml_path_is_recorded_absolute passes a relative path and changes the working directory.
- GENERATOR_VERSION is now 2. Docstrings (module, Block, plan_task_session, _segments, generate_session, _robot_contact) were updated to describe the new behaviour.
### reviewer remaining concerns
1. **Default layouts hide channel-order bugs.** The default layouts (glove_template, robot_hand_template) map channels in order (identity), so default synthetic sessions cannot catch a missing layout.by_channel in preprocessing. Only custom permuted layouts exercise it (covered in test_raw_is_channel_major_and_matches_layout). PRE and INTEG end-to-end tests should also generate at least one session with a permuted Layout object.

2. **manifest.layout path rules.** When a Layout object is passed, manifest.layout is an absolute path to <session>/layout.yaml, and meta.synthetic.layout_file gives the session-relative name. PRE should resolve manifest.layout relative to the session directory first, so datasets can be moved.

3. **Glove D1 block order differs from the protocol.** The glove D1 session starts with imu_calibration (flat hand) and then baseline_start. d1_motion.yaml does it the other way round (baseline_start, sync, imu_calibration). This was kept deliberately so that the first no_contact segment is artefact-free.

4. **Real data needs an explicit ΔS reference.** In real data the reference is the rest pose in both D1 and D2. Synthetic sessions now consistently use the flat hand / q = 0 instead. STAGE1 should keep in mind that real sessions' baselines include the rest-pose artefact.

5. **Label timing.** hand_pose.npz labels sit on the host clock with no offset. Only the camera timestamps carry the constant offset (meta.synthetic.camera_offset_s). Real vision labels would inherit the camera offset.

6. **Self-touch during glove power grasps.** In glove power grasps (grasp_lift_place, pour, wipe) the thumb pad comes within the 4 mm self-touch margin of the index finger for most of the manipulate phase. These frames are labelled and pressed as self_touch, consistently with pose.mano.self_touch_from_hand.

7. **Phenomenological contact and tactile models.**
   - Penetration uses margin-inflated capsules, so the press starts slightly before the surfaces touch.
   - Glove and robot pinch penetrations of 7–12 mm stand in for pad compression.
   - The tactile model's parameters are not fitted to real mk555 glove data.
   - Robot grasps are thumb–index(–middle) pinch grasps only.

8. **Global thread setting.** torch.set_num_threads(1) is process-global while generate_session runs. That is harmless for offline or test use, but a caller running torch in other threads at the same moment would be affected briefly.

9. **Interface requests from the implementer still stand.**
   - PRE should ignore the gt_synthetic.npz sidecar.
   - datasets/__init__ should not import synthetic eagerly.
   - ACQ should keep the {contact, labels} values in phase_start events stable.
   - Robot D2 object_pose is in the hand-base frame (meta.synthetic.object_frame).

10. **API change for existing callers.** SynthParams.robot_contact_radius was removed in favour of robot_link_radius (new: robot_pinch_depth). No other module used it.


## ACQ
### public_api
robot_skin.acquisition (lazy PEP 562 exports; `import robot_skin.acquisition` does not import torch; manifest.py unchanged):

instructions.py: template_slots(tpl)->tuple; check_template(tpl, available); humanize(v) ("red_cup"->"red cup"); normalize_instruction(text); render_instruction(tpl, slots)->str (KeyError if a slot is missing); sample_instruction(templates, slots, rng)->(idx, text).

protocol.py
- Constants: PROTOCOL_DIR; CONTACT_EXPECTATIONS=("none","self","object","any"); DEFAULT_CONTACT_LABELS {none:(no_contact,), self:(self_touch,), object:(), any:()}; STEP_KINDS; SPEEDS; ADVANCE_MODES; TASK_PHASES=(reach,grasp,manipulate,release,retreat); KO_NAMES.
- ProtocolError(ValueError).
- Frozen dataclasses: SyncSpec(taps=3, intervals_s=(0.6,1.2), lead_s, tail_s, finger) with .duration_s / .tap_offsets(); Timing(lead_in_s, transition_s, lead_out_s, min_step_s); Step(id, block, kind, duration_s, contact, labels, advance, speed, prompt, prompt_en, motion, scalable) with .event_value(); TimedStep(step, t0, t1); TaskSpec.
- EpisodePlan(index, steps, task) with .timeline(timing) and .duration_s(timing). SessionPlan with .episodes, .timing, .total_duration_s, .to_dict().
- Protocol with .blocks, .phases, .tasks, .expand_blocks(), .task_steps(), .is_task.
- load_protocol(name|path) validates the schema (unknown keys, contact/speed values, for_each, prompt placeholders, template slots, unique ids).
- list_protocols(); protocol_from_dict(d).
- plan_session(protocol, *, seed, tasks, objects, n_episodes, repetitions, time_scale, instruction, shuffle)->SessionPlan: D1 = one episode; D2 = one episode per (task, object, repetition), seeded target/template/order.
- scale_steps(steps, scale, min_step_s); scale_timing(timing, scale).
- format_script(plan, lang="ko"|"en", max_episodes).
- make_session_id(dataset, subject, when, task_id, obj, rep, index).
- Built-in protocols: d1_motion, d2_task, robot_sweep.

sources.py
- STREAM_KINDS=(pressure, imu, joint_state, hand_pose, object_pose, camera); SAMPLE_KEYS.
- Clocks: MonotonicClock; SimClock(t0) with .advance(dt) / .set(t).
- StreamSource Protocol: name, kind, rate_hz, start(clock), poll()->[(t_host, dict)], stop(), info(). validate_source(src): camera sources must be named camera_<name>.
- PlaybackSource(name, kind, t, data, rate_hz, info).
- Fake sources backed by a FakeScene (default scene if none is given): FakePressureSource(scene), FakeImuSource, FakeCameraSource(scene, camera), FakeHandPoseSource, FakeObjectPoseSource, FakeJointSource. fake_sources(scene).
- CsvLineParser(n_channels, sep, time_column); FrameParser Protocol.
- SerialPressureSource(port, n_channels, baudrate, parser, rate_hz): pyserial guarded; device time or back-filled stamps.
- CameraSource(camera, device, rate_hz, size): cv2 guarded, background grab thread.
- ImuSource / RosJointStateSource: raise NotImplementedError with a TODO in the docstring.

fake.py: FakeConfig (rates, latency_s, clock_ppm, jitter, drop, image_hw, imu_mount_deg, …); FakeScene(timeline, duration_s, *, kind glove|robot, layout, cameras, task, seed, config), .from_episode(ep, timing), .default(). Stream getters: .pressure_stream(), .imu_stream(), .camera_stream(cam), .hand_pose_stream(), .object_pose_stream(), .joint_stream(), .stream(name). Also .hand_state(t), .hand_pose(t), .truth (latencies, ppm, imu_mount, imu_world, tap_times), .imu_mount, .baseline_raw; default_fake_timeline(); ROBOT_JOINT_NAMES.

recorder.py
- EVENTS_NAME; EVENT_TYPES; STREAM_FILES; stream_file(name, kind).
- EventLog(path) with .log(t, type, name, value); writes events.jsonl line by line with a flush.
- load_events(path|dir) (sorted by time); phases_from_events(events, t_end)->[{name,t0,t1,value,closed}]; segments_from_events(events, t_end).
- Recorder(sources, session_dir, manifest, *, clock, camera_format auto|npy|jpg, jpeg_quality, overwrite, sim_dt, poll_interval_s)
  - Lifecycle: .start(threaded=None), .step(), .now(), .run_until(t), .run_for(dt), .stop()->SessionManifest; also usable as a context manager.
  - Events: .event(type, name, value), .phase_start / .phase_end / .phase(ctx), .marker, .instruction(text), .success(bool|None), .add_segment.
  - Inspection: .snapshot(stream, t0, t1), .counts(), .open_phases().
  - Writes pressure.npz {t, raw}, imu.npz {t, quat, gyro, acc, sites}, joint_state.npz {t, q, qd, tau, names}, hand_pose.npz (via pose.vision_hand.save_hand_labels), object_pose.npz {t, pos, quat}, camera_<n>/{timestamps.npy, frames.npy | %06d.jpg}, events.jsonl and session.json. Segments come from the phase labels; open phases are auto-closed at stop.

calibration.py: compute_imu_calibration(quat[T,S,4], sites, layout, gyro, skeleton, wrist_site)->(calib dict via imu_calibration_to_dict with keys imu_offsets/imu_world/imu_sites, quality {quat_spread_deg, gyro_rms, ok}); find_calibration_phase(events) (last attempt wins); calibrate_session_imu(session_dir, phase="imu_calibration", trim_s=0.5, save=True)->dict|None (quality stored under imu_calibration_quality).

sync.py
- OffsetEstimate(offset_s, score, lag_samples, at_limit); float() gives offset_s. Convention: t_b + offset_s aligns stream b with stream a.
- ClockModel(scale, offset_s) with .apply / .inverse / .drift_ppm / .to_dict / .from_dict.
- change_envelope(t, x, hz, t_range, smooth_s)->(grid, env, support).
- estimate_offset(t_a, x_a, t_b, x_b, max_lag_s=0.5, hz=200, *, window, smooth_s, envelope=True).
- detect_taps(t, x, n=3, window, …).
- fit_clock_drift(t_stream, t_ref)->ClockModel; apply_offset(t, off).
- sync_windows(events); load_sync_signal(...); stream_kind(manifest, name).
- apply_clock_models(session_dir, {stream: model}): keeps t_host / timestamps_host.npy, so re-applying is idempotent.
- sync_session(session_dir, reference="pressure", streams, max_lag_s, hz, min_score=0.3, smooth_s=None (per kind: pressure/IMU 0.02, camera 0.06), apply, save)->report, stored in manifest.calibration["sync"]. Drift is fitted only when it exceeds DRIFT_MIN_DELTA_S.

qc.py: DEFAULT_THRESHOLDS; stream_timing(t, nominal_hz, gap_factor)->{n, rate_hz, jitter_ms, max_gap_s, n_gaps, dropped_pct, monotonic}; session_qc(session_dir, thresholds, write)->report {streams, checks[{name, stream, passed, severity, value, limit, message}], passed, n_errors, n_warnings, sync, events}; format_report(rep); CLI `python -m robot_skin.acquisition.qc <dirs> [--write] [--json F] [--strict] [--set k=v]`.

session.py
- Operator Protocol; AutoOperator(success=True); ConsoleOperator(input_fn, print_fn, lang, metronome).
- adhoc_plan(duration|None, no_contact, kind, cameras); planned_streams(kind, cameras, rates, imu_sites, imu).
- write_dry_run(plan, out, kind, layout, subject, streams)->session.json | plan.json.
- fake_source_factory(plan, kind, layout, cameras, seed, hand_pose, config, imu); `.scenes` holds the ground truth.
- record_episode(plan, episode, session_dir, sources, *, kind, layout, subject, operator, clock, …, sync, calibrate, qc, sync_from, calibration_from)->{session_dir, manifest, sync, calibration, qc}.
- postprocess_session(dir, sync, calibrate, qc, sync_from, calibration_from); copy_imu_calibration(dir, src).
- run_plan(plan, *, kind, source_factory, out, root, subject, layout, operator, clock_factory, …)->list: one directory per episode.
- CLI `python -m robot_skin.acquisition.session <dirs>`.

glove_logger.main(argv) / robot_logger.main(argv)
- Shared flags: --out, --root, --protocol, --subject, --layout, --seed, --time-scale, --lang, --task/--object (repeatable or comma lists), --episodes, --repetitions, --instruction, --duration, --no-contact, --dry-run, --fake, --no-sync, --sync-from, --calibration-from, --no-qc, --camera-format.
- Glove only: --cameras (default ego,third), --camera-devices, --port, --baud, --n-channels, --pressure-format, --no-imu.
- Robot only: --joint-topic; the default protocol is robot_sweep.
- Return codes: 0 = all sessions pass QC, 3 = some QC failures, 1 = nothing recorded. Real devices raise NotImplementedError (IMU / ROS stubs).
### deviations
1. Added modules beyond the spec, all inside acquisition/*, which I own:
   - fake.py: FakeScene, the synthetic generator behind the Fake*Sources and --fake. I did not use datasets.synthetic because it was still being written concurrently.
   - calibration.py: IMU calibration from the flat-hand block.
   - session.py: protocol runner, operators, and post-processing (sync → IMU calibration → QC).
   - protocols/robot_sweep.yaml: the robot logger's default protocol.
2. Sync corrects timestamps in place. `t` (npz) and timestamps.npy are rewritten onto the pressure clock; the originals are kept as t_host / timestamps_host.npy, and details go in manifest.calibration["sync"]. StreamInfo.clock stays "host". Consumers therefore need no sync logic.
3. Sync method details:
   - Envelope smoothing is set per stream kind: 20 ms for pressure and IMU, 60 ms for cameras. Cameras see hand motion around a tap rather than the contact edges, so their precision is about one frame.
   - Drift is fitted only when the start-to-end offset change exceeds DRIFT_MIN_DELTA_S; otherwise the mean offset is used. Drift above 1000 ppm is rejected.
   - estimate_offset returns OffsetEstimate(offset_s, score, lag_samples, at_limit) rather than a bare float; float() still works.
4. The contact-expectation vocabulary is none/self/object/any, mapped to segment labels (none→no_contact, self→self_touch). Calibration blocks are labelled [calibration, no_contact] and sync blocks [sync]. D2 reach/retreat carry no labels because their boundaries are operator-marked, and a single "task" segment spans all task phases.
5. Each D2 episode now starts with: 3 s rest baseline (the same rest pose as D1, so the ΔS reference is consistent), 2 s flat-hand IMU calibration, then the 3-tap sync. This can be shortened with --sync-from / --calibration-from.
6. Sync taps use unequal spacing (0.6 s / 1.2 s) so the correlation peak is unique.
7. Legacy CLI changes:
   - The glove dry-run now lists camera_ego / camera_third instead of "camera", and pressure.npz instead of pressure.bin. test_acquisition.py was updated accordingly.
   - A protocol-free real run without --duration is an open-ended manual step. It still reaches the device stub, so it raises NotImplementedError as the old test expects.
   - Logger return codes: 0 / 3 / 1.
   - --fake on a D2 catalog defaults to 3 episodes.
   - --time-scale also shrinks pauses between blocks (floored at 0.25 s).
8. Events: instruction and success events are named literally "instruction" / "success". The other agent's synthetic writer uses the task_id as the name, so consumers should key on `type`. phase_start.value carries kind, contact, labels, block, speed, motion and finger/axis.
9. Fake --fake sessions include hand_pose.npz (vision-like labels on the reference clock). Real sessions get that file offline via HaMeR/WiLoR.
### interface_requests
1. [PRE] robot_skin/datasets/build.py and docs/DATA_FORMAT.md:
   - (a) Read `t` / timestamps.npy as-is. They are already sync-corrected when calibration["sync"]["applied"]; ignore t_host and timestamps_host.npy.
   - (b) Identify events by `type`, not by `name`. phase_start.value.contact ∈ none|self|object|any, and value.labels gives the segment labels.
   - (c) The phase vocabulary includes sync_start/sync_end (index-fingertip taps: real contact, labelled "sync", never no_contact), imu_calibration (flat hand, [calibration, no_contact]) and D2 baseline/imu_calibration/sync_start before reach…retreat.
   - (d) The first no_contact segment is always the rest pose (D1 baseline_start / D2 baseline), which is the right ΔS baseline reference.
   - (e) Please consider a config option to skip sessions whose qc.json has passed=false.
   - (f) manifest.task has extra keys: target, repetition, template_index, template, instruction_source, slots, success_criteria, grasp, manipulate. calibration has an extra imu_calibration_quality key.
2. [SYNTH] Optional: add the 2 s D2 `imu_calibration` block to synthetic D2 episodes so they match the new protocol. Its instruction/success event names differ from mine (task_id vs literal), which is harmless if PRE keys on type.
3. [INTEG]
   - robot_skin/__main__.py: `record` → acquisition.glove_logger.main / robot_logger.main; `qc` → acquisition.qc.main; optionally `postprocess` → acquisition.session.main.
   - robot_skin/README.md: the acquisition row still says "(--dry-run 만 동작)". It should list protocols, recorder, 3-tap sync, IMU calibration, QC, and the --fake/--dry-run CLIs, with the IMU hub and ROS joint state as stubs.
4. [DOCS] Link docs/DATA_ACQUISITION.md from ARCHITECTURE/TRAINING docs.
5. manifest.py: no change required.
### open_todos
- Real device IO:
  - ImuSource (glove IMU hub) and RosJointStateSource are NotImplementedError stubs; each docstring states what to implement.
  - mk555 binary frames need a FrameParser built on deformable_sats/sats/preprocessing/bin_merge.py and injected via SerialPressureSource(parser=...). glove_logger --pressure-format mk555 currently raises with that instruction.
  - CameraSource (cv2) is unexercised here because cv2 is not installed.
- Camera sync precision is about one frame (the fake shows 8–22 ms bias). Sub-frame sync would need an LED flash plus the brightness envelope. Per-tap uncertainty estimates are not computed.
- ConsoleOperator:
  - Manual D2 phases block on input() with no timeout.
  - There is no foot-pedal or skip-episode control (Ctrl-C stops the plan after saving and QC'ing the current episode).
- The offline hand_pose.npz generation (HaMeR/WiLoR on camera frames) is outside ACQ. Only the fake produces hand_pose.
- The recommended data amounts in docs/DATA_ACQUISITION.md are explicitly starting points to revisit after the pilot.
- The robot kind reuses the D2 pre-blocks (flat-hand calibration and taps), which mean little for a robot hand. A robot-specific D2 preamble could be added once teleop exists.
### reviewer fixes
- recorder.py: new `TIME_EPS` (exported). run_until uses it, and ConsoleOperator's countdown loops on the same tolerance, so it always terminates.
- session.py ConsoleOperator: every Enter of a manual chain now marks a phase START. before_step asks for the first phase's start (reach); perform asks for the next phase's start, or the chain end for the last phase (n phases → n + 1 presses). Prompts show the protocol `boundary` cue (end prompt: the '(끝: …)' part). protocol.Step has a new `boundary` field filled from the phase vocabulary. The ad-hoc prompt text is updated. docs/DATA_ACQUISITION.md §7.1, the d2_task.yaml header and the README now describe this.
- qc.py: new `rest_spans(phases)` (exported). Baseline drift now compares the first vs last rest-pose (static, no_contact) block: baseline_start vs baseline_end in D1, and not checked in D2, which has one rest block. When events carry no step kinds it falls back to the no_contact segments. Spans are sorted; the report adds `baseline_drift_spans`.
- qc.py: pressure checks use only the layout's channels (`_layout_channels`; report `checked_channels`, stuck channels given as board numbers). New error check `pressure_channels_layout` fails when the board has fewer channels than the layout uses.
- qc.py: new checks `recorder_errors` and `streams_recorded` (error, from meta.recorder), and `stream_coverage` (each stream must cover the recorded phases within the new threshold `coverage_tol_s` = 0.5 s; error, warning for hand_pose/object_pose). Module docstring, README and DATA_ACQUISITION QC table updated.
- sources.py SerialPressureSource: timestamps are strictly increasing. Burst back-fill spacing shrinks to fit between the previous stamp and the arrival time. Wrong-size frames are dropped before spacing (`n_bad_frames`). A device clock that jumps back or runs more than 0.5 s ahead is re-anchored (`n_reanchors`). Both counters appear in info().
- recorder.py: overwrite=True first deletes what the recording will write (stream npz files, camera directories, events.jsonl, qc.json) and keeps other files such as offline hand_pose.npz. Docstring updated.
- robot_logger.py: docstring corrected. Robot task episodes need a protocol of kind robot/any; d2_task's blocks are human-hand only. The example was replaced with a working one.
- fake.py: IMU world frame is now a heading-only rotation about gravity (z), and the module docstring says so.
- protocol.py: top-level and `episode` keys are validated; `_pos_int` covers repetitions; phase entries must be mappings; plan_session removes duplicate `--task` entries and normalises the operator instruction (blank → ProtocolError).
- glove_logger.py: `--camera-devices` pairs are validated with a clear SystemExit. `_Buffer` initialises its column dict in one assignment. `rest_spans` added to the lazy __init__ exports.
- New tests:
- test_recorder: serial bursts, bad frames and device clock reset; overwrite removes the previous take; fake IMU quaternions agree with gravity.
- test_acquisition: scripted ConsoleOperator D2 run under a SimClock (terminates, n + 1 start boundaries, contiguous phases, idle time kept out of reach, task segment).
- test_sync_qc: D2 drift not confused by pose (also shows the old rule gives > 3 %); D1 drift spans are baseline_start/baseline_end; recorder_errors / streams_recorded / stream_coverage; layout-channel checks (wide board passes, narrow board fails, dead channel reported by board index).
- test_protocol: 6 more schema-error cases, --task dedupe, instruction normalisation / blank rejection, Step.boundary.
### reviewer remaining concerns
1. Spec deviation (intentional): QC baseline drift uses rest-pose static blocks, not "first/last no_contact segments". For D1 these are the same segments (baseline_start/baseline_end). D2 episodes have one rest block, so no drift check. Sessions without step kinds in their events (e.g. datasets.synthetic) fall back to the spec rule.

2. Behaviour change for operators: a D2 manual chain now takes n + 1 Enter presses. The first press starts `reach` at the moment the hand leaves the start pose; idle time after the sync block is in no phase. PRE should not assume the task phases start right after sync_start.

3. [SYNTH], for PRE's attention (not my file): the synthetic D2 plan differs from d2_task.yaml. Its baseline is flat hand then relaxed pose, it adds a `baseline_end` block, and it has no imu_calibration/sync_start blocks. Its instruction/success events use task_id as the name. PRE must key on event `type` and should not assume the ACQ D2 block list.

4. The new QC `stream_coverage` requires each stream to span all recorded phases within 0.5 s. Cameras that take more than about 1 s to open after Recorder.start could clip the start of the first block when lead_in_s is short; revisit with real hardware.

5. Items unchanged from the implementer's open TODOs:
   - ImuSource and RosJointStateSource are stubs.
   - mk555 frames need a parser built on bin_merge.py.
   - CameraSource is untested here (no cv2).
   - Camera sync precision is about one frame.
   - The console has no timeout or foot pedal.
   - There is no robot task protocol; d2_task stays glove-only.

6. Interface requests from the implementer still stand ([PRE] a/b/c/d/e/f, [INTEG] CLI wiring + README row, [DOCS] links).

Files edited (all ACQ-owned; manifest.py, common/ and deformable_sats/ untouched; nothing committed):
- robot_skin/acquisition/{recorder,session,sources,protocol,qc,fake,glove_logger,robot_logger,__init__}.py
- robot_skin/acquisition/protocols/d2_task.yaml
- robot_skin/acquisition/README.md
- docs/DATA_ACQUISITION.md
- robot_skin/tests/test_{acquisition,recorder,protocol,sync_qc}.py


## PRE
### public_api
## robot_skin/datasets/build.py
Everything below is importable directly from `robot_skin.datasets.build`. `datasets/__init__` was not touched.

**Constants**
- `PREPROCESS_VERSION = "robot_skin.datasets.build/1"`
- `DEFAULTS` (dict): every preprocessing knob. `configs/stages/preprocess.yaml` mirrors it and a test enforces equality.
- `CONFIG_PATH`
- `LAYOUT_FILE = "layout.yaml"`
- `GT_FILE = "gt_synthetic.npz"`
- `GT_KEYS` maps a sidecar key to an optional episode array: `artefact_pct`→`gt_artefact_pct`, `press_pct`→`gt_press_pct`, `contact`→`gt_contact`, `self_touch`→`gt_self_touch`, `object_contact`→`gt_object_contact`.
- `HAND_Q_NAMES`: the 45 glove `q` column names (`index1_x` … `thumb3_z`, MANO_JOINTS[1:] × xyz).

**Main entry**
- `preprocess_session(session_dir, out_root=None, cfg=None, *, pressure_loader=None, overwrite=False) -> Episode`
  - Writes atomically (temp dir, then rename) to `<out_root>/<dataset>/<session_id>/`.
  - `out_root=None` returns the episode in memory only.
  - An existing episode raises `FileExistsError` unless `overwrite`; overwriting wipes `derived/` too.
  - Writes arrays `t`, `pressure_raw`, `delta_pct`, `saturated`, `taxel_pos`, `taxel_nrm`, `q`, `qd`, `imu_*`, `hand_*`, `hand_pose_valid`, `object_*`, `phase_id`, `self_touch`, `contact_label`, `cam_<name>_idx`, plus the optional `gt_*` arrays.
  - Writes static `baseline_raw` and `taxel_channels`, a `layout.yaml` copy, and the camera dirs (symlink by default, or copy / none).
  - Fills `meta.phases` / `phase_names` / `task` / `preprocessing` (contents listed in docs/DATA_FORMAT.md §2.6).

**Batch run and CLI**
- `build_all(raw_roots, out_root, cfg=None, *, force=False, fail_fast=False, pressure_loader=None) -> list[{session, episode, status: built|skipped|qc_failed|failed, stale?, error?}]`
  - A skipped episode is reported `stale` when the config hash, `PREPROCESS_VERSION` or raw file set (names + sizes) differs.
- `find_sessions(root) -> list[Path]`
- `main(argv) -> int`, run as `python -m robot_skin.datasets.build`.
  - Flags: `--raw` (repeatable; default: cfg `raw_root`), `--out` (default: cfg `out_root`), `--config`, `--set k.a=v`, `--force`, `--fail-fast`, `--json`, `-q`.
  - Exit code 1 if any session failed.

**Helpers**
- `load_preprocess_config(path=None, overrides=None) -> dict`: deep-merged over `DEFAULTS`; unknown keys raise `ValueError`.
- `resolve_layout(session_dir, manifest, override=None) -> (Layout, ref)`. Lookup order:
  1. the override;
  2. a relative YAML path in `manifest.layout`, resolved against the session dir;
  3. the session-local copy (`meta.synthetic.layout_file`, or the YAML's file name inside the session dir);
  4. an absolute path;
  5. a built-in layout name.
- `load_episode_layout(ep|dir) -> Layout`
- `load_pressure_npz(session_dir, manifest) -> (t, raw[T,C])`
- `joint_velocity(q[T,D], hz, *, method="savgol_causal"|"savgol"|"gradient", window_s=0.05, polyorder=2) -> float32 [T,D]`
- `camera_frame_index(frame_t, t, *, max_age_s=None) -> int32 [T]`: −1 before the first frame; handles unsorted timestamps.
- `phase_ids(phases, t) -> (int16 [T], names)`
- `contact_labels(t, segments, self_touch|None, n_taxels, *, conflict="unknown"|"segment"|"geometry") -> (int8 [T,N], counts)`
- `episode_dir(out_root, manifest) -> Path`

## robot_skin/datasets/splits.py
- `SPLITS = ("train", "val", "test")`; `GROUP_KEYS` = subject, session, object, task, dataset, kind, episode_id.
- `episode_group(ep|dir|meta, by)`: an episode missing the key gets its own group, `episode:<id>`.
- `make_splits(episode_dirs, by="subject"|tuple, val_frac=0.15, test_frac=0.15, seed=0, holdout=None) -> {train, val, test: [str]}`
  - Groups never cross splits.
  - Deterministic and independent of input order.
  - Groups are shuffled and assigned greedily; train always keeps at least one group.
  - `holdout` is either `{field: [values]}` (sent to test) or `{"val"|"test": {...}}`.
- `check_splits(splits, by, *, ignore=())`
- `save_splits(splits, path, *, root=None, meta=None)`: with `root`, paths are stored relative to it.
- `load_splits(path, *, root=None) -> {split: [Path]}`

## robot_skin/datasets/stats.py
- Constants:
  - `IMU_FEATURES = "imu_features"`: a pseudo-key for `pose.imu_model.imu_features`.
  - `SATURATION_MASKED`: keys whose saturated samples are excluded.
  - `HAND_MASKED`: keys restricted to `hand_pose_valid` frames.
- `compute_stats(episodes, keys=("q","qd"), method="std"|"robust"|"none", *, masks="auto"|None|{key: "auto"|None|callable}, eps=1e-6, max_samples=20000, seed=0, imu_kw=None) -> {key: NormStats}`
  - Statistics are per flattened feature.
  - `std` merges per-episode results with Chan's parallel update.
  - `masks="auto"` excludes saturated samples for ΔS-like keys, and non-`hand_pose_valid` frames for hand keys and glove `q`/`qd`.
- `default_mask(ep, key)`, `episode_array(ep, key, *, imu_kw)`, `as_episodes(eps)`
- `apply_stats(ns, x)` / `invert_stats(ns, x)` work for any leading shape.
- `save_stats(stats, path, *, meta)` / `load_stats(path, *, return_meta=False)`

## robot_skin/datasets/motion.py
All windows are causal: history only, edge-padded with frame 0 at the start.

**Helpers**
- `causal_window(t, W)`
- `imu_wrist_index(ep)`
- `episode_imu_features(ep, *, gyro=True, acc=True, vec_frame="sensor", wrist_index=None) -> float32 [T,F]`
- `q_valid_mask(ep)`

**`BaselineWindowDataset`**
- Signature: `(episodes, *, window=32, stride=1, only_labels=(0,), joint_stats=None|{"q","qd"}|tuple, exclude_saturated=True, min_valid=1, require_q_valid=True, label_key="contact_label")`
- Returns `q_hist[W,D]`, `qd_hist[W,D]`, `pos[N,3]`, `nrm[N,3]`, `y[N]` (ΔS at t, 0 where invalid), `valid[N]`, `episode`, `t_index`.

**`ContactWindowDataset`**
- Signature: `(episodes, *, window=16, stride=1, joint_stats=None, z_key="residual_z", label_key="contact_label", exclude_saturated=True, min_labelled=1)`
- Needs `derived/residual_z`.
- Returns `z_hist[W,N]` (non-finite values set to 0), `sat_hist[W,N]`, `q[D]`, `qd[D]`, `label[N]`, `label_mask[N]`, `episode`, `t_index`.

**`ImuPoseWindowDataset`**
- Signature: `(episodes, *, window=32, stride=1, imu_stats=None|NormStats|{"imu_features"}, gyro=True, acc=True, vec_frame="sensor", wrist_index=None)`
- Uses only `hand_pose_valid` frames.
- Returns `feat[W,F]`, `finger_pose[15,3]`, `global_orient[3]`, `episode`, `t_index`.

**Attributes:** `.index` [M,2] of (episode, frame); `.joint_dim`, `.n_taxels`, `.feature_dim`.
### deviations
1. **`qd` uses the causal Savitzky–Golay derivative by default.** The spec asked for "SG derivative"; the default here is `qd.method: savgol_causal`, a polynomial fit over the last 50 ms evaluated at the newest sample. The online processor (CTRL) must match the offline stages, and a centred SG needs future samples. `savgol` (centred) and `gradient` remain options. `joint_velocity` is public so CTRL can reuse it.
2. **Master clock.** The span is the overlap of the `clock.reference` streams (pressure, imu, joint_state). The grid starts rounded up to a multiple of 1/hz and stays on the session clock (not re-zeroed), so event and segment times apply directly. hand_pose, camera and object streams may not cover the whole span; hand pose gets `hand_pose_valid`, cameras get idx −1, object pose is edge-held.
3. **Baseline window.** The baseline is the median over the first `baseline.duration_s` (1 s) of the first no_contact segment, using only frames with no saturated taxel. It falls back to `manifest.baseline`, then to the start of the recording. `baseline.phase` can name another phase. A taxel with baseline ≤ 0 (dead channel) is marked saturated with ΔS = 0 instead of making the build fail.
4. **Saturation flag.** `saturated` also marks master frames that interpolate between a rail sample and a valid one, so dropouts do not leak into half-interpolated values.
5. **Contact-label conflicts.** A geometric self-touch inside a no_contact segment is set to −1 by default (`labels.conflict`, configurable). Robot sessions have no geometric self-touch, so their labels are only 0 or −1.
6. **Extra arrays and files in the episode.**
   - Optional `gt_*` arrays for synthetic sessions (`synthetic_gt.store`). These are not synonyms of `K_*` keys.
   - A `layout.yaml` copy in every episode. `meta.layout` is the built-in name, or the absolute path of that copy for custom layouts.
   - Camera dirs are symlinked by default (`cameras.copy_frames: symlink|copy|none`).
7. **Things the spec did not ask for.**
   - `meta.preprocessing` also records `source_fingerprint`.
   - `build_all` reports `stale` episodes.
   - `qc.skip_failed` skips sessions whose qc.json says `passed: false` (ACQ request).
   - `imu.calibrate_if_missing` computes the calibration with ACQ's `calibrate_session_imu(save=False)`.
   - The CLI has `--set`, `--json` and `--fail-fast`.
8. **`splits.py`.** Episodes without the group key get one group each. Holdouts override the grouping. `save_splits` can store paths relative to a root.
9. **`stats.py`.** Statistics are per flattened trailing feature (`hand_finger_pose` gives 45 features). `apply_stats` / `invert_stats` handle any leading shape. There is no minmax method; ActionNormalizer already has one.
10. **`motion.py` extras.** Samples also carry `episode` / `t_index`, and ContactWindowDataset returns `sat_hist`. BaselineWindowDataset by default requires `q` to be a measurement over the whole window (`require_q_valid`, i.e. `hand_pose_valid` for glove). Normalisation stats are never fit inside a dataset; `None` keeps raw units.
11. **Instruction precedence.** `manifest.task.instruction` wins. The last `instruction` event is used only when the manifest has none, and then `instruction_source: "events"` is added if absent. ACQ's own `instruction_source` is kept.
### interface_requests
None of these block anything.

1. **[INTEG]**
   - `robot_skin/__main__.py` `preprocess` should forward to `robot_skin.datasets.build.main(argv)`. Its flags are `--raw`, `--out`, `--config`, `--set`, `--force`, `--json`, `-q`; `--raw` and `--out` default to the cfg `raw_root` / `out_root`.
   - Add a datasets row to `robot_skin/README.md` covering build, splits, stats and motion, and link `docs/DATA_FORMAT.md`.
   - End-to-end tests can call `build_all(raw_root, processed_root)`. Synthetic episodes carry `gt_artefact_pct` and `gt_contact` for evaluation.
2. **[datasets/__init__.py, no named owner]** Optional: lazy (PEP 562) exports of build, splits, stats and motion. They are not exported today; import the submodules directly. `synthetic` must stay out of eager imports, as SYNTH asked.
3. **[STAGE1]**
   - Fit stats on the train split with `stats.compute_stats(train, keys=("q", "qd", IMU_FEATURES))`, then pass them to `BaselineWindowDataset(..., joint_stats=stats)`, `ImuPoseWindowDataset(..., imu_stats=stats)` and `ContactWindowDataset(..., joint_stats=stats)`.
   - ContactWindowDataset needs `derived/residual_z`, which is press-positive.
   - The ground-truth artefact for evaluation is in the `gt_artefact_pct` episode array; the raw sidecar is also reachable via `meta.source_session`.
4. **[CTRL]** To match offline stages online:
   - Compute qd with `datasets.build.joint_velocity(q_buffer, 200, method="savgol_causal", window_s=0.05, polyorder=2)`.
   - Compute the baseline as `common.signal.estimate_baseline` over the first no-contact second.
   - Compute ΔS with `relative_change`, and saturation as rails OR |ΔS| ≥ 90.
   - Read the effective values from `meta.preprocessing.config` of the training episodes.
5. **[SYNTH]**
   - The first no_contact segment of D2 sessions (glove and robot) and robot D1 `baseline_start` includes the move from the flat hand / q = 0 to the relaxed pose, where the artefact is non-zero. With the default 1 s baseline window this biases ΔS by about 0.03 % in short sessions; my tests use `baseline.duration_s` 0.2–0.3 s.
   - Suggestion: label only the flat hold `no_contact`, or split the transition into its own unlabelled phase.
6. **[ACQ]** `acquisition.calibration.calibrate_session_imu` and `qc` load `manifest.layout` with plain `load_layout`. Resolving it relative to the session dir, as `datasets.build.resolve_layout` does, would keep moved datasets working. build.py passes the layout explicitly, so it is unaffected.
### open_todos
1. **mk555 `.bin` sessions.** They need a thin loader wrapping `deformable_sats/sats/preprocessing/bin_merge.py`, set in `pressure.loader`. None exists yet, and I did not copy bin_merge.
2. **Non-causal hand labels.** `smooth_hand_labels` (6 Hz zero-phase low-pass) and the optional pressure low-pass are non-causal. That is fine for vision labels, which are offline-only. Pressure low-pass is off by default and flagged in `notes` when enabled.
3. **Glove sessions without `hand_pose.npz`.** Their episodes have no `q`/`qd` and use constant flat-rest taxel poses (`taxel_pose_source: "rest"`). After HaMeR/WiLoR labels are added, `build_all` reports these episodes `stale`; rebuild them with `--force`, or they could later be filled from the derived `hand_finger_pose_imu`.
4. **Robot self-touch.** No robot self-touch labels exist: there is no generic URDF capsule model. Robot `contact_label` is only 0 / −1.
5. **Real-data frames for `imu_quat`.** For real data the calibrated `imu_quat` world frame is the one defined by the flat-hand reference (`imu_reference_rotations`, global_orient = 0), not the vision world frame. The wrist-relative IMU features are unaffected; this is documented in DATA_FORMAT.
6. **Performance at scale.** Not profiled beyond a 63 s ACQ fake session: about 1 s of compute plus about 1.5 s one-time scipy import. Multi-process `build_all` (for example `--jobs`) could be added for large corpora.
### reviewer fixes
- robot_skin/datasets/build.py _taxel_poses: glove taxel poses and self-touch are now computed with global_orient = 0 and the wrist at the origin, i.e. the MANO wrist-joint frame. Self-touch is frame-invariant, so its results are unchanged. New meta.preprocessing.taxel_frame (mano_wrist | urdf_root | layout). The module docstring gives the world-pose formula R(hand_global_orient)*taxel_pos + hand_wrist_pos. PREPROCESS_VERSION bumped to robot_skin.datasets.build/2.
- build.py URDF resolution: _resolve_session_file(as_given=True) for cfg paths tries session-relative, then the file name inside the session, then as given (absolute or CWD-relative). A missing explicit cfg URDF raises FileNotFoundError. A missing manifest.meta.urdf still falls back, now with log.warning plus the note.
- build.py: new public qd_support(hz, method, window_s, polyorder) -> (past, future), the derivative filter footprint. It shares _qd_window with joint_velocity. stats.py: new q_valid_mask (moved from motion.py, still re-exported there) and qd_valid_mask (hand_pose_valid eroded by the footprint taken from meta.preprocessing.config.qd). default_mask uses it for glove qd. motion.BaselineWindowDataset requires qd_valid over the whole window (which implies q valid). ContactWindowDataset samples gain a 'q_valid' bool.
- build.py pressure: non-finite raw samples are bridged by interpolation in native time, OR-ed into the rail mask (so they are flagged saturated like dropouts), and noted. delta_pct is always finite and the baseline is unaffected. motion.BaselineWindowDataset also excludes non-finite targets.
- splits.py make_splits: a split's first group is the first shuffled group with at most 2*frac*n episodes, else the smallest. Behaviour on balanced corpora is unchanged (the old [9,3,3] test still holds). Docstring updated.
- build.py _jsonable handles Layout (via to_dict) and falls back to str for unknown objects. layout_source reports 'override:<name|path>'.
- build.py _process_hand: an empty hand_pose.npz is treated as absent (note; no q; rest-hand taxel poses).
- build.py build_all: detects a within-run <dataset>/<session_id> collision and reports the second session as failed ('episode id collision'); the first episode is left intact. find_sessions raises FileNotFoundError for a missing root. main() prints an error and returns exit code 2 for a missing --raw root.
- motion.py: BaselineWindowDataset checks that episodes share joint dim, meta.joint_names and n_taxels. ContactWindowDataset checks the derived z shape [T,N], joint dim and n_taxels. ImuPoseWindowDataset checks that meta.imu_sites match and warns about episodes with imu.calibrated == False.
- build.py: calibrate_if_missing passes skeleton=_skeleton(cfg).
- Docs: docs/DATA_FORMAT.md covers taxel_pos frame + taxel_frame, the q/qd validity masks, the saturated semantics including non-finite samples, the URDF resolution rules, the absolute camera symlink caveat, version /2 and layout_source. robot_skin/datasets/README.md covers taxel frame, qd mask, q_valid and the dataset consistency rule. robot_skin/configs/stages/preprocess.yaml robot.urdf comment updated (a test still checks the YAML equals DEFAULTS).
- Tests (all PRE-owned files):
- test_build.py:
  - Taxel poses are compared with ground-truth world poses mapped into the hand frame, R^T (p - wrist).
  - Glove D2 taxel centroid travel is < 0.1 m while the world centroid travels > 0.2 m.
  - A CWD-relative cfg URDF resolves (monkeypatch.chdir); a typo raises FileNotFoundError.
  - NaN-loader case: finite delta_pct, NaN neighbourhood saturated, baseline within 1e-4.
  - New test_layout_object_override_and_empty_hand_labels.
  - New test_build_all_detects_episode_id_collisions.
  - A missing --raw root gives exit code 2, and build_all raises.
- test_splits_stats.py:
  - qd_valid_mask equals frames 10..25 (derived by hand); qd stats equal NormStats.fit on that mask.
  - New test_splits_keep_a_dominant_group_in_train (8 seeds; the pre-fix code fails seeds 1 and 4).
- test_motion_datasets.py:
  - Hand-derived exclusions for causal (t 12..26) and centred savgol (t 7..20) qd footprints.
  - ContactWindowDataset q_valid flags (30..39) and the z-shape check.
  - New test_datasets_reject_mixed_episodes (joint_names, n_taxels, IMU sites, uncalibrated-IMU warning).
### reviewer remaining concerns
1. [CTRL / POSE] Online taxel poses must also be in the hand frame to match training.
   - Use ManoPoseProvider / GloveImu2ManoPoseProvider with global_orient = 0 and the wrist at the origin. GloveImu2ManoPoseProvider's default global_from_imu=True gives IMU-world poses.
   - [STAGE1] contact/pseudo_label hand-object proximity needs world poses: R(hand_global_orient)*taxel_pos + hand_wrist_pos, or ManoSkeleton FK on the hand_* arrays. object_pos stays in the world frame (hand base for synthetic robot sessions).

2. [STAGE1] New PRE outputs to consume:
   - ContactWindowDataset samples have an extra 'q_valid' bool key; consume or ignore it.
   - Use stats.qd_valid_mask / q_valid_mask when masking glove q/qd outside the provided datasets.
   - qd_support is exported for CTRL/STAGE1 if they need the filter footprint.

3. A missing manifest.meta.urdf still degrades to driver-order q and static poses (note + log warning), as the implementer designed and tested. Mixed-order episodes are now rejected by the motion datasets through meta.joint_names, but VTLA/REPR datasets do not check this. Consider making it an error.

4. The baseline is estimated on the master clock, which is cropped to the pressure ∩ IMU / joint_state overlap. If the IMU starts late, the baseline window shrinks (0.035 % ΔS change in my probe). Using native pressure samples of the first no_contact segment would be slightly more robust.

5. Camera dirs are absolute symlinks by default, so they break if the raw root moves (now documented). qd_support's causal footprint is w frames (w − 1 plus a one-frame margin for the no-scipy fallback), i.e. one frame conservative.

6. Kept as documented by the implementer:
   - Non-causal zero-phase smoothing of hand labels.
   - The SYNTH D2 / robot D1 first no_contact segment includes the move to the rest pose (about 0.02-0.03 % ΔS bias with the 1 s window).
   - No robot self-touch labels.
   - No mk555 .bin loader yet.

7. The implementer's interface requests still stand:
   - [INTEG] robot_skin/__main__.py 'preprocess' → datasets.build.main (exit codes now 0 / 1 / 2) and a README datasets row.
   - [datasets/__init__] Optional lazy exports.
   - [CTRL] qd via joint_velocity(method='savgol_causal').
   - [SYNTH] Label only the flat hold as no_contact.
   - [ACQ] Resolve manifest.layout relative to the session dir in calibration / QC.

8. Only PRE-owned files were edited (build/splits/stats/motion.py, datasets/README.md, configs/stages/preprocess.yaml, docs/DATA_FORMAT.md, and the three PRE test files). The modified tracked files under robot_skin/representation/ belong to the concurrent REPR agent and were not touched. Nothing was committed.


## REPR
### public_api
`robot_skin.representation` re-exports everything below and still exports `TaxelTokenizer`, `fourier_features` and `random_taxel_mask` unchanged.

**encoder.py**
- **Constants:**
  - `OBS_MODES` is the same tuple as `policy.OBS_MODES`.
  - `TACTILE_FRAME_DIMS = {full: 6, ordinal: 4, binary: 1, none: 0}`.
  - `Z_CLIP = 100.0`, `Z_SCALE = 2.0`, `ENCODER_STATE_NAME = "encoder_state.pt"`, `ENCODER_FORMAT`.
- `tactile_value_dim(obs_mode="full", history=1) -> int` returns frame_dim × history.
- `tactile_value_features(residual_z[...,N], level[...,N], saturated=None, *, obs_mode="full", z_clip, z_scale) -> [...,N,F]`
  - This is the single feature function for pretraining, VTLA and control. numpy input gives float32 numpy; torch input gives torch on the same device.
  - `full` = `[zf, onehot(NONE, WEAK, STRONG, SAT), sat]`, with `zf = asinh(clip(z, ±z_clip) / z_scale)`.
  - `ordinal` = one-hot 4 (identical to `OrdinalQuantizer.one_hot`).
  - `binary` = `1[WEAK|STRONG]` (identical to `policy` binary).
  - `none` returns `[...,N,0]`; consumers must disable the tactile branch.
  - A saturated taxel (flag or level SAT) gets zf = 0, level forced to SAT and sat = 1. NaN z gives zf = 0. A level outside 0..3 gives an all-zero one-hot.
- `z_feature(z, z_clip, z_scale)` and its inverse `z_from_feature(f, z_scale)`.
- `history_indices(t, history, stride) -> [..., k]`: oldest first, clipped at 0.
- `stack_history(frames[..., K, N, F]) -> [..., N, K·F]`.
- `episode_tactile_arrays(ep) -> (residual_z, contact_level, saturated|None)`. Raises KeyError pointing to the contact stage if the derived arrays are missing.
- `@dataclass(frozen) TactileFeatureSpec(obs_mode="full", history=1, stride=1, z_clip, z_scale)`
  - Properties: `.frame_dim`, `.dim`, `.span`.
  - `.features(z, lv, sat)`, `.from_arrays(z[T,N], lv, sat, t_index) -> [..., N, dim]`, `.from_episode(ep, t_index)`.
  - `.to_dict()` / `.from_dict()` (unknown keys raise ValueError).
- `TactileHistory(spec)` for online use: `.push(z[N], lv[N], sat[N]) -> [N, dim]`, which equals the offline result at the same tick (tested with numpy and torch); `.reset()`.
- `TaxelEncoder(value_dim, d_model=64, depth=2, heads=4, *, n_fourier=8, fourier_scale=0.05, n_taxels=None, ff_mult=4, dropout=0.0, feature_spec=None)`
  - Structure: TaxelTokenizer → pre-LN `nn.TransformerEncoder` (batch_first, GELU) → LayerNorm.
  - `forward(values[B,N,F], pos, nrm, mask=None, key_padding_mask=None) -> [B,N,D]`. In `key_padding_mask`, True means padding or hidden: those taxels are removed as keys and their outputs are zeroed. Hiding every taxel of a sample raises ValueError. Inputs are cast to the parameter dtype, so Fourier phases stay fp32 under bf16 autocast.
  - Properties: `.value_dim`, `.d_model`, `.out_dim`, `.config` (JSON-safe, includes `feature_spec`), `.feature_spec`, `.pretrain_meta`.
  - `from_config(cfg)`; `encode(residual_z, level, saturated, pos, nrm)` for history = 1.
  - value_dim 0 (obs_mode none) raises ValueError.
- `save_pretrained_encoder(path|dir, encoder, meta=None) -> Path`: atomic, via train.checkpoint. Writes `{format, version, config, state_dict, meta}`.
- `read_encoder_state(path|dir)`: uses `torch.load(weights_only=True)` and validates the format.
- `load_pretrained_encoder(path|dir, *, map_location="cpu", strict=True, freeze=False) -> TaxelEncoder`.

**pretrain.py** (MAE, He et al. 2022)
- `MASK_MODES = ("random", "group", "mixed")`; `random_taxel_mask` is unchanged.
- `sample_taxel_mask(B, N, ratio, *, pad_mask=None, groups[B,G,N]=None, mode, group_prob=0.5, generator=None, device=None) -> bool[B,N]`
  - Per sample, k = round(ratio·n), at least 1, capped at n−1 so one taxel stays visible.
  - `group` hides one eligible layout group whole, then tops up with random taxels to k. `mixed` uses group masking with probability `group_prob`.
  - Padding is never masked, and the result is deterministic for a given generator.
- `layout_group_matrix(layout, *, max_group_frac=0.5) -> (names, bool[G,N])`: drops groups larger than `max_group_frac` (e.g. `all`, and glove `fingertip` at 5/9) and duplicate memberships.
- `TaxelPretrainDataset(episodes|dirs, spec, *, frame_stride=1, contact_repeat=1, use_groups=True, max_group_frac=0.5, layout=None)`
  - Samples: `values [N,dim]`, `pos`/`nrm`, `target_z` (zf of the current frame), `z_valid` (finite and not saturated), `target_level` (SAT forced; −1 = unknown), `groups [G,N]`, `episode`, `t`.
  - Also provides `.level_counts()` and `.group_names`.
  - Warns when the layout cannot be resolved, and when `taxel_pos` looks world-framed (taxel centroid moves more than 0.15 m).
- `collate_pretrain(samples)`: pads the taxel axis (`pad_mask`, True = padding) and the group axis, so glove and robot layouts can share a batch.
- `level_class_weights(counts, *, power=0.5, max_weight=10) -> [4]`: weights ∝ freq^-power, normalised so the expected weight is 1.
- `MaskedTaxelPretrainer(encoder, value_dim=None, *, mask_ratio=0.3, mask_mode="random", group_prob=0.5, decoder_dim=None, decoder_depth=1, decoder_heads=None, ff_mult=4, dropout=0.0, huber_delta=1.0, z_weight=1.0, level_weight=1.0, level_class_weights=None, eval_seed=0)` (nn.Module)
  - The encoder sees only visible taxels: hidden taxels are key-padded and their values replaced by `[MASK]`. The decoder is a light pre-LN transformer with a mask token plus a decoder pose embedding. Heads predict zf (Huber) and level (weighted CE) on hidden taxels.
  - `forward(batch, mask=None) -> {loss, z_huber, level_ce, z_mae, level_acc, mask_frac}`.
  - Also `.predict(batch, mask=None, *, generator=None) -> {z_pred, level_logits, mask, pad}`, `.sample_mask()`, `.loss_from_predictions()` and `.config`.
  - In eval mode, masks are re-seeded from `eval_seed` on every call. Every trainable parameter receives a gradient, so it is DDP-safe.
- `pretrain_loss(model, batch)`: the Trainer loss_fn; calls forward only.
- `evaluate_reconstruction(model, data, *, batch_size=256, device=None, seed=0) -> dict`: exact count-weighted metrics.
  - Keys: `loss`, `z_huber`, `z_mae`, `level_ce`, `level_acc`, `level_bal_acc`, `recall_<level>`, `contact_precision/recall/f1`, `n_z`, `n_level`.
  - Baselines: `z_huber_zero`, `z_mae_zero`, `level_acc_majority`.

**stages/pretrain.py**
- `STAGE = "pretrain"`, `CONFIG_PATH`, `DEFAULTS` (mirrored by `configs/stages/pretrain.yaml`), `METRICS_NAME`.
- `load_stage_config(path=None, overrides=None)`; `resolve_config(cfg)`: merges over DEFAULTS, applies the hardware profile once (only if `hardware` is set and `hardware_applied` is not), and sets `out_dir` (default `train.out_dir`).
- `discover_episodes(data_cfg)`; `load_usable_episodes(dirs) -> (episodes, skipped)`.
- `split_episodes(eps, data_cfg)`: takes `data.splits` (a splits.json with train/val/test given as dirs or ids; test excluded unless `use_test`), or else a seeded `val_frac` split by episode or subject.
- `run(cfg) -> metrics`
  - Trains with `train.Trainer` (`TrainConfig.from_dict(cfg["train"])`, `collate_pretrain`, `extra_state` = encoder/pretrainer/feature configs).
  - Loads `ckpt_best.pt` (EMA weights if enabled) on rank 0, evaluates on val (train if there is no val set), and writes `<out_dir>/encoder_state.pt` and `<out_dir>/metrics.json`.
  - Metrics include top-level `val/loss` and the other `val/*` keys, `n_episodes`, `n_samples`, `skipped`, `best`, `steps`, `feature_spec`, `level_class_weights` and `encoder_path`.
- `main(argv)`: `python -m robot_skin.stages.pretrain --config y --set a.b=v`.

**configs/stages/pretrain.yaml** sections: `stage`, `hardware`, `hardware_applied`, `out_dir`, `data`, `features`, `model`, `pretrain`, `eval`, `train`.
### deviations
1. **MAE-style pretraining instead of BERT-style.** The spec asks for masking with `random_taxel_mask` and reconstructing masked taxels. I followed MAE (He et al. 2022), which the spec also cites.
   - The encoder never attends to hidden taxels: they are key-padded, so visible tokens are exactly the encoder applied to the visible subset. Their values are also swapped for the tokenizer `[MASK]`.
   - A light decoder with its own mask token and pose embedding does the reconstruction.
   - The tokenizer mask token stays in the autograd graph (its gradient is zero), so DDP needs no `find_unused_parameters`.
   - `random_taxel_mask` is kept unchanged. The new `sample_taxel_mask` adds padding, layout-group and mixed modes.
2. **`tactile_value_features` details.**
   - `full` is 6-D per taxel: `[zf, one-hot 4, sat]`.
   - z is compressed as `asinh(clip(z, ±100) / 2)` rather than a linear clip, so strong presses keep their magnitude.
   - `none` returns zero-width features, and `TaxelEncoder` refuses `value_dim` 0. Consumers disable the tactile branch in that case; I did not use an all-zeros-plus-flag encoding.
   - A saturated taxel gets zf = 0 and level SAT, the same rule as `policy.tactile_features`.
3. **Temporal stacking lives in `TactileFeatureSpec`** (history k, stride s, oldest frame first, causal edge padding), not in the encoder constructor. The spec is stored in the encoder config, so consumers rebuild exactly the same features from `enc.feature_spec`.
4. **Reconstruction targets.** The Huber target is the compressed zf of the current frame, not raw z; it is invertible with `z_from_feature`. Saturated and non-finite z are excluded from the Huber term. Level CE uses optional "balanced" class weights (inverse square-root frequency).
5. **Stage-runner extras beyond the spec.**
   - Episode discovery also accepts an explicit `data.episodes` list.
   - Splits come from a splits.json when given (test episodes excluded unless `use_test`); otherwise a seeded split by episode or subject.
   - Options `frame_stride` and `contact_repeat`.
   - A `hardware_applied` flag so the profile is applied only once.
   - A small CLI `main()`.
   - `metrics.json` `val/*` values come from `evaluate_reconstruction`, which is an exact count-weighted average with a single mask generator. The history `val/loss` is the Trainer's batch-weighted mean, so the two differ slightly.
6. **tokenizer.py was not edited.** It is not in my ownership list. The Tancik et al. Fourier-feature citation is in the encoder.py docstring instead.
### interface_requests
1. **[PRE] robot_skin/datasets/build.py `_taxel_poses` — high priority.**
   - Glove `taxel_pos`/`taxel_nrm` are computed with `taxel_poses_from_hand(layout, sk, go, fp, wp)`, which puts them in the world/camera frame. The Episode contract (`episode.py`) says "hand/robot base frame".
   - In a synthetic D2 task episode the taxel centroid travels 0.43 m; `TaxelPretrainDataset` now warns about this. With world-frame positions the Fourier position features mostly encode where the hand is in the room, and they will not match robot base-frame poses at deploy time.
   - Please store hand-frame poses by passing `go=None` and `wp=None`, or zeros; self-touch is frame-invariant and can keep using go/wp. If you keep world frame, update the contract and record the frame in `meta.preprocessing`.
   - Also, `meta.layout` should stay resolvable by `common.layouts.load_layout`: a built-in name, an absolute path, or a path relative to the episode dir. This is what enables group masking. It works today.
2. **[STAGE1] contact stage.** Write `derived/residual_z` (press-positive calibrated z) and `derived/contact_level` (int8 ContactLevel, saturation mapped to SATURATED) for every processed episode, D1 and D2. Pretraining skips episodes that lack them.
3. **[VTLA]**
   - Build tactile inputs with `enc = load_pretrained_encoder(pretrain_out_dir, freeze=...)`, then `spec = enc.feature_spec` and `spec.from_episode(ep, tick_indices) -> [.., N, spec.dim]`. Without a pretrained encoder, use `TactileFeatureSpec(obs_mode, history, stride)` and `TaxelEncoder(spec.dim, ..., feature_spec=spec)`.
   - For obs_mode `none`, `tactile_value_dim == 0`: disable the tactile branch.
   - `contact[N]` = level ≥ WEAK. Pass `key_padding_mask` for mixed-layout batches.
   - `TaxelEncoder.tokenizer.mask_token` gets no gradient unless you pass a `mask`. Under DDP, either set `find_unused_parameters` (which modality dropout already needs) or `enc.tokenizer.mask_token.requires_grad_(False)`.
4. **[CTRL] online.py.** Use one `TactileHistory(encoder.feature_spec)` per episode (reset at start) and call `.push(residual_z_t, level_t, saturated_t)` every 200 Hz tick; it is tested equal to the offline `from_arrays`. Send taxel poses in the same frame used for training (see item 1).
5. **[INTEG] robot_skin/__main__.py.**
   - `train pretrain` should call `robot_skin.stages.pretrain.run(cfg)` with a cfg from `load_stage_config(path, overrides)`.
   - If the CLI applies the hardware profile and its own overrides itself, it should set `cfg["hardware_applied"] = True`; otherwise `run()` applies `cfg.hardware` once.
   - robot_skin/README.md: change the representation row to implemented (encoder, tactile_value_features, MAE pretraining, stage runner).
   - configs/default.yaml `representation:` is not read by the stage, which uses configs/stages/pretrain.yaml.
6. **[REFS/DOCS] docs/REFERENCES.md.** The MAE row says the model reconstructs with a "[MASK] 값 벡터". The implementation is MAE-style: hidden taxels are removed from encoder attention and a decoder mask token does the reconstruction. Please adjust the wording. Separately, robot_skin/representation/tokenizer.py `fourier_features` should cite Tancik et al. arXiv:2006.10739; that file is not in my ownership.
### open_todos
1. **Pose frame.** The world-frame `taxel_pos` question (interface request 1) must be settled before real D2 pretraining. Until then, D2 glove episodes trigger the warning.
2. **Not tested on GPU or DDP here.**
   - The code is written for them: rank-0-only IO, barriers, and every parameter receives a gradient.
   - Under torch.compile, the mask sampling and key-padding checks cause graph breaks (`.any()` host syncs), which is harmless.
3. **Untuned hyperparameters.** None were tuned on real data: mask ratio 0.3, mixed group masking, z compression (100 / 2), class-weight power 0.5, and frame_stride 4.
4. **One-frame targets.** With history > 1, a hidden taxel's whole history is hidden, but only the current frame is reconstructed. Predicting future frames is a possible extension.
5. **Transductive option.** `use_test: true` allows pretraining on test episodes; it is off by default.
### reviewer fixes
- robot_skin/stages/pretrain.py:
- New apply_hardware(cfg): exports the profile env (setdefault), applies the train/suggest.pretrain keys and sets hardware_applied, handling 'auto' and inline mappings.
- load_stage_config now builds DEFAULTS ⊕ YAML ⊕ profile ⊕ overrides, so --set wins over the profile.
- resolve_config applies the profile once, only when it has not been applied.
- Module docstring states the precedence.
- robot_skin/stages/pretrain.py:
- _match takes several bases; split entries resolve against processed_root and the splits.json directory.
- _finite is recursive (numpy scalars too); run() returns and writes strict JSON (allow_nan=False).
- run() raises a clear ValueError when the train split yields zero frames.
- robot_skin/representation/pretrain.py:
- New _masked_target helper: unused targets are zeroed before the Huber term, in both loss_from_predictions and evaluate_reconstruction.
- evaluate_reconstruction returns loss = NaN when there is no z or level target at all.
- New MaskedTaxelPretrainer.loss(batch, mask=None) alias.
- robot_skin/representation/pretrain.py, TaxelPretrainDataset:
- _resolve_layout order: override, then the episode's own layout.yaml (EPISODE_LAYOUT_FILE), then meta.layout, then meta.layout relative to the episode dir. Loads are cached by resolved file path or built-in name, never by a relative reference.
- Frames with non-finite taxel_pos/nrm are skipped.
- Unresolved-layout, world-framed-pose and dropped-frame warnings are aggregated into one warning each.
- level_counts uses per-episode frame lists instead of an O(E·S) scan.
- Docstrings updated.
- robot_skin/representation/encoder.py:
- load_pretrained_encoder moves the module to map_location when it is a device.
- The forward() docstring explains that the tokenizer [MASK] is untrained after MAE pretraining and to use key_padding_mask for dead taxels.
- robot_skin/representation/README.md:
- [MASK] caveat and loss(batch) alias.
- NaN-safe targets, pose-frame and non-finite-frame warnings.
- Hardware precedence, split-entry resolution and the leakage note (use the VTLA splits.json).
- Fixed the encoder snippet (torch.as_tensor on pos/nrm).
- robot_skin/configs/stages/pretrain.yaml: comments for hardware_applied (set by load_stage_config) and data.splits (use the VTLA splits to keep test episodes unseen). Keys are unchanged, so the DEFAULTS sync test still holds.
- robot_skin/tests/test_pretrain.py, new tests:
- test_dataset_layout_resolution_per_episode: the relative-ref collision and the moved dataset (stale absolute meta.layout plus a local copy, with no warning allowed).
- test_dataset_skips_frames_with_non_finite_poses: skipped frames and level_counts over the kept frames.
- test_unused_non_finite_targets_do_not_poison_loss_or_grads.
- test_evaluate_reconstruction_empty_is_nan_not_zero, plus the level-only loss composition.
- test_stage_hardware_profile_precedence: CLI path, raw-dict path, no re-application, and the no-profile case.
- robot_skin/tests/test_pretrain.py, extended tests:
- The world-frame warning is now asserted to be a single aggregated warning.
- The MAE test also checks the eval/no_grad fast path with padding (finite and equal to the slow path).
- Spec loss() alias check.
- metrics.json parsed strictly (no NaN/Infinity).
- Split entries relative to the splits.json directory.
### reviewer remaining concerns
1. **[PRE] Glove taxel_pos is in the world frame, against the Episode contract.** The contract (datasets/episode.py K_TAXEL_POS) says hand/robot-base frame. robot_skin/datasets/build.py `_taxel_poses` still calls `taxel_poses_from_hand(layout, sk, go, fp, wp)`, which yields world-frame positions. The taxel centroid of a synthetic D2 glove episode travels 0.39 m, and TaxelPretrainDataset now raises one aggregated warning. This must be settled before real D2 pretraining. It is the implementer's high-priority interface request, and it is not something REPR should silently transform.

2. **Fourier position-feature defaults (design / hyperparameters, no code change).** The tokenizer's fourier_features uses `scale` as the period of the lowest octave, so features are exactly periodic with a 50 mm period per axis. A hand spans about 130 mm, and with 8 octaves the finest period is 0.39 mm. My probe showed 1 mm pose noise shifts the features by about 5, against a median taxel-pair distance of about 6.6, so the top octaves are noise for real, mm-accurate pose labels. For real data, consider fourier_scale ≥ 0.2–0.4 m with n_fourier around 5–6. Changing it involves tokenizer.py (not REPR-owned) and default.yaml `representation.fourier_scale_m` (INTEG), so I left the 0.05 default.

3. **Downstream leakage.** The default split (data.splits: null, val_frac) pretrains on episodes that may later be VTLA test episodes. This is unlabelled and transductive, and it is now documented. INTEG and the e2e pipeline should pass the same splits.json to pretrain and vtla.

4. **Interface notes from the implementer that still stand:**
   - **[STAGE1]** Write derived residual_z and contact_level for all episodes.
   - **[VTLA/CTRL]** Use enc.feature_spec / TactileHistory. For obs_mode none, disable the tactile branch. For dead taxels, use key_padding_mask rather than the untrained tokenizer [MASK].
   - **[INTEG]** Call load_stage_config(path, overrides) and then run(cfg). It now applies the hardware profile correctly (YAML < profile < overrides) and exports the profile env, so the CLI no longer needs to apply the profile itself. Update the README representation row.
   - **[REFS]** Adjust the MAE wording in REFERENCES.md, and add the Tancik citation in tokenizer.py.

5. **Not tested here:** GPU, bf16 autocast, DDP and torch.compile (no GPU available). With history > 1, only the current frame of a hidden taxel is reconstructed. Hyperparameters (mask ratio 0.3, z compression 100/2, class-weight power 0.5, frame_stride 4) are untuned.

**Ownership:** I edited only REPR-owned files:
- robot_skin/representation/{encoder,pretrain}.py and README.md
- robot_skin/stages/pretrain.py
- robot_skin/configs/stages/pretrain.yaml
- robot_skin/tests/test_pretrain.py

test_representation.py, `__init__.py` and stages/`__init__.py` are unchanged from the implementer. I did not touch common/, deformable_sats/, tokenizer.py or any other agent's files, and nothing was committed.


## STAGE1
### public_api
## baseline/temporal.py
Everything here is also re-exported from `robot_skin.baseline`. The v1 `BaselinePredictor` and the other v1 code are unchanged.

**`TemporalBaselinePredictor`**
- Signature: `(n_taxels, joint_dim, *, window=32, hidden=128, taxel_emb_dim=8, kernel=3, n_layers=4, arch="tcn"|"gru", head_hidden=64, dropout=0., pos_scale=10., use_pose=True, sigma0=1., logvar_min=-12., logvar_max=8., var_detach=True)`.
- `forward(q_hist[B,W,D], qd_hist[B,W,D], pos[B,N,3], nrm[B,N,3]) -> (mean[B,N] ΔS %, logvar[B,N] %²)`.
  - Inputs are in raw units; the model normalises internally with its own buffers.
  - There is no ΔS argument; a test pins the signature so ΔS can never become an input.
- Structure:
  - Joint path: causal dilated TCN (per-step LayerNorm; receptive field 1+(k−1)(2^L−1) = 31 with the defaults) or a GRU; the last step gives the context.
  - Per-taxel head: FiLM style, from `pos`/`nrm` at t plus a taxel embedding.
  - Initialisation: mean head zero-init; logvar bias = log σ0²; logvar is soft-clamped.
- Buffers: `set_joint_stats(q_stats, qd_stats)` and `set_target_scale(scale[N])`. `joint_stats_dict()` returns the stats; `.config`, `.receptive_field` and `from_config` are available.

**Losses and windows**
- `gaussian_nll(mean, logvar, y, valid=None, *, logvar_min, logvar_max, include_const)` (Kendall & Gal 2017).
- `baseline_loss(mean, logvar, y, valid, y_scale, *, mean_loss="mse"|"huber"|"none", mean_weight, nll_weight, detach_mean=True, huber_delta) -> {loss, nll, mean_loss, mae}`, computed in target-scale units.
- `soft_clamp(x, lo, hi)`.
- `causal_windows(x, W, index=None)`: edge-padded with frame 0.

**Inference**
- `predict_episode(model, episode, joint_stats=None, window=None, batch_size=2048, *, device, q_source="q") -> (mean[T,N], logvar[T,N])`. It is causal.
- `CausalBaselineStream(model).push(q[D], qd[D], pos[N,3], nrm[N,3]) -> (mean[N], logvar[N])`, plus `.reset()`. It matches the offline output (tested). This is what CTRL uses online.

**Joint-state source**
- `episode_joint_view(ep, q_source="q"|"hand_pose_imu")`.
- With `hand_pose_imu`: `q` = derived `hand_finger_pose_imu`, `qd` = `joint_velocity` with the episode's preprocessing `qd` config, and taxel poses are recomputed in the hand frame.

**Bundle**
- `save_baseline_model(path|dir, model, meta)` and `load_baseline_model(path|dir)`; the loaded model carries `.bundle_meta`.
- Constants: `BASELINE_MODEL_NAME="baseline_model.pt"`, `Q_SOURCES`, `MEAN_LOSSES`.

## contact (new modules)
The torch-based detector is exported lazily from `robot_skin.contact`, so `import robot_skin.contact` does not import torch.

**calibration.py**
- `ResidualCalibrator(sigma, center, gain, use_logvar, weak_z=3, strong_z=8, weak_floor_pct=.5, strong_floor_pct=3, fsm, info)`.
- `.fit(residual[T,N], valid, logvar=None, *, weak_z, strong_z, weak_floor_pct, strong_floor_pct, sigma_floor_pct=.02, center=True, min_samples=50, fsm)`
  - σ is the robust per-taxel spread (MAD·1.4826).
  - With logvar: σ² = max(MADσ² − median(exp(logvar)), floor²), and a per-taxel gain g makes MADσ(z) = 1.
- Transform methods:
  - `.press(r)` = −r − c.
  - `.scale(logvar)`.
  - `.transform(r, logvar) -> z` (press-positive).
  - `.levels(z, saturated, *, press_pct)`: WEAK needs z ≥ weak_z AND p ≥ floor; STRONG likewise; SAT overrides.
  - `.levels_from_residual`.
- Persistence: `to_dict` / `from_dict` / `save` / `load`, as a single JSON file.
- `robust_sigma(x, valid, center=None)`.
- `saturation_gate(residual, saturated, dt, *, ok_pct, ok_sec, max_recover_s, enabled) -> (corrected, untrusted, states)`: runs `SaturationFSM` offline.
- `residual_levels(cal, residual, saturated, logvar, *, dt) -> {residual_z, contact_level, untrusted, press_pct}`: the single offline entry point.

**detector.py**
- `ContactDetector(n_taxels=None, window=16, hidden=32, *, joint_dim, n_layers=3, kernel=3, taxel_emb_dim=4, use_motion=True, use_sat=True, z_scale=2, z_clip=100, prior=.01, dropout=0)`.
  - `forward(z_hist[B,W,N], sat_hist, qd[B,D], q_valid[B]) -> logits[B,N]`.
  - Per taxel: asinh(clip(z)/z_scale) and the sat flag go through a causal dilated conv shared across taxels.
  - A |q̇| summary (RMS and max of normalised qd, gated by q_valid) and an optional taxel embedding are added.
  - Output bias is initialised to the prior π (Lin et al. 2017).
  - `set_joint_stats(qd_stats)`.
- Losses: `focal_loss(logits, target, mask, *, gamma=2, alpha=.25)` (Lin et al. 2017) and `contact_loss(..., kind="focal"|"bce", pos_weight)`.
- Inference:
  - `predict_contact_prob(det, episode=None, *, z, saturated, qd, q_valid, window, batch_size) -> prob[T,N]`, causal. `predict_episode` is an alias.
  - `CausalDetectorStream(det).push(z[N], sat[N], qd[D], q_valid) -> prob[N]`, matches the offline output.
- `save_detector` / `load_detector`.

**hysteresis.py**
- `HysteresisFilter(on_thr=.6, off_thr=.4, min_on=1, min_off=1, n_taxels=None)` with `.step(p[N])`, `.run(p[T,N])`, `.reset()`, `.config`, `from_config`.
- Turns ON after min_on consecutive samples ≥ on_thr and OFF after min_off consecutive samples < off_thr.
- NaN counts as low.

**pseudo_label.py**
- `D_CONTACT_LABEL_PSEUDO = "contact_label_pseudo"`; `EXPECTATIONS`.
- `phase_expectation(phase|name)` and `frame_expectation(ep, overrides)`.
- `taxel_world_positions(ep) -> (pos, valid)`: glove taxels are mapped R(go)·p + wp.
- `pseudo_label_episode(ep, prob, *, hysteresis, neg_thr=.1, keep_existing=True, none_conflict="unknown"|"zero", unknown_phase="unknown"|"any"|"none", saturated_as_contact=True, proximity={"max_dist_m"}, phase_overrides) -> {label int8[T,N], on, expectation, counts}`.
- `pseudo_label_metrics(label, truth, ignore) -> {precision, recall, neg_precision, coverage}`.

## stages (all three have the same shape)
Each of `stages.imu_pose`, `stages.baseline` and `stages.contact` provides `STAGE`, `CONFIG_PATH`, `DEFAULTS` (equal to its YAML, enforced by a test), `load_stage_config(path=None, overrides=None)` (applies the hardware profile, then overrides; unknown keys raise), `resolve_config(cfg)`, `run(cfg) -> metrics` (also writes `out_dir/metrics.json` as strict JSON) and a `main(argv)` CLI (`python -m robot_skin.stages.<s> --config y --set a.b=v`).

**imu_pose**
- Trains `ImuHandPoseNet` on episodes with hand labels, using `hand_pose_loss(...)["loss"]`. Feature stats are stored inside the model.
- Writes derived `hand_finger_pose_imu` [T,15,3], plus `imu_pose_model.pt` (`load_imu_pose_model`) and `imu_stats.json`.
- Metrics: `val/rot_deg`, `tip_mm`, and their flat-hand references.

**baseline**
- Trains on D1 no-contact frames via `BaselineWindowDataset(joint_stats=None)` and `train.Trainer`.
- Writes derived `baseline_pred`, `baseline_logvar`, `residual` for the predict datasets, plus `baseline_model.pt` and `joint_stats.json`.
- Metrics per split (`val/`, `test/`, `task/`): mae_raw/mae_resid/resid_reduction, nll, coverage_2sigma, z_std, sep_auroc/dprime before/after, and `gt_artefact_*` on synthetic data.
- Options: `data.q_source` (q | hand_pose_imu) and `data.kind`.

**contact**
- Fits `ResidualCalibrator` on the D1 **val** split's no-contact frames.
- Writes derived `residual_z` / `contact_level`, then trains `ContactDetector` on D1 labels, writing `contact_prob`.
- Pseudo-labels D2 into derived `contact_label_pseudo`. The `contact_label` array is never modified.
- Outputs: `calibrator.json`, `contact_detector.pt`.
- Metrics: `{val,test}/{z,prob,hyst}_{hallucination_taxel, hallucination_frame, recall, precision, f1, auroc, gt_auroc, gt_hallucination, gt_recall}` and `pseudo/*`.
- Options: `detector.bootstrap` (auto | true | false), `fsm.*`, `data.kind`.

## stages/__init__.py (light helpers, no torch import)
- Config: `apply_stage_hardware`, `load_stage_yaml`, `resolve_stage_config`, `check_stage_keys`, `parse_overrides`.
- Output: `finite_json`, `write_json_atomic`.
- Episodes and training: `stage_episodes`, `split_stage_episodes` (delegates to `stages.pretrain.split_episodes`, so splits match), `fit_and_restore` (Trainer; falls back from val/* to train/* monitor; restores the best/EMA weights).
### deviations
1. **`baseline/calibrate.py` was not created (it was optional).** Rescaling σ after training is handled by the per-taxel gain `g` in `contact.calibration.ResidualCalibrator`, fitted on held-out no-contact frames. A second calibration layer would duplicate it.

2. **Baseline loss.** The default fits the mean with a target-scale MSE and trains the variance with a stop-gradient Gaussian NLL (`detach_mean`, and `var_detach`: the logvar head reads detached features).
   - I measured the alternatives. Joint NLL fitted the mean much more slowly: in-sample residual 0.39 % vs 0.07 % at 300 steps with grad_clip 1.
   - The pure Kendall & Gal objective is still available: `mean_loss: none`, `detach_mean: false`, `var_detach: false`.
   - All loss terms are in per-taxel target-scale units (`y_scale` buffer = RMS of no-contact ΔS).

3. **Normalisation lives inside the models.** q/qd stats and `y_scale` are buffers in the baseline; the qd stats are buffers in the detector.
   - Stages build the PRE datasets with `joint_stats=None`, and checkpoints are self-contained for the online side.
   - `predict_episode(joint_stats=...)` is kept only for models trained on inputs normalised outside the model.

4. **Pseudo labels are a derived array**, `contact_label_pseudo` (int8 −1/0/1). The preprocessing `contact_label` array is never modified, as the task asked.

5. **Shared stage helpers are in `stages/__init__.py`.** They are light and import no torch at import time. I put them there rather than in a private module, which would have been a file outside my ownership list. Splitting reuses REPR's public `stages.pretrain.split_episodes`, so every stage has identical `data.splits` / `val_frac` semantics.

6. **Additions beyond the spec:**
   - `data.q_source` (`q` | `hand_pose_imu`, the camera-free glove path) in baseline and contact.
   - `data.kind` filter; mixed kinds or taxel counts in one training pool raise a clear error.
   - `detector.bootstrap: auto`. Robot D1 has no geometric self-touch labels, so when the train split has no positives, STRONG calibrated levels inside contact-allowed phases become positives (self-training).
   - Optional `SaturationFSM` gate (`fsm.enabled`, default false). Its config is stored in `calibrator.json`.
   - Hysteresis-filtered detector metrics, and synthetic ground-truth metrics wherever `gt_*` arrays exist.
   - `CausalBaselineStream` and `CausalDetectorStream` for CTRL, each tested equal to the offline output.

7. **Contact levels** use z thresholds AND % floors (weak 0.5 %, strong 3 %). FSM-untrusted taxels are set to SATURATED. `residual_z` stays finite on saturated taxels; consumers mask with `level == SAT` or `saturated`.

8. **Detector naming and summary.** The spec's `predict_episode` is `predict_contact_prob`, with `predict_episode` kept as an alias. The |q̇| summary is (RMS, max) of the normalised qd plus a q_valid flag.

9. **test_stage1 does not use `generate_dataset`.** The synthetic generator draws the per-taxel artefact physics (gain, sign, lag) from each session's seed, so `generate_dataset` sessions are different "gloves" and a baseline cannot generalise across them. The test calls `generate_session` 3× with seed 7 (motion s0 and s1, task s0), giving one glove with different motion styles, and checks held-out generalisation: train s0, validate s1, predict the task.

10. **No FiLM citation.** The per-taxel head is described as FiLM-style, but that paper is not in the verified reference list. Citations are only Kendall & Gal 2017, Lin et al. 2017, and Yu et al. arXiv:2607.22964 (as related work).
### interface_requests
1. **[SYNTH] `robot_skin/datasets/synthetic.py`**
   - The tactile physics come from `_rng(seed, "tactile")`, which is per session, so every `generate_dataset` session is a different glove and a baseline cannot generalise across sessions.
   - Request: add an option such as `generate_dataset(..., shared_glove=True)` / `generate_session(glove_seed=...)` that draws the gains/signs/lags/press params (optionally the baselines) from a glove seed shared across sessions.
   - INTEG e2e can then evaluate the baseline on held-out sessions. Until then, use `generate_session` with one shared seed, as `test_stage1` does.

2. **[PRE] `datasets/motion.py`**
   - `ContactWindowDataset` / `BaselineWindowDataset` read `label_key` only from `ep.arrays`. Please fall back to `ep.derived(label_key)` so `contact_label_pseudo` (derived) can train on D2 directly. I work around this with in-memory views.
   - **[episode.py owner]** Consider adding `D_CONTACT_LABEL_PSEUDO = "contact_label_pseudo"` to the episode contract. It is currently defined in `contact/pseudo_label.py`.

3. **[CTRL] `control/online.py` must mirror the offline pipeline:**
   - (a) Load `baseline_model.pt` with `load_baseline_model`. Compute qd with `datasets.build.joint_velocity(q_buffer, hz, **model.bundle_meta["qd"])`, then `CausalBaselineStream.push(q, qd, pos, nrm)`. `pos`/`nrm` are in the HAND frame (go = 0, wrist at origin).
   - (b) `r = ΔS − mean`. Load `cal = ResidualCalibrator.load("calibrator.json")`. If `cal.fsm["enabled"]`, step a `SaturationFSM` with the same config: `nan_to_num(r)` into `step`, `corrected(r)` for the residual, untrusted = state ≠ OK.
   - (c) `z = cal.transform(r, logvar)`, `level = cal.levels(z, sat | untrusted, press_pct=cal.press(r))`.
   - (d) `CausalDetectorStream(load_detector(...)).push(z, sat, qd, q_valid)`, then `HysteresisFilter(**det.bundle_meta["hysteresis"]).step(prob)`.
   - Reset every stream at session start.

4. **[INTEG]**
   - `robot_skin/__main__.py`: `train imu_pose|baseline|contact` → `cfg = stages.<s>.load_stage_config(path, overrides)`; `stages.<s>.run(cfg)` (the hardware profile is applied inside, and `hardware_applied` is set).
   - Pipeline order: imu_pose → baseline → contact → pretrain → vtla. Pass the same `data.splits` to every stage.
   - e2e tests: call `torch.set_num_threads(1)` and give the detector ≥ 100 `max_steps`.
   - README: rows for baseline (temporal), contact (calibration / detector / hysteresis / pseudo_label) and the stage-1 runners.

5. **[REFS/DOCS]** `docs/TRAINING.md`: describe the stage-1 runners, outputs and metrics. Optionally add FiLM (Perez et al., AAAI 2018) to REFERENCES.md if a citation for the per-taxel head is wanted; I did not cite it.

6. **[REPR]** No change needed. `contact_level` includes SATURATED wherever `saturated` is set (and FSM-untrusted), consistent with `tactile_value_features`.
### open_todos
1. **The baseline's σ is aleatoric only.** Out-of-distribution poses (the D2 power grasp, not covered by D1 no-contact blocks) give large residuals with small σ, and hence false contact on uncovered taxels. In one synthetic run the pseudo-label gt precision was 0.34; it was 0.82 in the test run. Options:
   - add grasp-like no-contact holds to the D1 protocol;
   - make the variance OOD-aware (ensembles or MC dropout);
   - use the pseudo-label `proximity_m` veto.

2. **Hyperparameters are untuned for real mk555 data:** window 32, TCN depth 4, calibration thresholds 3/8 z with 0.5/3 % floors, focal α/γ, hysteresis 0.6/0.4 with 2/4 ticks.

3. **Untested here (no GPU):** GPU, bf16 autocast, DDP and torch.compile. Rank-0-only writes and barriers are implemented.

4. **`q_source: hand_pose_imu`.** IMU-model predictions on its own training episodes are optimistic. For a fair baseline, train the IMU model on a split disjoint from the baseline's, or accept the optimism.

5. **Robot detector bootstrap** needs more steps (~400 in a manual run gave pseudo-label precision 0.82 / recall 0.99). Robot val detection metrics have no positive labels, so AUROC is NaN; only the gt metrics apply.

6. **Minor efficiency:** `CausalBaselineStream` / `CausalDetectorStream` rebuild their ring-buffer arrays every tick, which is fine at 200 Hz. A preallocated ring would be leaner for CTRL.
### reviewer fixes
- robot_skin/baseline/temporal.py: new `qd_settings(preprocessing)` returns only the `joint_velocity` kwargs (method, window_s, polyorder), merged over PRE DEFAULTS. It is exported (`QD_KWARGS`, and from robot_skin.baseline). `episode_joint_view` now uses it, and it uses the preprocessing skeleton (`hand_pose.mano_model`) unless `skeleton=` is given. The docstring states the qd contract.
- robot_skin/stages/baseline.py: `bundle_meta["qd"] = qd_settings(pre)` and a new `bundle_meta["qd_source"]` ('derivative' | 'file'; 'file' means use the driver's velocities online). `mean_loss` is validated, and `mean_loss: none` now requires `detach_mean: false` and `nll_weight > 0`. Mixed master rates in the pool raise an error; mixed qd settings warn. Docstring updated.
- robot_skin/stages/__init__.py: new `seed_model_init(train_cfg)` (torch.manual_seed(train.seed) before model construction). imu_pose.py, baseline.py and contact.py call it right before building ImuHandPoseNet, TemporalBaselinePredictor and ContactDetector. Runs are now bit-identical; checked with two runs and with a mutation of the fix.
- robot_skin/stages/contact.py: `_usable(ep, use_logvar, q_source, need_qd)`. The vision `qd` is required only for q_source q with a motion-aware detector. For hand_pose_imu it requires derived hand_finger_pose_imu instead of crashing. With FSM enabled, the calibrator is fitted on the FSM-corrected residual of trusted samples only (saturation_gate). The bootstrap uses STRONG levels only. `predicted` counts written episodes. The docstring warns that val/ metrics are optimistic (calibration and early-stopping split) and recommends a splits.json test split.
- robot_skin/contact/pseudo_label.py: `taxel_world_positions` returns valid=False for every frame of a mano_wrist (glove) episode without hand pose, so no spurious proximity vetoes.
- robot_skin/contact/calibration.py: the σ formula in the docstring now matches the code.
- robot_skin/baseline/README.md and robot_skin/contact/README.md: documented the qd kwargs, qd_source and the None-safe fsm check in the CTRL recipe.
- tests/test_stage1.py:
- The bundle check now uses the CTRL recipe: exact `qd` kwargs, `joint_velocity(**qd)` equals ep.qd, and CausalBaselineStream fed by ring-buffer qd matches predict_episode.
- The docstring caveat on s1 is added, and the task transfer assertion is tightened to < 0.75·|gt|.
- The no-detector contact run now covers `q_source: hand_pose_imu` together with `fsm.enabled`.
- New tests: test_contact_usable_follows_q_source, test_baseline_rejects_a_loss_that_never_trains_the_mean, test_stage_runs_are_reproducible (fails without the fix, verified by mutation), test_detection_metrics_hand_values (hand-derived AUROC 7/8, gt_auroc 10/12, etc.), and a SATURATED case in the bootstrap test.
- tests/test_baseline_temporal.py: new test_qd_settings_are_joint_velocity_kwargs (source dropped, defaults filled, view honours the episode's qd config).
- tests/test_contact_learned.py: new test_proximity_veto_needs_a_hand_pose_for_gloves.
### reviewer remaining concerns
1. **Baseline σ is aleatoric only (implementer's open TODO, now quantified).** With seeds fixed per run, D2 pseudo-label gt precision still ranges 0.30–0.70 across train seeds.
   - The false positives sit on non-contact taxels in manipulate/release. There the true artefact is outside D1 pose coverage and the baseline under-predicts it by about 1.6 % with σ≈0.3, so z≈20.
   - Options: grasp-like no-contact holds in the D1 protocol, or an OOD-aware variance (ensembles / MC dropout).
   - Do not rely on D2 pseudo-label precision yet.

2. **Optimistic evaluation and a train/deploy gap.**
   - The contact-stage val/ metrics use the same split that fits the calibrator and early-stops the detector. Only test/ numbers from a splits.json are unbiased (now documented).
   - The detector trains on train-split residuals, which are in-sample for the baseline, so its negatives are cleaner than at deployment.

3. **Split consistency across stages.** Without `data.splits`, each stage re-splits its own usable pool. Different skip sets or `data.kind` settings can give a different val split, so the calibrator could land on baseline-train episodes. INTEG should always pass one splits.json to imu_pose, baseline, contact, pretrain and vtla.

4. **[SYNTH] The shared-glove option is still needed.** Motion and tactile RNGs share the session seed, so s1 is a time-warped replay of s0. The task episode is the only novel-motion check.

5. **[CTRL] follow-ups.**
   - Use `bundle_meta["qd"]` (joint_velocity kwargs only) and honour `bundle_meta["qd_source"] == "file"`.
   - For `q_source: hand_pose_imu`, CTRL must run the IMU pose model per tick itself: `ImuHandPoseNet.predict` on a ring buffer of `imu_features`, using the imu_pose bundle's window, features and wrist_index. STAGE1 provides no stream class for this.
   - Reset every stream at session start.

6. **[episode.py owner] Non-atomic derived writes.** `Episode.set_derived` overwrites derived .npy files in place (np.save truncates the same inode). A process holding a memmap of that file while a stage re-runs can read torn data. Consider write-to-temp + os.replace.

7. **Untested here:** GPU, bf16 and DDP (the rank-0 writes and barriers were only reviewed, not run). The `_preprocessing_skeleton` branch that loads a MANO model file is untested because no MANO model is available. Hyperparameters are untuned for real mk555 data.

All edits are in STAGE1-owned files:
- robot_skin/baseline/{temporal.py, __init__.py, README.md}
- robot_skin/contact/{calibration.py, pseudo_label.py, README.md}
- robot_skin/stages/{__init__.py, baseline.py, contact.py, imu_pose.py}
- robot_skin/tests/{test_stage1.py, test_baseline_temporal.py, test_contact_learned.py}

Nothing was committed.


## VTLA
### public_api
`robot_skin.vtla` re-exports everything below. The old `TactileTokenAdapter` and `ContactGate` API still works.

**adapter.py** (additive changes)
- `ContactGate.forward(tokens, contact, valid=None)`: padded taxels are neither contact nor counted in the soft gate's fraction.
- `TactileTokenAdapter.forward(tokens, contact, key_padding_mask=None)` passes `~key_padding_mask` to the gate.
- Docstring now cites Perceiver (Jaegle et al., arXiv:2103.03206).

**model.py**
- Constants: `MODALITIES = (language, vision, tactile, proprio, readout)`, `LANG`, `VISION`, `TACTILE`, `PROPRIO`, `READOUT`.
- `VTLAConfig` dataclass:
  - Sizes: `action_dim` 54, `horizon` 16, `proprio_dim`, `obs_history`, `d_model`, `fusion_depth`, `fusion_heads`, `ff_mult`, `dropout`, `n_readout`.
  - Branches: `cameras`, `vision` (build_vision_encoder cfg or None), `vision_frozen`, `text` (build_text_encoder cfg or None), `text_frozen`, `feature_spec` (TactileFeatureSpec dict), `tactile_encoder` (TaxelEncoder kwargs), `tactile_frozen`, `n_tactile_tokens`, `tactile_heads`, `tactile_gate`.
  - Head: `head` (chunk | flow), `head_depth`, `head_heads`, `flow_steps`, `flow_tau` (uniform | beta), `flow_tau_beta_b`.
  - Regularisation: `p_drop_tactile`, `p_drop_vision`, `p_drop_language`, `aux_contact_weight`, `contact_pos_weight`.
  - Helpers: `.to_dict()`, `.from_dict()` (unknown keys raise), `.spec`, `.use_tactile`, `.use_vision`, `.use_language`.
- `VTLAPolicy(cfg, *, tactile_encoder=None)`:
  - `tactile_encoder` takes a pretrained `TaxelEncoder`; its config replaces `cfg.tactile_encoder`.
  - Module names (usable as `lr_mult` prefixes): `text_encoder`, `text_proj`, `vision_encoder`, `vision_proj`, `camera_emb`, `history_emb`, `tactile_encoder`, `adapter`, `contact_head`, `proprio_mlp`, `readout`, `type_emb`, `fusion`, `head`.
  - `encode(batch)` returns `{memory [B,S,d], memory_mask [B,S], token_types [S], tactile_dropped / vision_dropped / language_dropped [B] | None, contact_logits [B,N] | None}`.
  - `forward(batch, *, return_encoding=False)` returns `{loss, action_loss, action_l1 | flow_mse, contact_loss?}`. Without `actions` in the batch it returns `{actions}` instead.
  - `predict(batch, n_steps=None, *, generator=None, noise=None)` returns normalized `[B,H,A]`, always in eval mode.
  - `per_sample_action_error(batch, actions, valid, **kw)` returns `[B]`.
  - Also: `.config`, `from_config`, `.cameras`, `.feature_spec`, `.obs_mode`. `train()` keeps frozen encoders in eval mode.
- Batch keys:
  - Observation: `proprio [B,k·P]`, `images {cam: [B,(k,)3,H,W]}` or `vision_feats {cam: [B,(k,)P,D]}`, `vision_valid {cam: [B,k]}`.
  - Tactile: `tactile_values [B,N,F]`, `taxel_pos`, `taxel_nrm`, `contact`, `taxel_pad`.
  - Language: `input_ids` + `text_pad_mask`, or `instruction` (list of str).
  - Training targets: `actions [B,H,A]`, `action_valid [B,H]`, `contact_target` / `contact_target_mask [B,N]`.
- Bundle:
  - Constants: `POLICY_BUNDLE_NAME = "policy_bundle.pt"`, `BUNDLE_FORMAT`, `BUNDLE_VERSION`.
  - `save_policy_bundle(path, policy, *, action, proprio, tactile, vision, language, timing, meta=None, state_dict=None) -> Path` writes atomically; the file loads with `torch.load(weights_only=True)`.
  - `read_policy_bundle(path | dir)` loads the file and validates it.
  - `build_policy_from_bundle(bundle_dict | path, *, map_location, device, strict) -> VTLAPolicy` returns the policy in eval mode. Encoders are rebuilt with `pretrained: false`; weights come from `state_dict`.
  - `bundle_components(bundle, *, device)` returns `{policy, action_spec, action_normalizer, rel_mode, chunk_offset, proprio_normalizer, feature_spec, contact_rule, eval_transform (EvalTransform), tokenizer, cameras, policy_hz, source_hz, stride, horizon, obs_history, head, flow_steps, tactile, vision, meta}`.

**heads.py**
- Constants and helpers: `HEAD_TYPES`, `TAU_DISTS`, `sinusoidal_embedding(x[B], dim)`, `build_head(kind, d_model, A, H, **kw)`.
- `ChunkRegressionHead(d_model, A, H, *, depth, heads, ff_mult, dropout)` (ACT): H learned queries go through a pre-LN TransformerDecoder with cross-attention to memory, then a zero-initialised Linear. Methods: `forward(memory, mask)`, `loss(...)` (masked L1), `sample`, `per_sample_error`.
- `FlowMatchingHead(d_model, A, H, *, depth, heads, ff_mult, dropout, n_steps=10, tau_dist="uniform", tau_beta_b=1.5)`. Methods: `velocity(x, tau, memory, mask)`, `sample_tau`, `loss(..., noise=, tau=, generator=)`, `per_sample_error`, `sample(memory, mask, *, n_steps, noise, generator)`.
- τ convention, stated in the docstring:
  - τ = 0 is noise and τ = 1 is data.
  - Path: `x_τ = τ·a + (1−τ)·ε`, target velocity `u = a − ε`.
  - Sampling: Euler from `x_0 = ε` in K steps.
  - `beta` samples τ from Beta(1, b) via the inverse CDF, which puts more weight on noisy τ.
  - Citations: Lipman et al. arXiv:2210.02747, Liu et al. arXiv:2209.03003, π0 arXiv:2410.24164.

**losses.py**
- `masked_error`, `masked_l1`, `masked_mse`: padded steps contribute no value and no gradient, even when the padding is NaN, and no host sync is needed.
- `masked_step_sums(err, valid) -> (sum[H], count[H])`.
- `contact_bce(logits, target, mask, *, pos_weight)`, `weighted_total`.
- `vtla_loss(model, batch)` is the Trainer `loss_fn`.

**dataset.py**
- Constants:
  - `TASK_PHASES` equals `acquisition.protocol.TASK_PHASES` (a test checks this).
  - `CONTACT_RULES = (level_ge_weak, weak_or_strong)`, `AUX_TARGETS = (label, level, gt)`, `TACTILE_SOURCES = (auto, derived, bootstrap)`, `BOOTSTRAP_DEFAULTS`.
  - `PSEUDO_LABEL_KEY = "contact_label_pseudo"`, equal to STAGE1's `D_CONTACT_LABEL_PSEUDO`.
- Functions:
  - `bootstrap_tactile_arrays(ep, **kw) -> (z, level, sat)`: test / bring-up only.
  - `episode_tactile(ep, source, bootstrap) -> (z, lv, sat, used)`.
  - `contact_from_level(level, saturated, rule)`, `sample_phase_mask(ep, phases)`, `history_ticks(t, k, stride)`.
  - `eval_transform_to_dict` / `eval_transform_from_dict`.
  - `make_observation(*, proprio_states[k,A], tactile_values, taxel_pos, taxel_nrm, contact, instruction, proprio_normalizer, images | vision_feats, vision_valid)`: shared with online control.
- `VTLADataset(episodes, *, cameras, policy_hz=20, horizon=16, obs_history=1, chunk_offset=1, action_spec="hand_mano", rel_mode="delta", feature_spec, image_transform, use_cached_vision, action_normalizer, proprio_normalizer, phases="task", sample_stride, min_valid_steps=1, require_valid_state=True, contact_rule, aux_target="label", tactile_source="auto", bootstrap)`.
  - Attributes and methods: `.index [M,2]`, `.stride`, `.action_spec`, `.tactile_source`, `.raw_targets()`, `.fit_normalizers(method, min_scale)`.
  - Sample keys: `proprio`, `tactile_values`, `taxel_pos`, `taxel_nrm`, `contact`, `taxel_pad`, `instruction`, `images` or `vision_feats`, `vision_valid`, `actions`, `action_valid`, `contact_target`, `contact_target_mask`, `episode`, `t_index`, `task_id`.
- `collate_vtla(samples, tokenizer=None)` pads the taxel axis and adds `taxel_pad`. `VTLACollator(tokenizer)` is a picklable wrapper.

**dpo.py**
- `dpo_loss(pc, pr, rc, rr, beta, *, reduction)` returns `{loss, reward_chosen, reward_rejected, reward_margin, reward_accuracy}`. Cites Rafailov et al. arXiv:2305.18290 and VTLA arXiv:2505.09577.
- `chunk_log_likelihood(policy, batch, actions, valid, *, sigma, noise, tau)`.
- `preference_loss(policy, reference, batch, *, beta, sigma, n_draws, generator)`: shares the (ε, τ) draws across all four evaluations.
- `make_reference_policy`, `PreferencePair`.
- `build_preference_pairs`: stub that raises NotImplementedError with the pipeline described.

**stages/vtla.py**
- `STAGE`, `CONFIG_PATH`, `METRICS_NAME`, `DEFAULTS` (mirrored by `configs/stages/vtla.yaml`; a test enforces this).
- `apply_hardware`, `load_stage_config(path, overrides)`, `resolve_config`. An encoder block that contains `type` replaces the default block; a partial block (`--set vision.encoder.out_dim=64`) merges.
- `discover_episodes`, `load_usable_episodes(dirs, cfg, spec)` (lists skipped episodes with reasons), `split_episodes` (a splits.json, or `datasets.splits.make_splits`).
- `evaluate_policy(model, ds, *, batch_size, device, seed, n_steps, collate_fn)` returns `{l1, l1_per_step[H], l1_by_task, l1_raw/<group>, n_samples, n_valid_steps}`.
- `run(cfg) -> metrics` writes `<out_dir>/policy_bundle.pt` and `metrics.json`.
  - Bundle sections: `model_config`; `state_dict` (best checkpoint, EMA if enabled); `action{spec, normalizer, rel_mode, chunk_offset}`; `proprio{normalizer, history, source}`.
  - `tactile{feature_spec, obs_mode, contact_rule, source, bootstrap, calibrator, calibrator_state (embedded calibrator.json), baseline_model, pretrained_encoder, frozen, layouts}`.
  - `vision{cameras, encoder, frozen, cached_features_key, eval_transform, transform_config}`; `language{encoder, frozen}`.
  - `timing{policy_hz, source_hz, stride, horizon, obs_history, sample_stride}`; `head`; `meta{episodes per split, phases, best, metrics, ensemble_k}`.
- `main(argv)` for `python -m robot_skin.stages.vtla --config y --set k=v`.
### deviations
1. **Default sampling phases.** The spec says to sample "inside phases != none".
   - The default here is `data.phases: task`, meaning only reach, grasp, manipulate, release and retreat.
   - Reason: ACQ's D2 episodes also contain a 3 s rest baseline, the flat-hand imu_calibration and the sync taps. Imitating those would teach the policy calibration motions.
   - `phases: all` gives exactly the spec's rule (phase_id ≥ 0). A list of phase names is also accepted.
2. **Proprio statistics.** Proprio is the current action-space state (hand action or robot q), so it is normalised with a second `ActionNormalizer`, fit on the train samples' current states. The spec suggested NormStats from datasets.stats. The bundle stores it as `proprio.normalizer` (an ActionNormalizer dict).
3. **Stage-1 fallback.** It is called "bootstrap". It uses a static per-taxel reference from the episode's `contact_label == 0` frames: median ΔS, σ = 1.4826·MAD floored at 1 %, z = press/σ, and WEAK/STRONG levels from z thresholds 3 / 8 with %-floors 3 % / 15 %. There is no motion-artefact model. It warns and is documented as tests / bring-up only. `tactile_source: derived` makes it a hard error.
4. **ContactGate input** is `level ≥ WEAK` (SATURATED included), as the spec says. `data.contact_rule: weak_or_strong` excludes saturated taxels.
5. **Modality dropout** hides a modality through the fusion key-padding mask, so the dropped tokens stay in the graph with zero gradient. The tokenizer's `[MASK]` parameter is set to `requires_grad=False` because it is only used in pretraining. With both, DDP does not need `find_unused_parameters`. obs_mode `none` builds no tactile branch at all.
6. **Additions beyond the spec:**
   - `obs_history` stacks proprio and camera frames at policy ticks. Tactile history stays in `TactileFeatureSpec` (master ticks).
   - `sample_stride` (densify samples), `require_valid_state`, `min_valid_steps` and `require_success`.
   - Aux targets `label` (the contact stage's `contact_label_pseudo` if present, else `contact_label`), `level` and `gt`.
   - A frozen-vision feature cache inside the stage (`vision.cache_features`). Its key includes a hash of the encoder weights so a stale cache is never reused. Cached training uses the eval transform, with no augmentation.
   - `evaluate_policy` also reports per-group L1 in raw units.
   - The bundle embeds the contact stage's `calibrator.json` content (`tactile.calibrator_state`) when `tactile.calibrator` points to it.
   - `make_observation` is shared with control; `bundle_components` is added.
7. **ContactGate API.** `ContactGate.forward` gained an optional `valid` argument, and the adapter passes it from `key_padding_mask`. The public API is unchanged and test_vtla.py still passes.
8. **Bundle helpers location.** The bundle helpers live in `vtla/model.py`, not a new file, to stay inside my ownership list.
9. **DPO likelihood surrogates** (documented in the dpo.py docstring):
   - Chunk head: a fixed-σ Gaussian.
   - Flow head: the negative flow-matching error at shared (ε, τ).
   - Neither is an exact likelihood.
10. **Flow-matching convention** is τ=0 noise, τ=1 data. It is stated explicitly and differs in naming from π0's own text and code.
### interface_requests
1. **[INTEG] `robot_skin/__main__.py`**
   - `train vtla` should call `cfg = robot_skin.stages.vtla.load_stage_config(path, overrides)` and then `run(cfg)`. `load_stage_config` applies `hardware` once, before the `--set` overrides, and sets `hardware_applied`.
   - Standalone CLI: `python -m robot_skin.stages.vtla --config ... --set k=v`.
   - The pipeline and e2e test should pass the same `data.splits` splits.json to pretrain and vtla.
   - `tactile.pretrained` should point at the pretrain out_dir; `tactile.calibrator` at the contact stage's out_dir or its calibrator.json.
2. **[INTEG] `robot_skin/configs/default.yaml`.** The `vtla:` block (n_query_tokens, d_out, gate) is not read by the stage, which uses `configs/stages/vtla.yaml` (`model.n_tactile_tokens`, `model.tactile_gate`, `model.d_model`). Please update or remove it.
3. **[INTEG / DOCS] `robot_skin/README.md`.** Change the vtla row to "implemented": model (fusion over lang/vision/tactile/proprio/readout), chunk + flow heads, dataset, DPO hook (stub pipeline), stages/vtla, policy_bundle.pt. docs/VTLA.md can link vtla/README.md.
4. **[CTRL] Online inference path.** To match training:
   - `comp = vtla.bundle_components(path)`.
   - One `representation.TactileHistory(comp["feature_spec"])` per episode. Push `(residual_z_t, level_t, saturated_t)` every 200 Hz tick, where saturated follows `datasets.episode.K_SATURATED` semantics.
   - `contact = vtla.contact_from_level(level_t, saturated_t, comp["contact_rule"])`.
   - Keep a proprio ring buffer of the absolute action-space state at policy ticks, `stride = comp["stride"]`, and pass it as `[k, A]` oldest first (`vtla.history_ticks` convention).
   - Frames: `comp["eval_transform"](frame[None])[0]`. With `vision.cached_features_key` set, run `policy.vision_encoder` online (frozen) instead of passing `vision_feats`.
   - `obs = vtla.make_observation(...)`, then `batch = vtla.collate_vtla([obs], comp["tokenizer"])`, then `policy.predict(batch)`.
   - Then: `action_normalizer.unnormalize`, `action.make_absolute(chunk, state_t, spec, rel_mode)`, and `TemporalEnsembler(k=bundle["meta"]["ensemble_k"])`.
   - Taxel poses must be in the hand / robot base frame (PRE v2 convention).
   - `bundle["tactile"]["source"] == "bootstrap"` means the policy was trained on bootstrap tactile levels and must not be deployed.
5. **[STAGE1]** Please keep `derived/contact_label_pseudo` (vtla `PSEUDO_LABEL_KEY`; a test checks it equals `contact.pseudo_label.D_CONTACT_LABEL_PSEUDO`) and `<out_dir>/calibrator.json` as the output names. The VTLA aux target and the bundle embedding rely on them.
6. **[DOCS]** docs/VTLA.md: describe the τ convention (τ=0 noise → τ=1 data), the bundle sections and the DPO surrogate caveats as in vtla/README.md.
### open_todos
1. **Untested hardware paths.** Not exercised here (no GPU): CUDA, bf16 autocast, DDP (a masked modality should need no `find_unused_parameters`, but not verified under torchrun), torch.compile, and HF / ResNet encoders.
   - `build_policy_from_bundle` rebuilds HF encoders with `pretrained: false`. That still needs the model's config from the HF cache or hub; the weights themselves come from the bundle.
2. **Proprio for hand_mano uses the absolute world-frame wrist position** from hand_pose.npz. Deploying on a robot needs a hand-equivalent state estimate (CTRL). An option to drop wrist position or use delta_pose proprio could be added.
3. **The DPO pipeline is a stub** (`build_preference_pairs`); it needs success/failure rollout data. The likelihood surrogates are heuristics.
4. **The bootstrap tactile path has no motion-artefact model.** Real training must set `data.tactile_source: derived` once the contact stage has run. Consider making `derived` the YAML default at that point.
5. **Hyperparameters are untuned.** Examples: d_model 128, horizon 16 at 20 Hz, p_drop_tactile 0.1, flow_steps 10, image_size 96×128, rel_mode delta.
6. **Evaluation scope.** The `vtla` stage evaluates only offline chunk L1. Closed-loop success rate needs the control / deploy stage.
7. **Unsupported inputs.** Tactile history in master ticks and obs_history at policy ticks are independent knobs. Vision history with cached features needs features for every history frame, which the cache provides. Mixed-layout batches (glove plus robot) work on the tactile side, but a dataset still uses a single action spec and master rate.
### reviewer fixes
- model.py: `history_emb` is built only when vision is used and `obs_history > 1`, so every trainable parameter gets a gradient. The docstring's lr_mult name list is completed and a ContactGate dead-channel caveat added.
- dpo.py: `preference_loss` encodes each model once and shares that encoding between the chosen and rejected chunks and across all draws. `chunk_log_likelihood` has a new `encoding=` argument. A new `disable_dropout=True` default runs the policy in eval mode with gradients kept (TRL-style) and restores its mode afterwards; the reference is also evaluated in eval mode. Loss is now exactly log 2 at π_θ = π_ref. Module docstring updated.
- heads.py: `FlowMatchingHead` gains `eval_seed=0`. In eval mode, a `loss()` call without explicit noise/tau/generator draws (ε, τ) from a generator re-seeded each call, so `val/loss` is comparable across epochs; training mode is unchanged. `build_head` drops `eval_seed` for the chunk head.
- heads.py: `ChunkRegressionHead.sample` returns float32; `FlowMatchingHead.sample(n_steps=0)` raises ValueError.
- stages/vtla.py: new `check_config` rejects unknown keys at the top level and one level into every section except the open `train`/`image` sections, and rejects non-mapping sections with a hint (`vision.encoder: null`). It is called from both `load_stage_config` and `resolve_config`.
- stages/vtla.py: `_weights_hash` hashes raw bytes plus dtype, so it works for bf16/fp16/bool weights.
- dataset.py: `make_observation` validates that taxel_pos/taxel_nrm are [N,3] and that tactile_values/contact match N. `VTLADataset` rejects `robot_joint` with `delta_pose` up front.
- README.md / vtla.yaml (comment only; the YAML still equals DEFAULTS): documented the dead-channel ContactGate caveat, the flow eval_seed, DPO shared encoding with disable_dropout, stage key validation, and that bundle `tactile.source` bootstrap or mixed must not be deployed.
- New tests in test_vtla_model.py:
- `test_flow_eval_loss_is_deterministic_train_loss_is_not`: checks the eval loss by hand from the eval_seed draws, and n_steps=0 raising.
- `test_obs_history_params_all_used_with_and_without_vision`
- `test_chunk_predict_is_float32_under_autocast`
- `test_preference_loss_disables_dropout_and_shares_encoding` (chunk and flow): exact log 2 with dropout configured, mode restored, and rewards/loss equal to a hand computation from one shared encoding. Verified that it fails with `disable_dropout=False` (loss 1.93 / 1.50).
- `make_observation` N-mismatch checks.
- New tests in test_vtla_dataset.py:
- `test_stage_config_rejects_unknown_keys`, including the bf16 hash.
- `test_stage_splits_json_shared_with_other_stages`: `save_splits` relative paths plus episode ids; unknown and unlisted entries are warned.
- `test_stage_run_with_pretrained_frozen_tactile_encoder`: the pretrained spec wins with a warning, frozen weights reach the bundle bit-exact, obs_history=2 with `history_emb`, and the bundle rebuilds.
- `robot_joint` + `delta_pose` raising.
### reviewer remaining concerns
1. **Proprio for `hand_mano` uses the absolute wrist position and orientation in the hand-label frame** (camera/world, from HaMeR; the ego camera moves with the head). Deploying on a robot needs a hand-equivalent state estimate (CTRL). Consider an option for relative or no-wrist proprio. The implementer's open TODO stands.
2. **PRE smooths hand labels non-causally (filtfilt).** The proprio at tick t therefore carries slight future information that an online causal estimate will not have, so train and deploy inputs can differ slightly.
3. **`sample_phase_mask('task')` fallback:** an episode that has labelled phases but none of the task phases (e.g. D1 motion, if a user sets `datasets: [motion, task]`) samples every labelled frame, including imu_calibration and sync. It only warns, which contradicts the "never imitate calibration" principle; stricter behaviour could be considered.
4. **The ContactGate default (`level_ge_weak`) opens during ADC dropouts and for dead channels.** This is now documented, not changed, because the spec prescribes level ≥ WEAK. On real hardware, prefer `weak_or_strong` or pad dead taxels.
5. **`data.tactile_source` defaults to `auto`.** It silently mixes bootstrap levels (with a warning) for episodes the contact stage missed; the bundle marks this `mixed` or `bootstrap`. Switch the YAML default to `derived` once STAGE1 is always run before VTLA (INTEG e2e).
6. **Weight decay on learned embeddings:** TRAIN's `param_groups` keywords do not match `camera_emb`, `readout` or the flow `pos` embedding, so these get weight decay. Minor; untuned.
7. **Not exercised here (no GPU):** CUDA, real DDP, torch.compile, HF/ResNet encoders.
8. **Interface requests from the implementer still stand:**
   - [INTEG] `__main__` `train vtla` → `load_stage_config` + `run`. `load_stage_config` now also rejects unknown keys, so INTEG must not inject extra top-level keys.
   - [INTEG] Remove or update the unused `configs/default.yaml` `vtla:` block, and update the README row.
   - [CTRL] Online path through `bundle_components`, `make_observation` and `collate_vtla`. `make_observation` now validates N.
   - [STAGE1] Keep the output names `contact_label_pseudo` and `calibrator.json`.
   - [DOCS] τ convention.

I edited only VTLA-owned files, and nothing was committed:
- robot_skin/vtla/{model,heads,dpo,dataset}.py
- robot_skin/vtla/README.md
- robot_skin/stages/vtla.py
- robot_skin/configs/stages/vtla.yaml (comment only)
- robot_skin/tests/test_vtla_model.py
- robot_skin/tests/test_vtla_dataset.py


## CTRL
### public_api
## robot_skin.control
Exports are lazy (PEP 562), so `import robot_skin.control` does not import torch.

### interfaces.py
- **Protocols**
  - `RobotHandInterface`: `joint_names`, `lower`, `upper`, `read_state() -> (t, q[D], qd|None)`, `read_pressure() -> (t, raw[C] in channel order)` and `send_joint_targets(q[D])`. Optional: `velocity_limits`, `start`/`stop`/`estop`, `layout`, `urdf_xml`/`urdf_path`. Timestamps are on the host clock.
  - `CameraInterface`: `name`, `read() -> (t, uint8[H,W,3]|None)`.
  - `check_robot` / `check_camera` validate an object against these.
- **Helpers**
  - `load_urdf_model(urdf=None) -> (URDFModel, xml)`; `None` gives the synthetic 16-DoF hand.
  - `finger_joint_groups(model)`, `taxel_coupling(layout, model) -> W[N,D]`.
  - `layout_tip_offsets(layout, tip_links)`, `urdf_tip_fk(model, tip_links, tip_offsets, base_link)`: a differentiable fk for `FingertipRetargeter`.
  - `SYNTHETIC_HAND_HUMAN_TO_ROBOT`: the 3x3 rotation from the MANO frame to the synthetic hand's frame.
- `VirtualObject(angle=0.9 | {finger: rad}, compliance_rad=0.25, fingers, palm=True)`.
- `FakeRobotHand(urdf=None, layout="robot_hand_template", *, clock=SimClock, sim_hz=400, tau_s=0.05, q0, velocity_limit, obj, n_channels, noise_pct, artefact_*, press_*, baseline_raw, seed)`
  - Joints track targets first-order and are clipped to the velocity limits.
  - The virtual object blocks fingers at `angle + compliance`.
  - The skin model: lagged artefact `g·u + c·u² + h·u̇` with `u = W q`, a press `−P_max(1−e^{−k·pen/P_max})`, noise, and ADC rails.
  - Methods: `read_state`, `read_pressure`, `send_joint_targets`, `estop`/`reset_estop`, `inject_dropout(taxels, s)`, `truth()`, `closure()`.
- `FakeCamera(name, hand, *, hw, rate_hz, clock, seed)` with `.read()`.

### online.py
- `CausalJointVelocity(hz, method="savgol_causal", window_s, polyorder)` with `.push(q) -> qd` and `.reset()`. Equals `datasets.build.joint_velocity`; non-causal methods raise.
- **Pose functions**: `glove_pose_fn(layout, skeleton)`, `robot_pose_fn(layout, urdf)` (numpy chain FK, same result as `URDFModel.fk` to 1e-8, about 0.2 ms), `static_pose_fn`, `make_pose_fn`.
- `load_calibrator(obj | dict | path | dir)`.
- `startup_calibrator(residual[T,N], saturated, logvar, **fit_kw)`: fits a bring-up calibrator from a still hold.
- `TactileFrame` fields: `t`, `raw`, `delta`, `saturated`, `baseline_mean`/`logvar`, `residual`, `residual_z`, `level`, `untrusted`, `press_pct`, `contact`, `features`, `prob`, `contact_on`, `q`, `qd`, `pos`, `nrm`; property `.any_contact`.
- `OnlineTactileProcessor(layout, baseline_model=None, calibrator=None, *, joint_stats, fsm, window, hz, qd, qd_source, feature_spec, contact_rule, pose_fn, urdf, skeleton, detector, hysteresis, pressure, baseline_raw, baseline_s, raw_order="channel"|"layout", device, joint_names)`
  - Construction: `.from_stage_outputs(layout, baseline=, calibrator=, detector=)`; `.from_policy_bundle(bundle, layout, ...)` takes the feature spec and contact rule from the bundle.
  - Baseline capture: `begin_baseline` / `add_baseline_sample(raw, t)` / `finish_baseline(duration)` / `capture_baseline(rows, t)`; `set_baseline`; `baseline_ready`.
  - Streaming: `step(raw, q, *, qd, pos, nrm, t, saturated, q_valid) -> TactileFrame`; `reset(keep_baseline=True)`; `set_calibrator`; `fsm_config`.
- `replay_episode(processor, episode, *, poses="episode"|"model", qd, baseline, extra_saturated, frames) -> {key: [T,...]}`.

### bundle.py
- `load_policy_bundle(path | dir | dict, *, device, allow_bootstrap=True) -> PolicyBundle`. It wraps `vtla.bundle_components`.
- `PolicyBundle` attributes: `policy`, `action_spec`, `action_normalizer`, `rel_mode`, `chunk_offset`, `proprio_normalizer`, `feature_spec`, `contact_rule`, `eval_transform`, `tokenizer`, `cameras`, `policy_hz`, `source_hz`, `stride`, `horizon`, `obs_history`, `head`, `flow_steps`, `tactile`, `vision`, `meta`, `path`.
- Properties: `action_kind`, `action_dim`, `ensemble_k`, `tactile_source`, `uses_tactile`.
- Methods: `calibrator()` (embedded state or path), `baseline_model_path()`, `check_deployable(allow_bootstrap=False)`, `summary()`.

### safety.py
- `SafetyFilter(lower, upper, *, dt, max_vel, max_acc, margin, tactile_stop={enabled, levels, min_ticks, release_ticks, mode: freeze_closing|hold|estop}, closing_sign, taxel_joints[N,D], watchdog={enabled, max_age_s: {stream_glob: s}, estop_after_s}, estop_callback, joint_names)`
  - Methods: `.reset(q)`, `.filter(q_target, q_current, *, t, level, stamps) -> q_cmd`, `.trigger_estop(reason, t)`, `.summary()`.
  - State: `.events` (a list of `SafetyEvent`), `.counts`, `.estopped`, `.stop_active`, `.stale`.
- `taxel_joint_mask(layout, model, joint_names) -> bool[N,D]`, `closing_signs(names, spec)`, `TACTILE_STOP_MODES`.

### runner.py
- `PolicyRunner(robot, bundle, processor, *, cameras, retargeter, safety, control_hz, policy_hz, instruction, ensembler, clock, logger, hand_state_fn, hand_state_init="mean"|"flat"|[54], interpolate=True, device, seed, allow_bootstrap=False, stop_on_estop=True, n_steps)`
  - Methods: `.startup(baseline_s, calib_s)` (baseline capture, optional bring-up calibrator, policy warm-up), `.warmup(n)`, `.begin_rollout()`, `.step() -> TactileFrame`, `.end_rollout()`, `.run(duration_s, *, startup, baseline_s, calib_s, success, close) -> metrics`, `.metrics()`.
  - Metrics: `n_ticks`, `n_policy_ticks`, `loop_hz`, `loop_hz_wall`, `latency_p50_ms`/`p95_ms`, `tick_p50/p95_ms`, `overruns`, `safety_counts`, `estop`, `contact_frac`, `retarget_ms`, `startup`, `session_dir`.
- `DeploymentLogger(session_dir, *, layout, joint_names, clock, cameras, dataset="other", subject, session_id, task_id="deploy", instruction, urdf_xml, camera_format, overwrite, meta, pressure_hz, joint_hz, camera_hz)`
  - Built on the acquisition `Recorder`.
  - Stream methods: `log_pressure`, `log_joint_state`, `log_camera`.
  - Event methods: `phase_start`/`phase_end`, `marker`, `instruction`, `set_baseline`, `log_tick`/`log_policy`.
  - `close(success, meta) -> SessionManifest` writes the RAW session plus `layout.yaml`, `robot.urdf` and `deploy_log.npz`.
- Helpers and constants: `joint_permutation(src, dst)`, `initial_hand_state(bundle, init)`, `DEPLOY_LOG_NAME`.

### latency.py
- `percentile_summary(ms)`, `LatencyMeter` (`.time()` context, `.record`, `.summary`).
- `example_batch(bundle, *, n_taxels, image_hw, instruction, seed)`.
- `benchmark_policy(bundle | path | policy, batch=None, *, device, n, warmup, n_steps, seed, **batch_kw) -> {n, mean_ms, p50_ms, p95_ms, p99_ms, max_ms, device, head, batch_size}`.
- `export_torchscript(module, example_inputs, path, *, method="trace"|"script", check)`: guarded; raises `RuntimeError` when the module cannot be exported.

## robot_skin.stages.deploy
- `STAGE`, `CONFIG_PATH`, `METRICS_NAME`, `DEFAULTS` (mirrored by `configs/stages/deploy.yaml`), `REAL_ROBOT_HELP`.
- `load_stage_config(path, overrides)`, `resolve_config(cfg)`; unknown keys raise.
- `build_retargeter(model, layout, retarget_cfg, *, synthetic)`: `scale="auto"` uses `estimate_scale` between the flat MANO hand and q = 0.
- `build_processor(cfg, bundle, layout, model, device) -> (proc, notes)`, `build_safety(cfg, robot, layout, model, dt)`.
- `run(cfg, *, robot=None, cameras=None) -> metrics`
  - `robot: fake` builds a `FakeRobotHand` and `FakeCamera`s on a SimClock (or a real clock with `realtime: true`).
  - Any other robot without an instance raises `NotImplementedError` with instructions.
  - Writes `<out_dir>/metrics.json` and `<out_dir>/sessions/<id>/`.
  - Metrics: `loop_hz`, `latency_p50_ms`/`p95_ms`, `tick_*`, `overruns`, `safety_counts`, `safety_events`, `estop`, `benchmark`, `notes`, …
- `main(argv)` for `python -m robot_skin.stages.deploy --config y --set k=v`.

## robot_skin.transfer
- `finger_group(name)`, `HAND_GROUPS`.
- `CapsuleSkeleton(p0, p1, radius, names, rank, palmar, groups)` with `.from_mano(skeleton, finger_pose)` (hand frame), `.from_urdf(model, q, *, radius, tip_length, palmar_axis)`, `.lengths`, `.lateral`, `.finger_u_offsets()`.
- `project_to_skeleton(pos[...,N,3], skeleton | (p0,p1,r[,names]), *, taxel_groups, allowed) -> SkeletonProjection`
  - Fields: `segment`, `names`, `finger`, `t`, `u` (along the finger, 0 to 1), `side` (palmar cosine), `v` (palm lateral position), `closest`, `offset`, `distance`, `surface_distance`; property `segment_names`.
  - `project_to_mano(pos, skeleton=None, finger_pose=None)` is the MANO shortcut.
- `taxel_groups(layout)`, `layout_rest_poses(layout, *, skeleton, urdf, q, finger_pose)`.
- `align_layouts(src, dst, *, src_pos, dst_pos, src_nrm, dst_nrm, src_skeleton, dst_skeleton, src_urdf, dst_urdf, k=1, by_group=True, side_weight=.5, lateral_weight=1., normal_weight=0., max_dist) -> LayoutAlignment`
  - Matching is done in skeleton space or in Euclidean space.
  - `LayoutAlignment` fields: `index`, `weight`, `distance`, `valid`, `n_src`, `groups`, `space`. Methods: `.matrix()`, `.to_dict()`/`.from_dict()`/`.save()`/`.load()`.
- `map_taxel_values(values, mapping, *, taxel_axis=-1, reduce="weighted"|"nearest"|"max", fill)`, `REDUCE_MODES`.
- `RobotToManoEstimator(forward_retargeter, skeleton, *, iters=10, method, jacobian="fd")`: lazy export. Methods: `.estimate(q) -> finger_pose[15,3]`; `__call__(q, commanded) -> hand_action[54]` (usable as `hand_state_fn`); `.robot_tips_in_mano(q)`.
- `mano_tip_fk(skeleton)`.
- `transfer.mano_projection` is kept as a compatibility import path.
### deviations
1. **Signatures differ from the spec.**
   - `OnlineTactileProcessor(layout, baseline_model, calibrator, *, joint_stats, fsm, window, ...)`: the calibrator is the third positional argument and everything else is keyword-only.
   - `PolicyRunner(robot, bundle, processor, *, cameras, retargeter, safety, control_hz, policy_hz, instruction, ensembler, ...)`: cameras are keyword-only.
   - `project_to_mano(taxel_pos, skeleton=None, finger_pose=None)` replaces the stub's `(taxel_pos, mano_vertices, mano_segments)`. It uses `ManoSkeleton` capsules rather than mesh vertices, since there is no MANO mesh.
2. **transfer is split into skeleton.py, align.py and reverse.py.** `mano_projection.py` stays as a compatibility import path.
   - Canonical coordinates are finger, `u` along the finger, palmar `side`, and palm lateral `v`.
   - Skeleton-space matching needs both skeletons; otherwise matching is Euclidean.
   - The optional reverse retargeting is implemented as `RobotToManoEstimator`: 15 flexion + 5 abduction parameters, fitted by LM on fingertip vectors.
3. **The tactile stop freezes closing joints at their measured position**, not the last command. A position servo lags its command, so freezing the command kept squeezing. With `taxel_joint_mask` it freezes only the triggering taxel's kinematic chain; palm taxels freeze everything.
4. **Online FK uses its own numpy chain**, which matches `URDFModel.fk` to 1e-8. This cut the processor tick from about 4 ms to about 2 ms on CPU, because torch FK takes about 1.8 ms per call.
5. **hand_mano proprio on a robot defaults to the commanded hand action.** It is initialised from the proprio normalizer centre; optionally `hand_state.estimate` or a `hand_state_fn` supplies it. The wrist part of the action is logged but not executed, since there is no arm.
6. **Deployment sessions default to `dataset: other`**, so they do not leak into VTLA `task` training.
   - Phases: `baseline` and `calibration` (both `no_contact`), then `rollout` (`task`).
   - Sidecar: `deploy_log.npz`.
   - `session.json` uses `layout: layout.yaml` and `meta.urdf: robot.urdf`.
7. **The deploy stage has its own `load_stage_config` / `resolve_config`** instead of `stages.load_stage_yaml`. That helper's `apply_hw_profile` injects a `train` section, which deploy does not have. `hardware` only exports the profile environment and resolves `device: auto`.
8. **The bundle's stage-1 references are used only when they fit the robot skin.** `use_bundle_refs` applies them only when the bundle's `tactile.layouts` names the robot layout (glove_template and robot_hand_template both have N = 9, so taxel count alone cannot tell them apart). Otherwise a note is added.
   - With no calibrator and `startup.calib_s > 0`, a bring-up calibrator is fitted during a still hold.
   - Without a baseline model, residual = ΔS.
9. **The runner does synchronous inference inside the control tick, with a warm-up.** The first `predict` took about 400 ms and would stall real-time loops, so two dummy inferences run during start-up and the deadline is reset at rollout start.
   - Between policy ticks the joint target is linearly interpolated from the last command.
   - Control ticks run at the bundle `source_hz` (warns otherwise).
10. **Extras beyond the spec:**
    - `CausalJointVelocity`, `replay_episode`, `startup_calibrator`, `load_calibrator`.
    - `FakeRobotHand.inject_dropout`/`truth`/`closure`; `load_urdf_model`, `urdf_tip_fk`, `layout_tip_offsets`, `SYNTHETIC_HAND_HUMAN_TO_ROBOT`.
    - `PolicyBundle.check_deployable`, which refuses bundles trained on bootstrap tactile levels.
    - `example_batch`, `export_torchscript`, `LatencyMeter`.
11. **test_config_and_stubs.py:** the transfer-stub `NotImplementedError` assertions were replaced by a test of the implemented behaviour.
### interface_requests
1. **[INTEG] `robot_skin/__main__.py`:** `deploy` should call `robot_skin.stages.deploy.main(argv)`, or `cfg = deploy.load_stage_config(path, overrides)` then `deploy.run(cfg)`.
   - The flags are `--config` and `--set k=v`; `bundle` is required.
   - The e2e test can call `deploy.run({"bundle": <vtla out_dir>, "duration_s": 0.5, "startup": {"baseline_s": 0.2}, "latency": {"benchmark": False}})`.
2. **[INTEG] `robot_skin/README.md`:**
   - The `transfer` row should become implemented: skeleton projection, align_layouts, map_taxel_values, RobotToManoEstimator.
   - Add a `control` row: interfaces with FakeRobotHand, OnlineTactileProcessor, SafetyFilter, PolicyRunner with DeploymentLogger, latency, and the deploy stage.
   - Optionally, `configs/default.yaml` could point to `configs/stages/deploy.yaml`.
3. **[PRE] `docs/DATA_FORMAT.md`:** document deployment sessions.
   - They are kind robot, dataset `other` by default, with `layout: layout.yaml` (session-relative) and `meta.urdf: robot.urdf`.
   - Phases are `baseline`/`calibration` (`no_contact`) and `rollout` (`task`).
   - `meta.deployment` holds the bundle, rates, metrics and safety summary.
   - The `deploy_log.npz` sidecar is not a stream and should be ignored, like `gt_synthetic.npz`. build.py already ignores it; re-ingestion is tested.
4. **[DOCS]** Link `docs/DEPLOYMENT.md` from ARCHITECTURE.md and TRAINING.md. The pipeline ends with `deploy`.
5. **[POSE] `pose/urdf.py`, optional:** `URDFModel.fk` costs about 1.8 ms per single-q call on CPU because of torch per-op overhead. `control.online._NumpyChainFK` is a numpy single-configuration path equal to 1e-8. It could move to `URDFModel.fk_numpy` as a fast path.
6. **[ACQ] `acquisition/sources.py`, optional:** `DeploymentLogger` uses a private duck-typed push source with `Recorder` in synchronous mode. A public `PushSource` (with push/poll) in `acquisition.sources` would make this an official pattern.
7. **[STAGE1]** Baselines trained with `data.q_source: hand_pose_imu` need an online IMU→pose stream, such as an `ImuHandPoseNet` ring buffer; the processor only takes q. A `CausalImuPoseStream` next to `CausalBaselineStream` would close this gap. Robot deployment uses `q_source: q`, so it is unaffected.
8. **[VTLA], optional:**
   - `hand_mano` proprio includes the absolute wrist pose in the camera/world frame, which a hand-only robot cannot observe. An option for wrist-relative or no-wrist proprio would make hand_mano bundles more deployable.
   - The bundle could carry a default instruction (or the task catalog) in `meta` for the deploy stage. Today it falls back to `meta.instruction`, which is absent, or "".
### open_todos
1. **No asynchronous inference.** Policy inference and retargeting run synchronously inside the 200 Hz tick. On CPU the default VTLA takes about 12 ms (chunk head) or 27 ms (flow head), and retargeting about 35 ms, so every policy tick overruns. A background inference thread with delay-compensated ensembling is not implemented.
2. **GPU latency not measured.** Neither the RTX 5090 numbers nor a CUDA sync path have been run here, since there is no GPU; the docs say to measure them from `metrics.json`.
3. **No real hand, IMU or ROS drivers.** Only the protocol and the `NotImplementedError` guidance exist. mk555 `.bin` parsing must wrap `deformable_sats/sats/preprocessing/bin_merge.py`.
4. **Phenomenological fake robot.** The `FakeRobotHand` skin and object model are not fitted to real mk555 data. The virtual object is a closure threshold, not contact geometry.
5. **hand_mano deployment gaps.** The wrist action is unused because there is no arm controller. `hand_state.estimate` (RobotToManoEstimator) adds about 30 ms per policy tick on CPU.
6. **Coarse palm matching.** Cross-embodiment palm alignment uses the wrist→knuckle rays. `side` needs a user-given `palmar_axis` for URDF hands; robot_hand_template taxels lie on the link axes, so their `side` is NaN.
7. **Synthetic-hand defaults only.** `human_to_robot` defaults exist only for the synthetic hand; real hands must configure `retarget.human_to_robot`, `tip_links`, `tip_offsets` and `safety.closing_sign`.
8. **TorchScript is deprecated.** It is deprecated in torch ≥ 2.9, and the whole VTLA policy (string/dict inputs) cannot be traced; only tensor-only submodules can be exported. A `torch.export` path is not implemented.
### reviewer fixes
- online.py: new _check_calibrator. A use_logvar calibrator without a baseline model raises a clear ValueError in the constructor and in set_calibrator. from_policy_bundle drops such a bundle calibrator with a warning. deploy.build_processor drops it with a note, and the start-up calibrator stands in (it also notes a missing bundle baseline file).
- online.py: robot_pose_fn(layout, urdf, joint_names=None) reorders q from joint_names order to URDF order, and make_pose_fn forwards joint_names. For URDF skins the processor's joint_names defaults to the URDF actuated-joint order, so PolicyRunner permutes the driver order into it. It validates joint_names length against baseline_model.joint_dim.
- online.py: add_baseline_sample times samples by tick index including skipped samples (_bl_n counter).
- online.py: a non-finite q holds the last finite reading and marks the tick q_valid=False (raises if there is no previous reading). step and module docstrings updated.
- bundle.py: new PolicyBundle.stage1_matches(layout), moved from deploy._stage1_ref_ok. from_policy_bundle and deploy.build_processor both use it.
- deploy.py: new _baseline_fits checks taxel count, model kind, and that the model's joints are robot joints. build_processor takes robot_joints; run passes robot.joint_names.
- runner.py: honours chunk_offset. Offset 0 drops chunk[0] before TemporalEnsembler.add; offset > 1 warns; horizon 1 with offset 0 raises.
- runner.py: new _blank_image(). Before a camera's first frame the runner uses the eval transform of a zero image with vision_valid False, the dataset's rule; it raises only if the transform has no fixed output size.
- runner.py: begin_rollout always resets the safety filter from the measured q, warning if this clears a latched e-stop. It also resets the hand_state_fn, the loop counters and the meters. metrics uses explicit None checks. The deadline is re-synced after the warm-up.
- runner.py: start-up warns about dead taxels. The manifest baseline uses the processor's baseline in channel order.
- runner.py DeploymentLogger: pressure and joint_state samples are logged once per new timestamp. qd is written only when the driver measures it; the first sample fixes the schema and later gaps become NaN.
- safety.py: trigger_estop(reason, t, q_hold=None). A tactile-mode e-stop holds the measured position.
- interfaces.py: FakeRobotHand integrates by an integer step count from t0 (t = t0 + k*sim_dt), so there is no drift.
- Tests added in test_control.py:
- test_processor_holds_the_last_joint_reading_on_a_driver_glitch
- test_baseline_capture_window_is_real_time_when_rail_samples_are_skipped
- test_runner_maps_driver_joint_order_and_logs_only_measured_qd (reverse-order driver without qd, pose equality with FK, no qd in joint_state.npz, strictly increasing stamps, re-ingestion gives URDF order)
- test_runner_honours_chunk_offset_and_blanks_cameras_before_their_first_frame (patched predict: offset 1 gives chunk[0], offset 0 gives chunk[1]; blank image equals the dataset stand-in)
- test_runner_metrics_describe_each_rollout
- Assertions added to existing tests:
- tactile e-stop holds the measured q
- FakeRobotHand keeps up with SimClock(12345.678)
- log-variance calibrator without a baseline raises
- deploy with a log-variance calibrator and a missing baseline falls back to the start-up calibrator
- from_policy_bundle skin matching
- make_bundle gained cal_logvar and baseline_ref
- the start-up capture check in the online≡offline test no longer builds a processor that would crash on step()
- test_transfer.py: test_projection_matches_hand_derived_coordinates checks t, u across a two-capsule finger, side, distances, closest point and the palm lateral coordinate v against hand-derived values.
- Docs:
- docs/DEPLOYMENT.md: chunk_offset handling, camera stand-in, log-variance calibrator rule, dead-channel behaviour, real-time baseline window, driver joint order and qd=None, the tactile e-stop at the measured position, and the e-stop latch cleared at rollout start
- control/README.md: joint-order and calibrator rules, stage1_matches
- runner module docstring
### reviewer remaining concerns
1. By design, not changed:
   - Policy inference and retargeting are synchronous inside the 200 Hz tick. On CPU every policy tick overruns; there is no async inference thread.
   - hand_mano proprio is the commanded hand action, and wrist actions are unused because there is no arm.
   - A dead channel or a SaturationFSM recovery (up to max_recover_s = 30 s) reads SATURATED, so the tactile stop keeps that chain from closing. This is fail-safe, and start-up now warns about dead channels.

2. begin_rollout now clears a latched software e-stop, with a warning, treating a new rollout as the operator's explicit restart. The hardware e-stop, for example FakeRobotHand.estopped, still needs its own reset.

3. deploy.yaml defaults closing_sign to 1.0 for every joint, including thumb_cmc_abd. The docs recommend 0 for abduction joints on real hands.

4. The online processor's glove path uses the default ManoSkeleton. Preprocessing uses hand_pose.mano_model when that is configured. This does not matter for robot deployment.

5. Untested here:
   - GPU / CUDA sync paths
   - a real-time clock with real hardware
   - CameraSource
   - TorchScript on newer torch (deprecated)

6. Nothing was committed. Every edit is in CTRL-owned files:
   - robot_skin/control/{online,runner,safety,interfaces,bundle}.py and control/README.md
   - robot_skin/stages/deploy.py
   - docs/DEPLOYMENT.md
   - robot_skin/tests/test_control.py and test_transfer.py

7. The implementer's interface requests still stand:
   - [INTEG] __main__ `deploy` and the README rows
   - [PRE] document deployment sessions
   - [DOCS] links
   - optional [POSE] numpy FK fast path
   - optional [ACQ] public PushSource
   - [STAGE1] IMU pose stream
   - [VTLA] wrist-free proprio and a default instruction
