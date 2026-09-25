# robot_skin framework — implementation contract (internal, for implementation agents)

Repo: <repo> (monorepo). Branch restructure/robot-skin. DO NOT commit/push — the
orchestrator commits. Python venv (use this interpreter for everything):
  V=python
  cd <repo> && $V -m pytest -q -p no:cacheprovider robot_skin/tests/test_<yours>.py
Installed: numpy 1.26, torch 2.14 (CPU only here; no GPU in this box), scipy, PyYAML, Pillow, pytest.
NOT installed (must be optional imports, guarded, never required by tests): torchvision, transformers,
timm, opencv (cv2), pyserial (serial), tensorboard, wandb, optuna, mujoco.

## Goal (user)
Framework from data acquisition → preprocessing → optimization → training-structure design → VTLA
model → robot control. Two datasets will be collected with a tactile glove (barometric taxels,
mk555-type) + 7 IMUs + cameras:
- D1 `motion`: vision + hand shape (IMU) + tactile while moving (free motion, no object; incl.
  self-touch sets). Purpose: IMU→hand-pose model, motion-induced tactile baseline predictor,
  contact detector calibration (self-touch = free contact labels).
- D2 `task`: vision + hand shape + tactile + object tasks with language instructions. Purpose:
  VTLA training (vision + tactile + language → action), later robot control.
Training runs on various GPUs (RTX 5090 etc.) reached via Tailscale → hardware profiles,
bf16 AMP, torch.compile opt-in, DDP via torchrun; everything must also run on CPU for tests.

Design decisions (defaults, configurable):
- Cameras: any number, named; default `ego` (head-mounted, egocentric) + `third` (fixed). Stream
  names `camera_<name>`.
- Canonical action = human hand action (MANO): wrist position (3) + wrist rotation 6D (6) + 15 finger
  joints axis-angle (45) = 54-D, chunked (ACT). Robot control retargets hand action → robot joint
  targets (fingertip-vector matching, AnyTeleop/DexPilot style). A robot-joint action space
  (`robot_joint`, D-dim) is also supported for when robot teleop data exists.
- Instructions: generated from task-catalog templates (acquisition/protocols/d2_task.yaml), operator
  may override per episode (events.jsonl `instruction` event or manifest.task.instruction).
- Tactile sign: ΔS% = (raw−baseline)/baseline·100 (SATS convention; press → NEGATIVE).
  `common.signal.press_intensity(x) = −x` is press-positive. Thresholds are press-positive.

## Hard rules
1. Dependency direction: robot_skin → common. robot_skin must NOT import sats/hitmap/deformable_sats.
   Do NOT modify common/ or deformable_sats/ (read-only). (common/tests/test_dependency_direction.py)
2. Only create/edit files you OWN (listed per agent). If you need a change in someone else's file,
   do not edit it — report it in `interface_requests` in your final output.
3. Every public module gets unit tests in robot_skin/tests/test_<area>*.py (you own those files).
   Tests: CPU, deterministic (seeded), fast (< ~10 s per file), no network, no GPU, no optional deps.
   Use tmp_path for any disk IO. Keep existing tests passing; if you replace a stub that an existing
   test asserts raises NotImplementedError, update that test (you own it if listed).
4. Match the style of existing code: `from __future__ import annotations`, dataclasses, numpy/torch,
   module docstring explaining purpose + paper grounding, concise comments. Korean is fine in READMEs.
5. Config: every stage runner reads a plain dict (from YAML via robot_skin.config.load_config /
   yaml.safe_load). Your stage YAML lives in robot_skin/configs/stages/<stage>.yaml.
6. No fabricated citations. Only cite papers listed in "References" below (verified).

## Existing code you can use (read it)
- common.signal: relative_change, press_intensity, PRESS_SIGN, estimate_baseline, NormStats(fit/apply/
  invert/to_dict/from_dict/save/load), saturation_mask, ADC_MIN/ADC_MAX.
- common.timeline: Stream(t, values, method="linear"|"zoh"), master_clock, resample, align_streams(
  streams: dict, hz=200, span="intersection"|"union") -> (t_master, values_by_name, valid_by_name).
- common.layouts: load_layout(name|path) -> Layout(.n, .channels, .positions [N,3] m, .normals,
  .parents, .groups, .imu_sites, .by_channel(raw)), MANO_SEGMENTS (wrist, palm, thumb1..3, index1..3,
  middle1..3, ring1..3, pinky1..3), grid_layout. Built-ins: sats_4x4, glove_template (9 taxels,
  parent_frame mano, 7 imu_sites wrist/palm/thumb/index/middle/ring/pinky with parents wrist,palm,
  thumb3,index3,...), robot_hand_template (parent_frame urdf, links thumb_distal_link, ...,
  palm_link).
- robot_skin.geometry.rotations (DONE, tested): as_tensor, skew, aa_to_matrix, matrix_to_aa,
  quat_normalize, quat_conj, quat_mul, quat_apply, quat_to_matrix, matrix_to_quat, quat_to_aa,
  aa_to_quat, quat_fix_continuity, matrix_to_6d, sixd_to_matrix, rpy_to_matrix (URDF: Rz@Ry@Rx),
  geodesic_distance, make_transform, invert_transform, transform_points. Quats wxyz; 6D = first two
  columns.
- robot_skin.acquisition.manifest (v2): SessionManifest(kind glove|robot|bench, layout, streams{name:
  StreamInfo(file, rate_hz, fields, clock, method)}, session_id, created_utc, master_hz, segments
  [{t0,t1,label}], baseline, notes, meta, dataset motion|task|other, subject, task{task_id,
  instruction, object, success}|None, calibration{}), .save(dir)/.load(dir), .spans(label),
  .add_segment, .stream_path, .cameras. Raw stream file formats are in its module docstring:
    pressure.npz: t[T] float64 s, raw[T,C]
    imu.npz: t, quat[T,S,4] wxyz, gyro[T,S,3], acc[T,S,3], sites[S] (str)
    joint_state.npz: t, q[T,D], qd?, tau?, names[D]
    hand_pose.npz: t, global_orient[T,3], finger_pose[T,15,3] (MANO order), wrist_pos[T,3], confidence[T]
    object_pose.npz: t, pos[T,3], quat[T,4]
    camera_<name>/timestamps.npy [F] + frames.npy uint8[F,H,W,3] or %06d.jpg
    events.jsonl: {"t","type": phase_start|phase_end|marker|success|instruction,"name","value"}
  Segment labels used: "no_contact" (baseline training), "self_touch", "calibration", "task".
- robot_skin.datasets.episode (DONE, tested): Episode(meta: EpisodeMeta, arrays, static, root),
  .T, .t, .has, [key], .phase_mask(name), .derived/.has_derived/.set_derived, .get_frame/.frame_at/
  .frame_timestamps, .save(root), Episode.load(root, mmap=True, keys=None), list_episodes(root,
  dataset). Keys: K_T, K_PRESSURE_RAW, K_DELTA, K_SATURATED, K_TAXEL_POS, K_TAXEL_NRM, K_Q, K_QD,
  K_IMU_QUAT, K_IMU_GYRO, K_IMU_ACC, K_HAND_GLOBAL, K_HAND_FINGERS, K_HAND_WRIST, K_HAND_VALID,
  K_OBJECT_POS, K_OBJECT_QUAT, K_PHASE, K_SELF_TOUCH, K_CONTACT_LABEL, cam_idx_key(name);
  static S_BASELINE_RAW, S_CHANNELS; derived D_BASELINE_PRED, D_BASELINE_LOGVAR, D_RESIDUAL,
  D_RESIDUAL_Z, D_CONTACT_PROB, D_LEVEL, D_HAND_POSE_IMU.
  Processed root: robot_skin/data/processed/<dataset>/<episode_id>/.
- robot_skin.pose.provider: TaxelPoseProvider protocol (n_taxels, pose_at(t)->(pos[N,3],nrm[N,3])),
  transform_taxels(layout, {parent: 4x4}), TransformPoseProvider, StaticPoseProvider, sample_poses.
- robot_skin.baseline (v1): BaselinePredictor(n_taxels, joint_dim, hidden, depth, taxel_embedding_dim)
  forward(pos[B,N,3], nrm, q[B,D], qd) -> [B,N] (zero-init); NoContactSession, NoContactWindowDataset,
  train_baseline, predict_session.
- robot_skin.contact: residual, contact_mask, OrdinalQuantizer(weak_pct, strong_pct) -> levels
  ContactLevel{NONE,WEAK,STRONG,SATURATED}, .one_hot; SaturationFSM(n, ok_pct, ok_sec,
  max_recover_s).step/.run/.trusted/.corrected/.offset; self_touch_labels(taxel_pos[T,N,3],
  taxel_parents, seg_p0[T,S,3], seg_p1, seg_radius, seg_names, margin), finger_of,
  point_segment_distance.
- robot_skin.representation: TaxelTokenizer(value_dim, d_model, n_fourier, fourier_scale, n_taxels)
  forward(values[B,N,F], pos[B,N,3], nrm[B,N,3], mask[B,N]|None) -> [B,N,D]; random_taxel_mask.
- robot_skin.policy: OBS_MODES (full, ordinal, binary, none), tactile_features(mode, resid, sat,
  quantizer, scale_pct), ObservationBuilder.
- robot_skin.vtla: TactileTokenAdapter(d_in, d_out, n_query, n_heads, gate) forward(tokens[B,N,d],
  contact[B,N]) -> [B,K,d_out]; ContactGate(hard|soft).
- robot_skin.sim.TaxelDomainRandomizer; robot_skin.eval metrics (hallucination_rate, auroc,
  motion_contact_separability, saturation_recovery_times).
- robot_skin.config: load_config(path=None, overrides=None), deep_merge.

## Module map and ownership

### WAVE 1 (foundations)

[POSE] owns: robot_skin/pose/mano.py, robot_skin/pose/urdf.py, robot_skin/pose/robot_fk.py (replace
stub), robot_skin/pose/imu_model.py, robot_skin/pose/glove_imu2mano.py (replace stub),
robot_skin/pose/vision_hand.py, robot_skin/pose/README.md, robot_skin/pose/__init__.py (may extend
exports), tests: robot_skin/tests/test_pose.py (update stub asserts), test_mano.py, test_urdf.py,
test_imu_model.py.
- mano.py: MANO_JOINTS (16, MANO order: wrist, index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3),
  MANO_PARENTS (-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14), FINGERS, SEGMENT_TO_JOINT mapping for every
  common.layouts.MANO_SEGMENTS name ("palm" → wrist joint frame with a fixed palm-centre offset;
  "wrist" → joint 0). Default rest skeleton (approximate adult right hand, metres, MANO canonical
  frame, wrist at origin) + fingertip offsets. class ManoSkeleton(rest_joints=None, tip_offsets=None):
  forward(global_orient[...,3], finger_pose[...,15,3], wrist_pos[...,3]) -> dict(joint_T[...,16,4,4],
  joint_pos[...,16,3], tip_pos[...,5,3]) (torch; accepts numpy via as_tensor);
  segment_transforms(fk) -> {segment_name: [...,4,4]} for all MANO_SEGMENTS;
  capsules(fk) -> (p0[...,S,3], p1[...,S,3], radius[S] np, names list[S]) for 15 finger segments +
  palm; skeleton-only (no mesh / no MANO pkl needed); optional `from_mano_pkl(path)` that reads rest
  joints if the user has MANO (guarded, may raise NotImplementedError with instructions).
  Also `ManoPoseProvider(layout, skeleton, pose_at_fn)` implementing TaxelPoseProvider via
  transform_taxels + segment_transforms, and `taxel_poses_from_hand(layout, skeleton, global_orient
  [T,3], finger_pose[T,15,3], wrist_pos[T,3]) -> (pos[T,N,3], nrm[T,N,3])` numpy (batched, used by
  preprocessing) and `self_touch_from_hand(layout, skeleton, ...same..., margin) -> bool[T,N]` wrapping
  contact.self_touch_labels with capsules.
- urdf.py: minimal URDF parser (xml.etree): links, joints (revolute, continuous, prismatic, fixed;
  origin xyz/rpy, axis, limits, mimic optional-ignore with warning). class URDFModel: from_string,
  from_file, joint_names (actuated, file order), link_names, root_link, lower/upper limits (np),
  fk(q[...,D] torch) -> {link: [...,4,4]} differentiable, fk_numpy(q) convenience.
- robot_fk.py: RobotFKPoseProvider(layout, urdf: URDFModel|path, joint_state_at: Callable[[float],
  np.ndarray]) implemented (parent_frame must be urdf); plus taxel_poses_from_joints(layout, model,
  q[T,D]) -> (pos, nrm) batched numpy for preprocessing.
- imu_model.py: imu_features(quat[...,W,S,4], gyro, acc, wrist_index=0) -> [...,W,S*F] with each
  site's orientation expressed relative to the wrist IMU (q_wrist⁻¹⊗q_site as 6D), gyro/acc rotated
  into the wrist frame; calibrate_imu_offsets(quat_calib[T,S,4], ref_rot[S,3,3]|None) -> offsets
  [S,4] (static flat-hand/T-pose calibration: q_offset = mean(q_meas)⁻¹ ⊗ q_ref), apply_imu_offsets.
  class ImuHandPoseNet(nn.Module): GRU (or causal TCN) over IMU feature windows → finger pose 15×6D
  (+ optional global orient 6D); forward(feat[B,W,F]) -> finger_rot6d[B,15,6]; helper
  to_axis_angle. Loss hand_pose_loss(pred6d, gt_aa, skeleton=None, tip_weight) = geodesic rotation
  loss + optional fingertip position loss via ManoSkeleton (Paper grounding: user-specified VIHand
  VIFNet-S is the intended pretrained backbone; this is the in-house baseline with the same I/O).
- glove_imu2mano.py: GloveImu2ManoPoseProvider(layout, model: ImuHandPoseNet, skeleton, imu arrays,
  t array, window) implemented offline (precompute poses); finetune_vifnet_s stays a documented stub
  (NotImplementedError: external weights) + `load_vifnet_s` stub.
- vision_hand.py: VisionHandEstimator Protocol (estimate(frames uint8[B,H,W,3]) -> dict
  global_orient[B,3], finger_pose[B,15,3], wrist_pos[B,3], confidence[B]); save_hand_labels(path,
  t, ...) / load_hand_labels(path) for hand_pose.npz; HaMeREstimator stub (NotImplementedError,
  explains running HaMeR/WiLoR offline to produce hand_pose.npz); `smooth_hand_labels` (confidence-
  gated SLERP/interp over gaps + low-pass) numpy.

[TRAIN] owns: robot_skin/train/{__init__,engine,hardware,distributed,optim,logging_utils,sweep,
checkpoint}.py, robot_skin/train/README.md, robot_skin/configs/hardware/{rtx5090,rtx4090,rtx3090,
a100,cpu}.yaml, tests robot_skin/tests/test_train_engine.py, test_hardware.py, test_sweep.py.
- hardware.py: resolve_device("auto"|"cpu"|"cuda"|"cuda:1"|"mps") ; resolve_precision(pref
  "auto"|"bf16"|"fp16"|"fp32", device) -> PrecisionPlan(autocast_dtype|None, use_grad_scaler) (auto:
  cuda with bf16 support → bf16; older cuda → fp16+GradScaler; cpu/mps → fp32);
  enable_tf32(); load_hw_profile(name|path) -> dict (profiles dir robot_skin/configs/hardware);
  apply_hw_profile(stage_cfg, profile) (profile keys under `train:` override); describe_environment()
  -> dict (torch/cuda versions, GPU names, capability, arch list); check_arch_support() -> list of
  warnings (e.g. GPU capability (12,0) Blackwell/RTX 5090 requires torch built with sm_120 / CUDA ≥
  12.8 — check torch.cuda.get_arch_list()).
  Profiles: rtx5090 (32 GB, bf16, compile true, batch/grad_accum suggestions, num_workers 8), rtx4090
  (24 GB), rtx3090 (24 GB, bf16 ok on Ampere), a100 (80 GB), cpu (fp32, compile false, workers 0).
- distributed.py: DistInfo(rank, world_size, local_rank, is_main, backend); init_distributed()
  (reads torchrun env RANK/WORLD_SIZE/LOCAL_RANK; nccl on cuda else gloo; no-op single process);
  wrap_ddp(model, info); make_sampler(dataset, info, shuffle, seed); barrier(); cleanup();
  all_reduce_mean(dict of floats).
- optim.py: param_groups(model, weight_decay) (no decay for bias, norm, embeddings, 1-D params,
  names containing "token"/"query"); build_optimizer(model, name adamw, lr, weight_decay, betas,
  fused auto on cuda); build_scheduler(opt, schedule cosine|constant|linear, warmup_steps,
  total_steps, min_lr_ratio) (LambdaLR); class EMA(model, decay) update/apply_to/state_dict.
- logging_utils.py: JsonlLogger(out_dir) log(step, dict), optional TensorBoard/W&B sinks (guarded).
- checkpoint.py: save_checkpoint(path, **state) atomic (tmp+rename); load_checkpoint(path,
  map_location); find_last(out_dir).
- engine.py: @dataclass TrainConfig(max_epochs=10, max_steps=None, batch_size=64, lr=3e-4,
  weight_decay=0.05, warmup_steps=100, schedule="cosine", min_lr_ratio=0.1, grad_clip=1.0,
  grad_accum=1, precision="auto", compile=False, ema_decay=None, num_workers=0, pin_memory=True,
  log_every=50, eval_every_epochs=1, ckpt_every_epochs=1, out_dir="robot_skin/runs/default", seed=0,
  device="auto", deterministic=False, early_stop_patience=None, monitor="val/loss");
  TrainConfig.from_dict(d) ignoring unknown keys (warn).
  class Trainer(model, loss_fn, cfg, train_data, val_data=None, collate_fn=None, extra_state=None):
  loss_fn(model, batch) -> dict[str, Tensor] with key "loss" (other entries logged). Handles device
  moves of nested batches (dict/list/tuple/Tensor), DDP, sampler set_epoch, AMP autocast + GradScaler,
  grad accumulation, clipping, scheduler step per optimizer step, EMA, torch.compile opt-in (skip on
  cpu if unsupported), best-by-monitor checkpoint (ckpt_best.pt) + ckpt_last.pt, resume(path),
  early stopping, history list of dicts, returns history from fit(). evaluate() averages loss dict
  over val set (uses EMA weights if enabled). Checkpoint dict keys: model, optimizer, scheduler,
  scaler, ema, step, epoch, config, extra. `Trainer.load_model_weights(model, path, use_ema=True)`
  static helper for inference.
  seed_everything(seed, deterministic).
- sweep.py: expand_grid(space: dict[path, list]) -> list[dict] of dotted-path overrides → nested
  dicts; sample_random(space, n, seed) (lists = choice, {"log_uniform":[a,b]}, {"uniform":[a,b]},
  {"int":[a,b]}); run_sweep(train_fn(cfg)->float metric, base_cfg, overrides, out_dir) writes
  results.jsonl and returns sorted results; optuna integration optional/guarded.

[VISLANG] owns: robot_skin/vision/{__init__,encoders,transforms,feature_cache}.py,
robot_skin/vision/README.md, robot_skin/language/{__init__,text_encoders}.py,
robot_skin/language/README.md, tests test_vision.py, test_language.py.
- vision.encoders: VisionEncoder base (nn.Module) with .out_dim, .n_tokens(h,w), forward(images
  float[B,3,H,W] normalized) -> tokens [B,P,D]; TinyConvEncoder(out_dim, patch grid) trainable
  (default for tests / scratch); ResNetEncoder(name resnet18|34, pretrained, frozen, out_dim) using
  torchvision if installed (guarded ImportError with message); HFVisionEncoder(model_id e.g.
  facebook/dinov2-small, google/siglip-base-patch16-224) via transformers if installed (guarded);
  build_vision_encoder(cfg dict) factory. SpatialSoftmax/keypoint pooling optional.
- vision.transforms: to_float_tensor(uint8 [B,H,W,3] np/torch) -> [B,3,H,W] in [0,1];
  Normalize(mean,std) (ImageNet defaults); resize_short / center_crop; TrainAugment (random resized
  crop small scale range, color jitter brightness/contrast, all torch ops, seeded generator);
  EvalTransform.
- vision.feature_cache: cache_episode_features(episode, camera, encoder, transform, batch_size,
  device, key) -> writes derived/vision_<key>_<camera>.npy [F,P,D] float16 (per frame, not per master
  tick) + returns path; load_cached(episode, camera, key). (Frozen-encoder training speed-up.)
- language.text_encoders: TextEncoder base: encode(list[str]) -> (tokens[B,L,D], pad_mask[B,L] True
  = padding); HashingTextEncoder(dim, max_len, vocab_size=2**16) — lowercase word tokenizer + stable
  hash (zlib.crc32) → nn.Embedding + learned positional embedding (trainable, deterministic, no
  deps); HFTextEncoder(model_id e.g. openai/clip-vit-base-patch32 / google/siglip / t5-small,
  frozen) guarded; build_text_encoder(cfg); InstructionCache (dict str→tensor) for frozen encoders.

[ACTION] owns: robot_skin/action/{__init__,space,chunking,retarget}.py, robot_skin/action/README.md,
tests test_action.py, test_retarget.py.
- space.py: @dataclass ActionSpec(kind "hand_mano"|"robot_joint", dim, names, rot_repr) ;
  HAND_MANO_DIM = 54; hand_action_from_arrays(global_orient[T,3], finger_pose[T,15,3],
  wrist_pos[T,3]) -> [T,54] (wrist pos 3 | wrist rot 6D 6 | fingers aa 45); hand_action_to_arrays(a)
  inverse (6D→matrix→aa); robot_action_from_q(q[T,D]); ActionNormalizer (wraps common NormStats;
  fit on train episodes; skip normalizing 6D? → normalize all dims with std method, keep eps) with
  to_dict/from_dict; relative/delta option: actions expressed relative to current state
  (make_relative(actions_chunk, state) / make_absolute) for wrist position.
- chunking.py: action_chunk(actions[T,A], t_index, horizon, stride) -> (chunk[H,A], valid[H] bool)
  (future steps t+stride, t+2·stride …, padded with last valid + mask False; ACT); downsample
  indices helper for policy_hz vs 200 Hz (stride = round(200/policy_hz)); TemporalEnsembler(horizon,
  action_dim, k=0.01) (ACT exponential weighting w_i = exp(−k·i), i=0 oldest) with add(chunk) /
  step() -> action, reset().
- retarget.py: class FingertipRetargeter(fk_fn: Callable[[q torch[...,D]], dict name->pos[...,3]]
  or URDFModel + tip_links mapping {finger: link}, lower, upper, human_tip_names order, scale=1.0,
  reg_weight, smooth_weight, iters, lr): retarget(human_tip_pos[5,3] relative to wrist,
  human_wrist_T?, q_init) -> q[D] via torch autograd (Adam or LBFGS) minimizing
  Σ||s·(p_r,i−p_r,base) − (p_h,i−p_h,wrist)||² over fingertip & thumb-to-finger vectors
  (DexPilot-style pair vectors) + reg||q−q_prev||² with clamping to limits; batch/sequence helper
  retarget_sequence with warm start. Must work with any fk callable (tests use a toy planar finger
  FK written in the test).

[REFS] owns docs/REFERENCES.md only. Web-verify each reference (title, authors, venue/year,
arXiv id/URL) and write a table mapping paper → what we take → where in code. Also try to verify the
user-named "VIHand"/"VIFNet-S" (IMU-based hand pose); if it can't be found, say so explicitly in the
doc (do not invent details).

### WAVE 2 (consumers; start after wave 1 is merged)

[ACQ] owns robot_skin/acquisition/* except manifest.py (you may ADD fields only if essential → report),
robot_skin/acquisition/protocols/{d1_motion,d2_task}.yaml, robot_skin/acquisition/README.md,
docs/DATA_ACQUISITION.md, tests test_acquisition.py (update), test_recorder.py, test_protocol.py,
test_sync_qc.py.
- sources.py: StreamSource protocol (name, start(), poll() -> list of (t_host, sample dict), stop());
  FakePressureSource / FakeImuSource / FakeCameraSource / FakeJointSource (deterministic synthetic,
  for tests & --fake), SerialPressureSource (pyserial guarded; frame parser injectable; note mk555
  .bin parsing canonical in deformable_sats/sats/preprocessing/bin_merge.py — do not copy),
  ImuSource / CameraSource (cv2 guarded) / RosJointStateSource stubs with clear TODO.
- recorder.py: Recorder(sources, session_dir, manifest) – threaded polling (or synchronous step()
  for tests), host monotonic clock with session t0, writes the raw file formats on stop(), EventLog
  (events.jsonl: phase_start/end, marker, success, instruction), manifest segments from events
  (phase names with contact expectation → segments labels no_contact/self_touch/task).
- protocol.py + YAML: D1 protocol = ordered blocks (static baseline 5 s no_contact; IMU calibration
  flat-hand pose; per-finger slow flex/extend; whole-hand open/close slow/fast; wrist rotations;
  random free motion; self-touch sets: thumb–index/middle/ring/pinky pinch, fist, finger crossing,
  palm touch; final static baseline), each with duration, repetitions, expected contact label
  (no_contact / self_touch), speed tag. D2 = task catalog: tasks (e.g. grasp_lift_place, pour,
  peg_insert, open_jar, wipe, handover, press_button, in-hand rotate) with objects list, instruction
  templates ("pick up the {object} and place it on the {target}"), phases (reach, grasp, manipulate,
  release, retreat with contact expectation), repetitions, success criteria; session planning:
  plan_session(protocol, seed) -> ordered list of steps with rendered instructions (randomized
  object/order), operator script printing, per-episode (one task repetition = one session dir)
  recommendations. Loader validates schema.
- sync.py: estimate_offset(t_a, x_a, t_b, x_b, max_lag_s, hz) via cross-correlation of event
  envelopes (e.g. pressure tap transient vs IMU acc-norm spike vs camera mean-brightness flash);
  sync_event_envelope helpers; apply_offset to stream timestamps; document the "3-tap sync"
  procedure at session start/end (also measures drift → linear clock model fit_clock_drift).
- qc.py: session_qc(session_dir) -> report dict: per-stream rate mean/jitter, gaps > k/rate,
  dropped %, pressure saturation %, baseline drift between first/last no_contact segments, IMU quat
  norm, camera fps, hand_pose confidence coverage; pass/fail with thresholds; CLI
  `python -m robot_skin.acquisition.qc <session_dir>`.
- glove_logger.py / robot_logger.py: rewritten on top of Recorder + protocol: `--fake` (synthetic
  sources, runs end-to-end quickly, used by tests), `--dry-run` (plan only), `--protocol d1_motion|
  d2_task`, `--task`, `--subject`; real hardware sources remain NotImplemented until devices exist.
  Camera stream names `camera_<name>` (default ego, third). instruction rendering.
- docs/DATA_ACQUISITION.md: operator procedure for D1 and D2 (hardware checklist, calibration,
  sync taps, per-block scripts, naming, QC gates, how many sessions/subjects recommended,
  anonymization note for video).

[PRE] owns robot_skin/datasets/{build,splits,stats,motion}.py, robot_skin/datasets/README.md,
robot_skin/configs/stages/preprocess.yaml, docs/DATA_FORMAT.md, tests test_build.py,
test_splits_stats.py, test_motion_datasets.py. (episode.py is fixed contract — do not edit; report
requests.)
- build.py: preprocess_session(session_dir, out_root, cfg) -> Episode. Steps: load manifest+streams
  (npz formats above; pressure loader injectable for mk555 .bin via callable — never copy bin_merge);
  reorder channels to layout (layout.by_channel); align all streams to master clock 200 Hz
  (common.timeline; pressure/imu/joints/hand linear — quats: interpolate then renormalize + continuity
  fix; camera frame index via zoh on frame timestamps → cam_<name>_idx, -1 before first frame);
  optional pressure low-pass (scipy butter filtfilt if available, else moving average) ; baseline from
  first no_contact segment (fallback initial baseline_duration_s) with estimate_baseline →
  static baseline_raw; ΔS via relative_change; saturated via saturation_mask(raw, ΔS); IMU offsets
  from manifest.calibration if present; hand pose from hand_pose.npz (confidence gate → hand_pose_valid);
  q/qd: robot = joint_state; glove = finger_pose flattened (45) when hand pose present (else IMU-model
  derived later); qd = Savitzky–Golay derivative (scipy if available) else central diff + smoothing;
  taxel poses: glove → pose.mano.taxel_poses_from_hand, robot → pose.robot_fk.taxel_poses_from_joints
  (urdf path from cfg), bench → static layout; phases from events (+ manifest.segments) → phase_id +
  meta.phases; self_touch from pose.mano.self_touch_from_hand; contact_label: 0 in no_contact
  segments, 1 where self_touch, else -1 (D2: contact during grasp/manipulate phases stays -1 unless
  labelled); camera frames: copy or symlink the camera dir into the episode (cfg.copy_frames);
  meta.preprocessing records params + git-free version string. CLI `python -m robot_skin.datasets.build
  --raw <dir> --out <dir> [--config]` processing many sessions, skipping already-built unless --force.
- splits.py: make_splits(episode_dirs, by="subject"|"session"|"object"|"task", val_frac, test_frac,
  seed, holdout: dict|None) -> {"train": [...], "val": [...], "test": [...]} leakage-safe (group
  never crosses splits), save/load splits.json.
- stats.py: compute_stats(episodes, keys, method) -> dict[key, NormStats] over the train split with
  masks (exclude saturated for ΔS); save/load stats json.
- motion.py (D1 datasets): BaselineWindowDataset(episodes, window W (history frames), stride,
  only_labels=(0,) no-contact frames via contact_label/segments, exclude saturated) returning dict
  q_hist[W,D], qd_hist[W,D], pos[N,3], nrm[N,3], y[N] (ΔS at t), valid[N]; ContactWindowDataset
  (episodes with derived residual_z, labels from contact_label ≥ 0) returning z_hist[W,N], q/qd at
  t, label[N], label_mask[N]; ImuPoseWindowDataset (imu features window → finger pose aa at t where
  hand_pose_valid). All use joint/imu NormStats from stats.py passed in.
- docs/DATA_FORMAT.md: raw session format + processed Episode format (all keys, dtypes, units,
  conventions: quats wxyz, ΔS sign, frames).

[SYNTH] owns robot_skin/datasets/synthetic.py, tests test_synthetic.py.
- generate_session(out_dir, kind glove|robot, dataset motion|task, duration_s, seed, cameras
  ("ego",), image_hw (24,32), n_taxels from layout, ...) writes a complete RAW session in the exact
  formats above (session.json v2 with segments/task/calibration, pressure.npz, imu.npz (glove),
  joint_state.npz (robot; uses a small synthetic URDF string it also writes as robot.urdf and
  records in manifest.meta["urdf"]), hand_pose.npz (glove), camera dirs with tiny frames whose content
  correlates with hand state, events.jsonl with phases, object_pose.npz (task)). Physics-ish model:
  finger flexion drives a smooth no-contact ΔS artefact per taxel (gain varies per taxel, depends
  on joint angle AND velocity with a first-order lag → rewards temporal models), contact events
  (self-touch in motion sessions per protocol blocks; object contact in task grasp/manipulate phases)
  add negative ΔS press (raw drop) with occasional saturation; noise; mild baseline drift; IMU quats
  consistent with pose.mano.ManoSkeleton segment orientations (+ offsets recorded in calibration).
  Returns manifest. generate_dataset(root, n_motion, n_task, subjects, seed) convenience.
  Used by integration smoke tests — keep default sizes tiny (e.g. 4–6 s, 200 Hz).

[STAGE1] owns robot_skin/baseline/{temporal,calibrate}.py (+ may extend baseline/__init__.py,
README), robot_skin/contact/{calibration,detector,hysteresis,pseudo_label}.py (+ contact/__init__.py
exports, README), robot_skin/stages/{__init__,imu_pose,baseline,contact}.py,
robot_skin/configs/stages/{imu_pose,baseline,contact}.yaml, tests test_baseline_temporal.py,
test_contact_learned.py, test_stage1.py.
- baseline/temporal.py: TemporalBaselinePredictor(n_taxels, joint_dim, window, hidden, taxel_emb_dim,
  kernel) – causal temporal conv (or GRU) over [q_hist, qd_hist] → joint context; per-taxel head on
  [context, pos, nrm, taxel_emb] → (mean, logvar) [B,N]; mean head zero-init; logvar init log(σ0²);
  gaussian_nll(mean, logvar, y, valid) (Kendall & Gal 2017) with logvar clamp; predict_episode(model,
  episode, joint_stats, window, batch) -> (mean[T,N], logvar[T,N]) causal (pads beginning).
  Keeps v1 BaselinePredictor untouched. Design principle from sats/bending: never feed observed ΔS as
  input (would learn to erase contact).
- contact/calibration.py: ResidualCalibrator.fit(residual[T,N], valid mask no-contact, logvar
  optional) → per-taxel robust σ (MAD·1.4826) (+ uses predicted σ if given: z = press_intensity(r)/
  sqrt(σ_taxel² + exp(logvar))), thresholds weak/strong in z units with % floor; transform(residual,
  logvar) -> z (press-positive); levels(z, saturated) -> ContactLevel via OrdinalQuantizer semantics;
  to_dict/from_dict.
- contact/detector.py: ContactDetector(n_taxels?, window, hidden) per-taxel temporal classifier on
  [z_hist, |q̇| summary, taxel emb optional] → logit; focal/BCE with pos_weight; predict_episode.
- contact/hysteresis.py: HysteresisFilter(on_thr, off_thr, min_on, min_off) streaming + batch.
- contact/pseudo_label.py: pseudo_label_episode(episode, prob, …) fusing detector prob, hysteresis,
  saturation, phase expectation (reach/retreat/no_contact phases → 0; grasp/manipulate → allow 1),
  optional vision hand–object proximity (object_pos vs fingertip positions) → contact_label for D2.
- stages/imu_pose.py run(cfg) trains pose.imu_model.ImuHandPoseNet on D1 (+D2) episodes with
  hand_pose_valid via train.Trainer, saves ckpt + stats, writes derived hand_finger_pose_imu for all
  episodes; stages/baseline.py run(cfg) trains TemporalBaselinePredictor on no-contact frames (D1),
  evaluates (val NLL, residual MAE on no-contact, motion_contact_separability before/after), writes
  derived baseline_pred/logvar/residual for all episodes; stages/contact.py run(cfg): fit
  ResidualCalibrator on D1 val no-contact, write residual_z/contact_level; train ContactDetector on
  D1 labels (self_touch=1, no_contact=0); write contact_prob; pseudo-label D2; report metrics
  (hallucination_rate on no-contact, AUROC/F1 on self-touch). Each run(cfg) returns a metrics dict and
  writes out_dir/metrics.json. cfg keys: data.processed_root, data.splits, train (TrainConfig dict),
  model{...}, hw profile name optional.

[REPR] owns robot_skin/representation/{encoder,pretrain}.py (replace stub), representation/__init__.py,
README, robot_skin/stages/pretrain.py, robot_skin/configs/stages/pretrain.yaml, tests
test_representation.py (update), test_pretrain.py.
- encoder.py: TaxelEncoder(value_dim, d_model, depth, heads, n_fourier, fourier_scale, temporal k
  frames stacked into value_dim) = TaxelTokenizer + nn.TransformerEncoder (pre-LN, batch_first) →
  tokens [B,N,D]; tactile_value_features(residual_z[...,N], level[...,N], saturated) -> [...,N,F]
  (z clipped/scaled, level one-hot 4, sat flag) — the SAME feature function must be used by VTLA and
  control (single source of truth; export it).
- pretrain.py: MaskedTaxelPretrainer(encoder, value_dim) – masks taxels (random_taxel_mask; option
  mask whole groups from layout.groups), reconstructs masked residual_z (Huber) + level (CE);
  loss(batch) dict; stage runner stages/pretrain.py trains on all D1+D2 episodes with derived
  residual_z/contact_level (Trainer), saves encoder weights. Grounding: MAE (He et al. 2022).

[VTLA] owns robot_skin/vtla/{model,heads,losses,dataset,dpo}.py (+ vtla/__init__.py exports, keep
adapter.py API; may add), robot_skin/vtla/README.md, robot_skin/stages/vtla.py,
robot_skin/configs/stages/vtla.yaml, tests test_vtla.py (keep existing tests passing),
test_vtla_model.py, test_vtla_dataset.py.
- dataset.py: VTLADataset(episodes (D2), cameras, policy_hz (default 20), horizon H (default 16),
  obs_history (default 1), action_spec (hand_mano via action.space), tactile feature fn from
  representation.encoder.tactile_value_features, image transform, use_cached_vision key|None,
  stats: dict from datasets.stats / ActionNormalizer) → sample dict: images {cam: float[3,H,W]} or
  vision_feats {cam: [P,D]}, tactile_values [N,F], taxel_pos [N,3], taxel_nrm [N,3], contact [N] bool
  (level ≥ WEAK), proprio [P] (current hand action / robot q, normalized), instruction str,
  actions [H,A] normalized, action_valid [H] bool, meta ids. collate_vtla(batch) stacks tensors and
  keeps instruction list. Sampling only at policy_hz ticks inside phases != none.
- model.py: VTLAConfig dataclass; VTLAPolicy(nn.Module): vision encoder (vision.build_vision_encoder;
  per-camera learned camera embedding; or cached features path), text encoder (language.
  build_text_encoder), tactile: TaxelEncoder (optionally initialized from stage-2 pretrain ckpt,
  optionally frozen) → TactileTokenAdapter (ContactGate) → K tactile tokens, proprio MLP → 1 token,
  modality type embeddings, fusion TransformerEncoder (pre-LN) over [lang, vision, tactile, proprio,
  readout], action head; modality dropout (p_drop_tactile, p_drop_vision) at train; obs_mode
  ablation (full/ordinal/binary/none controls tactile feature subset or disables tactile tokens);
  forward(batch) -> dict; predict(batch, n_steps) -> actions [B,H,A] (normalized); aux contact head
  (predict per-taxel contact from tactile tokens) optional.
- heads.py: ChunkRegressionHead (ACT-style: H learned queries cross-attend fused tokens → [B,H,A],
  L1 loss masked) and FlowMatchingHead (π0 / rectified flow: small transformer/MLP denoiser v_θ(x_τ,
  τ, cond) with sinusoidal τ embedding; train: x_τ = (1−τ)·ε + τ·a (or per π0 convention, state your
  convention), target velocity; inference: Euler K steps from noise; masked loss).
- losses.py: action losses (masked L1 / flow MSE), aux contact BCE, total weighting.
- dpo.py: documented hook for VTLA-paper-style preference optimization (DPO) on action chunks —
  implement dpo_loss(policy_logp_chosen, policy_logp_rejected, ref_*, beta) generic + NotImplemented
  pipeline stub explaining where preference pairs come from (success/failure rollouts).
- stages/vtla.py: run(cfg) → splits, stats (ActionNormalizer, proprio stats), datasets, model,
  Trainer, eval (masked L1 per horizon step, by task), save ckpt + normalizers + model config
  (everything control needs to reload: save a single `policy_bundle.pt` with config, weights,
  normalizers, tactile calibrator path refs).

[CTRL] owns robot_skin/control/{__init__,interfaces,online,runner,safety,bundle,latency}.py,
robot_skin/control/README.md, robot_skin/transfer/* (replace stubs), robot_skin/stages/deploy.py,
robot_skin/configs/stages/deploy.yaml, docs/DEPLOYMENT.md, tests test_control.py, test_transfer.py,
test_config_and_stubs.py (update transfer stub asserts).
- interfaces.py: RobotHandInterface Protocol (joint_names, read_state() -> (t, q, qd), read_pressure()
  -> (t, raw[C]), send_joint_targets(q_target), limits), CameraInterface Protocol (read() -> (t,
  uint8[H,W,3])), FakeRobotHand (first-order joint tracking sim with a synthetic tactile response:
  motion artefact + contact when a joint exceeds a "virtual object" angle), FakeCamera.
- online.py: OnlineTactileProcessor(layout, baseline_model|None, joint_stats, calibrator, fsm cfg,
  window) streaming: startup baseline estimation (N no-contact samples) → ΔS → baseline prediction with
  rolling q history → residual → SaturationFSM → z & levels → tactile_value_features. MUST match
  offline stages output on the same data (test: run offline stage functions and online processor on
  a synthetic episode → allclose).
- bundle.py: load_policy_bundle(path) → policy + normalizers + configs; PolicyBundle.
- runner.py: PolicyRunner(robot, cameras, bundle, processor, retargeter|None, safety, control_hz,
  policy_hz, instruction, ensembler) → step() does sensing, obs building (same as VTLADataset
  sample, single-sample), inference every policy tick (chunk), TemporalEnsembler, action →
  (hand_mano → retarget via action.retarget; robot_joint → direct), safety filter, send; logs every
  tick to a deployment session (reusable as data); run(duration) loop with rate keeping.
- safety.py: SafetyFilter(lower, upper, max_vel, max_acc?, tactile_stop: level STRONG/SATURATED
  sustained > k ticks on any taxel → freeze closing motion, estop callback), watchdog on sensor
  staleness.
- latency.py: benchmark_policy(bundle/policy, batch, device, n) → p50/p95 ms; optional TorchScript
  export (guarded).
- transfer/: project_to_skeleton(taxel_pos[N,3], skeleton capsules) → (segment idx, t along bone,
  offset) ; align_layouts(src Layout+poses, dst Layout+poses) → mapping dst→src (by finger group +
  nearest) ; map_taxel_values(values_src, mapping); robot_to_mano_retarget stub-free version via
  retarget for the reverse direction optional. Replace NotImplementedError stubs.
- stages/deploy.py run(cfg): builds FakeRobotHand when cfg.robot == "fake" (tests) else raises with
  instructions; runs N seconds; returns metrics (latency, safety events).
- docs/DEPLOYMENT.md: control loop diagram, rates, safety, bringing up a real hand (implement
  RobotHandInterface), latency budget on RTX 5090 vs CPU.

### WAVE 3
[INTEG] owns robot_skin/__main__.py (CLI: record, preprocess, qc, train <stage>, deploy, env,
sweep), robot_skin/configs/default.yaml (extend: paths, stages list, policy defaults), robot_skin/
README.md, robot_skin/tests/test_e2e_pipeline.py (synthetic D1+D2 → preprocess → imu_pose → baseline
→ contact → pretrain → vtla → deploy fake, tiny sizes, < 60 s), and MAY make minimal cross-module
fixes anywhere in robot_skin to make the pipeline run (list every such edit).
[DOCS] owns docs/ARCHITECTURE.md, docs/TRAINING.md (stages, configs, hardware profiles, RTX 5090
notes, multi-GPU & multi-machine via Tailscale), docs/VTLA.md, root README.md, module READMEs sync.

## References (verified; cite only these, REFS agent may add verified ones)
- VTLA: Vision-Tactile-Language-Action Model with Preference Learning for Insertion Manipulation,
  arXiv:2505.09577 (2025).
- 3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing, arXiv:2410.24091 (CoRL 2024).
- ActionSense (DelPreto et al.), NeurIPS 2022 Datasets & Benchmarks.
- OSMO: Open-Source Tactile Glove for Human-to-Robot Skill Transfer, arXiv:2512.08920.
- DexUMI, arXiv:2505.21864.
- ACT: Zhao et al., Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, arXiv:2304.13705.
- Diffusion Policy: Chi et al., arXiv:2303.04137.
- π0: Black et al., A Vision-Language-Action Flow Model for General Robot Control, arXiv:2410.24164.
- Zhou et al., On the Continuity of Rotation Representations in Neural Networks, CVPR 2019, arXiv:1812.07035.
- Kendall & Gal, What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?,
  NeurIPS 2017, arXiv:1703.04977.
- MANO: Romero et al., Embodied Hands, SIGGRAPH Asia 2017.
- MAE: He et al., Masked Autoencoders Are Scalable Vision Learners, CVPR 2022, arXiv:2111.06377.
- HaMeR: Pavlakos et al., Reconstructing Hands in 3D with Transformers, CVPR 2024, arXiv:2312.05251.
- AnyTeleop: Qin et al., arXiv:2307.04577.  DexPilot: Handa et al., ICRA 2020, arXiv:1910.03135.
- DexCap: Wang et al., arXiv:2403.07788. OpenVLA arXiv:2406.09246. Octo arXiv:2405.12213.
- Rectified flow / flow matching: Lipman et al., Flow Matching for Generative Modeling, arXiv:2210.02747.
(REFS agent verifies these; if one is wrong it corrects docs/REFERENCES.md and reports.)
- Internal: deformable_sats/sats/bending/baseline_restorer.py (deg→offset restorer; lesson: never
  feed observed ΔS into the baseline model), sats/inference/run_dashboard.py (quarantine/sign gate).

## Final output of every implementation agent (structured)
files_written, public_api (signatures summary), tests (command + result counts), deviations from
this spec (and why), interface_requests (changes needed in files you don't own), open_todos.
