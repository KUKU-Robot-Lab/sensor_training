# Wave 1 AS-BUILT API (authoritative over SPEC.md where they differ)

Read the actual source + package README for exact signatures; this digest summarizes implementer reports and reviewer notes.


## POSE
### public_api
All of the following are re-exported from robot_skin.pose (__all__ has 45 names).

mano.py
- Constants: MANO_JOINTS (16, MANO order: wrist, index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3); MANO_PARENTS (-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14); FINGERS=("thumb","index","middle","ring","pinky") (anatomical order, used for tips and per-finger arrays); FINGER_JOINTS {finger: (j1,j2,j3)}; TIP_NAMES; SEGMENT_TO_JOINT (covers every common.layouts.MANO_SEGMENTS name; wrist and palm map to 0); PALM_CAPSULES; CAPSULE_NAMES (15 phalanges + palm_index/middle/ring/pinky); DEFAULT_REST_JOINTS [16,3] (approximate right hand in MANO canonical axes: fingers -x, radial +z, palm faces -y, wrist at origin); DEFAULT_TIP_LENGTHS; DEFAULT_SEG_RADIUS [15]; DEFAULT_PALM_RADIUS; DEFAULT_SELF_TOUCH_EXCLUDE {"palm": ("thumb1",)}; bone_frame(direction, dorsal_hint).
- class ManoSkeleton(rest_joints=None, tip_offsets=None, *, seg_radius=None, palm_radius=0.012, palm_offset=(0,0,0), dorsal=(0,1,0), thumb_dorsal=(0,.5,1), bone_aligned=True)
  - forward(global_orient[...,3]|None, finger_pose[...,15,3]|None, wrist_pos[...,3]|None) -> {joint_T[...,16,4,4], joint_rot[...,16,3,3], joint_pos[...,16,3], tip_pos[...,5,3] (FINGERS order)}. Differentiable, accepts numpy, keeps the dtype of the input tensors. __call__ is an alias.
  - forward_matrices(global_rot[...,3,3], finger_rot[...,15,3,3], wrist_pos): the same FK but from rotation matrices.
  - rest_fk().
  - segment_transforms(fk) -> {seg: [...,4,4]} for all 17 MANO_SEGMENTS.
  - segment_transform_tensor(fk) -> [...,17,4,4].
  - capsules(fk) -> (p0[...,19,3], p1[...,19,3], radius[19] np, names list).
  - flexion_pose(flex[...,15] or [...,5], abduction[...,5]=None) -> finger_pose[...,15,3].
  - segment_local_rotation(seg).
  - Attributes: rest_joints, tip_offsets, bone_frames[17,3,3], segment_local[17,4,4].
  - classmethod from_mano_pkl(path .pkl|.npz, betas=None, *, tip_vertex_ids=None, **kw). Needs chumpy for the original pkl; otherwise raises ImportError with instructions.
- taxel_poses_from_hand(layout, skeleton, global_orient[T,3], finger_pose[T,15,3], wrist_pos[T,3]|None, *, chunk=8192) -> (pos[T,N,3], nrm[T,N,3]) float64. Single-frame input returns [N,3].
- self_touch_from_hand(layout, skeleton, go, fp, wp=None, *, margin=0.004, exclude=None, chunk=4096) -> bool[T,N]. Wraps contact.self_touch_labels and imports it lazily.
- class ManoPoseProvider(layout, skeleton|None, pose_at_fn): pose_at_fn(t) returns a (go, fp[, wp]) tuple or a dict. Provides n_taxels, pose_at(t) and hand_pose_at(t).

urdf.py
- @dataclass URDFJoint(name, type, parent, child, origin_xyz, origin_rpy, axis (unit), lower, upper, effort, velocity, mimic=(src, mult, off)|None), with .movable and .origin_matrix().
- class URDFModel(name, links, joints, *, mimic="follow"|"ignore")
  - from_string(xml, *, mimic) / from_file(path, *, mimic).
  - Attributes: name, link_names, joints, joint_names (actuated = movable and not mimic, file order), root_link, n_dof, lower/upper/velocity_limits [D] np (continuous → ±inf).
  - joint(name), parent_joint(link), chain(link), clamp(q).
  - reorder_q(q, names, *, fill=0.0): maps driver order to model order.
  - fk(q[...,D] torch|np, *, base_T=None, links=None) -> {link: [...,4,4]}: differentiable, closed-form Rodrigues.
  - fk_numpy(q, *, base_T, links).
  - Joint types: revolute, continuous, prismatic, fixed. floating/planar are treated as fixed with a warning. Mimic joints follow their source by default.

robot_fk.py
- class RobotFKPoseProvider(layout, urdf: URDFModel|str|Path, joint_state_at, *, base_T=None). Raises ValueError unless parent_frame is urdf; checks that the layout parents are URDF links. Provides n_taxels and pose_at(t).
- taxel_poses_from_joints(layout, model|path, q[T,D], *, base_T=None, chunk=8192) -> (pos[T,N,3], nrm[T,N,3]).

imu_model.py
- imu_feature_dim(n_sites, *, gyro=True, acc=True): 12 per site by default.
- imu_features(quat[...,W,S,4], gyro=None, acc=None, wrist_index=0, *, vec_frame="sensor"|"world") -> [...,W,S*F]. Per site: [6D(q_wrist^-1 q_site) | gyro in wrist frame | acc in wrist frame]. Numpy input returns float32.
- imu_windows(feat[T,F], window) -> [T,W,F]: causal, edge-padded view.
- mean_quaternion(q, axis=0).
- calibrate_imu_offsets(quat_calib[T,S,4], ref_rot[S,3,3]|None, *, world=None) -> offsets[S,4] = mean^-1 ⊗ G ⊗ q_ref.
- estimate_world_alignment(quat_calib, ref_rot=None, *, index=0) -> G[4].
- apply_imu_offsets(quat[...,S,4], offsets, *, world=None) = G^-1 ⊗ q ⊗ q_off.
- apply_imu_offsets_to_vectors(vec[...,S,3], offsets) = R_off^T v.
- Manifest calibration helpers: CALIB_OFFSETS_KEY="imu_offsets", CALIB_WORLD_KEY="imu_world", CALIB_SITES_KEY="imu_sites"; imu_calibration_to_dict(offsets, world=None, sites=None); imu_calibration_from_dict(calib, sites=None) -> (offsets|None, world|None).
- imu_site_segments(layout); imu_site_quats(layout, skeleton, go, fp) -> [T,S,4].
- synthesize_imu(layout, skeleton, t, go, fp, wp=None, *, gravity=(0,0,-9.81)) -> {t, quat, gyro, acc, sites}. Ideal IMU streams in the sensor (= segment) frame.
- class ImuHandPoseNet(in_dim, *, hidden=256, n_layers=2, arch="gru"|"tcn", dropout=0.1, kernel=3, predict_global=False)
  - forward(feat[B,W,F]) -> finger_rot6d[B,15,6].
  - forward_all(feat, *, all_steps=False) -> {finger_rot6d, global_rot6d?}.
  - predict(feat) -> adds finger_pose (axis-angle) and global_orient.
  - set_feature_stats(mean, std): stored as buffers feat_mean/feat_std.
  - .config / from_config(cfg).
  - The output head is zero-initialised around the identity rotation, so an untrained net predicts the flat hand.
- to_axis_angle(rot6d).
- rotation_geodesic(R1, R2): atan2-based, finite gradients.
- hand_pose_loss(pred6d[B,15,6], gt_aa[B,15,3], skeleton=None, tip_weight=10.0, *, valid=None, pred_global6d=None, gt_global=None, global_weight=1.0) -> dict {loss, rot, tip?, global?}.
- predict_finger_pose_sequence(model, quat[T,S,4], gyro=None, acc=None, *, window, wrist_index=0, vec_frame="sensor", batch_size=1024, device=None) -> {finger_pose[T,15,3], global_orient?}.

glove_imu2mano.py
- class GloveImu2ManoPoseProvider(layout, model, skeleton|None, t[T], quat[T,S,4], gyro=None, acc=None, *, window=32, wrist_index=None (defaults to the site whose parent is wrist), offsets=None, world=None, global_orient=None, wrist_pos=None, global_from_imu=True, vec_frame="sensor", batch_size=1024, device=None)
  - Runs offline and precomputes everything.
  - Attributes: finger_pose, global_orient, wrist_pos, t, poses.
  - pose_at(t) interpolates linearly between frames and renormalises normals.
- load_vifnet_s(checkpoint) and finetune_vifnet_s(train_sessions, out_dir): documented NotImplementedError stubs (external weights).

vision_hand.py
- VisionHandEstimator Protocol: estimate(frames uint8[B,H,W,3]) -> dict.
- HaMeREstimator: stub; the constructor raises NotImplementedError and the docstring gives the offline HaMeR/WiLoR procedure.
- save_hand_labels(path|dir, t, go, fp, wp, confidence=None) -> Path.
- load_hand_labels(path|session_dir) -> dict.
- estimate_sequence(estimator, frames, t, *, batch_size=16).
- smooth_hand_labels(t, go, fp, wp, confidence=None, *, min_conf=0.5, max_gap_s=0.25, cutoff_hz=6.0, order=2) -> dict with the same keys plus valid[T]. It gates by confidence, SLERP-fills short gaps and low-passes each valid run (scipy Butterworth filtfilt, falling back to a centred moving average).
### deviations from spec
1. Palm segment origin. SEGMENT_TO_JOINT["palm"] is 0, as specified, but the palm frame's default origin is the wrist joint, not a palm-centre offset.
   - Reason: glove_template palm taxels sit at z = 30–55 mm along the palm. Measured from a palm centre, they would land beyond the knuckles (MCPs).
   - The offset is configurable (ManoSkeleton(palm_offset=...), expressed in the palm frame). The palm frame's rotation is aligned with the palm (z points from the wrist toward middle1, y is dorsal).

2. Bone-aligned segment frames. Finger segment frames are the MANO joint frame times a fixed bone-aligned rotation: local +z along the bone, -y toward the palm pad.
   - This is what makes glove_template coordinates such as [0, -6, 12] mm fingertip pads meaningful.
   - bone_aligned=False gives the raw MANO joint frames. "wrist" is exactly joint 0.

3. Default rest skeleton. It is an approximate hand-made right hand in what I believe are MANO's canonical axes (fingers -x, radial +z, palm faces -y). It is not MANO's template and I could not verify the axes against the MANO files. from_mano_pkl loads real rest joints.

4. Capsules. There are 15 phalanx capsules plus 4 palm capsules (wrist to each MCP) instead of a single palm capsule, because one capsule cannot cover a roughly 65 mm wide palm.
   - By default, self_touch_from_hand makes palm taxels ignore the thumb1 capsule. Without this, the thenar region falsely flags the palm taxels in an open hand.
   - This is done by calling contact.self_touch_labels per finger group; contact/self_touch.py is not modified.

5. ImuHandPoseNet API.
   - forward() returns finger_rot6d only, as specified.
   - The optional global 6D output comes from forward_all() and predict().
   - hand_pose_loss returns a dict {loss, rot, tip?, global?} rather than a scalar, so it plugs straight into the Trainer's loss_fn contract.

6. imu_features. When gyro or acc is None, that stream is omitted (F = 6/9/12 per site); use imu_feature_dim() to get the width. There is an extra vec_frame="world" option for devices that output world-frame vectors.

7. Calibration. The spec's formula (q_off = mean^-1 ⊗ q_ref) is the default. I added an optional world alignment G (estimate_world_alignment, then the world= argument) and apply_imu_offsets_to_vectors for gyro and acc.

8. URDF mimic joints follow their source joint by default (q = multiplier·q_src + offset) instead of being ignored. mimic="ignore" keeps the spec's behaviour: held at 0, with a warning. Mimic joints are never in joint_names.

9. GloveImu2ManoPoseProvider's constructor takes t, quat, gyro, acc, offsets, global_orient and wrist_pos explicitly. Global orientation comes from the calibrated wrist IMU by default.

10. Additions not in the spec (additive only):
    - ManoSkeleton: forward_matrices, flexion_pose
    - imu_model: synthesize_imu, imu_site_quats, predict_finger_pose_sequence, the manifest calibration dict helpers
    - vision_hand: estimate_sequence
    - URDFModel: reorder_q, clamp, chain
### notes for consumers / interface_requests
1. robot_skin/pose/provider.py (not in my ownership list): the module docstring still describes robot_fk.RobotFKPoseProvider and glove_imu2mano.GloveImu2ManoPoseProvider as "Stub". Please change these to "Implemented" (orchestrator or DOCS).

2. robot_skin/README.md (INTEG/DOCS): the pose row lists robot_fk and glove_imu2mano as stubs. It should instead list mano, urdf, robot_fk, imu_model, glove_imu2mano and vision_hand as implemented, with the VIFNet-S loader and HaMeR as stubs.

3. PRE (docs/DATA_FORMAT.md, datasets/build.py), please state and use:
   - imu.npz gyro/acc are in each IMU's sensor frame. acc is specific force including gravity.
   - manifest.calibration IMU keys are imu_offsets / imu_world / imu_sites. Write them with pose.imu_model.imu_calibration_to_dict; read them with imu_calibration_from_dict(calib, sites) and apply with apply_imu_offsets and apply_imu_offsets_to_vectors.
   - hand_pose.npz finger_pose is relative to the flat MANO template (flat_hand_mean=True). wrist_pos is the world position of joint 0 (from MANO: transl + J_0(β)).
   - Use pose.vision_hand.smooth_hand_labels for the confidence gate; its valid output gives hand_pose_valid.
   - Glove taxel poses come from pose.mano.taxel_poses_from_hand, and self_touch from pose.mano.self_touch_from_hand.
   - Robot taxel poses come from pose.robot_fk.taxel_poses_from_joints(layout, URDFModel.from_file(urdf), model.reorder_q(q, names)).
   - The ImuPoseWindowDataset should build features with pose.imu_model.imu_features on the K_IMU_QUAT/GYRO/ACC arrays and window them with imu_windows(feat, W). The target is finger pose axis-angle at t.

4. SYNTH, to stay consistent with the pose model:
   - Generate IMU streams with pose.imu_model.synthesize_imu(layout, ManoSkeleton(), t, go, fp, wp). This gives ideal readings with sensor frame = segment frame.
   - Apply a mounting M as q_raw = q ⊗ M and v_raw = R_M^T v. The recorded offsets are then conj(M); store them via imu_calibration_to_dict.
   - ManoSkeleton.flexion_pose(flex[...,5|15], abduction) produces anatomical poses; positive flex curls toward the palm.

5. STAGE1 imu_pose stage:
   - Build the model as ImuHandPoseNet(imu_feature_dim(S), ...) and call model.set_feature_stats(NormStats.offset, NormStats.scale); otherwise the dataset must normalise the features.
   - Use hand_pose_loss(pred6d, gt_aa, skeleton, valid=...)["loss"] as the loss.
   - Save model.config with the checkpoint.
   - Write derived D_HAND_POSE_IMU with predict_finger_pose_sequence(model, quat, gyro, acc, window=W)["finger_pose"].

6. ACTION retarget:
   - Human fingertips are ManoSkeleton.forward(...)["tip_pos"] in FINGERS order (thumb, index, middle, ring, pinky), matching TIP_NAMES.
   - URDFModel.fk(q, links=[...]) is differentiable and can serve as the fk_fn.

7. CTRL transfer: skeleton capsules are available from ManoSkeleton.capsules(fk) (names in CAPSULE_NAMES), for project_to_skeleton.

8. REFS: I cite only MANO, Zhou et al. 2019 (6D) and HaMeR. VIHand / VIFNet-S is mentioned only as the user-specified backbone, with no bibliographic details. Please confirm or add the citation in docs/REFERENCES.md.

### reviewer: fixes applied
- mano.py: DEFAULT_SELF_TOUCH_EXCLUDE now also maps index1/middle1/ring1/pinky1 to their own palm_<finger> capsule. self_touch_from_hand accepts exclude keys that are finger groups or MANO segment names (both apply), validates the keys, and groups the self_touch_labels calls per parent segment. The public signature is unchanged. Docstrings updated, plus an immutability note on ManoSkeleton.
- robot_fk.py: chunk is normalised once (step = max(1, chunk)), so chunk<=0 gives correct output. base_T must be 4x4 in both RobotFKPoseProvider and taxel_poses_from_joints (clear ValueError). The docstring states the q column order and points to reorder_q.
- urdf.py: fk broadcasts base_T[...,4,4] against q's leading dims and validates its shape.
- imu_model.py:
- estimate_world_alignment normalises negative indices and validates shapes; it also documents that one static pose cannot separate G from the wrist mounting.
- _masked_mean broadcasts valid from the left, checks its shape, and uses an eps denominator.
- hand_pose_loss checks the gt_aa shape.
- ImuHandPoseNet.encode casts features to the buffer dtype.
- The _CausalTCN docstring gives the receptive-field formula.
- New exported helper imu_reference_rotations(layout, skeleton=None, global_orient=None, finger_pose=None) -> [S,3,3], the flat-hand ref_rot for calibration.
- glove_imu2mano.py: world= without offsets now applies world alignment with identity offsets. t must be 1-D, with an error message naming the argument order.
- vision_hand.py: save_hand_labels treats any non-.npz path as a session directory (<dir>/hand_pose.npz, created if needed). Corrected the hold-behaviour docstring of smooth_hand_labels.
- __init__.py exports imu_reference_rotations. README.md documents the new self-touch exclusion rule and the flat-hand calibration recipe.
- New tests:
- test_mano: test_fk_matches_hand_derived_mcp_rotation; test_self_touch_no_false_positive_on_proximal_phalanges, which also covers the old-rule repro and segment-key and bad-key handling.
- test_urdf: test_planar_two_link_closed_form; test_base_T_broadcasts_and_chunk_edge_cases.
- test_imu_model: test_flat_hand_calibration_pipeline_with_reference_rotations; test_hand_pose_loss_broadcasts_valid_over_steps_and_checks_shapes; test_net_accepts_float64_features; test_glove_provider_world_alignment_without_offsets.
- test_pose: test_save_hand_labels_directory_paths.
- The calibration pipeline test uses an unknown IMU world, per-site mounting and an unknown hand heading. It shows calibrated features equal the ideal ones and orientations are re-anchored at the calibration heading.
### reviewer: remaining concerns
1. Spec deviations kept as documented by the implementer; all are reasonable, but wave-2 agents must follow the code, not the spec wording:
   - hand_pose_loss returns a dict {loss, rot, tip?, global?}, not a scalar. STAGE1 must use ["loss"].
   - GloveImu2ManoPoseProvider takes (layout, model, skeleton, t, quat, gyro, acc, ...), with t before the IMU arrays. A swapped call now fails with a clear message.
   - URDF mimic joints follow their source by default.
   - The palm frame origin is the wrist; palm_offset is configurable.
   - There are 19 capsules (4 palm), not 16.

2. The default rest skeleton / MANO canonical axes are not verified against the real MANO_RIGHT model, which is licence-restricted. Index1 relative to the wrist does match remembered MANO template values closely. Check once with ManoSkeleton.from_mano_pkl.

3. A single static calibration pose cannot separate the IMU world alignment from the wrist IMU's mounting. If the wrist strap is not aligned with the wrist segment, wrist-relative features become a fixed per-session change of basis, and the IMU global orientation is off by that mounting. Fixing this needs a second pose or a vision global_orient. This is now documented in estimate_world_alignment.

4. The capsule self-touch model is crude:
   - An equal-angle fist flags only thumb and index tips, not middle/ring/pinky against the palm. These are false negatives, which are harmless because PRE leaves them at -1, but self-touch recall on fist blocks will be low.
   - Radii and thumb_dorsal need tuning on real glove geometry.

5. DDP caveat: with predict_global=True, training through forward() leaves global_head without gradients. DDP then requires find_unused_parameters; use forward_all.

6. Interface requests from the implementer still stand (files not owned by POSE):
   - provider.py docstring still says robot_fk and glove_imu2mano are "Stub";
   - the robot_skin/README.md pose row;
   - PRE/SYNTH/STAGE1 conventions.

   PRE, SYNTH and ACQ should now use pose.imu_model.imu_reference_rotations(layout, skeleton) as ref_rot for flat-hand calibration:

   ```
   R = imu_reference_rotations(L, sk)
   G = estimate_world_alignment(q_cal, R, index=wrist)
   q_off = calibrate_imu_offsets(q_cal, R, world=G)
   ```

   Store the result with imu_calibration_to_dict(q_off, G, sites).

Files touched (all POSE-owned):
- robot_skin/pose/{mano,urdf,robot_fk,imu_model,glove_imu2mano,vision_hand,__init__}.py
- robot_skin/pose/README.md
- robot_skin/tests/{test_pose,test_mano,test_urdf,test_imu_model}.py


## TRAIN
### public_api
robot_skin.train (lazy PEP 562 re-exports of everything below)

engine.py
- @dataclass TrainConfig: all spec fields with spec defaults (max_epochs=10, max_steps=None, batch_size=64, lr=3e-4, weight_decay=0.05, warmup_steps=100, schedule="cosine", min_lr_ratio=0.1, grad_clip=1.0, grad_accum=1, precision="auto", compile=False, ema_decay=None, num_workers=0, pin_memory=True, log_every=50, eval_every_epochs=1, ckpt_every_epochs=1, out_dir="robot_skin/runs/default", seed=0, device="auto", deterministic=False, early_stop_patience=None, monitor="val/loss"). Extra fields: monitor_mode="min", early_stop_min_delta=0.0, optimizer="adamw", betas=(0.9,0.999), eps=1e-8, lr_mult=None, drop_last=False, eval_batch_size=None, compile_mode=None, tf32=True, ema_warmup=True, ckpt_every_steps=None, resume=None ("auto"|path), find_unused_parameters=False, tensorboard=False, wandb_project=None.
  - TrainConfig.from_dict(d, *, strict=False): warns on unknown keys and ignores them. Coerces types, so YAML `lr: 3e-4` (read as a string), lists for betas, "none", and `resume: true` → "auto" all work. Also .to_dict() and .validate().
- Trainer(model, loss_fn, cfg, train_data, val_data=None, collate_fn=None, extra_state=None, *, optimizer=None|Optimizer|callable(model), callbacks=(), dist_info=None)
  - loss_fn(model, batch) -> dict with "loss" (a bare Tensor also works).
  - Attributes: model (bare), train_model (DDP/compiled), optimizer, scheduler, scaler, ema, step, epoch, batch_in_epoch, history, best_value/best_epoch/best_step, should_stop, stopped_early, device, precision, dist, resumed_from.
  - Methods: fit() -> list[dict]; evaluate(data=None, *, use_ema=True, prefix="val/") -> {"val/loss", ...}; eval_weights(use_ema) context manager; state_dict()/load_state_dict(); save_checkpoint(path=None); resume(path="auto") -> Path|None; current_lr(); @staticmethod load_model_weights(model, path, use_ema=True, strict=True, map_location="cpu").
  - Checkpoint keys: model, optimizer, scheduler, scaler, ema, step, epoch, config, extra, plus best, bad_epochs, history, rng, batch_in_epoch.
  - Run dir contents: config.json, env.json, metrics.jsonl, history.json, ckpt_last.pt, ckpt_best.pt, summary.json. summary.json exists only after fit() finishes and is deleted when a run restarts.
- Helpers: seed_everything(seed, deterministic=False), seed_worker(worker_id), move_to_device(batch, device, non_blocking=False) for dict/list/tuple/namedtuple/dataclass, maybe_compile(module, device, mode) -> (module, compiled).

hardware.py
- resolve_device(pref="auto"|"cpu"|"cuda"|"cuda:N"|"mps", local_rank=None) -> torch.device
- PrecisionPlan(name, autocast_dtype, use_grad_scaler, device_type), with .enabled, .autocast(), .make_scaler()
- resolve_precision(pref, device, *, capability=None) -> PrecisionPlan; cuda_capability(device)
- enable_tf32(enabled=True) -> previous precision string
- Profiles: PROFILES_DIR, list_hw_profiles(), load_hw_profile(name|path|"auto"), apply_hw_profile(stage_cfg, profile, stage=None), maybe_apply_hw_profile(cfg, stage=None) (reads cfg["hardware"]), apply_profile_env(profile, override=False), detect_hw_profile(gpu_names=None)
- describe_environment() -> dict; check_arch_support(capabilities=None, arch_list=None, torch_cuda=None, names=None) -> list[str]; main() CLI (`python -m robot_skin.train.hardware [--json]`)

distributed.py
- DistInfo(rank=0, world_size=1, local_rank=0, is_main=True, backend=None), with .distributed property and .single()
- init_distributed(backend=None, timeout_s=1800) (no-op without torchrun env); is_dist_initialized(); wrap_ddp(model, info, *, find_unused_parameters=False); unwrap_model(m)
- make_sampler(dataset, info=None, shuffle=True, seed=0, drop_last=False): always a DistributedSampler (num_replicas=1 in a single process); None for an IterableDataset. ShardSampler / make_eval_sampler(dataset, info) shard validation data without padding.
- barrier(info=None), cleanup(), all_reduce_sum(dict), all_reduce_mean(dict)

optim.py
- split_decay_params(model, keywords) -> (decay, no_decay) lists of (name, param)
- param_groups(model, weight_decay=0.05, *, lr=None, lr_mult=None, no_decay_keywords=("token","query","queries","embed")) -> groups carrying weight_decay, lr_mult, group_name and lr
- build_optimizer(model_or_params, name="adamw"|"adam"|"sgd", lr, weight_decay, betas, *, eps, momentum, fused="auto", lr_mult)
- lr_factor(step, schedule, warmup_steps, total_steps, min_lr_ratio): warmup factor is (s+1)/W. build_scheduler(opt, schedule, warmup_steps, total_steps, min_lr_ratio) -> LambdaLR stepped once per optimizer step.
- EMA(model, decay=0.999, *, warmup=True, device=None) with update(), apply_to(), swap() context manager, state_dict()/load_state_dict(), current_decay()

checkpoint.py
- save_checkpoint(path, **state) is atomic (write tmp file, fsync, os.replace); load_checkpoint(path|dir, map_location="cpu", *, weights_only=False); find_last(out_dir); find_best(out_dir); strip_state_dict_prefixes(sd); constants LAST_NAME="ckpt_last.pt", BEST_NAME="ckpt_best.pt"

logging_utils.py
- JsonlLogger(out_dir, filename="metrics.jsonl", *, enabled=True, tensorboard=False, wandb=None) with .log(step, metrics, *, kind=None), .log_config(), .close(); works as a context manager. TensorBoard and W&B imports are guarded and only warn if missing.
- read_jsonl(path, kind=None); to_float_dict(); setup_logging(level, rank)

sweep.py
- Path helpers: set_by_path, get_by_path, unflatten, flatten
- expand_grid(space) -> list of nested override dicts
- sample_random(space, n, seed): list = choice; {"log_uniform"|"uniform"|"int"|"choice": ...}
- shard(items, i, n)
- run_sweep(train_fn, base_cfg, overrides, out_dir=None, *, mode="min", metric_key=None, trial_dir_key="train.out_dir", resume=True, catch_errors=True, indices=None) -> results sorted best first; appends to results.jsonl as trials finish
- load_results(*paths, mode); suggest_from_space(trial, space); run_optuna(...) (optuna import guarded)
- main(argv): CLI `python -m robot_skin.train.sweep --fn mod:fn --base yaml --space yaml --out dir [--metric --direction --mode --n --seed --shard i/n --hardware]`

configs/hardware/{rtx5090,rtx4090,rtx3090,a100,cpu}.yaml
- Keys: name, gpu{arch, compute_capability, memory_gb, bf16, min_torch_cuda}, train{device, precision, tf32, compile, num_workers, pin_memory}, suggest{imu_pose|baseline|contact|pretrain|vtla: {batch_size, grad_accum}}, env{PYTORCH_CUDA_ALLOC_CONF}, notes.
- Every GPU profile keeps the same effective batch per stage (enforced by a test).
### deviations from spec
1. TrainConfig has extra optional fields beyond the spec: monitor_mode, early_stop_min_delta, optimizer, betas, eps, lr_mult, drop_last, eval_batch_size, compile_mode, tf32, ema_warmup, ckpt_every_steps, resume, find_unused_parameters, tensorboard, wandb_project. All spec fields and defaults are unchanged. from_dict also coerces types, because YAML 1.1 reads `3e-4` as a string.
2. max_steps, if set, takes precedence over max_epochs (HF-style). The scheduler is stepped per optimizer step. lr_factor warmup is (step+1)/warmup_steps, so the first update never uses LR 0.
3. Trainer has extra keyword-only args: optimizer (instance or builder), callbacks (epoch-end hooks whose returned metrics merge into the history row and can be the monitor; they can set should_stop), and dist_info.
   - evaluate() returns keys with a "val/" prefix.
   - Checkpoints carry extra keys: best, bad_epochs, history, rng, batch_in_epoch. This allows exact resume, including mid-epoch.
   - Under DDP, RNG is re-seeded per rank on resume rather than restored.
4. Hardware profiles put per-stage batch suggestions under `suggest.<stage>`, not under `train:`. Otherwise one profile would force the same batch_size on every stage. apply_hw_profile therefore takes an extra `stage=None` argument (defaults to cfg["stage"]). Precedence: stage train < suggest[stage] < profile train. It also records out["hardware"] = profile name.
5. Extra helpers not in the spec: maybe_apply_hw_profile (supports `hardware: auto`), detect_hw_profile, apply_profile_env, cuda_capability, and `python -m robot_skin.train.hardware`.
6. resolve_precision has a `capability=` kwarg (used by tests and for planning a remote GPU).
   - Explicit "bf16" on CPU is honoured with CPU autocast.
   - "fp16" on CPU or MPS falls back to fp32 with a warning.
   - "bf16" on a GPU below compute capability 8.0 falls back to fp16 + GradScaler with a warning.
   - resolve_device takes `local_rank`.
7. enable_tf32 uses only torch.set_float32_matmul_precision. In torch ≥ 2.9, mixing it with the new fp32_precision API raises RuntimeError (verified here on torch 2.14).
8. DistInfo.is_main is a field defaulting to True, plus a `.distributed` property. make_sampler always returns a DistributedSampler (num_replicas=1 in a single process, which needs no process group) for deterministic, resumable epoch shuffles. Validation uses a new non-padding ShardSampler so metrics stay exact under DDP.
9. EMA uses the standard warmup min(decay, (1+n)/(10+n)) by default (ema_warmup=True), and it averages buffers as well as parameters.
10. robot_skin/train/__init__.py uses lazy PEP 562 exports so `python -m robot_skin.train.hardware|sweep` runs without runpy double-import warnings.
11. run_sweep has extra kwargs: metric_key (dotted path allowed), trial_dir_key, resume, catch_errors and indices (global trial numbers for sharding). There are also shard(), load_results(), suggest_from_space() and a sweep CLI.
### notes for consumers / interface_requests
- [INTEG] robot_skin/__main__.py: the README (robot_skin/train/README.md §4–5) assumes the top-level CLI does three things:
  - `python -m robot_skin train <stage> --config <yaml> --hardware <name|auto|path> --set key=value ...`, working under `torchrun ... -m robot_skin train <stage>`.
  - Applies the profile via robot_skin.train.apply_hw_profile(cfg, hw, stage=<stage>) (or maybe_apply_hw_profile), then CLI overrides, and calls robot_skin.train.apply_profile_env(profile) before any CUDA call.
  - Has `env` → robot_skin.train.hardware.main(argv) and `sweep` → robot_skin.train.sweep.main(argv).
  If the chosen flag names differ (e.g. not `--set`), please update the command examples in robot_skin/train/README.md or tell me.
- [STAGE1 / REPR / VTLA] stage runners:
  - Build with `TrainConfig.from_dict(cfg["train"])` and `Trainer(model, loss_fn, tcfg, train_ds, val_ds, collate_fn=..., extra_state={normalizers, model config})`.
  - Add `stage: <name>` to each stage YAML (names: imu_pose, baseline, contact, pretrain, vtla) so profile `suggest.<stage>` batch sizes apply.
  - Return the metrics dict from run(cfg) so sweeps can pick `--metric val/loss`.
  - loss_fn must call model(...) forward only during training (DDP/compile wrapper). Use robot_skin.train.unwrap_model for custom methods.
- [VTLA] vtla.yaml: set `train.find_unused_parameters: true` when modality dropout can skip a branch under DDP. Use `train.lr_mult: {<vision encoder prefix>: 0.1}` for pretrained encoders (0 = excluded/frozen).
- [INTEG] configs/default.yaml could add `hardware: auto` (or null) as a top-level default.
- [DOCS] docs/TRAINING.md can link to or summarise robot_skin/train/README.md: profiles, RTX 5090 cu128 setup, DDP, Tailscale workflow.

### reviewer: fixes applied
- engine.py: each micro-batch loss is now weighted by its share of the group's samples. The per-rank micro-batch sizes come from the sampler, via new Trainer._micro_batch_sizes; user-supplied loaders fall back to equal weights. n=60 and n=62 now match the large batch to 2e-7.
- engine.py: when an unknown-length loader ends mid-group, the leftover gradients are rescaled by accum/pending before the step (new _scale_grads). Matches the hand-written reference to 0.0.
- engine.py: the EMA is now created after the DDP wrap / compile, so it starts from the weights DDP copied from rank 0. Confirmed identical EMA across 2 gloo ranks.
- engine.py: new _zero_grad clears optimizer and model grads after every step and at the start of fit(). This fixes stale grads when fit() is re-entered and the build-up on lr_mult-0 params.
- engine.py: the monitor fallback (val/* → train/*, with a warning) now also applies when eval_every_epochs == 0. Docstrings updated, including the module docstring on how accumulation is weighted.
- distributed.py: DistInfo.is_main defaults to rank == 0 (explicit values still win), and rank/world_size are validated. wrap_ddp uses the parameter's actual CUDA device index. Minor getattr cleanup.
- logging_utils.py: numpy arrays are written as lists; NaN/inf inside tensors and arrays become strings, so every line stays valid JSON.
- hardware.py: check_arch_support flags Blackwell GPUs (cc >= 10) with a CUDA < 12.8 build even when a PTX entry exists, with a PTX-specific message.
- optim.py: norm layers are now detected by the class-name suffix regex norm(\d+d)?$ (LayerNorm2d and RMSNorm still match; NormalEncoder and Normalizer do not). param_groups warns about lr_mult prefixes that match no trainable parameter.
- sweep.py: `--hardware` now calls load_hw_profile, then apply_profile_env (before CUDA init), then apply_hw_profile.
- New or strengthened tests in test_train_engine.py: accumulation for n in {64, 56, 60, 62} × {adamw, sgd}; stale grads on re-fit after an interrupt (checked by breaking the fix: the test fails); IterableDataset partial last group; lr_mult=0 freeze with no grad build-up; eval_every_epochs=0 fallback; fp16 GradScaler path on CPU with exact resume including scaler state; norm-by-class-name and lr_mult typo warning; JsonlLogger numpy/NaN output parsed strictly.
- New tests in test_hardware.py: DistInfo is_main defaults; arch check with PTX on an old toolkit; a real 2-process gloo DDP test. It runs 2 ranks × batch 8 × accum 2 with a different init per rank and checks against 1 process × batch 32: parameters, EMA shadow, per-epoch val/loss (exact over sharded 21-sample validation), step count, and rank-0-only files. I confirmed it fails with the old EMA ordering. It skips if gloo or loopback sockets are unavailable.
- New test in test_sweep.py: the CLI `--hardware cpu` applies the profile (device, suggest.vtla batch size), and trial overrides still win over the profile.
- README.md: documented how accumulation is weighted, the limits of bit-exact mid-epoch resume, and the new test coverage.
### reviewer: remaining concerns
- **Untested on GPU:** the paths that need CUDA are still unexercised here: NCCL DDP, fused AdamW, torch.compile on CUDA, and cross-node DDP over Tailscale. Check on the RTX 5090 machine with `python -m robot_skin.train.hardware` and a `torchrun --standalone --nproc_per_node=1` smoke run.
- **User-supplied loaders:** a DataLoader or IterableDataset passed in by the user still weights micro-batches equally within a group, because sizes are not known in advance. Accumulation with a short last batch is exact only for loaders the Trainer builds itself.
- **Resume is not always bit-exact:** mid-epoch resume (Ctrl-C or ckpt_every_steps) is bit-exact only for deterministic models. The random numbers for dropout in the replayed group and for worker-side augmentation change. DDP resume re-seeds RNG per rank. This is documented in the README.
- **For [INTEG]:** Trainer calls init_distributed() but never cleanup(). The CLI (`robot_skin/__main__.py`) should call robot_skin.train.cleanup() at exit under torchrun, and apply_profile_env(profile) before CUDA init. The implementer's interface requests still stand (CLI flag names, a `stage:` key in stage YAMLs, `find_unused_parameters` and `lr_mult` settings for VTLA).
- **Test runtime and sockets:** test_hardware.py now spawns 2 processes that talk over TCP on 127.0.0.1 (~5 s). It stays under the 10 s per-file budget and skips if gloo or loopback sockets are unavailable.
- **Cosmetic:** engine.py line 452 (the Trainer signature, from the implementer) is 114 characters. The F401 lint hits in `__init__.py` are intentional imports for type checkers.
- **Files edited, all owned by [TRAIN]:** robot_skin/train/{engine,distributed,hardware,optim,logging_utils,sweep}.py, robot_skin/train/README.md, and robot_skin/tests/{test_train_engine,test_hardware,test_sweep}.py. Nothing outside ownership was touched.


## VISLANG
### public_api
robot_skin.vision.encoders:
- VisionEncoder(nn.Module) base: .out_dim, .mean/.std (the normalization it expects), .n_tokens(h,w) (analytic for Tiny/ResNet; for HF, one cached dummy forward), .cache_key, .is_frozen, .freeze(), forward(images float[B,3,H,W] normalized) -> tokens [B,P,D].
- TinyConvEncoder(out_dim=128, grid=(4,4), *, pool="grid", channels=(32,64,128), n_keypoints=16, groups=8, pos_embed=True): a GroupNorm CNN. grid -> fixed P=gh*gw for any image size; none -> ceil(h/2^S)*ceil(w/2^S).
- ResNetEncoder(name="resnet18"|"resnet34"|"resnet50", *, pretrained=True, frozen=True, out_dim=None, pool="none", grid, n_keypoints=32, group_norm=False, pos_embed=True): uses torchvision (guarded import). frozen freezes the trunk only and keeps its BatchNorm in eval mode.
- HFVisionEncoder(model_id="facebook/dinov2-small", *, pretrained=True, frozen=True, out_dim=None, tokens="all"|"pooled", revision=None, local_files_only=False): uses transformers (guarded). For CLIPModel/SiglipModel it uses .vision_model. mean/std come from the image processor. Passes interpolate_pos_encoding when supported.
- TokenPool(in_ch, out_dim, pool in POOL_MODES=("none","grid","avg","spatial_softmax"), grid, n_keypoints, pos_embed); SpatialSoftmax(in_ch, n_keypoints, temperature, learnable_temperature) -> [B,K,2] in [-1,1]; sincos_pos_embed_2d(h,w,dim); sanitize_key(s).
- build_vision_encoder(cfg=None, **overrides): type tiny (default) | tiny_conv | conv | resnet | resnet18/34/50 | torchvision | hf | transformers | huggingface | dinov2 | siglip | clip. Unknown keys -> ValueError listing accepted keys; missing optional package -> ImportError with a pip hint and the 'tiny' alternative.

robot_skin.vision.transforms:
- Constants: IMAGENET_/CLIP_/SIGLIP_ MEAN/STD.
- to_float_tensor(images, *, channels_last=None, device=None): uint8 [...,H,W,3] (numpy/torch, including read-only memmaps) -> float32 [...,3,H,W] in [0,1].
- Normalize(mean,std) (nn.Module) with .inverse.
- resize(x,size), resize_short(x,size), center_crop(x,size), crop_box(...), crop_resize(x,fw,fh,cx,cy,out_hw) (per-sample affine crop, anti-alias prefilter).
- EvalTransform(out_size=None, *, crop_scale=1.0, stretch=False, mean, std): deterministic.
- TrainAugment(out_size=None, *, scale=(0.8,1.0), ratio=(1,1), brightness=0.2, contrast=0.2, saturation=0.0, stretch=False, mean, std, seed=None): random resized crop plus colour jitter, all torch ops, on any device. Params are drawn per first dim and shared across the other leading dims ([B,T,3,H,W] history gets one crop). Also: __call__(images, generator=None), sample_params, reseed, eval_transform(crop_scale=None -> mean train area). Picklable. Inside DataLoader workers it reseeds per worker and per epoch.
- build_transforms(cfg=None, encoder=None) -> (train, eval). Keys: image_size, scale, ratio, brightness, contrast, saturation, stretch, seed, eval_crop_scale, mean, std, augment, antialias.

robot_skin.vision.feature_cache:
- feature_path(episode|dir, camera, key) -> <ep>/derived/vision_<key>_<camera>.npy
- cache_episode_features(episode|dir, camera, encoder, transform=None, *, batch_size=64, device="cpu", key=None (=encoder.cache_key), overwrite=False, autocast_dtype=None) -> Path. Writes float16 [F,P,D] per camera frame, atomically, plus a .json sidecar. Skips if a valid cache exists. Warns if the encoder is not frozen.
- cache_features(episodes, cameras|None, encoder, transform, **kw) -> list[Path]
- load_cached(episode|dir, camera, key, *, mmap=True, validate=True) -> np.float16 [F,P,D]. Raises on stale frame count.
- has_cached, load_cache_info.
- gather_frame_features(feats, frame_idx) -> (float32 [...,P,D], valid bool[...]); -1 -> zeros/False.
- CLI: python -m robot_skin.vision.feature_cache --root --dataset --cameras --encoder JSON|YAML --image-size H W --crop-scale --key --device --batch-size --bf16 --overwrite

robot_skin.language.text_encoders:
- Constants PAD_ID=0, BOS_ID=1, N_SPECIAL=2.
- simple_word_tokenize(text, lowercase=True): NFKC, casefold, \w+ split; works for Korean.
- hash_token(tok, vocab_size, seed=0) = 2 + crc32(f"{seed}:{tok}") % (vocab-2).
- pool_tokens(tokens, pad_mask) -> [B,D].
- TextEncoder(nn.Module) base: get_tokenizer() (picklable callable, no weights), tokenize(texts) -> {input_ids long[B,L], pad_mask bool[B,L] (True = padding)}, forward(input_ids, pad_mask) -> [B,L,D] with padding rows zeroed, encode(texts, device=None) -> (tokens, pad_mask), .out_dim, .max_len, .fixed_len (int = static L, or None), .cache_key, .is_frozen, .freeze(), .device.
- HashingTokenizer(max_len=32, vocab_size=2**16, *, hash_seed=0, lowercase=True, pad_to_max=True): [BOS] always at position 0, so a row is never fully masked.
- HashingTextEncoder(dim=256, max_len=32, vocab_size=2**16, *, hash_seed=0, n_layers=0, n_heads=4, dropout=0.0, lowercase=True, pad_to_max=True): token and learned position embeddings, LayerNorm, optional pre-LN transformer.
- HFTokenizerFn; HFTextEncoder(model_id="openai/clip-vit-base-patch32", *, frozen=True, max_len=77, out_dim=None, padding="longest"|"max_length", pretrained=True, revision=None, local_files_only=False): uses text_model for CLIP/SigLIP and get_encoder() for T5; guarded import.
- build_text_encoder(cfg=None, **overrides): type hashing (default) | hash | hf | transformers | huggingface | clip | siglip (max_len 64, padding max_length) | t5 (t5-small).
- InstructionCache(encoder, *, batch_size=64, dtype=float32, allow_trainable=False): warm(texts), get(text) -> [L_i,D] valid tokens, encode(texts, device=None) -> (tokens, pad_mask) with the same shapes as encoder.encode (CPU by default), state_dict/load_state_dict, save(path), InstructionCache.load(path, encoder, strict=True), n_encoded, len/in.
### deviations from spec
- Feature-cache files are per camera frame, as the spec says, and are written directly to derived/vision_<key>_<camera>.npy without Episode.set_derived, because set_derived requires T rows. They must be read with vision.load_cached, not Episode.derived: Episode.derived would put them in _derived, and a later Episode.save would then fail set_derived's T-row check. This is documented in the module docstring and README.
- An explicit `key` is validated against [A-Za-z0-9][A-Za-z0-9_.-]* and raises on mismatch; it is not silently sanitized. Default keys come from encoder.cache_key, which is already sanitized.
- Additions beyond the spec: TokenPool; pool modes for Tiny/ResNet; `tokens` mode for HF; `group_norm` for ResNet; cache_features plus a CLI; gather_frame_features; has_cached/load_cache_info; HashingTokenizer and HFTokenizerFn picklable tokenizers via TextEncoder.get_tokenizer(), so collate/DataLoader workers can tokenize without holding the model; optional n_layers transformer in HashingTextEncoder; pool_tokens; InstructionCache save/load.
- HashingTextEncoder always puts a [BOS] token at position 0, and max_len includes it (up to max_len-1 words), so no row is ever fully masked. HFTokenizerFn also un-masks position 0 of an all-padding row.
- Defaults: TrainAugment scale=(0.8,1.0) with no flips. build_transforms' eval crop defaults to the mean train crop area so train and eval see the same zoom; pass eval_crop_scale=1.0 for the full view.
- `frozen=True` on ResNetEncoder/HFVisionEncoder/HFTextEncoder freezes the backbone only; an out_dim projection stays trainable. For caching, use out_dim=None.
### notes for consumers / interface_requests
None required. Notes for consumers:
- VTLA agent: vision uses build_vision_encoder(cfg) and build_transforms(cfg_image, encoder) -> (train_tf, eval_tf). Transforms accept uint8 [B,H,W,3] or [B,T,H,W,3] and return normalized [B,(T,)3,h,w]. For cached features use load_cached(ep, cam, key) and map ticks to frames with gather_frame_features(feats, ep[cam_idx_key(cam)][ticks]).
- VTLA agent, text: build_text_encoder(cfg). The dataset/collate can hold enc.get_tokenizer() and emit input_ids/pad_mask tensors; the model calls text_enc(input_ids, pad_mask), or text_enc.encode(list_of_str) inside forward. pad_mask is True = padding, matching nn.MultiheadAttention key_padding_mask. For frozen HF towers use InstructionCache.
- CTRL agent: at deployment use the same eval transform (train_tf.eval_transform() / build_transforms' eval) and the same encoder mean/std. These should be saved in the policy bundle config.
- INTEG/DOCS: config blocks `vision:` (type/pretrained/frozen/out_dim/pool...), `image:` (build_transforms keys) and `language:` (type/dim/max_len/n_layers) can be added to default.yaml or vtla.yaml; the examples are in the READMEs.

### reviewer: fixes applied
- transforms.crop_box: `scale` is now the area fraction of the largest crop with the (jittered) target aspect that fits the image: w_max = min(W, H*aspect), and both sides scale by sqrt(scale). Matching-aspect behaviour is unchanged. Pixel-exact centre crops were derived by hand and verified: 24x32 to (24,24) gives cols 4..28; crop_scale 0.25 to (12,12) gives rows 6..18, cols 10..22; (16,32) gives rows 4..20. Docstrings for crop_box, EvalTransform and TrainAugment updated.
- transforms.crop_resize: the anti-alias prefilter is now per axis (only downsamples the axes minified by more than 2x). Verified against F.interpolate(antialias=True): mean abs diff 5.6e-9 for 480x640 to 64x16 stretch, 3.7e-3 for 480x640 to 64x64 (vs 0.146 without the prefilter).
- transforms: empty batches handled (_flat/_prepare use an explicit N = prod(lead); early returns in TrainAugment/EvalTransform/crop_resize). EvalTransform.__repr__ is now deterministic and includes antialias and the mean/std.
- encoders.sincos_pos_embed_2d: computed once per (h,w,dim) in float64 on the CPU (lru_cache) and returned as a fresh copy on the requested device/dtype. This works on MPS, and mutating the result cannot corrupt the cache. Checked against a numpy MAE reference and hand values (dim=8, cell y=1,x=2).
- encoders: `frozen_backbone` is now a live property on VisionEncoder (no backbone parameter requires grad). ResNet and HF vision train() and forward use it, so manual unfreezing enables gradients and train-mode BN/dropout. Same change for HFTextEncoder. Verified with real torchvision 0.29 and transformers 5.17 from the implementer's scratch installs.
- encoders: the n_tokens fallback matches the parameter dtype. TinyConvEncoder accepts out_dim=None (D = last stage width). ResNet warns on group_norm+pretrained+frozen and its cache_key includes _gn. The HF vision cache_key includes _rev-<rev> and _scratch.
- feature_cache: an existing cache is reused only if its sidecar's encoder_cache_key, out_dim and transform (deterministic repr only; callables with object-address reprs are not compared) match; otherwise it raises ValueError asking for overwrite=True or another key. The sidecar also records mean/std and is written atomically. load_cached and has_cached fall back to the sidecar n_frames when the camera dir is absent and skip the check only if neither source exists. Module docstring updated.
- language: HFTextEncoder forces tokenizer.padding_side='right'. Its cache_key includes revision and scratch. HashingTextEncoder.cache_key now includes max_len and a '_cased' suffix.
- Tests added to test_vision.py (38 to 51): pixel-exact crops and crop_box hand values for 4:3 to square; zoom is active for mismatched aspects and boxes stay inside the image with the aspect honoured; anisotropic anti-aliasing plus empty batches; per-worker/per-epoch reseeding with a monkeypatched get_worker_info; sincos hand values and cache aliasing; base n_tokens fallback in float64; Tiny out_dim=None; a fake torchvision module covering ResNetEncoder n_tokens==forward for all 4 pools x 3 sizes, frozen BN stats unchanged, unfreeze gives grads, GN warning, cache keys; a fake transformers covering HFVisionEncoder (dual-tower selection, processor mean/std, no dropout when frozen in train mode, n_tokens fallback, pooled + projection, revision/scratch key, unfreeze); feature-cache signature mismatch, overwrite, lambda transform not compared, and loading without frames or without the camera dir.
- Tests added to test_language.py (14 to 19): the hash value 262 is hardcoded, with crc32('0:cup')=0xb049dea8 re-derived with a bitwise CRC-32 so it guards cross-machine stability; pool_tokens hand values; hashing cache_key tracks tokenization; a fake transformers covering HFTextEncoder (CLIP-style dual tower, left-padding tokenizer forced to right, pad masks, empty-string row unmasked, frozen means deterministic and no grad, attention mask honoured so no padding leak, InstructionCache equals encode, picklable tokenizer; SigLIP family max_length with no mask and fixed_len; T5 get_encoder with d_model, trainable path with grads, manual unfreeze).
- READMEs (vision, language) updated for the crop-scale semantics, cache staleness and frame-less validation, the live frozen-state behaviour, right padding and cache-key scope. Also added a note that VTLA training (cached) and deployment (online) must use the same eval transform (CLI default crop 1.0 vs build_transforms eval default = mean train scale).
### reviewer: remaining concerns
1. Pretrained weights are still untested: the hub and download.pytorch.org are blocked here, so only random/local checkpoints were run. Check on the GPU box (HF_HOME/TORCH_HOME).
2. Cache keys and the InstructionCache key identify the architecture and tokenization, not the trained weights. A trained-then-frozen Tiny or hashing encoder reusing a key cannot be detected; this is documented.
3. build_vision_encoder is strict on purpose: `type: tiny` with leftover `pretrained`/`frozen` keys from a ResNet YAML block raises ValueError listing the accepted keys. `out_dim: null` now works.
4. The lru_cache in sincos_pos_embed_2d may cause a torch.compile graph break. It is not an error, and compile is opt-in.
5. Multi-worker DataLoader reseeding is covered with a monkeypatched get_worker_info. The implementer's real 2-worker check was manual.
6. Consumers should know that cache-key strings changed: hashing keys now carry `_L<max_len>`, ResNet `_gn`, HF `_rev-*`/`_scratch`. No other module depends on them yet.
7. For VTLA/CTRL: the eval transform used for cached features (recorded in the sidecar `transform`) must equal the deployment transform. That should be stored in the policy bundle.


## ACTION
### public_api
robot_skin.action (re-exports everything below)

space.py
- Constants: HAND_MANO_DIM=54; MANO_FINGER_JOINTS (15 names in MANO order; a test checks it equals pose.mano.MANO_JOINTS[1:]); HAND_MANO_NAMES; WRIST_POS=slice(0,3), WRIST_ROT6D=slice(3,9), FINGERS_AA=slice(9,54); HAND_SLICES; ACTION_KINDS; REL_MODES=("abs","delta","delta_pose"); NORM_METHODS=("std","robust","minmax","none")
- @dataclass(frozen) ActionSpec(kind, dim, names=(), rot_repr): .hand_mano(), .robot_joint(names|int), .slices, .to_dict()/.from_dict()
- hand_action_from_arrays(global_orient[...,3], finger_pose[...,15,3], wrist_pos[...,3]) -> [...,54]. Layout: wrist pos | wrist rot 6D (first two columns of R) | finger axis-angle 45. numpy in gives float32 numpy out; torch in gives torch out with the same dtype.
- hand_action_to_arrays(a[...,54]) -> {"global_orient","finger_pose","wrist_pos"}. Decodes 6D via Gram-Schmidt, then matrix to axis-angle with angle in [0, pi]. The keys match ManoSkeleton.forward.
- wrist_rotation(a) -> [...,3,3]; robot_action_from_q(q[T,D]) -> float32 copy
- hand_action_from_episode(ep) -> (actions[T,54] float32, valid[T]) using hand_pose_valid; actions_from_episode(ep, spec | "hand_mano" | "robot_joint") -> (actions, valid)
- make_relative(actions[...,(H,)A], state[...,A], spec, mode="delta") and make_absolute(...), exact inverses. delta: hand subtracts the current wrist position (world frame); robot uses q - q_cur. delta_pose (hand only): position and rotation in the current wrist frame. abs: returns a copy.
- ActionNormalizer(stats: NormStats, spec=None, method): .fit(actions | [arrays], valid=None, *, spec, method="std"|"robust"|"minmax"|"none", eps=1e-6, min_scale=1e-2); .normalize/.unnormalize (numpy or torch, keeps dtype/device); .dim; .to_dict/.from_dict/.save/.load

chunking.py
- policy_stride(source_hz=200, policy_hz=20, *, tol=0.01) -> int. Warns when the ratio is not an integer.
- policy_tick_indices(T, stride, *, start=0, mask=None, min_future=0) -> int64 indices
- action_chunk(actions[T,A], t_index, horizon, stride=1, *, offset=1, valid=None) -> (chunk[H,A], valid[H]). chunk[i] = actions[t+(offset+i)*stride]. Steps past the episode end repeat the last frame and have valid=False. The optional source mask [T] (e.g. hand_pose_valid) is ANDed in.
- action_chunks(actions, t_indices[B], horizon, stride, *, offset=1, valid=None) -> ([B,H,A], [B,H]). Vectorised; works on numpy, memmap and torch.
- TemporalEnsembler(horizon, action_dim, k=0.01): add(chunk[h<=H, A]), step() -> float64 [A], reset(), weights(n), current_predictions(), properties t / n_chunks / ready. Weighting is ACT's w_i = exp(-k*i) with i=0 the oldest prediction. Calling add() twice in the same step replaces the earlier chunk. step() with nothing covering the current step raises RuntimeError.

retarget.py
- HAND_FINGERS = ("thumb","index","middle","ring","pinky") (= pose.mano.FINGERS), WRIST="wrist", RETARGET_METHODS=("lm","adam","lbfgs"), JACOBIAN_MODES=("autograd","fd"), VECTOR_SETS
- FingertipRetargeter(fk, tip_links=None, *, base_link=None, lower=None, upper=None, dof=None, human_tip_names=HAND_FINGERS, scale=1.0, human_to_robot=None, vectors="tips+pairs"|"tips"|"pairs"|[(origin,target)], tip_weight=1, pair_weight=1, reg_weight=1e-7, smooth_weight=1e-6, q_nominal=None (default mid-range), pinch_threshold=None, pinch_distance=0.0, pinch_weight=None, method="lm", iters=None (lm 50 / adam 300 / lbfgs 100), lr=0.02, tol=1e-8, ftol=1e-10, jacobian="autograd"|"fd", dtype=float64, device="cpu")
  - fk can be any callable q[B,D] -> {name: pos[B,3] or T[B,4,4]}, or a duck-typed model with .fk(q, links=...) and lower/upper (plus n_dof/joint_names if present). pose.urdf.URDFModel works and is never imported.
  - Methods: retarget(human_tip_pos[...,F,3], human_wrist_T=None, q_init=None, q_prev=None) -> q[...,D] (numpy in gives numpy out); retarget_sequence(tips[T,F,3], human_wrist_T=None, q_init=None, *, warm_start=True, smooth=True) -> [T,D]; step(tips, human_wrist_T=None) and reset(q=None) for streaming; vector_error(q, tips) -> [...,V] in metres; estimate_scale(tips_ref, q_ref=None) -> float; robot_points(q) / robot_vectors(q) / human_vectors(tips); from_config(fk, dict) (unknown keys raise); attributes dof, fingers, vectors, lower, upper, q_nominal, joint_names, last_info{method, iters, cost}
- human_fingertips(finger_pose[...,15,3], skeleton=None) -> [...,5,3]: wrist-frame fingertips via a lazy import of pose.mano.ManoSkeleton
- hand_action_fingertips(a[...,54], skeleton=None) -> [...,5,3]
### deviations from spec
1. Retarget scale convention. The spec cost is ||s*(p_r - p_r_base) - (p_h - p_h_wrist)||^2. I follow AnyTeleop instead: `scale` multiplies the human vectors (target = scale * R_hr * v_h), so scale = robot size / human size. This is the spec's s with s -> 1/scale; it is documented in the module docstring and README. estimate_scale() returns a value in this convention.
2. Solvers. Besides Adam and L-BFGS from the spec, the default is `lm`: projected Levenberg-Marquardt with an active set at the joint limits, Jacobian from torch autograd (vectorised reverse mode). Adam is projected (clamped every step); L-BFGS runs unconstrained and is clamped once at the end. There is also an optional `jacobian="fd"` (central differences in one batched FK call), about 2x faster on a Python-heavy URDF FK.
3. Weights. `reg_weight` pulls toward q_nominal (null-space regulariser, like DexPilot's small q term). `smooth_weight` pulls toward q_prev (temporal smoothing). The spec's reg ||q - q_prev||^2 is the smooth term.
4. q_nominal default is mid-range of the joint limits (0 for unbounded joints), not 0. Starting a flexion joint at its lower limit (a straight finger, singular Jacobian) trapped the solver on the boundary in testing.
5. Extras added: optional DexPilot-inspired pinch handling (off by default); human_to_robot frame rotation; human_wrist_T for world-frame fingertips; ActionNormalizer method "minmax" and a `min_scale` floor (default 0.01) for near-constant dims; relative mode `delta_pose`; action_chunk `offset` parameter (default 1, matching the spec's t+stride, ...) and a source `valid` mask; `action_chunks` batch version; `policy_tick_indices`; `actions_from_episode`; `human_fingertips` / `hand_action_fingertips` helpers; `FingertipRetargeter.from_config`.
6. hand_action_to_arrays returns a dict {global_orient, finger_pose, wrist_pos} rather than a tuple, so it can be splatted straight into ManoSkeleton.forward and hand_action_from_arrays.
### notes for consumers / interface_requests
None required.

Notes for consumers:
- VTLA agent: use actions_from_episode + policy_stride / policy_tick_indices + action_chunks(valid=hand_pose_valid). Fit ActionNormalizer on the same representation you train on (absolute, or make_relative output) and store norm.to_dict() and spec.to_dict() in policy_bundle.pt.
- CTRL agent: the runner should call TemporalEnsembler.reset() at episode start. For hand_mano, map the action through make_absolute (if a relative mode was used), then hand_action_fingertips(a), then FingertipRetargeter.step(tips).
- Real robots: set `human_to_robot`. The MANO canonical frame (fingers along -x, radial side +z, palm facing -y) almost never matches the robot palm-link axes.
- Real-time use: cap `iters` (about 5-10 is enough when warm-started) and consider jacobian="fd". On a synthetic 16-DoF URDF, warm-started frames took 4-5 LM iterations: roughly 40 ms per frame with autograd and roughly 20 ms with fd on this CPU.
- deploy.yaml can take a `retarget:` block that goes straight into FingertipRetargeter.from_config(urdf_model, cfg).

### reviewer: fixes applied
- robot_skin/action/chunking.py: added `_last_valid_index`. When `valid` is given, every masked step (past the end or invalid source) now takes the last valid source frame at or before it, or the first valid frame if none precedes. `action_chunks` now accepts lists. Docstrings updated.
- robot_skin/action/space.py: new `HandArrays` NamedTuple returned by `hand_action_to_arrays`. It unpacks as a tuple (go, fp, wp) and also supports h['key'], keys(), items(), `in` and ** splatting, so ManoSkeleton.forward(**h) still works. It stays differentiable for torch input.
- space.py: `make_relative`/`make_absolute` now take `spec='hand_mano'` by default; `_as_spec` accepts 'robot_joint' and infers the dim from the actions.
- space.py: ActionSpec `rot_repr` defaults to None and resolves per kind ('6d' / 'none'). Missing hand_mano names default to HAND_MANO_NAMES. Validates dim >= 1. from_dict tolerates a missing rot_repr.
- space.py: ActionNormalizer accepts str/dict specs; validates finite stats, scale > 0 and a known method; adds `apply`/`invert` aliases; from_dict accepts a bare NormStats dict. `actions_from_episode` checks for 'q' first and its docstring notes that glove q is the 45-D finger pose.
- robot_skin/action/retarget.py: constructor positional order now follows the spec (fk, tip_links, lower, upper, human_tip_names, scale, reg_weight, smooth_weight, iters, lr), with the rest keyword-only; all existing keyword calls and from_config are unaffected. Non-finite human_tip_pos, human_wrist_T and q_init/q_prev now raise ValueError. `_q_batch` accepts leading dims. The scale-convention note is added to the class docstring.
- robot_skin/action/__init__.py: now exports HandArrays, ACTION_KINDS and NORM_METHODS.
- robot_skin/action/README.md: documents HandArrays, the default spec, last-valid padding (NaN rationale), the constructor order, the NaN guard and the scale convention (scale = 1/s of the spec formula).
- robot_skin/tests/test_action.py: updated the dict assertion; added tests for tuple and mapping use of HandArrays and splatting into ManoSkeleton.forward, decode differentiability, ActionSpec defaults and validation, normalizer aliases / bare dict / NaN stats / str spec, hand-computed std and minmax stats, hand-computed delta_pose and default spec / 'robot_joint' string, and last-valid padding with NaN invalid frames (numpy and torch) plus list input.
- robot_skin/tests/test_retarget.py: added test_spec_positional_order_and_input_validation (positional order; NaN tips / inf wrist / NaN q_init raise for all solvers; q_init with leading dims) and test_hand_derived_planar_fingertip (closed-form tip positions, recovery, limits).
### reviewer: remaining concerns
1. Scale convention: FingertipRetargeter.scale multiplies the human vectors (AnyTeleop), so it is 1/s of the spec formula ||s*v_r - v_h||^2. The optimum is the same and this is documented, but please tell CTRL/deploy authors to calibrate it with estimate_scale() rather than setting it from the spec formula.
2. hand_action_to_arrays now returns HandArrays, a NamedTuple that also acts as a mapping. Iterating over it yields values, not keys. Code that did `for k in result` or `set(result)` against the implementer's dict description must use `.keys()`; `result['key']` and `**result` still work.
3. TemporalEnsembler.step() returns float64 numpy. retarget() returns self.dtype (float64 by default) even for float32 torch input.
4. LM cold starts on a 20-DoF hand can run to the 50-iteration cap, though vector errors are already about 1e-7 m. For real-time use, warm starts need about 5 iterations; set `iters` accordingly.
5. The ActionNormalizer default min_scale=0.01 is a heuristic floor. For robot_joint delta actions with very small per-step motion, VTLA may want a lower value.
No files outside [ACTION] ownership were edited; common/ and deformable_sats/ are untouched. Nothing was committed.


## REFS
### public_api
I wrote docs/REFERENCES.md. It is in Korean with English titles. Each table row lists the paper, a link, what robot_skin takes from it, and the robot_skin module path. There are 7 pipeline sections, a VIHand section, a section on internal files in this repo, and a list of corrections to the spec.

I checked title, first author, venue/year and arXiv id through web-search results and publisher/proceedings pages. arxiv.org, dl.acm.org, huggingface.co and semanticscholar were blocked by the network proxy, so no paper PDF was opened. The doc says this.

ALL SPEC REFERENCES CHECKED (title and id match the spec unless a correction is noted):
- VTLA: Vision-Tactile-Language-Action Model with Preference Learning for Insertion Manipulation — Zhang et al., arXiv:2505.09577 (2025). A journal version is in Biomimetic Intelligence and Robotics (2026, ScienceDirect).
- 3D-ViTac — Huang et al., CoRL 2024, arXiv:2410.24091.
- ActionSense — DelPreto et al., NeurIPS 2022 Datasets & Benchmarks. No arXiv id found; the doc links the NeurIPS proceedings page and OpenReview olvz0gAdGOX.
- OSMO: Open-Source Tactile Glove for Human-to-Robot Skill Transfer — Yin et al., arXiv:2512.08920 (Dec 2025). The glove has 12 three-axis sensors.
- DexUMI — Xu et al., CoRL 2025, arXiv:2505.21864.
- ACT — Zhao et al., RSS 2023, arXiv:2304.13705.
- Diffusion Policy — Chi et al., RSS 2023 (extended version in IJRR), arXiv:2303.04137.
- π0 — Black et al., RSS 2025 (roboticsproceedings rss21 p010), arXiv:2410.24164.
- Zhou et al., rotation continuity (6D) — CVPR 2019, arXiv:1812.07035.
- Kendall & Gal — NIPS 2017, arXiv:1703.04977.
- MANO (Embodied Hands) — Romero et al., ACM TOG 36(6), SIGGRAPH Asia 2017, DOI 10.1145/3130800.3130883, arXiv:2201.02610.
- MAE — He et al., CVPR 2022, arXiv:2111.06377.
- HaMeR — Pavlakos et al., CVPR 2024, arXiv:2312.05251.
- AnyTeleop — Qin et al., RSS 2023, arXiv:2307.04577.
- DexPilot — Handa et al., ICRA 2020, arXiv:1910.03135.
- DexCap — Wang et al., RSS 2024, arXiv:2403.07788.
- OpenVLA — Kim et al., CoRL 2024, arXiv:2406.09246.
- Octo — Octo Model Team (Ghosh et al.), RSS 2024, arXiv:2405.12213.
- Flow Matching — Lipman et al., ICLR 2023, arXiv:2210.02747.

ADDED AND CHECKED:
- VIHand: Enhancing 3D Hand Pose Estimation with Visual-Inertial Benchmark — Wang et al., ACM MM 2025, DOI 10.1145/3746027.3758215, project page shirley0118.github.io/VIHand. Only partly checked; details below.
- VIST, Visual-inertial hand motion tracking — Lee et al., Science Robotics 6(58) 2021, DOI 10.1126/scirobotics.abe1315.
- STAG — Sundaram et al., Nature 569:698–702 (2019), doi 10.1038/s41586-019-1234-z.
- WiLoR — Potamias et al., CVPR 2025, arXiv:2409.12259.
- Sparsh — Higuera et al., CoRL 2024, arXiv:2410.24090.
- Pose-Aware Modeling to Mitigate Pose-Related Artifacts in Tactile Gloves — Yu et al., arXiv:2607.22964 (Jul 2026). It tackles the same problem as the D1 no-contact baseline predictor.
- Focal Loss — Lin et al., ICCV 2017, arXiv:1708.02002.
- Fourier Features — Tancik et al., NeurIPS 2020, arXiv:2006.10739.
- Perceiver — Jaegle et al., ICML 2021, arXiv:2103.03206.
- Rectified Flow (Flow Straight and Fast) — Liu et al., ICLR 2023, arXiv:2209.03003.
- DPO — Rafailov et al., NeurIPS 2023, arXiv:2305.18290.
- DINOv2 — Oquab et al., TMLR 2024, arXiv:2304.07193.
- SigLIP — Zhai et al., ICCV 2023, arXiv:2303.15343.
- CLIP — Radford et al., ICML 2021, arXiv:2103.00020.

INTERNAL FILES (existence confirmed): deformable_sats/sats/bending/baseline_restorer.py, deformable_sats/sats/inference/run_dashboard.py, deformable_sats/sats/preprocessing/bin_merge.py.
### deviations from spec
Corrections and additions to the spec's reference list:

1. The spec line "Rectified flow / flow matching: Lipman et al., arXiv:2210.02747" mixes up two papers. 2210.02747 is Flow Matching for Generative Modeling (Lipman et al., ICLR 2023). Rectified flow is a separate paper: Flow Straight and Fast (Liu et al., ICLR 2023, arXiv:2209.03003). The doc lists both.
2. MANO: added arXiv:2201.02610 and DOI 10.1145/3130800.3130883. The original publication is ACM TOG 36(6), SIGGRAPH Asia 2017.
3. Added venues the spec leaves out: ACT RSS 2023; Diffusion Policy RSS 2023 (extended version in IJRR); AnyTeleop RSS 2023; DexCap RSS 2024; Octo RSS 2024; π0 RSS 2025; OpenVLA CoRL 2024; DexUMI CoRL 2025; VTLA journal version in Biomimetic Intelligence and Robotics (2026); OSMO is an arXiv preprint with first author Yin.
4. ActionSense: no arXiv id could be confirmed, so the doc uses the NeurIPS 2022 Datasets & Benchmarks proceedings page and OpenReview.
5. Octo's author line is "Octo Model Team (Ghosh et al.)".

VIHand / VIFNet-S: the paper exists and matches the user-supplied names. Checked: title, ACM MM 2025, DOI, and the abstract. The abstract describes a glove-worn visual-inertial dataset (15 subjects, over 1.4M synchronized RGB-D and IMU frames), a fusion model VIFNet, and an IMU-only distilled student VIFNet-S. The first author listed by the search index is Xinyi Wang; the ACM page could not be opened.

Not checked, and written as such in the doc: whether an arXiv version exists, the number and placement of IMUs, the IMU input format, the output format (MANO θ or joint positions), and whether code/weights are released and under what license.

Limitation: arxiv.org, dl.acm.org, huggingface.co, semanticscholar and the VIHand project page were blocked by the proxy. All checks went through WebSearch results. The doc states this and asks implementers to re-check paper details against the originals before quoting them in docstrings.
### notes for consumers / interface_requests
1. [POSE] robot_skin/pose/imu_model.py (module docstring and ImuHandPoseNet docstring) and robot_skin/pose/README.md describe ImuHandPoseNet as having "the same I/O" as VIFNet-S. That has not been checked; VIFNet-S's IMU count and placement, input format and output format are unknown. Suggested wording: "same role (IMU window → MANO finger pose); VIFNet-S I/O unverified — load_vifnet_s must wrap an input adapter (our 7 sites → VIFNet-S input) and an output conversion (→ finger_pose[15,3] axis-angle)". Also cite it as "VIHand (Wang et al., ACM MM 2025, DOI 10.1145/3746027.3758215)".
2. [POSE] robot_skin/pose/vision_hand.py and pose/README.md mention WiLoR without a citation. Add: Potamias et al., CVPR 2025, arXiv:2409.12259.
3. [VTLA] vtla/heads.py FlowMatchingHead should cite Lipman et al. arXiv:2210.02747 for flow matching and Liu et al. arXiv:2209.03003 for rectified flow as separate papers, and state its own τ convention. vtla/dpo.py should cite Rafailov et al. arXiv:2305.18290. vtla/adapter.py says "Perceiver-style"; cite Jaegle et al. arXiv:2103.03206.
4. [STAGE1] contact/detector.py focal loss: cite Lin et al. arXiv:1708.02002. baseline/temporal.py could mention Yu et al. arXiv:2607.22964 (pose-related artifacts in tactile gloves) as closely related work.
5. [REPR] representation/tokenizer.py fourier_features: cite Tancik et al. arXiv:2006.10739.
6. [DOCS] Other docs should link to docs/REFERENCES.md rather than re-listing citations.
