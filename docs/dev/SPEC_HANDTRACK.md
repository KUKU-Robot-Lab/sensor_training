# robot_skin — hand-motion tracking (IMU + vision) implementation contract

User requirement (verbatim intent): "IMU 센서와 비전을 통해서 손동작 추종하는 것들까지 모두 넣어져 있어야 한다.
앞으로 데이터를 통해서 만들 것이니까." → Every piece needed to turn recorded glove sessions (7 IMUs +
cameras) into hand-pose labels, train/run IMU hand tracking, fuse IMU with vision, and track the hand
online must be REAL, working code — no NotImplementedError stubs on the main path. External heavy
models (MediaPipe, HaMeR, WiLoR, VIFNet-S) are integrated through working adapters that run when the
package/weights are installed (guarded imports, clear install instructions), and every adapter's
conversion logic is unit-tested with mocked model outputs. Everything else is pure numpy/torch.

Repo <repo>. Read first: SPEC.md, WAVE1_API.md, WAVE2_API.md (docs/dev/ and docs/dev/as_built/), then the
CURRENT source (it changed after the final review): robot_skin/pose/{mano,imu_model,glove_imu2mano,
vision_hand,urdf,robot_fk}.py, robot_skin/datasets/{build,synthetic,episode}.py, robot_skin/stages/
imu_pose.py, robot_skin/control/*, robot_skin/action/retarget.py, robot_skin/__main__.py,
docs/DATA_FORMAT.md, docs/DATA_ACQUISITION.md. Python: the project env (tests must pass CPU-only; torch, numpy 1.26,
scipy, Pillow, matplotlib; NOT installed: mediapipe, cv2, torchvision, transformers, hamer, wilor).
Tests: CPU, deterministic, fast, no network, optional deps never required (mock them).
Conventions: MANO joint order (wrist, index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3); axis-angle;
quats wxyz; 6D first two columns; metres; right hand (left hand: document mirroring, implement
`mirror_hand_pose` if simple). Do not modify common/ or deformable_sats/. robot_skin must not import sats.

## Groups and ownership

[KPIK] owns robot_skin/pose/keypoints.py, robot_skin/pose/ik.py, robot_skin/eval/hand_metrics.py
(+ eval/__init__ exports), tests test_keypoints.py, test_ik.py, test_hand_metrics.py.
- keypoints.py: 21-keypoint hand convention = MediaPipe/OpenPose order (0 wrist; thumb 1 CMC,2 MCP,
  3 IP,4 TIP; index 5 MCP,6 PIP,7 DIP,8 TIP; middle 9-12; ring 13-16; pinky 17-20). KP_NAMES,
  KP_PARENTS, MANO_TO_KP index map (MANO 16 joints + 5 tips (ManoSkeleton FINGERS order) → 21 kp),
  skeleton_keypoints(fk) -> [...,21,3]; keypoints_from_hand(skeleton, go, fp, wp) -> [T,21,3];
  normalize/align helpers (root-relative, scale by palm size), handedness mirroring.
- ik.py: fit_hand_to_keypoints_3d(kp3d[T,21,3], conf[T,21]|None, skeleton, *, fit_global=True,
  fit_wrist=True, init=None, temporal_weight, prior_weight, iters, solver lm|adam, device) →
  dict(global_orient[T,3], finger_pose[T,15,3], wrist_pos[T,3], residual_mm[T], valid[T]); batched
  over T (vectorized; torch autograd; warm-start frame t from t−1 option); anatomical joint-limit prior
  (flexion ranges, small abduction/twist for PIP/DIP, thumb ranges) as soft penalty — document ranges
  as approximate; robust loss (Huber) + confidence weighting; optional bone-length/scale fitting:
  fit_skeleton_scale(kp3d over calibration frames) → per-bone lengths/global scale → ManoSkeleton
  with fitted rest joints (subject calibration). fit_hand_to_keypoints_2d(kp2d[V,T,21,2], conf, cameras
  (list of Camera from vision.cameras), skeleton, ...) multi-view reprojection fitting (needs wrist
  depth init from triangulation or given). Tests: synthetic poses → FK keypoints (+noise, missing
  kps) → IK recovers joint positions (MPJPE < few mm) and pose; temporal smoothing reduces jitter;
  prior keeps poses plausible under heavy noise; 2D multi-view fitting recovers 3D.
- eval/hand_metrics.py: mpjpe, pa_mpjpe (Procrustes with scale), pck(threshold), auc_pck,
  joint_angle_error (geodesic per joint, deg), jitter (mean |acceleration|), per-finger breakdown.

[CAM] owns robot_skin/vision/cameras.py (+ vision/__init__ exports additions), robot_skin/vision/
calibration.py, tests test_cameras.py.
- cameras.py: @dataclass Camera(name, K[3,3], dist (k1,k2,p1,p2,k3), width, height, T_world_cam
  [4,4] (camera pose in world/rig frame), model "pinhole"); project(points_world[...,3]) → (uv[...,2],
  depth, in_front mask) with distortion; undistort_points(uv) (iterative, pure numpy);
  pixel_ray(uv); triangulate_dlt(uv_views[V,...,2], conf[V,...], cameras) weighted multi-view DLT +
  optional nonlinear refinement (Gauss-Newton on reprojection); reprojection_error; load/save
  camera YAML/JSON (OpenCV-style keys: camera_matrix, dist_coeffs, image_size, T_world_cam) and a
  rig file (cameras.yaml per session: manifest.calibration["cameras"] or <session>/cameras.yaml).
  Head-mounted ego camera moves: support per-frame extrinsics T_world_cam[F,4,4] (optional
  camera_<name>/poses.npy) — document; if absent, hand pose is expressed in that camera frame.
- calibration.py: checkerboard/charuco intrinsic calibration + stereo extrinsics via OpenCV when cv2 is
  installed (guarded, clear error otherwise); a pure-numpy Zhang's-method fallback for planar
  checkerboard corners already detected (homographies → K, then LM refine of K, dist (k1,k2), poses) —
  test with synthetic corners projected by a known camera. CLI entry `python -m robot_skin.vision.calibration`.
- Tests: projection/undistortion roundtrip, triangulation accuracy with noise and a missing view,
  YAML roundtrip, Zhang calibration recovers synthetic K within tolerance.

[VBACK] owns robot_skin/pose/vision_hand.py (replace HaMeR stub with real adapters; keep existing
functions/APIs: VisionHandEstimator protocol, save/load_hand_labels, smooth_hand_labels),
robot_skin/pose/vision_backends.py, robot_skin/pose/extract.py, robot_skin/pose/viz.py,
robot_skin/configs/stages/handpose.yaml, tests test_vision_backends.py, test_extract.py.
- vision_backends.py: estimators returning a common per-frame result (kp2d[21,2], kp2d_conf[21],
  kp3d_cam[21,3] or None, mano {global_orient, finger_pose, betas?, cam_t?} or None, handedness,
  bbox, score):
  * MediaPipeHandEstimator (mediapipe Tasks/solutions API, guarded): 2D landmarks (pixels) + world
    landmarks (metric, hand-centred), handedness → pick right hand; min detection/tracking
    confidence params; runs on CPU/GPU delegate.
  * HaMeRAdapter / WiLoRAdapter (guarded imports of their repos' python packages; takes model config
    + checkpoint path; per-frame hand detection (their detector or given bboxes) → MANO params in
    their camera convention (global_orient/hand_pose as rotation matrices, betas, pred_cam→cam_t with
    focal length) → convert to our axis-angle + wrist position in camera frame (document the
    conversion; handle their flat-hand-mean convention vs ours; right-hand flip handling for left
    detections). Unit-test the conversion functions with mocked outputs (rotation matrices from our
    own FK) so conversion is verified without the packages.
  * KeypointFileEstimator: reads precomputed keypoints from npz/json (any external tool) — the
    universal fallback.
  * build_estimator(cfg) factory.
- extract.py: extract_session_hand_pose(session_dir, cfg) → reads camera frames (frames.npy or jpgs),
  timestamps, cameras.yaml; runs the estimator per camera (batched, device), selects the tracked hand
  (handedness + temporal continuity of bbox), multi-view: triangulate 2D keypoints (vision.cameras)
  when ≥2 calibrated views, else use the backend's 3D (metric world landmarks scaled/placed with
  camera depth from MANO cam_t or wrist depth heuristic — document limitations) → IK
  (pose.ik.fit_hand_to_keypoints_3d/2d, optional subject skeleton from calibration frames) or direct
  MANO from HaMeR/WiLoR (optionally refined by IK) → confidence per frame → smooth_hand_labels →
  write hand_pose.npz (save_hand_labels) + keypoints_<cam>.npz sidecar (kp2d, conf, t) +
  hand_pose_meta.json (backend, versions, params, per-frame residuals) and register the stream in the
  manifest (streams["hand_pose"]) so datasets.build picks it up (CHECK how build reads hand_pose after
  the final-review fix and match it). extract_dataset(raw_root, cfg, skip_existing) batch. CLI
  `python -m robot_skin.pose.extract --session ... | --raw-root ... --backend mediapipe|hamer|wilor|keypoints
  --cameras ego third --device cuda`.
- viz.py: draw_hand_2d(frame, kp2d, conf) (numpy/Pillow drawing, no cv2), plot_hand_3d(matplotlib,
  optional), render_overlay_video(session, out_dir, every_n) writing PNGs for QC; used by extract
  `--viz`.
- Tests: KeypointFileEstimator path end-to-end on a synthetic session whose keypoint files are
  generated by projecting ground-truth skeleton keypoints through synthetic cameras (coordinate with
  [SYNTHKP]: use datasets.synthetic option if present, else generate in-test) → hand_pose.npz →
  datasets.build episode has valid hand pose close to GT; HaMeR/WiLoR/MediaPipe conversion functions
  with mocked outputs; handedness selection; missing detections → low confidence gaps filled by
  smoothing.

[VIFUSE] owns robot_skin/pose/fusion.py, robot_skin/pose/tracking.py (online), robot_skin/pose/
external_model.py, robot_skin/pose/glove_imu2mano.py (replace load_vifnet_s/finetune_vifnet_s stubs
with working generic-adapter implementations), robot_skin/stages/imu_pose.py (extend: model.type
in_house|external, eval with eval.hand_metrics, label source vision|fused, export online bundle),
robot_skin/configs/stages/imu_pose.yaml, tests test_fusion.py, test_tracking.py, test_external_model.py
(and update test_stage1.py imu_pose parts if needed; keep them green).
- fusion.py: VisualInertialHandFusion (offline): inputs per-frame IMU-derived hand pose (from
  ImuHandPoseNet/external model, 200 Hz), calibrated IMU segment orientations (for global orient),
  vision hand pose with confidence (30 Hz, gaps), → fused hand pose [T] at 200 Hz: rotation-space
  complementary filter / error-state Kalman on SO(3) per joint (IMU = high-rate prediction with drift,
  vision = low-rate absolute correction weighted by confidence), forward pass (causal) + optional RTS-
  style backward smoothing (offline labels); wrist position from vision (+ IMU acceleration-aided
  interpolation optional) with confidence decay in gaps; outputs confidence. Also yaw-drift/world-
  alignment re-estimation between IMU world and camera frame (estimate R_cam_imuworld from overlapping
  confident frames by rotation averaging). Ground with VIST (Lee et al., Sci. Robot. 2021) and VIHand.
- tracking.py (online): CausalImuPoseStream(model bundle: ImuHandPoseNet or external model + feature
  stats + window + calibration offsets) push(t, quat[S,4], gyro, acc) → finger_pose at ≤200 Hz with
  exactly the offline windowing (test allclose to predict_finger_pose_sequence);
  OnlineHandTracker(imu_stream, vision_estimator|None (async thread/queue, drops stale frames),
  fusion params) → HandState(t, global_orient, finger_pose, wrist_pos, confidence, source flags);
  latency-aware (vision timestamp older than now → correct past state then re-propagate with buffered
  IMU (short replay buffer)). Streams from acquisition sources (FakeImuSource / real ImuSource) via an
  adapter. Also used by control/online glove path (q_source hand_pose_imu) — provide the API CTRL
  needs.
- external_model.py: ExternalImuPoseModel(nn.Module-compatible predictor) wrapping a TorchScript file
  or `module.path:factory` + state_dict; config-driven input adapter (site order map from our 7 sites
  to the model's IMU slots, input representation: quat|rotmat|rot6d|acc|gyro|ori+acc, frame:
  relative-to-wrist|world, normalization stats) and output adapter (aa|rot6d|rotmat|quat, joint order
  map → MANO 15, flat-hand-mean offset) → same predict() contract as ImuHandPoseNet. load_vifnet_s(
  checkpoint, config) now implemented via this (document that VIFNet-S I/O must be filled from the
  released code — provide a template YAML with TODO fields and validation that errors clearly when a
  field is missing); finetune_vifnet_s → runs stages.imu_pose with model.type external (fine-tune head
  or all layers). Tests with a tiny fake external model saved as TorchScript and as module:factory.
- stages/imu_pose.py: label_source: vision (hand_pose from extract) | fused (after fusion pass,
  iterative refinement option) ; metrics via eval.hand_metrics (MPJPE/PA-MPJPE mm using ManoSkeleton
  keypoints, angle error, jitter) ; writes derived hand_finger_pose_imu and optional fused pose
  derived key (add D_HAND_POSE_FUSED constant request to episode.py owner — you may add it yourself to
  episode.py derived keys as a small, additive change) ; exports imu_tracker_bundle.pt consumed by
  CausalImuPoseStream.

[SYNTHKP] — part of [KPIK]'s ownership:
robot_skin/datasets/synthetic.py additive option `vision_keypoints=True` writing <session>/cameras.yaml
(synthetic intrinsics/extrinsics for each camera; ego camera may be static in synthetic) and
keypoints_<cam>.npz (2D projected ground-truth 21 keypoints with noise, dropouts, confidence) so the
vision path can be tested end-to-end without real images. Must not change outputs when the option is
off (bit-identical), update test_synthetic accordingly.

[TELEOP+INTEG+DOCS] owns robot_skin/control/teleop.py, robot_skin/__main__.py (add subcommands:
`handpose extract`, `handpose calibrate-subject`, `track` (offline fusion over a raw/processed root),
`teleop`), robot_skin/configs/stages/teleop.yaml, docs/HAND_TRACKING.md (Korean: full pipeline from
recording to labels to IMU model to fusion to online tracking and teleop; backend install for
MediaPipe / HaMeR / WiLoR / VIFNet-S; camera calibration procedure; subject skeleton calibration;
QC with viz; expected accuracy caveats), docs/DATA_ACQUISITION.md + DATA_FORMAT.md + ARCHITECTURE.md
+ README updates, robot_skin/pose/README.md, tests test_teleop.py, test_e2e_handtrack.py.
- teleop.py: TeleopRunner(tracker: OnlineHandTracker, retargeter: FingertipRetargeter (estimate_scale),
  robot: RobotHandInterface, safety: SafetyFilter, logger) — live human hand → robot joint targets at
  control rate, with tactile safety, records robot raw sessions (joint_state + pressure) + the human
  hand pose stream → reusable as robot training data (D3 teleop). FakeRobotHand + fake IMU source in
  tests.
- e2e: synthetic glove session with vision_keypoints → handpose extract (keypoints backend) →
  build → imu_pose stage (label_source vision) → fusion → CausalImuPoseStream online replay matches
  offline → teleop on FakeRobotHand for 1 s.

## References to cite (verify new ones through web search before citing; list in docs/REFERENCES.md)
MANO (Romero et al. 2017), HaMeR (arXiv:2312.05251), WiLoR (arXiv:2409.12259), VIST (Lee et al.,
Science Robotics 2021, DOI 10.1126/scirobotics.abe1315), VIHand (ACM MM 2025, DOI
10.1145/3746027.3758215), MediaPipe Hands (Zhang et al. 2020 — VERIFY arXiv id before citing),
Zhang's camera calibration (Zhang 2000, TPAMI — verify), AnyTeleop / DexPilot (retargeting), DexCap.
