# pose/ — taxel 위치·법선 제공자와 그 뒤의 기구학 (MANO · URDF · 글러브 IMU · 비전 라벨)

`TaxelPoseProvider` 프로토콜: `pose_at(t) → (positions[N,3], normals[N,3])` (레이아웃 순서, m).
모든 구현은 `transform_taxels(layout, {parent: 4×4})` 로 귀결되며, 달라지는 것은 parent 포즈의 출처뿐이다.

| 모듈 | 내용 | parent 해석 |
|---|---|---|
| `provider.py` | `StaticPoseProvider`, `TransformPoseProvider`, `transform_taxels`, `sample_poses` | 고정 4×4 / 임의 콜러블 |
| `mano.py` | `ManoSkeleton` (MANO 16관절 FK, 세그먼트 프레임, 캡슐), `ManoPoseProvider`, `taxel_poses_from_hand`, `self_touch_from_hand` | MANO 세그먼트 (`common.layouts.MANO_SEGMENTS`) |
| `urdf.py` | 표준 라이브러리만 쓰는 최소 URDF 파서 + 미분 가능한 배치 FK (`URDFModel`) | — |
| `robot_fk.py` | `RobotFKPoseProvider`, `taxel_poses_from_joints` (전처리용 배치) | URDF 링크 |
| `imu_model.py` | IMU 특징(손목 기준 상대 자세), 보정(오프셋·월드 정렬), 합성 IMU, `ImuHandPoseNet`, `hand_pose_loss` | — |
| `glove_imu2mano.py` | `GloveImu2ManoPoseProvider` (IMU→MANO 오프라인 추론→taxel 포즈), `load_vifnet_s`/`finetune_vifnet_s` 스텁 | MANO 세그먼트 |
| `vision_hand.py` | `VisionHandEstimator` 프로토콜, `hand_pose.npz` 입출력, `smooth_hand_labels`, `HaMeREstimator` 스텁 | — |

## 규약

**MANO (mano.py)** — Romero et al., *Embodied Hands* (SIGGRAPH Asia 2017)의 기구학 체인을 스켈레톤만으로 재구현
(메쉬·블렌드쉐이프·라이선스 파일 불필요).

- 관절 순서 `MANO_JOINTS` = wrist, index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3, 부모 `MANO_PARENTS`.
- FK: `G_0 = [R(global_orient) | wrist_pos]`, `G_k = G_parent · [R(θ_k) | J_k − J_parent]` — MANO/SMPL LBS와 동일
  (휴지 자세에서 모든 관절 프레임은 정준 좌표축과 평행).
- 오른손, m, 축-각. `finger_pose` 는 **평평한 템플릿 기준** (MANO `flat_hand_mean=True`). `hands_mean` 기준 라벨은
  더해서 넣을 것. `wrist_pos` 는 관절 0의 월드 위치 (MANO `transl` 이면 `transl + J_0(β)`).
- 기본 정준 축(우리가 아는 MANO 오른손 템플릿 규약): 손가락 **−x**, 요측(검지·엄지) **+z**, 손바닥 **−y** (손등 +y).
  기본 휴지 관절(`DEFAULT_REST_JOINTS`)은 이 축으로 만든 **근사치 성인 오른손**이며 MANO 템플릿 값이 아니다.
  정확도가 필요하면 `ManoSkeleton.from_mano_pkl(MANO_RIGHT.pkl | 변환 .npz, betas)` 사용.
- 손가락별 배열(`tip_pos`, `FINGERS`)은 해부학 순서 thumb, index, middle, ring, pinky (IMU 사이트·`contact.self_touch` 와 동일).
- 세그먼트 프레임 (`segment_transforms`): `wrist` = 관절 0 프레임, `palm` = 관절 0 · 손바닥 정렬 회전 (원점은 손목,
  `palm_offset` 으로 이동 가능), `<finger><1|2|3>` = 해당 지골을 움직이는 관절 프레임 · **뼈 정렬** 회전
  (로컬 **+z = 뼈 방향**, **−y = 손바닥(패드) 쪽**, x = y×z). `glove_template` 좌표(손끝 패드 `[0,−6,10–12] mm`,
  손바닥 패드 z = 30–55 mm)가 이 규약을 따른다. 이 프레임에서 굴곡 = 로컬 +x 양의 회전
  (`ManoSkeleton.flexion_pose(flex, abduction)`).
- 캡슐 (`capsules`): 지골 15개(관절→자식 관절/손끝) + 손바닥 4개(손목→각 MCP). self-touch 기본 제외 규칙
  (`DEFAULT_SELF_TOUCH_EXCLUDE`, 키 = 손가락 그룹 또는 세그먼트 이름): 손바닥 taxel 은 엄지 중수골(thumb1, 무지구 안쪽)을,
  근위지골(`index1` 등) taxel 은 자기 MCP 에서 끝나는 손바닥 캡슐(`palm_index` 등)을 무시한다 (없으면 펼친 손에서도 오탐).

**URDF (urdf.py)** — revolute / continuous / prismatic / fixed, `origin xyz rpy` (`R = Rz·Ry·Rx`), `axis`, `limit`,
`mimic` (기본 추종; `mimic="ignore"` 이면 0 고정 + 경고). floating/planar 는 fixed 로 취급(경고). xacro 는 미리 전개.
`q[...,D]` 순서 = `URDFModel.joint_names` (mimic 아닌 가동 관절, 파일 순서). 드라이버 순서가 다르면
`model.reorder_q(q, names)`. `fk(q)` 는 torch autograd 로 미분 가능 (리타게팅·IK 에 사용).

**IMU (imu_model.py)** — 쿼터니언 wxyz, gyro(rad/s)·acc(m/s², 중력 포함 비력)는 **센서 프레임**.

- 특징 `imu_features(quat[...,W,S,4], gyro, acc, wrist_index)` → `[...,W,S·F]`, 사이트마다
  `[6D(q_wrist⁻¹⊗q_site) | 손목 프레임 gyro | 손목 프레임 acc]` (F = 12). 공통 전역 회전(IMU 월드 기준 차이)에 불변.
  윈도: `imu_windows(feat[T,F], W)` (인과적, 앞쪽 edge-padding).
- 보정 모델 `q_meas = G ⊗ q_segment ⊗ M`. 정적 자세(평평한 손 등, 세그먼트 자세 `ref_rot` 기지)에서
  `calibrate_imu_offsets(quat_calib, ref_rot, world=G)` → `q_off = mean(q_meas)⁻¹ ⊗ G ⊗ q_ref`,
  적용 `apply_imu_offsets(q, q_off, world=G)` = `G⁻¹ ⊗ q ⊗ q_off`, 벡터는 `apply_imu_offsets_to_vectors`.
  `G` 는 `estimate_world_alignment` (손목 IMU 장착 오프셋 = I 가정). 손목 기준 상대 특징에서는 `G` 가 상쇄된다.
  평평한 손 보정의 `ref_rot` 은 `imu_reference_rotations(layout, skeleton)` (스켈레톤 세그먼트 자세 → 보정 후 센서 프레임 =
  세그먼트 프레임, `synthesize_imu` 와 같은 규약):
  `R = imu_reference_rotations(L, sk); G = estimate_world_alignment(q_cal, R, index=wrist); q_off = calibrate_imu_offsets(q_cal, R, world=G)`.
  세션 매니페스트 저장 형식: `manifest.calibration.update(imu_calibration_to_dict(q_off, G, sites))`
  (키 `imu_offsets`, `imu_world`, `imu_sites`), 읽기 `imu_calibration_from_dict(calib, sites)`.
- `synthesize_imu(layout, skeleton, t, global_orient, finger_pose, wrist_pos, gravity)` — MANO 포즈 시퀀스로부터
  이상적인 IMU (합성 데이터·테스트용; 장착 오프셋과 잡음은 호출 측에서 추가).
- `ImuHandPoseNet(in_dim, hidden, n_layers, arch="gru"|"tcn", predict_global)` — IMU 윈도 → 15×6D
  (Zhou et al. 2019 연속 6D 표현). 입력 정규화 버퍼(`set_feature_stats`) 내장, 출력 헤드는 항등 회전(평평한 손)으로
  0-초기화. `hand_pose_loss` = 측지 각도 + `tip_weight`·손끝 거리(스켈레톤 FK) (+ 전역 자세).
- **VIHand VIFNet-S** (사용자 지정 사전학습 백본)는 외부 가중치라 `load_vifnet_s`/`finetune_vifnet_s` 는 문서화된 스텁.
  `ImuHandPoseNet` 이 같은 역할(IMU 윈도 → MANO 손가락 자세)의 사내 베이스라인이며 (VIFNet-S 의 실제 I/O 는 미검증 — 어댑터 필요), `in_dim` + `predict(feat) → {"finger_pose"}` 를 제공하는
  래퍼라면 `GloveImu2ManoPoseProvider` 에 그대로 꽂힌다. (VIFNet-S 서지 정보는 `docs/REFERENCES.md` 확인.)

**비전 라벨 (vision_hand.py)** — HaMeR (Pavlakos et al., CVPR 2024) / WiLoR (Potamias et al., CVPR 2025, arXiv:2409.12259) 를 **오프라인**으로 돌려
`hand_pose.npz` (`t, global_orient[T,3], finger_pose[T,15,3], wrist_pos[T,3], confidence[T]`) 를 만든다
(`HaMeREstimator` docstring 에 절차). `smooth_hand_labels`: 신뢰도 게이트 → 짧은 공백(≤ `max_gap_s`) 쿼터니언
SLERP / 위치 선형 보간 → 유효 구간별 영위상 저역통과(scipy Butterworth, 없으면 중심 이동평균). 긴 공백은 `valid=False`.

## 데이터 흐름

```
D1/D2 raw ─ camera ─▶ HaMeR/WiLoR (offline) ─▶ hand_pose.npz ─ smooth_hand_labels ─┐
          ─ imu.npz ─ calibrate/apply offsets ─ imu_features ─ ImuHandPoseNet ◀── 학습 라벨 (stages/imu_pose)
                                                                     │
glove:  hand pose ─▶ ManoSkeleton ─▶ taxel_poses_from_hand ─▶ taxel_pos/nrm (datasets.build)
                                  └▶ self_touch_from_hand  ─▶ self_touch / contact_label=1
robot:  joint_state q ─▶ URDFModel.fk ─▶ taxel_poses_from_joints ─▶ taxel_pos/nrm
```

포즈는 `baseline/`(입력), `contact/self_touch`(자동 라벨), `representation/`(pose 임베딩), `action/retarget`(URDF FK)에 쓰인다.

**taxel pose 의 프레임**: 전처리(`datasets.build`)는 글러브 taxel pose 를 **손 프레임**(`global_orient = 0`, 손목 =
원점; `meta.preprocessing.taxel_frame = mano_wrist`), 로봇은 URDF 루트 프레임(`urdf_root`)으로 저장한다. baseline·
pretrain·VTLA·온라인 처리기(`control.online.glove_pose_fn`)가 모두 이 규약을 쓴다. 아래 예처럼 `global_orient`·
`wrist_pos` 를 넣으면 월드(카메라) 프레임 자세가 나온다 — 손–물체 근접처럼 월드 좌표가 필요할 때만 쓴다
(`contact.pseudo_label.taxel_world_positions` = `R(hand_global_orient)·p + hand_wrist_pos`).
`GloveImu2ManoPoseProvider` 는 기본(`global_from_imu=True`)이 IMU 월드 프레임이므로, 이 모델들에 넣을 때는
`global_from_imu=False`(그리고 `wrist_pos` 없이)로 만든다.

## 예

```python
from common.layouts import load_layout
from robot_skin.pose import ManoSkeleton, taxel_poses_from_hand, self_touch_from_hand, URDFModel, taxel_poses_from_joints

L, sk = load_layout("glove_template"), ManoSkeleton()
pos, nrm = taxel_poses_from_hand(L, sk, global_orient, finger_pose, wrist_pos)   # [T,N,3]
touch = self_touch_from_hand(L, sk, global_orient, finger_pose, wrist_pos, margin=0.004)  # bool[T,N]

model = URDFModel.from_file("hand.urdf")          # 레이아웃 parent 이름 = URDF 링크 이름
pos, nrm = taxel_poses_from_joints(load_layout("my_robot_hand"), model, model.reorder_q(q, names))
```
