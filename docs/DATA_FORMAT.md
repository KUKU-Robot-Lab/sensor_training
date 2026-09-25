# 데이터 포맷 — raw 세션 → processed Episode

robot_skin 의 데이터는 두 단계로 저장된다.

1. **raw 세션**: 수집기(`robot_skin.acquisition`) 또는 합성기(`robot_skin.datasets.synthetic`)가 쓰는
   스트림별 원본 파일. 스트림마다 자기 시계와 레이트를 가진다.
2. **processed Episode**: 전처리(`python -m robot_skin.datasets.build`)가 모든 스트림을 200 Hz 마스터
   시계에 맞추고, 라벨을 붙여 만든 결과. **모든 학습 stage 는 Episode 만 읽는다.**

운영 절차는 `docs/DATA_ACQUISITION.md`, 수집 코드는 `robot_skin/acquisition/README.md`, 전처리·데이터셋
코드는 `robot_skin/datasets/README.md` 를 본다. 논문 근거는 `docs/REFERENCES.md` 에 모아 두었다.

## 0. 공통 규약

| 항목 | 규약 |
|---|---|
| 시간 | 초(s), float64. raw 는 **세션 시계**(`Recorder.start` 이후 경과, 3-탭 싱크 후 pressure 시계로 보정됨), processed `t` 도 같은 세션 시계 위의 `1/hz` 격자 |
| 길이 | m (layout YAML 은 `units: mm` 가능 → 로드 시 m 로 변환) |
| 각도 | rad. 회전은 axis-angle(rad, `‖aa‖` = 각도) 또는 쿼터니언 |
| 쿼터니언 | **wxyz**, 단위 노름. processed 에서는 시간 방향 연속성(인접 내적 > 0)이 보장됨 |
| 6D 회전 | 회전행렬의 **앞 두 열** (Zhou et al., CVPR 2019) — `geometry.rotations.matrix_to_6d` |
| ΔS | `ΔS[%] = (raw − baseline) / baseline · 100` (SATS 규약, `common.signal.relative_change`). mk555 기압 taxel 은 눌리면 raw 가 **감소** → **press = 음수 ΔS**, dropout 은 −100 % 방향. 누름 양수 뷰는 `common.signal.press_intensity(ΔS) = −ΔS`, 임계값은 누름 양수로 표기 |
| ADC | raw count, 레일 `ADC_MIN = 0`, `ADC_MAX = 2²⁴ − 1` (`common.signal`) |
| taxel 순서 | raw 는 **채널 순서**(열 = ADC 채널), processed 는 **layout 순서**(`layout.taxels` 순서, `layout.by_channel(raw)`) |
| MANO 관절 순서 | `wrist, index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3` (16개; 손가락 15개 = 1..15). 손끝·손가락별 배열은 `FINGERS = (thumb, index, middle, ring, pinky)` 순서 |
| MANO 정준 좌표 | 손가락 −x, 요측(엄지 쪽) +z, 손바닥 −y, 손목(joint 0) 원점 (`pose.mano`, 근사 오른손) |

## 1. raw 세션

```
robot_skin/data/raw/<dataset>/<subject>/<session_id>/
  session.json          SessionManifest v2
  pressure.npz          촉각 (필수)
  imu.npz               장갑 IMU               (glove)
  joint_state.npz       로봇 관절 상태          (robot)
  hand_pose.npz         MANO 손 자세 라벨       (glove; 오프라인 비전 또는 --fake)
  object_pose.npz       물체 자세              (선택, D2)
  camera_<name>/        timestamps.npy + frames.npy | 000000.jpg …
  events.jsonl          phase / marker / instruction / success 이벤트
  qc.json               수집 QC 보고서          (acquisition.qc)
  robot.urdf            (합성 로봇 세션; 실제 로봇은 manifest.meta.urdf 가 가리키는 파일)
  layout.yaml           (합성 세션에 Layout 객체를 넘긴 경우)
  gt_synthetic.npz      합성기 정답 (스트림 아님, synthetic 전용)
```

### 1.1 `session.json` (`acquisition.manifest.SessionManifest`, schema v2)

| 키 | 타입 | 의미 |
|---|---|---|
| `kind` | `glove` \| `robot` \| `bench` | 세션 종류 |
| `dataset` | `motion` \| `task` \| `other` | D1 / D2 |
| `layout` | str | 내장 layout 이름(`glove_template`, `robot_hand_template`, `sats_4x4`) 또는 YAML 경로. 상대 경로는 **세션 디렉터리 기준**으로 먼저 찾는다 (세션 이동 후에도 동작) |
| `streams` | `{name: {file, rate_hz, fields, clock, method}}` | `file` 은 세션 디렉터리 기준 상대 경로. `rate_hz` 는 명목값(파일 속 타임스탬프가 기준). `method`: `linear` \| `zoh` (카메라 = `zoh`). 카메라 스트림 이름은 `camera_<name>` |
| `session_id`, `subject`, `created_utc` | str | 피험자는 가명 ID 만 |
| `master_hz` | float | 전처리 마스터 시계 (기본 200) |
| `segments` | `[{t0, t1, label}]` | 라벨 구간 (§1.9) |
| `baseline` | `[C]` \| null | 채널별 raw baseline (알고 있을 때; 전처리 fallback) |
| `task` | dict \| null | D2: `task_id`, `instruction`, `object`, `success` (+ 수집기 추가 키 `target`, `repetition`, `template`, `template_index`, `instruction_source`, `slots`, `success_criteria`, `grasp`, `manipulate`) |
| `calibration` | dict | §1.8 |
| `meta` | dict | 자유 형식: `protocol`, `seed`, `fake`, `urdf`(로봇: URDF 경로, 세션 디렉터리 기준), `synthetic{…}`(합성기: `camera_offset_s`, `object_frame`, `layout_file` …) |
| `notes`, `schema_version` | | |

### 1.2 `pressure.npz`

| 키 | shape / dtype | 단위·규약 |
|---|---|---|
| `t` | `[T]` float64 | s, 세션 시계 |
| `raw` | `[T, C]` float64 | ADC count, **채널 순서**. layout taxel i 의 값은 열 `layout.channels[i]`. 레일 값(0 / 2²⁴−1)은 포화·dropout |

mk555 `.bin` 원본은 `deformable_sats/sats/preprocessing/bin_merge.py` 가 정본 파서다(복사 금지). npz 가 아닌
세션은 전처리 설정 `pressure.loader: "pkg.module:function"` 으로 `(session_dir, manifest) → (t, raw)` 로더를
주입한다.

### 1.3 `imu.npz` (glove)

| 키 | shape / dtype | 단위·규약 |
|---|---|---|
| `t` | `[T]` float64 | s |
| `quat` | `[T, S, 4]` float32 | wxyz. **센서 프레임의 자세를 IMU-world 에서 본 것**: `q_meas = G ⊗ q_segment ⊗ M` (`G` IMU-world ← 모델 world, `M` 장착 오프셋). 장치 관례상 `w ≥ 0` 반구로 뒤집혀 있을 수 있음 → 전처리에서 연속성 보정 |
| `gyro` | `[T, S, 3]` float32 | rad/s, **각 센서 프레임** (월드 프레임 벡터를 내는 장치는 전처리 `imu.vec_frame: world`, §1.8) |
| `acc` | `[T, S, 3]` float32 | m/s², **각 센서 프레임**, 비력(specific force) = **중력 포함** (정지 시 크기 ≈ 9.81, 위쪽 방향) |
| `sites` | `[S]` str | IMU 사이트 이름 (layout `imu_sites`: `wrist, palm, thumb, index, middle, ring, pinky`) |

### 1.4 `joint_state.npz` (robot)

`t [T]` float64, `q [T, D]` (rad / m), 선택 `qd [T, D]`, `tau [T, D]`, `names [D]` str (드라이버 순서). 전처리는
URDF 의 actuated 관절 순서(`URDFModel.joint_names`)로 재배열한다(`URDFModel.reorder_q`; URDF 에 없는 이름은
버림, 빠진 관절은 0 — `preprocessing.notes` 와 `preprocessing.zero_filled_joints` 에 기록; 이름이 **하나도** 맞지
않으면(네임스페이스 접두사 `rh_…` 등) q 가 전부 0 이 되므로 **에러**). URDF 는 설정 `robot.urdf` (세션 디렉터리 기준 → 그 파일 이름의 세션 안 사본 →
주어진 그대로 = 절대/CWD 기준; 못 찾으면 **에러**) 또는 `manifest.meta.urdf` (세션 디렉터리 기준; 못 찾으면
경고 + `q` 는 드라이버 순서, taxel 자세 정적 — 이런 episode 는 URDF 순서 episode 와 한 데이터셋에 섞을 수 없다:
`meta.joint_names` 가 다르면 `datasets.motion` 이 거부).

### 1.5 `hand_pose.npz` (glove 손 자세 라벨)

`pose.vision_hand.save_hand_labels` / `load_hand_labels` 로 읽고 쓴다.

| 키 | shape / dtype | 규약 |
|---|---|---|
| `t` | `[T]` float64 | s, 세션 시계 |
| `global_orient` | `[T, 3]` float32 | MANO 루트(손목) 회전 axis-angle, 추정기의 카메라/world 프레임 |
| `finger_pose` | `[T, 15, 3]` float32 | 손가락 관절 axis-angle, MANO 순서, **평평한 MANO 템플릿 기준** (`flat_hand_mean=True`; 추정기가 `hands_mean` 을 쓰면 더해서 저장) |
| `wrist_pos` | `[T, 3]` float32 | m, joint 0 의 world 위치 (MANO 출력이면 `transl + J_0(β)`) |
| `confidence` | `[T]` float32 | ∈ [0, 1]; 전처리 게이트 `hand_pose.min_conf` |

실제 세션은 기록 후 HaMeR (Pavlakos et al., CVPR 2024) / WiLoR 를 카메라 영상에 오프라인으로 돌려 만든다.
`save_hand_labels(<session_dir>, …)` 는 파일을 쓰고 `session.json` 에 `hand_pose` 스트림을 등록한다
(`register_hand_labels`). 전처리는 등록되지 않은 세션 디렉터리의 `hand_pose.npz` / `object_pose.npz` 도 읽는다
(`build.OFFLINE_STREAMS`, `preprocessing.notes` 에 기록). 단 레코더 세션의 기록 시작(`meta.recorder.started_utc`)보다
오래된 파일은 `overwrite` 재기록이 남긴 이전 take 의 라벨로 보고 쓰지 않는다(notes).

### 1.6 `object_pose.npz` (선택)

`t [T]`, `pos [T, 3]` m, `quat [T, 4]` wxyz. 프레임은 world (합성 로봇 세션은 손 base 프레임:
`meta.synthetic.object_frame = "hand_base"`; 전처리가 `meta.preprocessing.object_frame` 에 옮긴다).

### 1.7 `camera_<name>/`

`timestamps.npy [F]` float64 s + `frames.npy uint8 [F, H, W, 3]` (RGB) **또는** `000000.jpg …`. 싱크 보정 전
원본 시각은 `timestamps_host.npy` 에 남는다(전처리는 무시). 합성 세션의 카메라 시각에는 작은 상수 오프셋이
있다(`meta.synthetic.camera_offset_s`, 보정되지 않은 채로 둠).

### 1.8 `calibration`

| 키 | 값 | 쓰는 곳 / 읽는 곳 |
|---|---|---|
| `imu_offsets` | `[[w,x,y,z]] × S` | `q_off = M⁻¹` (IMU 사이트 순서) — `pose.imu_model.imu_calibration_to_dict` 로 쓰고 `imu_calibration_from_dict(calib, sites)` 로 읽는다 |
| `imu_world` | `[w,x,y,z]` | `G` (선택) |
| `imu_sites` | `[S]` | offsets 의 사이트 순서 (읽을 때 `sites` 로 재정렬·검증). 없으면 전처리는 offsets 를 `imu.npz` 의 `sites` 순서로 읽는다(개수가 같을 때) |
| `imu_calibration_quality` | dict | 수집기 평손 보정 품질 (`quat_spread_deg`, `gyro_rms`, `ok`, phase …) |
| `sync` | dict | 3-탭 싱크 보고서 (`applied`, `reference`, `streams{…}` …). `applied` 이면 모든 스트림 `t`/`timestamps.npy` 가 이미 pressure 시계로 보정됨 (`t_host`, `timestamps_host.npy` 에 원본 보존) |

보정 적용: `q_segment = G⁻¹ ⊗ q_meas ⊗ q_off` (`apply_imu_offsets`), 벡터 `v_segment = R_offᵀ · v_meas`
(`apply_imu_offsets_to_vectors`). 장치가 gyro/acc 를 IMU-world 프레임으로 낸다면 전처리 `imu.vec_frame: world` —
벡터에는 장착 오프셋이 아니라 `v_model = G⁻¹ · v_meas` 가 적용되고(`apply_imu_offsets_to_vectors(..., vec_frame="world",
world=G)`), IMU 특징은 `imu_features(vec_frame="world")` 로 계산해야 한다(`imu_pose` 의 `features.vec_frame`; 에피소드에
기록된 프레임과 다르면 `episode_imu_features` 가 오류). 평손 보정 절차는 `imu_reference_rotations` + `estimate_world_alignment` +
`calibrate_imu_offsets` (`acquisition.calibration.calibrate_session_imu`).

### 1.9 `events.jsonl` 와 segments

한 줄에 JSON 하나: `{"t": s, "type": …, "name": str, "value": any}`. **`type` 으로 식별**한다 (합성기는
instruction/success 의 `name` 에 task_id 를 쓰는 옛 관례가 있을 수 있음).

| type | name | value |
|---|---|---|
| `phase_start` / `phase_end` | 단계 id (`baseline_start`, `imu_calibration`, `sync_start`, `pinch_index`, `reach`, `grasp` …) | start 의 value: `contact` ∈ `none\|self\|object\|any`, `labels` (segment 라벨), 수집기는 `kind` (`static\|calibration\|sync\|motion\|self_touch\|task_phase`), `block`, `speed`, `motion`, `finger`/`axis`/`grasp` (for_each 값), `nominal_s` 추가 |
| `marker` | 자유 | 자유 |
| `instruction` | `instruction` | 지시문 문자열 |
| `success` | `success` | bool \| null |

**`events.jsonl` 이 phase segment 의 기준이다** (`acquisition.recorder.session_segments`): 레코더 세션
(`meta.recorder` 있음)은 전처리·후처리(`python -m robot_skin.acquisition.session`)가 segments 를 이벤트에서 다시
만들고, 이벤트에서 나오지 않는 segment(`Recorder.add_segment` — D2 `task` — 또는 기록 전에 manifest 에 있던 것)는
`meta.recorder.explicit_segments` 와, 어떤 phase 도 만들지 않는 라벨의 manifest segment 에서 가져온다(그 키가 없는
옛 기록도 이렇게 `task` 를 유지한다; phase 라벨(`no_contact` 등)의 구간을 `session.json` 에서 직접 고치면 이벤트 값으로
되돌아간다 — 이벤트를 고친다). 이벤트를 손으로 고치면 manifest 와 달라진 점이 `preprocessing.notes` 에 남는다. 합성 세션처럼 레코더가 쓰지
않은 manifest 의 segments 는 쓰인 그대로 쓴다(비어 있으면 이벤트에서).

segment 라벨: `no_contact` (baseline 학습 + ΔS 기준; 첫 구간은 휴지/평손 자세), `self_touch`, `calibration`
(평손 IMU 보정), `sync` (탭 = 실제 접촉, no_contact 아님), `task` (D2 reach…retreat 전체). 단계의 `contact`
기대값 `none → [no_contact]`, `self → [self_touch]` 가 기본 라벨이다. 합성 세션의 `no_contact` 구간은
기하 접촉 주변이 잘려 있어 항상 참이다.

### 1.10 `gt_synthetic.npz` (합성 전용 sidecar)

pressure 타임스탬프 위, layout 순서의 정답: `artefact_pct` (무접촉 움직임 artefact, 참 baseline 기준 ΔS %),
`press_pct`, `drift_pct`, `noise_pct`, `delta_true_pct`, bool `contact` / `self_touch` / `object_contact` /
`saturated`, `penetration_m`, `baseline_raw [N]`, `baseline_raw_all [C]`, `channels`, `taxel_pos`, 관절각
`joint_angle` + artefact 파라미터, 참 자세(`hand_*` 또는 `q`, `object_pos`). `datasets.synthetic.load_ground_truth`.
정답 `taxel_pos` 는 `object_pos` 와 같은 프레임이다 — glove 는 **world** (참 `global_orient`·`wrist_pos` 적용),
robot 은 손 base(URDF 루트) (`meta.synthetic.gt_taxel_frame` = `world` | `hand_base`). processed Episode 의
`taxel_pos`(손 프레임, §2)와 비교하려면 glove 는 `R(hand_global_orient)ᵀ · (taxel_pos − hand_wrist_pos)`.
`meta.synthetic.glove_seed`: 장갑(스킨) 물리 파라미터(artefact 이득·부호·지연, 누름 모델, baseline, 잡음 수준)를
뽑은 시드 — 같은 값의 세션들은 **같은 장갑**으로 기록된 것이다 (`generate_dataset` 기본 `shared_glove=True`);
`null` 이면 세션마다 다른 스킨.
전처리는 이 파일을 스트림으로 읽지 않고, 있으면 ΔS 를 대조하고(§2.6) 선택적으로 `gt_*` 배열로 옮긴다.

### 1.11 배포 세션 (`control.runner.DeploymentLogger`)

로봇 배포(`stages.deploy`, `control.PolicyRunner`)도 같은 raw 형식으로 기록된다 (acquisition `Recorder` 기반):
`kind: robot`, `dataset: other` (기본값 — VTLA 의 `task` 학습에 섞이지 않게), `layout: layout.yaml` (세션 기준
상대 경로, 세션 안에 사본), `meta.urdf: robot.urdf`, `meta.deployment` (번들 경로, 제어·정책 주기, 지표, 안전
요약). phase: `baseline`, `calibration` (둘 다 `contact: none` → `no_contact` segment), `rollout` (`contact: any`
→ `task` segment); instruction / success 이벤트와 안전 marker. 스트림: `pressure.npz`(raw 채널 순서),
`joint_state.npz` (q, 드라이버가 재는 경우에만 qd, names; 새 샘플 시각마다 1행), `camera_<name>/`.
`deploy_log.npz` 는 틱 단위 sidecar(목표·명령·레벨·정책 틱)이지 스트림이 아니다 — `gt_synthetic.npz` 처럼
`datasets.build` 가 무시하며, 배포 세션은 일반 로봇 세션처럼 재전처리된다 (성공/실패 rollout 재사용).

## 2. processed Episode

```
robot_skin/data/processed/<dataset>/<episode_id>/      (episode_id = session_id)
  episode.json            EpisodeMeta
  layout.yaml             세션 layout 사본 (mm) — episode 가 자기완결적이 되도록
  arrays/<key>.npy        시간축 배열 [T, …] (np.load(mmap_mode="r"))
  static/<key>.npy        시불변 배열
  derived/<key>.npy       이후 stage 출력 (baseline_pred, residual_z, contact_prob …)
  camera_<name>/          raw 카메라 디렉터리의 symlink(기본, 절대 경로 → raw 를 옮기면 끊김) 또는 사본
```

읽기: `datasets.Episode.load(root, mmap=True)`, 목록: `datasets.list_episodes(processed_root, dataset)`,
layout: `datasets.build.load_episode_layout(ep)`.

### 2.1 `episode.json` (`EpisodeMeta`)

| 키 | 의미 |
|---|---|
| `episode_id`, `dataset`, `kind`, `subject` | raw manifest 에서 |
| `layout` | 내장 이름, 또는 사용자 YAML 이면 episode 안 `layout.yaml` 의 절대 경로 |
| `n_taxels`, `hz` | N, 마스터 시계 (200) |
| `joint_names` | `q` 열 이름: robot = URDF actuated 순서, glove = `index1_x, index1_y, index1_z, …, thumb3_z` (45) |
| `imu_sites` | IMU 배열의 사이트 순서 (layout 순서) |
| `cameras` | 카메라 이름 (`cam_<name>_idx` 가 있는 것) |
| `phases` | `[{name, t0, t1, contact?, labels?, kind?, speed?, closed?}]` — events 의 단계 (없으면 segments) |
| `phase_names` | `phase_id` 의 어휘 (첫 등장 순서) |
| `task` | D2: manifest.task 전체 (+ 없으면 events 의 instruction/success). `meta.instruction` 속성 |
| `source_session` | raw 세션 절대 경로 |
| `preprocessing` | §2.6 |
| `created_utc`, `format_version` | |

### 2.2 시간축 배열 (`arrays/`, 모두 `[T, …]`, `episode.py` 의 `K_*`)

| 키 | shape / dtype | 단위·규약 |
|---|---|---|
| `t` | `[T]` float64 | s, 세션 시계, 균일 `1/hz` 격자 (값이 `1/hz` 의 배수) |
| `pressure_raw` | `[T, N]` float64 | ADC count, layout 순서 (선형 보간; 저역통과 설정 시 필터된 값) |
| `delta_pct` | `[T, N]` float32 | ΔS %, press → 음수 |
| `saturated` | `[T, N]` bool | 앞뒤 raw 샘플 중 하나라도 레일 근처 또는 non-finite (로더의 결손 샘플; 보간으로 메워 ΔS 는 항상 유한), pressure 샘플 갭 안(§2.3), 또는 `\|ΔS\| ≥ max_abs_pct` (90), 또는 죽은 채널 |
| `taxel_pos` | `[T, N, 3]` float32 | m, **손 / 로봇 base 프레임** (`episode.py` 계약). glove: MANO 손목 관절(joint 0) 프레임 — `global_orient`·`wrist_pos` 를 뺀, 손가락 자세만 반영한 위치 (손 자세 없으면 평손 rest 위치); robot: URDF 루트 링크 프레임; bench/정보 없음: layout 위치 그대로. 어느 쪽인지는 `preprocessing.taxel_frame` (`mano_wrist` \| `urdf_root` \| `layout`). glove world 위치가 필요하면 `R(hand_global_orient) · taxel_pos + hand_wrist_pos` |
| `taxel_nrm` | `[T, N, 3]` float32 | 단위 법선, 같은 프레임 |
| `q` | `[T, D]` float32 | robot: 관절 상태 (URDF 순서); glove: 평활된 `hand_finger_pose` 45-D (rad). glove 의 `hand_pose_valid` 가 아닌 프레임은 마지막 라벨을 붙잡아 둔 값, robot 의 `joint_state_valid` 가 아닌 프레임은 끝값 유지/갭 보간 → 측정값 마스크는 `datasets.stats.q_valid_mask` |
| `qd` | `[T, D]` float32 | 단위/s. Savitzky–Golay 미분, 기본 `savgol_causal` (과거 50 ms 만 사용 → 온라인에서 `datasets.build.joint_velocity` 로 동일 재현; `savgol` = 중앙, 오프라인 전용). glove 라벨 결손 뒤 붙잡힌 값이 참 자세로 점프하는 곳은 미분 필터 폭(`build.qd_support`) 만큼 속도 스파이크가 생기므로, 측정값 마스크는 `datasets.stats.qd_valid_mask` (= `q_valid_mask` 를 그 폭만큼 침식) |
| `imu_quat` | `[T, S, 4]` float32 | wxyz, 보정 후 **세그먼트 프레임** 자세 (`G⁻¹ ⊗ q ⊗ q_off`), 연속성 보정·재정규화. 보정이 없으면 raw 센서 프레임 (`preprocessing.imu.calibrated = false`) |
| `imu_gyro` | `[T, S, 3]` float32 | rad/s, 보정 후 세그먼트 프레임 (없으면 센서 프레임; `preprocessing.imu.vec_frame = world` 이면 보정 후 모델 world, 없으면 IMU world) |
| `imu_acc` | `[T, S, 3]` float32 | m/s², 중력 포함 비력, 같은 프레임 |
| `imu_valid` | `[T]` bool | IMU 가 실제로 잰 프레임: IMU 스트림 자체 구간 안(`clock.span: pressure` 면 밖은 끝값 유지) + 샘플 갭(§2.3) 밖. `ImuPoseWindowDataset` 은 창 전체가 유효한 프레임만, `imu_features` 통계도 이 프레임만 쓴다 |
| `joint_state_valid` | `[T]` bool | robot: `joint_state` 가 실제로 잰 프레임 (같은 규칙). `q_valid_mask` / `qd_valid_mask` (→ baseline·contact 데이터셋, `q`/`qd` 통계) 에 반영 |
| `hand_global_orient` | `[T, 3]` float32 | axis-angle (쿼터니언 보간 후 변환, 각도 ∈ [0, π]) |
| `hand_finger_pose` | `[T, 15, 3]` float32 | axis-angle, MANO 순서, 평평한 템플릿 기준 |
| `hand_wrist_pos` | `[T, 3]` float32 | m |
| `hand_pose_valid` | `[T]` bool | 라벨 존재 + 신뢰도 게이트 + 짧은 결손 보간(`smooth_hand_labels`) + 앞뒤 라벨 모두 유효 |
| `object_pos` / `object_quat` | `[T, 3]` / `[T, 4]` float32 | m / wxyz (구간 밖은 끝값 유지) |
| `phase_id` | `[T]` int16 | `phase_names` 인덱스, −1 = 단계 없음 (`t0 ≤ t < t1`, 중첩 시 안쪽 단계) |
| `self_touch` | `[T, N]` bool | glove: `pose.mano.self_touch_from_hand` (캡슐 거리 ≤ r + margin), 손 자세 무효 프레임은 False |
| `contact_label` | `[T, N]` int8 | −1 모름, 0 무접촉, 1 접촉 (§2.5) |
| `cam_<name>_idx` | `[T]` int32 | 시각 ≤ t 인 최신 프레임 인덱스(zoh), 첫 프레임 전 −1 (`cameras.max_age_s` 초과 시 −1) |
| `gt_artefact_pct`, `gt_press_pct` | `[T, N]` float32 | (합성 전용, 선택) 정답 artefact / press |
| `gt_contact`, `gt_self_touch`, `gt_object_contact` | `[T, N]` bool | (합성 전용, 선택) 정답 접촉 |

`static/`: `baseline_raw [N]` float64 (ΔS 기준 raw), `taxel_channels [N]` int64 (layout taxel → raw 채널).

`derived/` (이후 stage 가 `Episode.set_derived` 로 씀): `baseline_pred`, `baseline_logvar` (ΔS %, `[T, N]`),
`residual = delta_pct − baseline_pred`, `residual_z` (보정된 z, **누름 양수**), `contact_prob`,
`contact_level` (int8 `ContactLevel`), `hand_finger_pose_imu` (`[T, 15, 3]`), `contact_label_pseudo` (int8 −1/0/1
`[T, N]`, contact stage 의 D2 pseudo 라벨 = `episode.D_CONTACT_LABEL_PSEUDO`; `contact_label` 배열은 그대로 둔다 —
`datasets.motion` 데이터셋은 `label_key="contact_label_pseudo"` 로 이 derived 배열을 직접 학습에 쓴다). 쓰기는
원자적(임시 파일 + `os.replace`)이라 memmap 으로 읽는 중인 프로세스가 잘린 배열을 보지 않는다. 비전 특징 캐시
`derived/vision_<key>_<camera>.npy` 는 프레임 단위(`[F, P, D]`)라 `vision.load_cached` 로만 읽는다.

### 2.3 마스터 시계와 보간

- 범위: `clock.reference` 스트림(pressure, imu, joint_state 중 존재하는 것)의 **공통 구간**, 시작은
  `1/hz` 격자로 올림. hand_pose / 카메라 / object 는 이 구간을 다 덮지 않을 수 있다 → 유효 마스크로 표시.
  `clock.span: pressure` 면 IMU / joint_state 도 자기 구간 밖에서 끝값 유지 → `imu_valid` / `joint_state_valid` = False.
- 샘플 갭: 이웃한 native 샘플 간격이 `max(clock.max_gap_s (0.1 s), 4 × 중앙 간격)` 보다 크면 그 사이 프레임은
  보간으로 메운 값일 뿐 측정이 아니다 → pressure 는 `saturated`, IMU / joint_state 는 `imu_valid` /
  `joint_state_valid` = False. 개수는 `preprocessing.invalid_frames` 와 `notes`.
- 연속 신호: 선형 보간. 쿼터니언: 연속성 보정 → 성분별 선형 보간 → 재정규화 (≤ 100 Hz 입력을 200 Hz 로
  올릴 때 SLERP 와 차이 무시 가능). axis-angle 라벨도 쿼터니언을 거친다.
- 카메라: zoh 인덱스만 저장 (프레임은 raw 카메라 디렉터리를 가리킴).

### 2.4 baseline 과 ΔS

1. `baseline.phase` 가 있으면 그 단계, 아니면 **첫 `no_contact` segment** (D1: 휴지 `baseline_start`,
   합성 glove D1 은 평손 `imu_calibration`; D2: `baseline`).
2. 그 구간 시작(+`trim_s`)부터 `duration_s`(1 s) 동안, 어떤 taxel 도 포화되지 않은 프레임의 **중앙값**
   (`common.signal.estimate_baseline`).
3. 쓸 구간이 없으면 `manifest.baseline` → 녹화 시작 `fallback_s` 초 (`preprocessing.notes` 에 기록).
4. baseline ≤ 0 인 taxel(죽은 채널)은 전 구간 `saturated`, ΔS = 0.

주의: 실제 세션의 기준은 휴지 자세라 그 자세의 artefact 가 baseline 에 들어간다. 합성 세션은 평손/`q = 0`
(artefact 0) 을 기준으로 한다.

### 2.5 `contact_label` 정책

| 조건 | 값 |
|---|---|
| `no_contact` segment 안 (`t0 ≤ t < t1`) | 0 |
| 기하 `self_touch` | 1 |
| 그 외 (D2 grasp/manipulate 의 물체 접촉, sync 탭, 기하가 못 잡는 self-touch 블록 등) | −1 |
| `no_contact` 안인데 `self_touch` (충돌) | `labels.conflict`: `unknown` → −1 (기본), `segment` → 0, `geometry` → 1 |

D2 물체 접촉은 여기서 붙이지 않는다 (`contact.pseudo_label` 이 나중에 채움). robot 세션은 기하 self-touch
가 없어 0 / −1 만 있다.

### 2.6 `meta.preprocessing`

`version` (`robot_skin.datasets.build/3`), `config` (적용된 전체 설정), `config_hash`, `source_fingerprint`
(raw 파일 해시: manifest·events 내용, 스트림 파일 크기 + 시간 벡터 `t` / `timestamps.npy` 내용, 등록되지 않은
`hand_pose.npz`·`object_pose.npz`), `notes` (경고),
`master_hz`, `t_span`, `clock_reference`, `stream_spans`, `baseline {source, segment, t0, t1, n_frames}`,
`saturated_frac`, `dead_taxels?` (기준 ≤ 0 인 죽은 채널 — vtla 가 `taxel_pad` 로 가린다), `imu {calibrated, calibration_source (manifest | phase:<name>), wrist_index,
frame (segment | sensor), vec_frame (sensor | world — gyro/acc 의 프레임 종류, `imu_features(vec_frame=…)` 에 줄 값),
world_aligned, quality?}`, `hand_pose {valid_frac, label_valid_frac, n_labels}`,
`q_source` (`joint_state` | `hand_pose` | null), `qd_source`, `urdf`, `zero_filled_joints?` (URDF 관절 중
joint_state 에 없던 것, q = 0), `invalid_frames {pressure, imu, joint_state}` (§2.3), `taxel_pose_source` (`hand_pose` | `urdf`
| `rest` | `static`), `self_touch {margin_m, frac}`, `object_frame`, `contact_label {n_no_contact, n_contact,
n_unknown, n_conflict}`, `segments` (라벨에 쓴 segments, §1.9), `markers`, `taxel_frame` (`mano_wrist` | `urdf_root` | `layout`),
`layout_source` (내장 이름 | `override:<이름/경로>` | manifest 값), `layout_file`,
`synthetic` (`fake_recorder` = `record --fake` 세션 `meta.fake`, `generator` = `datasets.synthetic` 세션, `null` = 실제 기록;
이 키가 생기기 전에 만든 episode 에는 없다),
`synthetic_gt {layout_order_match, delta_abs_err_pct {median, p99}, baseline_rel_err, saturated_recall, stored}`.

## 3. 분할과 정규화 파일

`splits.json` (`datasets.splits.save_splits`):

```json
{"format": 1, "relative": true, "train": ["motion/<id>", …], "val": […], "test": […],
 "meta": {"by": "subject", "seed": 0, "val_frac": 0.15, "test_frac": 0.15}}
```

`relative: true` 이면 경로는 processed 루트 기준 (`load_splits(path, root=…)`, 기본은 파일이 있는 디렉터리).
그룹(`subject` | `session` | `object` | `task` …)은 split 을 넘지 않는다.

정규화 통계 (`datasets.stats.save_stats`, train split 에서만 fit):

```json
{"format": 1, "stats": {"q": {"offset": […], "scale": […]}, "qd": {…}, "imu_features": {…}}, "meta": {…}}
```

`x_norm = (x − offset) / scale`, 특징 = 배열의 뒤쪽 차원을 펼친 것 (`q` → D, `delta_pct` → N,
`hand_finger_pose` → 45, `imu_features` → `pose.imu_model.imu_feature_dim(S)`). `delta_pct` 류는 포화 샘플,
손 자세와 glove `q` 는 `hand_pose_valid` 가 아닌 프레임, `qd` 는 `qd_valid_mask` 가 아닌 프레임, robot `q` 는
`joint_state_valid` 가 아닌 프레임, `imu_features` 는 `imu_valid` 가 아닌 프레임을 제외한다.

## 4. 버전

- raw manifest: `schema_version` 2 (v1 은 기본값으로 로드).
- Episode: `format_version` 1 (`episode.py`); 전처리 출력 변경 시 `datasets.build.PREPROCESS_VERSION` 을 올린다.
  CLI 는 기존 episode 를 건너뛰되, 설정 해시 / 버전 / raw 파일 구성(예: 나중에 추가된 `hand_pose.npz`, 고친
  `events.jsonl`, 싱크 재적용 — `apply_clock_models` 처럼 파일 크기가 같은 타임스탬프 재작성도)이 다르면 `stale` 로
  보고한다 (`--force` 로 재생성). `qc.json` 이 `passed: false` 인 세션은 기본으로 건너뛴다(`qc_failed`,
  `qc.skip_failed`). `meta.dry_run` manifest(`record --dry-run` 계획)는 `plan` 으로 보고하고 건너뛴다.
