# datasets/ — raw 세션 → Episode (전처리) · 분할 · 정규화 · D1 데이터셋

raw/processed 포맷의 모든 키·dtype·단위·좌표 규약은 **`docs/DATA_FORMAT.md`** 에 있다. 이 문서는 코드 사용법이다.

| 파일 | 역할 |
|---|---|
| `episode.py` | **고정 계약**: `Episode` / `EpisodeMeta`, 키 상수 `K_*`/`S_*`/`D_*`, `cam_idx_key`, `list_episodes` (수정 금지) |
| `build.py` | `preprocess_session(session_dir, out_root, cfg)` → `Episode`; `build_all`; CLI `python -m robot_skin.datasets.build` |
| `splits.py` | `make_splits(episode_dirs, by, val_frac, test_frac, seed, holdout)` — 그룹이 split 을 넘지 않음; `save_splits`/`load_splits`/`check_splits` |
| `stats.py` | `compute_stats(episodes, keys, method)` → `{key: NormStats}` (train split, 포화·무효 프레임 마스크); `apply_stats`/`invert_stats`; `save_stats`/`load_stats` |
| `motion.py` | D1 데이터셋: `BaselineWindowDataset`, `ContactWindowDataset`, `ImuPoseWindowDataset` (인과 창) |
| `synthetic.py` | 합성 raw 세션 (`generate_session` / `generate_dataset` / `load_ground_truth`) — 무거운 import(torch) 때문에 `datasets/__init__` 에서 export 하지 않음 |

## 빠른 시작

```bash
PY=python   # repo 루트
# 1) raw → processed (이미 있는 episode 는 건너뜀; --force 로 재생성)
$PY -m robot_skin.datasets.build --raw robot_skin/data/raw --out robot_skin/data/processed \
    --config robot_skin/configs/stages/preprocess.yaml [--set baseline.duration_s=2 --set qd.method=savgol]
```

```python
from robot_skin.datasets import list_episodes
from robot_skin.datasets.synthetic import generate_dataset
from robot_skin.datasets.build import build_all
from robot_skin.datasets.splits import make_splits, save_splits
from robot_skin.datasets.stats import IMU_FEATURES, compute_stats, save_stats
from robot_skin.datasets.motion import BaselineWindowDataset, ImuPoseWindowDataset

generate_dataset("/tmp/raw", n_motion=4, n_task=2, kind="glove", duration_s=6.0)   # 하드웨어 없이
build_all("/tmp/raw", "/tmp/proc")                                                   # [{session, status}, …]
eps = list_episodes("/tmp/proc", "motion")
splits = make_splits(eps, by="subject", val_frac=0.2, test_frac=0.2, seed=0)
save_splits(splits, "/tmp/proc/splits.json", root="/tmp/proc", meta={"by": "subject"})
stats = compute_stats(splits["train"], keys=("q", "qd", IMU_FEATURES))           # train 에서만 fit
save_stats(stats, "/tmp/proc/stats_motion.json")
ds = BaselineWindowDataset(splits["train"], window=32, joint_stats=stats)       # val/test 도 같은 stats
imu = ImuPoseWindowDataset(splits["train"], window=32, imu_stats=stats)
```

## 전처리 단계 (`build.py`)

1. manifest + layout 해석 (`resolve_layout`: 설정 override → 세션 디렉터리 기준 상대 경로 → 세션 안 사본 →
   절대 경로 → 내장 이름; 세션을 옮겨도 동작). events/segments 로드.
2. pressure 로드 (기본 npz, `pressure.loader` 로 mk555 `.bin` 로더 주입 — `bin_merge.py` 복사 금지) →
   `layout.by_channel` (채널 순서 → layout 순서) → (선택) 저역통과.
3. 마스터 시계: `clock.reference` 스트림의 공통 구간, `1/hz` 격자 (세션 시계 그대로).
4. baseline = 첫 `no_contact` segment 의 처음 `duration_s` 중앙값 → ΔS (press → 음수) → `saturated`
   (레일 근처 raw 샘플이 보간에 섞인 프레임 포함, `|ΔS| ≥ 90 %`).
5. IMU: `manifest.calibration` 의 `imu_offsets`/`imu_world` 적용 (쿼터니언 `G⁻¹ ⊗ q ⊗ q_off`, gyro/acc
   `R_offᵀ v`) → 연속성 보정 → 보간 → 재정규화. 보정이 없으면 raw 로 두고 경고
   (`imu.calibrate_if_missing: true` 면 `imu_calibration` 단계에서 계산; raw 세션은 수정하지 않음).
6. hand_pose: `pose.vision_hand.smooth_hand_labels` (신뢰도 게이트, ≤ 0.25 s 결손 SLERP, 6 Hz 저역통과) →
   쿼터니언 보간 → `hand_pose_valid`.
7. `q`/`qd`: robot = `joint_state` 를 `URDFModel.reorder_q` 로 URDF 순서 (URDF = `robot.urdf` 설정 또는
   `manifest.meta.urdf`, 세션 디렉터리 기준); glove = `hand_finger_pose` 45-D. `qd` = `joint_velocity`
   (Savitzky–Golay, 기본 `savgol_causal`: 최근 50 ms 다항식 적합을 가장 새 샘플에서 평가 — 과거 샘플만 쓰므로
   온라인 제어기가 q 링버퍼로 **똑같이** 재현한다. `savgol` 은 더 매끈한 중앙 추정이지만 오프라인 전용).
8. taxel 자세 (**손 / 로봇 base 프레임**, `episode.py` 계약): glove `pose.mano.taxel_poses_from_hand` 를
   `global_orient = 0`, 손목 원점으로 (손가락 자세만 반영, `taxel_frame = mano_wrist`), robot
   `pose.robot_fk.taxel_poses_from_joints` (URDF 루트), 그 외 layout 정적 위치. glove `self_touch` =
   `pose.mano.self_touch_from_hand` (손 자세 유효 프레임만; 거리라 프레임과 무관).
9. `phase_id` + `meta.phases` (events), `contact_label` (0 = no_contact segment, 1 = self_touch, 그 외 −1,
   충돌은 −1), `cam_<name>_idx` (zoh, 첫 프레임 전 −1), 카메라 디렉터리 symlink/복사, `meta.task`.
10. `gt_synthetic.npz` 가 있으면 ΔS 를 정답과 대조 (`meta.preprocessing.synthetic_gt`) 하고 `gt_*` 배열 저장.
11. 임시 디렉터리에 저장 후 rename (중단돼도 반쯤 쓴 episode 가 남지 않음). `--force` 는 `derived/` 까지 지운다.
    이미 있는 episode 는 건너뛰고, 설정·버전·raw 파일 구성(나중에 추가한 `hand_pose.npz` 등)이 바뀌었으면
    `stale` 로 보고한다.

설정 키 전체와 기본값: `robot_skin/configs/stages/preprocess.yaml` (= `build.DEFAULTS`, 모르는 키는 에러).

## D1 데이터셋 (`motion.py`)

모든 샘플은 마스터 시계 프레임 `t` 에 고정되고 **과거 창** `t−W+1 … t` 만 본다 (시작 전은 프레임 0 으로
패딩). 정규화 통계는 반드시 밖에서(train split 에서 `compute_stats`) 넣는다 — `None` 이면 원 단위.

| 데이터셋 | 샘플 | 사용 프레임 |
|---|---|---|
| `BaselineWindowDataset` | `q_hist[W,D]`, `qd_hist[W,D]`, `pos[N,3]`, `nrm[N,3]`, `y[N]` (t 의 ΔS), `valid[N]` | `contact_label ∈ only_labels` (기본 0) 이고 포화 아닌 taxel ≥ `min_valid` 개, 창 전체의 `q`·`qd` 가 측정값 (glove: `stats.qd_valid_mask` = `hand_pose_valid` 를 미분 필터 폭만큼 침식) |
| `ContactWindowDataset` | `z_hist[W,N]` (derived `residual_z`, 누름 양수), `sat_hist[W,N]`, `q[D]`, `qd[D]`, `q_valid` (t 의 q/qd 가 측정값인지), `label[N]`, `label_mask[N]` | `contact_label ≥ 0` (포화 제외) 인 taxel ≥ `min_labelled` 개; derived `residual_z` `[T,N]` 필요 |
| `ImuPoseWindowDataset` | `feat[W,F]` (`pose.imu_model.imu_features`, 손목 IMU 기준), `finger_pose[15,3]`, `global_orient[3]` | `hand_pose_valid` |

한 데이터셋의 episode 들은 `n_taxels`, `meta.joint_names` (q 열 순서), IMU 사이트가 같아야 한다 (다르면
`ValueError` — 예: URDF 를 못 찾아 드라이버 순서로 남은 robot episode 가 섞이는 것을 막는다).
관측 ΔS 는 baseline 모델의 입력으로 절대 쓰지 않는다 (SATS bending restorer 의 교훈: ΔS 를 보면 접촉을
지우는 법을 배운다). 모든 샘플에 `episode`, `t_index` 가 들어 있어 예측을 episode 로 되돌려 쓸 수 있다.

## 분할과 통계

- `make_splits(by="subject")` 가 기본 — 새 사람에게 장갑을 씌웠을 때의 일반화. `by="object"`/`"task"` 는
  D2 일반화, 튜플은 복합 키. 키가 없는 episode (예: `by="object"` 의 D1) 는 episode 하나가 한 그룹.
  `holdout={"subject": ["S07"]}` → test, `{"val": {…}, "test": {…}}` 형태도 가능 (holdout 은 그룹 규칙보다 우선).
- `compute_stats(..., masks="auto")`: `delta_pct`/`pressure_raw`/residual 류는 포화 제외, 손 자세와 glove
  `q` 는 `hand_pose_valid` 만, glove `qd` 는 `qd_valid_mask` 만 (라벨 결손 뒤 점프의 속도 스파이크 제외). `method="std"` 는 episode 별 누적(Chan)이라 메모리에 전부 올리지 않는다.
