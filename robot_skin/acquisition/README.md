# acquisition/ — 데이터 수집 (프로토콜 · 소스 · 레코더 · 싱크 · 보정 · QC)

운영자 절차(장비 체크리스트, 장갑 착용, 3-탭 싱크, 블록별 스크립트, D2 과제 카탈로그, QC 실패 시 조치,
권장 수집량, 영상 개인정보)는 **`docs/DATA_ACQUISITION.md`** 에 있다. 이 문서는 코드 사용법이다.

| 파일 | 역할 |
|---|---|
| `manifest.py` | `SessionManifest` v2 (고정 계약, 수정 금지): kind, layout, streams, segments, dataset, subject, task, calibration. raw 파일 포맷은 모듈 docstring |
| `protocol.py` + `protocols/*.yaml` | 프로토콜 스키마/검증, `plan_session` (seed 고정 무작위화), `format_script` (운영자 대본), `make_session_id` |
| `instructions.py` | 지시문 템플릿 슬롯 `{object}` `{target}` … 렌더링 (`red_cup` → "red cup") |
| `sources.py` | `StreamSource` 프로토콜, `SimClock`/`MonotonicClock`, `PlaybackSource`, `Fake*Source`, `SerialPressureSource`(pyserial, 파서 주입), `CameraSource`(cv2), `ImuSource`·`RosJointStateSource` 스텁 |
| `fake.py` | `FakeScene`: 프로토콜 타임라인 → 물리적으로 일관된 합성 스트림 (압력·IMU·카메라·손 라벨·물체·관절). 누름은 접촉 부위별 대본이라 전처리의 기하 self-touch 라벨과 맞지 않는다 — `--fake` 는 소프트웨어 경로 확인용이고, 접촉 검출 학습·평가는 `datasets.synthetic` 으로 |
| `recorder.py` | `Recorder`: 호스트 단조 시계, 소스별 폴링 스레드(또는 `SimClock` 동기 모드), `events.jsonl` 실시간 기록, 종료 시 raw 파일 + `session.json` |
| `sync.py` | 3-탭 싱크: 이벤트 envelope 상호상관 → 스트림별 offset (+ 시작/끝 → drift), 타임스탬프 제자리 보정 |
| `calibration.py` | 평손 보정 블록 → `calibrate_imu_offsets` → `manifest.calibration` (`imu_offsets`/`imu_world`/`imu_sites`) |
| `qc.py` | `session_qc` + CLI: 레이트·지터·갭·드롭, 포화, baseline 드리프트, IMU, 카메라, 라벨, 싱크 게이트 |
| `session.py` | `run_plan`/`record_episode`: 에피소드마다 디렉터리, 운영자(`ConsoleOperator`/`AutoOperator`; D2 수동 phase 는 Enter 마다 *시작* 경계, 마지막 Enter 가 체인의 끝), 후처리(싱크→보정→segments 재생성(`refresh_segments`)→QC) |
| `glove_logger.py`, `robot_logger.py`, `_cli.py` | CLI (`--dry-run`, `--fake`, 실장비) |

## 빠른 시작

```bash
PY=python   # repo 루트에서
# 계획만: session.json(단일 세션) / plan.json(D2) + 한국어 운영자 대본 출력
$PY -m robot_skin.acquisition.glove_logger --protocol d1_motion --subject S01 --dry-run
# 합성 end-to-end (하드웨어 없이 수 초): 기록 → 3-탭 싱크 → IMU 보정 → QC
$PY -m robot_skin.acquisition.glove_logger --protocol d1_motion --subject S01 --fake --time-scale 0.05   # → robot_skin/data/synthetic
$PY -m robot_skin.acquisition.glove_logger --protocol d2_task --task pour --task wipe --episodes 4 --fake
$PY -m robot_skin.acquisition.robot_logger --fake --time-scale 0.1 --subject R01        # robot_sweep
# QC / 후처리 재실행
$PY -m robot_skin.acquisition.qc robot_skin/data/raw/motion/S01/<session_id> --write
$PY -m robot_skin.acquisition.session <session_dir> [--sync-from DIR] [--calibration-from DIR]
```

최상위 CLI 로도 같다: `python -m robot_skin record glove|robot <위 인자>` ≡ `glove_logger` / `robot_logger`,
`python -m robot_skin postprocess …` ≡ `acquisition.session`, `python -m robot_skin qc …` ≡ `acquisition.qc`.

주요 옵션: `--protocol d1_motion|d2_task|robot_sweep|<yaml>`, `--subject S01`(가명 ID만),
`--cameras ego,third|none`, `--task`/`--object`/`--episodes`/`--repetitions` (D2), `--instruction "..."`
(운영자 지시문 덮어쓰기), `--seed`, `--time-scale`, `--no-imu`, `--no-sync`, `--sync-from`,
`--calibration-from`, `--camera-format auto|npy|jpg`, `--lang ko|en`.

저장 위치: `--out` 은 단일 세션 프로토콜(D1, robot_sweep, `--duration`)이면 세션 디렉터리, D2 면 부모
디렉터리(에피소드마다 하위 디렉터리). 생략 시 `<--root>/<dataset>/<subject>/<session_id>` — `--root` 기본은
`configs/default.yaml` `paths.raw_root`(`robot_skin/data/raw`), **`--fake` 는 `paths.synthetic_root`**
(`robot_skin/data/synthetic`: 합성 세션이 실제 raw 와 섞여 전처리되지 않게). `--dry-run` 계획은 `<root>/<dataset>/<subject>/dry_run`
에 남고 `datasets.build` 는 이를 `plan` 으로 건너뛴다.
`session_id = <dataset>-<subject>-<YYYYMMDD>-<HHMMSS>[-e<NNN>-<task>-<object>-r<rep>]`.

반환 코드: 0 = 전 세션 QC 통과, 3 = 기록은 됐지만 QC 실패 세션 있음, 1 = 기록 없음.
실장비: `ImuSource`(장갑 IMU 허브)와 `RosJointStateSource` 는 장치가 생길 때까지 `NotImplementedError`
(무엇을 구현할지 docstring 에 명시). `--no-imu` 로 압력 + 카메라만 기록할 수 있다.

## raw 세션 포맷 (`datasets.build` 입력)

```
session.json      SessionManifest (streams, segments, dataset, subject, task, calibration, meta)
pressure.npz      t[T] (s, 세션 시계), raw[T,C] float64 (채널 순서; layout.by_channel 로 재배열)
imu.npz           t, quat[T,S,4] wxyz, gyro[T,S,3] rad/s, acc[T,S,3] m/s² (센서 프레임, 중력 포함), sites[S]
joint_state.npz   t, q[T,D], qd, tau, names[D]                                   (robot)
hand_pose.npz     t, global_orient[T,3], finger_pose[T,15,3], wrist_pos[T,3], confidence[T]  (오프라인 비전 라벨)
object_pose.npz   t, pos[T,3], quat[T,4]                                         (선택)
camera_<name>/    timestamps.npy[F] + frames.npy uint8[F,H,W,3] | 000000.jpg …
events.jsonl      {"t","type","name","value"}  type: phase_start|phase_end|marker|instruction|success
qc.json           session_qc 보고서
```

- 시계: 모든 샘플은 도착 시 **호스트 단조 시계**로 찍고 `t = t_host − t0`(레코더 시작) 로 저장.
  싱크 적용 후 `t` 는 기준(`pressure`) 시계로 보정되고 원본은 `t_host` (npz) / `timestamps_host.npy`
  (카메라) 로 남는다 — 재실행해도 누적되지 않는다. 기록 내용은 `manifest.calibration["sync"]`.
- `phase_start.value = {"kind", "contact": none|self|object|any, "labels": [...], "block", "speed", ...}`.
  `stop()` 시 phase 마다 라벨별 segment 가 생긴다 (`none`→`no_contact`, `self`→`self_touch`, 보정 블록은
  `[calibration, no_contact]`, 싱크 블록 `sync`). D2 는 과제 phase 전체를 덮는 `task` segment 가 추가된다
  (이벤트에서 나오지 않는 segment 는 `meta.recorder.explicit_segments` 에도 적힌다). 이벤트는 `type` 으로 식별한다
  (`instruction`/`success` 의 `name` 은 정보용).
- `events.jsonl` 이 segment 의 기준이다: 경계를 손으로 고친 뒤 `acquisition.session <dir>` 를 다시 돌리면
  `session.json` 의 segments 가 이벤트에서 다시 만들어지고(`recorder.session_segments` + explicit segment),
  `datasets.build` 도 같은 함수로 segments 를 만든다.
- D2 `manifest.task = {task_id, instruction, object, target, success, repetition, template_index, template,
  instruction_source, slots, success_criteria, grasp, manipulate}`.
- IMU 보정: `manifest.calibration` 의 `imu_offsets`/`imu_world`/`imu_sites` (+ `imu_calibration_quality`),
  읽기는 `pose.imu_model.imu_calibration_from_dict`.

## Python API

```python
from robot_skin.acquisition import (plan_session, format_script, run_plan, fake_source_factory,
                                    AutoOperator, SimClock, Recorder, SessionManifest, sync_session, session_qc)

plan = plan_session("d2_task", seed=0, tasks=["peg_insert"], n_episodes=3)
print(format_script(plan))
res = run_plan(plan, kind="glove", source_factory=fake_source_factory(plan), clock_factory=SimClock,
               operator=AutoOperator(), out="/tmp/d2", subject="S01")
print(res[0]["qc"]["passed"], res[0]["sync"]["streams"]["camera_ego"]["offset_s"])

# 직접 레코더 사용 (소스는 StreamSource 프로토콜: name, kind, rate_hz, start(clock), poll(), stop(), info())
rec = Recorder(sources, "sess_dir", SessionManifest(kind="glove", layout="glove_template", dataset="motion"))
rec.start()
with rec.phase("baseline_start", contact="none", labels=["no_contact"]):
    rec.run_for(5.0)
rec.stop()
```

실장비 연결: `StreamSource` 를 구현하면 된다 (`poll()` 은 논블로킹, `(t_host, sample)` 목록 반환, 장치 시각이
있으면 첫 프레임에서 호스트 시계에 고정). mk555 바이너리 보드는 `SerialPressureSource(parser=...)` 에
`deformable_sats/sats/preprocessing/bin_merge.py` 기반 파서를 주입한다 (**복사 금지**, 정본은 그쪽).

## 싱크 · 보정 · QC 요약

- **3-탭 싱크** (`sync.py`): 검지 끝으로 싱크 패드를 짧게-길게(0.6 s / 1.2 s) 3번. 각 스트림을 활동
  envelope(채널별 robust 스케일 `|dx/dt|` 합, 200 Hz, 가우시안 평활)로 바꿔 기준(압력)과 상호상관
  (정규화, 서브샘플 포물선 보간). 간격이 불균등해 ±짧은 간격 이내 lag 에서 피크가 유일하다. 압력/IMU 는
  모서리(충격·이륙)가 같은 모양이라 σ=20 ms, 카메라(30 Hz)는 손 *움직임*만 보이므로 σ=60 ms 로 탭
  단위 중심을 맞춘다 (오차 ≤ 1 프레임; 더 정밀하려면 LED 플래시). 시작+끝 → `t_ref = scale·t + offset`
  (>1000 ppm 은 기각). 합성 세션에서 IMU 지연 12 ms 를 ±6 ms, 카메라 30/45 ms 를 1 프레임 이내로 복원(테스트).
- **IMU 보정** (`calibration.py`): 평손 블록 → `imu_reference_rotations` → `estimate_world_alignment`(손목)
  → `calibrate_imu_offsets`. 정지 품질(gyro RMS ≤ 0.15 rad/s, 자세 흩어짐 ≤ 3°) 미달이면 콘솔 운영자가
  블록 반복을 제안한다. 합성 장착 오차(최대 20°)를 2° 이내로 복원(테스트).
- **QC** (`qc.py`, 기본 임계값 `DEFAULT_THRESHOLDS`, `--set key=value`): error = 소스 오류/빈 스트림,
  레이트 ±10 %, 드롭 > 2 % (카메라 5 %), 0.5 s 초과 갭, 비단조 시각, phase 구간을 못 덮는 스트림,
  no_contact 구간 포화 > 0.5 %, baseline 드리프트 > 3 % (첫·마지막 **휴식 자세** static 블록끼리만 비교 —
  평손 보정 등 다른 자세의 무접촉 블록은 굽힘 artefact 가 달라 드리프트가 아니다),
  quat 노름 오차 > 0.02, 프레임 수 불일치, D1 no_contact 없음, D2 task/instruction 없음, 싱크 |offset| > 0.5 s.
  warning = 전체 포화 > 10 %, IMU 보정 없음/정지 불량, 손 라벨 coverage < 0.7, 성공 판정 없음,
  싱크 점수 < 0.3, 탭 3개 미검출, 자동 종료된 phase.

## 테스트

```bash
python -m pytest -q -p no:cacheprovider robot_skin/tests/test_acquisition.py robot_skin/tests/test_recorder.py \
    robot_skin/tests/test_protocol.py robot_skin/tests/test_sync_qc.py
```

근거: 세션 = 스트림별 파일 + 매니페스트 + 이벤트 로그, 1인칭 + 고정 카메라 구성은 ActionSense
(DelPreto et al., NeurIPS 2022 D&B). D2 삽입 과제는 VTLA (arXiv:2505.09577), 닦기는 OSMO
(arXiv:2512.08920). 전체 목록은 `docs/REFERENCES.md`.
