# 로봇 배포 — 학습된 VTLA 정책을 촉각 스킨 로봇 핸드에서 실행

`vtla` 스테이지가 만든 `policy_bundle.pt` 를 로봇 핸드에서 폐루프로 돌리는 **운영 절차서**다. 코드 지도는
`robot_skin/control/README.md`, 데이터 포맷은 `docs/DATA_FORMAT.md`, 학습 스테이지는 각 `stages/*.py`,
논문 근거는 `docs/REFERENCES.md`.

```bash
# 시뮬레이션(가짜 로봇 핸드 + 가짜 카메라): 소프트웨어 경로 전체 점검
python -m robot_skin.stages.deploy --set bundle=robot_skin/runs/vtla --set duration_s=10
# 결과: robot_skin/runs/deploy/metrics.json, robot_skin/runs/deploy/sessions/deploy_0/ (RAW 세션)
```

---

## 1. 전체 흐름과 필요한 산출물

| 산출물 | 만드는 곳 | 배포에서의 역할 |
|---|---|---|
| `policy_bundle.pt` | `stages/vtla.py` | 정책 가중치 + 행동 공간/정규화 + proprio 정규화 + 촉각 특징 스펙(`TactileFeatureSpec`) + contact 규칙 + 카메라·eval 변환 + 토크나이저 + 타이밍(policy_hz, stride, horizon) |
| 로봇 스킨의 `baseline_model.pt` | `stages/baseline.py` (로봇 D1: `acquisition/protocols/robot_sweep.yaml`) | 움직임에 의한 무접촉 ΔS(artefact) 예측 → residual |
| 로봇 스킨의 `calibrator.json` (+ `contact_detector.pt`) | `stages/contact.py` | residual → z-score, NONE/WEAK/STRONG/SATURATED, (선택) FSM 게이트 |
| 로봇 URDF + 스킨 레이아웃 | 사용자 | 관절 순서·한계, taxel 자세(FK), 리타게팅 |

주의: 번들 안의 stage-1 참조(`tactile.calibrator_state`, `baseline_model`)는 **정책을 학습한 스킨**(보통 글러브)의
것이다. 로봇 스킨은 자기 D1 데이터로 baseline·contact 스테이지를 따로 돌려 `stage1.baseline_model` /
`stage1.calibrator` 로 넘긴다. 번들의 `tactile.layouts` 가 로봇 레이아웃 이름과 다르면 deploy 가 번들 참조를
쓰지 않고 notes 에 남긴다(`OnlineTactileProcessor.from_policy_bundle` 도 같은 규칙, `PolicyBundle.stage1_matches`).
contact 스테이지 기본값(`use_logvar`)으로 맞춘 calibrator 는 **자기 baseline 모델의 예측 분산**이 있어야 z 를 만든다 —
번들만 로봇 PC 로 복사해 baseline 모델 파일이 없으면 그 calibrator 는 쓰지 않고(notes) 시작 보정
(`startup.calib_s`)이 대신한다. 촉각 bootstrap(움직임 모델 없는 정적 기준)으로 학습된 번들은 거부한다
(`allow_bootstrap: true` 는 시뮬레이션 전용).

## 2. 제어 루프와 주기

```
 ┌──────────────── control tick: control_hz = 200 Hz (5 ms) ─────────────────────────────────┐
 │ robot.read_state() → (t, q, qd)        robot.read_pressure() → (t, raw[C])                  │
 │        │                                        │                                           │
 │        │        OnlineTactileProcessor.step: raw → ΔS% → saturation → qd(causal SG)         │
 │        │          → taxel pose(FK) → CausalBaselineStream → residual → FSM → z / level      │
 │        │          → TactileHistory 특징 [N,F] + contact(level 규칙) (+ detector/hysteresis)   │
 │        ▼                                        ▼                                           │
 │  every policy_every = 10 ticks (policy_hz = 20 Hz):                                         │
 │     cameras.read() → eval transform ─┐                                                      │
 │     proprio ring (k = obs_history) ──┼─ make_observation → collate_vtla → policy.predict    │
 │     instruction (tokenizer) ─────────┘      → chunk [H,A] (정규화) → unnormalize            │
 │     → make_absolute(state) → TemporalEnsembler.add/step → 다음 정책 틱의 목표                │
 │        hand_mano: hand_action_fingertips → FingertipRetargeter.step → 로봇 관절 목표         │
 │        robot_joint: 번들 action 이름 → 로봇 관절 순서                                        │
 │  매 틱: 마지막 명령 → 목표로 선형 보간 → SafetyFilter → robot.send_joint_targets(q_cmd)     │
 │  DeploymentLogger: pressure / joint_state / camera / events / deploy_log.npz               │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

| 주기 | 기본값 | 근거 |
|---|---|---|
| 촉각 센서 / 마스터 클록 | 200 Hz | 전처리 `master_hz`. `TactileFeatureSpec` 의 history stride, qd 창(50 ms), FSM 시간이 모두 이 틱 단위 → **control_hz = 번들 `source_hz`** 여야 학습과 같다(다르면 경고) |
| 제어 루프 | 200 Hz | 센서마다 한 번 처리 + 위치 목표 전송 |
| 정책 | 20 Hz (`timing.policy_hz`, stride 10) | 학습 샘플이 20 Hz 틱에서 만들어졌다. chunk[i] = 정책 틱 t + `chunk_offset` + i 의 행동 — offset 1(기본)이면 chunk[0] 이 다음 틱 목표, offset 0 번들은 chunk[0](현재)을 버리고 chunk[1] 부터 앙상블에 넣는다(offset > 1 은 경고) |
| 카메라 | ≈ 30 Hz | 정책 틱에서 최신 프레임 사용. 첫 프레임 전에는 학습 데이터셋과 같은 대체 이미지(0 영상의 eval 변환, `vision_valid` False) |

ACT 방식(Zhao et al., arXiv:2304.13705): 매 정책 틱에 새 chunk 를 `TemporalEnsembler` 에 넣고
`w_i = exp(−k·i)` (k = 번들 `meta.ensemble_k`, 기본 0.01) 로 겹치는 예측을 평균한다. 앙상블·리타게터·
처리기 스트림·proprio/카메라 링은 rollout 시작마다 리셋된다.

## 3. 시작 절차 (start-up)

1. **정지·무접촉 baseline** (`startup.baseline_s`, 기본 1 s): 손을 편 채 아무것도 닿지 않게 둔다. 이 창의
   raw 중앙값이 taxel 별 기준값이 된다 — 전처리와 같은 추정기(`common.signal.estimate_baseline`), 레일 샘플이
   있는 프레임 제외(창은 실제 시간: 건너뛴 샘플도 시간을 쓴다). 죽은 채널(기준 ≤ 0)은 ΔS = 0 + 항상 포화로
   표시되고 경고가 나온다 — SATURATED 가 촉각 정지 레벨이면 그 taxel 의 체인은 계속 닫히지 못한다(fail safe;
   채널을 고치거나 `safety.tactile_stop.levels` 에서 SATURATED 를 뺀다). 세션 로그에 `baseline` phase
   (`no_contact` 세그먼트)로 남아 재전처리 때도 같은 창이 baseline 이 된다.
2. **보정**: 로봇 스킨의 `calibrator.json` 이 있으면 그대로 쓴다(권장). 없고 `startup.calib_s > 0` 이면 정지 상태
   residual 로 **bring-up 용** calibrator 를 맞춘다(`control.online.startup_calibrator`; 움직임 artefact 를 모르므로
   움직일 때 σ 가 낙관적 → 실제 운용 전 contact 스테이지를 돌릴 것).
3. **rollout**: 모든 스트림 리셋 후 `rollout` phase(`task` 세그먼트) + instruction 이벤트, 정책 실행.
4. 종료: phase 닫기, (선택) 성공 판정 `success` 이벤트, 파일 기록.

로봇이 움직이기 전 baseline 을 잡으므로, **시작 자세 = 학습 데이터의 baseline 자세**(편 손 / q = 0 부근)로
맞추면 ΔS 기준이 학습과 일치한다.

## 4. 오프라인 ≡ 온라인

`OnlineTactileProcessor` 는 학습 데이터를 만든 함수를 틱마다 그대로 부른다(`control/README.md` 표).
`tests/test_control.py::test_online_processor_reproduces_offline_stage_outputs` 가 처리된 합성 에피소드를 흘려
ΔS·포화·레벨은 비트 단위로, baseline 평균/분산·z·특징·detector 확률은 1e-5 수준으로 오프라인 스테이지 출력과
같음을 확인한다. 배포 설정 검증에는 녹화된 에피소드를 `control.replay_episode` 로 흘려 derived 배열과 비교하면 된다.

설계상 남는 차이: (1) 압력은 최신 샘플(zero-order hold) vs 오프라인 선형 보간 — 프런트엔드를 200 Hz 로 읽으면
무시 가능; (2) 글러브 손 라벨의 비인과 평활은 온라인에 없음; (3) `hand_mano` 정책의 proprio 는 로봇에 사람 손
상태가 없어 명령한 손 행동을 쓴다(`hand_state.estimate: true` 면 손가락은 로봇 q 에서
`transfer.RobotToManoEstimator` 로 추정). 손목 자세 행동은 팔 제어기가 없으면 쓰이지 않는다.

## 5. 안전

`SafetyFilter` 가 모든 명령을 거른다 (먼저 해당하는 규칙이 우선):

| 순서 | 규칙 | 동작 | 설정 |
|---|---|---|---|
| 1 | e-stop 래치 | e-stop 순간 자세 유지(촉각 e-stop 은 **측정** 자세 — 마지막 명령을 유지하면 서보 지연만큼 계속 조인다), `robot.estop()` 1 회 호출 | `SafetyFilter.reset` 으로만 해제 (새 rollout 시작 = 운영자의 명시적 재시작, 경고 로그) |
| 2 | 센서 watchdog | 스트림이 `max_age_s` 보다 오래되면 마지막 명령 유지, `estop_after_s` 지속 시 e-stop | `safety.watchdog` (pressure/joint 50 ms, camera 0.5 s, e-stop 0.5 s) |
| 3 | 비유한 목표 | NaN/inf 관절 → 마지막 명령 | 항상 |
| 4 | **촉각 정지** | 어떤 taxel 이 STRONG/SATURATED 로 `min_ticks`(10 틱 = 50 ms) 지속 → 그 taxel 의 **운동 체인 관절만** 닫는 방향 동결(여는 방향은 허용), `release_ticks` 조용하면 해제. 동결 위치는 **측정 자세**(서보 지연만큼 더 조이지 않도록) | `safety.tactile_stop` (`mode`: freeze_closing / hold / estop), `closing_sign`, `per_taxel_joints` |
| 5 | 관절 한계 | `[lower+margin, upper−margin]` 로 자르기 | `safety.margin` |
| 6 | 속도 | 틱당 `|Δq| ≤ max_vel·dt` | `safety.max_vel` (3 rad/s) |
| 7 | 가속도 | `|Δv| ≤ max_acc·dt` | `safety.max_acc` |

SATURATED 에는 ADC 레일 드롭아웃도 포함되므로 센서 고장도 "강한 접촉"처럼 멈추는 쪽(fail-safe)이다. 이벤트
(`tactile_stop_on/off`, `stale_on/off`, `estop`)는 세션 `events.jsonl` 에 `safety` marker 로 남고, 클램프는
`metrics.safety_counts` 로 집계된다. **이 소프트웨어 층은 하드웨어 e-stop 과 핸드 자체의 전류/토크 제한을
대체하지 않는다.** 첫 실행은 `max_vel` 을 낮추고(예: 0.5 rad/s), 물체 없이 시작한다.

## 6. 지연 예산

200 Hz 에서 틱 하나는 **5 ms**. 이 개발 박스(CPU 4 코어, GPU 없음, `torch.set_num_threads(1)`)에서 측정한 값:

| 구성 요소 | 호출 주기 | 측정값 (CPU, p50 / p95) | 비고 |
|---|---|---|---|
| ΔS · 포화 · FSM · 보정 · 레벨 (numpy) | 200 Hz | 0.14 / 0.19 ms | |
| taxel FK (합성 16-DoF 손, 9 taxel, numpy 체인) | 200 Hz | 0.21 / 0.38 ms | torch FK 는 ≈ 1.8 ms 라 온라인은 numpy 경로 사용 |
| 처리기 1 틱 전체 (위 + `CausalBaselineStream` TCN W=32, hidden 64, 4 층) | 200 Hz | 2.1 / 2.4 ms | 신경망이 대부분 |
| 정책 추론, 기본 VTLA (d_model 128, 카메라 2, tiny CNN 96×128, chunk head) | 20 Hz | 12 / 14 ms | flow head 10 step: 27 / 38 ms |
| 리타게팅 (LM 10 iter, fd Jacobian, 16 DoF) | 20 Hz | ≈ 35 ms | `retarget.iters` 로 조절; 웜스타트면 5 iter 로 충분한 경우가 많다 |
| (선택) 손 상태 역추정 `hand_state.estimate` (MANO 20 파라미터, LM 10 iter, fd) | 20 Hz | ≈ 30 ms | |
| 테스트용 tiny 정책 | 20 Hz | ≈ 4 ms | |

- 첫 추론은 지연 초기화로 수백 ms 걸릴 수 있어, 시작 절차에서 baseline 캡처 직후 손을 멈춘 채 더미 추론
  2 회로 워밍업한다(`startup.warmup_ms` 로 기록). rollout 시작 시 마감 시각(deadline)을 다시 잡는다.
- 정책 추론·리타게팅은 **제어 틱 안에서 동기로** 돈다(학습 타이밍과 같고 결정적). 5 ms 를 넘기면 그 틱이 늦어지고
  `overruns` 가 오른다. ACT chunk + 앙상블 덕분에 로봇은 이전 목표를 계속 따라가므로 가끔의 overrun 은 매끄러움만
  해친다. CPU 에서는 정책 틱마다 overrun 이 생기므로 **GPU 권장**. 비동기 추론 스레드는 TODO.
- GPU(RTX 5090)에서의 값은 이 박스에서 측정하지 못했다 — 실행 후 `metrics.json` 의 `latency_p50_ms`,
  `latency_p95_ms`, `tick_p95_ms`, `overruns`, `benchmark`(`control.latency.benchmark_policy`) 를 확인한다.
  대략 CPU 보다 한 자릿수 빠를 것으로 예상하지만 **측정 전에는 가정하지 말 것**.
- TorchScript 내보내기(`control.latency.export_torchscript`)는 텐서만 받는 부분(baseline 예측기, 촉각 인코더)에만
  된다 — 문자열·dict 입력을 받는 VTLA 정책 전체는 trace 되지 않는다.

## 7. 실제 로봇 핸드 연결

1. **드라이버**: `robot_skin.control.interfaces.RobotHandInterface` 를 구현한다.
   - `joint_names` (구동 관절, 드라이버 순서 — URDF 순서와 달라도 된다: 러너가 이름으로 처리기(URDF / baseline
     모델 순서)·정책 행동 순서로 바꾼다), `lower`/`upper` (rad), (선택) `velocity_limits`
   - `read_state() -> (t, q[D], qd[D] | None)` (속도를 재지 않으면 None — 로그에 가짜 0 속도를 남기지 않는다;
     비유한 q 는 처리기가 직전 값으로 유지), `read_pressure() -> (t, raw[C])` — raw 는 촉각 프런트엔드의
     **채널 순서**(`pressure.npz` 와 같음). mk555 `.bin` 파싱은 `deformable_sats/sats/preprocessing/bin_merge.py` 가
     정본이다(복사하지 말고 파서를 감쌀 것)
   - `send_joint_targets(q[D])` — 위치 목표, 오래 막히지 않게
   - (선택) `estop()`, `start()`/`stop()`, `layout`, `urdf_xml` 또는 `urdf_path` (세션 로그에 `robot.urdf` 로 저장 →
     재전처리에서 관절 순서·FK 가 복원됨)
   - **타임스탬프는 러너와 같은 호스트 클록**(`time.monotonic`) 초 단위. 장치 시간을 받으면 변환하거나 도착 시각으로 찍는다.
   - `check_robot(robot)` 으로 형태를 확인한다.
2. **스킨 레이아웃**: `common/layouts/robot_hand_template.yaml` 을 복사해 `parent` 를 실제 URDF 링크 이름으로,
   `position`/`normal` 을 실측값으로, `channel` 을 실제 배선으로 바꾼다.
3. **로봇 스킨 stage 1**: `robot_logger --protocol robot_sweep` 으로 무접촉 스윕(D1) 기록 → `datasets.build` →
   `stages.baseline` → `stages.contact` (`data.kind: robot`). 결과를 `stage1.baseline_model` / `stage1.calibrator` 로.
4. **리타게팅** (`hand_mano` 정책): `retarget.tip_links` (손가락 → 끝 링크), `retarget.human_to_robot`
   (MANO 손 좌표계: 손가락 −x, 엄지쪽 +z, 손바닥 −y → 로봇 base 좌표계로의 3×3 회전; 합성 손의 값은
   `control.interfaces.SYNTHETIC_HAND_HUMAN_TO_ROBOT`), `scale: auto`(편 손 ↔ q = 0 에서 `estimate_scale`, 사람 벡터에
   곱하는 AnyTeleop 규약), `tip_offsets` (끝 링크 원점 → 실제 손가락 끝).
5. **안전 설정**: `closing_sign` (관절별 닫는 방향; 외전 관절은 0), `max_vel` 낮게 시작, watchdog 한계를 실제 센서
   주기에 맞게.
6. **실행**:
   ```python
   from robot_skin.stages import deploy
   cfg = deploy.load_stage_config("my_deploy.yaml")      # robot: my_hand, bundle, stage1.*, retarget.*, safety.*
   metrics = deploy.run(cfg, robot=MyHand(...), cameras={"ego": MyCamera(...)})
   ```
   `robot: <이름>` 인데 인스턴스를 넘기지 않으면 `NotImplementedError` 가 위 절차를 안내한다.
7. **검증 순서**: (a) `robot: fake` 로 같은 설정 실행, (b) 실제 핸드에서 정책 없이 baseline 캡처만 해 보고 ΔS 가
   0 근처인지, (c) 손으로 taxel 을 눌러 level 과 촉각 정지 확인, (d) 낮은 속도로 정책 실행.

## 8. RTX 5090 박스에서 실행

- 환경 확인: `python -m robot_skin.train.hardware` — sm_120 은 CUDA ≥ 12.8 빌드 torch 가 필요하다
  (`robot_skin/train/README.md` §3).
- 제어 루프는 **로봇이 연결된 머신에서** 돌린다. Tailscale 너머의 원격 GPU 로 매 틱 추론을 보내면 왕복 지연이
  5 ms 틱과 20 Hz 정책 주기를 쉽게 넘는다 — 5090 박스에 로봇(USB/이더넷)을 직접 붙이거나, 번들을 로봇 PC 로
  복사해서 그 PC 의 GPU 로 돌린다. Tailscale 은 번들·세션 로그 전송(rsync/ssh)용.
- 실행:
  ```bash
  python -m robot_skin.stages.deploy --config my_deploy.yaml \
      --set device=cuda --set hardware=rtx5090 --set duration_s=30
  ```
  `hardware` 는 프로파일의 환경 변수(`PYTORCH_CUDA_ALLOC_CONF` 등)만 적용하고, `device: auto` 면 프로파일 장치를 쓴다.
  첫 호출의 CUDA 초기화·커널 준비는 `latency.warmup` 과 시작 baseline 창 동안 끝난다.
- `metrics.json` 에서 `loop_hz_wall`(실제 처리량), `tick_p95_ms` (< 5 ms 목표), `overruns`, `latency_p95_ms` 를 본다.

## 9. 로그와 재사용

모든 실행은 RAW 세션 포맷(`acquisition.manifest`)으로 남는다: `pressure.npz`(raw, 채널 순서), `joint_state.npz`
(q, qd, names), `camera_<name>/`, `events.jsonl`(phases `baseline`/`calibration` → `no_contact`, `rollout` → `task`,
instruction, safety marker, success), `session.json`(kind robot, `layout: layout.yaml`, `meta.urdf: robot.urdf`,
`meta.deployment`: 번들·주기·지표·안전 요약), `layout.yaml`, `robot.urdf`, 그리고 틱별 사이드카 `deploy_log.npz`
(`tick_q`, `tick_q_des`, `tick_q_cmd`, `tick_level`, `tick_contact`, `policy_action`, `policy_q_target`,
`policy_inference_ms`, …; 매니페스트 스트림이 아니므로 전처리는 무시).

```bash
python -m robot_skin.datasets.build --raw robot_skin/runs/deploy/sessions --out robot_skin/data/processed
```

기본 `log.dataset: other` 라서 VTLA 학습 풀(`task`)에 자동으로 섞이지 않는다. 성공/실패 rollout 을 선호 학습
(`vtla/dpo.py`)에 쓰려면 `success` 를 기록하고 `dataset: task` 로 모은다.

## 10. 설정 요약 (`robot_skin/configs/stages/deploy.yaml`)

`robot`, `bundle`, `device`, `duration_s`, `control_hz`/`policy_hz` (null → 번들), `instruction`, `seed`,
`allow_bootstrap`, `realtime` (fake: 시뮬레이션 클록 vs 실시간), `layout`/`urdf`, `stage1.*`, `startup.*`,
`tactile.*` (전처리 pressure 설정과 같게), `fake_robot.*` (가상 물체 `object.angle/compliance_rad/palm`),
`cameras.*`, `retarget.*`, `hand_state.*`, `safety.*`, `log.*`, `latency.*`. 알 수 없는 키는 오류.

## 11. 한계 / TODO

- 비동기(백그라운드) 정책 추론 없음 — 추론이 5 ms 를 넘으면 overrun.
- 실제 핸드 드라이버, IMU 허브, ROS joint state 는 제공하지 않는다(프로토콜만).
- `FakeRobotHand` 의 스킨·물체 모델은 현상학적이며 실제 mk555 데이터에 맞춘 값이 아니다.
- 팔(손목 자세) 제어는 범위 밖 — `hand_mano` 행동의 손목 부분은 로그에만 남는다.
