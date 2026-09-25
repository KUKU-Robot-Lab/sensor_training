# control/ — 학습된 VTLA 정책을 로봇 핸드에서 실행 (촉각 스킨 포함 폐루프)

운영 절차(시작, 안전, 지연 예산, 실제 핸드 연결, RTX 5090 실행)는 `docs/DEPLOYMENT.md`. 여기는 코드 지도다.

```
RobotHandInterface ─ read_state / read_pressure ─▶ OnlineTactileProcessor  (≡ 오프라인 stage 1)
CameraInterface ─ read ─┐                              │ residual z · level · features · contact
                        ▼                              ▼
PolicyRunner ── make_observation / collate_vtla ─▶ VTLAPolicy.predict ─▶ TemporalEnsembler
     │  hand_mano: hand_action_fingertips → FingertipRetargeter │ robot_joint: 그대로
     └─▶ SafetyFilter ─▶ send_joint_targets        DeploymentLogger ─▶ RAW 세션 (datasets.build 재투입)
```

| 모듈 | 내용 |
|---|---|
| `interfaces.py` | `RobotHandInterface` / `CameraInterface` 프로토콜, `check_robot`, `FakeRobotHand`(URDF FK + 1차 추종 + 가상 물체 + 합성 스킨: 움직임 artefact·누름·노이즈·ADC 레일·드롭아웃 주입, `truth()`), `FakeCamera`, `urdf_tip_fk`, `SYNTHETIC_HAND_HUMAN_TO_ROBOT` |
| `online.py` | `OnlineTactileProcessor` (raw → ΔS → `CausalBaselineStream` → residual → `SaturationFSM` → `ResidualCalibrator` z/level → `TactileHistory` 특징 → contact 규칙, 선택: `CausalDetectorStream` + `HysteresisFilter`), `CausalJointVelocity`(= `datasets.build.joint_velocity` savgol_causal), 자세 함수(`glove_pose_fn`/`robot_pose_fn`), 시작 baseline 캡처, `startup_calibrator`, `replay_episode` |
| `bundle.py` | `load_policy_bundle` → `PolicyBundle` (`vtla.bundle_components` 래핑, `calibrator()`, `baseline_model_path()`, `stage1_matches(layout)`(번들의 stage-1 참조가 이 스킨 것인지), `taxel_frames`(촉각 인코더가 학습한 taxel pose 프레임: `mano_wrist` / `urdf_root`; 옛 번들은 `tactile.layouts` 로 추정), `check_deployable` — bootstrap 촉각으로 학습된 번들 거부) |
| `runner.py` | `PolicyRunner` (시작: 정지·무접촉 baseline, 선택적 bring-up 보정 → rollout: 200 Hz 틱, 20 Hz 정책, ACT 앙상블, 선형 보간, 안전 필터, 속도 유지·overrun 집계; `taxel_frame="auto"`: 정책 관측의 taxel pose 를 학습 프레임으로 옮김 — 아래), `DeploymentLogger` (acquisition `Recorder` 위에서 RAW 세션 기록), `initial_hand_state`, `joint_permutation` |
| `safety.py` | `SafetyFilter` (e-stop 래치 → 센서 watchdog(스트림별 마지막 타임스탬프를 매 틱 검사) → 비유한 목표 → 촉각 정지(STRONG/SATURATED 지속 → 닫는 방향 동결, 해당 체인만; 가속도 제한보다 우선) → 관절 한계(밴드 밖 시작은 속도 제한으로 진입) → 속도 → 가속도; 속도·가속도는 직전 명령 이후의 실제 시간 기준(최대 `dt`); `reset` 은 비유한 자세를 거부해 명령이 NaN 이 되지 않는다), `taxel_joint_mask`, `closing_signs` |
| `latency.py` | `LatencyMeter`, `percentile_summary`, `benchmark_policy`(p50/p95, CUDA 동기화), `example_batch`, `export_torchscript`(가드됨) |

Stage runner: `robot_skin/stages/deploy.py` (`configs/stages/deploy.yaml`).

## 오프라인 ≡ 온라인 (핵심 계약)

`OnlineTactileProcessor` 는 학습 데이터를 만든 함수를 **그대로** 틱 단위로 호출한다:
`common.signal.relative_change` / `saturation_mask` (전처리와 같은 레일·|ΔS|≥90 % 규칙, 죽은 채널 = ΔS 0 + 포화),
`datasets.build.joint_velocity` 의 causal Savitzky–Golay 계수, `pose.mano.taxel_poses_from_hand`
(go = 0, 손 좌표계) / `pose.robot_fk.taxel_poses_from_joints`, `baseline.temporal.CausalBaselineStream`,
`contact.calibration.ResidualCalibrator` + `SaturationFSM` (calibrator.fsm), `representation.TactileHistory`,
`vtla.contact_from_level`. `tests/test_control.py` 가 처리된 합성 에피소드(로봇·글러브, 드롭아웃 주입)를
틱마다 흘려 `predict_episode` → `residual_levels` → `TactileFeatureSpec.from_arrays` 와 모든 중간값이
같음을 확인한다 (ΔS·포화·레벨·미신뢰·히스테리시스는 비트 단위, q̇·baseline 출력·z·특징·검출 확률은 1e-4 이내).

남는 차이(설계상): 압력은 온라인에서 최신 샘플(zero-order hold), 오프라인은 마스터 격자 선형 보간
(+ 레일 샘플 인접 프레임 포화 처리) — 프런트엔드를 200 Hz 로 읽으면 무시할 수준. 글러브 손 라벨의
비인과 평활은 온라인에 없다.

**taxel pose 프레임**: 처리기는 스킨 자신의 프레임(로봇: URDF 루트)으로 pose 를 계산한다 — 로봇 baseline 모델이
학습한 프레임이다. 정책의 촉각 인코더는 **자기가 학습한 프레임**의 pose 를 받아야 한다(`PolicyBundle.taxel_frames`,
vtla stage 가 기록). 글러브 D2 로 학습한 번들(`mano_wrist`: 손가락 −x, 손바닥 −y)을 URDF 스킨(합성 손: 손가락 +z,
패드 +x)에서 돌리면 `PolicyRunner` 가 정책 틱마다 리타게터의 역변환 `transfer.RobotToManoTaxelFrame`
(`p ↦ R_hrᵀ R_bᵀ (p − t_b) / scale`, `n ↦ R_hrᵀ R_bᵀ n`)으로 옮겨 넣는다(`taxel_frame_info`, deploy `notes`).
리타게터가 없는 불일치(예: 글러브 pose 로 학습한 `robot_joint` 번들)는 경고하고, `taxel_frame=` 에 함수
`(pos, nrm, q_robot) → (pos, nrm)` 를 주거나 `None` 으로 끌 수 있다. 사상의 정확도는 `human_to_robot`/`scale`
과 두 손의 형태 차이에 달려 있다(합성 템플릿: 손끝 7–47 mm, 손바닥 ≤ 9 mm; 변환 전 57–266 mm).

**죽은 채널**: 번들이 `tactile.mask_dead_taxels` (vtla `data.mask_dead_taxels`, 기본 켬)로 학습됐으면 `PolicyRunner` 가
처리기의 죽은 채널(`processor.dead`, 시작 baseline 에서 레일에 붙은 채널)을 `make_observation(taxel_mask=...)` →
`taxel_pad` 로 정책 입력에서 가린다 — 학습 데이터셋이 `meta.preprocessing.dead_taxels` 를 가린 것과 같은 규칙이라,
늘 SATURATED 인 채널이 `level_ge_weak` ContactGate 를 열어 두지 못한다(안전 필터의 촉각 정지는 그대로 본다).

**flow 헤드 난수**: `PolicyRunner` / `benchmark_policy` 는 `torch.Generator(device=<정책 장치>)` 를 넘기고, flow 헤드는
노이즈를 그 생성기의 장치에서 뽑아 모델 장치로 옮긴다 — GPU 에서도 flow 번들이 돈다.

관절 순서: 처리기의 `q` 는 `joint_names` 순서 — baseline 모델의 `bundle_meta["joint_names"]`, 모델이 없으면
URDF 구동 관절 순서(전처리도 로봇 q 를 URDF 순서로 바꾼다). `PolicyRunner` 가 드라이버 순서를 이름으로 바꿔 넣는다.
calibrator 는 자기가 맞춰진 residual 과 짝이다: `use_logvar` calibrator 는 baseline 모델 없이 거부된다.

## 빠른 사용

```python
from robot_skin.control import (FakeRobotHand, FakeCamera, OnlineTactileProcessor, PolicyRunner,
                                SafetyFilter, load_policy_bundle, taxel_joint_mask)

b = load_policy_bundle("robot_skin/runs/vtla")                       # policy_bundle.pt
hand = FakeRobotHand(obj={"angle": 0.9})                              # 또는 실제 RobotHandInterface
proc = OnlineTactileProcessor.from_stage_outputs(hand.layout, baseline="runs/robot_baseline",
                                                 calibrator="runs/robot_contact", urdf=hand.model,
                                                 feature_spec=b.feature_spec, contact_rule=b.contact_rule)
safety = SafetyFilter(hand.lower, hand.upper, dt=1 / 200, max_vel=3.0, tactile_stop={},
                      taxel_joints=taxel_joint_mask(hand.layout, hand.model), joint_names=hand.joint_names)
runner = PolicyRunner(hand, b, proc, cameras={"ego": FakeCamera("ego", hand)}, safety=safety,
                      instruction="pick up the cup")
metrics = runner.run(5.0, baseline_s=1.0)        # loop_hz, latency_p50_ms/p95, safety_counts, …
```

CLI: `python -m robot_skin.stages.deploy --set bundle=robot_skin/runs/vtla --set duration_s=10`.
실제 로봇은 `run(cfg, robot=MyHand(), cameras={...})` — 드라이버가 없으면 `NotImplementedError` 로 구현 방법을 안내한다.

## 한계 / TODO

- 정책 추론은 제어 루프 안에서 **동기**로 돈다(학습과 같은 타이밍, 결정적). CPU 에서 기본 VTLA 는 5 ms
  틱을 넘겨 정책 틱마다 overrun 이 생긴다 → GPU 사용, 또는 비동기 추론 스레드(미구현). 밀린 틱은 이어서 실행되며
  처리기에는 예정 시각으로 보간한 샘플이 들어가고(`catchup_ticks`), 명령은 실제 시간 기준 `max_vel` 로 제한된다 —
  안전은 지켜지지만 추론 동안 명령이 없어 손이 느려진다(`docs/DEPLOYMENT.md` §6).
- 실패 처리: 비유한 관절 읽기는 직전 유한 값 유지(처리기·proprio·taxel 자세), 시작 읽기는 유한해질 때까지 재시도,
  비유한 chunk 는 버림(`nonfinite_chunk`), `run()` 중 예외 → e-stop·유지·로그 닫기·`robot.stop()` 후 다시 던짐
  (`PolicyRunner.abort`). 시작 baseline 창 내내 레일에 붙은 채널은 죽은 채널로 표시된다.
- `hand_mano` 정책의 proprio 는 로봇에 사람 손 상태가 없어 **명령한** 손 행동을 쓴다(기본). 손가락 부분을
  로봇 q 에서 추정하려면 `transfer.RobotToManoEstimator` (`hand_state.estimate: true`). 손목 자세는 팔이 없으면
  관측 불가 — 팔 제어기는 범위 밖.
- 실제 IMU 기반 글러브 텔레옵(사람 손 → 로봇) 경로는 여기서 다루지 않는다(정책 실행만).
