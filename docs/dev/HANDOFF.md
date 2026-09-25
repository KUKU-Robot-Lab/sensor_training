# HANDOFF — robot_skin 프레임워크 인수인계 (클라우드 세션 → 로컬 Claude Code)

> 로컬 Claude Code 에게: 이 파일을 먼저 끝까지 읽고, 아래 "지금 할 일" 을 이어서 진행하세요.
> 사용자 목표: D1(모션: 비전 + IMU 손모양 + 촉각)·D2(물체 태스크 + 지시문) 데이터 취득 → 전처리 →
> 학습(stage 1–3) → VTLA → 로봇 제어. 학습은 Tailscale 로 연결된 RTX 5090 등 여러 GPU 에서 한다.

## 1. 현재 상태 (2026-09-25)

- `main` 에 merge 완료 (PR #43, merge commit `6eda1ca`): 모노레포(`deformable_sats/`, `common/`,
  `robot_skin/`) + 프레임워크 전체 (acquisition, datasets, pose, train, vision, language, action,
  baseline, contact, representation, vtla, control, transfer, stages, CLI `python -m robot_skin`) + docs.
- 테스트: 루트 `pytest` → legacy 250 passed (main 과 동일한 12 failed/4 errors: raw 데이터·zarr 3 관련),
  robot_skin + common 695 passed (최종 리뷰 수정 반영 후).
- 브랜치 `restructure/robot-skin`: merge 후 후속 작업용으로 최신 main 에서 다시 시작함.

## 2. 작업 A — 최종 적대적 리뷰 수정 (완료)

6개 차원 리뷰어가 찾고 반박 검증자가 재현한 **51건** (high 8 / medium 13 / low 30):
`docs/dev/as_built/FINAL_REVIEW_FINDINGS.md`.

수정 상태: **완료 (클라우드 세션, 이 브랜치에 커밋됨)** — 48건 수정 + 2건은 앞선 수정에 이미 포함(CTRL-4, CLI-3),
1건 보류(DP-11, low: `--fake` 취득 장면의 누름 신호와 기하 self-touch 라벨 불일치 — 테스트 픽스처 충실도 문제).
각 수정에는 회귀 테스트가 붙어 있음 (robot_skin + common: 695 passed; legacy 결과는 main 과 동일).

주요 수정: glove 학습 정책의 로봇 배포 taxel 좌표계(URDF root → MANO wrist 변환), `hand_pose.npz` 전처리 누락,
events.jsonl 수정이 라벨에 반영, 안전필터 NaN 전파·카메라 watchdog·가속 제한 vs 촉각 정지, sweep/hardware
프로파일 덮어쓰기, resume=auto 오동작, CUDA 에서 flow head 노이즈 디바이스, pipeline 하위 단계 재실행 누락,
전처리 버전 `robot_skin.datasets.build/3` (기존 processed 데이터는 `--force` 로 재빌드 권장).

→ 로컬 세션: 이제 HANDOFF §3 의 모든 파일을 수정해도 된다 (`git pull --rebase` 후).

## 3. 지금 할 일 B — IMU + 비전 손동작 추종 (사용자 요구: 스텁 없이 전부 실제 동작)

명세: `docs/dev/SPEC_HANDTRACK.md` (소유권 단위 그룹 [KPIK] [CAM] [VBACK] [VIFUSE] [TELEOP+INTEG+DOCS]).

권장 순서 (충돌 없는 새 파일부터):
1. **[KPIK] + [CAM] — 지금 바로 시작 가능** (전부 새 파일):
   `robot_skin/pose/keypoints.py`, `robot_skin/pose/ik.py`, `robot_skin/eval/hand_metrics.py`,
   `robot_skin/vision/cameras.py`, `robot_skin/vision/calibration.py` + 테스트.
   (`datasets/synthetic.py` 의 `vision_keypoints` 옵션은 작업 A 의 data 수정이 올라온 뒤에.)
2. (작업 A 완료 — `git pull --rebase` 후 바로) **[VBACK]** (MediaPipe/HaMeR/WiLoR/키포인트파일 백엔드, 세션 일괄
   추출 → `hand_pose.npz`, viz) 와 **[VIFUSE]** (시각-관성 융합, 온라인 추적 `CausalImuPoseStream` /
   `OnlineHandTracker`, VIFNet-S 범용 어댑터 `ExternalImuPoseModel`, imu_pose stage 확장).
3. **[TELEOP+INTEG+DOCS]**: `control/teleop.py`, CLI `handpose`/`track`/`teleop`, `docs/HAND_TRACKING.md`,
   e2e 테스트.

5090 환경의 이점: `pip install mediapipe` 및 HaMeR/WiLoR 설치·가중치 다운로드가 가능하므로 어댑터를
**실제 모델로도** 검증할 것 (이 저장소 테스트는 CPU·모의 출력 기준으로 유지).

## 4. 반드시 지킬 규약

- 의존 방향: `robot_skin → common ← deformable_sats`. `common/`, `deformable_sats/` 수정 금지
  (`common/tests/test_dependency_direction.py`).
- ΔS% = (raw − baseline)/baseline × 100 — **SATS 규약, 누르면 음수** (`common.signal.PRESS_SIGN=-1`,
  `press_intensity = −ΔS`). 쿼터니언 wxyz, 6D = 회전행렬 앞 두 열, MANO 관절 순서, taxel pose 는 **손 좌표계**
  (손목 원점, global_orient=0) 로 저장.
- baseline 모델에 관측 ΔS 를 입력으로 넣지 말 것 (`sats/bending` 교훈: 접촉까지 지운다).
- `representation.encoder.tactile_value_features` 가 촉각 특징의 단일 출처 (pretrain·VTLA·control 공용).
- 오프라인 stage 와 온라인 control 은 수치적으로 일치해야 함 (테스트로 고정).
- 테스트: CPU, 결정적(seed), 빠르게, 선택 의존성(mediapipe, cv2, torchvision, transformers …)은 guarded import
  + 모의 테스트. 인용은 `docs/REFERENCES.md` 에 검증된 것만.
- 전체 테스트: `python -m pytest -q -p no:cacheprovider robot_skin/tests common/tests` (루트 `pytest` 는 legacy 포함).

## 5. 참고 문서

| 파일 | 내용 |
|---|---|
| `docs/dev/SPEC_FRAMEWORK.md` | 프레임워크 원 설계 명세 (모듈 소유권·계약). 일부는 as-built 로 대체됨 |
| `docs/dev/as_built/WAVE1_API.md`, `WAVE2_API.md` | 각 모듈 구현자/리뷰어 보고: 실제 API, 명세와의 차이, 남은 우려 |
| `docs/dev/as_built/FINAL_REVIEW_FINDINGS.md` | 최종 리뷰 확인 결함 51건 |
| `docs/dev/SPEC_HANDTRACK.md` | 손동작 추종 구현 명세 (작업 B) |
| `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, `docs/VTLA.md`, `docs/DEPLOYMENT.md`, `docs/DATA_ACQUISITION.md`, `docs/DATA_FORMAT.md`, `docs/REFERENCES.md` | 사용자 문서 |

## 6. 알려진 한계 (ARCHITECTURE.md §8 요약)

- MANO 축/rest 골격은 실제 MANO 파일로 미검증 (`ManoSkeleton.from_mano_pkl` 로 확인 필요).
- VIFNet-S 입출력 형식 미확인 (VIHand, ACM MM 2025 — 공개 코드 확인 후 어댑터 설정).
- baseline σ 는 aleatoric 만 — D2 에서 OOD 자세 오탐 가능 (D1 에 air-grasp 블록 추가됨, 앙상블은 미구현).
- 제어 루프의 정책 추론은 200 Hz tick 안에서 동기 실행 (GPU 필요).
- GPU 경로(bf16, DDP/NCCL, compile), 사전학습 가중치 다운로드는 CPU·오프라인 환경에서 미검증 → 5090 에서
  `python -m robot_skin env` 와 `torchrun --standalone --nproc_per_node=1 -m robot_skin train ...` 로 먼저 확인.
- 관련 선행연구: Yu et al. 2026, *Pose-Aware Modeling to Mitigate Pose-Related Artifacts in Tactile Gloves*
  (arXiv:2607.22964) — D1 baseline 과 같은 문제 설정. 차별점 정리는 ARCHITECTURE.md.
