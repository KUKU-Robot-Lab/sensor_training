# HANDOFF — robot_skin 프레임워크 인수인계 (클라우드 세션 → 로컬 Claude Code)

> 로컬 Claude Code 에게: 이 파일을 먼저 끝까지 읽고, 아래 "지금 할 일" 을 이어서 진행하세요.
> 사용자 목표: D1(모션: 비전 + IMU 손모양 + 촉각)·D2(물체 태스크 + 지시문) 데이터 취득 → 전처리 →
> 학습(stage 1–3) → VTLA → 로봇 제어. 학습은 Tailscale 로 연결된 RTX 5090 등 여러 GPU 에서 한다.

## 1. 현재 상태 (2026-09-25)

- `main` 에 merge 완료 (PR #43, merge commit `6eda1ca`): 모노레포(`deformable_sats/`, `common/`,
  `robot_skin/`) + 프레임워크 전체 (acquisition, datasets, pose, train, vision, language, action,
  baseline, contact, representation, vtla, control, transfer, stages, CLI `python -m robot_skin`) + docs.
- 테스트: 루트 `pytest` → legacy 250 passed (main 과 동일한 12 failed/4 errors: raw 데이터·zarr 3 관련),
  robot_skin + common 621 passed.
- 브랜치 `restructure/robot-skin`: merge 후 후속 작업용으로 최신 main 에서 다시 시작함.

## 2. 진행 중이던 작업 A — 최종 적대적 리뷰 수정 (클라우드 세션이 마무리 중)

6개 차원 리뷰어가 찾고 반박 검증자가 재현한 **51건** (high 8 / medium 13 / low 30):
`docs/dev/as_built/FINAL_REVIEW_FINDINGS.md`.

수정 상태 (클라우드 세션이 차원별로 직렬 수정 중; 완료분은 이 브랜치에 커밋됨):

| 차원 | 상태 |
|---|---|
| geometry | 수정 완료 (클라우드) |
| data | 수정 중 (클라우드) |
| online (control/safety) | 대기 |
| train (engine/hardware/sweep) | 대기 |
| vtla | 대기 |
| cli_docs | 대기 |

→ 클라우드 세션이 끝까지 마치면 이 표를 갱신해 push 한다. **로컬 세션은 이 커밋이 올라오기 전까지
아래 목록의 파일은 건드리지 말 것** (충돌 방지): `robot_skin/datasets/build.py`, `robot_skin/control/*`,
`robot_skin/train/*`, `robot_skin/vtla/*`, `robot_skin/__main__.py`, `robot_skin/stages/*`, docs/*.md(기존 파일).
클라우드 세션이 중단된 경우: FINAL_REVIEW_FINDINGS.md 의 남은 항목을 차원별로 재현 → 수정 → 회귀 테스트.

## 3. 지금 할 일 B — IMU + 비전 손동작 추종 (사용자 요구: 스텁 없이 전부 실제 동작)

명세: `docs/dev/SPEC_HANDTRACK.md` (소유권 단위 그룹 [KPIK] [CAM] [VBACK] [VIFUSE] [TELEOP+INTEG+DOCS]).

권장 순서 (충돌 없는 새 파일부터):
1. **[KPIK] + [CAM] — 지금 바로 시작 가능** (전부 새 파일):
   `robot_skin/pose/keypoints.py`, `robot_skin/pose/ik.py`, `robot_skin/eval/hand_metrics.py`,
   `robot_skin/vision/cameras.py`, `robot_skin/vision/calibration.py` + 테스트.
   (`datasets/synthetic.py` 의 `vision_keypoints` 옵션은 작업 A 의 data 수정이 올라온 뒤에.)
2. 작업 A 가 push 되면 `git pull` 후 **[VBACK]** (MediaPipe/HaMeR/WiLoR/키포인트파일 백엔드, 세션 일괄
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
