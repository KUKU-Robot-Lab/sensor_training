# robot_skin/ — 손 전체 촉각 스킨 (글러브 ⇄ 로봇 핸드)

```
acquisition ─▶ datasets.build ─▶ imu_pose ─▶ baseline ─▶ contact ─▶ pretrain ─▶ vtla ─▶ deploy
 (기록·동기화·QC)  (전처리 → Episode)  (IMU→MANO)  (무접촉 ΔS 예측)  (잔차 z·레벨·검출기)  (MAE 인코더)  (정책)  (로봇 제어)
                 pose(taxel pose) · train(학습 엔진·HW 프로파일) · vision/language · action(청크·리타겟) · transfer · eval
```

## 한 줄 명령 (`python -m robot_skin <command>`)

모든 하위 명령은 해당 모듈의 `main` / `load_stage_config` + `run` 에 그대로 위임한다. 설정을 받는 명령은 같은
플래그를 쓴다: `--config <yaml>` 과 `--set KEY=VALUE` (점 경로, 값은 YAML, 반복 가능; `preprocess`, `train`,
`pipeline`, `deploy`), `--hardware <rtx5090|rtx4090|rtx3090|a100|cpu|auto|profile.yaml>` (`train`, `pipeline`,
`deploy`; 프로파일 `env` 는 CUDA 초기화 전에 export; 우선순위 stage YAML `train` < 프로파일 `suggest.<stage>`/`train`
< `--set`). `record`, `postprocess`, `qc`, `env`, `sweep` 은 인자를 해당 모듈 CLI 에 그대로 넘긴다. 기본값(경로·stage
설정 파일·파이프라인 순서·split 정책)은 `configs/default.yaml` (`robot_skin.config.load_config`).
`python -m robot_skin <command> --help` 가 각 명령의 인자를 보여 준다.

| 명령 | 위임 대상 | 비고 |
|---|---|---|
| `record glove\|robot ...` | `acquisition.glove_logger` / `robot_logger` | `--protocol d1_motion\|d2_task\|robot_sweep`, `--fake`, `--dry-run` |
| `postprocess ...` | `acquisition.session` | 3-tap 동기화 → IMU 캘리브레이션 → QC 재실행 |
| `qc ...` | `acquisition.qc` | 세션 QC (`--write`, `--json`) |
| `synth` | `datasets.synthetic.generate_dataset` | 데모·dry run 용 합성 raw 세션 (한 장갑 공유) |
| `preprocess ...` | `datasets.build` | raw 세션 → Episode (`--raw`, `--out`, `--force`) |
| `splits` | `datasets.splits` | processed root 전체에 대한 splits.json 하나 (기본 subject 단위) |
| `train <stage>` | `stages.<stage>` | `imu_pose\|baseline\|contact\|pretrain\|vtla`; `torchrun` 지원 |
| `pipeline` | 위 stage 들 | splits(1회) → imu_pose → baseline → contact → pretrain → vtla, 재개 가능 |
| `deploy` | `stages.deploy` | `--set bundle=<runs>/vtla`; `robot: fake` = 시뮬레이션 |
| `env` / `sweep ...` | `train.hardware` / `train.sweep` | GPU·torch 점검 / 하이퍼파라미터 스윕 |

```bash
# 합성 데이터로 전체 경로 확인 (4 코어 CPU VM 에서 실행 확인, 합쳐 약 30 초)
R=/tmp/rs
python -m robot_skin synth --out $R/raw
python -m robot_skin preprocess --raw $R/raw --out $R/processed
python -m robot_skin pipeline --processed $R/processed --out $R/runs --hardware cpu \
    --set train.max_steps=30 --set train.warmup_steps=5 --set vtla.image.image_size='[24, 32]'
python -m robot_skin deploy --set bundle=$R/runs/vtla --set duration_s=2 --set out_dir=$R/deploy

# 실제 데이터 (RTX 5090 한 대 / 두 대)
python -m robot_skin pipeline --hardware rtx5090          # data/processed → runs/<stage>
torchrun --standalone --nproc_per_node=2 -m robot_skin train vtla --hardware rtx5090 --set data.splits=robot_skin/runs/splits.json
```

문서: 설계 [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) · 학습 실행 [`docs/TRAINING.md`](../docs/TRAINING.md) ·
정책 모델 [`docs/VTLA.md`](../docs/VTLA.md) · 수집 [`docs/DATA_ACQUISITION.md`](../docs/DATA_ACQUISITION.md) ·
포맷 [`docs/DATA_FORMAT.md`](../docs/DATA_FORMAT.md) · 배포 [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) ·
참고문헌 [`docs/REFERENCES.md`](../docs/REFERENCES.md).

**pipeline** (`robot_skin/__main__.py` `run_pipeline`)
- `<out>/splits.json` 을 한 번 만들고(`make_splits`, `configs/default.yaml` `pipeline.splits`; `--splits` 로
  기존 파일 사용) **모든 stage 의 `data.splits`** 로 넘긴다 → 어떤 stage 도 다른 stage 의 val/test 에피소드로
  학습하지 않는다. 다시 나누려면 파일을 지우고 `--force`.
- 각 stage: `out_dir = <out>/<stage>`, `data.processed_root` = `--processed`, 실제 설정은
  `<out>/<stage>/pipeline_config.yaml`, 실행 기록(설정·splits sha256·연결·소요 시간)은 `<out>/pipeline.json`.
- 연결: baseline 의 derived `residual`/`baseline_logvar` → contact; contact 의 `residual_z`/`contact_level`
  → pretrain·vtla (`data.tactile_source: derived` — bootstrap 대체 경로 금지); `calibrator.json`·`baseline_model.pt`
  → policy bundle 의 stage-1 참조(`tactile.calibrator` / `tactile.baseline_model`); `encoder_state.pt` →
  `tactile.pretrained`.
- `--set KEY=V` 는 그 키를 가진 모든 stage 에, `--set <stage>.KEY=V` 는 한 stage 에만 (stage 지정이 우선).
  어느 stage 에도 없는 키는 오류.
- 재개: `metrics.json` + 주 산출물이 있고 같은 processed root·splits 로 학습된 stage 는 건너뛴다(`--force` 로
  재학습). 한 stage 가 다시 돌면 그 뒤 stage 도 모두 다시 돈다(입력이 바뀌었으므로).

## 모듈

| 모듈 | 구현 | 스텁 / 미구현 |
|---|---|---|
| `hardware/` | 장비 문서 (코드 없음) | — |
| `geometry/` | 회전 유틸: 쿼터니언(wxyz)·axis-angle·행렬·6D(첫 두 열) 변환, URDF rpy, 측지 거리, 4×4 변환 | — |
| `config.py`, `configs/` | `load_config`/`deep_merge`; `default.yaml`(경로·stage 설정 파일·파이프라인), `stages/*.yaml`, `hardware/*.yaml` | — |
| `acquisition/` | 프로토콜(D1 motion, D2 task, robot_sweep), Recorder, 3-tap 동기화·드리프트, IMU 평손 캘리브레이션, QC, `--fake`/`--dry-run` 로거 CLI, `SerialPressureSource`(pyserial), `CameraSource`(cv2, 이 환경에서 미시험) | IMU 허브 `ImuSource`, ROS `RosJointStateSource`, mk555 `.bin` 파서(bin_merge 주입) |
| `datasets/` | `build`(전처리, 200 Hz Episode), `splits`, `stats`, D1 윈도 데이터셋(`motion`), 합성 세션(`synthetic`) — [`docs/DATA_FORMAT.md`](../docs/DATA_FORMAT.md) | — |
| `pose/` | MANO 스켈레톤·self-touch, URDF FK, IMU→손자세 모델(`imu_model`), `glove_imu2mano`, 비전 라벨 입출력·스무딩 | VIFNet-S 로더, HaMeR 추정기(오프라인 실행 안내) |
| `baseline/` | 시간 모델 `TemporalBaselinePredictor`(평균+분산, Kendall & Gal), 인과 스트림, v1 `BaselinePredictor` | — |
| `contact/` | `ResidualCalibrator`(z·레벨), `ContactDetector`(focal), 히스테리시스, D2 pseudo label, `OrdinalQuantizer`, `SaturationFSM`, self-touch | — |
| `representation/` | `TaxelTokenizer`, `TaxelEncoder`, `tactile_value_features`(단일 특징 함수), MAE 식 마스킹 사전학습 | — |
| `vision/`, `language/` | Tiny/ResNet/HF 비전 인코더, 변환, feature cache; hashing/HF 텍스트 인코더 | (선택 의존성: torchvision, transformers) |
| `action/` | MANO 54-D / robot_joint 행동 공간, 정규화, ACT 청크·시간 앙상블, 손끝 리타겟팅 | — |
| `vtla/` | `VTLAPolicy`(언어·비전·촉각·proprio 융합), chunk / flow-matching head, 데이터셋, `policy_bundle.pt` | DPO 선호쌍 생성 파이프라인 |
| `control/` | `FakeRobotHand`, `OnlineTactileProcessor`(오프라인과 동일), `SafetyFilter`, `PolicyRunner` + 배포 세션 로그, 지연 측정 — [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md) | 실제 로봇 드라이버(`RobotHandInterface` 구현 필요) |
| `transfer/` | 스켈레톤 사영, `align_layouts`, `map_taxel_values`, `RobotToManoEstimator` | — |
| `train/` | `Trainer`(AMP·DDP·EMA·재개), 하드웨어 프로파일, 스윕 — [`train/README.md`](train/README.md) | — |
| `stages/` | `imu_pose`, `baseline`, `contact`, `pretrain`, `vtla`, `deploy` 러너 (`run(cfg) -> metrics`) | — |
| `policy/`, `sim/`, `eval/` | 관측 ablation 빌더, per-taxel 도메인 랜덤화, 지표(환각률·분리도·포화 복구) | RL 학습, touch-grid 환경 |

설정: `configs/default.yaml`(최상위: 경로·stage 설정 파일·파이프라인), `configs/stages/<stage>.yaml`(각 stage 가
직접 읽고 키를 검증), `configs/hardware/*.yaml`. 데이터·산출물: `data/`, `runs/` (git-ignored).
인용은 [`docs/REFERENCES.md`](../docs/REFERENCES.md) 에 있는 논문만.
`deformable_sats` 는 import 하지 않는다 — 공유 규약은 전부 `common/`.
