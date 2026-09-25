# sensor_training (monorepo)

기압 기반 촉각 센서 연구 저장소. 평면 SATS 패드 연구(`deformable_sats/`)와 손 전체 촉각 스킨(`robot_skin/`)이
공유 규약 계층(`common/`)을 통해 함께 산다. `robot_skin` 은 촉각 글러브(기압 taxel + IMU 7개 + 카메라)로 모은
사람 손 데이터에서 **vision + tactile + language → action (VTLA)** 정책을 학습해 촉각 스킨 로봇 핸드를 제어하는
프레임워크다: 수집 → 전처리 → 촉각 해석(무접촉 baseline, 접촉 보정) → 촉각 표현 사전학습 → VTLA → 로봇 제어.

| 디렉터리 | 역할 |
|---|---|
| `robot_skin/` | 글러브 ⇄ 로봇 핸드 촉각 스킨 프레임워크 — 모듈 표는 [`robot_skin/README.md`](robot_skin/README.md) |
| `common/` | 공유 규약: ΔS% 신호·baseline·정규화·포화(`signal`), 스트림 시계 정렬(`timeline`), taxel 레이아웃(`layouts`) — [`common/README.md`](common/README.md) |
| `deformable_sats/` | **기존 저장소 전체**(sats, hitmap, cnn_lstm, scripts, skin_ws, learning_data, runs, history). 4×4 SATS 압력맵·XY/Z/Fz 회귀, 밴딩 보상, 논문 워크스페이스. 내부 구조·import 무변경 — [`deformable_sats/README.md`](deformable_sats/README.md) |
| `docs/` | 설계·운영 문서 (아래) |

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 목표(D1 motion / D2 task → VTLA → 로봇), 의존 방향, 전체 데이터 흐름, 핵심 설계 결정, 논문 매핑, 한계와 다음 단계 |
| [`docs/DATA_ACQUISITION.md`](docs/DATA_ACQUISITION.md) | 수집 운영 절차: 장비, 3-탭 싱크, D1/D2 스크립트, QC 게이트, 권장 수집량 |
| [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) | raw 세션과 processed Episode 의 모든 파일·키·단위·좌표 규약 |
| [`docs/TRAINING.md`](docs/TRAINING.md) | stage·파이프라인 실행, 설정, 하드웨어 프로파일(RTX 5090), torchrun, Tailscale 다중 머신, 재개, 스윕, 평가 |
| [`docs/VTLA.md`](docs/VTLA.md) | VTLA 모델 구조, 데이터 샘플링, 정책 번들, 논문과의 관계, VLM 백본 확장 |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | 정책을 로봇 핸드에서 실행: 제어 루프, 안전, 지연 예산, 실제 핸드 연결 |
| [`docs/REFERENCES.md`](docs/REFERENCES.md) | 검증된 참고문헌 — 코드·문서는 여기 있는 문헌만 인용한다 |
| [`robot_skin/train/README.md`](robot_skin/train/README.md) | 학습 엔진 세부: Trainer, precision 규칙, DDP, 체크포인트, 스윕 |

## 의존 방향

```
robot_skin  ──▶  common  ◀──  deformable_sats
```

`common` 은 두 패키지 어느 쪽도 import 하지 않고, `robot_skin` 은 `deformable_sats` 를 import 하지 않는다
(`common/tests/test_dependency_direction.py`).

## 빠른 시작

```bash
pip install -r requirements.txt          # = deformable_sats/requirements.txt (torch 2.9.0+cu128 고정) + PyYAML, pytest

# robot_skin + common 테스트 (CPU, 결정적; 루트 pytest.ini: pythonpath = . deformable_sats)
python -m pytest -q -p no:cacheprovider robot_skin/tests common/tests
# 저장소 전체 (deformable_sats 테스트 포함 — 아래 "알려진 테스트 실패")
pytest
```

### 하드웨어 없이 robot_skin 전체 경로 (합성 데이터)

```bash
R=/tmp/rs
python -m robot_skin synth --out $R/raw                     # 합성 glove raw 세션 8개 (D1 4 + D2 4)
python -m robot_skin preprocess --raw $R/raw --out $R/processed
python -m robot_skin pipeline --processed $R/processed --out $R/runs --hardware cpu \
    --set train.max_steps=30 --set train.warmup_steps=5 --set vtla.image.image_size='[24, 32]'
python -m robot_skin deploy --set bundle=$R/runs/vtla --set duration_s=2 --set out_dir=$R/deploy
```

`pipeline` 은 `splits.json` 하나를 만들어 imu_pose → baseline → contact → pretrain → vtla 를 순서대로 학습하고
산출물을 이어 준다(`$R/runs/<stage>/`, `$R/runs/pipeline.json`). `deploy` 는 정책 번들을 가짜 로봇 핸드 + 가짜 카메라로
200 Hz 폐루프 실행하고 `$R/deploy/metrics.json` 과 다시 전처리할 수 있는 raw 세션을 남긴다. 네 명령 모두 이 저장소의
4 코어 CPU VM(GPU 없음)에서 실행해 확인했고 합쳐 약 30 초 걸렸다. 모델이 작고 30 step 이라 지표에는 의미가 없다 —
배관 점검이다. 단계별 설명은 [`docs/TRAINING.md`](docs/TRAINING.md) §1.3.

### 실제 데이터

```bash
python -m robot_skin record glove --protocol d1_motion --subject S01 --dry-run   # 계획·운영자 대본만 (장비 없이)
python -m robot_skin record glove --protocol d1_motion --subject S01 --fake --time-scale 0.05   # 합성 소스로 end-to-end
                                                         #   → robot_skin/data/synthetic (실제 raw 와 분리)
python -m robot_skin preprocess                          # robot_skin/data/raw → robot_skin/data/processed
python -m robot_skin env                                 # GPU/torch 점검, 추천 하드웨어 프로파일
python -m robot_skin pipeline --hardware rtx5090         # → robot_skin/runs/<stage>/
python -m robot_skin deploy --set bundle=robot_skin/runs/vtla
```

`record` 는 `--out`/`--root` 를 생략하면 `configs/default.yaml` `paths.raw_root`(`robot_skin/data/raw/<dataset>/<subject>/…`)
에 쓴다. `--dry-run` 은 계획만 `…/dry_run/session.json` 으로 남기고, `preprocess` 는 이런 계획을 `plan` 으로 보고하고
건너뛴다. `--fake` 세션은 전부 합성이라 `paths.synthetic_root`(`robot_skin/data/synthetic/…`)로 가서 실제 데이터와 섞이지
않는다 — 경로 점검용으로 전처리하려면 `python -m robot_skin preprocess --raw robot_skin/data/synthetic --out /tmp/fake_proc`
(episode 의 `meta.preprocessing.synthetic` 에 출처가 남고, 합성과 실제 세션을 함께 전처리하면 경고한다). 앞의 세 줄과
`env` 는 이 VM 에서 실행해 확인했다.
실제 장비 입력(IMU 허브, ROS joint state, 로봇 핸드 드라이버, mk555 `.bin` 로더)은 아직 인터페이스만 있다 —
[`docs/DATA_ACQUISITION.md`](docs/DATA_ACQUISITION.md), [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) §7.

### 공유 규약 (Python)

```python
from common.signal import relative_change, estimate_baseline
from common.layouts import load_layout
from robot_skin.contact import OrdinalQuantizer

layout = load_layout("glove_template")          # 손끝 5 + 손바닥 2×2, parent = MANO 세그먼트, IMU 7개
delta = relative_change(raw, estimate_baseline(raw, n_samples=200))   # SATS 규약 ΔS% (누르면 음수)
levels = OrdinalQuantizer(weak_pct=3, strong_pct=15)(delta)          # 0 무접촉 / 1 약 / 2 강 / 3 포화
```

규약: ΔS SATS 부호(누르면 음수; 접촉 로직은 `press_intensity = −ΔS`), 쿼터니언 wxyz, 6D 회전 = 회전행렬 첫 두 열,
MANO 관절 순서, 시간 s, 위치 m.

### 기존 SATS 워크플로

```bash
cd deformable_sats                       # 커맨드는 예전과 동일, 디렉터리만 이동
python -m sats.training.train_e2e --help
```

## 알려진 테스트 실패 (main 과 동일, 이동과 무관)

- `deformable_sats/sats/training/tests/test_phase0_data_layout.py` — git-ignored raw 데이터 필요
- `deformable_sats/hitmap/tests/test_zarr_index_resolution.py`, `test_depth_contract.py` — zarr 2.16 고정,
  zarr 3 환경에서는 실패
