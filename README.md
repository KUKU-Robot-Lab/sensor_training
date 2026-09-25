# sensor_training (monorepo)

기압 기반 촉각 센서 연구 저장소. 평면 SATS 패드 연구(`deformable_sats/`)와 손 전체 촉각 스킨
(`robot_skin/`)이 공유 규약 계층(`common/`)을 통해 함께 산다.

| 디렉터리 | 역할 |
|---|---|
| `deformable_sats/` | **기존 저장소 전체**(sats, hitmap, cnn_lstm, scripts, skin_ws, learning_data, runs, history). 4×4 SATS 압력맵·XY/Z/Fz 회귀, 밴딩 보상, 논문 워크스페이스(`history/fig_data/`). 내부 구조·import 무변경 |
| `robot_skin/` | 글러브 ⇄ 로봇 핸드 촉각 스킨: 취득, taxel pose, 무접촉 baseline 예측, 접촉 해석, 토큰 표현, 정책/VTLA, sim, 평가 |
| `common/` | 공유 규약: ΔS% 신호·baseline·정규화·포화(`signal`), 스트림 시계 정렬(`timeline`), taxel 레이아웃(`layouts`) |
| `docs/ARCHITECTURE.md` | 설계 근거 |

## 의존 방향

```
robot_skin  ──▶  common  ◀──  deformable_sats
```

`common` 은 두 패키지 어느 쪽도 import 하지 않고, `robot_skin` 은 `deformable_sats` 를 import 하지 않는다
(`common/tests/test_dependency_direction.py`).

## 빠른 시작

```bash
pip install -r requirements.txt          # = deformable_sats/requirements.txt + PyYAML, pytest

# 전체 테스트 (루트 pytest.ini: pythonpath = . deformable_sats)
pytest

# 기존 SATS 워크플로 — 커맨드는 예전과 동일, 디렉터리만 이동
cd deformable_sats
python -m sats.training.train_e2e --help
```

```python
from common.signal import relative_change, estimate_baseline
from common.layouts import load_layout
from robot_skin.contact import OrdinalQuantizer, SaturationFSM

layout = load_layout("glove_template")          # 손끝 5 + 손바닥, parent = MANO 세그먼트
delta = relative_change(raw, estimate_baseline(raw, n_samples=200))   # SATS 규약 ΔS% (누르면 음수)
levels = OrdinalQuantizer(weak_pct=3, strong_pct=15)(delta)
```

## 알려진 테스트 실패 (main 과 동일, 이동과 무관)

- `deformable_sats/sats/training/tests/test_phase0_data_layout.py` — git-ignored raw 데이터 필요
- `deformable_sats/hitmap/tests/test_zarr_index_resolution.py`, `test_depth_contract.py` — zarr 2.16 고정,
  zarr 3 환경에서는 실패
