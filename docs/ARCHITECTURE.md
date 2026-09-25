# Architecture

## 1. 왜 모노레포인가

기존 저장소는 평면 4×4 SATS 패드(변형·밴딩 보상 포함)의 압력맵 학습 워크스페이스였다. 다음 단계는 같은
기압 taxel 을 **손 전체**(글러브, 로봇 핸드)에 붙이는 것이고, 두 작업은 신호 규약(ΔS%, baseline, 포화),
시계 정렬, taxel 배치 기술을 공유한다. 복붙 대신 공유 계층 `common/` 을 두고 두 패키지가 그것만 바라본다.

```
robot_skin  ──▶  common  ◀──  deformable_sats
```

- `common` 은 어느 쪽도 import 하지 않는다 (테스트로 강제).
- `robot_skin` 은 `deformable_sats` 를 import 하지 않는다. 연구 결과(예: bending restorer)는 **아이디어로** 가져와
  일반화하고, 규약은 `common` 에 올린다.
- `deformable_sats` 는 당장 `common` 을 쓰지 않아도 된다. 규약 일치는 `common/tests/test_signal.py` 가
  `sats/training/dataset.py` 의 수식을 텍스트로 고정해 감시한다. 점진적으로 `common` 으로 옮기면 된다.

## 2. deformable_sats 를 통째로 한 단계 내린 이유

`sats`/`hitmap` 코드에는 `from sats.…` import 가 수백 개, `Path(__file__).resolve().parents[N] / "learning_data/…"`
식 저장소-루트 상대 경로가 수십 개 있다. 내부 구조를 바꾸면 전부 깨지므로 **루트 전체를 `git mv` 로
`deformable_sats/` 아래로 옮기기만** 했다(history/fig_data 포함, 이름 변경 없음). `parents[N]` 은 파일 기준
상대라 그대로 `deformable_sats/` 를 가리키고, 루트 `pytest.ini` 의 `pythonpath = . deformable_sats` 로
`import sats` 도 그대로 동작한다. 기존 커맨드는 `cd deformable_sats` 후 그대로 쓴다.

## 3. 신호 규약 (`common.signal`)

- **ΔS% = (raw − baseline)/baseline × 100** — `sats/training/dataset.py` 와 비트 단위로 동일.
  mk555 기압 taxel 은 누르면 raw 가 **감소**하므로 접촉 ΔS 는 **음수**(드롭아웃은 −100 % 방향).
  frozen SATS 에 그대로 넣을 수 있도록 부호를 뒤집지 않고, 대신 `PRESS_SIGN = −1` 과
  `press_intensity = −ΔS` 를 두어 접촉 로직(임계값·ordinal)은 양수로 다룬다.
- baseline = 초기 무접촉 구간 **median** (초반 스파이크에 강건).
- `NormStats`(offset/scale) 는 학습 데이터로만 fit 하고 json 으로 모델과 함께 저장.
- 포화 = ADC 레일(24-bit) 또는 |ΔS| ≥ 상한.

## 4. 핵심 문제: 움직임 vs 접촉

손 위의 taxel 은 **접촉 없이도** 관절이 굽고 피부가 늘어나면 ΔS 가 변한다. 이것을 접촉으로 읽으면 환각이다.

1. `pose/` 가 시각 t 의 taxel 위치·법선을 준다 (로봇: URDF FK, 글러브: 7-IMU → MANO(VIFNet-S)).
2. `baseline/BaselinePredictor` 가 `[pose, q, q̇]` 로 **무접촉 ΔS** 를 예측한다. 이는
   `deformable_sats/sats/bending/` 의 deg→offset `BaselineRestorer` 의 일반화다: 조건 변수가 밴딩각 1개에서
   taxel pose + 관절 상태로 바뀌었을 뿐 구조(오프셋 예측 후 차감, zero-init 항등 웜스타트)는 같다.
   bending 에서 얻은 교훈 — **관측 ΔS 를 입력으로 넣으면 접촉까지 오프셋으로 학습해 지운다** — 을 그대로 따라
   입력은 운동학 변수만 쓴다. 학습 데이터는 무접촉 세션(`SessionManifest.segments: no_contact`).
3. `contact/` 는 잔차(관측 − 예측)를 해석한다: 연속값, `OrdinalQuantizer`(무접촉/약/강/포화),
   `SaturationFSM`(포화 후 복구 대기, 안정 복귀 없으면 re-zero — `sats/inference/run_dashboard.py` 채널 격리 규칙을
   per-taxel 상태기계로 정리), self-touch 자동 라벨(손가락 캡슐 거리).

## 5. 표현과 정책

- `TaxelTokenizer` 는 채널 인덱스가 아니라 **pose** 로 위치를 인코딩해 글러브·로봇 핸드가 토크나이저를 공유한다.
- 정책 관측 ablation(full/ordinal/binary/none)은 sim·실기 공용 빌더로 만들어, "정책이 촉각에서 무엇을 얻는가"만
  변수로 남긴다. sim 은 per-taxel 게인·포화 랜덤화로 절대값 의존을 깨뜨린다.
- VTLA 어댑터는 접촉 게이트로 무접촉 시 촉각 토큰을 0 으로 만든다 — 드리프트에서 오는 환각을 구조적으로 차단.

## 6. 평가

- 환각률(무접촉 frame/taxel 중 접촉 예측 비율)
- 모션-접촉 분리도(실접촉 vs 움직임만 구간의 AUROC·d′, baseline 차감 전/후)
- 포화 복구 시간(SATURATED 이탈 → OK)

## 7. 경계 규칙 요약

| 규칙 | 이유 |
|---|---|
| mk555 raw `.bin` 파서는 `deformable_sats/sats/preprocessing/bin_merge.py` 가 정본 | 포맷 두 벌 유지 금지 |
| 새 공유 규약은 `common/` 에, 테스트와 함께 | 두 패키지 드리프트 방지 |
| `robot_skin/data`, `robot_skin/runs` 는 git-ignored (README 만 추적) | 대용량 산출물 |
