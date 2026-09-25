# skin_ws — Raw Acquisition Archive

`skin_ws`는 SATS 센서 실험의 **원본 취득 데이터 아카이브**다. 여기 있는 raw BIN은
읽기 전용 입력으로만 다루며, 학습 산출물은 만들지 않는다. 실제 학습/GT는
`learning_data`만 참조한다(아래 흐름 참고).

## 디렉터리

```text
skin_ws/
├── raw_data/          # 원본 raw BIN 아카이브 (읽기 전용)
├── node/              # mk555 stage 스캔 프로그램 (.node)
├── acquisition_code/  # 취득 코드 (final_logger.py 통합 로거 + loadcell_bin_logger·convert_bins,
│                      #   tools/{device_tests,gui,node_generation,analysis}, legacy/ — 자체 README)
├── md/                # 센서 사양·실험 노트
└── backup/            # 펌웨어/구버전 백업
```

## raw_data 구조

```text
skin_ws/raw_data/sats/eco20 + mesh/
└── d5/
    ├── test1/
    └── test2/
        ├── due_raw_burst_*.bin          # DUE: 16센서 × 10 FIFO 프레임/버스트
        ├── ethermotion_encoder_*.bin    # stage 명령 x,y,z,u (가상축 u 포함)
        └── loadcell_raw_*.bin           # 단일축 loadcell kg
```

공식 입력은 위 3종 raw BIN이다. force는 loadcell `Fz = (kg - kg_baseline) * 9.80665`로
계산하며, AFD `Fx/Fy/Fz`는 사용하지 않는다.

## 타임라인 정렬 기준

학습 row의 기준은 DUE/loadcell의 의미 대역에 맞춘 200 Hz common timeline이다.

- DUE raw burst는 10 FIFO frame을 펼쳐 유효 200 Hz tactile stream으로 만든다.
- Loadcell은 같은 200 Hz timeline에 보간한다.
- EtherMotion은 1000 Hz 이상 고속 로그이며, `x/y/z/u`를 200 Hz row에 고정밀 보간한다.

따라서 EtherMotion row 수가 가장 많더라도 학습 row를 EtherMotion rate로 늘리지 않는다.
대신 각 200 Hz tactile/loadcell row에 `z_stage_mm`, `z_depth_mm`를 약 `0.0001 mm`
정밀도로 붙인다.

## mk555 node 기준값

| 항목 | d5 | d10 |
| --- | ---: | ---: |
| X/Y 범위 / step | -10.0~10.0 mm / 0.5 mm | 동일 |
| X/Y 위치 수 | 41 × 41 = 1681 | 동일 |
| 시작 Z | 13.0 mm | 12.0 mm |
| 최대 Z | 15.5 mm | 15.5 mm |
| Z step | 0.5 mm | 0.5 mm |
| 유효 stage depth | 2.50 mm | 3.50 mm |
| U 범위 | 0.0~3.0 mm | 0.0~4.0 mm |

기준 node는 `node/SATS_d5_mk555.node`, `node/SATS_d10_mk555.node`이다.

## U축은 가상축이다

`ethermotion`의 `u`(→ merged `u_mm`)는 **연속보간을 위한 가상축**이다. U가 "움직이는"
동안 stage는 물리적으로 정지(x, y, z 고정)한다. 실측상 각 (x, y)에서:

- `u_mm`: node 내부 대기/가상 축 로그 값이다. 실제 depth나 GT 계산 기준으로 쓰지 않는다.
- 실제 압입 깊이: `d5 = z_stage_mm - 13.0`, `d10 = z_stage_mm - 12.0`

따라서 `u_mm`은 물리 전단이 아니라 hold 표식이며, loading·hold 모두 수직 GT가
유효하므로 둘 다 학습에 사용한다.

## learning_data로 변환

raw 아카이브는 직접 학습하지 않고, 아래 명령으로 `learning_data`에 모은다.

```bash
python3 sats/preprocessing/prepare_learning_data.py \
  --source-root skin_ws/raw_data \
  --learning-root learning_data
```

매핑(현재 mk555):

```text
sats/eco20 + mesh/d5/test1  -> learning_data/sensor_raw_bin/ecomesh/d5/z_2.5mm/test1
sats/eco20 + mesh/d5/test2  -> learning_data/sensor_raw_bin/ecomesh/d5/z_2.5mm/test2
sats/eco20 + mesh/d10/test1 -> learning_data/sensor_raw_bin/ecomesh/d10/z_3.5mm/test1  # d10 수집 시
```

현재 d5 `test1/test2`는 `learning_data/gt`의 pressure-map GT와 row alignment가
검증되어 있다. SATS 학습에서는 0.5 mm hold 구간까지 포함되므로 `seq_len=1000`
이상을 사용한다.

새 trial은 `raw_data`에 `testN` 또는 `YYYYMMDD_testN` 폴더로 넣고 위 명령을 다시 돌리면 자동 추가된다.
자동 인식 조건: ① 소재 폴더(기본 `eco20 + mesh`), ② depth-map에 있는 직경(기본 d5/d10,
새 직경은 `--depth-map d7=2.0`), ③ `YYYYMMDD_testN` 형식, ④ raw BIN 3종 모두 존재.

출력 `testN` 번호는 `learning_data/trial_registry.json`에 영구 고정된다(git 추적). 한 번
부여된 번호는 유지되고 새 폴더만 뒤에 append되므로, **과거 날짜 폴더를 나중에 끼워넣어도
기존 trial 번호는 바뀌지 않는다.**

상세 파이프라인은 [`sats/README.md`](../sats/README.md),
[`sats/preprocessing/README.md`](../sats/preprocessing/README.md) 참고.
