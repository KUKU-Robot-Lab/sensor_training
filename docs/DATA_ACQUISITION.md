# 데이터 수집 운영 절차 — D1 `motion` / D2 `task`

촉각 장갑(mk555 계열 기압 taxel) + IMU 7개 + 카메라(`ego`, `third`)로 두 데이터셋을 모으는 **운영자용
절차서**다. 코드 사용법은 `robot_skin/acquisition/README.md`, raw/processed 포맷은 `docs/DATA_FORMAT.md`,
논문 근거는 `docs/REFERENCES.md` 를 본다.

| 데이터셋 | 내용 | 쓰임 |
|---|---|---|
| **D1 `motion`** (`protocols/d1_motion.yaml`) | 물체 없이 손만 움직임: 정지 baseline, IMU 보정, 손가락별 굴곡, 쥐기/펴기(느림·빠름), 손목 회전, 자유 동작, **에어 그립**(D2 쥐기 손모양을 물체 없이 공중에서, 무접촉), **자기접촉 세트**, 3-탭 싱크 | IMU → 손 자세 모델, 움직임에 의한 촉각 baseline 예측기(무접촉 구간), 접촉 검출기 보정(자기접촉 = 공짜 접촉 라벨) |
| **D2 `task`** (`protocols/d2_task.yaml`) | 물체 과제 + 언어 지시문. **에피소드 1개 = 과제 1회 = 세션 디렉터리 1개** | VTLA (vision + tactile + language → action) 학습, 이후 로봇 제어 |

기록 후 처리 흐름은 전부 자동이다: **기록 → 3-탭 싱크(타임스탬프 보정) → IMU 보정 → QC(`qc.json`)**
→ (나중에) `python -m robot_skin.datasets.build` 전처리. 손 자세 라벨(`hand_pose.npz`)은 기록 후
카메라 영상에 HaMeR/WiLoR 를 오프라인으로 돌려 만든다 (`robot_skin/pose/vision_hand.py` 참고).

세션 구조(스트림별 파일 + 매니페스트 + 이벤트 로그)와 1인칭 + 고정 카메라 구성은 ActionSense
(DelPreto et al., NeurIPS 2022 Datasets & Benchmarks)를 따랐다.

---

## 1. 역할

- **피험자**: 장갑을 끼고 동작을 수행한다. 이름 대신 가명 ID(`S01`, `S02` …)만 쓴다.
- **운영자**: 키보드로 블록 시작(Enter), D2 phase 경계(Enter), 성공 판정(y/n)을 입력하고 대본을 읽어 준다.
- **파트너**(D2 `handover` 만): 물체를 건네받는 실험자.

## 2. 장비 체크리스트

| 항목 | 확인 사항 |
|---|---|
| 촉각 장갑 + taxel 보드 | 펌웨어 버전 기록(`--notes`), 케이블 고정(당김이 taxel 을 누르지 않게), 전 채널 raw 값이 움직이는지 |
| IMU 7개 | 손목·손등(palm)·다섯 손가락 끝마디(layout `imu_sites`). 배터리/연결, 스트랩 단단히 |
| 카메라 `ego` | 머리 장착, 아래-앞을 비스듬히 봄. 작업 영역과 손이 화면 중하단. **자동 노출/화이트밸런스 끔**, 30 fps, ≥ 640×480 |
| 카메라 `third` | 삼각대 고정, 작업대 측면 약 45°, 1–1.5 m. 책상 전체 + 손. 얼굴이 안 나오게 구도 |
| 싱크 패드 | 단단한 평판(아크릴/나무) 위에 표시한 탭 위치. 두 카메라 모두에서 보여야 함 |
| 팔뚝 받침대 | 정지·보정 자세에서 팔뚝만 받치고 손은 공중에 뜨게 |
| 시작 마커 (D2) | 손 시작 위치 표시 (책상 위 테이프) |
| D2 물체 세트 | `d2_task.yaml` 의 objects/targets (없으면 YAML 을 실제 물체로 수정, id 는 세션 간 고정) |
| 호스트 PC | 절전/화면보호기 끔, 디스크 여유(카메라 JPEG 기준 대략 수 GB/시간), 다른 무거운 작업 금지 |
| 조명 | 고정 조명, 깜빡이는 조명·직사광선 피하기 |
| 서류 | 동의서, 피험자 ID 대응표(데이터와 **따로** 보관) |

기록 전 소프트웨어 점검 (하드웨어 없이 가능):

```bash
python -m robot_skin.acquisition.glove_logger --protocol d1_motion --subject S01 --dry-run   # 대본 출력
python -m robot_skin.acquisition.glove_logger --protocol d1_motion --subject TEST --fake --time-scale 0.05 --out /tmp/t
```

## 3. 장갑 착용과 워밍업

1. 손을 씻고 완전히 말린다 (땀·습기는 기압 taxel 드리프트 원인).
2. 장갑을 끼고 손가락 끝 taxel 이 손끝 지문면 중앙에 오는지 확인한다. 주름이 taxel 위에 접히지 않게.
3. **손목 IMU 는 손등 방향과 나란히** 붙인다. 단일 정지 자세 보정은 손목 IMU 장착 오차와 IMU 세계 좌표를
   구분하지 못한다(`pose.imu_model.estimate_world_alignment` 문서). 손가락 IMU 는 끝마디 등쪽 중앙.
4. **워밍업 3–5 분**: 손을 편히 두고 raw 값이 안정될 때까지 기다린다 (체온으로 장갑 내부 온도가 변하면
   기압 baseline 이 흐른다). 워밍업을 건너뛰면 QC `pressure_baseline_drift` 가 실패하기 쉽다.
5. 장갑을 벗었다가 다시 끼면 **새 세션**이고 보정/싱크를 다시 한다 (D2 는 에피소드마다 자동으로 함).

## 4. 기준 자세 두 가지

- **휴식(rest) 자세 = 촉각 baseline 기준**: 팔뚝만 받침대에 올리고 손은 힘을 빼고 공중에. 손가락은
  자연스럽게 약간 굽힘. 아무것도(책상, 몸, 옷, 반대 손) 닿지 않는다. ΔS 는 세션의 첫 `no_contact`
  구간을 기준으로 계산되므로 **D1 `baseline_start` 와 D2 `baseline` 은 반드시 같은 휴식 자세**로 한다
  (자세가 다르면 baseline 예측기의 기준점이 세션마다 달라진다).
- **평손(flat hand) 자세 = IMU 보정**: 손가락을 모두 붙여 곧게 펴고, 손바닥은 아래, 손은 수평, **공중**
  (책상에 대면 손바닥/손끝 taxel 이 눌려 `no_contact` 라벨이 오염된다). 5 초(D2 는 2 초) **완전 정지**.
  콘솔에 `보정 품질: spread=…° gyro_rms=…` 가 뜨고, spread ≤ 3° 이고 gyro RMS ≤ 0.15 rad/s 가 아니면
  반복을 제안한다. 결과는 `manifest.calibration` 의 `imu_offsets`/`imu_world`/`imu_sites` 로 저장된다.

## 5. 3-탭 싱크 (시작·끝)

모든 스트림은 도착 시각(호스트 단조 시계)으로 기록되지만 장치마다 지연(시리얼 버퍼, 카메라 노출→USB,
IMU 무선)과 시계 드리프트가 있다. 하나의 날카로운 물리 사건을 모든 센서가 동시에 보게 해서 이를 잰다.

**방법** (`sync_start`, `sync_end` 블록: D1 약 4.3 초. D2 는 1 s 대신 0.8 s 정지, 간격 0.5 s / 1.0 s, 끝 정지
1.0 s 로 약 3.3 초 — 값은 각 YAML 의 `sync:`)

1. 손을 싱크 패드 위 약 3 cm 에 두고 **1 초 정지** (검지만 펴고 나머지 손가락은 말아서 닿지 않게).
2. 검지 끝으로 패드를 **"탁 – 탁 ——— 탁"** 리듬(간격 약 0.6 s, 1.2 s)으로 3번 친다. 튕기듯 짧고
   분명하게(접촉 0.1 s 안팎). 간격이 불균등해야 상호상관 피크가 하나로 정해진다.
3. 마지막 탭 후 1.5 초 정지. 두 카메라에 손끝이 보여야 한다.

**언제**: D1 은 시작(`sync_start`)과 끝(`sync_end`) → 오프셋 + 드리프트(선형 시계 모델). D2 는
에피소드마다 시작에서 한 번 → 오프셋. 시간이 부족하면 YAML `episode.pre` 에서 빼고 같은 착용 회차의
다른 세션 값을 `--sync-from <세션>` 으로 적용할 수 있다 (장치 지연이 일정하다는 가정).

**처리** (`acquisition/sync.py`, 자동): 각 스트림을 활동 envelope 으로 바꿔 압력 스트림과 상호상관한다.
IMU 는 충격·이륙 가속도 스파이크가 압력 모서리와 같은 모양이라 수 ms 정밀도, 카메라(30 Hz)는 손 움직임만
보이므로 약 1 프레임(33 ms) 정밀도다. 프레임 이하 정밀도가 필요하면 탭과 동시에 켜지는 LED 를 화면에
넣는다. 보정된 시각은 파일에 바로 반영되고 원본은 `t_host` / `timestamps_host.npy` 로 남는다.

**확인**: QC `sync_taps_visible` (압력에서 탭 3개 검출), `sync_score` (≥ 0.3), `sync_offset` (≤ 0.5 s).

## 6. D1 `motion` 세션 스크립트

한 세션 ≈ 블록 5.6 분 (명목 휴지 1 s 포함 6.2 분) + 운영자 휴지 (실제 9–11 분). 각 블록 전 대본을 읽고 **Enter 로 시작**하면
정해진 시간 동안 기록되며 콘솔이 초 단위 카운트다운과 주기마다 메트로놈(벨)을 낸다.

| # | 블록 (phase 이름) | 시간 | 접촉 기대 → segment | 동작 (피험자에게) |
|---|---|---|---|---|
| 1 | `baseline_start` | 5 s | 없음 → `no_contact` | 휴식 자세로 정지 |
| 2 | `sync_start` | ≈4.3 s | 탭 → `sync` | 3-탭 싱크 (5절) |
| 3 | `imu_calibration` | 5 s | 없음 → `calibration`, `no_contact` | 평손 자세로 완전 정지 |
| 4–8 | `finger_flex_{thumb,index,middle,ring,pinky}` | 5 × 10 s | 없음 → `no_contact` | 한 손가락만 끝까지 굽혔다 폄 (2 s 주기 × 5회), 손끝이 손바닥에 닿지 않게 |
| 9 | `open_close_slow` | 16 s | 없음 | 손 전체 쥐기/펴기 2 s 주기 × 8회, 손끝이 손바닥 닿기 직전까지만 |
| 10 | `open_close_fast` | 8.4 s | 없음 | 같은 동작 0.7 s 주기 × 12회 |
| 11–13 | `wrist_rotation_{pronation_supination, flexion_extension, radial_ulnar}` | 3 × 12 s | 없음 | 손 모양 유지, 손목만 회내/회외 · 굴곡/신전 · 요측/척측 편위 (3 s 주기 × 4회) |
| 14 | `free_motion` | 30 s | 없음 | 손가락·손목 자유 동작, 속도·벌림 섞기, 아무것도 닿지 않게 |
| 15–18 | `air_grasp_slow_{power,precision,lateral,tripod}` | 4 × 12 s | 없음 → `no_contact` | **물체 없이** 공중에서 D2 쥐기 손모양(파워 그립 / 정밀 집기 / 옆(열쇠) 집기 / 세 손가락 집기)을 가상의 물체 둘레로 천천히 만들어 ≈ 1 s 유지 후 폄 (4 s × 3회). 손끝끼리·손바닥에 닿지 않게 (손끝 사이 1 cm 이상) |
| 19–22 | `air_grasp_fast_{power,precision,lateral,tripod}` | 4 × 6 s | 없음 | 같은 손모양을 빠르게 만들었다 폄 (1.5 s × 4회) |
| 23–26 | `pinch_{index,middle,ring,pinky}` | 4 × 10 s | 자기접촉 → `self_touch` | 엄지 끝과 각 손가락 끝을 살짝→꾹 눌렀다 뗌 (2 s × 5회) |
| 27 | `fist` | 12.5 s | 자기접촉 | 주먹 꽉 쥐기(손끝이 손바닥을 누름)/완전히 펴기 (2.5 s × 5회) |
| 28 | `finger_crossing` | 10 s | 자기접촉 | 검지·중지 꼬기/풀기 (2.5 s × 4회) |
| 29–33 | `palm_touch_{thumb,index,middle,ring,pinky}` | 5 × 8 s | 자기접촉 | 각 손가락 끝을 같은 손 손바닥에 굽혀 눌렀다 폄 (2 s × 4회) |
| 34 | `sync_end` | ≈4.3 s | 탭 | 3-탭 싱크 반복 (드리프트 측정) |
| 35 | `baseline_end` | 5 s | 없음 | 휴식 자세 정지 (baseline 드리프트 확인) |

**자기접촉 세트가 중요한 이유**: 손 기하(MANO 캡슐)로 계산한 자기접촉이 접촉 검출기의 "공짜" 양성
라벨이 된다. 전처리는 `self_touch` 구간 안에서 기하 자기접촉이 있는 taxel 만 1, 나머지는 −1(모름)로
둔다. 반대로 **무접촉 블록은 정말 무접촉이어야** baseline 예측기가 오염되지 않는다.

**에어 그립(`air_grasp_*`)이 필요한 이유**: baseline 예측기(무접촉 ΔS 아티팩트 = 손 자세·속도의 함수)는
D1 에서 본 손 자세 범위 안에서만 믿을 만하다. 쥐기/펴기·자유 동작만으로는 D2 의 쥐기 손모양(엄지 대립,
손가락마다 다른 굽힘)이 분포 밖이라 그 자세의 아티팩트를 **과소 예측**하고, 예측 σ 도 작게 나와(σ 는
aleatoric 만 표현) D2 의 비접촉 taxel 이 거짓 접촉으로 읽힌다 (합성 데이터 리뷰에서 확인된 D2 pseudo-label
정밀도 저하). 에어 그립은 같은 손모양을 **접촉 없이** 보여 줘서 이 공백을 메운다. 손모양이 흐트러지더라도
**접촉이 없는 것**이 우선이다 — 손끝끼리 닿았으면 그 블록을 다시 한다 (자기접촉이 섞인 무접촉 라벨은
baseline 을 오염시킨다; 전처리는 무접촉 segment 안의 기하 자기접촉을 −1 로 돌리지만 기하 모델이 놓친
접촉은 걸러지지 않는다).

**실수했을 때**: 무접촉 블록에서 무언가 닿았으면 운영자가 즉시 알려 주고 그 블록을 다시 한다 (같은
phase 이름이 두 번 기록되어도 된다. IMU 보정은 마지막 시도를 쓴다). 기록 중 중단(Ctrl-C)해도 그때까지의
파일은 저장되고 열린 phase 는 자동 종료(QC warning `phases_closed`)된다 — 이런 세션은 다시 찍는다.

## 7. D2 `task` 에피소드

### 7.1 에피소드 흐름 (디렉터리 1개)

1. 콘솔이 과제·물체·**지시문**·성공 기준을 보여 준다. 운영자가 지시문을 그대로 읽어 준다.
2. `baseline` 3 s (휴식 자세, 시작 마커 위 공중) → `imu_calibration` 2 s (평손) → `sync_start` (3-탭).
3. 과제 phase — **운영자가 각 phase 의 시작 경계 순간에 Enter** (발 페달로 대체 가능). 콘솔이 다음에 누를
   경계를 매번 보여 준다. 첫 Enter = `reach` 시작(손이 시작 자세를 떠나는 순간), 이후 Enter 는 진행 중인
   phase 를 끝내고 다음 phase 를 시작하며, **마지막 Enter = `retreat` 끝(손이 시작 자세에 도착)**.
   phase 가 n 개면 Enter 는 n + 1 번이다 (`press_button`: 5 번, 나머지: 6 번). 싱크 뒤 첫 Enter 전까지의
   대기 시간은 어떤 phase 에도 속하지 않는다.

| phase | 접촉 기대 | 시작 경계 (Enter 누르는 순간) |
|---|---|---|
| `reach` | 없음 | 손이 시작 자세를 떠나는 순간 |
| `grasp` | 물체 | 장갑이 물체에 **처음 닿는** 순간 |
| `manipulate` | 물체 | 파지가 안정되어 들어 올리거나 조작을 시작하는 순간 |
| `release` | 물체 | 물체를 내려놓고 손을 펴기 시작하는 순간 |
| `retreat` | 없음 | 손이 물체에서 **마지막으로 떨어지는** 순간 (끝 = 시작 자세 도착) |

   `press_button` 은 `grasp` 가 없다 (reach → manipulate(누름) → release → retreat).
   `reach`/`retreat` 는 운영자 입력이라 경계가 부정확하므로 `no_contact` 학습 데이터로 쓰지 않는다
   (segment 라벨 없음). 과제 phase 전체는 `task` segment 로 묶인다.
4. 끝나면 **성공 판정** y / n (Enter = 보류). **실패 에피소드도 지우지 않는다** — 행동 품질 분석과
   선호 학습(VTLA 논문의 DPO 방식; `vtla/dpo.py`)의 rejected 샘플로 쓸 수 있다.

### 7.2 과제 카탈로그 (`d2_task.yaml`)

| 과제 | 물체 → 목표 | 지시문 템플릿 (에피소드마다 1개 무작위) | 성공 기준 |
|---|---|---|---|
| `grasp_lift_place` | cup, block, ball, bottle → plate, tray, box_lid | "pick up the {object} and place it on the {target}" / "put the {object} on the {target}" / "move the {object} onto the {target}" | 놓은 뒤 목표 위에 안정적으로 있음 (떨어뜨리거나 넘어뜨리지 않음) |
| `pour` | cup_of_beads, small_pitcher, bottle_of_beads → bowl, glass | "pour the {object} into the {target}" 외 2 | 비즈 90 % 이상 목표에, 용기를 바로 세워 둠 |
| `peg_insert` | round_peg, square_peg, usb_plug → peg_board | "insert the {object} into the {target}" 외 2 | 맞는 구멍에 끝까지 삽입, 재파지 2회 이하 |
| `open_jar` | jar, water_bottle | "open the {object}" / "unscrew the lid of the {object}" / "take the lid off the {object}" | 뚜껑을 완전히 떼어 옆에 둠 |
| `wipe` | sponge, cloth → marked_area, plate | "wipe the {target} with the {object}" 외 2 | 표시 영역 전체를 3회 이상 문지름, 도구 원위치 |
| `handover` | cup, block, marker → experimenter | "hand the {object} to the {target}" 외 2 | 파트너가 잡은 뒤 놓음, 떨어뜨리지 않음 |
| `press_button` | red_button, light_switch, keypad_key | "press the {object}" / "push the {object}" / "press the {object} once" | 정확히 1회 작동 (클릭/LED), 두 번 누르지 않음 |
| `in_hand_rotate` | block, marker, ball (+ angle 90/180) | "rotate the {object} by {angle} degrees in your hand" 외 2 | 목표 방향 ±20°, 떨어뜨리거나 책상에 기대지 않음 |

- 슬롯 값의 밑줄은 단어로 바뀐다 (`cup_of_beads` → "cup of beads"). 지시문은 영어(텍스트 인코더 기본)다.
- **지시문과 실제 물체가 일치해야 한다.** 물체를 바꿨으면 `--object` 로 계획을 다시 만들거나
  `--instruction "…"` 으로 덮어쓴다 (`task.instruction_source = operator` 로 기록됨).
- 계획은 `--seed` 로 재현된다: 목표·템플릿·순서가 무작위(시드 고정)로 섞여 과제가 번갈아 나온다
  (피로·순서 효과 완화). 한 번의 카탈로그 전체 = 115 에피소드(약 1 시간); `--task`, `--object`,
  `--episodes`, `--repetitions` 로 나눠 찍는다.
- 삽입 과제는 VTLA (arXiv:2505.09577), 닦기는 OSMO (arXiv:2512.08920) 에서 온 과제군이다.

## 8. 명령어 요약

```bash
# D1 (세션 1개). --out 생략 시 robot_skin/data/raw/motion/S01/<session_id>
python -m robot_skin.acquisition.glove_logger --protocol d1_motion --subject S01
# D2 (에피소드마다 디렉터리). 과제 2개만, 8 에피소드
python -m robot_skin.acquisition.glove_logger --protocol d2_task --subject S01 --task pour --task wipe --episodes 8
# 카메라 지정 / IMU 없이 / 대본 영어
python -m robot_skin.acquisition.glove_logger --protocol d1_motion --subject S01 --cameras ego,third \
    --camera-devices ego=0,third=2 --lang en
# 로봇 핸드 무접촉 스윕
python -m robot_skin.acquisition.robot_logger --protocol robot_sweep --subject R01
# 후처리 재실행 (이벤트 수정 후 등), QC
python -m robot_skin.acquisition.session <session_dir> [--sync-from DIR] [--calibration-from DIR]
python -m robot_skin.acquisition.qc robot_skin/data/raw/task/S01/* --write --json qc_all.json
```

실장비 소스 중 장갑 IMU 허브(`sources.ImuSource`)와 ROS joint state 는 장치가 준비될 때까지
`NotImplementedError` 다 (구현 지침은 각 docstring). 압력만 먼저 쓰려면 `--no-imu`.
mk555 바이너리 보드는 `deformable_sats/sats/preprocessing/bin_merge.py` 기반 파서를
`SerialPressureSource(parser=…)` 로 주입한다 (복사 금지).

## 9. 파일·디렉터리 규칙

```
robot_skin/data/raw/<dataset>/<subject>/<session_id>/     # robot_skin/data/ 는 git 에 올라가지 않는다
  session.json  pressure.npz  imu.npz  camera_ego/  camera_third/  events.jsonl  qc.json  [hand_pose.npz]
session_id = <dataset>-<subject>-<YYYYMMDD>-<HHMMSS>[-e<NNN>-<task>-<object>-r<rep>]
   예) motion-S01-20260925-143012,  task-S01-20260925-150301-e004-pour-cup_of_beads-r03
```

- 피험자 ID 는 가명(`S01`…; 로봇 `R01`…)만. 이름·생년월일 등은 `--notes` 에도 쓰지 않는다.
- 세션 디렉터리는 **덮어쓰지 않는다** (비어 있지 않으면 레코더가 거부). 다시 찍으면 새 디렉터리.
- `hand_pose.npz` 는 기록 후 `pose.vision_hand.save_hand_labels` 로 같은 디렉터리에 추가한다.
- 이벤트 경계를 손으로 고쳐야 하면 `events.jsonl` 줄을 수정/추가하고(시간순 아니어도 됨)
  `python -m robot_skin.acquisition.session <dir>` 로 후처리를 다시 돌린다.

## 10. QC 게이트와 실패 시 조치

기록이 끝나면 자동으로 QC 가 돌고 `qc.json` 이 생긴다. `error` 가 하나라도 실패하면 세션 FAIL
(로거 종료 코드 3). 임계값은 `qc.DEFAULT_THRESHOLDS`, CLI `--set key=value` 로 바꿀 수 있다.

| 체크 | 의미 (기본 임계) | 조치 |
|---|---|---|
| `recorder_errors`, `streams_recorded` (error) | 기록 중 소스 오류 없음, 샘플이 0 개인 소스 없음 (`meta.recorder`) | 장치 연결/전원 확인 후 재기록 (카메라가 안 열렸거나 시리얼이 끊긴 경우) |
| `rate`, `dropped`, `max_gap` (error) | 레이트 ±10 %, 드롭 ≤ 2 % (카메라 5 %), 갭 ≤ 0.5 s | 케이블/USB 대역폭 확인 (카메라는 서로 다른 USB 컨트롤러, MJPEG), 다른 프로세스 종료 후 재기록 |
| `stream_coverage` (error; `hand_pose`/`object_pose` 는 warning) | 스트림이 기록된 모든 phase 를 ±0.5 s 안에서 덮음 (늦게 시작하거나 중간에 멈춘 스트림) | 장치 시작 지연/중단 원인 확인 후 재기록 |
| `timestamps_monotonic` (error) | 시각이 증가해야 함 | 장치 타임스탬프 변환 버그 — 소스 코드 확인 |
| `pressure_channels_layout` (error) | 보드 채널 수가 layout 이 쓰는 채널을 모두 포함 | `--n-channels`, 보드 펌웨어, layout 의 `channel` 확인 |
| `pressure_channels_alive` (error) | layout 이 쓰는 채널 중 분산 0 채널 없음 (연결 안 된 보드 채널은 검사하지 않음) | 커넥터/채널 단선 확인 |
| `pressure_saturation_no_contact` (error) | 무접촉 구간 포화 ≤ 0.5 % | 무접촉 블록 중 접촉이 있었거나 센서 이상 → 해당 블록/세션 재기록 |
| `pressure_saturation` (warning) | 전체 포화 ≤ 10 % | 너무 세게 누름 — 피험자에게 힘 조절 안내 |
| `pressure_baseline_drift` (error) | 첫·마지막 **휴식 자세** baseline 블록(`baseline_start`→`baseline_end`) raw 중앙값 차 ≤ 3 %. 다른 무접촉 블록(평손 보정, 손가락 굽힘)은 자세가 달라 굽힘 artefact 가 섞이므로 비교하지 않는다. D2 에피소드는 휴식 블록이 하나라 검사하지 않는다 | 워밍업 연장, 장갑 밀림 확인, 온도 변화(에어컨 바람) 제거 |
| `imu_quat_norm`, `imu_finite` (error) | 쿼터니언 노름 오차 ≤ 0.02 | IMU 펌웨어/연결 확인 |
| `imu_calibration_present`, `imu_calibration_still_*` (warning) | 보정 존재, 정지 품질 | 평손 블록 재기록 또는 `--calibration-from` |
| `camera_frame_count` (error) | 프레임 수 = 타임스탬프 수 | 디스크 가득/쓰기 오류 확인 |
| `hand_pose_coverage` (warning) | 신뢰도 ≥ 0.5 비율 ≥ 0.7 | 손이 화면 밖/가림 — 카메라 구도 조정 |
| `segments_no_contact` (error, D1) | 무접촉 segment ≥ 1 | 프로토콜로 기록했는지 확인 |
| `task_meta` (error, D2) / `task_success_marked` (warning) | task_id·지시문 / 성공 판정 | 누락 시 `session.json` 의 `task` 보완 |
| `sync_taps_visible`, `sync_score` (warning), `sync_offset` (error) | 탭 3개 검출, 상관 ≥ 0.3, 오프셋 ≤ 0.5 s | 탭을 더 분명히, 카메라 자동노출 끄기, 손끝이 화면에 보이게. 안 되면 `--sync-from` |
| `phases_closed` (warning) | 자동 종료된 phase 없음 | 중단된 세션 — 다시 찍는다 |

FAIL 세션은 지우지 말고 그대로 두고(원인 분석용) 새 디렉터리로 다시 찍는다. 전처리 대상을 고를 때는
`qc.json` 의 `passed` 로 거른다.

## 11. 권장 수집량 (시작점)

아래는 **초기 권장치**다. 파일럿 결과(학습 곡선, 분할별 성능)를 보고 조정한다.

- **파일럿**: 피험자 1명, D1 1 세션 + D2 10 에피소드 → `datasets.build` 까지 돌려 QC·라벨·싱크를 먼저
  확인한 뒤 본 수집을 시작한다.
- **D1**: 피험자 5명 이상(가능하면 8–10명), 피험자당 서로 다른 날/재착용 3 세션 이상 (≈ 15 분 이상의
  블록 데이터). 이유: 피험자 단위 분할(`datasets.splits`, `by="subject"`)에서 val/test 에 최소 1명씩
  두려면 5명 이상이 필요하고, 착용마다 달라지는 baseline 과 IMU 장착 변동을 모델이 봐야 한다.
- **D2**: 과제당 물체 2–4개 × 물체당 5 반복 → 카탈로그 1회 ≈ 115 에피소드. 피험자 3–5명이 각각 1–2회
  → 과제당 약 50–100 에피소드를 첫 목표로 한다. 과제마다 물체 1개는 **평가용으로 남겨** 물체 단위
  일반화를 본다 (`by="object"` 분할). 실패 에피소드도 보존한다.
- 15–20 분마다 휴식. 과제 순서는 시드로 섞고 시드를 기록한다(자동으로 `meta.seed`).

## 12. 영상과 개인정보

- 기관 규정(IRB 등)에 따른 **서면 동의**를 받는다: 영상(손·작업대, 경우에 따라 몸 일부) 기록, 보관 기간,
  연구 목적 사용, 철회 시 삭제.
- 카메라 구도는 **얼굴이 나오지 않게** 잡는다 (`third` 는 작업대, `ego` 는 손 쪽). 화면 안에 신분증,
  모니터 화면, 서류, 다른 사람이 들어오지 않게 정리한다. 오디오는 기록하지 않는다.
- 공유·공개 전에는 얼굴/문자를 오프라인으로 블러 처리하고, 원본은 접근 제한된(암호화) 저장소에만 둔다.
- 데이터에는 가명 ID 만 남기고, ID ↔ 실명 대응표는 데이터와 **분리해** 보관한다. `robot_skin/data/` 는
  git 에서 제외되어 있으니 데이터를 저장소에 커밋하지 않는다.
- 철회 요청 시 해당 피험자 ID 의 모든 세션 디렉터리와 파생 데이터(processed episode, 캐시된 특징)를 삭제한다.

## 13. 문제 해결

| 증상 | 확인 |
|---|---|
| 압력 한 채널이 고정값 | 커넥터, 보드 채널 매핑 (`layout` 의 `channel` 과 보드 순서) |
| baseline 이 계속 흐름 | 워밍업 부족, 땀, 장갑이 헐거움, 에어컨/햇빛 |
| 카메라 fps 저하 | USB 대역폭(카메라마다 다른 포트), 해상도/MJPEG, 디스크 쓰기 속도, `--camera-format jpg` |
| 싱크 점수 낮음 | 탭이 약함, 리듬이 균등함, 손끝이 화면 밖, 자동 노출 켜짐 |
| IMU 보정 품질 불량 | 보정 중 손 떨림/움직임, 책상에 손을 댐, 손목 IMU 스트랩 헐거움 |
