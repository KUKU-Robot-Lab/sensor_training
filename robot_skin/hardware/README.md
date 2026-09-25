# hardware/ — 하드웨어 문서 (코드 없음)

| 대상 | 내용 |
|---|---|
| 기압 taxel | mk555 계열 barometric 센서(24-bit ADC). **누르면 raw 가 감소** → ΔS% 음수 (`common.signal.PRESS_SIGN = -1`). 레일(0, 2²⁴−1)·|ΔS|≥상한은 포화 (`common.signal.saturation_mask`). |
| 글러브 | 손끝 5 + 손바닥 2×2 pad, IMU 7개(wrist/palm/thumb/index/middle/ring/pinky). 배치: `common/layouts/glove_template.yaml` (parent = MANO 세그먼트). |
| 로봇 핸드 | 동일 토폴로지, parent = URDF 링크 (`common/layouts/robot_hand_template.yaml`, 링크명은 실제 URDF로 교체). |
| 평면 SATS 패드 | 4×4, 6.5 mm 피치 (`common/layouts/sats_4x4.yaml`) — 기존 `deformable_sats/` 실험 장비. |

## 기록할 것 (빌드마다)
- 채널 ↔ taxel id 매핑(레이아웃 YAML의 `channel`), 각 pad 의 parent 프레임 내 위치·법선 실측값.
- 펌웨어/보드 버전, 샘플레이트, 시계 동기 방식(→ `acquisition.SessionManifest.meta`).
- raw `.bin` 포맷이 mk555 와 같으면 파서는 `deformable_sats/sats/preprocessing/bin_merge.py` 가 정본(복제 금지).

장비 체크리스트·장갑 착용·3-탭 싱크·세션 스크립트는 `docs/DATA_ACQUISITION.md`, 파일 포맷은 `docs/DATA_FORMAT.md`,
로봇 핸드 드라이버 연결(`control.RobotHandInterface` 구현)은 `docs/DEPLOYMENT.md` §7.
