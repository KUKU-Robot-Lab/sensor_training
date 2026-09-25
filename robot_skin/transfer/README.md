# transfer/ — 글러브 ⇄ 로봇 핸드 (스텁)

| 함수 | 계획 |
|---|---|
| `project_to_mano` | 로봇 핸드 taxel pose → MANO 표면/세그먼트 사영 (canonical hand) |
| `align_layouts` | 글러브 레이아웃 ↔ 로봇 레이아웃 대응 (세그먼트 + 최근접 위치) — 토큰·라벨 이식용 |

사람 글러브 데이터(`pose/glove_imu2mano`)와 로봇 데이터를 같은 손 좌표계에서 합치기 위한 모듈.
