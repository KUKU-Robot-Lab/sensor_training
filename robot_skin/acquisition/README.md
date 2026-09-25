# acquisition/ — 세션 기록

| 파일 | 상태 | 역할 |
|---|---|---|
| `manifest.py` | 구현 | `SessionManifest`(json): kind(glove/robot/bench), layout, streams{file, rate_hz, fields, clock, method}, segments(`no_contact` 등 라벨 구간), baseline, meta |
| `glove_logger.py` | CLI 스텁 | 기압 + 7-IMU + 카메라. `--dry-run` 은 계획된 `session.json` 만 작성 |
| `robot_logger.py` | CLI 스텁 | 기압 + joint state(q, q̇, τ). `--dry-run` 동일 |

- 스트림은 각자 타임스탬프로 저장하고 정렬은 나중에 `common.timeline.align_streams`(마스터 200 Hz).
- mk555 raw `.bin` 파서는 **복제하지 않는다**: `deformable_sats/sats/preprocessing/bin_merge.py` 가 정본.
- 무접촉 세션(`--no-contact` → `segments: [{label: no_contact}]`)이 `baseline/` 학습 데이터.
