# sim/ — 시뮬레이션

| 파일 | 상태 | 역할 |
|---|---|---|
| `domain_rand.py` | 구현 | `TaxelDomainRandomizer`: per-taxel 게인·오프셋·포화 레벨·노이즈·드롭아웃(−100 %) 랜덤화 → (obs ΔS, saturated) |
| `touch_grid_env.py` | 스텁 | touch-grid 과제 환경 (MuJoCo/Isaac) |

실센서의 게인 편차·포화를 sim 에서 미리 보여줘서 정책이 절대값 대신 ordinal/binary 신호에 기대도록 한다.
