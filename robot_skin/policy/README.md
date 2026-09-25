# policy/ — 정책 관측 ablation

| 파일 | 상태 | 역할 |
|---|---|---|
| `observation.py` | 구현 | `ObservationBuilder(mode)`: proprio ⊕ 촉각 특징. mode = `full`(연속 잔차+포화 플래그, 2/taxel) · `ordinal`(one-hot 4) · `binary`(접촉 1) · `none`(0) |
| `train_rl.py` | 스텁 | 모드별 RL 학습 (sim.TouchGridEnv 필요) |

sim/실기 모두 같은 빌더를 쓰므로 ablation 은 "정책이 받는 촉각 정보량"만 바꾼다.

이 빌더는 RL/sim ablation 용이다(잔차 %·고정 % 임계 `OrdinalQuantizer`, full = 2/taxel). pretrain·VTLA·온라인 제어의
촉각 입력은 이 빌더가 아니라 `representation.tactile_value_features` 하나로 만든다(contact stage 의 보정된 z·레벨
기반, full = 6/taxel; 모드 이름은 같다) — `representation/README.md`, `docs/ARCHITECTURE.md` §5.7.
