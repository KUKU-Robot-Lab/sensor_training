# policy/ — 정책 관측 ablation

| 파일 | 상태 | 역할 |
|---|---|---|
| `observation.py` | 구현 | `ObservationBuilder(mode)`: proprio ⊕ 촉각 특징. mode = `full`(연속 잔차+포화 플래그, 2/taxel) · `ordinal`(one-hot 4) · `binary`(접촉 1) · `none`(0) |
| `train_rl.py` | 스텁 | 모드별 RL 학습 (sim.TouchGridEnv 필요) |

sim/실기 모두 같은 빌더를 쓰므로 ablation 은 "정책이 받는 촉각 정보량"만 바꾼다.
