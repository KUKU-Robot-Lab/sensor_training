# robot_skin/ — 손 전체 촉각 스킨 (글러브 ⇄ 로봇 핸드)

```
acquisition ─▶ pose ─▶ baseline ─▶ contact ─▶ representation ─▶ policy / vtla
 (세션·스트림)  (taxel pose)  (무접촉 ΔS 예측)  (잔차·ordinal·FSM)  (taxel 토큰)
                           sim(도메인 랜덤화) · transfer(MANO) · eval(지표)
```

| 모듈 | 구현 | 스텁 |
|---|---|---|
| `hardware/` | 문서 | — |
| `acquisition/` | `SessionManifest` | `glove_logger`, `robot_logger` CLI (`--dry-run` 만 동작) |
| `pose/` | `TaxelPoseProvider`, `StaticPoseProvider`, `TransformPoseProvider` | `robot_fk`(URDF), `glove_imu2mano`(VIFNet-S) |
| `baseline/` | `BaselinePredictor`, 무접촉 윈도 데이터셋, 학습 루프 | — |
| `contact/` | residual, `OrdinalQuantizer`, `SaturationFSM`, 접촉 마스크, self-touch 자동 라벨 | — |
| `representation/` | `TaxelTokenizer`, `random_taxel_mask` | 마스킹 사전학습 |
| `sim/` | per-taxel 도메인 랜덤화 | touch-grid 환경 |
| `policy/` | 관측 ablation 빌더(full/ordinal/binary/none) | RL 학습 |
| `vtla/` | 촉각 토큰 어댑터 + 접촉 게이팅 | — |
| `transfer/` | — | MANO 사영·레이아웃 정렬 |
| `eval/` | 환각률, 모션-접촉 분리도, 포화 복구 시간 | — |

설정: `configs/default.yaml` (`robot_skin.config.load_config`). 데이터·산출물: `data/`, `runs/` (git-ignored).
`deformable_sats` 는 import 하지 않는다 — 공유 규약은 전부 `common/`.
