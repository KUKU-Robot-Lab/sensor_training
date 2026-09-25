# pose/ — taxel 위치·법선 제공자

`TaxelPoseProvider` 프로토콜: `pose_at(t) → (positions[N,3], normals[N,3])` (레이아웃 순서, m).

| 구현 | 상태 | parent 해석 |
|---|---|---|
| `StaticPoseProvider` | 구현 | 고정 4×4 (벤치 패드, 강체 마운트) |
| `TransformPoseProvider` | 구현 | 임의 `t → {parent: T}` 콜러블 |
| `robot_fk.RobotFKPoseProvider` | 스텁 | URDF FK (joint state q) — parent = URDF 링크 |
| `glove_imu2mano.GloveImu2ManoPoseProvider` | 스텁 | 7-IMU → MANO (VIHand **VIFNet-S** 파인튜닝) — parent = MANO 세그먼트 |

모두 `transform_taxels(layout, {parent: T})` 로 귀결되므로 소스만 바꾸면 된다.
포즈는 `baseline/`(입력), `contact/self_touch`(자동 라벨), `representation/`(pose 임베딩)에 쓰인다.
