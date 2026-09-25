# transfer/ — 글러브 ⇄ 로봇 핸드 (하나의 손 골격 위에서)

사람 글러브 데이터와 로봇 핸드 데이터를 **같은 손 좌표**에서 다루기 위한 모듈. 두 손은 크기·마디 수·링크
좌표계가 다르므로, 각 taxel 을 캡슐 골격(뼈 축 + 반지름) 위의 정규 좌표로 옮겨 대응시킨다.

| API | 내용 |
|---|---|
| `CapsuleSkeleton.from_mano(skeleton, finger_pose)` | `ManoSkeleton.capsules` (15 마디 + 손바닥 4) — 손 좌표계(go = 0, 손목 원점), palmar = 뼈 정렬 프레임의 −y |
| `CapsuleSkeleton.from_urdf(model, q, palmar_axis=...)` | URDF 링크 → 캡슐 (부모 원점 → 자식 원점, 말단 링크는 들어오는 뼈 방향으로 연장, 같은 위치 프레임은 건너뜀) |
| `project_to_skeleton(pos[...,N,3], skeleton, taxel_groups=...)` | 가장 가까운 캡슐: `segment`, `t`(뼈 위치 0–1), `closest`/`offset`, `distance`/`surface_distance`, 정규 좌표 `finger`, `u`(손가락 기저 0 → 끝 1), `side`(+1 바닥면 / −1 등), `v`(손바닥 가로 위치: 검지 0 → 새끼 1) |
| `project_to_mano(pos, skeleton=None, finger_pose=None)` | 위의 MANO 버전 (옛 스텁 이름 유지) |
| `taxel_groups(layout)` / `layout_rest_poses(layout, urdf=...)` | taxel 별 손가락 그룹(`finger_<f>`/`palm` 그룹 또는 부모 이름), 손 좌표계 기준 자세 |
| `align_layouts(src, dst, ...)` → `LayoutAlignment` | dst taxel → src taxel k 개 (역거리 가중), **같은 손가락 그룹 안에서만**. 두 골격을 주면 `skeleton` 공간(`|Δu| + side·|Δside|/2 + lateral·|Δv|`), 아니면 공통 좌표계의 유클리드 거리(+ 법선 항). `max_dist` 로 먼 대응 무효화. `matrix()`, `to_dict/save/load` |
| `map_taxel_values(values, mapping, taxel_axis, reduce)` | 값 이식: `weighted`(z·ΔS), `nearest`(임의 dtype), `max`(ordinal level·접촉 플래그); 무효 taxel 은 fill (float NaN / int −1 / bool False) |
| `RobotToManoEstimator(forward_retargeter)` | 역 리타게팅: 로봇 q → MANO 손가락 자세 (15 굴곡 + 5 외전 파라미터, `FingertipRetargeter` 목적함수를 역할만 바꿔 사용). `PolicyRunner(hand_state_fn=...)` 로 hand_mano 정책의 proprio 추정 |

```python
from robot_skin.transfer import CapsuleSkeleton, align_layouts, layout_rest_poses, map_taxel_values
from robot_skin.pose.urdf import URDFModel

model = URDFModel.from_file("robot.urdf")
gp, _ = layout_rest_poses("glove_template")
rp, _ = layout_rest_poses("robot_hand_template", urdf=model)
m = align_layouts("glove_template", "robot_hand_template", src_pos=gp, dst_pos=rp,
                  src_skeleton=CapsuleSkeleton.from_mano(),
                  dst_skeleton=CapsuleSkeleton.from_urdf(model, palmar_axis=(1, 0, 0)))
z_robot = map_taxel_values(z_glove, m)                         # [T, N_glove] → [T, N_robot]
lv_robot = map_taxel_values(level_glove, m, reduce="max")
```

근거: MANO (Romero et al., SIGGRAPH Asia 2017); 사람 손 데이터 → 로봇 손 이식은 OSMO (arXiv:2512.08920),
DexUMI (arXiv:2505.21864) 와 같은 방향(손 모델을 공유하고 센서 배치는 다름); 손끝 벡터 대응은 DexPilot
(arXiv:1910.03135) / AnyTeleop (arXiv:2307.04577). 참고문헌 표는 `docs/REFERENCES.md`.

한계: 손바닥은 캡슐 몇 개로 거칠게 근사한다. 로봇 링크의 palmar 축은 URDF 에 없으므로 `palmar_axis` 를
직접 줘야 `side` 가 생긴다(없으면 `side` 항은 건너뜀).
