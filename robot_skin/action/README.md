# action/ — 행동 공간 · 청킹 · 사람손 → 로봇손 리타게팅

VTLA 정책이 **무엇을 예측하는지**(행동 표현), **어떻게 학습 타깃을 만들고 실행하는지**(ACT 청킹 +
temporal ensembling), **사람 손 행동을 로봇 관절로 어떻게 바꾸는지**(fingertip-vector retargeting)를
한곳에 모은 모듈. 의존성: `common.signal.NormStats`, `robot_skin.geometry.rotations` 뿐 (URDF/MANO 는
duck typing · lazy import).

| 파일 | 역할 |
|---|---|
| `space.py` | `ActionSpec`, 54-D MANO 손 행동 ⇄ MANO 배열, 로봇 관절 행동, 상대 행동(`make_relative`/`make_absolute`), `ActionNormalizer` |
| `chunking.py` | `action_chunk(s)` (200 Hz master clock 위 미래 H 스텝 + 패딩 마스크), `policy_stride`, `policy_tick_indices`, `TemporalEnsembler` (ACT) |
| `retarget.py` | `FingertipRetargeter` (DexPilot/AnyTeleop 스타일 벡터 매칭, projected LM / Adam / L-BFGS), `human_fingertips`, `hand_action_fingertips` |

## 1. 행동 표현 (`space.py`)

**정준 행동 `hand_mano` = 54-D** (로봇 무관, D2 사람 시연에서 바로 계산):

```
index : 0:3        3:9                 9:54
        wrist_pos  wrist_rot (6D)      finger_pose axis-angle (15 관절 × 3, MANO 순서)
```

- 손목 회전은 **6D (R 의 첫 두 열)** — axis-angle/쿼터니언의 불연속을 피함 (Zhou et al., CVPR 2019,
  arXiv:1812.07035). 역변환은 Gram–Schmidt 라서 네트워크 출력(비정규 6D)도 항상 유효한 회전으로 복원.
- 손가락은 MANO axis-angle 그대로(각도 < π 영역 → 연속). MANO 관절 순서 index1..3, middle1..3,
  pinky1..3, ring1..3, thumb1..3 (`MANO_FINGER_JOINTS` = `pose.mano.MANO_JOINTS[1:]`).
- `hand_action_to_arrays(a)` → `HandArrays(global_orient, finger_pose, wrist_pos)` — 튜플
  (`go, fp, wp = hand_action_to_arrays(a)`, `hand_action_from_arrays` 인자 순서)이면서 읽기 전용 매핑
  (`h["finger_pose"]`, 키가 `ManoSkeleton.forward` 인자와 동일 → `skel.forward(**h)`). torch 입력은 미분 가능.
- `robot_joint` (D-dim) = 로봇 관절 목표 `q` (URDF actuated 순서) — 로봇 텔레옵 데이터가 있을 때.
- `actions_from_episode(ep, spec)` → `(actions[T,A], valid[T])` (`hand_pose_valid` 반영).

**상대 행동** `make_relative(chunk[...,H,A], state[...,A], spec="hand_mano", mode="delta")` /
`make_absolute` (정확한 역). `spec` 은 `ActionSpec`, 그 dict, `"hand_mano"`(기본) 또는 `"robot_joint"`(차원은
행동에서 추론):

| mode | hand_mano | robot_joint |
|---|---|---|
| `abs` | 그대로 | 그대로 |
| `delta` (기본) | 손목 위치 − 현재 손목 위치 (월드 프레임), 회전·손가락 절대 | `q − q_cur` |
| `delta_pose` | 손목 위치·회전을 **현재 손목 프레임**으로 (`Rsᵀ(p−ps)`, `RsᵀR`) | — |

에고 카메라만 있어 월드 프레임 손목 궤적이 머리 움직임에 섞이는 경우 `delta_pose` 권장.

**정규화** `ActionNormalizer.fit(actions | [arrays], valid, spec=, method=)` — `NormStats` 래핑.
`std`(기본, 6D 포함 모든 차원), `robust`, `minmax`(→[−1,1], flow/diffusion head 용), `none`.
`min_scale`(기본 0.01) 로 거의 상수인 차원의 노이즈 증폭 방지. **학습에 쓰는 표현(절대/상대)과 같은
표현으로 train split 에서만 fit** 하고 `to_dict`/`save` 로 정책 번들에 같이 저장.

## 2. 청킹 / 앙상블 (`chunking.py`) — ACT (Zhao et al., arXiv:2304.13705)

- 데이터는 200 Hz, 정책은 `policy_hz`(기본 20 Hz) → `stride = policy_stride(200, 20) = 10`.
- `action_chunk(actions, t, H, stride, offset=1)` → `chunk[i] = actions[t + (offset+i)·stride]`.
  state-as-action(사람 손 자세 / 로봇 q)에서는 `actions[t]` = 현재 상태이므로 기본 `offset=1`(순수 미래).
  에피소드 끝을 넘는 스텝은 마지막 프레임으로 패딩 + `valid=False` (ACT `is_pad`). `valid=` 로
  `hand_pose_valid` 결측도 마스킹하며, 이때 마스킹된 스텝(끝 넘김·결측)은 그 이전의 **마지막 유효 프레임** 값으로
  채움 → 결측 프레임의 NaN 이 masked loss 로 새지 않음 (NaN·0 = NaN). `action_chunks` 는 배치/벡터화 버전(memmap 에서 필요한 행만 읽음).
- `policy_tick_indices(T, stride, mask=phase_id>=0, min_future=stride)` — 학습 샘플 위치.
- `TemporalEnsembler(H, A, k=0.01)`: `add(chunk)` (정책 호출 때마다) → `step()` (매 실행 스텝).
  현재 스텝을 덮는 모든 예측을 오래된 것부터 `w_i = exp(−k·i)` 로 가중 평균 (ACT 와 동일; 테스트가
  ACT 의 `all_time_actions` 구현과 수치 일치 확인). `reset()` 은 에피소드 시작마다.

## 3. 리타게팅 (`retarget.py`)

```
q* = argmin_{lower≤q≤upper} Σ_v w_v‖v_r(q) − s·R_hr·v_h‖² + λ_reg‖q − q_nom‖² + λ_smooth‖q − q_prev‖²
```

- `v_h`: 사람 손목 프레임에서 손목→손끝(`tips`) + 엄지→각 손가락(`pairs`, DexPilot 식 쌍 벡터; 집기 여부
  결정). `v_r(q)`: 로봇 FK 의 같은 벡터를 base(palm) 링크 프레임에서.
- `scale` = 로봇/사람 크기비 (사람 벡터에 곱함, AnyTeleop 규약). `estimate_scale(tips_ref, q_ref)` 로
  기준 자세(둘 다 편 손)에서 추정. `human_to_robot` = 사람 손목 프레임 → 로봇 base 프레임 회전
  (MANO 정준 프레임: 손가락 −x, 요측 +z, 손바닥 −y — 로봇 URDF 축에 맞춰 설정 필수).
- `pinch_threshold/pinch_distance/pinch_weight` (선택, DexPilot 에서 착안): 사람 엄지–손가락 거리가
  임계 미만이면 로봇 목표 거리를 `pinch_distance` 로 당겨 크기 차이로 집기가 벌어지는 것을 방지.
- 생성자 위치 인자 순서(스펙): `FingertipRetargeter(fk, tip_links, lower, upper, human_tip_names, scale,
  reg_weight, smooth_weight, iters, lr, *, base_link=…, human_to_robot=…, …)`. 입력 손끝/손목 자세에 NaN·inf 가
  있으면 `ValueError` (로봇에 NaN 명령이 나가지 않도록).
- **`scale` 규약 주의**: 스펙 식 `‖s·v_r − v_h‖²` 의 `s` 와 역수 관계 (`scale = 1/s`). 여기서는 AnyTeleop 처럼
  사람 벡터에 곱함 → `scale` = 로봇/사람 크기비 (로봇 손이 작으면 < 1). 최소해는 동일.
- FK 는 추상화: `fk(q[B,D]) → {name: pos[B,3] | T[B,4,4]}` 아무 callable, 또는
  `fk(q, links=...)` + `lower/upper` 를 가진 URDF 모델 (`pose.urdf.URDFModel` 호환, import 하지 않음).
- 솔버: `lm`(기본, projected Levenberg–Marquardt + 관절 한계 active set; 도달 가능한 목표는 정확히 복원),
  `adam`(projected), `lbfgs`(무제약 후 투영). `jacobian="fd"` 는 배치 중앙차분(FK 1회 호출) — 파이썬
  오버헤드가 큰 FK 에서 약 2배 빠름.
- 초기값/정규화 목표 `q_nominal` 기본 = 관절 범위 중앙 (한계에 붙은 곧은 손가락은 특이자세라 경계에
  갇힐 수 있음).
- `retarget(tips[...,F,3], human_wrist_T=None, q_init, q_prev)` 배치 독립 풀이;
  `retarget_sequence(tips[T,F,3])` 프레임별 warm start + smoothness; `step()`/`reset()` 스트리밍 (제어
  루프). `vector_error()` 로 벡터 오차(m) 진단. `from_config(fk, dict)` 는 YAML 블록에서 생성(오타 키는 에러).
- `hand_action_fingertips(a[...,54])` / `human_fingertips(finger_pose)` → MANO 손목 프레임 손끝
  `[...,5,3]` (`pose.mano.ManoSkeleton`, lazy import). 행 순서 = `HAND_FINGERS` (thumb, index, middle,
  ring, pinky) = 기본 `human_tip_names`.

가중치 단위: 벡터 오차는 m, 관절은 rad → `reg_weight`/`smooth_weight` = "1 rad 편차가 몇 m² 오차와 같은가".
기본 `reg_weight=1e-7`(영공간만 결정, 바이어스 무시 가능), `smooth_weight=1e-6`.
16-DoF 합성 손 기준 warm start 는 프레임당 4–5 LM 반복 (CPU 수십 ms, `jacobian="fd"` 로 절반).
고속 제어 루프에서는 `iters` 를 작게(예: 5) 제한.

## 사용 흐름

```python
from robot_skin.action import *
spec = ActionSpec.hand_mano()
a, valid = actions_from_episode(ep, spec)                       # [T,54]
stride = policy_stride(200, 20)
ticks = policy_tick_indices(ep.T, stride, mask=ep["phase_id"] >= 0, min_future=stride)
chunks, cvalid = action_chunks(a, ticks, horizon=16, stride=stride, valid=valid)
rel = make_relative(chunks, a[ticks], spec, mode="delta")      # 선택
norm = ActionNormalizer.fit(rel, cvalid, spec=spec)             # train split 만

# 실행 (control/runner)
ens = TemporalEnsembler(16, 54, k=0.01)
ens.add(make_absolute(norm.unnormalize(pred_chunk), state, spec))
a_t = ens.step()
rt = FingertipRetargeter(urdf_model, {"thumb": "thumb_tip", "index": "index_tip", ...},
                         base_link="palm", human_to_robot=R_mano_to_palm, scale=s)
q_target = rt.step(hand_action_fingertips(a_t))                 # → safety filter → robot
```
