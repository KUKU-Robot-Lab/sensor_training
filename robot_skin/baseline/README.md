# baseline/ — 무접촉 ΔS 예측기 (motion artefact baseline)

**`deformable_sats/sats/bending/` 의 deg→offset `BaselineRestorer` 를 일반화한 것.**
bending 은 "밴딩 각도 1개 → 16채널 오프셋"이었다면, 여기서는 손이 움직일 때 각 taxel 이
**접촉 없이도** 보이는 ΔS%(늘어남·굽힘·공압 커플링)를 관절 상태와 taxel 자세로 예측한다.
접촉 신호는 `contact/` 에서 `residual = 관측 − 예측` 으로 분리한다.

| 파일 | 역할 |
|---|---|
| `temporal.py` | **v2 (stage `baseline`)**: `TemporalBaselinePredictor` — 인과(causal) 관절 이력 `q_hist, qd_hist [W,D]` → 시간 인코더(causal dilated TCN 또는 GRU) → FiLM 방식 per-taxel 헤드(`pos, nrm` at t + taxel 임베딩) → `(mean, logvar) [B,N]`. `gaussian_nll`(Kendall & Gal 2017), `baseline_loss`, `predict_episode`(오프라인, 인과), `CausalBaselineStream`(온라인 링버퍼, 오프라인과 동일 출력), `episode_joint_view`(`q_source`), `save/load_baseline_model` |
| `model.py` | v1 `BaselinePredictor`: 순간 상태 MLP (변경 없음) |
| `dataset.py` | v1 `NoContactSession` / `NoContactWindowDataset` (변경 없음) |
| `train.py` | v1 `train_baseline`, `predict_session` (변경 없음) |

## 설계 원칙

- **관측 ΔS 를 입력으로 넣지 않는다** — `forward(q_hist, qd_hist, pos, nrm)` 에는 ΔS 인자가 없다
  (넣으면 접촉까지 오프셋으로 학습해 지운다: bending `eval_contact_preservation` 에서 확인된 seq_deg 붕괴).
  테스트(`test_observed_delta_is_never_an_input`)가 시그니처를 고정한다.
- mean 헤드 zero-init(항등 웜스타트: 학습 전 residual == 관측), logvar 헤드 bias = `log σ0²`.
- 정규화는 모델 버퍼에: `set_joint_stats(q, qd)`(`datasets.stats.compute_stats`), `set_target_scale`(taxel별
  무접촉 ΔS RMS). `forward` 는 **원 단위**(rad, rad/s, m)를 받고 ΔS % 를 낸다 → 체크포인트 하나로 온라인 재현.
- 손실(기본): mean 은 target-scale MSE, 분산은 **stop-gradient** Gaussian NLL(`detach_mean`, `var_detach`).
  공동 NLL 학습은 mean 수렴이 매우 느렸다(1/σ² 가중과 큰 기울기가 공유 특징을 흔들고, grad clip 이 이를 더 늦춤).
  `mean_loss="none", detach_mean=False, var_detach=False` 로 순수 Kendall & Gal 목적함수 복원.
- 분산은 aleatoric(데이터 잡음) 만 표현한다. 학습에 없던 자세(예: D2 파워그립)의 외삽 오차는 σ 에 반영되지 않는다 →
  `contact.ResidualCalibrator` 의 % floor 가 최소한의 안전장치. 아래 "OOD 한계" 참고.

## OOD 한계: aleatoric σ 와 D2 거짓 접촉

`logvar` 는 **aleatoric**(같은 입력에서의 데이터 잡음) 분산이다 (Kendall & Gal 2017 의 구분). D1 무접촉 블록에서
본 적 없는 손 자세 — 대표적으로 D2 의 쥐기 손모양(엄지 대립, 손가락마다 다른 굽힘) — 에서는 mean 이 아티팩트를
**과소 예측**하는데 σ 는 작은 채로 남는다 (모델이 "모른다"는 것을 표현할 수단이 없음). 그러면
`z = press(r) / σ` 가 커져서 비접촉 taxel 이 접촉으로 읽힌다. 합성 데이터 리뷰: D2 manipulate/release 의 비접촉
taxel 에서 과소 예측 ≈ 1.6 %, σ ≈ 0.3 → z ≈ 20, D2 pseudo-label 정밀도 0.30–0.70 (학습 시드에 따라).

대응 (우선순위 순):
1. **데이터로 덮기** — D1 프로토콜의 `air_grasp_{slow,fast}_{power,precision,lateral,tripod}` 블록
   (`acquisition/protocols/d1_motion.yaml`): D2 쥐기 손모양을 물체 없이 공중에서 만들어 유지(무접촉 라벨). 합성 D1
   (`datasets.synthetic`, glove) 도 ≈ 6.4 s 보다 긴 세션부터 `air_grasp_slow_power`, 블록 순환이 한 바퀴 돌 만큼
   긴 세션(≈ 17.5 s 이상)에서는 `air_grasp_fast_precision` 도 갖는다. 가장 싸고 효과가 크다.
2. **epistemic σ (앙상블, 미구현 — 선택지)** — 서로 다른 `train.seed`(원하면 에피소드 bootstrap 재표집)로 K 개
   baseline 을 학습하고 `mean = mean_k μ_k`, `var = mean_k exp(logvar_k) + var_k(μ_k)` (모델 간 불일치 = 분포 밖에서
   커지는 epistemic 항)를 `logvar = log var` 로 `ResidualCalibrator(use_logvar)` 에 넣는다. 번들에 K 개 모델,
   온라인에 K 개 `CausalBaselineStream` 이 필요해 추론 비용이 K 배다 (제어 루프 예산 확인). 현재 stage 는
   단일 모델만 학습한다 — 도입 시 `stages/baseline.py` 에 `model.ensemble: K` 옵션과 번들/`control.online` 확장이
   함께 필요하다.
3. **라벨 단계 안전장치** — `contact.pseudo_label` 의 손–물체 근접 veto(`proximity.max_dist_m`), 페이즈 기대
   (reach/retreat = none), `ResidualCalibrator` 의 % floor. 거짓 접촉을 줄이지만 원인(외삽)은 그대로다.

점검: baseline stage 의 `task/` 지표와 D2 의 contact-none 페이즈(reach/retreat)에서 `residual_z` 분포를 본다 —
무접촉인데 z 꼬리가 길면 D1 자세 범위가 D2 를 덮지 못한 것이다.

## 인과 윈도 계약 (CTRL `control/online.py` 가 그대로 따라야 함)

프레임 t 의 예측 입력:
1. `q[t−W+1 … t]`, `qd[t−W+1 … t]` (오래된 것부터). t−W+1 < 0 인 인덱스는 **프레임 0 으로 edge-padding**
   (스트림은 첫 샘플을 반복). `W = model.window` (체크포인트 `config`).
2. `pos[t]`, `nrm[t]`: taxel 자세, **손/로봇 base 프레임**(glove: `global_orient=0`, 손목 원점 — `datasets.build` 규약).
3. `q` 는 에피소드와 같은 원 단위·열 순서(glove: MANO finger pose 45 = `HAND_Q_NAMES`, robot: URDF 관절 순서),
   `qd` = `datasets.build.joint_velocity(q_buffer, hz, **bundle_meta["qd"])` (기본 causal Savitzky–Golay 50 ms;
   `bundle_meta["qd"]` 는 `joint_velocity` kwargs 만 담는다 = `qd_settings(meta.preprocessing)`).
   `bundle_meta["qd_source"] == "file"` 이면(로봇, 드라이버 속도로 전처리) 드라이버 `qd` 를 그대로 쓴다.
4. 모델은 윈도의 순수 함수(상태 없음). TCN 기본 깊이의 receptive field(31) ≤ W(32).

```python
from robot_skin.baseline import load_baseline_model, CausalBaselineStream
model = load_baseline_model("robot_skin/runs/baseline")          # baseline_model.pt
stream = CausalBaselineStream(model)                              # 세션 시작마다 stream.reset()
mean, logvar = stream.push(q_t, qd_t, pos_t, nrm_t)               # [N], [N]  == predict_episode(...)[t]
residual_t = delta_t - mean
```

`model.bundle_meta`: `q_source`, `window`, `qd`(joint_velocity kwargs), `qd_source`, `joint_names`, `layout`, `joint_stats`, `target_scale`, 지표.

## q_source

- `q` (기본): 비전 손 라벨(glove) / joint_state(robot).
- `hand_pose_imu`: `imu_pose` stage 의 `derived/hand_finger_pose_imu` 로 q, qd, taxel 자세를 재계산한 view
  (`episode_joint_view`). 카메라 없는 glove 배포와 같은 입력 — IMU 자세 오차에 강건한 baseline.

## Stage

`python -m robot_skin.stages.baseline --config robot_skin/configs/stages/baseline.yaml` — D1 무접촉 프레임으로 학습,
모든 에피소드에 `derived/baseline_pred`, `baseline_logvar`, `residual` 기록, `metrics.json`
(`val/resid_reduction = 1 − MAE(ΔS−pred)/MAE(ΔS)`, NLL, `coverage_2sigma`, 움직임-접촉 분리도 before/after,
합성 데이터면 `gt_artefact_mae`).

관련 연구: Yu et al., *Pose-Aware Modeling to Mitigate Pose-Related Artifacts in Tactile Gloves* (arXiv:2607.22964),
불확실성: Kendall & Gal, NeurIPS 2017 (arXiv:1703.04977) — `docs/REFERENCES.md`.
