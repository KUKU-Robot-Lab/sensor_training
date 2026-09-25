# contact/ — 잔차 → 접촉 해석

부호: ΔS 는 SATS 규약(누르면 음수). 임계값·z 는 `common.signal.press_intensity`(= −ΔS) 기준의 양수.

| 파일 | 역할 |
|---|---|
| `residual.py` | `residual = 관측 ΔS − baseline 예측`, `contact_mask`(−residual ≥ 임계 & trusted) |
| `ordinal.py` | `OrdinalQuantizer`: {0 무접촉, 1 약, 2 강, 3 포화}, `one_hot` |
| `saturation_fsm.py` | per-taxel `SaturationFSM`: OK → SATURATED → RECOVERING → OK / 타임아웃 시 **re-zero**. 기본값은 `sats/inference/run_dashboard.py` 격리 규칙(1.5 %, 2 s, 30 s) |
| `self_touch.py` | 손가락 캡슐 거리 기반 **자가 접촉 자동 라벨** |
| `calibration.py` | **`ResidualCalibrator`**: D1 val 무접촉 프레임으로 taxel별 robust σ(MAD·1.4826), 중심 c, 이득 g 적합 → `z = (p − c) / (g·sqrt(σ² + exp(logvar)))` (baseline 예측 분산 사용 시). 레벨: `WEAK ⇔ z ≥ weak_z ∧ p ≥ weak_floor_pct`, `STRONG` 동일, 포화/FSM 미신뢰 → `SATURATED`. `saturation_gate`(FSM 오프라인 실행), `residual_levels`(오프라인 단일 진입점), `to_dict/save/load` |
| `detector.py` | **`ContactDetector`**: taxel별 인과 z 이력 `[W]`(+포화 플래그) → causal dilated conv(taxel 공유) + 관절속도 요약(RMS, max) + 선택적 taxel 임베딩 → logit. `focal_loss`(Lin et al. 2017)·`contact_loss`(focal/BCE), 출력 bias = prior π 초기화, `predict_contact_prob`(= `predict_episode`), `CausalDetectorStream`(온라인, 오프라인과 동일), `save/load_detector` |
| `hysteresis.py` | **`HysteresisFilter(on_thr, off_thr, min_on, min_off)`**: 두 임계 + 체류 틱 디바운스, `step`(온라인) ≡ `run`(오프라인). NaN = 낮음(가짜 접촉 방지) |
| `pseudo_label.py` | **`pseudo_label_episode`**: D2 미지 프레임에 detector 확률 + 히스테리시스 + 페이즈 기대(`none`→0, `object/self/any`→허용) + 포화(허용 페이즈에서 1) + 선택적 손–물체 근접 veto 를 융합 → `derived/contact_label_pseudo` (int8 −1/0/1). 기존 확실한 라벨은 유지. `phase_expectation`, `frame_expectation`, `taxel_world_positions`, `pseudo_label_metrics` |

`ContactDetector` 등 torch 모듈은 `contact/__init__` 에서 지연 import(PEP 562) — `ordinal`/`saturation_fsm`
사용자는 torch 를 import 하지 않는다.

## 오프라인 ≡ 온라인 (CTRL `control/online.py`)

stage `contact` 가 쓰는 값과 배포 시 값이 같으려면:

1. `calibrator.json` (`ResidualCalibrator.load`) 하나에 σ, c, g, 임계값, `use_logvar`, `fsm` 설정이 모두 있다.
2. 틱마다: `r = ΔS − baseline_mean` → (`(cal.fsm or {}).get("enabled")` 이면 같은 설정의 `SaturationFSM.step(r, sat, dt)`,
   `r ← fsm.corrected(r)`, 미신뢰 = state ≠ OK) → `z = cal.transform(r, logvar)`,
   `level = cal.levels(z, saturated | untrusted, press_pct=cal.press(r))`.
3. detector: `CausalDetectorStream(load_detector(...)).push(z, saturated, qd, q_valid)` → prob,
   이어서 `HysteresisFilter(**bundle_meta["hysteresis"]).step(prob)`.

## Stage

`python -m robot_skin.stages.contact --config robot_skin/configs/stages/contact.yaml`:
calibrator(D1 val) → 모든 에피소드 `derived/residual_z`, `contact_level` → detector 학습(D1 라벨: self-touch 1,
no-contact 0; `detector.bootstrap: auto` — train split 에 self-touch 라벨이 없으면(로봇 D1: 기하 self-touch 없음)
접촉 허용 페이즈(pinch 등)의 STRONG 레벨을 양성으로 자기학습) → `derived/contact_prob` → D2 `derived/contact_label_pseudo`. 에피소드의 `contact_label` **배열은 절대
수정하지 않는다**(전처리 산출물). 지표: 무접촉 hallucination(taxel/frame), self-touch recall/precision/F1/AUROC
(z 규칙, detector, 히스테리시스), 합성 GT AUROC, pseudo 라벨 통계.
