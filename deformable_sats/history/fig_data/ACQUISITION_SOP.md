# 새 센서 취득 SOP (2026-07-20 확정, 07-21 전이 방침 반영)

> 배경: 지표 재평가(`experiments_archive/reeval/`)로 ①rel RMSE 저force 왜곡 ②d5 무접촉(eco20 d5 fz>0.3N 0%) ③0.5mm×10 비효율이 드러남. 이 SOP는 셋을 한 번에 해결 — **공정 비교 가능 + 효율 + force 범위 확보**.
> 상위 전략: Notion "Bending-Aware Tactile Skin — Nature Communications" (G0~G3). 밴딩 취득은 `fig3_sats_bending/bending/README.md`.

## ★ 시나리오: 여러 ecomesh 센서 (센서 간 전이)

- 소재 비교(eco20/eco50)는 **종료**(d10 map 품질로 결론, 재취득 없음). 앞으로는 **ecomesh 센서 여러 개** 제작.
- 각 센서를 처음부터 다 모으지 않음. **기존 xy0.5 13-trial 학습 가중치(`runs/size_input/...`)를 베이스로 warm-start** → 새 센서마다 **소량만** 취득해 유닛 편차(게인·응답)만 적응.
- **센서당 목표 = d5×1 + d10×1 = 2 trial** (홀드아웃은 학습 trial 분할 or 다음 센서로 교차검증).
- ⚠️ "2개면 충분한가"는 **첫 새 센서에서 실측 검증 필요**(리허설 cross-warm은 프록시였음). 첫 센서: 2개 취득 → warm-start → map 품질(loc·peak상관) 확인 → 충분하면 확정, 부족하면 +2. 확정되면 이후 센서 전부 2개씩.

## 확정 파라미터

| 항목 | 값 | 근거 |
|---|---|---|
| **프로토콜** | **계단식 통일** (각 깊이에서 정지·측정) | rate 단일 → 도메인 간섭 없음(혼합 리허설 교훈). magnitude 라벨 정밀 |
| **xy 스캔** | **1 mm 격자** (±10mm, 21×21=441점) | 위치 정확도는 grid·GT가 결정, 스캔밀도 아님 — 전 모델 loc 0.5mm 실증. 0.5mm 대비 점수 1/4 |
| **인덴터** | **d5 + d10** (2종, 동일 z 프로토콜) | 크기입력(A)이 구분. d10 map 품질 최상(peak상관 0.976), d5는 충분히 눌러 force 실리게 |
| **센서당 반복** | **d5×1 + d10×1 = 2 trial** (전이 warm-start 전제) | 첫 센서만 검증차 +2 여유. 소재 비교 아님 → 반복 최소 |
| **깊이(z)** | 각 격자점에서 계단식 하강, **충분한 최대깊이까지** | d5 무접촉(얕은 press) 회피가 핵심. 저~고force 전 구간 커버 |

## ⚠️ 새 센서 z_start 보정 (필수 체크)

인덴터별 z_start(접촉 시작 stage z)는 `raw_merge.Z_START_BY_DIAMETER_MM`에 하드코딩(구 rig: d5=13.0, **d10=12.0**). **새 센서/rig는 이 값이 틀릴 수 있음** — v1(ecomesh)에서 d10=12.0을 쓰니 무접촉 프레임 z_depth가 1.3mm로 부풀려짐(실측). d5·d10 팁 높이가 같으면 z_start도 같아야 함. **검증법**: 병합 후 `Fz<0.05N`(무접촉) 프레임의 z_depth 중앙값이 ~0.3mm 이내여야 정상. 어긋나면 `prepare_learning_data.py --z-start dN=Z` 로 오버라이드 후 재병합. v1은 `--z-start d5=13.0 --z-start d10=13.0`으로 확정(z_stage≈13.5에서 force 시작 실측).

## z 계단 세부 (권장 기본값 — 실측 조정)

- d5: z 0 → 최대 **2.5mm+** (무른 재료라 충분히 깊게 눌러야 force 실림). 0.5mm 단위 정지 권장.
- d10: z 0 → 최대 **3.5mm+**. 넓은 접촉이라 얕으면 저force 노이즈 → 깊게.
- **각 단계 정지 시간**: 센서·로드셀 안정화 충분히(수백 ms+). loadcell 샘플레이트 확인.
- **핵심 원칙**: "얕은 저force 구간에 프레임이 몰리지 않게" — d5 무접촉·저force d10 약신호가 그동안 학습·평가를 오염시킨 근본 원인.

## 취득량 비교 (효율)

| | 기존 (비효율·불공정) | 새 SOP |
|---|---|---|
| xy 스캔 | 0.5mm (1681점) | 1mm (441점) → **1/4** |
| 반복 | 10~13 trial | 인덴터당 3 (d5·d10 = 6) → **약 1/2** |
| force | 얕은 press, d5 무접촉 | 각 점 저~고force 계단 → **전 범위** |
| 비교 | 소재/인덴터 force 제각각 | **동일 조건 → 공정 비교 자동** |

## 소재 비교 (eco20/eco50/ecomesh)

- 새 센서 = ecomesh 우선. 소재 비교는 eco20·eco50 센서도 **동일 SOP**로 취득해야 공정.
- 현재까지 유효 결론: **d10 map 품질 ecomesh(loc 0.5·corr 0.976) > eco20(0.71·0.944) > eco50(1.0·0.901)** — rel 아닌 map 품질이라 신뢰 가능. d5는 무접촉이라 그동안 비교 불가였음 → 새 SOP로 d5도 force 실려 비교 가능해짐.

## 학습·평가 (취득 후)

**핵심 원칙: d5+d10을 반드시 섞어서 학습** (분리 금지 — d10-only는 loc 1.58mm·corr 0.32로 최악, d5가 부족한 d10을 보완). 크기입력(A)으로 크기 구분.

> **방법 명칭(확정):** 이 학습 방식 = **cross-sensor calibration transfer** (센서 간 교정 전이). 소스 센서로 학습한 모델을 새 센서의 few-shot(d5×1+d10×1) 데이터로 **warm-start fine-tuning** → 유닛별 게인·응답 편차만 적응, 전량 재취득 불필요. 상위 우산은 transfer learning / domain adaptation, 코드상 구현은 `--init-ckpt` warm-start.

**③ warm-start 학습** (= calibration transfer, 기존 ecomesh 가중치에서, ~20ep/30분):
```bash
.venv/bin/python -m sats.training.train_e2e \
  --gt-mode gpu_on_the_fly --raw-dir learning_data/sensor_raw_bin \
  --gt-dir learning_data/trial_indices/<새센서인덱스> \
  --val-ratio 0 --val-trials <홀드아웃 d5·d10 각1> \
  --use-indenter-size-input \
  --init-ckpt sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3/best_model.pt \
  --epochs 20 --grid-step-mm 0.5 --run-name newsensor_warm
```
- 인덱스에 d5·d10 train trial 모두 포함(섞음), val-trials로 홀드아웃 분리.
- `--init-ckpt` = warm-start(계단식이라 xy0.5 가중치가 좋은 출발, 전이 리허설 검증). from-scratch도 되나 warm이 성능+시간 우위.
- 데모용 고해상도는 `--grid-step-mm 0.25`(또는 0.1)로 별도 학습(엔진 다해상도 지원).

### 출력 voxel 해상도 판정 (d5 다해상도 실측)

| 출력 grid | rel(정확도) | 위치오차 median | fz 추종 |
|---|---|---|---|
| 0.5mm | ~0.15 | 0.50mm | 0.15 |
| 0.1mm | ~0.15 | **0.22mm** | 0.15 |

- **정확도·fz는 해상도 무관 동일**(정보는 센서 16ch 결정, 해상도로 안 늘어남). **위치 읽기만 0.1mm가 2배 정밀**(argmax 양자화 감소). 0.22mm는 센서 sparsity 물리 한계 → 더 조밀해도 개선 없음.
- **권장**: 학습·기본 = 0.5mm(가볍고 성능 동일), 촘촘한 contact 표시·정밀 위치 = 0.1mm(같은 모델 grid만 변경). 논문 = "출력 grid 확장 ≠ 물리 분해능"(claim boundary).

**④ 평가**: 표준 지표 = **loc + peak 상관 + rel(저force 제외) + 절대 rmse** (rel 단독 금지, reeval 교훈). `reeval_map_quality.py`에 새 run 추가해 재사용.

**공정 비교**: 동일 SOP 데이터라 소재/인덴터를 같은 force 구간에서 직접 비교 가능.

**주의(관찰)**: 크기입력(A)은 magnitude 개선/위치 미세손해 trade-off가 있었으나 저force 홀드아웃 한계 탓 → 새 데이터는 force 커버리지가 좋아 A 유지 권장. 취득 후 A vs no-size 를 map 품질로 재확인.

## 세션 묶음 (한 셋업에)

새 센서 취득 세션에서 함께: ①이 SOP(위치+force, 6 trial) ②밴딩 G0/G1 세트(계단식 접촉과 정합) ③다점 2·3점 zero-shot. 센서 탈부착·온도 변화 최소화.
