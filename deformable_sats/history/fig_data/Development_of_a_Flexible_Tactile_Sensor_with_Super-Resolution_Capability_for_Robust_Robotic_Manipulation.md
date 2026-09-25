# Development of a Flexible Tactile Sensor with Super-Resolution Capability for Robust Robotic Manipulation

> 논문 작성용 본문 정리(working draft). 각 절은 그대로 따다 다듬어 쓸 수 있는 prose 위주로 작성.
> 센서 사양: **SATS 기반, 30 × 30 × 5.5 mm, FPCB + MEMS Barometer, 3중층 구조.**

---

## 0. 한 줄 요약 / Thesis statement

얇고 유연한 FPCB–MEMS Barometer 촉각 센서에 **mesh 압력전달층**을 도입해 sparse taxel 배치에서도 초해상도(SR) 압력·위치 추론을 가능하게 하고, 센서가 곡면에 부착되어 밴딩된 상태에서도 **baseline 변화로부터 곡률을 스스로 추정해 SR 입력 잔차를 보정**함으로써 곡률별 대규모 재학습 없이 SR 성능을 유지하는, **사람 손·로봇핸드 공용 촉각 센서 플랫폼**을 개발한다.

핵심 주장 세 가지:
1. **Mesh층이 receptive field overlap을 공학적으로 키워 SR 성능을 높인다** — 재료 비교(eco20 / eco50 / ecomesh)로 입증, 최종 선택 = **mesh20(ecomesh)**.
2. **밴딩은 baseline의 이동으로 나타난다.** SATS가 baseline 잔차를 학습하므로, 곡률을 추정해 잔차를 보정/conditioning하면 밴딩 상태에서도 SR이 동작한다(점진적 성능 저하).
3. **단일 접점 → 다중 접점 + 동적(시간) 해석**으로 확장한다.

---

## 1. Abstract (draft)

기민한 로봇 조작과 사람 손 동작 분석에는 접촉 위치·압력 분포·힘 변화·슬립을 실시간으로 해석하는 촉각 피드백이 필요하다. 그러나 고해상도 촉각센서는 대체로 taxel 밀도가 높아 배선이 취약하고 제작이 복잡하며, 두꺼운 폼팩터 탓에 곡면 부착이 어렵다. 본 연구는 **FPCB 기반 유연 회로**에 소형 **MEMS Barometer**를 sparse하게 배치하고, 상부에 **mesh 기반 압력전달층**을 두어 인접 taxel 간 수용장 중첩(receptive field overlap)을 유도함으로써, 적은 수의 센싱 유닛만으로 초해상도 압력 분포를 복원하는 30 × 30 × 5.5 mm 촉각 센서를 제안한다. 자기주의(self-attention) 기반 SATS 모델을 **baseline 잔차** 기준으로 학습시켜 평면(flat) 상태에서 고해상도 위치·압력을 복원하고, 센서가 밴딩되면 **무접촉 baseline의 변화 패턴으로부터 곡률을 추정**해 SATS 입력 잔차를 보정함으로써, 곡률별 대규모 재취득 없이 밴딩 상태에서도 SR을 유지한다. 마지막으로 동일 센서를 로봇핸드와 사람 손 같은 곡면에 부착해 실시간 접촉 감지·제어가 가능함을 보인다.

> 정량 수치(채울 것): flat SR RMSE(mm), R², SR scale factor / 소재별 ΔS·확산폭·SR 비교 / 곡률 추정 MAE·R² / bent SR 보정 전후 성능 저하폭.

---

## 2. Introduction

### 2.1 동기 — 조작에는 촉각이 필요하다
기민한 로봇핸드 조작에는 시각 정보만으로는 부족하다. 손 안 조작(in-hand manipulation), 안정 파지, 미끄럼 보상, 물체 재배치 같은 작업에서는 접촉이 손가락과 손바닥에 가려지므로, 외부 카메라 기반 관측만으로는 실제 접촉 상태를 정확히 추정하기 어렵다. 이러한 작업은 접촉 순간의 **위치, 압력 분포, 힘 변화, 미끄러짐, 국소 형상**을 실시간으로 해석할 수 있는 촉각 피드백을 요구한다. 따라서 로봇핸드가 인간 수준의 조작을 학습하려면 손 표면에 직접 통합 가능한 촉각 센서 플랫폼이 필수적이다.

### 2.2 한계 — 고해상도 촉각센서의 비용 구조
촉각센서는 접촉 위치 추정, 상대 압력·힘 추정, 형상 인식, 슬립 감지, 질감 분류를 목표로 광학식·정전용량식·저항식·자기장식·압전식·Barometric 방식으로 발전해 왔다. 일부 센서는 높은 공간 해상도를 달성하지만, 고해상도는 거의 항상 (i) 센싱 유닛(taxel)의 조밀한 배치, 또는 (ii) 카메라·광학 구조·복잡한 배선·정밀 보정을 대가로 한다. 그 결과 센서가 두꺼워지고 제작·유지보수 비용이 증가하며, 배선이 늘어 신뢰성이 떨어지고 신호 간섭(cross-talk)이 발생한다. 무엇보다 두껍고 평면 위주인 폼팩터는 로봇핸드의 곡면 링크나 좁은 손가락 표면에 넓게 적용하기 어렵다. 선행 연구도 고해상도 촉각센서의 한계로 **높은 비용, 취약한 배선, 복잡한 제조 공정, 제한된 폼팩터**를 반복적으로 지적한다.

### 2.3 한계 — 사람–로봇 촉각 데이터의 도메인 갭
현재의 촉각 데이터 기반 학습은 대부분 특정 센서 형상과 특정 로봇 구조에 종속된다. 사람 손과 로봇핸드는 부착 곡률, 피부 변형, 접촉 면적, 손가락 크기, 표면 재질, 열적 조건, 배선 구조가 모두 달라, 같은 센서를 부착해도 baseline과 접촉 응답이 달라진다. 인간 손은 부드럽고 곡률이 큰 피부를 가지며 파지 중 지속적으로 변형되는 반면, 로봇핸드는 링크 구조·관절 배치·표면 강성이 달라 동일 자극에 대한 응답이 어긋난다. 이 차이는 Real2Sim·Sim2Real 과정에서 촉각 신호의 도메인 갭을 만들고, 사람–로봇 간 데이터 공유를 어렵게 한다. 기존 장갑형 촉각센서는 사람 손 동작 분석에는 적합하나 인간 손 치수·관절 기준으로 설계되어, 로봇핸드에 적용하면 센서 위치·장력·접촉면·배선 경로가 달라지고 두께·발열이 관절 운동과 파지 성능을 저해할 수 있다.

이 두 한계를 함께 해소하려면 **얇고 유연하며, 곡면에 부착 가능하고, 센서 간격이 넓어도 고해상도 접촉 정보를 복원할 수 있는** 단일 플랫폼이 필요하다.

### 2.4 접근 — 초해상도 촉각 감지(SR tactile sensing)
SR tactile sensing은 물리적 taxel 수를 늘리는 대신, 소수의 센싱 유닛과 탄성 매질의 신호 확산 특성을 이용해 접촉 위치·압력 분포를 고해상도로 복원하는 접근이다. 이는 인간 피부에서 인접 기계수용기의 수용장(receptive field)이 서로 겹쳐, 물리적 분해능보다 미세한 자극을 감지하는 원리를 모사한 것이다. 수용장이 겹치면 하나의 접촉이 여러 유닛에 부분 정보를 남기고, 신경계(또는 학습 모델)는 이 다중 신호를 종합해 자극 위치를 물리 분해능 이하로 추정한다. 딥러닝은 이 다중소스 신호 통합을 효과적으로 모사한다. 선행 SR 연구(self-attention 기반 SATS, taxel value isoline 이론 등)는 sparse array에서도 손끝 수준에 근접한 위치 추정 성능을 보였다.

그러나 기존 SR 연구는 대부분 **평면 또는 고정 형상**에서 검증되었다. 센서가 사람 손·로봇핸드에 부착되어 **밴딩된 상태에서도 SR 성능을 유지하는지**는 충분히 다뤄지지 않았다. 곡면 부착은 본 플랫폼의 핵심 사용 시나리오이므로, "밴딩 상태에서의 SR"은 선택이 아니라 필수 검증 항목이다.

### 2.5 본 연구
본 연구는 사람 손·로봇핸드 공용의 **bendable super-resolution tactile sensor platform**을 제안한다. FPCB 유연 회로와 소형 MEMS Barometer로 밴딩 시 배선·유닛 손상 위험을 줄이고, 센서 간격을 충분히 확보해 곡면 부착성과 내구성을 높이면서도, **mesh 압력전달층**으로 수용장 중첩을 공학적으로 유도해 SR에 필요한 신호 분포를 형성한다. Barometric tactile sensor는 저비용·단순 구조·압력 직접 측정·실시간 추정 가능성 면에서 그리퍼·SR 연구에 적합한 기반 기술이다.

본 연구의 핵심 질문은 "고해상도 촉각센서를 만드는 것"이 아니라, **평면에서 학습한 SR 모델을 유지하면서도 실제 손 표면처럼 밴딩된 환경에서 동일하게 작동시킬 수 있는가**이다. 이를 위해 (i) 평면 접촉 데이터로 SATS를 학습하고, (ii) 곡률에 따라 변하는 무접촉 baseline을 별도로 모델링하며, (iii) 추정한 밴딩 상태를 기존 SR 추론에 연결한다. 그 결과 곡률별 대규모 데이터 재취득 없이도 밴딩 상태의 SR 압력 분포 추론이 가능하다.

### 2.6 기여(Contributions)
- **C1.** Mesh 압력전달층으로 sparse MEMS-barometric 센서의 수용장 중첩을 키워 SR 성능을 향상시키고, 재료 비교(eco20/eco50/ecomesh)와 SATS 학습 성능으로 그 이득을 정량 입증한다.
- **C2.** 센서가 밴딩되면 baseline 변화로부터 곡률을 자가 추정하고, 이를 SATS 입력 잔차 보정/conditioning에 사용해 밴딩 상태에서도 SR을 유지하는 **bending-aware SR** 기법을 제안·검증한다.
- **C3.** 단일 접점 SR을 **다중 접점 + 동적(시간) 해석**까지 확장한다.
- **C4.** 동일 센서를 로봇핸드와 사람 손 곡면에 부착해, 사람–로봇 도메인 갭을 줄이는 공용 플랫폼으로서의 실용성을 보인다.

---

## 3. Related Work / Background

### 3.1 고해상도 촉각센서와 트레이드오프
- **광학(비전) 기반(GelSight 등):** 카메라로 표면 변형을 관측해 매우 높은 공간 해상도를 얻지만, 구조 제약과 큰 부피 때문에 좁은 손가락·곡면 적용과 다지(多指) 확장이 어렵다.
- **정전용량·저항식 어레이:** 박형화에 유리하나 고해상도를 위해 taxel을 조밀히 배치하면 배선·cross-talk·신뢰성 문제가 커진다.
- **자기장(Hall) 기반:** 다중 접촉을 단일 접촉들로 분리(decouple)해 합성 데이터로 학습하는 접근이 제안되었으나, 일반화 성능이 학습 데이터 다양성에 제한된다.
- **Barometric(MEMS 기압):** 압력을 직접·저비용·간단히 측정. 단일 유닛 응답이 명확하고 시계열 처리에 적합해 SR과 그리퍼 통합에 유리하다. → **본 연구의 센싱 방식.**

### 3.2 초해상도 촉각 감지(SR)
- **수용장 중첩 원리:** 인접 유닛의 수용장이 겹칠 때, 하나의 접촉이 여러 유닛에 부분 정보를 남기고 모델이 이를 종합해 물리 분해능 이하의 위치를 복원한다. 따라서 SR의 전제는 "적절한 수용장 중첩"이며, **너무 좁으면 undersampling, 너무 넓으면 SNR 저하**가 발생한다(최적 중첩이 존재).
- **SATS (Self-Attention-assisted Tactile SR):** sparse taxel 배열을 그래프로 보고(노드=유닛, 간선=수용장 인접) 멀티채널 신호로부터 전 표면 압력 분포를 복원. per-taxel LSTM → self-attention 집계 → local map 구성 → CNN refine 구조. 평면 유연 센서에서 손끝 수준(서브-mm RMSE)에 근접하는 다중 접점 SR을 보고.
- **Taxel Value Isoline(TVI) 이론 / Barodome:** 특정 taxel 출력값을 만드는 (위치, 힘) 조합의 등고선으로 sparse 유닛의 SR을 설명. **전단력(shear)이 존재하면 순수 법선력 대비 위치 정확도가 본질적으로 저하**됨을 이론적으로 예측·검증. → 본 연구의 한계 논의와 향후 확장(다방향 힘)에 직접 연결.

### 3.3 본 연구의 차별점
| 항목 | 기존 SR 연구 | 본 연구 |
|---|---|---|
| 수용장 중첩 형성 | 재료/배치에 의존(암묵적) | **mesh층으로 명시적·공학적 형성** |
| 검증 형상 | 평면 또는 고정 곡률 | **임의 곡률(밴딩) 상태 SR 유지** |
| 곡률 처리 | 고려 안 함/재학습 | **baseline로 곡률 자가 추정 → 잔차 보정(재학습 최소화)** |
| 도메인 | 단일 센서/형상 | **사람 손·로봇핸드 공용 플랫폼** |

---

## 4. Sensor Design & Fabrication

### 4.1 전체 사양
외형 **30 × 30 × 5.5 mm**, FPCB 기반 유연 기판. 센싱 유닛은 소형 **MEMS Barometer**(압력 직접 측정)이며 sparse하게(넓은 간격) 배치해 곡면 부착성과 내구성을 확보한다. 신호는 taxel별 멀티채널 시계열이며, baseline 대비 잔차를 SATS 입력으로 사용한다.

### 4.2 3중층 구조(rationale 포함)
3층으로 분리한 이유는 **압력 전달 / 센싱 / 기계적 지지**의 역할을 분리해 각각을 독립적으로 튜닝하기 위함이다(선행 논문 근거).

| 층 | 구성(확정, 2026-07-12) | 두께 | 역할 |
|---|---|---|---|
| **Top** | Ecoflex 00-20 **·** 00-50 **·** 00-20+**mesh**(ecomesh) 중 택1 — **소재 비교군(C1)** | — | 외부 접촉 압력 수용 및 **인접 taxel로 분산(수용장 중첩 형성)** |
| **Mid** | Ecoflex 00-20, **MEMS embedded** | 2 mm | 센싱 유닛(챔버)층 |
| **Bot** | Ecoflex 00-20 | 1 mm | 베이스/기계적 지지·밀봉 |

> 표기 확정(D4): 각 층은 **층별 단일 소재 적층**(혼합비 아님). 기본 시편 = Bot eco20 / Mid eco20 / Top {eco20 | eco50 | ecomesh}. **변형 시편**: Mid를 Ecoflex 00-45로 한 버전, Bot·Top 동일 소재 버전 존재 → 본문에는 시편별 층 구성을 표로 명시. molding 변경·아랫면 소재 통일은 진행 중(D3).

### 4.3 Mesh가 SR을 돕는 메커니즘(C1의 물리적 근거)
점 접촉 하중이 표면에 가해지면, 그 압력이 하부의 sparse한 MEMS 챔버까지 전달되어야 한다.
- **무른 균질 엘라스토머(eco20):** 압력이 국소화되어 접촉점 근처 taxel만 반응한다. taxel 사이를 누르면 신호가 약하고 모호해 **수용장 중첩이 부족(undersampling)** → SR 복원이 불안정.
- **단단한 엘라스토머(eco50):** 하중이 더 넓게 퍼져 중첩은 늘지만, 민감도가 떨어지고 큰 하중에서 포화하기 쉽다 → **SNR 저하**.
- **Mesh 통합층(ecomesh):** mesh가 컴플라이언트 엘라스토머 내부의 **준강성 하중 분산 골격**으로 작용해, 표면 압력을 측방으로 채널링하여 다수 챔버로 전달한다. 결과적으로 각 taxel의 **유효 수용장이 커지고 인접 taxel 간 중첩이 증가**하되, 엘라스토머의 민감도는 유지된다 → **중첩과 SNR의 균형**으로 SR에 가장 유리.

요지: mesh = "**taxel을 늘리지 않고 수용장 중첩을 공학적으로 주입**"하는 구조. 이는 SR의 전제 조건을 재료/배치 운에 맡기지 않고 설계로 보장한다는 점에서 기여다.

---

## 5. Methods

### 5.1 SATS 요약과 본 연구의 입력 정의
SATS는 sparse taxel을 그래프로 모델링하고, per-taxel LSTM으로 각 유닛의 hysteresis·고유 응답을 인코딩한 뒤 self-attention으로 공간 통합, local map 구성, CNN refine으로 고해상도 압력 분포를 복원한다.

본 연구는 입력을 **baseline 잔차**로 정의한다:
$$ \mathbf{r} = \mathbf{p}_{\text{raw}} - \mathbf{p}_{\text{baseline}} $$
즉 SATS는 절대 압력이 아니라 "baseline 대비 변화량"을 학습한다. 이 정의가 **C2(밴딩 보정)의 출발점**이다 — 밴딩은 본질적으로 baseline의 이동이기 때문이다.

### 5.2 평면(flat) Ground-truth 취득
로봇팔 + 6축 F/T 센서(또는 loadcell)로 센싱 표면 각 위치를 법선 방향으로 목표 변위/힘까지 가압하고 해제·이동한다. 로봇팔 좌표·측정 힘 = ground truth, 센서 멀티채널 응답 = 입력. 타임스탬프 정렬 후 슬라이딩 윈도우(LSTM 입력 길이)로 분할한다. 인덴터/지그·격자 간격은 Fig.2/3 사양 참조.

### 5.3 밴딩의 관측 가능성과 곡률 자가 추정(C2-가정 1)
센서를 1축(중심축)으로 굽히는 경우를 가정한다.
- 곡률 `κ = 1/R`, taxel i의 중심축 거리 `z_i`, 탄성 변형률 `ε_i ≈ κ·z_i`.
- 변형률이 챔버 부피를 바꿔 무접촉 상태에서도 압력이 변한다:
$$ \Delta p_i^{\text{(bend)}} \approx k_i\,\kappa\,z_i $$
- `k_i`는 taxel i의 압력 민감도(재료·위치 의존). 중심축 근처(`z≈0`)는 변화 거의 없음, 먼 taxel일수록 `|Δp|`가 선형적으로 증가.
- `k_i`가 유사하면 1차원 스케일링 문제가 되어 닫힌형/리지 회귀로 곡률을 추정:
$$ \Delta \mathbf{p}^{\text{(bend)}} = \kappa\,(\mathbf{k}\odot \mathbf{z}), \qquad \hat{\kappa} \propto \frac{\sum_i z_i\,\Delta p_i}{\sum_i z_i^2} $$

→ **무접촉 baseline의 공간 패턴 자체가 곡률의 관측량**이다. 따라서 센서는 외부 센서 없이 자기 신호만으로 밴딩을 감지한다.

전제(본문 명시): 측정 시 접촉 하중 없음(pure bending), 온도·장기 drift 별도 보정, 단일 축 굽힘 가정(2축·비틀림은 제외 또는 별도 모델링).

### 5.4 Bending-aware SR(C2-가정 2): 잔차 분해와 점진적 저하
밴딩 + 접촉이 동시에 있으면 raw 신호는 다음으로 분해된다:
$$ \mathbf{p}_{\text{raw}}(\text{contact},\kappa) \approx \underbrace{\mathbf{p}_{\text{baseline}}(\kappa)}_{\text{밴딩 offset}} + \underbrace{\mathbf{r}(\text{contact},\kappa)}_{\text{접촉 잔차}} $$
평면 학습 SATS는 `r(contact, 0)`을 기대한다. 만약 flat baseline을 그대로 빼면 잔차에 밴딩 offset이 섞여 입력이 오염된다.

처리 전략(택일/혼합):
- **(a) 잔차 보정:** 추정한 `p_baseline(κ)`를 빼서 접촉 잔차 `r(contact,κ)`만 남긴다. 만약 baseline 제거 후 잔차가 곡률에 거의 불변(`r(contact,κ) ≈ r(contact,0)`)이면, **평면 모델을 그대로 재사용** 가능 → 가장 경량.
- **(b) Conditioned SR:** 곡률을 입력에 추가, `(x,y)=f(\mathbf{r},\kappa)`.
- **(c) 좌표 보정:** taxel 좌표계를 평면 기준으로 환산 후 평면 모델 적용.
- **(d) per-curvature submodel + interpolation.**

**점진적 저하 논거:** 밴딩이 baseline를 주로 *이동*시키고 민감도/기하를 크게 바꾸지 않는 1차 영역에서는 (a)만으로 충분하다. 잔차가 곡률에 의존하는 정도(기하·강성 2차 효과)가 곧 성능 저하의 크기이며, 본 연구는 이를 **flat vs bent SR(보정 전/후)** 비교로 정량화한다. 핵심 메시지: *"성능이 다소 떨어지더라도, 곡률별 대규모 재학습 없이 사용 가능하다."*

**도메인 갭과의 연결:** 평면 보정값과 곡면 배치(로봇핸드/사람 손) 사이의 차이는 결국 baseline 이동으로 환원된다. 따라서 5.3–5.4의 baseline 모델링은 **곡면 배치에 대한 경량 도메인 적응** 역할도 겸한다(C4와 연결).

### 5.5 다중 접점 + 동적 해석(C3)
- **다중 접점:** 단일 접점 합성/분리 또는 다중 접점 직접 학습으로 SR 맵에서 복수 피크를 복원.
- **동적(시간) 해석:** LSTM 시계열 특성을 활용해 접촉의 시간적 천이(접근–접촉–슬립–해제)를 추적 → 슬립/접촉 안정성 지표로 확장.
- **한계 연결(TVI):** 전단력이 동반되면 SR 위치 정확도가 본질적으로 저하되므로, 다방향 힘 상황은 별도 모델/향후 과제로 명시.

---

## 6. Figure Plan (요약)

- **Fig.1 — Concept:** 사람 손/로봇핸드 공용, 얇고 유연, 곡면 부착, sparse MEMS + mesh, SR. 컨셉 이미지 중심.
- **Fig.2 — C1(소재 ablation):** 인덴터 원형 d5/10/15·사각 5/10/15(fillet 2mm), xy 1mm 격자 압입, 소재별(eco20/eco50/ecomesh) 수집. 패널: (A) 셋업·격자 모식도, (B) 소재별 baseline-정규화 ΔS heatmap, (C) 총 |ΔS|·확산폭·활성 taxel 수 비교, (D) 소재별 3 set SATS 학습 RMSE/R². 결론: mesh20 최종 선택.
- **Fig.3 — C2(SATS 최종 + 밴딩):** ecomesh flat SR(gap 0.5mm, z=1.5mm, d5, raw fz, 10 set) + 각도별 bending baseline(10 set, jig). 패널: (A) jig·각도 정의, (B) 각도별 Δbaseline vs z_i, (C) κ·θ 추정 회귀(MAE/R²), (D) flat vs bent SR 보정 전/후 RMSE/R², (E) 추론 vs GT SR 맵.
- **Fig.4 — Application:** (1) 로봇핸드 부착 실시간 감지·제어, (2) 사람 손 곡면 부착 동작 시연.

---

## 7. Experiment / Data Checklist (2026-07-12 갱신 — 폴더 = §6 매핑, `PROJECT_STRUCTURE.md` 참조)

> **투고 로드맵 마스터 = `SUBMISSION_CHECKLIST.md`** (기여별 상태·P0 critical path·결정사항 D1~D4·§9 수치 확보 현황·실행 순서). 아래는 취득 항목 요약.
- **Fig.2:** [x] 인덴터/지그 제작(원형 d5/10/15, 사각 5/10/15 fillet2) · [x] 소재별 중앙점·baseline % 시각화 · [x] 인덴터별(d5·d10)·소재별 CSV 변환+패널 A/B/C 생성(→ `fig2_material_ablation/Analysis_Results/Fig2_report.md`) · [x] **소재별 SATS 학습 비교(패널 D)** — 크기입력(A) 모델, d10_rel ecomesh 0.182 < eco20 0.259 < eco50 0.336(→ `fig2_material_ablation/panelD_sats/`) · [~] molding 변경·아랫면 소재 통일(**진행 중**, D3) · [x] 소재별 3-set **재분석 완료**(2026-07-13 — set간 CV<2%로 기존 결론 유지, `Fig2C_metrics_3set`, eco50 d5 dead채널 캐빗)
- **Fig.3:** [x] ecomesh flat SR(xy0.5 13 trial 취득·A 모델 학습·패널 완료 → `fig3_sats_bending/flat_sr/`) · [ ] **각도별 bending baseline 10 set(jig, 무하중, signed deg)** — 취득 스펙 `fig3_sats_bending/bending/README.md`, 코드 `sats/bending/`(Phase 0 완료) · [ ] 밴딩+접촉 세트(대표 각도, xy0.5 프로토콜) · [ ] 곡률 회귀 MAE/R²(P1) · [ ] flat vs bent SR 보정 전/후 비교(P3)
- **Fig.4:** [ ] 로봇핸드+센서 실시간 데모 · [ ] 사람 손 곡면 부착 데모
- **보강 취득(성능):** [ ] 저force d10 반복취득(xy0.5 계단식 동일 프로토콜 — d10 magnitude 개선 유일 해법) · [ ] 다점 2·3점 zero-shot 테스트 세트(SATS 논문 Fig4E 재현, 재학습 불필요)
- 로깅 코드(2026-07 재구성): `skin_ws/acquisition_code/final_logger.py`(통합 raw binary 로거) + `loadcell_bin_logger.py` / 변환 `convert_bins.py` / 보조 도구는 `tools/`(device_tests·gui·node_generation·analysis), 구버전 `legacy/`. 상세 `skin_ws/acquisition_code/README.md`.

---

## 8. Discussion / Limitations
- **전단력(shear):** TVI 이론대로 전단 동반 시 위치 정확도 본질적 저하. 현 모델은 법선 하중 중심 → 다방향 힘은 향후 과제.
- **밴딩 가정:** 단일 축·무하중 baseline 가정. 2축 bending·비틀림·접촉 중 밴딩 변화(coupled)는 별도 모델 필요.
- **온도/drift:** barometric 특성상 장기 drift·온도 보정 전제. baseline 추정과 분리해 관리.
- **수용장 최적값:** mesh로 중첩을 키우되 과도하면 SNR 저하 → 최적 중첩(=최적 mesh 사양) 존재. mesh20 선택의 정량 근거를 Fig.2-C로 제시.
- **일반화:** 동일 플랫폼이 사람 손·로봇핸드 도메인 갭을 줄이나 완전 제거는 아님 → baseline 도메인 적응으로 완화.

---

## 9. 채울 정량 수치(Results 표용)

> **표준 지표(2026-07-20 확정)**: rel RMSE 단독 금지 — 저force(target 압력≈0)에서 분모 폭발 왜곡. **위치=loc(argmax 오차), 형태·강도=peak 상관, magnitude=peak 비율, + rel(저force 제외)·절대 rmse 병기.** 근거: `experiments_archive/reeval/`.

- **Flat SR 위치(loc)**: ecomesh **0.5 mm**(map 품질 재평가, d10) / S20 argmax 0.79 mm(xy1). peak 상관 0.976.
- **소재 서열(map 품질, d10)**: **Eco-mesh(loc 0.5·peak상관 0.976) > Eco20(0.71·0.944) > Eco50(1.0·0.901)** — rel 왜곡 배제, `Fig2D_B_mapquality.png`. (d5는 무접촉이라 소재 비교 불가 → 새 SOP로 해결 예정.)
- **SR scale factor**: ≈105× (41²/16), 다해상도 27×~2525× 안정.
- **d5/d10 섞음 효과**: 섞어 학습이 d10에 유리(d10-only loc 1.58 ≫ 섞음 0.71). 위치·해상도 추종 — 해상도별 loc median 0.5mm(0.5mm)→0.22mm(0.1mm), fz 적분 ~15% 해상도 불변.
- 곡률 추정: MAE(°/κ), R² — 밴딩 취득 후
- Bent SR(보정 전/후): loc·peak상관·rel(저force제외) 저하폭 — 밴딩 취득 후
- 다중 접점 분리 / 동적(슬립) 지표 — 취득 후

> ⚠️ 과거 초안의 "d10_rel 0.749 / 소재 d10_rel 0.182·0.259·0.336" 수치는 저force 왜곡 포함 → **재인용 금지**, 위 map 품질로 대체.

### 9.1 측정값 — 소재 ablation (Fig.2, xy_1mm, 2026-06-23 / 각 소재 d5·d10 1세트)
> 출처: `visualizing_scripts/xy_1mm/Analysis_Results/Fig2_report.md` (패널 B 수용장, 패널 C 배열 메트릭). 미충족 항목(SR RMSE/R²)은 패널 D 학습 후 추가.

**d5 (인덴터 작아 소재 대비 선명):**

| 메트릭 | eco20 | eco50 | ecomesh | 경향 |
|---|---|---|---|---|
| peak ΔS (%, 중앙4 평균) | 22.1 | **46.8** | 44.0 | 민감도 eco20 ≪ eco50≈ecomesh |
| 평균 수용장폭 (mm, 중앙4 half-max) | **3.56** | 3.61 | 3.78 | d5 국소라 셋 다 6.5mm pitch 미달 |
| Total \|ΔS\| (%) | 13.2 | **25.5** | 24.3 | 민감도 eco20 ≪ eco50≈ecomesh |
| Active taxels (N) | 1.38 | 3.77 | **3.95** | eco20 최저, ecomesh 최대 |
| Propagation σ_prop (mm) | 0.61 | 2.60 | **2.78** | eco20 최저, ecomesh 최대 |
| Response entropy H_norm | 0.034 | 0.213 | **0.230** | eco20 최저, ecomesh 최대 |

> per-press local baseline(접촉 직전 기준) 적용 → d5 eco20 σ_prop 가 3.22→0.61 로 교정(이전 전역 baseline 의 drift 부풀림 제거). 이제 eco20 이 4지표 모두 최저로 일관.

**d10 (⌀10mm > taxel 간격 6.5mm, 다중 taxel 관여):** 평균 수용장폭(중앙4) = eco20 **5.07** < ecomesh 6.25 ≈ eco50 **6.63 mm** → eco50·mesh 가 **pitch 6.5mm 에 근접/도달(이웃 수용장 overlap = SR 가능)**, eco20 은 미달(undersampling). 배열: **active = eco20 4.87 < eco50 6.19 < ecomesh 6.91 (mesh 최대)**; σ_prop·entropy 는 **eco50 ≈ ecomesh 동급**(3.61/0.349 ≈ 3.58/0.340); Total\|ΔS\| 은 eco50 최고(112), mesh 그 ~91%(102). peak 는 ~100% 포화.

**요지(C1, 정직판):** local baseline 으로 drift 부풀림을 제거하면 "mesh 가 σ·entropy 로 가장 넓고 고르게 퍼진다"는 **성립하지 않는다**(eco50 와 동급). 데이터가 지지하는 주장 = **"mesh 는 가장 많은 taxel 을 활성화(active 최대)하면서 eco50 급 민감도를 유지하고 eco20 의 undersampling 을 피한다"** → 다중-taxel 입력이 가장 풍부 = SR 유리, **mesh20 선택**. (이전 초안의 "확산 3지표 모두 mesh 최대"는 drift 인공물이었으므로 폐기.)
> 메트릭 정의(2026-06-25 갱신): 배열 지표는 |ΔS| 절댓값 + 절대 floor 0.5%, 수용장 σ는 임계 후 계산, peak 포화 대응으로 half-max 폭 추가. 통계 강건성 위해 소재당 3 set 반복 필요.

---

## 10. 참조(배경 기법)
- **SATS:** sparse taxel → 그래프 + per-taxel LSTM + self-attention + local map + CNN refine로 SR 압력 분포 복원(딥러닝 기반 tactile SR).
- **Taxel Value Isoline(TVI) / Barodome:** sparse 유닛으로 법선·전단 포함 SR, 전단 시 성능 저하의 이론적 설명·검증.
- **Barometric tactile sensing:** 저비용·단순·압력 직접 측정 → 그리퍼·SR 통합에 적합.
