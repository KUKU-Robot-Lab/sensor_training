아래 설계로 가는 것이 적절함.

핵심은 **옵션 B 유지** + **z/힘 보조 헤드 추가**임. 즉, 기존의 xy heatmap 기반 구조는 유지하고, point label만 soft target으로 바꾸며, 별도 회귀 헤드로 `z_depth`와 `fz`를 함께 학습시키는 방식이 가장 안전함. 현재 체크리스트의 방향과도 정확히 일치함. 

# 권장 로스 설계안

## 1. 출력 헤드 구성

### 기본 구조

* **shared backbone**

  * 입력: `16ch sensor signal`
  * 추가 입력: `z/힘 채널` 사용 시 concat 가능
* **head 1: xy heatmap**

  * 출력: `H x W` 1채널 heatmap
  * target: depth-aware soft label
* **head 2: z_depth regression**

  * 출력: scalar 1개
  * target: `z_depth`
* **head 3: fz regression**

  * 출력: scalar 1개
  * target: `fz_bc` 또는 정제된 force

즉 최종 출력은 아래와 같음.

```text
y_pred = {
  "xy_heatmap": [B, 1, H, W],
  "z_depth":    [B, 1],
  "fz":         [B, 1],
}
```

이 구성이 좋은 이유는 다음과 같음.

1. 기존 heatmap 학습 구조를 거의 건드리지 않음
2. xy는 분포 예측, z/fz는 스칼라 회귀로 역할 분리가 명확함
3. depth-aware label의 효과를 xy branch에만 집중시킬 수 있음
4. 이후 ablation이 쉬움

   * xy only
   * xy + z
   * xy + fz
   * xy + z + fz

체크리스트의 `옵션 B`와 `z/힘 채널 또는 z-head 추가` 방향에 그대로 부합함. 

---

## 2. xy heatmap loss

옵션 B에서는 **기존 heatmap 분류를 soft target 허용 형태로 바꾸는 것**이 핵심임. 

### 권장 1안: BCEWithLogitsLoss + soft target

가장 구현이 단순하고 기존 classification head와의 연속성이 좋음.

수식:

[
L_{xy} = \mathrm{BCEWithLogits}(M_{pred}, M_{gt}^{soft})
]

여기서

* `M_pred`: 모델 출력 logits
* `M_gt_soft`: depth-aware soft heatmap label

다만 heatmap 대부분이 background라서 foreground가 묻힐 수 있음.
따라서 실제로는 **가중 BCE** 형태가 더 적합함.

[
L_{xy} =
-\frac{1}{N}\sum_i
\left[
\alpha , M_i \log \sigma(\hat M_i)
+
(1-M_i)\log(1-\sigma(\hat M_i))
\right]
]

권장:

* foreground weight `α = 3 ~ 10`
* 또는 soft target 합이 1이 되도록 normalize 후 사용

### 권장 2안: weighted MSE

soft heatmap이 본질적으로 연속 분포이므로, 회귀 관점에서는 오히려 더 자연스러움.

[
L_{xy} = \frac{1}{N}\sum_i w_i(\hat M_i - M_i)^2
]

권장 weight:

* `w_i = 1 + \lambda M_i`
* 예: `λ = 4`

이 방식은 soft Gaussian label과 잘 맞고, peak shape 자체를 맞추는 데 유리함.

### 결론

현재 구조가 “분류형 heatmap”에 더 가깝다면:

* **1차 구현:** `BCEWithLogitsLoss`
* **2차 비교 실험:** `weighted MSE`

로 가는 것이 가장 실용적임.

---

## 3. z_depth loss

`z_depth`는 스칼라 회귀이므로 과도하게 복잡하게 갈 필요 없음.

### 권장

[
L_z = \mathrm{Huber}(\hat z, z)
]

이유:

* z는 전처리 오차, 접촉 시작점 오차, 드리프트 영향이 있음
* MSE는 이상치에 민감함
* Huber가 더 안정적임

권장 파라미터:

* `delta = 0.05 ~ 0.1` mm 수준에서 시작
* 정규화 후 학습 시 `delta = 1.0` 표준값 사용 가능

대안:

* 데이터가 매우 깨끗하면 MSE도 가능
* 그러나 현재 상황에서는 **Huber 우선**이 맞음

---

## 4. fz loss

힘 데이터도 회귀이나, 센서-기구물-재질 영향으로 노이즈와 편차가 있으므로 z와 동일하게 가는 것이 적절함.

[
L_{fz} = \mathrm{Huber}(\hat f_z, f_z)
]

권장:

* `fz`는 표준화해서 학습
* 추론 시 역정규화

이유:

* 힘 값 범위가 xy heatmap loss보다 스케일이 커질 가능성이 높음
* 반드시 normalization 또는 loss weight 조정 필요

---

## 5. 최종 총 loss

가장 추천하는 총 loss는 아래 형태임.

[
L_{total} = \lambda_{xy} L_{xy} + \lambda_z L_z + \lambda_f L_{fz}
]

### 초기 권장값

가장 먼저 돌릴 값:

[
\lambda_{xy}=1.0,\quad
\lambda_z=0.2,\quad
\lambda_f=0.2
]

이유:

* 현재 주목적은 xy 안정화임
* z/fz는 auxiliary role이어야 함
* 처음부터 z/fz를 세게 주면 heatmap branch 학습이 약해질 수 있음

### 2차 튜닝 후보

* `1.0 / 0.1 / 0.1`
* `1.0 / 0.3 / 0.3`
* uncertainty weighting
* gradient norm balancing

하지만 첫 실험에서는 단순 고정 가중치가 맞음.

---

## 6. 실전 추천안

현재 상황 기준, 바로 구현할 수 있는 설계는 아래임.

## 설계안 v1

```text
Head A: xy_heatmap   -> BCEWithLogitsLoss(soft target)
Head B: z_depth      -> HuberLoss
Head C: fz           -> HuberLoss

L_total = 1.0 * L_xy + 0.2 * L_z + 0.2 * L_fz
```

### 장점

* 구현 난이도 낮음
* 기존 파이프라인과 충돌 적음
* option flag 기반 분기 쉬움
* A/B 테스트 명확함

### 권장 플래그

```bash
--use_depth_aware_label
--depth_label_kernel gaussian
--depth_radius_model hertz   # or geom
--loss_xy bce
--loss_z huber
--loss_fz huber
--lambda_xy 1.0
--lambda_z 0.2
--lambda_fz 0.2
```

---

## 7. 추가로 꼭 넣어야 할 것

## (1) heatmap decode 방식

heatmap을 쓸 것이면 최종 xy는 단순 argmax보다 아래가 더 좋음.

* `soft-argmax`
* 또는 local soft-argmax
* 또는 argmax + subpixel refinement

이유:

* 0.5 mm grid에서 quantization error가 바로 생김
* soft label의 장점을 살리려면 subpixel decode가 맞음

즉 학습은 heatmap, 출력은 continuous xy로 가야 함.

---

## (2) z/fz를 입력으로도 넣을지, 출력만 둘지

질문에서 “z/힘 채널 필요”라고 했으므로 해석은 두 가지임.

### A. 입력 채널로 사용

* 현재 시점의 `z_depth`, `fz` 또는 관련 proxy를 backbone input에 concat
* 장점: xy 추정이 depth condition을 직접 받음
* 단점: 추론 시 동일 정보가 항상 있어야 함

### B. 출력 head로만 사용

* backbone은 sensor signal만 보고
* z/fz는 auxiliary target으로만 학습
* 장점: 구조 단순, 추론 유연성 높음
* 단점: condition 정보 활용은 간접적임

### 권장

**1차 실험은 output head only**가 맞음.
그 다음에 성능이 더 필요하면 input conditioning을 추가하는 것이 순서상 맞음.

즉 처음엔:

```text
input  = sensor 16ch
output = xy_heatmap + z_depth + fz
```

이렇게 가는 것이 가장 안정적임.

---

## 8. 권장 ablation 순서

순서는 반드시 이렇게 가는 것이 좋음.

### Stage 1

기존 baseline

```text
point label + xy only
```

### Stage 2

핵심 변경만 반영

```text
depth-aware soft heatmap + xy only
```

### Stage 3

멀티태스크 추가

```text
depth-aware soft heatmap + z head + fz head
```

### Stage 4

필요 시 입력 conditioning

```text
sensor + depth/force condition input
```

이 순서여야 어떤 변경이 실제로 성능 향상 원인인지 분리 가능함. 체크리스트의 A/B 실험 방향과 정확히 맞음. 

---

## 9. 추천하지 않는 설계

아래는 현재 단계에서 비추천임.

### 1) 처음부터 loss를 너무 많이 섞는 것

* focal + dice + mse + coord loss 동시 사용
* 해석 불가 상태가 됨

### 2) xy를 direct regression과 heatmap regression으로 동시에 크게 묶는 것

* 아직 option B로 방향이 정해졌으므로 불필요하게 구조가 무거워짐

### 3) z와 fz를 하나의 scalar head로 합치는 것

* 물리적으로 의미가 다름
* depth와 force의 noise 특성도 다름
* 분리 head가 맞음

---

## 최종 제안

이번 단계의 **권장 로스 설계안**은 아래로 확정하는 것이 좋음.

```text
[출력]
1. xy_heatmap : [B,1,H,W]
2. z_depth    : [B,1]
3. fz         : [B,1]

[손실]
L_xy = BCEWithLogitsLoss(soft depth-aware target)
L_z  = HuberLoss
L_fz = HuberLoss

L_total = 1.0 * L_xy + 0.2 * L_z + 0.2 * L_fz
```

## 한 줄 결론

**지금은 “soft heatmap + z/fz 보조 회귀 헤드”가 가장 맞는 설계이며, direct coordinate regression을 다시 크게 건드릴 단계는 아님.**

원하면 다음 답변에서 바로 이어서
`모델 출력 스펙 표`와 `argparse/config 항목 표`까지 바로 정리하겠음.

