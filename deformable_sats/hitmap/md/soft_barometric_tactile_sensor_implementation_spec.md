# Soft Barometric Tactile Sensor (Iterative Pressure Reconstruction)
## Implementation Specification (Corrected & Refined)

---

# 1. Problem Definition

본 센서는 **압력 센서 배열로부터 접촉 위치 및 힘을 추정하는 inverse problem**을 해결하는 것이 목적임.

- 입력: discrete pressure measurements (barometer array)
- 출력: continuous pressure distribution + contact parameters + force

핵심 특징:
- Machine Learning 기반이 아닌 **model-based optimization 방식**
- 매 timestep마다 **online parameter estimation 수행**

---

# 2. Sensor Input / Output

## Input

- pressure measurements:

```
p_hat ∈ R^N  (N = number of sensors, e.g., 32)
```

- sensor positions:

```
(x_i, y_i), i = 1...N
```

---

## Primary Output (Direct Reconstruction Output)

1. Pressure distribution parameters

```
θ = [p0, σ, lx, ly, κ_curve, α, x0, y0]
```

2. Reconstructed pressure field

```
p(x, y, θ*)
```

3. Normal force

```
Fz = ∬ p(x, y, θ*) dx dy
```

---

## Derived Output (Post-processing)

- contact count (multi-contact detection)
- dominant contact location
- orientation (via α)
- residual-based secondary contact

※ orientation, contact count는 **직접 출력이 아니라 후처리 결과**임

---

# 3. Pressure Distribution Model

## 3.1 Base Model (Rectangular + Gaussian)

```
p(x, y) =
    p0                        if |x| < lx/2 and |y| < ly/2
    p0 * exp(-d^2 / (2σ^2))   otherwise
```

- d: rectangle boundary까지의 최소 거리

---

## 3.2 Curvature Extension

- curvature parameter:

```
κ_curve
```

- 의미:

| κ_curve | 의미 |
|--------|------|
| 0      | flat distribution |
| ↑      | curved surface |

---

## 3.3 Coordinate Transform

```
[x', y'] = R(α) * ([x, y] - [x0, y0])
```

- α: orientation
- (x0, y0): center

---

# 4. Optimization Problem

## Objective Function

```
Err(θ) = Σ ( p(x_i, y_i, θ) - p̂_i )^2
```

- least-squares formulation

---

## Solver

- Powell’s Dog-Leg method
- nonlinear least squares

---

## Initialization Strategy

### Case 1: previous contact 존재

```
θ_init ← θ_prev
```

### Case 2: contact 없음

- max pressure 위치 기반 초기화

---

# 5. Iterative Reconstruction (Multi-contact)

## Step 1: First fit

```
θ1* = argmin Err(θ)
```

---

## Step 2: Residual 계산

```
p_res,i = p(x_i, y_i, θ1*) - p̂_i
```

---

## Step 3: Residual 기반 재적합

- residual이 큰 경우 → 새로운 contact 존재

```
θ2* = argmin Σ (p_residual error)
```

---

## Step 4: 반복

- 최대 5회

---

## Step 5: Termination

- residual 충분히 작음
- iteration limit 도달

---

# 6. Force Estimation

## Numerical Integration

```
Fz = ∬ p(x, y, θ*) dx dy
```

- closed-form 없음
- grid sampling 사용

---

# 7. Real-Time Execution Loop

```
while True:
    read pressure

    initialize θ

    optimize θ

    compute residual

    if residual > threshold:
        repeat fitting

    compute force

    reset pressure
```

---

# 8. Critical Implementation Details

## 8.1 Reset Requirement (IMPORTANT)

- 각 contact 이벤트 이후:

```
p_hat ← 0
```

- 이유:
- 이전 접촉 영향 제거
- iterative fitting 안정성 확보

---

## 8.2 Initialization Importance

- 이전 timestep 결과 사용 필수
- convergence 속도 결정 요소

---

## 8.3 Real-time Constraint

- max iteration: 5
- target frequency: ~20 Hz

---

# 9. Heuristic Constraints (Implementation-Level)

※ 아래는 논문 명시 조건이 아닌 **실무 구현 권장 사항**임

- lx, ly 최소값 제한 (degenerate 방지)
- σ lower bound 설정
- p0 saturation 방지

---

# 10. Limitations

## 10.1 Model Assumption

- pressure distribution 가정 기반
- 실제 contact shape와 차이 존재

---

## 10.2 Multi-contact Separation

- 약 5 mm 이상에서만 안정적 분리 가능

---

## 10.3 Shear Force 미고려

- normal force 중심 모델

---

# 11. Implementation Strategy (Strong Recommendation)

## Step 1: Python Reference Implementation

- 목적: 수식 검증

---

## Step 2: Test Cases

1. single point
2. line contact
3. dual contact
4. force integration 검증

---

## Step 3: Optimization Stability 확인

- 초기값 민감도
- noise robustness

---

## Step 4: ROS2 / C++ Porting

- Python 검증 이후 진행

---

# 12. Summary

이 시스템은 다음으로 정의됨:

```
pressure sensing → inverse optimization → parameter estimation → force integration
```

핵심 포인트:

- Machine Learning 아님
- Online nonlinear optimization
- Residual 기반 multi-contact 분해
- Real-time iterative reconstruction

---

(END)

