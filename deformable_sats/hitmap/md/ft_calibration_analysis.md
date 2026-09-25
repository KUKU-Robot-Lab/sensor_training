# Vensor Calibration Analysis (Fx, Fy R² and Sign Issue)

## 1. User Experiment Description

**Experiment setup:**  
1. Z-axis: move down 3 mm then return.  
2. Z-axis: move down 2 mm, then move X(Y) axis by +2 mm → 0 → −2 mm → 0, then return Z to 0.  
3. Total data points: 24 012  
4. Condition number: 42.54  
5. Observation: data clusters near each axis; Fx, Fy R² are low (< 0.9 target).  

**P1 sensor mapping:**  

| Direction | Sensors | Expected Behavior |
|------------|----------|------------------|
| ← (+X) | S2, S6 | Compression (Δs < 0) |
| → (−X) | S1, S5 | Tension (Δs > 0) |
| ↓ (+Y) | S5, S6 | Compression |
| ↑ (−Y) | S1, S2 | Tension |

Matrix relation:

```
Fx = c11*s1 + c12*s2 + c13*s3 + c14*s4
Fy = c21*s1 + c22*s2 + c23*s3 + c24*s4
Fz = c31*s1 + c32*s2 + c33*s3 + c34*s4
```

All s are bias-corrected Δs values (compression → output ↓ → Δs < 0).  

Expectation:  
- Fx + → (s2,s6) − , (s1,s5) + ⇒ (c12,c14) same sign; (c11,c13) same sign.  
- Fy follows similar pattern.  
- Fz → same sign for all.  

However, `cal_p1_2.csv` does **not** show this.  

---

## 2. Diagnostic Summary

| Issue | Description |
|--------|--------------|
| **Fz dominates** | Fx/Fy variation small → R²(Fx,Fy)≈0.35 – 0.4 ; R²(Fz)≈0.91. |
| **High collinearity** | s1–s2–s5 correlation ≈ 0.91–0.97 → OLS unstable, sign flipping likely. |
| **Non‑orthogonal stimuli** | X/Y motion coupled with Z‑load; shear + normal mixed. |
| **Model mismatch** | Bias removal, sign convention, and physical mapping may differ from code. |

---

## 3. Recommended Remedies

### 3.1 Stimulus Orthogonalization
- Keep Fz constant via feedback while sweeping ±X and ±Y.  
- Separate pure Fz loading sessions.  
- Re‑tare each block.  

### 3.2 Feature Transformation (Reduce Collinearity)

Define orthogonal features:  
```
u = s1 + s2 + s5 + s6        # Fz
v = (s2 + s6) - (s1 + s5)    # Fx
w = (s5 + s6) - (s1 + s2)    # Fy
```

Fit 3×3 model using [u,v,w]. Remove interaction terms unless justified.

### 3.3 Regularization
Apply **ridge (Tikhonov)** regression to reduce condition number < 15.  
α ≈ 1e‑3 – 1e‑1 typical.

### 3.4 Signal Pre‑processing
- Re‑bias per trial.  
- High‑pass or moving‑average detrend to remove drift.  

### 3.5 Expand Shear Range
Increase Fx/Fy actuation amplitude to raise SNR. Slightly deeper indentation helps multi‑taxel activation and accurate shear estimation.

---

## 4. Coefficient‑Sign Discussion

Example coefficients from `cal_p1_2.csv` (showing inconsistent signs):  
Fx row [− + + −], Fy [− + + −], Fz [− − − +].  

### Possible Causes
1. **Collinearity sign flipping** → regularize.  
2. **Δs definition mismatch** → ensure Δs = s − s₀ if compression → decrease.  
3. **Sensor mapping error** → re‑verify s1‑s6 channel order.  
4. **Fz leakage** → decouple via orthogonal motion or feedback control.

---

## 5. Re‑fitting Workflow

1. Re‑zero and z‑score Δs per trial.  
2. Transform to [u,v,w].  
3. Fit ridge regression (α grid).  
4. Select α via CV minimizing validation error.  
5. If physical sign constraints needed, use constrained least‑squares enforcing expected sign pattern.  
6. Acquire new data with orthogonal, balanced design; target Fx,Fy R² > 0.9.

---

## 6. Key Points

- Current low Fx,Fy R² due to *non‑orthogonal stimuli + collinearity*.  
- Use sum‑difference basis, ridge regression, orthogonal design, and stronger shear excitation.  
- Sign inconsistency stems from collinearity, preprocessing, mapping, or Fz mixing—solvable via above pipeline.

---

### References
- 2022 *Guiding the Design of Super‑Resolution Tactile Skins with Taxel Value Isolines Theory* — taxel overlap improves sensitivity and accuracy.
