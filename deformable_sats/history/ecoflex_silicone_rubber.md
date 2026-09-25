# Mechanical responses of Ecoflex silicone rubber: Compressible and incompressible behaviors

**Authors:** D. Steck¹, J. Qu¹, S. B. Kordmahale², D. Tscharnuter³, A. Muliana¹, J. Kameoka²

¹ Department of Mechanical Engineering, Texas A&M University, College Station, Texas 77853
² Department of Electrical Engineering, Texas A&M University, College Station, Texas 77853
³ Polymer Competence Center Leoben GmbH, Roseggerstrasse 12, 8700, Leoben, Austria

**Correspondence:** A. Muliana (amuliana@tamu.edu)

*J. Appl. Polym. Sci.* 2019, 136, 47025. DOI: 10.1002/app.47025
Received 14 December 2017; accepted 8 July 2018

---

## Abstract

Silicone rubbers are widely used in products ranging from cooking utensils and electronics to medical devices and implants, and have recently drawn interest among soft robotics researchers because they can be easily formed into various shapes and actuated quickly and easily. This article examines the nonlinear elastic response of the silicone rubber Ecoflex under both compressible and incompressible constraints. A uniaxial tension test indicates slight compressibility that increases with stretching. Five constitutive material models are considered to describe the nonlinear elastic responses of Ecoflex under both compressible and incompressible conditions. Finite element (FE) analysis is presented to analyze the multiaxial response of structures or devices made of Ecoflex under complex boundary conditions. The study highlights variations in the multiaxial response of structures at large deformations across different constitutive models and compressible/incompressible constraints. High-precision control in soft robotics applications requires understanding the multiaxial response of silicone rubbers, especially under large deformations.

---

## Introduction

Silicone rubbers are becoming the materials of choice for lightweight compliant systems requiring relatively fast and small actuation. Their soft characteristics make them appealing for soft robots and artificial limbs. Rubber pneumatic actuators are gaining popularity due to their simple actuation mechanisms: high power-to-weight ratios are attainable, relatively small input air pressures induce large deformations, and they can be formed into various shapes at low cost. Several pneumatic rubber actuators have been developed, such as the McKibben artificial muscle, bending actuators, biomimetic robots that mimic animal locomotion, and biomedical devices.

Despite the simple actuation mechanism, analyzing and designing silicone rubber pneumatic actuators is challenging due to the highly nonlinear responses of silicone rubbers. For analytical solutions, a simple constitutive model such as the neo-Hookean model is often used, but it is limited in capturing the response under large stretch. Finite element (FE) analyses are widely used to analyze pneumatic silicone rubber actuators, typically modeling the response as nonlinear elastic and incompressible using well-established models (neo-Hookean, Ogden, Mooney–Rivlin, and Yeoh). FE results generally agree with experiments at low-to-moderate stretch, but mismatches become more pronounced at higher pressures associated with large stretch.

Models for nonlinear elastic response generally involve several material constants, which can lead to nonuniqueness: numerous parameter sets fit experimental data over a wide range of stretches. Ogden et al. indicated that parameters calibrated from uniaxial response do not necessarily predict the biaxial response at larger stretch, and vice versa. In some models (e.g., Gent) mismatches are small, but in others (e.g., the six-parameter Ogden model) large mismatches appear. Simultaneously fitting uniaxial and biaxial data gives good overall fits. In many FE simulations the material parameters are obtained mainly from uniaxial data, yet pressurized actuators experience multiaxial deformations, so the effect of nonuniqueness may become more pronounced.

Silicone rubbers are often treated as isotropic, nonlinear elastic, incompressible or nearly incompressible. Typical experiments stretch specimens uniaxially without measuring lateral stretch, leading to an incompressible assumption. As shown later, volumetric changes under uniaxial stretching increase with stretch ratio, although the values remain small (determinant of the deformation gradient less than 1.1). The effect of this slightly compressible behavior on the overall multiaxial deformation of pressurized vessels has not been fully explored.

In this article, the compressibility and incompressibility constraints are examined in predicting the overall mechanical response of silicone rubbers. Experimental tests on Ecoflex specimens under uniaxial stretching (measuring axial and lateral response) are presented first. Different compressible and incompressible material models are then considered, followed by boundary value problems (a pressurized sphere and a pneumatic actuator device) and FE analyses.

---

## Ecoflex Specimen Fabrication

Uniaxial tensile tests were conducted on dog-bone specimens. A dog-bone mold was directly printed using a stereolithographic micro 3D printer. The mixed Ecoflex 00-30 solution was dispensed into the mold, then degassed in a desiccator for 20 min. A piece of polystyrene Petri dish film was placed on top of the dispensed Ecoflex to eliminate surface tension and produce a smooth, uniform-thickness surface. Curing time was 4 h at room temperature. Two batches of specimens with thicknesses of 2 mm and 3 mm were produced.

> **[Figure 1]** Production process for the Ecoflex dog-bone specimen. (a) Schematic of the dog-bone mold (3D and cross-sectional view). (b) Dispensing Ecoflex solution into the mold. (c) Curing with a thin plastic film placed on top to eliminate surface tension. (d) Specimen extracted from the mold. (e) Photograph of the produced specimen.

---

## Experimental Tests

Uniaxial tensile tests were conducted using an Instron 5500 electromechanical testing machine with a 10 N load cell. Axial and transverse strains up to a maximum axial stretch of approximately 5 were measured using the Mercury RT digital image correlation system (Sobriety s.r.o., Czech Republic) with two Prosilica GT6600 29-megapixel cameras. ISO 527-3 type 5 specimens were used. At an axial stretch ratio of 5, specimen width reduced from 6 mm to less than 3 mm. The cameras were set so the field of view covered twice the gauge length, balancing transverse-stretch resolution with axial-stretch field of view.

When specimens were stretched to 5× their initial length, significant hysteretic areas appeared upon unloading. Since only nonlinear elastic response is considered, tests were limited to a stretch level around 4, where specimens fully recover with negligible hysteresis.

Specimens were stretched at a constant rate of 1.2/min. Nine specimens from two batches with different thicknesses were tested under several loading–unloading cycles, showing full recovery upon unloading and good repeatability. The Cauchy stress measure is used to present the experimental data. Volume changes were examined by computing the determinant of the deformation gradient ($\det \mathbf{F}$). Ecoflex exhibits slight compressibility that increases with stretching. Based on these results, Ecoflex is treated as a nonlinear elastic material, and both compressible and incompressible conditions are assessed.

> **[Figure 2]** Experimental results from uniaxial stretching at a stretch rate of 1.2/min: (a) axial response, (b) lateral response, (c) volume changes ($\det \mathbf{F}$).

> **[Figure 3]** Loading–unloading cycles at a constant stretch rate (input stretch and output stress vs. time).

---

## Constitutive Models

A recently developed constitutive model is first considered to describe the mechanical responses of Ecoflex. The new model expresses the left Cauchy–Green deformation tensor ($\mathbf{B}$) in terms of the Cauchy stress ($\mathbf{T}$), whereas classical models express stress in terms of kinematical quantities (strain or stretch). The new model is described by four material constants for a compressible isotropic elastic body; under isochoric motion it is fully characterized by just two constants.

Let $\mathbf{x}$ denote the current position of a particle that is at $\mathbf{X}$ in a stress-free reference configuration. Let $\mathbf{x} = \boldsymbol{\chi}_{\kappa_R}(\mathbf{X}, t)$ be the motion of a particle at current time $t$ in configuration $\kappa_R$, with the deformation gradient $\mathbf{F}$ defined as

$$
\mathbf{F} = \frac{\partial \boldsymbol{\chi}_{\kappa_R}}{\partial \mathbf{X}}
\tag{4.1}
$$

The left and right Cauchy–Green deformation tensors $\mathbf{B}$ and $\mathbf{C}$ are defined through

$$
\mathbf{B} := \mathbf{F}\mathbf{F}^{T}, \qquad \mathbf{C} := \mathbf{F}^{T}\mathbf{F}
\tag{4.2}
$$

The new nonlinear elastic model for isotropic materials is given as

$$
\mathbf{B} = \mathbf{g}_a(\mathbf{T}) = \mathbf{I} - \kappa\left(1 - e^{-\alpha\sqrt{I_2}}\right)\mathbf{I} + \mu\,\frac{1 - e^{-\delta\sqrt{I_2}}}{\sqrt{I_2}}\,\mathbf{T}
\tag{4.3}
$$

where $\alpha$, $\delta$, $\kappa$, and $\mu$ are constants, and $I_2 = \mathrm{tr}(\mathbf{T}^2)$. When $\mathbf{T} = \mathbf{0}$, $\mathbf{B} = \mathbf{I}$. Upon linearization with small displacement gradients, the model reduces to the classical linearized elastic model. (Mansouri and Darijani also used an exponential strain-energy function of the first and second invariants of the Cauchy stretch tensor, capturing multiaxial nonlinear elastic behavior of soft materials with relatively few parameters.)

The uniaxial tensile test is defined by the deformation gradient

$$
\mathbf{F} = \begin{bmatrix} \Lambda_1 & 0 & 0 \\ 0 & \Lambda_2 & 0 \\ 0 & 0 & \Lambda_2 \end{bmatrix}
$$

where $\Lambda_1$ and $\Lambda_2$ are the stretches in the axial ($x_1$) and transverse ($x_2$) directions; isotropy implies the $x_3$ response equals the $x_2$ response. The only nonzero Cauchy stress component is $T_{11} = P_{11}/\Lambda_2^{2}$, where the first Piola–Kirchhoff stress in the axial direction is $P_{11} = F/A_0$ ($F$ = axial force, $A_0$ = initial cross-sectional area). Since $T_{11}$ is the only nonzero component, it is denoted $T$. The deformation tensor components in eq. (4.3) become

$$
\begin{aligned}
B_{11} = \Lambda_1^{2} &= \left[1 - \kappa\left(1 - e^{-\alpha T}\right)\right] + \mu\,\frac{1 - e^{-\delta T}}{T}\,T = \left[1 - \kappa\left(1 - e^{-\alpha T}\right)\right] + \mu\left(1 - e^{-\delta T}\right) \\
B_{22} = \Lambda_2^{2} = \Lambda_3^{2} &= 1 - \kappa\left(1 - e^{-\alpha T}\right)
\end{aligned}
\tag{4.4}
$$

Using the lateral response [Figure 2(b)], the two constants $\kappa$ and $\alpha$ are calibrated first; then, using the axial response [Figure 2(a)], $\mu$ and $\delta$ are obtained.

Imposing isochoric motion ($\det \mathbf{F} = 1$) gives $\Lambda_2 = \Lambda_3 = 1/\sqrt{\Lambda_1}$, and eq. (4.4) reduces to

$$
\Lambda_1^{2} - \frac{1}{\Lambda_1} = \mu\,\frac{1 - e^{-\delta T}}{T}\,T = \mu\left(1 - e^{-\delta T}\right)
\tag{4.5}
$$

so $\mu$ and $\delta$ are determined directly from the axial response.

### Ogden model

The Ogden stored energy in terms of principal stretches for a compressible material is

$$
W_O = \sum_{i=1}^{N}\frac{\mu_i}{\alpha_i}\left[\Lambda_1^{\alpha_i} + \Lambda_2^{\alpha_i} + \Lambda_3^{\alpha_i} - 3\right] + W_{\mathrm{vol}}
\tag{4.6}
$$

$$
W_{\mathrm{vol}} = \lambda\left(\frac{1}{2}J^{2} - J + \frac{1}{2}\right) - \sum_{i=1}^{N}\mu_i \ln J
\tag{4.7}
$$

where $J = \det \mathbf{F}$, and $\mu_i$, $\alpha_i$, $\lambda$ are material constants. The axial stress component is

$$
T_{kk} = \frac{\Lambda_k}{J}\frac{\partial W}{\partial \Lambda_k} + \frac{\partial W}{\partial J}; \qquad k = 1, 2, 3
\tag{4.8}
$$

For the uniaxial tensile test, the axial stresses in the compressible Ogden model are

$$
\begin{aligned}
T_{11} &= \sum_{i=1}^{N}\frac{\mu_i}{J}\left(\Lambda_1^{\alpha_i} - J^{\alpha_i/2}\Lambda_1^{-\alpha_i/2}\right) \\
T_{22} = T_{33} &= \sum_{i=1}^{N}\frac{\mu_i}{J}\left(J^{\alpha_i/2}\Lambda_1^{-\alpha_i/2} - 1\right) + \lambda(J - 1) = 0
\end{aligned}
\tag{4.9}
$$

When an incompressible condition is imposed ($J = 1$), the axial stress reduces to

$$
T_O = T_{O11} = \sum_{i=1}^{N}\mu_i\left(\Lambda_1^{\alpha_i} - \Lambda_1^{-\alpha_i/2}\right)
\tag{4.10}
$$

### Gent model

The Gent stored energy in terms of stretch for a compressible material is

$$
W_{GT} = -\frac{\mu}{2}J_m \ln\!\left(1 - \frac{J_1 - 3}{J_m}\right) + W_{\mathrm{vol}}
\tag{4.11}
$$

$$
W_{\mathrm{vol}} = c_1 \ln J + c_2 (\ln J)^{2} + c_3 (J^{2} - 1)
\tag{4.12}
$$

where $J_1$ is the first invariant of the stretch tensor, and $\mu$, $c_1$, $c_2$, $c_3$, $J_m$ are material parameters. The stress is obtained from

$$
\mathbf{T} = \frac{2}{J}\frac{\partial W}{\partial J_1}\mathbf{B} + 2J\frac{\partial W}{\partial J}\mathbf{I}
\tag{4.13}
$$

For uniaxial loading, where $J_1 = \Lambda_1^{2} + 2\Lambda_2^{2}$, $J = \Lambda_1 \Lambda_2^{2}$, and $T_{22} = T_{33} = 0$, the axial stress is

$$
T_{11} = \frac{\mu}{J}\,\frac{J_m}{J_m - J_1 + 3}\left(\Lambda_1^{2} - \frac{J}{\Lambda_1}\right)
\tag{4.14}
$$

When an incompressible condition is considered ($J_1 = \Lambda_1^{2} + 2/\Lambda_1$), eq. (4.14) reduces to

$$
T_{GT} = T_{GT11} = \frac{\mu J_m}{J_m - J_1 + 3}\left(\Lambda_1^{2} - \frac{1}{\Lambda_1}\right)
\tag{4.15}
$$

### Parameter identification (compressible vs. incompressible)

For incompressible conditions, the parameters in the Ogden and Gent models [eqs. (4.10) and (4.15)] are easily determined by fitting uniaxial data. For compressible conditions, parameter identification is tedious because the axial and lateral data must be fit concurrently:

$$
\begin{aligned}
T_{11} &= f_1(\Lambda_1, \Lambda_2) \\
T_{22} &= f_2(\Lambda_1, \Lambda_2) = 0
\end{aligned}
\tag{4.16}
$$

In the new model [eq. (4.4)], where stretch is expressed in terms of stresses, parameter identification for compressible bodies is straightforward.

### Yeoh and Mooney–Rivlin models

Two additional commonly used models are also considered. Their stored-energy functions are

$$
W = \sum_{i=1}^{3}C_i\left(\bar{I}_1 - 3\right)^{i} + \sum_{i=1}^{3}D_i\left(J - 1\right)^{2i}
\tag{4.17}
$$

$$
W = C_1\left(\bar{I}_1 - 3\right) + C_2\left(\bar{I}_2 - 3\right) + D\left(J - 1\right)^{2}
\tag{4.18}
$$

where $\bar{I}_1 = J^{-2/3}I_1$, and $C_i$, $D_i$ are material constants.

> **참고(번역자 노트):** 원문 PDF에서 식 (4.17)의 우변 두 번째 항이 OCR 손상으로 일부 불명확합니다($\sum_{k=1}^{3}D_i(J-3)^{2k}$로 표기됨). Yeoh 모델의 표준 체적항(volumetric term) 형태에 맞춰 위와 같이 복원했습니다. 원문 확인 후 수정 필요.

### Results

Figure 4 shows the uniaxial response of Ecoflex assumed incompressible, with several constitutive models; calibrated parameters are in **Table I**. Figure 5 treats Ecoflex as compressible (volumetric strain energy from eqs. (4.7) and (4.12) for Ogden and Gent), showing axial and transverse responses; calibrated parameters are in **Table II**. All models capture both axial and lateral responses except Mooney–Rivlin, which shows relatively large deviations. For the hyperelastic models the lateral stress is not exactly zero (Figure 6), unlike in the new model, though the values are negligible compared with the axial stress.

Figure 7 presents absolute percent errors for both incompressible and compressible assumptions. The present model gives relatively small errors, especially for the compressible condition; Mooney–Rivlin errors are large; other models show higher errors at early loading.

In summary, all hyperelastic models capture the nonlinear elastic response of Ecoflex under compressible and incompressible conditions. However, calibrating parameters under the compressible condition is challenging because the models involve more parameters and the axial and lateral conditions must be satisfied concurrently — so the traction-free lateral condition cannot be exactly satisfied (Figure 6). The new model has advantages: (1) relatively few material parameters, (2) no need to impose isochoric motion during calibration, and (3) direct satisfaction of boundary conditions.

### Finite element analysis of uniaxial response

The nonlinear elastic response of Ecoflex under uniaxial stretching was investigated with ABAQUS FE analyses using 3D continuum elements (C3D20RH). The Ogden constitutive model for both compressible and incompressible cases was used. The stored energy in terms of principal stretches is

$$
W_O = \sum_{i=1}^{N}\frac{\mu_i}{\alpha_i}\left[\hat{\Lambda}_1^{\alpha_i} + \hat{\Lambda}_2^{\alpha_i} + \hat{\Lambda}_3^{\alpha_i} - 3\right] + \sum_{j=1}^{M}\frac{1}{D_j}\left(J - 1\right)^{2j}
$$

where $J = \Lambda_1\Lambda_2\Lambda_3$ is the Jacobian, $\hat{\Lambda}_k = J^{-1/3}\Lambda_k$ ($k = 1, 2, 3$), and $\mu_i$, $\alpha_i$, $D_j$ are material parameters. After a convergence study, 5040 elements were used. Figure 8 compares the FE and analytical uniaxial responses for compressible and incompressible cases; parameters are in **Table III**. For the compressible case, responses using $D_1$ and $D_2$ versus only $D_1$ are presented. With only $D_1$, the transverse response is better captured but the axial response deviates; with $D_1$ and $D_2$, the axial response improves while the transverse response is slightly underpredicted. Overall, FE and analytical solutions for both conditions capture the experimental data.

> **[Figure 4]** Uniaxial response of Ecoflex treated as an incompressible material (axial stress vs. $\Lambda^2 - 1/\Lambda$).

> **[Figure 5]** Uniaxial response of Ecoflex treated as a compressible material (axial response vs. axial stretch; transverse response vs. transverse stretch).

> **[Figure 6]** Lateral stress during uniaxial stretching from the hyperelastic models (Ecoflex treated as compressible) — Ogden vs. Gent.

> **[Figure 7]** Absolute percent errors of several models vs. experimental data, under incompressible and compressible conditions.

> **[Figure 8]** Uniaxial response of Ecoflex: FE simulation vs. analytical solution (Ogden model), compressible and incompressible cases; Cauchy stress and transverse stretch vs. axial stretch.

#### Table I. Material parameters for Ecoflex as an incompressible material

| Present model | Ogden (N = 3) | Gent | Yeoh | Mooney–Rivlin |
|---|---|---|---|---|
| $\mu = 18$ | $\mu_1 = 22$ kPa | $J_m = 27$ | $C_1 = 17$ kPa | $C_1 = 48$ kPa |
| $\delta = 0.004$ kPa⁻¹ | $\alpha_1 = 1.3$ | $\mu = 17$ kPa | $C_2 = -0.2$ kPa | $C_2 = -152$ kPa |
| | $\mu_2 = 0.4$ kPa | | $C_3 = 0.023$ kPa | |
| | $\alpha_2 = 5$ | | | |
| | $\mu_3 = -2$ kPa | | | |
| | $\alpha_3 = -2$ | | | |

#### Table II. Material parameters for Ecoflex as a compressible material

| Present model | Ogden (N = 3) | Gent | Yeoh | Mooney–Rivlin |
|---|---|---|---|---|
| $\kappa = 0.715$ | $\mu_1 = 22$ kPa | $J_m = 27$ | $C_1 = 17$ kPa | $C_1 = 48$ kPa |
| $\alpha = 0.016$ kPa⁻¹ | $\alpha_1 = 1.3$ | $\mu = 17$ kPa | $C_2 = -0.2$ kPa | $C_2 = -152$ kPa |
| $\mu = 20$ | $\mu_2 = 0.4$ kPa | $c_1 = 50$ | $C_3 = 0.023$ kPa | $D = 3$ |
| $\delta = 0.003$ kPa⁻¹ | $\alpha_2 = 5$ | $c_2 = 20$ | $D_1 = 15$ | |
| | $\mu_3 = -2$ kPa | $c_3 = -25$ | $D_2 = 20$ | |
| | $\alpha_3 = -2$ | | $D_3 = 10$ | |
| | $\lambda = 375$ | | | |

> **참고:** 원문 Table II의 Yeoh / Mooney–Rivlin 열은 PDF 레이아웃 손상으로 항목 배치가 다소 뒤섞여 있어, 모델 구조(Yeoh: $C_1$–$C_3$ + 체적항 $D_1$–$D_3$, Mooney–Rivlin: $C_1$, $C_2$, $D$)에 맞춰 재배치했습니다.

#### Table III. Material parameters for Ecoflex used in FE simulations

| $i$ | $\mu_i$ (kPa) | $\alpha_i$ | $D_i$ |
|---|---|---|---|
| 1 | 16.9 | 1.3 | 1.156 |
| 2 | 0.08 | 5.0 | 0.0001 |
| 3 | 1.0 | −2.0 | – |

---

## Boundary Value Problems

Boundary value problems are presented to examine the multiaxial response of Ecoflex under compressible and incompressible conditions.

### Inflated thin-walled sphere

An inflated thin-walled sphere with initial radius $R$ and thickness $H$ ($R \gg H$) is studied; the wall is modeled as a membrane, allowing analytical solutions. Under lateral pressure $p$, the in-plane principal stretches are $\Lambda_1 = \Lambda_2 = r/R$ and the out-of-plane component is $\Lambda_3 = h/H$ ($r$, $h$ = inflated radius and thickness). The principal stresses are $\sigma_1 = \sigma_2 = \sigma$, $\sigma_3 \approx 0$. The in-plane stress relates to the pressure by

$$
\sigma = \frac{p}{2}\frac{\Lambda_1}{\Lambda_3}\frac{R}{H}
$$

**New model (isochoric, $\Lambda_3 = 1/\Lambda_1^{2}$):**

$$
\Lambda_1^{2} = \frac{1}{\Lambda_1^{4}} + \frac{1}{\sqrt{2}}\mu\left(1 - e^{-\frac{\sqrt{2}}{2}\delta p \Lambda_1^{3}\frac{R}{H}}\right)
\tag{5.1}
$$

**Ogden model (incompressible):**

$$
p_O = 2\frac{H}{R}\frac{1}{\Lambda_1^{3}}\sum_{i=1}^{N}\mu_i\left(\Lambda_1^{\alpha_i} - \Lambda_1^{-2\alpha_i}\right)
\tag{5.3}
$$

**Gent model (incompressible):**

$$
p_{GT} = 2\frac{H}{R}\frac{1}{\Lambda_1^{3}}\frac{\mu J_m}{J_m - I_1 + 3}\left(\Lambda_1^{2} - \frac{1}{\Lambda_1^{4}}\right); \qquad I_1 = 2\Lambda_1^{2} + \frac{1}{\Lambda_1^{4}}
\tag{5.4}
$$

For the incompressible case (R/H = 10), all models show a similar trend: pressure first increases with in-plane stretch, then changes slightly with continued stretching, then increases again with further stretching — indicating the expected limiting-stretch behavior.

**Compressible case — new model** (solve eqs. (5.6a) and (5.6b) simultaneously):

$$
\Lambda_1^{2} = \Lambda_2^{2} = 1 - \kappa\left(1 - e^{-\frac{\sqrt{2}}{2}\alpha p \frac{\Lambda_1}{\Lambda_3}\frac{R}{H}}\right) + \frac{1}{\sqrt{2}}\mu\left(1 - e^{-\frac{\sqrt{2}}{2}\delta p \frac{\Lambda_1}{\Lambda_3}\frac{R}{H}}\right)
\tag{5.6a}
$$

$$
\Lambda_3^{2} = 1 - \kappa\left(1 - e^{-\frac{\sqrt{2}}{2}\alpha p \frac{\Lambda_1}{\Lambda_3}\frac{R}{H}}\right)
\tag{5.6b}
$$

**Compressible case — Ogden model:**

$$
p_O = 2\frac{H}{R}\frac{1}{\Lambda_1^{3}}\sum_{i=1}^{N}\mu_i\left(\Lambda_1^{\alpha_i} - J^{\alpha_i}\Lambda_1^{-2\alpha_i}\right)
\tag{5.7a}
$$

$$
0 = \frac{1}{J}\sum_{i=1}^{N}\mu_i\left(\frac{J^{\alpha_i}}{\Lambda_1^{2\alpha_i}} - 1\right) + \lambda(J - 1)
\tag{5.7b}
$$

**Compressible case — Gent model:**

$$
p_{GT} = 2\frac{H}{R}\frac{1}{\Lambda_1^{3}}\frac{\mu J_m}{J_m - I_1 + 3}\left(\Lambda_1^{2} - \frac{J^{2}}{\Lambda_1^{4}}\right); \qquad I_1 = 2\Lambda_1^{2} + \frac{J^{2}}{\Lambda_1^{4}}
\tag{5.8a}
$$

$$
0 = \frac{1}{J}\frac{\mu J_m}{J_m - I_1 + 3}\frac{J^{2}}{\Lambda_1^{4}} + 2c_1 - \mu + 4c_2 \ln J + 4c_3 J^{2}
\tag{5.8b}
$$

For both Ogden and Gent compressible models, two equations are solved simultaneously. Equations (5.6)–(5.8) are solved numerically.

When compressible behavior is considered, the responses from the Ogden and new models differ from the incompressible assumption (Figure 9), even though all models capture the uniaxial response (Figures 4 and 5) because the tested Ecoflex shows only mild compressibility. For the multiaxial response, the Ogden and Gent models show a pressure drop as membrane stretch increases (Gent then increases rapidly with further stretch), while the new model's pressure increases monotonically. Examining the lateral (through-thickness) stretch (Figure 10): the new model gives a limiting lateral stretch near 0.5, whereas the Ogden and Gent models show a strong decrease (below 0.1 when in-plane stretch is around 3.5), similar to the incompressible condition. In the new model, a limiting in-plane stretch is accompanied by a limiting lateral stretch, as in the uniaxial experiment.

FE simulations of the pressurized sphere (Ogden model) considered compressible and incompressible conditions. Only a fraction of the sphere was modeled with symmetric boundary conditions; inner radius 10 mm, thickness 1 mm. After a convergence study, 1960 C3D20RH elements were used. At moderate-to-large stretch, some deviation between analytical and FE results appears (Figure 12). Stress contours (Figure 13) are similar at low stretch but differ significantly between compressible and incompressible cases at large stretch, since compressible behavior is more pronounced at large stretch. Shell elements (S4R, 168 elements) showed better agreement with the analytical solution (Figure 14). Across the three models, responses vary significantly at moderate-to-large multiaxial stretch even though uniaxial responses are similar — multiaxial experiments are needed to guide constitutive model choice.

> **[Figure 9]** Pressurized sphere response comparing compressible and incompressible material behaviors (inner pressure vs. circumferential stretch) for Ogden, Gent, and the present model.

> **[Figure 10]** Radial (through-thickness) stretch of the pressurized sphere using the compressible material models (Ogden, present, Gent).

> **[Figure 11]** Portion of a hollow sphere used in the FE simulation.

> **[Figure 12]** Sphere inflation response using the Ogden model (analytical and FE with 3D continuum elements), with deformed shapes.

> **[Figure 13]** Stress contours from the FE simulation (radial $\sigma_3$ and circumferential $\sigma_1, \sigma_2$), compressible vs. incompressible, plotted on the undeformed configuration at circumferential stretches of 1.49 and 4.0.

> **[Figure 14]** Sphere inflation response using the Ogden model (analytical and FE with shell elements).

### Pneumatic actuator soft micromold (PASMO) device

A more complex boundary condition is analyzed by FE only: a PASMO device used for releasing drugs as collagen microparticles. The device is a cylindrical body with three layers: a thin (1 mm) stiff bottom layer to prevent bottom expansion; a 7 mm second layer of Ecoflex containing a circular microchannel (4 mm height) where air pressure is applied; and a 1 mm Ecoflex top layer with wells (1 mm diameter, 0.3 mm depth). By symmetry, a quarter model was analyzed using 165,938 ten-node tetrahedral elements (C3D10H). The Ogden model for both compressible and incompressible conditions was used.

The area expansion of one well at different pressures (Figure 16) shows a similar trend for both behaviors, but the magnitude varies, indicating slight response changes with material behavior. Logarithmic circumferential strain contours and deformed shapes (Figure 17) show changes in the deformed shape of the PASMO device under slightly different material behaviors.

> **[Figure 15]** PASMO device for drug release (top view, side view, and quarter model used in FE analysis).

> **[Figure 16]** PASMO response under internal pressure: area of well vs. internal pressure (compressible $D_1$; compressible $D_1, D_2$; incompressible).

> **[Figure 17]** Logarithmic strain contours along the circumferential direction at 22 kPa internal pressure (incompressible; compressible $D_1$; compressible $D_1, D_2$).

---

## Conclusions

A nonlinear elastic response of Ecoflex silicone rubber under several loading conditions was studied. Uniaxial loading–unloading experiments on dog-bone specimens showed limiting stretch behavior in both axial and lateral directions and slight compressibility; up to an axial stretch ratio of 4, the response is elastic with negligible hysteresis, and compressibility increases with stretch ratio. While most studies treat Ecoflex as incompressible, this work examines the consequences of compressibility on the multiaxial response of devices made of Ecoflex. Five isotropic constitutive models were considered with compressible and incompressible constraints, all capable of capturing the uniaxial response; FE analysis based on the Ogden model was also performed.

Analytical and numerical solutions to boundary value problems revealed that models with similar uniaxial behavior give significantly different multiaxial responses, especially at large stretch, and these variations also appear in FE analyses. Different compressible/incompressible responses are seen at large stretch, as expected. Multiaxial experiments are necessary to better describe material response and characterize parameters. When modeling and simulation guide the design of Ecoflex structures or devices, proper material behaviors must be considered, especially when large deformations are expected.

---

## Acknowledgment

This research was sponsored by the Air Force Office of Scientific Research (AFOSR) under grant FA9550-14-1-0234.

---

## References

1. Tondu, B. *J. Intell. Material Syst. Struct.* **2012**, 23, 225.
2. Kordmahale, S.; Kameoka, J. *Ann. Mater. Sci. Eng.* **2015**, 2, 1021.
3. Shapiro, Y.; Wolf, A.; Gabor, K. *Sens. Actuators.* **2011**, 167, 484.
4. Sun, Y.; Song, Y. S.; Pail, K. In *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, **2013**, p 4446.
5. Suzumori, K.; Endo, S.; Kanda, T.; Kato, N.; Suzuki, H. In *Proceedings of IEEE International Conference on Robotics and Automation*, **2007**, p 4975.
6. Endo, S.; Suzumori, K.; Kanda, T.; Kato, N.; Suzuki, H.; Ando, Y. "Flexible and Functional Pectoral Fin Actuator for Underwater Robots," 3rd International Symposium on Aero Aqua Bio-mechanisms ISABMEC, **2006**, S42, p 55.
7. Konishi, S.; Kawai, F.; Cusin, P. *Sens. Actuators.* **2001**, 89, 28.
8. Suzumori, K. *Robotics Auton. Syst.* **1996**, 18, 135.
9. Suzumori, K.; Kondo, F.; Tanaka, H. *J. Robotics Mechatron.* **1993**, 5, 537.
10. Huang, P. J.; Chou, C. K.; Chen, C. T.; Yamaguchi, H.; Qu, J.; Muliana, A.; Hung, M. C.; Kameoka, J. *Soft Robot.* **2017a**, 4, 390–399.
11. Huang, P. J.; Qu, J.; Saha, P.; Muliana, A.; Kameoka, J. "Artificial islet: Microencapsulation for Beta cells in collagen microparticles via circular pneumatically actuated soft micro-mold (cPASMO) device," **2017b** (under review).
12. Noritsugu, T.; Tanaka, T. *IEEE ASME Trans. Mechatron.* **1997**, 2, 259.
13. Suzumori, K.; Hama, T.; Kanda, T. In *Proceedings of IEEE International Conference on Robotics and Automation*, **2006**, p 1824.
14. Wakimoto, S.; Ogura, K.; Suzumori, K.; Nishioka, Y. In *Proceedings of IEEE International Conference on Robotics and Automation*, **2009**, p 556.
15. Wakimoto, S.; Suzumori, K.; Ogura, K. *Adv. Robotics.* **2011**, 25, 1311.
16. Zhang, J.; Wang, H.; Tang, J.; Guo, H.; Hong, J. In *Proceedings of IEEE International Conference on Information and Automation*, **2015**, p 2460.
17. Polygerinos, P.; Wang, Z.; Overvelde, J. T. B.; Galloway, K. C.; Wood, R. J.; Bertoldi, K.; Walsh, C. J. *IEEE Trans. Robotics.* **2015**, 1, 1552–3098.
18. Ogden, R. W.; Saccomandi, G.; Sgura, I. *Comput. Mech.* **2004**, 34, 484.
19. Muliana, A. H.; Rajagopal, K. R.; Tscharnuter, D. "A Nonlinear Integral Model for Describing Responses of Viscoelastic Solids," *Int. J. Solids Struct.* **2015**, 58, 146–156.
20. Muliana, A.; Rajagopal, K. R.; Tscharnuter, D.; Pinter, G. *Int. J. Solids Struct.* **2016**, 100, 95.
21. Muliana, A.; Rajagopal, K. R.; Tscharnuter, D.; Schrittesser, B.; Saccomandi, G. *Rubber Chem. Technol.* **2018**, 91, 375–389.
22. Mansouri, M. R.; Darijani, H. *Int. J. Solids Struct.* **2014**, 51, 4316.
23. Ogden, R. W. *Proc. R. Soc. London A.* **1972a**, 326, 565.
24. Ogden, R. W. *Proc. R. Soc. London A.* **1972b**, 328, 567.
25. Gent, A. N. *Rubber Chem. Technol.* **1996**, 69, 59.
