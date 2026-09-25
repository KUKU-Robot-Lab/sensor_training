# Soft Barometric Tactile Sensor Utilizing Iterative Pressure Reconstruction — Formulas

> De Clercq, Sianov, Ostyn, Crevecoeur, *IEEE Access*, 2024.

## 1) Parameterized pressure distribution

연성 패드 하부의 직사각형 분포에 Gaussian drop-off를 결합:

\[
p(x, y) =
\begin{cases}
p_0, & \text{if } |x| < \dfrac{l_x}{2} \ \text{and}\ |y| < \dfrac{l_y}{2} \\
p_0 \, \exp\!\left(-\dfrac{d^2}{2\sigma^2}\right), & \text{otherwise}
\end{cases}
\tag{1}
\]

- \(p(x,y)\): 압력  
- \(p_0\) [Pa]: 중심 압력(최대값), 음수 허용(분포 빼기 가능)  
- \(l_x, l_y\) [mm]: 직사각형 주/부축 길이  
- \(d\): 직사각형 경계까지의 최소 카테시안 거리  
- \(\sigma\) [mm]: Gaussian drop-off 지수의 표준편차

곡률 도입: 중심을 \(y=10/\kappa_{\text{curve}}\)로 두고 극좌표 기반으로 직사각형을 만곡시켜 사용.  
- \(\kappa_{\text{curve}}\) [mm\(^{-1}\)]: 곡률. \(\kappa_{\text{curve}}\to 0\)이면 곡률 0, 값이 커질수록 곡률 증가.

## 2) Pose 및 회전 적용(좌표 변환)

분포의 위치/자세는 다음 변환으로 적용:

\[
\begin{bmatrix}
x' \\[2pt] y'
\end{bmatrix}
=
\begin{bmatrix}
\cos\alpha & -\sin\alpha \\
\sin\alpha & \ \cos\alpha
\end{bmatrix}
\!
\begin{bmatrix}
x - x_0 \\[2pt] y - y_0
\end{bmatrix}
\tag{2}
\]

- \(\alpha\) [deg]: 분포 주축과 \(x\)-축 사이 각도  
- \((x_0,y_0)\): 분포 중심의 카테시안 좌표

이때 최적화 파라미터 벡터는
\[
\theta=\big[p_0,\ \sigma,\ l_x,\ l_y,\ \kappa_{\text{curve}},\ \alpha,\ x_0,\ y_0\big].
\]

## 3) 파라미터 추정을 위한 최소제곱 목적함수

\[
\operatorname{Err}(\theta)
=
\sum_{i=1}^{n}
\Big[p(x_i, y_i;\ \theta) - \hat{p}_i\Big]^2
\tag{3}
\]

- \((x_i,y_i)\): 센서 \(i\)의 좌표  
- \(\hat{p}_i\): 센서 \(i\)의 측정 압력

## 4) 잔차 정의

\[
p_{\text{res},i} \;=\; p(x_i, y_i;\ \theta^\ast) \;-\; \hat{p}_i
\tag{4}
\]

- \(\theta^\ast\): (3)으로 구한 최적 파라미터

복합/다중 접촉에서는 (3)–(4)를 반복 적용해 여러 성분 분포의 합으로 최종 분포를 구성.

## 5) 수직 힘(법선력) 추정

최종 추정 분포를 접촉 패드 영역에 적분:

\[
F_z \;=\; \iint_{\text{pad}} p(x,y;\ \theta^\ast)\; dx\,dy
\tag{5}
\]

수치적분으로 계산.

## 6) Pressure Reconstruction Fit(PRF) 지표

도메인을 \(n\times n\) 격자로 표본화하고, 접촉 영역에서는 \(p>p_{\text{thres}}\), 비접촉 영역에서는 \(p<p_{\text{thres}}\)를 맞출 때 카운트를 증가. 평균하여 PRF 산출:

\[
\mathrm{PRF} \;=\; \frac{\text{fit}}{n^2}
\]

- \(p_{\text{thres}}\): 임계 압력  
- \(\text{fit}\): 일치 조건을 만족한 표본 지점의 수
