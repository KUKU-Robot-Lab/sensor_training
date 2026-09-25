# Super-resolution tactile sensor arrays with sparse units enabled by deep learning

![Fig. 2. The proposed tactile SR framework.
**그림 2. 제안된 촉각 초해상도(SR) 프레임워크.**
(A) 중첩된 수용영역을 활용하여 가상 taxel을 생성하는 원리.
(B) 수용영역 면적을 최대화하도록 최적화된 taxel 배열.
(C) 새로운 taxel을 확장하여 임의 형상의 두 영역을 커버하는 예시. 노란색과 파란색 음영 영역은 커버 대상 영역을 나타내며, 검은 실선과 점선은 그래프 구조를 구성한 후 인접 감지 유닛 사이를 연결하는 경계를 의미한다. 파란 점은 감지 유닛, 녹색 원은 각 감지 유닛의 수용영역을 나타낸다.
(D) 초해상도 감지 과정의 개략도. 촉각 센서 어레이로부터 다채널 신호를 수집하고, 이를 SATS 모델이 처리하여 압력 분포를 추론한다.
(E) 제안된 SATS 모델. 재귀적 특징 인코딩(recursive feature encoding), self-attention 기반 정보 집계, 고해상도 맵 생성 단계로 구성된다. 이 7노드 그래프의 각 노드는 특정 단계에서 해당 모듈이 인코딩한 taxel의 특징 벡터를 나타낸다.](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image.png)

Fig. 2. The proposed tactile SR framework.
**그림 2. 제안된 촉각 초해상도(SR) 프레임워크.**
(A) 중첩된 수용영역을 활용하여 가상 taxel을 생성하는 원리.
(B) 수용영역 면적을 최대화하도록 최적화된 taxel 배열.
(C) 새로운 taxel을 확장하여 임의 형상의 두 영역을 커버하는 예시. 노란색과 파란색 음영 영역은 커버 대상 영역을 나타내며, 검은 실선과 점선은 그래프 구조를 구성한 후 인접 감지 유닛 사이를 연결하는 경계를 의미한다. 파란 점은 감지 유닛, 녹색 원은 각 감지 유닛의 수용영역을 나타낸다.
(D) 초해상도 감지 과정의 개략도. 촉각 센서 어레이로부터 다채널 신호를 수집하고, 이를 SATS 모델이 처리하여 압력 분포를 추론한다.
(E) 제안된 SATS 모델. 재귀적 특징 인코딩(recursive feature encoding), self-attention 기반 정보 집계, 고해상도 맵 생성 단계로 구성된다. 이 7노드 그래프의 각 노드는 특정 단계에서 해당 모듈이 인코딩한 taxel의 특징 벡터를 나타낸다.

---

# INTRODUCTION

체화된 지각(embodied perception)은 체화 인공지능(embodied artificial intelligence)의 핵심 기반이며(1), 이를 통해 휴머노이드 로봇은 주변 환경과 인간의 의도를 이해하고, 복잡하고 고차원적인 작업을 수행할 수 있다(2, 3). 시각 지각도 중요하지만, 시각의 한계를 극복하기 위해 촉각 감각의 통합이 필수적이다(4). 특히 비정형 환경이나 어두운 공간, 가려진 공간처럼 시각 정보가 부족하거나 모호하거나 완전히 없는 상황에서 더욱 중요하다(5, 6). 촉각 센서는 일반적으로 감지 유닛(taxel)들이 격자 형태로 배열된 어레이 구조를 가진다. 그 작동 원리는 저항형(7–10), 정전용량형(11–13), 마찰전기형(14–16), 자기형(17, 18) 등 다양하다. 더 높은 공간 해상도는 더욱 정밀한 촉각 정보를 획득하고 정교한 조작을 가능하게 하기 위해 필수적이다. 그러나 물리적 수준에서 해상도를 높이려면 일반적으로 다양한 taxel을 조밀하게 배치하고, 많은 연결 배선과 복잡한 신호 처리 모듈이 필요하다. 이는 제작 난이도, 신뢰성 저하, 신호 간섭(cross-talk), 그리고 지속적으로 생성되는 대규모 촉각 데이터의 효율적 처리 문제를 야기한다.

카메라와 비전 알고리즘을 활용하는 시각 기반 촉각 센서는 고해상도 지각의 한 가지 해결책을 제공하지만(19, 20), 구조 설계의 제약과 큰 크기 때문에 다양한 로봇 촉각 응용 분야로의 확장성이 제한된다. 자연계에서는 인간의 피부가 이러한 문제 해결에 영감을 준다. 인간 피부의 기계수용기(mechanoreceptors, MRs)는 주변 자극을 감지하며(21, 22), 각 수용기는 수용영역(receptive field)이라 불리는 일정 영역 내 자극에 반응한다(23, 24). 인간 피부는 기계수용기의 물리적 분해능보다 더 미세한 공간 해상도로 자극을 감지할 수 있는데, 이를 촉각 초해상도(tactile super-resolution, SR)라 한다(25, 26). 이러한 능력은 인접한 기계수용기들의 수용영역이 서로 겹치기 때문에 가능하다(Fig. 1A). 이 중첩 구조는 여러 수용기 신호를 종합하여 자극 위치를 추정할 수 있게 한다(26, 27). 이 과정에서 신경계는 최소한의 센서 수로 외부 자극을 정확히 해석하는 신호 처리 장치 역할을 한다. 딥러닝 모델은 다중 소스 신호를 학습하고 통합함으로써 이러한 기능을 효과적으로 모사할 수 있으며, 이는 기존 연구들에서 폭넓게 탐구되어 왔다(28–30). 인간 피부의 촉각 초해상도 메커니즘을 바탕으로, 여러 촉각 센서들이 딥러닝을 이용해 단일 접점(single-point)(31–35) 또는 다중 접점(multi-point) 감지(table S1 요약)(36–39)에서 초해상도를 구현하였다. 실제 응용에서는 다중 접점 감지가 더 보편적이므로 더욱 중요하다. 그러나 다양한 접촉 상황에 대해 딥러닝 모델을 학습시키기 위한 대규모 데이터 수집은 여전히 어려운 과제이다(40, 41). 연구자들은 자성 필름과 홀 센서를 이용하여 다중 접촉 신호를 여러 단일 접촉 신호로 분리(decouple)하는 방식의 접근도 제안하였다(42, 43). 여러 단일 접촉 데이터를 수동으로 조합하여 다중 접촉 데이터를 합성하고, 이를 기반으로 다중 접촉 인식 모델을 학습시켰다. 이 방법은 대규모 데이터 수집 부담을 줄이지만, 학습된 모델의 일반화 성능은 여전히 학습 데이터 다양성에 의해 제한된다.

시뮬레이션 방법과 지식 전이(knowledge transfer) 기법이 일부 해결책을 제공했지만(36–39), 시뮬레이션 데이터와 실제 데이터 간의 도메인 차이(domain shift)는 지각 정확도를 떨어뜨린다. 또한 대면적 적용을 위해서는 최소한의 taxel 수로 고해상도 감각을 구현하는 것이 중요하다. 이를 위해서는 각 taxel의 수용영역을 최대한 활용하여 제한된 감지 자원을 최적으로 사용해야 한다. 그러나 현재 연구에서는 수용영역을 기반으로 taxel 배열을 최적화하는 문제를 충분히 다루지 못했으며, 그 결과 중복 배치와 자원 낭비가 발생하고 있다.

본 논문은 딥러닝 기법을 활용하여 촉각 센서에 초해상도 압력 분포 지각 능력을 부여하는 범용 계산 프레임워크를 제안한다. 먼저 촉각 센서 어레이의 배치를 최적화하기 위한 일반적인 위상(topological) 설계 방법을 제안하여, 제한된 감지 유닛을 가장 효율적으로 활용하고 최소 수의 유닛으로 최대 면적을 커버할 수 있도록 하였다. 이 배치를 기반으로 종단간(end-to-end) 자기주의 기반 촉각 초해상도 프레임워크(self-attention-assisted tactile SR, SATS)를 설계하여, 다채널 신호를 처리함으로써 감지 표면 전반의 자극(예: 압력) 분포를 추정한다.

이 프레임워크를 검증하기 위해 23개 노드를 가진 유연한 촉각 센서를 제작하고 SATS 모델로 보정(calibration)하여 다중 접점 초해상도 피부(multi-point SR skin, MSR-skin)를 구현하였다. 23개의 실제 taxel로부터 2700개의 가상 taxel을 생성하였으며, 초해상도 배율(scale factor, note S1 정의)은 약 117에 달했다. 감지 표면 전체에서 평균 위치 추정 오차는 0.73 mm [평균제곱근오차(RMSE)]로 나타났는데, 이는 인간 손끝의 위치 판별 정확도(0.3~1 mm)(25, 44)에 근접하며 기존 최고 수준(state-of-the-art) 기술보다 우수하다. 우리가 아는 한, 이는 유연형 센서에서 감지 표면 전체의 압력 분포를 인식할 수 있는 다중 접점 촉각 초해상도의 첫 번째 사례이다.

MSR-skin은 로봇 손에 통합되어 체화 촉각 지각 능력을 부여하였다(Fig. 1B). 또한 초해상도 능력을 직관적으로 보여주기 위해, 매우 조밀한 배열의 소프트 키보드를 감지 표면 위에 통합하여 다양한 인간-기계 상호작용 입력 장치로 활용하였다. 더 나아가, 접촉 지도(contact map) 기반의 초해상도 지각을 통해 센서는 다양한 형상을 성공적으로 영상화(imaging)하고 정확히 인식하였으며, 분류 정확도는 98.35%에 달했다. 제안된 초해상도 프레임워크의 범용성은 마찰전기 나노발전기(triboelectric nanogenerators, TENGs)를 기반으로 한 동적 초해상도 촉각 센서 설계에도 적용됨으로써 추가로 입증되었다.

---

# **RESULTS**

## Skin-­inspired design and layout optimization

![**Fig. 1. Skin-inspired design and SR sensing capability of the tactile sensor.**
**그림 1. 피부에서 영감을 받은 설계와 촉각 센서의 초해상도(SR) 감지 성능.**
(A) 인간 피부 구조와 피부 모사형 촉각 센서의 개략도. 두 구조 모두 외부 자극과 감지 유닛 사이에 탄성층을 전달 매체로 가지며, 중첩된 수용영역(receptive field)을 갖는다. SA와 RA는 각각 slowly adapting(느리게 적응하는 수용기)과 rapidly adapting(빠르게 적응하는 수용기)을 의미한다.
(B) 로봇 손에 통합된 촉각 센서의 개념도. 설계된 촉각 초해상도 시스템은 단일 접점(single-point)과 다중 접점(multi-point) 감지가 모두 가능하며, 고유의 공간 해상도를 크게 향상시킨다](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%201.png)

**Fig. 1. Skin-inspired design and SR sensing capability of the tactile sensor.**
**그림 1. 피부에서 영감을 받은 설계와 촉각 센서의 초해상도(SR) 감지 성능.**
(A) 인간 피부 구조와 피부 모사형 촉각 센서의 개략도. 두 구조 모두 외부 자극과 감지 유닛 사이에 탄성층을 전달 매체로 가지며, 중첩된 수용영역(receptive field)을 갖는다. SA와 RA는 각각 slowly adapting(느리게 적응하는 수용기)과 rapidly adapting(빠르게 적응하는 수용기)을 의미한다.
(B) 로봇 손에 통합된 촉각 센서의 개념도. 설계된 촉각 초해상도 시스템은 단일 접점(single-point)과 다중 접점(multi-point) 감지가 모두 가능하며, 고유의 공간 해상도를 크게 향상시킨다

인간 피부의 촉각 super-resolution(SR) 메커니즘에서 영감을 받아, 본 연구에서는 **인공 촉각 SR을 위한 일반 프레임워크**를 제안한다. Fig. 1A와 같이 인간 피부는 다층 구조를 가지며, 주로 **표피(epidermis), 진피(dermis), 피하조직(subcutaneous tissue)** 으로 구성된다. 촉각 지각을 담당하는 **mechanoreceptors(MRs)** 는 주로 진피층 내부에 존재한다(21). 표피는 주로 keratinocyte로 구성되며, 더 깊은 피부 구조와 신체 전체를 보호하는 역할을 한다. 진피는 collagen 및 elastic fiber가 풍부하여 탄성을 제공하는 동시에 내부의 MR들에게 기계적 지지 구조를 제공한다. 진피 아래의 피하조직은 지방층(adipose layer)과 느슨한 결합조직(loose connective tissue)으로 구성되며, 지지 및 완충 구조로 기능한다.

외력이 피부에 가해지면, 표피와 진피는 기계적 자극을 더 넓은 영역으로 분산시켜 여러 MR을 동시에 활성화한다. 이를 통해 체성감각 신경계(somatosensory nervous system)는 여러 MR로부터의 신호를 통합하여 힘의 위치 추정 정확도를 향상시킨다. 즉, 표피와 진피는 MR의 receptive field를 확장하고, 서로 겹치는 영역(overlapped area)을 형성하여 촉각 SR의 기반을 제공한다.

촉각 센서 배열의 구조 역시 인간 피부와 유사하게 볼 수 있으며, sensing unit과 각 sensing unit의 receptive field를 확장하는 전달 매체(transmission medium)로 구성된다. 예를 들어, elastic cover는 외력을 더 넓은 영역으로 분산시켜 sensing unit이 없는 영역에서도 자극 검출이 가능하게 한다(Fig. 2A). 활성화된 sensing unit들의 데이터를 종합하면 physical taxel 사이에 **virtual taxel**이 생성되며, 촉각 센서의 공간 해상도가 크게 향상된다.

촉각 SR을 달성하기 위해서는 receptive field 간 **중첩(overlap)** 이 필수적이며, 제한된 sensing resource를 최대한 활용할 수 있는 가장 효율적인 배열 구조를 탐색해야 한다. 2차원 평면에서 임의의 한 점 위치를 결정하려면, 최소한 **서로 일직선상에 있지 않은 세 점(noncollinear points)** 이 필요하다(45). 같은 원리로, 가해진 힘은 크기와 taxel과의 거리에 따라 각 taxel에 서로 다른 응답을 유도한다. 따라서 접촉 힘의 정확한 위치와 크기는 최소 세 개의 비공선 taxel의 위치와 응답을 분석함으로써 추론할 수 있다.

이에 따라 세 개의 taxel이 배열 최적화를 위한 기본 단위로 설정되었다(fig. S1A). 이 최적화 문제는 다음과 같이 정의된다.

$$
(d_1^,d_2^,d_3^*)=\arg\max_{d_1,d_2,d_3}(A_1\cup A_2\cup A_3),\quad \text{subject to } d_1\le r,\ d_2\le r,\ d_3\le r
$$

여기서

- $d_1, d_2, d_3$: taxel 간 거리
- $r$: receptive field 반지름
- $A_1=A_2=A_3=\pi r^2$: 각 taxel receptive field 면적

즉, $d_1,d_2,d_3$를 조절하여 세 receptive field가 세 taxel이 만드는 삼각형 영역을 완전히 덮도록 하면서, receptive field들의 union area를 최대화하는 것이 목적이다. 다시 말해, 최소 수의 taxel로 최대 면적을 덮는 것이 tactile SR의 핵심 목표이며, 이 문제를 해결함으로써 달성할 수 있다.

이 문제 해결에는 **particle swarm optimization(PSO)** 알고리즘이 사용되었으며, 최적 해는 다음과 같이 도출되었다.

$$
d_1^*=d_2^*=d_3^*=r
$$

(Fig. 2B, fig. S1B,C)

즉, 세 taxel은 한 변 길이가 receptive field 반지름과 동일한 정삼각형 구조로 배치되는 것이 최적임을 의미한다. 이 기본 구조를 기반으로 동일한 패턴을 반복 확장하면 임의 형상의 표면도 커버할 수 있다(Fig. 2C).

PSO 알고리즘은 빠른 수렴 속도와 우수한 성능 때문에 선택되었다. 다만 다른 machine learning 기반 최적화 기법들도 적용 가능하며, 여러 알고리즘 비교 결과는 note S2 및 fig. S2에 제시되어 있다.

또한 현재 설정에서는 세 receptive field 반지름이 동일하다고 가정하였다. 그러나 본 최적화 방법은 반지름이 서로 다른 경우에도 확장 가능하며, 이에 대한 상세 내용은 note S2 및 fig. S3에 제시되어 있다.

## The SATS deep learning model

- A tactile sensor array organized in the optimized layout can be conceptualized as a graph, where the nodes represent sensing units and the edges connect adjacent nodes, determined by the receptive fields, as shown in Fig. 2C. An SATS model, tailored to this structure, is proposed to process the original signals from the sensing units and estimate the pressure distribution across the entire sensing surface (Fig. 2D). The SATS model comprises four components: a long short-term
memory (LSTM)–based recurrent feature encoding module, a self-attention–
based information aggregation module, a local map construction module, and a convolutional neural network (CNN)–based refining module, as depicted in Fig. 2E.
    
    최적화된 배치로 구성된 촉각 센서 배열은 그래프로 개념화할 수 있으며, 여기서 노드는 감지 유닛(sensing unit)을 나타내고, 간선은 수용장(receptive field)에 의해 결정되는 인접 노드들을 연결한다(그림 2C). 이러한 구조에 맞추어 설계된 SATS 모델은 **감지 유닛들로부터의 원시 신호를 처리**하고 **전체 감지 표면에 걸친 압력 분포를 추정하기 위해 제안**되었다(그림 2D). SATS 모델은 네 가지 구성 요소로 이루어진다: 
    장단기 기억(long short-term memory, LSTM) 기반 순환 특징 인코딩 모듈, 
    자기 주의(self-attention) 기반 정보 집계 모듈, 
    국소 맵(local map) 구성 모듈, 그리고 
    합성곱 신경망(convolutional neural network, CNN) 기반 정제(refining) 모듈이며, 
    그림 2E에 도시되어 있다.
    
- LSTM networks are well-suited for encoding time-series data and have proven effective in modeling the hysteresis inherent in elastic sensing materials (46). Given that sensing units, even those from the same batch, have distinct response characteristics, a unique LSTM encoder is assigned to each sensing unit to accommodate its specific properties. The features encoded in this manner, called local features, cannot capture spatial touch information since a single point cannot define a plane. Therefore, a self-attention module, which has been proven effective in aggregating information from related parts (47), is introduced to facilitate information sharing among adjacent sensing units.  This module learns to aggregate information from neighbor units into the current unit, thereby integrating  multisource data to infer spatial touch information (Fig. 2E, middle). It allows for consideration of both the primary responsiveness of each sensing unit and the cooperation among adjacent sensing units. Besides, this approach naturally integrates the domain knowledge of the tactile SR mechanism, where the stimulus information on the sensing surface can be inferred by analyzing the responses of at least three sensing units.
    
    **LSTM 네트워크**는 시계열 데이터를 인코딩하는 데 적합하며, 탄성 감지 재료에 내재된 **이력현상(hysteresis)을 모델링하는 데 효과적인 것으로 입증**되었다(46). 동일한 배치에서 생산된 감지 유닛이라 하더라도 각 유닛은 서로 다른 응답 특성을 가지므로, **각 감지 유닛의 고유 특성을 반영하기 위해** **각각에 대해 별도의 LSTM 인코더가 할당**된다. 이러한 방식으로 인코딩된 특징은 **국소 특징(local features)**이라 불리며, 단일 점으로는 평면을 정의할 수 없기 때문에 공간적 촉각 정보를 포착할 수 없다. 따라서 관련된 부분들의 정보를 집계하는 데 효과적인 것으로 입증된 **자기 주의 모듈(47)을 도입하여 인접 감지 유닛 간 정보 공유를 촉진**한다. 이 모듈은 이웃 유닛들의 정보를 현재 유닛으로 집계하도록 학습되며, 이를 통해 다중 출처 데이터를 통합하여 공간적 촉각 정보를 추론한다(그림 2E, 가운데). 이는 각 **감지 유닛의 주된 응답성**과 **인접 감지 유닛 간 협력 효과를 모두 고려**할 수 있게 한다. 또한 이 접근법은 촉각 초해상도(SR) 메커니즘에 대한 도메인 지식을 자연스럽게 통합하는데, **감지 표면 위의 자극 정보는 최소 세 개 이상의 감지 유닛의 응답을 분석함으로써 추론**될 수 있다.
    
- Figure 2E (middle) illustrates the information aggregation process from six adjacent units to one central unit, with details provided in note S3. The features aggregated through this process were considered to contain stimulus information surrounding the central unit. These aggregated features are then concatenated with the original local features and fed into a multilayer perceptron (MLP)–based decoder (function gφ ) to construct local maps for each unit individually. All local maps are merged, according to the location of sensing units, to form the overall map. In this condition, each sensing unit is primarily responsible for constructing its own local map and partially contributes to the local maps of its neighboring units. When multiple contacts simultaneously stimulate areas of different local maps, this partitioning strategy enables local map inference in a relatively independent and parallel manner. This is analogous to decomposing a multi-point contact into several single-point contacts. From the perspective of deep models, the SATS mode extracts local features and constructs local maps. This design enables the local networks, including the LSTMs for each sensing unit and the MLP for local map construction, to focus on stimuli near a specific sensing unit and construct  corresponding local maps. In this manner, the local networks solve nearly identical problems for both single-point and multi-point presses, decomposing a multi-point contact into several single-point contacts while maintaining a consistent data distribution for the local networks. This allows the sensing  system to be directly applied to multi-point contact scenarios despite being calibrated with only single-point contact data. It effectively mitigates the domain shift issue (48) between single-point and multi-point contact data, substantially enhancing its generality to real-world applications. Last, two convolutional layers were used to refine the integrated map, providing more substantial fitting capabilities and smoother expressions to merge local pressure maps from individual taxels more effectively.
    
    그림 2E(가운데)는 여섯 개의 인접 유닛으로부터 하나의 중심 유닛으로 정보가 집계되는 과정을 보여주며, 자세한 내용은 주석 **S3**에 제시되어 있다. 이 과정을 통해 집계된 특징들은 중심 유닛 주변의 자극 정보를 포함하는 것으로 간주되었다. 이후 이러한 집계 특징들은 원래의 국소 특징(local features)과 연결(concatenation)되어 다층 퍼셉트론(multilayer perceptron, MLP) 기반 디코더(함수 gφ)에 입력되며, **각 유닛에 대한 국소 맵(local map)을 개별적으로 구성**한다. **모든 국소 맵은 감지 유닛의 위치에 따라 병합되어 전체 맵을 형성한다.**
    
    - Note S3. Formulation of the self-attention module
        
        Self-attention 모듈은 현재 taxel에 인접한 taxel들로부터 정보를 집계하도록 설계되었다. 서로 일직선상에 있지 않은 여러 sensing unit으로부터의 다중 정보를 결합함으로써, 공간 정보(예: 눌린 위치)를 추론할 수 있다.
        
        LSTM 모듈이 데이터를 인코딩한 후, 특정 sensing unit $i$의 local feature를 $h_i$라 한다. sensing unit $i$에 인접한 unit들의 집합은 $\mathcal{N}_i$로 나타낸다. 먼저, unit $i$와 $\mathcal{N}_i$에 속한 각 인접 unit 사이의 attention coefficient는 다음과 같이 계산된다.
        
        $$
        e_{ij}=a([Wh_i \parallel Wh_j]),\ \forall j\in\mathcal{N}_i \tag{S4}
        $$
        
        여기서 $W$는 선형 매핑을 위한 행렬이고, 기호 $\parallel$는 **concatenation 연산**을 나타내며, $a$는 결합된 feature를 실수값으로 변환하는 함수이다. $e_{ij}$는 계산된 **attention coefficient**로서, unit $j$의 feature가 unit $i$에 대해 **가지는 중요도**를 의미한다.
        
        그 다음, 모든 $j$에 대해 softmax 함수를 사용하여 정규화 연산을 수행한다 (56).
        
        $$
        \alpha_{ij}=\frac{\exp(\mathrm{LeakyReLU}(e_{ij}))}{\sum_{k\in\mathcal{N}i}\exp(\mathrm{LeakyReLU}(e_{ik}))} \tag{S5}
        $$
        
        정규화된 attention coefficient를 얻은 후, 다음과 같이 feature들의 선형 결합을 계산한다.
        
        $$
        h'_i=\omega\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}Wh_j\right) \tag{S6}
        $$
        
        여기서 $\omega$는 **비선형 활성화 함수**이며, exponential linear unit(ELU) (57)이다. $h'_i$는 unit $i$의 집계 feature로 사용된다. 이 전체 과정은 Figure 2E에 나타난 바와 같이 $f(\cdot)$로 표현된다.
        
    
    이 조건에서 각 감지 유닛은 주로 자신의 국소 맵을 구성하는 역할을 담당하며, 동시에 인접 유닛들의 국소 맵 구성에도 부분적으로 기여한다. **여러 접촉이 서로 다른 국소 맵 영역을 동시에 자극하는 경우, 이러한 분할 전략은 국소 맵 추론이 비교적 독립적이고 병렬적인 방식으로 이루어지도록 한다**. 이는 다점 접촉(multi-point contact)을 여러 개의 단일 점 접촉(single-point contact)으로 분해하는 것과 유사하다.
    
    딥러닝 모델의 관점에서 SATS 모델은 **국소 특징을 추출하고 국소 맵을 구성한다**. 이러한 설계는 각 감지 유닛에 대한 LSTM과 국소 맵 구성을 위한 MLP를 포함하는 **국소 네트워크들**이 **특정 감지 유닛 근처의 자극에 집중**하고, 이에 대응하는 **국소 맵을 구성할 수 있도록 한다**. 이 방식으로 국소 네트워크들은 단일 점 압력과 다점 압력 모두에 대해 거의 동일한 문제를 해결하게 되며, 다점 접촉을 여러 개의 단일 점 접촉으로 분해하면서도 국소 네트워크에 대해 일관된 데이터 분포를 유지한다.
    
    이를 통해 감지 시스템은 **단일 점 접촉 데이터만으로 보정(calibration)되었음에도 다점 접촉 상황에 직접 적용될 수 있다.** 이는 단일 점 접촉 데이터와 다점 접촉 데이터 사이의 도메인 시프트(domain shift) 문제(48)를 효과적으로 완화하여 실제 응용 환경에서의 범용성을 크게 향상시킨다. 마지막으로, 통합된 맵을 정제하기 위해 두 개의 합성곱 층(convolutional layers)이 사용되었으며, 이는 더 강력한 적합 능력과 더 매끄러운 표현을 제공하여 개별 택셀(taxel)로부터 생성된 국소 압력 맵들을 보다 효과적으로 병합하도록 한다.
    
- Through the above procedures, the SATS model takes raw multichannel signals from the sensor array as input and outputs a two-dimensional matrix representing the inferred pressure distribution across the sensing surface. This matrix has a size of 54 by 50, with each value corresponding to the pressure at a specific position on the sensing surface, which is considered a virtual taxel. Therefore, the SATS model generates 54 by 50 = 2700 virtual taxels in a single inference. The structural design of the SATS model incorporates both the general signal characteristics of tactile sensors and the inherent structure of sensor arrays, ensuring computational efficiency and broad applicability across various scenarios. This distinguishes SATS from existing approaches that use general machine learning models without explicitly considering SR scenarios, such as plain MLPs (8, 17, 49), interpolation models (42, 43), and CNNs (39). While these methods enable SR tactile sensing, they heavily rely on high-quality annotated data for model training. This requirement becomes even more demanding in multi-point SR tasks. In contrast, the proposed SATS model demonstrates superior learning efficiency by leveraging knowledge transfer from single-point to multi-point contact in a zero-shot manner.
    
    상기 절차를 통해 SATS 모델은 센서 배열로부터의 원시 다채널 신호(raw multichannel signals)를 입력으로 받아, **감지 표면 전체에 걸친 추정 압력 분포를 나타내는 2차원 행렬을 출력**한다. 이 행렬의 크기는 54 × 50이며, 각 값은 **감지 표면의 특정 위치에서의 압력에 대응**하고, **이는 가상 택셀(virtual taxel)로 간주**된다. 따라서 SATS 모델은 한 번의 추론으로 54 × 50 = 2700개의 가상 택셀을 생성한다.
    
    SATS 모델의 구조적 설계는 촉각 센서의 일반적인 신호 특성과 센서 배열의 고유 구조를 모두 반영하여, 계산 효율성과 다양한 상황에 대한 폭넓은 적용 가능성을 보장한다. 이는 초해상도(SR) 시나리오를 명시적으로 고려하지 않은 일반적인 기계학습 모델을 사용하는 기존 접근법들과 SATS를 구별짓는 요소이다. 이러한 기존 방법으로는 단순 다층 퍼셉트론(plain MLPs)(8, 17, 49), 보간(interpolation) 모델(42, 43), 그리고 합성곱 신경망(CNNs)(39) 등이 있다.
    
    이들 방법 역시 SR 촉각 감지를 가능하게 하지만, 모델 학습을 위해 고품질 주석 데이터(high-quality annotated data)에 크게 의존한다. 이러한 요구 사항은 다점(multi-point) SR 과제에서는 더욱 까다로워진다. 반면, 제안된 SATS 모델은 단일 점 접촉(single-point contact)에서 다점 접촉으로의 지식 전이(knowledge transfer)를 제로샷(zero-shot) 방식으로 활용함으로써, 더 우수한 학습 효율성을 보여준다.
    

## Fabrication and characterization of the soft tactile sensor

![Fig. 3. Fabrication and characterization of the tactile sensor.
**그림 3. 촉각 센서의 제작 및 특성 평가.**
(A) 제작된 감지 유닛의 분해도와 CNT가 표면에 매립된 재료 표면의 SEM 이미지. PI는 polyimide를 의미한다.
(B) 다양한 압력 조건에서의 상대 저항 변화. R₀는 압력이 가해지지 않았을 때의 저항값이다.
(C) 감지 유닛의 히스테리시스 특성.
(D) 12.7~318.3 kPa 압력 범위에서의 동적 응답 특성.
(E) 단일 감지 유닛의 수용영역 측정 장치와 그 결과.
(F) 센서 어레이 프로토타입 사진.
(G) 8 N 외력을 기준으로 보정된 중앙 감지 유닛의 수용영역. 파란 점은 실제 taxel을 의미한다.
(H) 모든 taxel의 수용영역을 중첩하여 얻은 전체 촉각 센서의 수용영역. 오른쪽 단면도는 가장 오른쪽에 위치한 5개 taxel의 다양한 힘 조건에서의 응답을 보여준다.
(I) 한 점, 두 점, 세 점에 각각 압력이 가해졌을 때 센서 어레이의 응답 특성.](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%202.png)

Fig. 3. Fabrication and characterization of the tactile sensor.
**그림 3. 촉각 센서의 제작 및 특성 평가.**
(A) 제작된 감지 유닛의 분해도와 CNT가 표면에 매립된 재료 표면의 SEM 이미지. PI는 polyimide를 의미한다.
(B) 다양한 압력 조건에서의 상대 저항 변화. R₀는 압력이 가해지지 않았을 때의 저항값이다.
(C) 감지 유닛의 히스테리시스 특성.
(D) 12.7~318.3 kPa 압력 범위에서의 동적 응답 특성.
(E) 단일 감지 유닛의 수용영역 측정 장치와 그 결과.
(F) 센서 어레이 프로토타입 사진.
(G) 8 N 외력을 기준으로 보정된 중앙 감지 유닛의 수용영역. 파란 점은 실제 taxel을 의미한다.
(H) 모든 taxel의 수용영역을 중첩하여 얻은 전체 촉각 센서의 수용영역. 오른쪽 단면도는 가장 오른쪽에 위치한 5개 taxel의 다양한 힘 조건에서의 응답을 보여준다.
(I) 한 점, 두 점, 세 점에 각각 압력이 가해졌을 때 센서 어레이의 응답 특성.

안정성과 내구성을 고려하여 감지 유닛 제작에는 압저항성(piezoresistive) 재료가 선택되었다. 그림 S4와 같이, 다공성 탄소나노튜브(CNT) 기반 스펀지는 기공률 110 pixels per inch(PPI)의 니켈(Ni) 폼을 다공성 템플릿으로 사용하는 일반적인 산 에칭 공정을 통해 제조되었다. 먼저 CNT와 클로로포름의 혼합물을 준비한 뒤 초음파 세포 파쇄 시스템으로 분산시켰다. 다음으로, 폴리디메틸실록산(PDMS) 전구체와 경화제 혼합물에 클로로포름을 첨가하여 희석하였다. 희석된 PDMS 전구체 용액을 CNT/클로로포름 혼합물에 넣고 상온에서 교반하였다. 이후 생성된 혼합물(CNT/클로로포름/PDMS)을 가열 및 교반하여 일부 클로로포름을 증발시켰다.

Ni 폼은 적외선 레이저 시스템을 이용해 원형으로 절단되었다. 이 Ni 폼을 CNT/클로로포름/PDMS 혼합물에 짧게 담근 후, 체 위에서 흔들어 불균일하거나 응집된 부분을 제거하였다. 열처리 후, 경화된 시료를 희석 염산(HCl) 용액에 담가 에칭하였다. 마지막으로 증류수로 여러 차례 세척하여 다공성 CNT/PDMS 스펀지를 얻었다. 그림 3A는 직경 10 mm, 두께 2 mm의 제작된 감지 유닛을 보여준다. 주사전자현미경(SEM) 이미지를 통해 감지 재료의 다공성 구조를 확인할 수 있다(그림 S5). 이 재료의 전도성은 PDMS 내부에 반쯤 매립된 CNT에 의해 제공되며, 압력이 가해질 때 CNT들의 접촉과 분리에 따라 전도도가 변화한다.

감지 재료의 특성 평가를 위해, 감지 유닛 상하부에 은 페이스트를 전도성 접착제로 사용하여 두 개의 전극을 부착하였다. 감지 유닛의 측정 범위 내 상대 저항 변화는 그림 3B에 나타나 있다. 5 N의 순간 자극 하에서 감지 유닛은 30 ms의 빠른 응답 속도를 보였다(그림 S6A). 또한 감지 유닛의 히스테리시스도 시험되었으며(그림 3C), 비교적 작은 범위(1.98%)를 나타냈다. SATS 모델의 LSTM은 하중 인가 및 제거 과정에서 나타나는 이러한 미세한 응답 차이를 모델링할 수 있다.

또한 이 감지 유닛은 동적 하중 조건에서 우수한 안정성과 반복성을 보였다(그림 3D). 외부 압력이 0.4 kPa에서 318 kPa까지 증가함에 따라 상대 저항 변화도 점진적으로 증가하였다. 특히 이 다공성 감지 재료는 매우 민감하여 0.4 kPa 수준의 작은 압력도 감지할 수 있었다(그림 3D 삽입도). 장치의 내구성을 검증하기 위해 3000회 이상의 반복 하중-제거 사이클 시험을 수행하였으며(그림 S6B), 촉각 센서 어레이 구성 요소로서 충분한 내구성을 입증하였다.

추가적으로, 서로 다른 기공률을 가진 세 종류의 Ni 폼으로 제작된 CNT/PDMS 재료의 성능도 비교 분석하였다(그림 S7). 기공률 80, 110, 130 PPI 시료는 80 kPa 이하 압력 구간에서는 유사한 감도를 보였다. 기공률이 낮고 기공 크기가 큰 CNT/PDMS는 저압 영역에서 약간 더 높은 감도를 나타냈다. 그러나 80 PPI와 130 PPI 시료는 고압 영역(100~300 kPa)에서는 110 PPI 시료보다 낮은 감도를 보였다. 따라서 본 연구에서는 넓은 압력 범위에서의 종합적인 감도 성능을 고려하여, 110 PPI 기공률의 Ni 폼을 CNT/PDMS 압저항 센서 제작용으로 선택하였다.

주목할 점은 Ni 폼이 압축 가능한 재료로 사용된 것이 아니라, 제한된 탄성 변형률과 높은 기계적 강도를 가지므로 산 에칭을 위한 템플릿으로 사용되었다는 점이다. 순수 Ni 폼과 제작된 CNT/PDMS 스펀지의 비교 결과는 그림 S8에 제시되어 있다.

수용영역(receptive field)을 확장하기 위해 두께 5 mm의 탄성체(elastomer)를 감지 유닛 상부에 배치하였다. 이 고무 기반 탄성체는 접촉역학(contact mechanics)에 따라 상부에 가해진 외력을 효과적으로 분산시켰다(50). 그림 3E는 탄성체로 덮인 감지 유닛의 수용영역 측정 장치를 보여준다. 통합된 감지 유닛은 이동 플랫폼 위에 놓였으며, 플랫폼 이동을 통해 인덴터가 대칭선을 따라 각 위치를 누를 수 있도록 하였다.

2 N에서 25 N 범위의 힘으로 서로 다른 위치를 누르면서 감지 유닛의 응답을 기록하였다. 감지 유닛은 중심으로부터 19 mm 떨어진 위치에서도 외력을 감지할 수 있었다. 그러나 명확하고 구별 가능한 신호 및 적절한 신호대잡음비(signal-to-noise ratio)를 확보하기 위해 최종적으로 수용영역 반경을 10.5 mm로 결정하였으며, 이는 이후 센서 어레이에서 인접 유닛 간 거리로 사용되었다.

그림 S9는 MSR-skin의 하드웨어 기반이 되는 23노드 센서 어레이 프로토타입의 제작 공정을 보여준다. 먼저 폴리이미드(polyimide) 필름 기판 위에 23개의 구리 전극으로 구성된 전극층을 제작하였다. 이후 마스크 역할을 하는 폴리에틸렌 테레프탈레이트(PET) 필름을 전극층 위에 덮고, 디지털 레이저로 정밀 절단하여 전극 부분이 노출되도록 하였다. 은 페이스트는 블레이드 코팅 방식으로 전극 위에 도포되어 전극과 감지 유닛 사이의 전기적 연결을 형성하였다.

가장자리 taxel은 중앙 taxel보다 인접 taxel 수가 적어 힘 조건이 달라지고 전극 연결 안정성이 떨어질 수 있다. 따라서 센서 어레이 외곽 둘레에는 추가 지지를 위해 스페이서를 배치하였다. 마지막으로 상부 전극층을 부착하여 제작을 완료하였고, 최종적으로 23노드 센서 어레이를 얻었다(그림 3F).

23채널 저항 신호를 전압 분배 방식으로 측정하기 위한 회로 기판도 설계 및 제작되었다. 회로도는 그림 S10에 제시되어 있다.

센서 어레이 성능 평가를 위해 감지 표면의 격자 위치들을 자극하여 응답을 측정하였다. 끝단에 힘 센서가 장착된 로봇 팔을 이용해 인덴터가 감지 표면 각 위치를 누르도록 프로그래밍하였다(그림 S11). 로봇 팔은 1 mm 간격으로 이동하도록 설정되어 전체 감지 표면(72 mm × 66 mm)을 72 × 66 행렬로 분할하였으며, 총 4891개(73 × 67)의 가압 지점을 생성하였다. 각 지점에서 힘 피드백 제어를 통해 압입력을 점진적으로 증가시켜 최대 10 N까지 가압하였다. 데이터 수집 과정 동안 인덴터 위치, 힘 센서 측정값, 센서 어레이 응답을 모두 기록하였다.

획득한 데이터를 바탕으로 각 taxel의 수용영역을 특성화하였다. 구체적으로, 감지 표면 각 위치를 8 N 힘으로 누를 때 각 taxel의 응답(상대 저항 변화)을 기록하였다. 결과는 그림 3G 및 그림 12에 제시되어 있다. 보다 명확한 시각화를 위해 5% 이하 값은 제거하고, 나머지 값은 1로 정규화하여 수용영역을 표시하였다(그림 S13).

각 taxel의 수용영역은 인접 유닛의 위치까지 포함하고 있어, 수용영역 중첩이라는 설계 목표를 충족하며 이후 초해상도 구현의 기반이 된다. 모든 taxel의 수용영역을 중첩하면 그림 3H와 같이 센서 어레이 전체의 수용영역을 얻을 수 있다. 또한 센서 어레이 오른쪽에 위치한 5개 taxel의 응답도 그림 3H(오른쪽)에 시각화하였다. 이는 이들 taxel의 공통 대칭선을 따라 각 위치를 누를 때의 응답을 보여준다. 하나의 taxel 중심을 눌렀을 때 인접 taxel들도 압력을 감지하였으며, 이는 해당 위치가 여러 taxel의 수용영역에 포함됨을 의미한다.

그림 3I는 센서 어레이 상의 한 점, 두 점, 세 점에 압력이 가해졌을 때 23채널(감지 유닛)의 상대 저항 변화를 보여준다. 서로 다른 접촉 패턴은 23채널 전체에서 서로 다른 응답 특성을 나타냈다. 이러한 시계열 응답 특징을 통해 SATS 모델은 감지 표면 전체의 압력 분포를 더욱 정밀하게 학습하고 추론할 수 있으며, 결과적으로 촉각 초해상도를 달성한다.

이 특성 평가 결과는, 고품질 감지 신호를 생성하는 센서 어레이와 이를 처리·해석하는 SATS 모델이 결합된 촉각 감지 시스템의 가능성을 보여준다.

## Calibration and validation of the MSR-skin system

- Ground truth, i.e., the expected pressure distribution, is essential for calibrating the MSR-skin comprising the sensor array and the SATS model. Conventionally, two approaches can be used to generate ground truth: direct measurement and simulation. However, measuring the pressure distribution across a surface directly is nearly impossible, and conventional simulation methods, such as the finite element model (FEM), are computationally expensive. To address this issue, a simulation method was proposed to approximate real scenarios based on the elastic half-space (EHS) model, using data recorded by the robot arm (coordinates and force). The EHS model describes how stress evolves inside an elastomer when its top surface is subjected to a point-contact pressure and has been used to model the mechanical properties of tactile sensors (51, 52). A simulation model based on this theory was constructed, as shown in fig. S14A. When a point on the elastomer’s surface is pressed, the stress at another point inside the elastomer is inversely proportional to the distance between them, which can be formulated as detailed in note S4. The pressure distribution was examined when a point load was applied. Figure S14 (B and C) illustrates the pressure distribution for varying external forces and elastomer thicknesses. It is observed that the elastomer’s thickness primarily influences the receptive field. A thicker elastomer leads to a larger receptive field by smoothing the pressure distribution. For a given thickness, a larger external force results in a higher pressure at the same location, thereby enhancing the sensor’s response and improving measurement precision. Figure S14D presents the pressure distribution in a two-dimensional format.
    
    실제 압력 분포, 즉 예상되는 압력 분포는 센서 어레이와 SATS 모델로 구성된 MSR-skin을 보정하는 데 필수적입니다. 일반적으로 실제 압력 분포를 생성하는 데는 직접 측정과 시뮬레이션의 두 가지 접근 방식이 사용됩니다. 그러나 표면 전체의 압력 분포를 직접 측정하는 것은 거의 불가능하며, 유한 요소 모델(FEM)과 같은 기존 시뮬레이션 방법은 계산 비용이 많이 듭니다. 이러한 문제를 해결하기 위해 로봇 팔에서 기록된 데이터(좌표 및 힘)를 사용하여 **탄성 반공간(EHS) 모델을 기반으로 실제 시나리오를 근사화하는 시뮬레이션 방법이 제안**되었습니다. EHS 모델은 엘라스토머의 윗면이 점 접촉 압력을 받을 때 **엘라스토머 내부의 응력이 어떻게 변화하는지 설명**하며, 촉각 센서의 **기계적 특성을 모델링하는 데 사용되어 왔습니다(51, 52)**. 이 이론을 기반으로 한 시뮬레이션 모델이 **그림 S14A**에 나타낸 바와 같이 구축되었습니다. 엘라스토머 표면의 한 지점을 누르면 엘라스토머 내부의 다른 지점에서의 응력은 두 지점 사이의 거리에 반비례하며, 이는 **주석 S4에 자세히 설명된 바와 같이 공식화**할 수 있습니다. 점하중을 가했을 때의 압력 분포를 조사했습니다. 그림 S14(B 및 C)는 다양한 외부 힘과 엘라스토머 두께에 따른 압력 분포를 보여줍니다. **엘라스토머의 두께가 수용 영역에 주로 영향을 미치는 것을 알 수 있습니다.** 엘라스토머가 두꺼울수록 압력 분포가 평활해져 수용 영역이 넓어집니다. 주어진 두께에서 외부 힘이 클수록 동일한 위치에서의 압력이 높아지므로 센서의 응답이 향상되고 측정 정밀도가 높아집니다. 그림 S14D는 압력 분포를 2차원 형식으로 나타낸 것입니다.
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%203.png)
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%204.png)
    
- Note that, in real-world scenarios, forces are applied over a surface rather than at a single point, which reflects the conditions during the data acquisition process. Therefore, we first discretized this surface contact area with a 5-mm radius into 80-point contacts (fig. S15A). The overall pressure distribution was then calculated on the basis of the previously established EHS model for a single-point contact. To mitigate potential errors introduced by discretization, an FEM of the elastomer was developed for rectification (fig. S15B). The discrepancy between the EHS model and the FEM (fig. S15C) was corrected using  a pressure-dependent correction factor, β. Figure S15D provides examples of the rectified EHS model and the corresponding β values. A second-order function was used to fit the relationship between β  and external pressure (fig. S16), enabling the EHS model to be corrected under varying pressure conditions (detailed in note S3). To be noted, the FEM was only used in the early process of rectification for fitting  β and excluded during generating ground-truth pressure maps, avoiding time-consuming simulations. Last, the rectified EHS model received the coordinates and force magnitude and generated a pressure map by calculating the pressure value at each position. This   map reflected the pressure distribution in the physical world and was used as the ground truth for training the SATS model. Notably, the rectified EHS model can calculate pressure values spatially continuously, allowing the constructed ground-truth pressure map to have any size or shape. Therefore, to align with the pressure map inferred by the SATS model, the ground-truth pressure map was constructed to correspond with each position in the inferred map. The  shape of each ground-truth pressure map was sized at 54 mm by 50 mm.
    
    실제 시나리오에서는 힘이 단일 지점이 아닌 **표면 전체에 작용한다는 점에 유의**해야 합니다. 이는 데이터 수집 과정 중의 조건을 반영합니다. 따라서 먼저 **반경 5mm의 접촉면을 80개의 접촉점으로 이산화**했습니다(그림 S15A). 그런 **다음 단일 접촉에 대해 이전에 확립된 EHS 모델을 기반으로 전체 압력 분포를 계산**했습니다. 이산화로 인해 발생할 수 있는 오류를 완화하기 위해 보정을 위해 엘라스토머의 유한 요소 모델(FEM)을 개발했습니다(그림 S15B). **EHS 모델과 FEM 간의 차이(그림 S15C)는 압력에 따라 달라지는 보정 계수 β를 사용하여 수정**했습니다. 그림 S15D는 보정된 EHS 모델과 해당 β 값의 예를 보여줍니다. **β와 외부 압력 간의 관계를 맞추기 위해 2차 함수를 사용**했으며(그림 S16), 이를 통해 **다양한 압력 조건에서도 EHS 모델을 보정할 수 있었습니다**(자세한 내용은 주석 S3 참조). 참고로, 유한요소법(FEM)은 초기 보정 과정에서 β 값을 맞추는 데에만 사용되었고, 시간 소모적인 시뮬레이션을 피하기 위해 실제 압력 분포도를 생성하는 과정에서는 제외되었습니다. 최종적으로, **보정된 EHS 모델은 좌표와 힘의 크기를 입력받아 각 위치에서의 압력 값을 계산하여 압력 분포도를 생성**했습니다. 이 분포도는 물리적 세계의 압력 분포를 반영하며, **SATS 모델 학습을 위한 실제 압력 분포도로 사용**되었습니다. 특히, 보정된 EHS 모델은 공간적으로 연속적인 압력 값 계산이 가능하므로, 생성된 실제 압력 분포도는 어떤 크기나 모양이라도 가질 수 있습니다. 따라서 SATS 모델이 추론한 압력 분포도와 일치하도록, **실제 압력 분포도는 추론된 분포도의 각 위치에 대응하도록 생성**되었습니다. 각 실제 압력 분포도의 크기는 54mm x 50mm였습니다.
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%205.png)
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%206.png)
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%207.png)
    
- The same dataset, collected through the robot arm and used to characterize the sensor array, was further used to train the SATS model along with the ground truths generated by the EHS model. The gradient descent method was used for training. The training process served as a calibration of the MSR-skin system. Figure S17 exhibits examples of the model’s inferences posttraining. The SATS model successfully estimated the pressure distribution across the entire sensing surface under varying forces and positions. It substantially enhanced the sensor array’s intrinsic spatial resolution by generating 54 by 50 = 2700 virtual taxels from 23 physical taxels, achieving a scale factor of 2700/23 ≈ 117. To comprehensively demonstrate the MSR-skin’s response, points along the symmetry line were sequentially pressed with a step size of 5 mm, applying forces ranging from 1 to 9 N. Responses from the five sensing units along this line were extracted and visualized (Fig. 4A). Sectional views of the inferred  pressure maps along their symmetry plane are presented for clarity. By integrating responses from multiple sensing units, the SATS model effectively decouples the position and magnitude of the external  force, thereby accurately estimating the pressure distribution.
    
    **로봇 팔을 통해 수집되어 센서 어레이의 특성을 파악하는 데 사용된 동일한 데이터 세트**를 **EHS 모델에서 생성된 정답 데이터와 함께 SATS 모델 학습에 사용**했습니다. 학습에는 경사 하강법을 사용했으며, 이 학습 과정은 MSR-skin 시스템의 보정 역할을 했습니다. 그림 S17은 학습 후 모델의 추론 예시를 보여줍니다. SATS 모델은 다양한 힘과 위치에서 전체 감지 표면의 압력 분포를 성공적으로 추정했습니다. **23개의 물리적 택셀에서 54 x 50 = 2700개의 가상 택셀을 생성**하여 센서 어레이의 고유 공간 해상도를 크게 향상시켰으며, 2700/23 ≈ 117의 스케일 팩터를 달성했습니다. MSR-skin의 반응을 종합적으로 보여주기 위해 **대칭선을 따라 5mm 간격으로 1~9N 범위의 힘을 순차적으로 가했습니다.** 이 선을 따라 위치한 5개의 센싱 유닛에서 얻은 반응을 추출하여 시각화했습니다(그림 4A). 추론된 압력 분포도를 대칭면을 따라 단면도로 제시하여 이해를 돕습니다. SATS 모델은 여러 센싱 장치의 응답을 통합함으로써 외부 힘의 위치와 크기를 효과적으로 분리하여 압력 분포를 정확하게 추정합니다.
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%208.png)
    
    ![fig4.a](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%209.png)
    
    fig4.a
    
- To quantitatively examine the performance of this sensing system, the inference error at each position on the sensing surface was calculated. Specifically, the RMSE was computed across the entire sensing surface for a given position under varying forces. Since each position was pressed with different forces, the errors for a specific position (x and y) were averaged over the force range. This calculation process is illustrated in fig. S18 and can be formulated as (2)
where N denotes the number of data samples collected at (x and y),
and $P^{gt}_i$and $P^{pred}_i$ are the ground-truth pressure and the model-inferred pressure at the ith press, respectively.
    
    이 센싱 시스템의 성능을 정량적으로 검토하기 위해 센싱 표면의 각 위치에서 추론 오류를 계산했습니다. 구체적으로, 주어진 위치에서 다양한 힘에 대해 전체 센싱 표면에 걸쳐 RMSE를 계산했습니다. 각 위치에 서로 다른 힘으로 압력을 가했기 때문에 특정 위치(x 및 y)에 대한 오류는 힘 범위에 걸쳐 평균화되었습니다. 이 계산 과정은 그림 S18에 나타나 있으며 다음과 같이 공식화할 수 있습니다.
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2010.png)
    
    여기서 N은 (x 및 y)에서 수집된 데이터 샘플 수를 나타내고,
    $P^{gt}_i$와 $P^{pred}_i$는 각각 i번째 압력에서의 실제 압력과 모델 추론 압력입니다.
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2011.png)
    
- Figure 4B illustrates the calculated error distribution, revealing that the maximal error is ~0.2 kPa, and the average error is 0.116 kPa. Further statistical analysis of this error map shows that 90% of the errors are below 0.15 kPa (Fig. 4C). Comparison of interpolation methods and the SATS model, as well as the results of ablation experiments, is detailed in note S5, table S2, and fig. S19. In addition to the pressing position, the RMSE error with respect to external force was also examined. Figure 4D shows an increasing trend in error as the force rises from 1 to 10 N, with saturation occurring when the force exceeds 5 N. Furthermore, the localization error was assessed, reflecting the discrepancy between the actual pressing position and the model-inferred position. Since the SATS model does not provide the pressing position directly but rather a pressure map, the location of the maximum pressure value was identified as the pressing position. Figure S20 displays the distribution of localization error (RMSE) in relation to pressing position and external force.
    
    그림 4B는 계산된 오차 분포를 보여주는데, **최대 오차는 약 0.2 kPa이고 평균 오차는 0.116 kPa임**을 알 수 있습니다. 이 오차 분포에 대한 추가적인 통계 분석 결과, **오차의 90%가 0.15 kPa 미만**인 것으로 나타났습니다(그림 4C). 보간법과 SATS 모델의 비교, 그리고 절제 실험 결과는 주석 S5, 표 S2, 그림 S19에 자세히 설명되어 있습니다. 압착 위치 외에도 외부 힘에 대한 RMSE 오차도 조사했습니다. 그림 4D는 **힘이 1 N에서 10 N으로 증가함에 따라 오차가 증가하는 경향**을 보이며, **힘이 5 N을 초과하면 포화 상태에 도달**합니다. 또한, 실제 압착 위치와 모델에서 추론한 위치 간의 차이를 반영하는 위치 오차를 평가했습니다. **SATS 모델은 압착 위치를 직접 제공하는 것이 아니라 압력 분포를 제공하기 때문에 최대 압력 값이 나타나는 위치를 압착 위치로 정의**했습니다. 그림 S20은 누르는 위치와 외부 힘에 따른 위치 오차(RMSE)의 분포를 보여줍니다.
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2012.png)
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2013.png)
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2014.png)
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2015.png)
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2016.png)
    
- Unlike the pressure map inference error, **the localization error** shows a descending trend as the external force increases from 1 to 10 N. This discrepancy arises from the trade-off between pressure map distortion and the signal-to-noise ratio of taxel responses. On the one hand, as the applied force increases, the corresponding pressure map exhibits greater distortion, indicating that the difference between the maximum and minimum values becomes more pronounced. This increases the difficulty of resolving fine pressure variations by the SATS model, leading to higher errors in the pressure map inference. On the other hand, force magnitude was not considered when evaluating localization errors, thereby eliminating errors introduced by  force estimation. Moreover, a larger applied force enhances taxel responses, increasing the signal-to-noise ratio and improving the accuracy of pressing position localization. The average localization error across all pressing positions and force ranges was 0.73 mm, slightly less than the step size for data collection. Therefore, it can be reasonably assumed that the localization error of the sensing system is strongly related to the step size in the data collection process. It is promising to further enhance the SATS model’s localization precision and reduce the localization error. This can be achieved by using finer grid segmentation (i.e., smaller step sizes) to collect data for training an SATS model with an increased latent dimension and larger inferred map size.
    
    압력 맵 추론 오차와는 달리, **위치 추정 오차**는 외부 힘이 1N에서 10N으로 증가함에 따라 감소하는 경향을 보입니다. 이러한 차이는 압력 맵 왜곡과 택셀 반응의 신호 대 잡음비 사이의 상충 관계에서 비롯됩니다. 한편, **가해지는 힘이 증가함에 따라 해당 압력 맵은 더 큰 왜곡을 나타내며, 이는 최대값과 최소값의 차이가 더욱 두드러짐을 의미**합니다. 이로 인해 SATS 모델이 미세한 압력 변화를 구분하기 어려워지고, 압력 맵 추론 오차가 증가합니다. 다른 한편으로, **위치 추정 오차를 평가할 때 힘의 크기를 고려하지 않았으므로 힘 추정으로 인한 오차가 제거**되었습니다. 또한, 가해지는 힘이 클수록 택셀 반응이 향상되어 신호 대 잡음비가 증가하고 누르는 위치 추정 정확도가 향상됩니다**. 모든 누르는 위치와 힘 범위에 걸친 평균 위치 추정 오차는 0.73mm**로, **데이터 수집 단계 크기보다 약간 작습니다**. 따라서 센싱 시스템의 위치 추정 오차는 데이터 수집 과정의 단계 크기와 밀접한 관련이 있다고 합리적으로 추정할 수 있습니다. SATS 모델의 위치 추정 정확도를 더욱 향상시키고 위치 추정 오류를 줄이는 것은 유망한 접근 방식입니다. 이를 위해서는 **더 세밀한 그리드 분할(즉, 더 작은 스텝 크기)을 사용하여 데이터를 수집하고, 잠재 차원을 증가시키고 추론된 지도 크기를 확대하여 SATS 모델을 학습시켜야 합니다.**
    
- A notable advantage of the proposed SATS model is the zero-shot knowledge transfer from single-point contacts to multi-point contacts. Figure 4E and movie S1 illustrate the sensing system’s inferences on pressure distributions under forces applied at one, two, and three points, respectively. Additional examples of the sensing system detecting multi-point contacts are presented in fig. S21, including the simultaneous detection of four, five, and six contact points, respectively. This capability substantially reduces the workload associated with data collection for different contact scenarios. In addition, an approach was proposed to identify the precise  coordinates of contact points based on the inferred pressure maps. Specifically, the nonmaximum suppression algorithm was used to locate local maxima, which were then recognized as contact positions. Details can be found in note S6.
    
    제안된 SATS 모델의 주목할 만한 장점은 단일 접촉점에서 다중 접촉점으로의 지식 전달이 제로샷으로 가능하다는 점입니다. **그림 4E와 동영상 S1**은 각각 한 점, 두 점, 세 점에 힘이 가해졌을 때 센싱 시스템이 압력 분포를 추론하는 과정을 보여줍니다. **그림 S21**에는 다중 접촉점을 감지하는 센싱 시스템의 추가적인 예시가 제시되어 있으며, 여기에는 각각 네 점, 다섯 점, 여섯 점의 접촉점을 동시에 감지하는 과정이 포함됩니다. 이러한 기능은 **다양한 접촉 시나리오에 대한 데이터 수집 작업량을 크게 줄여줍니다**. 또한, 추론된 압력 분포도를 기반으로 접촉점의 정확한 좌표를 식별하는 방법을 제안했습니다. 구체적으로, 비최대 억제 알고리즘을 사용하여 지역 최대값을 찾고, 이를 접촉 위치로 인식했습니다. 자세한 내용은 주석 S6을 참조하십시오.
    
    - Note S6. Recognition of contact points in multi-point contact scenarios
        
        제안된 센싱 시스템은 SATS 모델을 사용하여 압력 맵을 직접 생성함으로써 외부 접촉을 반영하는 풍부하고 독창적인 정보를 제공하여 로봇이 주변 환경을 더 잘 인식하도록 돕습니다. 단일 접촉의 경우, 압력 맵에서 전역 최대값을 식별하여 접촉 위치를 결정합니다. 다중 접촉의 경우, 여러 개의 지역 최대값을 검출해야 합니다. 본 연구에서는 이를 위해 두 가지 방법을 제시합니다. 일반적인 접근 방식으로는 비최대 억제(NMS) 알고리즘을, 두 지점 식별 실험에 특화된 방법으로는 K-평균 알고리즘을 사용합니다.
        
        NMS 알고리즘은 슬라이딩 윈도우 내에서 **압력 맵의 특정 위치가 지역 최대값인지 반복적으로 검사**합니다. 지역 최대값인 경우 해당 위치를 접촉점으로 기록하고, 그렇지 않은 경우 다음 위치로 진행합니다. 이러한 방식으로 **그림 S21A에 나타난 압력 맵의 접촉 위치를 그림 S21B에서와 같이 정확하게 식별**할 수 있었습니다. 또한, 평균 위치 오차가 접촉점의 수에 따라 증가하는 것을 확인할 수 있는데, 이는 여러 개의 동시 접촉을 검출하는 것이 더 어렵다는 것을 의미합니다. 접촉점의 수가 증가하고 접촉점 사이의 거리가 감소함에 따라 압력장 간의 간섭이 심해져 신호 중첩이 발생하고 결과적으로 위치 오차가 커집니다. 
        
        두 접촉점을 구분하기 위해 K-평균 알고리즘을 사용했습니다. 먼저 SATS 모델에서 추론된 압력 맵에서 평균값의 6배 미만인 값을 제거하여 필터링합니다. 이 과정은 작은 값을 제외하지만, 국소 최대값은 제외하지 않습니다. 남은 값들의 좌표를 K-평균 알고리즘에 입력하여 두 접촉 영역을 나타내는 두 개의 클러스터를 얻습니다. 각 클러스터에서 압력 맵의 최대값에 해당하는 위치를 접촉 위치로 간주합니다. 그림 S22B는 이 절차를 사용하여 계산된 결과를 보여줍니다.
        
         두 접촉점이 너무 가까우면 두 개의 최대값이 아닌 하나의 최대값이 생성될 수 있습니다. 이러한 경우 K-평균 알고리즘이 두 개의 클러스터를 식별하더라도 계산된 접촉점이 서로 가깝거나 심지어 겹칠 수 있습니다(그림 S22B, 왼쪽 하단). SATS 모델은 이러한 상황을 두 접촉점의 경우 구분할 수 없는 조건으로 간주했습니다.
        
        ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2017.png)
        
    
    [adv2124_movie_s1.mp4](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/adv2124_movie_s1.mp4)
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2018.png)
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2019.png)
    
- **To explore the capability of discriminating multiple contacts, an experiment on two-point discrimination, a crucial criterion of tactile sensation (53), was conducted**. Two points on the sensing surface were pressed while controlling the external force and interval (fig. S22A). The pressure maps inferred by the SATS model were processed  using the K-means algorithm (note S6) to identify two local maxima as the predicted contact positions. The results are shown in Fig. 4F and fig. S22B. The minimal interval the SATS model could discriminate was 8 mm (approximately the interval between adjacent sensing units) under an external force greater than 25 N. It is observed that a larger external force improves distinguishability, which is consistent with results reported for a single-point contact. By this experiment, some limitations are observed as well. The SATS model faced challenges to accurately distinguishing two closely located contacts. The reasons include three points. From the physical perspective, the pressure fields of two contacts overlap with each other, leading to hard decomposition directly at the physical level. From the SATS model’s perspective, adjacent local maps are semicoupled due to the working mechanism of the self-attention module, indicating that pressure variations in one map may influence the inferred pressure distribution in a neighboring one. From the signal overlap perspective, a single-point contact typically activates three taxels, implying that two close contacts may activate the same taxels, which can compromise the accuracy of multi-point contact inference.
    
    다중 접촉을 구별하는 능력을 탐구하기 위해 촉각 감각의 중요한 기준인 2점 식별 실험(53)을 수행했습니다. 감지 표면의 두 지점을 누르면서 외부 힘과 간격을 제어했습니다(그림 S22A). SATS 모델에서 추론된 압력 맵을 K-평균 알고리즘(주석 S6)을 사용하여 처리하여 예측된 접촉 위치로 두 개의 지역 최대값을 식별했습니다. 결과는 그림 4F와 그림 S22B에 나와 있습니다. SATS 모델이 구별할 수 있는 최소 간격은 25N 이상의 외부 힘에서 8mm(인접한 감지 장치 사이의 간격과 거의 같음)였습니다. 외부 힘이 클수록 구별력이 향상되는 것을 확인할 수 있으며, 이는 단일 지점 접촉에 대해 보고된 결과와 일치합니다. 이 실험을 통해 몇 가지 한계점도 관찰되었습니다. **SATS 모델은 서로 가까이 위치한 두 접촉을 정확하게 구별하는 데 어려움**을 겪었습니다. 그 이유는 세 가지입니다. 물리적 관점에서 **두 접촉의 압력장이 서로 겹쳐 물리적 수준에서 직접 분해하기 어렵습니다.** SATS 모델의 관점에서 볼 때, 자기주의 모듈의 작동 메커니즘으로 인해 인접한 로컬 맵들은 부분적으로 연결되어 있으며, 이는 한 맵의 압력 변화가 인접한 맵의 추론된 압력 분포에 영향을 미칠 수 있음을 의미합니다. 신호 중첩 관점에서 보면, 단일 접촉점은 일반적으로 세 개의 택셀을 활성화시키는데, 이는 두 개의 근접 접촉점이 동일한 택셀을 활성화시킬 수 있음을 의미하며, 이는 다중 접촉점 추론의 정확도를 저하시킬 수 있습니다.
    
    ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2017.png)
    

## Demonstration of SR tactile perception

![Fig. 4. Calibration of the MSR-skin.
**그림 4. MSR-skin의 보정(calibration).**
(A) 감지 표면의 대칭선을 따라 다양한 힘이 가해졌을 때 추론된 압력 맵의 단면도.
(B) 가압 위치에 따른 오차 분포. 각 위치의 막대는 해당 위치에 압력이 가해졌을 때 감지 표면 전체에서 계산된 평균 RMSE를 의미한다.
(C) (B)의 오차 분포에 대한 통계 분석.
(D) 외력에 따른 RMSE 변화.
(E) 한 점, 두 점, 세 점에 압력이 가해졌을 때 SATS 모델이 추론한 압력 분포.
(F) 두 점 식별(two-point discrimination) 실험 결과.
(G) 두 가지 윤곽 추적 응용에서 MSR-skin의 초해상도 성능. 두 윤곽선을 따라 접촉 위치를 정확히 감지한다.](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2020.png)

Fig. 4. Calibration of the MSR-skin.
**그림 4. MSR-skin의 보정(calibration).**
(A) 감지 표면의 대칭선을 따라 다양한 힘이 가해졌을 때 추론된 압력 맵의 단면도.
(B) 가압 위치에 따른 오차 분포. 각 위치의 막대는 해당 위치에 압력이 가해졌을 때 감지 표면 전체에서 계산된 평균 RMSE를 의미한다.
(C) (B)의 오차 분포에 대한 통계 분석.
(D) 외력에 따른 RMSE 변화.
(E) 한 점, 두 점, 세 점에 압력이 가해졌을 때 SATS 모델이 추론한 압력 분포.
(F) 두 점 식별(two-point discrimination) 실험 결과.
(G) 두 가지 윤곽 추적 응용에서 MSR-skin의 초해상도 성능. 두 윤곽선을 따라 접촉 위치를 정확히 감지한다.

개발된 촉각 센서는 공간적 초해상도(SR) 능력을 바탕으로 미세한 패턴을 감지하는 데 뛰어난 성능을 보인다. 그림 4G는 두 가지 윤곽 추적(contour-following) 응용에서 MSR-skin의 성능을 보여준다. 먼저 로봇 팔을 이용해 패턴의 윤곽선을 따라 각 위치를 누르도록 프로그래밍하였고, 동시에 센서 어레이의 응답과 해당 좌표를 기록하였다. 이후 센서 어레이에서 얻은 시계열 데이터를 SATS 모델에 입력하여 추론을 수행하였다. 압력 맵에서 최대값이 나타나는 위치를 추정된 접촉 위치로 간주하였으며, 평균 RMSE는 0.32 mm를 달성하였다.

주목할 점은 이 위치 오차가 검증 단계에서 보고된 평균 RMSE 0.73 mm보다 더 낮다는 것이다. 이는 평균화 전략(averaging strategy)에 기인한다. 즉, 각 가압 위치에서 SATS 모델이 여러 번 추론한 좌표들의 평균값을 계산함으로써 힘 차원(force dimension)의 시간 정보를 종합적으로 활용했기 때문이다.

![Fig. 5. Human-machine interaction applications enhanced by SR sensation.
**그림 5. 초해상도 촉각 감각으로 향상된 인간-기계 상호작용 응용.**
(A) 감지 표면 위에 소형 컴퓨터 키보드를 구현한 예시.
(B) 키보드를 활용한 타이핑 및 계산기 응용 구현.
(C) 로봇 손에 통합된 촉각 센서로 향상된 촉각 지각 및 상호작용 기능 제공.
(D) 다양한 형상의 영상화(imaging) 및 CNN 기반 분류 결과.
(E) t-SNE 알고리즘을 사용한 차원 축소 결과.
(i) 학습 데이터, (ii) 테스트 데이터.
(F) 다양한 형상 분류에 대한 혼동 행렬(confusion matrix) 결과.](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2021.png)

Fig. 5. Human-machine interaction applications enhanced by SR sensation.
**그림 5. 초해상도 촉각 감각으로 향상된 인간-기계 상호작용 응용.**
(A) 감지 표면 위에 소형 컴퓨터 키보드를 구현한 예시.
(B) 키보드를 활용한 타이핑 및 계산기 응용 구현.
(C) 로봇 손에 통합된 촉각 센서로 향상된 촉각 지각 및 상호작용 기능 제공.
(D) 다양한 형상의 영상화(imaging) 및 CNN 기반 분류 결과.
(E) t-SNE 알고리즘을 사용한 차원 축소 결과.
(i) 학습 데이터, (ii) 테스트 데이터.
(F) 다양한 형상 분류에 대한 혼동 행렬(confusion matrix) 결과.

또한 이 시스템은 높은 집적도와 맞춤형 설계가 필요한 응용에도 적합하며, 정확한 위치 인식이 요구되는 인간-기계 상호작용 작업을 수행할 수 있다. 예를 들어, 감지 표면의 작은 영역 위에 소형 키보드를 맞춤형으로 구현하였다(그림 5A). 작은 키 하나가 눌리면 SATS 모델은 가압 위치의 좌표를 정확히 추정하였다. 어떤 키가 눌렸는지는 미리 정의된 좌표 사전을 참조하여 판별하였다. 이 통합 키보드는 인간-기계 상호작용을 위한 입력 장치로 작동할 수 있으며, 로봇이 사용자의 의도를 이해하도록 돕는다. 그림 5B와 동영상 S2는 촉각 감지 시스템을 키보드로 활용한 사례를 보여준다. 이를 확장하여 계산기 응용도 설계되었으며, 동영상 S3에 제시되어 있다.

이 촉각 센서는 로봇에 통합되어 인간과 로봇 간 상호작용을 촉진하고 체화된 촉각 지각 능력을 제공할 수 있다. 그림 5C는 로봇 손에 통합된 촉각 센서를 보여준다. 이 센서를 통해 로봇 손은 외부 자극을 감지하고 접촉 유형에 따라 반응할 수 있다(동영상 S4). 이는 인간-기계 상호작용 응용의 기반이 된다.

예를 들어, SATS 모델이 제공하는 SR 능력을 활용하여 이 촉각 센서는 접촉 형상을 영상화(imaging)하는 데 사용되었다(그림 5D). 이에 비해 그림 S23에 나타난 센서 어레이의 원시 응답(raw response)은 압력 분포에 대한 제한적인 정보만 제공하였다. 주목할 점은 이러한 표면 접촉(surface contact) 감지는 SATS 모델 학습에 사용된 단일 점 접촉(single-point contact) 시나리오와 본질적으로 다르다는 것이다. 따라서 점 접촉에서 표면 접촉으로의 zero-shot 지식 전이라는 새로운 도전 과제가 발생하며, 이는 SATS 모델 성능이 완벽하지 못했던 이유가 될 수 있다. 그럼에도 불구하고, 서로 다른 응용 간 간극을 효율적인 방식으로 메우려는 중요한 연구 방향을 제시한다.

이후 생성된 접촉 이미지는 CNN 기반 분류기에 의해 정확히 인식되었으며, 학습 데이터에서 99.5%, 테스트 데이터에서 98.35%의 정확도를 달성하였다. CNN 기반 분류기는 계산 복잡도를 최소화하기 위해 소형 구조로 설계되었고, 자원 효율적인 배치와 실시간 임베디드 시스템에서의 낮은 지연 추론(샘플당 약 1 ms)을 보장하였다. 서로 다른 접촉 형상의 실시간 인식은 동영상 S5에서 확인할 수 있다.

또한 t-distributed stochastic neighbor embedding(t-SNE) 알고리즘을 사용하여 분류기 출력 공간의 차원을 축소하였다(그림 5E). 학습 데이터와 테스트 데이터 모두에서 동일 클래스 데이터는 군집화되었고, 서로 다른 클래스는 명확히 분리되었다. 테스트 데이터의 분류 성능은 혼동 행렬(confusion matrix)로 그림 5F에 제시되어 있다. 대부분의 형상은 정확히 식별되었으며, 오분류는 주로 “H” 형상과 “L” 형상 사이에서 발생하였다. 이에 대한 타당한 설명은 두 형상이 구조적으로 유사하기 때문이다. 예를 들어 H 형상은 네 개의 L 형상으로 구성된 것으로 볼 수 있다.

대형 형상뿐 아니라 소형 형상 인식도 추가적으로 조사되었으며, 자세한 내용은 note S7 및 그림 S24에 제시되어 있다. 테스트 세트에서 92.87%의 분류 정확도를 달성하였는데, 이는 대형 형상 인식보다 낮은 수치로, 더 어려운 과제임을 시사한다.

## Framework generality

- The proposed computational framework for tactile SR can be deployed into several conditions, including pressure map inference and conventional coordinate and force inference. Note S8 discussed the SATS model’s effectiveness and performance in the above conditions, leveraging simulation data (figs. S25 to S30). Besides, investigating the SATS model’s interpretability further proved its powerful capability of learning both temporal and spatial characteristics (fig. S29).
    
    본 연구에서 제안하는 촉각 SR(감각 인식)을 위한 계산 프레임워크는 압력 맵 추론, 기존 좌표 및 힘 추론을 포함한 여러 조건에 적용될 수 있습니다. 참고문헌 S8에서는 시뮬레이션 데이터를 활용하여 위의 조건에서 SATS 모델의 효과와 성능을 논의했습니다(그림 S25~S30). 또한, SATS 모델의 해석 가능성을 조사한 결과, 시간적 및 공간적 특성을 모두 학습하는 강력한 능력이 입증되었습니다(그림 S29).
    
    - Note S8. Further investigation using simulated data
        - **Inference of Pressure Map:**
            - The proposed tactile sensor, equipped with the SATS model, can directly generate pressure maps.  Furthermore, the elaborately designed SATS model enables the transfer of knowledge learned from single-point touch to multi-point touch without requiring extra training. To further evaluate the performance of the SATS model, a simulation model of the tactile sensor array, based on the EHS model, was developed, as shown in Fig. S25, A and B. The sensing surface was divided into a grid (Fig. S25C), with each point on the grid serving as a position for pressing to generate simulation data. At each position, the simulated force was applied following the pattern shown in Fig. S25D. The sensor response in the simulation model was consistent with that observed in real tests. Fig. S25E illustrates the simulated sensor response curve to pressure obtained by fitting the tested response data. Noting that 𝑦𝑥=0 ≈ 3.1 simulated the static measurement error. The ground truth pressure distribution was generated directly by the EHS model (e.g., Fig. S25F). It should be noted that only one position was pressed at a time during the simulation. All positions were pressed according to the above setup for data collection (generation).
                
                압력 맵 추론: SATS 모델이 탑재된 제안된 촉각 센서는 압력 맵을 직접 생성할 수 있습니다. 또한, 정교하게 설계된 SATS 모델은 추가 학습 없이 단일 지점 터치에서 학습된 지식을 다중 지점 터치로 전이할 수 있도록 합니다. SATS 모델의 성능을 더욱 평가하기 위해 EHS 모델을 기반으로 촉각 센서 어레이의 시뮬레이션 모델을 개발했습니다**(그림 S25, A 및 B 참조)**. 감지 표면은 격자로 나뉘었고(그림 S25C), 격자의 각 점은 시뮬레이션 데이터 생성을 위한 누름 위치로 사용되었습니다. 각 위치에서는 그림 S25D에 나타낸 패턴에 따라 시뮬레이션된 힘이 가해졌습니다. 시뮬레이션 모델에서의 센서 응답은 실제 테스트에서 관찰된 응답과 일치했습니다. 그림 S25E는 테스트 응답 데이터를 피팅하여 얻은 압력에 대한 시뮬레이션 센서 응답 곡선을 보여줍니다. $y_{x=0} ≈ 3.1$은 정적 측정 오차를 시뮬레이션한 것입니다. 실제 압력 분포는 EHS 모델에서 직접 생성되었습니다(예: 그림 S25F). 시뮬레이션 동안 한 번에 한 위치만 눌렀다는 점에 유의해야 합니다. 모든 위치는 데이터 수집(생성)을 위한 위의 설정에 따라 눌렀습니다.
                
                ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2022.png)
                
            - The SATS model was trained using the aforementioned simulation data. The RMSE for pressing each position was calculated as presented in Fig. S18, with an average value of 0.034 kPa, as shown in Fig. S26. Fig. S27 displays several examples of the SATS model’s  inferences, along with the corresponding ground truths and error distributions. It was observed that while larger pressures resulted in larger absolute errors, the relative error was maintained below 2.4% (maximum pressure to maximum pressure). Furthermore, Fig. S28 illustrates the SATS model’s performance when directly applied to multi-point touch scenarios. Although relatively larger errors were observed, the SATS model remained effective under these conditions,  reporting a relative error of less than 10% (maximum pressure to maximum pressure) in a three-point contact scenario. While obtaining ground truth pressure distribution under real-world conditions was challenging, the simulation results provided valuable insights
                
                SATS 모델은 앞서 언급한 시뮬레이션 데이터를 사용하여 학습되었습니다. 각 위치를 누를 때의 RMSE는 그림 S18에 제시된 바와 같이 계산되었으며, 그림 S26에서 볼 수 있듯이 평균값은 0.034 kPa였습니다. 그림 S27은 SATS 모델의 추론 결과 몇 가지 예시와 해당 실제 압력 분포 및 오차 분포를 보여줍니다. **압력이 클수록 절대 오차가 커지는 것을 확인할 수 있었지만, 상대 오차는 2.4% 미만(최대 압력 대 최대 압력)으로 유지되었습니다.** 또한, 그림 S28은 SATS 모델을 다중 접촉 시나리오에 직접 적용했을 때의 성능을 보여줍니다. 상대적으로 큰 오차가 관찰되었지만, SATS 모델은 이러한 조건에서도 효과적으로 작동하여 3점 접촉 시나리오에서 10% 미만의 상대 오차(최대 압력 대 최대 압력)를 나타냈습니다. 실제 환경에서 실제 압력 분포를 얻는 것은 어려웠지만, 시뮬레이션 결과는 유용한 통찰력을 제공했습니다.
                
                ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2023.png)
                
                ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2024.png)
                
            - Theoretically, the self-attention module enhances the SATS model’s ability for spatial perception. To validate the effectiveness of the self-attention module, a specific sensing unit (the orange one in Fig. S29A) and its corresponding encoded features were selected for analysis. The area surrounding this sensing unit was pressed to stimulate it, and the features encoded before and after the self-attention module were recorded, respectively. The t-SNE algorithm was used to reduce the dimensionality of these features to a two-dimensional space (Fig. S29, B and C). The features before the self-attention-based information sharing could only perceive distance and failed to distinguish the pressing positions (Fig. S29B). In contrast, the features after the self- attention module effectively decoupled spatial information. Fig. S29C shows the correspondence between data points in the two-dimensional space and pressing position, demonstrating the powerful spatial encoding capability of the self-attention module.
                
                이론적으로, 자기주의 모듈은 SATS 모델의 공간 지각 능력을 향상시킵니다. 자기주의 모듈의 효과를 검증하기 위해 특정 센싱 유닛(그림 S29A의 주황색 유닛)과 그에 해당하는 인코딩된 특징을 분석 대상으로 선정했습니다. 이 센싱 유닛 주변 영역을 눌러 자극을 가하고, 자기주의 모듈 적용 전후의 인코딩된 특징을 각각 기록했습니다. t-SNE 알고리즘을 사용하여 이러한 특징들의 차원을 2차원 공간으로 축소했습니다(그림 S29, B 및 C). 자기주의 기반 정보 공유 전의 특징은 거리만 인지할 수 있었고 누르는 위치를 구분하지 못했습니다(그림 S29B). 반면, 자기주의 모듈 적용 후의 특징은 공간 정보를 효과적으로 분리했습니다. 그림 S29C는 2차원 공간의 데이터 포인트와 누르는 위치 간의 대응 관계를 보여주며, 자기주의 모듈의 강력한 공간 인코딩 능력을 입증합니다.
                
                ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2025.png)
                
        - **Inference of Coordinates and Force:**
            - Existing studies on tactile super-resolution primarily focus on localization estimation, specifically determining the coordinates of a contact point. However, a significant limitation of these approaches is that they only function effectively in single-point contact scenarios, failing when faced with multi-point contacts. This is attributed to both the design of the model structure and training data. Without an  internal structure design, an end-to-end model outputs only coordinates and the force of an external stimulus in the form of a three-dimensional vector. This model can only handle single-point contacts after being trained with single-point contact data. Data of multi-point contact must be collected to train a new model whose output layer has been modified to accommodate multi-point contact scenarios. The proposed SATS model overcomes this limitation by effectively transferring knowledge learned from single-point contact to multi-point contact, leveraging the local receptive field enhanced by the self-attention mechanism. This capability is crucial for real-world applications where multipoint contact is prevalent. Additionally, the calibration data required could be significantly reduced since only single-point contact data is necessary. In practice, collecting data that encompasses all possible multi-point contact conditions is nearly impossible due to the curse of dimensionalities.
                
                기존의 촉각 초해상도 연구는 주로 접촉점의 좌표를 결정하는 등 접촉 위치 추정에 초점을 맞추고 있습니다. 그러나 이러한 접근 방식의 중요한 한계는 단일 접촉 시나리오에서만 효과적으로 작동하고 다중 접촉 상황에서는 제대로 작동하지 못한다는 점입니다. 이는 모델 구조 설계와 학습 데이터 모두에 기인합니다. 내부 구조 설계가 제대로 이루어지지 않은 경우, 엔드투엔드 모델은 외부 자극의 좌표와 힘만을 3차원 벡터 형태로 출력합니다. 이러한 모델은 단일 접촉 데이터로 학습된 후에는 단일 접촉만 처리할 수 있습니다. 다중 접촉 시나리오를 수용할 수 있도록 출력 레이어를 수정하고 다중 접촉 데이터를 수집하여 새로운 모델을 학습시켜야 합니다. 제안하는 SATS 모델은 셀프 어텐션 메커니즘으로 강화된 로컬 수용 영역을 활용하여 단일 접촉에서 학습한 지식을 다중 접촉으로 효과적으로 전달함으로써 이러한 한계를 극복합니다. 이러한 기능은 다중 접촉이 빈번하게 발생하는 실제 응용 분야에 매우 중요합니다. 또한, 단일 접촉 데이터만 필요하므로 필요한 보정 데이터의 양을 크게 줄일 수 있습니다. 실제로는 차원의 저주 때문에 가능한 모든 다점 접촉 조건을 포괄하는 데이터를 수집하는 것은 거의 불가능합니다.
                
            - Despite the limitations of estimating coordinates for single-point contact, this approach remains relevant to tactile super-resolution and often yields an impressive scale factor. Consequently, the feasibility of applying the proposed super-resolution framework to this task was also explored. First, the SATS model was modified by replacing the local map reconstruction module with a regression module (a three-layer MLP) to infer contact position (coordinates) and force. Subsequently, 5000 positions on the sensing surface were randomly sampled for pressing, with force varying according to the paradigm in Fig. S25D. The modified SATS model was then trained. It was observed that greater force resulted in reduced position error (Fig. S30A). The spatial distribution of position errors under different forces is shown in Fig. S30B. Position accuracy improves as the force increases, with an average error of 0.12 mm (RMSE) over the whole force range. Under an external force of 8 N, this system achieved a maximal SR scale factor of 19547, extensively surpassing the current state-of-the-art. A similar trend was observed in force inference, with an average force error of 0.035 N. With its localization capability, the system can accurately reconstruct complex and fine patterns (Fig. S30D) in contour-following applications. The results in the inference of coordinates and force further demonstrated the generality of the proposed tactile SR framework, illustrating its strong potential.
                
                단일 접촉점 좌표 추정의 한계에도 불구하고, 이 접근 방식은 촉각 초해상도에 여전히 유효하며 종종 인상적인 스케일 팩터를 제공합니다. 따라서 제안된 초해상도 프레임워크를 이 작업에 적용하는 타당성도 탐색했습니다. 먼저, SATS 모델에서 로컬 맵 재구성 모듈을 접촉 위치(좌표)와 힘을 추론하는 회귀 모듈(3계층 MLP)로 대체하여 모델을 수정했습니다. 그런 다음, 감지 표면에서 5000개의 위치를 무작위로 샘플링하여 누르는 동작을 수행했으며, 힘은 그림 S25D의 패러다임에 따라 변화시켰습니다. 수정된 SATS 모델을 학습시킨 결과, 힘이 클수록 위치 오차가 감소하는 것을 관찰했습니다(그림 S30A). 다양한 힘 조건에서의 위치 오차의 공간 분포는 그림 S30B에 나타나 있습니다. 힘이 증가함에 따라 위치 정확도가 향상되며, 전체 힘 범위에서 평균 오차는 0.12mm(RMSE)입니다. 8N의 외부 힘 하에서, 이 시스템은 최대 19547의 SR 스케일 팩터를 달성하여 현재 최첨단 기술을 크게 능가했습니다. 힘 추론에서도 유사한 경향이 관찰되었으며, 평균 힘 오차는 0.035N이었습니다. 위치 파악 기능을 통해 이 시스템은 윤곽선 추적 응용 분야에서 복잡하고 미세한 패턴을 정확하게 재구성할 수 있습니다(그림 S30D). 좌표 및 힘 추론 결과는 제안된 촉각 SR 프레임워크의 일반성을 더욱 입증하며, 그 강력한 잠재력을 보여줍니다.
                
                ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2026.png)
                
- The tactile sensor prototype used in this study is piezoresistive for pressure detection, with an elastomer used as the transmission medium to expand the receptive field. In the above investigation, the elastomer thickness was empirically set to 5 mm, demonstrating proper performance in subsequent tactile SR applications. The receptive field size of a sensing unit could be influenced by its intrinsic characteristics, signal-to-noise ratio, and elastomer thickness. Given an integrated sensing system where sensitivity and signal-to-noise ratio are predefined, the receptive field size can be adjusted by modifying the thickness of the elastic covering to achieve an optimal configuration. A detailed discussion of this aspect is provided in note S9 and fig. S31.
    
    본 연구에서 사용된 촉각 센서 프로토타입은 압력 감지를 위한 압저항 방식이며, 수용 영역을 확장하기 위해 엘라스토머를 전달 매체로 사용했습니다. 위의 조사에서 엘라스토머 두께는 5mm로 설정되었으며, 이는 이후 촉각 SR 응용 분야에서 적절한 성능을 보여주었습니다. 센싱 유닛의 수용 영역 크기는 고유 특성, 신호 대 잡음비 및 엘라스토머 두께의 영향을 받을 수 있습니다. 감도와 신호 대 잡음비가 미리 정의된 통합 감지 시스템의 경우, 탄성 커버의 두께를 조절하여 수용 영역 크기를 조정함으로써 최적의 구성을 얻을 수 있습니다. 이 측면에 대한 자세한 설명은 주석 S9 및 그림 S31에 제공됩니다.
    
    - Note S9. Investigation of the optimal receptive filed size
        - The receptive field size of a sensing unit is influenced by its intrinsic sensing properties, the signal-to-noise ratio (SNR), and the thickness of the elastic covering. Given a predefined fabrication process, the sensing properties, such as sensitivity, remain constant. The SNR is typically affected by the magnitude of the external stimulus, while the thickness of the elastic covering can be easily adjusted to modify the receptive field size. To account for environmental noise, we assume a response threshold beyond which the signal is considered valid. For instance, a threshold of 0.15 implies that only when the relative resistance change (Δ𝑅⁄𝑅0) exceeds 15% is the response deemed effective. Since the response of the sensing unit is negatively correlated with the distance from the applied force, the boundary of the receptive field is determined by the unit’s response threshold.
            
            감지 장치의 수용 영역 크기는 고유의 감지 특성, 신호 대 잡음비(SNR), 그리고 탄성 커버의 두께에 영향을 받습니다. 미리 정의된 제조 공정을 가정하면 감도와 같은 감지 특성은 일정하게 유지됩니다. SNR은 일반적으로 외부 자극의 크기에 영향을 받는 반면, 탄성 커버의 두께는 수용 영역 크기를 조절하기 위해 쉽게 조정할 수 있습니다. 환경 잡음을 고려하기 위해, 신호가 유효하다고 간주되는 응답 임계값을 설정합니다. 예를 들어, 임계값이 0.15인 경우 상대 저항 변화(Δ𝑅⁄𝑅0)가 15%를 초과할 때만 응답이 유효하다고 판단합니다. 감지 장치의 응답은 가해진 힘으로부터의 거리에 반비례하므로, 수용 영역의 경계는 장치의 응답 임계값에 의해 결정됩니다.
            
        - To explore the optimal receptive field size, we examined its dependence on the response threshold and the thickness of the elastic covering. In a specific setup where the threshold is set to 0.15, and an external force of 5 N is applied, the radius of the receptive field under different elastomer thicknesses is presented in Fig. S31A. The results indicate that the maximum radius of 11.6 mm occurs at an elastomer thickness of 5 mm. Subsequently, with the thickness fixed at 5 mm, the threshold was varied, producing the results shown in Fig. S31B. These results suggest that a lower threshold, corresponding to higher sensitivity, leads to a larger receptive field.
            
            최적의 수용장 크기를 알아보기 위해 반응 역치와 탄성 코팅 두께에 따른 수용장의 변화를 조사했습니다. 역치를 0.15로 설정하고 5N의 외부 힘을 가한 특정 조건에서, 탄성 코팅 두께에 따른 수용장 반경을 그림 S31A에 나타냈습니다. 결과에 따르면 탄성 코팅 두께가 5mm일 때 최대 반경인 11.6mm가 나타났습니다. 이어서, 두께를 5mm로 고정하고 역치를 변화시켜 얻은 결과는 그림 S31B에 제시되어 있습니다. 이러한 결과는 역치가 낮을수록(즉, 감도가 높을수록) 수용장이 커진다는 것을 시사합니다.
            
        
        ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2027.png)
        
        - In summary, the threshold and thickness exhibit a coupled influence on the receptive field size, as illustrated in Fig. S31C. Additionally, given  that external force influences the sensing unit’s response, the magnitude of the applied force is also considered in Fig. S31C. The results demonstrate that for a given threshold (primarily determined by the sensing unit’s sensitivity) and a specific force magnitude, an optimal elastomer thickness exists that maximizes the receptive field size.
            
            요약하자면, 그림 S31C에서 볼 수 있듯이 임계값과 두께는 수용 영역 크기에 상호 연관된 영향을 미칩니다. 또한, 외부 힘이 감지 장치의 반응에 영향을 미친다는 점을 고려하여 그림 S31C에서는 가해지는 힘의 크기도 함께 고려했습니다. 결과는 주어진 임계값(주로 감지 장치의 감도에 의해 결정됨)과 특정 힘의 크기에 대해 수용 영역 크기를 최대화하는 최적의 엘라스토머 두께가 존재함을 보여줍니다.
            

---

# DISCUSSION

### SR capability

본 연구는 계산 기반 촉각 감각(computational tactile sensation)의 가능성을 탐구하였으며, 딥러닝 기술을 기반으로 촉각 센서가 초해상도(SR) 압력 분포를 감지할 수 있도록 하는 프레임워크를 개발하였다. 특히 기존 감지 자원의 활용을 최적화하는 데 중점을 두었다. SATS 모델은 다채널 신호를 입력받아 압력 분포를 추정하도록 설계되었다. 이 프레임워크는 물리적 한계를 넘어 촉각 센서의 공간 해상도를 향상시켰다. 또한 하드웨어의 한계를 크게 보완하고, 촉각 센서 구조를 단순화하며, 신뢰성과 내구성을 개선하였다.

더불어 향상된 해상도는 다양한 인간-기계 상호작용 응용을 가능하게 하였고, 체화 지각(embodied perception)의 발전에도 기여하였다. 또한 제안된 프레임워크는 범용성이 높아 다양한 촉각 센서 설계에 적용될 수 있다. 수용영역(receptive field)이 가장자리 부분을 완전히 덮지 못해 감지 표면의 가장자리에서 더 큰 오차가 관찰되었지만, 이는 새로운 taxel을 추가 확장함으로써 해결할 수 있다.

또한 더욱 높은 해상도는 첨단 제조 기술을 통해 더 작고 조밀한 taxel을 제작함으로써 달성될 가능성이 있다. 이는 외부 자극의 미세한 세부 정보를 하드웨어 수준에서 더 잘 포착하게 해준다. 이후 더 높은 충실도의 데이터를 수집하고, 잠재 특징(latent feature) 차원과 추론 맵 크기를 조정한 SATS 모델을 학습시킴으로써 초해상도 촉각 감지는 추가적으로 향상될 수 있다.

### Training efficiency and scalability

SATS 모델은 단일 점 접촉(single-point contact)에서 다중 점 접촉(multi-point contact)으로의 zero-shot 지식 전이를 가능하게 한다. 학습에는 단일 점 접촉 데이터만 필요하지만, 동일한 모델이 재학습 없이도 다중 점 감지를 수행할 수 있다. 이러한 특성은 모델의 적응성을 높여 다양한 응용에 적합하게 만든다.

다만 두 개 이상의 접촉점이 서로 가까이 위치할 경우, 활성화되는 taxel이 동일해져 마치 하나의 점만 존재하는 것처럼 나타날 수 있다. 따라서 여러 지점에 동시에 압력이 가해질 때도 단일 점 접촉과 동일한 taxel이 활성화될 수 있으며, 이는 단일 점 접촉과 다중 점 접촉 사이의 도메인 차이(domain shift) 문제를 악화시킨다. 다시 말해, 다중 점 접촉이 매우 조밀하게 분포된 자극을 포함하는 경우 감지 성능이 저하될 수 있다.

이 문제는 그림 5D에 제시된 다양한 형상 영상화(imaging)와 같은 복잡한 접촉 상황에서 더욱 심화될 수 있으며, 이는 점 접촉에서 표면 접촉(surface contact)으로의 지식 전이 문제를 의미한다. 비록 이러한 도전 과제로 인해 압력 추론 기반 영상화 성능이 완벽하지는 않았지만, 소프트웨어와 하드웨어 양 측면에서 개선이 가능하다.

소프트웨어 측면에서는 다중 점 접촉 및 표면 접촉 감지를 위한 추가 데이터를 수집하여 SATS 모델을 미세 조정(fine-tuning)할 수 있다. 하드웨어 측면에서는 더 작고 조밀한 taxel을 사용하여 센서 어레이의 물리적 해상도를 높임으로써 더 높은 품질의 원시 데이터를 제공할 수 있다.

또한 시스템은 확장성이 높다. 제안된 배열 방식에 따라 새로운 감지 유닛을 추가할 경우, 해당 유닛을 통합하기 위해 새로운 LSTM 층만 추가하면 되며, self-attention, 로컬 맵 구성, CNN 층은 변경 없이 그대로 유지할 수 있다.

---

# Method

## Fabrication of the porous CNT/PDMS sponge

다공성 CNT/PDMS 스펀지는 Ni 폼을 다공성 템플릿으로 사용하는 표준 산 에칭 공정을 통해 제작되었다. 먼저 CNT(XFM19, XFNANO, 중국)와 클로로포름(10006862, Sinopharm Chemical Reagent Co. Ltd., 중국)을 중량비 1:20으로 혼합하였다. 이 혼합물은 초음파 세포 파쇄 시스템(LC-JY96-IIN, LICHEN, 중국)을 사용하여 출력 90 W에서 15분간 초음파 분산 처리하였다.

이후 PDMS 전구체(Sylgard 184, Dow Corning, 미국)와 경화제를 10:1 비율로 혼합한 뒤, 클로로포름을 추가하여 중량비 1:3으로 희석하였다. 희석된 PDMS 용액을 CNT/클로로포름 혼합물에 첨가하고, 실온에서 약 10분 동안 650 rpm 속도로 교반하였다. 생성된 혼합물은 이후 80°C에서 650 rpm으로 가열·교반하여, 흄 후드 내에서 클로로포름의 약 60%를 증발시켰다.

다음으로 Ni 폼(KSH-1011, Willtek Photoelectric Materials Co. Ltd., 중국)은 두께 2 mm의 시트를 사용하였으며, 적외선 레이저 시스템(파장: 10.6 μm; VLS 2.50, UNIVERSAL Laser System Inc.)을 이용하여 직경 10 mm의 원형으로 절단하였다. 절단된 Ni 폼은 CNT/클로로포름/PDMS 혼합물에 30초간 담근 뒤, 체 위에서 흔들어 불균일하거나 응집된 부분을 제거하였다.

이후 90°C에서 30분간 베이킹한 후, 경화된 시료를 3 M 염산(HCl) 용액에 20시간 동안 침지하여 Ni 폼을 에칭 제거하였다. 마지막으로 에칭된 시료를 증류수로 여러 차례 세척하여 다공성 CNT/PDMS 스펀지를 얻었다.

## Fabrication of the sensor array

센서 어레이는 층별 적층 방식(layer by layer)으로 제작되었다. 센서 어레이의 전극층은 중국 JLC Technology Group Co. Ltd.에서 맞춤 제작한 유연 인쇄회로기판(flexible printed circuit board, FPCB)을 사용하였다. 이후 두께 0.12 mm의 PET 층(중국 Dongguan Hongshun Adhesive Products Co. Ltd.)을 하부 전극층 위에 부착한 뒤, 레이저 가공기(FD-300, Fengying Computer Technology Development Co. Ltd., 중국)를 이용해 전극이 노출되도록 절단하였다.

이 가공된 PET 층은 은 페이스트(GC-SP360-A, Shandong Jiahui Material Technology Co. Ltd., 중국)를 블레이드 코팅(blade coating)하기 위한 마스크 역할을 하였다. 준비된 감지 유닛들은 하부 전극 위의 은 페이스트에 직접 부착되었다. 또한 센서 어레이 가장자리에는 감지 유닛과 동일한 재료로 제작된 스페이서를 배치하여, 균일한 기계적 특성을 유지하도록 하였다.

상부 전극층 역시 하부 전극층과 동일한 공정으로 제작한 후 감지 유닛 상부에 배치하였다. 은 페이스트를 경화시키기 위해, 완성된 통합 센서 어레이를 가열 스테이지(X1010T, Xinhaomai Electronic Technology Co. Ltd., Shenzhen, 중국) 위에서 열처리하였다.

마지막으로 Dragon Skin 10(Smooth-On, 미국) 소재의 두께 5 mm 탄성체(elastomer)를 센서 어레이 상부에 배치하였다. 이 탄성체는 보호 커버 역할과 동시에 힘 전달 매체(force transmission medium)로 기능하였다.

## Characterization

주사전자현미경(SEM; Gemini 300, Zeiss, 독일)을 사용하여 다공성 CNT/PDMS 스펀지의 표면 형태를 관찰하였다. 감지 재료의 기계적 특성은 만능 재료 시험기(universal testing machine; 34SC-05, Instron, 미국)를 사용하여 측정하였다. 이동 플랫폼(LY100-C, Jevid Hydraulic and Pneumatic Machinery Factory, 중국)은 수용영역(receptive field) 측정에 사용되었다.

특성 평가 과정 동안 데이터 수집 모듈의 프로세서로 ATmega328P 마이크로컨트롤러(Microchip Technology Inc., 미국)를 사용하였다. 저항값은 전압 분배기(voltage divider) 방식을 통해 측정하였으며, 샘플링 속도는 500 Hz로 설정하였다.

## Signal acquisition of sensor array

- For the 23-node tactile sensor prototype, a customized circuit board was designed and manufactured. Using the four-wire resistance measurement method, the STM32F microcontroller from STMicroelectronics (Switzerland) was used for resistance measurement and data acquisition. Each sensing unit was connected to a top electrode for current input and a bottom electrode for current output, resulting in 46 channels (23 × 2). Consequently, two multiplexers (CD74HC4067, Texas Instruments, USA) were used to control 23 channels. The schematic circuit diagram is shown in fig. S6. The sampling rate for data acquisition was set at 10 Hz.
    
    23개 노드로 구성된 촉각 센서 프로토타입을 위해 맞춤형 회로 기판을 설계 및 제작했습니다. 4선식 저항 측정 방식을 사용하여 STMicroelectronics(스위스)사의 STM32F 마이크로컨트롤러로 저항을 측정하고 데이터를 수집했습니다. 각 센싱 유닛은 전류 입력을 위한 상단 전극과 전류 출력을 위한 하단 전극에 연결되어 총 46개 채널(23 × 2)을 구성했습니다. 따라서 23개 채널을 제어하기 위해 두 개의 멀티플렉서(CD74HC4067, Texas Instruments, 미국)를 사용했습니다. 회로도는 그림 S6에 나타냈습니다. 데이터 수집 샘플링 속도는 10Hz로 설정했습니다.
    

## Data acquisition for DNN model training

- To collect data for calibrating the sensor array and training the SATS model, a robot arm (UR5, Universal Robots, Denmark) equipped with a six-dimensional force sensor (Gamma IP60, ATI Industrial Automation, USA) was programmed to press each position on the sensing surface. With force feedback, the robot arm was controlled at each position to gradually increase the displacement along the normal direction of the sensing surface until a force of 10 N was reached. The robot arm was then released and moved to the next position. During this process, the coordinates of the robot arm’s end effector and the force sensor’s measured force were recorded using the rosbag tool. In addition, the responses from the sensor array were recorded, resulting in 23-channel time-series data. These two data types were aligned along the time axis using recorded timestamps and segmented using a sliding window of size 10 sample points (aligned with the LSTM’s inputs). Data from the robot arm generated ground truths, while data from the sensor array served as inputs to the SATS model.
    
    센서 어레이 교정 및 SATS 모델 학습을 위한 데이터 수집을 위해, 6차원 힘 센서(Gamma IP60, ATI Industrial Automation, USA)가 장착된 로봇 팔(UR5, Universal Robots, 덴마크)을 사용하여 센싱 표면의 각 위치를 누르도록 프로그래밍했습니다. 힘 피드백을 통해 로봇 팔은 각 위치에서 센싱 표면의 법선 방향을 따라 변위를 점진적으로 증가시켜 10N의 힘이 가해질 때까지 제어했습니다. 그 후 로봇 팔을 놓아 다음 위치로 이동시켰습니다. 이 과정에서 rosbag 툴을 사용하여 로봇 팔 끝단의 좌표와 힘 센서에서 측정된 힘을 기록했습니다. 또한 센서 어레이의 응답도 기록하여 23채널 시계열 데이터를 얻었습니다. 이 두 가지 데이터 유형은 기록된 타임스탬프를 사용하여 시간 축을 따라 정렬하고, LSTM 입력에 맞춰 10개의 샘플 포인트 크기의 슬라이딩 윈도우를 사용하여 분할했습니다. 로봇 팔에서 얻은 데이터는 정답 데이터(ground truth)로 사용되었고, 센서 어레이에서 얻은 데이터는 SATS 모델의 입력으로 사용되었습니다.
    
- Data were collected in the large-scale shape recognition application by randomly placing three-dimensional printed models of various shapes on the sensor array and applying random normal force on these models. This process was completed by volunteers rather than a standard compression testing machine to ensure the diversity of data distribution. Approximately 4000 samples were collected for each shape, with 90% used for training and 10% for validation. The sensor array’s responses were recorded and processed by the SATS model to infer pressure maps, which were then used as inputs to the CNN-based shape classifier.
    
    대규모 형상 인식 응용 프로그램에서는 다양한 형상의 3D 프린팅 모델을 센서 어레이 위에 무작위로 배치하고 이 모델들에 무작위적인 수직력을 가하여 데이터를 수집했습니다. 데이터 분포의 다양성을 확보하기 위해 표준 압축 시험기 대신 자원봉사자들이 이 과정을 수행했습니다. 각 형상에 대해 약 4000개의 샘플을 수집했으며, 이 중 90%는 학습에, 10%는 검증에 사용했습니다. 센서 어레이의 응답을 기록하고 SATS 모델을 통해 처리하여 압력 맵을 도출했으며, 이 압력 맵은 CNN 기반 형상 분류기의 입력으로 사용되었습니다.
    

## The DNN models

- In the SATS model, a one-layer LSTM with a hidden size of 125 was
used to encode the time-series signals with a window size of 10. Each sensing unit was assigned a unique LSTM to accommodate its unique characteristics. The self-attention module was implemented on the basis of the graph attention model (GAT) (54), with two GAT layers, each having a hidden size of 125. Before input into the self- attention module, features from the LSTMs of all sensing units were structured as a graph, of which the vertices were encoded features, and an adjacent matrix defined the edges. For each sensing unit,  features from the LSTM model and the self-attention module were concatenated and input into the local map construction module, a three-layer MLP (250 by 375 by 500 by 195). Each layer of the MLP was followed by a LeakyReLU (55) activation function. The output from the MLP was resized into a two-dimensional matrix of size 13 by 15 to form the local pressure map. These local maps were then masked and merged according to their positions (the locations of the corresponding sensing units) to construct the overall pressure map. Last, the overall map was processed by a two-layer CNN to obtain the final inference of the pressure distribution, with each layer of the CNN using a convolutional kernel of size 3 by 3 and the first layer followed by a LeakyReLU activation function. The constructed overall pressure map is formulated as a matrix of shape 54 by 50, of which each value is regarded as a virtual taxel, contributing to 2700 virtual taxels in total. The SATS model was trained using the adaptive moment estimation (Adam) optimizer with a learning rate of 0.0064 and a batch size of 2048 for 200 epochs. The mean square error was used as the loss function.
    
    SATS 모델에서는 은닉 크기가 125인 단일 레이어 LSTM을 사용하여 윈도우 크기 10으로 시계열 신호를 인코딩했습니다. 각 센싱 유닛에는 고유한 특성을 수용하기 위해 고유한 LSTM이 할당되었습니다. 셀프 어텐션 모듈은 그래프 어텐션 모델(GAT)(54)을 기반으로 구현되었으며, 각각 은닉 크기가 125인 두 개의 GAT 레이어로 구성되었습니다. 셀프 어텐션 모듈에 입력하기 전에 모든 센싱 유닛의 LSTM에서 추출한 특징들을 그래프로 구성했는데, 그래프의 정점은 인코딩된 특징이고 인접 행렬은 간선을 정의했습니다. 각 센싱 유닛에 대해 LSTM 모델과 셀프 어텐션 모듈에서 추출한 특징들을 연결하여 3개 레이어로 구성된 MLP(250 x 375 x 500 x 195)인 로컬 맵 생성 모듈에 입력했습니다. MLP의 각 레이어에는 LeakyReLU(55) 활성화 함수가 적용되었습니다. MLP의 출력은 13x15 크기의 2차원 행렬로 변환되어 지역 압력 지도를 생성했습니다. 이 지역 지도들은 위치(해당 센싱 유닛의 위치)에 따라 마스킹 및 병합되어 전체 압력 지도를 구성했습니다. 마지막으로, 전체 지도는 2층 CNN을 통해 처리되어 최종 압력 분포 추론 결과를 얻었습니다. CNN의 각 층은 3x3 크기의 컨볼루션 커널을 사용했으며, 첫 번째 층에는 LeakyReLU 활성화 함수를 적용했습니다. 구성된 전체 압력 지도는 54x50 크기의 행렬로 표현되며, 각 값은 가상 택셀로 간주되어 총 2700개의 가상 택셀이 사용되었습니다. SATS 모델은 Adam(Adaptive Moment Estimation) 최적화기를 사용하여 학습률 0.0064, 배치 크기 2048, 200 에포크 동안 학습했습니다. 손실 함수로는 평균 제곱 오차(MSE)를 사용했습니다.
    
- In shape recognition, since the outputs from the SATS model were
two-dimensional matrices representing the pressure distributions induced by different shapes, a CNN-based classifier was developed to recognize these shapes. This classifier included three convolutional layers, with output channels of 16, 32, and 64, respectively, and a kernel size of 3 by 3. A max pooling layer with a kernel size of 2 by 2 followed each convolutional layer. The outputs from the last max pooling layer were flattened and fed into an MLP (256 by 64 by 5) for classification. All convolutional layers and each layer of the MLP, except for the final layer, were followed by the LeakyReLU activation function. The model was trained using the Adam optimizer with a learning rate of 0.001. Cross-entropy loss was used as the loss function.
    
    형상 인식에서 SATS 모델의 출력은 다양한 형상에 의해 유도되는 압력 분포를 나타내는 2차원 행렬이었기 때문에, 이러한 형상을 인식하기 위해 CNN 기반 분류기를 개발했습니다. 이 분류기는 출력 채널이 각각 16, 32, 64개이고 커널 크기가 3x3인 세 개의 컨볼루션 레이어로 구성되었습니다. 각 컨볼루션 레이어 뒤에는 커널 크기가 2x2인 맥스 풀링 레이어가 이어졌습니다. 마지막 맥스 풀링 레이어의 출력은 평탄화되어 분류를 위해 MLP(256x64x5)에 입력되었습니다. 모든 컨볼루션 레이어와 MLP의 각 레이어(마지막 레이어 제외) 뒤에는 LeakyReLU 활성화 함수가 적용되었습니다. 모델은 Adam 옵티마이저를 사용하여 학습률 0.001로 학습되었습니다. 손실 함수로는 교차 엔트로피 손실이 사용되었습니다.
    
- The models mentioned above were implemented in Python 3.11.5 using the PyTorch framework (version 2.1.1). Training was conducted on an Nvidia GeForce RTX 4090 graphics processing unit under Ubuntu 20.04. The time cost of training the SATS model for one epoch was about 16 s, leading to about 53 min for 200 epochs.
    
    위에서 언급한 모델들은 PyTorch 프레임워크(버전 2.1.1)를 사용하여 Python 3.11.5로 구현되었습니다. 학습은 Ubuntu 20.04 운영 체제에서 Nvidia GeForce RTX 4090 그래픽 처리 장치를 사용하여 수행되었습니다. SATS 모델의 1 에포크 학습에 소요된 시간은 약 16초였으며, 총 200 에포크 학습에는 약 53분이 소요되었습니다.
    

## Finite element analysis

엘라스토머는 초탄성 재료이므로 Mooney-Rivlin 모델을 사용하여 그 특성을 시뮬레이션했습니다. 밀도는 1 × 10⁻³ g/cm³로 설정했습니다. Mooney-Rivlin 모델의 매개변수는 C₁₀ = 0.144, C₁₀ = 0.036, D₁ = 0으로 설정했습니다. 시뮬레이션에서는 직경 100mm, 두께 5mm의 원통형 물체를 생성했습니다. 이 물체의 윗면 중앙에 직경 10mm의 원형 영역에 균일 하중을 가했습니다. 그런 다음 바닥면의 압력 분포를 기록했습니다.

---

# Supplementary

## Note S1. Calculation of the SR scale factor

Super-resolution 알고리즘을 활용하면, 물리적 taxel 사이의 영역에서도 virtual taxel을 생성함으로써 외부 자극을 감지할 수 있다. Super-resolution scale factor는 virtual taxel 수와 physical taxel 수의 비율로 정의된다. 생성된 virtual taxel의 밀도(개수)를 직접 보고한 방법들의 경우, scale factor는 다음과 같이 계산된다.

$$
\alpha=\frac{N_v}{N_r}
$$

여기서 $N_r$는 실제(real) taxel의 개수이고, $N_v$는 virtual taxel의 개수이다.

virtual taxel 수를 직접 제공하지 않고 대신 **localization error를 보고한 방법들의 경우,** scale factor는 다음과 같이 계산된다.

$$
\alpha=\frac{N_v}{N_r}=\frac{S}{N_r\pi \epsilon^2}
$$

여기서 S는 센서 배열의 sensing area를 나타내며, $\epsilon$은 root-mean-square error (RMSE) 형태의 localization error이다.

## Note S2. Layout optimization of general conditions

![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2028.png)

식 (1)에서는 PSO 알고리즘을 사용하여 센서 배열의 최적 배치를 탐색하기 위한 최적화 문제가 구성되었다. 그러나 PSO만이 유일한 접근법은 아니며, Bayesian optimization (BO), genetic algorithm (GA), simulated annealing (SA)과 같은 다른 일반적인 최적화 알고리즘들도 이 문제를 해결하는 데 사용할 수 있다. 본 연구에서는 PSO, BO, GA, SA를 포함한 여러 방법을 비교하였다. 단순화를 위해 세 개 receptive field의 반지름은 모두 (r=1)로 설정하였다. 각 알고리즘은 식 (1)에 정의된 문제를 푸는 최적화기로 사용되었다. 수렴 곡선은 Fig. S2A에 제시되어 있다.

몇 차례 반복 이후, 모든 알고리즘은 동일한 결과로 수렴하였으며, 세 개의 taxel은 한 변의 길이가 receptive field 반지름 $r$와 같은 **정삼각형의 꼭짓점에 배치**되었다(Fig. S2B). 비교된 방법들 중 PSO와 SA가 가장 우수한 성능을 보였으며, 가장 빠른 수렴 속도와 가장 큰 union area를 나타냈다. 최종적으로 SA보다 더 나은 성능을 보인 PSO가 최적화 알고리즘으로 선택되었다.

현재 설정에서는 세 개의 sensing unit을 배열 배치 최적화를 위한 기본 단위로 간주하며, 세 receptive field의 반지름이 동일하다고 가정하였다(즉, 반지름 비율 = 1:1:1). 이 최적화 접근법은 receptive field의 반지름이 서로 다른 조건에도 확장될 수 있다. 이를 확인하기 위해 다음 네 가지 반지름 비율 조건을 조사하였다.

- 1.5:1:1
- 1.5:1.5:1
- 1.5:1.2:1
- 1.5:1.3:1.1

![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2029.png)

최적화 과정에는 PSO 알고리즘이 사용되었다. 결과는 Fig. S3에 제시되어 있다. 각 조건에서 최적화 과정은 특정 배치로 수렴하였다. 그러나 특히 작은 반지름을 가진 receptive field의 경우 일부 sensing resource가 비효율적으로 활용되는 현상이 관찰되었다. 이를 정량화하기 위해 tactile super-resolution 맥락에서 sensing resource utilization rate를 다음과 같이 정의하였다.

$$
\alpha=\frac{A_1\cup A_2\cup A_3}{A_1+A_2+A_3}
$$

이 식은 receptive field 간 중첩 조건을 만족하면서 최적화 후 형성된 union area를 전체 receptive field 면적 합으로 나눈 비율을 의미한다. tactile super-resolution의 목표는 가능한 적은 sensing unit으로 최대 면적을 커버하는 것이며, 이는 resource utilization rate로 효과적으로 표현될 수 있다.

반지름 비율이 각각 다음과 같을 때 계산된 utilization rate는 아래와 같다.

- 1:1:1 → (\alpha=68.4%)
- 1.5:1:1 → (\alpha=58.7%)
- 1.5:1.5:1 → (\alpha=66.9%)
- 1.5:1.2:1 → (\alpha=62.6%)
- 1.5:1.3:1.1 → (\alpha=63.6%)

**따라서 sensing unit 간 균일성을 유지하여 receptive field 크기를 동일하게 맞추는 것(반지름 비율 = 1:1:1)이 resource utilization rate 향상에 유리한 것으로 권장된다.**

## Note S3. Formulation of the self-attention module

Self-attention 모듈은 현재 taxel에 인접한 taxel들로부터 정보를 집계하도록 설계되었다. 서로 일직선상에 있지 않은 여러 sensing unit으로부터의 다중 정보를 결합함으로써, 공간 정보(예: 눌린 위치)를 추론할 수 있다.

LSTM 모듈이 데이터를 인코딩한 후, 특정 sensing unit $i$의 local feature를 $h_i$라 한다. sensing unit $i$에 인접한 unit들의 집합은 $\mathcal{N}_i$로 나타낸다. 먼저, unit $i$와 $\mathcal{N}_i$에 속한 각 인접 unit 사이의 attention coefficient는 다음과 같이 계산된다.

$$
e_{ij}=a([Wh_i \parallel Wh_j]),\ \forall j\in\mathcal{N}_i \tag{S4}
$$

여기서 $W$는 선형 매핑을 위한 행렬이고, 기호 $\parallel$는 **concatenation 연산**을 나타내며, $a$는 결합된 feature를 실수값으로 변환하는 함수이다. $e_{ij}$는 계산된 **attention coefficient**로서, unit $j$의 feature가 unit $i$에 대해 **가지는 중요도**를 의미한다.

그 다음, 모든 $j$에 대해 softmax 함수를 사용하여 정규화 연산을 수행한다 (56).

$$
\alpha_{ij}=\frac{\exp(\mathrm{LeakyReLU}(e_{ij}))}{\sum_{k\in\mathcal{N}i}\exp(\mathrm{LeakyReLU}(e_{ik}))} \tag{S5}
$$

정규화된 attention coefficient를 얻은 후, 다음과 같이 feature들의 선형 결합을 계산한다.

$$
h'_i=\omega\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}Wh_j\right) \tag{S6}
$$

여기서 $\omega$는 **비선형 활성화 함수**이며, exponential linear unit(ELU) (57)이다. $h'_i$는 unit $i$의 집계 feature로 사용된다. 이 전체 과정은 Figure 2E에 나타난 바와 같이 $f(\cdot)$로 표현된다.

## Note S4. The elastic half-space model and its rectification for ground truth generation

![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2030.png)

Elastic half-space(EHS)는 엘라스토머로 채워진 반무한 공간(semi-infinite space)을 설명하는 모델이다(Fig. S14A). 이 공간 내부의 한 점에 압력이 가해지면, 엘라스토머 내부 다른 점에서의 응력은 두 점 사이 거리의 역수에 비례한다. Boussinesq 해에 따르면, 엘라스토머 내부 위치 ((x,y,z))에서의 수직 응력(normal stress) (\sigma_{zz})는 다음과 같이 표현된다 (58).

$$
\sigma_{zz}=-\frac{3Fz^3}{2\pi r^5} \tag{S7}
$$

여기서 $r=\sqrt{x^2+y^2+z^2}$ 이며, (F)는 Z축 방향의 수직 힘(normal force)을 의미한다.

위 식은 점 접촉(point contact)에 대한 식이므로, 이를 이용하여 면 접촉(surface contact)을 근사하기 위해 접촉 면을 (N)개의 점 접촉으로 이산화(discretize)할 수 있다. 그러면 면 접촉에 대한 근사 압력 분포 $\tilde{\sigma}_{zz}$는 다음과 같이 계산된다.

$$
\tilde{\sigma}{zz}=-\sum_{i=1}^{N}\frac{3F_i z^3}{2\pi r_i^5} \tag{S8}
$$

이산화 과정에서 발생할 수 있는 오차를 보정하기 위해, 보정 계수(rectification factor) $\beta$를 다음과 같이 도입한다.

$$
\bar{\sigma}_{zz}=\tilde{\sigma}_{zz}\beta(p)=-\sum_{i=1}^{N}\frac{3F_i z^3}{2\pi r_i^5}\beta(p)\approx \sigma_{zz} \tag{S9}
$$

여기서 $p$는 접촉 면 전체에 외력이 가해질 때 발생하는 평균 압력을 의미한다. $\beta$와 $p$ 사이의 관계는 유한요소해석(Finite Element Module)을 이용하여 구축되었다(Fig. S16).

![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2031.png)

## Note S5. Comparison results of interpolation methods and ablation experiments

SATS 모델 학습을 위해 수집된 데이터는 무작위로 **학습 세트(training set)** 와 **테스트 세트(test set)** 로 **0.85:0.15 비율**로 분할되었다. 학습 세트는 SATS 모델 학습에 사용되었다. 촉각 센서와 SATS 모델로 구성된 sensing system의 성능을 종합적으로 평가하기 위해, 사전 학습된 SATS 모델의 성능을 학습 세트, 테스트 세트, 그리고 두 세트를 합친 union set에서 평가하였다. Fig. 4(B~D) 및 Fig. S18은 union set에서 얻어진 결과를 제시한다.

![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2032.png)

학습 세트, 테스트 세트, union set에서의 평균 RMSE는 각각 다음과 같다.

- Training set: 0.113 kPa
- Test set: 0.13 kPa
- Union set: 0.116 kPa

또한 SATS 모델의 성능은 여러 변형 모델(variants) 및 보간(interpolation) 방법들과 비교되었다.

LSTM, self-attention module, CNN module을 각각 제거하여 다음 세 가지 변형 모델을 구성하였다.

- SATS-noLSTM
- SATS-noAttention
- SATS-noCNN

추가로, local map construction module을 직접 전체 압력 맵을 추정하는 end-to-end overall map construction module로 대체하여 **SATS-overall**이라는 또 다른 변형 모델을 구성하였다.

보간 방법으로는 다음 네 가지가 사용되었다.

- Linear interpolation
- Quadratic interpolation
- Cubic interpolation
- Gaussian interpolation

이들 보간법은 **23개 taxel의 응답값을 기반으로 압력 분포를 추정**하였다. 각 t**axel에서 계산된 저항 변화량**은 **먼저 Fig. S25E의 fitting curve를 이용하여 압력으로 변환**되었으며, **이후 sensing surface 전체의 압력 분포가 보간법으로 계산**되었다.

![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2033.png)

위 방법들은 각각 training set, test set, union set에서 평가되었으며, 결과는 Table S2에 정리되어 있다.

![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2034.png)

학습 기반 방법들과 비교할 때, 보간법들은 더 큰 오차를 보여 상대적으로 성능이 낮았다. 그 이유는 다음과 같다.

- 보간법은 일반적으로 단순 수학 함수에 의존하여 각 sensing unit의 응답 특성을 근사함
- 각 unit의 개별 특성에 적응하는 학습 과정이 없음
- 서로 다른 taxel 간 편차를 충분히 반영하지 못함
- 인접 taxel 간 협력 관계(cooperation)를 고려하지 못함

이 한계는 Fig. 19의 error distribution에서 명확히 나타난다. 모든 보간법은 sensing surface의 **좌측 가장자리(left edge)** 에서 큰 추론 오차를 보였다.

가능성 높은 설명은 해당 영역의 sensing unit들이 Fig. S25E의 calibration 결과와 잘 맞지 않는 독특한 응답 특성을 가지기 때문이다. 결과적으로, 이러한 보간법은 원래 센서 응답값을 효과적으로 활용하여 압력 분포를 추론하지 못하였다.

반면 SATS 변형 모델들은 훨씬 작은 오차를 보이며 우수한 성능을 나타냈다. 그중 **SATS-noAttention**이 가장 낮은 성능을 보여, **self-attention module의 중요성**을 강조하였다. 다른 변형 모델들 역시 원본 SATS 모델보다 성능이 낮았으며, 이는 SATS 모델 내 각 모듈이 모두 필수적임을 보여준다.

## Note S6. Recognition of contact points in multi-point contact scenarios

**Note S6. 다점 접촉 상황에서 접촉점 인식**

제안된 센싱 시스템은 SATS 모델을 사용하여 압력 맵(pressure map)을 직접 생성하며, 이는 외부 접촉을 반영하는 풍부하고 원시적인 정보를 제공하여 로봇이 주변 환경을 더 잘 인지하도록 돕는다. 단일 점 접촉(single-point contact)의 경우, 압력 맵에서 전역 최대값(global maximum)을 찾음으로써 접촉 위치를 결정한다. 다점 접촉(multi-point contact)의 경우에는 여러 개의 국소 최대값(local maxima)을 검출해야 한다. 본 연구에서는 이를 위해 두 가지 방법을 제시한다. 하나는 일반적인 접근 방식으로서 비최대 억제(non-maximum suppression, NMS) 알고리즘이고, 다른 하나는 두 점 식별(two-point discrimination) 실험에 특화된 K-means 알고리즘이다.

![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2035.png)

NMS 알고리즘은 압력 맵 내의 특정 위치가 슬라이딩 윈도우(sliding window) 내에서 국소 최대값인지 반복적으로 검사한다. 만약 그렇다면 해당 위치를 접촉점으로 기록하고, 그렇지 않다면 다음 위치로 이동한다. 이 과정을 통해 Fig. S21A에 제시된 압력 맵의 접촉 위치를 정확하게 식별할 수 있었으며, 그 결과는 Fig. S21B에 나타나 있다. 또한 접촉점 수가 증가할수록 평균 위치 오차가 커지는 것이 관찰되었는데, 이는 여러 접촉점을 동시에 검출하는 일이 더 어렵다는 것을 의미한다. 접촉점 수가 증가하고 접촉점 간 거리가 가까워질수록 각각의 압력장이 서로 간섭하게 되며, 이로 인해 신호 중첩(signal overlap)이 발생하여 위치 오차가 증가한다.

K-means 알고리즘은 두 점 식별 실험에 사용되었다. 먼저 SATS 모델이 추론한 압력 맵에서 평균값의 6배보다 작은 값들을 제거하여 필터링한다. 이 절차는 작은 값들을 제거하지만 국소 최대값은 제거하지 않는다. 이후 남은 값들의 좌표를 K-means 알고리즘에 입력하여 두 개의 클러스터를 얻는데, 이는 두 개의 접촉 영역을 나타낸다. 각 클러스터에서 압력 맵 값이 최대인 위치를 접촉 위치로 간주한다. Fig. S22B는 이 절차를 통해 계산된 결과를 보여준다.

![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2036.png)

두 접촉점이 지나치게 가까운 경우, 두 개의 최대값 대신 하나의 전체 최대값만 형성될 수 있다. 이러한 경우 K-means 알고리즘이 두 개의 클러스터를 식별하더라도 계산된 접촉점들은 서로 매우 가깝거나 심지어 겹칠 수 있다(Fig. S22B 좌하단). SATS 모델은 이러한 경우를 두 점 접촉에서 **구별 불가능한(non-distinguishable)** 상태로 판단하였다.

## Note S7. Recognition of small-scale shapes

**Note S7. 소규모 형상의 인식**

![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2037.png)

다중 스케일 촉각 지각 능력을 추가로 평가하기 위해, 소규모 형상 인식 실험을 수행하였다. 추가적으로 마름모(rhombus), 삼각형(triangle), 정사각형(square), 타원(ellipse), 오각별(pentagram)의 다섯 가지 소형 형상을 사용하여 센싱 표면에 압력을 가하였다. 이 형상들은 택셀(taxel) 간 거리보다 작도록 설계되었으며, 외접원의 지름은 10 mm였다(Fig. S24A). 센싱 어레이의 기하학적 구조를 고려하여 센싱 표면 위 여섯 개의 위치를 선택하였다. 구체적으로는 택셀 바로 위의 위치와 세 개의 택셀 사이의 위치를 선택하여 수집 데이터의 다양성을 확보하였다. 압축 시험기를 이용해 힘을 0 N에서 10 N까지 점진적으로 가하면서 동시에 센싱 어레이의 데이터를 기록하였다.

Fig. S24B(첫 번째 행)는 SATS 모델이 추론한 압력 맵을 보여주며, 시각적으로는 형상 간 구별이 어렵다. 그러나 이 맵들에 Laplace 및 Gaussian 합성곱 커널(convolution kernels)을 적용한 후에는 형상 간 차이가 더욱 뚜렷하게 나타났으며, 이는 Fig. S24B(두 번째 행)에 제시되어 있다. 마름모, 삼각형, 타원과 같은 단순한 형상에서는 모서리나 꼭짓점과 같은 일부 기하학적 특징을 관찰할 수 있었다. 반면 오각별과 같이 더 복잡한 형상에서는 SATS 모델이 다섯 개의 돌출부(horns)와 같은 뚜렷한 국소 특징을 포착하지 못했다. 대형 형상은 넓게 분포된 택셀들을 자극하는 반면, 소규모 형상은 주로 인접한 택셀만 활성화시키며 특히 택셀 간 거리보다 작은 경우 이러한 경향이 강하다. 이는 SATS 모델이 형상을 정확히 재구성하는 데 큰 어려움을 초래한다. 실제로 이는 대부분의 촉각 센싱 시스템에서도 어려운 과제이다. 또한 이러한 소형 형상의 일부 국소 특징은 가상 택셀(virtual taxels)의 해상도(1 mm)보다도 작아 정밀한 형상 복원을 더욱 어렵게 만든다.

각 형상마다 4,000개의 샘플을 수집하여 총 20,000개의 샘플을 확보하였으며, 이 중 85%는 학습용(training), 15%는 테스트용(testing)으로 무작위 분할하였다. 이후 이 데이터들을 사전 학습된 SATS 모델에 입력하여 압력 맵 특징(feature)을 추출하였다. 이어서 학습 데이터를 사용해 Fig. 5D에 제시된 것과 동일한 CNN 기반 분류기(classifier)를 학습시켰고, 이를 테스트 데이터에 대해 평가하였다. Fig. S24C는 분류기 출력 공간에 대한 차원 축소 결과를 보여주며, 대부분의 데이터 포인트가 각 클래스별로 잘 군집화되었음을 나타낸다. 동시에 일부는 오분류되어 소규모 형상 인식의 난이도를 보여준다. 최종적으로 테스트 데이터에서 **92.87%의 정확도**를 달성하였으며, 자세한 결과는 Fig. S24D에 제시되어 있다. 이 성능은 대형 형상 인식보다 낮아 더 어려운 인식 과제임을 시사한다. 그럼에도 불구하고, 더 조밀한 센싱 어레이를 사용하거나 SATS 모델 학습을 위해 더 높은 해상도의 데이터를 수집함으로써 이러한 한계를 완화할 수 있다.

## Note S8. Further investigation using simulated data

- **Inference of Pressure Map:**
    - The proposed tactile sensor, equipped with the SATS model, can directly generate pressure maps.  Furthermore, the elaborately designed SATS model enables the transfer of knowledge learned from single-point touch to multi-point touch without requiring extra training. To further evaluate the performance of the SATS model, a simulation model of the tactile sensor array, based on the EHS model, was developed, as shown in Fig. S25, A and B. The sensing surface was divided into a grid (Fig. S25C), with each point on the grid serving as a position for pressing to generate simulation data. At each position, the simulated force was applied following the pattern shown in Fig. S25D. The sensor response in the simulation model was consistent with that observed in real tests. Fig. S25E illustrates the simulated sensor response curve to pressure obtained by fitting the tested response data. Noting that 𝑦𝑥=0 ≈ 3.1 simulated the static measurement error. The ground truth pressure distribution was generated directly by the EHS model (e.g., Fig. S25F). It should be noted that only one position was pressed at a time during the simulation. All positions were pressed according to the above setup for data collection (generation).
        
        압력 맵 추론: SATS 모델이 탑재된 제안된 촉각 센서는 압력 맵을 직접 생성할 수 있습니다. 또한, 정교하게 설계된 SATS 모델은 추가 학습 없이 단일 지점 터치에서 학습된 지식을 다중 지점 터치로 전이할 수 있도록 합니다. SATS 모델의 성능을 더욱 평가하기 위해 EHS 모델을 기반으로 촉각 센서 어레이의 시뮬레이션 모델을 개발했습니다**(그림 S25, A 및 B 참조)**. 감지 표면은 격자로 나뉘었고(그림 S25C), 격자의 각 점은 시뮬레이션 데이터 생성을 위한 누름 위치로 사용되었습니다. 각 위치에서는 그림 S25D에 나타낸 패턴에 따라 시뮬레이션된 힘이 가해졌습니다. 시뮬레이션 모델에서의 센서 응답은 실제 테스트에서 관찰된 응답과 일치했습니다. 그림 S25E는 테스트 응답 데이터를 피팅하여 얻은 압력에 대한 시뮬레이션 센서 응답 곡선을 보여줍니다. $y_{x=0} ≈ 3.1$은 정적 측정 오차를 시뮬레이션한 것입니다. 실제 압력 분포는 EHS 모델에서 직접 생성되었습니다(예: 그림 S25F). 시뮬레이션 동안 한 번에 한 위치만 눌렀다는 점에 유의해야 합니다. 모든 위치는 데이터 수집(생성)을 위한 위의 설정에 따라 눌렀습니다.
        
        ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2022.png)
        
    - The SATS model was trained using the aforementioned simulation data. The RMSE for pressing each position was calculated as presented in Fig. S18, with an average value of 0.034 kPa, as shown in Fig. S26. Fig. S27 displays several examples of the SATS model’s  inferences, along with the corresponding ground truths and error distributions. It was observed that while larger pressures resulted in larger absolute errors, the relative error was maintained below 2.4% (maximum pressure to maximum pressure). Furthermore, Fig. S28 illustrates the SATS model’s performance when directly applied to multi-point touch scenarios. Although relatively larger errors were observed, the SATS model remained effective under these conditions,  reporting a relative error of less than 10% (maximum pressure to maximum pressure) in a three-point contact scenario. While obtaining ground truth pressure distribution under real-world conditions was challenging, the simulation results provided valuable insights
        
        SATS 모델은 앞서 언급한 시뮬레이션 데이터를 사용하여 학습되었습니다. 각 위치를 누를 때의 RMSE는 그림 S18에 제시된 바와 같이 계산되었으며, 그림 S26에서 볼 수 있듯이 평균값은 0.034 kPa였습니다. 그림 S27은 SATS 모델의 추론 결과 몇 가지 예시와 해당 실제 압력 분포 및 오차 분포를 보여줍니다. **압력이 클수록 절대 오차가 커지는 것을 확인할 수 있었지만, 상대 오차는 2.4% 미만(최대 압력 대 최대 압력)으로 유지되었습니다.** 또한, 그림 S28은 SATS 모델을 다중 접촉 시나리오에 직접 적용했을 때의 성능을 보여줍니다. 상대적으로 큰 오차가 관찰되었지만, SATS 모델은 이러한 조건에서도 효과적으로 작동하여 3점 접촉 시나리오에서 10% 미만의 상대 오차(최대 압력 대 최대 압력)를 나타냈습니다. 실제 환경에서 실제 압력 분포를 얻는 것은 어려웠지만, 시뮬레이션 결과는 유용한 통찰력을 제공했습니다.
        
        ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2023.png)
        
        ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2024.png)
        
    - Theoretically, the self-attention module enhances the SATS model’s ability for spatial perception. To validate the effectiveness of the self-attention module, a specific sensing unit (the orange one in Fig. S29A) and its corresponding encoded features were selected for analysis. The area surrounding this sensing unit was pressed to stimulate it, and the features encoded before and after the self-attention module were recorded, respectively. The t-SNE algorithm was used to reduce the dimensionality of these features to a two-dimensional space (Fig. S29, B and C). The features before the self-attention-based information sharing could only perceive distance and failed to distinguish the pressing positions (Fig. S29B). In contrast, the features after the self- attention module effectively decoupled spatial information. Fig. S29C shows the correspondence between data points in the two-dimensional space and pressing position, demonstrating the powerful spatial encoding capability of the self-attention module.
        
        이론적으로, 자기주의 모듈은 SATS 모델의 공간 지각 능력을 향상시킵니다. 자기주의 모듈의 효과를 검증하기 위해 특정 센싱 유닛(그림 S29A의 주황색 유닛)과 그에 해당하는 인코딩된 특징을 분석 대상으로 선정했습니다. 이 센싱 유닛 주변 영역을 눌러 자극을 가하고, 자기주의 모듈 적용 전후의 인코딩된 특징을 각각 기록했습니다. t-SNE 알고리즘을 사용하여 이러한 특징들의 차원을 2차원 공간으로 축소했습니다(그림 S29, B 및 C). 자기주의 기반 정보 공유 전의 특징은 거리만 인지할 수 있었고 누르는 위치를 구분하지 못했습니다(그림 S29B). 반면, 자기주의 모듈 적용 후의 특징은 공간 정보를 효과적으로 분리했습니다. 그림 S29C는 2차원 공간의 데이터 포인트와 누르는 위치 간의 대응 관계를 보여주며, 자기주의 모듈의 강력한 공간 인코딩 능력을 입증합니다.
        
        ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2025.png)
        
- **Inference of Coordinates and Force:**
    - Existing studies on tactile super-resolution primarily focus on localization estimation, specifically determining the coordinates of a contact point. However, a significant limitation of these approaches is that they only function effectively in single-point contact scenarios, failing when faced with multi-point contacts. This is attributed to both the design of the model structure and training data. Without an  internal structure design, an end-to-end model outputs only coordinates and the force of an external stimulus in the form of a three-dimensional vector. This model can only handle single-point contacts after being trained with single-point contact data. Data of multi-point contact must be collected to train a new model whose output layer has been modified to accommodate multi-point contact scenarios. The proposed SATS model overcomes this limitation by effectively transferring knowledge learned from single-point contact to multi-point contact, leveraging the local receptive field enhanced by the self-attention mechanism. This capability is crucial for real-world applications where multipoint contact is prevalent. Additionally, the calibration data required could be significantly reduced since only single-point contact data is necessary. In practice, collecting data that encompasses all possible multi-point contact conditions is nearly impossible due to the curse of dimensionalities.
        
        기존의 촉각 초해상도 연구는 주로 접촉점의 좌표를 결정하는 등 접촉 위치 추정에 초점을 맞추고 있습니다. 그러나 이러한 접근 방식의 중요한 한계는 단일 접촉 시나리오에서만 효과적으로 작동하고 다중 접촉 상황에서는 제대로 작동하지 못한다는 점입니다. 이는 모델 구조 설계와 학습 데이터 모두에 기인합니다. 내부 구조 설계가 제대로 이루어지지 않은 경우, 엔드투엔드 모델은 외부 자극의 좌표와 힘만을 3차원 벡터 형태로 출력합니다. 이러한 모델은 단일 접촉 데이터로 학습된 후에는 단일 접촉만 처리할 수 있습니다. 다중 접촉 시나리오를 수용할 수 있도록 출력 레이어를 수정하고 다중 접촉 데이터를 수집하여 새로운 모델을 학습시켜야 합니다. 제안하는 SATS 모델은 셀프 어텐션 메커니즘으로 강화된 로컬 수용 영역을 활용하여 단일 접촉에서 학습한 지식을 다중 접촉으로 효과적으로 전달함으로써 이러한 한계를 극복합니다. 이러한 기능은 다중 접촉이 빈번하게 발생하는 실제 응용 분야에 매우 중요합니다. 또한, 단일 접촉 데이터만 필요하므로 필요한 보정 데이터의 양을 크게 줄일 수 있습니다. 실제로는 차원의 저주 때문에 가능한 모든 다점 접촉 조건을 포괄하는 데이터를 수집하는 것은 거의 불가능합니다.
        
    - Despite the limitations of estimating coordinates for single-point contact, this approach remains relevant to tactile super-resolution and often yields an impressive scale factor. Consequently, the feasibility of applying the proposed super-resolution framework to this task was also explored. First, the SATS model was modified by replacing the local map reconstruction module with a regression module (a three-layer MLP) to infer contact position (coordinates) and force. Subsequently, 5000 positions on the sensing surface were randomly sampled for pressing, with force varying according to the paradigm in Fig. S25D. The modified SATS model was then trained. It was observed that greater force resulted in reduced position error (Fig. S30A). The spatial distribution of position errors under different forces is shown in Fig. S30B. Position accuracy improves as the force increases, with an average error of 0.12 mm (RMSE) over the whole force range. Under an external force of 8 N, this system achieved a maximal SR scale factor of 19547, extensively surpassing the current state-of-the-art. A similar trend was observed in force inference, with an average force error of 0.035 N. With its localization capability, the system can accurately reconstruct complex and fine patterns (Fig. S30D) in contour-following applications. The results in the inference of coordinates and force further demonstrated the generality of the proposed tactile SR framework, illustrating its strong potential.
        
        단일 접촉점 좌표 추정의 한계에도 불구하고, 이 접근 방식은 촉각 초해상도에 여전히 유효하며 종종 인상적인 스케일 팩터를 제공합니다. 따라서 제안된 초해상도 프레임워크를 이 작업에 적용하는 타당성도 탐색했습니다. 먼저, SATS 모델에서 로컬 맵 재구성 모듈을 접촉 위치(좌표)와 힘을 추론하는 회귀 모듈(3계층 MLP)로 대체하여 모델을 수정했습니다. 그런 다음, 감지 표면에서 5000개의 위치를 무작위로 샘플링하여 누르는 동작을 수행했으며, 힘은 그림 S25D의 패러다임에 따라 변화시켰습니다. 수정된 SATS 모델을 학습시킨 결과, 힘이 클수록 위치 오차가 감소하는 것을 관찰했습니다(그림 S30A). 다양한 힘 조건에서의 위치 오차의 공간 분포는 그림 S30B에 나타나 있습니다. 힘이 증가함에 따라 위치 정확도가 향상되며, 전체 힘 범위에서 평균 오차는 0.12mm(RMSE)입니다. 8N의 외부 힘 하에서, 이 시스템은 최대 19547의 SR 스케일 팩터를 달성하여 현재 최첨단 기술을 크게 능가했습니다. 힘 추론에서도 유사한 경향이 관찰되었으며, 평균 힘 오차는 0.035N이었습니다. 위치 파악 기능을 통해 이 시스템은 윤곽선 추적 응용 분야에서 복잡하고 미세한 패턴을 정확하게 재구성할 수 있습니다(그림 S30D). 좌표 및 힘 추론 결과는 제안된 촉각 SR 프레임워크의 일반성을 더욱 입증하며, 그 강력한 잠재력을 보여줍니다.
        
        ![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2026.png)
        

## Note S9. Investigation of the optimal receptive filed size

- The receptive field size of a sensing unit is influenced by its intrinsic sensing properties, the signal-to-noise ratio (SNR), and the thickness of the elastic covering. Given a predefined fabrication process, the sensing properties, such as sensitivity, remain constant. The SNR is typically affected by the magnitude of the external stimulus, while the thickness of the elastic covering can be easily adjusted to modify the receptive field size. To account for environmental noise, we assume a response threshold beyond which the signal is considered valid. For instance, a threshold of 0.15 implies that only when the relative resistance change (Δ𝑅⁄𝑅0) exceeds 15% is the response deemed effective. Since the response of the sensing unit is negatively correlated with the distance from the applied force, the boundary of the receptive field is determined by the unit’s response threshold.
    
    감지 장치의 수용 영역 크기는 고유의 감지 특성, 신호 대 잡음비(SNR), 그리고 탄성 커버의 두께에 영향을 받습니다. 미리 정의된 제조 공정을 가정하면 감도와 같은 감지 특성은 일정하게 유지됩니다. SNR은 일반적으로 외부 자극의 크기에 영향을 받는 반면, 탄성 커버의 두께는 수용 영역 크기를 조절하기 위해 쉽게 조정할 수 있습니다. 환경 잡음을 고려하기 위해, 신호가 유효하다고 간주되는 응답 임계값을 설정합니다. 예를 들어, 임계값이 0.15인 경우 상대 저항 변화(Δ𝑅⁄𝑅0)가 15%를 초과할 때만 응답이 유효하다고 판단합니다. 감지 장치의 응답은 가해진 힘으로부터의 거리에 반비례하므로, 수용 영역의 경계는 장치의 응답 임계값에 의해 결정됩니다.
    
- To explore the optimal receptive field size, we examined its dependence on the response threshold and the thickness of the elastic covering. In a specific setup where the threshold is set to 0.15, and an external force of 5 N is applied, the radius of the receptive field under different elastomer thicknesses is presented in Fig. S31A. The results indicate that the maximum radius of 11.6 mm occurs at an elastomer thickness of 5 mm. Subsequently, with the thickness fixed at 5 mm, the threshold was varied, producing the results shown in Fig. S31B. These results suggest that a lower threshold, corresponding to higher sensitivity, leads to a larger receptive field.
    
    최적의 수용장 크기를 알아보기 위해 반응 역치와 탄성 코팅 두께에 따른 수용장의 변화를 조사했습니다. 역치를 0.15로 설정하고 5N의 외부 힘을 가한 특정 조건에서, 탄성 코팅 두께에 따른 수용장 반경을 그림 S31A에 나타냈습니다. 결과에 따르면 탄성 코팅 두께가 5mm일 때 최대 반경인 11.6mm가 나타났습니다. 이어서, 두께를 5mm로 고정하고 역치를 변화시켜 얻은 결과는 그림 S31B에 제시되어 있습니다. 이러한 결과는 역치가 낮을수록(즉, 감도가 높을수록) 수용장이 커진다는 것을 시사합니다.
    

![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2027.png)

- In summary, the threshold and thickness exhibit a coupled influence on the receptive field size, as illustrated in Fig. S31C. Additionally, given  that external force influences the sensing unit’s response, the magnitude of the applied force is also considered in Fig. S31C. The results demonstrate that for a given threshold (primarily determined by the sensing unit’s sensitivity) and a specific force magnitude, an optimal elastomer thickness exists that maximizes the receptive field size.
    
    요약하자면, 그림 S31C에서 볼 수 있듯이 임계값과 두께는 수용 영역 크기에 상호 연관된 영향을 미칩니다. 또한, 외부 힘이 감지 장치의 반응에 영향을 미친다는 점을 고려하여 그림 S31C에서는 가해지는 힘의 크기도 함께 고려했습니다. 결과는 주어진 임계값(주로 감지 장치의 감도에 의해 결정됨)과 특정 힘의 크기에 대해 수용 영역 크기를 최대화하는 최적의 엘라스토머 두께가 존재함을 보여줍니다.
    

## Note S10. Validation of the proposed computational paradigm on a TENG-based tactile sensor

Single-electrode mode로 동작하는 **triboelectric nanogenerators(TENGs)** 는 동적 자극(dynamic stimuli) 감지에 매우 적합하며, 미세한 동적 터치 감지에 뛰어난 성능을 가진다. TENG는 **electrostatic induction(정전 유도)** 원리로 동작하며, 전기장이 전달 매체로 작용하기 때문에 각 sensing unit의 receptive field를 효과적으로 확장할 수 있다.

제안된 computational paradigm의 적용 가능성을 검증하기 위해, **23-taxel TENG 기반 촉각 센서 배열**이 제작되었다. 단일 taxel의 receptive field 반지름은 **15 mm**로 결정되었다. 이 센서는 **3층 구조의 flexible printed circuit board(FPCB)** 형태로 제작되었으며(Fig. S32A), 다양한 곡면에 부착할 수 있는 유연성을 제공한다(Fig. S32B).

![image.png](Super-resolution%20tactile%20sensor%20arrays%20with%20sparse/image%2038.png)

Arduino 개발 보드는 이 센서의 데이터를 수집하도록 프로그래밍되었으며(Fig. S32C), 이후 데이터 처리 및 super-resolution을 위해 **23채널 신호**를 출력하였다.

SATS 모델은 접촉 위치(좌표)를 추론하기 위해 수정되었다. 기존 local map reconstruction module 대신 **3층 MLP regression module**로 대체하였다. 수정된 SATS 모델은 다음 조건으로 학습되었다.

- Optimizer: Adam
- Learning rate: 0.002
- Batch size: 2048

학습 후 위치 추정 오차(localization error)는 **약 1.3 mm (RMSE)** 수준으로 유지되었으며, 약 **120배의 super-resolution scale factor**를 달성하였다.

TENG의 빠른 동적 응답 특성과 향상된 공간 해상도를 활용하여, 이 시스템은 다음 기능을 수행할 수 있었다.

- 외부 접촉 위치 정확 추정
- 접촉 궤적(contact trajectory) 추적
- 질량 3 g 이하의 튀는 탁구공(bouncing ping-pong ball) 위치 검출

이러한 성능은 Video S6에서 제시되었다.