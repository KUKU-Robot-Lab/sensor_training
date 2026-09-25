# References — robot_skin

robot_skin 의 코드·README·docstring 이 인용할 수 있는 **검증된** 문헌 목록이다. 새 문헌을 인용하려면 먼저
이 파일에 (검증 후) 추가하고, 코드에서는 `제1저자 et al., <학회> <연도>, arXiv:<ID>` 형식으로 쓴다.
목록에 없는 문헌은 인용하지 않는다.

**검증 방법 (2026-09-25)**

- 항목마다 제목·제1저자·학회/연도·arXiv ID 를 웹 검색 인덱스에서 교차 확인했다. 확인에 쓴 출처는 arXiv abs 페이지
  제목, 학회 proceedings 페이지(CVF Open Access, roboticsproceedings.org, NeurIPS proceedings), 출판사 페이지(Nature,
  Science, ACM, ScienceDirect), dblp/ADS 다.
- 이 작업 환경에서는 arxiv.org, dl.acm.org 등에 대한 직접 HTTP 접근이 egress 정책으로 막혀 있다. 그래서 원문 PDF
  본문을 열어 보지 않았다. **"robot_skin 이 가져오는 것" 열은 초록 수준에서 확인한 내용과 robot_skin 설계 의도를
  연결한 것이다.** 원문 세부(수식 기호, 하이퍼파라미터 등)를 docstring 에 옮길 때는 원문으로 다시 확인할 것.
- 저자는 `제1저자 et al.` 로만 적는다. 전체 저자 목록은 링크에서 확인한다.
- "코드 위치" 는 robot_skin 모듈 경로다(`robot_skin/` 기준; `deformable_sats/…` 는 저장소 루트 기준). 표에 적힌
  경로는 모두 현재 코드에 있다. 각 논문이 robot_skin 설계 전체에서 어디에 쓰였는지는 [`ARCHITECTURE.md`](ARCHITECTURE.md) §6.

---

## 1. 데이터 취득 · 인간 시연 · 촉각 글러브

| 논문 | 링크 | robot_skin 이 가져오는 것 | 코드 위치 |
|---|---|---|---|
| **ActionSense: A Multimodal Dataset and Recording Framework for Human Activities Using Wearable Sensors in a Kitchen Environment** — DelPreto et al., NeurIPS 2022 Datasets & Benchmarks | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5985e81d65605827ac35401999aea22a-Abstract-Datasets_and_Benchmarks.html) · [OpenReview](https://openreview.net/forum?id=olvz0gAdGOX) (arXiv 판 확인 안 됨) | 웨어러블 다중 스트림을 동시에 기록하는 프레임워크다(촉각 장갑, 손가락 추적 장갑, IMU 바디 트래킹, 1인칭 카메라 + 외부 카메라, 활동 라벨). robot_skin 은 여기서 세션 = 스트림별 파일 + 매니페스트 + 이벤트 로그 구조를 가져왔다. 기본 카메라 구성인 1인칭 `ego` + 고정 `third` 도 같은 근거다. | `acquisition/manifest.py`, `acquisition/recorder.py`, `acquisition/sync.py`, `acquisition/protocol.py`, `docs/DATA_ACQUISITION.md` |
| **Learning the signatures of the human grasp using a scalable tactile glove** (STAG) — Sundaram et al., *Nature* 569, 698–702 (2019) | [Nature](https://www.nature.com/articles/s41586-019-1234-z) | 손 전체에 분포한 촉각 글러브(약 550 센서)와 대규모 파지 데이터로 물체 식별·무게 추정이 가능하다는 근거다. robot_skin 은 글러브 taxel 을 손 전체 layout 으로 다룬다. D2 에서는 물체 메타(`task.object`)를 기록하고, 물체 단위 split 으로 누수를 막는다. | `common.layouts` (glove_template, 읽기 전용), `acquisition/protocols/d2_task.yaml`, `datasets/splits.py` (`by="object"`) |
| **OSMO: Open-Source Tactile Glove for Human-to-Robot Skill Transfer** — Yin et al., arXiv 2025 | [arXiv:2512.08920](https://arxiv.org/abs/2512.08920) · [code](https://github.com/jessicayin/osmo_tactile_glove) | 인간과 로봇이 **같은 촉각 글러브**를 끼면 시각·촉각 embodiment gap 이 줄어든다. 이렇게 모은 인간 시연만으로 접촉이 많은 정책(wiping)을 학습했다. robot_skin 은 여기서 두 가지를 가져왔다: 글러브(인간)와 로봇 핸드가 같은 pose 기반 taxel 토큰 표현을 쓰는 구조, 그리고 인간 손 행동 → 로봇 리타게팅 경로. D2 의 wipe 과제도 여기서 왔다. 차이점: OSMO 는 3축(법선+전단) 센서 12개를 쓴다. mk555 기압 taxel 은 법선 1축이라 전단 정보가 없다. | `representation/tokenizer.py`, `representation/encoder.py`, `transfer/`, `action/retarget.py`, `acquisition/protocols/d2_task.yaml` |
| **DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation** — Wang et al., RSS 2024 | [arXiv:2403.07788](https://arxiv.org/abs/2403.07788) | 휴대형 손 mocap(SLAM + 전자기장 글러브)으로 인간 데이터를 모으고, IK 리타게팅을 거쳐 로봇 정책(DexIL)을 학습하는 파이프라인이다. robot_skin 의 "D2 인간 시연 → `action.retarget` → 로봇 제어" 흐름이 이 구조를 따른다. | `acquisition/`, `action/retarget.py`, `control/runner.py` |
| **DexUMI: Using Human Hand as the Universal Manipulation Interface for Dexterous Manipulation** — Xu et al., CoRL 2025 | [arXiv:2505.21864](https://arxiv.org/abs/2505.21864) | 인간 손을 범용 조작 인터페이스로 삼아 여러 로봇 핸드로 기술을 옮기는 프레임워크다. robot_skin 은 여기서 인간 손과 로봇 손의 embodiment gap 을 명시적으로 다루는 관점을 가져왔다. 그래서 canonical action 을 인간 손(MANO)으로 두고, 로봇별 변환은 리타게팅 단계로 분리한다. | `action/space.py` (`hand_mano`), `action/retarget.py`, `docs/DEPLOYMENT.md` |

## 2. 손 자세 (비전 라벨 · IMU)

| 논문 | 링크 | robot_skin 이 가져오는 것 | 코드 위치 |
|---|---|---|---|
| **Embodied Hands: Modeling and Capturing Hands and Bodies Together** (MANO) — Romero et al., *ACM TOG* 36(6), SIGGRAPH Asia 2017 | [DOI 10.1145/3130800.3130883](https://doi.org/10.1145/3130800.3130883) · [arXiv:2201.02610](https://arxiv.org/abs/2201.02610) | 손목 + 15 손가락 관절로 된 16관절 기구학 체인과 관절별 axis-angle 자세(15×3 = 45)를 가져왔다. robot_skin 은 메시와 형상 블렌드 없이 **스켈레톤만** 다시 구현했다. MANO pkl 이 없어도 동작한다. | `pose/mano.py`, `action/space.py` (54-D hand action), `acquisition/manifest.py` (`hand_pose.npz`), `pose/vision_hand.py` |
| **Reconstructing Hands in 3D with Transformers** (HaMeR) — Pavlakos et al., CVPR 2024 | [arXiv:2312.05251](https://arxiv.org/abs/2312.05251) | 단안 영상 → MANO 파라미터 회귀기다. robot_skin 은 이를 **오프라인**으로 돌려 `hand_pose.npz` 라벨을 만든다. 이 라벨이 IMU→손자세 모델을 감독한다. | `pose/vision_hand.py` (`HaMeREstimator` stub, `smooth_hand_labels`), `stages/imu_pose.py` |
| **WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild** — Potamias et al., CVPR 2025 | [arXiv:2409.12259](https://arxiv.org/abs/2409.12259) | 손 검출과 3D 재구성을 한 번에 하는 파이프라인이다. HaMeR 대신 쓸 수 있는 오프라인 라벨러이며, 별도 손 검출기가 필요 없다. | `pose/vision_hand.py` |
| **VIHand: Enhancing 3D Hand Pose Estimation with Visual-Inertial Benchmark** (VIFNet / **VIFNet-S**) — Wang et al., ACM MM 2025 (**부분 검증**, §8 참조) | [DOI 10.1145/3746027.3758215](https://doi.org/10.1145/3746027.3758215) · [project](https://shirley0118.github.io/VIHand) | 글러브를 낀 visual-inertial 손 자세 데이터셋이다. 시각-관성 융합 모델(VIFNet)을 IMU 만 쓰는 학생 모델(VIFNet-S)로 증류한다. robot_skin D1 의 "비전 라벨로 IMU→손자세 모델을 감독한다"는 설계 근거이며, 사용자가 지정한 사전학습 백본 후보다. | `pose/glove_imu2mano.py` (`load_vifnet_s` / `finetune_vifnet_s` stub), `pose/imu_model.py` (`ImuHandPoseNet` 사내 베이스라인), `stages/imu_pose.py` |
| **Visual-inertial hand motion tracking with robustness against occlusion, interference, and contact** (VIST) — Lee et al., *Science Robotics* 6(58), 2021 | [DOI 10.1126/scirobotics.abe1315](https://doi.org/10.1126/scirobotics.abe1315) | 다중 IMU 글러브와 헤드마운트 스테레오 카메라를 융합해, 가림·자기장 간섭·접촉에도 버티는 손 추적을 보였다. robot_skin 이 ego 카메라와 글러브 IMU 를 함께 쓰는 근거다. IMU 는 가림에 강하고, 비전은 IMU 누적 오차를 잡아 준다. | `pose/imu_model.py`, `pose/vision_hand.py`, `acquisition/` (기본 `ego` 카메라) |

## 3. 촉각 신호 · 무접촉 기저선 · 접촉 판정

| 논문 | 링크 | robot_skin 이 가져오는 것 | 코드 위치 |
|---|---|---|---|
| **What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?** — Kendall & Gal, NIPS 2017 | [arXiv:1703.04977](https://arxiv.org/abs/1703.04977) | 입력에 따라 달라지는(heteroscedastic) aleatoric 분산을 함께 예측하는 Gaussian NLL 이다. robot_skin 은 무접촉 ΔS 기저선을 (평균, log 분산)으로 예측하고, 잔차 z-score 에 예측 분산을 반영한다. | `baseline/temporal.py` (`gaussian_nll`, `TemporalBaselinePredictor`), `contact/calibration.py` (`ResidualCalibrator`) |
| **Pose-Aware Modeling to Mitigate Pose-Related Artifacts in Tactile Gloves** — Yu et al., arXiv 2026 | [arXiv:2607.22964](https://arxiv.org/abs/2607.22964) | 유연 촉각 글러브는 접촉이 없어도 손 자세가 바뀌면 신호가 변한다(pose-related artifact). 이 논문은 손 자세 정보를 쓰는 잔차 예측 분기로 이를 보정했고, 글러브 3종·사용자 15명에서 최소 검출 힘(MDF)이 줄었다. robot_skin 의 D1 무접촉 기저선 예측(관절 상태·taxel pose → 움직임 유발 ΔS)과 **문제 설정이 같은 병행 연구**다. 비교 기준과 평가 지표(MDF) 참고용이다. robot_skin 설계는 `deformable_sats` bending restorer 를 일반화한 것으로, 이 논문과 독립적으로 나왔다. | `baseline/model.py`, `baseline/temporal.py`, `stages/baseline.py`, `eval/metrics.py` |
| **Focal Loss for Dense Object Detection** — Lin et al., ICCV 2017 | [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) | 쉬운 음성 샘플(무접촉)이 대부분인 극단적 클래스 불균형에서 손실을 재가중하는 방법이다. per-taxel 접촉 분류기의 focal 손실 옵션에 쓴다. | `contact/detector.py` |

## 4. 촉각 표현 · 사전학습

| 논문 | 링크 | robot_skin 이 가져오는 것 | 코드 위치 |
|---|---|---|---|
| **Masked Autoencoders Are Scalable Vision Learners** (MAE) — He et al., CVPR 2022 | [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) | 입력의 높은 비율을 가리고 재구성하는 자기지도 사전학습이다. robot_skin 은 이를 taxel 에 적용했다 (MAE 방식): 가린 taxel 은 인코더 attention 의 key 에서 제외하고(key padding), 가린 자리에 디코더 mask 토큰을 넣은 뒤 모든 자리에 taxel 위치 임베딩을 더해 가벼운 디코더가 가린 taxel 의 residual_z 와 접촉 level 을 재구성한다. 고정 2-D sin-cos 위치 임베딩도 여기서 가져왔다. | `representation/pretrain.py` (`MaskedTaxelPretrainer`), `stages/pretrain.py`, `representation/tokenizer.py`, `vision/encoders.py` |
| **Sparsh: Self-supervised touch representations for vision-based tactile sensing** — Higuera et al., CoRL 2024 | [arXiv:2410.24090](https://arxiv.org/abs/2410.24090) | 라벨 없는 대규모 촉각 데이터로 자기지도 사전학습(MAE/DINO/JEPA)한 범용 촉각 표현이, 과제별 end-to-end 학습보다 낫다는 근거다. 다만 Sparsh 는 비전 기반 촉각 센서(촉각 이미지)용이고 robot_skin 은 희소 기압 taxel 이다. 그래서 **방법론 근거로만** 쓴다. | `representation/pretrain.py`, `stages/pretrain.py` |
| **3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing** — Huang et al., CoRL 2024 | [arXiv:2410.24091](https://arxiv.org/abs/2410.24091) | 촉각 값을 시각과 같은 3D 공간에 점으로 놓아 공간 관계를 보존하는 통합 표현이다(정책은 diffusion policy). robot_skin 은 taxel 을 채널 인덱스가 아니라 **3D pose(위치·법선)** 로 토큰화한다. 그래서 글러브와 로봇 핸드가 같은 토크나이저를 쓴다. | `representation/tokenizer.py`, `representation/encoder.py`, `pose/provider.py`, `vtla/model.py` |
| **Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains** — Tancik et al., NeurIPS 2020 | [arXiv:2006.10739](https://arxiv.org/abs/2006.10739) | 저차원 좌표를 sin/cos 특징으로 사상해야 MLP 가 고주파 함수를 배울 수 있다는 결과다. robot_skin 은 taxel 3D 위치를 옥타브 주파수 sin/cos 로 인코딩한다(positional-encoding 형태). | `representation/tokenizer.py` (`fourier_features`) |
| **Perceiver: General Perception with Iterative Attention** — Jaegle et al., ICML 2021 | [arXiv:2103.03206](https://arxiv.org/abs/2103.03206) | 소수의 잠재(쿼리) 토큰이 cross-attention 으로 큰 입력을 압축하는 비대칭 어텐션이다. robot_skin 은 이를 써서 taxel 수와 무관하게 K 개 촉각 토큰을 만든다. | `vtla/adapter.py` (`TactileTokenAdapter`) |

## 5. 회전 · 행동 표현 · 정책 헤드

| 논문 | 링크 | robot_skin 이 가져오는 것 | 코드 위치 |
|---|---|---|---|
| **On the Continuity of Rotation Representations in Neural Networks** — Zhou et al., CVPR 2019 | [arXiv:1812.07035](https://arxiv.org/abs/1812.07035) | 연속 6D 회전 표현(회전행렬의 첫 두 열 + Gram–Schmidt)이다. 쿼터니언·오일러각은 불연속이라 회귀에 불리하다. | `geometry/rotations.py` (`matrix_to_6d`, `sixd_to_matrix`), `action/space.py` (손목 회전 6D), `pose/imu_model.py` (15×6D 출력) |
| **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware** (ACT) — Zhao et al., RSS 2023 | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) | action chunking(미래 H 스텝을 한 번에 예측)과 temporal ensembling(지수 가중, 가장 오래된 예측이 i = 0), L1 회귀 손실을 가져왔다. | `action/chunking.py` (`action_chunk`, `TemporalEnsembler`), `vtla/heads.py` (`ChunkRegressionHead`), `vtla/losses.py`, `control/runner.py` |
| **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** — Chi et al., RSS 2023 (확장판 IJRR) | [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) | 행동 시퀀스를 생성하고 receding horizon 으로 실행한다. 시각 조건화 방식(ResNet-18 + GroupNorm, spatial softmax, 작은 random crop)도 가져왔다. 생성형 행동 헤드를 쓰는 근거이기도 하다. | `vision/encoders.py`, `vision/transforms.py`, `vtla/heads.py` |

## 6. VTLA / VLA · 생성 모델 · 선호 학습

| 논문 | 링크 | robot_skin 이 가져오는 것 | 코드 위치 |
|---|---|---|---|
| **VTLA: Vision-Tactile-Language-Action Model with Preference Learning for Insertion Manipulation** — Zhang et al., arXiv 2025; 저널판 *Biomimetic Intelligence and Robotics* (2026) | [arXiv:2505.09577](https://arxiv.org/abs/2505.09577) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2667379726000616) | vision + tactile + language → action 이라는 문제 정의와 명칭, 삽입(peg-in-hole) 과제를 가져왔다. 이 논문은 next-token(분류) 손실과 연속 행동 사이의 간극을 DPO 선호 최적화로 보완한다. robot_skin 은 행동을 토큰화하지 않는다. 대신 연속 헤드(chunk 회귀 / flow matching)를 쓰고, DPO 는 선택적 후처리 훅으로 둔다. | `vtla/model.py`, `vtla/dpo.py`, `stages/vtla.py`, `docs/VTLA.md`, `acquisition/protocols/d2_task.yaml` (`peg_insert`) |
| **π0: A Vision-Language-Action Flow Model for General Robot Control** — Black et al., RSS 2025 | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) | 사전학습 VLM 위에 flow-matching 행동 헤드를 얹어 연속 action chunk 를 만든다. 시간 변수 τ 의 방향 규약은 `vtla/heads.py` docstring 에 robot_skin 규약으로 명시하며, π0 원문 표기와 다를 수 있다. | `vtla/heads.py` (`FlowMatchingHead`), `vtla/model.py` |
| **Flow Matching for Generative Modeling** — Lipman et al., ICLR 2023 | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) | 고정된 조건부 확률 경로의 벡터장을 회귀하는, 시뮬레이션 없는 학습 목적함수다. OT(직선) 경로를 쓴다. | `vtla/heads.py`, `vtla/losses.py` |
| **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow** — Liu et al., ICLR 2023 | [arXiv:2209.03003](https://arxiv.org/abs/2209.03003) | 노이즈와 데이터를 직선으로 잇는 보간 경로와, 적은 스텝의 Euler 적분 추론을 가져왔다. | `vtla/heads.py` |
| **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** — Rafailov et al., NeurIPS 2023 | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) | 보상 모델 없이, 선호 쌍(chosen / rejected)의 정책·참조 로그확률 차이에 분류 손실을 건다. VTLA 논문이 쓰는 선호 학습의 원형이다. | `vtla/dpo.py` (`dpo_loss`) |
| **OpenVLA: An Open-Source Vision-Language-Action Model** — Kim et al., CoRL 2024 | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) | 공개 VLA 기준선이다. DINOv2 + SigLIP 비전 타워 조합을 가져왔다. | `vision/encoders.py` (`HFVisionEncoder`), `docs/VTLA.md` |
| **Octo: An Open-Source Generalist Robot Policy** — Octo Model Team (Ghosh et al.), RSS 2024 | [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) | 모듈형 토큰 설계를 가져왔다: 언어 과제, 여러 카메라(손목·3인칭), proprio 관측, 여러 행동 공간을 한 트랜스포머가 처리한다. robot_skin 은 여기에 modality type embedding 과 readout 토큰을 쓰고, `hand_mano` / `robot_joint` 행동 공간을 둘 다 지원한다. | `vtla/model.py`, `action/space.py` |
| **DINOv2: Learning Robust Visual Features without Supervision** — Oquab et al., TMLR 2024 | [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | 고정(frozen) 비전 백본 옵션이다(`facebook/dinov2-*`). | `vision/encoders.py`, `vision/feature_cache.py` |
| **Sigmoid Loss for Language Image Pre-Training** (SigLIP) — Zhai et al., ICCV 2023 | [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) | 비전/텍스트 고정 인코더 옵션이다(`google/siglip-*`). | `vision/encoders.py`, `language/text_encoders.py` |
| **Learning Transferable Visual Models From Natural Language Supervision** (CLIP) — Radford et al., ICML 2021 | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | 고정 텍스트 인코더 옵션이다(`openai/clip-vit-base-patch32`). | `language/text_encoders.py` |

## 7. 리타게팅 · 로봇 제어

| 논문 | 링크 | robot_skin 이 가져오는 것 | 코드 위치 |
|---|---|---|---|
| **DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System** — Handa et al., ICRA 2020 | [arXiv:1910.03135](https://arxiv.org/abs/1910.03135) | 사람 손 → 로봇 핸드 운동학 리타게팅 비용함수를 가져왔다. 손끝끼리, 그리고 엄지–손가락 사이의 상대 벡터를 맞추고 정규화 항을 더한다. | `action/retarget.py` (`FingertipRetargeter`), `control/runner.py` |
| **AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System** — Qin et al., RSS 2023 | [arXiv:2307.04577](https://arxiv.org/abs/2307.04577) | 로봇 모델에 묶이지 않는 범용 리타게팅이다(여러 팔·손·카메라 구성 지원). robot_skin 에서는 FK 콜러블이나 URDF 만 바꿔 끼우면 되도록 설계했다. | `action/retarget.py`, `pose/urdf.py` |

## 8. VIHand / VIFNet-S — 무엇이 확인되고 무엇이 안 됐나

사용자가 이름으로 지정한 "VIHand / VIFNet-S" 는 실제 문헌으로 **찾았다**. 다만 이 환경에서는 원문과 프로젝트 페이지를
직접 열 수 없어 검증은 부분적이다.

**확인됨** (검색 인덱스, 출판사 DOI 레코드 기준)

- 제목: *VIHand: Enhancing 3D Hand Pose Estimation with Visual-Inertial Benchmark*
- 게재처: Proceedings of the 33rd ACM International Conference on Multimedia (ACM MM 2025), DOI `10.1145/3746027.3758215`
- 초록 수준 내용:
  - 글러브를 낀 visual-inertial 손 자세 추정용 대규모 데이터셋이다. 피험자 15명, 동기화된 RGB-D·IMU 프레임 140만 장 이상이다.
  - 두 모달리티를 쓰는 융합 모델 **VIFNet** 과, IMU 만으로 평가하는 증류 학생 모델 **VIFNet-S** 를 제안한다.
  - 희소 IMU 구성에서도, visual-inertial 감독으로 증류한 모델이 IMU-only 성능을 크게 올렸다고 보고한다.
- 프로젝트 페이지: <https://shirley0118.github.io/VIHand> (검색 인덱스에 있음; 직접 열람은 못 함)
- 저자: 검색 인덱스 기준 제1저자는 Xinyi Wang 이다. ACM 페이지를 직접 확인하지 못했으므로 코드에서는 "Wang et al." 로만 표기한다.

**확인하지 못함** (연동 전에 원문·공개 코드로 반드시 확인)

- arXiv 판이 있는지
- 글러브 IMU 개수와 부착 위치. robot_skin 글러브는 7 IMU 다(wrist, palm, thumb/index/middle/ring/pinky). 이와 같은지 모른다.
- IMU 입력 형식: 쿼터니언인지 raw 가속도·자이로인지, 샘플링 레이트, 윈도 길이, 좌표계 규약.
- 출력 표현: MANO θ 인지, 관절 3D 좌표인지. 관절 좌표라면 MANO IK 변환이 필요하다.
- 코드·사전학습 가중치 공개 여부와 라이선스.

**코드에 미치는 영향**

`pose/imu_model.py`, `pose/glove_imu2mano.py`, `pose/README.md` 는 `ImuHandPoseNet` 을 VIFNet-S 와 **같은 역할**
(IMU 윈도 → MANO 손가락 자세)의 사내 베이스라인으로만 적고, VIFNet-S 의 실제 입출력은 **미검증**이라고 명시한다.
`load_vifnet_s` 를 구현할 때는 두 변환을 감싸는 래퍼로 만들어야 한다: 입력 어댑터(robot_skin 7 사이트 특징 →
VIFNet-S 입력 배치)와 출력 변환(→ `finger_pose[15,3]` axis-angle). 그 전까지 `load_vifnet_s` / `finetune_vifnet_s` 는
`NotImplementedError` 스텁이고, `stages/imu_pose.py` 가 `ImuHandPoseNet` 을 학습한다.

## 9. 내부 참고 (이 저장소)

| 파일 | robot_skin 이 가져오는 것 | 코드 위치 |
|---|---|---|
| `deformable_sats/sats/bending/baseline_restorer.py` | deg→offset baseline restorer 다. 오프셋을 예측한 뒤 빼는 구조이고, zero-init 이라 처음엔 항등으로 동작한다. 여기서 얻은 교훈은 **관측 ΔS 를 기저선 모델 입력으로 넣지 말 것**이다. 넣으면 모델이 접촉까지 오프셋으로 학습해 지워 버린다. | `baseline/model.py`, `baseline/temporal.py` |
| `deformable_sats/sats/inference/run_dashboard.py` | 채널 격리(quarantine) 규칙과 부호 게이트다. robot_skin 은 이를 per-taxel 포화 상태기계로 정리했다. | `contact/saturation_fsm.py`, `contact/ordinal.py`, `control/online.py` |
| `deformable_sats/sats/preprocessing/bin_merge.py` | mk555 `.bin` 파싱의 기준 구현이다. **복사하지 않고** 주입 가능한 파서 콜러블로 연결한다. | `acquisition/sources.py`, `datasets/build.py` |

## 10. 구현 계약(SPEC) 참고문헌 목록 대비 정정 · 보강

1. **"Rectified flow / flow matching: Lipman et al., arXiv:2210.02747"** — 두 문헌이 한 줄에 섞여 있다. arXiv:2210.02747 은
   *Flow Matching for Generative Modeling* (Lipman et al., ICLR 2023)이다. Rectified flow 는 별도 문헌인 *Flow Straight and
   Fast* (Liu et al., ICLR 2023, arXiv:2209.03003)다. 이 파일에는 둘을 따로 실었다.
2. **MANO** — 원 논문은 ACM TOG 36(6), SIGGRAPH Asia 2017 이다. arXiv 판(2201.02610, 2022 게시)을 보강했다.
3. **학회·연도 보강**: ACT RSS 2023 · Diffusion Policy RSS 2023(확장판 IJRR) · AnyTeleop RSS 2023 · DexCap RSS 2024 ·
   Octo RSS 2024 · π0 RSS 2025 · OpenVLA CoRL 2024 · DexUMI CoRL 2025 · 3D-ViTac CoRL 2024(일치) · VTLA 저널판
   *Biomimetic Intelligence and Robotics* (2026) · OSMO 는 arXiv 프리프린트(2025-12, 제1저자 Yin).
4. **ActionSense** — arXiv ID 는 확인하지 못했다. NeurIPS 2022 D&B proceedings 와 OpenReview 링크를 쓴다.
5. **Octo** 저자 표기는 "Octo Model Team (Ghosh et al.)" 이다.
6. 나머지 항목(VTLA, 3D-ViTac, OSMO, DexUMI, ACT, Diffusion Policy, π0, Zhou, Kendall & Gal, MAE, HaMeR, AnyTeleop, DexPilot,
   DexCap, OpenVLA, Octo)은 제목과 arXiv ID 가 SPEC 과 일치한다.
7. **추가한 검증 문헌**: VIHand(부분 검증), VIST, STAG, WiLoR, Sparsh, Pose-Aware PRA(2026), Focal loss, Fourier features,
   Perceiver, Rectified flow, DPO, DINOv2, SigLIP, CLIP.
