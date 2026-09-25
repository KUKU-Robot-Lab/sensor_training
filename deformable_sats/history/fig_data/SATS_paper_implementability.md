# SATS 논문 전 figure/분석 — 우리 데이터 기준 구현 가능성 판정

기준 논문: *Super-resolution tactile sensor arrays with sparse units enabled by deep learning*
우리 자산: 단일점 grid press (xy1 eco20/eco50/ecomesh, xy0p5 ecomesh · d5/d10 인덴터 · 23ch 센서+좌표+힘) · 학습된 e2e SATS · EHS형 GT 생성기(gpu_on_the_fly) · fig2_material_ablation 소재특성 분석.

핵심 제약: **단일점 접촉 데이터만 보유**. 다점·형상·2점식별·키보드·로봇손 데이터 없음. 소재는 xy1 3종 + xy0p5 ecomesh.

---

## A. 이미 구현 완료 ✅

| 논문 | 내용 | 우리 산출물 |
|---|---|---|
| Fig4A | 대칭선 압력 프로파일(SR/수용영역 중첩) | `Fig2D_A_lineprofile_*` / `FinalA_*` |
| Fig4B | 위치별 오차 3D 막대 | `Fig2D_D_poserror3d_*` |
| Fig4C | 오차 히스토그램+KDE | `Fig2D_E_error_hist_*` |
| Fig4D | force별 오차 | `Fig2D_F_force_error_*` |
| FigS17 | 학습 후 추론 예시(GT/Pred) | `Fig2D_C_pressure3d_*` |
| Fig3 B·E·G·H | 상대저항·수용영역·커버리지·중첩 | `fig2_material_ablation/`(hitmap·coverage·overlap·centerline) |

## B. 기존 데이터로 즉시 구현 가능 (추가 취득·학습 불필요) ⭐

| 논문 | 내용 | 방법 | 가치 |
|---|---|---|---|
| **FigS20** | **위치추정 오차**(위치별 2D + force별 bar) | pred map argmax = 추정위치 vs 실제 좌표 RMSE | ★ 논문 헤드라인 지표(0.73mm)와 직접 대응 |
| Note S1 | SR scale factor | virtual/physical taxel 비 (=S/(N·π·ε²)) | 계산만 |
| FigS19(부분) | 보간 vs SATS 비교 | raw 23ch → linear/cubic 보간 맵 vs SATS RMSE | 학습 불필요 |
| **FigS29** | **self-attention 해석성** | attention 전후 feature 추출 + t-SNE(위치별 색) | ★ 학습된 모델서 바로 |
| Fig3I(부분) | 단일점 채널 응답 | 1-point 시계열 (2/3점은 불가) | |

## C. 추가 학습만 필요 (기존 데이터 재학습) 🔁

| 논문 | 내용 | 필요 작업 |
|---|---|---|
| **FigS30** | 좌표/힘 회귀 + localization scale factor(≈19547) | local-map 모듈 → 3층 MLP 회귀로 교체 후 재학습 (좌표·힘 GT 보유) |
| FigS19(ablation) | noLSTM / noAttention / noCNN 비교 | 모듈 제거 변형 각각 재학습 |
| ~~Note S8/시뮬~~ ✅ | 순수 시뮬레이션 학습 — **완료(2026-07-13)**: sim@sim 0.074 vs sim@real 1.109(전이 실패, 실측 필수 근거) | `supplementary/S8_sim_only/` |

## D. 추가 데이터 취득 필요 📥

| 논문 | 내용 | 필요 취득 | 비고 |
|---|---|---|---|
| Fig4E | 1/2/3점 압력 추론 | 2·3점 동시 press | **zero-shot → 재학습 불필요, 테스트 데이터만** |
| Fig4F | 2점 식별(분해능) | 간격별 2점 press(+고force) | K-means 후처리 |
| Fig4G | 윤곽 추적 이미징 | 형상 윤곽선 press | argmax 좌표 추적 |
| Fig5D–F | 형상 이미징+CNN 분류+t-SNE+혼동행렬 | 형상별 press ~4000×N | CNN 분류기 추가 학습 |
| Fig3C·D | 히스테리시스·동적응답 | load-unload/동적 로딩 | raw에 일부 존재 가능 — 확인 필요 |
| FigS21/S22 | 4/5/6점·2점상세 | 다점 press | |

## E. 범위 밖 / 하드웨어·응용 특화 ⛔

- Fig1·Fig2 개념도, Fig2E SATS 구조도 (일러스트)
- Fig3A SEM·분해도, Fig3F 프로토타입 사진, 제작/특성(S4–S8 재료·내구)
- Fig5A–C 소형 키보드·계산기·로봇손 통합 (별도 응용·하드웨어)
- FigS2 layout 최적화(PSO/GA/SA) — 센서 배열 설계 단계 (우리 배열 고정)
- FigS31 수용영역 최적 두께 — 단일 두께만이라 부분 분석만 가능
- FigS10/S11 회로·로봇팔 셋업 (하드웨어 문서)

---

## 권장 다음 단계 (가치·비용순)

1. **FigS20 위치추정 오차** — 즉시 가능, 논문 대표 지표와 직접 비교 가능. (기존 npz의 pred argmax vs 좌표)
2. **FigS29 attention 해석성 t-SNE** — 즉시 가능, SATS 구조 타당성 근거.
3. **FigS19 보간 비교** — 즉시 가능, "SATS > 보간" 정량 근거.
4. **FigS30 좌표/힘 회귀** — 재학습 1회, localization scale factor 확보.
5. (데이터 취득 후) **Fig4E 다점 zero-shot** — 소량 2·3점 취득만으로 논문 핵심 주장 재현.

> 단일점 데이터만으로도 논문 **정량 성능 축(Fig4A–D, S17, S19–S20, S29–S30)** 은 대부분 재현 가능.
> 다점·형상·응용(Fig4E–G, Fig5)은 추가 취득이 전제.
