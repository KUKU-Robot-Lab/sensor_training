# 논문 투고 로드맵 & 마스터 체크리스트

> **최상위 전략 = Notion "Bending-Aware Tactile Skin — Nature Communications"** (2026-07-18 확정):
> 타깃 NatComm(fallback AIS), **G0~G3 go/no-go 게이트**, 4-figure 스토리, claim boundaries.
> 이 문서는 그 아래의 **운영 체크리스트** — 아래 P0의 밴딩 취득은 G0/G1/G2 정합 스펙
> (`fig3_sats_bending/bending/README.md` 2026-07-18판) 기준. 첫 valid 취득 후 4주 내 G1~G2로 저널 분기.
>
> 기준 = 논문 md(`Development_...md`)의 기여 **C1~C4**, §6 Figure Plan, §7 데이터 체크리스트, §9 정량 수치.
> 갱신: **2026-07-18 v3** — NatComm 게이트 정합. 세부 데이터 위치는 `PROJECT_STRUCTURE.md`.
> 표기: ✅ 완료 · 🔄 진행/부분 · ⬜ 미착수

---

## 0. 한눈에 보는 상태

| 기여 | 주장 | 상태 |
|---|---|---|
| **C1** mesh 수용장 중첩 → SR 향상 | 소재 비교 + SATS 성능으로 입증 | ✅ **데이터·figure 완비** (3-set 통계는 기존 데이터 재분석만 남음, §2.4) |
| **C2** 밴딩 자가추정 → bending-aware SR | 곡률 추정 + 잔차 보정 | 🔄 **코드 Phase 0 완료, 데이터 취득 0건** ← 투고 critical path |
| **C3** 다중 접점 + 동적 해석 | 단일→다점·시간 확장 | ⬜ **유지 확정(D1)** — 재학습 불필요, **2·3점 테스트 취득 + zero-shot 추론 실험만** |
| **C4** 사람손·로봇핸드 공용 플랫폼 | 곡면 부착 데모 | ⬜ 하드웨어 데모 미착수 (Fig.4) |

**투고 성립의 최소 조건 = C1(완료) + C2(밴딩 실증)**. C3는 취득 부담 작음(zero-shot), C4는 데모.

---

## 1. 결정사항 확정 내역 (2026-07-12, D1~D4)

| # | 결정 | 확정 내용 |
|---|---|---|
| D1 | C3 다중접점 | **유지.** SATS 구조상 단일점 학습 모델로 다점 zero-shot 추론 가능(원논문 Fig4E, `SATS_paper_implementability.md` §D) → **재학습 없이 2·3점 동시 press 테스트 데이터만 취득해 추론 실험**. 2점 분해능(Fig4F류)은 간격별 2점 press + K-means 후처리 추가 |
| D2 | 소재 3-set 반복 | **추가 취득 불필요 — 데이터 이미 보유.** xy1 소재별 d5/d10 각 3 rep 취득 완료(learning_data 병합 기준). Fig.2 B/C 분석이 대표 1 set만 사용했을 뿐 → 나머지 test 폴더 `.bin→CSV` 변환 후 **3-set 평균±std 오차막대 재분석**(분석 작업, §2.4) |
| D3 | molding 변경·아랫면 통일 | **진행 중** (하드웨어 작업) — 완료 시 신규 시편 취득 여부 별도 판단 |
| D4 | §4.2 표기 | **확정: 층별 단일 소재 적층(혼합비 아님).** Bot=eco20 / Mid=eco20 / Top=eco20·eco50·ecomesh 중 택1(소재 비교군). 변형 시편: Mid=eco45 버전, Bot·Top 동일 소재 버전 존재 → 본문 §4.2 반영 완료 |

---

## 2. P0 — Critical Path (이것 없이는 투고 불가)

### 2.1 밴딩 데이터 취득 (C2·Fig.3) — 최우선 (⚠️ 새 센서 제작 완료 후, **G0/G1 정합 스펙**)
- [ ] **G0**: jig 각도 눈금 독립 검증·스트림 동기화·메타(센서ID/온도/retare/session) 기록·registry 추적
- [ ] **밴딩-only baseline (G1)**: −40°~+40° 10° 간격 × 무하중 10 set + **독립 remounting ≥3 session** + intermediate-angle·remount 홀드아웃 + 온도/drift 로깅
- [ ] **밴딩+접촉 (G2)**: 0°, ±20°, ±40° × d5, xy0.5 계단식 동일 프로토콜 + **same-session flat reference**
- [ ] 취득 직후 **사전 검증**: 밴딩·접촉 신호 시간스케일 분리 가능성 + 중첩 선형성 (ill-posed 리스크 게이트)
- 스펙 상세: `fig3_sats_bending/bending/README.md` (2026-07-18 G-게이트 정합판)

### 2.2 밴딩 모델 P1→P4 (`sats/bending`, Phase 0 완료)
- [ ] P1 estimator 학습 → **곡률 MAE/R²** (§9 수치, Fig.3C)
- [ ] P2 restorer (A안 오프셋 지도 / B안 end-to-end 중 데이터 보고 결정)
- [ ] P3 pipeline 검증 — **G2 5-비교군**(uncorrected/naive subtraction/proposed/curvature-conditioned/per-curvature retraining 상한) + paired 95% CI·remount 재현 판정 (§9 수치, Fig.3D)
- [ ] P4 figure: Fig.3 패널 B/C/D/E 생성 스크립트 + report (피규어 재현 코드 원칙)

### 2.3 Fig.3 완성
- [ ] 패널 (A) jig·각도 정의 모식도 (`algorithm/`의 알고리즘 구조도와 별개 — §6 정의 기준)
- [ ] 패널 (B)~(E) ← 2.2 산출물
- [x] flat 기준 패널 (`flat_sr/final_xy0p5/`, A 모델)

### 2.4 Fig.2 3-set 통계 재분석 (D2 확정 — 취득 불필요, 분석만) ✅ 2026-07-13 완료
- [x] 소재별 나머지 test 폴더 10개 `.bin→CSV` 변환 (18/18 세트 CSV 확보)
- [x] Fig.2C 3-set 평균±std 재생성 (`generate_panelC_3set.py` → `Fig2C_metrics_3set.{png,csv}` + taxel health)
  — **set간 재현성 매우 높음(CV<2%), 기존 결론 유지**. 캐빗: eco50 d5 전 세트 dead 채널 1~2개(d10은 깨끗)

---

## 3. P1 — 주장 강화 (투고 전 완료 권장)

### 3.1 C3 다점 zero-shot 실험 (D1 확정 — 재학습 불필요, **실시간 추론도 불필요=오프라인**)
- [ ] **2·3점 동시 press 테스트 데이터 취득만** (기존 로거로 raw .bin 저장이 하드웨어 작업의 전부)
  - GT 좌표 = **간격을 아는 이중/삼중 인덴터 지그**: 지그 중심 좌표+알려진 오프셋으로 각 접점 계산(원논문 방식)
  - loadcell은 합력만 측정 → 검증 지표는 **위치 분리·피크 개수** 중심(점별 압력은 정성)
- [ ] (취득 후, 오프라인) 동결 최종모델(A)로 zero-shot 추론 → 다점 피크 분리 확인 (SR 맵 argmax·K-means)
- [ ] (선택) 간격별 2점 press → 2점 분해능 곡선 (원논문 Fig4F 대응)
- [ ] (선택) 동적 해석: 기존 raw의 load-unload 구간으로 히스테리시스·시간 응답 분석(원논문 Fig3C·D 대응 — 취득 전 기존 데이터 확인 먼저)
- [ ] figure + §5.5 본문 실증 연결

### 3.2 Fig.4 Application (C4)
- [ ] 로봇핸드 부착 실시간 감지·제어 데모 (실시간 추론: `sats/inference` + `sats/bending/pipeline`)
- [ ] 사람 손 곡면 부착 동작 시연
- [ ] 데모 영상/스틸 → Fig.4 패널 구성

### 3.3 Fig.1 Concept
- [ ] 컨셉 일러스트 제작 (후보 소재: `fig1_concept/README.md` — archive/사진·pptx)

### 3.4 성능 보강 취득
- [ ] **저force d10 반복취득**(xy0.5 계단식) — d10 magnitude 과대예측(현 d10_rel 0.749)의 유일한 남은 해법. Fig.3E 맵 품질에 직결
- [ ] 재학습(A 모델, 동일 하이퍼) → 진단 재덤프 → flat figure 재생성

### 3.0 새 센서 취득 SOP (2026-07-20 확정 — `ACQUISITION_SOP.md`)
- **계단식 통일 · 1mm 격자 · d5+d10 · 인덴터당 3 trial · 각 점 저~고force**. 기존 0.5mm×10 대비 스캔점 1/4·반복 1/2, force 범위·공정비교 확보.
- 근거: 지표 재평가(rel 저force 왜곡·d5 무접촉·loc 0.5mm) + 전이 리허설(warm 2 trial) + 혼합 리허설(rate 통일).
- 취득 후: warm-start fine-tune(A) + 표준 지표(loc+peak상관+rel저force제외+절대rmse). 밴딩 G0/G1·다점과 한 세션에.

### 3.A 센서 전이/일반화 실험 (2026-07-17 신설 — **논문 소재 확정**, 센서 파손 → 새 센서 대응)
- [x] **데이터 효율/전이 리허설 완료(2026-07-19)**: zero-shot 불가(0.66/0.89) · 1-pair 부족(0.38~0.47) · **2-pair warm 0.132·cross-warm 0.107(전 조건 최고)** → **최소 취득량 N=2 pair(4 trial) 확정**, warm-start 이득(성능+시간 1/10) 확정 (`experiments_archive/transfer_efficiency/transfer_report.md`)
- [x] **xy1 취득 + 0.25mm 출력 검증 완료**: 4 trial로 0.121 — coarse 스캔+fine 출력 성립 직접 증거
- [ ] 새 센서 제작 후: zero-shot → per-taxel 게인 보정 → xy1 소량(d5×N+d10×N, N은 리허설로 확정) warm-start fine-tune 3단 평가 → **C4(공용 플랫폼)·취득 효율 주장 보강**
- 개념 정리(본문 반영용): 취득 스캔 간격(xy1/xy0.5)과 출력 가상 taxel 해상도(0.5/0.25/0.1mm)는 **독립** — 출력 해상도는 연속 GT 덕분에 자유, 위치 정확도 제약은 센서 sparsity+GT 충실도(xy1 학습 loc 0.79mm < 스캔 간격 1mm)

### 3.5 하드웨어 (진행 중)
- [x] ~~D3 결정~~ → molding 변경·아랫면 소재 통일 **진행 중**
- [ ] 완료 시: 신규 시편 재취득 범위 판단 (Fig.2 갱신 여부)

---

## 4. §9 정량 수치 확보 현황 (Results 표)

| §9 항목 | 값 | 상태 / 출처 |
|---|---|---|
| Flat SR 위치오차 | ecomesh_xy1 **0.79 mm** (최종 xy0.5 1.13 mm) | ✅ `supplementary/S20_localization/loc_summary.csv` |
| Flat SR 상대 RMSE | 최종(A): d5_rel **0.188** / d10_rel 0.749(저force magnitude 잔존) | ✅ `experiments_archive/sizeA_final_xy0p5_diag/` (§3.4 보강으로 개선 여지) |
| SR scale factor | **≈105×** (41²/16; 다해상도 27×~2525× 안정 실증) | ✅ `supplementary/summary_metrics/` + `d5_final/` |
| 소재별 SR 비교 | d10_rel **ecomesh 0.182 < eco20 0.259 < eco50 0.336** · loc **0.79 < 1.02 < 1.27 mm** | ✅ `fig2_material_ablation/panelD_sats/` |
| 소재별 ΔS·확산·active taxel | §9.1 표 확보 + **3-set 평균±std 완료**(CV<2%, 결론 유지) | ✅ `fig2_material_ablation/Analysis_Results/Fig2_report.md` |
| **곡률 추정 MAE(°)/R²** | — | ⬜ P0 §2.2 |
| **Bent SR 보정 전/후 RMSE/R²·저하폭** | — | ⬜ P0 §2.2 |
| 다중 접점 분리 / 동적(슬립) 지표 | — | ⬜ P1 §3.1 (zero-shot) |
| Abstract 정량 수치 채움 (§1 각주) | — | ⬜ 위 수치 확정 후 일괄 |

---

## 5. 원고(본문) 작업

- [x] §4.2 재료 표기 확정 (D4) — 층별 적층 반영, 변형 시편 명시
- [ ] §4.2 mesh 사양(재질·피치) 기재
- [ ] §1 Abstract 정량 수치 채움 (§4 표 완성 후)
- [ ] §5.4 처리 전략 (a)~(d) 중 실제 채택안으로 본문 확정 (P2 결과 반영)
- [ ] §5.5 다점 zero-shot 실증 연결 (D1 — §3.1 결과)
- [ ] §9 Results 표 완성 → 본문 §6 Figure 캡션과 수치 일치 검증
- [ ] Supplementary 재구성: 현 `supplementary/`는 SATS 원논문 번호(S19 등) → **우리 논문 Supp 번호로 재편성** + d5 다해상도를 Supp figure로 승격 검토
- [ ] 참고문헌 정리 (`reference/` 선행논문 → 인용 목록화)

## 6. 투고 준비 (원고 완성 후)

- [ ] 타깃 저널 확정 → figure 규격(포맷·해상도·컬러)·워드리밋 맞춤
- [ ] figure 최종본: HTML figure_set → 저널 규격 개별 파일 export (생성 스크립트 경로 고정)
- [ ] 재현성 패키지: 코드·모델·데이터 공개 범위 결정 (learning_data는 대용량 git-ignored)
- [ ] 커버레터 / 하이라이트

---

## 7. 권장 실행 순서 (의존성 기준)

```
① 밴딩 취득(2.1) ──→ ② P1~P4(2.2) ──→ ③ Fig.3 완성(2.3)     ← critical path
   ∥ (같은 셋업 기간에 병렬 취득 권장)
④ 다점 2·3점 테스트 취득(3.1) → zero-shot 추론 실험 (재학습 0)
⑤ 저force d10 보강취득(3.4) → flat 재학습·figure 갱신
   ∥ (취득과 무관하게 병렬)
⑥ Fig.2 3-set 재분석(2.4, 기존 데이터) · Fig.1 일러스트(3.3)
⑦ Fig.4 데모(3.2) · molding 완료 대응(3.5)
   ↓
⑧ §9 표·Abstract 수치 채움 → 본문 §5 확정(5장)
⑨ Supp 재편성 → 저널 규격 export → 투고(6장)
```

취득 세션 효율화: ①④⑤는 같은 로봇팔+jig 셋업 기간에 묶어서 취득 (센서 탈부착·온도 조건 변화 최소화).
