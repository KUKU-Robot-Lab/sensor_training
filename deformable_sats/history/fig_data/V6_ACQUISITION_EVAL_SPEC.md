# v6 취득·평가 스펙 (재구성 모델 평가 3종, 2026-08-02)

> 목적: v6 센서의 **3종 데이터로 학습된 재구성 모델(SATS+밴딩)의 추론 정확도**를 평가하고,
> 각 시나리오의 **어플리케이션 히트맵**을 남긴다. "정책=모델 추론(압력맵 재구성)".
> 상위: Notion "Bending-Aware Tactile Skin — NatComm". 밴딩 구조는 [G1_ACQUISITION_DESIGN](fig3_sats_bending/bending/G1_ACQUISITION_DESIGN.md).

## 3종 데이터 → 논문 매핑

| 데이터 | 평가 대상 | 게이트/그림 | 학습? |
|---|---|---|---|
| ① 단일점 랜덤 press | SATS 단일접촉 재구성(위치·깊이·힘), **연속 랜덤위치 일반화** | Fig 2/3 성능 | 학습 가능 or 홀드아웃 |
| ② 다중점 랜덤 press | 2·3점 분리·최소거리·force imbalance | Fig 4 (stretch) | **평가 전용(zero-shot)** |
| ③ 밴딩+접촉 press | **G2 — 재학습 없이 곡률 보정이 접촉 재구성 회복** | Fig 3 핵심 | **평가 전용(no-retraining)** |

★ **②③은 절대 학습에 넣지 말 것.** "다중점 zero-shot", "밴딩접촉 재학습 없이 회복" claim이 무너진다.

## 공통 필수 (모든 데이터)

- 취득 병합 포맷 = 기존 `*_merged.bin` (컬럼 `s1..s16, x_mm, y_mm, z_stage_mm, z_depth_mm, Fz`).
- **GT 동기화 로깅**: 접촉 위치(stage `x_mm,y_mm`) + 힘(loadcell `Fz`) 프레임 정합.
- 무접촉 baseline 확보(각 세션 시작 flat 구간) — pct 표현 `(s−base)/base×100`의 기준.
- 온도·retare 시각 로깅 추가 권장(G0 drift 통제 보완).

## ① 단일점 랜덤 press

- 취득: 랜덤 (x,y) 위치에서 계단식 z press(저~고 force). 격자 아님 = 위치 일반화 테스트.
- **평가**: 기존 `scripts/reeval_map_quality.py` 재사용 — 새 trial을 run의 val-trials로 지정.
  지표 = loc(argmax vs GT xy)·peak_corr·peak_ratio·force. 신규 코드 불필요.
- 사진: 추론 41×41(표시용 0.1mm) 히트맵 1개 피크.

## ② 다중점 랜덤 press

- 취득: 2점(→3점) 동시 접촉, 랜덤 위치. **각 접촉점의 위치·힘 GT 필수**.
  - 단일 stage로는 1 (x,y)만 기록됨 → **다접촉 GT는 사이드카로 제공**:
    세션당 `contacts_gt.csv` (컬럼 `frame_or_time_s, k, x_mm, y_mm, Fz`; k=접촉 인덱스).
    또는 고정 다인덴터 지그면 위치를 메타(json)로 1회 기록.
- **평가**: `scripts/eval_multicontact.py` (신규) — SATS 맵에서 peak 검출→GT 매칭.
  지표 = 검출율/오검출, 매칭 위치오차, **최소 분리거리**(두 점 구분 한계), force imbalance 견딤.
  순수 로직 = `sats/tools/multicontact_metrics.py` (테스트 포함).
- 사진: 여러 피크 히트맵(2·3점).

## ③ 밴딩+접촉 press (G2 — 최우선)

- 취득 (★ 핵심 요건):
  1. **곡률 단계 고정**(예 δ=Y−origin ∈ {0, 5, 10mm}; v5 방식 Z=12 후 Y로 δ) 각 단계에서 접촉 press.
  2. **같은 위치의 flat(δ=0) 기준 접촉을 같은 세션에 함께 취득** — "flat 등가 회복" 비교 불가 시 G2 무효.
  3. 접촉 위치·힘 GT + 곡률(δ) 로깅. 곡률은 sensor 자가추정(estimator)으로도 얻지만, 검증용 δ GT 병기 권장.
  4. 독립 remounting ≥2 세션 반복(G2도 remounting 재현 요구).
- **평가**: `sats/bending/eval_g2.py` (신규) — 4-조건 비교:
  - `reference` = 같은위치 flat 접촉의 SATS 맵(정답 상한)
  - `uncorrected` = 밴딩+접촉 pct → 동결 SATS (곡률 오염)
  - `corrected` = estimator(pct)→restorer→동결 SATS (제안 보정)
  - `flat-sub`(선택) = 단순 flat baseline 차감
  지표 = 보정 후 위치오차·force가 flat 수준 회복률, 곡률(δ)별 분해. **준-합성(eval_contact_preservation)을 실측으로 대체**.
- 사진: 같은 접촉의 무보정(가짜/왜곡) vs 보정(회복) 히트맵 나란히 — 논문 Fig 3 킬러 컷.

## 평가 vs 사진

- **정량(게이트 판정)**: 위 지표를 GT와 비교. G2는 "보정이 uncorrected보다 개선, flat의 1.5배 이내" 등.
- **사진(정성 데모)**: 같은 데이터에서 추론맵을 0.1mm grid로 렌더(엔진 다해상도). 텍스트 영어만.

## 준비된 하네스 (v6 도착 즉시 실행)

| 데이터 | 도구 | 상태 |
|---|---|---|
| ① | `scripts/reeval_map_quality.py` (기존, MODELS/val-trials에 v6 추가) | ✅ |
| ② | `sats/tools/multicontact_metrics.py` + `scripts/eval_multicontact.py` | ✅ 준비 |
| ③ | `sats/bending/eval_g2.py` | ✅ 준비 |

> 하네스는 위 데이터 포맷을 가정한다. v6 실제 컬럼/사이드카가 다르면 로더부만 조정(로직 재사용).
