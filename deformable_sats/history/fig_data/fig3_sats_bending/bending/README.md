# Fig.3 밴딩 파트 — 데이터 취득 준비 (논문 §6 Fig.3 B–E)

> 상태: **데이터 취득 대기** (코드 Phase 0 + **합성 리허설 완료** — `rehearsal/rehearsal_report.md`).
> 리허설 결과(2026-07-13, `scripts/rehearse_bending_pipeline.py`): P1 deg MAE 0.77°(밴딩-only) ·
> P2 오프셋 제거율 98.5% · P3 동결 SATS(A, size 전달) 출력 회복률 97.8%.
> 발견 2건 수정/문서화: ①pipeline size 미전달 버그 fix ②B안 e2e 학습 시 cudnn RNN backward 제약
> (`torch.backends.cudnn.flags(enabled=False)` 필요). ⚠️ 밴딩+접촉 혼합 시 deg MAE 16.8°로 저하
> (밴딩-only 학습 estimator 기준) — 실데이터 사전검증 항목 그대로 유효.
> 목표: flat 학습 SATS를 동결한 채, 밴딩 baseline에서 곡률(signed deg)을 자가 추정하고
> flat 등가 신호를 복원해 밴딩 상태에서도 SR을 유지 (논문 C2).

## 논문 패널 매핑 (§6 Fig.3)

| 패널 | 내용 | 필요 데이터 | 생성 코드(예정) |
|---|---|---|---|
| (A) | jig·각도 정의 모식도 | 없음(모식도) | 신규 스크립트 or Figma |
| (B) | 각도별 Δbaseline vs z_i | 밴딩-only baseline | Phase 4 스크립트 |
| (C) | 곡률 θ 추정 회귀 (MAE/R²) | 밴딩-only baseline | `sats/bending/train_bending.py` → 평가 |
| (D) | flat vs bent SR 보정 전/후 RMSE/R² | 밴딩+접촉 | `sats/bending/pipeline.py` 평가 |
| (E) | 추론 vs GT SR 맵 | 밴딩+접촉 | 진단 덤프 + 맵 시각화 |

## 취득 스펙 (2026-07-18 NatComm G0/G1/G2 게이트 정합판 — 마스터: Notion "Bending-Aware Tactile Skin — Nature Communications")

**G0 취득 유효성 (전 세트 공통, 미충족 시 학습 중단·프로토콜 수정)**:
- signed 곡률 GT **독립 측정**(jig 각도 눈금 검증 + 가능하면 IMU 교차확인)
- 무접촉/접촉 구간과 16ch 스트림 **동기화** 확인
- 메타 기록: 센서 ID·온도·retare 시각·mounting session·각도·인덴터·위치·힘
- raw→source data까지 trial registry 추적 (`learning_data/trial_registry.json` 규약 재사용)

1. **밴딩-only baseline** (Phase 1·패널 B/C — **G1 조건**):
   - −40°~+40°, 10° 간격, 각도별 무하중 10 set. flat(0°) 포함.
   - **독립 remounting ≥3 session** (센서 탈부착 후 재취득 — G1 held-out 축)
   - 평가 홀드아웃: **intermediate angle**(학습 각도 사이) + **remounting session** 단위
   - 온도/장기 drift 로깅 — drift가 각도 예측을 설명하지 않음을 보여야 함
2. **밴딩+접촉** (Phase 2·3·패널 D/E — **G2 조건**): 0°, ±20°, ±40° × d5 인덴터,
   xy0.5 계단식 동일 프로토콜(도메인 일치). 좌표·fz GT 동시 로깅. **same-session flat reference 필수**.
3. **flat 기준**: 동일 세션 flat 1 set (G2의 "corrected bent ≤ 1.5× same-session flat" 판정 기준).

## npz 컨트랙트 (`sats/bending/dataset.py`)

trial별 `.npz`:
- `sensor` float[N,16] — 16ch 시계열
- `bend_deg` float[N] — **signed** 곡률 각도 GT (지그/IMU)
- `contact` float[N,3] — 선택, (x, y, fz). 밴딩-only는 생략
- 저장 위치: `learning_data/sensor_raw_bin/<mat>_bend/`

## 취득 후 실행 순서 (P1→P4, G2 비교군 포함)

1. P1 estimator: `train_bending.train_estimator()` → deg MAE (intermediate-angle·remount 홀드아웃)
2. P2 restorer: 오프셋 지도(A안) 또는 end-to-end(B안, cudnn off 필요)
3. P3 pipeline — **G2 5-비교군**: ①uncorrected ②flat-baseline subtraction(naive)
   ③proposed correction ④curvature-conditioned 모델 ⑤per-curvature retraining(상한선).
   판정: 전 비-zero 곡률 집계에서 ③>① 개선, paired 개선 95% CI>0, 홀드아웃 trial ≥80% 개선,
   corrected rel RMSE ≤ 1.5× same-session flat, 독립 remount session에서 재현.
4. P4 figure: 패널 B–E + calibration burden 비교(⑤ 대비 취득량) — NatComm Fig.3 구성과 일치

## 사전 검증 (리스크 방어)

- 밴딩·접촉 신호의 **시간 스케일 차이·중첩 선형성**을 취득 초기 데이터로 먼저 확인
  (분리 ill-posed 리스크). 곡률 GT(지그 각도) 신뢰도 체크.
