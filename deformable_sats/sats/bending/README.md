# 밴딩 보상 프론트엔드 (`sats/bending/`)

flat 학습 SATS를 **동결**한 채, 밴딩 상태 센서 신호에서 ①밴딩 곡률(부호 있는 deg)을 추정하고
②flat 등가 baseline을 복원해 ③동결 SATS에 넣어 **재학습 없이** 밴딩+압력 추론을 한다.
목적 = **곡률마다 접촉 데이터를 재수집하지 않는 것**(무접촉 밴딩만으로 보정).

## 구조
```
밴딩% [T,16] → LSTM → MLP-A → signed deg      (BendingEstimator)
밴딩% + deg  → MLP-B → 오프셋; %−오프셋 = flat등가  (BaselineRestorer)
flat등가 → ❄️Frozen SATS → 압력맵            (BendingPipeline)
```
★ **SATS 입력 = 상대변화% `(s_raw−baseline)/baseline×100`** (`training/dataset.py:248`).
raw(~7e6)를 넣으면 포화되므로 밴딩도 반드시 이 %표현으로 변환해 다룬다.

## 파일
| 파일 | 역할 |
|---|---|
| `config.py` | `BendingConfig` (LSTM/MLP, deg_scale, 동결 SATS 경로) |
| `dataset.py` | 로더·윈도잉 + **`valid_only` 필터·`NormStats`·`build_windows`** |
| `geometry.py` | 원호 좌굴 δ→κ,θ,R + 자가추정 κ̂ + 캘리브레이션(순수함수) |
| `bending_estimator.py` | LSTM+MLP → signed deg |
| `baseline_restorer.py` | deg 조건 오프셋 → flat 등가(zero-init 항등) |
| `pipeline.py` | estimator→restorer→동결 SATS, `load_frozen_sats()` |
| `train_bending.py` | **Phase1 estimator + Phase2 restorer 학습** + CLI, save/load |
| `eval_pipeline.py` | **Phase3** 검증 — 환각 억제·곡률별 오차(e2e) + 그림 |
| `eval_g1.py` | **G1 관측성 하네스** — leave-one-session-out MAE·Spearman·drift |
| `eval_contact_preservation.py` | **준-합성 접촉 보존 검증**(기존 데이터만) — restorer 모드 비교 |
| `tests/` | Phase 0 TDD (합성) |

전처리(raw→npz)는 `sats/preprocessing/prepare_bending_data.py`. 취득 기구·곡률 정의·유효범위는
`history/fig_data/fig3_sats_bending/bending/G1_ACQUISITION_DESIGN.md`.

## 데이터 사양
trial별 `.npz`(`learning_data/bending/<ver>/`): `sensor`[N,16] raw, `baseline`[16],
`bend_deg`[N] signed θ, `kappa_geo`·`kappa_hat`·`delta_mm`·`valid`(|δ|≤상한). 곡률 GT는
**스테이지 압축 원호모델**(지그/IMU 아님). ★ **유효 |δ|≤~10mm** — 초과 시 센서 포화(κ̂ 비단조 붕괴).

## 단계 (진행 현황, v0 = 단일세션 무접촉 밴딩)
- **Phase 0** ✅ 스캐폴드·모듈·테스트.
- **Phase 1** ✅ estimator: valid+표준화+trial split. holdout θ **MAE 6.3°**(기준선 30°).
  `python -m sats.bending.train_bending --phase estimator --val-trials <holdout>`
- **Phase 2** ✅ restorer(A 오프셋지도). **restorer_mode=deg_only 기본**(§5.3, 접촉 보존):
  무접촉 오프셋 제거 43%(deg_only) / 67.9%(seq_deg). `... --phase restorer`
- **Phase 3** ✅(예비) 밴딩→flat→동결SATS: 무접촉 환각 억제 **60%**(deg_only) / 88%(seq_deg),
  e2e(예측곡률)=oracle 동등.
  `python -m sats.bending.eval_pipeline --analyze --estimator <ckpt> --figure out.png`
- **접촉 보존(준-합성)** ✅ 기존 데이터만으로 검증: 실측 접촉+v0 밴딩 오프셋 합성 →
  **★ seq_deg restorer는 접촉 파괴(위치 2.7→4.6mm, 회복 −127%), deg_only는 보존+개선
  (2.7→2.3mm, +16%)**. 이 결함 발견으로 restorer 기본을 deg_only로 교정.
  `python -m sats.bending.eval_contact_preservation --figure out.png`
  ⚠ 가법성 가정·다른 유닛 합성 → 상대 비교가 핵심, 절대 회복률은 지시적. 독립 증명 = G2.
- **G1(관측성)**: 하네스 완비. **정식 판정 = remounting ≥3세션 무접촉 취득 필요**(v0는 스모크).
- **Phase 4**: figure/README 최종.

## 참고
- 그래프 텍스트는 **영어만**(matplotlib 한글 □ 깨짐).
- 원호 근사는 clamped-BC라 엄밀 균일호 아님 → κ̂ 교차검증이 유효범위 실증(valid R²=0.82).

## 재현
```bash
.venv/bin/python -m pytest sats/bending/tests/ -q
```
