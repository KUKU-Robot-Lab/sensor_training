# runs — SATS 학습 실행 기록 (폴더별 의미와 결과)

> 2026-07-16 정리, **2026-08-10 3-tier 분류 추가**. 수치 = 각 run `history.json`의
> **best val RMSE**(절대, a.u.). 세부 진단(d5/d10 분리·상대오차)은
> `history/fig_data/experiments_archive/` 각 폴더 참조. 실험 서사 전체는 Notion 연구일지.

## ★ 3-tier 분류 (2026-08-10)

| 티어 | 위치 | 내용 | 원칙 |
|---|---|---|---|
| **① 데모(deploy)** | 리포 루트 `runs/` (git 추적, `runs/README.md`) | 센서별 g05nsc SATS + deform 복원기 + estimator_v6_2 | `scripts/stage_deploy_runs.py`로 여기서 복사. 대시보드는 `runs/`를 1순위 스캔 |
| **② 논문 평가/실험** | 여기(`sats/training/runs/`) **제자리 유지** | 아래 "실험별 run 그룹" + 배포 계열 전체 | ★피규어/평가 스크립트가 경로를 직접 참조 — **이동 금지**(재현성) |
| **③ 잔재/중복** | — | arm4090 구 레이아웃 `sats/runs/`(로컬과 전수 해시 일치 확인) | 2026-08-10 `~/.trash_sats_runs_dup_20260810`로 격리(삭제 대기) |

## 배포(deploy) 계열 — 센서별 실기 추론용 (② 원본, ①은 여기서 복사)

| run | 의미 | 소재 |
|---|---|---|
| `ecomesh_v{5..9}_deploy_g05nsc` | ★현행 표준 — 클린 계보(v0 no-size 베이스 재학습 → warm), 41²@0.5mm, 크기입력 없음 | 양쪽 (v8/v9는 arm4090 학습 후 sync) |
| `ecomesh_v{5..9}_deploy_g05ns` | no-size 1세대(혼합 계보) — g05nsc 비교군 | 양쪽 |
| `(ecomesh_)v{5..9}_deploy_g025` | 81²@0.25mm 고해상(크기입력 있음, 십자 아티팩트 이슈) | 양쪽 (arm4090은 접두사 없음) |
| `(ecomesh_)v{5..9}_deploy_g01` | 201²@0.1mm | 양쪽 |
| `ecomesh_v{3,5,6}_deploy_all4` | 초기 배포(4-trial, 크기입력 있음) | 로컬(v6은 양쪽) |
| `ecomesh_xy0p5_base_nosize` | ★클린 warm 베이스(v0 데이터, no-size, val 0.2906) | 양쪽 |
| `ecomesh_v{1,3,5,6}_calibtransfer_*` | cross-sensor calibration transfer 실험(§전이) | 로컬 |

## bending 계열 (`sats/bending/runs/`, 양쪽 동일화 2026-08-10)

| 그룹 | 분류 | 내용 |
|---|---|---|
| `deform_restorer_v{5,7,8,9}` | ①데모 | 각도-프리 변형 복원기(z=32) — `runs/deform/<v*>/restorer.pt`로 복사됨 |
| `estimator_v6_2` | ①데모 | 현행 theta estimator — `runs/theta/v6/`로 복사됨 |
| `estimator_{v0,v5,v6,v6new,both}` | ②논문 | estimator 세대별 비교(G1 게이트 이력) |
| `est_{lstm,cnn2d,cnn2d_shape,cnn_lstm,mlp_frame,v6_2}` | ②논문 | 밴딩 구조 실험(LSTM 최선 결론의 근거) |
| `restorer_{v0,degonly}` | ②논문 | 구 restorer(deg_cnn 70.4% 비교군) |

## ★ 추론(infer)에 쓸 모델

| 용도 | run | 체크포인트 |
|---|---|---|
| **최종 flat 모델 (xy0.5, 크기입력 A)** | `size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3` | `best_model.pt` (d5_rel 0.188 / d10_rel 0.749 / loc 1.13mm) |
| 소재 비교 대표 (xy1, A) | `size_input_material/sizeA_{eco20_f2, eco50_f1, ecomesh_f3}` | Fig.2D 소스 |
| d5 전용 (β 물성보정, 0.5mm) | `d5_only_multires/d5only_beta_g0p5` | d5_rel ~0.15 |

로드 방법: `sats.bending.pipeline.load_frozen_sats(run_dir)` 또는 `sats.tools.eval_diagnostics._load_model`.
**주의**: A 모델은 추론 시 `model(sensor, lengths, size)`로 **인덴터 지름(size)을 반드시 전달**
(`eval_diagnostics.diagnose` 참조). 오프라인 진단 = `python -m sats.tools.eval_diagnostics --run-dir <run>`.

**실시간 추론** (`inference_engine`이 A 모델 자동 감지, d5 디폴트 조건 — size는 추론 대상이 아니라 magnitude 컨디셔닝):

```bash
.venv/bin/python -m sats.inference.run_realtime \
  --run-dir sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3 \
  --port /dev/ttyACM0 --viz 2d   # (옵션) --indenter-diameter-mm 5.0
```

## 실험별 run 그룹 (시간순)

| 그룹 | 무엇을 학습/검증 | 결과 요약 |
|---|---|---|
| `xy1_material_d5d10/` (9+1) | **Experiment A(구모델)**: xy1 소재 3종 × 3 fold | fold collapse 다수(best@e1~2). eco50_f1은 loadcell tare 교정 후 재학습(0.299), `_precorr_bak`=교정 전 백업(0.335) |
| `reg_probe/` | fold collapse에 정규화(dropout0.3+wd1e-3) 시도 | **무효 실증**(underfit 0.808) — 원인은 과적합이 아니라 데이터 다양성 |
| `ecomesh_resolution_controlled_d5d10/` (6) | **Experiment B**: ecomesh xy1 vs xy0.5, 6-trial controlled | xy1 healthy fold 0.133~0.143, xy0.5 0.319~0.426 (프로토콜 도메인 차이 확인의 시작) |
| `datarich_probe/` | ecomesh xy0.5 11-trial data-rich (구 최종 후보) | 0.283@e14, collapse 해소. d10 약점 발견의 기준 모델 |
| `pool_d10/` | xy0.5에 xy1 d10 3 rep 추가(pooling) | **null result**(홀드아웃 d10 불변) — xy1/xy0.5 프로토콜 도메인 불일치 실증 |
| `d10_only/` | d10만 학습(d5 편중 가설 검증) | 0.718 — d5 빼면 **더 나쁨** → "d5 편중이 원인" 가설 반증 |
| `size_input/` | **★크기입력(A, FiLM) 최종 flat 모델** | 0.2995@e4. d10 pedestal 개선(peak비율 1.35→1.14), 전 구간 최선 → **최종 확정** |
| `size_beta/`, `size_beta_gentle/`, `size_beta_physical/` | β(p) GT 물성보정 3변형 (A 위에) | 전부 **무이득 확정**(d10 저force 악화) — 인프라 보존용 |
| `size_input_material/` (3) | A로 소재 대표 fold 재학습 (eco20 f2 / eco50 f1 / ecomesh f3) | d10_rel **ecomesh 0.182 < eco20 0.259 < eco50 0.336** = Fig.2D |
| `ablation_ecomesh/` (3) | S19 모듈 ablation (구모델 베이스) | noCNN 0.209 < noLSTM 0.307 < noAttention 0.345 |
| `ablation_ecomesh_A/` (3) | S19 ablation (**A 베이스, 현행**) | noCNN 0.151 ≈ full < noLSTM 0.287 < noAttention 0.328 → attention 최핵심 |
| `d5_only_multires/` (4) | d5-only + β physical, 출력 grid 1.0/0.5/0.25/0.1mm | d5_rel ~0.15 해상도 무관 안정(SR 27×~2525×) = `supplementary/d5_final` |
| `sim_only_s8/` | Note S8 — 순수 EHS 시뮬 학습 | sim@sim 0.074 vs **sim@real 1.109(전이 실패)** → 실측 필수 근거 |

## 이름 규칙

`{실험}_{소재}_{취득스캔}_{fold|val 구성}_e2e_g05`
- `xy1`/`xy0p5` = 취득 스캔 간격(출력 해상도 아님 — 출력은 g05=0.5mm grid)
- `fold{N}` = trial-level 3-fold의 홀드아웃 선택 / `val_d5t10_d10t3` = d5 test10·d10 test3 홀드아웃
- `sizeA`/`sizeinput` = 크기입력(A) 모델, `beta*` = β GT보정 변형
