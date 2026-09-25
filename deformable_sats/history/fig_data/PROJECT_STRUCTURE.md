# fig_data 프로젝트 구조 (매 세션 탐색 단축용)

> 유연 촉각 센서(SATS 기반 super-resolution) 논문의 **Figure 제작 + 데이터 분석** 워크스페이스.
> 논문 본문: `Development_of_a_Flexible_Tactile_Sensor_with_Super-Resolution_Capability_for_Robust_Robotic_Manipulation.md`
> 센서 사양: SATS 기반 30×30×5.5mm, FPCB + MEMS Barometer, 3중층(Top: Ecoflex+mesh / Mid: MEMS embedded / Bot: 베이스). taxel 16개 = 4×4 격자(간격 6.5mm, ±9.75mm).
>
> **2026-07-12 재배치**: 폴더 = 논문 §6 Figure Plan 1:1 대응. 과정 산출물은 `experiments_archive/`, 원본은 `archive/`.

## 최상위 디렉토리 (논문 §6 매핑)

| 경로 | 논문 | 역할 · 상태 |
|---|---|---|
| `Development_...md` | 본문 | working draft (Fig.1~4 계획 §6, 체크리스트 §7) |
| `fig1_concept/` | Fig.1 | 컨셉 일러스트 — **미제작**, 후보 소재는 README |
| `fig2_material_ablation/` | Fig.2 (C1) | 소재 ablation: 패널 A~C(`Analysis_Results/`) + **패널 D**(`panelD_sats/` — 소재별 SATS 성능, 구 fig3 소재비교 이동) |
| `fig3_sats_bending/` | Fig.3 (C2) | `algorithm/`(구조도) + `flat_sr/final_xy0p5/`(완료) + `bending/`(**취득 대기**, 스펙 README) |
| `fig4_application/` | Fig.4 | 로봇핸드/사람손 데모 — 미착수 |
| `supplementary/` | Supp. | S19 ablation·S20 localization·S29 t-SNE·S30 regression·summary_metrics(SATS 원논문 번호, README 참조) + **d5_final**(d5-only 다해상도, §6 비패널이라 여기 배치) |
| `experiments_archive/` | — | 과정 진단·null result 보존 (**sizeA_diag = 현행 A 모델 진단 데이터 소스**) |
| `visualizing_scripts/` | — | Figure 생성 Python (figure↔스크립트 매핑은 각 폴더 README) |
| `reference/` | — | 선행 논문(SATS, TVI/Barodome 등) |
| `archive/` | — | 사진/·pptx·구모델 백업 |
| `SATS_paper_implementability.md` | — | SATS 논문 figure별 재현 가능성 판정표 |

## 모델 현황 (2026-07-07 확정)

- **최종 베이스 = 크기입력(A, FiLM indenter-size conditioning) 단독**. β GT보정은 인프라만 보존(무이득 확정).
- 소재 서열(d10 상대오차): **ecomesh 0.182 < eco20 0.259 < eco50 0.336** (가설 지지).
- 최종 flat: ecomesh xy0.5 `runs/size_input/`, 소재비교: `runs/size_input_material/` (대표 fold: eco20 f2·eco50 f1·ecomesh f3).
- d5-only 다해상도: d5_rel ~0.15로 SR 27×~2525× 안정 (`experiments_archive/d5_multires_diag/`).

## fig2_material_ablation 데이터 (구 fig2_heatmap)

세 실험 조건:
- `CenterLine/` — 1D 라인 스캔(y 고정, x 스위프). CSV 처리·PNG 완료. `combined_centerline.csv` 병합본.
- `CenterPress/` — 중심 고정 깊이 스윕. PNG 결과만 존재.
- `xy_1mm/` — **2D 격자(±10mm, 1mm 간격) 압입. Fig.2B 핵심 데이터.** 대부분 raw `.bin`, 일부만 CSV 변환.
- `hitmap/` — coverage/overlap/spreading 분석 PNG + md.
- `Analysis_Results/` — 패널 A~C 산출물 + `Fig2_report.md`(통합 보고서).
- `panelD_sats/xy1_material/` — **패널 D**: 소재별 SATS 패널 `Fig2D_{A~G}_*`(2026-07-12 구 Fig3*에서 개명·재생성) + `shared_axes/` + `Fig2D_report.md`.

소재/인덴터 폴더 규칙: `xy_1mm/{ec020|eco50|ecomesh}/d{5|10}/YYYYMMDD_testN/`
(주의: `ec020`은 eco20 오타. mesh는 `ecomesh`/`eco20 + mesh` 혼용.)

### CSV 포맷 (각 test 폴더)
- `due_data.csv`: `elapsed_ns,time_s,burst_index,frame_index,Skin1..Skin16` — 센서 raw ADC(~7.0e6 offset). **1 burst = 10 frame**.
- `ethermotion_data.csv`: `...,X,Y,Z,U,...,Z_lCmd_mm` — 인덴터 위치. **X/Y/Z 단위 = encoder pulse, ×1e-4 = mm** (±100000 pulse = ±10mm; Z max 14mm).
- `loadcell_data.csv`: `elapsed_ns,time_s,kg` (×9.80665 = N).
- `afd50_data.csv`: 6축 F/T — 거의 미사용(0행).
- `manifest.json`: 변환 메타(일부 폴더). `due_payload_layout: sensor[16][frame10]`.

### ΔS(센서 응답) 정의
`ΔS_i = -(raw_i - baseline_i) / baseline_i * 100` (%). baseline = 무접촉 초기 구간 평균. 부호 반전으로 압력 증가를 양수로.

## visualizing_scripts (주요)

- `figure_set/generate_fig3_sats.py` — **★SATS 패널 생성기** `--figset {xy1_material|d5_final|xy0p5_final}` (+`--shared-axes`, `--ref-limits`). 출력: xy1_material→`fig2_material_ablation/panelD_sats/xy1_material/`, 나머지→`fig3_sats_bending/flat_sr/`.
- `figure_set/generate_fig3_forcematched_d10.py` — force-matched d10 소재 비교 → panelD.
- `figure_set/generate_supp_{ablation,localization,attention_tsne,regression,summary}.py` — supplementary S19/S20/S29/S30/summary → `supplementary/`.
- `figure_set/viz_d5_*.py`, `compare_*.py`, `verify_d10_magnitude.py`, `analyze_data_quality.py` — 진단·검증 → `experiments_archive/`.
- `figure_set/build_figure_set_html.py` — 통합 HTML figure set(base64 내장, pptx 흡수). **2026-07-12 §6 번호 재편**: Fig.2(C1)→Fig.3(밴딩 C2)→Fig.4(application placeholder)→Supp. Fig. B1(모델 벤치마크, 구 Fig.3).
- `xy_1mm/generate_{2d_heatmap,panelC_metrics,panelA_schematic,centerpress_depth,betweentaxel,...}.py` — Fig.2 A~C → `fig2_material_ablation/Analysis_Results/`.
- `CenterLine/*.py` — 1D 라인 스캔 처리·시각화.

원칙: **모든 figure는 생성 스크립트와 매핑 기록** (애드혹 금지).

## sensor_training 저장소 (취득 + 학습)

- `skin_ws/acquisition_code/` — 멀티센서 로깅(`final_logger*`), `convert_bins.py`(.bin→CSV).
- `sats/` — 공식 SATS 파이프라인: preprocessing(bin_merge→GT) → training(4단계 + e2e) → inference(realtime) → tools(eval_diagnostics 등).
- `sats/bending/` — **밴딩 보상 프론트엔드** (Phase 0 완료): 곡률 signed deg 추정 + flat 등가 복원 + 동결 SATS.
- `learning_data/` — merged.bin + GT meta cache(대용량, git-ignored). `trial_registry.json`·`trial_indices/`.
- `hitmap/`, `cnn_lstm/` — legacy/사이드.

## 데이터 파이프라인 요약
```
취득(skin_ws final_logger) → raw .bin
  → convert_bins.py → due/ethermotion/loadcell CSV   (Figure 분석용)
  → sats/preprocessing/bin_merge.py → merged.bin → GT(gpu_on_the_fly)   (학습용)
  → SATS e2e 학습(A 크기입력) → eval_diagnostics --dump-samples → npz
시각화: visualizing_scripts/* → fig2/fig3/supplementary 폴더 PNG + report
```

## 다음 할 일

**→ `SUBMISSION_CHECKLIST.md`** (투고 로드맵 마스터 — P0 critical path = 밴딩 취득→P1~P4→Fig.3, 결정 필요 D1~D4, §9 수치 확보 현황 포함).
