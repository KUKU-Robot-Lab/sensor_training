# scripts — 실험 러너·검증 스크립트 (구 저장소 루트 scratchpad_*)

> 2026-07-12 루트에서 이동. 모두 **저장소 루트에서 실행** (`bash scripts/xxx.sh` / `.venv/bin/python scripts/xxx.py`).
> GPU 학습 스크립트는 [[sats-env-venv]] 규칙대로 `.venv` 필수, 실행 전 GPU 점유 확인.
> 결과 해석은 `history/fig_data/experiments_archive/` 각 폴더 README/report 참조.

## 학습 러너 (시간순)

| 스크립트 | 용도 | 산출 run / 결과 |
|---|---|---|
| `scratchpad_run_expA.sh` | Experiment A — xy1 소재 비교 9 runs (구모델) | `runs/xy1_material_d5d10/` → `experiments_archive/xy1_material_d5d10/` |
| `scratchpad_run_expB.sh` | Experiment B — ecomesh xy1 vs xy0p5 controlled 6 runs | `runs/ecomesh_resolution_controlled_d5d10/` |
| `scratchpad_rerun_all.sh` | eco50 tare 교정 재학습 + ecomesh pooled + diag + figure 일괄 재생성 | `experiments_archive/pool_diag/pool_result.md` |
| `scratchpad_rollout_A.sh` | **A(크기입력) 최종 롤아웃** — 소재 대표 fold 재학습 (eco20 f2·eco50 f1·ecomesh f3) | `runs/size_input_material/` = 현행 소재 비교 모델 |
| `scratchpad_d5_multires.sh` | d5-only 다해상도 학습 (출력 1.0/0.5/0.25/0.1 mm, β physical) | `runs/d5_only_multires/` → `supplementary/d5_final/` |
| `scratchpad_run_ablation.py` | S19 ablation 3변형 재학습 (구모델 베이스 — legacy) | `runs/ablation_ecomesh/` |
| `scratchpad_run_ablation_A.py` | S19 ablation 3변형 재학습 (**A 베이스** — 현행) | `runs/ablation_ecomesh_A/` → `supplementary/S19_ablation/` |

## 시각화·검증

| 스크립트 | 용도 |
|---|---|
| `scratchpad_viz_maps.py` | 대표 GT/Pred 맵 생성 → `experiments_archive/final_maps/` |
| `check_gt.py` | (legacy) `gt_output_v1` GT 메타 sanity check — 현행 파이프라인은 gt_meta_cache 사용 |
| `verify_v2_matching.py` | (legacy) `gt_output_v2` 매칭 검증 — 동일 |

## logs/

러너 실행 로그 보관(git-ignored, `*.log`). 결과 수치는 이미 `experiments_archive/`·`supplementary/` report에 반영됨 — 로그는 디버깅 참고용.
