# runs_comparison 후속 학습 결과 분석

작성일: 2026-04-10

## 요약 판단

후속 변경으로 평가 프레임워크의 주요 누락은 해결됐다. `training/runs_comparison/metrics_multi_head_field_stage3_dlabel-gaussian-hertz_xybce1_zhuber0p2_fzhuber0p2_decsoftargmax.json`은 이제 `[x, y, z, fz]` 순서의 metric schema와 `per_output.fz`를 포함한다. 같은 실행에서 `comparison_results.json`도 `multi_head_field`의 4D metric을 담고 있다.

현재 워크트리에 존재하는 `ecomesh` 후속 결과 기준 최선은 Stage3(`depth-aware soft heatmap + z/Fz 보조 head`)다. MAE는 `[x, y, z, fz] = [0.669, 0.482, 0.086, 0.375]`, R2는 `[0.960, 0.986, 0.758, 0.289]`이다. xy/z는 1차 분석보다 더 신뢰 가능한 형태로 저장됐고, Fz도 이전 `ecemesh` 문서 수치보다 개선되어 양수 R2가 나왔다. 다만 Fz R2 0.289는 아직 강한 일반화 성능으로 보기 어렵다.

이번 결과는 trial-level split과 zarr별 index 구조를 반영한다. `preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr` 기준 seed 42 split은 train samples 79,437개, val samples 16,072개, test samples 0개다. train trials는 `[ecomesh_d10_z1.0_test1, ecomesh_d10_z1.0_test2, ecomesh_d10_z1.0_test3, ecomesh_d10_z1.5_test1, ecomesh_d5_z1.0_test1, ecomesh_d5_z1.0_test2, ecomesh_d5_z1.5_test1, ecomesh_d5_z1.5_test2, ecomesh_d5_z1.5_test3, ecomesh_d5_z1.5_test4, ecomesh_d5_z1.5_test6, ecomesh_d5_z1.5_test7, ecomesh_d5_z1.5_test8]`, val trials는 `[ecomesh_d5_z1.0_test3, ecomesh_d5_z1.5_test5, ecomesh_d5_z1.5_test9]`, test trials는 `[]`이다. metric 집계는 `depth_min_for_label` 필터 후 depth bin count 기준 14,543개다.

주의: 이전 분석 문서의 `dataset_ecemesh.zarr` 및 `preprocessing/processed_data_reindexed` 경로는 현재 워크트리에 존재하지 않는다. 따라서 이 문서는 현재 재학습된 `ecomesh` 산출물을 기준으로 갱신한다.

## 변경 전 이슈와 후속 상태

| 항목 | 1차 상태 | 후속 상태 |
|---|---|---|
| Fz metric | loss에는 있었지만 JSON metric 누락 | `[x,y,z,fz]` schema, `per_output.fz`, `fz_summary_*.csv` 저장 |
| train/val leakage | sequence sample 단위 random split | `sample_trial_ids` 기반 trial-level split |
| Zarr/index 불일치 | 부모 `dataset_index.json` 덮어쓰기/자동 zarr 선택 위험 | zarr 내부 `dataset_index.json` 우선 사용, index의 `zarr_path` 검증, 여러 zarr 자동 선택 거부 |
| Stage 구분 | tag가 stage/lambda를 충분히 드러내지 않음 | `stage1/2/3`, label, loss, lambda, decode가 파일명에 반영 |
| 문서 명령 | 일부 예시가 실제 산출물과 불일치 | `README.md`, `training/README.md`에서 명시적 `--zarr-path`와 실제 stage3 checkpoint 반영 |

## 후속 실험 결과

Metric 순서는 모두 `[x, y, z, fz]`이다.

| 결과 파일 | MAE | RMSE | R2 | 판단 |
|---|---:|---:|---:|---|
| `metrics_multi_head_field_stage1_point_xybce1_zoff_fzoff.json` | `[0.782, 0.506, 1.097, 1.124]` | `[1.533, 0.977, 1.132, 1.251]` | `[0.931, 0.975, -19.110, -4.024]` | point label baseline, z/Fz scalar는 실패 |
| `metrics_multi_head_field_stage2_dlabel-gaussian-hertz_xybce1_zoff_fzoff_decsoftargmax.json` | `[0.741, 0.518, 1.140, 0.952]` | `[1.231, 0.924, 1.175, 1.081]` | `[0.955, 0.978, -20.668, -2.752]` | xy는 유지/개선, z/Fz head off라 scalar는 의미 없음 |
| `metrics_multi_head_field_stage3_dlabel-gaussian-hertz_xybce1_zhuber0p2_fzhuber0p2_decsoftargmax.json` | `[0.669, 0.482, 0.086, 0.375]` | `[1.173, 0.739, 0.124, 0.470]` | `[0.960, 0.986, 0.758, 0.289]` | 현재 최선, Fz는 개선됐지만 추가 검증 필요 |

Depth bin 기준 Stage3:

| depth bin | count | xy MAE | success <= 1 cell |
|---|---:|---:|---:|
| 0.8-1.1 mm | 6,102 | 1.158 mm | 21.8% |
| 1.1-1.4 mm | 5,084 | 0.610 mm | 46.7% |
| 1.4-1.7 mm | 3,357 | 0.615 mm | 51.4% |

해석은 명확하다. Stage3는 Stage1/2 대비 xy, z, Fz를 모두 개선했고 `selection_mae_xyz`도 0.412로 가장 낮다. 0.5 mm 한 셀 이내 성공률은 깊이 1.1 mm 이상에서 46-51%까지 올라왔지만, 얕은 0.8-1.1 mm 구간은 21.8%로 아직 약하다. Stage3의 z는 MAE 0.086 mm, R2 0.758로 의미 있는 개선이고, Fz는 MAE 0.375, R2 0.289로 이전보다 나아졌지만 별도 holdout test 없이 충분하다고 판단하기는 이르다.

## 재현 명령

후속 결과의 stage3 파일명과 맞는 명령은 다음과 같다. `_hnorm`이 붙은 체크포인트를 재현하려면 `--normalize-heatmap`을 추가해야 하지만, 현재 저장된 후속 산출물명에는 `_hnorm`이 없다.

```bash
python -m training.pipelines.train_comparison \
  --data-dir preprocessing/processed_data \
  --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
  --models multi_head_field \
  --use-depth-aware-label \
  --loss-xy bce --loss-z huber --loss-fz huber \
  --lambda-xy 1.0 --lambda-z 0.2 --lambda-fz 0.2 \
  --depth-label-kernel gaussian --depth-radius-model hertz \
  --heatmap-size 40 --fg-weight 8.0 --heatmap-sigma-scale 0.35 \
  --decode-xy softargmax \
  --depth-fallback-mm 1.0 --depth-min-for-label 0.05 \
  --save-heatmap-overlay --overlay-batches 1 --overlay-samples 4 \
  --epochs 100 --batch-size 1024
```

추론/overlay 확인:

```bash
python inference/run_inference.py \
  --checkpoint training/runs_comparison/best_multi_head_field_stage3_dlabel-gaussian-hertz_xybce1_zhuber0p2_fzhuber0p2_decsoftargmax.pth \
  --data-dir preprocessing/processed_data \
  --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
  --decode-xy softargmax \
  --heatmap-sigma-scale 0.35 \
  --depth-fallback-mm 1.0 --depth-min-for-label 0.05 \
  --batch-size 256 --max-batches 2 \
  --save-heatmap-overlay --overlay-batches 1 --overlay-samples 4
```

## 검증 요약

- `env PYTHONNOUSERSITE=1 conda run -n sensor python -m unittest tests.test_trial_split tests.test_zarr_index_resolution tests.test_multi_head_fz_metrics`: 17 tests 통과.
- 현재 `ecomesh` sequence split 확인: total 95,509, train 79,437, val 16,072, test 0. `fz_summary_*.csv`는 header 포함 각 16,073 lines로 val sample 수와 일치한다.
- overlay 확인: `training/runs_comparison/overlays/multi_head_field_e100_overlay_b0_i0.png`부터 `i3.png`까지 780 x 300 RGBA PNG로 생성됐다.
- `env PYTHONNOUSERSITE=1 conda run -n sensor python -m py_compile ...`: 변경된 Python 파일 및 테스트 파일 컴파일 통과.
- `python3 -m pytest ...`: 실패. 원인은 코드가 아니라 현재 기본 `python3` 환경의 NumPy 2.2.6과 시스템 SciPy/sklearn ABI 불일치(`numpy.core.multiarray failed to import`)다.
- secrets scan: 실제 secret 없음. `api_key`, `secret`, `token` 검색 결과는 문서 프롬프트 문구와 변수명 `token`뿐이다.
- unsafe file operation review: 새 쓰기 경로는 metrics JSON, Fz summary CSV, overlay PNG, zarr 내부 `dataset_index.json` 생성이다. 임의 삭제/외부 경로 삭제는 없다.

## 남은 리스크

- Fz는 개선됐지만 아직 약하다. Stage3 Fz MAE 0.375, R2 0.289라 별도 holdout test 없이 일반화 성능을 확정하기 어렵다.
- 현재 test split은 비어 있다. val trials는 `ecomesh_d5_z1.0_test3`, `ecomesh_d5_z1.5_test5`, `ecomesh_d5_z1.5_test9`이며, 다음 검증에서는 명시적 `--test-trials`가 필요하다.
- `training/runs_comparison/`와 `preprocessing/processed_data/`의 zarr는 큰 산출물이다. 커밋 대상인지, 재생성 대상인지 별도 정책이 필요하다.
- 이전 문서의 `preprocessing/processed_data_reindexed/dataset_ecemesh.zarr` 기준 결과는 현재 워크트리에서 재현할 수 없다.
- 기본 `python3` 환경은 테스트 실행에 부적합하다. 다음 실험자는 `sensor` conda 환경에서 `PYTHONNOUSERSITE=1`을 켜고 실행해야 한다.

## 다음 데이터 수집/실험 제안

1. `--test-trials ecomesh_d5_z1.5_test9`처럼 holdout trial을 명시하고 Stage2/Stage3를 다시 돌린다.
2. leave-one-trial-out으로 `ecomesh` trial들을 한 번씩 val/test로 둔다.
3. Fz는 소재별/깊이별 normalization을 분리하거나, Fz head loss weight와 target scale을 grid search한다.
4. `ecemesh` zarr를 복구할 수 있으면 material holdout(`ecemesh` train -> `ecomesh` test 또는 반대)을 따로 측정한다.
5. success <= 1 cell이 낮은 얕은 깊이 구간을 대상으로 softargmax temperature, argmax_refine, heatmap sigma를 같은 split에서 비교한다.
