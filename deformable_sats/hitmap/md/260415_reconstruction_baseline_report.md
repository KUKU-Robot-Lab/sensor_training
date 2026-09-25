# Reconstruction Baseline Report

## Official Baseline Table

| Group | Run | Role | Split | Condition | x MAE | y MAE | z MAE | fz MAE | Note | Source |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| current_xyzf | multi_head_field_stage3_cv5 | official | cv5_val_only | direct_model_output | 1.3308 | 0.5442 | 0.1612 | 0.4249 | trial-aware 5-fold CV aggregate | `/home/user/sensor_training/training/runs/runs_comparison/comparison_results.json` |
| current_zfz | z_fz_regressor_gt_xy | reference_only | cv5_val_only | gt_xy+gt_radius | - | - | 0.136 | 0.354 | upper-bound reference; not deployment condition | `/home/user/sensor_training/training/runs/runs_z_fz/cv_summary_z_fz_regressor.json` |
| current_zfz | z_fz_regressor_predicted_xy | separated_upper_bound | cv5_val_only | pred_xy+gt_radius | - | - | 0.1356 | 0.356 | radius still GT; keep separate from main score table | `/home/user/sensor_training/training/runs/runs_z_fz/cv_summary_z_fz_regressor.json` |
| legacy_0409 | min10_cnnlstm | official | aggregate_legacy | legacy_direct_regression | 1.5895 | 1.2908 | 0.1738 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/min10/comparison_results.json` |
| legacy_0409 | min10_sats | official | aggregate_legacy | legacy_direct_regression | 0.7426 | 0.351 | 0.141 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/min10/comparison_results.json` |
| legacy_0409 | min10_sats_xy | official | aggregate_legacy | legacy_direct_regression | 0.8199 | 0.3808 | 0.1281 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/min10/comparison_results.json` |
| legacy_0409 | min20_cnnlstm | official | aggregate_legacy | legacy_direct_regression | 1.6113 | 1.2174 | 0.1339 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/min20/comparison_results.json` |
| legacy_0409 | min20_sats | official | aggregate_legacy | legacy_direct_regression | 4.8252 | 5.7913 | 0.1414 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/min20/comparison_results.json` |
| legacy_0409 | min20_sats_xy | official | aggregate_legacy | legacy_direct_regression | 4.8272 | 5.8102 | 0.1287 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/min20/comparison_results.json` |
| legacy_0409 | min8_cnnlstm | official | aggregate_legacy | legacy_direct_regression | 1.6652 | 1.6703 | 0.2063 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/min8/comparison_results.json` |
| legacy_0409 | min8_sats | official | aggregate_legacy | legacy_direct_regression | 0.6867 | 0.3368 | 0.1347 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/min8/comparison_results.json` |
| legacy_0409 | min8_sats_xy | official | aggregate_legacy | legacy_direct_regression | 0.7733 | 0.3219 | 0.1168 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/min8/comparison_results.json` |
| legacy_0409 | runs_comparison_1-6_ms8_sats_xy | official | aggregate_legacy | legacy_direct_regression | 0.8726 | 0.4227 | 0.11 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/runs_comparison_1-6_ms8/comparison_results.json` |
| legacy_0409 | runs_comparison_1-9_ms8_cnnlstm | official | aggregate_legacy | legacy_direct_regression | 0.7772 | 0.3527 | 0.1517 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/runs_comparison_1-9_ms8/comparison_results.json` |
| legacy_0409 | runs_comparison_1-9_ms8_mlp | official | aggregate_legacy | legacy_direct_regression | 1.9885 | 1.2106 | 0.2117 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/runs_comparison_1-9_ms8/comparison_results.json` |
| legacy_0409 | runs_comparison_1-9_ms8_sats | official | aggregate_legacy | legacy_direct_regression | 0.5024 | 0.1893 | 0.0803 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/runs_comparison_1-9_ms8/comparison_results.json` |
| legacy_0409 | runs_comparison_7-9_ms8_cnnlstm | official | aggregate_legacy | legacy_direct_regression | 0.7348 | 0.3457 | 0.0961 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/runs_comparison_7-9_ms8/comparison_results.json` |
| legacy_0409 | runs_comparison_7-9_ms8_mlp | official | aggregate_legacy | legacy_direct_regression | 2.5203 | 1.8871 | 0.1939 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/runs_comparison_7-9_ms8/comparison_results.json` |
| legacy_0409 | runs_comparison_7-9_ms8_sats | official | aggregate_legacy | legacy_direct_regression | 0.4662 | 0.1743 | 0.0509 | - | legacy aggregate without explicit split metadata | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/학습결과/runs_comparison_7-9_ms8/comparison_results.json` |
| legacy_0409_doc | legacy_sr_ff_report_reference | reference_only | document_reference | legacy_pipeline_doc | 0.62 | 0.31 | 0.08 | 0.602 | doc-only reference; not split-aligned json output | `/home/user/sensor_training/training/runs/접촉점 기준 학습_0409/md/sensor_learning_report_final_20260406.md` |

## Reclassification Notes

- `eval-split all` heatmap 결과는 exploratory로만 분류한다. 현재 공식 baseline 표에는 포함하지 않는다.
- `predicted_xy + GT radius`는 pseudo end-to-end upper-bound이므로 메인 성능 표에서 분리한다.
- `z_fz_regressor_gt_xy`도 deployment condition이 아니므로 reference-only로 유지한다.

