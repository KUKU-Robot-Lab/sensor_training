# robot_skin/runs/ (git-ignored)

학습 산출물. 이 README 외에는 커밋하지 않는다. `python -m robot_skin pipeline` 기본 출력:

```
runs/splits.json              모든 stage 가 공유하는 train/val/test (datasets.splits)
runs/pipeline.json            stage 별 실행 기록 (설정 파일, splits sha256, 연결, 소요 시간)
runs/<stage>/                 imu_pose | baseline | contact | pretrain | vtla: metrics.json, 체크포인트(ckpt_best/last.pt),
                              주 산출물(imu_pose_model.pt, baseline_model.pt, calibrator.json, encoder_state.pt,
                              policy_bundle.pt), pipeline_config.yaml (실제 사용된 stage 설정)
runs/deploy/                  python -m robot_skin deploy: metrics.json + sessions/<id>/ (다시 전처리 가능한 raw 세션)
```
