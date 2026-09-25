# robot_skin/data/ (git-ignored)

이 README 외에는 커밋하지 않는다. 경로 기본값은 `robot_skin/configs/default.yaml` `paths`.

```
data/raw/<dataset>/<subject>/<session_id>/          기록된 raw 세션 (session.json + 스트림; docs/DATA_ACQUISITION.md)
data/processed/<dataset>/<episode_id>/              전처리된 Episode (python -m robot_skin preprocess; docs/DATA_FORMAT.md)
data/synthetic/<dataset>/<subject>/<session_id>/    python -m robot_skin synth 의 합성 raw 세션 (실제 데이터와 섞지 않는다)
```
`<dataset>` = `motion` (D1) | `task` (D2) | `other` (배포 세션 등).
