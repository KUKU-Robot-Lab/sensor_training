# runs/ — 대시보드 배포 가중치 폴더 규약

`run_dashboard` 는 이 폴더를 1순위로 스캔한다. **v\* 폴더 안의 .pt/.pth 는 이름과
무관하게 전부** UI 콤보에 후보로 뜬다. (가중치 파일은 gitignore — clone 후 직접 배치)

```
runs/
├── sats/v5/g05nsc/best_model.pt   # S1·S3 SATS — 같은 폴더에 config.json 필수
│            └── config.json
├── deform/v5/restorer.pt          # S2·S3 변형 복원기(자가수록 체크포인트)
└── theta/v6/estimator_v6_2.pt     # S2 밴딩각 estimator
             └── estimator_v6_2.pt_ref_baseline.npy   # (있으면) 재앵커 기준
```

- **S1**: UI 에서 v\* 선택 → `runs/sats/<v*>/` 의 pth 전부 표시.
- **S2**: est 콤보 = `runs/theta/`, restore 콤보 = v\* 선택 시 `runs/deform/<v*>/`.
- **S3**: SATS(run 콤보)와 restore 콤보 모두 v\* 기준 — S1·S2 와 동일 규칙.
- 구 위치(`sats/training/runs/*_deploy_*`, `sats/bending/runs/{deform_restorer*,estimator_*}`)도
  폴백으로 계속 인식된다.

## 기존 가중치를 이 규약으로 복사

```bash
.venv/bin/python scripts/stage_deploy_runs.py
```

멱등(같은 크기면 스킵)이며 원본은 옮기지 않는다. Windows clone 에는 이 폴더 구조
그대로 가중치를 복사해 넣으면 된다(예: `runs\sats\v8\g05nsc\best_model.pt`).
