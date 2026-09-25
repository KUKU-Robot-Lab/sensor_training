# Codex 재구성 실행 프롬프트 목록

작성일: 2026-04-10

이 문서는 다른 터미널에서 Codex를 새로 실행해 `sensor_training` 학습 프레임워크를 재구성할 때 사용할 순차 실행 프롬프트다.

기준 문서:

- `md/260410_runs_comparison_analysis.md`
- `md/260409_loss_design.md`
- `md/260409_depth_aware_contact_labeling.md`
- `README.md`
- `training/README.md`

## 사용 방법

1. 새 터미널에서 repo로 이동한다.

```bash
cd /home/user/sensor_training
```

2. Codex를 실행한다.
3. 아래 프롬프트를 `Prompt 0`부터 순서대로 하나씩 붙여넣는다.
4. 각 단계가 끝날 때마다 Codex가 제시한 diff, 테스트 결과, 남은 리스크를 확인한 뒤 다음 프롬프트로 넘어간다.

## 공통 운영 규칙

모든 프롬프트에서 다음 규칙을 유지한다.

- `md/260410_runs_comparison_analysis.md`의 결론을 기준으로 작업한다.
- 기존 사용자 변경을 되돌리지 않는다. 특히 untracked `AGENTS.md`가 있으면 건드리지 않는다.
- ECC skill/agent를 적극 사용한다. 가능하면 `planner`, `tdd-guide`, `python-reviewer`, `code-reviewer`, `build-error-resolver`, `doc-updater`, `verification-loop`를 직접 언급하고 활용한다.
- 코드 변경은 TDD 방식으로 진행한다. 먼저 재현 테스트 또는 최소 smoke test를 추가하고, 그 다음 구현한다.
- Python 환경은 사용자가 말한 대로 `conda activate sensor`를 우선 사용한다. 환경 충돌이 있으면 우회하지 말고 원인과 해결안을 보고한다.
- 최종 변경 후에는 `git diff`, 관련 테스트, 실행 명령, 실패한 검증을 요약한다.

현재 주요 문제 요약:

- Fz는 학습 loss에는 들어가지만 저장 metric에 빠져 있다.
- `comparison_results.json`이 핵심 `multi_head_field` 결과를 통합하지 않는다.
- `preprocessing/processed_data/zarr_data/dataset_index.json`이 마지막 소재 index만 보존해 소재별 zarr와 불일치할 수 있다.
- `train_comparison.py`의 자동 zarr 선택은 여러 `.zarr` 중 첫 번째만 선택한다.
- train/val split이 trial 단위가 아니라 sequence sample 단위라 leakage 가능성이 있다.
- `--use-depth-aware-label`이 실제 loss/label 분기를 충분히 제어하지 않는다.

---

## Prompt 0: Repo Grounding & 실행 계획 확정

ECC 매칭:

- Agent: `planner`, `python-reviewer`
- Skill: `tdd-workflow`, `verification-loop`

붙여넣을 프롬프트:

```text
나는 /home/user/sensor_training repo에서 tactile sensor 학습 프레임워크를 재구성하려고 한다.

ECC 매칭을 적극 사용해라:
- planner 관점으로 작업 순서를 먼저 확정
- python-reviewer 관점으로 학습/전처리 코드 리스크 점검
- tdd-workflow로 테스트 우선 구현
- verification-loop로 각 단계 검증

먼저 아래 문서를 읽고, repo를 비파괴적으로 탐색해 실제 수정 범위와 테스트 전략을 확정해라.

반드시 읽을 문서:
- md/260410_runs_comparison_analysis.md
- md/260409_loss_design.md
- md/260409_depth_aware_contact_labeling.md
- README.md
- training/README.md

반드시 확인할 코드:
- training/pipelines/train_comparison.py
- preprocessing/preprocess.py
- training/data/dataset_zarr.py
- training/data/dataset_unified.py
- training/models/multi_head_field_model.py
- inference/run_inference.py

목표:
1. Zarr/index 불일치 문제 해결 범위 확정
2. train/val split leakage 제거 방향 확정
3. Fz metric 누락 해결 방향 확정
4. depth-aware label flag 정리 방향 확정
5. overlay 진단 개선 방향 확정
6. 각 단계별 테스트 계획 확정

제약:
- 아직 코드 수정하지 말고 읽기/분석만 수행
- 기존 사용자 변경을 되돌리지 말 것
- untracked AGENTS.md가 있으면 건드리지 말 것

출력:
- 파일별 변경 예정 요약
- 테스트 우선순위
- 예상 실행 명령
- 진행 중 막힐 수 있는 환경 리스크
```

완료 기준:

- Codex가 실제 파일 근거를 들어 수정 범위를 확정한다.
- 다음 단계에서 바로 TDD 구현에 들어갈 수 있을 정도로 테스트 대상이 명확하다.

---

## Prompt 1: Zarr index/data source 재구성

ECC 매칭:

- Agent: `tdd-guide`, `python-reviewer`, `code-reviewer`
- Skill: `tdd-workflow`, `verification-loop`

붙여넣을 프롬프트:

```text
이제 Zarr/index/data source 문제를 TDD로 수정해라.

ECC 매칭:
- tdd-guide로 먼저 실패 테스트를 설계하고 실행
- python-reviewer 관점으로 dataset loader correctness 검토
- code-reviewer 관점으로 변경 후 자체 리뷰
- verification-loop로 관련 테스트와 diff 확인

문제:
- preprocessing/preprocess.py의 export_to_zarr()가 소재별 zarr를 저장할 때 부모 zarr_data/dataset_index.json을 매번 덮어쓴다.
- training/pipelines/train_comparison.py의 _resolve_zarr_path()는 여러 .zarr 중 첫 번째만 자동 선택한다.
- 현재 dataset_ecemesh.zarr는 772,567행, dataset_ecomesh.zarr는 192,686행인데 dataset_index.json은 ecomesh_d5_7~9만 포함한다.

원하는 동작:
1. 각 zarr가 자기 index를 명확히 가지게 하거나, loader가 zarr 내부/동일 디렉토리의 정확한 index만 읽게 해라.
2. 여러 zarr가 있을 때 자동으로 첫 번째를 조용히 고르지 말고, 명시적 --zarr-path 또는 통합 zarr만 허용해라.
3. 기존 csv data source fallback은 유지해라.
4. 기존 processed_data를 바로 파괴적으로 재생성하지 말고, 코드와 테스트를 먼저 고쳐라.

테스트:
- zarr별 index 파일/경로 resolution 테스트 추가
- 여러 zarr가 있을 때 _resolve_zarr_path()가 조용히 첫 번째를 고르지 않는지 테스트
- index trial_id와 zarr shape가 불일치할 때 명확한 에러가 나는지 최소 smoke test 추가

제약:
- 기존 사용자 변경 되돌리지 말 것
- 테스트가 먼저 실패하는 RED 상태를 확인한 뒤 구현
- 구현 후 GREEN 확인

출력:
- 변경 파일
- RED/GREEN 테스트 명령과 결과
- 남은 데이터 재생성 필요 여부
```

완료 기준:

- zarr와 index가 소재별로 섞이지 않는다.
- 다중 zarr 자동 선택이 조용한 오동작을 만들지 않는다.
- 테스트가 추가되어 같은 문제가 재발하면 실패한다.

---

## Prompt 2: Trial-level split / leave-one-trial-out split

ECC 매칭:

- Agent: `tdd-guide`, `python-reviewer`
- Skill: `tdd-workflow`, `verification-loop`

붙여넣을 프롬프트:

```text
이제 train/val split leakage 문제를 TDD로 수정해라.

ECC 매칭:
- tdd-guide로 split leakage 재현 테스트를 먼저 작성
- python-reviewer 관점으로 dataset/split correctness 검토
- verification-loop로 테스트와 diff 확인

문제:
- ZarrSequenceDataset은 (trial_id, x_mm, y_mm)별 sequence를 만든다.
- build_shared_data()는 전체 sequence index를 무작위 80/20 split한다.
- 그래서 같은 trial/좌표의 겹치는 sequence가 train과 val에 동시에 들어갈 수 있다.

원하는 동작:
1. 기본 split은 trial 단위로 한다.
2. CLI 옵션으로 --val-trials와 가능하면 --test-trials를 받을 수 있게 한다.
3. leave-one-trial-out 평가가 가능하도록 명시적 trial list를 지원한다.
4. 기존 간단 실행이 깨지지 않도록, 명시 trial이 없으면 seed 기반 trial split을 사용한다.
5. split 요약 로그에 train/val/test trial_id 목록과 sample 수를 출력한다.

테스트:
- synthetic dataset 또는 lightweight fixture에서 같은 trial_id가 train_idx와 val_idx에 동시에 들어가지 않음을 검증
- --val-trials 지정 시 해당 trial만 val로 가는지 검증
- trial 수가 매우 적을 때 명확한 에러 또는 documented fallback이 동작하는지 검증

제약:
- 먼저 실패 테스트를 실행하고 RED를 확인
- 구현 후 같은 테스트가 GREEN인지 확인
- 기존 사용자 변경 되돌리지 말 것

출력:
- 변경 파일
- split 정책 요약
- 테스트 명령과 결과
- 후속 재학습 명령 예시
```

완료 기준:

- train/val에 같은 trial이 섞이지 않는다.
- trial 지정 실험이 재현 가능하다.
- 기존 metric의 낙관적 leakage 리스크가 제거된다.

---

## Prompt 3: Fz metric/evaluation 저장

ECC 매칭:

- Agent: `tdd-guide`, `python-reviewer`, `code-reviewer`
- Skill: `tdd-workflow`, `verification-loop`

붙여넣을 프롬프트:

```text
이제 multi_head_field의 Fz metric 누락 문제를 TDD로 수정해라.

ECC 매칭:
- tdd-guide로 metric shape/key 테스트를 먼저 작성
- python-reviewer 관점으로 torch/numpy metric 변환 검토
- code-reviewer 관점으로 변경 후 자체 리뷰
- verification-loop로 테스트와 diff 확인

문제:
- MultiHeadFieldModel은 scalar_vec [z, Fz]를 출력한다.
- train_comparison.py는 l_fz를 loss에 포함하지만 validation metric에는 [x, y, z]만 저장한다.
- 그래서 metrics_*.json만으로 Fz 학습 품질을 판단할 수 없다.

원하는 동작:
1. multi_head_field validation에서 [x, y, z, fz] metric을 저장한다.
2. 기존 다른 모델의 [x, y, z] metric과 충돌하지 않도록 결과 schema를 명확히 한다.
3. Fz MAE/RMSE/R2를 metrics JSON에 포함한다.
4. 가능하면 z-Fz 또는 Fz pred/target summary CSV를 저장한다.
5. comparison_results.json에도 multi_head_field의 확장 metric이 보존되게 한다.

테스트:
- synthetic prediction/target으로 calculate_metrics가 4D 입력을 처리하는지 검증
- multi_head_field validation path가 fz target을 버리지 않는지 검증
- JSON에 fz 관련 key 또는 4번째 metric 값이 저장되는지 검증

제약:
- 먼저 실패 테스트로 RED 확인
- 구현 후 GREEN 확인
- 기존 metric 소비 코드가 있으면 호환성 검토

출력:
- 변경 파일
- 새 metrics JSON schema 설명
- 테스트 명령과 결과
- Fz 평가를 포함한 재학습/재평가 명령 예시
```

완료 기준:

- 새 metric 파일에서 Fz 성능을 직접 확인할 수 있다.
- z만 좋고 Fz가 나쁜 상황을 구분할 수 있다.

---

## Prompt 4: Depth-aware stage flag 정리

ECC 매칭:

- Agent: `tdd-guide`, `python-reviewer`, `code-reviewer`
- Skill: `tdd-workflow`, `verification-loop`

붙여넣을 프롬프트:

```text
이제 --use-depth-aware-label과 Stage1/2/3 실험 분리를 TDD로 정리해라.

ECC 매칭:
- tdd-guide로 flag branch 테스트를 먼저 작성
- python-reviewer 관점으로 loss/label branch correctness 검토
- code-reviewer 관점으로 CLI와 naming review
- verification-loop로 테스트와 diff 확인

문제:
- 현재 --use-depth-aware-label은 이름상 depth-aware label on/off 플래그다.
- 하지만 multi_head_field 학습 로직은 이 플래그와 무관하게 _build_soft_heatmap()을 쓰는 형태라 Stage1/Stage2 A/B 비교가 명확하지 않다.

원하는 동작:
1. Stage1: baseline point label 또는 기존 argmax 방식이 명확히 실행된다.
2. Stage2: depth-aware soft heatmap + xy only가 명확히 실행된다. lambda_z=0, lambda_fz=0인 경우 scalar head는 metric 목적 외 학습에 영향 주지 않게 한다.
3. Stage3: depth-aware soft heatmap + z/Fz scalar head가 명확히 실행된다.
4. checkpoint/tag 이름이 실제 stage와 loss 구성을 정확히 반영한다.
5. README.md와 training/README.md의 예시 명령이 실제 동작과 일치한다.

테스트:
- --use-depth-aware-label off일 때 soft heatmap branch가 사용되지 않는지 검증
- --use-depth-aware-label on일 때 soft target branch가 사용되는지 검증
- lambda_z/lambda_fz 0일 때 scalar loss가 total loss에 기여하지 않는지 검증
- checkpoint tag가 stage/loss 구성을 반영하는지 검증

제약:
- 먼저 RED 확인 후 구현
- 기존 실험 결과 파일을 수정/삭제하지 말 것
- README 업데이트는 코드 동작 확정 후 수행

출력:
- 변경 파일
- Stage1/2/3 최종 CLI 예시
- 테스트 명령과 결과
- 남은 실험 재실행 필요 여부
```

완료 기준:

- Stage1/2/3가 같은 코드에서 명확히 분리된다.
- depth-aware label flag가 이름과 실제 동작이 일치한다.

---

## Prompt 5: Overlay 진단 개선

ECC 매칭:

- Agent: `python-reviewer`
- Skill: `verification-loop`

붙여넣을 프롬프트:

```text
이제 overlay 진단 이미지를 개선해라.

ECC 매칭:
- python-reviewer 관점으로 matplotlib/torch tensor 변환 안정성 검토
- verification-loop로 smoke test와 diff 확인

문제:
- 현재 overlay PNG는 pred heatmap과 target heatmap만 보여준다.
- metric과 overlay가 왜 다르게 보이는지 바로 확인하기 어렵다.

원하는 동작:
1. overlay에 target center와 predicted center를 표시한다.
2. 각 샘플 제목 또는 annotation에 xy error, target z, pred z를 표시한다.
3. Prompt 3에서 Fz metric이 구현된 뒤라면 target Fz와 pred Fz도 표시한다.
4. 기존 --save-heatmap-overlay 옵션으로만 저장되게 유지한다.
5. 저장 파일명은 기존 패턴과 호환되게 유지하거나, 새 suffix를 붙여 구분한다.

테스트:
- 작은 fake fmap/target batch로 overlay 저장 함수가 파일을 생성하는지 smoke test
- headless 환경에서 matplotlib backend 문제가 없는지 확인

제약:
- 기존 결과 PNG를 삭제하지 말 것
- 테스트/스모크는 새 임시 output dir를 사용

출력:
- 변경 파일
- 새 overlay 예시 경로
- smoke test 명령과 결과
```

완료 기준:

- overlay만 봐도 heatmap 품질과 좌표 metric 차이를 추적할 수 있다.

---

## Prompt 6: Controlled re-run 실험

ECC 매칭:

- Agent: `planner`, `build-error-resolver`
- Skill: `verification-loop`

붙여넣을 프롬프트:

```text
이제 수정된 프레임워크로 controlled re-run 실험을 준비하고 실행해라.

ECC 매칭:
- planner 관점으로 실행 순서와 리소스 리스크를 먼저 정리
- build-error-resolver 관점으로 환경/런타임 오류를 해결
- verification-loop로 실행 결과와 출력물을 확인

목표:
- 같은 데이터 split 기준으로 Stage1/2/3을 다시 실행해 비교 가능하게 만든다.
- Fz metric까지 포함한 새 결과를 생성한다.

환경:
- conda activate sensor를 우선 사용
- torch/numpy/zarr 환경 충돌이 있으면 원인과 해결안을 먼저 보고하고, 임의로 우회하지 말 것

실행 전 확인:
1. zarr/index가 올바른 소재와 trial을 가리키는지
2. train/val trial 목록이 출력되는지
3. Fz metric 저장 schema가 동작하는지
4. overlay 저장이 새 annotation을 포함하는지

실험:
- Stage1 baseline
- Stage2 depth-aware soft xy only
- Stage3 depth-aware soft xy + z/Fz

출력 위치:
- 기존 training/runs/runs_comparison을 덮어쓰지 말고 새 run dir를 사용해라.
- 예: training/runs/runs_rebuild_260410 또는 Codex가 timestamp 기반으로 정한 새 디렉토리.

제약:
- 실행 시간이 너무 길면 먼저 --epochs 1 또는 tiny smoke로 통과 확인 후 본 실험 명령을 제안해라.
- OOM 가능성이 있으면 batch-size를 줄이고 근거를 남겨라.

출력:
- 실제 실행한 명령
- 생성된 metrics JSON 경로
- Stage별 핵심 metric 표
- 실패한 명령과 원인
```

완료 기준:

- 새 run dir에 Stage1/2/3 결과가 생성된다.
- 새 metric에 Fz가 포함된다.
- trial-level split 기준의 성능을 확인할 수 있다.

---

## Prompt 7: Final review & 문서 업데이트

ECC 매칭:

- Agent: `code-reviewer`, `python-reviewer`, `security-reviewer`, `doc-updater`
- Skill: `verification-loop`

붙여넣을 프롬프트:

```text
이제 전체 변경을 최종 리뷰하고 문서화해라.

ECC 매칭:
- code-reviewer 관점으로 correctness/regression review
- python-reviewer 관점으로 Python/PyTorch/data pipeline review
- security-reviewer 관점으로 secrets/unsafe file operations 여부 확인
- doc-updater 관점으로 README와 md 문서 업데이트
- verification-loop로 최종 테스트/검증 요약

목표:
1. 전체 diff를 리뷰해 의도하지 않은 변경이 없는지 확인
2. 테스트/스모크/재학습 실행 결과를 정리
3. md/260410_runs_comparison_analysis.md를 후속 결과 기준으로 업데이트하거나, 새 후속 분석 문서를 작성
4. README.md와 training/README.md의 명령 예시가 실제 코드와 일치하는지 확인

반드시 확인:
- git status --short
- git diff --stat
- 관련 테스트 결과
- 새 metrics JSON 내용
- Fz metric 포함 여부
- train/val trial leakage 제거 여부
- Zarr/index 불일치 제거 여부

출력:
- 변경 요약
- 테스트 결과
- 새 실험 결과 요약
- 남은 리스크
- 다음 데이터 수집/실험 제안
```

완료 기준:

- 코드 변경, 테스트 결과, 새 학습 결과, 문서 업데이트가 한 번에 추적 가능하다.
- 다음 실험자가 같은 명령으로 재현할 수 있다.

## 권장 진행 순서 요약

1. Prompt 0으로 repo 상태와 수정 범위를 다시 확정한다.
2. Prompt 1-4로 core correctness를 먼저 고친다.
3. Prompt 5로 진단 시각화를 개선한다.
4. Prompt 6으로 controlled re-run을 수행한다.
5. Prompt 7로 최종 리뷰와 문서화를 끝낸다.

가장 중요한 순서는 `Zarr/index -> trial split -> Fz metric -> stage flag -> overlay -> re-run`이다. 모델 구조를 더 복잡하게 바꾸는 것은 이 순서가 끝난 뒤 판단하는 것이 맞다.
