# Hitmap Training Pipeline

이 README는 legacy/experimental `hitmap` 학습 경로를 설명한다. 현재 공식
SATS 학습은 `learning_data`의 merged BIN과 on-the-fly GT를 사용하는 별도 경로이며,
현행 엔트리는 `python3 -m sats.training.train_e2e`(+인덴터 크기 입력 A)다.
SATS의 DUE/loadcell 200 Hz 정렬, EtherMotion 고정밀 보간, `test1/test2` row alignment,
GPU batch 권장은 [`sats/README.md`](../../sats/README.md)를 기준으로 본다.

`hitmap`은 CSV/Zarr 기반 XY heatmap, contact 기준 Z depth, Fz 회귀 실험용이다. 여기의
`--seq-len 50`, `--batch-size 1024`, `heatmap-size 40`, loading-only 조건은 SATS
pressure-map GT 학습 설정과 직접 호환되는 값이 아니다.

이 디렉토리는 전처리된 tactile dataset으로 XY heatmap, contact 기준 Z depth, Fz를 학습하고 평가하는 스크립트를 담고 있습니다.  
현재 실험 흐름에서 우선순위는 `train_z_fz_regressor.py`이고, `train_comparison.py` Stage1/2/3는 XY checkpoint 생성 및 비교 실험용입니다.

## Main Pipelines
- `pipelines/train_z_fz_regressor.py`
  - GT XY 조건 기반 Z/Fz 전용 시퀀스 회귀, 기본 trial-aware 5-fold CV
- `pipelines/train_comparison.py`
  - 여러 모델 동시 비교, `multi_head_field` Stage1/2/3 포함, 기본 trial-aware 5-fold CV
- `pipelines/evaluate_comparison_heatmap.py`
  - fold metadata를 재사용하는 모델별 XY/Z/Fz heatmap 평가

## Legacy Surface
- `pipelines/train_sr_zarr.py`
- `pipelines/train_ff_zarr.py`
- `pipelines/train_sr_class_zarr.py`
- `pipelines/train_unified.py`
- `pipelines/evaluate_zarr_sr.py`
- `pipelines/evaluate_sr.py`

위 스크립트들은 legacy/experimental surface입니다. 공식 경로로 취급하지 않습니다.

## 1. Z/Fz Regressor
목적:
XY는 주어진 조건으로 두고 tactile sequence에서 contact 기준 Z/Fz만 분리 학습합니다. 이 경로는 hysteresis 보존을 위해 loading phase만 사용합니다.

대표 명령:
```bash
python -m training.pipelines.train_z_fz_regressor \
  --data-dir preprocessing/processed_data \
  --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
  --out-dir training/runs_z_fz \
  --xy-checkpoint training/runs_comparison/folds/fold_0/best_multi_head_field_stage2_dlabel-gaussian-hertz_xybce1_zoff_fzoff_decsoftargmax.pth \
  --decode-xy softargmax \
  --xy-noise-std-mm 0.5 \
  --cv-folds 5 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --epochs 100 \
  --batch-size 1024 \
  --device cuda
```

실제 semantics:
- 학습은 항상 GT XY를 condition으로 사용합니다.
- 전처리 zarr의 `depth_mm`는 현재 `z_contact_mm`를 뜻합니다.
- Z/Fz sequence 회귀는 loading phase만 사용합니다. unloading/all은 이 스크립트에서 허용하지 않습니다.
- `--xy-checkpoint`는 학습 경로에 들어가지 않습니다.
- `--xy-checkpoint`를 주면 validation에서 frozen XY heatmap checkpoint를 decode한 `predicted_xy` 지표를 추가 계산합니다.
- validation은 두 세트를 기록합니다.
  - `gt_xy`: GT XY + GT radius
  - `predicted_xy`: frozen XY decode + GT radius
- 따라서 현재 `predicted_xy`는 완전한 end-to-end 평가가 아닙니다. radius는 여전히 GT를 사용합니다.
- best checkpoint 선택 기준은 validation `predicted_xy`의 `z_mae + fz_mae`입니다.
- `--test-trials`를 줘도 test metric을 별도로 계산하지는 않고 split metadata만 저장합니다.

주요 옵션:
- `--zarr-path`
  - 사용할 dataset을 명시합니다. `.zarr`가 여러 개면 필수입니다.
- `--xy-checkpoint`
  - frozen XY checkpoint를 써서 `predicted_xy` validation을 활성화합니다.
  - 5-fold 실행에서는 `fold_0` template 경로를 넣어도 현재 fold에 맞는 checkpoint로 자동 치환합니다.
- `--decode-xy {softargmax,argmax_refine}`
  - XY checkpoint heatmap을 실제 XY로 바꾸는 방법
- `--heatmap-size`
  - XY checkpoint 구조와 맞아야 합니다.
- `--xy-noise-std-mm`
  - 학습 시 GT XY에 넣는 gaussian noise 크기
- `--seq-len`, `--stride`
  - depth sequence를 윈도우로 자르는 규칙
- `--phase loading`
  - hysteresis 보존을 위해 loading phase만 허용합니다.
- `--val-trials`, `--test-trials`, `--seed`
  - 수동 split 제어. 이 옵션을 주면 k-fold 대신 단일 split을 사용합니다.
- `--cv-folds`, `--fold-index`
  - 기본 5-fold trial-aware CV. `--fold-index`를 주면 특정 fold만 실행합니다.
- `--loss {huber,mse}`, `--huber-delta`
  - Z/Fz 회귀 손실 정의
- `--optimizer {adam,adamw}`, `--weight-decay`, `--dropout`
  - 공식 regularization surface
- `--max-samples`
  - 빠른 smoke run용 trial-balanced sample cap
- `--device`
  - `cuda`일 때는 preloaded dataset 전체를 VRAM으로 옮깁니다.

출력:
- `best_z_fz_regressor.pth`
- `metrics_z_fz_regressor.json`
- `history_z_fz_regressor.json`

운영 팁:
- `--xy-noise-std-mm`는 GT upper-bound와 현실 추론 사이의 간극을 줄이기 위한 강건성 옵션입니다.
- `--max-samples`를 이용하면 작은 smoke run을 빠르게 돌릴 수 있습니다.

## 2. multi_head_field Comparison
목적:
XY heatmap checkpoint 생성, depth-aware label 비교, multi-model ablation을 수행합니다.

권장 순서:
1. Stage1: point label + xy only
2. Stage2: depth-aware soft label + xy only
3. Stage3: soft label + z/fz auxiliary heads
4. 필요 시 Stage2 checkpoint를 `train_z_fz_regressor.py`의 `--xy-checkpoint`로 사용

Stage1:
```bash
python -m training.pipelines.train_comparison \
  --data-dir preprocessing/processed_data \
  --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
  --models multi_head_field \
  --cv-folds 5 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --epochs 100 \
  --batch-size 1024 \
  --seq-len 50 \
  --lambda-z 0.0 \
  --lambda-fz 0.0 \
  --decode-xy none
```

Stage2:
```bash
python -m training.pipelines.train_comparison \
  --data-dir preprocessing/processed_data \
  --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
  --models multi_head_field \
  --use-depth-aware-label \
  --depth-label-kernel gaussian \
  --depth-radius-model hertz \
  --heatmap-size 40 \
  --fg-weight 8.0 \
  --heatmap-sigma-scale 0.35 \
  --lambda-z 0.0 \
  --lambda-fz 0.0 \
  --decode-xy softargmax \
  --depth-fallback-mm 1.0 \
  --depth-min-for-label 0.05 \
  --save-heatmap-overlay \
  --overlay-batches 1 \
  --overlay-samples 4 \
  --epochs 100 \
  --batch-size 1024
```

Stage3:
```bash
python -m training.pipelines.train_comparison \
  --data-dir preprocessing/processed_data \
  --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
  --models multi_head_field \
  --use-depth-aware-label \
  --loss-xy bce \
  --loss-z huber \
  --loss-fz huber \
  --lambda-xy 1.0 \
  --lambda-z 0.2 \
  --lambda-fz 0.2 \
  --depth-label-kernel gaussian \
  --depth-radius-model hertz \
  --heatmap-size 40 \
  --fg-weight 8.0 \
  --heatmap-sigma-scale 0.35 \
  --decode-xy softargmax \
  --depth-fallback-mm 1.0 \
  --depth-min-for-label 0.05 \
  --save-heatmap-overlay \
  --overlay-batches 1 \
  --overlay-samples 4 \
  --epochs 100 \
  --batch-size 1024
```

주요 옵션:
- `--use-depth-aware-label`
  - point label 대신 depth-aware soft label 사용
- `--depth-label-kernel {gaussian,linear}`
  - heatmap label kernel
- `--depth-radius-model {hertz,geom}`
  - label radius 생성 모델
- `--heatmap-size`
  - output heatmap 해상도
- `--heatmap-sigma-scale`
  - gaussian label 폭
- `--fg-weight`
  - foreground 가중치
- `--decode-xy {softargmax,argmax_refine,none}`
  - XY decode 방식
- `--cv-folds`, `--fold-index`
  - 기본 5-fold trial-aware CV
- `--optimizer {adam,adamw}`, `--weight-decay`, `--dropout`
  - 공식 regularization surface
- `--save-heatmap-overlay`, `--overlay-batches`, `--overlay-samples`
  - overlay sanity check PNG 저장

주의:
- `preprocess.py`의 radius model 옵션은 `geo`, `train_comparison.py`의 depth label 옵션은 `geom`입니다. 이름이 다르므로 그대로 복사할 때 구분해야 합니다.

## 3. Evaluation
heatmap 평가:
```bash
python3 -m training.pipelines.evaluate_comparison_heatmap \
  --runs-dir training/runs_comparison \
  --models multi_head_field \
  --batch-size 512 \
  --device cuda \
  --eval-split all \
  --decode-xy softargmax \
  --fill-missing neighbor
```

일반 체크포인트 추론 예시:
```bash
python inference/run_inference.py \
  --checkpoint training/runs_comparison/folds/fold_0/best_multi_head_field_stage3_dlabel-gaussian-hertz_xybce1_zhuber0p2_fzhuber0p2_decsoftargmax.pth \
  --data-dir preprocessing/processed_data \
  --zarr-path preprocessing/processed_data/zarr_data/dataset_ecomesh.zarr \
  --decode-xy softargmax \
  --heatmap-sigma-scale 0.35 \
  --depth-fallback-mm 1.0 \
  --depth-min-for-label 0.05 \
  --batch-size 256 \
  --max-batches 2 \
  --save-heatmap-overlay \
  --overlay-batches 1 \
  --overlay-samples 4
```

출력:
- comparison checkpoint: `training/runs_comparison/best_<model>*.pth`
- comparison summary: `training/runs_comparison/comparison_results.json`
- CV manifest: `training/runs_comparison/cv_manifest_comparison.json`
- heatmap 시각화: `training/runs_comparison/heatmaps/`
