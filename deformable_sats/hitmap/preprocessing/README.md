# Hitmap Preprocessing Pipeline

이 README는 legacy/experimental `hitmap` 경로를 설명한다. 현재 SATS 학습의 공식
입력은 `learning_data/sensor_raw_bin/**/*_merged.bin` + GT meta cache
(`learning_data/gt_meta_cache_xy_d5d10_g05`, GPU on-the-fly GT)이며, 200 Hz common
timeline으로 정렬된 BIN을 사용한다. SATS pressure-map GT 생성/학습은 [`sats/README.md`](../../sats/README.md)와
[`sats/preprocessing/README.md`](../../sats/preprocessing/README.md)를 기준으로 본다.

`hitmap` 경로는 CSV/Zarr 기반 XY heatmap 및 Z/Fz 회귀 실험용이다. AFD 3축 force,
100 Hz resample, `processed_data/zarr_data` 같은 설정은 이 경로에만 해당하며,
현재 `eco20 + mesh` d5 `test1/test2` SATS 학습 설정과는 다르다.

이 디렉토리는 raw 장비 로그를 학습 가능한 feature/Zarr dataset으로 바꾸는 단계입니다.  
실행 순서는 `raw_merge.py` 다음 `preprocess.py`입니다.

## Input / Output Layout
입력:
- `preprocessing/raw_data/` 아래의 실험 trial 폴더
- 각 trial에는 `due*.csv`, `ethermotion*.csv`, `afd*.csv`가 있어야 합니다.

출력:
- `processed_data/baselines/`: baseline JSON
- `processed_data/grid/`: grid-filtered raw CSV
- `processed_data/features/`: normalized feature CSV
- `processed_data/zarr_data/`: training용 Zarr
- `depth_mm`는 현재 학습 기준 depth이며 `z_contact_mm`로 저장됩니다.
- `processed_data/peak_summary/`: 좌표별 최대 depth/Fz 요약 CSV
- `processed_data/label_preview/`: label sanity-check PNG

## 1. raw_merge.py
목적:
세 장비 로그를 공통 타임라인으로 정렬해 `*_merged.csv`를 생성합니다.

기본 예시:
```bash
python3 preprocessing/raw_merge.py \
  --raw-root preprocessing/raw_data \
  --align-mode resample \
  --resample-hz 100 \
  --min-match-ratio 0.9 \
  --force-round-dp 2
```

주요 산출물:
- `*_merged.csv`
- sync plot PNG
- baseline/summary JSON

자주 조정하는 옵션:
- `--align-mode {nearest,resample}`
  - `nearest`: 원본 타임스탬프 기준 최근접 병합
  - `resample`: 공통 주기로 다시 샘플링해서 병합
- `--resample-hz`
  - `resample` 모드의 목표 주파수
- `--window-ms`, `--window-agg`
  - 리샘플 후 시간 윈도우 smoothing 설정
- `--max-dt-ms`
  - 최근접 정렬 시 허용할 최대 시차. 초과 샘플은 버립니다.
- `--lag-due-ms`, `--lag-ethermotion-ms`, `--lag-afd-ms`
  - 장비별 전역 시간 오프셋 보정
- `--min-match-ratio`
  - due/ethermotion/afd 정렬 성공률 최소 기준
- `--force-round-dp`
  - Fx/Fy/Fz 소수점 반올림 자릿수

운영 팁:
- 데이터 구조가 nested여도 `test*` leaf 디렉토리까지 자동 탐색합니다.
- 병합이 끝난 뒤에는 각 trial에 `*_merged.csv`가 생성됐는지 먼저 확인하세요.

## 2. preprocess.py
목적:
merged CSV를 baseline-corrected grid/feature/Zarr dataset으로 바꿉니다.

동작 개요:
1. `**/*_merged.csv`를 수집합니다.
2. 무부하 구간을 찾아 trial별 baseline을 계산합니다.
3. 0.5 mm grid에 스냅되는 정지 샘플만 남깁니다.
4. `z_stage_mm`, `z_contact_mm`, `z_depth_mm`, `phase`, `s_norm_i`, `fz_bc`를 계산합니다.
5. 필요하면 contact radius와 label preview를 생성합니다.
6. feature CSV, Zarr, peak summary를 저장합니다.

주요 옵션

입출력 / 실행 제어:
- `--raw-dir`
  - merged CSV를 찾을 루트 경로
- `--out-dir`
  - `baselines`, `grid`, `features`, `zarr_data`를 저장할 경로
- `--glob`
  - 수집할 merged CSV 패턴. 기본값은 `**/*_merged.csv`
- `--workers`
  - trial 단위 병렬 처리 수. 결과 의미는 같고 처리 속도만 바뀝니다.
- `--no-zarr`
  - feature CSV만 만들고 Zarr 저장은 건너뜁니다.

샘플 필터:
- `--contact-threshold`
  - 접촉으로 볼 최소 센서 변화량 기준
- `--z-bin-mm`
  - `z_depth_mm`를 이 간격으로 binning 후 `(trial, x, y, z_bin)` 단위로 집계합니다.
  - 예를 들어 `0.001`이면 0.001 mm 단위로 depth를 반올림한 뒤 같은 bin만 집계합니다.
  - 값이 커질수록 깊이 해상도는 낮아지고 샘플 안정성은 높아집니다.
  - 데이터가 없는 bin은 새로 만들지 않고, 보간도 하지 않습니다.
- `--min-signal`
  - 개별 샘플의 `max(|s_norm_i|)`가 이 값보다 작으면 제거합니다.
  - 값을 주지 않으면 baseline 구간 노이즈에서 trial별 자동 threshold를 계산합니다.
- `--min-reliable-s`
  - 좌표 단위 일관성 필터 기준입니다.
  - `min-signal`이 샘플 제거 기준이라면, `min-reliable-s`는 좌표 유지 기준입니다.
  - 둘은 동시에 사용할 수 있으며, 순서는 `min-signal` 후 `min-reliable-s`입니다.

baseline 탐색:
- `--baseline-z-thresh`
  - baseline 후보에서 허용할 `|z_mm|` 최대값
- `--baseline-force-thresh`
  - baseline 후보에서 허용할 `|Fz|` 최대값
- `--baseline-min-consec`
  - baseline으로 인정할 최소 연속 샘플 수

depth-aware radius:
- `--use-depth-aware-radius`
  - contact radius를 계산해 `contact_radius_mm`, `contact_radius_cell`을 추가합니다.
  - 이 옵션을 켜면 Zarr `aux_feat[:,3]`의 의미가 `diameter_mm`가 아니라 `contact_radius_mm`가 됩니다.
  - 인덴터 반경은 `raw_data/<material>/d5`, `d10` 같은 폴더명에서 자동 추론합니다.
- `--radius-model {hertz,geo}`
  - `hertz`: `a = sqrt(R * delta)`, 작은 압입에서의 탄성 접촉 근사
  - `geo`: `a = sqrt(2R*delta - delta^2)`, 구형 인덴터 기하 모델
- `--fallback-depth-mode {none,mean,const}`
  - depth가 0/음수일 때 반경 계산용 depth 대체 규칙
- `--fallback-depth-mm`
  - `const` 모드에서 쓸 상수 depth

label preview:
- `--export-label-heatmap`
  - 학습용 출력이 아니라 라벨 sanity check PNG를 생성합니다.
- `--label-samples`
  - 시각화 샘플 수
- `--label-kernel {gaussian,linear}`
  - 라벨 히트맵 커널
- `--sigma-scale`
  - gaussian kernel의 sigma 배율

## Recommended Runs
학습용 기본 전처리:
```bash
python3 preprocessing/preprocess.py \
  --raw-dir preprocessing/raw_data \
  --out-dir preprocessing/processed_data \
  --use-depth-aware-radius \
  --radius-model hertz \
  --z-bin-mm 0.001 \
  --min-reliable-s 0.001 \
  --baseline-z-thresh 0.001 \
  --baseline-force-thresh 0.5 \
  --baseline-min-consec 40 \
  --fallback-depth-mode none
```

수동 `min-signal`을 쓰는 완화된 필터:
```bash
python3 preprocessing/preprocess.py \
  --raw-dir preprocessing/raw_data \
  --out-dir preprocessing/processed_data_min10 \
  --use-depth-aware-radius \
  --radius-model hertz \
  --z-bin-mm 0.001 \
  --min-signal 0.01 \
  --min-reliable-s 0.001 \
  --baseline-z-thresh 0.001 \
  --baseline-force-thresh 0.5 \
  --baseline-min-consec 40 \
  --fallback-depth-mode none
```

label sanity-check 포함 전처리:
```bash
python3 preprocessing/preprocess.py \
  --raw-dir preprocessing/raw_data \
  --out-dir preprocessing/processed_data \
  --use-depth-aware-radius \
  --radius-model hertz \
  --z-bin-mm 0.001 \
  --min-reliable-s 0.001 \
  --baseline-z-thresh 0.001 \
  --baseline-force-thresh 0.5 \
  --baseline-min-consec 40 \
  --fallback-depth-mode none \
  --export-label-heatmap \
  --label-kernel gaussian \
  --label-samples 3 \
  --sigma-scale 1.0
```

라벨 프리뷰만 별도 생성:
```bash
python3 preprocessing/label_preview.py \
  --grid-file preprocessing/processed_data/grid/ecomesh_d5_z1.0_test1_grid.csv \
  --samples 3 \
  --kernel gaussian \
  --sigma-scale 1.0
```

## Peak Summary
전처리 마지막에는 소재별 `peak_summary/<material>_peak_summary.csv`를 추가로 생성합니다.

주요 컬럼:
- `material`, `diameter_mm`, `z_max_indentation_mm`, `x_mm`, `y_mm`
- `max_depth_mm`
- `max_fz_bc`
- `max_fz_raw`
- `n_trials`, `n_samples`

용도:
- 좌표별 최대 압입 깊이와 최대 정압을 파악
- `d5` vs `d10`, `z_1.0mm` vs `z_1.5mm` 비교
- 이후 Z/Fz 학습 한계와 난이도 분석
