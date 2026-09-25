# SATS xy 소재/해상도 성능평가 실험 계획

> 목적: `xy_1mm` 데이터로 소재별 SATS 성능을 정량 비교하고, `ecomesh`에 대해 `xy_1mm`와 `xy_0.5mm` 학습 조건을 비교하여 논문에 보고할 최종 SATS 성능을 확정한다. 모든 주요 실험은 d5와 d10 인덴터를 함께 사용한다.

---

## 1. 논문에서 검증할 주장

### 1.1 소재 ablation: xy_1mm

Fig.2의 receptive field 분석은 소재가 SATS 학습 입력의 품질을 바꾼다는 가설을 제시한다. `eco20_xy1`은 특히 d5에서 2번째/3번째 센서 신호가 약해 SR 학습에 불리하고, `eco50_xy1`과 `ecomesh_xy1`은 다중센서 신호가 살아 있다. SATS 성능평가는 이 물리적 가설을 모델 성능으로 검증한다.

검증 질문:

- 동일한 `xy_1mm` 스캔 조건에서 `eco20`, `eco50`, `ecomesh` 중 어느 소재가 가장 낮은 pressure-map RMSE를 보이는가.
- d5와 d10 조건에서 소재 순위가 같은가, 아니면 작은 접촉(d5)에서 차이가 더 커지는가.
- Fig.2의 coverage/overlap 지표가 SATS의 error map, localization error, force-conditioned RMSE와 일관되는가.

기대되는 결론 형태:

- `eco20_xy1`은 다중센서 coverage가 낮아 d5 holdout에서 높은 오류를 보일 가능성이 크다.
- `eco50_xy1`과 `ecomesh_xy1`은 `eco20_xy1` 대비 낮은 오류를 보여야 한다.
- `ecomesh_xy1`이 최종 소재로 채택되려면 단순 평균 RMSE뿐 아니라 d5/d10 모두에서 안정적이고, error map이 특정 taxel/edge에 치우치지 않아야 한다.

### 1.2 ecomesh resolution comparison: xy_1mm vs xy_0.5mm

`xy_0.5mm`는 더 조밀한 스캔을 제공하지만, 논문에서 비교할 때는 두 효과가 섞이지 않도록 구분한다.

- **Controlled comparison**: `ecomesh_xy1`과 `ecomesh_xy0p5` 모두 공통 반복인 d5/d10 `test1-3`만 사용한다. 이 비교는 scan pitch 자체의 효과를 본다.
- **Data-rich final calibration**: 최종 데모/대표 모델은 선택된 조건에서 가능한 데이터를 최대한 사용하되, 논문 성능값은 반드시 사전에 정의한 holdout 또는 controlled CV 결과와 분리해 보고한다.

결론 기준:

- 최종 논문 성능은 controlled comparison의 holdout 평균과 95% CI를 우선한다.
- 모든 데이터를 사용한 final model은 대표 추론 그림, realtime/demo, qualitative panel 생성용으로 사용한다.
- `xy_0.5mm`가 controlled comparison에서 `xy_1mm` 대비 평균 RMSE를 의미 있게 낮추지 못하면, dense scan의 비용 대비 이득이 작다고 기록한다.

---

## 2. 데이터 범위와 trial 정의

### 2.1 원천 데이터

원천 raw archive:

```text
skin_ws/raw_data/sats/
```

학습용 merged artifact:

```text
learning_data/sensor_raw_bin/{material}/d{5|10}/z_{2.5|3.5}mm/test{n}/
```

대상 material key:

| 목적 | material key | 입력 scan pitch | 인덴터 |
|---|---:|---:|---:|
| 소재 비교 | `eco20_xy1` | 1.0 mm | d5, d10 |
| 소재 비교 | `eco50_xy1` | 1.0 mm | d5, d10 |
| 소재 비교 / 최종 후보 | `ecomesh_xy1` | 1.0 mm | d5, d10 |
| 해상도 비교 / 최종 후보 | `ecomesh_xy0p5` | 0.5 mm | d5, d10 |

깊이 convention:

```text
d5  -> z2.5
d10 -> z3.5
```

### 2.2 split 원칙

trial-level holdout을 사용한다. 한 fold에서 같은 반복 번호의 d5와 d10을 함께 validation으로 둔다.

예: fold 3

```text
validation = *_d5_z2.5_test3 + *_d10_z3.5_test3
train      = 같은 material의 나머지 d5/d10 trials
```

이렇게 해야 동일 날짜/반복 조건이 d5와 d10 사이에서 섞이는 leakage를 줄이고, 논문에 “held-out scan repetitions”로 설명할 수 있다.

### 2.3 controlled ecomesh comparison용 trial index

`ecomesh_xy0p5`는 d5 반복이 test1-10까지 있어 그대로 사용하면 `xy_0.5mm`가 데이터양에서 유리해진다. Controlled comparison에서는 두 해상도 모두 아래 6개 trial만 사용한다.

```text
{material}_d5_z2.5_test1
{material}_d5_z2.5_test2
{material}_d5_z2.5_test3
{material}_d10_z3.5_test1
{material}_d10_z3.5_test2
{material}_d10_z3.5_test3
```

이를 위해 custom `dataset_index.json`을 만든다.

`train_e2e.py`는 `dataset_index_path`를 `--gt-dir/dataset_index.json`로 구성한다. 본 실험은 `gpu_on_the_fly`를 사용하므로 `--gt-dir`는 precomputed GT 저장소가 아니라 controlled trial index를 전달하는 용도로 사용한다.

---

## 3. 공통 실험 세팅

### 3.1 SATS 모델

학습 대상은 end-to-end SATS이다.

- Input: 16 taxel normalized time-series window
- Model: sensor-wise LSTM encoder, self-attention aggregation, local map decoder, CNN refiner
- Output: dense pressure map on fixed output grid
- GT: robot stage/loadcell 좌표와 힘으로 생성한 on-the-fly pressure map

논문 Methods에는 “raw multichannel signals are directly mapped to a dense pressure distribution by an end-to-end SATS model”로 기술한다.

### 3.2 고정 hyperparameter

모든 비교 실험에서 아래 값을 고정한다.

| 항목 | 값 | 이유 |
|---|---:|---|
| `gt_mode` | `gpu_on_the_fly` | 동일한 물리 GT를 GPU batch에서 생성, 저장 GT mismatch 방지 |
| output `grid_step_mm` | `0.5` | `[-10, 10]` 영역에서 `41 x 41` virtual taxel |
| `grid_size` | auto `41` | `grid_step_mm=0.5`에서 자동 계산 |
| `window_size` | `10` | SATS paper-style sliding window |
| `seq_len` | `1000` | d5/d10 loading cycle 보존 |
| `batch_size` | `2048` | 기존 SATS 설정과 비교 가능 |
| `epochs` | `50` primary, 필요 시 `100` final | 소재/해상도 비교는 비용 절감, 최종 모델은 확장 가능 |
| `seed` | `42` | fold 간 재현성 |
| `val_ratio` | `0` | trial-level split 사용 |
| `local_map_size` | auto | grid-step에 맞춰 기존 물리 범위 유지 |
| `use_gt_meta_cache` | true | 반복 학습 속도 안정화 |

주의:

- `xy_1mm`와 `xy_0.5mm`는 원천 스캔 pitch이다.
- 본 계획에서 output resolution은 둘 다 0.5 mm로 고정한다.
- output grid를 0.25 mm로 높이는 실험은 secondary/supplementary로 분리한다.

---

## 4. 실행 준비

### 4.1 merged BIN 생성

```bash
python3 sats/preprocessing/prepare_learning_data.py \
  --source-root skin_ws/raw_data \
  --source-material all \
  --learning-root learning_data \
  --stage merge
```

사전 확인:

```bash
python3 sats/preprocessing/prepare_learning_data.py \
  --source-root skin_ws/raw_data \
  --source-material all \
  --learning-root learning_data \
  --stage merge \
  --dry-run
```

기대값:

```text
planned trials: 31
```

### 4.2 공통 GT meta cache 생성

d5/d10 모두 사용하므로 `--exclude-diameters`를 쓰지 않는다.

```bash
python3 -m sats.training.build_gt_meta_cache \
  --raw-dir learning_data/sensor_raw_bin \
  --out-dir learning_data/gt_meta_cache_xy_d5d10_g05 \
  --include-materials eco20_xy1 eco50_xy1 ecomesh_xy1 ecomesh_xy0p5 \
  --grid-step-mm 0.5
```

cache manifest 확인:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("learning_data/gt_meta_cache_xy_d5d10_g05/manifest.json")
data = json.loads(path.read_text())
print("trials:", len(data["trials"]))
for row in data["trials"]:
    print(row["trial_id"], row["created"])
PY
```

---

## 5. Experiment A: xy_1mm 소재별 SATS 성능평가

### 5.1 실험 matrix

| Run group | Materials | Folds | Holdout per fold | 목적 |
|---|---|---:|---|---|
| `xy1_material_d5d10` | eco20, eco50, ecomesh | 3 | d5/d10 같은 test 번호 | 소재 효과 정량화 |

총 run 수:

```text
3 materials x 3 folds = 9 runs
```

### 5.2 학습 명령어

```bash
set -euo pipefail

OUT_DIR="sats/training/runs/xy1_material_d5d10"
CACHE_DIR="learning_data/gt_meta_cache_xy_d5d10_g05"

for MAT in eco20_xy1 eco50_xy1 ecomesh_xy1; do
  for FOLD in 1 2 3; do
    python3 -m sats.training.train_e2e \
      --gt-mode gpu_on_the_fly \
      --raw-dir learning_data/sensor_raw_bin \
      --gt-meta-cache-dir "$CACHE_DIR" \
      --include-materials "$MAT" \
      --val-ratio 0 \
      --val-trials "${MAT}_d5_z2.5_test${FOLD}" "${MAT}_d10_z3.5_test${FOLD}" \
      --grid-step-mm 0.5 \
      --epochs 50 \
      --seed 42 \
      --out-dir "$OUT_DIR" \
      --run-name "xy1_d5d10_${MAT}_fold${FOLD}_e2e_g05"
  done
done
```

### 5.3 결과 수집

```bash
mkdir -p history/fig_data/experiments_archive/xy1_material_d5d10

RUN_DIRS=$(find sats/training/runs/xy1_material_d5d10 \
  -mindepth 1 -maxdepth 1 -type d | sort)

python3 -m sats.tools.compare_sats_runs \
  --run-dirs $RUN_DIRS \
  --out history/fig_data/experiments_archive/xy1_material_d5d10/summary_by_run.csv
```

taxel-wise RMSE maps:

```bash
python3 -m sats.tools.analyze_taxel_rmse \
  --run-dirs $RUN_DIRS \
  --out-dir history/fig_data/experiments_archive/xy1_material_d5d10/taxel_rmse
```

### 5.4 논문용 분석

Run-level summary를 material별로 aggregate한다.

필수 표:

| Table | 내용 |
|---|---|
| Table A1 | material별 fold 평균 best val RMSE, std, 95% CI |
| Table A2 | d5/d10 holdout을 분리한 RMSE |
| Table A3 | material별 final train loss, best epoch, overfit gap |

필수 그림:

| Figure | 내용 |
|---|---|
| Fig. SATS-A | 소재별 mean RMSE bar plot with fold dots |
| Fig. SATS-B | 소재별 per-taxel RMSE heatmap |
| Fig. SATS-C | representative GT/pred/error maps for d5 and d10 |
| Fig. SATS-D | Fig.2 coverage metric vs SATS RMSE scatter |

논문 문장 skeleton:

```text
To test whether the mechanically broadened receptive fields translate into
computational super-resolution, we trained identical SATS models on each
xy_1mm material dataset using held-out scan repetitions for validation.
Both d5 and d10 indentation conditions were included in each fold.
```

---

## 6. Experiment B: ecomesh xy_1mm vs xy_0.5mm controlled comparison

### 6.1 custom dataset index 생성

Controlled comparison에서는 각 material별로 공통 6개 trial만 노출한다.

```bash
mkdir -p learning_data/trial_indices/ecomesh_xy1_common
mkdir -p learning_data/trial_indices/ecomesh_xy0p5_common

python3 - <<'PY'
import json
from pathlib import Path

def write_index(material: str, out_dir: str) -> None:
    trials = []
    for dia, z in [("d5", "2.5"), ("d10", "3.5")]:
        for n in [1, 2, 3]:
            trials.append({"trial_id": f"{material}_{dia}_z{z}_test{n}"})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_index.json").write_text(
        json.dumps({"trials": trials}, indent=2),
        encoding="utf-8",
    )

write_index("ecomesh_xy1", "learning_data/trial_indices/ecomesh_xy1_common")
write_index("ecomesh_xy0p5", "learning_data/trial_indices/ecomesh_xy0p5_common")
PY
```

### 6.2 controlled 학습 명령어

```bash
set -euo pipefail

OUT_DIR="sats/training/runs/ecomesh_resolution_controlled_d5d10"
CACHE_DIR="learning_data/gt_meta_cache_xy_d5d10_g05"

for MAT in ecomesh_xy1 ecomesh_xy0p5; do
  if [ "$MAT" = "ecomesh_xy1" ]; then
    GT_DIR="learning_data/trial_indices/ecomesh_xy1_common"
  else
    GT_DIR="learning_data/trial_indices/ecomesh_xy0p5_common"
  fi

  for FOLD in 1 2 3; do
    python3 -m sats.training.train_e2e \
      --gt-mode gpu_on_the_fly \
      --raw-dir learning_data/sensor_raw_bin \
      --gt-dir "$GT_DIR" \
      --gt-meta-cache-dir "$CACHE_DIR" \
      --include-materials "$MAT" \
      --val-ratio 0 \
      --val-trials "${MAT}_d5_z2.5_test${FOLD}" "${MAT}_d10_z3.5_test${FOLD}" \
      --grid-step-mm 0.5 \
      --epochs 50 \
      --seed 42 \
      --out-dir "$OUT_DIR" \
      --run-name "ecomesh_controlled_d5d10_${MAT}_fold${FOLD}_e2e_g05"
  done
done
```

### 6.3 controlled 결과 수집

```bash
mkdir -p history/fig_data/experiments_archive/ecomesh_resolution_controlled_d5d10

RUN_DIRS=$(find sats/training/runs/ecomesh_resolution_controlled_d5d10 \
  -mindepth 1 -maxdepth 1 -type d | sort)

python3 -m sats.tools.compare_sats_runs \
  --run-dirs $RUN_DIRS \
  --out history/fig_data/experiments_archive/ecomesh_resolution_controlled_d5d10/summary_by_run.csv

python3 -m sats.tools.analyze_taxel_rmse \
  --run-dirs $RUN_DIRS \
  --out-dir history/fig_data/experiments_archive/ecomesh_resolution_controlled_d5d10/taxel_rmse
```

### 6.4 해석 기준

Controlled comparison에서 아래를 확인한다.

- `ecomesh_xy0p5`가 `ecomesh_xy1` 대비 평균 RMSE를 얼마나 낮추는가.
- 개선이 d5에서만 나타나는가, d10에서도 유지되는가.
- error map 개선이 전체 면적에 퍼져 있는가, 특정 taxel/edge에서만 나타나는가.
- best epoch가 크게 늦어지거나 train/val gap이 커지는지 확인한다.

채택 기준:

| 조건 | 해석 |
|---|---|
| xy0p5가 mean RMSE를 10% 이상 낮추고 fold 3개 모두 일관 | 최종 성능 조건으로 xy0p5 우선 |
| xy0p5 개선이 5% 미만이거나 fold별 순위가 뒤집힘 | xy1이 더 경제적인 calibration 조건 |
| xy0p5는 d5만 개선, d10은 동일 | 작은 접촉 해상도 개선으로 제한해 보고 |
| xy0p5 train/val gap 증가 | dense scan overfit 가능성, regularization/epoch 재검토 |

---

## 7. Experiment C: ecomesh final calibration

### 7.1 목적

Experiment A/B로 최종 소재와 scan pitch를 정한 뒤, 논문 대표 그림과 demo에 사용할 final SATS 모델을 만든다. Final model의 성능값은 controlled holdout 결과와 혼동하지 않는다.

### 7.2 후보 1: ecomesh_xy1 final

`xy_1mm`를 최종 조건으로 채택할 경우:

```bash
python3 -m sats.training.train_e2e \
  --gt-mode gpu_on_the_fly \
  --raw-dir learning_data/sensor_raw_bin \
  --gt-meta-cache-dir learning_data/gt_meta_cache_xy_d5d10_g05 \
  --include-materials ecomesh_xy1 \
  --val-ratio 0 \
  --val-trials ecomesh_xy1_d5_z2.5_test3 ecomesh_xy1_d10_z3.5_test3 \
  --grid-step-mm 0.5 \
  --epochs 100 \
  --seed 42 \
  --out-dir sats/training/runs/ecomesh_final_d5d10 \
  --run-name ecomesh_xy1_d5d10_final_e2e_g05
```

### 7.3 후보 2: ecomesh_xy0p5 data-rich final

`xy_0.5mm`를 최종 조건으로 채택할 경우, d5의 추가 반복을 활용한다. Holdout은 마지막 반복을 사전에 고정한다.

```bash
python3 -m sats.training.train_e2e \
  --gt-mode gpu_on_the_fly \
  --raw-dir learning_data/sensor_raw_bin \
  --gt-meta-cache-dir learning_data/gt_meta_cache_xy_d5d10_g05 \
  --include-materials ecomesh_xy0p5 \
  --val-ratio 0 \
  --val-trials ecomesh_xy0p5_d5_z2.5_test9 ecomesh_xy0p5_d5_z2.5_test10 ecomesh_xy0p5_d10_z3.5_test3 \
  --grid-step-mm 0.5 \
  --epochs 100 \
  --seed 42 \
  --out-dir sats/training/runs/ecomesh_final_d5d10 \
  --run-name ecomesh_xy0p5_d5d10_final_e2e_g05
```

### 7.4 final 결과 산출

```bash
RUN_DIRS=$(find sats/training/runs/ecomesh_final_d5d10 \
  -mindepth 1 -maxdepth 1 -type d | sort)

mkdir -p history/fig_data/experiments_archive/ecomesh_final_d5d10

python3 -m sats.tools.compare_sats_runs \
  --run-dirs $RUN_DIRS \
  --out history/fig_data/experiments_archive/ecomesh_final_d5d10/summary_by_run.csv

python3 -m sats.tools.analyze_taxel_rmse \
  --run-dirs $RUN_DIRS \
  --out-dir history/fig_data/experiments_archive/ecomesh_final_d5d10/taxel_rmse
```

---

## 8. Metric 정의

### 8.1 primary metric

Primary metric은 holdout validation set에서 pressure map RMSE이다.

$$
RMSE = \sqrt{\frac{1}{N H W}\sum_{i=1}^{N}\sum_{x=1}^{W}\sum_{y=1}^{H}(P^{pred}_{i,x,y}-P^{gt}_{i,x,y})^2}
$$

보고 단위:

- model output scale 기준 RMSE
- 필요 시 GT scale과 물리 단위 변환을 함께 표기

### 8.2 secondary metrics

| Metric | 정의 | 논문에서의 역할 |
|---|---|---|
| per-taxel RMSE map | holdout 전체에서 taxel별 squared error 누적 | Fig.4B 스타일 error distribution |
| per-sample RMSE distribution | sample별 map RMSE histogram | Fig.4C 스타일 통계 |
| d5-only RMSE | d5 holdout만 집계 | 작은 접촉 SR 성능 |
| d10-only RMSE | d10 holdout만 집계 | 큰 접촉/포화 조건 성능 |
| localization error | predicted map peak 좌표와 GT/loadcell 좌표 차이 | “위치 추정” supplementary |
| integrated force error | map sum 또는 calibrated integral과 loadcell Fz 차이 | 힘 크기 decoupling 검증 |
| edge/interior RMSE | 가장자리 taxel 인접 영역 vs 중앙 영역 | boundary artifact 확인 |
| best epoch / train-val gap | best val epoch, final train loss 비교 | overfitting 및 안정성 |

### 8.3 fold aggregate

각 실험군은 run-level CSV를 만든 뒤 아래 방식으로 집계한다.

```text
mean = fold 평균
std  = fold 표준편차
95% CI = 1.96 * std / sqrt(n_folds)
```

fold 수가 3으로 작으므로 p-value 중심 해석은 피하고, effect size와 fold 일관성을 함께 제시한다.

---

## 9. 결과 디렉터리 규칙

학습 run:

```text
sats/training/runs/
  xy1_material_d5d10/
    xy1_d5d10_{material}_fold{1|2|3}_e2e_g05/
      config.json
      history.json
      best_model.pt
      last_model.pt
  ecomesh_resolution_controlled_d5d10/
    ecomesh_controlled_d5d10_{material}_fold{1|2|3}_e2e_g05/
  ecomesh_final_d5d10/
    ecomesh_{xy1|xy0p5}_d5d10_final_e2e_g05/
```

논문/분석 산출물:

```text
history/fig_data/experiments_archive/
  xy1_material_d5d10/
    summary_by_run.csv
    summary_by_material.csv
    taxel_rmse/
    figures/
  ecomesh_resolution_controlled_d5d10/
    summary_by_run.csv
    summary_by_resolution.csv
    taxel_rmse/
    figures/
  ecomesh_final_d5d10/
    summary_by_run.csv
    taxel_rmse/
    representative_maps/
```

각 run의 `config.json`은 논문 Methods 재현성의 source of truth로 보존한다. `history.json`은 learning curve와 best epoch를 만드는 데 사용한다.

---

## 10. 분석 절차

### 10.1 run sanity check

각 run 종료 후 다음을 확인한다.

```bash
python3 - <<'PY'
import json
from pathlib import Path

for rd in sorted(Path("sats/training/runs").glob("*/*")):
    cfg = rd / "config.json"
    hist = rd / "history.json"
    if not cfg.exists() or not hist.exists():
        continue
    c = json.loads(cfg.read_text())
    h = json.loads(hist.read_text())
    best = min(h, key=lambda r: r["val_rmse"])
    print(rd)
    print("  include_materials:", c.get("include_materials"))
    print("  val_trials:", c.get("val_trials"))
    print("  best:", best["epoch"], best["val_rmse"])
PY
```

검출해야 할 문제:

- validation trial이 train pool에 섞였는지
- fold별 val trials가 d5/d10 모두 포함되는지
- `grid_step_mm=0.5`, `gt_mode=gpu_on_the_fly`가 모든 run에서 유지되는지
- best epoch가 1-2 epoch에 고정되어 학습 실패처럼 보이는지
- final train loss가 낮지만 val RMSE가 나쁜 overfit run이 있는지

### 10.2 material 결과 해석 순서

1. fold별 best val RMSE를 확인한다.
2. material별 mean/std/CI를 계산한다.
3. d5-only, d10-only로 나누어 같은 순위가 유지되는지 본다.
4. per-taxel RMSE heatmap에서 edge artifact, 특정 sensor 주변 오류를 확인한다.
5. Fig.2의 coverage, overlap, active taxel 결과와 SATS RMSE의 방향성을 비교한다.

### 10.3 ecomesh resolution 결과 해석 순서

1. controlled 3-fold에서 `xy1`과 `xy0p5`의 평균 RMSE를 비교한다.
2. d5와 d10으로 나누어 dense scan의 이득이 어느 접촉 크기에서 나타나는지 확인한다.
3. error map이 전체적으로 낮아졌는지, sparse location만 좋아졌는지 본다.
4. data-rich final model은 controlled 결과보다 낮은 오류를 보일 수 있으나, 논문 결론에서는 별도 문단으로 분리한다.

---

## 11. 논문 figure/table 계획

### Main figure 후보

| Panel | 내용 | 입력 파일 |
|---|---|---|
| Fig.3A | SATS pipeline schematic, raw 16-channel sequence to 41x41 pressure map | schematic |
| Fig.3B | xy_1mm 소재별 holdout RMSE bar with fold dots | `xy1_material_d5d10/summary_by_material.csv` |
| Fig.3C | 소재별 representative d5 GT/pred/error maps | best fold checkpoints |
| Fig.3D | 소재별 per-taxel RMSE heatmaps | `taxel_rmse/*.npy`, png |
| Fig.3E | ecomesh xy1 vs xy0p5 controlled RMSE | `ecomesh_resolution_controlled_d5d10/summary_by_resolution.csv` |
| Fig.3F | final ecomesh prediction examples for d5 and d10 | final run |

### Supplementary figure 후보

| Figure | 내용 |
|---|---|
| Fig. S-SATS1 | all training curves by material/fold |
| Fig. S-SATS2 | d5-only vs d10-only RMSE split |
| Fig. S-SATS3 | per-sample RMSE histograms |
| Fig. S-SATS4 | localization error maps from pressure peak |
| Fig. S-SATS5 | force-conditioned RMSE, low-to-high Fz bins |
| Fig. S-SATS6 | ablation: output grid 0.25 mm if performed |

### Table 후보

| Table | 내용 |
|---|---|
| Table 1 | material ablation, xy_1mm, d5+d10, 3-fold mean/std |
| Table 2 | ecomesh resolution comparison, controlled 3-fold |
| Table S1 | all run configs: material, fold, val trials, best epoch, seed |
| Table S2 | final model performance and qualitative figure source |

---

## 12. 논문 Methods 초안

아래 문단은 결과가 채워지면 바로 Methods에 들어갈 수 있는 형태다.

```text
For computational calibration, synchronized robot-stage coordinates, load-cell
force readings, and 16-channel barometric sensor signals were merged into a
trial-level dataset. Ground-truth pressure maps were generated on the fly from
the measured contact position and normal force using a spherical-indenter
elastic half-space approximation. The output pressure map covered a
20 mm x 20 mm sensing area with 0.5 mm spacing, yielding a 41 x 41 virtual
taxel grid.

To evaluate the effect of the pressure-transfer material, we trained identical
end-to-end SATS models for eco20, eco50, and ecomesh using the xy_1mm datasets.
Both d5 and d10 indenters were included during training. Model selection and
evaluation were performed with trial-level holdout folds, where the d5 and d10
trials with the same repetition index were held out together.

To isolate the effect of scan pitch, ecomesh models trained from xy_1mm and
xy_0.5mm datasets were compared using the same number of scan repetitions
(test1-test3) and identical d5/d10 holdout folds. A separate data-rich final
model was trained only after the controlled comparison was completed.
```

---

## 13. Decision log

실험 완료 후 아래 항목을 채운다.

| 결정 | 기준 | 결과 |
|---|---|---|
| 최종 소재 | xy_1mm material CV에서 lowest/stable RMSE | TBD |
| 최종 scan pitch | ecomesh controlled CV + cost/benefit | TBD |
| 논문 대표 model | final calibration run | TBD |
| main figure에 넣을 d5/d10 example | median-error sample 또는 representative position | TBD |
| supplementary로 보낼 항목 | fold curves, extra heatmaps, output grid ablation | TBD |

---

## 14. 실행 체크리스트

- [ ] `prepare_learning_data.py --dry-run`에서 planned trials 31 확인
- [ ] merged BIN 생성 완료
- [ ] GT meta cache manifest 생성 및 trial 누락 확인
- [ ] Experiment A 9 runs 완료
- [ ] Experiment A summary/taxel RMSE 생성
- [ ] Experiment B custom index 생성
- [ ] Experiment B 6 runs 완료
- [ ] Experiment B summary/taxel RMSE 생성
- [ ] Controlled result로 final ecomesh 조건 결정
- [ ] Experiment C final run 완료
- [ ] representative GT/pred/error maps 생성
- [ ] 논문용 table/figure export
- [ ] Methods 문단의 숫자와 config 재검증
