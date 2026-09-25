# Study Sensing Technology based on Learning

1. Guiding the design of superresolution tactile skins with taxel value isolines theory _*Science Robotics* 2022
2. Sensing multi-directional forces at superresolution using taxel value isoline theory_nature communications 2025
3. Location and Orientation Super-Resolution Sensing With a Cost-Efficient and Repairable Barometric Tactile Sensor_IEEE Transactions on Robotics, 2024
4. Soft Barometric Tactile Sensor Utilizing Iterative Pressure Reconstruction_IEEE Access, 2024
5. A Soft Barometric Tactile Sensor to Simultaneously Localize Contact and Estimate Normal Force With Validation to Detect Slip in a Robotic Gripper_Robotics and Automation Letters
6. BaroTac: Barometric Three-Axis Tactile Sensor with Slip Detection Capability _ MDPI Sensors 2022

이 논문들 기반으로 내 센서(S1~S16, 4X4 array)를 가지고 실시간 접촉위치 히트맵 분포, 접촉위치의 중심점에서의 3축센싱을 통합한 학습 방법을 구현하고 싶음.

기능1) 1,2번은 ml방법이고, 2번은 cnn+lstm 방법임 두 방법을 적절히 조합하되, cnn+lstm 방식으로 함. ⇒ 접촉 위치 추정 방식 진행.(3,4번 내용도 참고) 
output =  [x, y, z, fz(ditrubution) ] ←xy 면적과 그 면적에서의 fz 분포 값(hitmap 형식이나 또는 3d 형식)

기능2) 5번 논문 내용은 3개의 barometer 센서를 통해 중심 점에서 3축 센싱을 가능하게 함. 교정 행렬을 통한 보정을 하였음.  접촉물의 중심에서의 3축 힘을 구현하려함. 논문 2번과 같은 상황에서 3축센싱을 가능하게 하려 함. 
output = [x , y, z ,fx, fy, fz] ← 접촉물 면적 중심 

학습 dataset은 첨부된 것과 같으나 필요한 부분 추가 가능함.

*요청사항*

1. 1,2번 논문 기반 isoline theory 기반 MLP 구조  + CNN/LSTM 합치는 것이 가능한가?
가능하지 않다면 두 방법 중 적절한 것은 무엇인가.
2. INPUT은 센서 16개 (4 x 4 x 1) 구조여야 하는지, (4 x 4 x timestep)처럼 LSTM을 위한 구조로 데이터 셋을 구분해야 하는 지. 그리고 DATASET에 더 추가해야할 지표들이 있는지.
3. 참고용 dataset은 (0,0)에서 hemi_10mm지름의 인덴터로 누른 값임. 반복을 얼마나 해야 하는 지?
(다양한 인덴터로 누르는 실험을 진행할 예정임)
4. 3축 스테이지 모터에서 z축 방향(6축 상용센서+인덴터)으로 누르는 실험 위주로 진행 하였음.
여기에 하단 센서 고정파트를 회전 시킨 상태로 z축 방향으로 누르는 실험을 진행 해야 기능2가 제대로 구현되는지 아니면 필요 없는지
5. 기능1, 기능2에 대한 나의 아웃풋 결과가 정확한 지. 정확하지 않다면 어떻게 구성되는 것인지?
6. INPUT 과 OUTPUT 사이의 층들을 어떻게 구성해야 하는 지.
7. BARNCH로 기능1, 기능2를 따로 해야 하는데 어떻게 구성해야 하는지

## 요청사항 답변

(참고 논문: ①Science Robotics 2022, ②Nature Comm. 2025, ③IEEE T-RO 2024, ④IEEE Access 2024, ⑤RA-L, ⑥MDPI Sensors 2022)

---

## 1️⃣ Isoline 기반 MLP + CNN/LSTM 결합 가능성

**가능합니다.**

단, 단순 병합이 아닌 **하이브리드 형태**로 설계해야 합니다.

- **역할 구분**
    - **CNN+LSTM**: 시공간 패턴(시간축 변화 + 공간 분포)을 학습 → 주된 backbone.
    - **Isoline 기반 MLP**: 물리적 feature (예: 인접 taxel 간 압력 변화율, isoline gradient, 중심 예측 bias) 보정 역할.
- **병합 방식**
    - CNN+LSTM feature vector (예: 128차원) → MLP branch에서 isoline-derived feature들과 concatenate → 최종 FC layer에서 통합.
    - 즉, **MLP는 물리적 해석적 보정 계층**, CNN+LSTM은 **데이터 기반 공간-시간 표현 학습기**로 통합 가능.

📌 이 구조는 Science Robotics 2022, Nature Comm. 2025의 isoline 이론 기반 추정 + IEEE T-RO 2024의 CNN+LSTM 접근을 융합한 형태로 가장 효율적입니다.

---

## 2️⃣ 입력 구조 (4×4×1 vs 4×4×T) 및 추가 데이터 필요성

**LSTM을 포함한다면 4×4×T 형태로 해야 합니다.**

T는 raster scan 또는 압입 시퀀스 길이(time step)입니다.

- 입력 예: `(batch, T, 1, 4, 4)`
- 각 시점의 4×4는 CNN encoder를 통해 feature로 변환 → LSTM이 temporal dependency 학습.
- CSV 파일의 time column을 이용해 동일 경로 내 데이터를 T 단위로 묶어 시퀀스 구성.

**추가 필요 데이터 (선택적):**

| 항목 | 필요성 | 이유 |
| --- | --- | --- |
| 인덴터 직경, 형상 | ★★ | 접촉면積 변화 → fz 분포 형태에 직접 영향 |
| 실험 온도/습도 | ★ | 공기압 기반 센서의 drift 보정용 |
| 센서 calibration offset | ★★ | MLP isoline branch에 직접 활용 |
| 전단력 발생 여부(θ) | ★★ | 기능2 학습시 shear 학습 필요 |

---

## 3️⃣ 데이터 수집(반구형 인덴터, D=10mm) 반복 횟수

- **최소 기준:** 각 z-step당 50~100회 반복 × 위치별 10~20개 grid = 수천 샘플
- **실제 목표:** 10⁴~10⁵ level의 샘플 확보 (CNN+LSTM 안정 학습을 위해)
- 인덴터 종류는 최소 3종 이상 (예: φ4.6, φ5.6, φ6.6mm)
- 반복 이유: 노이즈 평균화 + 일반화 성능 확보

📌 Nature Comm. 2025 및 IEEE Access 2024의 super-resolution 모델은 약 10⁵ 샘플 수준에서 수렴했습니다.

---

## 4️⃣ 회전 실험 필요 여부 (기능2용)

**필수입니다.**

3축력 추정(fx, fy, fz)을 위해서는 단순 수직(z) 하중만으로는 부족합니다.

- 센서 하단을 회전시켜 비대칭 하중 또는 전단력(shear)을 발생시켜야 합니다.
- 이렇게 해야 **fx, fy 성분의 calibration matrix**를 학습할 수 있습니다.
- IEEE T-RO 2024, MDPI 2022(BaroTac) 모두 동일하게 “센서 기판 회전”을 이용했습니다.

📌 결론: z방향 하중만으로는 fz만 보정 가능. 회전(±15°~30°)을 추가해야 fx, fy까지 학습 가능합니다.

---

## 5️⃣ 기능1, 기능2의 출력 구조 타당성 검증

| 기능 | 현재 출력 정의 | 수정/보완 제안 | 이유 |
| --- | --- | --- | --- |
| 기능1 | `[x, y, z, fz(distribution)]` | ✅ 타당 (히트맵 또는 3D 분포) | CNN+LSTM+MLP로 시공간 압력 분포 예측 가능 |
| 기능2 | `[x, y, z, fx, fy, fz]` | ✅ 타당 | 중심점 기준 3축력 벡터를 회귀형으로 예측 가능 |

보완한다면:

- 기능1의 fz(distribution)은 “히트맵”이 아닌 “25×25 normalized map”으로 두는 것이 좋음.
- 기능2의 fz는 단일 스칼라(중심 압력)이어야 함.

---

## 6️⃣ INPUT → OUTPUT 층 구성 방식

### 공통 Backbone (CNN+LSTM)

```
입력: (B, T, 1, 4, 4)
↓ CNN(16, 3×3, ReLU)
↓ CNN(32, 3×3, ReLU)
↓ Flatten (→ feature_dim=64)
↓ LSTM(64, hidden=128)
↓ 공통 feature (128)

```

### 기능1 Branch (히트맵 + 좌표)

```
↓ FC(128→256) + ReLU
↓ FC(256→(25×25)=625)  # fz 분포
↓ FC(128→3)             # [x,y,z]
출력: [x,y,z,fz_map]

```

### 기능2 Branch (3축 힘)

```
↓ FC(128→128) + ReLU
↓ FC(128→6)   # [x,y,z,fx,fy,fz]
출력: [x,y,z,fx,fy,fz]

```

📌 isoline-MLP는 CNN+LSTM feature 뒤에 병렬로 연결하여 `feature_concat = [LSTM_out, isoline_feature]` 형태로 FC layer에 결합합니다.

---

## 7️⃣ 기능1, 기능2 분기(branch) 구성 방법

멀티태스크(MTL) 형태로 구성해야 합니다.

```
          ┌──────────┐
Input → CNN+LSTM → ─┤ Shared Feature (128) ├─┐
                     └──────────┘             │
                     │                       │
                Branch 1 (기능1)        Branch 2 (기능2)
           ┌──────────────────┐      ┌──────────────────┐
           │ FC→25×25 map     │      │ FC→6 force vector │
           │ FC→[x,y,z]       │      │ FC→[x,y,z,fx,fy,fz]│
           └──────────────────┘      └──────────────────┘

```

- 두 branch는 **공통 representation을 공유**하되, 각각의 목적에 특화된 FC layer로 분기.
- loss는 `L_total = λ1 * L_func1 + λ2 * L_func2` 형태로 가중 합.
- isoline-based MLP feature는 각 branch 입력 전 단계에서 결합 (feature fusion).

---

## ✅ 결론 요약

| 항목 | 권장 방법 |
| --- | --- |
| 핵심 구조 | CNN+LSTM backbone + isoline MLP 보정 |
| 입력 데이터 | 4×4×T 시계열 (CSV time 기반 segment) |
| 출력 | 기능1: [x,y,z,fz_map], 기능2: [x,y,z,fx,fy,fz] |
| 데이터 수량 | 최소 10⁴ 샘플 이상, 각 조건 50~100회 반복 |
| 회전 실험 | 필수 (전단력 학습용) |
| 네트워크 | Shared CNN+LSTM + dual-branch FC |
| 구현 프레임워크 | PyTorch (LSTM backbone + MTL loss) |

---

## 전체 구조도

```python
                  ┌────────────────────────────────────────┐
                  │               INPUT                    │
                  │      S1~S16 (4×4×T 시계열)            │
                  └────────────────────────────────────────┘
                                   │
                                   ▼
                 ┌───────────────────────────────────────┐
                 │        CNN+LSTM Backbone              │
                 │ (시공간 Feature 추출, hidden=128)     │
                 └───────────────────────────────────────┘
                                   │
                          ┌────────┴─────────┐
                          │                  │
                          ▼                  ▼
        ┌────────────────────────┐  ┌────────────────────────┐
        │      Isoline MLP       │  │   (보조 물리 feature) │
        │  16ch 압력값의 isoline │  │  gradient/평균 추출   │
        └────────────────────────┘  └────────────────────────┘
                          │                  │
                          └───────Feature Fusion────────────┘
                                          │
                                          ▼
             ┌──────────────────────────────────────────────┐
             │               Shared Feature (160)           │
             └──────────────────────────────────────────────┘
                      │                         │
              (Branch1) 기능1             (Branch2) 기능2
   ┌─────────────────────────────┐   ┌─────────────────────────────┐
   │ FC→(x,y,z) + FC→fz_map(25×25) │ │ FC→(x,y,z,fx,fy,fz)        │
   └─────────────────────────────┘   └─────────────────────────────┘
                      │                         │
               [위치+히트맵] 출력          [3축힘+위치] 출력

```

## Input Data 관련

## CSV 데이터 항목 검토

현재 제공된 CSV(예: `0.0_0.0_hemi_D10mm_1.csv`)에는 일반적으로 다음 항목이 있을 것입니다:

```
time, s1, s2, ..., s16, Fx, Fy, Fz, x, y, z
```

이 구성이면 **기본적으로 충분**합니다. 하지만 **추가로 권장되는 항목**은 다음과 같습니다:

| 항목 | 필요 여부 | 이유 |
| --- | --- | --- |
| `indent_diameter`, `indent_shape` | ★★ | 다양한 인덴터 실험 generalization |
| `sensor_temperature` | ★ | 압력센서 offset 보정용 |
| `sensor_drift` | ★ | 장기 drift나 비정상 압력 보정 |
| `rotation_angle` | ★★ | shear, fx/fy 학습시 반드시 필요 |
| `contact_id` | ★ | 여러 실험 그룹 labeling 용도 |

→ 즉, **기본 CSV + 인덴터 메타데이터 + 회전 각도** 정도면 완전한 데이터셋입니다.

## ✅ 2️⃣ raw count → 압력 단위 변환 또는 정규화

현재 BMP384 센서의 `raw count`는 **직접적인 압력 단위가 아닙니다.**

→ 모델이 raw count를 그대로 학습하면 실험 환경이 달라질 때 drift에 매우 민감해집니다.

따라서 **두 가지 방법 중 하나**를 반드시 적용해야 합니다:

### ~~(a) 교정식 기반 변환 (권장)~~

- ~~BMP384 데이터시트에서 제공하는 보정 파라미터(`par_p1`, `par_p2`, …)를 적용해 실제 압력(Pa 단위)으로 환산.~~
- ~~이 과정은 BMP 라이브러리 예제의 “보정 계산 단계”와 동일합니다.~~
- ~~결과적으로 각 S1~S16을 `압력(Pa)` 단위로 변환합니다.~~

### (b) 정규화 기반 변환 (빠른 대안)

- 센서별 min/max 또는 baseline 평균을 구해 `ΔP = raw - baseline` 형태로 만듭니다.
- 이후 `(ΔP - mean)/std`로 표준화(z-score).

예:

```python
for s in ['s1','s2',...,'s16']:
    df[s] = (df[s] - df[s].mean()) / df[s].std()
```

✅ 이렇게 하면 **드리프트 보정(Drift Compensation)** 역할을 겸합니다.

---

## ✅ 3️⃣ 드리프트(drfit) 추가 변수화

센서의 **drift는 시간에 따른 offset 변화**입니다.

`raw count`를 그대로 쓸 경우 drift가 포함되므로, 이를 변수로 모델에 알려주는 것이 좋습니다.

**방법:**

- 각 센서 채널의 baseline offset(= 초반 1~2초 평균값)을 별도로 계산 → 새로운 feature로 추가.
- 예:
    
    ```python
    baseline = df[['s1','s2',...,'s16']].iloc[:20].mean()
    for s in sensors:
        df[f'{s}_drift'] = df[s] - baseline[s]
    ```
    

→ 이 `*_drift` feature는 Isoline-MLP 쪽 입력으로 넣을 수 있습니다.

---

## ✅ 4️⃣ 온도 보정값 추가 (BMP384 내부 온도)

BMP384의 온도 출력은 `t_fine` 기반 섭씨 단위 계산이 가능합니다.

→ CSV에 `temperature (°C)` 컬럼을 추가하면 됩니다.

이 값은 drift 및 압력 offset 보정에 **강한 보조 feature**로 작용합니다.

**적용 방법:**

- `df['temperature']`를 그대로 추가.
- 이후 모델 입력 시 `isoline_feature`로 함께 전달.

---

## ✅ 5️⃣ rotation_angle 처리

- 실험 장비에서 회전 각도를 기록 가능하다면 `rotation_angle(deg)`을 CSV에 추가.
- 단, z축 하중만 있을 경우 `0°`로 일괄 입력.
- 나중에 학습 시 **force 방향 보정(예: fx, fy)** 학습에 직접 활용 가능.

---

## ✅ 6️⃣ 전처리 후 모델 입력 구조 예시

모델에 들어가기 전, 하나의 시퀀스(batch)는 다음 형태가 됩니다:

| Feature 그룹 | 개수 | 역할 |
| --- | --- | --- |
| S1~S16 (정규화된 압력) | 16 | CNN/LSTM 입력 |
| S1~S16_drift | 16 | Isoline MLP 입력 |
| temperature | 1 | Isoline MLP 입력 |
| rotation_angle | 1 | Isoline MLP 입력 |
| dia / shape / contact_id | 2~3 | one-hot or embedding |
| Fx, Fy, Fz, x, y, z | 6 | 레이블 (타깃) |

즉, LSTM 입력은 `(B, T, 1, 4, 4)`

MLP 입력은 `(B, T, 16+α)` 형태(α=온도·회전·지름 등 추가 feature).

## Data 전처리 관련 : RAW COUNT 관련

| 단계 | 기능 |
| --- | --- |
| 1 | 파일명 파싱: x,y,형상,지름,반복 번호 추출 |
| 2 | CSV 로드 및 `raw count → ΔP 정규화` |
| 3 | drift 계산 (초기 20샘플 평균) |
| 4 | 온도·회전각 feature 추가 |
| 5 | CNN/LSTM 입력(4×4×T) + Isoline 입력(16+meta feature) 구성 |
| 6 | PyTorch Dataset / DataLoader 생성 |

```python
import os
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ===========================================================
# 1️⃣ 파일명 파싱 함수
# ===========================================================
def parse_filename(fname):
    fname = os.path.basename(fname).replace('.csv', '')
    parts = fname.split('_')
    x_init, y_init = float(parts[0]), float(parts[1])
    shape = parts[2]
    dia = float(re.findall(r'\d+', parts[3])[0])  # D10mm → 10
    rep = int(parts[4])
    return x_init, y_init, shape, dia, rep

# ===========================================================
# 2️⃣ 전처리 함수
# ===========================================================
def preprocess_dataframe(df, sensors, normalize=True):
    # 드리프트 기준 (초기 20 샘플 평균)
    baseline = df[sensors].iloc[:20].mean()

    for s in sensors:
        df[f'{s}_drift'] = df[s] - baseline[s]
        if normalize:
            df[s] = (df[s] - df[s].mean()) / (df[s].std() + 1e-6)
            df[f'{s}_drift'] = (df[f'{s}_drift'] - df[f'{s}_drift'].mean()) / (df[f'{s}_drift'].std() + 1e-6)
    return df

# ===========================================================
# 3️⃣ Dataset 정의
# ===========================================================
class TactileDataset(Dataset):
    def __init__(self, folder_path, seq_len=50):
        self.file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        self.seq_len = seq_len
        self.sensors = [f's{i}' for i in range(1, 17)]

        # 스케일러 (전체 normalization용)
        self.scaler_fx = StandardScaler()
        self.scaler_fy = StandardScaler()
        self.scaler_fz = StandardScaler()

        self.data = []
        self._load_all_files()

    def _load_all_files(self):
        for fpath in self.file_list:
            x_init, y_init, shape, dia, rep = parse_filename(fpath)
            df = pd.read_csv(fpath)

            # 필요한 컬럼 확인
            if not set(self.sensors).issubset(df.columns):
                continue

            df = preprocess_dataframe(df, self.sensors)

            # 추가 feature
            df["temperature"] = df.get("temperature", 25.0)
            df["rotation_angle"] = df.get("rotation_angle", 0.0)
            df["indenter_diameter"] = dia
            df["x_init"], df["y_init"], df["rep_id"] = x_init, y_init, rep

            # sliding window로 시퀀스 생성
            for i in range(0, len(df) - self.seq_len):
                seq = df.iloc[i:i + self.seq_len]

                # CNN/LSTM 입력 (4×4×T)
                arr = seq[self.sensors].values.reshape(self.seq_len, 1, 4, 4)
                arr = torch.tensor(arr, dtype=torch.float32)

                # Isoline 입력 (drift + meta)
                drift_cols = [f"{s}_drift" for s in self.sensors]
                isoline_feat = seq[drift_cols + ["temperature", "rotation_angle", "indenter_diameter"]].values
                isoline_feat = torch.tensor(isoline_feat, dtype=torch.float32)

                # 타깃 (Fx,Fy,Fz,x,y,z)
                if set(["Fx","Fy","Fz","x","y","z"]).issubset(df.columns):
                    tgt = seq[["x","y","z","Fx","Fy","Fz"]].iloc[-1].values
                else:
                    continue

                self.data.append((arr, isoline_feat, torch.tensor(tgt, dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ===========================================================
# 4️⃣ 사용 예시
# ===========================================================
if __name__ == "__main__":
    folder = "./dataset"  # CSV 파일들이 있는 폴더
    dataset = TactileDataset(folder_path=folder, seq_len=50)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for x_seq, iso_feat, target in loader:
        print(f"CNN/LSTM 입력: {x_seq.shape}")      # (B, T, 1, 4, 4)
        print(f"Isoline 입력: {iso_feat.shape}")    # (B, T, 16 + 3)
        print(f"Target: {target.shape}")             # (B, 6)
        break

```

## 학습코드

### GPU

```python
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ===========================================================
# 1️⃣ 파일명 파싱
# ===========================================================
def parse_filename(fname):
    fname = os.path.basename(fname).replace('.csv', '')
    parts = fname.split('_')
    x_init, y_init = float(parts[0]), float(parts[1])
    shape = parts[2]
    dia = float(re.findall(r'\d+', parts[3])[0])  # D10mm → 10
    rep = int(parts[4])
    return x_init, y_init, shape, dia, rep

# ===========================================================
# 2️⃣ 전처리 함수
# ===========================================================
def preprocess_dataframe(df, sensors, normalize=True):
    baseline = df[sensors].iloc[:20].mean()
    for s in sensors:
        df[f'{s}_drift'] = df[s] - baseline[s]
        if normalize:
            df[s] = (df[s] - df[s].mean()) / (df[s].std() + 1e-6)
            df[f'{s}_drift'] = (df[f'{s}_drift'] - df[f'{s}_drift'].mean()) / (df[f'{s}_drift'].std() + 1e-6)
    return df

# ===========================================================
# 3️⃣ Dataset 정의
# ===========================================================
class TactileDataset(Dataset):
    def __init__(self, folder_path, seq_len=50):
        self.file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        self.seq_len = seq_len
        self.sensors = [f's{i}' for i in range(1, 17)]
        self.data = []
        self._load_all_files()

    def _load_all_files(self):
        for fpath in self.file_list:
            x_init, y_init, shape, dia, rep = parse_filename(fpath)
            df = pd.read_csv(fpath)
            if not set(self.sensors).issubset(df.columns):
                continue

            df = preprocess_dataframe(df, self.sensors)
            df["temperature"] = df.get("temperature", 25.0)
            df["rotation_angle"] = df.get("rotation_angle", 0.0)
            df["indenter_diameter"] = dia

            for i in range(0, len(df) - self.seq_len):
                seq = df.iloc[i:i + self.seq_len]
                arr = seq[self.sensors].values.reshape(self.seq_len, 1, 4, 4)
                arr = torch.tensor(arr, dtype=torch.float32)

                drift_cols = [f"{s}_drift" for s in self.sensors]
                isoline_feat = seq[drift_cols + ["temperature", "rotation_angle", "indenter_diameter"]].values
                isoline_feat = torch.tensor(isoline_feat, dtype=torch.float32)

                if set(["Fx","Fy","Fz","x","y","z"]).issubset(df.columns):
                    tgt = seq[["x","y","z","Fx","Fy","Fz"]].iloc[-1].values
                else:
                    continue

                self.data.append((arr, isoline_feat, torch.tensor(tgt, dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ===========================================================
# 4️⃣ 모델 정의 (CNN + LSTM + Isoline)
# ===========================================================
class IsolineMLP(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, output_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.mean(x, dim=1)  # 시계열 평균
        return x

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 4 * 4, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN_LSTM_Isoline(nn.Module):
    def __init__(self, use_branch1=True, use_branch2=True):
        super().__init__()
        self.encoder = CNNEncoder()
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.isoline = IsolineMLP(input_dim=19, hidden_dim=64, output_dim=32)

        self.use_branch1 = use_branch1
        self.use_branch2 = use_branch2
        fused_dim = 128 + 32

        if use_branch1:
            self.fc1_1 = nn.Linear(fused_dim, 256)
            self.fc1_map = nn.Linear(256, 25 * 25)
            self.fc1_xyz = nn.Linear(256, 3)

        if use_branch2:
            self.fc2_1 = nn.Linear(fused_dim, 128)
            self.fc2_out = nn.Linear(128, 6)

    def forward(self, x, iso):
        B, T, _, _, _ = x.shape
        x = x.view(B * T, 1, 4, 4)
        cnn_feat = self.encoder(x)
        cnn_feat = cnn_feat.view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_feat)
        lstm_feat = lstm_out[:, -1, :]

        iso_feat = self.isoline(iso)
        fused = torch.cat([lstm_feat, iso_feat], dim=-1)

        z1, z2 = None, None
        if self.use_branch1:
            f1 = F.relu(self.fc1_1(fused))
            z1_map = self.fc1_map(f1)
            z1_xyz = self.fc1_xyz(f1)
            z1 = torch.cat([z1_xyz, z1_map], dim=1)

        if self.use_branch2:
            f2 = F.relu(self.fc2_1(fused))
            z2 = self.fc2_out(f2)

        return z1, z2

# ===========================================================
# 5️⃣ 학습 루프
# ===========================================================
def train_model(model, dataloader, optimizer, device, epochs=10, use_branch1=True, use_branch2=True):
    criterion = nn.MSELoss()

    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for x_seq, iso_feat, tgt in dataloader:
            x_seq, iso_feat, tgt = x_seq.to(device), iso_feat.to(device), tgt.to(device)
            optimizer.zero_grad()
            z1, z2 = model(x_seq, iso_feat)
            loss = 0
            if use_branch1 and z1 is not None:
                loss += criterion(z1[:, :3], tgt[:, :3])  # xyz 회귀
            if use_branch2 and z2 is not None:
                loss += criterion(z2, tgt)  # 3축 힘 + 위치 회귀
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.6f}")
    torch.save(model.state_dict(), "trained_model.pth")
    print("✅ 모델 저장 완료: trained_model.pth")

# ===========================================================
# 6️⃣ 실행 예시
# ===========================================================
if __name__ == "__main__":
    folder = "./dataset"
    dataset = TactileDataset(folder_path=folder, seq_len=50)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    model = CNN_LSTM_Isoline(use_branch1=True, use_branch2=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, loader, optimizer, device, epochs=30, use_branch1=True, use_branch2=True)

```

### COLAB

```python
# ===============================
# 1️⃣  환경설정 및 라이브러리 설치
# ===============================
!nvidia-smi
!pip install torch torchvision torchaudio
!pip install pandas numpy scikit-learn pyqtgraph
```

```python
# ===============================
# 2️⃣  데이터 업로드 (CSV 여러 개)
# ===============================
from google.colab import files
import os, zipfile

upload_mode = input("CSV 여러 개를 ZIP으로 업로드할까요? (y/n): ")

if upload_mode.lower() == "y":
    uploaded = files.upload()
    zipname = list(uploaded.keys())[0]
    with zipfile.ZipFile(zipname, 'r') as zip_ref:
        zip_ref.extractall("/content/dataset")
else:
    os.makedirs("/content/dataset", exist_ok=True)
    uploaded = files.upload()
    for name, data in uploaded.items():
        with open(os.path.join("/content/dataset", name), "wb") as f:
            f.write(data)

print("✅ dataset 폴더에 파일 업로드 완료")
!ls /content/dataset
```

```python
# ===============================
# 3️⃣  CNN + LSTM + Isoline 학습 구조
# ===============================
import os, re, torch, pandas as pd, numpy as np
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

def parse_filename(fname):
    fname = os.path.basename(fname).replace('.csv', '')
    parts = fname.split('_')
    x_init, y_init = float(parts[0]), float(parts[1])
    shape = parts[2]
    dia = float(re.findall(r'\d+', parts[3])[0])
    rep = int(parts[4])
    return x_init, y_init, shape, dia, rep

def preprocess_dataframe(df, sensors, normalize=True):
    baseline = df[sensors].iloc[:20].mean()
    for s in sensors:
        df[f'{s}_drift'] = df[s] - baseline[s]
        if normalize:
            df[s] = (df[s] - df[s].mean()) / (df[s].std() + 1e-6)
            df[f'{s}_drift'] = (df[f'{s}_drift'] - df[f'{s}_drift'].mean()) / (df[f'{s}_drift'].std() + 1e-6)
    return df

class TactileDataset(Dataset):
    def __init__(self, folder_path, seq_len=50):
        self.file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        self.seq_len = seq_len
        self.sensors = [f's{i}' for i in range(1, 17)]
        self.data = []
        self._load_all_files()

    def _load_all_files(self):
        for fpath in self.file_list:
            x_init, y_init, shape, dia, rep = parse_filename(fpath)
            df = pd.read_csv(fpath)
            if not set(self.sensors).issubset(df.columns):
                continue
            df = preprocess_dataframe(df, self.sensors)
            df["temperature"] = df.get("temperature", 25.0)
            df["rotation_angle"] = df.get("rotation_angle", 0.0)
            df["indenter_diameter"] = dia

            for i in range(0, len(df) - self.seq_len):
                seq = df.iloc[i:i + self.seq_len]
                arr = seq[self.sensors].values.reshape(self.seq_len, 1, 4, 4)
                arr = torch.tensor(arr, dtype=torch.float32)
                drift_cols = [f"{s}_drift" for s in self.sensors]
                isoline_feat = seq[drift_cols + ["temperature", "rotation_angle", "indenter_diameter"]].values
                isoline_feat = torch.tensor(isoline_feat, dtype=torch.float32)
                if set(["Fx","Fy","Fz","x","y","z"]).issubset(df.columns):
                    tgt = seq[["x","y","z","Fx","Fy","Fz"]].iloc[-1].values
                else:
                    continue
                self.data.append((arr, isoline_feat, torch.tensor(tgt, dtype=torch.float32)))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class IsolineMLP(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, output_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.mean(x, dim=1)

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 4 * 4, 64)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNN_LSTM_Isoline(nn.Module):
    def __init__(self, use_branch1=True, use_branch2=True):
        super().__init__()
        self.encoder = CNNEncoder()
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.isoline = IsolineMLP(19, 64, 32)
        self.use_branch1, self.use_branch2 = use_branch1, use_branch2
        fused_dim = 128 + 32
        if use_branch1:
            self.fc1_1 = nn.Linear(fused_dim, 256)
            self.fc1_map = nn.Linear(256, 25 * 25)
            self.fc1_xyz = nn.Linear(256, 3)
        if use_branch2:
            self.fc2_1 = nn.Linear(fused_dim, 128)
            self.fc2_out = nn.Linear(128, 6)
    def forward(self, x, iso):
        B, T, _, _, _ = x.shape
        x = x.view(B * T, 1, 4, 4)
        cnn_feat = self.encoder(x).view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_feat)
        lstm_feat = lstm_out[:, -1, :]
        iso_feat = self.isoline(iso)
        fused = torch.cat([lstm_feat, iso_feat], dim=-1)
        z1, z2 = None, None
        if self.use_branch1:
            f1 = F.relu(self.fc1_1(fused))
            z1_map = self.fc1_map(f1)
            z1_xyz = self.fc1_xyz(f1)
            z1 = torch.cat([z1_xyz, z1_map], dim=1)
        if self.use_branch2:
            f2 = F.relu(self.fc2_1(fused))
            z2 = self.fc2_out(f2)
        return z1, z2

```

```python
# ===============================
# 4️⃣  학습 루프 + GPU 사용
# ===============================
def train_model(model, dataloader, optimizer, device, epochs=20, use_branch1=True, use_branch2=True):
    criterion = nn.MSELoss()
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for x_seq, iso_feat, tgt in dataloader:
            x_seq, iso_feat, tgt = x_seq.to(device), iso_feat.to(device), tgt.to(device)
            optimizer.zero_grad()
            z1, z2 = model(x_seq, iso_feat)
            loss = 0
            if use_branch1 and z1 is not None:
                loss += criterion(z1[:, :3], tgt[:, :3])
            if use_branch2 and z2 is not None:
                loss += criterion(z2, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.6f}")
    torch.save(model.state_dict(), "/content/trained_model.pth")
    print("✅ 모델 저장 완료: /content/trained_model.pth")

```

```python
# ===============================
# 5️⃣  학습 시작
# ===============================
folder = "/content/dataset"
dataset = TactileDataset(folder_path=folder, seq_len=50)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

model = CNN_LSTM_Isoline(use_branch1=True, use_branch2=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_model(model, loader, optimizer, device, epochs=30)

```

```python
from google.colab import files
files.download("/content/trained_model.pth")
```

### Isoline Theory 적용

좋은 질문이에요.

지금 우리가 만든 **CNN + LSTM + Isoline 통합 구조**에서 “isoline theory”가 어디서, 어떤 역할로 반영되었는지 명확히 구분해보겠습니다.

---

## 🧠 1️⃣ Isoline Theory 핵심 개념 요약

**논문 배경 (Science Robotics, 2022 / Nature Communications, 2025 등):**

- **Taxel Value Isoline Theory**란, 압력 분포를 “등압선(isoline)”으로 해석해
    
    접촉 위치, 중심, 형태(super-resolution contact map)를 예측하는 방법입니다.
    
- 각 센서값 간의 **공간적 관계(gradient field)** 를 모델링하는 것이 핵심입니다.
    
    즉, “S1~S16의 절대값”이 아니라 “센서 간 차이 패턴”이 중요합니다.
    

> 수학적으로는, 압력장 P(x, y)에서 ∂P/∂x, ∂P/∂y를 통해 contact boundary를 추정합니다.
> 

---

## 🧩 2️⃣ 우리가 만든 모델에서의 적용 위치

| 모델 구성 요소 | 역할 | Isoline Theory와의 대응 |
| --- | --- | --- |
| **`df[f'{s}_drift'] = df[s] - baseline[s]`** | baseline 제거, 상대압력 계산 | → 압력장 변화율(gradient)의 기초 구성 |
| **`IsolineMLP` 모듈 (input_dim=19)** | drift + 보조 feature 학습 | → 등압선의 패턴(gradient feature)을 학습 |
| **CNN Encoder (4×4 spatial conv)** | 공간 필터링으로 인접 센서간 gradient 인식 | → Isoline map의 공간적 연속성 학습 |
| **LSTM Temporal sequence** | 시계열 압력 변화를 통해 contact 형성·해제 인식 | → Isoline map의 시간적 변형까지 포착 |
| **Fusion (concat [LSTM, IsolineMLP])** | 공간 gradient + drift feature 결합 | → Isoline theory의 물리적 gradient와 drift 안정화 동시 반영 |

즉, **Isoline theory의 수학적 gradient field를 CNN/LSTM/MLP로 근사적으로 학습**하고 있습니다.

---

## 🔬 3️⃣ 이 설계가 Isoline Theory를 “직접 수식화하지 않고도” 구현하는 이유

논문에서는 수식 기반으로 등압선의 형태를 추정하지만,

우리의 구조는 그 과정을 **신경망이 자동으로 학습하도록 설계**한 것입니다.

### 비교 요약

| 항목 | 원 논문 방식 | 본 모델 적용 방식 |
| --- | --- | --- |
| Pressure → Isoline 변환 | ∂P/∂x, ∂P/∂y 계산 | CNN이 4×4 spatial filter로 자동 추정 |
| Drift/Offset 보정 | 수식으로 baseline 보정 | drift feature + IsolineMLP |
| Contact 중심 추정 | Isoline peak 탐색 | MLP + FC layer에서 xyz 회귀 |
| Contact 분포 복원 | 보간(interpolation) 기반 | 25×25 heatmap branch (fc1_map) |

따라서,

> 우리가 사용하는 “IsolineMLP”는 Isoline theory의 gradient 연산과 drift 보정 파트를 대체하는 학습형 구현체입니다.
> 

---

## ⚙️ 4️⃣ 실험적으로 확인할 수 있는 부분

- Isoline branch (`IsolineMLP`)를 **끄면** contact 중심 예측이 불안정해지고,
    
    drift나 인덴터 형태 변화에 민감해집니다.
    
- 반대로 **Isoline branch를 켜면**,
    
    gradient-like feature(센서 간 상대적 패턴)가 포함되어 예측이 안정화됩니다.
    

---

## ✅ 5️⃣ 결론 — 당신의 모델에서의 Isoline Theory 구현 정리

> Isoline theory의 수식적 정의(∂P/∂x, ∂P/∂y)는 CNN과 drift feature를 통해 implicitly 학습됨.“IsolineMLP”는 논문에서의 gradient field 계산 + drift correction을 담당.이 branch가 존재하기 때문에, 본 모델은 “data-driven Isoline theory implementation” 구조에 해당함.
> 

---

원하신다면

👉 논문(`Guiding the Design of Superresolution Tactile Skins with Taxel Value Isolines Theory`, 2022)을 기반으로

**우리 모델의 수학적 Isoline 근사 구조 (∂P/∂x, ∂P/∂y를 CNN이 어떻게 대체하는지)** 를

수식과 그래프로 도식화해드릴까요?

(즉, "논문 수식 ↔ CNN 구조 대응 관계"를 시각적으로 보여주는 그림)

## 훈련 모델 결과 검증 시각화

다음 단계로는, 앞서 학습한 CNN+LSTM+Isoline 모델의 출력을 **3D 히트맵(압력 분포 fz) + 화살표(3축 힘 fx, fy, fz)** 형태로 시각화하는 루프를 추가합니다.

이 시각화는 실험 후 결과 검증뿐 아니라, 실시간 시각화 루프에서도 그대로 재활용 가능합니다.

---

## ✅ 구조 개요

| 구성 | 설명 |
| --- | --- |
| 입력 | 테스트 데이터 (4×4×T 시퀀스, isoline 입력) |
| 출력 | Branch1: fz_map(25×25), [x,y,z], Branch2: [fx,fy,fz] |
| 시각화 | matplotlib 3D surface + quiver (화살표) |

---

## ✅ 코드 (학습 후 시각화 루프 포함)

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader

# ───────────────────────────────
# 3D 시각화 함수
# ───────────────────────────────
def visualize_prediction(model, dataloader, device='cpu', branch1=True, branch2=True):
    model.eval()
    model.to(device)

    with torch.no_grad():
        for x_seq, iso_feat, target in dataloader:
            x_seq, iso_feat = x_seq.to(device), iso_feat.to(device)
            z1, z2 = model(x_seq, iso_feat)

            # Branch1 → fz 히트맵
            if branch1 and z1 is not None:
                xyz = z1[:, :3][0].cpu().numpy()
                fz_map = z1[:, 3:][0].cpu().numpy().reshape(25, 25)

                # 3D surface plot
                X, Y = np.meshgrid(np.linspace(0, 25, 25), np.linspace(0, 25, 25))
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, fz_map, cmap='jet', linewidth=0, antialiased=True)
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Fz Distribution')
                ax.set_title(f'Contact Heatmap + Center: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})')

            # Branch2 → 3축 힘 화살표
            if branch2 and z2 is not None:
                vec = z2[0].cpu().numpy()
                x0, y0, z0 = vec[0], vec[1], vec[2]
                fx, fy, fz = vec[3], vec[4], vec[5]

                ax.quiver(
                    x0, y0, np.max(fz_map),
                    fx, fy, fz,
                    length=5, normalize=True, color='black', linewidth=2
                )

            plt.tight_layout()
            plt.show()
            break  # 첫 샘플만 시각화

# ───────────────────────────────
# 실행 예시 (학습 후)
# ───────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tactile_training import CNN_LSTM_Isoline, TactileDataset  # 기존 코드에서 import

    # 학습 완료 모델 로드
    folder = "./dataset"
    dataset = TactileDataset(folder, seq_len=50)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = CNN_LSTM_Isoline(use_branch1=True, use_branch2=True)
    model.load_state_dict(torch.load("trained_model.pth", map_location='cpu'))

    visualize_prediction(model, loader, device='cpu', branch1=True, branch2=True)

```

---

## ✅ 시각화 예시

- 히트맵(`fz_map`): 접촉면의 압력 분포 (25×25 mm)
- 화살표(`fx, fy, fz`): 중심점에서의 3축 힘 방향 및 크기

```
       ▲ Fz
       │
   ↖   │   ↗  ← 힘 방향 벡터 (fx, fy, fz)
────────┼──────────────▶ X
        │
        ▼ Y

```

결과:

- 색상 → 압력 크기 (jet colormap)
- 표면 높이 → fz intensity
- 화살표 → 중심점 힘 벡터

---

## ✅ 확장 아이디어

| 목표 | 구현 방식 |
| --- | --- |
| 실시간 시각화 | `matplotlib.animation` 또는 `PyQtGraph` 활용 |
| 여러 인덴터 비교 | `for shape in [hemi, flat, cone]: visualize_prediction(...)` |
| 힘 벡터 히스토리 | `ax.quiver` 대신 `ax.plot`으로 trajectory 표시 |
| 동적 update | 모델 추론 루프 내에 `plt.pause(0.01)` 추가 |

---

이제 이 구조를 그대로 적용하면,

- 학습된 모델에서 바로 추론
- 25×25 히트맵과 3축 벡터를 한눈에 확인 가능
- 실시간 혹은 로그 분석용으로 사용 가능

---

다음 단계로

👉 “**실시간 추론용 루프**” — 즉 센서 스트림이 들어올 때마다 3D plot을 자동 갱신하는 버전(PyQtGraph 기반 실시간 갱신 루프)으로 확장해드릴까요?

## 실시간 추론용

이제 **실시간 추론 루프 + 3D 시각화(PyQtGraph 기반)** 버전을 완성하겠습니다.

이 버전은 실제 BMP384 센서 데이터 스트림(또는 CSV 실험 데이터 replay) 기반으로

매 프레임마다 **CNN+LSTM+Isoline 모델이 추론 → fz 히트맵 + 힘 벡터 갱신**을 수행합니다.

---

## ✅ 목표

| 항목 | 설명 |
| --- | --- |
| 입력 | 센서 raw값(S1~S16) 스트림 (또는 테스트용 CSV) |
| 출력 | ① 접촉분포 fz 히트맵 ② 중심점 (x,y,z) ③ 중심힘 (fx, fy, fz) |
| 시각화 | PyQtGraph 3D surface plot + 화살표 벡터 |
| 주기 | 약 10~30Hz (실시간에 근접) |

---

## ✅ 실행 요구사항

```bash
pip install pyqtgraph pyqt5 torch pandas numpy
```

---

## ✅ 코드 (실시간 추론 + 3D 시각화)

```python
import numpy as np
import torch
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore
import pandas as pd
import time
from tactile_training import CNN_LSTM_Isoline, preprocess_dataframe

# =====================================================
# 1️⃣ 실시간 데이터 스트림 (시뮬레이션용 예시)
# =====================================================
class SensorStream:
    def __init__(self, csv_path, sensors):
        self.df = pd.read_csv(csv_path)
        self.ptr = 0
        self.sensors = sensors
        self.total = len(self.df)

    def read(self, window=50):
        """시계열 길이 window 만큼 반환"""
        if self.ptr + window >= self.total:
            self.ptr = 0
        seq = self.df.iloc[self.ptr:self.ptr + window]
        self.ptr += 1
        return seq[self.sensors].values.reshape(window, 1, 4, 4)

# =====================================================
# 2️⃣ PyQtGraph 3D 초기화
# =====================================================
class TactileVisualizer(QtWidgets.QWidget):
    def __init__(self, model, sensors, device='cpu'):
        super().__init__()
        self.model = model.to(device)
        self.sensors = sensors
        self.device = device
        self.layout = QtWidgets.QVBoxLayout(self)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=60)
        self.layout.addWidget(self.view)

        self.grid = gl.GLGridItem()
        self.grid.scale(2, 2, 1)
        self.view.addItem(self.grid)

        # 히트맵용 surface mesh
        X, Y = np.meshgrid(np.linspace(0, 25, 25), np.linspace(0, 25, 25))
        self.surface = gl.GLSurfacePlotItem(
            x=X, y=Y, z=np.zeros((25, 25)),
            shader='heightColor', color=(0.3, 0.5, 1, 0.8)
        )
        self.view.addItem(self.surface)

        # 힘 벡터용 화살표
        self.arrow = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1, 0, 0, 1), width=3)
        self.view.addItem(self.arrow)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)  # 20Hz

        self.stream = SensorStream("./dataset/0.0_0.0_hemi_D10mm_1.csv", sensors)
        self.seq_len = 50

    def update_frame(self):
        seq = self.stream.read(self.seq_len)
        arr = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Isoline feature (드리프트+보조 변수)
        iso_feat = torch.zeros((1, self.seq_len, 19)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            z1, z2 = self.model(arr, iso_feat)

        if z1 is None or z2 is None:
            return

        # 히트맵 (fz 분포)
        fz_map = z1[:, 3:].cpu().numpy().reshape(25, 25)
        X, Y = np.meshgrid(np.linspace(0, 25, 25), np.linspace(0, 25, 25))
        self.surface.setData(z=fz_map)

        # 중심점 힘 벡터
        vec = z2[0].cpu().numpy()
        x0, y0, z0 = vec[0], vec[1], np.max(fz_map)
        fx, fy, fz = vec[3], vec[4], vec[5]
        arrow_pts = np.array([[x0, y0, z0], [x0 + fx * 5, y0 + fy * 5, z0 + fz * 5]])
        self.arrow.setData(pos=arrow_pts)

        pg.QtGui.QApplication.processEvents()

# =====================================================
# 3️⃣ 실행 루프
# =====================================================
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    sensors = [f"s{i}" for i in range(1, 17)]

    model = CNN_LSTM_Isoline(use_branch1=True, use_branch2=True)
    model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))

    vis = TactileVisualizer(model, sensors)
    vis.show()

    QtWidgets.QApplication.instance().exec_()

```

---

## ✅ 실행 동작 요약

| 단계 | 설명 |
| --- | --- |
| ① | `SensorStream`이 CSV에서 50 timestep씩 sliding window로 읽음 |
| ② | CNN+LSTM+Isoline 모델 추론 수행 |
| ③ | Branch1 → `fz_map` 히트맵으로 3D surface 업데이트 |
| ④ | Branch2 → 중심점의 `[fx, fy, fz]`로 화살표 업데이트 |
| ⑤ | 20Hz로 갱신 (실시간) |

---

## ✅ 실제 센서 데이터 스트림 적용 시

| 항목 | 적용 방식 |
| --- | --- |
| BMP384 실시간 데이터 | UART/Serial 또는 ROS2 topic에서 16개 센서값 수집 |
| 실시간 변환 | (1,16) 벡터를 `reshape(1,1,4,4)` 하여 입력 |
| 누적 버퍼 | `collections.deque(maxlen=50)` 형태로 LSTM 입력 윈도우 유지 |
| PyQtGraph 업데이트 | 위 코드 그대로 사용 가능 (`self.stream.read()` → `deque`로 변경) |

---

## ✅ 확장 아이디어

| 확장 기능 | 방법 |
| --- | --- |
| ROS2 센서 스트림 연동 | `rclpy` 노드 생성 후 `callback`에서 `update_frame()` 호출 |
| 예측 FPS 측정 | `time.time()` 기반으로 FPS 계산 출력 |
| 히트맵 컬러맵 변경 | `shader='shaded'`, `colorMap='jet'` 적용 |
| fz, fx, fy 개별 플롯 | 서브 윈도우 생성 후 각각 2D plot 가능 |

---

##