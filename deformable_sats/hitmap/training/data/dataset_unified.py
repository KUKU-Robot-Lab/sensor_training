
import os
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

def parse_filename(fname):
    fname = os.path.basename(fname).replace('.csv', '')
    parts = fname.split('_')
    try:
        x_init, y_init = float(parts[0]), float(parts[1])
        shape = parts[2]
        dia_match = re.findall(r'\d+', parts[3])
        dia = float(dia_match[0]) if dia_match else 10.0
        rep = int(parts[4]) if len(parts) > 4 else 1
    except (IndexError, ValueError):
        return 0.0, 0.0, "unknown", 10.0, 1
    return x_init, y_init, shape, dia, rep

def preprocess_dataframe(df, sensors, normalize=True):
    baseline = df[sensors].iloc[:min(20, len(df))].mean()
    for s in sensors:
        drift_col = f'{s}_drift'
        df[drift_col] = df[s] - baseline[s]
        if normalize:
            std = df[s].std()
            df[s] = (df[s] - df[s].mean()) / (std + 1e-6)
            d_std = df[drift_col].std()
            df[drift_col] = (df[drift_col] - df[drift_col].mean()) / (d_std + 1e-6)
    return df

class UnifiedTactileDataset(Dataset):
    """
    메모리 절약형 데이터셋.
    - 전체 시퀀스를 미리 메모리에 적재하지 않고, (파일 경로, 시작 인덱스)만 인덱스 테이블에 저장한다.
    - 실제 텐서 생성은 __getitem__ 호출 시 해당 파일을 읽거나(캐시 미스) 캐시된 DataFrame에서 슬라이스한다.
    """

    def __init__(self, folder_path, seq_len=50, stride=5, augment=False, file_glob="**/*_grid.csv"):
        self.folder_path = folder_path
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment
        self.file_glob = file_glob

        self.sensors = [f's{i}' for i in range(1, 17)]
        self.drift_cols = [f"{s}_drift" for s in self.sensors]

        # 샘플 인덱스 테이블: (fpath, start_idx, dia)
        self.samples: list[tuple[str, int, float]] = []
        # 파일별 전처리 캐시 (최근 접근 위주로 소량 유지 권장)
        self._cache: dict[str, pd.DataFrame] = {}

        if os.path.exists(folder_path):
            self._index_all_files()

    # ── 내부 유틸 ────────────────────────────────────────────────────────
    def _get_df(self, fpath: str, dia: float) -> pd.DataFrame:
        if fpath in self._cache:
            return self._cache[fpath]

        df = pd.read_csv(fpath)
        rename_map = {}
        if "z_depth_mm" in df.columns and "z_mm" not in df.columns:
            rename_map["z_depth_mm"] = "z_mm"
        if "fx" in df.columns and "Fx" not in df.columns:
            rename_map["fx"] = "Fx"
        if "fy" in df.columns and "Fy" not in df.columns:
            rename_map["fy"] = "Fy"
        if "fz" in df.columns and "Fz" not in df.columns:
            rename_map["fz"] = "Fz"
        if rename_map:
            df = df.rename(columns=rename_map)

        if not set(self.sensors).issubset(df.columns):
            raise ValueError(f"[{fpath}] missing sensor columns")

        df = preprocess_dataframe(df, self.sensors)
        # 필요한 컬럼만 보관해 메모리 절약 (float32 변환)
        keep_cols = self.sensors + self.drift_cols + [c for c in ["x_mm", "y_mm", "z_mm", "Fx", "Fy", "Fz"] if c in df.columns]
        df = df[keep_cols].astype(np.float32)
        if "diameter_mm" in df.columns:
            dia_val = float(df["diameter_mm"].iloc[0])
        else:
            dia_val = float(dia)
        df["indenter_diameter"] = np.float32(dia_val)

        # 간단한 캐시 정책: 최근 2개 파일만 유지
        self._cache[fpath] = df
        if len(self._cache) > 2:
            # 가장 오래된 항목 제거
            oldest = next(iter(self._cache))
            if oldest != fpath:
                self._cache.pop(oldest, None)
        return df

    def _index_all_files(self):
        import glob
        patterns = [self.file_glob]
        if self.file_glob == "**/*_grid.csv":
            patterns.append("**/*_merged.csv")
        file_list = []
        for pat in patterns:
            file_list.extend(glob.glob(os.path.join(self.folder_path, pat), recursive=True))
        file_list = sorted(set(file_list))
        if not file_list:
            print(f"Warning: No CSV files found in {self.folder_path}")
            return

        for fpath in file_list:
            try:
                _, _, _, dia, _ = parse_filename(fpath)
                df = pd.read_csv(fpath, nrows=5)
                if not set(self.sensors).issubset(df.columns):
                    continue

                n_rows = len(pd.read_csv(fpath, usecols=[self.sensors[0]]))  # 빠른 row count
                max_start = n_rows - self.seq_len
                if max_start <= 0:
                    continue
                for start in range(0, max_start, self.stride):
                    self.samples.append((fpath, start, dia))
            except Exception as e:
                print(f"Error indexing {fpath}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, start, dia = self.samples[idx]
        df = self._get_df(fpath, dia)
        seq = df.iloc[start : start + self.seq_len]

        grid_input = seq[self.sensors].values.reshape(self.seq_len, 1, 4, 4)
        drift_vals = seq[self.drift_cols].values
        dia_col = seq[["indenter_diameter"]].values.astype(np.float32)
        iso_feat = np.concatenate([drift_vals, dia_col], axis=1)

        target_cols = ["x_mm", "y_mm", "z_mm", "Fx", "Fy", "Fz"]
        available_targets = [c for c in target_cols if c in seq.columns]
        if len(available_targets) >= 3:
            tgt_vals = seq[available_targets].iloc[-1].values.astype(np.float32)
        else:
            tgt_vals = np.zeros(6, dtype=np.float32)
        full_tgt = np.zeros(6, dtype=np.float32)
        full_tgt[: len(tgt_vals)] = tgt_vals

        grid = torch.tensor(grid_input, dtype=torch.float32)
        iso = torch.tensor(iso_feat, dtype=torch.float32)
        tgt = torch.tensor(full_tgt, dtype=torch.float32)

        if self.augment:
            if torch.rand(1) > 0.5:
                grid = torch.flip(grid, dims=[-1]); tgt[0] = -tgt[0]; tgt[3] = -tgt[3]
            if torch.rand(1) > 0.5:
                grid = torch.flip(grid, dims=[-2]); tgt[1] = -tgt[1]; tgt[4] = -tgt[4]
            grid += torch.randn_like(grid) * 0.01

        return grid, iso, tgt
