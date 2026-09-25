"""
dataset_sr.py

features.csv 기반 SR 학습용 Dataset.

SRDataset     : point-wise (MLP, CNN용)
SRSeqDataset  : depth-axis sequence (CNN-LSTM용)

반환 dict 공통:
  s16     (16,)  float32  - 16채널 s_norm 전체
  diam    (1,)   float32  - diameter_norm
  target  (4,)   float32  - [x_mm, y_mm, z_contact_mm, fz]
  x_mm, y_mm     float    - 평가용 그리드 좌표

SRSeqDataset 추가:
  s16    (T, 16), diam (T, 1), target (T, 4)  - 시퀀스
  mask_len  int  - 실제 유효 프레임 수
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── 상수 ────────────────────────────────────────────────────────────────────
S_NORM_COLS  = [f"s_norm_{i}" for i in range(1, 17)]   # s_norm_1 .. s_norm_16
TARGET_COLS  = ["x_mm", "y_mm", "z_mm", "fz"]


def _prepare_depth_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "z_contact_mm" in out.columns:
        out["z_mm"] = out["z_contact_mm"]
    elif "z_depth_mm" in out.columns:
        out["z_mm"] = out["z_depth_mm"]
    else:
        raise KeyError("Expected one of z_contact_mm or z_depth_mm in features.csv")
    return out


# ── 데이터 로드 + 분할 ──────────────────────────────────────────────────────
def load_split(
    features_csv: Path,
    split: str = "train",       # "train" | "val" | "test" | "all"
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    phase_filter: int = 0,      # 0=loading, 1=unloading, None=전체
) -> pd.DataFrame:
    """CSV → phase 필터 → trial-based split → DataFrame 반환"""
    df = _prepare_depth_columns(pd.read_csv(features_csv))

    if phase_filter is not None:
        df = df[df["phase"] == phase_filter].copy()

    trials = sorted(df["trial_id"].unique())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(trials))
    trials = [trials[i] for i in order]

    n = len(trials)

    if n == 1:
        # trial 1개: 같은 trial을 모든 split에 사용 (소규모 데이터셋 fallback)
        sel = trials
    else:
        # trial 2개 이상: trial-based split (test는 n>=3일 때만 독립 분리)
        n_test  = max(1, round(n * test_ratio)) if n >= 3 else 0
        n_val   = max(1, round(n * val_ratio))
        n_train = max(1, n - n_val - n_test)

        train_t = trials[:n_train]
        val_t   = trials[n_train : n_train + n_val]
        test_t  = trials[n_train + n_val :] or val_t  # 없으면 val 재사용

        split_map = {
            "train": train_t,
            "val":   val_t,
            "test":  test_t,
            "all":   trials,
        }
        sel = split_map.get(split, trials)
    return df[df["trial_id"].isin(sel)].reset_index(drop=True)


# ── Point-wise Dataset (MLP, CNN) ────────────────────────────────────────────
class SRDataset(Dataset):
    """
    features.csv 한 행 = 하나의 샘플.
    MLP / CNN 학습에 사용.
    """

    def __init__(self, df: pd.DataFrame):
        self.s16    = df[S_NORM_COLS].values.astype(np.float32)        # (N, 16)
        self.diam   = df["diameter_norm"].values.astype(np.float32).reshape(-1, 1)  # (N,1)
        self.target = df[TARGET_COLS].values.astype(np.float32)        # (N, 4)
        self.x_mm   = df["x_mm"].values.astype(np.float32)
        self.y_mm   = df["y_mm"].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.s16)

    def __getitem__(self, idx: int) -> dict:
        return {
            "s16":    torch.from_numpy(self.s16[idx]),
            "diam":   torch.from_numpy(self.diam[idx]),
            "target": torch.from_numpy(self.target[idx]),
            "x_mm":   float(self.x_mm[idx]),
            "y_mm":   float(self.y_mm[idx]),
        }

    def to_gpu(self, device: torch.device) -> dict:
        """GPU 캐시용 전체 텐서 반환"""
        return {
            "s16":    torch.from_numpy(self.s16).to(device),
            "diam":   torch.from_numpy(self.diam).to(device),
            "target": torch.from_numpy(self.target).to(device),
            "x_mm":   torch.from_numpy(self.x_mm).to(device),
            "y_mm":   torch.from_numpy(self.y_mm).to(device),
        }


# ── Sequence Dataset (CNN-LSTM) ───────────────────────────────────────────────
class SRSeqDataset(Dataset):
    """
    (trial_id, x_mm, y_mm) 그룹별 depth-axis 시퀀스.
    z_contact_mm 기준 오름차순으로 정렬 → [0:seq_len] 사용, 부족하면 zero-pad.
    CNN-LSTM 학습에 사용.
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 32):
        self.seq_len = seq_len
        self.seqs: list = []

        for _, grp in df.groupby(["trial_id", "x_mm", "y_mm"], sort=False):
            grp = grp.sort_values("z_mm").reset_index(drop=True)
            s16   = grp[S_NORM_COLS].values.astype(np.float32)         # (T, 16)
            diam  = grp["diameter_norm"].values.astype(np.float32)[:, None]  # (T,1)
            tgt   = grp[TARGET_COLS].values.astype(np.float32)         # (T, 4)
            x     = float(grp["x_mm"].iloc[0])
            y     = float(grp["y_mm"].iloc[0])
            self.seqs.append((s16, diam, tgt, x, y))

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> dict:
        s16, diam, tgt, x, y = self.seqs[idx]
        T = len(s16)
        L = self.seq_len

        if T >= L:
            s16  = s16[:L]
            diam = diam[:L]
            tgt  = tgt[:L]
            mask = L
        else:
            pad = L - T
            s16  = np.pad(s16,  ((0, pad), (0, 0)))
            diam = np.pad(diam, ((0, pad), (0, 0)))
            tgt  = np.pad(tgt,  ((0, pad), (0, 0)))
            mask = T

        return {
            "s16":      torch.from_numpy(s16.astype(np.float32)),    # (L, 16)
            "diam":     torch.from_numpy(diam.astype(np.float32)),   # (L, 1)
            "target":   torch.from_numpy(tgt.astype(np.float32)),    # (L, 4)
            "mask_len": mask,
            "x_mm":     x,
            "y_mm":     y,
        }
