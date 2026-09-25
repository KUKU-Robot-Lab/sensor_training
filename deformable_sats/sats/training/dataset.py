"""
sats/training/dataset.py

SATSSequenceDataset
-------------------
각 압입 사이클 (trial 내 한 grid 위치의 time-series)을 하나의 학습 샘플로 구성한다.

데이터 흐름
-----------
raw merged BIN/CSV  →  on-grid 필터  →  s_norm 계산  →  (x,y)별 그룹화
                                                         │
GT targets.npy (mmap) ──────────────────────────────────┘
                                 __getitem__: [sensor_seq, gt_seq]

반환 형식 (__getitem__)
-----------------------
sensor_seq : Tensor[T, 16]       s_norm 시계열, float32
gt_seq     : Tensor[T, 41, 41]   GT 압력맵 시계열, float32
length     : int                 실제 시퀀스 길이 T

반환 형식 (collate_fn)
-----------------------
sensor_batch : Tensor[B, T_max, 16]
gt_batch     : Tensor[B, T_max, 41, 41]
lengths      : Tensor[B]  int64

온-그리드 필터 (GT generation과 동일 기준)
------------------------------------------
그리드: np.linspace(-10.0, 10.0, 41), step=0.5mm
스냅:   xs = lo + round((x - lo) / step) * step   (lo=-10.0 기준 offset 사용)
조건:   |x - xs| < tol  AND  |y - ys| < tol
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from sats.preprocessing.merged_bin import merged_bin_to_frame, read_merged_bin_header
from .config import SATSConfig, filter_trial_ids

log = logging.getLogger(__name__)

# ── 센서 컬럼 상수 ────────────────────────────────────────────────────────────
_SENSOR_COLS = [f"s{i}" for i in range(1, 17)]          # merged CSV 컬럼명
_SKIN_MEAN_KEYS = [f"Skin{i}_mean" for i in range(1, 17)]  # baseline JSON 키


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _baseline_array_from_dict(data: dict) -> np.ndarray:
    return np.array([data[k] for k in _SKIN_MEAN_KEYS], dtype=np.float64)


def _load_baseline(path: Path, merged_bin: Path | None = None) -> np.ndarray | None:
    """Load baseline from JSON, falling back to merged BIN metadata."""

    if path.exists():
        with open(path) as f:
            return _baseline_array_from_dict(json.load(f))
    if merged_bin is not None and merged_bin.exists():
        header, _ = read_merged_bin_header(merged_bin)
        baseline = header.get("metadata", {}).get("baseline")
        if isinstance(baseline, dict) and all(k in baseline for k in _SKIN_MEAN_KEYS):
            return _baseline_array_from_dict(baseline)
    return None


def _resolve_merged_input(paths: dict, cfg: SATSConfig) -> Path | None:
    merged_bin = paths.get("merged_bin")
    merged_csv = paths.get("merged_csv")
    if cfg.prefer_merged_bin and merged_bin is not None and merged_bin.exists():
        return merged_bin
    if merged_csv is not None and merged_csv.exists():
        return merged_csv
    if merged_bin is not None and merged_bin.exists():
        return merged_bin
    return None


def _load_merged_training_frame(path: Path, cfg: SATSConfig) -> pd.DataFrame | None:
    required_cols = _SENSOR_COLS + ["x_mm", "y_mm", "Fz"]
    optional_cols = ["u_mm"]
    if path.suffix == ".bin":
        columns = required_cols + optional_cols
        try:
            return merged_bin_to_frame(
                path,
                columns=columns,
                u_zero_only=cfg.use_u_zero_only,
                u_zero_tol_mm=cfg.u_zero_tol_mm,
            )
        except ValueError as exc:
            log.warning("[%s] merged BIN 로드 실패: %s", path.name, exc)
            return None

    header_cols = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in required_cols if c not in header_cols]
    if missing:
        log.warning("[%s] merged CSV 필수 컬럼 없음: %s", path.name, missing)
        return None
    usecols = required_cols + [c for c in optional_cols if c in header_cols]
    df = pd.read_csv(path, usecols=usecols, dtype=np.float64)
    if "u_mm" not in df.columns:
        df["u_mm"] = 0.0
    return df


def _on_grid_mask(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    step: float = 0.5,
    tol: float  = 0.05,
    lo: float   = -10.0,
    hi: float   = 10.0,
) -> np.ndarray:
    """
    on-grid bool mask.

    올바른 스냅 공식: lo를 기준으로 offset을 계산하여 반올림.
    (단순 round(x/step)*step 은 그리드 원점 오프셋을 무시하므로 잘못된 결과를 냄)

    예: lo=-10.0, step=0.5 → grid = [-10.0, -9.5, ..., 10.0]
    x = -10.0: (x-lo)/step = 0.0 → snap = -10.0
    x = -9.75: half-grid 위치이므로 tol=0.05에서는 제외
    """
    n_steps = round((hi - lo) / step)
    kx = np.clip(np.round((x_arr - lo) / step), 0, n_steps)
    ky = np.clip(np.round((y_arr - lo) / step), 0, n_steps)
    xs = lo + kx * step
    ys = lo + ky * step
    return (np.abs(x_arr - xs) < tol) & (np.abs(y_arr - ys) < tol)


def _snap_coords(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    step: float = 0.5,
    lo: float   = -10.0,
    hi: float   = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """on-grid 통과 후 좌표를 그리드 좌표로 스냅 (float 비교 안정성)."""
    n_steps = round((hi - lo) / step)
    xs = lo + np.clip(np.round((x_arr - lo) / step), 0, n_steps) * step
    ys = lo + np.clip(np.round((y_arr - lo) / step), 0, n_steps) * step
    return xs, ys


def _load_trial(
    trial_id: str,
    cfg: SATSConfig,
) -> Optional[Tuple[List[dict], np.ndarray]]:
    """
    한 trial을 처리하여 (시퀀스 목록, GT mmap)을 반환한다.

    Parameters
    ----------
    trial_id : str  예) "ecomesh_d5_z1_test1"
    cfg      : SATSConfig

    Returns
    -------
    (sequences, gt_mmap) 또는 None (파일 오류 시)

    sequences: List[dict], 각 원소 =
        {
          "trial_id"   : str,
          "x_mm"       : float,
          "y_mm"       : float,
          "sensor_seq" : np.ndarray [T, 16] float32,
          "row_indices": np.ndarray [T]     int64,
        }

    gt_mmap: np.ndarray (mmap read-only) shape [N_total, grid_size, grid_size]
    """
    paths = cfg.trial_paths(trial_id)
    merged_input  = _resolve_merged_input(paths, cfg)
    merged_bin    = paths.get("merged_bin")
    baseline_json = paths["baseline_json"]
    gt_npy        = cfg.gt_npy_path(trial_id)

    # 파일 존재 확인
    if merged_input is None:
        log.warning("[%s] merged BIN/CSV 없음: %s / %s", trial_id, paths.get("merged_bin"), paths.get("merged_csv"))
        return None
    for label, p in [("GT npy", gt_npy)]:
        if not p.exists():
            log.warning("[%s] %s 없음: %s", trial_id, label, p)
            return None

    # ── 1. baseline 로드 ──────────────────────────────────────────────────
    baseline = _load_baseline(baseline_json, merged_bin=merged_bin)   # [16] float64
    if baseline is None:
        log.warning("[%s] baseline JSON/metadata 없음: %s", trial_id, baseline_json)
        return None

    # ── 2. merged input 로드 (필요 컬럼 + optional u_mm) ───────────────────
    df = _load_merged_training_frame(merged_input, cfg)
    if df is None:
        return None
    fz_all = df["Fz"].to_numpy()

    x_arr = df["x_mm"].to_numpy()
    y_arr = df["y_mm"].to_numpy()

    # ── 3. on-grid 필터 ────────────────────────────────────────────────────
    mask = _on_grid_mask(
        x_arr, y_arr,
        step=cfg.grid_step_mm,
        tol=cfg.grid_tol_mm,
        lo=cfg.grid_min_mm,
        hi=cfg.grid_max_mm,
    )
    if cfg.use_u_zero_only:
        u_arr = df["u_mm"].to_numpy(dtype=np.float64)
        mask &= np.abs(u_arr) <= cfg.u_zero_tol_mm
    n_grid = int(mask.sum())
    log.info("[%s] on-grid: %d / %d rows", trial_id, n_grid, len(df))

    if n_grid == 0:
        log.warning("[%s] on-grid rows 없음, 건너뜀", trial_id)
        return None

    # ── 4. GT mmap 로드 & 행 수 검증 ─────────────────────────────────────
    gt_mmap = np.load(str(gt_npy), mmap_mode="r")  # [N_total, 41, 41]
    if gt_mmap.shape[0] != n_grid:
        log.warning(
            "[%s] GT 행 수 불일치 (GT=%d, on-grid=%d). trial 건너뜀.",
            trial_id, gt_mmap.shape[0], n_grid,
        )
        return None

    df_grid = df[mask].reset_index(drop=True)

    # ── 5. s_norm 계산 및 100배 스케일 업 ───────────────────────────────
    s_raw  = df_grid[_SENSOR_COLS].to_numpy(dtype=np.float64)   # [N, 16]
    s_norm = (((s_raw - baseline) / baseline) * 100.0).astype(np.float32)  # [N, 16]

    # ── 6. 그리드 좌표로 스냅 → (x,y) 별 그룹화 ─────────────────────────
    xg, yg = _snap_coords(
        df_grid["x_mm"].to_numpy(),
        df_grid["y_mm"].to_numpy(),
        step=cfg.grid_step_mm,
        lo=cfg.grid_min_mm,
        hi=cfg.grid_max_mm,
    )
    # 부동소수점 키: 4자리 반올림으로 안정화
    groups: Dict[Tuple[float, float], List[int]] = defaultdict(list)
    for i in range(n_grid):
        key = (round(float(xg[i]), 4), round(float(yg[i]), 4))
        groups[key].append(i)

    # ── 6b. on-grid 행의 Fz 추출 ──────────────────────────────────────────
    fz_grid = fz_all[mask]  # [n_grid]

    # ── 7. 시퀀스 생성 ─────────────────────────────────────────────────────
    sequences = []
    skipped_short = 0
    skipped_no_contact = 0
    for (x, y), idxs in groups.items():
        T = len(idxs)
        if T < cfg.min_seq_len:
            skipped_short += 1
            continue
        # seq_len 초과 시 앞부분(loading phase 포함) 사용
        if T > cfg.seq_len:
            idxs = idxs[: cfg.seq_len]
            T = cfg.seq_len
        row_idx = np.array(idxs, dtype=np.int64)

        # Fz > 0인 timestep이 하나도 없으면 순수 비접촉 → 학습 제외
        if fz_grid[row_idx].max() <= 0.0:
            skipped_no_contact += 1
            continue

        sequences.append({
            "trial_id":    trial_id,
            "x_mm":        x,
            "y_mm":        y,
            "sensor_seq":  s_norm[row_idx],             # [T, 16] float32 - pre-loaded
            "row_indices": row_idx,                     # [T] int64 - for GT lookup
            "fz_seq":      fz_grid[row_idx].astype(np.float32),  # [T] float32 - Fz 마스킹용
        })

    log.info(
        "[%s] 시퀀스: %d개 (짧음=%d, 비접촉=%d)",
        trial_id, len(sequences), skipped_short, skipped_no_contact,
    )
    return sequences, gt_mmap


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SATSSequenceDataset(Dataset):
    """
    SATS 학습용 시퀀스 데이터셋.

    각 샘플 = 한 grid 위치에서의 압입 사이클 전체 시계열.

    Parameters
    ----------
    trial_ids  : 학습/검증에 사용할 trial_id 목록. _preloaded 사용 시 무시.
    cfg        : SATSConfig
    _preloaded : {"sequences": List[dict], "gt_mmaps": Dict[str, ndarray]}
                 build_dataloaders 의 val_ratio 모드에서 pre-split 데이터를 주입할 때 사용.
    """

    def __init__(
        self,
        trial_ids: List[str],
        cfg: SATSConfig,
        *,
        _preloaded: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self._index: List[dict] = []
        self._gt_mmaps: Dict[str, np.ndarray] = {}

        if _preloaded is not None:
            self._index   = _preloaded["sequences"]
            self._gt_mmaps = _preloaded["gt_mmaps"]
            log.info("Dataset (preloaded): %d 시퀀스", len(self._index))
        else:
            n_failed = 0
            for tid in trial_ids:
                result = _load_trial(tid, cfg)
                if result is None:
                    n_failed += 1
                    continue
                seqs, gt_mmap = result
                self._index.extend(seqs)
                self._gt_mmaps[tid] = gt_mmap

            log.info(
                "Dataset 완료: %d 시퀀스, %d trial 성공, %d 실패",
                len(self._index), len(self._gt_mmaps), n_failed,
            )

        if len(self._index) == 0:
            raise RuntimeError("유효한 시퀀스가 없습니다. 경로와 설정을 확인하세요.")

    # ── Dataset 인터페이스 ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns
        -------
        sensor_seq : Tensor[T, 16]      s_norm 시계열
        gt_seq     : Tensor[T, 41, 41]  GT 압력맵 시계열
        length     : int                실제 시퀀스 길이 T
        """
        item = self._index[idx]
        trial_id    = item["trial_id"]
        row_indices = item["row_indices"]              # [T] int64
        sensor_seq  = item["sensor_seq"]               # [T, 16] float32 (pre-loaded)
        fz_seq      = item["fz_seq"]                   # [T] float32

        # GT: mmap에서 해당 행만 읽어 복사 후 100배 스케일 업
        gt_seq = (self._gt_mmaps[trial_id][row_indices].copy() * 100.0)   # [T, 41, 41]

        # abs 모드 보정: Fz <= 0인 timestep은 비접촉 → GT를 zero로 강제
        gt_seq[fz_seq <= 0.0] = 0.0

        return (
            torch.from_numpy(sensor_seq),           # [T, 16]
            torch.from_numpy(gt_seq),               # [T, 41, 41]
            len(row_indices),                       # T (int)
        )

    # ── 편의 메서드 ───────────────────────────────────────────────────────────

    def trial_ids(self) -> List[str]:
        """로드된 trial_id 목록."""
        return list(self._gt_mmaps.keys())

    def seq_lengths(self) -> List[int]:
        """전체 시퀀스 길이 목록 (통계 확인용)."""
        return [len(item["row_indices"]) for item in self._index]


# ─────────────────────────────────────────────────────────────────────────────
# Window Dataset (논문 방식: loading phase + sliding window)
# ─────────────────────────────────────────────────────────────────────────────

class SATSWindowDataset(Dataset):
    """
    슬라이딩 윈도우 기반 SATS 데이터셋.

    논문 방식: LSTM은 window_size=10 타임스텝 윈도우를 입력받는다.
    - 압입 loading phase(GT 합이 최대가 되는 시점까지)만 사용한다.
    - 각 샘플 = 10 타임스텝 윈도우 + 윈도우 마지막 타임스텝의 GT 맵.

    Parameters
    ----------
    trial_ids  : 학습/검증에 사용할 trial_id 목록. _preloaded 사용 시 무시.
    cfg        : SATSConfig
    _preloaded : {"sequences_by_trial": Dict[str, List[dict]], "gt_mmaps": Dict[str, ndarray]}
                 build_dataloaders 의 val_ratio 모드에서 pre-split 데이터를 주입할 때 사용.
    """

    def __init__(
        self,
        trial_ids: List[str],
        cfg: SATSConfig,
        *,
        _preloaded: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self._index: List[Tuple[str, int, int, int]] = []
        self._sequences: Dict[str, List[dict]] = {}
        self._gt_mmaps: Dict[str, np.ndarray] = {}

        if _preloaded is not None:
            self._gt_mmaps  = _preloaded["gt_mmaps"]
            self._sequences = _preloaded["sequences_by_trial"]
            for tid, seqs in self._sequences.items():
                self._build_windows_for_trial(tid, seqs, self._gt_mmaps[tid])
            log.info("WindowDataset (preloaded): %d 윈도우", len(self._index))
        else:
            n_failed = 0
            for tid in trial_ids:
                result = _load_trial(tid, cfg)
                if result is None:
                    n_failed += 1
                    continue
                seqs, gt_mmap = result
                self._gt_mmaps[tid] = gt_mmap
                self._sequences[tid] = seqs
                self._build_windows_for_trial(tid, seqs, gt_mmap)

            log.info(
                "WindowDataset 완료: %d 윈도우, %d trial 성공, %d 실패",
                len(self._index), len(self._gt_mmaps), n_failed,
            )

        if len(self._index) == 0:
            raise RuntimeError("유효한 윈도우가 없습니다. 경로와 설정을 확인하세요.")

    def _build_windows_for_trial(
        self,
        tid: str,
        seqs: List[dict],
        gt_mmap: np.ndarray,
    ) -> None:
        """한 trial의 시퀀스 목록으로 슬라이딩 윈도우 인덱스를 생성한다."""
        n_windows = 0
        n_short   = 0
        for local_idx, seq in enumerate(seqs):
            row_indices = seq["row_indices"]
            fz_seq      = seq["fz_seq"]

            gt_slice = gt_mmap[row_indices]
            gt_sums  = gt_slice.sum(axis=(1, 2)).copy()
            gt_sums[fz_seq <= 0.0] = -1.0
            peak_t = int(np.argmax(gt_sums))

            loading_len = peak_t + 1
            if loading_len < self.cfg.window_size:
                n_short += 1
                continue

            for t in range(self.cfg.window_size - 1, loading_len):
                self._index.append((tid, local_idx, t, int(row_indices[t])))
            n_windows += loading_len - self.cfg.window_size + 1

        log.info("[%s] 윈도우 %d개 생성 (짧음=%d)", tid, n_windows, n_short)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tid, local_idx, t, gt_row_idx = self._index[idx]
        seq = self._sequences[tid][local_idx]
        sensor_seq = seq["sensor_seq"]   # [T, 16] float32

        w = self.cfg.window_size
        sensor_window = torch.from_numpy(
            sensor_seq[t - w + 1 : t + 1].copy()
        )  # [window_size, 16]

        gt_map = torch.from_numpy(
            (self._gt_mmaps[tid][gt_row_idx].copy() * 100.0).astype(np.float32)
        )  # [41, 41]

        # Fz <= 0이면 비접촉 → GT zero
        if seq["fz_seq"][t] <= 0.0:
            gt_map = torch.zeros_like(gt_map)

        return sensor_window, gt_map


# ─────────────────────────────────────────────────────────────────────────────
# collate_fn
# ─────────────────────────────────────────────────────────────────────────────

def sats_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    가변 길이 시퀀스를 배치로 패딩.

    Parameters
    ----------
    batch : List of (sensor_seq [T_i, 16], gt_seq [T_i, 41, 41], length T_i)

    Returns
    -------
    sensor_batch : Tensor[B, T_max, 16]        zero-padded
    gt_batch     : Tensor[B, T_max, 41, 41]    zero-padded
    lengths      : Tensor[B]  int64            실제 길이
    """
    sensor_seqs, gt_seqs, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.int64)

    # 센서 시퀀스: pad_sequence 사용 ([T, 16] → [B, T_max, 16])
    sensor_batch = pad_sequence(
        sensor_seqs, batch_first=True, padding_value=0.0
    )  # [B, T_max, 16]

    # GT: 수동 패딩 (4D tensor)
    B = len(gt_seqs)
    T_max = int(lengths.max().item())
    G = gt_seqs[0].shape[-1]
    gt_batch = torch.zeros(B, T_max, G, G, dtype=torch.float32)
    for i, (gs, ln) in enumerate(zip(gt_seqs, lengths)):
        gt_batch[i, : int(ln)] = gs

    return sensor_batch, gt_batch, lengths


def window_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SATSWindowDataset 배치 구성.

    모든 윈도우가 동일 길이이므로 패딩 불필요.
    시퀀스 모드와 동일한 3-tuple (sensor, gt, lengths)을 반환하여
    학습 루프 인터페이스를 통일한다.

    Returns
    -------
    sensor_batch : Tensor[B, window_size, 16]
    gt_batch     : Tensor[B, 41, 41]        (3D — 시퀀스 모드의 4D와 구분)
    lengths      : Tensor[B]  (모두 window_size)
    """
    sensor_wins, gt_maps = zip(*batch)
    B = len(sensor_wins)
    window_size = sensor_wins[0].shape[0]
    lengths = torch.full((B,), window_size, dtype=torch.int64)
    return torch.stack(sensor_wins), torch.stack(gt_maps), lengths


# ─────────────────────────────────────────────────────────────────────────────
# 팩토리 함수
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: SATSConfig,
    all_trial_ids: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    train / val DataLoader를 생성한다.

    cfg.val_ratio > 0 이면 논문 방식: 전체 시퀀스를 랜덤하게 (1-val_ratio)/val_ratio 로 분리.
    cfg.val_ratio == 0 이면 cfg.val_trials 기반 trial-level split 사용.
    """
    if cfg.gt_mode in {"on_the_fly", "gpu_on_the_fly"}:
        from .dataset_on_the_fly import build_dataloaders_on_the_fly

        return build_dataloaders_on_the_fly(cfg, all_trial_ids=all_trial_ids)
    if cfg.gt_mode != "precomputed":
        raise ValueError(f"unknown gt_mode: {cfg.gt_mode!r}")

    if all_trial_ids is None:
        with open(cfg.dataset_index_path) as f:
            idx = json.load(f)
        all_trial_ids = [t["trial_id"] for t in idx["trials"]]

    if cfg.include_materials or cfg.exclude_diameters:
        before = len(all_trial_ids)
        all_trial_ids = filter_trial_ids(
            all_trial_ids,
            include_materials=cfg.include_materials,
            exclude_diameters=cfg.exclude_diameters,
        )
        log.info(
            "trial filters include_materials=%s exclude_diameters=%s 적용: %d → %d trials",
            cfg.include_materials, cfg.exclude_diameters, before, len(all_trial_ids),
        )

    if cfg.val_ratio > 0:
        return _build_dataloaders_random_split(cfg, all_trial_ids)

    # ── trial-level split (기존 방식) ─────────────────────────────────────────
    val_set   = set(cfg.val_trials)
    train_ids = [t for t in all_trial_ids if t not in val_set]
    val_ids   = [t for t in all_trial_ids if t in val_set]

    log.info("Train trials (%d): %s", len(train_ids), train_ids)
    log.info("Val   trials (%d): %s", len(val_ids),   val_ids)

    if cfg.use_window_dataset:
        log.info("Window 데이터셋 모드 (window_size=%d, loading phase only)", cfg.window_size)
        train_ds   = SATSWindowDataset(train_ids, cfg)
        val_ds     = SATSWindowDataset(val_ids,   cfg)
        collate_fn = window_collate_fn
    else:
        train_ds   = SATSSequenceDataset(train_ids, cfg)
        val_ds     = SATSSequenceDataset(val_ids,   cfg)
        collate_fn = sats_collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.effective_device() == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.effective_device() == "cuda"),
        drop_last=False,
    )
    return train_loader, val_loader


def _build_dataloaders_random_split(
    cfg: SATSConfig,
    all_trial_ids: List[str],
) -> Tuple[DataLoader, DataLoader]:
    """
    논문 방식: 전체 시퀀스를 랜덤하게 (1-val_ratio)/val_ratio 로 분리.

    sequence 단위 split이므로 같은 trial의 다른 (x,y) 위치가 train/val에 섞일 수 있다.
    인접 window 간 data leakage를 피하기 위해 window가 아닌 sequence 단위로 split한다.
    """
    log.info(
        "랜덤 sequence-level split 모드 (val_ratio=%.2f, seed=%d)",
        cfg.val_ratio, cfg.seed,
    )

    # ── 1. 전체 trial 로드 ────────────────────────────────────────────────────
    all_seqs_flat: List[Tuple[str, dict]] = []   # (trial_id, seq_dict)
    gt_mmaps: Dict[str, np.ndarray] = {}

    for tid in all_trial_ids:
        result = _load_trial(tid, cfg)
        if result is None:
            continue
        seqs, gt_mmap = result
        gt_mmaps[tid] = gt_mmap
        for seq in seqs:
            all_seqs_flat.append((tid, seq))

    n_total = len(all_seqs_flat)
    log.info("전체 시퀀스: %d개", n_total)

    # ── 2. 랜덤 split ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * cfg.val_ratio))
    val_mask = set(perm[:n_val].tolist())

    train_seqs: Dict[str, List[dict]] = defaultdict(list)
    val_seqs:   Dict[str, List[dict]] = defaultdict(list)
    for i, (tid, seq) in enumerate(all_seqs_flat):
        if i in val_mask:
            val_seqs[tid].append(seq)
        else:
            train_seqs[tid].append(seq)

    log.info(
        "Split: train=%d 시퀀스 (%d trials), val=%d 시퀀스 (%d trials)",
        sum(len(v) for v in train_seqs.values()), len(train_seqs),
        sum(len(v) for v in val_seqs.values()),   len(val_seqs),
    )

    # ── 3. Dataset 생성 ───────────────────────────────────────────────────────
    if cfg.use_window_dataset:
        log.info("Window 데이터셋 모드 (window_size=%d)", cfg.window_size)
        train_ds = SATSWindowDataset(
            [], cfg,
            _preloaded={"sequences_by_trial": train_seqs, "gt_mmaps": gt_mmaps},
        )
        val_ds = SATSWindowDataset(
            [], cfg,
            _preloaded={"sequences_by_trial": val_seqs, "gt_mmaps": gt_mmaps},
        )
        collate_fn = window_collate_fn
    else:
        train_flat = [seq for seqs in train_seqs.values() for seq in seqs]
        val_flat   = [seq for seqs in val_seqs.values()   for seq in seqs]
        train_ds = SATSSequenceDataset(
            [], cfg,
            _preloaded={"sequences": train_flat, "gt_mmaps": gt_mmaps},
        )
        val_ds = SATSSequenceDataset(
            [], cfg,
            _preloaded={"sequences": val_flat, "gt_mmaps": gt_mmaps},
        )
        collate_fn = sats_collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.effective_device() == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.effective_device() == "cuda"),
        drop_last=False,
    )
    return train_loader, val_loader
