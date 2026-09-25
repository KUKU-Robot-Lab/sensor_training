"""
sats/inference/viz_postprocess.py

실시간 시각화 전용 후처리 유틸.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_OUTPUT_SCALE = 100.0


def to_nmm2(pred_map_scaled: np.ndarray) -> np.ndarray:
    """
    모델 출력 스케일(N/mm² x 100) -> 물리 단위(N/mm²) 변환.
    """
    return pred_map_scaled.astype(np.float32, copy=False) / MODEL_OUTPUT_SCALE


def apply_viz_threshold_nmm2(pred_map_nmm2: np.ndarray, threshold_nmm2: float) -> np.ndarray:
    """
    시각화용 하한 임계값 적용. threshold 이하 값을 0으로 클리핑.
    """
    if threshold_nmm2 <= 0:
        return pred_map_nmm2
    out = pred_map_nmm2.copy()
    out[out < threshold_nmm2] = 0.0
    return out


@lru_cache(maxsize=8)
def compute_gt_global_max_nmm2(gt_dir: str) -> float:
    """
    gt_dir의 *_targets.npy 전체를 스캔해 GT global max(N/mm²)를 반환한다.
    """
    root = Path(gt_dir)
    files = sorted(root.glob("*_targets.npy"))
    if not files:
        raise FileNotFoundError(f"GT targets 파일이 없습니다: {root}")

    global_max = 0.0
    for npy in files:
        arr = np.load(str(npy), mmap_mode="r")
        global_max = max(global_max, float(arr.max()))
    return global_max
