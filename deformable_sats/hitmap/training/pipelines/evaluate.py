"""
evaluate.py

평가 지표:
  - centroid_error_mm : 예측 map centroid vs GT (cx, cy) 거리
  - fz_mae            : map 적분 기반 예측 Fz vs measured Fz MAE
  - iou               : threshold 기반 IoU
  - dice              : Dice coefficient
  - sensor_l1         : D(M_pred) vs tactile_raw L1
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .sensor_layout import downsample_to_sensor_batch


@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target_map: torch.Tensor,
    tactile_raw: torch.Tensor,
    fz: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    x_bounds: torch.Tensor,
    y_bounds: torch.Tensor,
    threshold_ratio: float = 0.2,
    sensor_spacing_mm: float = 6.5,
    sensor_origin_x_mm: float = 0.0,
    sensor_origin_y_mm: float = 0.0,
) -> Dict[str, float]:
    """
    Args:
        pred:         (B, 1, H, W) 예측 HR map
        target_map:   (B, 1, H, W) pseudo GT
        tactile_raw:  (B, 16) 정규화 sensor 응답
        fz:           (B,) measured Fz [N]
        cx, cy:       (B,) contact center [mm]
        x_bounds:     (B, 2)
        y_bounds:     (B, 2)

    Returns:
        metrics dict (평균값)
    """
    B, _, H, W = pred.shape
    device = pred.device

    metrics: Dict[str, float] = {}

    # --- centroid error ---
    pred_cx, pred_cy = compute_centroid(pred, x_bounds, y_bounds)  # (B,), (B,)
    cx = cx.to(device)
    cy = cy.to(device)
    centroid_err = torch.sqrt((pred_cx - cx) ** 2 + (pred_cy - cy) ** 2)
    metrics["centroid_error_mm"] = centroid_err.mean().item()

    # --- Fz MAE ---
    dx = (x_bounds[:, 1] - x_bounds[:, 0]).to(device) / max(W - 1, 1)
    dy = (y_bounds[:, 1] - y_bounds[:, 0]).to(device) / max(H - 1, 1)
    pix_area = dx * dy  # (B,)
    pred_fz = pred.reshape(B, -1).sum(dim=1) * pix_area
    fz_mae = (pred_fz - fz.to(device).clamp(min=0.0)).abs().mean()
    metrics["fz_mae"] = fz_mae.item()

    # --- IoU & Dice ---
    pred_thresh = _threshold(pred, threshold_ratio)
    gt_thresh = _threshold(target_map, threshold_ratio)
    iou, dice = iou_dice(pred_thresh, gt_thresh)
    metrics["iou"] = iou.item()
    metrics["dice"] = dice.item()

    # --- sensor L1 ---
    pred_sensor = downsample_to_sensor_batch(
        pred, x_bounds, y_bounds,
        spacing_mm=sensor_spacing_mm,
        origin_x_mm=sensor_origin_x_mm,
        origin_y_mm=sensor_origin_y_mm,
    )  # (B, 16)
    metrics["sensor_l1"] = F.l1_loss(pred_sensor, tactile_raw.to(device)).item()

    # --- map L1 (참고용) ---
    metrics["map_l1"] = F.l1_loss(pred, target_map).item()

    return metrics


def compute_centroid(
    hr_map: torch.Tensor,
    x_bounds: torch.Tensor,
    y_bounds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map의 weighted centroid를 mm 단위로 반환.

    Args:
        hr_map:   (B, 1, H, W)
        x_bounds: (B, 2)
        y_bounds: (B, 2)
    Returns:
        cx: (B,), cy: (B,) in mm
    """
    B, _, H, W = hr_map.shape
    device = hr_map.device

    # pixel 중심 좌표 생성
    # x: left→right, y: top→bottom
    xs = torch.linspace(0, 1, W, device=device)  # (W,)
    ys = torch.linspace(0, 1, H, device=device)  # (H,)

    weight = hr_map[:, 0, :, :]  # (B, H, W)
    total = weight.reshape(B, -1).sum(dim=1).clamp(min=1e-8)  # (B,)

    # centroid in [0,1]
    cx_norm = (weight * xs.unsqueeze(0)).reshape(B, -1).sum(dim=1) / total  # (B,)
    cy_norm = (weight * ys.unsqueeze(1)).reshape(B, -1).sum(dim=1) / total  # (B,)

    # [0,1] → mm
    x_bounds = x_bounds.to(device)
    y_bounds = y_bounds.to(device)
    cx_mm = x_bounds[:, 0] + cx_norm * (x_bounds[:, 1] - x_bounds[:, 0])
    cy_mm = y_bounds[:, 0] + cy_norm * (y_bounds[:, 1] - y_bounds[:, 0])

    return cx_mm, cy_mm


def _threshold(hr_map: torch.Tensor, ratio: float) -> torch.Tensor:
    """peak 값 대비 ratio 이상인 픽셀을 1로 설정."""
    B = hr_map.shape[0]
    peak = hr_map.reshape(B, -1).max(dim=1).values  # (B,)
    thresh = (peak * ratio).reshape(B, 1, 1, 1)
    return (hr_map >= thresh).float()


def iou_dice(pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B = pred_bin.shape[0]
    p = pred_bin.reshape(B, -1)
    g = gt_bin.reshape(B, -1)
    intersection = (p * g).sum(dim=1)
    union = (p + g - p * g).sum(dim=1).clamp(min=1e-8)
    sum_pg = (p.sum(dim=1) + g.sum(dim=1)).clamp(min=1e-8)
    iou = (intersection / union).mean()
    dice = (2 * intersection / sum_pg).mean()
    return iou, dice
