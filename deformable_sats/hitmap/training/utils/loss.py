"""
loss.py

Physics-informed 복합 loss:
  L_total = λ1·L_map + λ2·L_sensor + λ3·L_Fz + λ4·L_smooth

  L_map    : pseudo GT HR map 과의 L1 차이
  L_sensor : D(M_pred)를 sensor grid로 내렸을 때 실제 sensor 응답과 L1 일관성
  L_Fz     : map 적분값 ↔ 측정 Fz 일관성
  L_smooth : Total Variation (고주파 artifact 억제)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sensor_layout import downsample_to_sensor_batch


class SkinLoss(nn.Module):
    """
    Args:
        lambda_map:    L_map 가중치
        lambda_sensor: L_sensor 가중치
        lambda_fz:     L_Fz 가중치
        lambda_smooth: L_smooth (TV) 가중치
        pixel_area_mm2: HR map 픽셀 1개의 면적 (mm²). None이면 배치에서 계산.
        sensor_positions: (N, 2) 고정 센서 위치 텐서. None이면 매 forward에서 x/y_bounds로부터 계산.
        sensor_spacing_mm: sensor_positions=None일 때 사용.
        sensor_origin_x_mm: sensor_positions=None일 때 사용.
        sensor_origin_y_mm: sensor_positions=None일 때 사용.
    """

    def __init__(
        self,
        lambda_map: float = 1.0,
        lambda_sensor: float = 0.5,
        lambda_fz: float = 0.3,
        lambda_smooth: float = 0.1,
        pixel_area_mm2: Optional[float] = None,
        sensor_positions: Optional[torch.Tensor] = None,
        sensor_spacing_mm: float = 6.5,
        sensor_origin_x_mm: float = 0.0,
        sensor_origin_y_mm: float = 0.0,
    ) -> None:
        super().__init__()
        self.lam_map = lambda_map
        self.lam_sensor = lambda_sensor
        self.lam_fz = lambda_fz
        self.lam_smooth = lambda_smooth
        self.pixel_area_mm2 = pixel_area_mm2

        if sensor_positions is not None:
            self.register_buffer("sensor_positions", sensor_positions)
        else:
            self.sensor_positions = None
            self.sensor_spacing_mm = sensor_spacing_mm
            self.sensor_origin_x_mm = sensor_origin_x_mm
            self.sensor_origin_y_mm = sensor_origin_y_mm

    def forward(
        self,
        pred: torch.Tensor,
        target_map: torch.Tensor,
        tactile_raw: torch.Tensor,
        fz: torch.Tensor,
        x_bounds: torch.Tensor,
        y_bounds: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred:         (B, 1, H, W) 예측 HR map
            target_map:   (B, 1, H, W) pseudo GT
            tactile_raw:  (B, 16) 정규화된 실제 sensor 응답 (dead ch 포함)
            fz:           (B,) 측정 Fz [N]
            x_bounds:     (B, 2) canvas x 범위 [mm]
            y_bounds:     (B, 2) canvas y 범위 [mm]

        Returns:
            total_loss: scalar
            components: dict with individual loss values
        """
        # --- L_map: pseudo GT와 L1 ---
        l_map = F.l1_loss(pred, target_map)

        # --- L_sensor: D(M_pred) ↔ 실제 sensor 응답 ---
        l_sensor = self._sensor_consistency(pred, tactile_raw, x_bounds, y_bounds)

        # --- L_Fz: map 적분 ↔ measured Fz ---
        l_fz = self._force_consistency(pred, fz, x_bounds, y_bounds)

        # --- L_smooth: Total Variation ---
        l_smooth = total_variation(pred)

        total = (
            self.lam_map * l_map
            + self.lam_sensor * l_sensor
            + self.lam_fz * l_fz
            + self.lam_smooth * l_smooth
        )

        components = {
            "loss": total.item(),
            "l_map": l_map.item(),
            "l_sensor": l_sensor.item(),
            "l_fz": l_fz.item(),
            "l_smooth": l_smooth.item(),
        }
        return total, components

    def _sensor_consistency(
        self,
        pred: torch.Tensor,
        tactile_raw: torch.Tensor,
        x_bounds: torch.Tensor,
        y_bounds: torch.Tensor,
    ) -> torch.Tensor:
        pred_sensor = downsample_to_sensor_batch(
            pred,
            x_bounds,
            y_bounds,
            sensor_positions=getattr(self, "sensor_positions", None),
            spacing_mm=getattr(self, "sensor_spacing_mm", 6.5),
            origin_x_mm=getattr(self, "sensor_origin_x_mm", 0.0),
            origin_y_mm=getattr(self, "sensor_origin_y_mm", 0.0),
        )  # (B, 16)
        return F.l1_loss(pred_sensor, tactile_raw.to(pred.device))

    def _force_consistency(
        self,
        pred: torch.Tensor,
        fz: torch.Tensor,
        x_bounds: torch.Tensor,
        y_bounds: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H, W = pred.shape
        if self.pixel_area_mm2 is not None:
            pix_area = self.pixel_area_mm2
        else:
            dx = (x_bounds[:, 1] - x_bounds[:, 0]) / max(W - 1, 1)
            dy = (y_bounds[:, 1] - y_bounds[:, 0]) / max(H - 1, 1)
            pix_area = dx * dy  # (B,)

        map_sum = pred.reshape(B, -1).sum(dim=1)  # (B,)
        if isinstance(pix_area, torch.Tensor):
            predicted_fz = map_sum * pix_area
        else:
            predicted_fz = map_sum * pix_area

        fz_pos = fz.clamp(min=0.0)
        return F.l1_loss(predicted_fz, fz_pos.to(pred.device))


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV loss."""
    diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w
