"""
sensor_layout.py

4×4 균등 그리드 센서 레이아웃 유틸리티.

센서 위치 (mm, 센서 프레임):
  Skin1  : (0.0,  0.0)   Skin2 : (6.5,  0.0) [dead]
  Skin3  : (13.0, 0.0)   Skin4 : (19.5, 0.0)
  Skin5  : (0.0,  6.5)   Skin6 : (6.5,  6.5)
  Skin7  : (13.0, 6.5)   Skin8 : (19.5, 6.5)
  Skin9  : (0.0,  13.0) [dead]  Skin10: (6.5,  13.0)
  Skin11 : (13.0, 13.0)  Skin12: (19.5, 13.0)
  Skin13 : (0.0,  19.5)  Skin14: (6.5,  19.5)
  Skin15 : (13.0, 19.5)  Skin16: (19.5, 19.5)

D(M) operator:
  HR map (B, 1, H, W) → sensor readings (B, 16)
  각 센서 위치에서 bilinear interpolation.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


SENSOR_SPACING_MM: float = 6.5
SENSOR_GRID_ROWS: int = 4
SENSOR_GRID_COLS: int = 4
N_SENSORS: int = SENSOR_GRID_ROWS * SENSOR_GRID_COLS  # 16

# Dead channel indices (0-indexed): No dead channels for current experiment
DEAD_CHANNEL_INDICES: Tuple[int, ...] = ()
LIVE_CHANNEL_INDICES: Tuple[int, ...] = tuple(range(N_SENSORS))  # All 16 channels


def build_sensor_positions(
    spacing_mm: float = 6.5,
    n_rows: int = 4,
    n_cols: int = 4,
) -> torch.Tensor:
    """
    사용자 제공 4x4 센서 레이아웃 (26.04 기준)
    S1~S4 (Y: -9.75), S5~S8 (Y: -3.25), S9~S12 (Y: 3.25), S13~S16 (Y: 9.75)
    X축은 S1(9.75)에서 S4(-9.75)로 감소하는 방향.
    """
    # X coordinates: S1(9.75), S2(3.25), S3(-3.25), S4(-9.75)
    xs = torch.tensor([9.75, 3.25, -3.25, -9.75], dtype=torch.float32)
    # Y coordinates for each row
    ys = torch.tensor([-9.75, -3.25, 3.25, 9.75], dtype=torch.float32)
    
    positions = []
    for y in ys:
        for x in xs:
            positions.append(torch.tensor([x, y]))
            
    return torch.stack(positions)  # (16, 2)


def downsample_to_sensor(
    hr_map: torch.Tensor,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    sensor_positions: Optional[torch.Tensor] = None,
    spacing_mm: float = SENSOR_SPACING_MM,
    origin_x_mm: float = 0.0,
    origin_y_mm: float = 0.0,
) -> torch.Tensor:
    """
    HR map을 센서 응답 벡터로 다운샘플링 (bilinear interpolation).
    canvas bounds가 모든 배치에서 동일할 때 사용.

    Args:
        hr_map:           (B, 1, H, W) float32
        x_bounds:         (x_min_mm, x_max_mm) — canvas x 범위 (배치 공통)
        y_bounds:         (y_min_mm, y_max_mm) — canvas y 범위 (배치 공통)
        sensor_positions: (N, 2) 센서 절대 위치 mm. None이면 기본값 사용.
        spacing_mm:       센서 간격 (sensor_positions=None일 때 사용)
        origin_x_mm:      센서 원점 x (sensor_positions=None일 때 사용)
        origin_y_mm:      센서 원점 y (sensor_positions=None일 때 사용)

    Returns:
        sensor_vals: (B, N) float32 — N개 센서의 HR map 보간값
    """
    if sensor_positions is None:
        sensor_positions = build_sensor_positions(spacing_mm, origin_x_mm, origin_y_mm)

    sensor_positions = sensor_positions.to(hr_map.device)

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])

    # 센서 위치를 [-1, 1] 정규화 좌표로 변환 (grid_sample 규약)
    norm_x = 2.0 * (sensor_positions[:, 0] - x_min) / (x_max - x_min) - 1.0
    norm_y = 2.0 * (sensor_positions[:, 1] - y_min) / (y_max - y_min) - 1.0

    # grid_sample: grid shape (B, H_out, W_out, 2), (x, y) 순서
    N = sensor_positions.shape[0]
    B = hr_map.shape[0]
    grid = torch.stack([norm_x, norm_y], dim=1)  # (N, 2)
    grid = grid.unsqueeze(0).unsqueeze(0).expand(B, 1, N, 2)  # (B, 1, N, 2)

    sampled = F.grid_sample(
        hr_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # (B, 1, 1, N)
    return sampled.squeeze(1).squeeze(1)  # (B, N)


def downsample_to_sensor_batch(
    hr_map: torch.Tensor,
    x_bounds: torch.Tensor,
    y_bounds: torch.Tensor,
    sensor_positions: Optional[torch.Tensor] = None,
    spacing_mm: float = SENSOR_SPACING_MM,
    origin_x_mm: float = 0.0,
    origin_y_mm: float = 0.0,
) -> torch.Tensor:
    """
    배치 내 샘플마다 canvas bounds가 다를 때의 벡터화 버전.
    GPU에서 루프 없이 전체 배치를 한 번에 처리.

    Args:
        hr_map:    (B, 1, H, W) float32
        x_bounds:  (B, 2) — 각 샘플의 canvas x 범위
        y_bounds:  (B, 2) — 각 샘플의 canvas y 범위
        sensor_positions: (N, 2) 절대 위치 mm. None이면 기본값.

    Returns:
        sensor_vals: (B, N) float32
    """
    if sensor_positions is None:
        sensor_positions = build_sensor_positions(spacing_mm, origin_x_mm, origin_y_mm)

    device = hr_map.device
    dtype = hr_map.dtype
    sensor_positions = sensor_positions.to(device=device, dtype=dtype)  # (N, 2)
    x_bounds = x_bounds.to(device=device, dtype=dtype)   # (B, 2)
    y_bounds = y_bounds.to(device=device, dtype=dtype)   # (B, 2)

    B = hr_map.shape[0]
    N = sensor_positions.shape[0]

    # 각 샘플별 bounds로 센서 위치를 [-1, 1]로 정규화
    # sensor_positions: (N, 2) → (1, N, 2)
    # x_bounds: (B, 2) → x_min (B, 1), x_max (B, 1)
    x_min = x_bounds[:, 0:1]  # (B, 1)
    x_max = x_bounds[:, 1:2]  # (B, 1)
    y_min = y_bounds[:, 0:1]  # (B, 1)
    y_max = y_bounds[:, 1:2]  # (B, 1)

    sx = sensor_positions[:, 0].unsqueeze(0)  # (1, N)
    sy = sensor_positions[:, 1].unsqueeze(0)  # (1, N)

    norm_x = 2.0 * (sx - x_min) / (x_max - x_min) - 1.0  # (B, N)
    norm_y = 2.0 * (sy - y_min) / (y_max - y_min) - 1.0  # (B, N)

    # grid_sample grid: (B, 1, N, 2)
    grid = torch.stack([norm_x, norm_y], dim=2)   # (B, N, 2)
    grid = grid.unsqueeze(1)                        # (B, 1, N, 2)

    sampled = F.grid_sample(
        hr_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # (B, 1, 1, N)
    return sampled.squeeze(1).squeeze(1)  # (B, N)


def get_live_channels(tactile: torch.Tensor) -> torch.Tensor:
    """
    (B, 16) 또는 (16,) 텐서에서 dead channel 제거 → (B, 14) 또는 (14,)
    """
    idx = list(LIVE_CHANNEL_INDICES)
    return tactile[..., idx]
