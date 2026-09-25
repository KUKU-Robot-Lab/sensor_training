"""
sats/training/local_map_module.py

SATS Local Map Construction 모듈.

논문 근거 (Note S3 / Figure 2E)
--------------------------------
각 센서 i에 대해:
  combined_i = [local_feat_i ∥ agg_feat_i]          (LSTM 출력 + Self-Attention 출력)
  local_map_i = g_phi(combined_i)                    (공유 MLP 디코더)

16개 local_map을 센서 물리 좌표에 따라 41×41 전체 맵에 배치 & 합산 → merged_map.

센서 레이아웃 (4×4 grid, 6.5mm 간격)
--------------------------------------
  col→  0      1      2      3
row 0:  S1     S2     S3     S4    (y=-9.75mm → grid_row=0)
row 1:  S5     S6     S7     S8    (y=-3.25mm → nearest grid_row=13)
row 2:  S9    S10    S11    S12    (y=+3.25mm → nearest grid_row=27)
row 3: S13    S14    S15    S16    (y=+9.75mm → grid_row=40)

local_map_size=15 (홀수), half=7:
  내부 센서: local map 전체 사용
  경계 센서: 전체 맵 범위 밖으로 나가는 영역 클리핑
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from .config import SATSConfig
from .lstm_module import SensorLSTMEncoder
from .attention_module import SATSSelfAttention


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: 센서 물리 좌표 → 그리드 인덱스 매핑
# ─────────────────────────────────────────────────────────────────────────────

def build_sensor_physical_positions(
    n_sensors: int = 16,
    sensor_spacing_mm: float = 6.5,
) -> torch.Tensor:
    """
    Return physical sensor coordinates as ``[x_mm, y_mm]``.

    S1부터 x가 증가한다:
      S1=(-9.75,-9.75), S4=(9.75,-9.75), S16=(9.75,9.75).
    """
    side = int(math.sqrt(n_sensors))
    if side * side != n_sensors:
        raise ValueError(f"n_sensors must form a square grid: {n_sensors}")

    origin = -0.5 * (side - 1) * sensor_spacing_mm
    positions = torch.zeros(n_sensors, 2, dtype=torch.float32)
    for i in range(n_sensors):
        row_phys = i // side
        col_phys = i % side
        positions[i, 0] = origin + col_phys * sensor_spacing_mm
        positions[i, 1] = origin + row_phys * sensor_spacing_mm
    return positions


def _nearest_grid_index_outward_tie(
    coord_mm: float,
    grid_min_mm: float,
    grid_step_mm: float,
    grid_size: int,
) -> int:
    raw = (coord_mm - grid_min_mm) / grid_step_mm
    lower = math.floor(raw)
    upper = math.ceil(raw)
    lower_dist = abs(raw - lower)
    upper_dist = abs(upper - raw)
    if math.isclose(lower_dist, upper_dist, rel_tol=0.0, abs_tol=1e-9):
        idx = lower if coord_mm < 0.0 else upper
    elif lower_dist < upper_dist:
        idx = lower
    else:
        idx = upper
    return max(0, min(grid_size - 1, int(idx)))


def build_sensor_grid_positions(
    n_sensors: int = 16,
    grid_size: int = 41,
    grid_min_mm: float = -10.0,
    sensor_spacing_mm: float = 6.5,
    grid_step_mm: float = 0.5,
) -> torch.Tensor:
    """
    4×4 센서 그리드의 물리 좌표를 41×41 맵의 grid 인덱스로 변환한다.

    Parameters
    ----------
    n_sensors         : 센서 수 (기본 16)
    grid_size         : 전체 맵 한 변 크기 (기본 41)
    grid_min_mm       : 그리드 최솟값 (mm, 기본 -10.0)
    sensor_spacing_mm : 센서 간 물리 간격 (mm, 기본 6.5)
    grid_step_mm      : 그리드 step (mm, 기본 0.5)

    Returns
    -------
    positions : LongTensor[n_sensors, 2]
        각 행은 (grid_row, grid_col) — 0-indexed.
    """
    physical_xy = build_sensor_physical_positions(
        n_sensors=n_sensors,
        sensor_spacing_mm=sensor_spacing_mm,
    )
    positions = torch.zeros(n_sensors, 2, dtype=torch.long)
    for i in range(n_sensors):
        x_mm = float(physical_xy[i, 0].item())
        y_mm = float(physical_xy[i, 1].item())
        grid_r = _nearest_grid_index_outward_tie(y_mm, grid_min_mm, grid_step_mm, grid_size)
        grid_c = _nearest_grid_index_outward_tie(x_mm, grid_min_mm, grid_step_mm, grid_size)
        positions[i, 0] = grid_r
        positions[i, 1] = grid_c
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: local map 배치 슬라이스 계산 (경계 클리핑 포함)
# ─────────────────────────────────────────────────────────────────────────────

def build_placement_slices(
    sensor_positions: torch.Tensor,
    local_map_size: int,
    grid_size: int,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    각 센서에 대해 local map → global map 배치 슬라이스를 계산한다.

    경계 처리:
      - local map이 global map 범위를 벗어나면 해당 부분을 클리핑한다.
      - src 슬라이스: local map에서 실제 사용할 영역
      - dst 슬라이스: global map에서 배치할 위치

    Parameters
    ----------
    sensor_positions : LongTensor[n_sensors, 2] — (grid_row, grid_col)
    local_map_size   : local map 한 변 크기 (홀수)
    grid_size        : global map 한 변 크기

    Returns
    -------
    slices : list of (src_r, src_c, dst_r, dst_c)
        각 원소는 (start, end) 튜플.
        src_r = (src_r_start, src_r_end)  — local map row 슬라이스
        src_c = (src_c_start, src_c_end)  — local map col 슬라이스
        dst_r = (dst_r_start, dst_r_end)  — global map row 슬라이스
        dst_c = (dst_c_start, dst_c_end)  — global map col 슬라이스
    """
    half = local_map_size // 2
    n_sensors = sensor_positions.shape[0]
    result = []

    for i in range(n_sensors):
        cr = sensor_positions[i, 0].item()   # center grid row
        cc = sensor_positions[i, 1].item()   # center grid col

        # local map을 global map에 배치할 원래 범위 (클리핑 전)
        dst_r0 = cr - half
        dst_r1 = cr - half + local_map_size
        dst_c0 = cc - half
        dst_c1 = cc - half + local_map_size

        # 클리핑: global map 범위 [0, grid_size)
        src_r0 = max(0, -dst_r0)
        src_c0 = max(0, -dst_c0)
        dst_r0_clipped = max(0, dst_r0)
        dst_c0_clipped = max(0, dst_c0)
        dst_r1_clipped = min(grid_size, dst_r1)
        dst_c1_clipped = min(grid_size, dst_c1)

        actual_h = dst_r1_clipped - dst_r0_clipped
        actual_w = dst_c1_clipped - dst_c0_clipped

        src_r1 = src_r0 + actual_h
        src_c1 = src_c0 + actual_w

        result.append((
            (src_r0, src_r1),
            (src_c0, src_c1),
            (dst_r0_clipped, dst_r1_clipped),
            (dst_c0_clipped, dst_c1_clipped),
        ))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: 공유 MLP 디코더 g_phi
# ─────────────────────────────────────────────────────────────────────────────

class _LocalMapMLP(nn.Module):
    """
    공유 MLP 디코더 g_phi.

    모든 센서가 동일한 MLP 가중치를 공유한다 (weight sharing).
    논문 구조: 3-layer MLP, 250→375→500→195 (LeakyReLU).
    우리 구조: combined_dim → ×1.5 → ×2 → local_map_pixels (LeakyReLU).

    Input  : [B*n_sensors, combined_dim]
    Output : [B*n_sensors, local_map_size, local_map_size]
    """

    def __init__(self, combined_dim: int, local_map_size: int) -> None:
        super().__init__()
        self.local_map_size = local_map_size
        out_pixels = local_map_size * local_map_size
        h1 = int(combined_dim * 1.5)
        h2 = combined_dim * 2

        self.net = nn.Sequential(
            nn.Linear(combined_dim, h1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(h1, h2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(h2, out_pixels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [N, combined_dim]  (N = B * n_sensors)

        Returns
        -------
        out : [N, local_map_size, local_map_size]
        """
        N = x.size(0)
        return self.net(x).view(N, self.local_map_size, self.local_map_size)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: SATSLocalMapDecoder — 통합 디코더
# ─────────────────────────────────────────────────────────────────────────────

class SATSLocalMapDecoder(nn.Module):
    """
    [B, n_sensors, combined_dim] → merged_map [B, grid_size, grid_size]

    동작 순서:
      1. 16개 센서 feature를 공유 MLP로 통과 → 16개 local map
      2. 센서 물리 좌표에 따라 local map을 전체 맵에 배치 & 합산

    Parameters
    ----------
    combined_dim      : LSTM out + attn_dim
    local_map_size    : 각 센서 local map 한 변 크기
    grid_size         : 전체 맵 한 변 크기 (41)
    n_sensors         : 센서 수 (16)
    grid_min_mm       : 그리드 최솟값 (mm)
    sensor_spacing_mm : 센서 간 물리 간격 (mm)
    grid_step_mm      : 그리드 step (mm)
    """

    def __init__(
        self,
        combined_dim: int,
        local_map_size: int = 15,
        grid_size: int = 41,
        n_sensors: int = 16,
        grid_min_mm: float = -10.0,
        sensor_spacing_mm: float = 6.5,
        grid_step_mm: float = 0.5,
    ) -> None:
        super().__init__()
        self.local_map_size = local_map_size
        self.grid_size = grid_size
        self.n_sensors = n_sensors

        self.mlp = _LocalMapMLP(combined_dim, local_map_size)

        # 센서 위치 buffer (GPU로 자동 이동)
        positions = build_sensor_grid_positions(
            n_sensors=n_sensors,
            grid_size=grid_size,
            grid_min_mm=grid_min_mm,
            sensor_spacing_mm=sensor_spacing_mm,
            grid_step_mm=grid_step_mm,
        )
        self.register_buffer("sensor_positions", positions)  # [n_sensors, 2]

        # 배치 슬라이스 사전 계산 (Python 튜플 — buffer 불필요)
        self._placement_slices = build_placement_slices(
            positions, local_map_size, grid_size
        )

    def forward(self, combined_feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        combined_feat : [B, n_sensors, combined_dim]

        Returns
        -------
        merged_map : [B, grid_size, grid_size]
        """
        B = combined_feat.size(0)
        device = combined_feat.device

        # 1. 공유 MLP: 모든 센서 feature를 한 번에 처리
        flat = combined_feat.reshape(B * self.n_sensors, -1)          # [B*n, combined_dim]
        local_maps = self.mlp(flat)                                    # [B*n, H, W]
        local_maps = local_maps.view(B, self.n_sensors,
                                     self.local_map_size, self.local_map_size)  # [B, n, H, W]

        # 2. 전체 맵에 배치 & 합산
        merged = torch.zeros(B, self.grid_size, self.grid_size,
                             dtype=combined_feat.dtype, device=device)
        for i, (src_r, src_c, dst_r, dst_c) in enumerate(self._placement_slices):
            merged[:, dst_r[0]:dst_r[1], dst_c[0]:dst_c[1]] += (
                local_maps[:, i, src_r[0]:src_r[1], src_c[0]:src_c[1]]
            )

        return merged

    def extra_repr(self) -> str:
        return (
            f"local_map_size={self.local_map_size}, "
            f"grid_size={self.grid_size}, "
            f"n_sensors={self.n_sensors}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: SATSLocalMapStage — 학습용 통합 모듈
# ─────────────────────────────────────────────────────────────────────────────

class SATSLocalMapStage(nn.Module):
    """
    LSTM 인코더 + Self-Attention + Local Map Decoder.

    학습 전략
    ----------
    train_local_map.py에서 LSTM 인코더 & Self-Attention 가중치를 로드 후 동결,
    Local Map Decoder만 학습한다.

    Input
    -----
    sensor_seq : [B, T, 16]
    lengths    : [B]

    Output
    ------
    pred_map     : [B, 41, 41]        ← MSE 손실 대상
    combined_feat: [B, 16, combined]  ← 다음 단계(CNN) 연계용
    """

    def __init__(self, cfg: SATSConfig) -> None:
        super().__init__()

        self.encoder = SensorLSTMEncoder(
            n_sensors     = cfg.n_sensors,
            hidden_dim    = cfg.hidden_dim,
            num_layers    = cfg.num_layers,
            dropout       = cfg.dropout,
            bidirectional = cfg.bidirectional,
        )
        self.attention = SATSSelfAttention(
            in_dim    = self.encoder.out_dim,
            attn_dim  = cfg.attn_dim,
            n_sensors = cfg.n_sensors,
            n_layers  = cfg.n_gat_layers,
        )
        combined_dim = self.encoder.out_dim + cfg.attn_dim
        self.local_map_decoder = SATSLocalMapDecoder(
            combined_dim      = combined_dim,
            local_map_size    = cfg.local_map_size,
            grid_size         = cfg.grid_size,
            n_sensors         = cfg.n_sensors,
            grid_min_mm       = cfg.grid_min_mm,
            sensor_spacing_mm = cfg.sensor_spacing_mm,
            grid_step_mm      = cfg.grid_step_mm,
        )

    def forward(
        self,
        sensor_seq: torch.Tensor,   # [B, T, 16]
        lengths: torch.Tensor,      # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_feat    = self.encoder(sensor_seq, lengths)           # [B, 16, lstm_out]
        agg_feat      = self.attention(local_feat)                  # [B, 16, attn_dim]
        combined_feat = torch.cat([local_feat, agg_feat], dim=-1)  # [B, 16, combined]
        pred_map      = self.local_map_decoder(combined_feat)       # [B, grid, grid]
        return pred_map, combined_feat
