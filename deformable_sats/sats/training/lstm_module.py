"""
sats/training/lstm_module.py

SensorLSTMEncoder
-----------------
16개 센서 각각에 독립 LSTM 인코더를 할당한다.

논문 근거
---------
"LSTM networks are well-suited for encoding time-series data and have proven
 effective in modeling the hysteresis inherent in elastic sensing materials.
 Given that sensing units, even those from the same batch, have distinct
 response characteristics, a unique LSTM encoder is assigned to each sensing
 unit to accommodate its specific properties."
— Super-resolution tactile sensor arrays with sparse units enabled by deep learning

구조
----
                 ┌─ LSTM_1 ─┐  h_1 [B, hidden_dim]
sensor_seq       ├─ LSTM_2 ─┤  h_2
[B, T, 16] ─────┤   ...     ├──────────────────────► local_feat [B, 16, hidden_dim]
                 └─ LSTM_16─┘  h_16

각 LSTM_i:
  input  [B, T, 1]  →  LSTM(input_size=1, hidden_size=hidden_dim, num_layers)
  output h_i [B, hidden_dim]  ← 마지막 레이어, 마지막 유효 timestep의 hidden state

가변 길이 시퀀스 처리
--------------------
pack_padded_sequence / pad_packed_sequence를 사용하여
패딩 영역이 hidden state에 영향을 주지 않도록 한다.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SensorLSTMEncoder(nn.Module):
    """
    16개 센서 독립 LSTM 인코더.

    Parameters
    ----------
    n_sensors   : 센서 수 (기본 16)
    hidden_dim  : LSTM hidden size per sensor
    num_layers  : LSTM 레이어 깊이
    dropout     : 레이어 간 dropout (num_layers > 1 일 때만 유효)
    bidirectional : True면 양방향 LSTM (hidden_dim이 2배)
                    False(기본)면 단방향 → 실시간 추론 호환

    Input
    -----
    sensor_seq : Tensor[B, T, n_sensors]   s_norm 시계열
    lengths    : Tensor[B]                 int64, 각 샘플의 실제 길이

    Output
    ------
    local_feat : Tensor[B, n_sensors, out_dim]
                 out_dim = hidden_dim × (2 if bidirectional else 1)
    """

    def __init__(
        self,
        n_sensors: int    = 16,
        hidden_dim: int   = 64,
        num_layers: int   = 2,
        dropout: float    = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.n_sensors     = n_sensors
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.out_dim       = hidden_dim * (2 if bidirectional else 1)

        # 16개 독립 LSTM (각각 고유 가중치)
        # input_size=1 : 센서당 하나의 s_norm 값
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size  = 1,
                hidden_size = hidden_dim,
                num_layers  = num_layers,
                batch_first = True,
                dropout     = dropout if num_layers > 1 else 0.0,
                bidirectional = bidirectional,
            )
            for _ in range(n_sensors)
        ])

    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        sensor_seq: torch.Tensor,   # [B, T, n_sensors]
        lengths: torch.Tensor,      # [B]  int64
    ) -> torch.Tensor:
        """
        Returns
        -------
        local_feat : Tensor[B, n_sensors, out_dim]
        """
        B = sensor_seq.size(0)
        device = sensor_seq.device

        local_feats = []  # 각 센서의 h_i: [B, out_dim]

        # CPU에서 pack_padded_sequence를 위해 lengths는 CPU tensor여야 함
        lengths_cpu = lengths.cpu()

        for i, lstm in enumerate(self.lstms):
            # 센서 i의 시계열 추출: [B, T, 1]
            xi = sensor_seq[:, :, i : i + 1].contiguous()

            # 가변 길이 패킹
            packed = pack_padded_sequence(
                xi,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,   # 정렬 불필요 (DataLoader shuffle)
            )

            # LSTM 순전파
            # h_n: [num_layers * num_dir, B, hidden_dim]
            _, (h_n, _) = lstm(packed)

            # 마지막 레이어의 hidden state 추출
            if self.bidirectional:
                # h_n[-2]: 마지막 레이어 forward,  h_n[-1]: backward
                h_i = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, hidden_dim*2]
            else:
                h_i = h_n[-1]   # [B, hidden_dim]

            local_feats.append(h_i)

        # [n_sensors, B, out_dim] → [B, n_sensors, out_dim]
        local_feat = torch.stack(local_feats, dim=1)
        return local_feat

    # ─────────────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"n_sensors={self.n_sensors}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, out_dim={self.out_dim}, "
            f"bidirectional={self.bidirectional}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 학습용 임시 디코더 (LSTM 단독 학습 전용)
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMProxyDecoder(nn.Module):
    """
    LSTM 단독 학습을 위한 임시 선형 디코더.

    LSTM local_feat [B, 16, out_dim] → flatten → Linear → [B, 41, 41]

    이 모듈은 train_lstm.py에서만 사용되며,
    이후 Self-Attention + Local Map 모듈로 대체된다.
    """

    def __init__(self, n_sensors: int, lstm_out_dim: int, grid_size: int = 41) -> None:
        super().__init__()
        self.grid_size = grid_size
        in_features = n_sensors * lstm_out_dim

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, grid_size * grid_size),
        )

    def forward(self, local_feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        local_feat : [B, n_sensors, lstm_out_dim]

        Returns
        -------
        pred_map : [B, grid_size, grid_size]
        """
        B = local_feat.size(0)
        x = local_feat.reshape(B, -1)              # [B, n_sensors * out_dim]
        x = self.head(x)                            # [B, grid_size^2]
        return x.view(B, self.grid_size, self.grid_size)


class SATSLSTMStage(nn.Module):
    """
    LSTM 인코더 + 임시 디코더를 하나로 묶은 학습 모듈.

    train_lstm.py에서 사용. 이후 단계에서는 LSTMEncoder만 재사용한다.

    Input
    -----
    sensor_seq : [B, T, 16]
    lengths    : [B]

    Output
    ------
    pred_map   : [B, 41, 41]  마지막 유효 timestep 기준 GT와 MSE 비교
    local_feat : [B, 16, out_dim]  다음 모듈 연계용
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        from .config import SATSConfig
        assert isinstance(cfg, SATSConfig)

        self.encoder = SensorLSTMEncoder(
            n_sensors   = cfg.n_sensors,
            hidden_dim  = cfg.hidden_dim,
            num_layers  = cfg.num_layers,
            dropout     = cfg.dropout,
            bidirectional = cfg.bidirectional,
        )
        self.decoder = _LSTMProxyDecoder(
            n_sensors    = cfg.n_sensors,
            lstm_out_dim = self.encoder.out_dim,
            grid_size    = cfg.grid_size,
        )

    def forward(
        self,
        sensor_seq: torch.Tensor,   # [B, T, 16]
        lengths: torch.Tensor,      # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_feat = self.encoder(sensor_seq, lengths)   # [B, 16, out_dim]
        pred_map   = self.decoder(local_feat)            # [B, grid, grid]
        return pred_map, local_feat
