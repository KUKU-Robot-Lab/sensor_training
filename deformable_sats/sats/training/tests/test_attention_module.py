"""
tests/test_attention_module.py

SATSSelfAttention / SATSAttentionStage TDD
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[3]))

from sats.training.config import SATSConfig
from sats.training.attention_module import (
    SATSSelfAttention,
    SATSAttentionStage,
    build_adjacency_4x4,
)


# ─────────────────────────────────────────────────────────────────────────────
# 인접 행렬 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAdjacency4x4:
    """4×4 센서 그리드 8-connected 인접 행렬."""

    def test_shape(self):
        adj = build_adjacency_4x4()
        assert adj.shape == (16, 16), "인접 행렬 shape은 [16, 16]"

    def test_dtype_bool(self):
        adj = build_adjacency_4x4()
        assert adj.dtype == torch.bool

    def test_self_loops_all_true(self):
        """대각선 원소(자기 자신)는 모두 True."""
        adj = build_adjacency_4x4()
        assert adj.diagonal().all(), "모든 self-loop는 True여야 한다"

    def test_symmetric(self):
        """인접 관계는 대칭이어야 한다."""
        adj = build_adjacency_4x4()
        assert (adj == adj.T).all(), "adj는 대칭행렬"

    def test_adjacent_row_neighbors(self):
        """S1(0)–S5(4): 같은 열, 인접 행 → True"""
        adj = build_adjacency_4x4()
        assert adj[0, 4].item() is True   # S1 ↔ S5

    def test_adjacent_col_neighbors(self):
        """S1(0)–S2(1): 같은 행, 인접 열 → True"""
        adj = build_adjacency_4x4()
        assert adj[0, 1].item() is True   # S1 ↔ S2

    def test_diagonal_neighbors(self):
        """S1(0)–S6(5): 대각 방향 → True"""
        adj = build_adjacency_4x4()
        assert adj[0, 5].item() is True   # S1 ↔ S6 (대각)

    def test_non_adjacent_same_row(self):
        """S1(0)–S3(2): 같은 행, col 차이=2 → False"""
        adj = build_adjacency_4x4()
        assert adj[0, 2].item() is False

    def test_non_adjacent_different_row(self):
        """S1(0)–S9(8): row 차이=2 → False"""
        adj = build_adjacency_4x4()
        assert adj[0, 8].item() is False

    def test_corner_sensor_neighbor_count(self):
        """코너 센서(idx=0)의 이웃 수: 4 (자기 포함)."""
        adj = build_adjacency_4x4()
        assert adj[0].sum().item() == 4

    def test_edge_sensor_neighbor_count(self):
        """엣지 센서(idx=1)의 이웃 수: 6 (자기 포함)."""
        adj = build_adjacency_4x4()
        assert adj[1].sum().item() == 6

    def test_interior_sensor_neighbor_count(self):
        """내부 센서(idx=5, S6)의 이웃 수: 9 (자기 포함)."""
        adj = build_adjacency_4x4()
        assert adj[5].sum().item() == 9


# ─────────────────────────────────────────────────────────────────────────────
# SATSSelfAttention 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestSATSSelfAttention:
    def _make_module(self, in_dim=64, attn_dim=64):
        return SATSSelfAttention(in_dim=in_dim, attn_dim=attn_dim)

    def _rand_input(self, B=4, n=16, in_dim=64):
        return torch.randn(B, n, in_dim)

    def test_output_shape(self):
        mod = self._make_module()
        x = self._rand_input()
        out = mod(x)
        assert out.shape == (4, 16, 64), f"예상 [4,16,64], 실제 {out.shape}"

    def test_output_shape_custom_attn_dim(self):
        mod = self._make_module(in_dim=64, attn_dim=32)
        x = self._rand_input()
        out = mod(x)
        assert out.shape == (4, 16, 32)

    def test_no_nan_inf(self):
        mod = self._make_module()
        x = self._rand_input()
        out = mod(x)
        assert not torch.isnan(out).any(), "NaN 발생"
        assert not torch.isinf(out).any(), "Inf 발생"

    def test_output_dtype_float32(self):
        mod = self._make_module()
        x = self._rand_input()
        out = mod(x)
        assert out.dtype == torch.float32

    def test_batch_independence(self):
        """배치 내 샘플 간 출력이 독립적이어야 한다."""
        mod = self._make_module()
        mod.eval()
        x = self._rand_input(B=4)
        with torch.no_grad():
            out_full = mod(x)
            out_single = mod(x[2:3])
        torch.testing.assert_close(out_full[2:3], out_single)

    def test_gradient_flows(self):
        """역전파 시 gradient가 모든 파라미터에 흐르는지 확인."""
        mod = self._make_module()
        x = self._rand_input()
        out = mod(x)
        loss = out.sum()
        loss.backward()
        for name, p in mod.named_parameters():
            assert p.grad is not None, f"{name} gradient가 없음"

    def test_adj_buffer_registered(self):
        """인접 행렬이 buffer로 등록되어 state_dict에 포함되어야 한다."""
        mod = self._make_module()
        sd = mod.state_dict()
        assert "adj" in sd, "adj buffer가 state_dict에 없음"

    def test_different_inputs_different_outputs(self):
        """서로 다른 입력은 서로 다른 출력을 내야 한다."""
        mod = self._make_module()
        mod.eval()
        x1 = torch.randn(2, 16, 64)
        x2 = torch.randn(2, 16, 64)
        with torch.no_grad():
            o1 = mod(x1)
            o2 = mod(x2)
        assert not torch.allclose(o1, o2), "다른 입력에서 같은 출력"


# ─────────────────────────────────────────────────────────────────────────────
# SATSAttentionStage 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestSATSAttentionStage:
    def _make_stage(self):
        cfg = SATSConfig(hidden_dim=32, num_layers=1, attn_dim=32)
        return SATSAttentionStage(cfg)

    def _rand_batch(self, B=2, T=10, n=16):
        sensor_seq = torch.randn(B, T, n)
        lengths = torch.tensor([T, T - 2])
        return sensor_seq, lengths

    def test_pred_map_shape(self):
        stage = self._make_stage()
        x, l = self._rand_batch()
        pred_map, _ = stage(x, l)
        assert pred_map.shape == (2, 41, 41), f"pred_map shape 오류: {pred_map.shape}"

    def test_agg_feat_shape(self):
        stage = self._make_stage()
        x, l = self._rand_batch()
        _, agg_feat = stage(x, l)
        assert agg_feat.shape == (2, 16, 32), f"agg_feat shape 오류: {agg_feat.shape}"

    def test_no_nan_in_pred_map(self):
        stage = self._make_stage()
        x, l = self._rand_batch()
        pred_map, _ = stage(x, l)
        assert not torch.isnan(pred_map).any()

    def test_gradient_flows_through_attention(self):
        """Self-Attention / 디코더 파라미터에 gradient가 흘러야 한다."""
        stage = self._make_stage()
        x, l = self._rand_batch()
        pred_map, _ = stage(x, l)
        pred_map.sum().backward()
        # attention 파라미터 확인
        for name, p in stage.attention.named_parameters():
            assert p.grad is not None, f"attention.{name} grad 없음"
        # decoder 파라미터 확인
        for name, p in stage.decoder.named_parameters():
            assert p.grad is not None, f"decoder.{name} grad 없음"

    def test_frozen_encoder_no_grad(self):
        """encoder가 frozen 상태면 grad가 없어야 한다."""
        stage = self._make_stage()
        # encoder 동결
        for p in stage.encoder.parameters():
            p.requires_grad_(False)
        x, l = self._rand_batch()
        pred_map, _ = stage(x, l)
        pred_map.sum().backward()
        for name, p in stage.encoder.named_parameters():
            assert p.grad is None, f"encoder.{name} grad가 있어서는 안 됨"
