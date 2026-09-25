"""
tests/test_local_map_module.py

SATSLocalMapDecoder / SATSLocalMapStage TDD
"""
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[3]))

from sats.training.config import SATSConfig
from sats.training.local_map_module import (
    build_sensor_physical_positions,
    build_sensor_grid_positions,
    build_placement_slices,
    SATSLocalMapDecoder,
    SATSLocalMapStage,
)


# ─────────────────────────────────────────────────────────────────────────────
# 공용 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _small_cfg(**kwargs):
    defaults = dict(
        hidden_dim=16,
        num_layers=1,
        attn_dim=16,
        dropout=0.0,
        grid_size=41,
        n_sensors=16,
        local_map_size=15,
        sensor_spacing_mm=6.5,
        clip_grad=1.0,
    )
    defaults.update(kwargs)
    return SATSConfig(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: build_sensor_grid_positions
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSensorGridPositions:
    """4×4 센서의 물리 좌표 → 그리드 인덱스 변환."""

    def _pos(self, **kwargs):
        return build_sensor_grid_positions(**kwargs)

    def test_physical_positions_s1_to_s16_x_increasing(self):
        xy = build_sensor_physical_positions()
        assert xy[0].tolist() == [-9.75, -9.75]
        assert xy[3].tolist() == [9.75, -9.75]
        assert xy[15].tolist() == [9.75, 9.75]

    def test_shape(self):
        pos = self._pos()
        assert pos.shape == (16, 2), f"shape 오류: {pos.shape}"

    def test_dtype_long(self):
        pos = self._pos()
        assert pos.dtype == torch.long

    def test_sensor0_top_left(self):
        """S1(idx=0): x=-9.75, y=-9.75 → nearest grid (0, 0)."""
        pos = self._pos()
        assert pos[0, 0].item() == 0   # grid_row (y)
        assert pos[0, 1].item() == 0   # grid_col (x)

    def test_sensor5_interior(self):
        """S6(idx=5): x=-3.25, y=-3.25 → nearest grid (13, 13)."""
        pos = self._pos()
        assert pos[5, 0].item() == 13
        assert pos[5, 1].item() == 13

    def test_sensor15_bottom_right(self):
        """S16(idx=15): x=+9.75, y=+9.75 → nearest grid (40, 40)."""
        pos = self._pos()
        assert pos[15, 0].item() == 40
        assert pos[15, 1].item() == 40

    def test_sensor3_top_right(self):
        """S4(idx=3): x=+9.75, y=-9.75 → nearest grid (0, 40)."""
        pos = self._pos()
        assert pos[3, 0].item() == 0
        assert pos[3, 1].item() == 40

    def test_sensor12_bottom_left(self):
        """S13(idx=12): x=-9.75, y=+9.75 → nearest grid (40, 0)."""
        pos = self._pos()
        assert pos[12, 0].item() == 40
        assert pos[12, 1].item() == 0

    def test_all_values_in_range(self):
        """모든 grid 인덱스가 [0, 40] 범위."""
        pos = self._pos()
        assert (pos >= 0).all()
        assert (pos <= 40).all()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: build_placement_slices
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPlacementSlices:
    """
    16개 센서 각각의 (src_slice, dst_slice) 쌍 검증.

    local_map_size=15, half=7, grid_size=41.
    sensor_positions: [0,0], [0,13], ..., [40,40]
    """

    def _slices(self):
        pos = build_sensor_grid_positions()
        return build_placement_slices(pos, local_map_size=15, grid_size=41)

    def test_returns_16_entries(self):
        slices = self._slices()
        assert len(slices) == 16

    def test_each_entry_has_4_tuples(self):
        """각 항목은 (src_r, src_c, dst_r, dst_c) — 각각 (start, end) 2-tuple."""
        slices = self._slices()
        for i, entry in enumerate(slices):
            assert len(entry) == 4, f"센서 {i}: 항목 수 오류"
            for tup in entry:
                assert len(tup) == 2, f"센서 {i}: slice tuple 길이 오류"

    def test_src_dst_sizes_match(self):
        """src와 dst의 실제 크기(end-start)가 모든 센서에서 일치해야 한다."""
        slices = self._slices()
        for i, (src_r, src_c, dst_r, dst_c) in enumerate(slices):
            src_h = src_r[1] - src_r[0]
            src_w = src_c[1] - src_c[0]
            dst_h = dst_r[1] - dst_r[0]
            dst_w = dst_c[1] - dst_c[0]
            assert src_h == dst_h, f"센서 {i}: 높이 불일치 src={src_h} dst={dst_h}"
            assert src_w == dst_w, f"센서 {i}: 너비 불일치 src={src_w} dst={dst_w}"

    def test_dst_slice_in_bounds(self):
        """dst 슬라이스는 항상 [0, grid_size) 범위."""
        slices = self._slices()
        for i, (_, _, dst_r, dst_c) in enumerate(slices):
            assert dst_r[0] >= 0,  f"센서 {i}: dst_r 시작 음수"
            assert dst_r[1] <= 41, f"센서 {i}: dst_r 끝 초과"
            assert dst_c[0] >= 0,  f"센서 {i}: dst_c 시작 음수"
            assert dst_c[1] <= 41, f"센서 {i}: dst_c 끝 초과"

    def test_src_slice_in_bounds(self):
        """src 슬라이스는 항상 [0, local_map_size) 범위."""
        slices = self._slices()
        for i, (src_r, src_c, _, _) in enumerate(slices):
            assert src_r[0] >= 0,  f"센서 {i}: src_r 시작 음수"
            assert src_r[1] <= 15, f"센서 {i}: src_r 끝 초과"
            assert src_c[0] >= 0,  f"센서 {i}: src_c 시작 음수"
            assert src_c[1] <= 15, f"센서 {i}: src_c 끝 초과"

    def test_corner_sensor0_clipped(self):
        """
        S1(idx=0): center=(0,0), half=7
        → local_map row [7:15], col [7:15] (src) → dst row [0:8], col [0:8]
        """
        slices = self._slices()
        src_r, src_c, dst_r, dst_c = slices[0]
        assert src_r == (7, 15), f"S1 src_r 오류: {src_r}"
        assert src_c == (7, 15), f"S1 src_c 오류: {src_c}"
        assert dst_r == (0, 8),  f"S1 dst_r 오류: {dst_r}"
        assert dst_c == (0, 8),  f"S1 dst_c 오류: {dst_c}"

    def test_interior_sensor5_no_clip(self):
        """
        S6(idx=5): center=(13,13), half=7
        → local_map 전체 [0:15, 0:15] (src) → dst row [6:21], col [6:21]
        """
        slices = self._slices()
        src_r, src_c, dst_r, dst_c = slices[5]
        assert src_r == (0, 15), f"S6 src_r 오류: {src_r}"
        assert src_c == (0, 15), f"S6 src_c 오류: {src_c}"
        assert dst_r == (6, 21), f"S6 dst_r 오류: {dst_r}"
        assert dst_c == (6, 21), f"S6 dst_c 오류: {dst_c}"

    def test_corner_sensor15_clipped(self):
        """
        S16(idx=15): center=(40,40), half=7
        → local_map row [0:8], col [0:8] (src) → dst row [33:41], col [33:41]
        """
        slices = self._slices()
        src_r, src_c, dst_r, dst_c = slices[15]
        assert src_r == (0, 8),   f"S16 src_r 오류: {src_r}"
        assert src_c == (0, 8),   f"S16 src_c 오류: {src_c}"
        assert dst_r == (33, 41), f"S16 dst_r 오류: {dst_r}"
        assert dst_c == (33, 41), f"S16 dst_c 오류: {dst_c}"

    def test_size_positive(self):
        """모든 슬라이스 크기가 양수 (0 크기 슬라이스 없음)."""
        slices = self._slices()
        for i, (src_r, src_c, dst_r, dst_c) in enumerate(slices):
            assert src_r[1] - src_r[0] > 0, f"센서 {i}: src 높이 0"
            assert src_c[1] - src_c[0] > 0, f"센서 {i}: src 너비 0"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: _LocalMapMLP (내부 모듈 — SATSLocalMapDecoder를 통해 간접 테스트)
# ─────────────────────────────────────────────────────────────────────────────

# _LocalMapMLP는 SATSLocalMapDecoder 내부 구현 세부사항이므로
# TestSATSLocalMapDecoder에서 통합 검증


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: SATSLocalMapDecoder
# ─────────────────────────────────────────────────────────────────────────────

class TestSATSLocalMapDecoder:

    def _make_decoder(self, combined_dim=32, local_map_size=15, grid_size=41):
        return SATSLocalMapDecoder(
            combined_dim=combined_dim,
            local_map_size=local_map_size,
            grid_size=grid_size,
            n_sensors=16,
            grid_min_mm=-10.0,
            sensor_spacing_mm=6.5,
            grid_step_mm=0.5,
        )

    def _rand_combined(self, B=2, combined_dim=32):
        return torch.randn(B, 16, combined_dim)

    def test_output_shape(self):
        dec = self._make_decoder()
        x = self._rand_combined()
        out = dec(x)
        assert out.shape == (2, 41, 41), f"shape 오류: {out.shape}"

    def test_no_nan_inf(self):
        dec = self._make_decoder()
        x = self._rand_combined()
        out = dec(x)
        assert not torch.isnan(out).any(), "NaN 발생"
        assert not torch.isinf(out).any(), "Inf 발생"

    def test_output_dtype_float32(self):
        dec = self._make_decoder()
        x = self._rand_combined()
        out = dec(x)
        assert out.dtype == torch.float32

    def test_gradient_flows(self):
        """역전파 시 MLP 파라미터에 gradient가 흘러야 한다."""
        dec = self._make_decoder()
        x = self._rand_combined()
        out = dec(x)
        out.sum().backward()
        for name, p in dec.named_parameters():
            assert p.grad is not None, f"{name} gradient 없음"

    def test_batch_independence(self):
        """배치 내 샘플 간 출력이 독립적이어야 한다."""
        dec = self._make_decoder()
        dec.eval()
        x = self._rand_combined(B=4)
        with torch.no_grad():
            out_full = dec(x)
            out_single = dec(x[2:3])
        torch.testing.assert_close(out_full[2:3], out_single)

    def test_different_inputs_different_outputs(self):
        dec = self._make_decoder()
        dec.eval()
        x1 = torch.randn(2, 16, 32)
        x2 = torch.randn(2, 16, 32)
        with torch.no_grad():
            o1 = dec(x1)
            o2 = dec(x2)
        assert not torch.allclose(o1, o2), "다른 입력에서 같은 출력"

    def test_sensor_positions_buffer_registered(self):
        """sensor_positions가 buffer로 등록되어 state_dict에 포함되어야 한다."""
        dec = self._make_decoder()
        sd = dec.state_dict()
        assert "sensor_positions" in sd, "sensor_positions buffer가 state_dict에 없음"

    def test_single_sensor_activation(self):
        """
        센서 0(S1, top-left, grid=(0,0)) 활성화 시 해당 영역이 더 활성화되는지 검증.
        S1: dst_r=(0,8), dst_c=(0,8)
        """
        dec = self._make_decoder(combined_dim=32)
        dec.eval()
        # 모든 입력을 0으로 시작
        x = torch.zeros(1, 16, 32)
        # 센서 0 (S1) 활성화
        x[0, 0] = torch.randn(32).abs() + 1.0
        with torch.no_grad():
            out = dec(x)  # [1, 41, 41]
        # S1 영역: (0,0) 근처 (dst: (0,8),(0,8))
        s1_region = out[0, 0:8, 0:8].abs().mean()
        # 반대편 영역: (32,32) 근처
        other_region = out[0, 33:41, 33:41].abs().mean()
        # S1 영역이 더 활성화되어야 함
        assert s1_region >= other_region, (
            f"S1 영역({s1_region:.4f})이 반대 영역({other_region:.4f})보다 작음"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: SATSLocalMapStage
# ─────────────────────────────────────────────────────────────────────────────

class TestSATSLocalMapStage:

    def _make_stage(self):
        cfg = _small_cfg()
        return SATSLocalMapStage(cfg)

    def _rand_batch(self, B=2, T=10, n=16):
        sensor_seq = torch.randn(B, T, n)
        lengths = torch.tensor([T, T - 2])
        return sensor_seq, lengths

    def test_pred_map_shape(self):
        stage = self._make_stage()
        x, l = self._rand_batch()
        pred_map, _ = stage(x, l)
        assert pred_map.shape == (2, 41, 41), f"pred_map shape 오류: {pred_map.shape}"

    def test_combined_feat_shape(self):
        """combined_feat shape = [B, 16, lstm_out + attn_dim]."""
        stage = self._make_stage()
        x, l = self._rand_batch()
        _, combined = stage(x, l)
        expected_combined_dim = stage.encoder.out_dim + stage.attention.attn_dim
        assert combined.shape == (2, 16, expected_combined_dim), (
            f"combined shape 오류: {combined.shape}"
        )

    def test_no_nan_in_pred_map(self):
        stage = self._make_stage()
        x, l = self._rand_batch()
        pred_map, _ = stage(x, l)
        assert not torch.isnan(pred_map).any()

    def test_gradient_flows_through_decoder(self):
        """local_map_decoder 파라미터에 gradient가 흘러야 한다."""
        stage = self._make_stage()
        x, l = self._rand_batch()
        pred_map, _ = stage(x, l)
        pred_map.sum().backward()
        for name, p in stage.local_map_decoder.named_parameters():
            assert p.grad is not None, f"local_map_decoder.{name} grad 없음"

    def test_frozen_encoder_attention_no_grad(self):
        """encoder + attention 동결 시 grad 없어야 한다."""
        stage = self._make_stage()
        for p in stage.encoder.parameters():
            p.requires_grad_(False)
        for p in stage.attention.parameters():
            p.requires_grad_(False)

        x, l = self._rand_batch()
        pred_map, _ = stage(x, l)
        pred_map.sum().backward()

        for name, p in stage.encoder.named_parameters():
            assert p.grad is None, f"encoder.{name} grad 있어서는 안 됨"
        for name, p in stage.attention.named_parameters():
            assert p.grad is None, f"attention.{name} grad 있어서는 안 됨"

    def test_init_from_config(self):
        """SATSConfig로부터 올바르게 초기화되는지 확인."""
        cfg = _small_cfg(hidden_dim=32, attn_dim=24, local_map_size=11)
        stage = SATSLocalMapStage(cfg)
        # local_map_decoder의 local_map_size 확인
        assert stage.local_map_decoder.local_map_size == 11
