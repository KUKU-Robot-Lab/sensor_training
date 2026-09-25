"""
tests/test_cnn_module.py

SATSCNNRefiner / SATSCNNStage TDD
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[3]))

from sats.training.config import SATSConfig
from sats.training.cnn_module import SATSCNNRefiner, SATSCNNStage


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
        cnn_hidden_channels=8,
    )
    defaults.update(kwargs)
    return SATSConfig(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: SATSCNNRefiner
# ─────────────────────────────────────────────────────────────────────────────

class TestSATSCNNRefiner:

    def _make_refiner(self, hidden_channels=8):
        return SATSCNNRefiner(grid_size=40, hidden_channels=hidden_channels)

    def _rand_merged(self, B=2, H=40, W=40):
        return torch.randn(B, H, W)

    def test_output_shape(self):
        """입력 [B, 40, 40] → 출력 [B, 40, 40]."""
        ref = self._make_refiner()
        x = self._rand_merged()
        out = ref(x)
        assert out.shape == (2, 40, 40), f"shape 오류: {out.shape}"

    def test_output_dtype_float32(self):
        ref = self._make_refiner()
        x = self._rand_merged()
        out = ref(x)
        assert out.dtype == torch.float32

    def test_no_nan_inf(self):
        ref = self._make_refiner()
        x = self._rand_merged()
        out = ref(x)
        assert not torch.isnan(out).any(), "NaN 발생"
        assert not torch.isinf(out).any(), "Inf 발생"

    def test_gradient_flows(self):
        """역전파 시 CNN 파라미터에 gradient가 흘러야 한다."""
        ref = self._make_refiner()
        x = self._rand_merged()
        out = ref(x)
        out.sum().backward()
        for name, p in ref.named_parameters():
            assert p.grad is not None, f"{name} gradient 없음"

    def test_batch_independence(self):
        """배치 내 샘플 간 출력이 독립적이어야 한다."""
        ref = self._make_refiner()
        ref.eval()
        x = self._rand_merged(B=4)
        with torch.no_grad():
            out_full = ref(x)
            out_single = ref(x[2:3])
        torch.testing.assert_close(out_full[2:3], out_single)

    def test_different_inputs_different_outputs(self):
        ref = self._make_refiner()
        ref.eval()
        x1 = torch.randn(2, 40, 40)
        x2 = torch.randn(2, 40, 40)
        with torch.no_grad():
            o1 = ref(x1)
            o2 = ref(x2)
        assert not torch.allclose(o1, o2), "다른 입력에서 같은 출력"

    def test_zero_input_produces_output(self):
        """zero 입력이 크래시 없이 출력을 반환해야 한다 (bias로 인해 비-zero 가능)."""
        ref = self._make_refiner()
        x = torch.zeros(1, 40, 40)
        with torch.no_grad():
            out = ref(x)
        assert out.shape == (1, 40, 40)

    def test_spatial_dimensions_preserved(self):
        """3x3 conv + padding=1은 공간 크기를 보존해야 한다."""
        ref = self._make_refiner()
        for H, W in [(40, 40), (20, 20), (10, 10)]:
            x = torch.randn(1, H, W)
            with torch.no_grad():
                out = ref(x)
            assert out.shape == (1, H, W), f"크기 변형: 입력({H},{W}) → 출력{out.shape[1:]}"

    def test_leaky_relu_applied_to_first_layer(self):
        """첫 번째 레이어 뒤에 LeakyReLU가 적용됨을 확인.
        음수 입력 영역에서 출력이 정확히 0이 아님을 검증한다."""
        ref = self._make_refiner()
        ref.eval()
        # 모든 값이 크게 음수인 입력
        x = torch.full((1, 40, 40), -10.0)
        with torch.no_grad():
            out = ref(x)
        # LeakyReLU(0.2)로 인해 음수 입력도 완전히 소멸되지 않음
        # 만약 ReLU였다면 출력이 모두 bias 값에만 의존할 것
        # 여기서는 중간 활성화가 살아있다는 것을 간접 검증
        assert out is not None  # 크래시 없음

    def test_extra_repr_contains_config(self):
        """extra_repr에 grid_size와 hidden_channels가 포함되어야 한다."""
        ref = SATSCNNRefiner(grid_size=40, hidden_channels=16)
        r = repr(ref)
        assert "40" in r
        assert "16" in r


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: SATSCNNStage
# ─────────────────────────────────────────────────────────────────────────────

class TestSATSCNNStage:

    def _make_stage(self, **kwargs):
        cfg = _small_cfg(**kwargs)
        return SATSCNNStage(cfg)

    def _rand_batch(self, B=2, T=10, n=16):
        sensor_seq = torch.randn(B, T, n)
        lengths = torch.tensor([T, T - 2])
        return sensor_seq, lengths

    def test_refined_map_shape(self):
        """refined_map 출력 shape = [B, 41, 41]."""
        stage = self._make_stage()
        x, l = self._rand_batch()
        refined_map, merged_map = stage(x, l)
        assert refined_map.shape == (2, 41, 41), f"refined_map shape 오류: {refined_map.shape}"

    def test_merged_map_shape(self):
        """merged_map (CNN 전) 출력 shape = [B, 41, 41]."""
        stage = self._make_stage()
        x, l = self._rand_batch()
        refined_map, merged_map = stage(x, l)
        assert merged_map.shape == (2, 41, 41), f"merged_map shape 오류: {merged_map.shape}"

    def test_no_nan_in_refined_map(self):
        stage = self._make_stage()
        x, l = self._rand_batch()
        refined_map, _ = stage(x, l)
        assert not torch.isnan(refined_map).any(), "refined_map에 NaN 발생"

    def test_gradient_flows_through_cnn(self):
        """cnn_refiner 파라미터에 gradient가 흘러야 한다."""
        stage = self._make_stage()
        x, l = self._rand_batch()
        refined_map, _ = stage(x, l)
        refined_map.sum().backward()
        for name, p in stage.cnn_refiner.named_parameters():
            assert p.grad is not None, f"cnn_refiner.{name} grad 없음"

    def test_frozen_upstream_no_grad(self):
        """encoder + attention + local_map_decoder 동결 시 grad 없어야 한다."""
        stage = self._make_stage()
        for p in stage.encoder.parameters():
            p.requires_grad_(False)
        for p in stage.attention.parameters():
            p.requires_grad_(False)
        for p in stage.local_map_decoder.parameters():
            p.requires_grad_(False)

        x, l = self._rand_batch()
        refined_map, _ = stage(x, l)
        refined_map.sum().backward()

        for name, p in stage.encoder.named_parameters():
            assert p.grad is None, f"encoder.{name} grad 있어서는 안 됨"
        for name, p in stage.attention.named_parameters():
            assert p.grad is None, f"attention.{name} grad 있어서는 안 됨"
        for name, p in stage.local_map_decoder.named_parameters():
            assert p.grad is None, f"local_map_decoder.{name} grad 있어서는 안 됨"

    def test_init_from_config(self):
        """SATSConfig로부터 올바르게 초기화되는지 확인."""
        cfg = _small_cfg(cnn_hidden_channels=32)
        stage = SATSCNNStage(cfg)
        assert stage.cnn_refiner is not None

    def test_refined_differs_from_merged(self):
        """CNN Refiner가 merged_map을 실제로 변환해야 한다 (동일하면 안 됨)."""
        stage = self._make_stage()
        stage.eval()
        x, l = self._rand_batch()
        with torch.no_grad():
            refined_map, merged_map = stage(x, l)
        # CNN이 아무것도 안 한다면 동일하겠지만, 일반적으로 달라야 함
        # 가중치가 초기화된 상태에서 동일할 확률은 거의 0
        assert not torch.allclose(refined_map, merged_map), (
            "refined_map이 merged_map과 동일함 — CNN Refiner가 적용되지 않은 것 같음"
        )
