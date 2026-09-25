"""
tests/test_train_local_map.py

train_local_map.py TDD
"""
import sys
import math
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[3]))

from sats.training.config import SATSConfig
from sats.training.attention_module import SATSAttentionStage
from sats.training.local_map_module import SATSLocalMapStage
from sats.training.train_local_map import (
    load_attention_weights,
    train_epoch,
    val_epoch,
)


# ─────────────────────────────────────────────────────────────────────────────
# 공용 픽스처
# ─────────────────────────────────────────────────────────────────────────────

def _small_cfg(**kwargs):
    defaults = dict(
        hidden_dim=16,
        num_layers=1,
        attn_dim=16,
        dropout=0.0,
        grid_size=40,
        n_sensors=16,
        local_map_size=15,
        sensor_spacing_mm=6.5,
        clip_grad=1.0,
    )
    defaults.update(kwargs)
    return SATSConfig(**defaults)


def _make_local_map_stage(cfg=None):
    cfg = cfg or _small_cfg()
    return SATSLocalMapStage(cfg)


def _save_attn_ckpt(cfg, path: Path):
    """SATSAttentionStage 가중치를 체크포인트 형식으로 저장한다."""
    attn_stage = SATSAttentionStage(cfg)
    torch.save(
        {
            "epoch": 1,
            "model": attn_stage.state_dict(),
            "metrics": {},
        },
        path,
    )
    return attn_stage


def _fake_loader(B=2, T=8, n=16, n_batches=3):
    """(sensor_seq, gt_batch, lengths) 튜플을 n_batches개 반환하는 이터러블."""
    data = []
    for _ in range(n_batches):
        sensor_seq = torch.randn(B, T, n)
        gt_batch   = torch.rand(B, T, 40, 40)
        lengths    = torch.tensor([T, T - 1])
        data.append((sensor_seq, gt_batch, lengths))
    return data


# ─────────────────────────────────────────────────────────────────────────────
# load_attention_weights 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAttentionWeights:
    def test_encoder_weights_loaded_correctly(self):
        """체크포인트의 encoder 가중치가 stage.encoder에 정확히 로드된다."""
        cfg = _small_cfg()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            attn_stage = _save_attn_ckpt(cfg, Path(f.name))
            lm_stage = _make_local_map_stage(cfg)

            load_attention_weights(f.name, lm_stage, freeze=False)

            for (n1, p1), (n2, p2) in zip(
                attn_stage.encoder.named_parameters(),
                lm_stage.encoder.named_parameters(),
            ):
                assert n1 == n2
                torch.testing.assert_close(p1, p2), f"encoder.{n1} 가중치 불일치"

    def test_attention_weights_loaded_correctly(self):
        """체크포인트의 attention 가중치가 stage.attention에 정확히 로드된다."""
        cfg = _small_cfg()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            attn_stage = _save_attn_ckpt(cfg, Path(f.name))
            lm_stage = _make_local_map_stage(cfg)

            load_attention_weights(f.name, lm_stage, freeze=False)

            for (n1, p1), (n2, p2) in zip(
                attn_stage.attention.named_parameters(),
                lm_stage.attention.named_parameters(),
            ):
                assert n1 == n2
                torch.testing.assert_close(p1, p2), f"attention.{n1} 가중치 불일치"

    def test_freeze_true_disables_encoder_attention_grad(self):
        """freeze=True이면 encoder & attention 파라미터가 requires_grad=False."""
        cfg = _small_cfg()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            _save_attn_ckpt(cfg, Path(f.name))
            stage = _make_local_map_stage(cfg)
            load_attention_weights(f.name, stage, freeze=True)

            for name, p in stage.encoder.named_parameters():
                assert not p.requires_grad, f"encoder.{name} requires_grad가 True"
            for name, p in stage.attention.named_parameters():
                assert not p.requires_grad, f"attention.{name} requires_grad가 True"

    def test_local_map_decoder_unaffected(self):
        """로드 후 local_map_decoder 파라미터는 requires_grad=True를 유지해야 한다."""
        cfg = _small_cfg()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            _save_attn_ckpt(cfg, Path(f.name))
            stage = _make_local_map_stage(cfg)
            load_attention_weights(f.name, stage, freeze=True)

            for name, p in stage.local_map_decoder.named_parameters():
                assert p.requires_grad, f"local_map_decoder.{name} grad 꺼짐"


# ─────────────────────────────────────────────────────────────────────────────
# train_epoch 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainEpoch:
    def _setup(self):
        cfg = _small_cfg()
        stage = _make_local_map_stage(cfg)
        # encoder + attention 동결
        for p in stage.encoder.parameters():
            p.requires_grad_(False)
        for p in stage.attention.parameters():
            p.requires_grad_(False)
        trainable = [p for p in stage.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-3)
        loader = _fake_loader()
        return stage, optimizer, cfg, loader

    def test_returns_loss_key(self):
        stage, opt, cfg, loader = self._setup()
        metrics = train_epoch(stage, loader, opt, "cpu", cfg)
        assert "loss" in metrics, "metrics에 'loss' 키가 없음"

    def test_loss_is_finite(self):
        stage, opt, cfg, loader = self._setup()
        metrics = train_epoch(stage, loader, opt, "cpu", cfg)
        assert math.isfinite(metrics["loss"]), "loss가 NaN 또는 Inf"

    def test_loss_is_non_negative(self):
        stage, opt, cfg, loader = self._setup()
        metrics = train_epoch(stage, loader, opt, "cpu", cfg)
        assert metrics["loss"] >= 0.0

    def test_local_map_decoder_params_updated(self):
        """train_epoch 이후 local_map_decoder 파라미터가 변해야 한다."""
        stage, opt, cfg, loader = self._setup()
        before = {
            n: p.clone()
            for n, p in stage.local_map_decoder.named_parameters()
        }
        train_epoch(stage, loader, opt, "cpu", cfg)
        changed = any(
            not torch.equal(before[n], p)
            for n, p in stage.local_map_decoder.named_parameters()
        )
        assert changed, "local_map_decoder 파라미터가 업데이트되지 않음"

    def test_frozen_encoder_attention_unchanged(self):
        """동결된 encoder + attention 파라미터는 변하지 않아야 한다."""
        stage, opt, cfg, loader = self._setup()
        enc_before  = {n: p.clone() for n, p in stage.encoder.named_parameters()}
        attn_before = {n: p.clone() for n, p in stage.attention.named_parameters()}

        train_epoch(stage, loader, opt, "cpu", cfg)

        for n, p in stage.encoder.named_parameters():
            assert torch.equal(enc_before[n], p), f"encoder.{n} 값이 변했음"
        for n, p in stage.attention.named_parameters():
            assert torch.equal(attn_before[n], p), f"attention.{n} 값이 변했음"


# ─────────────────────────────────────────────────────────────────────────────
# val_epoch 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestValEpoch:
    def _setup(self):
        cfg = _small_cfg()
        stage = _make_local_map_stage(cfg)
        loader = _fake_loader()
        return stage, loader

    def test_returns_mse_rmse_keys(self):
        stage, loader = self._setup()
        metrics = val_epoch(stage, loader, "cpu")
        assert "mse" in metrics
        assert "rmse" in metrics

    def test_rmse_equals_sqrt_mse(self):
        stage, loader = self._setup()
        metrics = val_epoch(stage, loader, "cpu")
        assert math.isclose(metrics["rmse"], math.sqrt(metrics["mse"]), rel_tol=1e-5)

    def test_metrics_finite(self):
        stage, loader = self._setup()
        metrics = val_epoch(stage, loader, "cpu")
        assert math.isfinite(metrics["mse"])
        assert math.isfinite(metrics["rmse"])

    def test_no_grad_during_val(self):
        """val_epoch 후 파라미터에 gradient가 없어야 한다."""
        stage, loader = self._setup()
        val_epoch(stage, loader, "cpu")
        for name, p in stage.named_parameters():
            assert p.grad is None, f"{name}에 grad가 남아있음"
