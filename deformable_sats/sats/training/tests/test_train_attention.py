"""
tests/test_train_attention.py

train_attention.py TDD
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
from sats.training.lstm_module import SATSLSTMStage
from sats.training.train_attention import (
    load_lstm_encoder,
    train_epoch,
    val_epoch,
)


# ─────────────────────────────────────────────────────────────────────────────
# 공용 픽스처
# ─────────────────────────────────────────────────────────────────────────────

def _small_cfg(**kwargs):
    """테스트용 최소 설정."""
    defaults = dict(
        hidden_dim=16,
        num_layers=1,
        attn_dim=16,
        dropout=0.0,
        grid_size=40,
        n_sensors=16,
        clip_grad=1.0,
    )
    defaults.update(kwargs)
    return SATSConfig(**defaults)


def _make_attention_stage(cfg=None):
    cfg = cfg or _small_cfg()
    return SATSAttentionStage(cfg)


def _save_lstm_ckpt(cfg, path: Path):
    """SATSLSTMStage 가중치를 체크포인트 형식으로 저장한다."""
    lstm_stage = SATSLSTMStage(cfg)
    torch.save(
        {
            "epoch": 1,
            "model": lstm_stage.state_dict(),
            "metrics": {},
        },
        path,
    )
    return lstm_stage


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
# load_lstm_encoder 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadLSTMEncoder:
    def test_weights_loaded_correctly(self):
        """체크포인트의 LSTM 인코더 가중치가 stage.encoder에 정확히 로드된다."""
        cfg = _small_cfg()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            lstm_stage = _save_lstm_ckpt(cfg, Path(f.name))
            attn_stage = _make_attention_stage(cfg)

            load_lstm_encoder(f.name, attn_stage, freeze=False)

            # encoder 파라미터가 일치해야 한다
            for (n1, p1), (n2, p2) in zip(
                lstm_stage.encoder.named_parameters(),
                attn_stage.encoder.named_parameters(),
            ):
                assert n1 == n2
                torch.testing.assert_close(p1, p2), f"{n1} 가중치 불일치"

    def test_freeze_true_disables_grad(self):
        """freeze=True이면 encoder 파라미터가 requires_grad=False."""
        cfg = _small_cfg()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            _save_lstm_ckpt(cfg, Path(f.name))
            stage = _make_attention_stage(cfg)
            load_lstm_encoder(f.name, stage, freeze=True)

            for name, p in stage.encoder.named_parameters():
                assert not p.requires_grad, f"{name} requires_grad가 True"

    def test_freeze_false_keeps_grad(self):
        """freeze=False이면 encoder 파라미터가 requires_grad=True."""
        cfg = _small_cfg()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            _save_lstm_ckpt(cfg, Path(f.name))
            stage = _make_attention_stage(cfg)
            load_lstm_encoder(f.name, stage, freeze=False)

            for name, p in stage.encoder.named_parameters():
                assert p.requires_grad, f"{name} requires_grad가 False"

    def test_attention_params_unaffected(self):
        """LSTM 로드 후 attention / decoder 파라미터는 requires_grad=True."""
        cfg = _small_cfg()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            _save_lstm_ckpt(cfg, Path(f.name))
            stage = _make_attention_stage(cfg)
            load_lstm_encoder(f.name, stage, freeze=True)

            for name, p in stage.attention.named_parameters():
                assert p.requires_grad, f"attention.{name} grad 꺼짐"
            for name, p in stage.decoder.named_parameters():
                assert p.requires_grad, f"decoder.{name} grad 꺼짐"


# ─────────────────────────────────────────────────────────────────────────────
# train_epoch 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainEpoch:
    def _setup(self):
        cfg = _small_cfg()
        stage = _make_attention_stage(cfg)
        # encoder 동결
        for p in stage.encoder.parameters():
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

    def test_loss_is_positive(self):
        stage, opt, cfg, loader = self._setup()
        metrics = train_epoch(stage, loader, opt, "cpu", cfg)
        assert metrics["loss"] >= 0.0

    def test_attention_params_updated(self):
        """train_epoch 이후 attention 파라미터가 변해야 한다."""
        stage, opt, cfg, loader = self._setup()
        before = {n: p.clone() for n, p in stage.attention.named_parameters()}
        train_epoch(stage, loader, opt, "cpu", cfg)
        changed = any(
            not torch.equal(before[n], p)
            for n, p in stage.attention.named_parameters()
        )
        assert changed, "attention 파라미터가 업데이트되지 않음"

    def test_encoder_params_unchanged_when_frozen(self):
        """frozen encoder 파라미터는 train_epoch 후에도 변하지 않아야 한다."""
        stage, opt, cfg, loader = self._setup()
        before = {n: p.clone() for n, p in stage.encoder.named_parameters()}
        train_epoch(stage, loader, opt, "cpu", cfg)
        for n, p in stage.encoder.named_parameters():
            assert torch.equal(before[n], p), f"encoder.{n} 값이 변했음"


# ─────────────────────────────────────────────────────────────────────────────
# val_epoch 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestValEpoch:
    def _setup(self):
        cfg = _small_cfg()
        stage = _make_attention_stage(cfg)
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
