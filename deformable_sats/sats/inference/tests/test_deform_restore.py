"""데모용 변형 복원기 래퍼 — 마스킹 일관성과 실패 시 안전한 동작."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from sats.bending.baseline_restorer import BaselineRestorer
from sats.bending.config import BendingConfig
from sats.inference.deform_restore import DeformRestorer, try_load


def _make_ckpt(tmp_path, *, dead=(7, 11, 15), k=8, mode="latent"):
    cfg = replace(BendingConfig(), restorer_mode="latent", latent_dim=k)
    m = BaselineRestorer(cfg)
    p = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "latent_dim": k, "restorer_mode": mode,
                "dead_channels_1based": list(dead), "sensor": "v7"}, p)
    return p


def test_restores_window_shape_preserved(tmp_path):
    r = DeformRestorer(_make_ckpt(tmp_path))
    win = np.random.default_rng(0).normal(0, 3, (r.window_size, 16)).astype(np.float32)
    assert r.restore(win).shape == win.shape


def test_batch_input_supported(tmp_path):
    r = DeformRestorer(_make_ckpt(tmp_path))
    win = np.zeros((4, r.window_size, 16), np.float32)
    assert r.restore(win).shape == (4, r.window_size, 16)


def test_dead_channels_masked_from_checkpoint(tmp_path):
    """★추론 마스킹은 체크포인트를 따라야 한다 — 학습과 달라지면 오프셋 추정이 어긋난다."""
    r = DeformRestorer(_make_ckpt(tmp_path, dead=(7, 11, 15)))
    assert r.dead == (6, 10, 14)
    win = np.full((r.window_size, 16), 5.0, np.float32)
    assert np.all(r.mask(win)[:, [6, 10, 14]] == 0.0)
    assert np.all(r.mask(win)[:, 0] == 5.0)


def test_mask_does_not_mutate_input(tmp_path):
    r = DeformRestorer(_make_ckpt(tmp_path))
    win = np.full((r.window_size, 16), 5.0, np.float32)
    r.restore(win)
    assert np.all(win == 5.0)                    # 원본 불변


def test_no_dead_channels_is_fine(tmp_path):
    r = DeformRestorer(_make_ckpt(tmp_path, dead=()))
    assert r.dead == ()
    assert "없음" in r.describe()


def test_missing_checkpoint_raises_with_guidance(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_deform_restorer"):
        DeformRestorer(tmp_path / "nope.pt")


def test_wrong_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="latent"):
        DeformRestorer(_make_ckpt(tmp_path, mode="deg_only"))


def test_try_load_never_raises(tmp_path):
    """데모는 복원기가 없어도 떠야 한다 — 예외 대신 상태 문구."""
    r, msg = try_load(tmp_path / "nope.pt")
    assert r is None and "복원기 없음" in msg
    r, msg = try_load(_make_ckpt(tmp_path))
    assert r is not None and "latent_dim" in msg
