"""Tests for optional indenter-size (diameter) conditioning of the E2E model.

Motivation: the model input is only the 16-taxel window + lengths, so it cannot
tell a 5 mm from a 10 mm indenter given similar sensor signals and outputs a
size-blended pressure peak (d10 low-force over-prediction). Feeding the known
indenter diameter as a FiLM condition lets it produce size-appropriate peaks.

Design guarantees under test:
- Off by default → forward ignores any size argument (backward compatible).
- On + identity init → output equals the no-size output (safe warm start).
- On + perturbed film → different diameters yield different outputs.
"""
import torch

from sats.training.cnn_module import SATSCNNStage
from sats.training.config import SATSConfig


def _dummy_batch(cfg: SATSConfig, batch: int = 4, t: int = 10):
    torch.manual_seed(0)
    seq = torch.randn(batch, t, cfg.n_sensors)
    lengths = torch.full((batch,), t, dtype=torch.long)
    size = torch.tensor([5.0, 10.0, 5.0, 10.0])[:batch]
    return seq, lengths, size


def test_size_input_off_by_default_ignores_size():
    cfg = SATSConfig()
    model = SATSCNNStage(cfg).eval()
    seq, lengths, size = _dummy_batch(cfg)
    with torch.no_grad():
        out_none, _ = model(seq, lengths)
        out_size, _ = model(seq, lengths, size)
    assert torch.allclose(out_none, out_size, atol=1e-6)


def test_size_input_identity_init_matches_no_size():
    cfg = SATSConfig(use_indenter_size_input=True)
    model = SATSCNNStage(cfg).eval()
    seq, lengths, size = _dummy_batch(cfg)
    with torch.no_grad():
        out_none, _ = model(seq, lengths)          # size=None → no FiLM
        out_size, _ = model(seq, lengths, size)    # identity FiLM → same
    assert torch.allclose(out_none, out_size, atol=1e-6)


def test_size_input_changes_output_when_film_active():
    cfg = SATSConfig(use_indenter_size_input=True)
    model = SATSCNNStage(cfg).eval()
    # break identity init so the condition actually modulates features
    with torch.no_grad():
        for p in model.size_film.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    seq, lengths, _ = _dummy_batch(cfg)
    d5 = torch.full((seq.shape[0],), 5.0)
    d10 = torch.full((seq.shape[0],), 10.0)
    with torch.no_grad():
        out_d5, _ = model(seq, lengths, d5)
        out_d10, _ = model(seq, lengths, d10)
    assert not torch.allclose(out_d5, out_d10, atol=1e-5)
