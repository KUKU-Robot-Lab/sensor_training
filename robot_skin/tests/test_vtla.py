import pytest
import torch

from robot_skin.vtla import ContactGate, TactileTokenAdapter


@pytest.mark.parametrize("gate", ["hard", "soft"])
def test_adapter_shapes_and_no_contact_is_zero(gate):
    torch.manual_seed(0)
    ad = TactileTokenAdapter(d_in=32, d_out=48, n_query=4, gate=gate)
    x = torch.randn(3, 9, 32)
    contact = torch.zeros(3, 9, dtype=torch.bool)
    contact[1, 2] = True
    out = ad(x, contact)
    assert out.shape == (3, 4, 48)
    assert torch.count_nonzero(out[0]) == 0 and torch.count_nonzero(out[2]) == 0
    assert torch.count_nonzero(out[1]) > 0
    # taxel-count agnostic
    assert ad(torch.randn(1, 16, 32), torch.ones(1, 16)).shape == (1, 4, 48)


def test_soft_gate_grows_with_contact_fraction_and_is_trainable():
    g = ContactGate("soft")
    tok = torch.ones(2, 1, 1)
    c = torch.zeros(2, 10)
    c[0, :1] = 1
    c[1, :8] = 1
    out = g(tok, c)
    assert 0 < out[0].item() < out[1].item() <= 1
    out.sum().backward()
    assert g.w.grad is not None
    with pytest.raises(ValueError):
        ContactGate("fuzzy")
