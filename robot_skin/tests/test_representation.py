import pytest
import torch

from robot_skin.representation import MaskedTaxelPretrainer, TaxelTokenizer, random_taxel_mask


def test_tokenizer_shapes_and_pose_sensitivity():
    tok = TaxelTokenizer(value_dim=4, d_model=32, n_fourier=4)
    v = torch.randn(2, 9, 4)
    pos, nrm = torch.randn(2, 9, 3) * 0.05, torch.nn.functional.normalize(torch.randn(2, 9, 3), dim=-1)
    out = tok(v, pos, nrm)
    assert out.shape == (2, 9, 32)
    assert not torch.allclose(out, tok(v, pos + 0.01, nrm))
    # taxel-count agnostic without id embedding
    assert tok(torch.randn(1, 16, 4), torch.zeros(1, 16, 3), torch.zeros(1, 16, 3)).shape == (1, 16, 32)


def test_mask_replaces_value_only():
    torch.manual_seed(0)
    tok = TaxelTokenizer(value_dim=2, d_model=16, n_taxels=5)
    pos, nrm = torch.randn(1, 5, 3), torch.randn(1, 5, 3)
    m = torch.tensor([[True, False, False, False, False]])
    a = tok(torch.randn(1, 5, 2), pos, nrm, mask=m)
    b = tok(torch.randn(1, 5, 2), pos, nrm, mask=m)
    assert torch.allclose(a[:, 0], b[:, 0])          # masked taxel ignores its value
    assert not torch.allclose(a[:, 1], b[:, 1])


def test_random_taxel_mask():
    g = torch.Generator().manual_seed(0)
    m = random_taxel_mask(4, 10, 0.3, generator=g)
    assert m.shape == (4, 10) and (m.sum(1) == 3).all()
    assert random_taxel_mask(2, 10, 0.0).sum() == 0
    with pytest.raises(ValueError):
        random_taxel_mask(1, 10, 1.0)
    with pytest.raises(NotImplementedError):
        MaskedTaxelPretrainer()
