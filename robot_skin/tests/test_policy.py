import numpy as np
import pytest

from robot_skin.policy import OBS_MODES, ObservationBuilder, tactile_features
from robot_skin.policy import train_rl


@pytest.mark.parametrize("mode,k", [("full", 2), ("ordinal", 4), ("binary", 1), ("none", 0)])
def test_builder_dims(mode, k):
    b = ObservationBuilder(mode, n_taxels=9, proprio_dim=16)
    obs = b(np.zeros((5, 16)), np.zeros((5, 9)), np.zeros((5, 9), bool))
    assert obs.shape == (5, 16 + 9 * k) == (5, b.dim)
    assert obs.dtype == np.float32


def test_feature_semantics():
    b = ObservationBuilder("ordinal", 3, 0)
    r = np.array([0.0, -5.0, -30.0])
    sat = np.array([False, False, True])
    np.testing.assert_array_equal(tactile_features("binary", r, sat, b.quantizer).ravel(), [0, 1, 0])
    np.testing.assert_array_equal(b(np.zeros(0), r, sat).reshape(3, 4).argmax(-1), [0, 1, 3])
    full = tactile_features("full", r, sat, b.quantizer, scale_pct=10.0).reshape(3, 2)
    np.testing.assert_allclose(full, [[0, 0], [0.5, 0], [0, 1]])


def test_modes_and_stub():
    assert OBS_MODES == ("full", "ordinal", "binary", "none")
    with pytest.raises(ValueError):
        ObservationBuilder("depth", 3, 0)
    with pytest.raises(NotImplementedError):
        train_rl.main(["--obs-mode", "binary"])
