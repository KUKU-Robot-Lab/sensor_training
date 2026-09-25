import numpy as np
import pytest

from robot_skin.contact import SatState, SaturationFSM
from robot_skin.eval import (
    auroc, hallucination_rate, motion_contact_separability, saturation_recovery_times,
)


def test_hallucination_rate_frame_and_taxel():
    g = np.array([[0, 0], [0, 0], [1, 0], [0, 0]], bool)
    p = np.array([[1, 0], [0, 0], [1, 0], [0, 1]], bool)
    assert hallucination_rate(p, g) == pytest.approx(2 / 3)
    assert hallucination_rate(p, g, level="taxel") == pytest.approx(2 / 7)
    assert np.isnan(hallucination_rate(p[2:3], g[2:3]))
    with pytest.raises(ValueError):
        hallucination_rate(p, g[:2])


def test_auroc_known_values():
    assert auroc([2, 3], [0, 1]) == 1.0
    assert auroc([0, 1], [2, 3]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert auroc([1, 2, 3], [0, 2]) == pytest.approx(4.5 / 6)


def test_motion_contact_separability_improves_after_baseline():
    rng = np.random.default_rng(0)
    T, N = 400, 4
    moving = np.ones(T, bool)
    contact = np.zeros((T, N), bool)
    contact[200:260, 0] = True
    motion = -8.0 * np.abs(np.sin(np.linspace(0, 10, T)))[:, None] * np.ones(N)   # motion artefact
    obs = motion + np.where(contact, -4.0, 0.0) + rng.normal(0, 0.3, (T, N))
    raw_sep = motion_contact_separability(obs, contact, moving)
    res_sep = motion_contact_separability(obs - motion, contact, moving)
    assert res_sep["auroc"] > 0.99 and res_sep["auroc"] > raw_sep["auroc"]
    assert res_sep["dprime"] > raw_sep["dprime"]


def test_saturation_recovery_times_from_fsm():
    dt = 0.01
    f = SaturationFSM(1, ok_sec=0.05)
    sat = np.zeros((40, 1), bool)
    sat[5:10] = True
    states = f.run(np.zeros((40, 1)), sat, dt)
    rt = saturation_recovery_times(states, dt)
    assert rt.shape == (1,) and rt[0] == pytest.approx(0.05)  # 5 RECOVERING ticks = ok_sec
    open_ep = np.array([[SatState.OK], [SatState.SATURATED], [SatState.RECOVERING]])
    assert saturation_recovery_times(open_ep, dt).size == 0
