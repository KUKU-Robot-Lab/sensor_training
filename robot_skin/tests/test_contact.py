import numpy as np
import pytest

from robot_skin.contact import (
    ContactLevel, OrdinalQuantizer, SatState, SaturationFSM, contact_mask, finger_of,
    point_segment_distance, residual, self_touch_labels,
)


def test_residual_and_mask_use_press_sign():
    r = residual(np.array([-10.0, 2.0, -2.0]), np.array([-1.0, 1.0, 0.0]))
    np.testing.assert_allclose(r, [-9.0, 1.0, -2.0])
    np.testing.assert_array_equal(contact_mask(r, 3.0), [True, False, False])
    np.testing.assert_array_equal(contact_mask(r, 3.0, trusted=[False, True, True]), [False, False, False])


def test_ordinal_levels():
    q = OrdinalQuantizer(3.0, 15.0)
    lv = q(np.array([0.0, -4.0, -20.0, -20.0, +20.0]), saturated=np.array([0, 0, 0, 1, 0], bool))
    np.testing.assert_array_equal(lv, [ContactLevel.NONE, ContactLevel.WEAK, ContactLevel.STRONG,
                                       ContactLevel.SATURATED, ContactLevel.NONE])
    oh = OrdinalQuantizer.one_hot(lv)
    assert oh.shape == (5, 4) and np.all(oh.sum(-1) == 1)
    with pytest.raises(ValueError):
        OrdinalQuantizer(5.0, 5.0)


def test_fsm_release_after_stable():
    f = SaturationFSM(1, ok_pct=1.5, ok_sec=0.1, max_recover_s=10.0)
    dt = 0.01
    assert f.step([-95.0], [True], dt)[0] == SatState.SATURATED
    assert f.step([0.5], [False], dt)[0] == SatState.RECOVERING
    states = [f.step([0.5], [False], dt)[0] for _ in range(12)]
    assert states[-1] == SatState.OK and SatState.RECOVERING in states
    assert f.rezero_count[0] == 0 and f.offset[0] == 0.0


def test_fsm_rezero_on_timeout_and_resaturation():
    f = SaturationFSM(2, ok_pct=1.5, ok_sec=0.5, max_recover_s=1.0)
    dt = 0.1
    f.step([-95.0, 0.0], [True, False], dt)
    for _ in range(3):
        f.step([8.0, 0.0], [False, False], dt)  # settled at a new level, outside ok band
    assert f.state[0] == SatState.RECOVERING and f.trusted.tolist() == [False, True]
    f.step([-95.0, 0.0], [True, False], dt)     # re-saturates during recovery
    assert f.state[0] == SatState.SATURATED
    states = f.run(np.tile([[8.0, 0.0]], (15, 1)), np.zeros((15, 2), bool), dt)
    assert states[-1, 0] == SatState.OK and f.rezero_count[0] == 1
    assert f.offset[0] == pytest.approx(8.0)
    np.testing.assert_allclose(f.corrected([8.0, 0.0]), [0.0, 0.0])


def test_point_segment_distance():
    a, b = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    np.testing.assert_allclose(point_segment_distance(np.array([[0.5, 1, 0], [2, 0, 0], [-1, 0, 0]]), a, b),
                               [1.0, 1.0, 1.0])
    assert point_segment_distance(np.array([1.0, 1, 0]), a, a) == pytest.approx(np.sqrt(2))


def test_finger_of():
    assert finger_of("index3") == "index" and finger_of("thumb_distal_link") == "thumb"
    assert finger_of("palm_link") == "palm" and finger_of("wrist") == "palm"


def test_self_touch_labels_excludes_own_finger():
    seg_names = ["index3", "thumb3"]
    # index tip taxel sits on its own segment; thumb segment approaches over time
    taxel_pos = np.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    p0 = np.array([[[-0.01, 0, 0], [0.0, 0.05, 0]], [[-0.01, 0, 0], [0.0, 0.009, 0]]])
    p1 = np.array([[[0.01, 0, 0], [0.02, 0.05, 0]], [[0.01, 0, 0], [0.02, 0.009, 0]]])
    lab = self_touch_labels(taxel_pos, ["index3"], p0, p1, 0.005, seg_names, margin=0.004)
    np.testing.assert_array_equal(lab, [[False], [True]])
