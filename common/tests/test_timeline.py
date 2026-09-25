import numpy as np
import pytest

from common.timeline import MASTER_HZ, Stream, align_streams, master_clock, resample


def test_master_clock_200hz():
    t = master_clock(1.0, 2.0)
    assert MASTER_HZ == 200.0
    assert t.shape == (201,)
    np.testing.assert_allclose(np.diff(t), 1 / 200.0)


def test_linear_resample_is_exact_on_ramp():
    s = Stream(t=np.array([0.0, 0.1, 0.3]), values=np.array([[0.0], [1.0], [3.0]]))
    v, valid = resample(s, np.array([0.05, 0.2, 0.3, 0.5]))
    np.testing.assert_allclose(v[:3, 0], [0.5, 2.0, 3.0])
    np.testing.assert_array_equal(valid, [True, True, True, False])


def test_zoh_holds_last_value():
    s = Stream(t=np.array([0.0, 1.0]), values=np.array([10, 20]), method="zoh")
    v, _ = resample(s, np.array([0.0, 0.5, 0.999, 1.0, 1.5]))
    np.testing.assert_array_equal(v, [10, 10, 10, 20, 20])


def test_sanitize_sorts_and_dedups():
    s = Stream(t=np.array([0.2, 0.0, 0.1, 0.1]), values=np.array([2.0, 0.0, 9.0, 1.0])).sanitize()
    np.testing.assert_array_equal(s.t, [0.0, 0.1, 0.2])
    np.testing.assert_array_equal(s.values, [0.0, 1.0, 2.0])


def test_align_intersection_and_union():
    pres = Stream(np.linspace(0.0, 2.0, 401), np.random.default_rng(0).normal(size=(401, 16)))
    imu = Stream(np.linspace(0.5, 3.0, 251), np.zeros((251, 7, 4)))
    cam = Stream(np.arange(0.0, 2.5, 1 / 30), np.arange(75), method="zoh")
    tm, vals, valid = align_streams({"pressure": pres, "imu": imu, "cam": cam})
    assert tm[0] == pytest.approx(0.5) and tm[-1] <= 2.0 + 1e-9
    assert vals["imu"].shape == (tm.shape[0], 7, 4)
    assert vals["pressure"].shape == (tm.shape[0], 16)
    assert all(v.all() for v in valid.values())
    tm_u, _, valid_u = align_streams({"pressure": pres, "imu": imu}, span="union")
    assert tm_u[0] == 0.0 and not valid_u["imu"][0] and not valid_u["pressure"][-1]


def test_align_rejects_disjoint():
    a = Stream(np.array([0.0, 1.0]), np.zeros(2))
    b = Stream(np.array([2.0, 3.0]), np.zeros(2))
    with pytest.raises(ValueError):
        align_streams({"a": a, "b": b})


def test_stream_shape_validation():
    with pytest.raises(ValueError):
        Stream(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        Stream(np.zeros(3), np.zeros(3), method="cubic")
