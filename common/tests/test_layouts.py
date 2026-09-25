import numpy as np
import pytest
import yaml

from common.layouts import (
    MANO_SEGMENTS, available_layouts, grid_layout, layout_from_dict, load_layout,
)


def test_builtins_present():
    assert {"sats_4x4", "glove_template", "robot_hand_template"} <= set(available_layouts())


def test_sats_4x4_geometry_and_channel_order():
    L = load_layout("sats_4x4")
    assert L.n == 16 and L.parent_frame == "sensor"
    p_mm = L.positions * 1e3
    np.testing.assert_allclose(p_mm[0, :2], [-9.75, -9.75])   # S1
    np.testing.assert_allclose(p_mm[3, :2], [9.75, -9.75])    # S4
    np.testing.assert_allclose(p_mm[15, :2], [9.75, 9.75])    # S16
    np.testing.assert_allclose(np.abs(p_mm[:, :2]).max(), 9.75)
    xs = np.unique(np.round(p_mm[:, 0], 6))
    np.testing.assert_allclose(np.diff(xs), 6.5)
    np.testing.assert_array_equal(L.channels, np.arange(16))
    np.testing.assert_allclose(L.normals, np.tile([0, 0, 1.0], (16, 1)))


def test_sats_yaml_matches_grid_layout():
    L = load_layout("sats_4x4")
    G = grid_layout(4, 4, 6.5, name="sats_4x4", parent="sats_pad")
    np.testing.assert_allclose(L.positions, G.positions, atol=1e-12)
    assert [t.id for t in L.taxels] == [t.id for t in G.taxels]
    assert L.groups == G.groups


def test_glove_template():
    L = load_layout("glove_template")
    assert L.parent_frame == "mano"
    assert set(L.parents) <= set(MANO_SEGMENTS)
    assert len(L.groups["fingertip"]) == 5 and len(L.groups["palm"]) >= 1
    assert [m.name for m in L.imu_sites] == ["wrist", "palm", "thumb", "index", "middle", "ring", "pinky"]
    np.testing.assert_allclose(np.linalg.norm(L.normals, axis=1), 1.0)


def test_robot_hand_template_uses_urdf_links():
    L = load_layout("robot_hand_template")
    assert L.parent_frame == "urdf"
    assert all(p.endswith("_link") for p in L.parents)


def test_by_channel_reorders():
    d = {"name": "t", "units": "m", "taxels": [
        {"id": "a", "channel": 2, "parent": "p", "position": [0, 0, 0]},
        {"id": "b", "channel": 0, "parent": "p", "position": [1, 0, 0]}]}
    L = layout_from_dict(d)
    raw = np.array([[10.0, 11.0, 12.0]])
    np.testing.assert_array_equal(L.by_channel(raw), [[12.0, 10.0]])


def test_validation_errors(tmp_path):
    base = {"name": "t", "taxels": [{"id": "a", "channel": 0, "parent": "p", "position": [0, 0, 0]}]}
    with pytest.raises(ValueError):
        layout_from_dict({**base, "taxels": base["taxels"] * 2})
    with pytest.raises(ValueError):
        layout_from_dict({**base, "parent_frame": "mano"})  # 'p' is not a MANO segment
    with pytest.raises(ValueError):
        layout_from_dict({**base, "units": "cm"})
    with pytest.raises(FileNotFoundError):
        load_layout("does_not_exist")
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(grid_layout(2, 3, 1.0).to_dict()))
    assert load_layout(p).n == 6
