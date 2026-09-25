import numpy as np

from common.layouts import load_layout
from robot_skin.config import load_config
from robot_skin.transfer import align_layouts, project_to_mano


def test_default_config_loads_and_is_consistent(tmp_path):
    cfg = load_config()
    assert cfg["master_hz"] == 200.0
    assert cfg["policy"]["obs_mode"] in ("full", "ordinal", "binary", "none")
    assert 0 < cfg["contact"]["weak_pct"] < cfg["contact"]["strong_pct"]
    load_layout(cfg["layout"])
    p = tmp_path / "o.yaml"
    p.write_text("baseline: {epochs: 3}\n")
    cfg2 = load_config(p, overrides={"policy": {"obs_mode": "none"}})
    assert cfg2["baseline"]["epochs"] == 3 and cfg2["baseline"]["hidden"] == cfg["baseline"]["hidden"]
    assert cfg2["policy"]["obs_mode"] == "none"


def test_transfer_is_implemented():
    """The former transfer stubs (NotImplementedError) are implemented: glove taxels project onto
    their own MANO segments and a layout aligns to itself."""
    glove = load_layout("glove_template")
    al = align_layouts(glove, glove)
    np.testing.assert_array_equal(al.index[:, 0], np.arange(glove.n))
    from robot_skin.transfer import layout_rest_poses

    pos, _ = layout_rest_poses(glove)
    pr = project_to_mano(pos)
    assert pr.segment_names[:2] == ["thumb3", "index3"]
    assert pr.u.shape == (glove.n,)
