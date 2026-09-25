import pytest

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


def test_transfer_stubs():
    with pytest.raises(NotImplementedError):
        align_layouts(load_layout("glove_template"), load_layout("robot_hand_template"))
    with pytest.raises(NotImplementedError):
        project_to_mano(None, None, None)
