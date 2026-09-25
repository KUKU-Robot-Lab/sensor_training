from pathlib import Path

import numpy as np

from common.layouts import load_layout
from robot_skin.config import load_config
from robot_skin.transfer import align_layouts, project_to_mano


def test_default_config_loads_and_is_consistent(tmp_path):
    """configs/default.yaml = the top-level CLI defaults (paths, stage config files, pipeline stage
    list + split policy); stage hyper-parameters live only in configs/stages/<stage>.yaml."""
    import yaml

    from robot_skin.__main__ import STAGES, stage_config_path
    from robot_skin.acquisition._cli import base_parser
    from robot_skin.datasets.build import DEFAULTS as PRE
    from robot_skin.datasets.splits import GROUP_KEYS

    cfg = load_config()
    assert set(cfg) == {"paths", "hardware", "stages", "pipeline", "synthetic"}
    paths = cfg["paths"]
    assert paths["raw_root"] == PRE["raw_root"] == "robot_skin/data/raw"
    assert paths["processed_root"] == PRE["out_root"] == "robot_skin/data/processed"
    assert paths["runs_root"] == "robot_skin/runs"
    assert base_parser("x", "y").parse_args([]).root == Path(paths["raw_root"])   # `record` passes through
    assert tuple(cfg["pipeline"]["stages"]) == STAGES
    sp = cfg["pipeline"]["splits"]
    assert sp["by"] in GROUP_KEYS and 0 < sp["val_frac"] + sp["test_frac"] < 1
    for stage, ref in cfg["stages"].items():                 # every referenced stage YAML exists and is its own
        p = stage_config_path(stage, None, cfg)
        assert p.is_file() and yaml.safe_load(p.read_text())["stage"] == stage, (stage, ref)
    for stage in STAGES:                                     # each stage's processed root = the shared path
        mod = __import__(f"robot_skin.stages.{stage}", fromlist=["DEFAULTS"])
        assert mod.DEFAULTS["data"]["processed_root"] == paths["processed_root"]
        assert mod.DEFAULTS["train"]["out_dir"] == f"{paths['runs_root']}/{stage}"
    load_layout("glove_template")
    p = tmp_path / "o.yaml"
    p.write_text("paths: {runs_root: /tmp/runs}\npipeline: {stages: [imu_pose]}\n")
    cfg2 = load_config(p, overrides={"hardware": "cpu"})
    assert cfg2["paths"]["runs_root"] == "/tmp/runs" and cfg2["paths"]["raw_root"] == paths["raw_root"]
    assert cfg2["pipeline"]["stages"] == ["imu_pose"] and cfg2["pipeline"]["splits"] == sp
    assert cfg2["hardware"] == "cpu" and load_config()["hardware"] is None


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
