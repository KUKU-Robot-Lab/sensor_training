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


def test_bare_stage_config_names_never_resolve_against_the_working_directory(tmp_path, monkeypatch):
    """Regression: ``stage_config_path`` returned ``Path("baseline.yaml")`` when a file of that name existed in
    the working directory (e.g. a sweep space), so ``train`` / ``pipeline`` / ``deploy`` silently used it as
    the stage config; default.yaml says bare names are ``robot_skin/configs/stages/<name>``."""
    from robot_skin.__main__ import stage_config_path
    from robot_skin.config import DEFAULT_CONFIG

    monkeypatch.chdir(tmp_path)
    (tmp_path / "baseline.yaml").write_text("mode: grid\nspace:\n  train.lr: [1.0e-3]\n")
    assert stage_config_path("baseline") == DEFAULT_CONFIG.parent / "stages" / "baseline.yaml"
    cfg = {"stages": {"baseline": "my/baseline.yaml", "vtla": str(tmp_path / "v.yaml")}}
    assert stage_config_path("baseline", None, cfg) == Path("my/baseline.yaml")      # a path: as given
    assert stage_config_path("vtla", None, cfg) == tmp_path / "v.yaml"
    assert stage_config_path("contact", "x.yaml") == Path("x.yaml")                  # explicit --config


def test_default_config_keys_are_validated_and_paths_reach_every_command(tmp_path, monkeypatch):
    """Regression: a misspelled default.yaml key was silently ignored, and ``paths.raw_root`` /
    ``paths.processed_root`` reached neither ``preprocess`` nor the ``record`` loggers (they kept
    robot_skin/data/raw|processed) while ``pipeline`` read ``paths.processed_root``."""
    import pytest

    import robot_skin.__main__ as M
    from robot_skin.acquisition import _cli
    from robot_skin.config import SCHEMA

    cfg = load_config()
    assert set(cfg) == set(SCHEMA)
    for sec in ("paths", "stages", "synthetic"):
        assert set(cfg[sec]) == set(SCHEMA[sec]), sec
    assert set(cfg["pipeline"]["splits"]) == set(SCHEMA["pipeline"]["splits"])
    for bad in ({"pipeline": {"splits": {"val_fraction": 0.2}}}, {"path": {}}, {"paths": {"raw": "x"}}):
        with pytest.raises(ValueError, match="unknown key"):
            load_config(overrides=bad)
    p = tmp_path / "typo.yaml"
    p.write_text("synthetic: {n_motions: 3}\n")
    with pytest.raises(ValueError, match="synthetic.n_motions"):
        load_config(p)
    # default.yaml paths → preprocess --raw / --out and the loggers' --root (unless given / set in preprocess.yaml)
    raw, proc, syn = tmp_path / "disk" / "raw", tmp_path / "disk" / "processed", tmp_path / "disk" / "synthetic"
    moved = load_config(overrides={"paths": {"raw_root": str(raw), "processed_root": str(proc),
                                             "synthetic_root": str(syn)}})
    monkeypatch.setattr(M, "_defaults", lambda: moved)
    monkeypatch.setattr("robot_skin.config.load_config", lambda *a, **k: moved)
    assert M._preprocess_paths([]) == ["--raw", str(raw), "--out", str(proc)]
    assert M._preprocess_paths(["--raw", "r", "--out", "o"]) == []
    assert M._preprocess_paths(["--set", "raw_root=elsewhere"]) == ["--out", str(proc)]
    assert _cli.default_root() == raw and _cli.default_root(fake=True) == syn
    assert _cli.base_parser("x", "y").parse_args([]).root == raw
    raw.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)                                        # no robot_skin/data/raw here
    assert M.main(["preprocess", "-q"]) == 0                           # reads paths.raw_root (empty: 0 sessions)


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
