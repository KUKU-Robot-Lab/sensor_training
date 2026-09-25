import json

import pytest

from robot_skin.acquisition import MANIFEST_NAME, SessionManifest, StreamInfo
from robot_skin.acquisition import glove_logger, robot_logger


def test_manifest_roundtrip(tmp_path):
    m = SessionManifest(kind="glove", layout="glove_template",
                        streams={"pressure": StreamInfo(file="pressure.bin", rate_hz=200.0)})
    m.add_segment(0.0, 5.0, "no_contact")
    p = m.save(tmp_path / "S1")
    assert p.name == MANIFEST_NAME
    m2 = SessionManifest.load(tmp_path / "S1")
    assert m2 == m
    assert m2.spans("no_contact") == [(0.0, 5.0)]
    assert m2.stream_path(tmp_path / "S1", "pressure") == tmp_path / "S1" / "pressure.bin"


@pytest.mark.parametrize("bad", [
    dict(kind="car"),
    dict(streams={"x": StreamInfo(file="/abs/path.bin")}),
    dict(streams={"x": StreamInfo(file="a.bin", method="cubic")}),
    dict(segments=[{"t0": 2.0, "t1": 1.0, "label": "x"}]),
])
def test_manifest_validation(bad):
    kw = dict(kind="robot", layout="robot_hand_template") | bad
    with pytest.raises(ValueError):
        SessionManifest(**kw)


def test_manifest_rejects_newer_schema(tmp_path):
    d = SessionManifest(kind="bench", layout="sats_4x4").to_dict()
    d["schema_version"] = 999
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(d))
    with pytest.raises(ValueError):
        SessionManifest.load(tmp_path)


def test_glove_logger_dry_run(tmp_path):
    assert glove_logger.main(["--out", str(tmp_path / "g"), "--dry-run", "--no-contact",
                              "--duration", "10"]) == 0
    m = SessionManifest.load(tmp_path / "g")
    assert m.kind == "glove" and m.layout == "glove_template"
    assert set(m.streams) == {"pressure", "imu", "camera"}
    assert len([f for f in m.streams["imu"].fields if f.endswith(".quat")]) == 7
    assert m.spans("no_contact") == [(0.0, 10.0)]


def test_robot_logger_dry_run_and_stub(tmp_path):
    assert robot_logger.main(["--out", str(tmp_path / "r"), "--dry-run"]) == 0
    assert set(SessionManifest.load(tmp_path / "r").streams) == {"pressure", "joint_state"}
    with pytest.raises(NotImplementedError):
        robot_logger.main(["--out", str(tmp_path / "r2")])
    with pytest.raises(NotImplementedError):
        glove_logger.main(["--out", str(tmp_path / "g2")])
