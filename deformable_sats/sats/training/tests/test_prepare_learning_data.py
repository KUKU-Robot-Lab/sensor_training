import json

import pytest

from sats.preprocessing.prepare_learning_data import (
    discover_planned_trials,
    parse_depth_map,
)


def _touch_bin_set(test_dir, suffix):
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / f"due_raw_burst_{suffix}.bin").write_bytes(b"")
    (test_dir / f"ethermotion_encoder_{suffix}.bin").write_bytes(b"")
    (test_dir / f"loadcell_raw_{suffix}.bin").write_bytes(b"")


def test_discover_planned_trials_maps_skin_ws_archive_to_learning_data(tmp_path):
    source_root = tmp_path / "skin_ws" / "raw_data"
    learning_root = tmp_path / "learning_data"
    d5_t1 = source_root / "sats" / "eco20" / "xy_1mm" / "d5" / "20260619_test5"
    d10_t1 = source_root / "sats" / "ecomesh" / "xy_0.5mm" / "d10" / "20260610_test1"
    d5_t2 = source_root / "sats" / "ecomesh" / "xy_1mm" / "d5" / "20260622_test1"
    d5_bad = source_root / "sats" / "eco20" / "xy_1mm" / "d5" / "20260619_test6"
    _touch_bin_set(d5_t1, "20260601")
    _touch_bin_set(d10_t1, "20260601")
    _touch_bin_set(d5_t2, "20260602")
    d5_bad.mkdir(parents=True)
    (d5_bad / "due_raw_burst_20260602.bin").write_bytes(b"")

    planned, skipped = discover_planned_trials(
        source_root,
        learning_root,
        source_material="all",
        material="auto",
        depth_map={"d5": 2.5, "d10": 3.5},
    )

    assert [p.trial_id for p in planned] == [
        "eco20_xy1_d5_z2.5_test1",
        "ecomesh_xy0p5_d10_z3.5_test1",
        "ecomesh_xy1_d5_z2.5_test1",
    ]
    assert planned[0].output_dir == learning_root / "sensor_raw_bin/eco20_xy1/d5/z_2.5mm/test1"
    assert planned[1].output_dir == learning_root / "sensor_raw_bin/ecomesh_xy0p5/d10/z_3.5mm/test1"
    assert any("20260619_test6" in item for item in skipped)


def test_discover_planned_trials_preserves_legacy_layout(tmp_path):
    source_root = tmp_path / "skin_ws" / "raw_data"
    learning_root = tmp_path / "learning_data"
    d5_t1 = source_root / "sats" / "eco20 + mesh" / "d5" / "test1"
    d10_t1 = source_root / "sats" / "eco20 + mesh" / "d10" / "test1"
    _touch_bin_set(d5_t1, "20260601")
    _touch_bin_set(d10_t1, "20260602")

    planned, skipped = discover_planned_trials(
        source_root,
        learning_root,
        source_material="eco20 + mesh",
        material="ecomesh",
        depth_map={"d5": 2.5, "d10": 3.5},
    )

    assert skipped == []
    assert [p.trial_id for p in planned] == [
        "ecomesh_d5_z2.5_test1",
        "ecomesh_d10_z3.5_test1",
    ]
    assert planned[0].output_dir == learning_root / "sensor_raw_bin/ecomesh/d5/z_2.5mm/test1"


def test_discover_planned_trials_natural_sorts_undated_test_dirs(tmp_path):
    source_root = tmp_path / "skin_ws" / "raw_data"
    learning_root = tmp_path / "learning_data"
    d_dir = source_root / "sats" / "ecomesh" / "xy_0.5mm" / "d5"
    for name in ["test1", "test10", "test2"]:
        _touch_bin_set(d_dir / name, name)

    planned, skipped = discover_planned_trials(
        source_root,
        learning_root,
        source_material="ecomesh",
        material="auto",
        depth_map={"d5": 2.5},
    )

    assert skipped == []
    assert [(p.source_dir.name, p.trial_id) for p in planned] == [
        ("test1", "ecomesh_xy0p5_d5_z2.5_test1"),
        ("test2", "ecomesh_xy0p5_d5_z2.5_test2"),
        ("test10", "ecomesh_xy0p5_d5_z2.5_test3"),
    ]


def test_discover_planned_trials_uses_manifest_when_multiple_sets_exist(tmp_path):
    source_root = tmp_path / "skin_ws" / "raw_data"
    learning_root = tmp_path / "learning_data"
    test_dir = source_root / "sats" / "eco20" / "xy_1mm" / "d10" / "20260602_test1"
    for suffix in ["a", "b"]:
        _touch_bin_set(test_dir, suffix)
    (test_dir / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "due": r"C:\x\due_raw_burst_b.bin",
                    "ethermotion": r"C:\x\ethermotion_encoder_b.bin",
                    "loadcell": r"C:\x\loadcell_raw_b.bin",
                }
            }
        )
    )

    planned, skipped = discover_planned_trials(
        source_root,
        learning_root,
        source_material="eco20",
        material="auto",
        depth_map={"d10": 3.5},
    )

    assert skipped == []
    assert len(planned) == 1
    assert planned[0].trial_info()["source_trial_dir"].endswith("20260602_test1")


def test_parse_depth_map_validates_diameter_keys():
    assert parse_depth_map(["d5=2.5", "d10=3.5"]) == {"d5": 2.5, "d10": 3.5}
    with pytest.raises(ValueError):
        parse_depth_map(["x5=2.5"])
