"""Phase 0 data layout contract: skin_ws raw BIN archive -> learning_data plan."""

from pathlib import Path

import pytest

from sats.preprocessing.bin_merge import (
    DUE_MAGIC,
    ETHERMOTION_MAGIC,
    LOADCELL_MAGIC,
    find_bin_set,
    read_bin_header,
)
from sats.preprocessing.prepare_learning_data import (
    REGISTRY_FILENAME,
    discover_planned_trials,
    load_trial_registry,
    save_trial_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "skin_ws" / "raw_data"
SATS_SOURCE = SOURCE_ROOT / "sats"
# ecomesh 원본은 센서 유닛별 버전 폴더(v0=원본, v1=신규)로 재편됨 → v0 명시 경로.
D5_SMOKE_TRIAL = SATS_SOURCE / "ecomesh" / "v0" / "xy_0.5mm" / "d5" / "test1"
D10_SMOKE_TRIAL = SATS_SOURCE / "ecomesh" / "v0" / "xy_1mm" / "d10" / "20260622_test4"


class TestSkinWsRawBinArchive:
    def test_sats_source_root_exists(self):
        assert SATS_SOURCE.is_dir(), f"SATS source archive missing: {SATS_SOURCE}"

    @pytest.mark.parametrize(
        "trial_dir",
        [
            D5_SMOKE_TRIAL,
            SATS_SOURCE / "ecomesh" / "v0" / "xy_1mm" / "d5" / "20260622_test1",
            SATS_SOURCE / "eco20" / "xy_1mm" / "d5" / "20260619_test5",
        ],
    )
    def test_required_raw_bin_set_exists(self, trial_dir):
        paths = find_bin_set(trial_dir)
        assert paths["due"].name.startswith("due_raw_burst_")
        assert paths["ethermotion"].name.startswith("ethermotion_encoder_")
        assert paths["loadcell"].name.startswith("loadcell_raw_")

class TestRawBinHeaders:
    @pytest.fixture(scope="class")
    def d5_paths(self):
        return find_bin_set(D5_SMOKE_TRIAL)

    @pytest.fixture(scope="class")
    def d10_paths(self):
        return find_bin_set(D10_SMOKE_TRIAL)

    def test_d5_due_header(self, d5_paths):
        magic, header, _ = read_bin_header(d5_paths["due"])
        assert magic == DUE_MAGIC
        assert "sensor-major uint32 little-endian: sensor[16][frame10]" in header["payload_layout"]
        assert header["num_sensors"] == 16
        assert header["fifo_frames"] == 10

    def test_d10_due_header(self, d10_paths):
        if d10_paths is None:
            pytest.skip("d10 smoke archive is not present in this checkout")
        magic, header, _ = read_bin_header(d10_paths["due"])
        assert magic == DUE_MAGIC
        assert "sensor-major uint32 little-endian: sensor[16][frame10]" in header["payload_layout"]

    def test_ethermotion_header_supports_record_size_in_archive(self, d5_paths, d10_paths):
        for paths in [p for p in [d5_paths, d10_paths] if p is not None]:
            magic, header, _ = read_bin_header(paths["ethermotion"])
            assert magic == ETHERMOTION_MAGIC
            assert int(header["record_bytes"]) in {40, 56}
            assert "u_cmd" in header["columns"] or "U" in header["columns"]

    def test_loadcell_header(self, d5_paths, d10_paths):
        for paths in [p for p in [d5_paths, d10_paths] if p is not None]:
            magic, header, _ = read_bin_header(paths["loadcell"])
            assert magic == LOADCELL_MAGIC
            assert header["record_bytes"] == "variable"


class TestLearningDataPlan:
    def test_current_archive_plans_all_force_trials(self, tmp_path):
        # ecomesh가 버전 폴더(v0/v1)로 재편된 뒤: 'all' 은 flat 소재(eco20/eco50)만
        # 자동 계획하고, 버전 폴더는 재귀하지 않는다(명시-버전 설계). skipped 도 없음.
        planned, skipped = discover_planned_trials(
            SOURCE_ROOT,
            tmp_path / "learning_data",
            source_material="all",
            material="auto",
            depth_map={"d5": 2.5, "d10": 3.5},
        )
        trial_ids = {p.trial_id for p in planned}
        assert len(planned) == 12
        assert skipped == []
        assert {"eco20_xy1_d5_z2.5_test1", "eco50_xy1_d10_z3.5_test1"} <= trial_ids
        assert all(p.source_dir.is_dir() for p in planned)

    def test_ecomesh_planned_via_explicit_version(self, tmp_path):
        # ecomesh 는 버전 명시(source_material="ecomesh/v0")로 취득한다(재편 후 설계).
        planned, _ = discover_planned_trials(
            SOURCE_ROOT,
            tmp_path / "learning_data",
            source_material="ecomesh/v0",
            material="ecomesh",
            depth_map={"d5": 2.5, "d10": 3.5},
        )
        trial_ids = {p.trial_id for p in planned}
        assert {"ecomesh_xy0p5_d5_z2.5_test1", "ecomesh_xy1_d10_z3.5_test1"} <= trial_ids
        assert all(p.source_dir.is_dir() for p in planned)


class TestTrialRegistryStableNumbering:
    """test 번호는 registry로 영구 고정되어, 과거 날짜 폴더를 끼워넣어도 안 밀린다."""

    @staticmethod
    def _make_trial(d_dir: Path, name: str) -> None:
        trial = d_dir / name
        trial.mkdir(parents=True)
        for stem in ("due_raw_burst", "ethermotion_encoder", "loadcell_raw"):
            (trial / f"{stem}_{name}.bin").write_bytes(b"")

    def _plan(self, source_root: Path, learning_root: Path) -> dict:
        registry = load_trial_registry(learning_root / REGISTRY_FILENAME)
        planned, _ = discover_planned_trials(
            source_root,
            learning_root,
            source_material="ecomesh",
            material="auto",
            depth_map={"d10": 3.5},
            registry=registry,
        )
        save_trial_registry(learning_root / REGISTRY_FILENAME, registry)
        return {p.source_dir.name: p.test_no for p in planned}

    def test_inserting_earlier_dated_trial_does_not_renumber(self, tmp_path):
        source_root = tmp_path / "raw"
        d_dir = source_root / "sats" / "ecomesh" / "xy_1mm" / "d10"
        learning_root = tmp_path / "learning_data"

        self._make_trial(d_dir, "20260601_test1")
        self._make_trial(d_dir, "20260602_test1")
        first = self._plan(source_root, learning_root)
        assert first == {"20260601_test1": 1, "20260602_test1": 2}

        # 더 빠른 날짜 폴더를 나중에 추가 → 기존 번호는 유지, 새 폴더만 append.
        self._make_trial(d_dir, "20260531_test1")
        second = self._plan(source_root, learning_root)
        assert second["20260601_test1"] == 1
        assert second["20260602_test1"] == 2
        assert second["20260531_test1"] == 3

    def test_numbering_is_idempotent_across_reruns(self, tmp_path):
        source_root = tmp_path / "raw"
        d_dir = source_root / "sats" / "ecomesh" / "xy_1mm" / "d10"
        learning_root = tmp_path / "learning_data"
        self._make_trial(d_dir, "20260601_test1")
        self._make_trial(d_dir, "20260602_test1")
        assert self._plan(source_root, learning_root) == self._plan(source_root, learning_root)

    def test_xy_resolution_has_separate_numbering(self, tmp_path):
        source_root = tmp_path / "raw"
        learning_root = tmp_path / "learning_data"
        self._make_trial(source_root / "sats" / "ecomesh" / "xy_0.5mm" / "d5", "test1")
        self._make_trial(source_root / "sats" / "ecomesh" / "xy_1mm" / "d5", "test1")

        registry = load_trial_registry(learning_root / REGISTRY_FILENAME)
        planned, skipped = discover_planned_trials(
            source_root,
            learning_root,
            source_material="ecomesh",
            material="auto",
            depth_map={"d5": 2.5},
            registry=registry,
        )

        assert skipped == []
        assert [p.trial_id for p in planned] == [
            "ecomesh_xy0p5_d5_z2.5_test1",
            "ecomesh_xy1_d5_z2.5_test1",
        ]
        assert registry["ecomesh_xy0p5/d5"]["sats/ecomesh/xy_0.5mm/d5/test1"] == 1
        assert registry["ecomesh_xy1/d5"]["sats/ecomesh/xy_1mm/d5/test1"] == 1


class TestForceConversionContract:
    def test_fz_conversion_formula(self):
        baseline_kg = 0.10
        press_kg = 0.30
        fz = (press_kg - baseline_kg) * 9.80665
        assert fz > 0
        assert abs(fz - 1.9613) < 0.001
