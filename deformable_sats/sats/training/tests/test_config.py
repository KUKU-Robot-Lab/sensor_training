"""
tests/test_config.py

SATSConfig — attn_dim / lstm_ckpt / local_map 필드 TDD
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

from sats.training.config import (
    SATSConfig,
    _parse_trial_id,
    filter_trial_ids,
    trial_id_to_paths,
)


class TestSATSConfigAttentionFields:
    """Phase 1: attn_dim, lstm_ckpt 신규 필드 검증."""

    def test_attn_dim_default(self):
        cfg = SATSConfig()
        assert cfg.attn_dim == 64, "기본 attn_dim은 64여야 한다"

    def test_attn_dim_custom(self):
        cfg = SATSConfig(attn_dim=128)
        assert cfg.attn_dim == 128

    def test_lstm_ckpt_default_empty(self):
        cfg = SATSConfig()
        assert cfg.lstm_ckpt == "", "lstm_ckpt 기본값은 빈 문자열이어야 한다"

    def test_lstm_ckpt_custom(self):
        cfg = SATSConfig(lstm_ckpt="sats/training/runs/lstm_v1/best_model.pt")
        assert cfg.lstm_ckpt == "sats/training/runs/lstm_v1/best_model.pt"

    def test_existing_fields_unchanged(self):
        """기존 필드가 그대로 유지되는지 회귀 검증."""
        cfg = SATSConfig()
        assert cfg.hidden_dim == 64
        assert cfg.num_layers == 2
        assert cfg.n_sensors == 16
        assert cfg.grid_size == 41
        assert cfg.bidirectional is False


class TestSATSConfigLocalMapFields:
    """Phase 1 (Local Map): local_map_size, sensor_spacing_mm, attn_ckpt 필드 검증."""

    def test_local_map_size_default(self):
        cfg = SATSConfig()
        assert cfg.local_map_size == 15, "기본 local_map_size는 15여야 한다"

    def test_local_map_size_custom(self):
        cfg = SATSConfig(local_map_size=11)
        assert cfg.local_map_size == 11

    def test_local_map_size_is_odd(self):
        """기본값이 홀수여야 local map 중앙 배치가 정확하다."""
        cfg = SATSConfig()
        assert cfg.local_map_size % 2 == 1, "local_map_size는 홀수여야 한다"

    def test_sensor_spacing_mm_default(self):
        cfg = SATSConfig()
        assert cfg.sensor_spacing_mm == 6.5, "기본 sensor_spacing_mm는 6.5여야 한다"

    def test_sensor_spacing_mm_custom(self):
        cfg = SATSConfig(sensor_spacing_mm=5.0)
        assert cfg.sensor_spacing_mm == 5.0

    def test_attn_ckpt_default_empty(self):
        cfg = SATSConfig()
        assert cfg.attn_ckpt == "", "attn_ckpt 기본값은 빈 문자열이어야 한다"

    def test_attn_ckpt_custom(self):
        cfg = SATSConfig(attn_ckpt="sats/training/runs/attn_v1/best_model.pt")
        assert cfg.attn_ckpt == "sats/training/runs/attn_v1/best_model.pt"

    def test_existing_fields_unchanged_after_local_map_addition(self):
        """Local Map 필드 추가 후 기존 필드 회귀 검증."""
        cfg = SATSConfig()
        assert cfg.attn_dim == 64
        assert cfg.lstm_ckpt == ""
        assert cfg.hidden_dim == 64
        assert cfg.grid_size == 41
        assert cfg.grid_step_mm == 0.5
        assert cfg.grid_min_mm == -10.0
        assert cfg.grid_max_mm == 10.0


class TestSATSConfigCNNFields:
    """Phase 1 (CNN): local_map_ckpt, cnn_hidden_channels 필드 검증."""

    def test_local_map_ckpt_default_empty(self):
        cfg = SATSConfig()
        assert cfg.local_map_ckpt == "", "local_map_ckpt 기본값은 빈 문자열이어야 한다"

    def test_local_map_ckpt_custom(self):
        cfg = SATSConfig(local_map_ckpt="sats/training/runs/local_map_v1/best_model.pt")
        assert cfg.local_map_ckpt == "sats/training/runs/local_map_v1/best_model.pt"

    def test_cnn_hidden_channels_default(self):
        cfg = SATSConfig()
        assert cfg.cnn_hidden_channels == 16, "기본 cnn_hidden_channels는 16이어야 한다"

    def test_cnn_hidden_channels_custom(self):
        cfg = SATSConfig(cnn_hidden_channels=32)
        assert cfg.cnn_hidden_channels == 32

    def test_existing_fields_unchanged_after_cnn_addition(self):
        """CNN 필드 추가 후 기존 필드 회귀 검증."""
        cfg = SATSConfig()
        assert cfg.attn_dim == 64
        assert cfg.lstm_ckpt == ""
        assert cfg.attn_ckpt == ""
        assert cfg.local_map_size == 15
        assert cfg.hidden_dim == 64
        assert cfg.grid_size == 41


class TestSATSConfigMk555Grid:
    """mk555 grid/output defaults."""

    def test_default_grid_contract(self):
        cfg = SATSConfig()
        assert cfg.grid_size == 41
        assert cfg.grid_min_mm == -10.0
        assert cfg.grid_max_mm == 10.0
        assert cfg.grid_step_mm == 0.5
        assert cfg.raw_dir == "learning_data/sensor_raw_bin"
        assert cfg.gt_dir == "learning_data/gt"
        assert cfg.dataset_index_path == "learning_data/gt/dataset_index.json"
        assert cfg.prefer_merged_bin is True
        assert cfg.val_trials == []
        assert cfg.val_ratio == 0.2
        assert cfg.include_materials == []

    def test_training_includes_hold_rows_by_default(self):
        # u_mm은 가상 보간축(max깊이 hold 표식)이라 물리 필터가 아니다.
        # hold(=최대 압입 정지·relaxation) 행도 수직 GT가 유효하므로 기본 포함한다.
        cfg = SATSConfig()
        assert cfg.use_u_zero_only is False

    def test_training_defaults_follow_paper_window_mode(self):
        cfg = SATSConfig()
        assert cfg.use_window_dataset is True
        assert cfg.window_size == 10
        assert cfg.seq_len == 1000
        assert cfg.batch_size == 2048
        assert cfg.num_workers == 2

    def test_trial_paths_include_merged_bin(self, tmp_path):
        cfg = SATSConfig(raw_dir=str(tmp_path / "raw_data"))
        paths = cfg.trial_paths("ecomesh_d5_z2.5_test1")
        assert paths["merged_bin"].name == "ecomesh_d5_z2.5_test1_merged.bin"
        assert paths["merged_csv"].name == "ecomesh_d5_z2.5_test1_merged.csv"

    def test_trial_id_parser_allows_resolution_in_material_key(self):
        parsed = _parse_trial_id("ecomesh_xy0p5_d5_z2.5_test1")
        assert parsed == {"material": "ecomesh_xy0p5", "d": 5, "z": 2.5, "n": 1}

    def test_trial_paths_include_resolution_material_key(self, tmp_path):
        paths = trial_id_to_paths("ecomesh_xy0p5_d5_z2.5_test1", raw_dir=str(tmp_path / "raw_data"))
        assert paths["merged_bin"] == (
            tmp_path
            / "raw_data"
            / "ecomesh_xy0p5"
            / "d5"
            / "z_2.5mm"
            / "test1"
            / "ecomesh_xy0p5_d5_z2.5_test1_merged.bin"
        )

    def test_include_materials_custom(self):
        cfg = SATSConfig(include_materials=["eco20_xy1"])
        assert cfg.include_materials == ["eco20_xy1"]

    def test_filter_trial_ids_keeps_only_included_materials(self):
        trial_ids = [
            "eco20_xy1_d5_z2.5_test1",
            "eco50_xy1_d5_z2.5_test1",
            "ecomesh_xy1_d5_z2.5_test1",
        ]

        assert filter_trial_ids(trial_ids, include_materials=["eco20_xy1"]) == [
            "eco20_xy1_d5_z2.5_test1",
        ]

    def test_filter_trial_ids_combines_material_and_diameter_filters(self):
        trial_ids = [
            "eco20_xy1_d5_z2.5_test1",
            "eco20_xy1_d5_z2.5_test2",
            "eco20_xy1_d10_z3.5_test1",
            "eco50_xy1_d5_z2.5_test1",
        ]

        assert filter_trial_ids(
            trial_ids,
            include_materials=["eco20_xy1"],
            exclude_diameters=[10],
        ) == [
            "eco20_xy1_d5_z2.5_test1",
            "eco20_xy1_d5_z2.5_test2",
        ]
