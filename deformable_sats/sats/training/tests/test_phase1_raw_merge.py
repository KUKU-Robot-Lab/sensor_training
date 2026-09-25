"""Phase 1 검증: raw_merge.py 신규 CSV 포맷 + loadcell force source.

변경점:
- 타임스탬프: Timestamp → time_s
- force: afd(Fx,Fy,Fz[N]) → loadcell(kg) → Fz=(kg-baseline)*9.80665
- ethermotion: X,Y,Z,U → x_mm,y_mm,z_mm,u_mm
- build_export_frame 시작점 영점: (9.75,-9.75) → (-10.0,-10.0)
- 그리드: 40/±9.75 → 41/±10.0
"""
import importlib.util
import os
import numpy as np
import pandas as pd
import pytest

# sats/preprocessing/raw_merge.py 직접 로드 (공식 SATS raw merge)
_RM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../preprocessing/raw_merge.py"
)
_spec = importlib.util.spec_from_file_location("raw_merge_mod", _RM_PATH)
rm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rm)

GRAVITY = 9.80665


# ── 1.1 타임스탬프 컬럼: time_s ─────────────────────────────────────────────
class TestTimestampColumn:
    def test_load_due_uses_time_s(self, tmp_path):
        p = tmp_path / "due_data.csv"
        cols = ["elapsed_ns", "time_s", "burst_index", "frame_index"] + [
            f"Skin{i}" for i in range(1, 17)
        ]
        row = [100, 0.001, 0, 0] + [6000000 + i for i in range(16)]
        pd.DataFrame([row, [200, 0.002, 0, 1] + [6000010 + i for i in range(16)]],
                     columns=cols).to_csv(p, index=False)
        out = rm.load_due_csv(p)
        assert "timestamp_due" in out.columns
        assert np.allclose(out["timestamp_due"].to_numpy(), [0.001, 0.002])
        for i in range(1, 17):
            assert f"Skin{i}" in out.columns

    def test_load_ethermotion_uses_time_s_and_u(self, tmp_path):
        p = tmp_path / "ethermotion_data.csv"
        cols = ["elapsed_ns", "time_s", "X", "Y", "Z", "U"]
        df = pd.DataFrame(
            [
                [100, 0.001, -100000, -100000, 125000, 0],
                [200, 0.002, -100000, -100000, 155000, 20000],
            ],
            columns=cols,
        )
        df.to_csv(p, index=False)
        out = rm.load_ethermotion_csv(p)
        assert "timestamp_ethermotion" in out.columns
        assert np.allclose(out["timestamp_ethermotion"].to_numpy(), [0.001, 0.002])
        # 스케일 ×1e-4
        assert np.allclose(out["x_mm"].to_numpy(), [-10.0, -10.0])
        assert np.allclose(out["z_mm"].to_numpy(), [12.5, 15.5])
        # U 컬럼 보존(전단 변위)
        assert "u_mm" in out.columns
        assert np.allclose(out["u_mm"].to_numpy(), [0.0, 2.0])


# ── 1.2 force loader: loadcell kg → N ──────────────────────────────────────
class TestLoadcellLoader:
    def test_load_loadcell_csv_exists(self):
        assert hasattr(rm, "load_loadcell_csv"), "load_loadcell_csv 함수 필요"

    def test_load_loadcell_columns(self, tmp_path):
        p = tmp_path / "loadcell_data.csv"
        pd.DataFrame(
            {"elapsed_ns": [100, 200], "time_s": [0.001, 0.002], "kg": [0.10, 0.30]}
        ).to_csv(p, index=False)
        out = rm.load_loadcell_csv(p)
        assert "timestamp_loadcell" in out.columns
        assert "kg" in out.columns
        assert np.allclose(out["timestamp_loadcell"].to_numpy(), [0.001, 0.002])

    def test_kg_to_newton_conversion(self):
        """N = (kg - baseline) × 9.80665"""
        fz = rm.kg_to_newton(0.30, baseline_kg=0.10)
        assert abs(fz - 0.20 * GRAVITY) < 1e-9

    def test_kg_to_newton_zero_at_baseline(self):
        assert abs(rm.kg_to_newton(0.10, baseline_kg=0.10)) < 1e-9

    def test_header_only_loadcell_is_rejected(self, tmp_path):
        """헤더뿐(0행)인 loadcell CSV는 force merge 입력에서 제외되어야 한다."""
        p = tmp_path / "loadcell_data.csv"
        p.write_text("time_s,kg\n")
        with pytest.raises(ValueError, match="all loadcell rows invalid"):
            rm.load_loadcell_csv(p)


# ── 1.3 baseline: loadcell kg 기반 ─────────────────────────────────────────
class TestLoadcellBaseline:
    def _make_streams(self):
        # ethermotion: 앞 3행 idle(XYZ=0 head), 이후 이동
        ether = pd.DataFrame(
            {
                "timestamp_ethermotion": [0.0, 0.01, 0.02, 0.03, 0.04],
                "X": [0, 0, 0, -100000, -100000],
                "Y": [0, 0, 0, -100000, -100000],
                "Z": [0, 0, 0, 125000, 155000],
                "x_mm": [0, 0, 0, -10.0, -10.0],
                "y_mm": [0, 0, 0, -10.0, -10.0],
                "z_mm": [0, 0, 0, 12.5, 15.5],
                "u_mm": [0, 0, 0, 0, 0],
            }
        )
        due = pd.DataFrame(
            {
                "timestamp_due": np.linspace(0.0, 0.04, 5),
                **{f"Skin{i}": np.linspace(6e6, 6e6 + 100, 5) for i in range(1, 17)},
                "due_mean": np.linspace(6e6, 6e6 + 100, 5),
                "due_std": np.zeros(5),
            }
        )
        lc = pd.DataFrame(
            {
                "timestamp_loadcell": np.linspace(0.0, 0.04, 5),
                "kg": [0.10, 0.10, 0.10, 0.25, 0.40],
            }
        )
        return due, lc, ether

    def test_baseline_uses_loadcell_kg(self):
        due, lc, ether = self._make_streams()
        baseline = rm.compute_baseline(due, lc, ether, fallback_sec=2.0)
        # idle head 구간(XYZ=0) kg 평균 = 0.10
        assert "kg_baseline" in baseline
        assert abs(baseline["kg_baseline"] - 0.10) < 1e-6

    def test_baseline_no_afd_keys(self):
        due, lc, ether = self._make_streams()
        baseline = rm.compute_baseline(due, lc, ether, fallback_sec=2.0)
        # afd(Fx,Fy) 잔재 없음
        assert "Fx_mean" not in baseline
        assert "Fy_mean" not in baseline


# ── 1.4 build_export_frame: 시작점 (-10,-10) 영점 + u_mm + Fz[N] ────────────
class TestExportFrame:
    def _make_merged(self):
        # 시작점(-10,-10): idle z=12.5, press z=15.5
        # 다른 격자(0,0): press z=15.5
        n = 6
        merged = pd.DataFrame(
            {
                "timestamp": np.arange(n) * 0.01,
                "time_rel_sec": np.arange(n) * 0.01,
                "x_mm": [-10.0, -10.0, -10.0, 0.0, 0.0, 0.0],
                "y_mm": [-10.0, -10.0, -10.0, 0.0, 0.0, 0.0],
                "z_mm": [12.5, 15.5, 12.5, 12.5, 15.5, 15.5],
                "u_mm": [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                "kg": [0.10, 0.40, 0.10, 0.10, 0.40, 0.40],
                **{f"Skin{i}": np.full(n, 6e6 + i) for i in range(1, 17)},
            }
        )
        return merged

    def test_export_has_u_mm_column(self):
        merged = self._make_merged()
        out = rm.build_export_frame(merged, baseline_kg=0.10, force_round_dp=None)
        assert "u_mm" in out.columns

    def test_export_preserves_z_stage_and_computes_depth(self):
        """z_stage_mm는 절대 command Z, z_depth_mm만 시작 Z 기준 depth."""
        merged = self._make_merged()
        out = rm.build_export_frame(merged, baseline_kg=0.10, force_round_dp=None, z_start_mm=12.5)
        assert "z_stage_mm" in out.columns
        assert "z_depth_mm" in out.columns
        assert "z_mm" not in out.columns
        start_idle = out[(np.isclose(out["x_mm"], -10.0)) & (np.isclose(out["y_mm"], -10.0))]
        assert start_idle["z_stage_mm"].min() == pytest.approx(12.5, abs=1e-6)
        assert start_idle["z_stage_mm"].max() == pytest.approx(15.5, abs=1e-3)
        assert start_idle["z_depth_mm"].min() == pytest.approx(0.0, abs=1e-6)
        assert start_idle["z_depth_mm"].max() == pytest.approx(3.0, abs=1e-3)

    def test_export_fz_from_loadcell(self):
        """Fz = (kg - baseline) × 9.80665"""
        merged = self._make_merged()
        out = rm.build_export_frame(merged, baseline_kg=0.10, force_round_dp=None)
        assert "Fz" in out.columns
        # press kg=0.40 → Fz=(0.40-0.10)*9.80665
        press = out[np.isclose(out["kg"] if "kg" in out.columns else merged["kg"], 0.40)] \
            if "kg" in out.columns else None
        # kg 컬럼이 export에 없을 수 있으므로 최대 Fz로 검증
        expected_max = (0.40 - 0.10) * GRAVITY
        assert out["Fz"].max() == pytest.approx(expected_max, abs=1e-3)
        # idle Fz ≈ 0
        assert out["Fz"].min() == pytest.approx(0.0, abs=1e-3)

    def test_export_no_fx_fy(self):
        """단일축 loadcell → Fx/Fy 없음 (또는 전부 NaN)"""
        merged = self._make_merged()
        out = rm.build_export_frame(merged, baseline_kg=0.10, force_round_dp=None)
        if "Fx" in out.columns:
            assert out["Fx"].isna().all(), "Fx는 단일축이므로 값이 없어야 함"
        if "Fy" in out.columns:
            assert out["Fy"].isna().all()
