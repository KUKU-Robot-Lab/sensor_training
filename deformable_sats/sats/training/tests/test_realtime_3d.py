from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np

from sats.inference.realtime_3d import RealtimeViz3D


class _DummyEngine:
    # 실 엔진 계약: 출력 grid 속성 노출 (뷰어가 메시·좌표 구성에 사용)
    grid_size = 40
    grid_min_mm = -10.0
    grid_max_mm = 10.0
    grid_step_mm = 20.0 / 39
    taxel_area = grid_step_mm ** 2

    def get_taxel_value(self, pred_map: np.ndarray, x_mm: float, y_mm: float) -> float:
        return float(pred_map.max())


def test_realtime_3d_update_uses_plot_surface(monkeypatch) -> None:
    viz = RealtimeViz3D(engine=_DummyEngine(), vmax=1.0)

    calls = {"bar3d": 0, "plot_surface": 0}

    def _fake_bar3d(*args, **kwargs):
        calls["bar3d"] += 1

        class _DummyCollection:
            def remove(self) -> None:
                return None

        return _DummyCollection()

    def _fake_plot_surface(*args, **kwargs):
        calls["plot_surface"] += 1

        class _DummyCollection:
            def remove(self) -> None:
                return None

        return _DummyCollection()

    monkeypatch.setattr(viz.ax, "bar3d", _fake_bar3d)
    monkeypatch.setattr(viz.ax, "plot_surface", _fake_plot_surface)

    pred = np.full((40, 40), 0.5, dtype=np.float32)
    viz.update(pred, peak=(0.0, 0.0, 0.5), fz=0.8)

    assert calls["bar3d"] == 0
    assert calls["plot_surface"] == 1

    viz.fig.clf()
