"""
sats/inference/realtime_3d.py

실시간 3D 압력 맵 시각화.

analyze_taxel_rmse.py 색상 스타일을 실시간 surface로 구현.

레이아웃
--------
  ┌──────────────────────────────────────────────────────┐
  │  3D Pressure Surface (Blues colormap)                │
  │  ✚ peak 위치 vertical line + 텍스트                  │
  │  ○ query 위치 vertical line + 텍스트 (클릭)          │
  └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .inference_engine import (
    SATSInferenceEngine,
    GRID_MIN_MM,
    GRID_MAX_MM,
    GRID_STEP_MM,
    GRID_SIZE,
    TAXEL_AREA,
)

# 그리드 좌표 메쉬 (mm)
_COORDS = np.linspace(GRID_MIN_MM, GRID_MAX_MM, GRID_SIZE)
_XX, _YY = np.meshgrid(_COORDS, _COORDS)   # 각각 [41, 41]


# ─────────────────────────────────────────────────────────────────────────────
# RealtimeViz3D
# ─────────────────────────────────────────────────────────────────────────────

class RealtimeViz3D:
    """
    실시간 3D 압력 맵 시각화 창.

    Parameters
    ----------
    engine      : SATSInferenceEngine
    query_xy_mm : 초기 query 위치 (x_mm, y_mm). None 이면 미지정.
    vmax        : z축 최댓값 고정(N/mm²). None 이면 자동.
    elev / azim : 3D 뷰 각도
    title       : 창 제목
    """

    def __init__(
        self,
        engine: SATSInferenceEngine,
        query_xy_mm: Optional[Tuple[float, float]] = None,
        vmax: Optional[float] = None,
        elev: float = 30.0,
        azim: float = -60.0,
        title: str = "SATS Realtime 3D",
    ) -> None:
        self.engine      = engine
        self.query_xy_mm = query_xy_mm
        self.vmax        = vmax

        # 표면 메시를 engine 출력 grid 에 맞춘다 (0.5mm=41², 0.1mm=201² 등 임의 해상도).
        _c = np.linspace(engine.grid_min_mm, engine.grid_max_mm, engine.grid_size)
        self._XX, self._YY = np.meshgrid(_c, _c)

        plt.ion()
        self.fig = plt.figure(figsize=(10, 7))
        self.fig.canvas.manager.set_window_title(title)

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_xlabel("x [mm]", labelpad=6)
        self.ax.set_ylabel("y [mm]", labelpad=6)
        self.ax.set_zlabel("Pressure (N/mm²)", labelpad=6)
        self.ax.set_title("Predicted Pressure Map (3D)")

        # 컬러바용 ScalarMappable (초기 dummy)
        self._sm = plt.cm.ScalarMappable(
            cmap=plt.cm.Blues,
            norm=plt.Normalize(vmin=0, vmax=1),
        )
        self._sm.set_array([])
        self._cbar = self.fig.colorbar(self._sm, ax=self.ax, shrink=0.5, pad=0.1)
        self._cbar.set_label("Pressure (N/mm²)", fontsize=9)

        # surface 초기 placeholder
        self._surf = None

        # 마우스 클릭 이벤트
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self._frame = 0

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    # ── 업데이트 ──────────────────────────────────────────────────────────────

    def update(
        self,
        pred_map: np.ndarray,
        *,
        peak: Optional[Tuple[float, float, float]] = None,
        fz: Optional[float] = None,
        query_val_nmm2: Optional[float] = None,
    ) -> None:
        """
        새로운 예측 맵으로 3D 시각화를 갱신한다.
        """
        self._frame += 1
        ax = self.ax

        # 기존 surface 제거
        if self._surf is not None:
            self._surf.remove()

        Z = np.clip(pred_map.astype(np.float64), 0.0, None)
        z_max = float(Z.max()) if Z.max() > 1e-9 else 1.0
        eff_vmax = self.vmax if self.vmax is not None else z_max

        # Blues: 낮은 값 → 하늘색, 높은 값 → 남색 (analyze_taxel_rmse 동일)
        norm_z = np.clip(Z / (eff_vmax + 1e-12), 0, 1)
        facecolors = plt.cm.Blues(0.25 + 0.75 * norm_z)

        self._surf = ax.plot_surface(
            self._XX, self._YY, Z,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            alpha=0.9,
        )

        # z축 범위
        ax.set_zlim(0, max(eff_vmax * 1.1, 0.1))

        # 컬러바 갱신
        self._sm.set_clim(0, eff_vmax)
        self._cbar.update_normal(self._sm)

        # peak 마커 (vertical line)
        has_peak = peak is not None
        if peak is None:
            x_mm, y_mm, peak_val = 0.0, 0.0, 0.0
        else:
            x_mm, y_mm, peak_val = peak
        if fz is None:
            fz = float(pred_map.clip(0).sum()) * self.engine.taxel_area

        # 기존 마커 제거 후 재드로우
        for attr in ("_peak_line", "_query_line", "_peak_ann", "_query_ann"):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                try:
                    getattr(self, attr).remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        # peak vertical line
        if has_peak:
            self._peak_line = ax.plot(
                [x_mm, x_mm], [y_mm, y_mm], [0, peak_val * 1.05],
                color="red", linewidth=2, zorder=5,
            )[0]

            peak_label = (
                f"Peak\n"
                f"({x_mm:+.1f},{y_mm:+.1f})\n"
                f"P={peak_val:.5f} N/mm²\n"
                f"Fz={fz:.3f}N"
            )
            self._peak_ann = ax.text(
                x_mm, y_mm, peak_val * 1.08,
                peak_label,
                color="red", fontsize=8, ha="center",
                fontweight="bold",
            )
        else:
            self._peak_line = None
            self._peak_ann = ax.text(
                GRID_MIN_MM + 0.5, GRID_MIN_MM + 0.5, max(eff_vmax * 0.05, 0.01),
                f"No contact\nFz={fz:.3f}N",
                color="red", fontsize=9, ha="left", va="bottom", fontweight="bold",
            )

        # query 마커
        self._query_line = None
        self._query_ann  = None
        if self.query_xy_mm is not None:
            qx, qy = self.query_xy_mm
            qv = query_val_nmm2
            if qv is None:
                qv = self.engine.get_taxel_value(pred_map, qx, qy)
            self._query_line = ax.plot(
                [qx, qx], [qy, qy], [0, qv * 1.05],
                color="lime", linewidth=2, zorder=5,
            )[0]
            self._query_ann = ax.text(
                qx, qy, qv * 1.08,
                f"Q({qx:+.1f},{qy:+.1f})\nP={qv:.5f} N/mm²",
                color="lime", fontsize=8, ha="center",
            )

        ax.set_title(f"Predicted Pressure Map (3D)  |  frame={self._frame}")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    # ── 이벤트 ────────────────────────────────────────────────────────────────

    def _on_click(self, event) -> None:
        """3D 축 클릭 시 query 위치 업데이트 (xdata, ydata 이용)."""
        if event.inaxes is not self.ax:
            return
        # 3D plot 에서 xdata/ydata 는 투영 좌표 → 근사 처리
        if event.xdata is None or event.ydata is None:
            return
        x_mm = event.xdata
        y_mm = event.ydata
        x_mm = max(GRID_MIN_MM, min(GRID_MAX_MM, float(x_mm)))
        y_mm = max(GRID_MIN_MM, min(GRID_MAX_MM, float(y_mm)))
        # 그리드 스냅
        x_mm = round(round((x_mm - self.engine.grid_min_mm) / self.engine.grid_step_mm) * self.engine.grid_step_mm + self.engine.grid_min_mm, 4)
        y_mm = round(round((y_mm - self.engine.grid_min_mm) / self.engine.grid_step_mm) * self.engine.grid_step_mm + self.engine.grid_min_mm, 4)
        self.query_xy_mm = (x_mm, y_mm)
        print(f"[3D] query 위치 선택: ({x_mm:+.2f}, {y_mm:+.2f}) mm")
