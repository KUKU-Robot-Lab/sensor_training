"""
sats/inference/realtime_2d.py

실시간 2D 압력 맵 시각화.

visualize.py 의 plot_sample 스타일을 실시간으로 구현.

레이아웃
--------
  ┌──────────────────────────────────┬──────────────────┐
  │  Pressure Map (hot colormap)     │   Info Panel     │
  │  [41×41 imshow, 실시간 update]   │   Peak (x, y):   │
  │                                  │   Peak P:        │
  │  ✚  peak 위치 마커               │   Fz:            │
  │  ○  query 위치 마커 (클릭)       │   Query P:       │
  └──────────────────────────────────┴──────────────────┘
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .inference_engine import (
    SATSInferenceEngine,
    GRID_MIN_MM,
    GRID_MAX_MM,
    GRID_STEP_MM,
    GRID_SIZE,
    TAXEL_AREA,
)

_EXTENT = [GRID_MIN_MM, GRID_MAX_MM, GRID_MIN_MM, GRID_MAX_MM]


# ─────────────────────────────────────────────────────────────────────────────
# RealtimeViz2D
# ─────────────────────────────────────────────────────────────────────────────

class RealtimeViz2D:
    """
    실시간 2D 압력 맵 시각화 창.

    Parameters
    ----------
    engine       : SATSInferenceEngine  (peak, fz 계산용)
    query_xy_mm  : 초기 query 위치 (x_mm, y_mm). None 이면 미지정.
    vmax         : 컬러바 최댓값 고정(N/mm²). None 이면 자동 스케일.
    title        : 창 제목
    """

    def __init__(
        self,
        engine: SATSInferenceEngine,
        query_xy_mm: Optional[Tuple[float, float]] = None,
        vmax: Optional[float] = None,
        title: str = "SATS Realtime 2D",
    ) -> None:
        self.engine      = engine
        self.query_xy_mm = query_xy_mm
        self.vmax        = vmax

        plt.ion()
        self.fig = plt.figure(figsize=(11, 5))
        self.fig.canvas.manager.set_window_title(title)
        gs = gridspec.GridSpec(1, 2, figure=self.fig, width_ratios=[3, 1], wspace=0.3)

        self.ax_map  = self.fig.add_subplot(gs[0])
        self.ax_info = self.fig.add_subplot(gs[1])

        # imshow 초기화
        empty = np.zeros((self.engine.grid_size, self.engine.grid_size), dtype=np.float32)
        self.im = self.ax_map.imshow(
            empty, origin="lower", extent=_EXTENT,
            cmap="hot", vmin=0, vmax=(vmax if vmax else 1.0),
            interpolation="nearest",
        )
        plt.colorbar(self.im, ax=self.ax_map, fraction=0.046, label="Pressure (N/mm²)")
        self.ax_map.set_xlabel("x [mm]")
        self.ax_map.set_ylabel("y [mm]")
        self.ax_map.set_title("Predicted Pressure Map")

        # peak 마커
        self.peak_marker, = self.ax_map.plot(
            [], [], marker="+", color="cyan", markersize=14, markeredgewidth=2, lw=0,
        )
        # query 마커
        self.query_marker, = self.ax_map.plot(
            [], [], marker="o", color="lime", markersize=8, markeredgewidth=2,
            fillstyle="none", lw=0,
        )

        # info 텍스트
        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(
            0.05, 0.5, "",
            transform=self.ax_info.transAxes,
            fontsize=10, verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
        )

        # 마우스 클릭으로 query 위치 선택
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        # 프레임 카운터
        self._frame = 0

        self.fig.subplots_adjust(left=0.08, right=0.95, wspace=0.35)
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
        새로운 예측 맵으로 시각화를 갱신한다.

        Parameters
        ----------
        pred_map : ndarray[41, 41]
        """
        self._frame += 1

        # 컬러바 자동 스케일
        if self.vmax is None:
            cur_max = float(pred_map.max())
            if cur_max > 1e-6:
                self.im.set_clim(0, max(cur_max, 0.1))

        self.im.set_data(pred_map)

        # peak (없으면 no-contact로 표시)
        has_peak = peak is not None
        if peak is None:
            self.peak_marker.set_data([], [])
            x_mm, y_mm, peak_val = 0.0, 0.0, 0.0
        else:
            x_mm, y_mm, peak_val = peak
            self.peak_marker.set_data([x_mm], [y_mm])

        if fz is None:
            fz = float(pred_map.clip(0).sum()) * self.engine.taxel_area

        # query
        query_val = query_val_nmm2
        if self.query_xy_mm is not None:
            qx, qy = self.query_xy_mm
            self.query_marker.set_data([qx], [qy])
            if query_val is None:
                query_val = self.engine.get_taxel_value(pred_map, qx, qy)
        else:
            self.query_marker.set_data([], [])

        # info 패널
        lines = [
            f" Frame: {self._frame}",
            f"",
            f" Peak pos:" if has_peak else " No contact",
            f"   x = {x_mm:+.2f} mm" if has_peak else "",
            f"   y = {y_mm:+.2f} mm" if has_peak else "",
            f"" if has_peak else "",
            f" Peak P: {peak_val:.5f} N/mm²",
            f"",
            f" Fz est: {fz:.4f} N",
        ]
        if query_val is not None:
            qx, qy = self.query_xy_mm
            lines += [
                f"",
                f" Query ({qx:+.1f},{qy:+.1f}):",
                f"   P = {query_val:.5f} N/mm²",
            ]

        self.info_text.set_text("\n".join(lines))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    # ── 이벤트 ────────────────────────────────────────────────────────────────

    def _on_click(self, event) -> None:
        """맵 영역 클릭 시 query 위치 업데이트."""
        if event.inaxes is not self.ax_map:
            return
        x_mm = float(event.xdata)
        y_mm = float(event.ydata)
        # 그리드에 스냅
        x_mm = round(round((x_mm - self.engine.grid_min_mm) / self.engine.grid_step_mm) * self.engine.grid_step_mm + self.engine.grid_min_mm, 4)
        y_mm = round(round((y_mm - self.engine.grid_min_mm) / self.engine.grid_step_mm) * self.engine.grid_step_mm + self.engine.grid_min_mm, 4)
        x_mm = max(GRID_MIN_MM, min(GRID_MAX_MM, x_mm))
        y_mm = max(GRID_MIN_MM, min(GRID_MAX_MM, y_mm))
        self.query_xy_mm = (x_mm, y_mm)
        print(f"[2D] query 위치 선택: ({x_mm:+.2f}, {y_mm:+.2f}) mm")
