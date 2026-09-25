"""통합 대시보드 패널 렌더러 — 주어진 matplotlib ax 위에 그린다.

3종:
  HeatmapPanel  : SATS 초록 압력맵 + 접촉 마커(x,y,z,fz) + 상태 배너. (contacts·bending 공용)
  ThetaGaugePanel: 밴딩각 게이지(유효 20–150° 밴드) + live 값·노이즈. (theta)
  UnitsInset    : 16-taxel(4×4) 원본 미니맵(bending 패널 부속, 선택).

색·마커는 demo_viz 와 통일(초록 램프·고대비 마커). 상태 배너는 demo_contacts.state_banner.
show=False(Agg) 헤드리스에서도 예외 없이 동작(테스트).
"""
from __future__ import annotations

import numpy as np

from sats.bending.geometry import TAXEL_XY_MM
from sats.inference.demo_contacts import Contact, ContactState, StateBanner
from sats.inference.demo_viz import _MARKER_COLORS, _green_cmap, taxel_grid

_THETA_FULL = (0.0, 160.0)     # 게이지 전체 범위
_THETA_VALID = (20.0, 150.0)   # 유효 관측 밴드

# ── 패널 테마 ── 기본 = 라이트(구 데모·mpl 뷰 호환). tk 대시보드는 시작 시
#   apply_dark_gold_theme() 를 호출해 블랙+골드 전시 테마로 전환한다.
#   ★색 정의(demo_contacts·demo_viz)는 그대로 두고 표시 직전에만 매핑 —
#   run_demo 등 라이트 배경 화면들이 영향받지 않는다.
_T = {
    "dark": False,
    "axes_bg": "#ffffff",
    "muted": "#5a6b80",
    "gauge_band": "#0033cc",
    "banner_map": {},              # 라이트용 색 → 다크 배경 가시성 색
    "marker_map": {},
}


def apply_dark_gold_theme() -> None:
    """블랙+골드 전시 테마. 다크 배경에서 안 보이는 색만 명도 보정한다."""
    _T.update(
        dark=True, axes_bg="#0d0d10", muted="#9a958a", gauge_band="#d4af37",
        banner_map={
            "#bbbbbb": "#8f8f96",   # OFFLINE
            "#888888": "#a8a29a",   # NO_CONTACT
            "#d81e00": "#ff5c33",   # CONTACT(적) — 다크에서 채도 유지·명도 업
            "#0033cc": "#6fa0ff",   # BENDING(청)
            "#c8a200": "#e6c34d",   # 진행(황)
        },
        marker_map={
            "#d81e00": "#ff5c33",
            "#0033cc": "#6fa0ff",
            "#111111": "#ffffff",   # 검정 마커는 다크 맵에서 소실 → 흰색
        },
    )


def theme_color(color: str) -> str:
    """라이트 기준 상태색 → 현재 테마 가시성 색(라이트면 그대로)."""
    return _T["banner_map"].get(color, color)


def _gold_cmap():
    """0=근흑 → 최고=거의 흰 골드. 블랙+골드 압력 램프.

    ★밝기(luminance)가 전 구간 단조·광대역이어야 41² 픽셀 값 차이가 그라데이션으로
    보인다 — 구 램프는 중간(a8842a)~상단(d4af37) 밝기 차가 작아 blob 이 단색
    덩어리로 뭉개졌다(실기 피드백)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "sats_gold", ["#0d0d10", "#33230b", "#6b4c12", "#a8842a", "#e0b53f", "#ffefad"])


def _dark_diverging_cmap():
    """raw Δp% 미니맵용 발산 램프 — 0=근흑(다크 카드와 융화), −=청, +=골드.

    RdBu 는 중심이 흰색이라 다크 테마에서 값 0 의 미니맵이 흰 판으로 떠 보인다(실측)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "sats_div_dark", ["#7fb3ff", "#3d5f9e", "#15151a", "#9e7d24", "#ffd75e"])


def _clear_dynamic(ax) -> None:
    """동적 아티스트(마커·라벨) 제거 — imshow(image)는 유지."""
    for txt in list(ax.texts):
        txt.remove()
    for ln in list(ax.lines):
        ln.remove()


def _banner_title(ax, banner: StateBanner, prefix: str) -> None:
    """상태 배너를 색 있는 제목으로(테마 가시성 매핑 적용)."""
    ax.set_title(f"{prefix}\n{banner.label}", fontsize=10,
                 color=theme_color(banner.color), fontweight="bold")


class HeatmapPanel:
    """SATS 초록 압력맵 + 접촉 마커. contacts/bending 공용(bending은 theta 병기).

    flip_y(기본 True): 센서 +y가 물리적으로 아래쪽이면 화면도 아래로 → 손 이동과 마커
    이동 방향 일치(마운팅 방향 보정). flip_x 는 좌우 반전.
    """

    def __init__(self, ax, grid_min_mm: float, grid_max_mm: float, prefix: str = "SATS",
                 draw_without_contacts: bool = False,
                 flip_x: bool = False, flip_y: bool = True, compact: bool = False) -> None:
        self.ax = ax
        self.prefix = prefix
        self.draw_without_contacts = draw_without_contacts
        self.gmin, self.gmax = grid_min_mm, grid_max_mm
        self.ext = [grid_min_mm, grid_max_mm, grid_min_mm, grid_max_mm]
        self.cmap = _gold_cmap() if _T["dark"] else _green_cmap()
        ax.set_facecolor(_T["axes_bg"])
        self.im = ax.imshow(np.zeros((2, 2)), origin="lower", extent=self.ext, cmap=self.cmap,
                            vmin=0, vmax=1.0, aspect="equal", interpolation="bicubic")
        self._vmax_ref: float | None = None      # 절대 스케일 기준(세션 러닝-맥스)
        # 물리 마운팅 방향에 맞춰 축 반전(이미지+마커 함께 반전 → 방향 일치)
        ax.set_xlim(grid_max_mm, grid_min_mm) if flip_x else ax.set_xlim(grid_min_mm, grid_max_mm)
        ax.set_ylim(grid_max_mm, grid_min_mm) if flip_y else ax.set_ylim(grid_min_mm, grid_max_mm)
        if compact:
            # ★S2 미니맵용 — 전체 크기 패널의 축 라벨("y (mm) (down = +)")이
            #   옆의 raw 미니맵을 침범한다(실측). 미니맵은 라벨·눈금 없이 그린다.
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)  (down = +)" if flip_y else "y (mm)")
            # 색 범례 — 관람자가 색=압력 세기임을 알 수 있게(절대 스케일: 세션 최대 기준)
            cb = ax.figure.colorbar(self.im, ax=ax, fraction=0.046, pad=0.03,
                                    ticks=[0.0, 1.0])
            cb.ax.set_yticklabels(["0", "max"])
            cb.set_label("pressure (norm. to session max)", fontsize=7,
                         color=_T["muted"])
            cb.ax.tick_params(colors=_T["muted"], labelsize=7)
            cb.outline.set_edgecolor(_T.get("axes_bg", "#ffffff"))
        _banner_title(ax, StateBanner(ContactState.OFFLINE, "SENSOR OFFLINE",
                                      "#bbbbbb"), prefix)

    def _draw_contacts(self, contacts: list[Contact]) -> None:
        for i, c in enumerate(contacts[:3]):
            col = _T["marker_map"].get(_MARKER_COLORS[i], _MARKER_COLORS[i])
            self.ax.plot([c.x_mm], [c.y_mm], marker="+", color=col, markersize=15,
                         markeredgewidth=2.4, lw=0)
            z = "n/a" if np.isnan(c.z_mm) else f"{c.z_mm:.2f}"
            self.ax.annotate(f"#{i+1} ({c.x_mm:+.1f},{c.y_mm:+.1f})\n z{z} fz{c.fz_n:.2f}N",
                             (c.x_mm, c.y_mm), color=col, fontsize=8, ha="left", va="bottom",
                             xytext=(3, 3), textcoords="offset points")

    def update(self, pred_map, contacts: list[Contact], banner: StateBanner,
               theta_deg: float | None = None, note: str = "",
               show_markers: bool = True) -> None:
        _clear_dynamic(self.ax)
        prefix = self.prefix if theta_deg is None else f"{self.prefix}   θ={theta_deg:+.0f}°"
        if note:
            prefix = f"{prefix}   {note}"
        blank = (banner.state is ContactState.OFFLINE or pred_map is None
                 or (not contacts and not self.draw_without_contacts))
        if blank:
            self.im.set_data(np.zeros((2, 2)))     # OFFLINE·무접촉 → blank
        else:
            pm = np.clip(pred_map, 0, None)
            peak = float(pm.max())
            # ★절대 스케일(세션 러닝-맥스) — 프레임 peak 상대 정규화는 0.3N 이든 3N
            #   이든 항상 풀 스케일로 그려 힘 차이가 색으로 안 보인다(실기 피드백).
            #   세션 최대 peak 을 기준으로 나누면 약접촉=어둡게·강접촉=밝게.
            #   기준은 즉시 상승·완만 감쇠(0.998/frame, 10fps 기준 반감 ~35s) —
            #   센서를 바꿔 약한 세션이 되면 서서히 재적응한다.
            if peak > 1e-6:
                self._vmax_ref = peak if self._vmax_ref is None \
                    else max(peak, self._vmax_ref * 0.998)
            ref = max(self._vmax_ref or peak, 1e-6)
            # 감마 0.7 — blob 스커트(저값)의 색 변화를 키워 픽셀 그라데이션이 보이게.
            # 절대 스케일(ref)은 유지되므로 힘 크기 구분은 그대로다.
            self.im.set_data(np.clip(pm / ref, 0, 1) ** 0.7)
            if contacts and show_markers:
                self._draw_contacts(contacts)
        _banner_title(self.ax, banner, prefix)


class ThetaGaugePanel:
    """밴딩각 게이지 — 유효밴드(20–150°) 음영 + 현재값 마커 + 큰 텍스트."""

    def __init__(self, ax, prefix: str = "bending θ (live)") -> None:
        self.ax = ax
        self.prefix = prefix
        ax.set_xlim(*_THETA_FULL); ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_facecolor(_T["axes_bg"])
        # xlabel 없음 — 제목에 단위가 있고, 라벨이 아래 행(restored 맵 제목)과 겹친다(실측)
        band = _T["gauge_band"]
        ax.axvspan(*_THETA_VALID, color=band, alpha=0.12, zorder=0)  # 유효 관측 밴드
        ax.axvline(_THETA_VALID[0], color=band, lw=1, ls="--", alpha=0.5)
        ax.axvline(_THETA_VALID[1], color=band, lw=1, ls="--", alpha=0.5)
        _banner_title(ax, StateBanner(ContactState.OFFLINE, "SENSOR OFFLINE", "#bbbbbb"), prefix)

    def update(self, theta_deg: float | None, noise: float | None, banner: StateBanner) -> None:
        _clear_dynamic(self.ax)
        if banner.state is ContactState.OFFLINE or theta_deg is None:
            _banner_title(self.ax, banner, self.prefix)
            return
        th = float(np.clip(theta_deg, *_THETA_FULL))
        col = theme_color(banner.color)
        self.ax.plot([th, th], [0, 1], color=col, lw=4, solid_capstyle="round")  # 현재각 바
        self.ax.plot([th], [1.02], marker="v", color=col, markersize=12, clip_on=False)
        ntxt = "" if noise is None else f"  (±{noise:.1f})"
        self.ax.text(0.5, 0.5, f"{theta_deg:+.1f}°{ntxt}", transform=self.ax.transAxes,
                     ha="center", va="center", fontsize=22, fontweight="bold", color=col)
        _banner_title(self.ax, banner, self.prefix)


class UnitsInset:
    """16-taxel(4×4) 원본 미니맵 — bending 패널 부속(밴딩 공간 gradient 관찰)."""

    def __init__(self, ax, title: str = "16-taxel (raw)",
                 flip_x: bool = False, flip_y: bool = True) -> None:
        self.ax = ax
        cmap = _dark_diverging_cmap() if _T["dark"] else "RdBu_r"
        # ★bicubic 보간 — nearest 는 4×4 블록이 각져 보인다(taxel 사이 그라데이션 요청).
        #   셀 중앙 숫자는 그대로 두므로 정량 판독은 유지된다.
        self.im = ax.imshow(np.zeros((4, 4)), origin="lower", cmap=cmap,
                            vmin=-1, vmax=1, extent=[-13, 13, -13, 13], interpolation="bicubic")
        ax.set_xlim(13, -13) if flip_x else ax.set_xlim(-13, 13)   # heatmap 과 방향 일치
        ax.set_ylim(13, -13) if flip_y else ax.set_ylim(-13, 13)
        ax.set_title(title, fontsize=8, color=_T["muted"])
        ax.set_xticks([]); ax.set_yticks([])
        self._ema = None          # 표시 평활(★수시로 색이 바뀌는 깜빡임 억제)
        self._ema_vmax = 1e-6

    def update(self, pct_window, vmax: float | None = None) -> None:
        """vmax: 컬러 스케일 고정값. ★before/after 비교 시 두 미니맵이 같은 스케일을
        써야 한다 — 복원 결과(≈0)를 자체 스케일로 그리면 잔여 노이즈가 꽉 찬 것처럼
        보여 복원이 안 된 것으로 오독된다."""
        _clear_dynamic(self.ax)
        if pct_window is None:
            # ★빈 흰 판은 "복원이 잘 됨"과 구분되지 않는다 — 미장착임을 명시.
            self.im.set_data(np.zeros((4, 4)))
            self.ax.text(0, 0, "select restore", ha="center", va="center",
                         fontsize=7, color="#999999")
            return
        dp = np.asarray(pct_window, float).mean(0)      # [16] 윈도우 평균 Δp%
        # EMA(α=0.25): 프레임 노이즈로 색·숫자가 수시로 바뀌지 않게. 스케일도 함께
        # 평활 — 자동 스케일이 매 프레임 튀면 값이 같아도 색이 흔들린다.
        self._ema = dp if self._ema is None else 0.75 * self._ema + 0.25 * dp
        dp = self._ema
        raw_vmax = vmax if vmax else max(float(np.abs(dp).max()), 1e-6)
        self._ema_vmax = max(0.9 * self._ema_vmax, raw_vmax)   # 상승 즉시·하강 완만
        vmax = self._ema_vmax
        self.im.set_data(taxel_grid(dp)); self.im.set_clim(-vmax, vmax)
        for i in range(16):
            x, y = TAXEL_XY_MM[i + 1]
            if _T["dark"]:   # 중심=근흑 램프 → 작은 값엔 밝은 글자, 극값(밝은 셀)엔 어두운 글자
                col = "#e8e6e0" if abs(dp[i]) < vmax * 0.6 else "#111111"
            else:            # RdBu(중심 흰색) → 반대
                col = "k" if abs(dp[i]) < vmax * 0.6 else "w"
            self.ax.text(x, y, f"{dp[i]:+.1f}", ha="center", va="center", fontsize=6,
                         color=col)
