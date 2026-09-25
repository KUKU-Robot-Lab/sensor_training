"""통합 대시보드 figure — 3채널 payload를 3패널에 렌더 + 키 이벤트 루프.

레이아웃 GridSpec(2,3, height=[5,1]):
  [0,0] contacts heatmap  [0,1] theta gauge  [0,2] bending→SATS heatmap
  [1,0:2] spec footer(D10)                   [1,2] 16-taxel raw inset

run_dashboard 에서 SensorChannel 리스트를 받아 구동. show=False 면 Agg(헤드리스 테스트).
"""
from __future__ import annotations

import time

from sats.inference.dashboard_panels import (
    HeatmapPanel, ThetaGaugePanel, UnitsInset,
)

# 포스터 footer(영문 — matplotlib 한글 깨짐 회피)
_SPEC_FOOTER = ("SATS 41x41 @ 0.5mm (20x20mm)  |  raw 16-taxel 200Hz / 250k baud"
                "  |  infer 129Hz  |  Fz 0-3.9N  |  theta 20-150 deg (G1 MAE 1.78)")
_CONTACT_CYCLE = (1, 2, 3)


class Dashboard:
    """3채널 통합 뷰. channels = [contacts, theta, bending] SensorChannel."""

    def __init__(self, channels, engine, args, show: bool = True) -> None:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.show = show
        self.args = args
        self.engine = engine
        self.channels = {c.role: c for c in channels}
        self._quit = False
        self._payloads: dict = {}

        self.fig = plt.figure(figsize=(16, 6))
        try:
            self.fig.canvas.manager.set_window_title("SATS v6 — unified demo dashboard")
        except Exception:
            pass
        gs = self.fig.add_gridspec(2, 3, height_ratios=[5, 1], hspace=0.35, wspace=0.25)
        gmin, gmax = engine.grid_min_mm, engine.grid_max_mm
        fx, fy = getattr(args, "flip_x", False), getattr(args, "flip_y", True)  # 마운팅 방향 보정
        self.p_contacts = HeatmapPanel(self.fig.add_subplot(gs[0, 0]), gmin, gmax,
                                       "S1  SATS contacts", flip_x=fx, flip_y=fy)
        self.p_theta = ThetaGaugePanel(self.fig.add_subplot(gs[0, 1]), "S2  bending theta (live)")
        self.p_bending = HeatmapPanel(self.fig.add_subplot(gs[0, 2]), gmin, gmax,
                                      "S3  bending -> SATS", flip_x=fx, flip_y=fy)
        self.p_units = UnitsInset(self.fig.add_subplot(gs[1, 2]), flip_x=fx, flip_y=fy)
        ax_foot = self.fig.add_subplot(gs[1, 0:2]); ax_foot.axis("off")
        ax_foot.text(0.0, 0.7, _SPEC_FOOTER, fontsize=9, va="center", family="monospace")
        self._hint = ax_foot.text(0.0, 0.2, self._hint_str(), fontsize=9, va="center",
                                  color="#0033cc", family="monospace")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        if show:
            plt.ion(); plt.show(block=False)

    def _hint_str(self) -> str:
        """현재 설정 + 키 안내(동적). D10=단일 정밀, D5=다중접촉(2·3) 권장."""
        return (f"contacts={self.args.contacts}  indenter=D{int(self.args.diameter)}   |   "
                f"[c] contacts 1>2>3   [d] d5/d10   [r] 복원 ON/OFF   [b] arm bending   "
                f"[z] re-zero   [q] quit")

    # ── 키 이벤트 ──────────────────────────────────────────────────────────────
    def _on_key(self, event) -> None:
        k = (event.key or "").lower()
        if k == "q":
            self._quit = True
        elif k == "c":
            cur = self.args.contacts
            nxt = _CONTACT_CYCLE[(_CONTACT_CYCLE.index(cur) + 1) % len(_CONTACT_CYCLE)] \
                if cur in _CONTACT_CYCLE else 1
            self.args.contacts = nxt
            print(f"[dashboard] max contacts = {nxt}"
                  + ("   (다중접촉은 [d]로 D5 권장)" if nxt > 1 and self.args.diameter >= 7.5 else ""))
        elif k == "r":
            ch = self.channels["bending"]
            if ch.restorer is None:
                print("[dashboard] 변형 복원기 없음 — train_deform_restorer 로 학습 필요")
            else:
                on = ch.toggle_restore()
                print(f"[dashboard] 변형 복원 {'ON' if on else 'OFF'}"
                      + ("  (유령이 사라져야 함)" if on else "  (유령이 보여야 함)"))
        elif k == "d":
            self.args.diameter = 5.0 if self.args.diameter >= 7.5 else 10.0
            self.args.min_distance_mm = self.args.diameter
            setf = getattr(self.engine, "set_diameter", None)
            if setf is not None:
                setf(self.args.diameter)                     # size 조건(_size_t) 교체
            print(f"[dashboard] indenter = D{int(self.args.diameter)} "
                  f"(min-dist {self.args.diameter:g}mm, size 조건 갱신)")
        elif k == "b":
            print("\n[dashboard] bending 재장착 — 무접촉·고정 유지(~1s)...")
            self.channels["bending"].arm_bending()
            ch = self.channels["bending"]
            print(f"[dashboard] armed. theta={ch.theta_fixed:+.1f} deg (bent-baseline 고정)")
        elif k == "z":
            print("\n[dashboard] theta 재영점...")
            self.channels["theta"].rezero_theta()
            print("[dashboard] theta0 갱신")

    # ── 렌더 ──────────────────────────────────────────────────────────────────
    def render_once(self) -> None:
        """현재 _payloads 를 3패널에 반영(1회)."""
        pc = self._payloads.get("contacts")
        if pc is not None:
            self.p_contacts.update(pc["pred_map"], pc["contacts"], pc["banner"], theta_deg=None)
        pt = self._payloads.get("theta")
        if pt is not None:
            self.p_theta.update(pt["theta"], pt["noise"], pt["banner"])
        pb = self._payloads.get("bending")
        if pb is not None:
            self.p_bending.update(pb["pred_map"], pb["contacts"], pb["banner"],
                                  theta_deg=pb.get("theta"), note=pb.get("note", ""))
            self.p_units.update(pb.get("units"))
        self._hint.set_text(self._hint_str())          # 현재 contacts/d 설정 반영
        if self.show:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)

    def poll_all(self) -> None:
        self._payloads = {role: ch.poll() for role, ch in self.channels.items()}

    def run(self) -> None:
        print("\n[dashboard] 구동 — 창에서 [b]/[z]/[q]. (Ctrl+C 도 종료)\n")
        infer_int = 0.0 if self.args.infer_max_fps <= 0 else 1.0 / self.args.infer_max_fps
        viz_int = 0.0 if self.args.viz_fps <= 0 else 1.0 / self.args.viz_fps
        last_infer, last_viz = 0.0, 0.0
        try:
            while not self._quit:
                now = time.time()
                if now - last_infer >= infer_int:
                    self.poll_all(); last_infer = now
                if now - last_viz >= viz_int:
                    self.render_once(); last_viz = now
                self.plt.pause(0.005)          # GUI 키 이벤트 처리
        except KeyboardInterrupt:
            print("\n[dashboard] 종료")

    def save(self, path) -> None:
        self.fig.savefig(path, dpi=110, bbox_inches="tight")
