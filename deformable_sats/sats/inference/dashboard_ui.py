"""Tkinter 통합 데모 UI — 패널별 센서 컨트롤(포트 선택·연결/끊기·리셋) + 캔버스.

레이아웃(1창):
  [ 상단바 ] 제목 · 접촉 수(1/2/3) · 인덴터(D5/D10) · 종료
  [ 3패널 ] S1 SATS contacts | S2 bending theta(live) | S3 bending(jig)+SATS
            각 패널 = 상태배너 + matplotlib 캔버스 + (포트▾ · 연결/끊기 · 리셋)
  [ 하단바 ] 스펙 요약(D10 · 41²/81² · 200Hz …)

리셋 의미: S1=접촉 필터 초기화 · S2=flat 재영점 · S3=재장착(무접촉 유지, θ+bent-baseline).
S2/S3 리셋은 ~1초 캡처라 스레드로 실행(UI 비차단, 배너에 진행 표시).
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sats.inference.dashboard_panels import HeatmapPanel, ThetaGaugePanel, UnitsInset

# ── 디자인 토큰(전시용) ── 블랙+골드 계기판 테마(연구실 CI).
#    단색 배경에 기본 tk 위젯을 그대로 두면 시제품처럼 보인다 — 크롬(흑)/카드(암회)/
#    상태·압력(골드) 역할로 색을 분리한다. 압력맵·게이지 색은 dashboard_panels 의
#    apply_dark_gold_theme() 가 함께 전환한다.
_C = {
    "win":       "#000000",   # 창 배경(카드 사이 여백으로 보임)
    "card":      "#141418",   # 패널 카드(암회 — 순흑 위에서 한 단 떠 보임)
    "field":     "#1d1d23",   # 입력창/콤보 내부
    "border":    "#33333c",   # 축 스파인·경계
    "header":    "#000000",   # 헤더/푸터(순흑)
    "header_fg": "#f2efe6",
    "header_dim": "#9a958a",
    "accent":    "#d4af37",   # 골드
    "text":      "#e8e6e0",
    "muted":     "#9a958a",
}
_BG = _C["card"]              # 패널 내부(캔버스·컨트롤) 배경
_ACCENT = _C["accent"]


def list_port_choices() -> list[str]:
    """포트 드롭다운 후보 — auto + 현재 보이는 시리얼 포트(Linux ttyACM/USB, Windows COM)."""
    try:
        from serial.tools import list_ports
        devs = [p.device for p in list_ports.comports()
                if any(k in p.device for k in ("ttyACM", "ttyUSB"))
                or p.device.upper().startswith("COM")]
    except ImportError:
        devs = []
    return ["auto"] + devs


class PanelUI(ttk.Frame):
    """패널 1개 = 배너 + 캔버스 + 컨트롤(포트·연결·리셋) + 설정 요약."""

    def __init__(self, master, channel, title: str, flip_x: bool, flip_y: bool,
                 grid_min: float, grid_max: float, engines=None, sensors=None) -> None:
        super().__init__(master, padding=10)
        self.channel = channel
        self.title = title
        self.engines = engines                      # EngineCache — 센서 교체용

        # 상태 배너
        self.banner_var = tk.StringVar(value="SENSOR OFFLINE")
        self.banner = tk.Label(self, textvariable=self.banner_var, font=("TkDefaultFont", 11, "bold"),
                               fg="#888888", bg=_BG, anchor="center")
        self.banner.pack(fill="x", pady=(0, 4))

        # matplotlib 캔버스(역할별 구성)
        self.fig = Figure(figsize=(4.6, 4.8), dpi=90, facecolor=_C["card"])
        role = channel.role
        if role == "theta":
            # ★S2 = theta 게이지(위) + [변형 raw %(4×4) | 복원 후 SATS 맵](아래).
            #   복원 결과는 taxel 값이 아니라 SATS 압력맵(run_demo 2d 와 동일 형태)으로 —
            #   "복원하면 SATS 가 flat 을 본다(유령 없음)"가 그대로 보인다.
            gs = self.fig.add_gridspec(2, 2, height_ratios=[1.6, 1.5], hspace=0.45, wspace=0.2)
            self.p_theta = ThetaGaugePanel(self.fig.add_subplot(gs[0, :]), title)
            self.p_units = UnitsInset(self.fig.add_subplot(gs[1, 0]),
                                      title="raw (deformed) %", flip_x=flip_x, flip_y=flip_y)
            self.p_restored_map = HeatmapPanel(self.fig.add_subplot(gs[1, 1]),
                                               grid_min, grid_max, "restored -> SATS",
                                               draw_without_contacts=True,
                                               flip_x=flip_x, flip_y=flip_y, compact=True)
            self.p_units_restored = None
            self.p_heat = None
        else:
            # S1(FLAT)·S3(DEFORMED) = SATS 맵 전체 크기
            self.p_heat = HeatmapPanel(self.fig.add_subplot(111), grid_min, grid_max, title,
                                       flip_x=flip_x, flip_y=flip_y)
            self.p_units = self.p_theta = None
            self.p_units_restored = None
            self.p_restored_map = None
        # 축 타이포 통일 — 기본 검정 스파인/틱은 계기판 톤과 어긋난다
        for ax in self.fig.axes:
            for sp in ax.spines.values():
                sp.set_color(_C["border"])
            ax.tick_params(colors=_C["muted"], labelsize=8)
            ax.xaxis.label.set_color(_C["muted"])
            ax.yaxis.label.set_color(_C["muted"])
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ★컨트롤 2행 구성 — 한 줄에 다 넣으면 패널 폭을 넘겨 뒤쪽 버튼이 잘린다.
        #   1행(row): port · 연결 · 리셋 (동작)
        #   2행(sel): v* · run/est · restore (모델 선택)
        row = ttk.Frame(self); row.pack(fill="x", pady=(6, 0))
        sel = ttk.Frame(self); sel.pack(fill="x", pady=(3, 0))
        self.sensor_var = self.run_var = None
        self.run_box = None
        # ★파손 채널 수동 지정(1-based, 예 7,11,15) — Enter 로 적용, 표시 통계에서 0 고정
        ttk.Label(sel, text="dead").pack(side="left")
        self.dead_var = tk.StringVar(value="")
        dbox = ttk.Entry(sel, textvariable=self.dead_var, width=8)
        dbox.pack(side="left", padx=(2, 8))
        dbox.bind("<Return>", lambda e: self._apply_dead())
        dbox.bind("<FocusOut>", lambda e: self._apply_dead())
        ttk.Label(row, text="port").pack(side="left")
        self.port_var = tk.StringVar(value="auto")
        self.port_box = ttk.Combobox(row, textvariable=self.port_var, width=16,
                                     values=list_port_choices(), state="readonly")
        self.port_box.pack(side="left", padx=4)
        self.port_box.bind("<Button-1>", lambda e: self.port_box.configure(values=list_port_choices()))
        if engines is not None and sensors and role == "theta":
            # ★S2: 다른 센서의 theta(est) + deform 복원기 선택.
            #   restore 콤보는 v* 선택 시 runs/deform/<v*>/ 스캔으로 갱신된다.
            from sats.inference import model_registry as registry
            ttk.Label(sel, text="v*").pack(side="left")
            self.sensor_var = tk.StringVar(value="")
            sbox = ttk.Combobox(sel, textvariable=self.sensor_var, width=3,
                                values=list(sensors), state="readonly")
            sbox.pack(side="left", padx=2)
            sbox.bind("<<ComboboxSelected>>", lambda e: self._set_theta_sensor())
            ttk.Label(sel, text="est").pack(side="left", padx=(6, 0))
            self.est_var = tk.StringVar(value="")
            self._est_map = {lab: str(p) for lab, p in registry.estimators().items()}
            ebox = ttk.Combobox(sel, textvariable=self.est_var, width=8,
                                values=list(self._est_map), state="readonly")
            ebox.pack(side="left", padx=2)
            ebox.bind("<<ComboboxSelected>>", lambda e: self._apply_estimator())
            ttk.Label(sel, text="restore").pack(side="left", padx=(6, 0))
            self.restore_var = tk.StringVar(value="")
            self._restore_map = {lab: str(p)
                                 for lab, p in registry.deform_restorers().items()}
            self.restore_box = ttk.Combobox(sel, textvariable=self.restore_var, width=14,
                                            values=list(self._restore_map), state="readonly")
            self.restore_box.pack(side="left", padx=2)
            self.restore_box.bind("<<ComboboxSelected>>", lambda e: self._apply_restorer())
        if engines is not None and sensors and role in ("contacts", "bending"):
            ttk.Label(sel, text="v*").pack(side="left")
            self.sensor_var = tk.StringVar(value="")
            sbox = ttk.Combobox(sel, textvariable=self.sensor_var, width=3,
                                values=list(sensors), state="readonly")
            sbox.pack(side="left", padx=2)
            sbox.bind("<<ComboboxSelected>>", lambda e: self._on_sensor_selected())
            ttk.Label(sel, text="run").pack(side="left", padx=(6, 0))
            self.run_var = tk.StringVar(value="")
            self.run_box = ttk.Combobox(sel, textvariable=self.run_var, width=18,
                                        values=[], state="readonly")
            self.run_box.pack(side="left", padx=2)
            self.run_box.bind("<<ComboboxSelected>>", lambda e: self._apply_run())
            if role == "bending":
                ttk.Label(sel, text="restore").pack(side="left", padx=(6, 0))
                self.restore_var = tk.StringVar(value="")
                self._restore_map = {}
                self.restore_box = ttk.Combobox(sel, textvariable=self.restore_var, width=7,
                                                values=[], state="readonly")
                self.restore_box.pack(side="left", padx=2)
                self.restore_box.bind("<<ComboboxSelected>>", lambda e: self._apply_restorer())
                self._refresh_restorers(None)        # v* 선택 전에는 전체 목록
        self.btn_conn = ttk.Button(row, text="연결", width=6, command=self._toggle_connect)
        self.btn_conn.pack(side="left", padx=2)
        reset_label = {"contacts": "리셋", "theta": "재영점", "bending": "재장착"}[role]
        self.btn_reset = ttk.Button(row, text=reset_label, width=7, command=self._reset_async)
        self.btn_reset.pack(side="left", padx=2)
        # ★S3 변형 복원 토글 — 데모의 '변형 → 정지 → 복원' 구분 동작
        self.btn_restore = None
        if role == "bending" and getattr(channel, "restorer", None) is not None:
            self.btn_restore = ttk.Button(row, width=9, command=self._toggle_restore)
            self.btn_restore.pack(side="left", padx=2)
            self._sync_restore_label()

        # 설정 요약(파트별 세팅 정리)
        self.info_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.info_var, font=("TkDefaultFont", 8),
                  style="Muted.TLabel").pack(fill="x", pady=(4, 0))

    # ── 컨트롤 동작 ──────────────────────────────────────────────────────────
    def _toggle_connect(self) -> None:
        if (not self.channel.connected and self.channel.role in ("contacts", "bending")
                and self.channel.engine is None):
            print(f"[dashboard] {self.channel.role}: 먼저 v* 와 run 을 선택하세요")
            return
        ch = self.channel
        if ch.connected:
            ch.disconnect()
            self.btn_conn.configure(text="연결")
            return
        err = ch.connect(self.port_var.get())
        if err:
            self.banner_var.set(err)
            self.banner.configure(fg="#c22")
        else:
            self.btn_conn.configure(text="끊기")

    def _apply_dead(self) -> None:
        spec = self.dead_var.get().strip()
        err = self.channel.set_dead_channels(spec)
        if err:
            print(f"[dashboard] {self.channel.role} dead: {err}")
        elif spec:
            print(f"[dashboard] {self.channel.role} dead 채널 = {spec} (pct 0 고정)")

    def _refresh_restorers(self, sensor: str | None) -> None:
        """restore 콤보 갱신 — v* 선택 시 runs/deform/<v*>/ 의 pth 전부."""
        from sats.inference import model_registry as registry
        if getattr(self, "restore_box", None) is None:
            return
        self._restore_map = {lab: str(p)
                             for lab, p in registry.deform_restorers(sensor).items()}
        self.restore_box.configure(values=list(self._restore_map))
        if self.restore_var.get() not in self._restore_map:
            self.restore_var.set("")
        if not self._restore_map and sensor:
            print(f"[dashboard] {sensor}: 복원기 없음 — runs/deform/{sensor}/ 에 pth 배치")

    def _set_theta_sensor(self) -> None:
        sensor = self.sensor_var.get()
        err = self.channel.apply_theta_sensor(sensor, self.engines)
        self._refresh_restorers(sensor)
        print(f"[dashboard] theta 센서 {sensor}"
              + (f" — {err}" if err else " (복원 SATS 맵 활성)"))

    def _apply_estimator(self) -> None:
        path = getattr(self, "_est_map", {}).get(self.est_var.get(), self.est_var.get())
        err = self.channel.apply_estimator(path)
        print(f"[dashboard] {self.channel.role}: estimator {self.est_var.get()}"
              + (f" — {err}" if err else " 적용"))

    def _apply_restorer(self) -> None:
        path = getattr(self, "_restore_map", {}).get(self.restore_var.get(),
                                                     self.restore_var.get())
        err = self.channel.apply_restorer(path)
        if err:
            print(f"[dashboard] {self.channel.role}: {err}")
        if self.btn_restore is not None:
            self.btn_restore.configure(state="normal" if self.channel.restorer else "disabled")
            self._sync_restore_label()

    def _on_sensor_selected(self) -> None:
        """v* 선택 → runs/sats/<v*>/ 스캔으로 추론 파일 목록(첫 항목 자동 선택).

        ★콤보에는 축약 라벨(g05nsc 등)만 — 실제 경로는 _run_map 이 들고 있다.
        """
        from sats.inference import model_registry as registry
        sensor = self.sensor_var.get()
        self._run_map = {lab: str(p) for lab, p in registry.sats_runs(sensor).items()}
        self.run_box.configure(values=list(self._run_map))
        if self.channel.role == "bending":
            self._refresh_restorers(sensor)
        if self._run_map:
            self.run_var.set(next(iter(self._run_map)))
            self._apply_run()
        else:
            self.run_var.set("")
            print(f"[dashboard] {sensor}: SATS 체크포인트 없음 — runs/sats/{sensor}/ 에"
                  " (config.json 과 함께) 배치하거나 학습 여부 확인")

    def _apply_run(self) -> None:
        """run(추론 파일) 확정 — 엔진 로드(캐시) + 복원기 유효성 갱신."""
        sensor, run = self.sensor_var.get(), self.run_var.get()
        run = getattr(self, "_run_map", {}).get(run, run)   # 축약 라벨 → 체크포인트 경로
        if not sensor or not run:
            return
        err = self.channel.apply_run(sensor, run, self.engines)
        print(f"[dashboard] {self.channel.role}: {sensor}/{run}"
              + (f" — {err}" if err else " 적용"))
        if self.btn_restore is not None:
            state = "normal" if self.channel.restorer is not None else "disabled"
            self.btn_restore.configure(state=state)
            self._sync_restore_label()

    def _sync_restore_label(self) -> None:
        if self.btn_restore is not None:
            self.btn_restore.configure(
                text="복원 ON" if self.channel.restore_on else "복원 OFF")

    def _toggle_restore(self) -> None:
        self.channel.toggle_restore()
        self._sync_restore_label()

    def _reset_async(self) -> None:
        if not self.channel.connected or self.channel.busy:
            return
        threading.Thread(target=self.channel.reset, daemon=True).start()

    # ── 렌더 ─────────────────────────────────────────────────────────────────
    def render(self, payload: dict, args) -> None:
        from sats.inference.dashboard_panels import theme_color
        b = payload["banner"]
        self.banner_var.set(b.label)
        self.banner.configure(fg=theme_color(b.color))
        role = self.channel.role
        if role == "theta":
            self.p_theta.update(payload.get("theta"), payload.get("noise"), b)
            if self.p_units is not None:
                self.p_units.update(payload.get("units"))          # 센서 % 그대로(자체 스케일)
            if getattr(self, "p_restored_map", None) is not None:
                self.p_restored_map.update(payload.get("restored_map"), [], b)
            self.info_var.set(f"smooth={args.theta_smooth}  deadband={args.theta_deadband:g}°"
                              f"  (valid 20–150°)")
        else:
            theta = payload.get("theta") if role == "bending" else None
            self.p_heat.update(payload.get("pred_map"), payload.get("contacts", []), b,
                               theta_deg=theta, note=payload.get("note", ""),
                               show_markers=getattr(args, "show_markers", True))
            if self.p_units is not None:
                u = payload.get("units")
                shared = (max(float(abs(__import__("numpy").asarray(u, float)).max()), 1e-6)
                          if u is not None else None)   # raw 기준 공유 스케일
                self.p_units.update(u, vmax=shared)
                if getattr(self, "p_units_restored", None) is not None:
                    ur = payload.get("units_restored")
                    self.p_units_restored.update(ur, vmax=shared)
                    if ur is not None:
                        import numpy as _np
                        self.p_units_restored.ax.set_title(
                            f"restored (res {float(_np.abs(ur).max()):.2f}%)", fontsize=8)
            extra = (f"  armed θ={self.channel.theta_fixed:+.1f}°" if role == "bending"
                     and self.channel.armed and self.channel.restorer is None else "")
            self.info_var.set(f"contacts≤{args.contacts}  D{int(args.diameter)}"
                              f"  fz_on={args.fz_on:g}N{extra}")
        if self.channel.connected:
            self.btn_conn.configure(text="끊기")
        else:
            self.btn_conn.configure(text="연결")
        self.canvas.draw_idle()


class DashboardApp:
    """Tk 루트 — 3 PanelUI + 전역 설정(접촉 수·인덴터) + 폴링 루프."""

    def __init__(self, channels, engine, args, engines=None) -> None:
        from sats.inference.dashboard_panels import apply_dark_gold_theme
        apply_dark_gold_theme()                  # ★패널 생성 전에 — cmap·축색 전환
        from sats.inference.model_registry import list_sensors
        self.engines = engines
        self.sensors = list_sensors() if engines is not None else []
        self.args = args
        self.engine = engine
        self.channels = {c.role: c for c in channels}
        self.root = tk.Tk()
        self.root.title("SATS — unified demo dashboard")
        self.root.configure(bg=_C["win"])
        # 전시용 전체화면(1920x1080) — F11 토글, Esc 해제(콤보 조작 등은 창 모드가 편함)
        if getattr(args, "fullscreen", False):
            self.root.attributes("-fullscreen", True)
        self.root.bind("<F11>", lambda e: self.root.attributes(
            "-fullscreen", not self.root.attributes("-fullscreen")))
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        # ttk 기본 배경 = 카드 — 패널 내부 위젯이 전부 카드 위에 얹힌다(다크)
        style.configure(".", background=_C["card"], foreground=_C["text"])
        style.configure("TButton", padding=(10, 4), relief="flat",
                        background="#26262e", foreground=_C["text"],
                        bordercolor=_C["border"])
        style.map("TButton", background=[("active", "#32323c")],
                  foreground=[("disabled", "#5c5c64")])
        for w in ("TCombobox", "TEntry"):
            style.configure(w, fieldbackground=_C["field"], foreground=_C["text"],
                            background=_C["card"], bordercolor=_C["border"],
                            arrowcolor=_C["muted"], insertcolor=_C["text"])
        style.map("TCombobox", fieldbackground=[("readonly", _C["field"])],
                  foreground=[("readonly", _C["text"])])
        style.configure("Muted.TLabel", foreground=_C["muted"])
        # 콤보 펼침 목록(tk Listbox)도 다크로
        self.root.option_add("*TCombobox*Listbox*Background", _C["field"])
        self.root.option_add("*TCombobox*Listbox*Foreground", _C["text"])
        self.root.option_add("*TCombobox*Listbox*selectBackground", _C["accent"])
        self.root.option_add("*TCombobox*Listbox*selectForeground", "#000000")

        # ── 헤더(순흑): 제목(골드) + 전역 토글 ──
        top = tk.Frame(self.root, bg=_C["header"], padx=16, pady=10)
        top.pack(fill="x")
        tk.Label(top, text="Tactile Super-Resolution — Live",
                 font=("TkDefaultFont", 15, "bold"), fg=_C["accent"],
                 bg=_C["header"]).pack(side="left")
        tk.Button(top, text="종료", command=self.root.destroy, bg=_C["header"],
                  fg=_C["header_fg"], activebackground="#1d3a5c",
                  activeforeground="#ffffff", relief="flat", bd=0, padx=12,
                  highlightthickness=0).pack(side="right")

        def _hdr_radio(text, value, variable, command):
            return tk.Radiobutton(top, text=text, value=value, variable=variable,
                                  command=command, bg=_C["header"], fg=_C["header_fg"],
                                  selectcolor=_C["header"], activebackground=_C["header"],
                                  activeforeground="#ffffff", relief="flat",
                                  highlightthickness=0)
        # 인덴터 토글
        self.dia_var = tk.StringVar(value=f"D{int(args.diameter)}")
        for d in ("D10", "D5"):
            _hdr_radio(d, d, self.dia_var, self._set_diameter).pack(side="right", padx=1)
        tk.Label(top, text="indenter", fg=_C["header_dim"], bg=_C["header"]
                 ).pack(side="right", padx=(18, 4))
        # 접촉 수 토글
        self.n_var = tk.IntVar(value=args.contacts)
        for n in (3, 2, 1):
            _hdr_radio(str(n), n, self.n_var, self._set_contacts).pack(side="right", padx=1)
        tk.Label(top, text="contacts", fg=_C["header_dim"], bg=_C["header"]
                 ).pack(side="right", padx=(18, 4))
        # 마커(+ 좌표·힘 주석) 표시 토글 — 깨끗한 압력맵만 보여주는 데모 모드
        args.show_markers = True
        self.marker_var = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="markers", variable=self.marker_var,
                       command=lambda: setattr(self.args, "show_markers",
                                               bool(self.marker_var.get())),
                       bg=_C["header"], fg=_C["header_fg"], selectcolor=_C["header"],
                       activebackground=_C["header"], activeforeground="#ffffff",
                       relief="flat", highlightthickness=0).pack(side="right", padx=(18, 4))

        tk.Frame(self.root, bg=_C["accent"], height=2).pack(fill="x")  # 골드 구분선

        # ── 3패널(암회 카드, 순흑 창 배경이 카드 사이 여백으로 보임) ──
        mid = tk.Frame(self.root, bg=_C["win"], padx=10, pady=10)
        mid.pack(fill="both", expand=True)
        gmin, gmax = ((engine.grid_min_mm, engine.grid_max_mm) if engine is not None
                      else (-10.0, 10.0))          # 물리 범위는 모든 run 동일
        fx, fy = getattr(args, "flip_x", False), getattr(args, "flip_y", True)
        titles = {"contacts": "S1  FLAT SATS", "theta": "S2  theta (deg) + restore",
                  "bending": "S3  DEFORMED SATS"}
        self.panels: dict[str, PanelUI] = {}
        for i, role in enumerate(("contacts", "theta", "bending")):
            p = PanelUI(mid, self.channels[role], titles[role], fx, fy, gmin, gmax,
                        engines=self.engines, sensors=self.sensors)
            p.grid(row=0, column=i, sticky="nsew", padx=6)
            mid.columnconfigure(i, weight=1)
            self.panels[role] = p
        mid.rowconfigure(0, weight=1)

        tk.Frame(self.root, bg=_C["accent"], height=2).pack(fill="x")  # 골드 구분선

        # ── 푸터(순흑): 스펙 요약 ──
        grid_txt = (f"{engine.grid_size}x{engine.grid_size}@{engine.grid_step_mm:g}mm"
                    if engine is not None else "run 선택 대기")
        self._spec = (f"SATS {grid_txt} (20x20mm)   ·   raw 16-taxel 200 Hz / 250 kbaud   ·   "
                      f"Fz 0–3.9 N (D10)   ·   theta 20–150° (MAE 1.78°)")
        self.spec_var = tk.StringVar(value=self._spec)
        tk.Label(self.root, textvariable=self.spec_var, font=("TkDefaultFont", 9),
                 fg=_C["header_dim"], bg=_C["header"], padx=16, pady=6, anchor="w"
                 ).pack(fill="x")

        self._tick_ms = max(30, int(1000 / max(args.viz_fps, 1)))
        # ★성능 계측 — 느린 PC 에서 "무엇이" 느린지(추론 poll vs 캔버스 draw) 푸터에
        #   보여준다. 틱이 목표 주기를 넘으면 재영점·게이지 반응까지 굼떠 보이므로
        #   원인 분리는 현장에서 즉시 필요하다.
        self._perf = {"poll": 0.0, "draw": 0.0, "n": 0, "t_last": 0.0}
        self._idle_drawn: set = set()          # 미연결 패널 재렌더 생략용

    # ── 전역 설정 ────────────────────────────────────────────────────────────
    def _set_contacts(self) -> None:
        self.args.contacts = int(self.n_var.get())

    def _set_diameter(self) -> None:
        d = 5.0 if self.dia_var.get() == "D5" else 10.0
        self.args.diameter = d
        self.args.min_distance_mm = d
        engines = ({self.engine} if self.engine is not None else set()) \
            | {c.engine for c in self.channels.values() if c.engine is not None}
        for e in engines:                            # ★패널별 엔진 전부에 적용
            setf = getattr(e, "set_diameter", None)
            if setf is not None:
                setf(d)

    # ── 루프 ─────────────────────────────────────────────────────────────────
    def _tick(self) -> None:
        import time as _time
        t0 = _time.perf_counter()
        for role, ch in self.channels.items():
            try:
                payload = ch.poll()
            except Exception as e:                 # 리더 급단선 등 — 패널만 오프라인
                ch.disconnect()
                from sats.inference.demo_contacts import state_banner
                payload = {"kind": "heatmap", "banner": state_banner(None, connected=False),
                           "pred_map": None, "contacts": [], "theta": None,
                           "noise": None, "units": None}
                print(f"[ui] {role} 리더 오류 → 연결 해제: {e}")
            t1 = _time.perf_counter()
            # 미연결 패널은 첫 OFFLINE 렌더 후 생략 — 약한 PC 에서 draw 비용 절약
            if ch.connected:
                self._idle_drawn.discard(role)
                self.panels[role].render(payload, self.args)
            elif role not in self._idle_drawn:
                self.panels[role].render(payload, self.args)
                self._idle_drawn.add(role)
            t2 = _time.perf_counter()
            self._perf["poll"] += t1 - t0
            self._perf["draw"] += t2 - t1
            t0 = t2
        self._perf["n"] += 1
        now = _time.perf_counter()
        if now - self._perf["t_last"] > 2.0 and self._perf["n"]:
            n = self._perf["n"]
            poll_ms = self._perf["poll"] / n * 1000
            draw_ms = self._perf["draw"] / n * 1000
            fps = n / max(now - self._perf["t_last"], 1e-6) \
                if self._perf["t_last"] else 1000 / self._tick_ms
            self.spec_var.set(f"{self._spec}   ·   tick {poll_ms + draw_ms:.0f}ms "
                              f"(infer {poll_ms:.0f} + draw {draw_ms:.0f}) {fps:.1f}fps")
            self._perf.update(poll=0.0, draw=0.0, n=0, t_last=now)
        self.root.after(self._tick_ms, self._tick)

    def run(self) -> None:
        self.root.after(self._tick_ms, self._tick)
        self.root.mainloop()
        for ch in self.channels.values():
            ch.disconnect()
