"""통합 대시보드 헤드리스(Agg) 테스트 — 패널 렌더 + Dashboard 구동 없이 1틱 렌더."""
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sats.inference.dashboard import Dashboard  # noqa: E402
from sats.inference.dashboard_panels import (  # noqa: E402
    HeatmapPanel, ThetaGaugePanel, UnitsInset,
)
from sats.inference.demo_contacts import Contact, state_banner  # noqa: E402

GRID = 41


def _map(cx, cy, amp=80.0, sig=1.5):
    c = np.linspace(-10, 10, GRID); xx, yy = np.meshgrid(c, c)
    return amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))


def test_heatmap_panel_contact_and_blank(tmp_path):
    fig = plt.figure(); ax = fig.add_subplot(111)
    p = HeatmapPanel(ax, -10, 10, "S1")
    # 접촉 → 맵 표시(peak>0)
    p.update(_map(-3, 2), [Contact(-3, 2, 1.4, 3.0, 80)], state_banner([Contact(-3, 2, 1.4, 3.0, 80)]))
    assert float(np.asarray(p.im.get_array()).max()) > 0.0
    # 무접촉 → blank(0)
    p.update(_map(0, 0), [], state_banner([]))
    assert float(np.asarray(p.im.get_array()).max()) == 0.0
    fig.savefig(tmp_path / "hm.png"); plt.close(fig)


def test_heatmap_panel_flip_y_reverses_axis():
    """flip_y=True → y축 반전(위=−, 아래=+), flip_x=False → x 정상."""
    fig = plt.figure(); ax = fig.add_subplot(111)
    HeatmapPanel(ax, -10, 10, "S", flip_x=False, flip_y=True)
    ylo, yhi = ax.get_ylim()
    assert ylo > yhi                          # 반전(위쪽이 작은 값)
    xlo, xhi = ax.get_xlim()
    assert xlo < xhi                          # x 정상
    plt.close(fig)
    fig2 = plt.figure(); ax2 = fig2.add_subplot(111)
    HeatmapPanel(ax2, -10, 10, "S", flip_y=False)
    assert ax2.get_ylim()[0] < ax2.get_ylim()[1]   # 반전 해제 시 정상
    plt.close(fig2)


def test_theta_gauge_panel_renders(tmp_path):
    fig = plt.figure(); ax = fig.add_subplot(111)
    p = ThetaGaugePanel(ax)
    p.update(92.0, 0.7, state_banner([], theta_deg=92.0))
    assert any("92" in t.get_text() for t in ax.texts)
    p.update(None, None, state_banner(None, connected=False))   # OFFLINE 안전
    fig.savefig(tmp_path / "th.png"); plt.close(fig)


def test_units_inset_renders():
    fig = plt.figure(); ax = fig.add_subplot(111)
    p = UnitsInset(ax)
    p.update(np.random.randn(10, 16).astype(np.float32))
    assert np.asarray(p.im.get_array()).shape == (4, 4)
    p.update(None)                                              # None 안전
    plt.close(fig)


class _StubChannel:
    """poll()가 고정 payload 반환. Dashboard 렌더 경로 검증용."""

    def __init__(self, role, payload):
        self.role = role
        self._p = payload
        self.reader = None
        self.theta_fixed = 0.0
        self.rezeroed = False

    def poll(self):
        return self._p

    def arm_bending(self):
        self.theta_fixed = 90.0

    def rezero_theta(self):
        self.rezeroed = True


def _args():
    return SimpleNamespace(infer_max_fps=20.0, viz_fps=10.0,
                           contacts=1, diameter=10.0, min_distance_mm=10.0)


class _StubEngine(SimpleNamespace):
    def __init__(self):
        super().__init__(grid_min_mm=-10.0, grid_max_mm=10.0, diameter_set=None)

    def set_diameter(self, d):
        self.diameter_set = d


def _engine():
    return _StubEngine()


def test_dashboard_renders_offline_and_data(tmp_path):
    cs = [Contact(-3, 2, 1.4, 3.0, 80)]
    channels = [
        _StubChannel("contacts", {"kind": "heatmap", "banner": state_banner(cs),
                                  "pred_map": _map(-3, 2), "contacts": cs, "theta": None, "units": None}),
        _StubChannel("theta", {"kind": "theta", "banner": state_banner([], theta_deg=92.0),
                               "theta": 92.0, "noise": 0.6}),
        _StubChannel("bending", {"kind": "heatmap", "banner": state_banner(cs, theta_deg=90.0),
                                 "pred_map": _map(4, -1), "contacts": cs, "theta": 90.0,
                                 "units": np.random.randn(10, 16).astype(np.float32)}),
    ]
    dash = Dashboard(channels, _engine(), _args(), show=False)
    dash.poll_all()
    dash.render_once()
    out = tmp_path / "dash.png"; dash.save(out)
    assert out.exists() and out.stat().st_size > 0


def test_dashboard_keys_arm_and_rezero():
    channels = [
        _StubChannel("contacts", {"kind": "heatmap", "banner": state_banner([]),
                                  "pred_map": None, "contacts": [], "theta": None, "units": None}),
        _StubChannel("theta", {"kind": "theta", "banner": state_banner([]), "theta": 0.0, "noise": 0.0}),
        _StubChannel("bending", {"kind": "heatmap", "banner": state_banner([]),
                                 "pred_map": None, "contacts": [], "theta": None, "units": None}),
    ]
    dash = Dashboard(channels, _engine(), _args(), show=False)
    dash._on_key(SimpleNamespace(key="b"))
    dash._on_key(SimpleNamespace(key="z"))
    dash._on_key(SimpleNamespace(key="q"))
    assert dash.channels["bending"].theta_fixed == 90.0
    assert dash.channels["theta"].rezeroed is True
    assert dash._quit is True


def test_sensor_channel_connect_disconnect_and_busy(monkeypatch):
    """UI 런타임 연결/해제 + baseline/busy 배너 상태."""
    from types import SimpleNamespace as NS
    import sats.inference.run_dashboard as rd

    class FakeReader:
        baseline_ready = False
        baseline_progress = 0.4
        baseline = None
        def stop(self): self.stopped = True
        def get_latest_window_with_seq(self): return None, 0

    monkeypatch.setattr(rd, "_build_reader", lambda port, args, w: FakeReader())
    args = NS(diameter=10.0, contacts=1, min_distance_mm=10.0, rel_threshold=0.3,
              min_fz=0.1, theta_smooth=7, theta_deadband=20.0,
              fz_on=0.2, fz_off=0.1, on_frames=2, off_frames=5, pos_smooth=3)
    engine = NS(window_size=10)
    ch = rd.SensorChannel("contacts", None, engine, None, None, args)
    assert ch.poll()["banner"].label == "SENSOR OFFLINE"
    assert ch.connect("auto") is None and ch.connected
    assert "BASELINE 40%" in ch.poll()["banner"].label      # 수집 중 표시
    ch.busy = "arming..."
    assert "ARMING" in ch.poll()["banner"].label            # 작업 중 표시
    ch.busy = ""
    ch.disconnect()
    assert not ch.connected and ch.poll()["banner"].label == "SENSOR OFFLINE"


def test_dashboard_toggle_contacts_and_diameter():
    """[c] contacts 1→2→3→1, [d] d10↔d5(min-dist·엔진 size 조건 동반)."""
    channels = [_StubChannel(r, {"kind": "heatmap", "banner": state_banner([]),
                                 "pred_map": None, "contacts": [], "theta": None, "units": None})
                for r in ("contacts", "theta", "bending")]
    args, eng = _args(), _engine()
    dash = Dashboard(channels, eng, args, show=False)
    assert args.contacts == 1
    for expect in (2, 3, 1):
        dash._on_key(SimpleNamespace(key="c"))
        assert args.contacts == expect
    # d10 → d5 → d10, 엔진 size 조건·min-dist 동반
    dash._on_key(SimpleNamespace(key="d"))
    assert args.diameter == 5.0 and args.min_distance_mm == 5.0 and eng.diameter_set == 5.0
    dash._on_key(SimpleNamespace(key="d"))
    assert args.diameter == 10.0 and eng.diameter_set == 10.0
    assert "contacts=1" in dash._hint_str() and "D10" in dash._hint_str()
