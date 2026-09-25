"""demo_viz 헤드리스(Agg) 렌더 테스트 — 창 없이 save 되는지."""
import numpy as np

from sats.inference.demo_contacts import Contact
from sats.inference.demo_viz import DemoViz2D, DemoViz3D


def _map(cx, cy):
    c = np.linspace(-10, 10, 41); xx, yy = np.meshgrid(c, c)
    return 80.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.5 ** 2))


def test_viz2d_renders_multicontact(tmp_path):
    v = DemoViz2D(-10, 10, show=False)
    contacts = [Contact(-5, 0, 1.4, 3.2, 80), Contact(5, 0, 1.2, 2.1, 56)]
    v.update(_map(-5, 0) + _map(5, 0), contacts, best_map=_map(-5, 0) + _map(5, 0), best_contacts=contacts)
    out = tmp_path / "v2d.png"; v.save(out)
    assert out.exists() and out.stat().st_size > 0


def test_viz2d_with_theta_and_empty_contacts(tmp_path):
    v = DemoViz2D(-10, 10, show=False)
    v.update(np.zeros((41, 41)), [], theta_deg=12.5)
    out = tmp_path / "v2d_empty.png"; v.save(out)
    assert out.exists()


def test_viz2d_live_blank_when_no_contact():
    """무접촉(contacts=[]): 신호가 있어도 live는 빈 화면(0)으로 표시."""
    v = DemoViz2D(-10, 10, show=False)
    v.update(_map(0, 0), [])                     # 접촉 신호 있으나 검출 0건
    assert float(np.asarray(v.im_live.get_array()).max()) == 0.0
    assert v.ax_live.get_title().startswith("live (no contact)")
    # 접촉 검출되면 다시 맵 표시
    v.update(_map(0, 0), [Contact(0, 0, 1.4, 3.0, 80)])
    assert float(np.asarray(v.im_live.get_array()).max()) > 0.0


def test_viz3d_renders(tmp_path):
    v = DemoViz3D(-10, 10, 41, show=False)
    v.update(_map(0, 0), [Contact(0, 0, 1.5, 3.0, 80)], theta_deg=-8.0)
    out = tmp_path / "v3d.png"; v.save(out)
    assert out.exists() and out.stat().st_size > 0
