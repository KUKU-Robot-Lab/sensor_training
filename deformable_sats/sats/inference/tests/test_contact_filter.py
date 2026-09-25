"""ContactFilter — 히스테리시스·디바운스·무접촉 리셋·스무딩 로직."""
import numpy as np

from sats.inference.contact_filter import ContactFilter, FilterConfig
from sats.inference.demo_contacts import Contact


def _c(x=0.0, y=0.0, fz=0.5, z=1.0, pv=80.0):
    return Contact(x_mm=x, y_mm=y, z_mm=z, fz_n=fz, peak_val=pv)


def test_debounce_on_requires_consecutive():
    cf = ContactFilter(FilterConfig(fz_on=0.3, on_frames=2, pos_smooth=1))
    assert cf.update([_c(fz=0.5)]) == ([], False)     # 1프레임 → 아직 no-contact
    out, released = cf.update([_c(fz=0.5)])            # 2프레임 → contact 확정
    assert released is False and len(out) == 1 and cf.contact is True


def test_weak_below_on_stays_no_contact():
    cf = ContactFilter(FilterConfig(fz_on=0.3, on_frames=2, pos_smooth=1))
    for _ in range(10):
        out, released = cf.update([_c(fz=0.1)])        # 총 fz 0.1 < 0.3
        assert out == [] and released is False
    assert cf.contact is False


def test_release_triggers_no_contact_reset():
    cf = ContactFilter(FilterConfig(fz_on=0.3, fz_off=0.15, on_frames=1, off_frames=3, pos_smooth=1))
    cf.update([_c(fz=0.5)])                             # contact ON
    assert cf.contact is True
    assert cf.update([]) == ([], False)                # off 1
    assert cf.update([]) == ([], False)                # off 2
    out, released = cf.update([])                       # off 3 → 릴리즈=리셋
    assert out == [] and released is True and cf.contact is False


def test_hysteresis_midband_holds_contact():
    """fz_off ≤ fz < fz_on 구간은 ON 유지(깜빡임 방지)."""
    cf = ContactFilter(FilterConfig(fz_on=0.3, fz_off=0.15, on_frames=1, off_frames=3, pos_smooth=1))
    cf.update([_c(fz=0.5)])
    for _ in range(10):
        out, released = cf.update([_c(fz=0.2)])         # 0.15 ≤ 0.2 < 0.3 → 유지
        assert released is False and len(out) == 1
    assert cf.contact is True


def test_position_smoothing_reduces_jitter():
    cf = ContactFilter(FilterConfig(fz_on=0.1, on_frames=1, pos_smooth=5, match_mm=4.0))
    xs_in, xs_out = [], []
    for x in [-1, 1, -1, 1, -1, 1, -1, 1]:             # ±1 떨림
        out, _ = cf.update([_c(x=float(x), fz=0.5)])
        xs_in.append(x); xs_out.append(out[0].x_mm)
    assert np.std(xs_out[3:]) < np.std(xs_in[3:])       # 스무딩 후 분산↓


def test_multicontact_tracks_matched_independently():
    cf = ContactFilter(FilterConfig(fz_on=0.1, on_frames=1, pos_smooth=3, match_mm=3.0))
    cf.update([_c(x=-5, fz=0.4), _c(x=5, fz=0.3)])
    out, _ = cf.update([_c(x=-5.4, fz=0.4), _c(x=5.4, fz=0.3)])
    xs = sorted(c.x_mm for c in out)
    assert xs[0] < -3 and xs[1] > 3                     # 두 트랙 분리 유지
