"""변형 크기별 평가 — 평균 하나로 가려지는 성능 차이를 드러낸다."""
import numpy as np
import torch

from sats.bending.deform_data import MAG_BINS, window_magnitude
from sats.bending.train_deform_restorer import evaluate_by_magnitude

W, C = 10, 16


class _IdentitySats:
    """SATS 대역: 입력 pct 를 그대로 작은 맵으로 펼치는 최소 대역품."""

    def __call__(self, x):
        b = x.shape[0]
        return x.mean(dim=1).reshape(b, 4, 4)


class _NoopRestorer(torch.nn.Module):
    def forward(self, x):
        return x * 0.5                      # 절반만 억제하는 더미


def _windows(mags):
    """지정한 **채널 평균 |pct|** 를 갖는 윈도우들."""
    out = []
    for m in mags:
        out.append(np.full((W, C), m, np.float32))
    return np.stack(out)


def _run(val, pool):
    import sats.bending.eval_contact_preservation as E
    orig_map, orig_peak = E._sats_map, E._peak_xy
    E._sats_map = lambda sats, x: sats(x)
    E._peak_xy = lambda m: np.zeros((len(m), 2))
    try:
        return evaluate_by_magnitude(_NoopRestorer(), val, pool, _IdentitySats(), "cpu")
    finally:
        E._sats_map, E._peak_xy = orig_map, orig_peak


def test_splits_into_magnitude_bins():
    val = _windows([1.0] * 20 + [7.0] * 20)           # 0-2% 와 5-10% 구간에만 표본
    res = _run(val, _windows([1.0] * 40))
    assert [(r["mag_lo"], r["mag_hi"]) for r in res] == [(0.0, 2.0), (5.0, 10.0)]
    assert all(r["n_windows"] == 20 for r in res)


def test_skips_bins_with_too_few_samples():
    """표본 8개 미만 구간은 수치가 의미 없으므로 보고하지 않는다."""
    val = _windows([1.0] * 20 + [7.0] * 3)
    res = _run(val, _windows([1.0] * 40))
    assert [(r["mag_lo"], r["mag_hi"]) for r in res] == [(0.0, 2.0)]


def test_single_broken_channel_does_not_dominate_magnitude():
    """★파손 채널 하나가 80% 를 찍어도 '센서 전체 변형'은 그만큼 커지면 안 된다."""
    w = np.zeros((1, W, C), np.float32)
    w[0, :, 5] = 80.0
    assert window_magnitude(w)[0] == 5.0             # 최댓값(80)이 아니라 채널 평균


def test_bins_cover_all_magnitudes():
    assert MAG_BINS[0][0] == 0.0 and np.isinf(MAG_BINS[-1][1])
    for (_, hi), (lo, _) in zip(MAG_BINS[:-1], MAG_BINS[1:]):
        assert hi == lo                                # 경계에 빈틈 없음


def test_contact_contaminated_windows_are_dropped():
    """★10% 초과(손가락 압박)는 학습에서 빠져야 한다 — L_suppress 에 들어가면
    모델이 '접촉은 지워라'를 배운다."""
    from sats.bending.train_deform_restorer import _drop_contact_contaminated
    win = _windows([1.0] * 50 + [14.0] * 7)
    kept = _drop_contact_contaminated(win, 10.0, quiet=True)
    assert len(kept) == 50
    assert window_magnitude(kept).max() <= 10.0


def test_threshold_zero_keeps_everything():
    from sats.bending.train_deform_restorer import _drop_contact_contaminated
    win = _windows([1.0] * 10 + [14.0] * 5)
    assert len(_drop_contact_contaminated(win, 0.0, quiet=True)) == 15


def test_dropout_windows_are_removed():
    """★raw=0 드롭아웃(−100%)이 든 윈도우는 학습에서 빠져야 한다."""
    from sats.bending.train_deform_restorer import drop_dropout_windows
    win = _windows([3.0] * 20)
    win[5, 4, 13] = -100.0                    # S14 한 프레임만 드롭아웃
    kept = drop_dropout_windows(win, quiet=True)
    assert len(kept) == 19


def test_normal_deformation_survives():
    from sats.bending.train_deform_restorer import drop_dropout_windows
    win = _windows([9.5] * 20)                # 강하지만 정상 범위
    assert len(drop_dropout_windows(win, quiet=True)) == 20
