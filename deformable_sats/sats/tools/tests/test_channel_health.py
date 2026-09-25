"""16채널 건강도 진단 — 무신호·폭주·무효값(0/범위이탈)과 발생 시각 탐지."""
import numpy as np
import pytest

from sats.tools.channel_health import analyze_channels, bad_indices

FS = 200.0
BASE = 6.9e6


def _raw(n=6000, n_ch=16, seed=0):
    """정상 센서 raw: 채널마다 다른 진폭의 변동 + 노이즈."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    sig = np.sin(2 * np.pi * t / 3.0)[:, None] * (BASE * 0.02)
    return BASE + sig * np.linspace(0.8, 1.2, n_ch)[None, :] + rng.normal(0, 300, (n, n_ch))


def test_all_healthy_reports_ok():
    r = analyze_channels(_raw(), fs=FS)
    assert all(c.label == "ok" for c in r.channels)
    assert bad_indices(r) == []


def test_rare_zero_glitches_still_ok():
    """정상 데이터에도 0 글리치가 ~0.07% 존재 — 이걸 고장으로 보면 안 된다."""
    x = _raw()
    rng = np.random.default_rng(1)
    x[rng.choice(len(x), 5, replace=False), 3] = 0.0
    assert analyze_channels(x, fs=FS).channels[3].label == "ok"


def test_zero_flooded_channel_is_faulty():
    """★raw=0 다발 → 추론에서 pct −100% 로 나타나는 고장."""
    x = _raw()
    x[:, 10] = 0.0
    r = analyze_channels(x, fs=FS)
    assert r.channels[10].label == "faulty"
    assert bad_indices(r) == [10]


def test_out_of_band_garbage_is_faulty():
    """유효 밴드를 크게 벗어난 쓰레기 값(이상한 숫자)."""
    x = _raw()
    x[:, 7] = BASE * 4.0
    assert analyze_channels(x, fs=FS).channels[7].label == "faulty"


def test_wild_swing_beyond_band_is_faulty():
    """★폭주 — 유효 밴드를 벗어나므로 faulty 로 잡힌다(별도 범주 불필요)."""
    x = _raw()
    rng = np.random.default_rng(2)
    x[:, 14] = BASE + rng.normal(0, BASE * 0.9, len(x))
    r = analyze_channels(x, fs=FS)
    assert r.channels[14].label == "faulty"
    assert 14 in bad_indices(r)


def test_localized_press_is_not_a_fault():
    """★접촉 스캔에서 눌린 taxel 은 다른 채널의 수백 배로 뛴다 — 고장이 아니다."""
    x = _raw()
    t = np.arange(len(x)) / FS
    x[:, 6] += np.where((t > 5) & (t < 25), BASE * 0.08, 0.0)   # 20초간 국소 압박
    r = analyze_channels(x, fs=FS)
    assert r.channels[6].label == "ok"
    assert bad_indices(r) == []


def test_fully_dead_channel_detected():
    x = _raw()
    x[:, 10] = BASE                        # 밴드 안이지만 완전 고착
    r = analyze_channels(x, fs=FS)
    assert r.channels[10].label == "dead"
    assert bad_indices(r) == [10]


def test_failure_midway_reports_onset_time():
    """★세션 중간 고장 — 시각을 알려줘야 앞부분만 살려 쓸 수 있다."""
    x = _raw(n=6000)
    x[3000:, 14] = 0.0                     # 15초 지점부터 무효값
    c = analyze_channels(x, fs=FS).channels[14]
    assert c.label == "faulty"
    assert c.bad_from_t_s is not None and abs(c.bad_from_t_s - 15.0) < 2.0


def test_healthy_channel_has_no_onset_time():
    assert analyze_channels(_raw(), fs=FS).channels[0].bad_from_t_s is None


def test_weak_channel_flagged_but_not_masked():
    x = _raw()
    x[:, 5] = BASE + (x[:, 5] - BASE) * 0.10     # 진폭 10% 로 약화
    r = analyze_channels(x, fs=FS)
    assert r.channels[5].label == "weak"
    assert bad_indices(r) == []                  # weak 는 살아있으므로 마스킹 대상 아님


def test_multiple_bad_channels():
    x = _raw()
    x[:, 10] = 0.0
    x[:, 14] = BASE
    assert bad_indices(analyze_channels(x, fs=FS)) == [10, 14]


def test_short_session_is_flagged_in_summary():
    """★duration 이 비정상적으로 짧은 세션은 '정상'으로 넘기면 안 된다."""
    r = analyze_channels(_raw(n=400), fs=FS)     # 2초
    assert "짧" in r.summary()


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match="16"):
        analyze_channels(np.zeros((300, 8)), fs=FS)


def test_short_input_rejected():
    with pytest.raises(ValueError, match="너무 짧"):
        analyze_channels(_raw(n=50), fs=FS)


def test_summary_lists_bad_channels():
    x = _raw()
    x[:, 10] = 0.0
    s = analyze_channels(x, fs=FS).summary()
    assert "S11" in s and "faulty" in s


def test_transient_dropout_is_glitchy_not_masked():
    """★수십 초 끊겼다 회복한 채널을 마스킹하면 멀쩡한 taxel 을 버리는 것."""
    x = _raw(n=48000)                    # 240초
    x[4000:8000, 9] = 0.0                # 20~40초 드롭아웃 후 회복(세션의 8%)
    r = analyze_channels(x, fs=FS)
    assert r.channels[9].label == "glitchy"
    assert bad_indices(r) == []
    assert r.channels[9].bad_from_t_s is not None


def test_failure_that_persists_to_end_is_masked():
    """같은 크기의 이상이라도 끝까지 지속되면 영구 고장."""
    x = _raw(n=12000)
    x[6000:, 9] = 0.0
    r = analyze_channels(x, fs=FS)
    assert r.channels[9].label == "faulty"
    assert bad_indices(r) == [9]


def test_mostly_bad_channel_is_faulty_even_if_it_recovers_at_end():
    """★세션 대부분이 무효인데 마지막만 멀쩡한 채널을 '일시적'으로 넘기면 안 된다."""
    x = _raw(n=12000)
    x[:9000, 9] = 0.0                    # 75% 무효, 끝에서만 회복
    r = analyze_channels(x, fs=FS)
    assert r.channels[9].label == "faulty"
    assert bad_indices(r) == [9]
