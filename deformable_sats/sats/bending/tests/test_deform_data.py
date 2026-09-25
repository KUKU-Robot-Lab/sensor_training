"""각도-프리 변형 로더 — 드리프트 보정·구간 분리·윈도우 생성."""
import struct

import numpy as np
import pytest

from sats.bending.deform_data import (
    DeformSession, deform_windows, discover_sessions, load_deform_session,
)

FS = 200.0


def _write_due_v2(path, raw, t):
    """[N,16] raw + t → DUE_V2 bin (burst 10프레임)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"DUE_V2\n")
        for i in range(0, len(raw) - 9, 10):
            f.write(struct.pack("<Q", int(t[i] * 1e9)))
            f.write(raw[i:i + 10].T.astype("<u4").tobytes())


def _session(tmp_path, *, dur_s=60.0, drift=0.001, deform_amp=0.02):
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    base = 6.9e6 * np.ones(16)
    raw = base[None, :] * (1 + (t / t[-1])[:, None] * drift)      # 선형 드리프트
    m = (t > 10) & (t < dur_s - 10)                                # 변형 구간
    amp = np.zeros(n)
    amp[m] = deform_amp * np.sin(2 * np.pi * t[m] / 7)
    raw = raw + base[None, :] * amp[:, None] * np.linspace(-1, 1, 16)[None, :]
    d = tmp_path / "s1"
    _write_due_v2(d / "due_v2_x.bin", raw, t)
    return d


def test_drift_corrected_baseline_near_zero(tmp_path):
    """앞뒤 baseline 구간은 드리프트 보정 후 ≈0% 여야 한다."""
    s = load_deform_session(_session(tmp_path, drift=0.002))
    b = deform_windows(s, 10, include_baseline=True)
    assert len(b) > 0
    assert np.abs(b).mean() < 0.05          # 드리프트(0.2%)가 보정돼 거의 0
    assert s.drift_pct > 0.1                # 원 드리프트는 기록으로 남음


def test_deform_window_has_signal(tmp_path):
    """변형 구간 윈도우는 baseline 구간보다 신호가 훨씬 크다."""
    s = load_deform_session(_session(tmp_path))
    w = deform_windows(s, 10)
    b = deform_windows(s, 10, include_baseline=True)
    assert len(w) > len(b)                  # 변형 구간이 더 김(40s vs 20s)
    assert np.abs(w).mean() > 10 * np.abs(b).mean()


def test_windows_shape_and_stride(tmp_path):
    s = load_deform_session(_session(tmp_path))
    w = deform_windows(s, 10)
    assert w.ndim == 3 and w.shape[1:] == (10, 16)
    w2 = deform_windows(s, 10, stride=5)
    assert len(w2) < len(w)                 # stride 로 샘플 감소


def test_short_session_rejected(tmp_path):
    with pytest.raises(ValueError, match="너무 짧음"):
        load_deform_session(_session(tmp_path, dur_s=15.0))


def test_discover_sessions(tmp_path):
    _session(tmp_path)
    found = discover_sessions(tmp_path)
    assert len(found) == 1 and found[0].name == "s1"


def test_no_due_bin_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_deform_session(tmp_path / "empty")


def test_stage_times_override_fixed_baseline(tmp_path):
    """★가변 단계 프로토콜: session_meta 의 stage_times_s 가 고정 10초보다 우선."""
    import json
    from sats.bending.deform_data import stage_bounds
    d = _session(tmp_path, dur_s=55.0)
    (d / "session_meta.json").write_text(json.dumps(
        {"stage_times_s": {"BASE_HEAD": 0.0, "DEFORM": 7.0, "BASE_TAIL": 45.0}}))
    s = load_deform_session(d)
    assert stage_bounds(s) == (7.0, 45.0)            # 고정값(10.0, 45.0) 아님
    w = deform_windows(s, 10)
    b = deform_windows(s, 10, include_baseline=True)
    assert len(w) > 0 and len(b) > 0
    assert np.abs(w).mean() > 10 * np.abs(b).mean()  # 경계가 맞아야 성립


def test_stage_times_absent_falls_back(tmp_path):
    """메타 없으면 고정 baseline_sec 폴백(구 데이터 호환)."""
    from sats.bending.deform_data import stage_bounds
    s = load_deform_session(_session(tmp_path, dur_s=60.0))
    lo, hi = stage_bounds(s)
    assert lo == 10.0
    assert abs(hi - (float(s.t[-1]) - 10.0)) < 1e-6   # 마지막 샘플 기준(dur−10s)


def test_drift_pct_property():
    s = DeformSession(name="t", sensor_pct=np.zeros((5, 16), np.float32),
                      t=np.arange(5) / FS,
                      baseline_head=np.full(16, 100.0), baseline_tail=np.full(16, 101.0))
    assert abs(s.drift_pct - 1.0) < 1e-6    # +1%


def test_pre_start_frames_are_trimmed(tmp_path):
    """★로거가 시작 전 잔여 프레임을 기록해 시간축이 역행한 세션을 살린다."""
    d = _session(tmp_path, dur_s=60.0)
    bin_p = next(d.glob("due_v2_*.bin"))
    raw_bytes = bin_p.read_bytes()
    head, body = raw_bytes.split(b"\n", 1)
    rec = 8 + 16 * 10 * 4
    bogus = struct.pack("<Q", int(3.7 * 86400 * 1e9)) + body[8:rec]   # 부팅기준 거대 타임스탬프
    bin_p.write_bytes(head + b"\n" + bogus + body)
    s = load_deform_session(d)                       # 이전에는 음수 길이로 실패
    assert float(s.t[-1]) > 50.0
    assert np.all(np.diff(s.t) >= 0)


def test_burst_overlap_jitter_preserves_session(tmp_path):
    """★버스트가 45ms 보다 빠르게 오면 수 ms 역행이 생긴다 — 정상 지터이므로 버리면 안 된다."""
    d = _session(tmp_path, dur_s=60.0)
    bin_p = next(d.glob("due_v2_*.bin"))
    head, body = bin_p.read_bytes().split(b"\n", 1)
    rec = 8 + 16 * 10 * 4
    out = bytearray()
    for i in range(len(body) // rec):                 # 버스트 간격을 40ms 로 압축
        r = bytearray(body[i * rec:(i + 1) * rec])
        r[:8] = struct.pack("<Q", int(i * 0.040 * 1e9))
        out += r
    bin_p.write_bytes(head + b"\n" + bytes(out))
    s = load_deform_session(d)
    assert len(s.sensor_pct) > 10000                  # 세션이 살아남아야 한다
    assert np.all(np.diff(s.t) >= 0)                  # 단조화는 되어야 한다
