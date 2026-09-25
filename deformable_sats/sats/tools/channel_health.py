"""16채널 건강도 진단 — 파손된 taxel 을 취득 직후에 잡아낸다.

barometric taxel 은 과도한 변형·압박으로 파손된다. 이를 모르고 학습하면 조용히 망가진다:

  1. **드리프트 보정 오염** — 세션 *중간*에 고장 나면 앞/뒤 baseline 이 크게 달라져,
     선형 보정이 그 채널에 존재하지 않는 램프를 만들어 넣는다(가짜 변형 신호).
  2. **학습/추론 분포 불일치** — 16채널 정상으로 학습한 SATS 에 14채널만 들어오면
     죽은 taxel 근처 접촉이 살아있는 이웃 쪽으로 끌려가 위치가 편향된다.

실측된 고장 형태는 둘이다:

  | 형태 | 증상 | 판정 근거 |
  |---|---|---|
  | `dead`    | 값이 고착 — 변화 없음 | 같은 창 다른 채널 대비 변동량 |
  | `faulty`  | raw=0 다발(추론 pct −100%) · 밴드 밖 쓰레기 값 | 전채널 median 대비 절대 밴드 |
  | `glitchy` | 위 증상이 있었으나 **회복** — 일시적 드롭아웃 | 세션 끝에서 정상인지 |

★판정 기준을 이렇게 나눈 이유(모두 실측 근거):

  - **변동량은 상대로만 의미 있다.** 접촉 스캔에서 눌린 taxel 의 변동량은 정상적으로
    다른 채널의 **최대 1300배**까지 간다. 절대 기준을 쓰면 정상 접촉이 고장으로 잡힌다.
  - **값의 타당성은 절대로만 의미 있다.** raw=0 이나 쓰레기 값은 오히려 변동량을
    키우므로, 상대 기준으로는 "활발한 채널"로 읽혀 절대 잡히지 않는다.
  - **단발 이상은 고장이 아니다.** 정상 데이터에도 전송 글리치로 밴드 밖 샘플이
    0.6% 존재한다. 그래서 연속 `_MIN_BAD_RUN` 창 이상 지속될 때만 고장으로 본다.
  - **변동량은 이상치에 강해야 한다.** 글리치 몇 개가 표준편차를 부풀리므로,
    유효 샘플만 골라 사분위 범위로 잰다.
  - **마스킹은 영구 고장에만 의미가 있다.** 중간에 수십 초 끊겼다가 회복한 채널을
    빼버리면 멀쩡한 taxel 을 버리는 것이다. 그래서 일시적인 것은 `glitchy` 로
    보고만 한다(마스킹 대상 아님).
  - **단, 끝에서 회복했다고 정상은 아니다.** 세션의 63% 가 무효인데 마지막 창만
    멀쩡한 채널을 "일시적"으로 넘기면 고장을 놓친다(실측 사례). 그래서 영구 판정은
    **세션 끝 상태 또는 나쁜 구간 비율**(`_PERSIST_BAD_FRAC`) 중 하나만 걸려도 성립.

그리고 "고장인가"뿐 아니라 **언제부터 고장인가**(`bad_from_t_s`)를 보고한다 —
세션 앞부분을 살려 쓰기 위해서다.

사용:
    python -m sats.tools.channel_health skin_ws/raw_data/deform/v7
    python -m sats.tools.channel_health learning_data/sensor_raw_bin/ecomesh_v7_xy1
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_SENSORS = 16
_WIN_SEC = 5.0            # 판정 창
_DEAD_RATIO = 0.02        # 같은 창 다른 채널 대비 이 이하면 신호 없음
_WEAK_RATIO = 0.20        # 살아는 있으나 현저히 약함
_VALID_LO, _VALID_HI = 0.5, 1.5   # 전채널 median 대비 유효 raw 밴드
_WIN_VALID_MIN = 0.5      # 창의 절반 이상이 무효면 그 창은 고장
_REPORT_INVALID = 0.02    # 보고에 무효 비율을 덧붙이는 기준(정상은 ~0.6%)
_MIN_BAD_RUN = 3          # ★연속 이만큼 나빠야 고장 — 단발 글리치와 구분
_PERSIST_BAD_FRAC = 0.25  # 세션의 이 비율 이상이 나쁘면, 끝에서 회복해도 영구 고장
_MIN_FRAMES = 200
_SHORT_SESSION_SEC = 30.0

_MASK_LABELS = ("dead", "faulty")


@dataclass(frozen=True)
class ChannelStatus:
    """채널 1개 진단. label ∈ {ok, weak, glitchy, dead, faulty}."""
    index: int                         # 0-based
    label: str
    activity: float                    # 채널 중앙값 대비 변동 비율
    invalid_frac: float = 0.0          # raw 가 유효 밴드를 벗어난 비율
    bad_from_t_s: float | None = None  # 세션 도중 고장 시작 시각(초)

    @property
    def name(self) -> str:
        return f"S{self.index + 1:02d}"

    def describe(self) -> str:
        when = f"@{self.bad_from_t_s:.0f}s" if self.bad_from_t_s is not None else ""
        inv = (f" 무효 {self.invalid_frac * 100:.0f}%"
               if self.invalid_frac > _REPORT_INVALID else "")
        return f"{self.name} {self.label}{when}(활동 {self.activity:.2f}{inv})"


@dataclass(frozen=True)
class HealthReport:
    channels: tuple[ChannelStatus, ...]
    duration_s: float

    def summary(self) -> str:
        note = (f" ★세션 너무 짧음({self.duration_s:.0f}s)"
                if self.duration_s < _SHORT_SESSION_SEC else "")
        bad = [c for c in self.channels if c.label != "ok"]
        if not bad:
            return f"16채널 정상 ({self.duration_s:.0f}s){note}"
        return (f"★이상 {len(bad)}개 — " + " · ".join(c.describe() for c in bad)
                + f" [{self.duration_s:.0f}s]{note}")


def bad_indices(report: HealthReport) -> list[int]:
    """마스킹해야 할 채널. weak 는 제외 — 약하지만 유효한 신호를 낸다."""
    return [c.index for c in report.channels if c.label in _MASK_LABELS]


def _windowed(x: np.ndarray, win: int) -> np.ndarray:
    """[N, C] → [W, win, C]. 창에 못 채우는 뒷부분은 버린다."""
    n_win = max(len(x) // win, 1)
    return x[: n_win * win].reshape(n_win, win, x.shape[1])


def _robust_spread(w: np.ndarray, axis: int = 1) -> np.ndarray:
    """사분위 범위 기반 변동량 — 글리치 몇 개에 흔들리지 않는다(NaN = 무효 샘플)."""
    with warnings.catch_warnings():                # 전부 무효인 채널은 정상적으로 NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        q25, q75 = np.nanpercentile(w, [25, 75], axis=axis)
    return np.nan_to_num((q75 - q25) / 1.349)


def _sustained(mask: np.ndarray, min_run: int) -> np.ndarray:
    """연속 min_run 이상인 구간만 남긴다 — 단발 글리치를 고장으로 오판하지 않도록."""
    out = np.zeros_like(mask)
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_run:
            out[i:j] = True
        i = j
    return out


def _onset(bad_w: np.ndarray, win: int, fs: float) -> float | None:
    """세션 도중 나빠지기 시작한 시각. 처음부터 나빴으면 None."""
    if not bad_w.any() or bad_w[0]:
        return None
    return float(int(np.flatnonzero(bad_w)[0]) * win / fs)


def analyze_channels(raw: np.ndarray, *, fs: float = 200.0, win_sec: float = _WIN_SEC,
                     dead_ratio: float = _DEAD_RATIO, weak_ratio: float = _WEAK_RATIO
                     ) -> HealthReport:
    """raw[N, 16] → 채널별 진단. 판정 근거는 모듈 docstring 참조."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != N_SENSORS:
        raise ValueError(f"raw 는 [N, {N_SENSORS}] 여야 함 — 받은 shape {raw.shape}")
    if len(raw) < _MIN_FRAMES:
        raise ValueError(f"프레임이 너무 짧음({len(raw)} < {_MIN_FRAMES})")

    win = min(max(int(win_sec * fs), 10), len(raw))   # 짧은 세션도 창 1개는 성립
    med_all = float(np.median(raw[raw > 0])) if (raw > 0).any() else 1.0
    valid = (raw > _VALID_LO * med_all) & (raw < _VALID_HI * med_all)
    masked = np.where(valid, raw, np.nan)             # 무효 샘플은 변동량에서 제외

    valid_w = _windowed(valid.astype(float), win).mean(axis=1)      # [W, 16]
    spread_w = _robust_spread(_windowed(masked, win))               # [W, 16]
    ratio_w = spread_w / np.maximum(np.median(spread_w, axis=1, keepdims=True), 1e-12)
    bad_w = (valid_w < _WIN_VALID_MIN) | (ratio_w < dead_ratio)

    spread_all = _robust_spread(masked[None, ...])[0]               # [16]
    overall = spread_all / max(float(np.median(spread_all)), 1e-12)
    invalid_frac = 1.0 - valid.mean(0)

    out = []
    for i in range(N_SENSORS):
        bi = _sustained(bad_w[:, i], min(_MIN_BAD_RUN, len(bad_w)))
        if not bi.any():
            label = "weak" if overall[i] < weak_ratio else "ok"
            out.append(ChannelStatus(i, label, float(overall[i]), float(invalid_frac[i])))
            continue
        # ★끝까지 나쁘거나, 세션의 상당 부분이 나쁘면 영구 고장
        if bi[-1] or float(bi.mean()) > _PERSIST_BAD_FRAC:
            label = "faulty" if float(np.median(valid_w[bi, i])) < _WIN_VALID_MIN else "dead"
        else:
            label = "glitchy"
        out.append(ChannelStatus(i, label, float(overall[i]), float(invalid_frac[i]),
                                 bad_from_t_s=_onset(bi, win, fs)))
    return HealthReport(tuple(out), duration_s=len(raw) / fs)


# ── CLI ────────────────────────────────────────────────────────────────────
def _iter_inputs(root: Path):
    """(이름, raw[N,16], fs) 를 yield — deform 세션과 merged bin 을 모두 지원."""
    from sats.bending.deform_data import discover_sessions, read_due_raw
    try:
        sessions = discover_sessions(root)
    except FileNotFoundError:
        sessions = []
    if sessions:
        for d in sessions:
            t, raw = read_due_raw(d)
            span = float(t[-1] - t[0])
            yield d.name, raw, (len(t) / span if span > 1e-6 else 200.0)
        return
    from sats.preprocessing.merged_bin import open_merged_bin
    bins = sorted(root.rglob("*_merged.bin"))
    if not bins:
        raise FileNotFoundError(f"deform 세션도 merged bin 도 없음: {root}")
    for b in bins:
        _, arr = open_merged_bin(b)
        raw = np.stack([np.asarray(arr[f"s{i}"], float) for i in range(1, 17)], 1)
        yield b.stem, raw, 200.0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    bad_union: set[int] = set()
    for name, raw, fs in _iter_inputs(root):
        try:
            rep = analyze_channels(raw, fs=fs)
        except ValueError as e:
            print(f"[{name}] ★건너뜀 — {e}")
            continue
        print(f"[{name}] {rep.summary()}")
        bad_union |= set(bad_indices(rep))
    if bad_union:
        order = sorted(bad_union)
        print(f"\n★마스킹 대상(합집합): {','.join(f'S{i + 1:02d}' for i in order)}")
        print(f"  학습·추론 양쪽에 동일 적용 — "
              f"--dead-channels {','.join(str(i + 1) for i in order)}")
    else:
        print("\n전 세션 16채널 정상.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
