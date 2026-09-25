"""응력완화 파일럿 분석 — 지그에 한 각도로 고정·유지한 raw 기록의 완화 곡선.

목적: 긴 hold 취득이 유효한지 판정. 신호가 (a)일정수준 정착=각도정보 유지→학습 가능,
(b)0으로 수렴=정보 소실→모델 불가(peak-hold만). τ(시간상수)로 dwell 길이도 산출.

취득: 센서 flat 몇 초 → 지그 고정(한 각도) → 그대로 60초+ 유지. due_raw_burst 기록.
실행: .venv/bin/python scripts/analyze_bend_relaxation.py --dir <기록폴더> [--flat-sec 3]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sats.preprocessing.bin_merge import load_due_bin


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="due_raw_burst_*.bin 있는 기록 폴더")
    p.add_argument("--flat-sec", type=float, default=3.0, help="앞부분 flat(무밴딩) 구간 초 → baseline")
    p.add_argument("--out", default=None, help="완화 곡선 png(기본 <dir>/relaxation.png)")
    args = p.parse_args()
    d = Path(args.dir)
    due = load_due_bin(glob.glob(str(d / "due_raw_burst_*.bin"))[0])
    s = np.asarray(due.sensors, np.float64)          # [N,16]
    t = np.asarray(due.time_s, np.float64); t -= t[0]
    base = s[t < args.flat_sec].mean(0)              # flat 프리픽스 평균
    dev = np.abs((s - base) / base * 100.0).mean(1)  # 프레임별 평균 |Δp|%

    # 클램프 시점 = 신호가 flat 노이즈의 5배 넘는 첫 지점
    flat_noise = dev[t < args.flat_sec].mean()
    onset = int(np.argmax(dev > max(flat_noise * 5, 0.3)))
    t0 = t[onset]
    peak = float(dev[onset:onset + 20].max())        # 클램프 직후 최대(fresh)
    tail = float(np.median(dev[t > t[-1] - 5.0]))     # 마지막 5초 중앙(정착)
    frac = tail / max(peak, 1e-9)
    # 지수완화 τ 추정: dev(t)=tail+(peak-tail)exp(-(t-t0)/τ) → 63% 지점
    after = dev[onset:]; ta = t[onset:] - t0
    target = tail + (peak - tail) * np.exp(-1)
    tau = float(ta[np.argmax(after <= target)]) if np.any(after <= target) else float("nan")

    print(f"기록 {t[-1]:.0f}초, {len(s)}프레임.  flat 노이즈 {flat_noise:.2f}%")
    print(f"클램프 t0={t0:.1f}s  fresh peak |Δp|={peak:.2f}%  정착(끝5s)={tail:.2f}%")
    print(f"★정착/fresh 비율 = {frac*100:.0f}%   완화 시간상수 τ≈{tau:.1f}s")
    if frac > 0.4:
        print(f"→ 판정: 정착 신호가 각도정보 유지({frac*100:.0f}%>40%). **긴 hold 취득 유효**.")
        print(f"   권장 dwell = ~{max(3*tau,10):.0f}s/스텝(3τ, steady 도달). 학습=dwell 끝 1τ 프레임.")
    else:
        print(f"→ 판정: 정착이 fresh의 {frac*100:.0f}%로 낮음(<40%). 정보 소실 위험 →")
        print(f"   모델 hold 추정 어려움, **peak-hold 표시** 권장. (angle=magnitude 한계)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(args.out) if args.out else d / "relaxation.png"
        plt.figure(figsize=(9, 4))
        plt.plot(t, dev, lw=0.8)
        plt.axvline(t0, color="g", ls="--", lw=0.8, label="clamp")
        plt.axhline(peak, color="r", ls=":", lw=0.8, label="fresh peak")
        plt.axhline(tail, color="b", ls=":", lw=0.8, label="settled")
        plt.xlabel("time (s)"); plt.ylabel("mean |dp| (%)")
        plt.title("bend stress-relaxation: settled/fresh=%.0f%%  tau~%.1fs" % (frac * 100, tau))
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out, dpi=120)
        print(f"곡선 저장: {out}")
    except Exception as e:  # noqa: BLE001
        print(f"(플롯 생략: {e})")


if __name__ == "__main__":
    main()
