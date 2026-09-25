#!/usr/bin/env python3
"""잔상(유령 접촉) 진단 로그 분석 — run_dashboard --diag-log 산출물(NDJSON)용.

무접촉인데 접촉이 그려지는 구간에 대해 다음을 판정한다:
  1) 그때 raw 잔차(드리프트 보정 후 |pct|)가 실제로 얼마나 작은가
     → 작은데 fz 가 크면 = SATS 가 미세 잔차 패턴을 증폭(할루시네이션)
  2) 유령 흡수기가 왜 발동하지 않았나 — 조건별 차단 횟수(moved/감쇠/비증가/span)
  3) ±2% 베이스라인 이탈(채널별 raw 궤적)과 drift_ref 추종이 어긋나는 구간

사용:
  .venv/bin/python -m sats.tools.ghost_log_analysis /tmp/ghost.ndjson --out /tmp/ghost_fig
그림 텍스트는 영어(한글 폰트 깨짐 방지).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path, role: str) -> list[dict]:
    rows = []
    for ln in path.read_text().splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("role") == role:
            rows.append(r)
    if not rows:
        raise SystemExit(f"role={role} 레코드 없음: {path}")
    return rows


def summarize_blockers(rows: list[dict]) -> dict[str, int]:
    """흡수기 판정 히스토그램 — 'blocked' 는 세부 원인으로 분해한다."""
    from sats.inference.run_dashboard import SensorChannel as SC
    hist: dict[str, int] = {}
    for r in rows:
        g = r.get("ghost") or {}
        why = g.get("why", "-")
        if why == "blocked":
            if g.get("moved", 0.0) >= SC._GHOST_POS_MM:
                why = "blocked:moved"                 # 위치가 떠돎(드리프트성)
            elif g.get("fzmax_ratio", 1.0) > 1.05:
                why = "blocked:fz_oscillating"        # ★출렁임 — 비증가 조건 위반
            else:
                why = "blocked:decay_insufficient"    # 감쇠 부족(느린경로 span 미달)
        hist[why] = hist.get(why, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def ghost_episodes(rows: list[dict], t0: float) -> list[tuple[float, float, float, float]]:
    """접촉 표시가 이어진 구간 (시작, 끝, 최대 raw잔차%, 최대 fz)."""
    eps, start, rmax, fmax = [], None, 0.0, 0.0
    prev_t = None
    for r in rows:
        t = r["t"] - t0
        # ★기록 공백(>3s) = 세션 경계(append 재실행) — 구간을 끊는다. 안 끊으면
        #   이전 세션 마지막 접촉이 다음 세션까지 이어진 것처럼 보인다(실측 9385s).
        if prev_t is not None and t - prev_t > 3.0 and start is not None:
            eps.append((start, prev_t, rmax, fmax))
            start = None
        prev_t = t
        has = bool(r["post"])
        resid = float(np.max(np.abs(np.asarray(r["raw"]) - np.asarray(r["drift"]))))
        fz = max((c[2] for c in r["post"]), default=0.0)
        if has and start is None:
            start, rmax, fmax = t, resid, fz
        elif has:
            rmax, fmax = max(rmax, resid), max(fmax, fz)
        elif start is not None:
            eps.append((start, t, rmax, fmax))
            start = None
    if start is not None:
        eps.append((start, rows[-1]["t"] - t0, rmax, fmax))
    return eps


def make_figure(rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = rows[0]["t"]
    t = np.array([r["t"] - t0 for r in rows])
    raw = np.array([r["raw"] for r in rows])            # [T,16]
    drift = np.array([r["drift"] for r in rows])
    resid = np.abs(raw - drift).max(axis=1)             # 보정 후 최대 |pct|
    fz_post = np.array([max((c[2] for c in r["post"]), default=np.nan) for r in rows])
    fz_pre = np.array([max((c[2] for c in r["pre"]), default=np.nan) for r in rows])
    has_post = np.array([bool(r["post"]) for r in rows])
    absorbed = np.array([str((r.get("ghost") or {}).get("why", "")).startswith("absorb")
                         for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax = axes[0]
    for i in range(16):
        ax.plot(t, raw[:, i], lw=0.6, alpha=0.6)
    ax.plot(t, drift.mean(axis=1), "k--", lw=1.2, label="drift_ref (mean)")
    ax.set_ylabel("raw pct (%)")
    ax.set_title("16-ch raw window mean vs drift_ref  (baseline wander check)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(t, resid, lw=1.0, color="#7a5f00", label="max |raw - drift| (%)")
    ax.set_ylabel("residual pct (%)")
    ax.set_title("drift-corrected residual amplitude (what SATS actually sees)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    ax.plot(t, fz_pre, lw=0.8, alpha=0.5, label="fz pre-filter (N)")
    ax.plot(t, fz_post, lw=1.2, label="fz displayed (N)")
    ax.fill_between(t, 0, np.nanmax(fz_pre) if np.isfinite(fz_pre).any() else 1.0,
                    where=has_post, alpha=0.10, color="red", label="contact shown")
    if absorbed.any():
        ax.plot(t[absorbed], np.zeros(absorbed.sum()), "g^", ms=8, label="ghost absorbed")
    ax.set_ylabel("fz (N)")
    ax.set_xlabel("time (s)")
    ax.set_title("contact force timeline (red shade = something is drawn)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    print(f"그림 저장: {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--role", default="contacts", choices=["contacts", "bending"])
    ap.add_argument("--out", type=Path, default=None, help="그림 출력 폴더(기본: 로그 옆)")
    a = ap.parse_args()
    rows = load_rows(a.log, a.role)
    t0 = rows[0]["t"]
    dur = rows[-1]["t"] - t0
    print(f"레코드 {len(rows)}개, {dur:.1f}s ({len(rows) / max(dur, 1e-9):.1f} tick/s)")

    eps = ghost_episodes(rows, t0)
    print(f"\n접촉 표시 구간 {len(eps)}개 (시작~끝 s, 최대 raw잔차%, 최대 fz N):")
    for s, e, rmax, fmax in eps:
        tag = " ← 잔상 의심(잔차 3% 미만인데 표시 지속 8s+)" \
            if (e - s) > 8.0 and rmax < 3.0 else ""
        print(f"  {s:7.1f} ~ {e:7.1f}  ({e - s:5.1f}s)  resid {rmax:4.1f}%  fz {fmax:.2f}N{tag}")

    print("\n유령 흡수기 판정 히스토그램:")
    for k, v in summarize_blockers(rows).items():
        print(f"  {k:28s} {v}")

    out = a.out or a.log.parent
    make_figure(rows, Path(out) / f"ghost_{a.role}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
