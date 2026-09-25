#!/usr/bin/env python3
"""취득 데이터 품질 분석 — meta cache 의 target_ts(실제 학습 프레임) 기준.

health 판정은 **중앙값이 아니라** 물리 근거로 한다(초기 중앙값 판정은 취득 프로토콜 차이를
손상으로 오판했음 — xy1 straight-press 는 저force·0근처 프레임이 많아 중앙값이 낮을 뿐 정상):
  - tare_error : 무접촉(센서<0.5) 프레임의 force 가 0 이 아님 → 로드셀 영점 오류(대개 상수 offset, 복구 가능)
  - dead_sensor: 힘이 실렸는데(fz>0.5) 센서가 안 움직임 → 센서 고장
  - ok         : 무접촉에서 force≈0, 힘 실릴 때 센서 반응
force 범위(최대값)는 health 가 아니라 '커버리지'로 별도 보고(저force = 정상, 커버리지 좁을 뿐).
"""
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
CACHE = REPO / "learning_data/gt_meta_cache_xy_d5d10_g05"
OUT = REPO / "history/fig_data/experiments_archive/data_quality"

TARE_TOL_N = 0.3       # 무접촉 force 가 이보다 크게 벗어나면 tare_error
NOCONTACT_SENS = 0.5   # 이 미만이면 무접촉으로 간주


def trial_files():
    files = []
    for f in sorted(CACHE.glob("*_meta_cache.pt")):
        name = f.name.replace("_870bc4ac6f_meta_cache.pt", "")
        files.append((name, f))
    return sorted(files, key=lambda t: t[0])


def analyze(path: Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    seqs = d["sequences"]
    fz_all, sens_all = [], []
    n_pos, n_pos_total = 0, len(seqs)
    for s in seqs:
        tt = np.asarray(s["target_ts"])
        if tt.size == 0:
            continue
        n_pos += 1
        fz_all.append(np.asarray(s["fz_seq"])[tt].astype(float))
        sens_all.append(np.abs(np.asarray(s["sensor_seq"])[tt]).max(axis=1).astype(float))
    fz = np.concatenate(fz_all); sens = np.concatenate(sens_all)

    # 로드셀 영점: 무접촉 프레임의 force (참값 0 이어야)
    nc = sens < NOCONTACT_SENS
    offset = float(np.median(fz[nc])) if nc.sum() > 50 else 0.0
    fz_corr = fz - offset  # tare 보정 후 force
    # 힘 실릴 때 센서 반응(force-matched): 보정 fz 0.5~1.0N 구간의 센서 중앙
    hi = (fz_corr >= 0.5) & (fz_corr < 1.0)
    sens_at_force = float(np.median(sens[hi])) if hi.sum() > 50 else np.nan
    has_highforce = bool(hi.sum() > 50)

    # health 판정
    if abs(offset) > TARE_TOL_N:
        health = "tare_error"      # 로드셀 영점 오류(대개 복구 가능)
    elif has_highforce and sens_at_force < 2.0:
        health = "dead_sensor"     # 힘 실렸는데 센서 무반응
    else:
        health = "ok"

    return {
        "health": health, "n_frames": fz.size, "n_pos": n_pos, "n_pos_total": n_pos_total,
        "tare_offset": offset, "sens_at_0p5_1N": sens_at_force,
        "fz_max_corr": float(fz_corr.max()), "fz_p90_corr": float(np.percentile(fz_corr, 90)),
        "fz_med_corr": float(np.median(fz_corr)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = trial_files()
    print(f"분석 대상 {len(files)} trial (health=물리근거, force범위=커버리지)\n")
    hdr = (f"{'trial':<34}{'health':>12}{'tare_off':>9}{'sens@.5-1N':>11}"
           f"{'fz_p90':>8}{'fz_max':>8}  (보정 force)")
    print(hdr); print("-" * len(hdr))
    rows = []
    for name, f in files:
        r = analyze(f)
        rows.append((name, r))
        sa = f"{r['sens_at_0p5_1N']:.1f}" if not np.isnan(r["sens_at_0p5_1N"]) else "  -"
        print(f"{name:<34}{r['health']:>12}{r['tare_offset']:>+9.2f}{sa:>11}"
              f"{r['fz_p90_corr']:>8.2f}{r['fz_max_corr']:>8.2f}")

    print("\n=== 판정 요약 ===")
    from collections import Counter
    print(" ", dict(Counter(r["health"] for _, r in rows)))
    for name, r in rows:
        if r["health"] == "tare_error":
            print(f"  ⚠️ {name}: tare_error offset={r['tare_offset']:+.2f}N "
                  f"→ fz+{-r['tare_offset']:.2f} 재보정으로 복구 가능")
        elif r["health"] == "dead_sensor":
            print(f"  ❌ {name}: dead_sensor (fz>0.5 인데 센서 {r['sens_at_0p5_1N']:.1f})")
    # 저커버리지(정상이나 force 범위 좁음) 참고
    low = [n for n, r in rows if r["health"] == "ok" and r["fz_max_corr"] < 1.0]
    if low:
        print(f"  ℹ️ 저force 커버리지(정상, 범위만 좁음): {len(low)}개 — {', '.join(low[:4])}...")

    _save_csv(rows); _plot(rows)


def _save_csv(rows):
    import csv
    keys = ["health", "tare_offset", "sens_at_0p5_1N", "fz_med_corr", "fz_p90_corr",
            "fz_max_corr", "n_frames", "n_pos", "n_pos_total"]
    with open(OUT / "data_quality_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["trial", *keys])
        for name, r in rows:
            w.writerow([name, *[r[k] for k in keys]])
    print("saved:", OUT / "data_quality_summary.csv")


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {"ok": "#2ca25f", "tare_error": "#d62728", "dead_sensor": "#000000"}
    names = [n for n, _ in rows]
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharey=True)

    # 좌: tare offset (0 이어야 정상)
    for (name, r), yi in zip(rows, y):
        axes[0].barh(yi, r["tare_offset"], color=color[r["health"]], alpha=0.85, height=0.7)
    axes[0].axvline(0, color="k", lw=0.8)
    for xv in (-TARE_TOL_N, TARE_TOL_N):
        axes[0].axvline(xv, color="gray", lw=0.6, ls=":")
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=7)
    axes[0].set_xlabel("loadcell tare offset [N]  (no-contact force; 0=OK)")
    axes[0].set_title("Loadcell zero (|offset|>0.3 = tare error)")
    axes[0].grid(axis="x", ls=":", alpha=0.4)

    # 우: force 커버리지 (보정 후 fz max) — health 아님, 정보용
    for (name, r), yi in zip(rows, y):
        axes[1].barh(yi, r["fz_max_corr"], color=color[r["health"]], alpha=0.85, height=0.7)
    axes[1].set_xlabel("force coverage: corrected fz max [N]")
    axes[1].set_title("Force range covered (low = narrow, still valid)")
    axes[1].grid(axis="x", ls=":", alpha=0.4)

    handles = [plt.Line2D([0], [0], marker="s", ls="", color=color[k],
               label={"ok": "OK", "tare_error": "tare error (recoverable)",
                      "dead_sensor": "dead sensor"}[k])
               for k in ("ok", "tare_error", "dead_sensor")]
    axes[1].legend(handles=handles, loc="lower right", fontsize=9, frameon=True)
    fig.suptitle("Acquisition data quality (physics-based: tare + force-matched sensor)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "data_quality_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", OUT / "data_quality_overview.png")


if __name__ == "__main__":
    main()
