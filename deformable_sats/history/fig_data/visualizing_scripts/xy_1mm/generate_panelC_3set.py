"""Fig.2C 3-set 통계판 — 소재별 3 반복(test set) 평균±std 오차막대 (체크리스트 §2.4, D2).

기존 `generate_panelC_metrics.py`(대표 1 set)의 메트릭 정의를 그대로 재사용하되,
소재/인덴터당 3개 test set 각각을 계산해 **set 간 평균±std** 로 통계 강건성을 보인다.
막대 = 3-set 평균, 오차막대 = set 간 std(n=3), 점 = 개별 set 값(정직 리포팅).

추가로 set 별 **taxel health 스크린**(접촉 중 |ΔS| max)을 남긴다 — dead/stuck 채널이
Active 지표를 편향시킬 수 있으므로 캐빗으로 기록 (알려진 사례: eco50 d5 test1 Skin16 stuck).

실행: history/fig_data/visualizing_scripts/xy_1mm 에서
  ../../../../.venv/bin/python generate_panelC_3set.py --diameter all
산출: Analysis_Results/{d5,d10}/Fig2C_metrics_3set.{png,csv} + taxel_health_3set.csv
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import generate_2d_heatmap as g2  # noqa: E402
from generate_panelC_metrics import ORDER, COLORS, array_metrics  # noqa: E402

BASE = g2.BASE

DATASETS_3SET = {
    "d5": {
        "eco20":   [f"{BASE}/ec020/d5/20260619_test{n}" for n in (5, 6, 7)],
        "eco50":   [f"{BASE}/eco50/d5/20260620_test{n}" for n in (1, 2, 3)],
        "ecomesh": [f"{BASE}/ecomesh/d5/20260622_test{n}" for n in (1, 2, 3)],
    },
    "d10": {
        "eco20":   [f"{BASE}/ec020/d10/20260619_test{n}" for n in (2, 3, 4)],
        "eco50":   [f"{BASE}/eco50/d10/20260620_test{n}" for n in (1, 2, 3)],
        "ecomesh": [f"{BASE}/ecomesh/d10/20260622_test{n}" for n in (4, 5, 6)],
    },
}

DEAD_MAX_DS = 1.0   # % — 접촉 전체에서 |ΔS| max 가 이 미만이면 dead 채널 의심


def taxel_health(df, mask):
    """접촉 구간 taxel별 |ΔS| max → dead 채널 리스트."""
    a = df.loc[mask, [f"dS_{c}" for c in g2.SKIN_COLS]].abs().to_numpy()
    mx = a.max(axis=0)
    dead = [g2.SKIN_COLS[i] for i in np.nonzero(mx < DEAD_MAX_DS)[0]]
    return mx, dead


def run(diameter: str) -> None:
    out_dir = os.path.join(g2.OUT_ROOT, diameter)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n########## Panel C 3-set  diameter={diameter} ##########")

    per_set: dict[str, list[dict]] = {}
    health_rows = []
    for name in ORDER:
        per_set[name] = []
        for path in DATASETS_3SET[diameter][name]:
            tag = os.path.basename(path)
            df, kg_base = g2.load_material(path)
            m = g2.contact_mask(df, kg_base)
            mets = array_metrics(df[m])
            per_set[name].append({k: v[0] for k, v in mets.items()})
            mx, dead = taxel_health(df, m)
            health_rows.append({"material": name, "set": tag,
                                "dead_channels": ";".join(dead) or "-",
                                **{f"max_dS_{c}": round(float(v), 2)
                                   for c, v in zip(g2.SKIN_COLS, mx)}})
            print(f"[{name}/{tag}] contact={int(m.sum())} dead={dead or '-'}  "
                  + "  ".join(f"{k.split(' ')[0]}={v[0]:.2f}" for k, v in mets.items()))

    metric_names = list(per_set[ORDER[0]][0].keys())

    # ---- 막대 (3-set 평균 ± set간 std, 개별 set 점) ----
    fig, axes = plt.subplots(1, len(metric_names), figsize=(16, 4.5), constrained_layout=True)
    for ax, mk in zip(axes, metric_names):
        for xi, name in enumerate(ORDER):
            vals = np.array([s[mk] for s in per_set[name]])
            mean, std = vals.mean(), vals.std(ddof=1)
            ax.bar(xi, mean, yerr=min(std, mean), capsize=4,
                   color=COLORS[name], alpha=0.85)
            ax.scatter([xi] * len(vals), vals, color="black", s=14, zorder=3)
            ax.text(xi, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER)
        ax.set_title(mk, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Fig.2C  Material metrics, 3-set mean ± std ({diameter})  "
                 f"[dots = individual sets]", fontsize=14, fontweight="bold")
    out_png = os.path.join(out_dir, "Fig2C_metrics_3set.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_png}")

    # ---- CSV: 소재×set 값 + 집계 ----
    rows = []
    for name in ORDER:
        for path, s in zip(DATASETS_3SET[diameter][name], per_set[name]):
            rows.append({"material": name, "set": os.path.basename(path), **s})
        vals = pd.DataFrame(per_set[name])
        rows.append({"material": name, "set": "MEAN", **vals.mean().to_dict()})
        rows.append({"material": name, "set": "STD(n=3)", **vals.std(ddof=1).to_dict()})
    out_csv = os.path.join(out_dir, "Fig2C_metrics_3set.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    out_h = os.path.join(out_dir, "taxel_health_3set.csv")
    pd.DataFrame(health_rows).to_csv(out_h, index=False)
    print(f"[saved] {out_h}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--diameter", default="all", choices=["d5", "d10", "all"])
    args = ap.parse_args()
    for d in (["d5", "d10"] if args.diameter == "all" else [args.diameter]):
        run(d)
