"""20260629.pptx 의 정량 결과를 atomic 막대 패널로 재렌더링 (Fig.3 SR 벤치마크).

수치 출처: 20260629.pptx
  - slide 26: Indenter 5mm 모델별 위치추정 (Mean xy Err, R2_x, R2_y) / 깊이 z (MAE,RMSE,R2_z)
  - slide 9·23: 소재별 SR — About[x]/[z] (Ours=ECO20+MESH / ECO20 / ECO30 / ECO50)
출력: panels/bench_*.png  (제목 없는 단위 이미지)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panels")
matplotlib.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans",
                            "axes.unicode_minus": False, "savefig.dpi": 200,
                            "savefig.bbox": "tight", "savefig.pad_inches": 0.06})

# ── slide26: 위치추정 리더보드 (Indenter 5mm) ──
LOC = [  # (model, Mean xy Err mm, R2_x, R2_y)
    ("SATS", 0.5771, 0.9909, 0.998), ("Multi-Head", 0.7643, 0.984, 0.9965),
    ("CNN-LSTM", 0.8688, 0.9817, 0.995), ("Unified", 1.8363, 0.9304, 0.9705),
    ("MLP", 2.1142, 0.945, 0.9723), ("CNN", 2.1721, 0.9376, 0.972),
    ("Transformer", 2.2855, 0.9332, 0.9663), ("Isoline-GNN", 2.3377, 0.9341, 0.9638),
    ("GNN-GAT", 2.6334, 0.9114, 0.9481),
]
# ── 소재별 SR R² ──
MAT = ["Ours", "ECO20", "ECO30", "ECO50"]
R2X = [0.9880, 0.9784, 0.9583, 0.9865]
R2Z = [0.7313, 0.6681, 0.6944, 0.6888]
MCOL = {"Ours": "#2e8b57", "ECO20": "#3b6ea5", "ECO30": "#e08a1e", "ECO50": "#9467bd"}


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("[saved]", name)


def panel_loc_leaderboard():
    """모델별 Mean xy Err (낮을수록 우수) — SATS 최저."""
    names = [m[0] for m in LOC]; err = [m[1] for m in LOC]
    cols = ["#2e8b57" if n == "SATS" else "#9bb0c7" for n in names]
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    y = np.arange(len(names))[::-1]
    ax.barh(y, err, color=cols, alpha=0.92)
    for yi, e in zip(y, err):
        ax.text(e + 0.03, yi, f"{e:.2f}", va="center", fontsize=9.5)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Mean xy localization error (mm),  D5")
    ax.set_xlim(0, max(err) * 1.16); ax.grid(axis="x", alpha=0.3)
    ax.text(0.97, 0.06, "lower = better", transform=ax.transAxes, ha="right",
            fontsize=9, style="italic", color="#2e8b57")
    _save(fig, "bench_loc_leaderboard.png")


def panel_material_r2():
    """소재별 SR R² (x 위치 / z 깊이) — Ours(ECO20+MESH) 최고."""
    x = np.arange(len(MAT)); w = 0.38
    fig, ax = plt.subplots(figsize=(4.7, 3.9))
    b1 = ax.bar(x - w / 2, R2X, w, label="R²ₓ (location)", color="#3b6ea5", alpha=0.9)
    b2 = ax.bar(x + w / 2, R2Z, w, label="R²_z (depth)", color="#e08a1e", alpha=0.9)
    for bars, vals in [(b1, R2X), (b2, R2Z)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(MAT)
    ax.set_ylabel("SR R²"); ax.set_ylim(0.6, 1.02)
    ax.legend(fontsize=9, frameon=False, loc="lower center", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.axvspan(-0.5, 0.5, color="#2e8b57", alpha=0.07)
    _save(fig, "bench_material_r2.png")


if __name__ == "__main__":
    panel_loc_leaderboard()
    panel_material_r2()
    print("[done] benchmark panels ->", OUT)
