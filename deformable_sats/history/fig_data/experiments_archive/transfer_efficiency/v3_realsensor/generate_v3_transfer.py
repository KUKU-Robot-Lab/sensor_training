#!/usr/bin/env python3
"""Real-sensor (ecomesh v3) cross-sensor calibration-transfer confirmation figure.

Resolves the honest caveat of the 2026-07-19 transfer rehearsal (`../transfer_report.md`):
that study was a PROXY (same acquisition rig, different material/fold; holdout was a
within-sensor trial split). v3 is the first REAL new sensor with a genuine held-out
trial: train = test1 (d5+d10), holdout = test2 (entirely unseen during training).

Metric = SATS map quality on the test2 holdout (contact fz>0.3N), which is the trustworthy
scale-invariant signal (NOT rel RMSE — that is a low-force artifact). Numbers come from
`scripts/reeval_map_quality.eval_model` on the two runs below.

Data → figure mapping (this dict is the single source of truth; edit here to regenerate):
  before_transfer  = source model  sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3
                     evaluated on v3 test2 holdout (transfer NOT applied)
  after_transfer   = sats/training/runs/ecomesh_v3_calibtransfer_warm (2 trial warm-start)
                     evaluated on the same v3 test2 holdout

Run:  .venv/bin/python history/fig_data/experiments_archive/transfer_efficiency/v3_realsensor/generate_v3_transfer.py
Out:  v3_realsensor_transfer.png  (English labels only — matplotlib has no Korean glyph)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- single source of truth: holdout (test2) map quality, fz>0.3N ---
RESULTS = {
    "d5": {
        "before": {"loc_mm": 2.062, "peak_corr": 0.701},
        "after": {"loc_mm": 0.707, "peak_corr": 0.802},
    },
    "d10": {
        "before": {"loc_mm": 2.062, "peak_corr": 0.735},
        "after": {"loc_mm": 1.414, "peak_corr": 0.789},
    },
}
INDENTERS = ["d5", "d10"]
BEFORE_COLOR, AFTER_COLOR = "#b0b0b0", "#2a7fff"


def _grouped_bars(ax, metric: str, ylabel: str, title: str) -> None:
    x = np.arange(len(INDENTERS))
    w = 0.36
    before = [RESULTS[d]["before"][metric] for d in INDENTERS]
    after = [RESULTS[d]["after"][metric] for d in INDENTERS]
    b1 = ax.bar(x - w / 2, before, w, label="before transfer (source model)", color=BEFORE_COLOR)
    b2 = ax.bar(x + w / 2, after, w, label="after transfer (v3 warm, 2 trials)", color=AFTER_COLOR)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([d + " indenter" for d in INDENTERS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def main() -> None:
    out = Path(__file__).resolve().parent / "v3_realsensor_transfer.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    _grouped_bars(ax1, "loc_mm", "contact localization error (mm)",
                  "Localization (lower = better)")
    _grouped_bars(ax2, "peak_corr", "peak correlation (GT vs pred)",
                  "Peak correlation (higher = better)")
    fig.suptitle("ecomesh v3 real-sensor calibration transfer — test2 holdout (unseen), fz>0.3N",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
