#!/usr/bin/env python3
"""
label_preview.py

깊이 기반 접촉 라벨 히트맵을 소량 생성하여 PNG로 저장하는 스크립트.
기존 파이프라인을 변경하지 않고 검증용으로만 사용한다.

예:
  python3 preprocessing/label_preview.py --grid-file preprocessing/processed_data/grid/ecemesh_d5_1_grid.csv --samples 3
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.utils.contact_geometry import contact_radius
from training.utils.contact_label import make_radial_label
from preprocessing.preprocess import GRID_STEP_MM, GRID_MIN_MM, GRID_MAX_MM


def parse_args():
    p = argparse.ArgumentParser(description="깊이 기반 라벨 히트맵 프리뷰 생성")
    p.add_argument("--grid-file", type=Path, required=True, help="*_grid.csv 경로")
    p.add_argument("--out-dir", type=Path, default=Path("preprocessing/processed_data/label_preview"))
    p.add_argument("--samples", type=int, default=3, help="시각화할 샘플 수")
    p.add_argument("--kernel", choices=["gaussian", "linear"], default="gaussian")
    p.add_argument("--sigma-scale", type=float, default=1.0, help="Gaussian 시 sigma = a * scale")
    p.add_argument("--radius-model", choices=["hertz", "geo"], default="hertz")
    p.add_argument("--indenter-radius-mm", type=float, default=2.5)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.grid_file)
    if df.empty:
        print("grid 파일이 비어 있습니다.")
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 깊이가 큰 순으로 샘플 선택
    df_sorted = df.sort_values("z_depth_mm", ascending=False).head(args.samples)

    for i, row in df_sorted.iterrows():
        a_mm = contact_radius(
            float(row["z_depth_mm"]),
            R_mm=args.indenter_radius_mm,
            model=args.radius_model,
        )
        label = make_radial_label(
            center_xy_mm=(row["x_mm"], row["y_mm"]),
            grid_min_mm=GRID_MIN_MM,
            grid_max_mm=GRID_MAX_MM,
            grid_step_mm=GRID_STEP_MM,
            radius_mm=a_mm,
            kernel=args.kernel,
            sigma_scale=args.sigma_scale,
        )
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(
            label,
            extent=[GRID_MIN_MM, GRID_MAX_MM, GRID_MIN_MM, GRID_MAX_MM],
            origin="lower",
            cmap="magma",
        )
        ax.scatter([row["x_mm"]], [row["y_mm"]], color="cyan", s=30, label="center")
        ax.set_title(f"{args.grid_file.stem} | z={row['z_depth_mm']:.3f}mm, a={a_mm:.3f}mm")
        ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight")
        out_path = out_dir / f"{args.grid_file.stem}_sample{i}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
