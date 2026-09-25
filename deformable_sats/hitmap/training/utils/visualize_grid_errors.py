import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from training.dataset_zarr import ZarrDataset
from training.models.mlp_sr import MLPSR

def visualize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, default="training/runs_sr_ecomesh/grid_errors.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로드 (Val split)
    GRID_MIN = -9.75
    val_ds = ZarrDataset(args.zarr_path, split="val", val_ratio=0.2, seed=42)
    val_tactile = val_ds.tactile_data.to(device)
    val_radius = (val_ds.aux_data[:, 3:4] / 2.0).to(device)
    val_cx = val_ds.cx_data.cpu().numpy()
    val_cy = val_ds.cy_data.cpu().numpy()
    val_cz = val_ds.depth_data.cpu().numpy()

    # 2. 모델 로드 및 추론
    ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    tn = ckpt.get("target_norm")

    # 모델 구조 추론 (기본값 또는 체크포인트 기반)
    model = MLPSR(in_dim=17, out_dim=3, hidden=[512, 512, 256, 128, 64]).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        pred_norm = model(val_tactile, val_radius)
        
    if tn:
        pred_x = pred_norm[:, 0].cpu().numpy() * (tn["x_max"] - tn["x_min"]) + tn["x_min"]
        pred_y = pred_norm[:, 1].cpu().numpy() * (tn["y_max"] - tn["y_min"]) + tn["y_min"]
        pred_z = pred_norm[:, 2].cpu().numpy() * (tn["z_max"] - tn["z_min"]) + tn["z_min"]
    else:
        pred_x, pred_y, pred_z = pred_norm[:, 0].cpu().numpy(), pred_norm[:, 1].cpu().numpy(), pred_norm[:, 2].cpu().numpy()

    # 3. 오차 집계
    df = pd.DataFrame({
        'x': val_cx,
        'y': val_cy,
        'err_x': np.abs(pred_x - val_cx),
        'err_y': np.abs(pred_y - val_cy),
        'err_z': np.abs(pred_z - val_cz)
    })

    grid_stats = df.groupby(['y', 'x']).mean().reset_index()

    # 4. 시각화 (3 Subplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['X-axis MAE (mm)', 'Y-axis MAE (mm)', 'Z-depth MAE (mm)']
    metrics = ['err_x', 'err_y', 'err_z']
    cmaps = ['Reds', 'Blues', 'Greens']

    for i, (ax, metric, title, cmap) in enumerate(zip(axes, metrics, titles, cmaps)):
        pivot = grid_stats.pivot(index='y', columns='x', values=metric)
        im = ax.imshow(pivot.values, cmap=cmap, origin='lower', extent=[GRID_MIN, 9.75, GRID_MIN, 9.75])
        ax.set_title(title)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        fig.colorbar(im, ax=ax, label='Error (mm)')

    plt.tight_layout()
    plt.savefig(args.out_path, dpi=150)
    print(f"Visualization saved to: {args.out_path}")

if __name__ == "__main__":
    visualize()
