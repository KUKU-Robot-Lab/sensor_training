import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from training.dataset_zarr import ZarrDataset
from training.models.mlp_ff import MLPFF

def visualize_ff():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str, required=True)
    parser.add_argument("--ff-model-path", type=str, required=True)
    parser.add_argument("--sr-model-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, default="training/runs_ff_zarr/ff_grid_errors.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로드
    val_ds = ZarrDataset(args.zarr_path, split="val", val_ratio=0.2, seed=42)
    val_tactile = val_ds.tactile_data.to(device)
    val_radius = (val_ds.aux_data[:, 3:4] / 2.0).to(device)
    val_fz_raw = val_ds.fz_data.cpu().numpy()
    val_cx = val_ds.cx_data.cpu().numpy()
    val_cy = val_ds.cy_data.cpu().numpy()

    # 2. SR 정규화 파라미터 로드
    sr_ckpt = torch.load(args.sr_model_path, map_location="cpu")
    sn = sr_ckpt["target_norm"]
    
    def normalize_sr(cx, cy, depth):
        nx = (cx - sn["x_min"]) / (sn["x_max"] - sn["x_min"])
        ny = (cy - sn["y_min"]) / (sn["y_max"] - sn["y_min"])
        nz = (depth - sn["z_min"]) / (sn["z_max"] - sn["z_min"])
        return torch.stack([nx, ny, nz], dim=1)

    val_sr_pos = normalize_sr(val_ds.cx_data, val_ds.cy_data, val_ds.depth_data).to(device)

    # 3. FF 모델 로드 및 추론
    ff_ckpt = torch.load(args.ff_model_path, map_location=device)
    fn = ff_ckpt["ff_norm"]
    model = MLPFF(in_dim=20, out_dim=1, hidden=[512, 512, 256, 128]).to(device)
    model.load_state_dict(ff_ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_norm = model(val_tactile, val_radius, val_sr_pos)
        pred_fz = pred_norm.cpu().numpy().squeeze() * (fn["fz_max"] - fn["fz_min"]) + fn["fz_min"]

    # 4. 오차 집계
    df = pd.DataFrame({
        'x': val_cx,
        'y': val_cy,
        'err_fz': np.abs(pred_fz - val_fz_raw)
    })
    grid_stats = df.groupby(['y', 'x']).mean().reset_index()

    # 5. 시각화
    plt.figure(figsize=(8, 6))
    pivot = grid_stats.pivot(index='y', columns='x', values='err_fz')
    im = plt.imshow(pivot.values, cmap='YlOrRd', origin='lower', extent=[-9.75, 9.75, -9.75, 9.75])
    plt.title('Force Field (Fz) MAE Grid (N)')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.colorbar(im, label='Error (Newton)')
    
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=150)
    print(f"FF Visualization saved to: {args.out_path}")

if __name__ == "__main__":
    visualize_ff()
