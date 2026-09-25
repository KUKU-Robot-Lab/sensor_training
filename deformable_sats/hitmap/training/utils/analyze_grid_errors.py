import argparse
import torch
import numpy as np
import pandas as pd
from training.dataset_zarr import ZarrDataset
from training.models.mlp_sr import MLPSR

def analyze():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로드 (Val split)
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

    model = MLPSR(in_dim=17, out_dim=3, hidden=[512, 512, 256, 128, 64]).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        pred_norm = model(val_tactile, val_radius)
        
    if tn:
        pred_x = pred_norm[:, 0].cpu().numpy() * (tn["x_max"] - tn["x_min"]) + tn["x_min"]
        pred_y = pred_norm[:, 1].cpu().numpy() * (tn["y_max"] - tn["y_min"]) + tn["y_min"]
    else:
        pred_x, pred_y = pred_norm[:, 0].cpu().numpy(), pred_norm[:, 1].cpu().numpy()

    # 3. 오차 계산 및 그리드 집계
    err_x = np.abs(pred_x - val_cx)
    err_y = np.abs(pred_y - val_cy)

    df = pd.DataFrame({
        'x': val_cx,
        'y': val_cy,
        'err_x': err_x,
        'err_y': err_y
    })

    # (x, y)별 평균 오차 계산
    grid_stats = df.groupby(['y', 'x']).agg({
        'err_x': 'mean',
        'err_y': 'mean'
    }).reset_index()

    # 4. 결과 출력 (Pivot Table 형태)
    print("\n[X-axis MAE Grid (mm)]")
    pivot_x = grid_stats.pivot(index='y', columns='x', values='err_x')
    print(pivot_x.round(3))

    print("\n[Y-axis MAE Grid (mm)]")
    pivot_y = grid_stats.pivot(index='y', columns='x', values='err_y')
    print(pivot_y.round(3))

    # 오차 상위 5개 지점
    print("\n[Top 5 High Error Points (X-axis)]")
    print(grid_stats.sort_values('err_x', ascending=False).head(5))

if __name__ == "__main__":
    analyze()
