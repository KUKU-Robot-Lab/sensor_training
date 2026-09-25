"""
train_ff_zarr.py

Zarr 데이터셋을 이용한 Force Field (FF) MLP 학습 스크립트.
Input: s1..16 (16) + x, y, z_depth (3) + radius (1) = 20
Output: fz_bc (1)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training.dataset_zarr import ZarrDataset
from training.models.mlp_ff import MLPFF


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str, required=True, help="Path to .zarr directory")
    parser.add_argument("--out-dir", type=Path, default=Path("training/runs_ff_zarr"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr-model-path", type=str, help="Path to best SR model to get normalization params")
    args = parser.parse_args()

    # 1. 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    # 2. 데이터셋 로드 (메모리로 전체 로드)
    train_ds = ZarrDataset(args.zarr_path, split="train", val_ratio=args.val_ratio, seed=args.seed)
    val_ds = ZarrDataset(args.zarr_path, split="val", val_ratio=args.val_ratio, seed=args.seed)

    # SR 정규화 파라미터 로드 (없으면 현재 데이터셋에서 계산)
    sr_norm = None
    if args.sr_model_path:
        sr_ckpt = torch.load(args.sr_model_path, map_location="cpu")
        sr_norm = sr_ckpt.get("target_norm")
    
    if sr_norm is None:
        sr_norm = {
            "x_min": train_ds.cx_data.min().item(), "x_max": train_ds.cx_data.max().item(),
            "y_min": train_ds.cy_data.min().item(), "y_max": train_ds.cy_data.max().item(),
            "z_min": train_ds.depth_data.min().item(), "z_max": train_ds.depth_data.max().item()
        }

    # Fz 정규화 파라미터 (Train set 기준)
    fz_min, fz_max = train_ds.fz_data.min().item(), train_ds.fz_data.max().item()
    fz_range = (fz_max - fz_min) if fz_max != fz_min else 1.0
    ff_norm = {"fz_min": fz_min, "fz_max": fz_max}

    print(f"SR Norm Ranges: X:[{sr_norm['x_min']:.2f}, {sr_norm['x_max']:.2f}], Z:[{sr_norm['z_min']:.2f}, {sr_norm['z_max']:.2f}]")
    print(f"FF Norm Range: Fz:[{fz_min:.2f}, {fz_max:.2f}]")

    def normalize_sr(cx, cy, depth):
        nx = (cx - sr_norm["x_min"]) / (sr_norm["x_max"] - sr_norm["x_min"])
        ny = (cy - sr_norm["y_min"]) / (sr_norm["y_max"] - sr_norm["y_min"])
        nz = (depth - sr_norm["z_min"]) / (sr_norm["z_max"] - sr_norm["z_min"])
        return torch.stack([nx, ny, nz], dim=1)

    def normalize_fz(fz):
        return (fz - fz_min) / fz_range

    def denormalize_fz(norm_fz):
        return norm_fz * fz_range + fz_min

    print("Moving data to VRAM...")
    train_tactile = train_ds.tactile_data.to(device)
    train_radius = (train_ds.aux_data[:, 3:4] / 2.0).to(device)
    train_sr_pos = normalize_sr(train_ds.cx_data, train_ds.cy_data, train_ds.depth_data).to(device)
    train_fz = normalize_fz(train_ds.fz_data).to(device).unsqueeze(1)

    val_tactile = val_ds.tactile_data.to(device)
    val_radius = (val_ds.aux_data[:, 3:4] / 2.0).to(device)
    val_sr_pos = normalize_sr(val_ds.cx_data, val_ds.cy_data, val_ds.depth_data).to(device)
    val_fz = normalize_fz(val_ds.fz_data).to(device).unsqueeze(1)
    val_fz_raw = train_ds.fz_data.to(device) # Validation용 원본 Fz (평가용) - 오타 수정 (val_ds)

    # 3. 모델 (FF 전용: in_dim=20, out_dim=1 [fz_bc])
    model = MLPFF(in_dim=20, out_dim=1, hidden=[512, 512, 256, 128]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 4. 학습 루프
    history = {"train_loss": [], "val_loss": [], "val_mae_n": []}
    best_val_mae = float("inf")

    n_train = len(train_ds)
    n_val = len(val_ds)
    val_fz_raw = val_ds.fz_data.to(device) # Corrected

    print(f"Starting Force Field training (Batch Size: {args.batch_size})...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = torch.randperm(n_train, device=device)
        
        train_losses = []
        for i in range(0, n_train, args.batch_size):
            batch_idx = indices[i : i + args.batch_size]
            b_tactile = train_tactile[batch_idx]
            b_radius = train_radius[batch_idx]
            b_sr_pos = train_sr_pos[batch_idx]
            b_fz = train_fz[batch_idx]
            
            optimizer.zero_grad()
            pred = model(b_tactile, b_radius, b_sr_pos)
            loss = criterion(pred, b_fz)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)

        # 검증
        model.eval()
        with torch.no_grad():
            pred_norm = model(val_tactile, val_radius, val_sr_pos)
            val_loss = criterion(pred_norm, val_fz).item()
            
            # MAE (Newton 단위)
            pred_n = denormalize_fz(pred_norm).squeeze()
            mae_n = torch.abs(pred_n - val_fz_raw).mean().item()
            
            history["train_loss"].append(float(avg_train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_mae_n"].append(float(mae_n))

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_train_loss:.6f}/{val_loss:.6f} | Fz MAE: {mae_n:.4f} N")

        if mae_n < best_val_mae:
            best_val_mae = mae_n
            save_dict = {
                "model_state_dict": model.state_dict(),
                "sr_norm": sr_norm,
                "ff_norm": ff_norm,
                "history": history
            }
            torch.save(save_dict, args.out_dir / "best_model.pt")

    torch.save(model.state_dict(), args.out_dir / "last_model.pt")
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Finished. Best Fz MAE: {best_val_mae:.4f} N")
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    train()
