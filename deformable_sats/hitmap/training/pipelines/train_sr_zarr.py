"""
train_sr_zarr.py

Zarr 데이터셋을 이용한 Super-resolution (SR) MLP 학습 스크립트.
Input: s1..16 (16) + radius (1) = 17
Output: x, y, z_depth (3)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.dataset_zarr import ZarrDataset
from training.models.mlp_sr import MLPSR


def _serialize_args(args):
    serialized = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=Path, required=True, help="Path to .zarr directory")
    parser.add_argument("--out-dir", type=Path, default=Path("training/runs_sr_zarr"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    # 2. 데이터셋 로드 (메모리로 전체 로드)
    train_ds = ZarrDataset(
        args.zarr_path,
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        drop_dead_channels=False,
    )
    val_ds = ZarrDataset(
        args.zarr_path,
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
        drop_dead_channels=False,
    )

    # 타겟 정규화 파라미터 계산 (Train set 기준)
    # X, Y는 고정 범위 [-9.75, 9.75]를 사용하거나 실제 데이터 범위를 사용
    x_min, x_max = train_ds.cx_data.min().item(), train_ds.cx_data.max().item()
    y_min, y_max = train_ds.cy_data.min().item(), train_ds.cy_data.max().item()
    z_min, z_max = train_ds.depth_data.min().item(), train_ds.depth_data.max().item()
    
    # 안전을 위해 아주 작은 값 추가
    x_range = (x_max - x_min) if x_max != x_min else 1.0
    y_range = (y_max - y_min) if y_max != y_min else 1.0
    z_range = (z_max - z_min) if z_max != z_min else 1.0

    target_norm = {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max
    }
    print(f"Target Normalization Ranges:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}] | Y: [{y_min:.2f}, {y_max:.2f}] | Z: [{z_min:.2f}, {z_max:.2f}]")

    def normalize_target(cx, cy, depth):
        nx = (cx - x_min) / x_range
        ny = (cy - y_min) / y_range
        nz = (depth - z_min) / z_range
        return torch.stack([nx, ny, nz], dim=1)

    def denormalize_target(norm_target):
        # (B, 3) -> [x, y, z]
        dx = norm_target[:, 0] * x_range + x_min
        dy = norm_target[:, 1] * y_range + y_min
        dz = norm_target[:, 2] * z_range + z_min
        return torch.stack([dx, dy, dz], dim=1)

    print(f"Moving data to {device} for maximum efficiency...")
    # 전체 데이터를 GPU로 미리 이동
    train_tactile = train_ds.tactile_data.to(device)
    train_radius = (train_ds.aux_data[:, 3:4] / 2.0).to(device)
    train_target = normalize_target(train_ds.cx_data, train_ds.cy_data, train_ds.depth_data).to(device)
    
    val_tactile = val_ds.tactile_data.to(device)
    val_radius = (val_ds.aux_data[:, 3:4] / 2.0).to(device)
    val_target = normalize_target(val_ds.cx_data, val_ds.cy_data, val_ds.depth_data).to(device)
    # 평가용 (mm 단위)
    val_target_mm = torch.stack([val_ds.cx_data, val_ds.cy_data, val_ds.depth_data], dim=1).to(device)

    n_train = len(train_ds)
    n_val = len(val_ds)
    print(f"Train samples: {n_train:,}, Val samples: {n_val:,}")

    # 3. 모델 (조금 더 깊은 모델 시도 가능)
    in_dim = train_tactile.shape[1] + 1  # tactile channels + radius
    # 3. 모델 (안정적인 구조로 복귀 및 가중치 손실 적용)
    model = MLPSR(in_dim=17, out_dim=3, hidden=[512, 512, 256, 128, 64]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # X축 가중치 설정을 위한 텐서 (X, Y, Z 순서)
    loss_weights = torch.tensor([2.0, 1.0, 1.0], device=device)

    def weighted_mse_loss(pred, target, weights):
        # (B, 3) 각 차원별 MSE 계산 후 가중치 곱함
        return (weights * (pred - target)**2).mean()

    # 4. 학습 루프
    history = {"train_loss": [], "val_loss": [], "val_mae_mm": []}
    best_val_mae = float("inf")

    print(f"Starting weighted training (X-Weight: 2.0, Batch Size: {args.batch_size})...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = torch.randperm(n_train, device=device)

        train_losses = []
        for i in range(0, n_train, args.batch_size):
            batch_idx = indices[i : i + args.batch_size]
            b_tactile = train_tactile[batch_idx]
            b_radius = train_radius[batch_idx]
            b_target = train_target[batch_idx]

            optimizer.zero_grad()
            pred = model(b_tactile, b_radius)
            # 일반 MSE 대신 가중치 MSE 적용
            loss = weighted_mse_loss(pred, b_target, loss_weights)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        # 검증
        model.eval()
        with torch.no_grad():
            val_losses = []
            val_preds_norm = []
            for i in range(0, n_val, args.batch_size):
                b_tactile = val_tactile[i : i + args.batch_size]
                b_radius = val_radius[i : i + args.batch_size]
                b_target = val_target[i : i + args.batch_size]
                
                pred = model(b_tactile, b_radius)
                # 정의된 weighted_mse_loss 사용
                loss = weighted_mse_loss(pred, b_target, loss_weights)
                val_losses.append(loss.item())
                val_preds_norm.append(pred)
            
            avg_val_loss = np.mean(val_losses)
            
            # mm 단위 오차 계산
            val_preds_mm = denormalize_target(torch.cat(val_preds_norm, dim=0))
            mae_mm = torch.abs(val_preds_mm - val_target_mm).mean(dim=0)
            avg_mae_mm = mae_mm.mean().item()
            xy_mae_mm = mae_mm[:2].mean().item()
        
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["val_mae_mm"].append(float(avg_mae_mm))

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_train_loss:.6f}/{avg_val_loss:.6f} | MAE: {avg_mae_mm:.3f}mm (XY: {xy_mae_mm:.3f}mm)")

        # Best 모델 저장 (MAE 기준)
        if avg_mae_mm < best_val_mae:
            best_val_mae = avg_mae_mm
            save_dict = {
                "model_state_dict": model.state_dict(),
                "target_norm": target_norm,
                "args": _serialize_args(args),
                "epoch": epoch,
                "history": history
            }
            torch.save(save_dict, args.out_dir / "best_model.pt")

    # 최종 모델 저장 및 히스토리 기록
    torch.save(model.state_dict(), args.out_dir / "last_model.pt")
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training finished. Best Val MAE: {best_val_mae:.6f} mm")
    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    train()
