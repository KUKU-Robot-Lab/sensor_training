import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.dataset_zarr import ZarrDataset
from training.models.mlp_sr_class import MLPSRClass

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("training/runs_sr_class"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Classification Mode)")

    # 1. 데이터셋 로드
    train_ds = ZarrDataset(args.zarr_path, split="train", val_ratio=args.val_ratio, seed=args.seed)
    val_ds = ZarrDataset(args.zarr_path, split="val", val_ratio=args.val_ratio, seed=args.seed)

    # 2. 그리드 매핑 설정 (0.5mm 간격)
    GRID_MIN = -9.75
    GRID_STEP = 0.5
    GRID_CLASSES = 40 # (-9.75 ~ 9.75, 0.5 step)

    def to_class(val):
        idx = torch.round((val - GRID_MIN) / GRID_STEP).long()
        return torch.clamp(idx, 0, GRID_CLASSES - 1)

    def from_class(idx):
        return idx.float() * GRID_STEP + GRID_MIN

    # 데이터 GPU 이동 및 분류용 정수 라벨 생성
    print("Moving data to GPU...")
    train_tactile = train_ds.tactile_data.to(device)
    train_radius = (train_ds.aux_data[:, 3:4] / 2.0).to(device)
    train_x_class = to_class(train_ds.cx_data).to(device)
    train_y_class = to_class(train_ds.cy_data).to(device)
    train_z_target = train_ds.depth_data.to(device)

    val_tactile = val_ds.tactile_data.to(device)
    val_radius = (val_ds.aux_data[:, 3:4] / 2.0).to(device)
    val_x_class = to_class(val_ds.cx_data).to(device)
    val_y_class = to_class(val_ds.cy_data).to(device)
    val_z_target = val_ds.depth_data.to(device)

    # 3. 모델 및 옵티마이저
    model = MLPSRClass(in_dim=17, x_classes=GRID_CLASSES, y_classes=GRID_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # 4. 학습 루프
    history = {"train_loss": [], "val_loss": [], "val_mae_mm": []}
    best_val_mae = float("inf")

    n_train = len(train_ds)
    n_val = len(val_ds)

    print(f"Starting Grid Classification Training (Classes: {GRID_CLASSES})...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = torch.randperm(n_train, device=device)
        
        train_losses = []
        for i in range(0, n_train, args.batch_size):
            batch_idx = indices[i : i + args.batch_size]
            
            b_tactile = train_tactile[batch_idx]
            b_radius = train_radius[batch_idx]
            b_x_cls = train_x_class[batch_idx]
            b_y_cls = train_y_class[batch_idx]
            b_z_reg = train_z_target[batch_idx]

            optimizer.zero_grad()
            x_logits, y_logits, z_pred = model(b_tactile, b_radius)
            
            loss_x = criterion_cls(x_logits, b_x_cls)
            loss_y = criterion_cls(y_logits, b_y_cls)
            loss_z = criterion_reg(z_pred.squeeze(), b_z_reg)
            
            loss = loss_x + loss_y + loss_z
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # 검증
        model.eval()
        with torch.no_grad():
            x_logits, y_logits, z_pred = model(val_tactile, val_radius)
            
            # 클래스 -> mm 단위 환산
            pred_x_mm = from_class(torch.argmax(x_logits, dim=1))
            pred_y_mm = from_class(torch.argmax(y_logits, dim=1))
            pred_z_mm = z_pred.squeeze()
            
            mae_x = torch.abs(pred_x_mm - val_ds.cx_data.to(device)).mean().item()
            mae_y = torch.abs(pred_y_mm - val_ds.cy_data.to(device)).mean().item()
            mae_z = torch.abs(pred_z_mm - val_z_target).mean().item()
            avg_mae = (mae_x + mae_y + mae_z) / 3.0

            history["train_loss"].append(np.mean(train_losses))
            history["val_mae_mm"].append(avg_mae)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {history['train_loss'][-1]:.4f} | MAE(mm): X={mae_x:.3f}, Y={mae_y:.3f}, Z={mae_z:.3f}")

            if avg_mae < best_val_mae:
                best_val_mae = avg_mae
                torch.save(model.state_dict(), args.out_dir / "best_model.pt")

    print(f"Training finished. Best Val MAE: {best_val_mae:.4f} mm")
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    train()
