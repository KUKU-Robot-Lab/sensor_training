
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse

from training.models.unified_model import UnifiedSensorModel
from training.dataset_unified import UnifiedTactileDataset

def train_unified(args):
    # 1. Dataset & DataLoader
    dataset = UnifiedTactileDataset(args.data_dir, seq_len=args.seq_len)
    if len(dataset) == 0:
        print(f"Error: No data found in {args.data_dir}")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = UnifiedSensorModel(use_branch1=True, use_branch2=True, seq_len=args.seq_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training Loop
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for grid_seq, iso_feat, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            grid_seq, iso_feat, targets = grid_seq.to(device), iso_feat.to(device), targets.to(device)
            
            optimizer.zero_grad()
            z1, z2 = model(grid_seq, iso_feat)
            
            loss = 0
            if z1 is not None:
                # xyz loss
                loss += criterion(z1["xyz"], targets[:, :3])
                # fz_map loss (dummy zero for now as GT fz_map is missing)
                # If GT fz_map exists, it would be targets[:, 6:]
            
            if z2 is not None:
                # Full [x, y, z, Fx, Fy, Fz] loss
                loss += criterion(z2, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for grid_seq, iso_feat, targets in val_loader:
                grid_seq, iso_feat, targets = grid_seq.to(device), iso_feat.to(device), targets.to(device)
                z1, z2 = model(grid_seq, iso_feat)
                
                loss = 0
                if z1 is not None:
                    loss += criterion(z1["xyz"], targets[:, :3])
                if z2 is not None:
                    loss += criterion(z2, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_unified_model.pth"))

    # Save final results
    torch.save(model.state_dict(), os.path.join(args.out_dir, "last_unified_model.pth"))
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="preprocessing/raw_data")
    parser.add_argument("--out-dir", type=str, default="training/runs_unified")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train_unified(args)
