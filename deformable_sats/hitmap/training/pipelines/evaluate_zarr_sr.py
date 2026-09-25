import argparse
import torch
from training.dataset_zarr import ZarrDataset
from training.models.mlp_sr import MLPSR


def _infer_mlp_dims_from_state_dict(state_dict):
    linear_layers = []
    for key, value in state_dict.items():
        if key.startswith("net.") and key.endswith(".weight") and value.ndim == 2:
            try:
                idx = int(key.split(".")[1])
            except ValueError:
                continue
            linear_layers.append((idx, value.shape[0], value.shape[1]))
    linear_layers.sort(key=lambda x: x[0])
    if not linear_layers:
        raise RuntimeError("Could not infer MLP dimensions from checkpoint state_dict.")

    in_dim = int(linear_layers[0][2])
    out_dim = int(linear_layers[-1][1])
    hidden = [int(out_features) for _, out_features, _ in linear_layers[:-1]]
    return in_dim, out_dim, hidden


@torch.no_grad()
def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16384)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로드 (Val/Test 분할)
    # 학습 시와 동일한 split 설정을 위해 seed 고정
    val_ds = ZarrDataset(
        args.zarr_path, split="val", val_ratio=0.2, seed=42, drop_dead_channels=False
    )
    
    val_tactile = val_ds.tactile_data.to(device)
    val_radius = (val_ds.aux_data[:, 3:4] / 2.0).to(device)
    val_target = torch.stack([val_ds.cx_data, val_ds.cy_data, val_ds.depth_data], dim=1).to(device)

    # 2. 모델 로드
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    target_norm = None
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        target_norm = ckpt.get("target_norm")
    else:
        # backward compatibility: plain state_dict checkpoint
        state_dict = ckpt

    in_dim, out_dim, hidden = _infer_mlp_dims_from_state_dict(state_dict)
    model = MLPSR(in_dim=in_dim, out_dim=out_dim, hidden=hidden).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 추론
    preds = []
    for i in range(0, len(val_ds), args.batch_size):
        b_tactile = val_tactile[i : i + args.batch_size]
        b_radius = val_radius[i : i + args.batch_size]
        
        pred = model(b_tactile, b_radius)
        preds.append(pred)
    
    preds = torch.cat(preds, dim=0)

    # 정규화 체크포인트인 경우 mm 단위로 역정규화
    if target_norm is not None:
        x_min = float(target_norm["x_min"])
        x_max = float(target_norm["x_max"])
        y_min = float(target_norm["y_min"])
        y_max = float(target_norm["y_max"])
        z_min = float(target_norm["z_min"])
        z_max = float(target_norm["z_max"])

        x_range = (x_max - x_min) if x_max != x_min else 1.0
        y_range = (y_max - y_min) if y_max != y_min else 1.0
        z_range = (z_max - z_min) if z_max != z_min else 1.0

        preds = torch.stack(
            [
                preds[:, 0] * x_range + x_min,
                preds[:, 1] * y_range + y_min,
                preds[:, 2] * z_range + z_min,
            ],
            dim=1,
        )
    
    # 4. 메트릭 계산 (mm 단위)
    diff = preds - val_target
    mae = torch.abs(diff).mean(dim=0).cpu().numpy()
    rmse = torch.sqrt((diff**2).mean(dim=0)).cpu().numpy()
    
    euclidean_err = torch.norm(diff[:, :2], p=2, dim=1).mean().item()

    print("\n" + "="*30)
    print("      Validation Results")
    print("="*30)
    print(f"MAE (mm)  | X: {mae[0]:.3f}, Y: {mae[1]:.3f}, Z: {mae[2]:.3f}")
    print(f"RMSE (mm) | X: {rmse[0]:.3f}, Y: {rmse[1]:.3f}, Z: {rmse[2]:.3f}")
    print(f"XY Euclidean Error: {euclidean_err:.3f} mm")
    print("="*30)

if __name__ == "__main__":
    evaluate()
