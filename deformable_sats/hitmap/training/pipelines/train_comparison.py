
import os
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp

# Avoid "Too many open files" when using many DataLoader workers / large preload.
# Use file_system sharing to reduce FD usage per worker on Linux.
mp.set_sharing_strategy("file_system")
from tqdm import tqdm
import json
import argparse
import numpy as np
from sklearn.metrics import r2_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.data.dataset_unified import UnifiedTactileDataset
from training.data.dataset_zarr import ZarrDataset
from training.models.unified_model import UnifiedSensorModel
from training.models.mlp_baseline import MLPBaseline
from training.models.cnn_sr import CNNSR
from training.models.cnnlstm_sr import CNNLSTMSR
from training.models.cnn_bilstm import CNNBiLSTM
from training.models.sats_model import SATSModel
from training.models.sats_xy_multihead import SATSXYMultiHead
from training.models.tactile_transformer import TactileTransformer
from training.models.isoline_gnn import IsolineGNN
from training.models.tactile_gnn_gat import TactileGAT
from training.models.multi_head_field_model import MultiHeadFieldModel
from training.utils.contact_geometry import contact_radius_tensor
from training.pipelines import runtime_common as rt

def get_model(name, seq_len=50, heatmap_size=40, dropout: float = 0.1):
    if name == "unified": return UnifiedSensorModel(seq_len=seq_len)
    elif name == "mlp": return MLPBaseline()
    elif name == "cnn": return CNNSR()
    elif name == "cnnlstm": return CNNLSTMSR()
    elif name == "cnnbilstm": return CNNBiLSTM()
    elif name == "sats": return SATSModel()
    elif name == "sats_xy": return SATSXYMultiHead()
    elif name == "transformer": return TactileTransformer()
    elif name == "isoline_gnn": return IsolineGNN()
    elif name == "tactile_gnn_gat": return TactileGAT()
    elif name == "multi_head_field": return MultiHeadFieldModel(seq_len=seq_len, heatmap_size=heatmap_size, dropout=dropout)
    else: raise ValueError(f"Unknown model: {name}")

GRID_STEP = 0.5
GRID_MIN = -9.75

def _metric_output_names(width: int) -> list[str]:
    if width == 3:
        return ["x", "y", "z"]
    if width == 4:
        return ["x", "y", "z", "fz"]
    return [f"output_{i}" for i in range(width)]


def calculate_metrics(preds, targets):
    mse = np.mean((preds - targets)**2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets), axis=0)
    r2 = r2_score(targets, preds, multioutput='raw_values')
    output_names = _metric_output_names(preds.shape[1])
    per_output = {}
    for i, name in enumerate(output_names):
        per_output[name] = {
            "mse": float(mse[i]),
            "rmse": float(rmse[i]),
            "mae": float(mae[i]),
            "r2": float(r2[i]),
        }
    return {
        "metric_schema": {
            "outputs": output_names,
            "array_order": output_names,
            "units": {
                "x": "mm",
                "y": "mm",
                "z": "mm",
                "fz": "source_units",
            },
        },
        "output_names": output_names,
        "mse": mse.tolist(),
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "r2": r2.tolist(),
        "per_output": per_output,
    }


def build_metric_report(raw_preds: np.ndarray, targets: np.ndarray, args) -> dict:
    raw_metrics = calculate_metrics(raw_preds, targets)
    raw_metrics["depth_bins"] = depth_bin_metrics(raw_preds, targets, args.depth_bin_edges)
    raw_metrics["selection_mae_xyz"] = float(np.mean(raw_metrics["mae"][:3]))

    calibrated_preds = raw_preds
    if raw_preds.shape[1] >= 2:
        calibrated_preds = apply_linear_calib(torch.from_numpy(raw_preds).float(), args).cpu().numpy()

    calibrated_metrics = calculate_metrics(calibrated_preds, targets)
    calibrated_metrics["depth_bins"] = depth_bin_metrics(calibrated_preds, targets, args.depth_bin_edges)
    calibrated_metrics["selection_mae_xyz"] = float(np.mean(calibrated_metrics["mae"][:3]))

    report = dict(raw_metrics)
    report["metric_selection"] = {
        "primary": "raw",
        "secondary": "calibrated",
    }
    report["metric_variants"] = {
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
    }
    return report


def depth_bin_metrics(preds, targets, bin_edges):
    """
    preds, targets: np arrays (N,3) where [:,2]=depth
    bin_edges: list of edges including upper edge
    Returns list of dicts per bin.
    """
    results = []
    depth = targets[:, 2]
    xy_err = np.linalg.norm(preds[:, :2] - targets[:, :2], axis=1)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (depth >= lo) & (depth < hi)
        if mask.sum() == 0:
            results.append({"range": [lo, hi], "count": 0})
            continue
        mae = float(xy_err[mask].mean())
        succ = float((xy_err[mask] <= GRID_STEP).mean())
        results.append({"range": [float(lo), float(hi)], "count": int(mask.sum()), "xy_mae": mae, "success<=1cell": succ})
    return results


def apply_linear_calib(pred: torch.Tensor, args):
    """
    Optional post-calibration to correct systematic scale/offset between
    predicted [x,y] and ground truth. Coefficients are user-tunable.
    """
    if not getattr(args, "apply_linear_calib", False):
        return pred
    if pred.size(-1) < 2:
        return pred
    out = pred.clone()
    px = pred[:, 0]
    py = pred[:, 1]
    out[:, 0] = args.calib_x_ax * px + args.calib_x_by * py + args.calib_x_bias
    out[:, 1] = args.calib_y_ax * px + args.calib_y_by * py + args.calib_y_bias
    return out


def _multi_head_metric_tensors(
    scalar_pred: torch.Tensor,
    fmap: torch.Tensor,
    targets: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    if args.decode_xy != "none":
        x_dec, y_dec = _decode_xy_from_heatmap(fmap, args.decode_xy)
        pred_xy = torch.stack([x_dec, y_dec], dim=1)
    else:
        flat = fmap.view(fmap.size(0), -1)
        argmax = flat.argmax(dim=1)
        iy = argmax // args.heatmap_size
        ix = argmax % args.heatmap_size
        xs = torch.arange(args.heatmap_size, device=fmap.device, dtype=fmap.dtype) * GRID_STEP + GRID_MIN
        ys = torch.arange(args.heatmap_size, device=fmap.device, dtype=fmap.dtype) * GRID_STEP + GRID_MIN
        pred_xy = torch.stack([xs[ix], ys[iy]], dim=1)

    pred_concat = torch.cat(
        [
            pred_xy,
            scalar_pred[:, 0:1],  # z
            scalar_pred[:, 1:2],  # Fz
        ],
        dim=1,
    )
    return pred_concat, targets[:, :4]


def _loss_component_tag(weight: float, loss_name: str) -> str:
    if float(weight) == 0.0:
        return "off"
    weight_text = f"{float(weight):g}".replace("-", "m").replace(".", "p")
    return f"{loss_name}{weight_text}"


def _multi_head_stage(args) -> str:
    if not args.use_depth_aware_label:
        return "stage1"
    if float(args.lambda_z) == 0.0 and float(args.lambda_fz) == 0.0:
        return "stage2"
    return "stage3"


def _multi_head_run_tag(model_name: str, args) -> str:
    if model_name != "multi_head_field":
        return ""

    label_tag = (
        f"dlabel-{args.depth_label_kernel}-{args.depth_radius_model}"
        if args.use_depth_aware_label
        else "point"
    )
    tag = f"_{_multi_head_stage(args)}_{label_tag}"
    tag += f"_xy{_loss_component_tag(args.lambda_xy, args.loss_xy)}"
    tag += f"_z{_loss_component_tag(args.lambda_z, args.loss_z)}"
    tag += f"_fz{_loss_component_tag(args.lambda_fz, args.loss_fz)}"
    if args.decode_xy != "none":
        tag += f"_dec{args.decode_xy}"
    if args.use_depth_aware_label and args.normalize_heatmap:
        tag += "_hnorm"
    return tag


def _write_fz_summary_csv(path: Path, preds: np.ndarray, targets: np.ndarray) -> None:
    if preds.shape[1] < 4 or targets.shape[1] < 4:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pred_z",
        "target_z",
        "error_z",
        "pred_fz",
        "target_fz",
        "error_fz",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pred_row, target_row in zip(preds, targets):
            writer.writerow(
                {
                    "pred_z": float(pred_row[2]),
                    "target_z": float(target_row[2]),
                    "error_z": float(pred_row[2] - target_row[2]),
                    "pred_fz": float(pred_row[3]),
                    "target_fz": float(target_row[3]),
                    "error_fz": float(pred_row[3] - target_row[3]),
                }
            )


def _forward_model(model_name, model, grid, iso, return_field: bool = False):
    # grid: (B, T, 1, 4, 4), iso: (B, T, 17=[drift16 + radius1])
    if model_name == "unified":
        # Unified model expects iso dim 19; pad two zeros when metadata is unavailable.
        if iso.size(-1) < 19:
            pad = torch.zeros(iso.size(0), iso.size(1), 19 - iso.size(-1), device=iso.device, dtype=iso.dtype)
            iso_u = torch.cat([iso, pad], dim=-1)
        else:
            iso_u = iso[..., :19]
        res1, _ = model(grid, iso_u)
        return res1["xyz"] if isinstance(res1, dict) else res1

    if model_name == "cnnlstm":
        radius_seq = iso[:, :, -1:]
        return model(grid, radius_seq)

    if model_name in ["sats", "sats_xy"]:
        s16_seq = grid[:, :, 0].reshape(grid.size(0), grid.size(1), -1)
        return model(s16_seq)

    if model_name == "cnnbilstm":
        return model(grid)

    if model_name == "multi_head_field":
        scalar_vec, field_map = model(grid)
        if return_field:
            return scalar_vec, field_map
        return scalar_vec

    if model_name == "mlp":
        s16 = grid[:, -1, 0].reshape(grid.size(0), -1)
        radius = iso[:, -1, -1:]
        return model(s16, radius)

    if model_name == "isoline_gnn":
        s16 = grid[:, -1, 0].reshape(grid.size(0), -1)
        radius = iso[:, -1, -1:]
        return model(s16, radius)

    if model_name == "tactile_gnn_gat":
        s16 = grid[:, -1, 0].reshape(grid.size(0), -1)
        return model(s16)

    if model_name in ["cnn", "transformer"]:
        s16 = grid[:, -1, 0].reshape(grid.size(0), -1)
        if model_name == "cnn":
            radius = iso[:, -1, -1:]
            return model(grid[:, -1], radius)
        return model(s16)

    raise ValueError(f"Unsupported model: {model_name}")


def _effective_batch_size(model_name: str, requested_bs: int) -> int:
    # LSTM-based models need much smaller batches than MLP/CNN.
    if model_name == "cnnlstm":
        return min(requested_bs, 1024)
    if model_name == "cnnbilstm":
        return min(requested_bs, 1024)
    if model_name == "multi_head_field":
        return min(requested_bs, 1024)
    if model_name == "unified":
        return min(requested_bs, 4096)
    if model_name in ["sats", "sats_xy"]:
        return min(requested_bs, 512)
    if model_name == "tactile_gnn_gat":
        return min(requested_bs, 2048)
    return requested_bs

def _resolve_device(force: str) -> torch.device:
    if force == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA device is not available")
        return torch.device("cuda")
    if force == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_optimizer(model: nn.Module, args):
    if args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def _decode_xy_from_heatmap(fmap_logits: torch.Tensor, decode: str):
    """
    fmap_logits: (B,1,H,W)
    Returns (x_mm, y_mm) tensors (B,)
    """
    B, _, H, W = fmap_logits.shape
    device = fmap_logits.device
    dtype = fmap_logits.dtype
    xs = torch.arange(W, device=device, dtype=dtype) * GRID_STEP + GRID_MIN
    ys = torch.arange(H, device=device, dtype=dtype) * GRID_STEP + GRID_MIN

    if decode == "softargmax":
        prob = torch.softmax(fmap_logits.view(B, -1), dim=1)
        exp_x = torch.sum(prob * xs.view(1, 1, W).repeat(1, H, 1).view(1, -1), dim=1)
        exp_y = torch.sum(prob * ys.view(1, H, 1).repeat(1, 1, W).view(1, -1), dim=1)
        return exp_x, exp_y

    # argmax_refine: coarse argmax + local softmax 3x3 window
    flat_idx = torch.argmax(fmap_logits.view(B, -1), dim=1)
    iy = flat_idx // W
    ix = flat_idx % W
    # local window
    x0 = torch.clamp(ix - 1, 0, W - 1)
    x1 = torch.clamp(ix + 1, 0, W - 1)
    y0 = torch.clamp(iy - 1, 0, H - 1)
    y1 = torch.clamp(iy + 1, 0, H - 1)
    # gather 3x3
    coords_x = []
    coords_y = []
    weights = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            cx = torch.clamp(ix + dx, 0, W - 1)
            cy = torch.clamp(iy + dy, 0, H - 1)
            coords_x.append(xs[cx])
            coords_y.append(ys[cy])
            weights.append(fmap_logits[torch.arange(B, device=device), 0, cy, cx])
    w = torch.stack(weights, dim=1)
    prob_local = torch.softmax(w, dim=1)
    cx = torch.stack(coords_x, dim=1)
    cy = torch.stack(coords_y, dim=1)
    exp_x = (prob_local * cx).sum(dim=1)
    exp_y = (prob_local * cy).sum(dim=1)
    return exp_x, exp_y


def _build_soft_heatmap(
    x_mm: torch.Tensor,
    y_mm: torch.Tensor,
    depth_mm: torch.Tensor,
    heatmap_size: int = 40,
    radius_model: str = "hertz",
    kernel: str = "gaussian",
    normalize: bool = False,
    indenter_radius_mm: float | torch.Tensor = 2.5,
    fallback_depth_mm: float = 1.0,
    sigma_scale: float = 1.0,
) -> torch.Tensor:
    """
    Depth-aware soft target heatmap.
    Returns (B,1,H,W) on the same device/dtype as inputs.
    """
    device = x_mm.device
    dtype = x_mm.dtype
    xs = torch.arange(heatmap_size, device=device, dtype=dtype) * GRID_STEP + GRID_MIN
    ys = torch.arange(heatmap_size, device=device, dtype=dtype) * GRID_STEP + GRID_MIN
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    dx = xx.unsqueeze(0) - x_mm.view(-1, 1, 1)
    dy = yy.unsqueeze(0) - y_mm.view(-1, 1, 1)
    dist2 = dx * dx + dy * dy

    depth_eff = torch.where(depth_mm > 0, depth_mm, torch.full_like(depth_mm, fallback_depth_mm))
    if torch.is_tensor(indenter_radius_mm):
        radius_mm = indenter_radius_mm.to(device=device, dtype=dtype).view(-1)
        a = contact_radius_tensor(depth_eff, R_mm=radius_mm, model=radius_model)
    else:
        a = contact_radius_tensor(depth_eff, R_mm=indenter_radius_mm, model=radius_model)

    a = a * sigma_scale

    if kernel == "linear":
        dist = torch.sqrt(dist2 + 1e-12)
        target = torch.relu(1.0 - dist / a.view(-1, 1, 1))
    else:
        denom = 2.0 * (a.view(-1, 1, 1) ** 2) + 1e-12
        target = torch.exp(-dist2 / denom)

    target = torch.where(depth_eff.view(-1, 1, 1) > 0, target, torch.zeros_like(target))

    if normalize:
        s = target.flatten(1).sum(dim=1, keepdim=True) + 1e-6
        target = target / s.view(-1, 1, 1)

    return target.unsqueeze(1)


def _build_point_heatmap(
    x_mm: torch.Tensor,
    y_mm: torch.Tensor,
    heatmap_size: int = 40,
) -> torch.Tensor:
    target = torch.zeros(
        (x_mm.size(0), 1, heatmap_size, heatmap_size),
        device=x_mm.device,
        dtype=x_mm.dtype,
    )
    ix = torch.round((x_mm - GRID_MIN) / GRID_STEP).long().clamp(0, heatmap_size - 1)
    iy = torch.round((y_mm - GRID_MIN) / GRID_STEP).long().clamp(0, heatmap_size - 1)
    target[torch.arange(x_mm.size(0), device=x_mm.device), 0, iy, ix] = 1.0
    return target


def _build_multi_head_target_map(targets: torch.Tensor, args) -> torch.Tensor:
    if args.use_depth_aware_label:
        indenter_radius_mm = targets[:, 4] if targets.size(1) > 4 else args.indenter_radius_mm
        return _build_soft_heatmap(
            targets[:, 0],
            targets[:, 1],
            targets[:, 2],
            heatmap_size=args.heatmap_size,
            radius_model=args.depth_radius_model,
            kernel=args.depth_label_kernel,
            normalize=args.normalize_heatmap,
            indenter_radius_mm=indenter_radius_mm,
            fallback_depth_mm=args.depth_fallback_mm,
            sigma_scale=args.heatmap_sigma_scale,
        )
    return _build_point_heatmap(targets[:, 0], targets[:, 1], heatmap_size=args.heatmap_size)


def _combine_multi_head_loss(args, l_xy: torch.Tensor, l_z: torch.Tensor, l_fz: torch.Tensor) -> torch.Tensor:
    return args.lambda_xy * l_xy + args.lambda_z * l_z + args.lambda_fz * l_fz


def _heatmap_coord_from_mm(value: torch.Tensor, heatmap_size: int) -> torch.Tensor:
    return ((value - GRID_MIN) / GRID_STEP).detach().cpu().clamp(0, heatmap_size - 1)


def _save_overlay(
    batch_idx,
    fmap_logits,
    target_map,
    out_dir,
    prefix="val",
    max_samples=4,
    pred_values: Optional[torch.Tensor] = None,
    target_values: Optional[torch.Tensor] = None,
):
    """
    Save overlay images with pred/target centers and scalar diagnostics.
    """
    os.makedirs(out_dir, exist_ok=True)
    prob = torch.sigmoid(fmap_logits).detach().cpu()
    tgt = target_map.detach().cpu()
    pred_diag = pred_values.detach().cpu() if pred_values is not None else None
    target_diag = target_values.detach().cpu() if target_values is not None else None
    b = min(prob.size(0), max_samples)
    if pred_diag is not None:
        b = min(b, pred_diag.size(0))
    if target_diag is not None:
        b = min(b, target_diag.size(0))
    for i in range(b):
        heatmap_size = prob.size(-1)
        if pred_diag is not None and pred_diag.size(1) >= 2:
            pred_x = _heatmap_coord_from_mm(pred_diag[i, 0], heatmap_size).item()
            pred_y = _heatmap_coord_from_mm(pred_diag[i, 1], heatmap_size).item()
        else:
            flat = prob[i, 0].reshape(-1).argmax()
            pred_y = float((flat // heatmap_size).item())
            pred_x = float((flat % heatmap_size).item())

        if target_diag is not None and target_diag.size(1) >= 2:
            target_x = _heatmap_coord_from_mm(target_diag[i, 0], heatmap_size).item()
            target_y = _heatmap_coord_from_mm(target_diag[i, 1], heatmap_size).item()
        else:
            flat = tgt[i, 0].reshape(-1).argmax()
            target_y = float((flat // heatmap_size).item())
            target_x = float((flat % heatmap_size).item())

        annotation = ""
        if pred_diag is not None and target_diag is not None and pred_diag.size(1) >= 3 and target_diag.size(1) >= 3:
            xy_err = torch.linalg.vector_norm(pred_diag[i, :2] - target_diag[i, :2]).item()
            parts = [
                f"xy_err={xy_err:.3f} mm",
                f"z tgt/pred={target_diag[i, 2].item():.3f}/{pred_diag[i, 2].item():.3f}",
            ]
            if pred_diag.size(1) >= 4 and target_diag.size(1) >= 4:
                parts.append(f"Fz tgt/pred={target_diag[i, 3].item():.3f}/{pred_diag[i, 3].item():.3f}")
            annotation = " | ".join(parts)

        fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))
        for ax, image, title, cmap in (
            (axes[0], prob[i, 0], "pred heatmap", "inferno"),
            (axes[1], tgt[i, 0], "target heatmap", "viridis"),
        ):
            im = ax.imshow(image, origin="lower", cmap=cmap, vmin=0, vmax=1)
            ax.scatter([target_x], [target_y], marker="o", facecolors="none", edgecolors="cyan", linewidths=1.8, label="target")
            ax.scatter([pred_x], [pred_y], marker="x", c="white", linewidths=1.8, label="pred")
            ax.set_title(title)
            ax.legend(loc="upper right", fontsize=6, framealpha=0.75)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if annotation:
            fig.suptitle(annotation, fontsize=8)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{prefix}_overlay_b{batch_idx}_i{i}.png")
        plt.savefig(path)
        plt.close()


class PreloadedDataset(Dataset):
    def __init__(
        self,
        base_ds: Dataset,
        indices,
        store_device: torch.device,
        desc: str = "preload",
        preload_workers: int = 8,
        preload_batch_size: int = 2048,
    ):
        subset = torch.utils.data.Subset(base_ds, indices)
        loader_kwargs = {
            "batch_size": preload_batch_size,
            "shuffle": False,
            "num_workers": preload_workers,
            "pin_memory": (store_device.type == "cuda"),
        }
        if preload_workers > 0:
            loader_kwargs["persistent_workers"] = True
        loader = DataLoader(subset, **loader_kwargs)

        grid_chunks, iso_chunks, tgt_chunks = [], [], []
        for grid, iso, tgt in tqdm(loader, desc=desc):
            grid_chunks.append(grid.to(store_device, non_blocking=True))
            iso_chunks.append(iso.to(store_device, non_blocking=True))
            tgt_chunks.append(tgt.to(store_device, non_blocking=True))

        self.grid = torch.cat(grid_chunks, dim=0).contiguous()
        self.iso = torch.cat(iso_chunks, dim=0).contiguous()
        self.tgt = torch.cat(tgt_chunks, dim=0).contiguous()

    def __len__(self):
        return self.grid.size(0)

    def __getitem__(self, idx):
        return self.grid[idx], self.iso[idx], self.tgt[idx]


class ZarrSequenceDataset(Dataset):
    """
    Build sequence samples from preprocessed Zarr data.
    Sequence key: (trial_id, x_mm, y_mm), sorted by depth.
    """

    def __init__(self, zarr_path: str, seq_len: int = 50, stride: int = 5, phase: str = "all"):
        self.seq_len = seq_len
        self.stride = stride

        zds = ZarrDataset(zarr_path=zarr_path, split="all", phase=phase)
        tactile = zds.tactile_data.float()  # (N,16)
        if getattr(zds, "aux_last_field", "diameter_mm") == "contact_radius_mm":
            radius = zds.aux_data[:, 3:4].float()
        else:
            radius = (zds.aux_data[:, 3:4] / 2.0).float()
        cx = zds.cx_data.float()
        cy = zds.cy_data.float()
        depth = zds.depth_data.float()
        fz = zds.fz_data.float()
        trial_ids = zds.trial_ids

        groups = defaultdict(list)
        for i in range(len(trial_ids)):
            key = (str(trial_ids[i]), round(float(cx[i].item()), 3), round(float(cy[i].item()), 3))
            groups[key].append(i)

        self.samples = []
        self.sample_trial_ids = []
        for key, idxs in groups.items():
            idxs = sorted(idxs, key=lambda j: float(depth[j].item()))
            trial_id = key[0]
            t = len(idxs)
            if t <= 0:
                continue
            if t <= seq_len:
                self.samples.append(idxs)
                self.sample_trial_ids.append(trial_id)
            else:
                max_start = t - seq_len
                for s in range(0, max_start + 1, stride):
                    self.samples.append(idxs[s : s + seq_len])
                    self.sample_trial_ids.append(trial_id)

        self.tactile = tactile
        self.radius = radius
        self.cx = cx
        self.cy = cy
        self.depth = depth
        self.fz = fz

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idxs = self.samples[idx]
        t = len(idxs)

        s16 = self.tactile[idxs]  # (t,16)
        r = self.radius[idxs]     # (t,1)

        if t < self.seq_len:
            pad = self.seq_len - t
            s16 = torch.cat([s16, torch.zeros(pad, 16, dtype=s16.dtype)], dim=0)
            r = torch.cat([r, torch.zeros(pad, 1, dtype=r.dtype)], dim=0)

        s16 = s16[: self.seq_len]
        r = r[: self.seq_len]
        grid = s16.reshape(self.seq_len, 1, 4, 4)

        # iso layout expected by current model router: [:16]=aux seq, [-1]=radius
        iso = torch.zeros(self.seq_len, 17, dtype=s16.dtype)
        iso[:, :16] = s16
        iso[:, 16:17] = r

        last_i = idxs[-1]
        tgt = torch.zeros(5, dtype=s16.dtype)  # [x, y, z, fz, indenter_radius_mm]
        tgt[0] = self.cx[last_i]
        tgt[1] = self.cy[last_i]
        tgt[2] = self.depth[last_i]
        tgt[3] = self.fz[last_i]
        tgt[4] = self.radius[last_i]
        return grid, iso, tgt


def _resolve_zarr_path(data_dir: str, zarr_path: str = ""):
    if zarr_path:
        p = Path(zarr_path)
        return str(p)
    p = Path(data_dir)
    if p.suffix == ".zarr":
        return str(p)
    cands = sorted((p / "zarr_data").glob("*.zarr")) + sorted(p.glob("*.zarr"))
    cands = sorted(set(cands))
    if len(cands) == 1:
        return str(cands[0])
    if len(cands) > 1:
        formatted = ", ".join(str(c) for c in cands)
        raise RuntimeError(
            "Found multiple .zarr datasets; pass --zarr-path explicitly or provide a single integrated zarr. "
            f"Candidates: {formatted}"
        )
    return ""


@dataclass(frozen=True)
class TrialSplit:
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    train_trials: list[str]
    val_trials: list[str]
    test_trials: list[str]


def _parse_trial_list(values: Optional[Sequence[str]]) -> Optional[list[str]]:
    if values is None:
        return None

    parsed: list[str] = []
    for raw in values:
        parsed.extend(token.strip() for token in str(raw).split(",") if token.strip())
    return parsed


def _dataset_split_ids(dataset: Dataset) -> list[str]:
    explicit_ids = getattr(dataset, "sample_trial_ids", None)
    if explicit_ids is not None:
        if len(explicit_ids) != len(dataset):
            raise RuntimeError(
                f"Dataset split metadata length mismatch: {len(explicit_ids)} ids for {len(dataset)} samples"
            )
        return [str(x) for x in explicit_ids]

    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise RuntimeError(
            "Dataset does not expose trial split metadata; expected sample_trial_ids or samples"
        )

    split_ids = []
    for sample in samples:
        if isinstance(sample, dict) and "trial_id" in sample:
            split_ids.append(str(sample["trial_id"]))
        elif isinstance(sample, tuple) and sample:
            split_ids.append(str(sample[0]))
        else:
            raise RuntimeError(f"Cannot infer trial id for dataset sample: {sample!r}")
    return split_ids


def _split_indices_by_trial(
    dataset: Dataset,
    seed: int,
    val_trials: Optional[Sequence[str]] = None,
    test_trials: Optional[Sequence[str]] = None,
    val_ratio: float = 0.2,
) -> TrialSplit:
    sample_trials = _dataset_split_ids(dataset)
    trial_ids = sorted(set(sample_trials))
    if len(trial_ids) < 2 and val_trials is None and test_trials is None:
        raise RuntimeError(
            "Trial-level split needs at least 2 distinct trials for train/val. "
            f"Found: {trial_ids}"
        )

    all_trials = set(trial_ids)
    requested_val = set(val_trials or [])
    requested_test = set(test_trials or [])
    unknown = (requested_val | requested_test) - all_trials
    if unknown:
        raise RuntimeError(f"Requested split trials are not present in dataset: {sorted(unknown)}")

    overlap = requested_val & requested_test
    if overlap:
        raise RuntimeError(f"Trials cannot be both val and test: {sorted(overlap)}")

    rng = np.random.default_rng(seed)
    remaining_trials = [t for t in trial_ids if t not in requested_val and t not in requested_test]

    if val_trials is None:
        if len(remaining_trials) < 2:
            raise RuntimeError(
                "Seed-based trial split needs at least 2 train/val candidate trials after test exclusion. "
                f"Candidates: {remaining_trials}"
            )
        shuffled = np.array(remaining_trials, dtype=object)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        n_val = min(n_val, len(shuffled) - 1)
        requested_val = set(str(t) for t in shuffled[:n_val])
        remaining_trials = [str(t) for t in shuffled[n_val:]]

    train_trials = sorted(t for t in remaining_trials if t not in requested_val)
    val_trials_sorted = sorted(requested_val)
    test_trials_sorted = sorted(requested_test)

    if not train_trials:
        raise RuntimeError("Trial split leaves no train trials. Adjust --val-trials/--test-trials.")
    if not val_trials_sorted:
        raise RuntimeError("Trial split leaves no val trials. Provide --val-trials or more trials.")

    split_by_trial = {
        "train": set(train_trials),
        "val": set(val_trials_sorted),
        "test": set(test_trials_sorted),
    }
    train_indices = [i for i, trial in enumerate(sample_trials) if trial in split_by_trial["train"]]
    val_indices = [i for i, trial in enumerate(sample_trials) if trial in split_by_trial["val"]]
    test_indices = [i for i, trial in enumerate(sample_trials) if trial in split_by_trial["test"]]

    return TrialSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_trials=train_trials,
        val_trials=val_trials_sorted,
        test_trials=test_trials_sorted,
    )


def _format_trials(trials: Sequence[str]) -> str:
    return "[" + ", ".join(trials) + "]"


def build_shared_data(args, device: torch.device):
    base_ds = None
    if args.data_source in ["auto", "zarr"]:
        zarr_path = rt.resolve_zarr_path(args.data_dir, args.zarr_path)
        if zarr_path:
            print(f"[INFO] data source: zarr ({zarr_path})")
            base_ds = rt.ZarrSequenceDataset(
                zarr_path=zarr_path,
                seq_len=args.seq_len,
                stride=args.stride,
                phase=args.phase,
            )
        elif args.data_source == "zarr":
            raise RuntimeError(f"Requested --data-source zarr but no .zarr found under {args.data_dir}")

    if base_ds is None:
        print(f"[INFO] data source: csv ({args.data_dir})")
        base_ds = UnifiedTactileDataset(args.data_dir, seq_len=args.seq_len, augment=False)

    if len(base_ds) == 0:
        raise RuntimeError(f"Dataset at {args.data_dir} is empty. Check your data path!")

    total = len(base_ds)
    splits = rt.build_cv_splits(
        base_ds,
        seed=args.seed,
        cv_folds=args.cv_folds,
        val_trials=rt.parse_trial_list(getattr(args, "val_trials", None)),
        test_trials=rt.parse_trial_list(getattr(args, "test_trials", None)),
        depth_bin_edges=args.depth_bin_edges,
        stratify_diameter_depth=args.stratify_diameter_depth,
        auto_test_trials=args.auto_test_trials,
    )
    if args.fold_index is not None:
        splits = [split for split in splits if split.fold_index == args.fold_index]
        if not splits:
            raise RuntimeError(f"Requested --fold-index {args.fold_index} but no such fold exists.")
    rt.save_cv_manifest(
        Path(args.out_dir) / "cv_manifest_comparison.json",
        splits,
        dataset=base_ds,
        depth_bin_edges=args.depth_bin_edges,
        min_depth_bin_samples=args.min_depth_bin_samples,
        stratify_diameter_depth=args.stratify_diameter_depth,
    )
    preload_indices = np.arange(total, dtype=np.int64)

    if args.preload_vram and device.type != "cuda":
        raise RuntimeError("--preload-vram requires CUDA device")
    preload_device = device if args.preload_vram else torch.device("cpu")
    where = "VRAM" if args.preload_vram else "RAM"
    print(f"[INFO] Indexed {total:,} samples")
    print(f"[INFO] Preloading all samples to {where} once...")
    preloaded_ds = PreloadedDataset(
        base_ds,
        preload_indices,
        store_device=preload_device,
        desc="preload all",
        preload_workers=args.preload_workers,
        preload_batch_size=args.preload_batch_size,
    )

    for split in splits:
        print(
            f"[INFO] Fold {split.fold_index+1}/{split.num_folds} samples: "
            f"train={len(split.train_indices):,}, val={len(split.val_indices):,}, test={len(split.test_indices):,}"
        )
        print(f"[INFO] Train trials: {_format_trials(split.train_trials)}")
        print(f"[INFO] Val trials: {_format_trials(split.val_trials)}")
        print(f"[INFO] Test trials: {_format_trials(split.test_trials)}")
    return preloaded_ds, splits


def _iter_batches(preloaded_ds, index_tensor, batch_size, shuffle, device, augment):
    if shuffle:
        order = torch.randperm(index_tensor.numel(), device=index_tensor.device)
        idx = index_tensor[order]
    else:
        idx = index_tensor

    num_batches = (idx.numel() + batch_size - 1) // batch_size
    for i in range(num_batches):
        bi = idx[i * batch_size : (i + 1) * batch_size]
        src_dev = preloaded_ds.grid.device
        if bi.device != src_dev:
            bi = bi.to(src_dev, non_blocking=True)

        grid = torch.index_select(preloaded_ds.grid, 0, bi)
        iso = torch.index_select(preloaded_ds.iso, 0, bi)
        tgt = torch.index_select(preloaded_ds.tgt, 0, bi)

        if augment:
            bsz = grid.size(0)
            flip_x = torch.rand(bsz, device=grid.device) > 0.5
            flip_y = torch.rand(bsz, device=grid.device) > 0.5
            if flip_x.any():
                grid[flip_x] = torch.flip(grid[flip_x], dims=[-1])
                tgt[flip_x, 0] = -tgt[flip_x, 0]
            if flip_y.any():
                grid[flip_y] = torch.flip(grid[flip_y], dims=[-2])
                tgt[flip_y, 1] = -tgt[flip_y, 1]
            grid = grid + torch.randn_like(grid) * 0.01

        if grid.device != device:
            grid = grid.to(device, non_blocking=True)
            iso = iso.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
        yield grid, iso, tgt, num_batches


def _evaluate_model_metrics(model_name, model, args, device, preloaded_ds, index_tensor, effective_bs):
    all_raw_preds, all_targets = [], []
    with torch.no_grad():
        eval_iter = _iter_batches(
            preloaded_ds,
            index_tensor,
            batch_size=effective_bs,
            shuffle=False,
            device=device,
            augment=False,
        )
        for batch_idx, (grid, iso, tgt, _) in enumerate(eval_iter):
            if model_name == "multi_head_field":
                depth_mask = tgt[:, 2] > args.depth_min_for_label
                if not depth_mask.any():
                    continue
                grid_m = grid[depth_mask]
                iso_m = iso[depth_mask]
                tgt_m = tgt[depth_mask]

                pred, fmap = _forward_model(model_name, model, grid_m, iso_m, return_field=True)
                if args.save_heatmap_overlay and batch_idx < args.overlay_batches:
                    target_map = _build_multi_head_target_map(tgt_m, args)
                    raw_pred_use, target_use = _multi_head_metric_tensors(pred, fmap, tgt_m, args)
                    _save_overlay(
                        batch_idx,
                        fmap,
                        target_map,
                        os.path.join(args.out_dir, "overlays"),
                        prefix=f"{model_name}_eval",
                        max_samples=args.overlay_samples,
                        pred_values=apply_linear_calib(raw_pred_use, args),
                        target_values=target_use,
                    )
                raw_pred_use, target_use = _multi_head_metric_tensors(pred, fmap, tgt_m, args)
            else:
                pred = _forward_model(model_name, model, grid, iso)
                raw_pred_use = pred[:, :3]
                target_use = tgt[:, :3]

            all_raw_preds.append(raw_pred_use.cpu().numpy())
            all_targets.append(target_use.cpu().numpy())

    raw_preds = np.concatenate(all_raw_preds)
    targets = np.concatenate(all_targets)
    return build_metric_report(raw_preds, targets, args)


def train_one_model(model_name, args, device, preloaded_ds, train_idx, val_idx, test_idx, out_dir: str):
    print(f"\n--- Training Model: {model_name} ---")
    print(f"  [INFO] device: {device}")
    effective_bs = _effective_batch_size(model_name, args.batch_size)
    if effective_bs != args.batch_size:
        print(f"  [INFO] batch_size override for {model_name}: {args.batch_size} -> {effective_bs} (OOM prevention)")

    model = get_model(model_name, args.seq_len, args.heatmap_size, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = _build_optimizer(model, args)
    bce_pos_weight = torch.tensor(args.fg_weight, device=device) if args.fg_weight > 0 else None
    bce_heatmap = None
    huber = None
    mse = nn.MSELoss()
    if model_name == "multi_head_field":
        if args.loss_xy == "bce":
            bce_heatmap = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
        huber = nn.SmoothL1Loss(beta=args.huber_delta)

    best_val_mae = float('inf')
    best_metrics = {}
    best_save_path = None

    for epoch in range(args.epochs):
        model.train()
        train_iter = _iter_batches(
            preloaded_ds,
            train_idx,
            batch_size=effective_bs,
            shuffle=True,
            device=device,
            augment=True,
        )
        train_bar = tqdm(
            train_iter,
            total=(train_idx.numel() + effective_bs - 1) // effective_bs,
            desc=f"{model_name} train {epoch+1}/{args.epochs}",
            leave=False,
        )
        for grid, iso, tgt, _ in train_bar:
            optimizer.zero_grad()

            if model_name == "multi_head_field":
                depth_mask = tgt[:, 2] > args.depth_min_for_label
                if not depth_mask.any():
                    continue  # skip batch with no contact

                grid_m = grid[depth_mask]
                iso_m = iso[depth_mask]
                tgt_m = tgt[depth_mask]

                pred, fmap = _forward_model(model_name, model, grid_m, iso_m, return_field=True)
                target_map = _build_multi_head_target_map(tgt_m, args)
                if args.use_depth_aware_label and args.loss_xy == "bce" and args.normalize_heatmap:
                    maxv = target_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0].clamp(min=1e-6)
                    target_map = target_map / maxv
                if args.loss_xy == "bce":
                    l_xy = bce_heatmap(fmap, target_map)
                else:
                    weight = 1.0 + args.wmse_alpha * target_map
                    l_xy = (weight * (fmap - target_map) ** 2).mean()

                pred_z = (pred[:, 0] - args.z_mean) / args.z_std
                tgt_z = (tgt_m[:, 2] - args.z_mean) / args.z_std
                pred_fz = (pred[:, 1] - args.fz_mean) / args.fz_std
                tgt_fz = (tgt_m[:, 3] - args.fz_mean) / args.fz_std

                if args.loss_z == "huber":
                    l_z = huber(pred_z, tgt_z)
                else:
                    l_z = mse(pred_z, tgt_z)
                if args.loss_fz == "huber":
                    l_fz = huber(pred_fz, tgt_fz)
                else:
                    l_fz = mse(pred_fz, tgt_fz)

                loss = _combine_multi_head_loss(args, l_xy, l_z, l_fz)
            else:
                pred = _forward_model(model_name, model, grid, iso)
                loss = criterion(pred[:, :3], tgt[:, :3])  # Position loss
                if args.lambda_offdiag > 0.0:
                    xc = pred[:, 0] - pred[:, 0].mean()
                    yc = pred[:, 1] - pred[:, 1].mean()
                    cov = (xc * yc).mean()
                    loss = loss + args.lambda_offdiag * (cov * cov)

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation & Metrics
        model.eval()
        with torch.no_grad():
            val_iter = _iter_batches(
                preloaded_ds,
                val_idx,
                batch_size=effective_bs,
                shuffle=False,
                device=device,
                augment=False,
            )
            val_bar = tqdm(
                val_iter,
                total=(val_idx.numel() + effective_bs - 1) // effective_bs,
                desc=f"{model_name} val   {epoch+1}/{args.epochs}",
                leave=False,
            )
            val_losses = []
            all_raw_preds, all_targets = [], []
            for batch_idx, (grid, iso, tgt, _) in enumerate(val_bar):
                if model_name == "multi_head_field":
                    depth_mask = tgt[:, 2] > args.depth_min_for_label
                    if not depth_mask.any():
                        continue
                    grid_m = grid[depth_mask]
                    iso_m = iso[depth_mask]
                    tgt_m = tgt[depth_mask]

                    pred, fmap = _forward_model(model_name, model, grid_m, iso_m, return_field=True)
                    target_map = _build_multi_head_target_map(tgt_m, args)
                    if args.use_depth_aware_label and args.loss_xy == "bce" and args.normalize_heatmap:
                        maxv = target_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0].clamp(min=1e-6)
                        target_map = target_map / maxv
                    if args.loss_xy == "bce":
                        l_xy = bce_heatmap(fmap, target_map)
                    else:
                        weight = 1.0 + args.wmse_alpha * target_map
                        l_xy = (weight * (fmap - target_map) ** 2).mean()
                    pred_z = (pred[:, 0] - args.z_mean) / args.z_std
                    tgt_z = (tgt_m[:, 2] - args.z_mean) / args.z_std
                    pred_fz = (pred[:, 1] - args.fz_mean) / args.fz_std
                    tgt_fz = (tgt_m[:, 3] - args.fz_mean) / args.fz_std
                    l_z = huber(pred_z, tgt_z) if args.loss_z == "huber" else mse(pred_z, tgt_z)
                    l_fz = huber(pred_fz, tgt_fz) if args.loss_fz == "huber" else mse(pred_fz, tgt_fz)
                    val_loss = _combine_multi_head_loss(args, l_xy, l_z, l_fz)
                    if args.save_heatmap_overlay and batch_idx < args.overlay_batches:
                        overlay_pred, overlay_target = _multi_head_metric_tensors(pred, fmap, tgt_m, args)
                        _save_overlay(
                            batch_idx,
                            fmap,
                            target_map,
                            os.path.join(args.out_dir, "overlays"),
                            prefix=f"{model_name}_e{epoch+1}",
                            max_samples=args.overlay_samples,
                            pred_values=overlay_pred,
                            target_values=overlay_target,
                        )
                else:
                    pred = _forward_model(model_name, model, grid, iso)
                    val_loss = criterion(pred[:, :3], tgt[:, :3])

                if model_name == "multi_head_field":
                    raw_pred_use, target_use = _multi_head_metric_tensors(pred, fmap, tgt_m, args)
                else:
                    raw_pred_use = pred[:, :3]
                    target_use = tgt[:, :3]

                all_raw_preds.append(raw_pred_use.cpu().numpy())
                all_targets.append(target_use.cpu().numpy())
                val_losses.append(val_loss.item())

        all_preds = np.concatenate(all_raw_preds)
        all_targets = np.concatenate(all_targets)
        metrics = build_metric_report(all_preds, all_targets, args)
        avg_mae = float(metrics["selection_mae_xyz"])

        if avg_mae < best_val_mae:
            best_val_mae = avg_mae
            best_metrics = metrics
            tag = _multi_head_run_tag(model_name, args)
            save_path = os.path.join(out_dir, f"best_{model_name}{tag}.pth")
            best_save_path = save_path
            torch.save({
                "state_dict": model.state_dict(),
                "metrics": metrics,
                "model_name": model_name,
                "args": vars(args),
            }, save_path)
            with open(os.path.join(out_dir, f"metrics_{model_name}{tag}.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            if model_name == "multi_head_field":
                _write_fz_summary_csv(
                    Path(out_dir) / f"fz_summary_{model_name}{tag}.csv",
                    all_preds,
                    all_targets,
                )

        print(
            f"  [EPOCH {epoch+1:03d}/{args.epochs}] "
            f"val_mae={avg_mae:.4f}mm (best={best_val_mae:.4f}mm)"
        )

    print(f"Best MAE for {model_name}: {best_val_mae:.4f} mm")

    # Free model-specific memory before next experiment.
    del model
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if best_save_path and test_idx.numel() > 0:
        best_model = get_model(model_name, args.seq_len, args.heatmap_size, dropout=args.dropout).to(device)
        checkpoint = torch.load(best_save_path, map_location=device)
        best_model.load_state_dict(checkpoint["state_dict"])
        best_model.eval()
        best_metrics["test_metrics"] = _evaluate_model_metrics(
            model_name,
            best_model,
            args,
            device,
            preloaded_ds,
            test_idx,
            effective_bs,
        )
        tag = _multi_head_run_tag(model_name, args)
        with open(os.path.join(out_dir, f"metrics_{model_name}{tag}.json"), "w") as f:
            json.dump(best_metrics, f, indent=2)

    return best_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="preprocessing/processed_data")
    parser.add_argument("--out-dir", type=str, default="training/runs_comparison")
    parser.add_argument("--models", nargs="+", default=["mlp", "cnn", "unified", "cnnlstm"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--preload-workers", type=int, default=8)
    parser.add_argument("--preload-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--phase", choices=["loading", "unloading", "all"], default="all")
    # calibration (optional)
    parser.add_argument("--apply-linear-calib", action="store_true", help="Apply post linear calibration to [x,y] outputs")
    parser.add_argument("--calib-x-ax", type=float, default=1.0)
    parser.add_argument("--calib-x-by", type=float, default=0.23)
    parser.add_argument("--calib-x-bias", type=float, default=-1.06)
    parser.add_argument("--calib-y-ax", type=float, default=0.0)
    parser.add_argument("--calib-y-by", type=float, default=0.60)
    parser.add_argument("--calib-y-bias", type=float, default=1.80)
    # cross coupling regularization
    parser.add_argument("--lambda-offdiag", type=float, default=0.0, help="Penalty weight for cov(x_pred, y_pred)")
    parser.add_argument("--data-source", choices=["auto", "zarr", "csv"], default="auto")
    parser.add_argument("--zarr-path", type=str, default="")
    parser.add_argument(
        "--val-trials",
        nargs="*",
        default=None,
        help="Explicit validation trial_id list. Accepts space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--test-trials",
        nargs="*",
        default=None,
        help="Explicit test trial_id list to exclude from train/val. Accepts space-separated or comma-separated values.",
    )
    parser.add_argument("--auto-test-trials", type=int, default=1, help="Automatically hold out this many full trials when --test-trials is omitted.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--stratify-diameter-depth", action="store_true", default=True, help="Balance folds using per-trial diameter and dominant depth regime.")
    parser.add_argument("--no-stratify-diameter-depth", dest="stratify_diameter_depth", action="store_false")
    parser.add_argument("--min-depth-bin-samples", type=int, default=16, help="Minimum target count reported per depth bin for held-out coverage checks.")
    parser.add_argument("--fold-index", type=int, default=None, help="Run only one fold index from the CV manifest.")
    parser.add_argument("--preload-vram", action="store_true", default=True, help="Preload all samples into VRAM")
    parser.add_argument("--no-preload-vram", dest="preload_vram", action="store_false", help="Preload all samples into RAM")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="cuda",
        help="'auto'=cuda if available else cpu; 'cuda'=require GPU; 'cpu'=force cpu",
    )
    # depth-aware heatmap option B
    parser.add_argument("--use-depth-aware-label", action="store_true", help="Use depth-aware soft heatmap (option B) for multi_head_field")
    parser.add_argument("--depth-label-kernel", choices=["gaussian", "linear"], default="gaussian")
    parser.add_argument("--depth-radius-model", choices=["hertz", "geom"], default="hertz")
    parser.add_argument("--indenter-radius-mm", type=float, default=2.5)
    parser.add_argument("--heatmap-size", type=int, default=40)
    parser.add_argument("--heatmap-sigma-scale", type=float, default=1.0, help="Scale factor applied to contact radius when building soft heatmap")
    parser.add_argument("--normalize-heatmap", action="store_true", help="Normalize soft heatmap to sum=1")
    parser.add_argument("--depth-fallback-mm", type=float, default=1.0, help="Depth to use when depth is missing/<=0")
    parser.add_argument("--depth-min-for-label", type=float, default=0.05, help="Ignore samples with depth <= this when computing heatmap losses/metrics")
    parser.add_argument("--fg-weight", type=float, default=5.0, help="Foreground weight for BCEWithLogitsLoss")
    parser.add_argument("--loss-xy", choices=["bce", "wmse"], default="bce")
    parser.add_argument("--wmse-alpha", type=float, default=4.0, help="weight = 1 + alpha * target for wmse")
    parser.add_argument("--loss-z", choices=["huber", "mse"], default="huber")
    parser.add_argument("--loss-fz", choices=["huber", "mse"], default="huber")
    parser.add_argument("--decode-xy", choices=["none", "softargmax", "argmax_refine"], default="none")
    parser.add_argument("--lambda-xy", type=float, default=1.0)
    parser.add_argument("--lambda-z", type=float, default=0.2)
    parser.add_argument("--lambda-fz", type=float, default=0.2)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--z-mean", type=float, default=0.0)
    parser.add_argument("--z-std", type=float, default=1.0)
    parser.add_argument("--fz-mean", type=float, default=0.0)
    parser.add_argument("--fz-std", type=float, default=1.0)
    parser.add_argument("--depth-bins", type=str, default="0.8,1.1,1.4,1.7", help="comma-separated depth bin edges (mm)")
    parser.add_argument("--save-heatmap-overlay", action="store_true", help="Save pred/target heatmap overlays on validation")
    parser.add_argument("--overlay-samples", type=int, default=4)
    parser.add_argument("--overlay-batches", type=int, default=1)
    parser.add_argument("--smoke-off-baseline", action="store_true", help="Run one forward path with depth-aware flag off to ensure compatibility")

    args = parser.parse_args()
    try:
        edges = [float(x) for x in args.depth_bins.split(",") if x.strip() != ""]
        if len(edges) >= 2:
            args.depth_bin_edges = edges + [float("inf")]
        else:
            args.depth_bin_edges = [0.0, float("inf")]
    except Exception:
        args.depth_bin_edges = [0.0, float("inf")]
    os.makedirs(args.out_dir, exist_ok=True)

    device = _resolve_device(args.device)
    print(f"[INFO] device: {device}")
    preloaded_ds, splits = build_shared_data(args, device)

    results: dict[str, list[dict]] = {m: [] for m in args.models}
    for split in splits:
        train_idx = torch.as_tensor(split.train_indices, dtype=torch.long, device=preloaded_ds.grid.device)
        val_idx = torch.as_tensor(split.val_indices, dtype=torch.long, device=preloaded_ds.grid.device)
        test_idx = torch.as_tensor(split.test_indices, dtype=torch.long, device=preloaded_ds.grid.device)
        fold_out_dir = os.path.join(args.out_dir, "folds", f"fold_{split.fold_index}")
        os.makedirs(fold_out_dir, exist_ok=True)

        for m in args.models:
            metrics = train_one_model(m, args, device, preloaded_ds, train_idx, val_idx, test_idx, fold_out_dir)
            metrics["fold_index"] = split.fold_index
            metrics["train_trials"] = split.train_trials
            metrics["val_trials"] = split.val_trials
            metrics["test_trials"] = split.test_trials
            results[m].append(metrics)

            if args.smoke_off_baseline and args.use_depth_aware_label and m == "multi_head_field":
                args.use_depth_aware_label = False
                try:
                    small_iter = _iter_batches(preloaded_ds, train_idx[:1], batch_size=1, shuffle=False, device=device, augment=False)
                    grid, iso, tgt, _ = next(small_iter)
                    _ = _forward_model(m, get_model(m, args.seq_len, args.heatmap_size, dropout=args.dropout).to(device), grid, iso)
                    print("[SMOKE] flag-off forward pass succeeded.")
                except Exception as e:
                    print(f"[SMOKE] flag-off forward failed: {e}")
                args.use_depth_aware_label = True

    summary = {}
    for model_name, folds in results.items():
        if not folds:
            continue
        mae = np.array([fold["mae"] for fold in folds], dtype=np.float64)
        rmse = np.array([fold["rmse"] for fold in folds], dtype=np.float64)
        summary[model_name] = {
            "num_folds": len(folds),
            "per_fold": folds,
            "mae": {"mean": mae.mean(axis=0).tolist(), "std": mae.std(axis=0, ddof=0).tolist()},
            "rmse": {"mean": rmse.mean(axis=0).tolist(), "std": rmse.std(axis=0, ddof=0).tolist()},
        }

    with open(os.path.join(args.out_dir, "comparison_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
