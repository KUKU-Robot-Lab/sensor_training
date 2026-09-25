"""
Simple inference script for multi_head_field checkpoint.

Features
- Loads zarr sequence data (same loader as train_comparison).
- Runs forward pass to get heatmap + [z, Fz].
- Decodes xy from heatmap (softargmax or argmax).
- Computes MAE for x,y,z and saves optional overlay PNGs.
"""

import argparse
import numpy as np
import torch

from training.models.multi_head_field_model import MultiHeadFieldModel
from training.pipelines.train_comparison import (
    ZarrSequenceDataset,
    _resolve_zarr_path,
    _build_soft_heatmap,
    _decode_xy_from_heatmap,
    _save_overlay,
    GRID_MIN,
    GRID_STEP,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to .pth checkpoint")
    ap.add_argument("--data-dir", type=str, default="preprocessing/processed_data")
    ap.add_argument("--zarr-path", type=str, default="")
    ap.add_argument("--seq-len", type=int, default=50)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--phase", choices=["loading", "unloading", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--decode-xy", choices=["softargmax", "argmax_refine", "none"], default="softargmax")
    ap.add_argument("--depth-label-kernel", choices=["gaussian", "linear"], default="gaussian")
    ap.add_argument("--depth-radius-model", choices=["hertz", "geom"], default="hertz")
    ap.add_argument("--indenter-radius-mm", type=float, default=2.5)
    ap.add_argument("--heatmap-size", type=int, default=40)
    ap.add_argument("--heatmap-sigma-scale", type=float, default=1.0)
    ap.add_argument("--normalize-heatmap", action="store_true")
    ap.add_argument("--depth-fallback-mm", type=float, default=1.0)
    ap.add_argument("--depth-min-for-label", type=float, default=0.05)
    ap.add_argument("--overlay-dir", type=str, default="training/runs_comparison/inference_overlays")
    ap.add_argument("--overlay-batches", type=int, default=1)
    ap.add_argument("--overlay-samples", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0, help="0=all")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    # Data
    zarr_path = _resolve_zarr_path(args.data_dir, args.zarr_path)
    if not zarr_path:
        raise RuntimeError(f"No .zarr found under {args.data_dir}")
    ds = ZarrSequenceDataset(
        zarr_path=zarr_path,
        seq_len=args.seq_len,
        stride=args.stride,
        phase=args.phase,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = MultiHeadFieldModel(seq_len=args.seq_len, heatmap_size=args.heatmap_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_preds = []
    all_tgts = []

    with torch.no_grad():
        for b, (grid, iso, tgt) in enumerate(loader):
            if args.max_batches and b >= args.max_batches:
                break
            depth_mask = tgt[:, 2] > args.depth_min_for_label
            if not depth_mask.any():
                continue
            grid = grid[depth_mask].to(device)
            iso = iso[depth_mask].to(device)
            tgt = tgt[depth_mask].to(device)

            scalar, fmap = model(grid)
            # xy decode
            if args.decode_xy == "softargmax":
                x_dec, y_dec = _decode_xy_from_heatmap(fmap, "softargmax")
            elif args.decode_xy == "argmax_refine":
                x_dec, y_dec = _decode_xy_from_heatmap(fmap, "argmax_refine")
            else:
                flat = fmap.view(fmap.size(0), -1)
                argmax = flat.argmax(dim=1)
                iy = argmax // args.heatmap_size
                ix = argmax % args.heatmap_size
                xs = torch.arange(args.heatmap_size, device=fmap.device, dtype=fmap.dtype) * GRID_STEP + GRID_MIN
                ys = torch.arange(args.heatmap_size, device=fmap.device, dtype=fmap.dtype) * GRID_STEP + GRID_MIN
                x_dec = xs[ix]
                y_dec = ys[iy]

            pred_xy = torch.stack([x_dec, y_dec], dim=1)
            pred_zf = scalar  # [z, fz]
            pred_concat = torch.cat([pred_xy, pred_zf], dim=1)  # [x,y,z,fz]

            all_preds.append(pred_concat.cpu().numpy())
            all_tgts.append(tgt.cpu().numpy())

            if b < args.overlay_batches:
                target_map = _build_soft_heatmap(
                    tgt[:, 0],
                    tgt[:, 1],
                    tgt[:, 2],
                    heatmap_size=args.heatmap_size,
                    radius_model=args.depth_radius_model,
                    kernel=args.depth_label_kernel,
                    normalize=args.normalize_heatmap,
                    indenter_radius_mm=args.indenter_radius_mm,
                    fallback_depth_mm=args.depth_fallback_mm,
                    sigma_scale=args.heatmap_sigma_scale,
                )
                _save_overlay(
                    b,
                    fmap,
                    target_map,
                    args.overlay_dir,
                    prefix="batch",
                    max_samples=args.overlay_samples,
                    pred_values=pred_concat,
                    target_values=tgt[:, :4],
                )

    if not all_preds:
        print("No samples passed depth_min_for_label; nothing to report.")
        return

    all_preds = np.concatenate(all_preds)
    all_tgts = np.concatenate(all_tgts)

    mae = np.mean(np.abs(all_preds - all_tgts), axis=0)
    print(f"MAE [x,y,z,fz]: {mae}")
    rmse = np.sqrt(np.mean((all_preds - all_tgts) ** 2, axis=0))
    print(f"RMSE [x,y,z,fz]: {rmse}")


if __name__ == "__main__":
    main()
