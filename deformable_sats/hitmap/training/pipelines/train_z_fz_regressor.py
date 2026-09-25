import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import time
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.models.multi_head_field_model import MultiHeadFieldModel
from training.models.z_fz_sequence_regressor import ZFzSequenceRegressor
from training.pipelines.runtime_common import (
    ZarrSequenceDataset,
    build_cv_splits,
    parse_trial_list,
    resolve_zarr_path,
    save_cv_manifest,
)
from training.pipelines.train_comparison import _decode_xy_from_heatmap, _resolve_device


XY_SCALE_MM = 10.0
RADIUS_SCALE_MM = 5.0


def _log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class ScalarNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor) -> "ScalarNormalizer":
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False).clamp(min=1e-6)
        return cls(mean=mean, std=std)

    @classmethod
    def from_dict(cls, payload: dict, device: torch.device | None = None) -> "ScalarNormalizer":
        return cls(
            mean=torch.tensor(payload["mean"], dtype=torch.float32, device=device),
            std=torch.tensor(payload["std"], dtype=torch.float32, device=device),
        )

    def to_dict(self) -> dict:
        return {
            "mean": [float(x) for x in self.mean.detach().cpu().tolist()],
            "std": [float(x) for x in self.std.detach().cpu().tolist()],
            "outputs": ["z", "fz"],
        }

    def to(self, device: torch.device) -> "ScalarNormalizer":
        return ScalarNormalizer(mean=self.mean.to(device), std=self.std.to(device))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values.device)) / self.std.to(values.device)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(values.device) + self.mean.to(values.device)


def build_condition_features(xy_mm: torch.Tensor, radius_mm: torch.Tensor) -> torch.Tensor:
    radius = radius_mm.view(-1, 1)
    return torch.cat([xy_mm / XY_SCALE_MM, radius / RADIUS_SCALE_MM], dim=1)


def metric_dict(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    diff = preds_np - targets_np
    mse = np.mean(diff * diff, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff), axis=0)
    names = ["z", "fz"]
    return {
        "metric_schema": {
            "outputs": names,
            "array_order": names,
            "units": {"z": "mm", "fz": "source_units"},
        },
        "output_names": names,
        "mse": mse.tolist(),
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "per_output": {
            name: {
                "mse": float(mse[i]),
                "rmse": float(rmse[i]),
                "mae": float(mae[i]),
            }
            for i, name in enumerate(names)
        },
    }


def _resolve_fold_xy_checkpoint(raw_path: str, fold_index: int) -> Path:
    requested = Path(raw_path)
    fold_token = f"fold_{fold_index}"
    has_fold_segment = any(re.fullmatch(r"fold_\d+", part) for part in requested.parts)

    def _replace_fold(path: Path) -> Path:
        parts = list(path.parts)
        for i, part in enumerate(parts):
            if re.fullmatch(r"fold_\d+", part):
                parts[i] = fold_token
                return Path(*parts)
        return path

    direct_fold = _replace_fold(requested)
    if has_fold_segment and direct_fold.exists():
        return direct_fold
    if requested.exists() and (has_fold_segment or fold_index == 0):
        return requested

    basename = requested.name
    candidate = requested.parent / "folds" / fold_token / basename
    if candidate.exists():
        return candidate

    recursive = sorted(requested.parent.glob(f"**/{basename}"))
    fold_matches = [path for path in recursive if f"/{fold_token}/" in str(path)]
    if len(fold_matches) == 1:
        return fold_matches[0]

    searched = [str(direct_fold), str(candidate)] + [str(path) for path in fold_matches[:5]]
    if requested.exists() and not has_fold_segment:
        raise FileNotFoundError(
            f"Resolved base XY checkpoint {requested}, but could not derive a fold-specific checkpoint for fold {fold_index}. "
            f"Refusing to reuse one checkpoint across folds. Searched: {searched}"
        )
    raise FileNotFoundError(f"Could not resolve XY checkpoint for fold {fold_index}: {raw_path}. Searched: {searched}")


def _load_xy_model(args, device: torch.device, fold_index: int) -> MultiHeadFieldModel | None:
    if not args.xy_checkpoint:
        return None
    ckpt_path = _resolve_fold_xy_checkpoint(args.xy_checkpoint, fold_index)
    _log(f"[INFO] Using XY checkpoint for fold {fold_index+1}: {ckpt_path}")
    model = MultiHeadFieldModel(seq_len=args.seq_len, heatmap_size=args.heatmap_size, dropout=args.dropout).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _xy_condition(
    grid: torch.Tensor,
    targets: torch.Tensor,
    xy_model: MultiHeadFieldModel | None,
    args,
) -> torch.Tensor:
    if xy_model is None:
        return targets[:, :2]
    with torch.no_grad():
        _, fmap = xy_model(grid)
        x_dec, y_dec = _decode_xy_from_heatmap(fmap, args.decode_xy)
        return torch.stack([x_dec, y_dec], dim=1)


def _batch_indices(indices: torch.Tensor, batch_size: int, shuffle: bool) -> list[torch.Tensor]:
    if shuffle:
        indices = indices[torch.randperm(indices.numel(), device=indices.device)]
    return [indices[i : i + batch_size] for i in range(0, indices.numel(), batch_size)]


def _evaluate(model, dataset, indices, normalizer, xy_model, args, device, use_predicted_xy: bool) -> dict:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx in _batch_indices(indices, args.batch_size, shuffle=False):
            grid = dataset.grid[batch_idx].to(device)
            tgt = dataset.tgt[batch_idx].to(device)
            xy_model_for_batch = xy_model if use_predicted_xy else None
            xy = _xy_condition(grid, tgt, xy_model_for_batch, args)
            cond = build_condition_features(xy, tgt[:, 4:5])
            pred_norm = model(grid, cond)
            preds.append(normalizer.denormalize(pred_norm))
            targets.append(tgt[:, 2:4])
    return metric_dict(torch.cat(preds, dim=0), torch.cat(targets, dim=0))


def _preload_dataset(ds: ZarrSequenceDataset, device: torch.device):
    start = time.perf_counter()
    _log(f"[INFO] Preloading {len(ds):,} sequence samples to {device.type.upper()}...")
    index_matrix = torch.zeros((len(ds), ds.seq_len), dtype=torch.long)
    valid_mask = torch.zeros((len(ds), ds.seq_len), dtype=torch.bool)
    last_indices = torch.zeros(len(ds), dtype=torch.long)

    for row, sample_indices in enumerate(ds.samples):
        count = min(len(sample_indices), ds.seq_len)
        index_matrix[row, :count] = torch.tensor(sample_indices[:count], dtype=torch.long)
        valid_mask[row, :count] = True
        last_indices[row] = int(sample_indices[-1])

    s16 = ds.tactile[index_matrix]
    s16 = torch.where(valid_mask.unsqueeze(-1), s16, torch.zeros_like(s16))
    ds.grid = s16.reshape(len(ds), ds.seq_len, 1, 4, 4).to(device)
    ds.tgt = torch.stack(
        [
            ds.cx[last_indices],
            ds.cy[last_indices],
            ds.depth[last_indices],
            ds.fz[last_indices],
            ds.radius[last_indices, 0],
        ],
        dim=1,
    ).to(device)
    elapsed = time.perf_counter() - start
    _log(f"[INFO] Preload complete in {elapsed:.1f}s")
    return ds


def _limit_samples_for_smoke(ds: ZarrSequenceDataset, max_samples: int) -> ZarrSequenceDataset:
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds

    by_trial: dict[str, list[int]] = {}
    for idx, trial_id in enumerate(ds.sample_trial_ids):
        by_trial.setdefault(str(trial_id), []).append(idx)

    selected = []
    trial_ids = sorted(by_trial)
    offset = 0
    while len(selected) < max_samples:
        added = False
        for trial_id in trial_ids:
            trial_samples = by_trial[trial_id]
            if offset < len(trial_samples):
                selected.append(trial_samples[offset])
                added = True
                if len(selected) >= max_samples:
                    break
        if not added:
            break
        offset += 1

    selected = sorted(selected)
    ds.samples = [ds.samples[i] for i in selected]
    ds.sample_trial_ids = [ds.sample_trial_ids[i] for i in selected]
    return ds


def _build_optimizer(model: nn.Module, args):
    if args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def _mean_std_metrics(per_fold: list[dict]) -> dict:
    metric_keys = ["mse", "rmse", "mae"]
    summary = {}
    for condition_key in ("gt_xy", "predicted_xy"):
        condition_summary = {}
        for key in metric_keys:
            arr = np.array([fold[condition_key][key] for fold in per_fold], dtype=np.float64)
            condition_summary[key] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0, ddof=0).tolist(),
            }
        summary[condition_key] = condition_summary
    return summary


def _train_one_fold(args, ds: ZarrSequenceDataset, split, normalizer_device: torch.device) -> dict:
    train_idx = torch.tensor(split.train_indices, dtype=torch.long, device=normalizer_device)
    val_idx = torch.tensor(split.val_indices, dtype=torch.long, device=normalizer_device)

    normalizer = ScalarNormalizer.fit(ds.tgt[train_idx, 2:4]).to(normalizer_device)
    xy_model = _load_xy_model(args, normalizer_device, split.fold_index)
    model = ZFzSequenceRegressor(seq_len=args.seq_len, dropout=args.dropout).to(normalizer_device)
    criterion = nn.SmoothL1Loss(beta=args.huber_delta) if args.loss == "huber" else nn.MSELoss()
    optimizer = _build_optimizer(model, args)
    best_metric = float("inf")
    best_metrics = {}
    best_ckpt_path = args.current_out_dir / "best_z_fz_regressor.pth"
    history = {"train_loss": [], "val_gt_xy_mae": [], "val_pred_xy_mae": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_idx in _batch_indices(train_idx, args.batch_size, shuffle=True):
            grid = ds.grid[batch_idx].to(normalizer_device)
            tgt = ds.tgt[batch_idx].to(normalizer_device)
            xy = tgt[:, :2]
            if args.xy_noise_std_mm > 0:
                xy = xy + torch.randn_like(xy) * args.xy_noise_std_mm
            cond = build_condition_features(xy, tgt[:, 4:5])
            target_norm = normalizer.normalize(tgt[:, 2:4])

            optimizer.zero_grad()
            loss = criterion(model(grid, cond), target_norm)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        gt_metrics = _evaluate(model, ds, val_idx, normalizer, xy_model, args, normalizer_device, use_predicted_xy=False)
        pred_metrics = _evaluate(
            model,
            ds,
            val_idx,
            normalizer,
            xy_model,
            args,
            normalizer_device,
            use_predicted_xy=xy_model is not None,
        )
        history["train_loss"].append(float(np.mean(losses)))
        history["val_gt_xy_mae"].append(gt_metrics["mae"])
        history["val_pred_xy_mae"].append(pred_metrics["mae"])

        select_mae = pred_metrics["mae"][0] + pred_metrics["mae"][1]
        if select_mae < best_metric:
            best_metric = select_mae
            best_metrics = {"gt_xy": gt_metrics, "predicted_xy": pred_metrics}
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "target_norm": normalizer.to_dict(),
                    "metrics": best_metrics,
                    "history": history,
                    "args": vars(args),
                    "split": {
                        "fold_index": split.fold_index,
                        "num_folds": split.num_folds,
                        "split_mode": split.split_mode,
                        "train_trials": split.train_trials,
                        "val_trials": split.val_trials,
                        "test_trials": split.test_trials,
                    },
                },
                best_ckpt_path,
            )
            with (args.current_out_dir / "metrics_z_fz_regressor.json").open("w", encoding="utf-8") as f:
                json.dump(best_metrics, f, indent=2)

        print(
            f"[FOLD {split.fold_index+1}/{split.num_folds}] "
            f"[EPOCH {epoch:03d}/{args.epochs}] "
            f"loss={history['train_loss'][-1]:.6f} "
            f"gt_xy_mae[z,fz]={gt_metrics['mae']} pred_xy_mae[z,fz]={pred_metrics['mae']}"
        )

    if split.test_indices:
        best_model = ZFzSequenceRegressor(seq_len=args.seq_len, dropout=args.dropout).to(normalizer_device)
        checkpoint = torch.load(best_ckpt_path, map_location=normalizer_device)
        best_model.load_state_dict(checkpoint["state_dict"])
        best_model.eval()
        normalizer = ScalarNormalizer.from_dict(checkpoint["target_norm"], device=normalizer_device)
        xy_model = _load_xy_model(args, normalizer_device, split.fold_index)
        test_idx = torch.tensor(split.test_indices, dtype=torch.long, device=normalizer_device)
        best_metrics["test_gt_xy"] = _evaluate(
            best_model,
            ds,
            test_idx,
            normalizer,
            xy_model,
            args,
            normalizer_device,
            use_predicted_xy=False,
        )
        best_metrics["test_predicted_xy"] = _evaluate(
            best_model,
            ds,
            test_idx,
            normalizer,
            xy_model,
            args,
            normalizer_device,
            use_predicted_xy=xy_model is not None,
        )
        with (args.current_out_dir / "metrics_z_fz_regressor.json").open("w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)

    with (args.current_out_dir / "history_z_fz_regressor.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "fold_index": split.fold_index,
        "split_mode": split.split_mode,
        "train_trials": split.train_trials,
        "val_trials": split.val_trials,
        "test_trials": split.test_trials,
        "gt_xy": best_metrics["gt_xy"],
        "predicted_xy": best_metrics["predicted_xy"],
        "test_gt_xy": best_metrics.get("test_gt_xy"),
        "test_predicted_xy": best_metrics.get("test_predicted_xy"),
    }


def train(args) -> dict:
    overall_start = time.perf_counter()
    device = _resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = resolve_zarr_path(args.data_dir, args.zarr_path)
    if not zarr_path:
        raise RuntimeError(f"No .zarr found under {args.data_dir}")
    if args.phase != "loading":
        raise RuntimeError(
            "train_z_fz_regressor.py is restricted to loading-phase data to preserve hysteresis semantics. "
            "Use --phase loading."
        )

    _log(f"[INFO] device: {device}")
    _log(f"[INFO] data source: zarr ({zarr_path})")
    _log(
        f"[INFO] Building loading-phase sequences (seq_len={args.seq_len}, stride={args.stride}, "
        f"cv_folds={args.cv_folds}, batch_size={args.batch_size})"
    )
    ds_build_start = time.perf_counter()
    ds = ZarrSequenceDataset(zarr_path=zarr_path, seq_len=args.seq_len, stride=args.stride, phase=args.phase)
    _log(f"[INFO] Sequence build complete in {time.perf_counter() - ds_build_start:.1f}s: {len(ds):,} samples")
    ds = _limit_samples_for_smoke(ds, args.max_samples)
    if args.max_samples > 0:
        _log(f"[INFO] Smoke sample cap applied: {len(ds):,} samples")
    splits = build_cv_splits(
        ds,
        seed=args.seed,
        cv_folds=args.cv_folds,
        val_trials=parse_trial_list(args.val_trials),
        test_trials=parse_trial_list(args.test_trials),
        depth_bin_edges=args.depth_bin_edges,
        stratify_diameter_depth=args.stratify_diameter_depth,
        auto_test_trials=args.auto_test_trials,
    )
    if args.fold_index is not None:
        splits = [split for split in splits if split.fold_index == args.fold_index]
        if not splits:
            raise RuntimeError(f"Requested --fold-index {args.fold_index} but no such fold exists.")
    save_cv_manifest(
        args.out_dir / "cv_manifest_z_fz_regressor.json",
        splits,
        dataset=ds,
        depth_bin_edges=args.depth_bin_edges,
        min_depth_bin_samples=args.min_depth_bin_samples,
        stratify_diameter_depth=args.stratify_diameter_depth,
    )
    _log(f"[INFO] Saved CV manifest with {len(splits)} fold(s): {args.out_dir / 'cv_manifest_z_fz_regressor.json'}")
    for split in splits:
        _log(
            f"[INFO] Fold {split.fold_index+1}/{split.num_folds} samples: "
            f"train={len(split.train_indices):,} val={len(split.val_indices):,} test={len(split.test_indices):,}"
        )

    ds = _preload_dataset(ds, device)
    per_fold = []
    for split in splits:
        args.current_out_dir = args.out_dir / "folds" / f"fold_{split.fold_index}"
        args.current_out_dir.mkdir(parents=True, exist_ok=True)
        _log(
            f"[INFO] Fold {split.fold_index+1}/{split.num_folds} "
            f"train={split.train_trials} val={split.val_trials} test={split.test_trials}"
        )
        per_fold.append(_train_one_fold(args, ds, split, device))

    metric_summary = _mean_std_metrics(per_fold)
    summary = {
        "num_folds": len(per_fold),
        "per_fold": per_fold,
        "gt_xy_summary": metric_summary["gt_xy"],
        "predicted_xy_summary": metric_summary["predicted_xy"],
    }
    with (args.out_dir / "cv_summary_z_fz_regressor.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _log(f"[INFO] Training complete in {time.perf_counter() - overall_start:.1f}s")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a separate z/Fz regressor from tactile sequence + xy/radius condition.")
    parser.add_argument("--data-dir", type=str, default="preprocessing/processed_data")
    parser.add_argument("--zarr-path", type=str, default="")
    parser.add_argument("--out-dir", type=Path, default=Path("training/runs_z_fz"))
    parser.add_argument("--xy-checkpoint", type=str, default="", help="Optional frozen multi_head_field checkpoint for end-to-end xy-conditioned eval")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument(
        "--phase",
        choices=["loading"],
        default="loading",
        help="Z/Fz 회귀는 hysteresis 보존을 위해 loading phase만 사용합니다.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-trials", nargs="*", default=None)
    parser.add_argument("--test-trials", nargs="*", default=None)
    parser.add_argument("--auto-test-trials", type=int, default=1, help="Automatically hold out this many full trials when --test-trials is omitted.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--stratify-diameter-depth", action="store_true", default=True, help="Balance folds using per-trial diameter and dominant depth regime.")
    parser.add_argument("--no-stratify-diameter-depth", dest="stratify_diameter_depth", action="store_false")
    parser.add_argument("--depth-bins", type=str, default="0.8,1.1,1.4,1.7", help="comma-separated depth bin edges (mm) used for split stratification/reporting")
    parser.add_argument("--min-depth-bin-samples", type=int, default=16, help="Minimum target count reported per depth bin for held-out coverage checks.")
    parser.add_argument("--fold-index", type=int, default=None, help="Run only one fold index from the CV manifest.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--xy-noise-std-mm", type=float, default=0.5)
    parser.add_argument("--decode-xy", choices=["softargmax", "argmax_refine"], default="softargmax")
    parser.add_argument("--heatmap-size", type=int, default=40)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional balanced sample cap for quick smoke runs. 0 uses all samples.")
    args = parser.parse_args()
    try:
        edges = [float(x) for x in args.depth_bins.split(",") if x.strip() != ""]
        args.depth_bin_edges = edges + [float("inf")] if edges else [0.0, float("inf")]
    except Exception:
        args.depth_bin_edges = [0.0, float("inf")]
    return args


if __name__ == "__main__":
    train(parse_args())
