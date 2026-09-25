"""
evaluate_sr.py

SR 모델 평가 + 위치별 오차 히트맵 시각화.

사용법:
  python3 training/evaluate_sr.py \
    --model-path training/runs_sr_eco20_mlp/best_xy.pt \
    --features-dir preprocessing/processed_data \
    --material eco20 \
    --out-dir training/runs_sr_eco20_mlp/eval

출력:
  eval/metrics_grid.csv       - (x_mm, y_mm) 별 MAE/RMSE/R2
  eval/metrics_summary.json   - 전체 요약 + edge vs interior
  eval/heatmap_x_mae.png      - x 오차 히트맵
  eval/heatmap_y_mae.png      - y 오차 히트맵
  eval/heatmap_z_mae.png      - z_depth 오차 히트맵
  eval/heatmap_fz_mae.png     - fz 오차 히트맵
  eval/heatmap_xy_err.png     - XY 유클리드 오차 히트맵
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset_sr import load_split, SRDataset, SRSeqDataset
from training.models import MLPSR, CNNSR, CNNLSTMSR


# ── 그리드 설정 ───────────────────────────────────────────────────────────────
GRID_STEP  = 0.5          # mm
GRID_RANGE = np.arange(-9.75, 9.76, GRID_STEP)  # 40 values
N_GRID     = len(GRID_RANGE)  # 40

# Edge 정의: x 또는 y가 첫/마지막 2개 그리드 포인트
EDGE_THRESH = 2  # 양 끝에서 몇 포인트까지를 edge로 볼 것인지


def _grid_key(x: float, y: float) -> tuple:
    """(x_mm, y_mm) → 0.5mm 그리드 스냅 키 (소수점 2자리로 고정)"""
    return (round(x, 2), round(y, 2))


def _is_edge(xi: int, yi: int, n: int = N_GRID) -> bool:
    return (xi < EDGE_THRESH or xi >= n - EDGE_THRESH or
            yi < EDGE_THRESH or yi >= n - EDGE_THRESH)


def _grid_idx(val: float) -> int:
    """mm 값 → 그리드 인덱스 (0 ~ N_GRID-1)"""
    return int(round((val + 9.75) / GRID_STEP))


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    model_name = args.get("model", "mlp")
    # model_config가 저장된 경우 우선 사용, 없으면 학습 기본값으로 복원
    cfg = ckpt.get("model_config", {})
    if model_name == "mlp":
        model = MLPSR(
            in_dim=17,
            hidden=cfg.get("hidden", [256, 256, 128, 64]),
            out_dim=4,
            dropout=cfg.get("dropout", 0.2),  # eval mode에서 Dropout은 비활성화됨
        )
    elif model_name == "cnn":
        model = CNNSR(out_dim=4)
    else:
        model = CNNLSTMSR(
            lstm_hidden=cfg.get("lstm_hidden", 128),
            lstm_layers=cfg.get("lstm_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            out_dim=4,
        )

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    t_mean = torch.tensor(ckpt["target_mean"], dtype=torch.float32, device=device)
    t_std  = torch.tensor(ckpt["target_std"],  dtype=torch.float32, device=device)

    return model, t_mean, t_std, model_name, args


# ── 추론 (point-wise) ─────────────────────────────────────────────────────────
@torch.no_grad()
def infer_pointwise(model, loader, device, t_mean, t_std):
    """
    Returns:
        pred_mm : (N, 4) float32 numpy
        tgt_mm  : (N, 4) float32 numpy
        x_coords: (N,)   float32 numpy
        y_coords: (N,)   float32 numpy
    """
    preds, tgts, xs, ys = [], [], [], []

    for batch in loader:
        s16  = batch["s16"].to(device)
        diam = batch["diam"].to(device)
        tgt  = batch["target"].to(device)

        pred_norm = model(s16, diam)
        pred_mm   = pred_norm.float() * t_std + t_mean

        preds.append(pred_mm.cpu().numpy())
        tgts.append(tgt.cpu().numpy())
        xs.extend(batch["x_mm"] if isinstance(batch["x_mm"], list) else batch["x_mm"].tolist())
        ys.extend(batch["y_mm"] if isinstance(batch["y_mm"], list) else batch["y_mm"].tolist())

    return (np.concatenate(preds), np.concatenate(tgts),
            np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32))


@torch.no_grad()
def infer_seq(model, loader, device, t_mean, t_std):
    """CNN-LSTM 시퀀스 추론: 유효 timestep만 추출"""
    preds, tgts, xs, ys = [], [], [], []

    for batch in loader:
        s16      = batch["s16"].to(device)
        diam     = batch["diam"].to(device)
        tgt      = batch["target"].to(device)
        mask_len = batch["mask_len"]
        x_mm     = batch["x_mm"]
        y_mm     = batch["y_mm"]

        pred_seq = model(s16, diam)  # (B, T, 4)

        for i, ml in enumerate(mask_len):
            ml = int(ml)
            p_mm = (pred_seq[i, :ml].float() * t_std + t_mean).cpu().numpy()
            t_mm = tgt[i, :ml].float().cpu().numpy()
            x = float(x_mm[i]) if hasattr(x_mm, '__getitem__') else float(x_mm)
            y = float(y_mm[i]) if hasattr(y_mm, '__getitem__') else float(y_mm)
            preds.append(p_mm)
            tgts.append(t_mm)
            xs.extend([x] * ml)
            ys.extend([y] * ml)

    return (np.concatenate(preds), np.concatenate(tgts),
            np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32))


# ── Grid-point 오차 집계 ──────────────────────────────────────────────────────
def aggregate_by_grid(pred_mm, tgt_mm, x_coords, y_coords):
    """
    Returns:
        grid_data: dict[(x_key, y_key)] = {
            'err': np.array (N, 4) - 절대 오차 per sample,
            'xi': int, 'yi': int
        }
    """
    grid_data = defaultdict(lambda: {"errs": [], "xi": 0, "yi": 0})

    for i in range(len(pred_mm)):
        key = _grid_key(float(x_coords[i]), float(y_coords[i]))
        err = np.abs(pred_mm[i] - tgt_mm[i])  # (4,)
        grid_data[key]["errs"].append(err)
        grid_data[key]["xi"] = _grid_idx(key[0])
        grid_data[key]["yi"] = _grid_idx(key[1])

    return grid_data


def build_heatmaps(grid_data) -> dict:
    """
    Returns:
        maps: dict[metric_name] = (N_GRID, N_GRID) float array (NaN for missing)
    """
    names = ["x_mae", "y_mae", "z_mae", "fz_mae", "xy_err"]
    maps  = {n: np.full((N_GRID, N_GRID), np.nan) for n in names}

    rows = []
    for (xk, yk), info in grid_data.items():
        errs = np.stack(info["errs"])  # (N, 4)
        xi, yi = info["xi"], info["yi"]

        x_mae  = float(errs[:, 0].mean())
        y_mae  = float(errs[:, 1].mean())
        z_mae  = float(errs[:, 2].mean())
        fz_mae = float(errs[:, 3].mean())
        xy_err = float(np.sqrt(errs[:, 0] ** 2 + errs[:, 1] ** 2).mean())

        # 그리드 인덱스 범위 클리핑
        xi = max(0, min(xi, N_GRID - 1))
        yi = max(0, min(yi, N_GRID - 1))

        maps["x_mae"][yi, xi]  = x_mae
        maps["y_mae"][yi, xi]  = y_mae
        maps["z_mae"][yi, xi]  = z_mae
        maps["fz_mae"][yi, xi] = fz_mae
        maps["xy_err"][yi, xi] = xy_err

        rows.append({
            "x_mm": xk, "y_mm": yk,
            "xi": xi, "yi": yi,
            "n_samples": len(info["errs"]),
            "x_mae": x_mae, "y_mae": y_mae, "z_mae": z_mae,
            "fz_mae": fz_mae, "xy_err_mm": xy_err,
            "is_edge": _is_edge(xi, yi),
        })

    return maps, rows


# ── 통계 요약 ─────────────────────────────────────────────────────────────────
def summarize(pred_mm, tgt_mm, rows):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # type: ignore

    def _r2(a, b):
        try:
            return float(r2_score(a, b))
        except Exception:
            return float("nan")

    summary = {}
    for i, name in enumerate(["x", "y", "z", "fz"]):
        p, t = pred_mm[:, i], tgt_mm[:, i]
        summary[f"{name}_mae"]  = float(mean_absolute_error(t, p))
        summary[f"{name}_rmse"] = float(np.sqrt(mean_squared_error(t, p)))
        summary[f"{name}_r2"]   = _r2(t, p)

    xy_err = np.sqrt((pred_mm[:, 0] - tgt_mm[:, 0]) ** 2 +
                     (pred_mm[:, 1] - tgt_mm[:, 1]) ** 2)
    summary["xy_err_mean"] = float(xy_err.mean())
    summary["xy_err_p95"]  = float(np.percentile(xy_err, 95))

    # Edge vs Interior
    edge_xy, int_xy = [], []
    for r in rows:
        if r["is_edge"]:
            edge_xy.append(r["xy_err_mm"])
        else:
            int_xy.append(r["xy_err_mm"])

    summary["edge_xy_err_mean"]     = float(np.mean(edge_xy))  if edge_xy  else float("nan")
    summary["interior_xy_err_mean"] = float(np.mean(int_xy))   if int_xy   else float("nan")
    summary["n_edge_pts"]   = len(edge_xy)
    summary["n_interior_pts"] = len(int_xy)

    # 최악 포인트 Top-10
    sorted_rows = sorted(rows, key=lambda r: r["xy_err_mm"], reverse=True)
    summary["worst_10_pts"] = [
        {"x_mm": r["x_mm"], "y_mm": r["y_mm"],
         "xy_err_mm": r["xy_err_mm"], "n_samples": r["n_samples"]}
        for r in sorted_rows[:10]
    ]

    return summary


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_heatmaps(maps: dict, out_dir: Path, title_prefix: str = ""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 없음 — 히트맵 저장 생략")
        return

    cfg = {
        "x_mae":  ("X Position Error (mm)",   "RdYlGn_r", 0, 5.0),
        "y_mae":  ("Y Position Error (mm)",   "RdYlGn_r", 0, 5.0),
        "z_mae":  ("Z Depth Error (mm)",      "RdYlBu_r", 0, 0.5),
        "fz_mae": ("Fz Error (N)",            "PuRd",     0, 1.0),
        "xy_err": ("XY Euclidean Error (mm)", "RdYlGn_r", 0, 6.0),
    }

    for key, (label, cmap, vmin, vmax) in cfg.items():
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(maps[key], origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       extent=[-9.75 - 0.25, 9.75 + 0.25,
                               -9.75 - 0.25, 9.75 + 0.25])
        plt.colorbar(im, ax=ax, label=label)

        # 센서 위치 표시 (6.5mm 간격, 4×4 grid at -9.75+k*6.5 offset)
        sensor_pos = [-9.75 + i * 6.5 for i in range(4)]
        for sx in sensor_pos:
            for sy in sensor_pos:
                ax.plot(sx, sy, "wo", markersize=5, markeredgecolor="gray", linewidth=0.5)

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"{title_prefix}{label}")
        ax.set_xticks(np.arange(-9.75, 10, 2.5))
        ax.set_yticks(np.arange(-9.75, 10, 2.5))
        plt.tight_layout()
        fig.savefig(out_dir / f"heatmap_{key}.png", dpi=150)
        plt.close(fig)
        print(f"  저장: heatmap_{key}.png")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SR 모델 평가 + 히트맵")
    p.add_argument("--model-path",    type=Path, required=True)
    p.add_argument("--features-dir",  type=Path, default=Path("preprocessing/processed_data"))
    p.add_argument("--material",      type=str,  required=True)
    p.add_argument("--out-dir",       type=Path, default=None)
    p.add_argument("--split",         type=str,  default="test",
                   choices=["test", "val", "all"])
    p.add_argument("--batch-size",    type=int,  default=4096)
    p.add_argument("--num-workers",   type=int,  default=4)
    p.add_argument("--device",        type=str,  default="auto")
    p.add_argument("--seq-len",       type=int,  default=32)
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 출력 디렉터리
    out_dir = args.out_dir or (args.model_path.parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 체크포인트 로드: {args.model_path}")
    model, t_mean, t_std, model_name, ckpt_args = load_model(args.model_path, device)

    # 체크포인트 학습 설정에서 split 파라미터 추출
    ckpt_val_ratio  = ckpt_args.get("val_ratio",  0.15)
    ckpt_test_ratio = ckpt_args.get("test_ratio", 0.15)
    ckpt_seed       = ckpt_args.get("seed",       42)

    csv_path = args.features_dir / f"{args.material}_features.csv"
    is_seq = model_name == "cnnlstm"

    df = load_split(
        features_csv=csv_path,
        split=args.split,
        val_ratio=ckpt_val_ratio,
        test_ratio=ckpt_test_ratio,
        seed=ckpt_seed,
        phase_filter=0,
    )
    print(f"[INFO] {args.split} 데이터: {len(df):,}행")

    if is_seq:
        ds = SRSeqDataset(df, seq_len=args.seq_len)
    else:
        ds = SRDataset(df)

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = 2
        loader_kw["persistent_workers"] = True

    loader = DataLoader(ds, **loader_kw)

    # 추론
    print("[INFO] 추론 중...")
    if is_seq:
        pred_mm, tgt_mm, x_coords, y_coords = infer_seq(model, loader, device, t_mean, t_std)
    else:
        pred_mm, tgt_mm, x_coords, y_coords = infer_pointwise(model, loader, device, t_mean, t_std)

    print(f"[INFO] 총 샘플: {len(pred_mm):,}")

    # Grid-point 집계
    grid_data = aggregate_by_grid(pred_mm, tgt_mm, x_coords, y_coords)
    maps, rows = build_heatmaps(grid_data)

    # 통계 요약
    try:
        summary = summarize(pred_mm, tgt_mm, rows)
    except ImportError:
        # scikit-learn 없을 경우 간단 요약
        xy_err = np.sqrt((pred_mm[:, 0] - tgt_mm[:, 0]) ** 2 +
                         (pred_mm[:, 1] - tgt_mm[:, 1]) ** 2)
        summary = {
            "xy_err_mean": float(xy_err.mean()),
            "x_mae": float(np.abs(pred_mm[:, 0] - tgt_mm[:, 0]).mean()),
            "y_mae": float(np.abs(pred_mm[:, 1] - tgt_mm[:, 1]).mean()),
            "z_mae": float(np.abs(pred_mm[:, 2] - tgt_mm[:, 2]).mean()),
            "fz_mae": float(np.abs(pred_mm[:, 3] - tgt_mm[:, 3]).mean()),
        }

    # 출력
    print("\n[평가 결과 요약]")
    print(f"  XY 오차  : {summary.get('xy_err_mean', '?'):.3f} mm (mean)")
    print(f"  X MAE    : {summary.get('x_mae', '?'):.3f} mm")
    print(f"  Y MAE    : {summary.get('y_mae', '?'):.3f} mm")
    print(f"  Z MAE    : {summary.get('z_mae', '?'):.3f} mm")
    print(f"  Fz MAE   : {summary.get('fz_mae', '?'):.3f} N")
    if "edge_xy_err_mean" in summary:
        print(f"  Edge XY  : {summary['edge_xy_err_mean']:.3f} mm "
              f"(n={summary['n_edge_pts']}포인트)")
        print(f"  Interior : {summary['interior_xy_err_mean']:.3f} mm "
              f"(n={summary['n_interior_pts']}포인트)")

    # 파일 저장
    import pandas as pd  # type: ignore
    pd.DataFrame(rows).to_csv(out_dir / "metrics_grid.csv", index=False)
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] CSV 저장: {out_dir}/metrics_grid.csv")
    print(f"[INFO] JSON 저장: {out_dir}/metrics_summary.json")

    # 히트맵 시각화
    print("[INFO] 히트맵 생성...")
    plot_heatmaps(maps, out_dir, title_prefix=f"[{args.material.upper()} {model_name.upper()}] ")

    print(f"\n[완료] 출력 디렉터리: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
