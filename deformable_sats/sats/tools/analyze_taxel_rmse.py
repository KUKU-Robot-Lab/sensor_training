#!/usr/bin/env python3
"""
sats/tools/analyze_taxel_rmse.py

논문 Fig.4.b/c 스타일 per-taxel RMSE 분석 도구.

기능
----
val_trials 전체 순회 → 각 샘플에서 예측 & GT 추출
→ 40×40 per-taxel RMSE heatmap (Fig.4.b)
→ per-taxel RMSE 3D 막대 그래프
→ RMSE 분포 히스토그램 + 통계 (Fig.4.c)

지원 stage: lstm, attn, local_map, cnn, e2e (CNN 체크포인트는 cnn/e2e 모두 동일 구조)

실행 예시
----------
# run 디렉터리 자동 로드 (권장)
python3 -m sats.tools.analyze_taxel_rmse \
    --run-dir sats/training/runs/04.25-sats-test4-e2e_v1

# 체크포인트 직접 지정 (구형)
python3 -m sats.tools.analyze_taxel_rmse \
    --ckpt sats/training/runs/04.25-sats-test4-e2e_v1/best_model.pt \
    --stage e2e

# 여러 모델 side-by-side 비교
python3 -m sats.tools.analyze_taxel_rmse \
    --run-dirs sats/training/runs/04.24-sats-test3\ --paper\ 동일/e2e_paper \
               sats/training/runs/04.25-sats-test4-e2e_v1 \
    --labels "Paper-E2E" "Test4-E2E"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sats.training.config import SATSConfig  # noqa: E402
from sats.training.dataset import build_dataloaders  # noqa: E402
from sats.training.gt_gpu import BatchGPUTargetGenerator  # noqa: E402
from sats.training.train_lstm import get_target  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# config 로드
# ─────────────────────────────────────────────────────────────────────────────

def _valid_fields() -> set[str]:
    return {f.name for f in dataclasses.fields(SATSConfig)}


def load_cfg_from_dir(cfg_dir: Path, explicit_overrides: dict) -> SATSConfig:
    """
    cfg_dir/config.json 을 읽어 SATSConfig 를 복원한다.
    explicit_overrides 에 있는 키만 덮어쓴다 (CLI 기본값은 제외).
    """
    cfg_path = cfg_dir / "config.json"
    base: dict = {}
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text())
        base = {k: v for k, v in raw.items() if k in _valid_fields()}
        print(f"  config 로드: {cfg_path}")
        print(f"  use_window_dataset : {base.get('use_window_dataset', False)}")
        print(f"  window_size        : {base.get('window_size', 10)}")
        print(f"  include_materials  : {base.get('include_materials', [])}")
        print(f"  val_trials         : {base.get('val_trials')}")
        print(f"  exclude_diameters  : {base.get('exclude_diameters', [])}")
    else:
        print(f"  [경고] config.json 없음: {cfg_path}  — 기본값 사용")
    base.update(explicit_overrides)
    return SATSConfig(**base)


def load_cfg_from_ckpt(ckpt_path: Path, explicit_overrides: dict) -> SATSConfig:
    return load_cfg_from_dir(ckpt_path.parent, explicit_overrides)


# ─────────────────────────────────────────────────────────────────────────────
# 모델 로더
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_KEYWORDS = [
    ("e2e",       "cnn"),
    ("cnn",       "cnn"),
    ("local_map", "local_map"),
    ("attn",      "attn"),
    ("lstm",      "lstm"),
]


def detect_stage(run_dir: Path, cfg: SATSConfig) -> str:
    for text in [run_dir.name.lower(), cfg.run_name.lower()]:
        for kw, stage in _STAGE_KEYWORDS:
            if kw in text:
                return stage
    return "cnn"


def load_model(stage: str, ckpt_path: Path, cfg: SATSConfig, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if stage == "lstm":
        from sats.training.lstm_module import SATSLSTMStage
        model = SATSLSTMStage(cfg)
    elif stage == "attn":
        from sats.training.attention_module import SATSAttentionStage
        model = SATSAttentionStage(cfg)
    elif stage == "local_map":
        from sats.training.local_map_module import SATSLocalMapStage
        model = SATSLocalMapStage(cfg)
    elif stage in ("cnn", "e2e"):
        from sats.training.cnn_module import SATSCNNStage
        model = SATSCNNStage(cfg)
    else:
        raise ValueError(f"알 수 없는 stage: {stage!r}")

    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            raise RuntimeError(
                f"체크포인트 아키텍처가 현재 코드와 맞지 않습니다.\n{e}"
            ) from None
        result = model.load_state_dict(ckpt["model"], strict=False)
        if result.missing_keys:
            print(f"  [경고] 초기화되지 않은 키: {result.missing_keys}")

    model.eval()
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 핵심 RMSE 계산
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_taxel_rmse(
    model,
    val_loader,
    device: str,
    grid_size: int = 40,
    target_generator: BatchGPUTargetGenerator | None = None,
) -> Tuple[np.ndarray, int, List[float]]:
    """
    val 데이터 전체에서 taxel별 squared error를 누적 → per-taxel RMSE 반환.

    Returns
    -------
    per_taxel_rmse  : [H, W] float64
    n_samples       : 처리된 샘플 수
    per_sample_rmse : List[float]
    """
    sq_err_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
    n_samples  = 0
    per_sample_rmse: List[float] = []

    model.eval()
    for sensor_b, gt_b, lengths in val_loader:
        sensor_b = sensor_b.to(device)
        gt_b     = gt_b.to(device)
        lengths  = lengths.to(device)

        if target_generator is None:
            target = get_target(gt_b, lengths)           # [B, H, W]
        else:
            target = target_generator(gt_b)              # [B, H, W]
        out    = model(sensor_b, lengths)
        pred   = out[0] if isinstance(out, tuple) else out

        err = (pred - target).cpu().numpy()              # [B, H, W]
        sq_err_sum += (err ** 2).sum(axis=0)
        for b in range(err.shape[0]):
            per_sample_rmse.append(float(np.sqrt(np.mean(err[b] ** 2))))
        n_samples += err.shape[0]

    per_taxel_rmse = np.sqrt(sq_err_sum / max(n_samples, 1))
    return per_taxel_rmse, n_samples, per_sample_rmse


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

_EXTENT = [-9.75, 9.75, -9.75, 9.75]


def plot_heatmap_comparison(
    results: List[Tuple[str, np.ndarray]],
    out_path: Path,
) -> None:
    """모델별 per-taxel RMSE heatmap (논문 Fig.4.b 스타일)."""
    n    = len(results)
    vmax = max(r.max() for _, r in results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5))
    if n == 1:
        axes = [axes]

    for ax, (label, rmse_map) in zip(axes, results):
        im = ax.imshow(
            rmse_map, origin="lower", extent=_EXTENT,
            cmap="hot", vmin=0, vmax=vmax,
        )
        ax.set_title(
            f"{label}\nmean={rmse_map.mean():.4f}  max={rmse_map.max():.4f}",
            fontsize=9,
        )
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        plt.colorbar(im, ax=ax, fraction=0.046, label="RMSE (N/mm²)")

    fig.suptitle("Per-taxel RMSE Heatmap", fontsize=12)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap 저장: {out_path}")


def plot_3d_bar(
    results: List[Tuple[str, np.ndarray]],
    out_dir: Path,
) -> None:
    """
    per-taxel RMSE 3D 막대 그래프.

    색상: 낮은 RMSE → 하늘색(연한 파랑), 높은 RMSE → 남색(진한 파랑).
    """
    # 40×40 그리드 → mm 좌표
    grid_size = results[0][1].shape[0]
    mm_coords = np.linspace(-9.75, 9.75, grid_size)
    step_mm   = mm_coords[1] - mm_coords[0]      # ≈ 0.5 mm
    bar_width = step_mm * 0.85

    xpos_1d, ypos_1d = np.meshgrid(mm_coords, mm_coords)
    xpos = xpos_1d.flatten()
    ypos = ypos_1d.flatten()
    zpos = np.zeros_like(xpos)

    for label, rmse_map in results:
        dz = rmse_map.flatten().astype(np.float64)

        # 0=최솟값(하늘색), 1=최댓값(남색)
        norm_vals = (dz - dz.min()) / (dz.max() - dz.min() + 1e-12)
        # Blues: 0.0 → 거의 흰색, 1.0 → 진한 남색
        # 하늘색 시작을 위해 [0.25, 1.0] 구간 사용
        colors = plt.cm.Blues(0.25 + 0.75 * norm_vals)

        fig = plt.figure(figsize=(12, 8))
        ax  = fig.add_subplot(111, projection="3d")

        ax.bar3d(
            xpos, ypos, zpos,
            bar_width, bar_width, dz,
            color=colors,
            alpha=0.9,
            shade=True,
        )

        ax.set_xlabel("x [mm]", labelpad=8)
        ax.set_ylabel("y [mm]", labelpad=8)
        ax.set_zlabel("RMSE (N/mm²)", labelpad=8)
        ax.set_title(f"{label}\nPer-taxel RMSE 3D", fontsize=11)
        ax.view_init(elev=30, azim=-60)

        # 컬러바 (실제 RMSE 값 기준)
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.Blues,
            norm=plt.Normalize(vmin=rmse_map.min(), vmax=rmse_map.max()),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label("RMSE (N/mm²)", fontsize=9)

        safe = label.replace(" ", "_").replace("/", "_")
        out_path = out_dir / f"{safe}_taxel_rmse_3d.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  3D bar 저장: {out_path}")


def plot_distribution_comparison(
    results: List[Tuple[str, np.ndarray, List[float]]],
    out_path: Path,
) -> None:
    """per-taxel / per-sample RMSE 분포 (논문 Fig.4.c 스타일)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10.colors

    for i, (label, rmse_map, sample_rmse) in enumerate(results):
        c    = colors[i % len(colors)]
        flat = rmse_map.flatten()

        axes[0].hist(flat, bins=50, alpha=0.6, color=c, label=label, density=True)
        axes[0].axvline(flat.mean(), color=c, linestyle="--", linewidth=1.5,
                        label=f"{label} mean={flat.mean():.4f}")

        axes[1].hist(sample_rmse, bins=50, alpha=0.6, color=c, label=label, density=True)
        axes[1].axvline(float(np.mean(sample_rmse)), color=c, linestyle="--", linewidth=1.5,
                        label=f"{label} mean={np.mean(sample_rmse):.4f}")

    axes[0].set_title("Per-taxel RMSE Distribution")
    axes[0].set_xlabel("RMSE (N/mm²)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Per-sample RMSE Distribution")
    axes[1].set_xlabel("RMSE (N/mm²)")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  분포 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 통계 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(label: str, rmse_map: np.ndarray, sample_rmse: List[float]) -> None:
    flat = rmse_map.flatten()
    print(f"\n{'─'*54}")
    print(f"  모델: {label}")
    print(f"{'─'*54}")
    print(f"  처리 샘플 수:        {len(sample_rmse)}")
    print("  Per-taxel RMSE:")
    print(f"    평균              {flat.mean():.6f}")
    print(f"    중앙값            {float(np.median(flat)):.6f}")
    print(f"    최대              {float(flat.max()):.6f}")
    print(f"    90th percentile   {float(np.percentile(flat, 90)):.6f}")
    print("  Per-sample RMSE:")
    print(f"    평균              {float(np.mean(sample_rmse)):.6f}")
    print(f"    중앙값            {float(np.median(sample_rmse)):.6f}")
    print(f"    90th percentile   {float(np.percentile(sample_rmse, 90)):.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 분석 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    entries: List[Tuple[Path, str, str, Path]],  # (ckpt, stage, label, cfg_dir)
    explicit_overrides: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_data: List[Tuple[str, np.ndarray]]              = []
    dist_data:    List[Tuple[str, np.ndarray, List[float]]] = []

    for ckpt_path, stage, label, cfg_dir in entries:
        print(f"\n[{label}]  ckpt: {ckpt_path}")
        cfg    = load_cfg_from_dir(cfg_dir, explicit_overrides)
        device = cfg.effective_device()

        _, val_loader = build_dataloaders(cfg)
        model = load_model(stage, ckpt_path, cfg, device)
        target_generator = (
            BatchGPUTargetGenerator(cfg, device)
            if cfg.gt_mode == "gpu_on_the_fly"
            else None
        )

        mode = "window" if cfg.use_window_dataset else "sequence"
        print(f"  추론 모드: {mode}  stage={stage}  device={device}")

        rmse_map, n, sample_rmse = compute_taxel_rmse(
            model, val_loader, device, cfg.grid_size, target_generator=target_generator,
        )
        print(f"  처리 완료: {n}개 샘플")
        print_stats(label, rmse_map, sample_rmse)

        safe = label.replace(" ", "_").replace("/", "_")
        np.save(out_dir / f"{safe}_taxel_rmse.npy", rmse_map)

        heatmap_data.append((label, rmse_map))
        dist_data.append((label, rmse_map, sample_rmse))

    plot_heatmap_comparison(heatmap_data, out_dir / "taxel_rmse_heatmap.png")
    plot_3d_bar(heatmap_data, out_dir)
    plot_distribution_comparison(dist_data, out_dir / "taxel_rmse_dist.png")
    print(f"\n출력 디렉터리: {out_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SATS per-taxel RMSE 분석 (논문 Fig.4.b/c)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── run 디렉터리 자동 로드 (권장) ─────────────────────────────────────────
    p.add_argument("--run-dir",  default="",
                   help="단일 run 디렉터리 (config.json + best_model.pt 자동 로드)")
    p.add_argument("--run-dirs", nargs="+", default=[],
                   help="비교할 run 디렉터리 목록")

    # ── 체크포인트 직접 지정 (구형 호환) ──────────────────────────────────────
    p.add_argument("--ckpt",   default="", help="단일 모델 체크포인트 경로")
    p.add_argument("--stage",  default="cnn",
                   choices=["lstm", "attn", "local_map", "cnn", "e2e"],
                   help="모델 종류 (--run-dir 사용 시 자동 감지)")
    p.add_argument("--label",  default="", help="표시 이름")
    p.add_argument("--ckpts",  nargs="+", default=[])
    p.add_argument("--stages", nargs="+", default=[])
    p.add_argument("--labels", nargs="+", default=[])

    # ── 명시적 오버라이드 (config.json 값 우선, 지정 시만 덮어씀) ─────────────
    p.add_argument("--device",     default=None, help="cuda / cpu")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--val-trials", nargs="+", default=None,
                   help="명시 시에만 config.json 값 대체")
    p.add_argument("--exclude-diameters", nargs="+", type=int, default=None)

    # ── 출력 ──────────────────────────────────────────────────────────────────
    p.add_argument("--out-dir", default="", help="출력 디렉터리 (기본: run-dir/taxel_rmse)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # 명시된 인자만 오버라이드에 포함
    explicit_overrides: dict = {}
    if args.device is not None:
        explicit_overrides["device"] = args.device
    if args.batch_size is not None:
        explicit_overrides["batch_size"] = args.batch_size
    if args.val_trials is not None:
        explicit_overrides["val_trials"] = args.val_trials
    if args.exclude_diameters is not None:
        explicit_overrides["exclude_diameters"] = args.exclude_diameters

    # device 미지정 시 기본 cuda
    if "device" not in explicit_overrides:
        explicit_overrides["device"] = "cuda"

    # ── run-dir 방식 ───────────────────────────────────────────────────────────
    if args.run_dirs:
        entries = []
        for rd in args.run_dirs:
            run_dir = Path(rd)
            cfg_tmp = load_cfg_from_dir(run_dir, {})
            stage   = detect_stage(run_dir, cfg_tmp)
            label   = run_dir.name
            entries.append((run_dir / "best_model.pt", stage, label, run_dir))
        if args.labels:
            for i, lbl in enumerate(args.labels[:len(entries)]):
                ckpt, stg, _, cfg_d = entries[i]
                entries[i] = (ckpt, stg, lbl, cfg_d)

    elif args.run_dir:
        run_dir = Path(args.run_dir)
        cfg_tmp = load_cfg_from_dir(run_dir, {})
        stage   = args.stage if args.stage != "cnn" else detect_stage(run_dir, cfg_tmp)
        label   = args.label or run_dir.name
        entries = [(run_dir / "best_model.pt", stage, label, run_dir)]

    # ── ckpt 직접 지정 방식 ────────────────────────────────────────────────────
    elif args.ckpts:
        ckpt_paths = [Path(c) for c in args.ckpts]
        stages = args.stages if args.stages else ["cnn"] * len(ckpt_paths)
        labels = args.labels if args.labels else [p.parent.name for p in ckpt_paths]
        entries = [(p, s, label_name, p.parent) for p, s, label_name in zip(ckpt_paths, stages, labels)]

    elif args.ckpt:
        ckpt_path = Path(args.ckpt)
        entries = [(ckpt_path, args.stage, args.label or ckpt_path.parent.name,
                    ckpt_path.parent)]

    else:
        raise SystemExit("--run-dir / --run-dirs / --ckpt / --ckpts 중 하나를 지정하세요.")

    # 출력 디렉터리
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.run_dir:
        out_dir = Path(args.run_dir) / "taxel_rmse"
    elif args.run_dirs:
        out_dir = Path("sats/tools/viz_output/taxel_rmse")
    else:
        out_dir = Path("sats/tools/viz_output/taxel_rmse")

    run_analysis(entries, explicit_overrides, out_dir)


if __name__ == "__main__":
    main()
