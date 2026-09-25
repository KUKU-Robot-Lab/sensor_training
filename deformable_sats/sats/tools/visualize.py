#!/usr/bin/env python3
"""
sats/tools/visualize.py

예측 맵 vs GT 맵 시각화 도구.

run 디렉터리를 지정하면 config.json을 자동으로 불러와 학습 조건(window/sequence,
val_trials, stage)에 맞게 추론한다.

실행:
    cd <repo>

    # run 디렉터리 자동 로드 (권장)
    python3 -m sats.tools.visualize \
        --run-dir sats/training/runs/04.25-sats-test4-e2e_v1

    # run 디렉터리 + stage 명시 (자동 감지 무시)
    python3 -m sats.tools.visualize \
        --run-dir sats/training/runs/04.23-sats-test1/cnn_v1 \
        --stage cnn

    # 구형 방식 (ckpt + stage 직접 지정)
    python3 -m sats.tools.visualize \
        --ckpt sats/training/runs/cnn_v1/best_model.pt \
        --stage cnn

    # 출력: run-dir/viz_output/ 또는 --out-dir 경로에 PNG 저장
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.utils.data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sats.training.config import SATSConfig
from sats.training.dataset import (
    SATSWindowDataset,
    build_dataloaders,
    sats_collate_fn,
    window_collate_fn,
)
from sats.training.train_lstm import get_target


# ─────────────────────────────────────────────────────────────────────────────
# config.json → SATSConfig
# ─────────────────────────────────────────────────────────────────────────────

def _cfg_field_names() -> set[str]:
    return {f.name for f in fields(SATSConfig)}


def load_config_from_run_dir(run_dir: Path) -> SATSConfig:
    """run_dir/config.json 을 읽어 SATSConfig 로 복원한다."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json 없음: {cfg_path}")

    raw = json.loads(cfg_path.read_text())
    known = _cfg_field_names()
    filtered = {k: v for k, v in raw.items() if k in known}
    return SATSConfig(**filtered)


# ─────────────────────────────────────────────────────────────────────────────
# stage 자동 감지
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_KEYWORDS = [
    ("e2e",       "cnn"),     # e2e 학습도 SATSCNNStage 사용
    ("cnn",       "cnn"),
    ("local_map", "local_map"),
    ("attn",      "attn"),
    ("lstm",      "lstm"),
]


def detect_stage(run_dir: Path, cfg: SATSConfig) -> str:
    """run_dir 이름과 config.run_name 에서 stage 를 추론한다."""
    candidates = [run_dir.name.lower(), cfg.run_name.lower()]
    for text in candidates:
        for keyword, stage in _STAGE_KEYWORDS:
            if keyword in text:
                return stage
    return "cnn"  # 기본값


# ─────────────────────────────────────────────────────────────────────────────
# 모델 로더
# ─────────────────────────────────────────────────────────────────────────────

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
    elif stage == "cnn":
        from sats.training.cnn_module import SATSCNNStage
        model = SATSCNNStage(cfg)
    else:
        raise ValueError(f"알 수 없는 stage: {stage}")

    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as strict_err:
        # size mismatch 는 strict=False 로도 해결 안 됨 → 즉시 종료
        if "size mismatch" in str(strict_err):
            raise RuntimeError(
                f"체크포인트 아키텍처가 현재 코드와 맞지 않습니다.\n"
                f"이 체크포인트는 이전 버전의 모델로 학습됐을 가능성이 높습니다.\n"
                f"원본 에러: {strict_err}"
            ) from None
        # 키 이름만 다른 경우 → strict=False 시도
        print(f"  [경고] strict 로드 실패 → strict=False 로 재시도\n  {strict_err}")
        result = model.load_state_dict(ckpt["model"], strict=False)
        if result.missing_keys:
            print(f"  [경고] 초기화되지 않은 키 (랜덤 초기화): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"  [경고] 무시된 키: {result.unexpected_keys}")
    model.eval()
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 시각화 (한 샘플)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample(
    pred: np.ndarray,
    gt: np.ndarray,
    title: str,
    save_path: Path,
) -> float:
    """pred/gt 비교 이미지를 저장하고 RMSE 를 반환한다."""
    error = pred - gt
    vmax = max(float(gt.max()), float(pred.max()), 1e-9)
    err_abs = max(float(np.abs(error).max()), 1e-9)

    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    ax_gt   = fig.add_subplot(gs[0])
    ax_pred = fig.add_subplot(gs[1])
    ax_err  = fig.add_subplot(gs[2])
    ax_info = fig.add_subplot(gs[3])

    extent = [-9.75, 9.75, -9.75, 9.75]

    im_gt = ax_gt.imshow(gt, origin="lower", extent=extent,
                         vmin=0, vmax=vmax, cmap="hot")
    ax_gt.set_title("GT", fontsize=9)
    ax_gt.set_xlabel("x [mm]")
    ax_gt.set_ylabel("y [mm]")
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.046)

    im_pred = ax_pred.imshow(pred, origin="lower", extent=extent,
                             vmin=0, vmax=vmax, cmap="hot")
    ax_pred.set_title("Prediction", fontsize=9)
    ax_pred.set_xlabel("x [mm]")
    plt.colorbar(im_pred, ax=ax_pred, fraction=0.046)

    im_err = ax_err.imshow(error, origin="lower", extent=extent,
                           vmin=-err_abs, vmax=err_abs, cmap="bwr")
    ax_err.set_title("Error (pred − GT)", fontsize=9)
    ax_err.set_xlabel("x [mm]")
    plt.colorbar(im_err, ax=ax_err, fraction=0.046)

    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae  = float(np.mean(np.abs(error)))
    rel  = rmse / (float(gt.max()) + 1e-12) * 100

    info_text = (
        f"GT peak:   {gt.max():.5f} N/mm²\n"
        f"Pred peak: {pred.max():.5f} N/mm²\n"
        f"RMSE:      {rmse:.6f}\n"
        f"MAE:       {mae:.6f}\n"
        f"RMSE/peak: {rel:.1f}%"
    )
    ax_info.axis("off")
    ax_info.text(0.05, 0.5, info_text, transform=ax_info.transAxes,
                 fontsize=9, verticalalignment="center",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(title, fontsize=10)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {save_path}  (RMSE={rmse:.6f}, RMSE/peak={rel:.1f}%)")
    return rmse


# ─────────────────────────────────────────────────────────────────────────────
# peak 윈도우 인덱스 추출
# ─────────────────────────────────────────────────────────────────────────────

def _get_peak_indices(dataset: SATSWindowDataset) -> list[int]:
    """각 (trial, seq) 위치에서 t 가 최대인 윈도우 인덱스만 반환한다."""
    best: dict[tuple, tuple] = {}  # (tid, local_idx) → (t, flat_idx)
    for flat_idx, (tid, local_idx, t, _gt_row) in enumerate(dataset._index):
        key = (tid, local_idx)
        if key not in best or t > best[key][0]:
            best[key] = (t, flat_idx)
    return [flat_idx for _t, flat_idx in best.values()]


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SATS 예측 시각화",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── 기본 인자 ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--run-dir",
        help="학습 run 디렉터리 (config.json + best_model.pt 자동 로드).\n"
             "지정 시 --ckpt / --val-trials / --raw-dir / --gt-dir 자동 결정.",
    )
    parser.add_argument(
        "--ckpt",
        help="체크포인트 파일 경로 (--run-dir 없이 사용할 때 필수).",
    )
    parser.add_argument(
        "--stage",
        choices=["lstm", "attn", "local_map", "cnn"],
        help="모델 stage. --run-dir 사용 시 자동 감지되므로 생략 가능.",
    )
    parser.add_argument("--n-samples",   type=int,   default=20)
    parser.add_argument("--min-gt-peak", type=float, default=0.05,
                        help="GT 맵 최댓값이 이 값 미만인 샘플은 건너뜀 (N/mm², 기본: 0.05)")
    parser.add_argument("--peak-only", action="store_true",
                        help="window 모드 전용: 각 위치의 peak 시점 윈도우만 시각화")
    parser.add_argument("--out-dir",   help="PNG 저장 경로 (기본: run-dir/viz_output)")
    parser.add_argument("--device",    default="cuda")

    # ── 구형 호환 인자 (--run-dir 없을 때) ────────────────────────────────────
    parser.add_argument("--raw-dir",    default="raw_data")
    parser.add_argument("--gt-dir",     default="sats/preprocessing/gt_output_v1")
    parser.add_argument("--val-trials", nargs="+",
                        default=["ecomesh_d5_z1_test3", "ecomesh_d5_z1.5_test9"])

    args = parser.parse_args()

    # ── 디바이스 ───────────────────────────────────────────────────────────────
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA 없음, CPU 사용")

    # ── config 및 ckpt 경로 결정 ───────────────────────────────────────────────
    if args.run_dir:
        run_dir = Path(args.run_dir)
        print(f"run 디렉터리: {run_dir}")
        cfg = load_config_from_run_dir(run_dir)
        cfg.device = device

        ckpt_path = Path(args.ckpt) if args.ckpt else run_dir / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

        stage = args.stage or detect_stage(run_dir, cfg)
        out_dir = Path(args.out_dir) if args.out_dir else run_dir / "viz_output"

        print(f"  stage     : {stage}")
        print(f"  val_trials: {cfg.val_trials}")
        print(f"  window    : use_window_dataset={cfg.use_window_dataset}"
              f"  window_size={cfg.window_size}")
        print(f"  exclude_d : {cfg.exclude_diameters}")

    else:
        if not args.ckpt:
            parser.error("--run-dir 또는 --ckpt 중 하나는 반드시 지정해야 합니다.")
        if not args.stage:
            parser.error("--ckpt 사용 시 --stage 를 명시해야 합니다.")

        ckpt_path = Path(args.ckpt)
        stage = args.stage
        cfg = SATSConfig(
            raw_dir    = args.raw_dir,
            gt_dir     = args.gt_dir,
            val_trials = args.val_trials,
            device     = device,
        )
        out_dir = Path(args.out_dir) if args.out_dir else Path("sats/tools/viz_output")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    print(f"\n모델 로드: {ckpt_path}")
    model = load_model(stage, ckpt_path, cfg, device)

    # ── Val 데이터로더 (학습 조건 그대로) ─────────────────────────────────────
    print("Val 데이터로더 구성 중...")
    _, val_loader = build_dataloaders(cfg)

    collate_fn = window_collate_fn if cfg.use_window_dataset else sats_collate_fn
    dataset = val_loader.dataset

    if args.peak_only and cfg.use_window_dataset and isinstance(dataset, SATSWindowDataset):
        # 각 (trial, seq) 위치에서 t 가 최대인 윈도우(= peak timestep)만 선택
        peak_indices = _get_peak_indices(dataset)
        dataset = torch.utils.data.Subset(dataset, peak_indices)
        print(f"  peak-only: {len(peak_indices)}개 위치의 peak 윈도우만 사용")

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(cfg.batch_size, 256),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── 추론 루프 ──────────────────────────────────────────────────────────────
    count = 0
    rmse_list: list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            sensor_b, gt_b, lengths = batch
            sensor_b = sensor_b.to(device)
            gt_b     = gt_b.to(device)
            lengths  = lengths.to(device)

            # 학습 때와 동일한 get_target 사용
            target = get_target(gt_b, lengths)   # [B, 40, 40]

            out = model(sensor_b, lengths)
            pred_map = out[0] if isinstance(out, tuple) else out

            for b in range(pred_map.shape[0]):
                if count >= args.n_samples:
                    break

                pred_np = pred_map[b].cpu().numpy()
                gt_np   = target[b].cpu().numpy()

                if float(gt_np.max()) < args.min_gt_peak:
                    continue

                mode_tag = "peak" if args.peak_only else "rand"
                save_path = out_dir / f"{stage}_{mode_tag}{count:03d}.png"
                rmse = plot_sample(
                    pred_np, gt_np,
                    title=f"stage={stage}  [{mode_tag}]  sample={count}",
                    save_path=save_path,
                )
                rmse_list.append(rmse)
                count += 1

            if count >= args.n_samples:
                break

    if rmse_list:
        print(f"\n샘플 {len(rmse_list)}개 평균 RMSE: {np.mean(rmse_list):.6f}")
    print(f"출력 디렉터리: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
