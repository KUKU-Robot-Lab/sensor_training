#!/usr/bin/env python3
"""
sats/training/train_lstm.py

SATS LSTM 인코더 단독 학습 스크립트.

학습 전략
----------
기본값은 논문 방식의 sliding window 학습이다.

입력  : sensor_window [B, 10, 16]   s_norm 시계열
타겟  : target GT map [B, 41, 41]   window 마지막 timestep의 압력맵
손실  : MSELoss(pred_map, target_gt)

`--no-use-window-dataset`을 주면 legacy peak-map 실험을 수행한다:
sensor_seq [B, 1000, 16] -> peak GT map [B, 41, 41].

이후 단계
----------
Self-Attention, Local Map 모듈을 추가할 때
SATSLSTMStage.encoder 가중치를 그대로 재사용한다.

실행
----
python3 -m sats.training.train_lstm
python3 -m sats.training.train_lstm --epochs 100 --hidden-dim 128 --batch-size 2048
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from .config import SATSConfig
from .dataset import build_dataloaders
from .lstm_module import SATSLSTMStage

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_peak_gt(
    gt_batch: torch.Tensor,   # [B, T_max, 40, 40]
    lengths: torch.Tensor,    # [B]
) -> torch.Tensor:
    """
    각 샘플에서 GT 총합이 최대인 timestep의 40×40 맵을 반환한다.

    패딩 영역(GT=0)은 자동으로 배제된다.
    모든 GT가 0인 샘플(비접촉 시퀀스)은 timestep 0을 사용한다.

    Returns
    -------
    peak_gt : Tensor[B, 40, 40]
    """
    B, T_max = gt_batch.shape[:2]
    device = gt_batch.device

    # 각 timestep의 GT 총합 [B, T_max]
    gt_sum = gt_batch.sum(dim=(-2, -1))

    # 패딩 timestep 마스킹 (유효 범위 밖은 -inf)
    time_idx = torch.arange(T_max, device=device).unsqueeze(0)   # [1, T_max]
    pad_mask  = time_idx >= lengths.unsqueeze(1)                  # [B, T_max]
    gt_sum = gt_sum.masked_fill(pad_mask, -1.0)

    peak_t = gt_sum.argmax(dim=1)                               # [B]
    peak_gt = gt_batch[torch.arange(B, device=device), peak_t]  # [B, 40, 40]
    return peak_gt


def get_target(
    gt_b: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    시퀀스 모드([B, T, 40, 40])는 find_peak_gt로 피크를 선택하고,
    윈도우 모드([B, 40, 40])는 그대로 반환한다.
    """
    if gt_b.dim() == 4:
        return find_peak_gt(gt_b, lengths)
    return gt_b


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """MSE의 제곱근 (스칼라)."""
    return math.sqrt(F.mse_loss(pred, target).item())


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 10.0,
    threshold: float = 1e-4,
) -> torch.Tensor:
    """
    접촉 영역에 가중치를 부여한 MSE loss.

    target > threshold인 픽셀에 (1 + alpha) 배 가중치를 적용한다.
    Fz < 0 구간의 노이지한 GT와 비접촉 zero-GT 픽셀의 영향을 줄이기 위함.
    """
    weight = 1.0 + alpha * (target > threshold).float()
    return (weight * (pred - target) ** 2).mean()


def write_history(path: Path, history: list[dict]) -> None:
    """Epoch 종료마다 history.json을 원자적으로 갱신한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(history, indent=2))
    tmp_path.replace(path)


def _loader_total(loader) -> int | None:
    try:
        return len(loader)
    except TypeError:
        return None


def _progress(loader, desc: str | None):
    if desc is None:
        return loader
    return tqdm(loader, total=_loader_total(loader), desc=desc, dynamic_ncols=True, leave=False)


# ─────────────────────────────────────────────────────────────────────────────
# 체크포인트
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict() if scheduler is not None else {},
            "metrics":    metrics,
        },
        path,
    )
    log.info("체크포인트 저장: %s", path)


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    log.info("체크포인트 로드: %s (epoch %d)", path, ckpt["epoch"])
    return ckpt["epoch"], ckpt.get("metrics", {})


# ─────────────────────────────────────────────────────────────────────────────
# 학습 / 검증 루프
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: SATSLSTMStage,
    loader,
    optimizer,
    device: str,
    cfg: SATSConfig,
    progress_desc: str | None = None,
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    progress_iter = _progress(loader, progress_desc)
    for sensor_b, gt_b, lengths in progress_iter:
        sensor_b = sensor_b.to(device, non_blocking=True)    # [B, T, 16]
        gt_b     = gt_b.to(device, non_blocking=True)        # [B, T, 40, 40]
        lengths  = lengths.to(device, non_blocking=True)     # [B]

        target = get_target(gt_b, lengths).detach()           # [B, 40, 40]

        # 순전파
        pred_map, _ = model(sensor_b, lengths)               # [B, 40, 40]
        loss = F.mse_loss(pred_map, target)

        # 역전파
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        if progress_desc is not None:
            progress_iter.set_postfix(loss=f"{total_loss / n_batches:.6f}")

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def val_epoch(
    model: SATSLSTMStage,
    loader,
    device: str,
    progress_desc: str | None = None,
) -> dict:
    model.eval()
    total_mse  = 0.0
    n_batches  = 0

    progress_iter = _progress(loader, progress_desc)
    for sensor_b, gt_b, lengths in progress_iter:
        sensor_b = sensor_b.to(device, non_blocking=True)
        gt_b     = gt_b.to(device, non_blocking=True)
        lengths  = lengths.to(device, non_blocking=True)

        target = get_target(gt_b, lengths)
        pred_map, _ = model(sensor_b, lengths)

        total_mse += F.mse_loss(pred_map, target).item()
        n_batches += 1
        if progress_desc is not None:
            running_mse = total_mse / n_batches
            progress_iter.set_postfix(mse=f"{running_mse:.6f}", rmse=f"{math.sqrt(running_mse):.6f}")

    mse  = total_mse / max(n_batches, 1)
    rmse = math.sqrt(mse)
    return {"mse": mse, "rmse": rmse}


# ─────────────────────────────────────────────────────────────────────────────
# 메인 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: SATSConfig) -> None:
    # ── 초기화 ────────────────────────────────────────────────────────────
    set_seed(cfg.seed)
    device = cfg.effective_device()
    log.info("device: %s", device)

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    # config 저장
    cfg_path = run_dir / "config.json"
    cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    cfg_path.write_text(json.dumps(cfg_dict, indent=2, default=str))
    log.info("설정 저장: %s", cfg_path)

    # ── 데이터 로더 ───────────────────────────────────────────────────────
    log.info("데이터 로더 구성 중...")
    train_loader, val_loader = build_dataloaders(cfg)

    # ── 모델 ──────────────────────────────────────────────────────────────
    model = SATSLSTMStage(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("모델 파라미터: %d", n_params)

    # ── 옵티마이저 / 스케줄러 ─────────────────────────────────────────────
    optimizer = Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = (
        ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
        )
        if cfg.use_lr_scheduler else None
    )
    if not cfg.use_lr_scheduler:
        log.info("고정 LR 모드 (use_lr_scheduler=False, lr=%.6f)", cfg.lr)

    # ── 학습 이력 ─────────────────────────────────────────────────────────
    history   = []
    best_rmse = float("inf")
    best_ckpt = run_dir / "best_model.pt"
    last_ckpt = run_dir / "last_model.pt"
    hist_path = run_dir / "history.json"

    # ── 학습 루프 ──────────────────────────────────────────────────────────
    log.info("학습 시작 (epochs=%d)", cfg.epochs)
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg,
            progress_desc=f"train {epoch}/{cfg.epochs}",
        )
        val_metrics = val_epoch(
            model,
            val_loader,
            device,
            progress_desc=f"val   {epoch}/{cfg.epochs}",
        )

        lr_now = optimizer.param_groups[0]["lr"]
        if cfg.use_lr_scheduler and scheduler is not None:
            scheduler.step(val_metrics["mse"])

        elapsed = time.time() - t0
        row = {
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "val_mse":    val_metrics["mse"],
            "val_rmse":   val_metrics["rmse"],
            "lr":         lr_now,
            "elapsed_s":  elapsed,
        }
        history.append(row)
        write_history(hist_path, history)

        log.info(
            "Epoch %3d/%d  train_loss=%.6f  val_rmse=%.6f  lr=%.2e  (%.1fs)",
            epoch, cfg.epochs,
            row["train_loss"], row["val_rmse"], lr_now, elapsed,
        )

        # best 체크포인트
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, row)
            log.info("  ★ best val_rmse=%.6f → %s", best_rmse, best_ckpt)

        # 주기 체크포인트
        if epoch % cfg.save_every == 0:
            ckpt_path = run_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, row)

    # 마지막 체크포인트
    save_checkpoint(last_ckpt, cfg.epochs, model, optimizer, scheduler, history[-1])

    # 이력 저장
    log.info("학습 이력 저장: %s", hist_path)
    log.info("완료. best val_rmse=%.6f", best_rmse)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SATS LSTM 학습")
    p.add_argument("--raw-dir",      default="learning_data/sensor_raw_bin",
                   help="merged BIN 학습 입력 루트 디렉터리")
    p.add_argument("--gt-dir",       default="learning_data/gt",
                   help="GT npy 디렉터리")
    p.add_argument("--gt-mode",      choices=["precomputed", "on_the_fly"], default="precomputed",
                   help="GT 공급 방식. precomputed=기존 npy, on_the_fly=merged BIN에서 즉석 생성")
    p.add_argument("--out-dir",      default="sats/training/runs",
                   help="결과 저장 디렉터리")
    p.add_argument("--run-name",     default="lstm_v1")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=2048)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--hidden-dim",   type=int,   default=64)
    p.add_argument("--num-layers",   type=int,   default=2)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--seq-len",      type=int,   default=1000)
    p.add_argument("--num-workers",  type=int,   default=2)
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="DataLoader worker당 미리 준비할 batch 수(num_workers>0일 때)")
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True,
                   help="DataLoader worker를 epoch 사이에 유지")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--val-trials",        nargs="+",
                   default=[],
                   help="검증에 사용할 trial_id 목록")
    p.add_argument("--val-ratio",         type=float, default=0.2,
                   help=">0: 랜덤 sequence-level split. 0: --val-trials 기반 trial split.")
    p.add_argument("--exclude-diameters", nargs="+", type=int, default=[],
                   help="학습/검증 풀에서 제외할 인덴터 직경(mm). 예: --exclude-diameters 10")
    p.add_argument("--window-size",       type=int,   default=10,
                   help="슬라이딩 윈도우 크기 (논문: 10)")
    p.add_argument("--on-the-fly-patch-step-mm", type=float, default=0.1,
                   help="on_the_fly GT 원형 접촉면 이산화 간격")
    p.add_argument("--contact-radius-step-mm", type=float, default=0.05,
                   help="구형 인덴터 접촉 반경 커널 캐시 양자화 간격")
    p.add_argument("--min-contact-radius-mm", type=float, default=0.05,
                   help="z_depth가 작을 때 사용할 최소 접촉 반경")
    p.add_argument("--z-depth-min-mm", type=float, default=0.001,
                   help="이 이하 z_depth는 비접촉 GT zero로 처리")
    p.add_argument("--z-balance-bin-width-mm", type=float, default=0.005,
                   help="on_the_fly balanced_contact에서 z_depth 균형 샘플링 bin 폭")
    p.add_argument("--plateau-stride", type=int, default=10,
                   help="on_the_fly balanced_contact에서 plateau/static 샘플 downsample stride")
    p.add_argument("--loading-stride", type=int, default=1,
                   help="on_the_fly balanced_contact에서 loading 샘플 stride")
    p.add_argument("--saturation-stride", type=int, default=2,
                   help="on_the_fly balanced_contact에서 high-force/saturation 샘플 stride")
    p.add_argument("--use-window-dataset", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="윈도우 데이터셋 사용 (논문 방식, loading phase만). 끄려면 --no-use-window-dataset")
    p.add_argument("--no-lr-scheduler",   action="store_true",
                   help="고정 LR 사용 (ReduceLROnPlateau 비활성)")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _build_parser().parse_args()

    cfg = SATSConfig(
        raw_dir            = args.raw_dir,
        gt_dir             = args.gt_dir,
        dataset_index_path = f"{args.gt_dir}/dataset_index.json",
        out_dir            = args.out_dir,
        run_name           = args.run_name,
        epochs             = args.epochs,
        batch_size         = args.batch_size,
        lr                 = args.lr,
        hidden_dim         = args.hidden_dim,
        num_layers         = args.num_layers,
        dropout            = args.dropout,
        seq_len            = args.seq_len,
        num_workers        = args.num_workers,
        dataloader_prefetch_factor = args.prefetch_factor,
        persistent_workers = args.persistent_workers,
        device             = args.device,
        seed               = args.seed,
        val_trials         = args.val_trials,
        val_ratio          = args.val_ratio,
        exclude_diameters  = args.exclude_diameters,
        window_size        = args.window_size,
        use_window_dataset = args.use_window_dataset,
        use_lr_scheduler   = not args.no_lr_scheduler,
        gt_mode            = args.gt_mode,
        on_the_fly_patch_step_mm = args.on_the_fly_patch_step_mm,
        contact_radius_step_mm   = args.contact_radius_step_mm,
        min_contact_radius_mm    = args.min_contact_radius_mm,
        z_depth_min_mm           = args.z_depth_min_mm,
        z_balance_bin_width_mm   = args.z_balance_bin_width_mm,
        plateau_stride           = args.plateau_stride,
        loading_stride           = args.loading_stride,
        saturation_stride        = args.saturation_stride,
    )

    train(cfg)


if __name__ == "__main__":
    main()
