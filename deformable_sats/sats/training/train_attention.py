#!/usr/bin/env python3
"""
sats/training/train_attention.py

SATS Self-Attention 단계 학습 스크립트.

학습 전략
----------
1. SATSLSTMStage 체크포인트에서 LSTM 인코더 가중치를 로드한다.
2. 인코더를 동결(freeze)하고, Self-Attention + 프록시 디코더만 학습한다.
3. 기본 타겟: window 마지막 timestep의 GT map (논문식 sliding window)
4. 손실: MSELoss(pred_map, target_gt)

`--no-use-window-dataset`을 주면 legacy peak-map 타겟을 사용한다.

실행
----
python3 -m sats.training.train_attention \
    --lstm-ckpt sats/training/runs/lstm_v1/best_model.pt \
    --run-name  attn_v1

python3 -m sats.training.train_attention \
    --lstm-ckpt sats/training/runs/lstm_v1/best_model.pt \
    --run-name  attn_v1 --epochs 50 --attn-dim 64
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .attention_module import SATSAttentionStage
from .config import SATSConfig
from .dataset import build_dataloaders
from .train_lstm import find_peak_gt, get_target, set_seed, save_checkpoint, write_history, _progress, weighted_mse_loss

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM 인코더 가중치 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_lstm_encoder(
    ckpt_path: Union[str, Path],
    stage: SATSAttentionStage,
    freeze: bool = True,
) -> None:
    """
    SATSLSTMStage 체크포인트에서 LSTM 인코더 가중치를 로드한다.

    SATSLSTMStage.encoder 와 SATSAttentionStage.encoder 는 동일한
    SensorLSTMEncoder 구조이므로, 체크포인트의 "encoder.*" 키를 그대로 사용한다.

    Parameters
    ----------
    ckpt_path : 체크포인트 파일 경로 (SATSLSTMStage로 저장된 것)
    stage     : 가중치를 주입할 SATSAttentionStage
    freeze    : True이면 인코더 파라미터를 동결 (requires_grad=False)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    full_sd = ckpt["model"]

    # "encoder." 접두사를 가진 파라미터 / 버퍼만 추출
    encoder_sd = {
        k[len("encoder."):]: v
        for k, v in full_sd.items()
        if k.startswith("encoder.")
    }
    stage.encoder.load_state_dict(encoder_sd, strict=True)
    log.info("LSTM 인코더 가중치 로드 완료: %s", ckpt_path)

    if freeze:
        for p in stage.encoder.parameters():
            p.requires_grad_(False)
        log.info("LSTM 인코더 동결 완료 (requires_grad=False)")


# ─────────────────────────────────────────────────────────────────────────────
# 학습 / 검증 루프
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: SATSAttentionStage,
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
        sensor_b = sensor_b.to(device, non_blocking=True)
        gt_b     = gt_b.to(device, non_blocking=True)
        lengths  = lengths.to(device, non_blocking=True)

        target = get_target(gt_b, lengths).detach()     # [B, 40, 40]

        pred_map, _ = model(sensor_b, lengths)           # [B, 40, 40]
        loss = F.mse_loss(pred_map, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.clip_grad:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.clip_grad,
            )
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        if progress_desc is not None:
            progress_iter.set_postfix(loss=f"{total_loss / n_batches:.6f}")

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def val_epoch(
    model: SATSAttentionStage,
    loader,
    device: str,
    progress_desc: str | None = None,
) -> dict:
    model.eval()
    total_mse = 0.0
    n_batches = 0

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

    # 데이터 로더
    log.info("데이터 로더 구성 중...")
    train_loader, val_loader = build_dataloaders(cfg)

    # 모델 생성
    model = SATSAttentionStage(cfg).to(device)

    # LSTM 인코더 가중치 로드 + 동결
    if cfg.lstm_ckpt:
        load_lstm_encoder(cfg.lstm_ckpt, model, freeze=True)
    else:
        log.warning("lstm_ckpt 미지정 — LSTM 인코더를 무작위 초기화로 학습합니다.")

    n_total    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("전체 파라미터: %d  학습 가능: %d", n_total, n_trainable)

    # 옵티마이저 (학습 가능한 파라미터만)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = (
        ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience,
        )
        if cfg.use_lr_scheduler else None
    )
    if not cfg.use_lr_scheduler:
        log.info("고정 LR 모드 (lr=%.6f)", cfg.lr)

    history   = []
    best_rmse = float("inf")
    best_ckpt = run_dir / "best_model.pt"
    last_ckpt = run_dir / "last_model.pt"
    hist_path = run_dir / "history.json"

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

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, row)
            log.info("  ★ best val_rmse=%.6f → %s", best_rmse, best_ckpt)

        if epoch % cfg.save_every == 0:
            ckpt_path = run_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, row)

    save_checkpoint(last_ckpt, cfg.epochs, model, optimizer, scheduler, history[-1])

    log.info("학습 이력 저장: %s", hist_path)
    log.info("완료. best val_rmse=%.6f", best_rmse)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SATS Self-Attention 학습")
    p.add_argument("--lstm-ckpt",   default="",
                   help="사전학습된 LSTM 체크포인트 경로 (SATSLSTMStage)")
    p.add_argument("--raw-dir",     default="learning_data/sensor_raw_bin")
    p.add_argument("--gt-dir",      default="learning_data/gt")
    p.add_argument("--out-dir",     default="sats/training/runs")
    p.add_argument("--run-name",    default="attn_v1")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=2048)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden-dim",  type=int,   default=64)
    p.add_argument("--attn-dim",    type=int,   default=64)
    p.add_argument("--num-layers",  type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--seq-len",     type=int,   default=1000)
    p.add_argument("--num-workers", type=int,   default=2)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--val-trials",        nargs="+", default=[])
    p.add_argument("--val-ratio",         type=float, default=0.2,
                   help=">0: 랜덤 sequence-level split. 0: --val-trials 기반 trial split.")
    p.add_argument("--exclude-diameters", nargs="+", type=int, default=[],
                   help="학습/검증 풀에서 제외할 인덴터 직경(mm). 예: --exclude-diameters 10")
    p.add_argument("--window-size",       type=int,   default=10)
    p.add_argument("--use-window-dataset", action=argparse.BooleanOptionalAction, default=True,
                   help="윈도우 데이터셋 사용 (논문 방식). 끄려면 --no-use-window-dataset")
    p.add_argument("--no-lr-scheduler",   action="store_true")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args()

    cfg = SATSConfig(
        lstm_ckpt          = args.lstm_ckpt,
        raw_dir            = args.raw_dir,
        gt_dir             = args.gt_dir,
        out_dir            = args.out_dir,
        run_name           = args.run_name,
        epochs             = args.epochs,
        batch_size         = args.batch_size,
        lr                 = args.lr,
        hidden_dim         = args.hidden_dim,
        attn_dim           = args.attn_dim,
        num_layers         = args.num_layers,
        dropout            = args.dropout,
        seq_len            = args.seq_len,
        num_workers        = args.num_workers,
        device             = args.device,
        seed               = args.seed,
        val_trials         = args.val_trials,
        val_ratio          = args.val_ratio,
        exclude_diameters  = args.exclude_diameters,
        window_size        = args.window_size,
        use_window_dataset = args.use_window_dataset,
        use_lr_scheduler   = not args.no_lr_scheduler,
    )
    train(cfg)


if __name__ == "__main__":
    main()
