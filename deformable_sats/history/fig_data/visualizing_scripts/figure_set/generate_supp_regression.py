#!/usr/bin/env python3
"""FigS30 — 좌표/힘 회귀 (localization). SATS head를 회귀 MLP로 교체해 (x,y,fz) 직접 추론.

논문 FigS30 대응: local-map/CNN 대신 회귀 head → 접촉 좌표·힘 추정. 위치오차·힘오차·
localization scale factor(Note S1: S/(N·π·ε²)) 산출.

encoder + attention 은 SATS 와 동일 구조 재사용(가중치는 새로 학습).

사용::

    .venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_supp_regression.py \
        --epochs 40                 # 학습 + 평가 + 그림
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT_DIR = REPO / "history/fig_data/supplementary/S30_regression"
CKPT = OUT_DIR / "regression_ecomesh.pt"
BASE_CFG = REPO / "sats/training/runs/xy1_material_d5d10/xy1_d5d10_ecomesh_xy1_fold3_e2e_g05/config.json"
XY_SCALE = 10.0    # x,y ∈ [-10,10] → [-1,1]
FZ_SCALE = 5.0     # fz 정규화 스케일 [N]
SENSING_AREA_MM2 = (2 * 10.0) ** 2  # 감지면 면적(≈ 20×20 mm²) — scale factor용
N_PHYSICAL = 16


class RegressionSATS(nn.Module):
    """encoder(LSTM) + self-attention + 회귀 MLP → (x,y,fz) 정규화 벡터."""

    def __init__(self, cfg) -> None:
        super().__init__()
        from sats.training.lstm_module import SensorLSTMEncoder
        from sats.training.attention_module import SATSSelfAttention

        self.encoder = SensorLSTMEncoder(
            n_sensors=cfg.n_sensors, hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers, dropout=cfg.dropout, bidirectional=cfg.bidirectional)
        self.attention = SATSSelfAttention(
            in_dim=self.encoder.out_dim, attn_dim=cfg.attn_dim,
            n_sensors=cfg.n_sensors, n_layers=cfg.n_gat_layers)
        combined = self.encoder.out_dim + cfg.attn_dim
        self.head = nn.Sequential(
            nn.Linear(combined, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2), nn.Linear(64, 3))

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lf = self.encoder(seq, lengths)
        af = self.attention(lf)
        feat = torch.cat([lf, af], dim=-1).mean(dim=1)   # 16 노드 평균 풀
        return self.head(feat)                            # [B,3] = (x,y,fz)/scale


def _load_cfg():
    import json
    from dataclasses import fields
    from sats.training.config import SATSConfig
    data = json.loads(BASE_CFG.read_text())
    valid = {f.name for f in fields(SATSConfig)}
    return SATSConfig(**{k: v for k, v in data.items() if k in valid})


def _target(meta: torch.Tensor) -> torch.Tensor:
    """meta(diameter,x,y,z,fz) → 정규화 (x,y,fz)."""
    return torch.stack([meta[:, 1] / XY_SCALE, meta[:, 2] / XY_SCALE,
                        meta[:, 4] / FZ_SCALE], dim=1)


def train_regression(epochs: int) -> None:
    from sats.training.dataset import build_dataloaders

    cfg = _load_cfg()
    device = cfg.effective_device()
    train_loader, _ = build_dataloaders(cfg)
    model = RegressionSATS(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0; n = 0
        for sensor_b, meta_b, lengths in train_loader:
            sensor_b, meta_b, lengths = (t.to(device) for t in (sensor_b, meta_b, lengths))
            pred = model(sensor_b, lengths)
            loss = F.mse_loss(pred, _target(meta_b))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"  epoch {ep:3d}/{epochs}  train_mse={tot / max(n, 1):.5f}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, CKPT)
    print("saved:", CKPT)


@torch.no_grad()
def evaluate() -> dict[str, np.ndarray]:
    from sats.training.dataset import build_dataloaders

    cfg = _load_cfg()
    device = cfg.effective_device()
    _, val_loader = build_dataloaders(cfg)
    model = RegressionSATS(cfg).to(device).eval()
    model.load_state_dict(torch.load(CKPT, map_location=device)["model"])
    gx, gy, gf, pe, fe = [], [], [], [], []
    for sensor_b, meta_b, lengths in val_loader:
        sensor_b, meta_b, lengths = (t.to(device) for t in (sensor_b, meta_b, lengths))
        pred = model(sensor_b, lengths).cpu().numpy()
        tx = meta_b[:, 1].cpu().numpy(); ty = meta_b[:, 2].cpu().numpy(); tf = meta_b[:, 4].cpu().numpy()
        px = pred[:, 0] * XY_SCALE; py = pred[:, 1] * XY_SCALE; pf = pred[:, 2] * FZ_SCALE
        contact = tf > 0.1  # 접촉만
        gx.append(tx[contact]); gy.append(ty[contact]); gf.append(tf[contact])
        pe.append(np.hypot(tx[contact] - px[contact], ty[contact] - py[contact]))
        fe.append(np.abs(tf[contact] - pf[contact]))
    return {"gx": np.concatenate(gx), "gy": np.concatenate(gy), "gf": np.concatenate(gf),
            "pos_err": np.concatenate(pe), "force_err": np.concatenate(fe)}


def plot(d: dict[str, np.ndarray]) -> None:
    from scipy.stats import binned_statistic_2d

    pos_mean = float(d["pos_err"].mean())
    force_mean = float(d["force_err"].mean())
    scale = SENSING_AREA_MM2 / (N_PHYSICAL * np.pi * pos_mean ** 2)  # Note S1 alt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    # 위치오차 2D 맵
    stat, _, _, _ = binned_statistic_2d(d["gx"], d["gy"], d["pos_err"], statistic="mean",
                                        bins=20, range=[(-10, 10), (-10, 10)])
    im = axes[0].imshow(stat.T, origin="lower", extent=[-10, 10, -10, 10], aspect="equal",
                        cmap="Greens", vmin=0, vmax=float(np.nanquantile(d["pos_err"], 0.95)))
    axes[0].set_title("position error by location"); axes[0].set_xlabel("X [mm]"); axes[0].set_ylabel("Y [mm]")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="position error [mm]")
    # force별 위치오차
    edges = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0]); xt, mn, se = [], [], []
    for i in range(len(edges) - 1):
        s = (d["gf"] >= edges[i]) & (d["gf"] < edges[i + 1])
        if s.sum() > 20:
            v = d["pos_err"][s]; xt.append(f"{edges[i]:.2g}–{edges[i+1]:.2g}")
            mn.append(v.mean()); se.append(v.std() / np.sqrt(v.size))
    axes[1].bar(range(len(mn)), mn, yerr=se, capsize=3, color="#8856a7", edgecolor="k", alpha=0.85)
    axes[1].set_xticks(range(len(xt))); axes[1].set_xticklabels(xt, rotation=30, ha="right", fontsize=8)
    axes[1].set_title(f"position error vs force (mean={pos_mean:.2f} mm)")
    axes[1].set_xlabel("force fz [N]"); axes[1].set_ylabel("position error [mm]"); axes[1].grid(axis="y", ls=":", alpha=0.4)
    # 힘 추정 산점
    axes[2].hist(d["force_err"], bins=50, color="#8856a7", alpha=0.8)
    axes[2].axvline(force_mean, ls="--", c="k")
    axes[2].set_title(f"force error (mean={force_mean:.3f} N)")
    axes[2].set_xlabel("|fz − fẑ| [N]"); axes[2].set_ylabel("count"); axes[2].grid(ls=":", alpha=0.4)

    fig.suptitle(f"Coordinate/force regression (ecomesh) — "
                 f"loc {pos_mean:.2f} mm · force {force_mean:.3f} N · SR scale ≈ {scale:.0f}", y=1.03)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "S30_regression_ecomesh.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}  (pos={pos_mean:.3f}mm force={force_mean:.3f}N scale={scale:.0f})")


def main() -> None:
    p = argparse.ArgumentParser(description="FigS30 좌표/힘 회귀")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--skip-train", action="store_true", help="기존 체크포인트로 평가만")
    args = p.parse_args()
    if not args.skip_train:
        train_regression(args.epochs)
    plot(evaluate())


if __name__ == "__main__":
    main()
