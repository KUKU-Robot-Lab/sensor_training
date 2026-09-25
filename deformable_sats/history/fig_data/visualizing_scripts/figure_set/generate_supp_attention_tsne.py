#!/usr/bin/env python3
"""FigS29 — self-attention 해석성 (t-SNE). attention 전/후 feature 공간 비교.

한 감지 유닛(node)의 인코딩 feature를 attention 모듈 전/후로 각각 t-SNE 2D 축소하고,
press 위치로 색을 매핑한다. attention 후 feature가 위치(공간정보)로 정렬되면(=색이 구조화),
self-attention 이 공간 디커플링을 수행함을 뜻한다(논문 FigS29 논지).

산출(모델당 1 파일, 좌=before / 우=after):

사용::

    .venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_supp_attention_tsne.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "history/fig_data/supplementary/S29_attention_tsne"
N_TSNE = 2500      # t-SNE 입력 표본 수(과밀·속도 균형)
CONTACT_HALF_MM = 9.75

RUNS: dict[str, Path] = {
    "eco20_xy1": REPO / "sats/training/runs/size_input_material/sizeA_eco20_xy1_fold2_e2e_g05",
    "eco50_xy1": REPO / "sats/training/runs/size_input_material/sizeA_eco50_xy1_fold1_e2e_g05",
    "ecomesh_xy1": REPO / "sats/training/runs/size_input_material/sizeA_ecomesh_xy1_fold3_e2e_g05",
    "ecomesh_xy0p5_final": REPO / "sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3",
}


def collect_features(run_dir: Path) -> dict[str, np.ndarray]:
    """attention 전/후 feature와 press 위치를 수집. 가장 활성 높은 node 를 자동 선택."""
    import torch

    from sats.tools.eval_diagnostics import load_cfg, _load_model
    from sats.training.gt_gpu import BatchGPUTargetGenerator
    from sats.training.dataset import build_dataloaders

    cfg = load_cfg(run_dir)
    device = cfg.effective_device()
    _, val_loader = build_dataloaders(cfg)
    model = _load_model(run_dir, cfg, device)
    tgen = BatchGPUTargetGenerator(cfg, device)

    before, after, px, py = [], [], [], []
    with torch.no_grad():
        for sensor_b, meta_b, lengths in val_loader:
            sensor_b, meta_b, lengths = (t.to(device) for t in (sensor_b, meta_b, lengths))
            target = tgen(meta_b)
            contact = target.amax(dim=(1, 2)) > 1e-3
            if not bool(contact.any()):
                continue
            lf = model.encoder(sensor_b, lengths)     # [B, nodes, D] attention 전
            af = model.attention(lf)                  # [B, nodes, D] attention 후
            before.append(lf[contact].cpu().numpy())
            after.append(af[contact].cpu().numpy())
            px.append(meta_b[contact, 1].cpu().numpy())
            py.append(meta_b[contact, 2].cpu().numpy())
    before = np.concatenate(before)   # [N, nodes, D]
    after = np.concatenate(after)
    px, py = np.concatenate(px), np.concatenate(py)

    # 활성 node = attention 전 feature 노름 분산이 가장 큰 유닛
    node_var = before.reshape(before.shape[0], before.shape[1], -1)
    node = int(np.linalg.norm(node_var, axis=2).var(axis=0).argmax())
    # 그 node 응답이 강한(=근처 press) 표본만
    norm = np.linalg.norm(before[:, node, :], axis=1)
    strong = norm >= np.quantile(norm, 0.5)
    idx = np.where(strong)[0]
    rng = np.random.default_rng(0)
    if idx.size > N_TSNE:
        idx = rng.choice(idx, N_TSNE, replace=False)
    return {
        "before": before[idx, node, :], "after": after[idx, node, :],
        "px": px[idx], "py": py[idx], "node": np.array([node]),
    }


def _pos_to_rgb(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """press 위치(x,y)를 2D→RGB 로 인코딩(가까운 위치=가까운 색)."""
    xn = (px - px.min()) / (np.ptp(px) + 1e-9)
    yn = (py - py.min()) / (np.ptp(py) + 1e-9)
    return np.stack([xn, yn, 0.6 * (1 - xn)], axis=1)


def plot_tsne(label: str, d: dict[str, np.ndarray]) -> None:
    from sklearn.manifold import TSNE

    colors = _pos_to_rgb(d["px"], d["py"])
    node = int(d["node"][0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, (tag, feat) in zip(axes, [("before attention", d["before"]),
                                      ("after attention", d["after"])]):
        emb = TSNE(n_components=2, init="pca", perplexity=30,
                   random_state=0).fit_transform(feat)
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=6, alpha=0.75)
        ax.set_title(f"{label}  ·  node {node}  ·  {tag}", fontsize=10)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.set_xticks([]); ax.set_yticks([])
    # 위치→색 범례(작은 2D 컬러 패치)
    cax = fig.add_axes([0.46, 0.16, 0.08, 0.16])
    gx, gy = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    patch = np.stack([gx, gy, 0.6 * (1 - gx)], axis=2)
    cax.imshow(patch, origin="lower", extent=[0, 1, 0, 1])
    cax.set_title("press pos", fontsize=7)
    cax.set_xlabel("x", fontsize=7); cax.set_ylabel("y", fontsize=7)
    cax.set_xticks([]); cax.set_yticks([])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"S29_attention_tsne_{label}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}  (node={node}, n={d['px'].size})")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="FigS29 attention 해석성 t-SNE")
    p.add_argument("--models", nargs="+", default=list(RUNS), choices=list(RUNS))
    args = p.parse_args()
    for label in args.models:
        print(f"--- {label} ---")
        plot_tsne(label, collect_features(RUNS[label]))


if __name__ == "__main__":
    main()
