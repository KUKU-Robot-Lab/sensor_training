"""v6 밴딩 데이터 live 스타일(초록 heatmap) 시각화 — 밴딩→복원→SATS 데모용.

4패널(모두 동결 SATS 41×41 출력, live와 동일한 초록 컬러맵):
  (1) 밴딩(무접촉) → SATS           : 밴딩이 만드는 가짜 접촉 환각
  (2) 밴딩 → restorer 복원 → SATS   : 환각 억제(≈비어야 정상)
  (3a) flat + 접촉 → SATS           : 밴딩 없이 같은 지점(reference)
  (3b) 밴딩 + 접촉 → 복원 → SATS    : 같은 지점, 밴딩 상태서 복원 결과

(3)은 준-합성(§5.4 가법): 실측 flat 접촉 pct + 실측 밴딩 오프셋 pct → 복원.
같은 접촉이라 (3a)와 (3b)를 공통 스케일로 나란히 비교. 텍스트는 영어만(한글 □ 깨짐).

실행: .venv/bin/python -m sats.inference.visualize_bending_correction \
        [--out out.png] [--sats-run ...] [--bending-dir learning_data/bending/v6] \
        [--contact-trial learning_data/sensor_raw_bin/ecomesh_v6_xy1/d10/z_3.5mm/test1]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from sats.bending.config import BendingConfig
from sats.bending.eval_contact_preservation import contact_windows
from sats.bending.eval_pipeline import _sats_map, _windows
from sats.bending.pipeline import load_frozen_sats
from sats.inference.bending_infer import load_restorer

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SATS = _ROOT / "sats/training/runs/ecomesh_v6_deploy_all4"
_DEFAULT_BEND = _ROOT / "learning_data/bending/v6_new" if (_ROOT / "learning_data/bending/v6_new").exists() \
    else _ROOT / "learning_data/bending/v6"      # 신규 v6(y23-33) 우선
_DEFAULT_CONTACT = _ROOT / "learning_data/sensor_raw_bin/ecomesh_v6_xy1/d10/z_3.5mm/test1"


def _map_np(sats, seq_np: np.ndarray) -> np.ndarray:
    """pct 윈도우[1,W,16] → SATS 41×41(양압)."""
    t = torch.from_numpy(seq_np.astype(np.float32)).to(next(sats.parameters()).device)
    m = _sats_map(sats, t)[0].cpu().numpy()
    return np.clip(m, 0, None)


def _map_mean(sats, seq_np: np.ndarray) -> np.ndarray:
    """pct 윈도우[K,W,16] → SATS 맵 평균(대표 맵, 단일프레임 잡음 완화)."""
    t = torch.from_numpy(seq_np.astype(np.float32)).to(next(sats.parameters()).device)
    m = _sats_map(sats, t).cpu().numpy()                 # [K,41,41]
    return np.clip(m, 0, None).mean(0)


def build_panels(sats_run: Path, bending_dir: Path, contact_trial: Path, device: str,
                 target_delta: float = 10.0) -> dict:
    """4패널 SATS 맵 + 메타 계산."""
    sats = load_frozen_sats(sats_run, device)
    cfg = BendingConfig(restorer_mode="deg_cnn")   # 공간CNN 복원(억제율↑)
    W = cfg.window_size
    trials = sorted(p.stem for p in bending_dir.glob("*.npz"))
    if not trials:
        raise FileNotFoundError(f"밴딩 npz 없음: {bending_dir}")

    print(f"[viz] restorer 학습(v6 buckling, e2e through frozen SATS, ~1분)...")
    restorer = load_restorer(sats_run, bending_dir, device=device, cfg=cfg)

    # target δ 근처 밴딩 윈도우들 — 단일프레임은 edge 잔차로 대표성↓ → 여러장 평균(대표 맵)
    bX, bdeg, bdelta, _ = _windows(bending_dir, trials, W)
    sel = np.where(np.abs(bdelta - target_delta) < 1.5)[0]
    if len(sel) == 0:
        sel = np.array([int(np.argmin(np.abs(bdelta - target_delta)))])
    sel = sel[np.linspace(0, len(sel) - 1, min(len(sel), 60)).astype(int)]
    bent = bX[sel]                                       # [K,W,16]
    deg_k = bdeg[sel]
    theta_mean = float(np.mean(deg_k))
    print(f"[viz] 밴딩 프레임 {len(sel)}장 평균  δ≈{np.mean(bdelta[sel]):.1f}mm  θ≈{theta_mean:.0f}°")
    dg = torch.from_numpy(deg_k.astype(np.float32)).to(device)

    # 접촉(flat) 윈도우 — off-center 하나 선택(peak 뚜렷). 같은 접촉에 각 밴딩 오프셋 합성.
    cwin, cxy = contact_windows(contact_trial, W, n=80, fz_min=0.5)
    off = np.linalg.norm(cxy, axis=1)
    k = int(np.argmax(off * (off < 8)))                 # 중심서 떨어졌지만 grid 안
    contact = cwin[k:k + 1]
    print(f"[viz] 선택 접촉 위치: ({cxy[k,0]:+.1f},{cxy[k,1]:+.1f}) mm")

    restored_nc = restorer(torch.from_numpy(bent).to(device), dg).detach().cpu().numpy()
    synth = contact + bent                               # [K,W,16] 같은 접촉 + 각 밴딩
    restored_c = restorer(torch.from_numpy(synth).to(device), dg).detach().cpu().numpy()

    p1 = _map_mean(sats, bent)
    p2 = _map_mean(sats, restored_nc)
    supp = 100.0 * (1.0 - p2.mean() / max(p1.mean(), 1e-9))
    print(f"[viz] 밴딩 환각 억제율(평균 |맵|): {supp:.1f}%")
    return {
        "bent_units": bent.reshape(-1, 16).mean(0),      # 밴딩 상태 16-taxel Δp(SATS 입력 원본)
        "p1_bent_nc": p1,                                # 밴딩 무접촉 → 환각(평균)
        "p2_restored_nc": p2,                            # 복원 → 억제(평균)
        "p3a_flat_contact": _map_np(sats, contact),      # flat 접촉(reference)
        "p3b_bent_restored_contact": _map_mean(sats, restored_c),  # 밴딩+접촉 복원(평균)
        "contact_xy": (float(cxy[k, 0]), float(cxy[k, 1])),
        "theta": theta_mean, "delta": float(np.mean(bdelta[sel])), "suppress": supp,
    }


def _draw_units(ax, fig, dp16: np.ndarray, theta: float) -> None:
    """(0) 밴딩 상태 16-taxel 센싱유닛 heatmap(SATS 입력 원본, 발산형)."""
    from sats.bending.geometry import TAXEL_XY_MM
    from sats.inference.demo_viz import taxel_grid
    vmax = max(float(np.abs(dp16).max()), 1e-6)
    im = ax.imshow(taxel_grid(dp16), origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[-13, 13, -13, 13], interpolation="nearest")
    for i in range(16):
        x, y = TAXEL_XY_MM[i + 1]
        ax.text(x, y, f"{dp16[i]:+.1f}", ha="center", va="center", fontsize=7,
                color="k" if abs(dp16[i]) < vmax * 0.6 else "w")
    ax.set_title(f"(0) bent sensing units (16 taxel)\nSATS input, theta~{theta:.0f}deg", fontsize=10)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("dp (%)", fontsize=8)


def render(panels: dict, out_png: Path, grid_min: float = -10.0, grid_max: float = 10.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sats.inference.demo_viz import _green_cmap
    cmap = _green_cmap()
    ext = [grid_min, grid_max, grid_min, grid_max]
    cx, cy = panels["contact_xy"]

    g12 = max(panels["p1_bent_nc"].max(), 1e-6)          # (1,2) 밴딩 환각 비교
    g3 = max(panels["p3a_flat_contact"].max(), panels["p3b_bent_restored_contact"].max(), 1e-6)  # (3a,3b)
    specs = [
        ("p1_bent_nc", g12, f"(1) bent, no contact -> SATS\n(hallucination, theta~{panels['theta']:.0f}deg)", None),
        ("p2_restored_nc", g12, "(2) bent -> restored -> SATS\n(hallucination suppressed)", None),
        ("p3a_flat_contact", g3, "(3a) flat + contact -> SATS\n(reference, no bending)", (cx, cy)),
        ("p3b_bent_restored_contact", g3, "(3b) bent + contact -> restored -> SATS\n(same point, corrected)", (cx, cy)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    _draw_units(axes[0, 0], fig, panels["bent_units"], panels["theta"])  # (0) 센싱유닛
    slots = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    for ax, (key, vmax, title, mark) in zip(slots, specs):
        im = ax.imshow(np.clip(panels[key] / vmax, 0, 1), origin="lower", extent=ext,
                       cmap=cmap, vmin=0, vmax=1.0, aspect="equal", interpolation="bicubic")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        if mark is not None:
            ax.plot([mark[0]], [mark[1]], marker="+", color="#d81e00", markersize=15, markeredgewidth=2.2)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("relative intensity", fontsize=8)
    axes[1, 2].axis("off")                               # 빈 슬롯
    fig.suptitle("v6 bending correction: sensing units -> SATS -> restored (live-style)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"[viz] 저장: {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="v6 밴딩 보정 live 스타일 시각화")
    p.add_argument("--out", default=str(_ROOT / "sats/inference/bending_correction_viz.png"))
    p.add_argument("--sats-run", default=str(_DEFAULT_SATS))
    p.add_argument("--bending-dir", default=str(_DEFAULT_BEND))
    p.add_argument("--contact-trial", default=str(_DEFAULT_CONTACT))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target-delta", type=float, default=10.0, help="시각화할 밴딩 세기(δ mm). 클수록 강한 밴딩")
    args = p.parse_args()
    panels = build_panels(Path(args.sats_run), Path(args.bending_dir), Path(args.contact_trial),
                          args.device, target_delta=args.target_delta)
    render(panels, Path(args.out))


if __name__ == "__main__":
    main()
