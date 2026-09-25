"""SATS 원논문 Note S8 재현 — 순수 시뮬레이션 학습 → 실측 sim2real 갭 측정.

시뮬 데이터 = EHS(Boussinesq) 커널 하나로 GT 맵과 **taxel 응답을 동시 생성**:
  GT[B,41,41]   = BatchGPUTargetGenerator(meta)              (학습 GT와 동일 물리)
  s_i           = GT 맵을 16개 taxel 물리좌표에서 bilinear 샘플 × 글로벌 게인
  window[t]     = s_i · (t/W) 로딩 램프 + 가우시안 노이즈      (히스테리시스·점탄성 없음)

meta(diameter,x,y,z_depth,fz)는 실측 val 분포에서 리샘플(공정한 커버리지) + 위치는 ±10mm 균일.
학습 = SATSCNNStage(최종 A 설정과 동일 config, 크기입력 on), sim 배치 스트리밍.
평가 = 실측 val holdout(d5 test10 · d10 test3) rel RMSE — datarich 실측학습(0.188/0.749) 대비.

한계(정직): 시뮬 센서 모델이 조악(램프·무히스테리시스·선형 게인)하므로 결과는
"EHS 시뮬만으로 어디까지 가는가"의 하한 성격. 원논문 Note S8 대응 supplementary.

실행: .venv/bin/python scripts/sim_only_s8.py
산출: history/fig_data/supplementary/S8_sim_only/{S8_report.md, S8_sim_only.png, s8_result.csv}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sats.tools.eval_diagnostics import load_cfg, _load_model  # noqa: E402
from sats.training.cnn_module import SATSCNNStage  # noqa: E402
from sats.training.dataset import build_dataloaders  # noqa: E402
from sats.training.gt_gpu import BatchGPUTargetGenerator  # noqa: E402

RUN_DIR = REPO / "sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3"
OUT_DIR = REPO / "history/fig_data/supplementary/S8_sim_only"
CKPT_DIR = REPO / "sats/training/runs/sim_only_s8"

SEED = 11
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 3000
BATCH = 512
LR = 1e-3
NOISE_FRAC = 0.05          # 시뮬 노이즈 std = 신호 std × 이 값
TAXEL_MM = [-9.75, -3.25, 3.25, 9.75]


def taxel_grid_coords(cfg) -> torch.Tensor:
    """taxel 물리좌표(mm) → grid_sample 정규화 좌표 [-1,1], [16,2](x,y)."""
    xs = torch.tensor(TAXEL_MM)
    gx = xs.repeat(4)                 # Skin(r*4+c+1): c가 빠른 축
    gy = xs.repeat_interleave(4)
    half = float(cfg.grid_max_mm)
    return torch.stack([gx / half, gy / half], dim=1)  # [16,2] in [-1,1]


def sample_taxels(maps: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """GT 맵[B,H,W]을 taxel 좌표에서 bilinear 샘플 → [B,16]."""
    b = maps.shape[0]
    grid = coords.view(1, 1, 16, 2).expand(b, 1, 16, 2).to(maps.device)
    out = F.grid_sample(maps.unsqueeze(1), grid, align_corners=True)  # [B,1,1,16]
    return out.view(b, 16)


def collect_real_meta(val_loader, n_batches: int = 8):
    metas, sensors = [], []
    for i, (sensor_b, meta_b, _len) in enumerate(val_loader):
        metas.append(meta_b)
        sensors.append(sensor_b)
        if i + 1 >= n_batches:
            break
    return torch.cat(metas), torch.cat(sensors)


def make_sim_batch(meta_pool, tgen, coords, gain, W, rng):
    """meta 풀에서 fz/z/dia 리샘플 + 위치 균일 → (window, meta, lengths, target)."""
    idx = torch.randint(0, meta_pool.shape[0], (BATCH,), generator=rng)
    meta = meta_pool[idx].clone()
    meta[:, 1] = (torch.rand(BATCH, generator=rng) * 2 - 1) * 10.0   # x 균일
    meta[:, 2] = (torch.rand(BATCH, generator=rng) * 2 - 1) * 10.0   # y 균일
    meta_d = meta.to(DEVICE)
    target = tgen(meta_d)                                            # [B,41,41]
    s = sample_taxels(target, coords) * gain                         # [B,16]
    ramp = torch.linspace(1.0 / W, 1.0, W, device=DEVICE).view(1, W, 1)
    win = s.unsqueeze(1) * ramp
    win = win + NOISE_FRAC * win.std() * torch.randn_like(win)
    lengths = torch.full((BATCH,), W, device=DEVICE, dtype=torch.long)
    return win, meta_d, lengths, target


@torch.no_grad()
def eval_real(model, cfg, tgen, val_loader) -> dict:
    per_se, tgt_ms, dias = [], [], []
    for sensor_b, meta_b, lengths in val_loader:
        sensor_b = sensor_b.to(DEVICE)
        meta_b = meta_b.to(DEVICE)
        lengths = lengths.to(DEVICE)
        target = tgen(meta_b)
        size = meta_b[:, 0] if getattr(cfg, "use_indenter_size_input", False) else None
        pred, _ = model(sensor_b, lengths, size)
        per_se.append(((pred - target) ** 2).mean(dim=(1, 2)).cpu())
        tgt_ms.append((target ** 2).mean(dim=(1, 2)).cpu())
        dias.append(meta_b[:, 0].cpu())
    se = torch.cat(per_se).numpy()
    tms = torch.cat(tgt_ms).numpy()
    dia = torch.cat(dias).numpy()
    d5 = dia < 7.5

    def rel(m):
        return float(np.sqrt(se[m].mean()) / np.sqrt(tms[m].mean())) if m.sum() else float("nan")

    return {"d5_rel": rel(d5), "d10_rel": rel(~d5), "overall_rel": rel(np.ones_like(d5, bool))}


def main() -> None:
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(RUN_DIR)
    tgen = BatchGPUTargetGenerator(cfg, DEVICE)
    coords = taxel_grid_coords(cfg)

    _, val_loader = build_dataloaders(cfg)
    meta_pool, real_sensors = collect_real_meta(val_loader)
    W = int(real_sensors.shape[1])

    # 게인 캘리브레이션: 시뮬 taxel 응답 스케일을 실측 윈도우 std에 맞춤(1스칼라만 실측 사용)
    with torch.no_grad():
        t0 = tgen(meta_pool[:2048].to(DEVICE))
        s0 = sample_taxels(t0, coords)
        gain = float(real_sensors.std() / (s0.std() + 1e-9))
    print(f"gain={gain:.3f} (real std {real_sensors.std():.3f} / sim std {s0.std():.3f})")

    model = SATSCNNStage(cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(1, STEPS + 1):
        win, meta_d, lengths, target = make_sim_batch(meta_pool, tgen, coords, gain, W, rng)
        size = meta_d[:, 0] if getattr(cfg, "use_indenter_size_input", False) else None
        pred, _ = model(win, lengths, size)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"step {step}/{STEPS}  sim train mse={loss.item():.5f}")

    torch.save({"model": model.state_dict()}, CKPT_DIR / "sim_only_model.pt")
    model.eval()

    # 평가 1: 시뮬 자체 val(동일 분포) — 학습이 됐는지
    with torch.no_grad():
        win, meta_d, lengths, target = make_sim_batch(meta_pool, tgen, coords, gain, W, rng)
        size = meta_d[:, 0] if getattr(cfg, "use_indenter_size_input", False) else None
        pred, _ = model(win, lengths, size)
        sim_rel = float(((pred - target) ** 2).mean().sqrt() / (target ** 2).mean().sqrt())
    # 평가 2: 실측 val holdout — sim2real
    _, val_loader2 = build_dataloaders(cfg)
    real = eval_real(model, cfg, tgen, val_loader2)
    # 참조: 실측 학습(datarich A) 모델
    ref_model = _load_model(RUN_DIR, cfg, DEVICE)
    _, val_loader3 = build_dataloaders(cfg)
    ref = eval_real(ref_model, cfg, tgen, val_loader3)

    rows = [
        ("sim-trained @ sim val", sim_rel, float("nan"), float("nan")),
        ("sim-trained @ real val", real["overall_rel"], real["d5_rel"], real["d10_rel"]),
        ("real-trained(A) @ real val", ref["overall_rel"], ref["d5_rel"], ref["d10_rel"]),
    ]
    import csv as _csv
    with open(OUT_DIR / "s8_result.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["setting", "overall_rel", "d5_rel", "d10_rel"])
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["sim@sim", "sim@real d5", "sim@real d10", "real(A)@real d5", "real(A)@real d10"]
    vals = [sim_rel, real["d5_rel"], real["d10_rel"], ref["d5_rel"], ref["d10_rel"]]
    colors = ["#888", "#e07b39", "#e07b39", "#2ca25f", "#2ca25f"]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("relative RMSE")
    ax.set_title("Note S8 rehearsal — sim-only training vs real training")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "S8_sim_only.png", dpi=160)

    report = f"""# S8 — 순수 시뮬레이션 학습 (SATS 원논문 Note S8 대응)

> 생성: `scripts/sim_only_s8.py` (seed {SEED}, {STEPS} step, batch {BATCH}).
> 시뮬 = EHS 커널로 GT·taxel 응답 동시 생성(taxel 좌표 bilinear 샘플 × 게인 {gain:.3f},
> 로딩 램프 + {NOISE_FRAC:.0%} 노이즈). **히스테리시스·점탄성 없음** — 하한 성격.
> 실측 사용은 게인 1스칼라·meta 분포 리샘플뿐(센서 신호는 미사용).

| setting | overall_rel | d5_rel | d10_rel |
|---|---|---|---|
| sim-trained @ sim val | {sim_rel:.3f} | — | — |
| **sim-trained @ real val** | {real['overall_rel']:.3f} | {real['d5_rel']:.3f} | {real['d10_rel']:.3f} |
| real-trained(A) @ real val (참조) | {ref['overall_rel']:.3f} | {ref['d5_rel']:.3f} | {ref['d10_rel']:.3f} |

해석 가이드:
- sim@sim 이 낮으면 "구조는 시뮬 과제를 학습 가능".
- sim@real vs real@real 차 = **sim2real 갭** — 실센서의 히스테리시스·비선형·크로스토크가
  EHS 선형 시뮬에 없기 때문. 갭이 크면 "실측 데이터 필수" 근거, 작으면 "시뮬 사전학습 가치" 근거.
- 체크포인트: `sats/training/runs/sim_only_s8/sim_only_model.pt`.
"""
    (OUT_DIR / "S8_report.md").write_text(report, encoding="utf-8")
    for r in rows:
        print(r)
    print("saved:", OUT_DIR)


if __name__ == "__main__":
    main()
