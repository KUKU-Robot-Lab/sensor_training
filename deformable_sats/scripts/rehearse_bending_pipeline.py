"""밴딩 P1~P3 합성 리허설 — 동결 SATS 실모델(A) 연결 사전 검증.

데이터 취득 전에 파이프라인 전체를 실제 체크포인트로 리허설한다:
  P1: 합성 밴딩-only 윈도우 → BendingEstimator 학습 → deg MAE
  P2: 실제 flat 윈도우 + 합성 밴딩 오프셋 → BaselineRestorer(오프셋 지도) → 복원 잔차
  P3: BendingPipeline(동결 SATS, size 전달) — 보정 전/후 SATS 출력이 flat 기준을
      얼마나 회복하는지 + gradient 경로(restorer만 흐르고 SATS 동결) 검증

합성 밴딩 모델(§5.3): Δs_i = k_i · (deg/90) · z_i  (z_i = 굽힘축 기준 taxel 거리,
k_i = taxel별 감도). 오프셋은 SATS 입력 정규화 공간에 직접 주입 — 리허설 목적은
물리 충실도가 아니라 **배관·인터페이스·복원 회복률** 검증이다.

실행: .venv/bin/python scripts/rehearse_bending_pipeline.py
산출: history/fig_data/fig3_sats_bending/bending/rehearsal/{rehearsal_report.md, rehearsal_summary.png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sats.bending.baseline_restorer import BaselineRestorer  # noqa: E402
from sats.bending.bending_estimator import BendingEstimator  # noqa: E402
from sats.bending.config import BendingConfig  # noqa: E402
from sats.bending.pipeline import BendingPipeline, load_frozen_sats  # noqa: E402
from sats.tools.eval_diagnostics import load_cfg  # noqa: E402
from sats.training.dataset import build_dataloaders  # noqa: E402

RUN_DIR = REPO / "sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3"
OUT_DIR = REPO / "history/fig_data/fig3_sats_bending/bending/rehearsal"

SEED = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_REAL_BATCHES = 4          # 실 val 윈도우 배치 수 (P2/P3용)
DEG_RANGE = 40.0            # 합성 각도 범위 ±40°
OFFSET_SCALE = 0.5          # 40°에서 오프셋 크기 ≈ 신호 std × 이 배율
P1_STEPS, P2_STEPS = 600, 600
N_SYN = 20000               # P1 밴딩-only 합성 윈도우 수


def taxel_z() -> np.ndarray:
    """굽힘축(x축, 중심선) 기준 taxel 수직거리 z_i [16] — 4x4, ±9.75/±3.25 mm."""
    ys = np.array([-9.75, -3.25, 3.25, 9.75], dtype=np.float32)
    return np.repeat(ys, 4)  # Skin(r*4+c+1) → 행 순서


def synth_offset(deg: torch.Tensor, k: torch.Tensor, z: torch.Tensor,
                 scale: float) -> torch.Tensor:
    """Δs[B,16] = scale · k_i · (deg/90) · (z_i/9.75)."""
    d = (deg / 90.0).view(-1, 1)
    return scale * d * (k * (z / 9.75)).view(1, -1)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    # ---- 동결 SATS(A) 로드 --------------------------------------------------
    sats = load_frozen_sats(RUN_DIR, device=DEVICE)
    cfg = load_cfg(RUN_DIR)
    assert getattr(sats, "use_size_input", False), "A 모델이어야 함(use_indenter_size_input)"
    report.append(f"- frozen SATS: `{RUN_DIR.name}` (use_size_input=True), device={DEVICE}")

    # ---- 실제 val 윈도우 확보 (P2/P3) ---------------------------------------
    _, val_loader = build_dataloaders(cfg)
    sensors, sizes, lens = [], [], []
    for i, (sensor_b, meta_b, lengths) in enumerate(val_loader):
        sensors.append(sensor_b)
        sizes.append(meta_b[:, 0])
        lens.append(lengths)
        if i + 1 >= N_REAL_BATCHES:
            break
    real_x = torch.cat(sensors).to(DEVICE)          # [N, T, 16] 정규화 공간
    real_size = torch.cat(sizes).to(DEVICE)         # [N]
    real_len = torch.cat(lens).to(DEVICE)           # [N]
    sig_std = real_x.std().item()
    n_real = real_x.shape[0]
    report.append(f"- 실 val 윈도우 {n_real}개 (shape {tuple(real_x.shape)}), 신호 std={sig_std:.4f}")

    # ---- 합성 밴딩 파라미터 -------------------------------------------------
    z = torch.tensor(taxel_z(), device=DEVICE)
    k = (1.0 + 0.3 * torch.randn(16, device=DEVICE)).clamp(0.5, 1.5)  # taxel별 감도
    off_scale = OFFSET_SCALE * sig_std  # 40°(=deg/90≈0.44)에서 ≈0.22·std per unit(z=9.75)

    bcfg = BendingConfig(window_size=int(real_x.shape[1]), device=DEVICE)

    # ============================ P1: estimator ==============================
    est = BendingEstimator(bcfg).to(DEVICE)
    opt = torch.optim.Adam(est.parameters(), lr=1e-3)
    W = int(real_x.shape[1])
    noise_std = 0.05 * sig_std
    for step in range(P1_STEPS):
        deg = (torch.rand(512, device=DEVICE) * 2 - 1) * DEG_RANGE
        off = synth_offset(deg, k, z, off_scale)                       # [B,16]
        seq = off.unsqueeze(1).expand(-1, W, -1) + noise_std * torch.randn(512, W, 16, device=DEVICE)
        lengths = torch.full((512,), W, device=DEVICE, dtype=torch.long)
        pred = est(seq, lengths)
        loss = F.smooth_l1_loss(pred / 90.0, deg / 90.0)
        opt.zero_grad(); loss.backward(); opt.step()
    est.eval()
    with torch.no_grad():
        deg_v = (torch.rand(4096, device=DEVICE) * 2 - 1) * DEG_RANGE
        off_v = synth_offset(deg_v, k, z, off_scale)
        seq_v = off_v.unsqueeze(1).expand(-1, W, -1) + noise_std * torch.randn(4096, W, 16, device=DEVICE)
        lengths_v = torch.full((4096,), W, device=DEVICE, dtype=torch.long)
        mae = (est(seq_v, lengths_v) - deg_v).abs().mean().item()
    report.append(f"- **P1 estimator**: {P1_STEPS} step 학습 → 밴딩-only deg MAE = **{mae:.2f}°** (±{DEG_RANGE:.0f}° 범위)")

    # ============================ P2: restorer ===============================
    res = BaselineRestorer(bcfg).to(DEVICE)
    opt = torch.optim.Adam(res.parameters(), lr=1e-3)
    for step in range(P2_STEPS):
        idx = torch.randint(0, n_real, (512,), device=DEVICE)
        flat = real_x[idx]
        deg = (torch.rand(512, device=DEVICE) * 2 - 1) * DEG_RANGE
        bent = flat + synth_offset(deg, k, z, off_scale).unsqueeze(1)
        restored = res(bent, deg)                                     # 참 deg 조건(지도)
        loss = F.mse_loss(restored, flat)
        opt.zero_grad(); loss.backward(); opt.step()
    res.eval()
    with torch.no_grad():
        deg_t = (torch.rand(n_real, device=DEVICE) * 2 - 1) * DEG_RANGE
        off_t = synth_offset(deg_t, k, z, off_scale).unsqueeze(1)
        bent_t = real_x + off_t
        rest_t = res(bent_t, deg_t)
        off_rms = off_t.expand_as(real_x).pow(2).mean().sqrt().item()
        resid_rms = (rest_t - real_x).pow(2).mean().sqrt().item()
    report.append(
        f"- **P2 restorer(오프셋 지도)**: 오프셋 RMS {off_rms:.4f} → 복원 잔차 RMS **{resid_rms:.4f}** "
        f"(제거율 {100 * (1 - resid_rms / off_rms):.1f}%)"
    )

    # ============================ P3: pipeline ===============================
    pipe = BendingPipeline(bcfg, sats).to(DEVICE)
    pipe.estimator.load_state_dict(est.state_dict())
    pipe.restorer.load_state_dict(res.state_dict())
    pipe.estimator.eval(); pipe.restorer.eval()

    with torch.no_grad():
        ref, _ = sats(real_x, real_len, real_size)                    # flat 기준 출력
        raw_bent, _ = sats(bent_t, real_len, real_size)               # 보정 없이 밴딩 입력
        deg_hat, corr = pipe(bent_t, real_len, real_size)             # 파이프라인 보정
        rmse_unc = (raw_bent - ref).pow(2).mean().sqrt().item()
        rmse_cor = (corr - ref).pow(2).mean().sqrt().item()
        ref_rms = ref.pow(2).mean().sqrt().item()
        deg_mae_mix = (deg_hat - deg_t).abs().mean().item()
    report.append(
        f"- **P3 pipeline(동결 SATS + size 전달)**: SATS 출력 RMSE vs flat 기준 — "
        f"보정 전 **{rmse_unc:.4f}** → 보정 후 **{rmse_cor:.4f}** "
        f"(회복률 {100 * (1 - rmse_cor / rmse_unc):.1f}%, 기준 출력 RMS {ref_rms:.4f})"
    )
    report.append(
        f"- P3 부수 관찰: 밴딩+접촉 혼합 신호에서 estimator deg MAE = {deg_mae_mix:.2f}° "
        f"(밴딩-only {mae:.2f}° 대비 — 접촉 중첩 시 저하 정도 = 실데이터에서 검증할 리스크)"
    )

    # ---- gradient 경로 검증 (end-to-end B안 가능성) --------------------------
    # 발견: 동결 SATS(eval)의 LSTM 은 cudnn 제약("RNN backward only in training mode")
    # 으로 backward 불가 → e2e(B안) 학습 시 cudnn RNN 을 끄고 통과시켜야 한다.
    pipe.restorer.train()
    with torch.backends.cudnn.flags(enabled=False):
        deg_g, pmap_g = pipe(bent_t[:64], real_len[:64], real_size[:64])
        loss_g = F.mse_loss(pmap_g, ref[:64])
        loss_g.backward()
    r_grads = [p.grad.abs().sum().item() for p in pipe.restorer.parameters() if p.grad is not None]
    s_grads = [p.grad for p in pipe.sats.parameters() if p.grad is not None]
    assert sum(r_grads) > 0, "restorer 로 gradient 가 흐르지 않음"
    assert len(s_grads) == 0, "동결 SATS 에 gradient 가 생김(동결 실패)"
    report.append(
        "- **gradient 검증**: 동결 SATS 통과 backward → restorer grad 흐름 OK, SATS grad 없음(동결 유지) → "
        "**end-to-end(B안) 학습 가능 확인**. 단, **cudnn 제약 발견**: eval 모드 LSTM 은 backward 불가 → "
        "B안 학습 루프는 `torch.backends.cudnn.flags(enabled=False)` 로 감싸야 함(속도 저하 감수)"
    )

    # ---- 요약 figure ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].bar(["uncorrected", "corrected"], [rmse_unc, rmse_cor], color=["#c0392b", "#2ca25f"])
    axes[0].set_ylabel("SATS output RMSE vs flat reference")
    axes[0].set_title("P3: frozen-SATS recovery (synthetic bending)")
    for i, v in enumerate([rmse_unc, rmse_cor]):
        axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    axes[1].bar(["bending-only", "bending+contact"], [mae, deg_mae_mix], color=["#5b8def", "#e07b39"])
    axes[1].set_ylabel("deg MAE (deg)")
    axes[1].set_title("P1: curvature estimation")
    for i, v in enumerate([mae, deg_mae_mix]):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rehearsal_summary.png", dpi=160)

    # ---- 리포트 --------------------------------------------------------------
    header = [
        "# 밴딩 P1~P3 합성 리허설 리포트",
        "",
        f"> 생성: `scripts/rehearse_bending_pipeline.py` (seed {SEED}) — 취득 전 배관·인터페이스 검증.",
        f"> 합성 모델: Δs_i = k_i·(deg/90)·(z_i/9.75)·{OFFSET_SCALE}·σ_signal, ±{DEG_RANGE:.0f}°, "
        f"오프셋은 SATS 정규화 입력 공간에 주입(물리 충실도 아님 — 배관 검증 목적).",
        "",
    ]
    footer = [
        "",
        "## 발견/조치",
        "- `BendingPipeline`이 동결 SATS에 **size(인덴터 지름) 미전달** → A 모델에서 FiLM 누락 버그. "
        "forward 에 `size` 인자 추가 + A 모델에서 누락 시 명시 에러로 수정(이 리허설에서 발견).",
        "",
        "## 실데이터에서 확인할 리스크 (리허설로 대체 불가)",
        "- 밴딩+접촉 중첩 시 deg 추정 저하 폭 (위 P3 부수 관찰 항목의 실측판)",
        "- 실제 오프셋의 z_i 선형성·k_i 안정성 (§5.3 가정)",
        "- 취득 신호의 정규화: bending npz 는 **SATS 학습과 동일한 정규화 공간**으로 변환 후 입력해야 함",
    ]
    (OUT_DIR / "rehearsal_report.md").write_text("\n".join(header + report + footer), encoding="utf-8")
    print("\n".join(report))
    print("saved:", OUT_DIR / "rehearsal_report.md")


if __name__ == "__main__":
    main()
