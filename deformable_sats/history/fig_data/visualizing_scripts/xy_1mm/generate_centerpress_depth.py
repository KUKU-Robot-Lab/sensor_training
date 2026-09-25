"""Fig.2 — center press (0,0) 깊이별 16-taxel 응답 (|ΔS| 절댓값, 인덴터별).

xy_1mm 격자 압입 중 **센터(0,0) 셀**(|x|,|y|<0.7mm)만 뽑아, 침투 깊이 두 단계
  · ≈1 mm  (접촉 onset +1 mm)
  · max    (최대 침투 ≈2 mm)
에서 16 taxel 응답을 4×4 센서 배열 맵으로 그린다.

신호: ΔS_i = (raw_i − baseline_i)/baseline_i × 100  → **절댓값 |ΔS| 사용**.
  이유: baseline 차감으로 +/− 부호가 생기는데, 압입 시 폴리머 챔버가 인장/압축
  어느 쪽으로 측정될지 불확실하므로 "변화의 크기"인 |ΔS| 로 비교한다.
  실측 무접촉 노이즈 std ≈ 0.01% → floor 0.1%(=10×noise) 이상만 신호로 본다.

스케일: **인덴터별(figure별) 자체 vmax** — d5(약함)와 d10(강함)을 각자 스케일로 보여
  패턴이 보이게 함. 같은 figure 안 1mm/max 두 깊이는 공통 스케일(깊이 효과 보존).

출력: Analysis_Results/Fig2_centerpress_d5.png , _d10.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import generate_2d_heatmap as g2

ORDER = ["eco20", "eco50", "ecomesh"]
LABEL = {"eco20": "eco20", "eco50": "eco50", "ecomesh": "ecomesh (mesh20)"}
COLORS = {"eco20": "#3b6ea5", "eco50": "#e08a1e", "ecomesh": "#2e8b57"}
SK = g2.SKIN_COLS
DS = [f"dS_{c}" for c in SK]
SENSOR_POS = np.array([g2.SENSOR_XY[c] for c in SK])
CENTER_R = 0.7
NOISE = 0.10          # % peak 이게 안 되면 신호 없음으로 early-return
FLOOR_ABS = 0.50      # % active/σ 절대임계 (local baseline 후 잔류 노이즈~0.1~0.2% 의 ~3배;
                      #   d5 처럼 약신호일 때 먼 taxel 노이즈가 σ 를 부풀리는 것 방지. panelC 와 동일)
CENTRAL = [6, 7, 10, 11]   # (0,0) 주변 taxel
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


def to_grid(vec16):
    g = np.full((4, 4), np.nan)
    for i in range(16):
        r, c = divmod(i, 4)
        g[r, c] = vec16[i]
    return g


def array_metrics(vec16):
    """|ΔS| 기준. active/σ 는 절대 floor(0.5%)+상대(0.2·peak) 임계로 노이즈 억제."""
    a = np.abs(vec16)
    peak = float(a.max())
    total = float(a.sum())
    if peak < NOISE:
        return peak, 0, 0.0, total
    thr = max(0.20 * peak, FLOOR_ABS)
    keep = a > thr
    active = int(keep.sum())
    w = np.where(keep, a, 0.0)
    if w.sum() > 0:
        cen = (w[:, None] * SENSOR_POS).sum(0) / w.sum()
        d2 = ((SENSOR_POS - cen) ** 2).sum(1)
        sigma = float(np.sqrt((w * d2).sum() / w.sum()))
    else:
        sigma = 0.0
    return peak, active, sigma, total


def _rep(window, k=10):
    """깊이 구간의 대표 응답 = 응답 상위 k 샘플 평균.

    한 z-구간에는 접근/하중/해제 통과 샘플이 섞여 단순 평균은 '실제 압입된' 응답을
    희석한다(특히 eco50 처럼 깊이 임계가 가파른 경우). 그 구간에서 실제로 눌린
    상태(=총 |ΔS| 상위)만 평균해 '그 깊이까지 눌렀을 때'의 응답을 대표한다.
    """
    if len(window) == 0:
        return np.full(16, np.nan)
    tot = window[DS].abs().sum(axis=1).values
    k = min(k, len(window))
    idx = np.argsort(tot)[-k:]
    return window[DS].iloc[idx].mean().values


def slices(dia):
    """깊이별 '압입으로 유발된' per-taxel 응답 (국소 drift 제거).

    글로벌 baseline(스캔 첫 구간) 대비 ΔS 는 센터 셀 도달 시점까지 누적된 drift 가
    섞여(특히 약신호 d5), 압입과 무관한 offset 이 taxel 마다 남는다. 그래서 센터 셀의
    무접촉(=총|ΔS| 하위 30%) 샘플 per-taxel 평균을 국소 drift 로 보고 빼서, 압입이
    실제로 유발한 변화만 남긴다. (예: eco20 d5 는 중앙=코너로 깊이 무관 평평 → 보정 후 ≈0.)
    """
    out = {}
    for name in ORDER:
        df, _ = g2.load_material(g2.DATASETS[dia][name])
        c = df[(df.x_mm.abs() < CENTER_R) & (df.y_mm.abs() < CENTER_R)].copy()
        tot = c[DS].abs().sum(axis=1)
        drift = c[tot <= tot.quantile(0.30)][DS].mean().values     # 국소 무접촉 per-taxel
        cc = c[c.z_mm > 5]
        onset, zmax = cc.z_mm.min(), cc.z_mm.max()
        v1 = _rep(c[(c.z_mm >= onset + 0.5) & (c.z_mm <= onset + 1.5)]) - drift
        vx = _rep(c[c.z_mm >= zmax - 0.6]) - drift
        out[name] = {"1mm": v1, "max": vx}
    return out


def main(dia):
    data = slices(dia)
    DEPTHS = [("1mm", "≈1 mm penetration"), ("max", "max depth (≈2 mm)")]
    NOISE_VIS = 0.6   # % : 이 아래 패널은 정규화하면 노이즈 증폭 → '≈noise' 표시

    fig = plt.figure(figsize=(13, 11))
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.85], hspace=0.5, wspace=0.28,
                  left=0.07, right=0.9, top=0.9, bottom=0.08)
    last_im = None
    for ri, (dk, dlabel) in enumerate(DEPTHS):
        for ci, name in enumerate(ORDER):
            ax = fig.add_subplot(gs[ri, ci])
            vec = data[name][dk]
            a = np.abs(vec)
            peak, active, sigma, total = array_metrics(vec)
            data[name][dk + "_m"] = (peak, active, sigma, total)
            # ★ 소재별 자체 정규화: 각 패널을 자기 |peak|=1 로 → '퍼짐(모양)' 공정 비교
            denom = peak if peak > 1e-6 else 1.0
            gridn = np.abs(to_grid(vec)) / denom
            noisy = peak < NOISE_VIS
            last_im = ax.imshow(gridn, origin="lower", cmap="magma", vmin=0, vmax=1,
                                extent=[-13, 13, -13, 13], interpolation="nearest")
            for i in range(16):
                r, cc = divmod(i, 4)
                sx, sy = SENSOR_POS[i]
                ax.text(sx, sy, f"{a[i]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if gridn[r, cc] < 0.6 else "black")
            for t in CENTRAL:
                sx, sy = SENSOR_POS[t - 1]
                ax.add_patch(plt.Rectangle((sx - 3.0, sy - 3.0), 6.0, 6.0, fill=False,
                                           ec="#39d0ff", lw=1.0, ls=":"))
            tcol = "#b03a2e" if noisy else "#1d2330"
            note = "  (≈noise)" if noisy else ""
            ax.set_title(f"{LABEL[name]}{note}\n|peak| {peak:.1f}%  active {active}  σ {sigma:.1f}mm",
                         fontsize=10.5, color=tcol)
            ax.set_xticks([-9.75, -3.25, 3.25, 9.75]); ax.set_yticks([-9.75, -3.25, 3.25, 9.75])
            ax.tick_params(labelsize=7)
            if ci == 0:
                ax.set_ylabel(f"{dlabel}\n\nsensor y (mm)", fontsize=10)
            ax.set_xlabel("sensor x (mm)", fontsize=8)

    cb = fig.colorbar(last_im, ax=fig.axes[:6], fraction=0.016, pad=0.015)
    cb.set_label("fraction of each panel's own |peak|  (shape / spread)", fontsize=9)

    # ── 하단: 퍼짐(SR 관련) 2개 + 절대 민감도 1개 ──
    ax_a = fig.add_subplot(gs[2, 0]); ax_s = fig.add_subplot(gs[2, 1]); ax_t = fig.add_subplot(gs[2, 2])
    actives = [data[n]["max_m"][1] for n in ORDER]
    sigmas = [data[n]["max_m"][2] for n in ORDER]
    totals = [data[n]["max_m"][3] for n in ORDER]
    peaks = [data[n]["max_m"][0] for n in ORDER]
    panels = [(ax_a, actives, "SPREAD — active taxels", "N", "{:d}", "#2e8b57"),
              (ax_s, sigmas, "SPREAD — σ (mm)", "mm", "{:.1f}", "#2e8b57"),
              (ax_t, totals, "ABS. sensitivity — total |ΔS|", "%", "{:.1f}", "#888")]
    for ax, vals, ttl, ylab, fmt, tc in panels:
        bars = ax.bar(ORDER, vals, color=[COLORS[n] for n in ORDER], alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    fmt.format(int(v) if ylab == "N" else v), ha="center", va="bottom", fontsize=9)
        ax.set_title(ttl + "  (max)", fontsize=10, color=tc, fontweight="bold")
        ax.set_ylabel(ylab); ax.grid(axis="y", alpha=0.3); ax.margins(y=0.2)
        ax.tick_params(axis="x", labelsize=9)
    # 퍼짐에서 mesh 최대면 강조
    for ax, vals in [(ax_a, actives), (ax_s, sigmas)]:
        if vals[2] == max(vals) and vals[2] > 0:
            ax.annotate("mesh ↑", xy=(2, vals[2]), ha="center", va="bottom", fontsize=9.5,
                        color="#2e8b57", fontweight="bold", xytext=(2, max(vals) * 1.04))

    fig.suptitle(f"Fig. 2  Center press (0,0) — per-material normalized receptive field  ({dia})",
                 fontsize=14.5, fontweight="bold", y=0.965)

    fig.text(0.5, 0.018,
             "Each panel is normalized to its OWN |peak| (color = receptive-field SHAPE/spread, not absolute magnitude) so the three materials "
             "are compared by how WIDELY one press spreads — eco20 localized · eco50 concentrated/strong · mesh widest. "
             "Dotted boxes = central taxels (6,7,10,11). Cell numbers = absolute |press-induced ΔS| (local drift removed); absolute sensitivity is the separate bottom-right bar.",
             ha="center", va="bottom", fontsize=8.6, color="#555")

    weak = max(peaks) < 3.0
    if weak:
        fig.text(0.5, -0.024,
                 f"⚠ low amplitude ({dia}): ⌀5 mm indenter at (0,0) sits between taxels (6.5 mm pitch) → weak coupling (|peak| < 3%); "
                 "panels marked (≈noise) are below the reliable floor. The clean mesh-spread demonstration is at d10.",
                 ha="center", va="top", fontsize=9.2, color="#b03a2e", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fdecea", ec="#b03a2e"))

    out = os.path.join(g2.OUT_ROOT, f"Fig2_centerpress_{dia}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}  (per-material normalized)")
    for n in ORDER:
        p, a, s, t = data[n]["max_m"]
        print(f"   {dia} {n:8s} max: |peak|={p:.2f} active={a} sigma={s:.2f} total|ΔS|={t:.1f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--diameter", choices=["d5", "d10", "all"], default="all")
    a = ap.parse_args()
    for dia in (["d5", "d10"] if a.diameter == "all" else [a.diameter]):
        main(dia)
