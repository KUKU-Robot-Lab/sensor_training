"""Fig.2C - 소재 ablation 정량 메트릭 비교 (xy_1mm).

논문 §6 패널 C: "총 |ΔS|·확산폭·활성 taxel 수 비교".
센서 16-taxel 배열 응답을 접촉 샘플마다 계산해 소재별 평균으로 막대그래프 비교.

메트릭 (각 접촉 샘플 = 인덴터 1회 압입 시점의 16-taxel ΔS 벡터에서 계산 후 평균):
  - Total |ΔS| (%)        : Σ_i |ΔS_i|  — 총 응답 세기(민감도)
  - Active taxels (N)     : ΔS_i > 5% 인 taxel 수 — 수용장 중첩 정도
  - Propagation σ (mm)    : 응답 가중, taxel 물리 위치 기준 RMS 확산폭 — 신호가 인접 taxel로 퍼진 정도
  - Response entropy H_norm: -Σ p_i log p_i / log16 — 응답 분산도(0=한 taxel 집중, 1=균등)

해석: mesh 가 수용장 중첩을 키우면 Active·σ_prop 이 eco20 보다 크고, eco50 대비 Total|ΔS|(민감도) 유지.

기존 로딩/접촉 로직은 generate_2d_heatmap.py 에서 재사용한다.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate_2d_heatmap as g2  # load_material, contact_mask, SENSOR_XY, SKIN_COLS, DATASETS, OUT_ROOT

ORDER = ["eco20", "eco50", "ecomesh"]
COLORS = {"eco20": "#1f77b4", "eco50": "#ff7f0e", "ecomesh": "#2ca02c"}
FLOOR_ABS = 0.5  # %, |ΔS| 절대 노이즈 임계 (실측 noise std ~0.01% 의 50배)
SENSOR_POS = np.array([g2.SENSOR_XY[c] for c in g2.SKIN_COLS])  # (16,2) mm


def peak_per_cell(sub):
    """접촉 샘플을 인덴터 1mm 격자 셀로 묶어 셀별 peak 압입(최대 Σ|ΔS|) 1행만 대표로 추출.

    전 로딩 시계열을 평균하면 약한 초기 구간이 섞여 활성/확산이 희석되므로,
    각 격자점의 '완전 압입' 순간만 모아 배열 응답을 특성화한다.
    """
    ds_all = sub[[f"dS_{c}" for c in g2.SKIN_COLS]].to_numpy()
    tot = np.abs(ds_all).sum(axis=1)
    ix = np.digitize(sub["x_mm"].to_numpy(), g2.EDGES) - 1
    iy = np.digitize(sub["y_mm"].to_numpy(), g2.EDGES) - 1
    ok = (ix >= 0) & (ix < g2.NBIN) & (iy >= 0) & (iy < g2.NBIN)
    tmp = pd.DataFrame({"cell": (iy * g2.NBIN + ix)[ok],
                        "tot": tot[ok], "idx": np.nonzero(ok)[0]})
    rep = tmp.loc[tmp.groupby("cell")["tot"].idxmax(), "idx"].to_numpy()
    return ds_all[rep]  # (Ncell, 16) 격자점별 peak 압입


def array_metrics(sub):
    """셀별 peak 압입 -> 4개 메트릭 (평균, 표준편차).

    중요(수정): 활성/확산 임계를 **상대(15%·peak)** 가 아니라 **절대 |ΔS|>0.5%** 로 둔다.
      d5 처럼 한 taxel 이 압도적인 경우 상대임계는 그 1개만 남겨(active=1) σ·entropy 가
      0 으로 퇴화했다. 또 부호 있는 ds>thr 는 인접 taxel 의 **음수** 응답(인장측)을 버렸다.
      |ΔS| 절대값 + 절대 floor 로 바꿔 실제로 반응한 약한 이웃까지 포함한다.
    """
    ds = peak_per_cell(sub)                          # (M,16)
    a = np.abs(ds)                                   # |ΔS| (인장/압축 부호 무관)
    peak = np.clip(a.max(axis=1), 1e-6, None)        # 샘플별 최대 taxel
    keep = peak > 1.0                                # 유효 압입(>1%)만
    a, peak = a[keep], peak[keep]

    keepm = a > FLOOR_ABS                            # 절대 노이즈플로어 초과 taxel
    w = np.where(keepm, a, 0.0)                      # 노이즈 제거 가중
    wsum = np.clip(w.sum(axis=1), 1e-6, None)

    total = a.sum(axis=1)
    active = keepm.sum(axis=1).astype(float)
    cen = (w @ SENSOR_POS) / wsum[:, None]
    d2 = ((SENSOR_POS[None, :, :] - cen[:, None, :]) ** 2).sum(axis=2)
    sigma = np.sqrt((w * d2).sum(axis=1) / wsum)
    p = w / wsum[:, None]
    H = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1) / np.log(len(g2.SKIN_COLS))

    out = {}
    for k, v in [("Total |ΔS| (%)", total), ("Active taxels (N)", active),
                 ("Propagation σ (mm)", sigma), ("Entropy H_norm", H)]:
        out[k] = (float(v.mean()), float(v.std()))
    return out


def run(diameter):
    materials = g2.DATASETS[diameter]
    out_dir = os.path.join(g2.OUT_ROOT, diameter)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n########## Panel C  diameter={diameter} ##########")

    results = {}
    for name in ORDER:
        df, kg_base = g2.load_material(materials[name])
        m = g2.contact_mask(df, kg_base)
        results[name] = array_metrics(df[m])
        print(f"[{name}] contact={int(m.sum())}  "
              + "  ".join(f"{k}={v[0]:.2f}" for k, v in results[name].items()))

    metric_names = list(next(iter(results.values())).keys())

    # ---- 막대그래프 (메트릭별 subplot, 소재 3바) ----
    fig, axes = plt.subplots(1, len(metric_names), figsize=(16, 4.5), constrained_layout=True)
    for ax, mk in zip(axes, metric_names):
        means = np.array([results[n][mk][0] for n in ORDER])
        stds = np.array([results[n][mk][1] for n in ORDER])
        lower = np.minimum(stds, means)  # 비음수 메트릭: 오차막대를 0에서 클리핑
        bars = ax.bar(ORDER, means, yerr=[lower, stds], capsize=4,
                      color=[COLORS[n] for n in ORDER], alpha=0.85)
        ax.set_title(mk, fontsize=12)
        ax.set_ylabel(mk)
        ax.set_ylim(bottom=0)
        for b, mv in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{mv:.2f}", ha="center", va="bottom", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Fig.2C  Material-wise quantitative metrics ({diameter})  "
                 f"[contact-sample mean ± std]", fontsize=14, fontweight="bold")
    out_png = os.path.join(out_dir, "Fig2C_metrics.png")
    fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"[saved] {out_png}")

    # ---- CSV ----
    rows = {n: {mk: results[n][mk][0] for mk in metric_names} for n in ORDER}
    summ = pd.DataFrame(rows).T[metric_names]
    out_csv = os.path.join(out_dir, "Fig2C_metrics.csv")
    summ.to_csv(out_csv)
    print(summ.to_string())
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fig.2C 소재별 정량 메트릭")
    ap.add_argument("--diameter", choices=["d5", "d10", "all"], default="all")
    a = ap.parse_args()
    for dia in (["d10", "d5"] if a.diameter == "all" else [a.diameter]):
        run(dia)
