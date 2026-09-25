"""Fig.2B - 소재 ablation 2D ΔS heatmap (xy_1mm 격자 압입).

논문 §4.3 주장 검증용: mesh 압력전달층이 인접 taxel 수용장 중첩을 키운다.
3개 소재(eco20 / eco50 / ecomesh, 동일 d10 인덴터)에 대해
인덴터 격자 위치 (x,y) 대비 baseline-정규화 ΔS 를 2D 맵으로 그린다.

생성물:
  - Fig2B_aggregate.png   : Σ_i |ΔS_i| (총 활성) vs 인덴터 (x,y)  [메인]
  - Fig2B_receptive.png   : 단일 중심 taxel ΔS vs 인덴터 (x,y)   [수용장 직접 근거]
  - 콘솔: 소재별 활성 격자 수 / 확산 σ / peak

데이터 포맷:
  due_data.csv        : elapsed_ns,time_s,burst_index,frame_index,Skin1..Skin16 (raw ADC)
  ethermotion_data.csv: ...,X,Y,Z,...  (X/Y/Z = encoder pulse, ×1e-4 = mm)
  loadcell_data.csv   : elapsed_ns,time_s,kg
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fig_data 루트를 파일 위치 기준으로 해석(환경 무관: 구 Windows 절대경로 하드코딩 제거)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BASE = os.path.join(REPO, "fig2_material_ablation/xy_1mm")
OUT_ROOT = os.path.join(REPO, "fig2_material_ablation/Analysis_Results")  # 출력 일원화(2026-06-25): 데이터 폴더 곁

# 인덴터 지름별 소재 라벨 -> test 폴더 (폴더명 ec020 오타 흡수). 비교 공정성 위해
# 한 지름 안에서는 모든 소재가 동일 인덴터 1세트씩.
DATASETS = {
    "d10": {
        "eco20":   os.path.join(BASE, "ec020/d10/20260619_test2"),
        "eco50":   os.path.join(BASE, "eco50/d10/20260620_test2"),
        "ecomesh": os.path.join(BASE, "ecomesh/d10/20260622_test4"),
    },
    "d5": {
        "eco20":   os.path.join(BASE, "ec020/d5/20260619_test5"),
        "eco50":   os.path.join(BASE, "eco50/d5/20260620_test3"),  # test1 은 Skin16 stuck → test3(Skin1 만 dead)
        "ecomesh": os.path.join(BASE, "ecomesh/d5/20260622_test1"),
    },
}
SKIN_COLS = [f"Skin{i}" for i in range(1, 17)]

PULSE_TO_MM = 1e-4          # encoder pulse -> mm
GRID_STEP_MM = 1.0          # 1mm 격자
GRID_LIM = 10.0             # ±10mm 스캔
Z_PRECONTACT = 13.0         # mm, 이 stage-z 미만 = 접촉 직전(travel 높이~12, 접촉 onset~13.3)
EDGES = np.arange(-GRID_LIM - GRID_STEP_MM / 2, GRID_LIM + GRID_STEP_MM, GRID_STEP_MM)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
NBIN = len(CENTERS)

# 4x4 센서 물리 좌표 (간격 6.5mm, ±9.75mm) — visualize.py 관례와 동일
_xs = [-9.75, -3.25, 3.25, 9.75]
SENSOR_XY = {}
for r in range(4):
    for c in range(4):
        # 데이터 검증 결과 Skin10(r=2,c=1) blob 이 (x=-3.25, y=+3.25)에 형성됨 -> y=_xs[r]
        SENSOR_XY[f"Skin{r*4 + c + 1}"] = (_xs[c], _xs[r])
CENTER_TAXELS = ["Skin6", "Skin7", "Skin10", "Skin11"]


def load_material(test_dir):
    """due를 burst 단위로 평균 -> ethermotion/loadcell 시간 동기화 -> ΔS 계산."""
    due = pd.read_csv(os.path.join(test_dir, "due_data.csv"))
    ether = pd.read_csv(os.path.join(test_dir, "ethermotion_data.csv"))
    load = pd.read_csv(os.path.join(test_dir, "loadcell_data.csv"))

    # burst 평균 (10 frame -> 1 sample), burst 대표 시간 = 평균 time_s
    g = due.groupby("burst_index")
    burst = g[SKIN_COLS].mean()
    burst["time_s"] = g["time_s"].mean()
    burst = burst.reset_index().sort_values("time_s").reset_index(drop=True)

    # ethermotion 보간 (시간축은 단조 증가 가정)
    ether = ether.sort_values("time_s")
    for col_src, col_dst in [("X", "x_mm"), ("Y", "y_mm"), ("Z", "z_mm")]:
        burst[col_dst] = np.interp(burst["time_s"], ether["time_s"], ether[col_src]) * PULSE_TO_MM

    # loadcell 보간
    load = load.sort_values("time_s")
    burst["kg"] = np.interp(burst["time_s"], load["time_s"], load["kg"])

    # ── per-press LOCAL baseline (압입 직전 기준) ──
    # 긴 xy 스캔(수천 초·수백 압입)에서 '스캔 시작 30 burst' 전역 baseline 은 센서 drift
    # (점탄성 회복·온도)로 후반/원거리에서 어긋나, 누르지 않은 taxel 이 ~0.5~1% 떠
    # 확산지표(σ_prop·active·entropy)를 부풀린다(특히 약신호 d5 eco20). 그래서 각 격자
    # 셀(=1회 압입 사이클)의 **접촉 직전**(travel 높이, z<Z_PRECONTACT 무접촉) per-taxel
    # 평균을 그 압입의 baseline 으로 써 drift 를 국소적으로 상쇄한다.
    ix = np.digitize(burst["x_mm"].to_numpy(), EDGES) - 1
    iy = np.digitize(burst["y_mm"].to_numpy(), EDGES) - 1
    inb = (ix >= 0) & (ix < NBIN) & (iy >= 0) & (iy < NBIN)
    cell = np.where(inb, iy * NBIN + ix, -1)
    raw = burst[SKIN_COLS].to_numpy()
    z = burst["z_mm"].to_numpy()
    gbase = burst.iloc[:30][SKIN_COLS].mean().to_numpy()      # fallback(전역)
    kg_base = burst.iloc[:30]["kg"].median()
    base_mat = np.tile(gbase, (len(burst), 1)).astype(float)  # 기본=전역, 셀별로 덮어씀
    for cid in np.unique(cell[cell >= 0]):
        m = cell == cid
        pre = m & (z < Z_PRECONTACT)
        if pre.sum() >= 5:                                   # 접촉 직전 표본 충분할 때만
            base_mat[m] = raw[pre].mean(axis=0)

    # ΔS = -(raw - base)/base * 100  (압력 증가를 양수로)
    ds = pd.DataFrame(-(raw - base_mat) / base_mat * 100.0,
                      columns=[f"dS_{c}" for c in SKIN_COLS], index=burst.index)
    out = pd.concat([burst[["time_s", "x_mm", "y_mm", "z_mm", "kg"]], ds], axis=1)
    return out, kg_base


DS_CONTACT_THR = 5.0  # %, 어떤 taxel이라도 이만큼 반응하면 접촉으로 간주


def contact_mask(df, kg_base):
    """접촉 구간: 센서 응답 기반(소재·인덴터 무관 robust).

    loadcell 절대 임계값은 d5/연질에서 힘이 약해 0 검출되는 문제가 있어,
    어떤 taxel이라도 ΔS 가 노이즈플로어(5%)를 넘는 샘플을 접촉으로 본다.
    """
    ds = df[[f"dS_{c}" for c in SKIN_COLS]]
    return ds.max(axis=1) > DS_CONTACT_THR


def build_grid(df, value):
    """(x,y) 1mm 격자에 value 의 셀별 최대값을 집계 (peak press 응답)."""
    ix = np.digitize(df["x_mm"], EDGES) - 1
    iy = np.digitize(df["y_mm"], EDGES) - 1
    grid = np.full((NBIN, NBIN), np.nan)
    ok = (ix >= 0) & (ix < NBIN) & (iy >= 0) & (iy < NBIN)
    tmp = pd.DataFrame({"ix": ix[ok], "iy": iy[ok], "v": np.asarray(value)[ok]})
    agg = tmp.groupby(["iy", "ix"])["v"].max()
    for (iy_, ix_), v in agg.items():
        grid[iy_, ix_] = v
    return grid


SIGMA_FLOOR_ABS = 2.0   # %, σ 계산 시 절대 노이즈 임계 (저SNR 맵의 σ 부풀림 방지)


def receptive_metrics(grid):
    """단일 taxel 수용장 맵의 정량 지표.

    σ 기준점은 레이아웃 가정이 아닌 데이터 기반 응답 가중 centroid 사용.
    중요: σ 와 centroid 는 **임계(≥ max(0.2·peak, 2%)) 를 넘은 셀만** 가중에 사용한다.
      임계 없이 전체 양수 셀을 쓰면 저SNR 맵(예: d5 eco20, peak~20%)에서 멀리 흩어진
      노이즈가 RMS 거리를 부풀려 σ 가 거꾸로 커지는 인공결과가 생긴다.
    halfmax_diam: 응답이 **peak 의 50% 이상**인 영역의 등가 지름(mm). peak 가 포화(=100%)
      해도 구분되는, 포화에 강건한 수용장 폭 지표(반치폭 FWHM 등가).
    반환: peak, centroid(cx,cy), sigma(임계 후 가중 RMS 거리 mm),
          active(peak 20% 이상 셀 수), halfmax_diam(반치폭 등가 지름 mm),
          radial(centroid 로부터 거리별 평균 ΔS).
    """
    g = np.nan_to_num(grid, nan=0.0)
    g = np.clip(g, 0, None)
    peak = float(g.max())
    yy, xx = np.meshgrid(CENTERS, CENTERS, indexing="ij")
    if peak > 0:
        thr = max(0.2 * peak, SIGMA_FLOOR_ABS)
        w = np.where(g >= thr, g, 0.0)          # 임계 후 가중 (저SNR 부풀림 차단)
        if w.sum() > 0:
            cx = float((w * xx).sum() / w.sum()); cy = float((w * yy).sum() / w.sum())
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            sigma = float(np.sqrt((w * dist ** 2).sum() / w.sum()))
        else:
            cx = cy = 0.0; dist = np.sqrt(xx ** 2 + yy ** 2); sigma = 0.0
        # 반치폭(50% peak) 면적 → 등가 지름
        hm_area = float((g >= 0.5 * peak).sum()) * (GRID_STEP_MM ** 2)
        halfmax_diam = float(2.0 * np.sqrt(hm_area / np.pi))
    else:
        cx = cy = 0.0; dist = np.sqrt(xx ** 2 + yy ** 2); sigma = 0.0; halfmax_diam = 0.0
    active = int((g > peak * 0.2).sum()) if peak > 0 else 0
    # 반경 프로파일 (1mm bin)
    rbins = np.arange(0, GRID_LIM + 1.0, 1.0)
    rc = (rbins[:-1] + rbins[1:]) / 2
    ri = np.digitize(dist.ravel(), rbins) - 1
    vals = g.ravel()
    radial = np.array([vals[ri == k].mean() if np.any(ri == k) else np.nan
                       for k in range(len(rc))])
    return {"peak": peak, "centroid": (cx, cy), "sigma": sigma, "active": active,
            "halfmax_diam": halfmax_diam, "radial": (rc, radial)}


def main(diameter):
    materials = DATASETS[diameter]
    out_dir = os.path.join(OUT_ROOT, diameter)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n########## diameter = {diameter}  ->  {out_dir} ##########")
    data = {}
    for name, d in materials.items():
        print(f"[load] {name}: {d}")
        df, kg_base = load_material(d)
        m = contact_mask(df, kg_base)
        data[name] = (df, m, kg_base)
        print(f"       bursts={len(df)} contact={int(m.sum())} kg_base={kg_base:.3f}")

    # 대표 중심 taxel: 전 소재 합산 |ΔS| 최대인 중심 taxel
    score = {t: 0.0 for t in CENTER_TAXELS}
    for name, (df, m, _) in data.items():
        for t in CENTER_TAXELS:
            score[t] += df.loc[m, f"dS_{t}"].abs().mean()
    rep = max(score, key=score.get)
    print(f"[rep taxel] {rep}  (center scores: "
          + ", ".join(f"{t}:{score[t]:.1f}" for t in CENTER_TAXELS) + ")")

    # 대표 taxel 수용장 격자 + 메트릭
    # 격자는 전체 샘플로 구성(셀별 max) -> 인덴터가 누른 모든 (x,y)에서 rep taxel 응답.
    # 멀리 떨어진 셀은 max≈0 으로 채워져 깔끔한 수용장 맵이 된다(접촉마스크로 비우지 않음).
    rep_xy = SENSOR_XY[rep]
    rep_grids, metrics = {}, {}
    for name, (df, m, _) in data.items():
        rep_grids[name] = build_grid(df, df[f"dS_{rep}"])
        metrics[name] = receptive_metrics(rep_grids[name])

    ORDER = ["eco20", "eco50", "ecomesh"]
    COLORS = {"eco20": "#1f77b4", "eco50": "#ff7f0e", "ecomesh": "#2ca02c"}

    # ---- Fig.2B-1: 수용장 2D 맵 (3소재 공통 스케일) ----
    vmax = max(metrics[n]["peak"] for n in ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for ax, name in zip(axes, ORDER):
        im = ax.imshow(rep_grids[name], origin="lower", cmap="viridis",
                       extent=[CENTERS[0], CENTERS[-1], CENTERS[0], CENTERS[-1]],
                       vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        ax.set_title(f"{name}\nσ={metrics[name]['sigma']:.2f}mm  "
                     f"peak={metrics[name]['peak']:.0f}%  active={metrics[name]['active']}",
                     fontsize=11)
        ax.set_xlabel("indenter x (mm)"); ax.set_ylabel("indenter y (mm)")
        for t, (sx, sy) in SENSOR_XY.items():
            ax.plot(sx, sy, "+", color="white", ms=6, mew=1.0, alpha=0.5)
        ax.plot(*rep_xy, "o", mfc="none", mec="red", ms=12, mew=1.5)  # 대표 taxel
    fig.colorbar(im, ax=axes, shrink=0.8, label=f"ΔS of {rep} (%)")
    fig.suptitle(f"Fig.2B  Receptive field of central taxel {rep} "
                 f"vs indenter position ({diameter})\n"
                 f"red ○ = {rep} location, white + = taxels",
                 fontsize=13, fontweight="bold")
    out1 = os.path.join(out_dir, "Fig2B_receptive.png")
    fig.savefig(out1, dpi=150); plt.close(fig)
    print(f"[saved] {out1}")

    # ---- Fig.2B-2: 반경 감쇠 곡선 (수용장 폭 직접 비교) ----
    fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)
    for name in ORDER:
        rc, rv = metrics[name]["radial"]
        ax.plot(rc, rv, "-o", color=COLORS[name], label=f"{name} (σ={metrics[name]['sigma']:.2f}mm)")
    ax.set_xlabel(f"distance from {rep} receptive-field centroid (mm)")
    ax.set_ylabel(f"mean ΔS of {rep} (%)")
    ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"Fig.2B  Radial attenuation of taxel response ({diameter})\n"
                 "(slower decay = wider receptive field = stronger overlap)",
                 fontsize=12, fontweight="bold")
    out2 = os.path.join(out_dir, "Fig2B_radial.png")
    fig.savefig(out2, dpi=150); plt.close(fig)
    print(f"[saved] {out2}")

    # ---- per-taxel 수용장 평균 (Skin10 단일 → 16 taxel 중 '중앙 4개' 평균) ----
    # Skin10 하나만 쓰면 그 taxel 의 특이성(약신호 flat-top 등)이 섞여 d5 eco20 이 거꾸로
    # 넓어 보이는 착시가 생겼다. 각 taxel 의 수용장을 따로 구해 평균하면 대표성이 높다.
    # half-max 는 표준 정의(각 taxel **자기 peak 의 50%** 영역 지름)를 쓴다.
    # 가장자리 taxel(±9.75mm)은 수용장이 스캔 경계(±10mm) 밖으로 잘려 폭이 과소평가되므로,
    # 수용장이 온전히 포착되는 **중앙 4개(±3.25mm: Skin6,7,10,11)** 만 평균한다.
    # 비교 기준: taxel 간격 6.5mm — 평균 수용장 지름이 6.5mm 에 근접/초과하면 이웃 수용장이
    # 겹쳐 super-resolution 보간이 가능(overlap), 그보다 작으면 undersampling 간극.
    print(f"\n===== Fig.2B metrics ({diameter}, per-taxel mean over central 4) =====")
    rows = {}
    for name, (df, _, _) in data.items():
        hm, sg, ac, pk = [], [], [], []
        for t in CENTER_TAXELS:
            mm = receptive_metrics(build_grid(df, df[f"dS_{t}"]))
            hm.append(mm["halfmax_diam"]); sg.append(mm["sigma"])
            ac.append(mm["active"]); pk.append(mm["peak"])
        rows[name] = {"peak_%": float(np.mean(pk)),
                      "halfmax_diam_mm": float(np.mean(hm)),
                      "sigma_mm": float(np.mean(sg)),
                      "active_cells": float(np.mean(ac))}
    summ = pd.DataFrame(rows).T[["peak_%", "halfmax_diam_mm", "sigma_mm", "active_cells"]].loc[ORDER]
    print(summ.to_string())
    summ.to_csv(os.path.join(out_dir, "Fig2B_metrics.csv"))
    print(f"[saved] {os.path.join(out_dir, 'Fig2B_metrics.csv')}  (vs taxel pitch 6.5mm)")
    print("\n해석 가이드: 평균 수용장 폭 -> eco20(연질) 가장 좁음 / eco50·mesh 넓어 6.5mm(pitch)에 근접 "
          "=> 이웃 수용장 overlap(SR 유리). eco20 은 pitch 미달(undersampling).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fig.2B 2D 수용장 heatmap (소재 ablation)")
    ap.add_argument("--diameter", choices=["d5", "d10", "all"], default="all",
                    help="인덴터 지름 선택 (기본: all = d5,d10 모두)")
    a = ap.parse_args()
    targets = ["d10", "d5"] if a.diameter == "all" else [a.diameter]
    for dia in targets:
        main(dia)
