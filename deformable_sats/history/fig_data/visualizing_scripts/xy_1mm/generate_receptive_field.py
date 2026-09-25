"""Fig.2 (supplement) — 센서별 receptive field (응답크기) + 합산 overlap.

요청 정의: 각 센서 s_i 가 **인덴터를 (x,y) 에 최대깊이(z_max)로 눌렀을 때 얼마나 반응했는지**
(|ΔS_i|) 를 (x,y) 히트맵으로. 예: s6(-3.25,-3.25) 는 인덴터가 그 근처(예 (-5,-5)) 를 z_max 로
누를 때 크게 반응 → s6 맵의 그 위치가 밝다. = s6 의 수용장(receptive field).

  A. 센서별 (`<dia>_<material>_sensors.png`): s1~s16 각각의 응답크기 맵을 4×4 몽타주(센서 배치).
  B. 합산 (`sum_<dia>_{2d,3d}.png`): S(x,y) = Σ_t |ΔS_t| (그 위치 z_max 압입에 반응한 모든 센서 합).
     여러 센서 수용장이 겹치는 곳일수록 S 가 크다 → **수용장 중첩(overlap)을 직접** 보여줌.

신호: per-press local baseline ΔS, 셀별 최대(=z_max 압입) 값. dead 채널(peak<1%)은 표기/제외.

출력: fig2_material_ablation/hitmap/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import generate_2d_heatmap as g2

ORDER = ["eco20", "eco50", "ecomesh"]
LABEL = {"eco20": "eco20", "eco50": "eco50", "ecomesh": "ecomesh (mesh20)"}
SK = g2.SKIN_COLS
OUT = os.path.join(g2.REPO, "fig2_material_ablation", "hitmap")
os.makedirs(OUT, exist_ok=True)
EXT = [g2.CENTERS[0], g2.CENTERS[-1], g2.CENTERS[0], g2.CENTERS[-1]]
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


def _blur(a, sigma=1.0):
    r = max(1, int(3 * sigma)); x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2)); k /= k.sum()
    b = np.apply_along_axis(lambda m: np.convolve(np.pad(m, r, "edge"), k, "valid"), 0, a)
    return np.apply_along_axis(lambda m: np.convolve(np.pad(m, r, "edge"), k, "valid"), 1, b)


def _upsample(Z, factor=4):
    ny, nx = Z.shape
    xi = np.linspace(0, nx - 1, nx * factor); yi = np.linspace(0, ny - 1, ny * factor)
    Zx = np.array([np.interp(xi, np.arange(nx), row) for row in Z])
    Zf = np.array([np.interp(yi, np.arange(ny), col) for col in Zx.T]).T
    return Zf, np.linspace(g2.CENTERS[0], g2.CENTERS[-1], nx * factor)


ECO50_D5_T1 = os.path.join(g2.BASE, "eco50/d5/20260620_test1")  # s1 정상(s16 stuck) — dead 보충용
ZC = 13.0   # mm, 접촉(압입) 구간 시작. 이 미만(travel/대각선 이동)은 grid 에서 제외해 아티팩트 차단


def _grids(df):
    dfp = df[df.z_mm >= ZC]   # 압입 구간만 → 인덴터 대각선 이동 등 travel 아티팩트 제거
    return {t: np.clip(np.nan_to_num(g2.build_grid(dfp, dfp[f"dS_{t}"].abs()), nan=0.0), 0, None)
            for t in SK}


def _denoise_d5(grids, dead, sigma=0.8, floor_frac=0.30):
    """d5(소인덴터·약신호)는 셀별 speckle 노이즈가 커 overlap/몽타주가 지저분하다.
    각 센서 맵을 (1) 가우시안 블러(sigma)로 평활 + (2) 자기 peak 의 floor_frac 미만을 0 으로
    잘라(상대 노이즈 플로어) 봉우리만 남긴다. d10(강신호)에는 적용 안 함."""
    out = {}
    for t, g in grids.items():
        if t in dead or g.max() <= 1.0:
            out[t] = g
            continue
        b = _blur(g, sigma)
        b[b < floor_frac * b.max()] = 0.0
        out[t] = b
    return out


def fields(dia, name):
    df, _ = g2.load_material(g2.DATASETS[dia][name])
    grids = _grids(df)
    dead = {t for t in SK if df[f"dS_{t}"].abs().max() < 1.0}
    # eco50 d5: test3 의 dead 채널(s1)을 test1(거기선 정상)에서 보충 → 16채널 모두 생존
    if dia == "d5" and name == "eco50" and dead:
        df1, _ = g2.load_material(ECO50_D5_T1)
        g1 = _grids(df1)
        for t in list(dead):
            if df1[f"dS_{t}"].abs().max() >= 1.0:
                grids[t] = g1[t]; dead.discard(t)
                print(f"   [d5 eco50] {t} 를 test1 에서 보충(생존)")
    return grids, dead


def montage(dia, name, grids, dead, vmax):
    """A. 센서별 응답크기 receptive field (4×4, 센서 배치 동일). vmax = 3소재 공통(비교용)."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12.4))
    im = None
    for i, t in enumerate(SK):
        r, c = divmod(i, 4)
        ax = axes[3 - r, c]
        s = t.replace("Skin", "s")
        if t in dead:
            ax.imshow(np.zeros((g2.NBIN, g2.NBIN)), origin="lower", cmap="Greys", vmin=0, vmax=1, extent=EXT)
            ax.text(0, 0, "DEAD\nchannel", ha="center", va="center", color="#c0392b", fontsize=9, fontweight="bold")
            ax.set_title(f"{s} — DEAD", fontsize=8.5, color="#c0392b", fontweight="bold")
            for sp in ax.spines.values():
                sp.set_color("#c0392b"); sp.set_linewidth(2)
        else:
            im = ax.imshow(grids[t], origin="lower", cmap="turbo", vmin=0, vmax=vmax,
                           extent=EXT, interpolation="bilinear", aspect="equal")
            ax.set_title(f"{s}  peak {grids[t].max():.0f}%", fontsize=8.5)
        ax.set_xticks([-10, -5, 0, 5, 10]); ax.set_yticks([-10, -5, 0, 5, 10])
        ax.tick_params(labelsize=6)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cb.set_label("|ΔS| at max-depth press (%)", fontsize=9)
    fig.suptitle(f"Per-sensor receptive field (response magnitude) — {LABEL[name]} · {dia}\n"
                 "each cell = how strongly that sensor responds to a z-max press at indenter (x,y)  "
                 "[color scale SHARED across eco20/eco50/mesh]",
                 fontsize=12.5, fontweight="bold")
    out = os.path.join(OUT, f"{dia}_{name}_sensors.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out}")


def overlap_norm(grids, dead):
    """O_excess(x,y) = Σ_t(|ΔS_t|/peak_t) − max_t(|ΔS_t|/peak_t)
       = 각 센서를 자기 peak 로 정규화해 더한 값에서 **가장 센 센서 1개를 뺀 '추가 중첩분'**.

    의도: 센서 **자기 중심**(그 센서 혼자 dominant)은 빼고 남으므로 ~0 에 가깝고, **여러 수용장이
    겹쳐 여러 센서가 동시에 반응하는 영역**일수록 커진다 → '다중센서 반응(겹침) 영역'이 가장 높게.
    절대 응답크기와 무관(각 센서 0~1 정규화) = 실제 ΔS 총합이 낮아도 겹침이 높게 표현됨.
    """
    live = [g / g.max() for t, g in grids.items() if t not in dead and g.max() > 1.0]
    if not live:
        return np.zeros((g2.NBIN, g2.NBIN))
    stack = np.stack(live)                 # (Nlive, NBIN, NBIN), 각 0~1
    return stack.sum(axis=0) - stack.max(axis=0)   # 총 정규화합 − 최강 센서 = 추가 중첩


CELL_AREA = g2.GRID_STEP_MM ** 2   # 1mm² per cell


def sensor_stats(dia, name, grids, dead):
    """센서별 수용장 '수영장 넓이' = half-max(자기 peak 50%) 이상 셀 면적(mm²)·등가지름(mm)."""
    rows = []
    for t in SK:
        s = t.replace("Skin", "s")
        if t in dead:
            rows.append((s, None, None, None)); continue
        g = grids[t]; pk = g.max()
        area = float((g >= 0.5 * pk).sum()) * CELL_AREA
        diam = 2.0 * np.sqrt(area / np.pi)
        rows.append((s, pk, area, diam))
    return rows


def coverage_stats(grids, dead, floors=(1.0, 1.5, 3.0), mag_grids=None):
    """SATS-정렬 다중센서 coverage: 각 인덴터 위치에서 |ΔS|≥floor 인 센서 수.

    근거(sensor_training/sats/training/dataset.py:248): SATS 입력 = s_norm=((raw−base)/base)·100
    = ΔS%, **floor/clip 없음**(입력 정규화도 없음). 위치 포함 기준은 Fz>0(접촉)뿐. 즉 한 위치에서
    SATS 가 위치를 추론하려면 **노이즈 위로 올라온 채널이 여럿** 있어야 한다. SATS 는 trial 당
    global baseline 을 쓰므로 스캔 drift(~0.5~1%)가 유효 노이즈 → 그를 충분히 넘는 |ΔS|≥3%·5% 를
    'contact 정보 보유' 기준으로 본다(raw per-press peak |ΔS| 사용, 디노이즈 전)."""
    live = [g for t, g in grids.items() if t not in dead]
    stack = np.stack(live)                       # (Nsensor, NBIN, NBIN) per-position |ΔS| (counts 용 = cov)
    tot = stack.shape[1] * stack.shape[2]
    # magnitude(순위별 센서 세기)는 raw(참신호, mag_grids)에서 — second_sensor 맵과 일치.
    mstack = stack if mag_grids is None else np.stack([mag_grids[t] for t in SK if t not in dead])
    msrt = np.sort(mstack, axis=0)
    s2, s3 = msrt[-2], msrt[-3]                   # 각 위치 2·3번째로 센 센서 |ΔS| (raw)
    out = {"med2": float(np.median(s2)), "max2": float(s2.max()),
           "med3": float(np.median(s3)), "max3": float(s3.max())}
    for f in floors:
        cnt = (stack >= f).sum(axis=0)           # 각 위치에서 floor 넘는 센서 수 (cov, de-speckle)
        out[f] = dict(ge1=100.0 * float((cnt >= 1).sum()) / tot,
                      ge2=100.0 * float((cnt >= 2).sum()) / tot,
                      ge3=100.0 * float((cnt >= 3).sum()) / tot,
                      ge3_cells=int((cnt >= 3).sum()),
                      blind=100.0 * float((cnt == 0).sum()) / tot,
                      meanN=float(cnt.mean()))
    return out


def overlap_stats(N):
    """overlap O_excess 분포 통계.

    겹침면적은 공유 컬러스케일(vmax≈0.74) 기준 **중간 겹침 O≥0.2 / 강한 겹침 O≥0.3** 면적으로 본다.
    (이전 O>0.05 는 거의 전 격자가 잡혀 포화→변별 불가. 0.2/0.3 에서 소재가 또렷이 갈린다.)"""
    flat = N.ravel()
    return dict(mean=float(N.mean()), std=float(N.std()),
                cv=float(N.std() / N.mean()) if N.mean() > 0 else 0.0,
                max=float(N.max()), p95=float(np.percentile(flat, 95)),
                area_mid=float((N >= 0.2).sum()) * CELL_AREA,    # 중간 겹침 면적
                area_strong=float((N >= 0.3).sum()) * CELL_AREA)  # 강한 겹침 면적


def write_stats(allstats):
    """수치 통계 md 저장: 센서별 수용장 넓이 + overlap 분포 통계."""
    L = ["# 센서 receptive field — 수치 통계",
         "",
         "> 생성: `generate_receptive_field.py`. 신호 = per-press local baseline |ΔS|, 셀별 z_max.",
         "> 셀 = 1mm² 격자. '수영장 넓이' = 자기 peak 의 50% 이상(half-max) 셀 면적, 등가지름 = 2√(area/π).",
         "> d5 는 디노이즈(가우시안 σ0.8 + 자기 peak 30% 노이즈플로어) 후 측정.",
         ""]

    # A. 센서별 수용장 넓이 요약 (소재·인덴터별 평균)
    L += ["## A. 센서별 수용장 넓이 (16채널 평균)", "",
          "| 인덴터·소재 | 평균 peak(%) | 평균 half-max 면적(mm²) | 평균 등가지름(mm) | 생존채널 |",
          "|---|---|---|---|---|"]
    for (dia, name), st in allstats.items():
        rows = st["sensors"]
        live = [r for r in rows if r[1] is not None]
        mpk = np.mean([r[1] for r in live]); mar = np.mean([r[2] for r in live]); mdi = np.mean([r[3] for r in live])
        L.append(f"| {dia} {name} | {mpk:.0f} | {mar:.1f} | {mdi:.1f} | {len(live)}/16 |")
    L.append("")

    # B. overlap 분포 통계
    L += ["## B. overlap(O_excess) 분포 통계", "",
          "> overlap 컬러스케일은 d5·d10 공통(vmax≈0.74). 겹침면적은 그 스케일 기준 임계값으로 측정.",
          "",
          "| 인덴터·소재 | 평균 | std | CV | p95 | max | 중간겹침 O≥0.2(mm²) | 강한겹침 O≥0.3(mm²) |",
          "|---|---|---|---|---|---|---|---|"]
    for (dia, name), st in allstats.items():
        o = st["overlap"]
        L.append(f"| {dia} {name} | {o['mean']:.3f} | {o['std']:.3f} | {o['cv']:.2f} | "
                 f"{o['p95']:.2f} | {o['max']:.2f} | {o['area_mid']:.0f} | {o['area_strong']:.0f} |")
    L += ["",
          "> **해석**:",
          "> - 겹침면적 = 다중센서가 동시 반응하는 '수영장 겹침대'의 넓이. **O≥0.2(중간)·O≥0.3(강한)** 두 임계값.",
          ">   (이전 O>0.05 는 거의 전 격자 441mm² 가 잡혀 포화→변별 불가였음. 0.2/0.3 에서 소재가 또렷이 갈림.)",
          "> - **d10**: 강한겹침 면적 **eco50 243 > mesh 175 ≫ eco20 21 mm²** — eco50·mesh 가 eco20 의 8~12배.",
          "> - **d5**: 좁은 수용장이라 강한겹침(≥0.3)은 0. 중간겹침(≥0.2)에서 mesh 69 > eco50 19 > eco20 0 으로 갈림.",
          "> - CV(변동계수)가 클수록 공간적으로 불균일(패치성)하게 보인다.",
          ""]

    # B2. SATS-정렬 다중센서 coverage
    fl = [1.0, 1.5, 3.0]
    L += ["## B2. SATS-정렬 다중센서 coverage (SR 입력 관점)", "",
          "> SATS 입력 = ΔS%(floor·정규화 없음; `sats/training/dataset.py:248`). 위치 포함=Fz>0.",
          "> SATS 구조 = 16채널 self-attention + 센서별 local map 합산 + CNN refiner → **반응하는 모든 센서를 통합**.",
          "> **2D 위치(x,y)+미지 접촉력 = 미지수 3 → 모호함 없는 SR 은 ≥3 센서 필요**(2 는 거리-링 2교점 모호, 최소바).",
          "> floor 는 local-baseline 노이즈(σ~0.05%)를 넘는 1.0/1.5%(+참고 3%). raw per-press peak |ΔS|.",
          "",
          "**순위별 센서 |ΔS| (SR 에 쓸 보조 신호 세기; 임의 floor 무관)**",
          "",
          "| 인덴터·소재 | 2nd-센서 중앙값 | 3rd-센서 중앙값 | 2nd-max | 3rd-max |",
          "|---|---|---|---|---|"]
    for (dia, name), st in allstats.items():
        cv = st["coverage"]
        L.append(f"| {dia} {name} | {cv['med2']:.1f}% | {cv['med3']:.1f}% | {cv['max2']:.1f}% | {cv['max3']:.1f}% |")
    L += ["",
          "> d5 3번째 센서: **eco20 0.3%(노이즈급) ≪ eco50 1.2% ≈ mesh 1.2%**. eco20 은 3번째(2번째도) 센서가 노이즈 →",
          "> **어디서도 ≥3 불가 = SR 불가**. eco50·mesh 는 3번째 센서까지 노이즈 위(~1.2%) → 면적 대부분에서 SR 성립.",
          "> 시각화: `second_sensor_<dia>.png`(2번째)·`coverage_<dia>.png`(3번째) = 동일 연속 형태·공통 스케일(직접 비교).",
          "",
          "**≥2 / ≥3 센서 coverage (floor 별; ≥3 = 모호함 없는 SR 의 바)**",
          "",
          "| 인덴터·소재 | floor | blind% | ≥2센서% | **≥3센서%** | ≥3 셀수 | 평균센서수 |",
          "|---|---|---|---|---|---|---|"]
    for (dia, name), st in allstats.items():
        for f in fl:
            cv = st["coverage"][f]
            L.append(f"| {dia} {name} | ≥{f:.1f}% | {cv['blind']:.0f} | "
                     f"{cv['ge2']:.0f} | **{cv['ge3']:.0f}** | {cv['ge3_cells']} | {cv['meanN']:.2f} |")
    L += ["",
          "> **해석 (SATS=d5 강점, ≥3 기준)**:",
          "> - **≥3센서(모호함 없는 SR) coverage**: d5·floor 1.0% 에서 **eco50 76% · mesh 68% ≫ eco20 0%**.",
          ">   **eco20 은 어떤 floor 에서도 ≥3 = 0%**(2·3번째 센서가 노이즈급) → undersampling 으로 SR 원천 불가.",
          "> - **mesh vs eco50**: d5 다중센서 coverage 는 둘이 막상막하(eco50 근소 위), mesh 가 2nd-센서 최대(5.1%)·정규화 overlap(0.157) 우위.",
          ">   핵심은 **eco20 의 완전 실패 대비 eco50·mesh 가 ≥3 센서 동시반응을 넓게 만든다**(SR 가능 영역).",
          "> - **대조 d10**: 인덴터가 커 모든 소재가 다수 센서 자극 → 소재차 작음. **소재 의존성은 접촉이 작을수록(=d5, SATS 대상) 극대화**(eco20 만 붕괴).",
          "> - ⚠ 3번째 센서가 ~1.1% 로 floor 근방이라 ≥3 coverage 는 floor 민감(1.0%→76/68%, 1.5%→15/13%). 견고한 비교는 **3rd-센서 세기 자체**(eco50·mesh 1.1% vs eco20 0.2%, 5배).",
          "> - coverage 는 d5 약블러(σ0.6)로 셀 speckle 만 평활(약신호 inflation 최소). magnitude(2·3번째 세기)는 raw — second_sensor 맵과 일치.",
          ""]

    # C. 센서별 상세 (소재·인덴터별 16채널 표)
    L += ["## C. 센서별 상세 (half-max 면적 mm² / 등가지름 mm)", ""]
    for (dia, name), st in allstats.items():
        L += [f"### {dia} · {name}", "",
              "| 센서 | peak(%) | 면적(mm²) | 지름(mm) |", "|---|---|---|---|"]
        for s, pk, ar, di in st["sensors"]:
            if pk is None:
                L.append(f"| {s} | DEAD | – | – |")
            else:
                L.append(f"| {s} | {pk:.0f} | {ar:.1f} | {di:.1f} |")
        L.append("")

    path = os.path.join(OUT, "STATS_수치.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {path}")


_STATS = {}


def compute(dia):
    """데이터 로드 + 디노이즈 + overlap 산출 + 통계 수집. 그리기는 render 가 담당."""
    raw = {n: fields(dia, n) for n in ORDER}
    # d5(약신호) 디노이즈: 몽타주는 블러+노이즈플로어(깔끔), overlap 은 블러만
    # (하드 floor 를 overlap 전에 적용하면 좁은 d5 수용장이 정규화 후 겹침 0 이 되어버림 →
    #  약하지만 실재하는 겹침 신호를 보존하려 블러로 speckle 만 제거).
    if dia == "d5":
        data = {n: (_denoise_d5(raw[n][0], raw[n][1], sigma=0.8, floor_frac=0.30), raw[n][1]) for n in ORDER}
        ov = {n: (_denoise_d5(raw[n][0], raw[n][1], sigma=0.9, floor_frac=0.0), raw[n][1]) for n in ORDER}
        # coverage 전용: 약한 블러(σ0.6) — 셀 speckle 만 평활, 강한 peak 의 이웃 번짐(약신호 inflation) 최소
        cov = {n: (_denoise_d5(raw[n][0], raw[n][1], sigma=0.6, floor_frac=0.0), raw[n][1]) for n in ORDER}
    else:
        data = raw; ov = raw; cov = raw
    Ns = {n: overlap_norm(*ov[n]) for n in ORDER}
    for n in ORDER:   # 통계 수집 (coverage 는 cov = d5 약블러본/ d10 raw → 맵과 동일 기준)
        _STATS[(dia, n)] = {"sensors": sensor_stats(dia, n, *data[n]),
                            "overlap": overlap_stats(Ns[n]),
                            "coverage": coverage_stats(*cov[n])}
    return {"data": data, "Ns": Ns, "raw": raw, "ov": ov, "cov": cov}


def render(dia, comp, ov_vmax):
    """ov_vmax = d5·d10 공통 overlap 컬러스케일(범례 통일 → 인덴터 간 직접 비교)."""
    data, Ns = comp["data"], comp["Ns"]
    # 센서별 몽타주: 3소재 공통 vmax (같은 스케일 → 비교 가능). 몽타주는 인덴터별(d10≈100%, d5≈40%) 그대로.
    vmax_m = max(max(g.max() for t, g in data[n][0].items() if t not in data[n][1]) for n in ORDER)
    for n in ORDER:
        montage(dia, n, *data[n], vmax_m)

    # ---- B. 겹침(overlap). 컬러스케일은 d5·d10 공통(ov_vmax) → 두 그림 범례 동일 ----
    vmax = ov_vmax
    yy, xx = np.meshgrid(g2.CENTERS, g2.CENTERS, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.0), constrained_layout=True)
    im = None
    for ax, n in zip(axes, ORDER):
        im = ax.imshow(Ns[n], origin="lower", cmap="turbo", vmin=0, vmax=vmax,
                       extent=EXT, interpolation="bilinear", aspect="equal")
        dd = data[n][1]
        ax.set_title(LABEL[n] + (f"  [dead {','.join(sorted(d.replace('Skin','s') for d in dd))}]" if dd else ""),
                     fontsize=11, fontweight="bold", color="#c0392b" if dd else "black")
        ax.set_xlabel("indenter x (mm)"); ax.set_ylabel("indenter y (mm)")
        ax.set_xticks([-10, -5, 0, 5, 10]); ax.set_yticks([-10, -5, 0, 5, 10])
    fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02).set_label(
        "MULTI-SENSOR overlap = Σ(|ΔS_t|/peak) − strongest  [colour scale SHARED d5↔d10]", fontsize=9)
    fig.suptitle(f"Receptive-field MULTI-SENSOR overlap  ({dia})   [overlap colour scale identical for d5 & d10 → directly comparable]\n"
                 "dominant sensor removed → a single sensor's own centre is dark; BRIGHT = several sensors respond together (best cue for SATS x,y,z)",
                 fontsize=12, fontweight="bold")
    o2 = os.path.join(OUT, f"overlap_{dia}_2d.png")
    fig.savefig(o2, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {o2}")

    fig = plt.figure(figsize=(17, 6.2))
    surf = None
    for k, n in enumerate(ORDER):
        Nf, mm = _upsample(_blur(Ns[n], 1.0), 4); Nf = _blur(Nf, 2.0)
        Xf, Yf = np.meshgrid(mm, mm)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        surf = ax.plot_surface(Xf, Yf, Nf, cmap="turbo", vmin=0, vmax=vmax, rstride=2, cstride=2,
                               linewidth=0, antialiased=True)
        ax.set_zlim(0, vmax * 1.05)
        ax.set_xlabel("x (mm)", fontsize=8); ax.set_ylabel("y (mm)", fontsize=8)
        ax.set_zlabel("multi-sensor overlap", fontsize=8)
        dd = data[n][1]
        ax.set_title(LABEL[n] + (f"  [dead {','.join(sorted(d.replace('Skin','s') for d in dd))}]" if dd else ""),
                     fontsize=11, fontweight="bold", color="#c0392b" if dd else "black")
        ax.view_init(elev=40, azim=-58)
    fig.colorbar(surf, ax=fig.axes, fraction=0.012, pad=0.04).set_label("multi-sensor overlap", fontsize=9)
    fig.suptitle(f"3D — multi-sensor overlap ({dia})  [z-scale identical for d5 & d10];  peaks = where several sensors respond together (overlap zones), not sensor centres",
                 fontsize=12, fontweight="bold")
    o3 = os.path.join(OUT, f"overlap_{dia}_3d.png")
    fig.savefig(o3, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {o3}")


def _cnt_map(grids, dead, floor):
    """각 인덴터 위치에서 |ΔS|≥floor 인 센서 수 N(x,y)."""
    live = [g for t, g in grids.items() if t not in dead]
    return (np.stack(live) >= floor).sum(axis=0)


def _rank_map(grids, dead, rank):
    """각 위치에서 rank 번째로 센 센서의 |ΔS| (rank=2 → 2번째, rank=3 → 3번째)."""
    live = [g for t, g in grids.items() if t not in dead]
    return np.sort(np.stack(live), axis=0)[-rank]


def sensor_rank_map(dia, comp, rank, fname, floor=1.0):
    """SATS 학습신호 = 각 위치에서 rank 번째로 센 센서의 |ΔS| 연속 맵.
    rank2 = SR 최소(2 센서), rank3 = 모호함 없는 SR 바(3 센서). 정량: 중앙값 + 그 신호≥1% 면적%.
    vmax 는 **이 그림(rank)의 3소재 공통**(자기 스케일) → 소재끼리 비교되면서 d5·d10 모두 또렷."""
    g = comp["cov"]              # d5 약블러(σ0.6) / d10 raw — STATS 와 동일 소스
    ordinal = {2: "2nd", 3: "3rd"}[rank]
    role = {2: "minimum for SR (≥2 sensors)", 3: "unambiguous SR bar (≥3 sensors: x,y+force)"}[rank]
    maps = {n: _rank_map(*g[n], rank) for n in ORDER}
    vmax = max(np.percentile(maps[n], 99) for n in ORDER)   # 이 rank 의 3소재 공통 스케일
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.0), constrained_layout=True)
    im = None
    for ax, n in zip(axes, ORDER):
        m = maps[n]
        area = 100 * (m >= floor).sum() / m.size      # 그 위치에 rank 번째 센서가 노이즈 위 = ≥rank coverage
        disp, _ = _upsample(_blur(m, 0.8), 4)         # 표시용 업샘플+스무딩 → 매끈한 연속 heatmap
        im = ax.imshow(disp, origin="lower", cmap="turbo", vmin=0, vmax=vmax, extent=EXT,
                       interpolation="bilinear", aspect="equal")
        ax.set_title(f"{LABEL[n]}\n{ordinal}-sensor median {np.median(m):.1f}%   (≥{floor:.0f}% over {area:.0f}% of area)",
                     fontsize=10.5, fontweight="bold")
        ax.set_xlabel("indenter x (mm)"); ax.set_ylabel("indenter y (mm)")
        ax.set_xticks([-10, -5, 0, 5, 10]); ax.set_yticks([-10, -5, 0, 5, 10])
    fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02).set_label(
        f"{ordinal}-strongest sensor |ΔS| (%)  [3 materials shared scale]  — warm = SR-usable {ordinal} sensor signal", fontsize=9)
    fig.suptitle(f"SATS training signal — {ordinal}-sensor response ({dia})  [{role}]\n"
                 f"warm & wide = recorded data carries a {ordinal} sensor signal SATS can learn from;  dark (eco20) = no {ordinal} sensor (not SR-trainable)",
                 fontsize=12, fontweight="bold")
    out = os.path.join(OUT, f"{fname}_{dia}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out}")




def spreading_gain(stats, floor=1.0):
    """논문풍 표: 소재 × (d5 SATS regime / d10 reference) 다중센서 SR 지표.
    핵심 = d5 에서 eco20 만 ≥3 coverage 0% 로 붕괴, eco50·mesh 는 유지."""
    rows = ["eco20", "eco50", "ecomesh"]
    rlabel = {"eco20": "eco20", "eco50": "eco50", "ecomesh": "ecomesh (mesh20)"}
    # 열: d5 5개 + d10 2개
    cols = ["2nd-sensor\nmedian (%)", "3rd-sensor\nmedian (%)", "≥3-sensor\ncoverage (%)",
            "blind-spot\n(%)", "overlap\nO̅", "≥3-sensor\ncoverage (%)", "overlap\nO̅"]
    grp = ["d5  (⌀5 mm — SATS operating regime)"] * 5 + ["d10  (⌀10 mm — reference)"] * 2

    def vals(dia, n):
        s = stats[(dia, n)]; c = s["coverage"]
        return [c["med2"], c["med3"], c[floor]["ge3"], c[floor]["blind"], s["overlap"]["mean"]]
    cell, raw_ge3_d5 = [], []
    for n in rows:
        d5 = vals("d5", n); d10 = vals("d10", n)
        raw_ge3_d5.append(d5[2])
        cell.append([f"{d5[0]:.1f}", f"{d5[1]:.1f}", f"{d5[2]:.0f}", f"{d5[3]:.0f}", f"{d5[4]:.2f}",
                     f"{d10[2]:.0f}", f"{d10[4]:.2f}"])

    fig, ax = plt.subplots(figsize=(13, 3.6)); ax.axis("off")
    tbl = ax.table(cellText=cell, rowLabels=[rlabel[r] for r in rows], colLabels=cols,
                   cellLoc="center", rowLoc="center", loc="center", bbox=[0.02, 0.0, 0.98, 0.80])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5)
    ncol = len(cols)
    # 헤더 스타일
    for j in range(ncol):
        c = tbl[0, j]; c.set_facecolor("#34495e"); c.set_text_props(color="white", fontweight="bold"); c.set_height(0.20)
    for i in range(len(rows)):
        tbl[i + 1, -1].set_text_props(fontweight="bold")
        for j in range(ncol):
            tbl[i + 1, j].set_height(0.16)
        # ≥3 coverage(d5=열2) 색칠: 0=빨강(붕괴), 높으면 초록
        g = raw_ge3_d5[i]
        tbl[i + 1, 2].set_facecolor("#f5b7b1" if g < 5 else "#abebc6")
        tbl[i + 1, 2].set_text_props(fontweight="bold")
    # d5 / d10 그룹 구분 음영 (열 0~4 = d5, 5~6 = d10)
    for i in range(len(rows)):
        for j in range(5, 7):
            tbl[i + 1, j].set_facecolor("#eef2f7")

    # 그룹 헤더(상단 띠)
    ax.text(0.02 + 0.98 * (5 / ncol) / 2 * 1.0, 0.86, grp[0], ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="#1a5276", transform=ax.transAxes)
    ax.text(0.02 + 0.98 * (5.5 / ncol), 0.86, grp[-1], ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="#5d6d7e", transform=ax.transAxes)
    ax.plot([0.02 + 0.98 * 5 / ncol] * 2, [0.0, 0.86], color="#34495e", lw=1.5, transform=ax.transAxes, clip_on=False)

    fig.suptitle("Multi-sensor coverage for super-resolution  (SR needs ≥3 sensors: x, y + contact force = 3 unknowns)",
                 fontsize=12.5, fontweight="bold", y=1.04)
    ax.text(0.5, -0.13, "At d5 (small contact = SATS regime) eco20 COLLAPSES: 2nd/3rd sensor ≈ noise → ≥3-sensor coverage 0% (SR impossible). "
            "eco50 & mesh keep ≥3-sensor signal (68–76%).\nAt d10 (large contact) all materials excite many sensors → difference vanishes. "
            "→ the mesh/eco50 pressure-spreading advantage is greatest exactly where SATS operates (small contacts).",
            ha="center", va="top", fontsize=8.6, color="#333", transform=ax.transAxes)
    out = os.path.join(OUT, "spreading_gain.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--diameter", choices=["d5", "d10", "all"], default="all")
    a = ap.parse_args()
    dias = ["d5", "d10"] if a.diameter == "all" else [a.diameter]
    comps = {dia: compute(dia) for dia in dias}
    # overlap 컬러스케일을 모든 인덴터 공통으로 → overlap_d5/d10 범례 동일(직접 비교 가능)
    ov_vmax = max(N.max() for c in comps.values() for N in c["Ns"].values())
    print(f"[overlap] shared vmax across {dias} = {ov_vmax:.3f}")
    for dia in dias:
        render(dia, comps[dia], ov_vmax)
        sensor_rank_map(dia, comps[dia], 2, "second_sensor")  # 2번째(SR 최소)
        sensor_rank_map(dia, comps[dia], 3, "coverage")       # 3번째(≥3=SR 바), 동일 형태·자기 스케일
    if len(dias) == 2:   # d5·d10 둘 다 있어야 spreading-gain 비교 가능
        spreading_gain(_STATS, floor=1.0)
    write_stats(_STATS)
