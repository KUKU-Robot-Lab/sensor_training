"""논문용 figure set HTML 빌더 (단일 자가완결 파일) — 3 figure 확장판.

20260629.pptx 의 기존 figure 작업 + xy_1mm atomic 패널을 통합:
  - Fig.2 (C1 소재 ablation): atomic PNG 8종 + mesh 메커니즘 inline SVG → 3×3
  - Fig.3 (SR 학습구조 & 모델 벤치마크): SATS 파이프라인 SVG + pptx error/model/3D 크롭
       + pptx 정량 수치(모델 리더보드·소재별 SR) HTML 표/막대 → 3×3
  - Fig.4 (C2 Bending-aware SR): pptx Module A~D 멀티모듈 네트워크 inline SVG + 잔차보정
       원리 카드 + 미취득 패널 placeholder → 3×3

이미지는 base64 내장(단일 파일). pptx figure 는 PowerPoint COM 으로 렌더 후 크롭한 panels/pptx_*.png.
재현: generate_atomic_panels.py → generate_benchmark_panels.py → (pptx 크롭) → build_figure_set_html.py
출력: visualizing_scripts/figure_set/figure_set.html
"""
import os
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = os.path.join(HERE, "panels")
OUT = os.path.join(HERE, "figure_set.html")


def b64(name):
    with open(os.path.join(PANELS, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# ───────────────────────── SVG 헬퍼 ─────────────────────────
ARROW_DEFS = """
  <defs>
    <marker id="ah" markerWidth="9" markerHeight="9" refX="7.5" refY="4"
            orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L8,4 L0,8 z" fill="#3a3a44"/>
    </marker>
    <marker id="ahp" markerWidth="9" markerHeight="9" refX="7.5" refY="4"
            orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L8,4 L0,8 z" fill="#7c5bb0"/>
    </marker>
    <linearGradient id="sats" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#7257ad"/><stop offset="1" stop-color="#473371"/>
    </linearGradient>
    <filter id="sh" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="1.5" stdDeviation="2" flood-color="#1b1b2b" flood-opacity="0.18"/>
    </filter>
  </defs>"""


def box(x, y, w, h, lines, fill, stroke, tcolor="#1d2330", fs=13, bold=False, rx=11):
    n = len(lines); lh = fs + 4
    y0 = y + h / 2 - (n - 1) * lh / 2
    tspans = "".join(
        f'<tspan x="{x + w/2:.0f}" y="{y0 + i*lh:.1f}">{ln}</tspan>' for i, ln in enumerate(lines))
    fw = 'font-weight="700"' if bold else 'font-weight="500"'
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.6" filter="url(#sh)"/>'
            f'<text text-anchor="middle" font-size="{fs}" {fw} fill="{tcolor}" '
            f'font-family="Inter,Segoe UI,Helvetica,Arial,sans-serif">{tspans}</text>')


def arrow(x1, y1, x2, y2, mid=None, color="#3a3a44", marker="ah", dash=None, fs=12,
          tcolor=None, lift=7, via=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    if via:
        el = (f'<path d="M{x1},{y1} L{via[0]},{via[1]} L{x2},{y2}" fill="none" '
              f'stroke="{color}" stroke-width="2"{d} marker-end="url(#{marker})"/>')
    else:
        el = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
              f'stroke-width="2"{d} marker-end="url(#{marker})"/>')
    if mid:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        tc = tcolor or color
        el += (f'<text x="{mx}" y="{my - lift}" text-anchor="middle" font-size="{fs}" '
               f'font-style="italic" fill="{tc}" font-family="Inter,Segoe UI,sans-serif">{mid}</text>')
    return el


def band(x, y, w, h, stroke, fill):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.6" stroke-dasharray="7 5" opacity="0.95"/>')


def lock(cx, cy):
    return (f'<g transform="translate({cx-9},{cy-9})">'
            f'<rect x="2" y="8" width="14" height="10" rx="2" fill="#f3c04a" stroke="#9c7613" stroke-width="1"/>'
            f'<path d="M5,8 V5 a4,4 0 0 1 8,0 V8" fill="none" stroke="#9c7613" stroke-width="1.6"/></g>')


C_IN, S_IN = "#e8f0fb", "#3b6ea5"
C_NEW, S_NEW = "#e4f4e8", "#2e8b57"
C_OUT, S_OUT = "#fde7d6", "#d98324"


# ───────────────────────── Fig.3a : SATS SR 파이프라인 ─────────────────────────
def sats_pipeline_svg():
    s = ['<svg viewBox="0 0 1200 250" xmlns="http://www.w3.org/2000/svg" '
         'font-family="Inter,Segoe UI,sans-serif">', ARROW_DEFS]
    s.append(box(30, 92, 132, 66, ["sensor seq", "[B, T, 16]"], C_IN, S_IN))
    s.append(box(206, 92, 150, 66, ["per-taxel LSTM", "(unit response)"], C_IN, S_IN, fs=12))
    s.append(box(400, 80, 168, 90, ["Self-Attention", "(4×4 GAT)", "receptive-field graph"],
                 "url(#sats)", "#3a2d63", tcolor="#fff", fs=12, bold=True))
    s.append(box(612, 92, 150, 66, ["Local-Map", "assembly"], "url(#sats)", "#3a2d63", tcolor="#fff", fs=12))
    s.append(box(806, 92, 150, 66, ["CNN refine", "(SR up-sample)"], "url(#sats)", "#3a2d63", tcolor="#fff", fs=12))
    s.append(box(1000, 60, 170, 60, ["Head — (x, y)"], C_OUT, S_OUT, fs=12, bold=True))
    s.append(box(1000, 132, 170, 60, ["Head — z, Fz, Area"], C_OUT, S_OUT, fs=12, bold=True))
    s.append(arrow(162, 125, 204, 125))
    s.append(arrow(356, 125, 398, 125, "s_norm", tcolor="#2f5d8f"))
    s.append(arrow(568, 125, 610, 125))
    s.append(arrow(762, 125, 804, 125))
    s.append(arrow(956, 118, 998, 92))
    s.append(arrow(956, 132, 998, 158))
    s.append('<text x="600" y="226" text-anchor="middle" font-size="12.5" fill="#54616f">'
             'sparse 16-taxel graph → super-resolved pressure / localization (deep-learning SR)</text>')
    s.append("</svg>")
    return "\n".join(s)


# ───────────────────────── Fig.4a : Module A~D 밴딩 네트워크 ─────────────────────────
def module_abcd_svg():
    """pptx slide18/19 의 멀티모듈 학습구조 재작성."""
    s = ['<svg viewBox="0 0 1200 480" xmlns="http://www.w3.org/2000/svg" '
         'font-family="Inter,Segoe UI,sans-serif">', ARROW_DEFS]

    # ── Module A : Baseline → Bending-State Estimator → θ̂ (상단) ──
    s.append(band(16, 30, 660, 118, "#2e8b57", "#f4fbf6"))
    s.append('<text x="30" y="54" font-size="14" font-weight="700" fill="#23704a">'
             'Module A — Bending-State Estimator (self-supervised)</text>')
    s.append(box(34, 70, 168, 60, ["no-load baseline", "shift Δp(κ)"], C_NEW, S_NEW, fs=12))
    s.append(box(232, 70, 196, 60, ["Baseline → θ̂ regressor", "(MLP, online)"], C_NEW, S_NEW, fs=12, bold=True))
    s.append(box(458, 70, 92, 60, ["Head — θ̂"], C_OUT, S_OUT, fs=12, bold=True))
    s.append(arrow(202, 100, 230, 100))
    s.append(arrow(428, 100, 456, 100))
    # θ̂ 를 Input 으로 주입 (아래로)
    s.append(arrow(504, 130, 150, 250, color="#23704a", dash="6 5", via=(504, 200), mid="θ̂", tcolor="#23704a"))

    # ── 메인 레인 : Input → B → C → D ──
    s.append(box(40, 232, 132, 78, ["Input", "[s₁..s₁₆, θ̂]"], C_IN, S_IN, fs=12, bold=True))
    s.append(box(212, 214, 198, 114,
                 ["Module B", "Tactile Feature Encoder", "CNN → corrected", "tactile map"],
                 C_IN, S_IN, fs=12, bold=True))
    s.append(box(452, 214, 198, 114,
                 ["Module C", "Localization Encoder", "(SATS) → latent"],
                 "url(#sats)", "#3a2d63", tcolor="#fff", fs=12, bold=True))
    s.append(box(692, 214, 210, 114,
                 ["Module D", "Shared Contact", "Mechanics Block (MLP)"],
                 C_NEW, S_NEW, fs=12, bold=True))
    s.append(arrow(172, 271, 210, 271))
    s.append(arrow(410, 271, 450, 271, "corrected map", tcolor="#3b6ea5"))
    s.append(arrow(650, 271, 690, 271, "latent", tcolor="#7c5bb0"))

    # ── Heads ──
    s.append(box(452, 372, 198, 56, ["Head — (x, y)"], C_OUT, S_OUT, fs=12, bold=True))
    s.append(arrow(551, 328, 551, 370, "localization", tcolor="#d98324", lift=4, fs=11))
    s.append(box(948, 206, 150, 44, ["Head — z"], C_OUT, S_OUT, fs=12, bold=True))
    s.append(box(948, 258, 150, 44, ["Head — Fz"], C_OUT, S_OUT, fs=12, bold=True))
    s.append(box(948, 310, 150, 44, ["Head — Area_eff"], C_OUT, S_OUT, fs=12, bold=True))
    for yy in (228, 280, 332):
        s.append(arrow(902, 271 if yy == 280 else (250 if yy == 228 else 292), 946, yy))
    # frozen SATS 표시 (Module C)
    s.append(lock(636, 224))
    s.append('<text x="551" y="456" text-anchor="middle" font-size="12" fill="#54616f">'
             'flat-trained SATS core (Module C) reused; θ̂ from Module A conditions the bent-state input</text>')
    s.append("</svg>")
    return "\n".join(s)


# ───────────────────────── Fig.2c : mesh 압력전달 메커니즘 ─────────────────────────
def mesh_mechanism_svg():
    def cell(ox, title, spread, strength, mesh, subtitle):
        g = [f'<g transform="translate({ox},0)">']
        block_fill = "#cfe3f5" if not mesh else "#bcd3ee"
        g.append(f'<rect x="6" y="40" width="150" height="60" rx="6" fill="{block_fill}" '
                 f'stroke="#5a6b80" stroke-width="1.3"/>')
        if mesh:
            for hx in range(14, 150, 13):
                g.append(f'<line x1="{6+hx}" y1="42" x2="{6+hx-18}" y2="98" stroke="#33619b" stroke-width="0.8" opacity="0.6"/>')
                g.append(f'<line x1="{6+hx-18}" y1="42" x2="{6+hx}" y2="98" stroke="#33619b" stroke-width="0.8" opacity="0.6"/>')
        g.append('<circle cx="81" cy="28" r="9" fill="#c0392b"/>')
        g.append('<path d="M81,38 L81,44" stroke="#c0392b" stroke-width="2.4" marker-end="url(#ah)"/>')
        half = spread / 2; op = 0.16 + 0.5 * strength
        g.append(f'<path d="M81,46 L{81-half},100 L{81+half},100 z" fill="#e8533a" opacity="{op:.2f}"/>')
        for tx in range(18, 150, 24):
            active = abs((6 + tx) - 81) <= half + 4
            fc = "#1b2a41" if active else "#9aa6b4"
            g.append(f'<rect x="{6+tx-7}" y="104" width="14" height="9" rx="2" fill="{fc}"/>')
        g.append(f'<text x="81" y="132" text-anchor="middle" font-size="13" font-weight="700" fill="#1d2330">{title}</text>')
        g.append(f'<text x="81" y="149" text-anchor="middle" font-size="11" fill="#54616f">{subtitle}</text>')
        g.append("</g>")
        return "".join(g)
    s = ['<svg viewBox="0 0 540 165" xmlns="http://www.w3.org/2000/svg" '
         'font-family="Inter,Segoe UI,sans-serif">', ARROW_DEFS]
    s.append(cell(0, "eco20", 34, 0.55, False, "localized → undersampling"))
    s.append(cell(182, "eco50", 92, 0.35, False, "wide but weak → SNR loss"))
    s.append(cell(364, "ecomesh", 96, 0.7, True, "wide + sensitive → best SR"))
    s.append("</svg>")
    return "\n".join(s)


# ───────────────────────── Fig.4 placeholder / 수식 카드 ─────────────────────────
def placeholder_svg(xlabel, ylabel, kind):
    s = ['<svg viewBox="0 0 320 240" xmlns="http://www.w3.org/2000/svg" '
         'font-family="Inter,Segoe UI,sans-serif">']
    s.append('<line x1="44" y1="196" x2="300" y2="196" stroke="#9aa6b4" stroke-width="1.4"/>')
    s.append('<line x1="44" y1="28" x2="44" y2="196" stroke="#9aa6b4" stroke-width="1.4"/>')
    g = "#c7cfd9"
    if kind == "baseline":
        s.append(f'<polyline points="60,70 110,120 172,180 234,118 290,66" fill="none" stroke="{g}" stroke-width="2.4" stroke-dasharray="6 5"/>')
        for px, py in [(60,70),(110,120),(172,180),(234,118),(290,66)]:
            s.append(f'<circle cx="{px}" cy="{py}" r="3.4" fill="{g}"/>')
    elif kind == "regress":
        s.append(f'<line x1="56" y1="186" x2="292" y2="44" stroke="{g}" stroke-width="2" stroke-dasharray="6 5"/>')
        import random; random.seed(3)
        for t in [0.1,0.25,0.4,0.55,0.7,0.85]:
            px = 56 + t*236; py = 186 - t*142 + random.uniform(-10,10)
            s.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="3.4" fill="{g}"/>')
    elif kind == "bars":
        s.append(f'<rect x="92" y="120" width="48" height="76" fill="{g}"/>')
        s.append(f'<rect x="196" y="92" width="48" height="104" fill="{g}"/>')
        s.append('<text x="116" y="214" text-anchor="middle" font-size="11" fill="#7a8694">flat</text>')
        s.append('<text x="220" y="214" text-anchor="middle" font-size="11" fill="#7a8694">bent</text>')
    elif kind == "maps":
        for ox, lab in [(70,"pred"),(186,"GT")]:
            s.append(f'<rect x="{ox}" y="70" width="64" height="64" rx="4" fill="{g}"/>')
            s.append(f'<circle cx="{ox+32}" cy="102" r="16" fill="#aeb8c4"/>')
            s.append(f'<text x="{ox+32}" y="150" text-anchor="middle" font-size="11" fill="#7a8694">{lab}</text>')
    s.append(f'<text x="172" y="224" text-anchor="middle" font-size="12" fill="#54616f">{xlabel}</text>')
    s.append(f'<text x="16" y="112" text-anchor="middle" font-size="12" fill="#54616f" transform="rotate(-90,16,112)">{ylabel}</text>')
    s.append('<rect x="96" y="14" width="150" height="26" rx="13" fill="#fff4e0" stroke="#e0a93a" stroke-width="1.2"/>')
    s.append('<text x="171" y="31" text-anchor="middle" font-size="12" font-weight="700" fill="#b5821f">DATA PENDING · jig</text>')
    s.append("</svg>")
    return "\n".join(s)


def design_summary_svg():
    s = ['<svg viewBox="0 0 320 240" xmlns="http://www.w3.org/2000/svg" font-family="Inter,Segoe UI,sans-serif">']
    s.append('<rect x="8" y="14" width="304" height="212" rx="10" fill="#f6f4fb" stroke="#7c5bb0" stroke-width="1.4"/>')
    s.append('<text x="160" y="42" text-anchor="middle" font-size="13" font-weight="700" fill="#4b3a78">Residual decomposition (§5.4)</text>')
    rows = [("raw(contact, κ) ≈", "#1d2330", 74), ("  p_baseline(κ)  +  r(contact, κ)", "#3b6ea5", 96),
            ("subtract Δp_bend(θ̂):", "#23704a", 132), ("  r(contact, κ) ≈ r(contact, 0)", "#23704a", 154),
            ("→ reuse flat SATS (Module C)", "#7c5bb0", 190)]
    for txt, col, y in rows:
        w = "700" if txt.startswith(("raw", "subtract", "→")) else "500"
        s.append(f'<text x="26" y="{y}" font-size="13" font-weight="{w}" fill="{col}" font-family="Georgia,serif">{txt}</text>')
    s.append("</svg>")
    return "\n".join(s)


# ───────────────────────── HTML 표 (pptx 정량 수치) ─────────────────────────
def table(title, headers, rows, hi=None, lowbest=None):
    """hi=강조 행 인덱스 집합, lowbest=낮을수록 좋은 열 인덱스(굵게 최솟값)."""
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for ri, r in enumerate(rows):
        cls = ' class="hi"' if hi and ri in hi else ""
        tds = "".join(f"<td>{c}</td>" for c in r)
        body.append(f"<tr{cls}>{tds}</tr>")
    return (f'<div class="tbl"><div class="tbl-t">{title}</div>'
            f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def loc_table():
    rows = [["SATS","0.577","0.991","0.998"],["Multi-Head","0.764","0.984","0.997"],
            ["CNN-LSTM","0.869","0.982","0.995"],["Unified","1.836","0.930","0.971"],
            ["MLP","2.114","0.945","0.972"],["CNN","2.172","0.938","0.972"],
            ["Transformer","2.286","0.933","0.966"],["Isoline-GNN","2.338","0.934","0.964"],
            ["GNN-GAT","2.633","0.911","0.948"]]
    return table("Localization @ D5  (mean xy err mm, R²ₓ, R²_y)",
                 ["Model","xy err","R²ₓ","R²_y"], rows, hi={0})


def depth_table():
    rows = [["CNN-LSTM","0.145","0.155","0.548"],["Multi-Head","0.174","0.158","0.532"],
            ["SATS","0.147","0.161","0.516"],["MLP","0.137","0.177","0.414"],
            ["Transformer","0.197","0.196","0.277"],["CNN","0.233","0.200","0.254"],
            ["GNN-GAT","0.226","0.204","0.217"],["Unified","0.198","0.220","0.093"],
            ["Isoline-GNN","0.220","0.228","0.029"]]
    return table("Depth z @ D5  (MAE, RMSE, R²_z)",
                 ["Model","MAE","RMSE","R²_z"], rows, hi={0,2})


def material_tables():
    ax = table("Material SR — About [x]  (Ours = ECO20+MESH)",
               ["Mat","MSE","RMSE","MAE","R²"],
               [["Ours","0.394","0.628","0.483","0.988"],["ECO20","0.713","0.845","0.569","0.978"],
                ["ECO30","1.375","1.173","0.725","0.958"],["ECO50","0.444","0.667","0.488","0.987"]], hi={0})
    az = table("Material SR — About [z]",
               ["Mat","MSE","RMSE","MAE","R²"],
               [["Ours","0.288","0.536","0.477","0.731"],["ECO20","0.352","0.593","0.535","0.668"],
                ["ECO30","0.334","0.578","0.509","0.694"],["ECO50","0.332","0.576","0.504","0.689"]], hi={0})
    return ax + az


# ───────────────────────── 셀 / figure 조립 ─────────────────────────
def img_cell(letter, src, cap):
    return (f'<figure class="cell"><span class="lbl">({letter})</span>'
            f'<img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')


def node_cell(letter, inner, cap, hero=False):
    cls = "cell hero" if hero else "cell"
    return (f'<figure class="{cls}"><span class="lbl">({letter})</span>'
            f'<div class="svgwrap">{inner}</div><figcaption>{cap}</figcaption></figure>')


def build():
    P = {n: b64(f"fig2_{n}.png") for n in ["a_topview","b_xsection","g_radial","h_metrics","i_sigma"]}
    RF = {m: b64(f"fig2_rf_d5_{m}.png") for m in ["eco20","eco50","ecomesh"]}
    PX = {n: b64(f"pptx_{n}.png") for n in ["error_maps","model_maps","rf3d_matrix"]}
    BN = {n: b64(f"bench_{n}.png") for n in ["loc_leaderboard","material_r2"]}

    fig2 = [
        img_cell("a", P["a_topview"], "Sensor top view — 16 sparse MEMS taxels (6.5 mm pitch), 1 mm xy scan grid, d5/d10 footprints."),
        img_cell("b", P["b_xsection"], "Three-layer cross-section (5.5 mm): mesh-loaded top / MEMS-embedded mid / base."),
        node_cell("c", mesh_mechanism_svg(), "Mechanism (C1): mesh = semi-rigid load-spreading skeleton — wide spread <i>and</i> retained sensitivity."),
        img_cell("d", RF["eco20"], "Receptive field, eco20 (d5): weak, localized (peak 21%)."),
        img_cell("e", RF["eco50"], "Receptive field, eco50 (d5): stronger, stiffer spread (peak 49%)."),
        img_cell("f", RF["ecomesh"], "Receptive field, ecomesh (d5): highest peak (51%), most active cells (24)."),
        img_cell("g", P["g_radial"], "Radial attenuation (d5): eco20 under-coupled; mesh/eco50 carry signal outward."),
        img_cell("h", P["h_metrics"], "Array metrics (d5): spread indices rise eco20<eco50<ecomesh; ecomesh keeps eco50-level sensitivity."),
        img_cell("i", '<img src="' + P["i_sigma"] + '" style="width:100%">' if False else P["i_sigma"], ""),
    ]
    # (i) 는 이미지 셀로 (위 라인 단순화)
    fig2[-1] = img_cell("i", P["i_sigma"], "Receptive-field σ at saturated d10 — monotonic spread eco20<eco50<ecomesh (d5 σ omitted: eco20 inflated at low SNR).")

    supp_bench = [
        node_cell("a", sats_pipeline_svg(), "SATS SR pipeline: sparse 16-taxel graph → per-taxel LSTM → self-attention (4×4 GAT) → local-map → CNN refine → localization & pressure heads.", hero=True),
        img_cell("b", PX["error_maps"], "Spatial error maps across the array — X / Y / Z / Fz MAE (lighter = lower error)."),
        img_cell("c", PX["model_maps"], "Localization quality across 9 learning structures (MLP, CNN, CNN-LSTM, Isoline-GNN, Unified, Multi-Head, Transformer, GNN-GAT, SATS)."),
        img_cell("d", PX["rf3d_matrix"], "3D receptive-field surfaces, material (ECO20+MESH/ECO20/ECO30/ECO50) × indenter (D10/D5/D4/D3) — input richness for SR."),
        img_cell("e", BN["loc_leaderboard"], "Localization leaderboard @ D5 — SATS lowest mean xy error (0.58 mm); attention/recurrent models beat MLP/GNN baselines."),
        node_cell("f", loc_table() + depth_table(), "Quantitative leaderboard (pptx): SATS best localization; CNN-LSTM/Multi-Head/SATS lead depth-z."),
        img_cell("g", BN["material_r2"], "Material-wise SR R² (location & depth): Ours (ECO20+MESH) highest — mesh selection confirmed on the learned task, not only receptive field."),
        node_cell("h", material_tables(), "Material SR error tables (pptx) — Ours best R² for both x and z."),
        node_cell("i", sr_takeaway_svg(), "Takeaway: SATS attention over the mesh-shaped receptive-field graph turns sparse 16-taxel input into sub-mm localization."),
    ]

    fig3_bending = [
        node_cell("a", module_abcd_svg(),
                  "Bending-aware multi-module network (pptx Module A–D). <b>Module A</b> regresses curvature θ̂ from the no-load baseline shift and conditions the input; <b>B</b> CNN tactile encoder yields a corrected map; <b>C</b> the flat-trained SATS localization encoder is reused; <b>D</b> a shared contact-mechanics MLP outputs z / Fz / effective area. Curvature self-estimation lets the flat SR model run under bending without curvature-specific retraining.", hero=True),
        node_cell("b", placeholder_svg("taxel axial distance zᵢ (mm)", "|Δbaseline| (a.u.)", "baseline"),
                  "Per-taxel baseline shift vs zᵢ under pure bending — expected Δpᵢ ∝ κ·zᵢ (§5.3)."),
        node_cell("c", placeholder_svg("jig angle θ (deg)", "estimated θ̂ (deg)", "regress"),
                  "Module A curvature regression (θ̂ vs jig θ): report MAE / R²."),
        node_cell("d", placeholder_svg("condition", "SR RMSE (mm)", "bars"),
                  "Flat vs bent SR error, correction on/off — graceful degradation (§5.4)."),
        node_cell("e", placeholder_svg("", "", "maps"),
                  "Reconstructed vs ground-truth SR map on a bent contact."),
        node_cell("f", design_summary_svg(),
                  "Residual-decomposition basis: subtracting Δp_bend(θ̂) restores the flat residual the frozen SATS core (Module C) expects."),
    ]

    css = """
    :root{--ink:#1d2330;--mut:#54616f;--line:#d7dde5;--bg:#ffffff;}
    *{box-sizing:border-box;}
    body{margin:0;background:#eef1f5;color:var(--ink);font-family:Inter,'Segoe UI',Helvetica,Arial,sans-serif;line-height:1.5;}
    .page{max-width:1300px;margin:0 auto;background:var(--bg);padding:46px 54px 70px;box-shadow:0 1px 14px rgba(0,0,0,.07);}
    header.doc{border-bottom:2px solid var(--ink);padding-bottom:16px;margin-bottom:30px;}
    header.doc h1{font-family:Georgia,'Times New Roman',serif;font-size:24px;margin:0 0 6px;}
    header.doc p{margin:2px 0;color:var(--mut);font-size:13.5px;}
    .badge{display:inline-block;background:#1d2330;color:#fff;font-size:11px;padding:2px 9px;border-radius:10px;margin-right:7px;letter-spacing:.3px;}
    section.fig{margin:42px 0 8px;}
    .fig-head{font-family:Georgia,'Times New Roman',serif;margin:0 0 4px;}
    .fig-head .fno{font-size:19px;font-weight:700;} .fig-head .ftitle{font-size:19px;}
    .fig-sub{color:var(--mut);font-size:13px;margin:0 0 18px;}
    .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;}
    .cell{position:relative;margin:0;border:1px solid var(--line);border-radius:10px;padding:30px 14px 14px;background:#fcfdfe;display:flex;flex-direction:column;}
    .cell.hero{grid-column:1 / -1;}
    .cell .lbl{position:absolute;top:9px;left:12px;font-family:Georgia,serif;font-weight:700;font-size:15px;color:var(--ink);}
    .cell img{width:100%;height:auto;display:block;border-radius:4px;}
    .svgwrap{width:100%;} .svgwrap svg{width:100%;height:auto;display:block;}
    figcaption{font-size:12px;color:var(--mut);margin-top:10px;line-height:1.45;text-align:justify;}
    .cap{font-size:12.5px;color:var(--ink);margin:16px 2px 0;text-align:justify;border-top:1px solid var(--line);padding-top:12px;}
    .cap b{font-weight:700;}
    .tbl{margin:2px 0 10px;} .tbl-t{font-size:11px;font-weight:700;color:var(--ink);margin-bottom:3px;}
    table{border-collapse:collapse;width:100%;font-size:10.5px;}
    th,td{border:1px solid #e2e7ee;padding:2px 5px;text-align:right;}
    th:first-child,td:first-child{text-align:left;}
    thead th{background:#eef2f7;color:#1d2330;font-weight:700;}
    tr.hi td{background:#e7f5ec;font-weight:700;color:#1d6b3f;}
    .foot{margin-top:50px;border-top:1px solid var(--line);padding-top:14px;font-size:11.5px;color:var(--mut);}
    """

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Figure set — Flexible Tactile Sensor with Super-Resolution</title>
<style>{css}</style></head><body><div class="page">

<header class="doc">
  <h1>Development of a Flexible Tactile Sensor with Super-Resolution Capability</h1>
  <p><span class="badge">FIGURE SET</span> Working draft — numbering follows paper §6: Fig.2 (material ablation, C1; panel D = material-wise SATS, <code>Fig2D_*</code> set), Fig.3 (bending-aware SR, C2), Fig.4 (application, pending). The SR learning-structure benchmark is kept as a supplementary set.</p>
  <p>Integrates xy_1mm receptive-field analysis with the quantitative results and module architecture from <code>20260629.pptx</code>. SATS = Self-Attention-assisted Tactile Super-Resolution.</p>
</header>

<section class="fig">
  <p class="fig-head"><span class="fno">Fig. 2</span> &nbsp;<span class="ftitle">| Mesh pressure-transfer layer engineers receptive-field overlap (C1).</span></p>
  <p class="fig-sub">xy 1 mm grid indentation over ±10 mm; eco20 / eco50 / ecomesh; indenters d5 (sharp) &amp; d10 (saturated). Central taxel Skin10.</p>
  <div class="grid">{''.join(fig2)}</div>
  <p class="cap"><b>Fig. 2.</b> SR presumes overlapping receptive fields (too narrow under-samples, too wide loses SNR). (a–c) Sparse barometric sensor, decoupled stack, and the mesh load-spreading mechanism. (d–f) The central-taxel receptive field widens and strengthens eco20→eco50→ecomesh; (g) radial profiles show eco20 is under-coupled; (h) spread indices increase monotonically while ecomesh keeps eco50-level sensitivity; (i) the monotonic spread also holds at d10. Selection: <b>mesh20 (ecomesh)</b>.</p>
</section>

<section class="fig">
  <p class="fig-head"><span class="fno">Fig. 3</span> &nbsp;<span class="ftitle">| Bending-aware super-resolution without curvature-specific retraining (C2).</span></p>
  <p class="fig-sub">Bending appears as a baseline shift; curvature is self-estimated from the no-load baseline and used to condition the flat-trained SATS model. Flat-SR reference panels: <code>fig3_sats_bending/flat_sr/final_xy0p5/</code>.</p>
  <div class="grid">{''.join(fig3_bending)}</div>
  <p class="cap"><b>Fig. 3.</b> (a) The Module A–D network (from 20260629.pptx): Module A self-estimates curvature θ̂ from the no-load baseline shift (Δpᵢ ≈ kᵢ·κ·zᵢ, §5.3) and conditions the input; Module B encodes a corrected tactile map; Module C reuses the flat-trained SATS localization encoder; Module D, a shared contact-mechanics MLP, outputs depth, normal force and effective area. (b–e) Validation panels pending the jig acquisition (per-angle no-load baselines + bent contacts): baseline–zᵢ law, θ̂ regression MAE/R², flat-vs-bent SR with correction on/off, and prediction-vs-GT maps. (f) The residual-decomposition principle underlying Module A.</p>
</section>

<section class="fig">
  <p class="fig-head"><span class="fno">Fig. 4</span> &nbsp;<span class="ftitle">| Application — robot hand &amp; human hand (pending).</span></p>
  <p class="fig-sub">Real-time contact sensing/control on a robot hand and demonstration on the curved human hand (paper §6 Fig.4). Panels pending hardware demos; plan in <code>fig4_application/README.md</code>.</p>
</section>

<section class="fig">
  <p class="fig-head"><span class="fno">Supp. Fig. B1</span> &nbsp;<span class="ftitle">| SR learning structure and model benchmark.</span></p>
  <p class="fig-sub">Self-attention over the sparse taxel graph maps 16 channels to a super-resolved field; benchmarked against 8 alternative structures and 3 alternative materials (data from 20260629.pptx, indenter D5). Supports Fig.2(d) material selection and the Fig.3 SATS core; not a main-text figure in paper §6.</p>
  <div class="grid">{''.join(supp_bench)}</div>
  <p class="cap"><b>Supp. Fig. B1.</b> (a) The SATS pipeline. (b) Per-axis spatial error maps; (c) localization quality across nine learning structures; (d) 3D receptive-field surfaces over material × indenter. (e–f) SATS attains the lowest mean xy error (<b>0.58 mm</b> at D5, R²ₓ 0.991 / R²_y 0.998), with CNN-LSTM / Multi-Head / SATS leading depth-z. (g–h) On the learned SR task, <b>Ours (ECO20+MESH)</b> gives the best R² for both location and depth — confirming the mesh choice end-to-end, not just by receptive field. (i) Net message of C1+the learner.</p>
</section>

<p class="foot">References informing the design: SATS (sparse taxel graph + per-taxel LSTM + self-attention + local-map + CNN refine); Taxel-Value-Isoline / Barodome (sparse-unit SR; shear degrades localization — §8); barometric tactile sensing (low-cost, direct pressure). Built from <code>visualizing_scripts/figure_set/panels/</code>; pptx figures rendered via PowerPoint and cropped; self-contained (images embedded).</p>

</div></body></html>"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[saved] {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")


# ───────────────────────── Fig.3 (i) 요약 카드 ─────────────────────────
def sr_takeaway_svg():
    s = ['<svg viewBox="0 0 320 240" xmlns="http://www.w3.org/2000/svg" font-family="Inter,Segoe UI,sans-serif">']
    s.append('<rect x="8" y="14" width="304" height="212" rx="10" fill="#eef6f1" stroke="#2e8b57" stroke-width="1.4"/>')
    s.append('<text x="160" y="44" text-anchor="middle" font-size="13" font-weight="700" fill="#23704a">Why SATS wins</text>')
    rows = [("attention over the", "#1d2330", 78), ("mesh-shaped RF graph (Fig.2)", "#2e8b57", 100),
            ("+ per-taxel LSTM dynamics", "#3b6ea5", 134),
            ("→ 0.58 mm xy @ D5", "#23704a", 174), ("   (21-fold SR, 16 taxels)", "#54616f", 196)]
    for txt, col, y in rows:
        w = "700" if txt.startswith(("→",)) else "500"
        s.append(f'<text x="26" y="{y}" font-size="12.5" font-weight="{w}" fill="{col}" font-family="Georgia,serif">{txt}</text>')
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    build()
