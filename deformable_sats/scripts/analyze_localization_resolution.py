#!/usr/bin/env python3
"""미세 스텝 위치 분해능 분석 (D10, 오프라인) — 모터 상대변위 vs SATS centroid 상대변위.

★좌표 규약(중요): 모터 원점과 센서 원점은 정렬 불가 → **상대 변위**로 평가한다.
  - pos_mm = 모터 **절대 좌표**(원점 임의). 첫 압입점(모터 0,0 등)이 기준점.
  - EM_V2 인코더 단위 = **0.1μm**(mm = 값/10000). node z14.0 → EM 140000 실측 확정.
  - 예측·GT 모두 기준점 대비 변위(Δ)로 바꿔 비교 → 원점 불일치(상수 offset)는
    소거된다. slope·R²·σ·분해능은 원래 offset-불변(회귀 절편이 흡수).

취득 절차(사용자 프로토콜): 모터 (0,0)에서 1회 압입(기준) → 절대 좌표로 0.1mm 등
스텝 이동 → 동일 z 압입 반복. 각 점 dwell 중 ≥20프레임 기록.

산출(SPEC.md 갱신 근거):
  - slope·R²      : 예측 변위가 실제 모터 변위를 얼마나 추종하나 (1=완벽)
  - σ (per-pos)   : 같은 위치 반복 예측의 흔들림 = 반복 노이즈
  - rel err       : 기준점 정렬 후 |예측Δ−GTΔ| median (상대 정확도)
  - resolution    : 2σ/slope = 두 위치를 분간 가능한 최소 간격(분해능)

────────────────────────────────────────────────────────────────────────
입력 형식 (둘 중 하나):
  A) --data-dir DIR : 위치당 npz 1개(파일명 순=취득 순, 첫 파일=기준점). 각 npz =
        sensor   float32 [F,16]  (F≥window_size, pct 상대% = 라이브 데모 윈도우값)
        pos_mm   float32 [2]     (모터 절대 x,y[mm] — 센서 원점과 무관)
        baseline float32 [16]    (선택; sensor가 raw면 --input raw 로 pct 변환)
  B) --data FILE.npz : 전 프레임 한 파일.
        sensor   [N,16],  pos_mm [N,2]   (동일 위치 연속 프레임끼리 묶음)
  지름은 D10 고정(--diameter 10).
────────────────────────────────────────────────────────────────────────

실행(4090):
  .venv/bin/python scripts/analyze_localization_resolution.py --data-dir learning_data/xy_fine/d10
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sats.inference.inference_engine import SATSInferenceEngine       # noqa: E402
from sats.tools.multicontact_metrics import detect_peaks              # noqa: E402


def _read_em_v2(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """EM_V2 bin → (t[s], xyzu[N,4]).  x,y,z,u 단위=μm."""
    import struct
    rows = []
    with open(path, "rb") as f:
        f.readline()                               # magic line "EM_V2"
        while True:
            d = f.read(8 + 32)
            if len(d) < 40:
                break
            rows.append(struct.unpack("<Qdddd", d))
    a = np.asarray(rows, float)
    if len(a) == 0:
        raise ValueError(f"EM_V2 레코드 없음: {path}")
    return a[:, 0] / 1e9, a[:, 1:]


def _read_due_v2(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """DUE_V2 bin → (t[s] per frame, raw[N,16]). burst=16센서×10프레임."""
    import struct
    rec = 8 + 16 * 10 * 4
    ts, vals = [], []
    with open(path, "rb") as f:
        f.readline()                               # magic line "DUE_V2"
        while True:
            d = f.read(rec)
            if len(d) < rec:
                break
            ns = struct.unpack("<Q", d[:8])[0]
            v = np.frombuffer(d[8:], dtype="<u4").reshape(16, 10).T   # [frame,sensor]
            ts.append(ns / 1e9)
            vals.append(v)
    if not ts:
        raise ValueError(f"DUE_V2 레코드 없음: {path}")
    t_burst = np.asarray(ts)
    raw = np.concatenate(vals).astype(np.float64)
    # burst 내 10프레임 시각을 선형 보간(200Hz, burst당 50ms)
    t = np.repeat(t_burst, 10) + np.tile(np.arange(10) * 0.005, len(t_burst))
    return t, raw


def _load_v2_session(data_dir: Path, *, hold_min_s: float = 0.3,
                     trim_frac: float = 0.25) -> list[tuple[np.ndarray, np.ndarray]]:
    """V2 취득 세션(due+em bin) → 압입(hold)별 (sensor_pct[F,16], pos_mm[2]).

    - z 정지 구간 중 z가 압입 깊이(최댓값 근처)인 세그먼트 = 압입 hold.
    - hold 앞뒤 trim_frac 씩 잘라 과도(진입/이탈) 제거.
    - baseline = 최초 비압입 대기 구간(z 하위) 채널 평균 → pct 변환.
    - 같은 (x,y) 반복 압입은 프레임을 합쳐 한 위치로(반복 노이즈 포함 측정).
    """
    due = sorted(data_dir.glob("due_v2_*.bin"))
    em = sorted(data_dir.glob("em_v2_*.bin"))
    if due and em:                                         # V2 포맷(신규 로거)
        t_s, raw = _read_due_v2(due[0])
        t_e, xyz = _read_em_v2(em[0])
        z_mm = xyz[:, 2] / 10000.0                         # 0.1μm → mm (EM 단위=0.1μm)
        xy_scale = 10000.0
    else:                                                  # V1 포맷(학습 취득과 동일)
        due1 = sorted(data_dir.glob("due_raw_burst_*.bin"))
        em1 = sorted(data_dir.glob("ethermotion_encoder_*.bin"))
        if not due1 or not em1:
            raise FileNotFoundError(f"due/em bin 없음(V2·V1 모두): {data_dir}")
        from sats.preprocessing.bin_merge import load_due_bin, load_ethermotion_bin
        d = load_due_bin(due1[0])
        e = load_ethermotion_bin(em1[0])
        t_s, raw = np.asarray(d.time_s, float), np.asarray(d.sensors, float)
        t_e = np.asarray(e.time_s, float)
        xyz = np.stack([e.x_mm, e.y_mm, e.z_mm], axis=1).astype(float)   # 이미 mm
        z_mm = xyz[:, 2]
        xy_scale = 1.0
    z_top = float(z_mm.max())
    # ★절대 허용치 판정(상대 분위수는 세션에 이동 구간(z=0 등)이 섞이면 오분류):
    #   압입 = z가 최댓값 −0.3mm 이내 정지 / baseline = 접촉영역보다 1.2mm↓ 정지
    z_press_lo = z_top - 0.3
    # 정지 세그먼트(위치 반올림 불변)
    key = np.round(np.stack([xyz[:, 0] / xy_scale, xyz[:, 1] / xy_scale, z_mm], 1), 2)
    change = np.any(np.diff(key, axis=0) != 0, axis=1)
    starts = np.concatenate([[0], np.where(change)[0] + 1])
    ends = np.concatenate([np.where(change)[0], [len(xyz) - 1]])
    # baseline 판정은 '정지 세그먼트 z의 최저 클러스터'(=safe_z) 기준 — 압입 깊이와 무관하게 강건
    seg_z = np.array([z_mm[s] for s in starts])
    z_idle_hi = float(seg_z[seg_z > 0.1].min()) + 0.3 if (seg_z > 0.1).any() else z_top - 1.2
    # baseline: 최초 비접촉 정지 구간(≥1s)
    base = None
    for s, e in zip(starts, ends):
        if z_mm[s] < z_idle_hi and t_e[e] - t_e[s] >= 1.0:
            m = (t_s >= t_e[s] + 0.2) & (t_s <= t_e[e] - 0.2)
            if m.sum() >= 50:
                base = raw[m].mean(0)
                break
    if base is None:
        raise ValueError("baseline(비접촉 대기 ≥1s) 구간 없음")
    # 압입 hold 수집 → (x,y)별 pct 프레임 합침
    groups: dict[tuple[float, float], list[np.ndarray]] = {}
    order: list[tuple[float, float]] = []
    for s, e in zip(starts, ends):
        dur = t_e[e] - t_e[s]
        if z_mm[s] < z_press_lo or dur < hold_min_s:
            continue
        trim = dur * trim_frac
        m = (t_s >= t_e[s] + trim) & (t_s <= t_e[e] - trim)
        if m.sum() < 12:
            continue
        pct = ((raw[m] / base[None, :] - 1.0) * 100.0).astype(np.float32)
        pos = (round(xyz[s, 0] / xy_scale, 5), round(xyz[s, 1] / xy_scale, 5))  # 포맷별 단위→mm
        if pos not in groups:
            groups[pos] = []
            order.append(pos)
        groups[pos].append(pct)
    if not groups:
        raise ValueError("압입 hold 구간 없음(z 왕복 확인)")
    out = [(np.concatenate(groups[p]), np.asarray(p, np.float32)) for p in order]
    n_press = sum(len(v) for v in groups.values())
    print(f"[v2] {len(out)} 위치, 압입 {n_press}회, baseline {base.mean():.0f} "
          f"(z 압입 {z_top:.1f} / 압입판정 ≥{z_press_lo:.1f} / 대기판정 <{z_idle_hi:.1f} mm)")
    return out


def _to_pct(sensor: np.ndarray, baseline: np.ndarray | None, is_raw: bool) -> np.ndarray:
    """raw면 baseline으로 pct(상대%) 변환, 이미 pct면 그대로."""
    if not is_raw:
        return sensor.astype(np.float32)
    if baseline is None:
        raise ValueError("--input raw 인데 baseline 없음")
    return ((sensor / baseline[None, :] - 1.0) * 100.0).astype(np.float32)


def _load_positions(args) -> list[tuple[np.ndarray, np.ndarray]]:
    """(sensor_pct[F,16], pos_mm[2]) 리스트로 정규화 로드."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    if args.v2_dir:
        return _load_v2_session(Path(args.v2_dir))
    if args.data_dir:
        files = sorted(glob.glob(str(Path(args.data_dir) / "*.npz")))
        if not files:
            raise FileNotFoundError(f"npz 없음: {args.data_dir}")
        for f in files:
            z = np.load(f)
            sensor = _to_pct(z["sensor"].astype(np.float32),
                             z["baseline"].astype(np.float32) if "baseline" in z else None,
                             args.input == "raw")
            pos = np.asarray(z["pos_mm"], np.float32).reshape(-1)[:2]
            out.append((sensor, pos))
    else:
        z = np.load(args.data)
        sensor = _to_pct(z["sensor"].astype(np.float32),
                         z["baseline"].astype(np.float32) if "baseline" in z else None,
                         args.input == "raw")
        pos = np.asarray(z["pos_mm"], np.float32).reshape(-1, 2)
        # 연속 동일 위치(반올림) 구간으로 묶음
        key = np.round(pos / args.group_round).astype(np.int64)
        start = 0
        for i in range(1, len(pos) + 1):
            if i == len(pos) or np.any(key[i] != key[start]):
                out.append((sensor[start:i], pos[start:i].mean(0)))
                start = i
    return out


def _predict_positions(engine, sensor_pct, subpixel):
    """윈도우 슬라이딩 → 각 윈도우 centroid(또는 argmax) 예측 [P,2]."""
    W = engine.window_size
    if len(sensor_pct) < W:
        return np.zeros((0, 2))
    preds = []
    for i in range(W - 1, len(sensor_pct)):
        pmap = engine.predict(sensor_pct[i - W + 1:i + 1])
        pk = detect_peaks(pmap, grid_min_mm=engine.grid_min_mm, grid_step_mm=engine.grid_step_mm,
                          max_peaks=1, subpixel=subpixel)
        if len(pk):
            preds.append(pk[0, :2])
    return np.asarray(preds, float).reshape(-1, 2)


def analyze(positions, engine):
    """상대변위 기반 분석 — 모터·센서 원점 불일치를 기준점 정렬로 소거.

    GT/예측 모두 첫 위치(기준 압입점) 대비 Δ로 비교. slope·R²·σ·분해능은
    원래 offset-불변(회귀 절편), rel err 는 기준점 정렬 후 상대 정확도.
    """
    gt = np.array([p for _, p in positions], float)                    # [K,2] 모터 절대
    # 스윕 방향 = GT 분산 주성분(PCA) — σ(반복 노이즈) 측정 축
    c = gt - gt.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    axis = vt[0] / (np.linalg.norm(vt[0]) + 1e-9)
    rows = []
    for sensor, pos in positions:
        pr = _predict_positions(engine, sensor, subpixel=True)
        if len(pr) == 0:
            continue
        mu = pr.mean(0)
        sig_axis = float(np.std(pr @ axis))                           # 스윕축 방향 반복 노이즈
        rows.append({"gt": pos, "mu": mu, "sig": sig_axis,
                     "gt_proj": float(pos @ axis), "pred_proj": float(mu @ axis), "n": len(pr)})
    # ★2D 유사변환 정합(스케일드 프로크루스테스, 반사 허용): pred ≈ s·R·gt + t
    #   모터축↔센서축은 원점·회전·반전이 모두 다를 수 있음(마운팅). 이를 적합해
    #   소거한 뒤의 잔차·스케일이 진짜 추종성/정확도.
    G = np.array([r["gt"] for r in rows]); P = np.array([r["mu"] for r in rows])
    Gc, Pc = G - G.mean(0), P - P.mean(0)
    U, S, Vt = np.linalg.svd(Gc.T @ Pc)
    R = (U @ Vt).T                                                    # 회전(+반사 허용)
    denom = float((Gc ** 2).sum())
    scale = float(S.sum() / max(denom, 1e-12))                        # 예측/모터 스케일(1=정확)
    fit = (scale * (R @ Gc.T)).T                                      # 정합된 GT
    resid2d = Pc - fit
    resid_rms = float(np.sqrt((resid2d ** 2).sum(1).mean()))
    rel_med = float(np.median(np.sqrt((resid2d ** 2).sum(1))))
    ss_tot = float((Pc ** 2).sum())
    r2 = 1.0 - float((resid2d ** 2).sum()) / max(ss_tot, 1e-12)
    rot_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    reflected = bool(np.linalg.det(R) < 0)
    sig_med = float(np.median([r["sig"] for r in rows]))
    res_2s = 2.0 * sig_med / max(abs(scale), 1e-6)
    # 그림용 1D 투영(정합 후)
    gp = (scale * (R @ Gc.T)).T @ axis
    pp = Pc @ axis
    for r, res in zip(rows, np.sqrt((resid2d ** 2).sum(1))):
        r["rel"] = float(res)
    return {"rows": rows, "axis": axis, "scale": scale, "rot_deg": rot_deg,
            "reflected": reflected, "r2": float(r2), "sig_med": sig_med,
            "rel_med": rel_med, "resid_rms": resid_rms,
            "resolution_2sigma": float(res_2s), "gp": np.asarray(gp), "pp": np.asarray(pp)}


def analyze_1d(positions, engine):
    """1D 스윕 전용 — 2D 프로크루스테스는 1D에서 회전이 부족결정이라 gain을 왜곡한다.

    모델: p(s) = p0 + s·v  (s=모터 스윕 좌표, v=예측공간 이동벡터)
      gain      = |v|  (모터 1mm당 예측 이동, 1=정확). 축 회전은 v 방향이 흡수.
      resid     = 직선에서의 이탈(계통 비선형)
      ★실효 분해능 = 인접 위치가 노이즈를 넘어 구분되는지 — 플래토(구분 불가 구간) 폭.
        두 위치 구분 기준: 예측 거리 > 2σ (σ=두 위치 반복 노이즈 평균).
    """
    S = np.array([float(p[0] if abs(p[0]) > abs(p[1]) else p[1]) for _, p in positions])
    P, SD = [], []
    for sensor, _ in positions:
        pr = _predict_positions(engine, sensor, subpixel=True)
        P.append(pr.mean(0))
        SD.append(float(np.hypot(pr[:, 0].std(), pr[:, 1].std())))
    P = np.asarray(P); SD = np.asarray(SD)
    o = np.argsort(S); S, P, SD = S[o], P[o], SD[o]
    ax, bx = np.polyfit(S, P[:, 0], 1)
    ay, by = np.polyfit(S, P[:, 1], 1)
    gain = float(np.hypot(ax, ay))
    resid = np.hypot(P[:, 0] - (ax * S + bx), P[:, 1] - (ay * S + by))
    # 인접 위치 구분 판정(플래토)
    d = np.hypot(*(P[1:] - P[:-1]).T)
    thr = (SD[1:] + SD[:-1])                       # = 2 × 평균 σ
    resolved = d > np.maximum(thr, 1e-6)
    widths, w = [], float(np.median(np.diff(S)))
    step = w
    for ok in resolved:
        if ok:
            widths.append(w); w = step
        else:
            w += step
    widths.append(w)
    return {"S": S, "P": P, "SD": SD, "gain": gain, "slope_xy": (float(ax), float(ay)),
            "resid": resid, "resid_rms": float(np.sqrt((resid ** 2).mean())),
            "sig_med": float(np.median(SD)), "step_mm": step,
            "resolved_frac": float(resolved.mean()),
            "plateau_med": float(np.median(widths)), "plateau_max": float(max(widths)),
            "n_distinct": int(len(widths))}


def make_figure_1d(r, out_png, diameter):
    """1D: (좌) 모터 vs 예측 이동거리 + σ, (우) 인접 스텝 이동 vs 노이즈 문턱."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S, P, SD = r["S"], r["P"], r["SD"]
    proj = (P - P[0]) @ (np.array(r["slope_xy"]) / max(r["gain"], 1e-9))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.5))
    a0.errorbar(S - S[0], proj, yerr=SD, fmt="o-", ms=4, lw=1, capsize=2, color="#0a7")
    lo, hi = 0, float(S[-1] - S[0])
    a0.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal gain 1")
    a0.set_xlabel("motor sweep (mm)"); a0.set_ylabel("predicted along sweep (mm)")
    a0.set_title(f"D{diameter:g}  gain={r['gain']:.2f}  residRMS={r['resid_rms']:.2f}mm")
    a0.legend(fontsize=8)
    d = np.hypot(*(P[1:] - P[:-1]).T)
    thr = SD[1:] + SD[:-1]
    mid = (S[1:] + S[:-1]) / 2 - S[0]
    a1.semilogy(mid, np.maximum(d, 1e-4), "o-", ms=4, color="#0a7", label="step move")
    a1.semilogy(mid, np.maximum(thr, 1e-4), "r--", lw=1, label="noise threshold (2 sigma)")
    a1.set_xlabel("motor sweep (mm)"); a1.set_ylabel("distance (mm, log)")
    a1.set_title(f"resolved {r['resolved_frac']*100:.0f}%  plateau med {r['plateau_med']:.2f}mm")
    a1.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def make_figure(res, out_png, diameter):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["rows"]
    gp, pp = res["gp"], res["pp"]
    sig = np.array([r["sig"] for r in rows])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.5))
    a0.errorbar(gp, pp, yerr=sig, fmt="o", ms=4, capsize=2, color="#0a7", label="predicted (centroid)")
    lo, hi = gp.min(), gp.max()
    a0.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal (slope 1)")
    a0.set_xlabel("aligned motor displacement (mm)"); a0.set_ylabel("predicted displacement (mm)")
    a0.set_title(f"D{diameter:g}  scale={res['scale']:.3f}  R2={res['r2']:.4f}"
                 f"  rot={res['rot_deg']:.1f}deg" + ("  reflected" if res["reflected"] else ""))
    a0.legend(fontsize=8)
    a1.plot(gp, pp - gp, "o-", ms=3, color="#c33")
    a1.axhline(0, color="k", lw=0.6)
    a1.set_xlabel("aligned motor displacement (mm)"); a1.set_ylabel("residual (mm)")
    a1.set_title(f"sigma_med={res['sig_med']:.3f}mm  resolution(2σ)={res['resolution_2sigma']:.3f}mm")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="미세 스텝 위치 분해능 분석(D10)")
    ap.add_argument("--data-dir", default=None, help="위치당 npz 디렉토리(형식 A)")
    ap.add_argument("--data", default=None, help="전 프레임 단일 npz(형식 B)")
    ap.add_argument("--v2-dir", default=None,
                    help="V2 취득 세션 디렉토리(due_v2_*.bin + em_v2_*.bin) — 압입 자동 추출")
    ap.add_argument("--run-dir", default=str(_ROOT / "sats/training/runs/ecomesh_v6_deploy_all4"))
    ap.add_argument("--diameter", type=float, default=10.0)
    ap.add_argument("--input", choices=["pct", "raw"], default="pct", help="sensor가 pct인지 raw인지")
    ap.add_argument("--group-round", type=float, default=0.05, help="[B] 위치 묶기 반올림(mm)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(_ROOT / "history/fig_data/experiments_archive/reeval/loc_resolution_d10.png"))
    args = ap.parse_args()
    if not args.data_dir and not args.data and not args.v2_dir:
        ap.error("--data-dir / --data / --v2-dir 중 하나 필요")

    engine = SATSInferenceEngine(args.run_dir, device=args.device, indenter_diameter_mm=args.diameter)
    positions = _load_positions(args)
    print(f"[분석] {len(positions)} 위치, window={engine.window_size}, D{args.diameter:g}")
    gt = np.array([p for _, p in positions], float)
    c = gt - gt.mean(0)
    sv = np.linalg.svd(c, compute_uv=False)
    is_1d = len(positions) >= 5 and sv[1] < 0.05 * sv[0]      # 2번째 주성분이 미미 = 1D 스윕
    if is_1d:
        r1 = analyze_1d(positions, engine)
        print("\n" + "=" * 62)
        print(f"  1D 스윕 위치 분해능 (D{args.diameter:g}, {len(positions)}점, "
              f"스텝 {r1['step_mm']:.2f}mm)")
        print("=" * 62)
        print(f"  추종 gain = {r1['gain']:.3f} mm/mm (1=정확)  "
              f"[dx/ds={r1['slope_xy'][0]:+.3f}, dy/ds={r1['slope_xy'][1]:+.3f}]")
        print(f"  직선 잔차 RMS = {r1['resid_rms']:.3f} mm (계통 비선형)")
        print(f"  반복 노이즈 sigma median = {r1['sig_med']:.4f} mm")
        print(f"  ★인접 {r1['step_mm']:.2f}mm 스텝 구분 성공률 = {r1['resolved_frac']*100:.0f}%"
              f"  (구분되는 위치 {r1['n_distinct']}/{len(positions)})")
        print(f"  ★실효 분해능(플래토 폭) median = {r1['plateau_med']:.2f} mm, "
              f"worst = {r1['plateau_max']:.2f} mm")
        print("=" * 62)
        make_figure_1d(r1, args.out, args.diameter)
        print(f"  그림: {args.out}\n")
        return
    res = analyze(positions, engine)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    make_figure(res, args.out, args.diameter)

    flip = "  (반사=축 반전 포함)" if res["reflected"] else ""
    print("\n" + "=" * 60)
    print(f"  위치 분해능 — 2D 정합(회전·반전 소거) 기준 (D{args.diameter:g}, {len(res['rows'])} 위치)")
    print("=" * 60)
    print(f"  추종성   scale = {res['scale']:.3f} (1=정확)   R2 = {res['r2']:.4f}")
    print(f"  축 정렬  rot = {res['rot_deg']:+.1f}°{flip}")
    print(f"  반복노이즈 sigma(축) median = {res['sig_med']:.3f} mm")
    print(f"  상대정확도 잔차 median = {res['rel_med']:.3f} mm  (RMS {res['resid_rms']:.3f})")
    print(f"  ★분해능  2σ/scale = {res['resolution_2sigma']:.3f} mm  "
          f"(이보다 가까운 두 위치는 노이즈에 묻힘)")
    print("=" * 60)
    print(f"  그림: {args.out}")
    print("  → SPEC.md §3 x/y resolution 을 이 수치로 갱신.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
