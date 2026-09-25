#!/usr/bin/env python3
"""
sats_data 바이너리 파싱 + s5,s6,s9,s10 부호 일관성 검사 + Mahalanobis Distance 계산

지원 포맷:
  DUE_RAW_BURST_BIN_V1         (센서: 16채널 × 10 frame / burst)
  ETHERMOTION_ENCODER_BIN_V1   (위치: x,y,z,u 각도 0.0001mm 단위)

부호 일관성(consistent):
  loading / holding 구간에서 s5,s6,s9,s10의 Δsensor 부호가
  모두 같으면 True (원점 근처 4 센서가 같은 방향으로 반응하는지 확인).

MD (Mahalanobis Distance):
  [ds5, ds6, ds9, ds10] 4차원 공간에서 해당 구간 전체 평균(μ),
  공분산(Σ)을 기준으로 각 포인트까지의 마할라노비스 거리.

사용:
    python scripts/check_sats_md.py
    python scripts/check_sats_md.py --data fig2_material_ablation/sats_data/ecomesh/d5
    python scripts/check_sats_md.py --data ... --out result.csv --stride 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── 상수 ─────────────────────────────────────────────────────────────────────
SENSOR_COLS   = [f"s{i}" for i in range(1, 17)]
CHECK_SENSORS = ["s5", "s6", "s9", "s10"]
CHECK_DELTA   = ["ds5", "ds6", "ds9", "ds10"]

# phase 검출 파라미터 — z_range 비율 기반 (절대값 아님)
_HOLD_FRAC  = 0.05   # holding: z >= z_max - z_range * 0.05
_BASE_FRAC  = 0.08   # base:    z <= z_low  + z_range * 0.08
_UNLOD_FRAC = 0.50   # unloaded: z < z_low  - z_range * 0.50


# ── 바이너리 파싱 ─────────────────────────────────────────────────────────────

def _header_offset(path: Path) -> tuple[int, dict]:
    """헤더 2줄(포맷명 + JSON)을 읽고 (데이터 시작 오프셋, 메타) 반환."""
    with open(path, "rb") as f:
        f.readline()                         # 포맷명 줄
        meta = json.loads(f.readline())      # JSON 메타 줄
        return f.tell(), meta


def parse_due(path: Path) -> pd.DataFrame:
    """DUE_RAW_BURST_BIN_V1 → DataFrame(elapsed_ns, s1..s16).

    각 burst의 10 frame을 센서별로 평균하여 1행으로 집계.
    """
    offset, meta = _header_offset(path)
    n_sen  = meta["num_sensors"]    # 16
    n_frm  = meta["fifo_frames"]    # 10
    rbytes = meta["record_bytes"]   # 648  (= 8 + 16*10*4)

    with open(path, "rb") as f:
        f.seek(offset)
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    n_rec = len(raw) // rbytes
    raw   = raw[: n_rec * rbytes].reshape(n_rec, rbytes)

    # 레코드 앞 8 bytes = uint64 elapsed_ns
    elapsed = np.frombuffer(np.ascontiguousarray(raw[:, :8]).tobytes(), dtype="<u8")

    # 나머지 640 bytes = sensor[16][frame10] uint32 little-endian
    payload = (
        np.frombuffer(np.ascontiguousarray(raw[:, 8:]).tobytes(), dtype="<u4")
        .reshape(n_rec, n_sen, n_frm)
    )
    means = payload.mean(axis=2)   # (n_rec, 16)

    df = pd.DataFrame(means, columns=SENSOR_COLS)
    df.insert(0, "elapsed_ns", elapsed)
    t0, t1 = int(elapsed[0]), int(elapsed[-1])
    print(f"  DUE: {n_rec:,} bursts  elapsed[0]={t0:,}  elapsed[-1]={t1:,}")
    print(f"       hex[0]={t0:#018x}  hex[-1]={t1:#018x}")
    return df


def parse_ethermotion(path: Path, stride: int = 10) -> pd.DataFrame:
    """ETHERMOTION_ENCODER_BIN_V1 → DataFrame(elapsed_ns, x,y,z mm).

    memmap 대신 struct.unpack 기반 청크 읽기 사용 (대용량 파일 안전).
    stride 레코드 간격으로 샘플링.
    """
    import struct as _struct

    offset, meta = _header_offset(path)
    rbytes  = meta["record_bytes"]   # 56
    FMT     = "<Qddddiiii"           # elapsed_ns, x,y,z,u (f64), x,y,z,u_lcmd (i32)
    assert _struct.calcsize(FMT) == rbytes, (
        f"struct 크기 불일치: {_struct.calcsize(FMT)} vs {rbytes}"
    )

    stride_bytes = rbytes * stride          # 560 bytes per stride group
    CHUNK        = stride_bytes * 10_000    # ~5.6 MB per read

    # ── 진단: 첫 5 레코드 출력 ──────────────────────────────────────────────
    with open(path, "rb") as f:
        f.seek(offset)
        for i in range(5):
            raw = f.read(rbytes)
            if len(raw) < rbytes:
                break
            v = _struct.unpack(FMT, raw)
            print(f"  [진단 {i}] t={v[0]:,}  "
                  f"x_raw={v[1]:.2f}  y_raw={v[2]:.2f}  z_raw={v[3]:.2f}")

    # ── 메인 파싱 ─────────────────────────────────────────────────────────────
    out_t, out_x, out_y, out_z = [], [], [], []

    with open(path, "rb") as f:
        f.seek(offset)
        while True:
            chunk = f.read(CHUNK)
            if not chunk or len(chunk) < rbytes:
                break
            pos = 0
            while pos + rbytes <= len(chunk):
                v = _struct.unpack_from(FMT, chunk, pos)
                out_t.append(v[0])
                out_x.append(v[1] * 1e-4)
                out_y.append(v[2] * 1e-4)
                out_z.append(v[3] * 1e-4)
                pos += stride_bytes   # stride 레코드만큼 건너뜀

    df = pd.DataFrame({"elapsed_ns": out_t, "x": out_x, "y": out_y, "z": out_z})
    print(f"  Ethermotion: stride={stride} → {len(df):,} records")
    return df


# ── Phase 라벨링 ──────────────────────────────────────────────────────────────

def label_phases(ether: pd.DataFrame) -> pd.DataFrame:
    """ethermotion z 거동 기반으로 phase 컬럼 추가.

    고정 임계값 대신 z_range 비율 기반으로 적응형 검출.
    스무딩 gradient로 loading/unloading 방향을 판별.
    """
    e = ether.sort_values("elapsed_ns").reset_index(drop=True)
    z = e["z"].to_numpy()
    y = e["y"].to_numpy()

    z_min, z_max = z.min(), z.max()
    z_range = z_max - z_min
    print(f"  z: {z_min:.4f} ~ {z_max:.4f} mm  (range = {z_range:.4f} mm)")
    print(f"  y: {y.min():.4f} ~ {y.max():.4f} mm")

    if z_range < 1e-3:
        print("  [경고] z 이동 범위 < 0.001 mm → 전체를 base로 처리")
        e["phase"]   = "base"
        e["z_depth"] = 0.0
        return e

    # 적응형 임계값 (z_range 비율)
    thr_hold = z_max - z_range * _HOLD_FRAC    # holding 하한
    thr_base = z_min + z_range * _BASE_FRAC    # base 상한
    thr_unld = z_min - z_range * _UNLOD_FRAC   # unloaded 상한

    # 노이즈 제거용 rolling median (레코드 수의 0.1%, 최소 5)
    win = max(5, len(z) // 1000)
    z_s = pd.Series(z).rolling(win, center=True, min_periods=1).median().to_numpy()
    dz_s = np.gradient(z_s)

    # loading 임계값: 양의 gradient 중앙값 × 0.3 (노이즈 수준 적응)
    pos_dz = dz_s[dz_s > 0]
    inc_thr = float(np.median(pos_dz) * 0.3) if len(pos_dz) > 10 else 0.0
    print(f"  loading dz 임계값(적응형): {inc_thr:.6f} mm")

    # y 이동 임계값 (y 이동 실험인 경우)
    dy_s = np.abs(np.gradient(y))
    y_range = y.max() - y.min()
    if y_range > 0.5:
        y_thr = float(np.percentile(dy_s[dy_s > 0], 80)) if (dy_s > 0).any() else 1e9
    else:
        y_thr = 1e9   # y 고정 실험 → y 이동 판정 비활성

    # 역우선순위 적용 (나중에 적용될수록 우선순위 높음)
    ph = np.full(len(e), "transition", dtype=object)
    ph[z_s <= thr_base]                                           = "base"
    ph[z_s >= thr_hold]                                           = "holding"
    ph[(dz_s < -inc_thr) & (z_s < thr_hold)]                    = "transition"   # 하강
    ph[(dz_s > inc_thr) & (z_s > thr_base) & (z_s < thr_hold)] = "loading"
    ph[dy_s >= y_thr]                                            = "transition"   # y 이동
    ph[z_s < thr_unld]                                           = "unloaded"

    e["phase"]   = ph
    e["z_depth"] = np.clip(z_s - z_min, 0.0, None).round(3)
    return e


# ── 시간 동기화 ───────────────────────────────────────────────────────────────

def sync_to_ether(due: pd.DataFrame, eth: pd.DataFrame) -> pd.DataFrame:
    """DUE elapsed_ns 기준으로 ethermotion x,y,z,phase 보간."""
    td = due["elapsed_ns"].to_numpy(float)
    te = eth["elapsed_ns"].to_numpy(float)

    # 시간 범위 겹침 확인
    overlap_pct = 100 * ((td <= te[-1]) & (td >= te[0])).mean()
    print(f"  DUE-Ethermotion 시간 겹침: {overlap_pct:.1f}%  "
          f"(DUE {td[0]/1e9:.1f}~{td[-1]/1e9:.1f}s  "
          f"Ether {te[0]/1e9:.1f}~{te[-1]/1e9:.1f}s)")

    out = due.copy()
    for col in ("x", "y", "z", "z_depth"):
        out[col] = np.interp(td, te, eth[col].to_numpy())

    # phase: 최근접 ethermotion 행
    idx     = np.clip(np.searchsorted(te, td), 1, len(te) - 1)
    nearest = np.where(np.abs(td - te[idx - 1]) <= np.abs(td - te[idx]), idx - 1, idx)
    out["phase"] = eth["phase"].to_numpy()[nearest]
    return out


# ── Baseline + Delta 계산 ─────────────────────────────────────────────────────

def compute_delta(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """base phase 평균을 baseline으로 삼아 Δsensor 컬럼 추가."""
    base = df[df["phase"] == "base"][SENSOR_COLS]
    if base.empty:
        print("  [경고] base phase 없음 → 전체 평균으로 baseline 대체")
        baseline = df[SENSOR_COLS].mean()
    else:
        baseline = base.mean()

    out = df.copy()
    for s in SENSOR_COLS:
        out[f"d{s}"] = df[s] - baseline[s]
    return out, baseline


# ── 일관성 검사 + MD 계산 ─────────────────────────────────────────────────────

def analyse(df: pd.DataFrame) -> pd.DataFrame:
    """loading/holding 구간에서 부호 일관성 + Mahalanobis Distance 계산."""
    active = df[df["phase"].isin(["loading", "holding"])].copy().reset_index(drop=True)
    if active.empty:
        print("  [경고] loading/holding 데이터 없음")
        return active

    # 부호 일관성: CHECK_SENSORS 4개가 모두 같은 부호인가
    signs = np.sign(active[CHECK_DELTA].to_numpy(float))
    active["consistent"] = np.all(signs == signs[:, :1], axis=1)
    active["n_agree"]    = (signs == signs[:, :1]).sum(axis=1)

    # Mahalanobis Distance (CHECK_DELTA 4차원)
    data = active[CHECK_DELTA].to_numpy(float)
    mu   = data.mean(axis=0)
    cov  = np.cov(data.T)
    try:
        Vi = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print("  [경고] 공분산 행렬 singular → 유사역행렬(pinv) 사용")
        Vi = np.linalg.pinv(cov)

    diff         = data - mu
    active["md"] = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", diff, Vi, diff)))

    return active


# ── 보고서 출력 ───────────────────────────────────────────────────────────────

def print_report(active: pd.DataFrame, baseline: pd.Series) -> None:
    if active.empty or "consistent" not in active.columns:
        print("\n[오류] loading/holding 데이터가 없습니다.")
        print("  → Phase 검출 실패: ethermotion z 범위를 확인하세요.")
        return
    n  = len(active)
    nc = int(active["consistent"].sum())

    print(f"\n{'='*65}")
    print(f" Baseline  s5={baseline['s5']:.0f}  s6={baseline['s6']:.0f}"
          f"  s9={baseline['s9']:.0f}  s10={baseline['s10']:.0f}")
    print(f"{'='*65}")
    print(f" s5,s6,s9,s10  부호 일관성  (loading+holding, n={n:,})")
    print(f"   일치(consistent)  : {nc:,}  ({100*nc/n:.1f}%)")
    print(f"   불일치            : {n-nc:,}  ({100*(n-nc)/n:.1f}%)")

    for label, mask in [("일치", active["consistent"]), ("불일치", ~active["consistent"])]:
        sub = active.loc[mask, "md"]
        if len(sub):
            print(f"\n [MD 통계 — {label}]  n={len(sub):,}")
            print(f"   mean={sub.mean():.4f}  median={sub.median():.4f}"
                  f"  min={sub.min():.4f}  max={sub.max():.4f}")

    print(f"\n{'='*65}")
    print(f" z_depth 구간별 요약")
    print(f"{'='*65}")
    cp       = active.copy()
    cp["zb"] = (cp["z_depth"] / 0.1).round() * 0.1
    hdr = (f"  {'z(mm)':>6}  {'n':>5}  {'일관(%)':>8}  "
           f"{'MD':>7}  {'ds5':>8}  {'ds6':>8}  {'ds9':>8}  {'ds10':>8}")
    print(hdr)
    for zb, g in cp.groupby("zb"):
        pct = 100 * g["consistent"].sum() / len(g)
        md  = g["md"].mean()
        dm  = [g[c].mean() for c in CHECK_DELTA]
        print(f"  {zb:6.2f}  {len(g):5d}  {pct:8.1f}  "
              f"{md:7.4f}  {dm[0]:8.1f}  {dm[1]:8.1f}  {dm[2]:8.1f}  {dm[3]:8.1f}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    here    = Path(__file__).resolve().parent
    default = here.parent / "fig2_material_ablation" / "sats_data" / "ecomesh" / "d5"

    ap = argparse.ArgumentParser(description="sats_data MD 검사 (s5,s6,s9,s10)")
    ap.add_argument("--data",   type=Path, default=default,
                    help="sats_data 폴더 (bin 파일 포함)")
    ap.add_argument("--out",    type=Path, default=None,
                    help="결과 CSV 저장 경로 (기본: <data>/md_check.csv)")
    ap.add_argument("--stride", type=int, default=10,
                    help="ethermotion 다운샘플 간격 (기본 10, 대용량 파일 대응)")
    args = ap.parse_args()

    d = args.data.resolve()
    print(f"데이터 폴더: {d}\n")

    due_f   = next(d.glob("due_raw_burst_*.bin"),        None)
    ether_f = next(d.glob("ethermotion_encoder_*.bin"),  None)
    if not due_f or not ether_f:
        raise FileNotFoundError(
            f"필요한 bin 파일을 찾지 못했습니다.\n"
            f"  due    : {due_f}\n"
            f"  ether  : {ether_f}"
        )

    print("[1] 바이너리 파싱...")
    due   = parse_due(due_f)
    ether = parse_ethermotion(ether_f, stride=args.stride)

    print("\n[2] Phase 라벨링...")
    ether = label_phases(ether)
    print(f"  {ether['phase'].value_counts().to_dict()}")

    print("\n[3] 시간 동기화 (DUE ↔ Ethermotion)...")
    synced = sync_to_ether(due, ether)
    print(f"  DUE phase 분포: {synced['phase'].value_counts().to_dict()}")

    print("\n[4] Baseline + Δsensor 계산...")
    synced, baseline = compute_delta(synced)
    print(f"  Baseline  s5={baseline['s5']:.0f}  s6={baseline['s6']:.0f}"
          f"  s9={baseline['s9']:.0f}  s10={baseline['s10']:.0f}")

    print("\n[5] 일관성 + MD 분석...")
    active = analyse(synced)

    print_report(active, baseline)

    out = args.out or (d / "md_check.csv")
    if active.empty or "consistent" not in active.columns:
        print(f"\nCSV 저장 건너뜀 (데이터 없음)")
        return
    save_cols = (
        ["elapsed_ns", "phase", "x", "y", "z_depth"]
        + SENSOR_COLS
        + CHECK_DELTA
        + ["consistent", "n_agree", "md"]
    )
    active[save_cols].to_csv(out, index=False, float_format="%.2f")
    print(f"\n결과 저장: {out}")
    print("  컬럼: phase, z_depth, s1..s16, ds5/ds6/ds9/ds10, consistent, n_agree, md")


if __name__ == "__main__":
    main()
