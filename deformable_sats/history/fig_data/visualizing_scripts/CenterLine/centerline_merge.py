#!/usr/bin/env python3
"""CenterLine 실험 CSV 통합 전처리.

각 테스트 폴더의 due_data.csv(센서) + ethermotion_data.csv(모션) +
loadcell_data.csv(하중)를 하나의 시계열 테이블로 병합한다.

출력 스키마:
    timestep, phase, material, depth, x, y, z, s1..s16, fz

핵심 처리:
  1) 단위 변환: ethermotion X/Y/Z/U 는 0.1µm 단위 -> mm (÷10000), z는 0.01mm 정렬.
  2) due burst 구조 복원: 1 burst = 10 frame 이 동일 timestamp 로 찍혀 있으나
     실제로는 ~9.84ms(약 100Hz) 간격의 별개 물리 시점이다.
     burst 간격을 frame 수로 나눠 frame 별 실제 시간을 재구성한다.
  3) phase 세그멘테이션 (ethermotion Z 거동 기반, U는 가상 보간축이라 무시):
        - loading : Z 단조 증가 구간
        - holding : 최고 Z 플래토 (Z >= Zmax - tol)
        - base    : 저 Z 안정 플래토
        - transition(제외) : Z 하강 / Y 이동 구간
  4) 집계:
        - loading : (y, 0.01mm z-bin) 별 평균  -> z 스텝마다 1행
        - base    : (y) 별 평균                -> 1행
        - holding : (y) 별 평균 (hysteresis 로 base 와 별도) -> 1행
     센서값은 raw counts 그대로. fz = loadcell kg × 9.80665 (N).

사용:
    python3 centerline_merge.py
    python3 centerline_merge.py --root <CenterLine 경로> --out <combined.csv>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── 상수 ────────────────────────────────────────────────────────────────
UM01_TO_MM = 1.0 / 10000.0       # 0.1µm -> mm
KG_TO_N = 9.80665                # loadcell kg -> N
Z_BIN_MM = 0.01                  # loading z 정렬 해상도
SKIN_COLS = [f"Skin{i}" for i in range(1, 17)]
OUT_COLS = (
    ["timestep", "phase", "material", "depth", "x", "y", "z"]
    + [f"s{i}" for i in range(1, 17)]
    + ["fz"]
)

# phase 검출 임계값 (ethermotion 원시 0.1µm 단위 기준)
Z_HIGH_TOL = 1500.0   # 최고 Z 로부터 0.15mm 이내 -> holding
Z_INC_THR = 30.0      # 인접 행 Z 증가량이 이 값 초과 -> loading
Y_STABLE_THR = 100.0  # 인접 행 Y 변화가 이 값 미만 -> 위치 안정(이동 아님)
BASE_PLATEAU_TOL = 2000.0  # 저 Z 플래토 레벨로부터 0.2mm 이내 -> base
UNLOADED_MARGIN = 25000.0  # base 레벨보다 2.5mm 이상 낮으면 접촉 전 무부하
LOADCELL_SENTINEL = 95.0  # loadcell 첫 마커 값


# ── 경로/메타 ───────────────────────────────────────────────────────────
def find_tests(root: Path) -> list[Path]:
    """due_data.csv 를 가진 테스트 폴더들을 반환."""
    return sorted({p.parent for p in root.glob("*/*/*/due_data.csv")})


def parse_meta(test_dir: Path, root: Path) -> tuple[str, str]:
    """테스트 폴더 경로에서 (material, depth) 추출.

    구조: <root>/<material>/<depth>/<testname>/
    """
    rel = test_dir.relative_to(root)
    material = rel.parts[0].replace(" ", "").lower()  # 'eco20 + mesh' -> 'eco20+mesh'
    depth = rel.parts[1]
    return material, depth


# ── ethermotion: phase 세그멘테이션 ─────────────────────────────────────
def label_phases(ether: pd.DataFrame) -> pd.DataFrame:
    """ethermotion 각 행에 phase 라벨과 위치 정보를 부여.

    반환 컬럼: time_s, x_mm, y_mm, z_mm, phase
    """
    e = ether.sort_values("time_s").reset_index(drop=True)
    z = e["Z"].to_numpy(dtype=float)
    y = e["Y"].to_numpy(dtype=float)

    dz = np.diff(z, prepend=z[0])
    dy = np.abs(np.diff(y, prepend=y[0]))
    z_high = z.max()

    # 저 Z 안정 플래토(base 레벨)를 데이터에서 추정: 최고 Z 가 아닌 안정 구간의 최빈 Z.
    stable = np.abs(dz) <= Z_INC_THR
    low_sel = stable & (z < z_high - Z_HIGH_TOL)
    zr = np.round(z[low_sel] / 1000.0) * 1000.0     # 0.1mm 양자화 후 최빈값
    vals, cnts = np.unique(zr, return_counts=True)
    z_low = float(vals[cnts.argmax()])

    # dz(증감)를 먼저 판정해야 loading 막판의 Z 증가 구간이 holding 으로 새지 않는다.
    phase = np.empty(len(e), dtype=object)
    for i in range(len(e)):
        if z[i] < z_low - UNLOADED_MARGIN:
            phase[i] = "unloaded"          # 접촉 전 무부하 (시작 approach / 끝 복귀)
        elif dy[i] >= Y_STABLE_THR:
            phase[i] = "transition"        # Y 이동 중
        elif dz[i] > Z_INC_THR:
            phase[i] = "loading"           # Z 증가 (max 직전까지 전부 loading)
        elif dz[i] < -Z_INC_THR:
            phase[i] = "transition"        # Z 하강
        elif z[i] >= z_high - Z_HIGH_TOL:
            phase[i] = "holding"           # 최고 Z 에서 고정(dz≈0)
        elif abs(z[i] - z_low) <= BASE_PLATEAU_TOL:
            phase[i] = "base"              # loading 직전 저 Z 플래토
        else:
            phase[i] = "transition"        # 그 외 저속 구간(approach/복귀 등) 제외

    out = pd.DataFrame(
        {
            "time_s": e["time_s"].to_numpy(),
            "x_mm": e["X"].to_numpy() * UM01_TO_MM,
            "y_mm": e["Y"].to_numpy() * UM01_TO_MM,
            "z_mm": e["Z"].to_numpy() * UM01_TO_MM,
            "phase": phase,
        }
    )
    return out


# ── due: frame 별 실제 시간 재구성 ──────────────────────────────────────
def reconstruct_frame_time(due: pd.DataFrame) -> pd.Series:
    """burst 구조를 펼쳐 frame 별 실제 물리 시간을 추정.

    burst b 의 frame 들을 [t_burst(b), t_burst(b+1)) 에 균등 배치.
    측정 갭(비정상적으로 큰 간격)은 전역 median frame 간격으로 클램프.
    """
    bt = due.groupby("burst_index")["time_s"].first()
    n_per = due.groupby("burst_index").size()
    dt_burst = (bt.shift(-1) - bt) / n_per          # burst 내 frame 간격
    med_dt = dt_burst[dt_burst > 0].median()
    # 갭/마지막 burst/이상치 -> median 으로 대체
    bad = ~((dt_burst > 0) & (dt_burst < med_dt * 5))
    dt_burst = dt_burst.mask(bad, med_dt)

    dt_map = due["burst_index"].map(dt_burst)
    bt_map = due["burst_index"].map(bt)
    return bt_map + due["frame_index"] * dt_map


# ── ether 시간축으로 매핑 ───────────────────────────────────────────────
def map_to_ether(t: np.ndarray, ph: pd.DataFrame) -> dict[str, np.ndarray]:
    """임의 시각 t 들에 ethermotion 의 phase/x/y/z 를 부여.

    x/y/z 는 선형 보간, phase 는 최근접 ether 행에서 가져온다.
    ether 시간 범위 밖 샘플은 phase='__out__' 로 표시.
    """
    et = ph["time_s"].to_numpy()
    x = np.interp(t, et, ph["x_mm"].to_numpy())
    y = np.interp(t, et, ph["y_mm"].to_numpy())
    z = np.interp(t, et, ph["z_mm"].to_numpy())

    idx = np.searchsorted(et, t)
    idx = np.clip(idx, 1, len(et) - 1)
    left = et[idx - 1]
    right = et[idx]
    nearest = np.where(np.abs(t - left) <= np.abs(t - right), idx - 1, idx)
    phase = ph["phase"].to_numpy()[nearest]

    out_of_range = (t < et[0]) | (t > et[-1])
    phase = phase.astype(object)
    phase[out_of_range] = "__out__"
    return {"x": x, "y": y, "z": z, "phase": phase}


# ── 그룹 키 ─────────────────────────────────────────────────────────────
def assign_group(df: pd.DataFrame) -> pd.DataFrame:
    """phase 별 집계 그룹 키(gkey)와 z-bin 을 부여.

    loading : (y, z-bin) 별, base/holding : (y) 별.
    """
    df = df.copy()
    # Y 는 0.1mm 스텝으로 이동하므로 같은 위치가 쪼개지지 않게 0.1mm 로 양자화
    y_key = df["y"].round(1)
    z_bin = (df["z"] / Z_BIN_MM).round() * Z_BIN_MM

    gkey = []
    zbin_col = []
    for ph, yk, zb in zip(df["phase"], y_key, z_bin):
        if ph == "loading":
            gkey.append(f"loading|{yk:.1f}|{zb:.2f}")
            zbin_col.append(zb)
        elif ph == "unloaded":
            gkey.append("unloaded")        # 위치 무관, 테스트당 1행
            zbin_col.append(np.nan)
        else:  # base / holding
            gkey.append(f"{ph}|{yk:.1f}")
            zbin_col.append(np.nan)
    df["gkey"] = gkey
    df["z_bin"] = zbin_col
    return df


# ── 테스트 1개 처리 ─────────────────────────────────────────────────────
def process_test(test_dir: Path, root: Path) -> pd.DataFrame:
    material, depth = parse_meta(test_dir, root)
    print(f"  [{material} {depth}] {test_dir.name}")

    ether = pd.read_csv(test_dir / "ethermotion_data.csv")
    ph = label_phases(ether)
    z_max_mm  = round(ph["z_mm"].max(), 2)                              # holding 레벨 (mm)
    z_low_mm  = round(ph.loc[ph["phase"] == "base", "z_mm"].mean(), 2) # base 플래토 레벨 (mm)
    z_max_depth = round(z_max_mm - z_low_mm, 2)                        # 실제 침투 깊이 범위 (mm)
    n = ph["phase"].value_counts().to_dict()
    print(f"    ether rows={len(ph)} phase={n} z_max={z_max_mm}mm z_low={z_low_mm}mm depth={z_max_depth}mm")

    # --- due: frame 펼치고 ether 에 매핑 ---
    due = pd.read_csv(test_dir / "due_data.csv")
    ft = reconstruct_frame_time(due).to_numpy()
    mp = map_to_ether(ft, ph)
    s = pd.DataFrame(
        {
            "t": ft,
            "x": mp["x"],
            "y": mp["y"],
            "z": mp["z"],
            "phase": mp["phase"],
        }
    )
    s[[f"s{i}" for i in range(1, 17)]] = due[SKIN_COLS].to_numpy()
    s = s[s["phase"].isin(["unloaded", "base", "loading", "holding"])].reset_index(drop=True)
    s = assign_group(s)

    # 센서 집계: 그룹별 평균 + 대표 시간/위치
    agg_cols = {f"s{i}": "mean" for i in range(1, 17)}
    agg_cols.update({"t": "mean", "x": "mean", "y": "mean", "z": "mean"})
    sg = s.groupby(["gkey", "phase"], sort=False).agg(agg_cols).reset_index()

    # --- loadcell: 동일 그룹 기준 fz 집계 ---
    lc_path = test_dir / "loadcell_data.csv"
    fz_map: dict[str, float] = {}
    lc = pd.read_csv(lc_path)
    lc = lc[lc["kg"] != LOADCELL_SENTINEL]
    if len(lc):
        lmp = map_to_ether(lc["time_s"].to_numpy(), ph)
        lf = pd.DataFrame(
            {"y": lmp["y"], "z": lmp["z"], "phase": lmp["phase"], "fz": lc["kg"].to_numpy() * KG_TO_N}
        )
        lf = lf[lf["phase"].isin(["unloaded", "base", "loading", "holding"])].reset_index(drop=True)
        if len(lf):
            lf = assign_group(lf)
            lg = lf.groupby("gkey", sort=False)["fz"].mean()
            fz_map = lg.to_dict()
    else:
        print("    loadcell 비어있음 -> fz=NaN")

    # --- 병합 ---
    sg["fz"] = sg["gkey"].map(fz_map).astype(float)
    sg["material"] = material
    sg["depth"] = depth
    sg["timestep"] = sg["t"]
    z_binned = np.where(
        sg["phase"] == "loading",
        (sg["z"] / Z_BIN_MM).round() * Z_BIN_MM,   # loading 은 bin 중심으로 정렬
        sg["z"].round(2),
    )
    # 기저면(base) 기준 침투 깊이: base≈0, holding≈z_max_depth
    sg["z"] = np.clip(z_binned - z_low_mm, 0.0, None).round(2)
    sg["x"] = sg["x"].round(3)
    sg["y"] = sg["y"].round(1)
    # 무부하 행은 측정 시작 무부하 기준이므로 맨 앞으로 정렬되게 timestep 을 최소시각으로
    if (sg["phase"] == "unloaded").any():
        t_min = s.loc[s["phase"] == "unloaded", "t"].min()
        sg.loc[sg["phase"] == "unloaded", "timestep"] = t_min

    # base fz offset 차감: y별 base 평균을 모든 행에서 제거
    base_fz_by_y = sg[sg["phase"] == "base"].groupby("y")["fz"].mean()
    fallback_offset = float(base_fz_by_y.mean()) if len(base_fz_by_y) else 0.0
    fz_offset = sg["y"].map(base_fz_by_y).fillna(fallback_offset)
    sg["fz"] = sg["fz"] - fz_offset

    # NaN fz 보간: phase·y 그룹별 z축 선형 보간 후 경계 외삽(ffill/bfill)
    for idx in sg.groupby(["phase", "y"]).groups.values():
        sub = sg.loc[idx].sort_values("z")
        sg.loc[sub.index, "fz"] = sub["fz"].interpolate(method="index").ffill().bfill().values

    # 센서값은 노이즈 제거 평균을 정수(raw counts)로 반올림
    skin_out = [f"s{i}" for i in range(1, 17)]
    sg[skin_out] = sg[skin_out].round().astype("Int64")
    sg["fz"] = sg["fz"].round(4)

    out = sg.sort_values("timestep").reset_index(drop=True)[OUT_COLS]
    vc = out["phase"].value_counts().to_dict()
    print(f"    -> rows={len(out)} {vc}")
    return out


# ── 메인 ────────────────────────────────────────────────────────────────
def main() -> None:
    here = Path(__file__).resolve().parent
    default_root = here.parent / "fig2_material_ablation" / "CenterLine"
    ap = argparse.ArgumentParser(description="CenterLine CSV 통합 전처리")
    ap.add_argument("--root", type=Path, default=default_root,
                    help="CenterLine 루트 폴더")
    ap.add_argument("--out", type=Path, default=None,
                    help="통합 combined.csv 경로 (기본: <root>/combined_centerline.csv)")
    args = ap.parse_args()

    root = args.root.resolve()
    out_path = args.out or (root / "combined_centerline.csv")

    tests = find_tests(root)
    if not tests:
        raise SystemExit(f"테스트 폴더를 찾지 못함: {root}/*/*/*/due_data.csv")
    print(f"테스트 {len(tests)}개 발견")

    combined = []
    for test_dir in tests:
        df = process_test(test_dir, root)
        # 테스트별 unified.csv 저장
        df.to_csv(test_dir / "unified.csv", index=False)
        combined.append(df)

    all_df = pd.concat(combined, ignore_index=True)
    all_df.to_csv(out_path, index=False)
    print(f"\n통합 저장: {out_path}  (총 {len(all_df)}행)")
    print(f"테스트별 unified.csv 도 각 폴더에 저장 완료")


if __name__ == "__main__":
    main()
