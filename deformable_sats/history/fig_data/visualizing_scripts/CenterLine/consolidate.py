#!/usr/bin/env python3
"""모든 unified.csv를 하나의 통합 CSV로 정리.

출력 컬럼: ecomesh, depth, phase, x, y, z, s1..s16, fz
  - ecomesh : 재료명 (eco20, eco50, eco20 + mesh …)
  - depth   : 압입 깊이 레벨 (d5, d10)
  - phase   : base / loading / holding  (unloaded 제외)
  - x, y, z : 위치 및 침투 깊이 (mm)
  - s1..s16 : 스킨 센서 raw counts
  - fz      : 하중 (N)

사용:
    python scripts/consolidate.py
    python scripts/consolidate.py --root <CenterLine 경로> --out <출력 파일>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ── 상수 ────────────────────────────────────────────────────────────────────
ROOT_DEFAULT = Path(__file__).resolve().parent.parent / "fig2_material_ablation" / "CenterLine"
OUT_DEFAULT  = Path(__file__).resolve().parent.parent / "fig2_material_ablation" / "consolidated.csv"

# loading phase z 간격 (mm) — 변경 가능
LOADING_Z_STEP_MM = 0.5

# 허용 x 위치 목록 (mm) — None 이면 base phase 에서 자동 감지
# 예: X_POSITIONS = [-0.7, -0.4, 0.23, 0.35, 0.4, 0.45]
X_POSITIONS: list[float] | None = None

# y 추출 간격 (mm) — 이 간격의 배수인 y값만 남김
# 예: Y_STEP_MM = 5.0  →  -10, -5, 0, 5, 10 만 추출
#     Y_STEP_MM = 0.1  →  0.1mm 단위 전체 추출
Y_STEP_MM = 5.0

PHASES_KEEP = {"base", "loading", "holding"}

OUT_COLS = (
    ["ecomesh", "depth", "phase", "x", "y", "z"]
    + [f"s{i}" for i in range(1, 17)]
    + ["fz"]
)

PHASE_ORDER = {"base": 0, "loading": 1, "holding": 2}


def collect_all(root: Path) -> pd.DataFrame:
    """root 아래 모든 unified.csv 를 찾아 하나로 합친다."""
    frames: list[pd.DataFrame] = []

    for csv_path in sorted(root.rglob("unified.csv")):
        # 경로에서 material / depth 추출
        # 예: .../CenterLine/eco20/d10/20260529_test2/unified.csv
        parts = csv_path.relative_to(root).parts   # ('eco20', 'd10', '20260529_…', 'unified.csv')
        if len(parts) < 3:
            print(f"  [skip] 경로 구조 인식 불가: {csv_path}")
            continue

        material = parts[0]   # eco20 / eco50 / eco20 + mesh …
        depth    = parts[1]   # d5 / d10

        df = pd.read_csv(csv_path)

        # unloaded 제외
        df = df[df["phase"].isin(PHASES_KEEP)].copy()

        # ecomesh 컬럼 추가
        df["ecomesh"] = material

        frames.append(df[["ecomesh", "depth", "phase", "x", "y", "z"]
                         + [f"s{i}" for i in range(1, 17)]
                         + ["fz"]])

        print(f"  loaded  {material}/{depth}  rows={len(df):,}")

    if not frames:
        raise FileNotFoundError(f"unified.csv 파일을 찾지 못했습니다: {root}")

    combined = pd.concat(frames, ignore_index=True)
    return combined


def filter_loading(df: pd.DataFrame) -> pd.DataFrame:
    """loading phase를 LOADING_Z_STEP_MM 간격 z값만 남긴다.
    각 grid point마다 가장 가까운 행 1개만 선택하고,
    (ecomesh, depth, x, y) 그룹의 최대 z(= holding 레벨)는 항상 포함."""
    loading = df[df["phase"] == "loading"].copy()
    rest    = df[df["phase"] != "loading"]

    if loading.empty:
        return df

    def _filter_group(grp):
        z_max = grp["z"].max()
        g = grp.copy()
        # 각 행을 가장 가까운 grid point에 할당
        g["_grid"] = (g["z"] / LOADING_Z_STEP_MM).round() * LOADING_Z_STEP_MM
        g["_dist"] = (g["z"] - g["_grid"]).abs()
        # grid point별 가장 가까운 행 1개
        on_grid = g.loc[g.groupby("_grid")["_dist"].idxmin()]
        # z_max 행 (이미 포함된 경우 중복 제거)
        at_max  = g[(g["z"] - z_max).abs() <= 0.005]
        return pd.concat([on_grid, at_max]).drop_duplicates().drop(columns=["_grid", "_dist"])

    parts: list[pd.DataFrame] = []
    for idx in loading.groupby(["ecomesh", "depth", "x", "y"]).groups.values():
        parts.append(_filter_group(loading.loc[idx]))
    filtered = pd.concat(parts, ignore_index=True) if parts else loading.iloc[:0]
    return pd.concat([rest, filtered], ignore_index=True)


def filter_x(df: pd.DataFrame) -> pd.DataFrame:
    """X_POSITIONS 에 정확히 일치하는 x값만 남긴다.
    X_POSITIONS 가 None 이면 base phase x값을 기준으로 자동 감지."""
    targets = X_POSITIONS
    if targets is None:
        targets = sorted(df[df["phase"] == "base"]["x"].unique().tolist())
        print(f"  x 자동 감지: {targets}")

    target_set = set(round(v, 6) for v in targets)
    mask = df["x"].round(6).isin(target_set)
    removed = (~mask).sum()
    if removed:
        print(f"  x 필터: {removed:,}행 제거 (drift 값)")
    return df[mask].reset_index(drop=True)


def filter_y(df: pd.DataFrame) -> pd.DataFrame:
    """Y_STEP_MM 간격의 y값만 남긴다. grid point마다 가장 가까운 행 1개 선택."""
    tol = Y_STEP_MM / 20
    on_grid = ((df["y"] / Y_STEP_MM).round() * Y_STEP_MM - df["y"]).abs() <= tol
    filtered = df[on_grid].copy()
    # grid point로 y 스냅
    filtered["y"] = ((filtered["y"] / Y_STEP_MM).round() * Y_STEP_MM).round(6) + 0.0
    removed = len(df) - len(filtered)
    if removed:
        print(f"  y 필터: {removed:,}행 제거 (Y_STEP_MM={Y_STEP_MM}mm 기준)")
    return filtered.reset_index(drop=True)


def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """ecomesh → depth → phase(base/loading/holding 순) → x → y → z 정렬."""
    df = df.copy()
    df["_phase_order"] = df["phase"].map(PHASE_ORDER).fillna(99)
    df = df.sort_values(
        ["ecomesh", "depth", "x", "y", "_phase_order", "z"],
        kind="stable",
    ).drop(columns=["_phase_order"])
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="unified.csv 통합")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT,
                        help="CenterLine 루트 경로")
    parser.add_argument("--out",  type=Path, default=OUT_DEFAULT,
                        help="출력 CSV 경로")
    args = parser.parse_args()

    print(f"root : {args.root}")
    print(f"out  : {args.out}")
    print()

    combined = collect_all(args.root)
    combined = filter_loading(combined)
    combined = filter_x(combined)
    combined = filter_y(combined)
    combined = sort_df(combined)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)

    # ── 요약 출력 ────────────────────────────────────────────────────────────
    print()
    print(f"총 행수  : {len(combined):,}")
    print(f"NaN fz   : {combined['fz'].isna().sum():,}")
    print()
    print("ecomesh × depth × phase 분포:")
    summary = (
        combined.groupby(["ecomesh", "depth", "phase"], sort=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    summary["_po"] = summary["phase"].map(PHASE_ORDER).fillna(99)
    summary = summary.sort_values(["ecomesh", "depth", "_po"]).drop(columns=["_po"])
    print(summary.to_string(index=False))
    print()
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
