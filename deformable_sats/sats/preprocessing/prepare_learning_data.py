#!/usr/bin/env python3
"""Prepare SATS learning_data from the skin_ws raw BIN archive.

Canonical layout:

    learning_data/
      sensor_raw_bin/<material>/d<D>/z_<Z>mm/test<N>/*_merged.bin
      gt/*_targets.npy

The source archive under skin_ws/raw_data is read-only input. This script writes
only learning_data artifacts and optional preview CSV files for inspection.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from sats.preprocessing.bin_merge import find_bin_set, process_trial_dir
    from sats.preprocessing.merged_bin import export_merged_bin_csv
except ImportError:  # pragma: no cover - direct script execution fallback
    from bin_merge import find_bin_set, process_trial_dir  # type: ignore[no-redef]
    from merged_bin import export_merged_bin_csv  # type: ignore[no-redef]


D_DIR_RE = re.compile(r"^d(?P<diameter>\d+(?:\.\d+)?)$", re.IGNORECASE)
XY_DIR_RE = re.compile(r"^xy_(?P<step>\d+(?:\.\d+)?)mm$", re.IGNORECASE)
SOURCE_TEST_DIR_RE = re.compile(r"^(?:\d{8}_)?test\d+$", re.IGNORECASE)
SOURCE_TEST_SORT_RE = re.compile(r"^(?:(?P<date>\d{8})_)?test(?P<test_no>\d+)$", re.IGNORECASE)
DEFAULT_DEPTH_MAP = {"d5": 2.5, "d10": 3.5}
AUTO_MATERIAL = "auto"
ALL_SOURCE_MATERIALS = {"all", "*"}

# test 번호를 영구 고정하기 위한 registry. learning_root에 저장한다.
# 구조: {"<material>/<diameter_key>": {"<source 상대경로>": test_no}}
# 한 번 부여된 번호는 재사용하고, 새 source 폴더만 max+1로 append한다(append-only).
REGISTRY_FILENAME = "trial_registry.json"


def load_trial_registry(path: Path) -> dict:
    """learning_root의 trial_registry.json을 읽는다. 없거나 손상 시 빈 dict."""
    if not Path(path).exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_trial_registry(path: Path, registry: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False, sort_keys=True)


def _assign_test_no(group_map: dict, source_key: str) -> int:
    """source_key에 안정적인 test 번호를 부여한다.

    이미 등록된 source는 기존 번호를 그대로 쓰고, 새 source는 그룹 내 최대 번호+1을
    부여한다. 따라서 과거 날짜 폴더를 나중에 끼워넣어도 기존 번호는 절대 바뀌지 않는다.
    """
    if source_key in group_map:
        return int(group_map[source_key])
    next_no = max((int(v) for v in group_map.values()), default=0) + 1
    group_map[source_key] = next_no
    return next_no


@dataclass(frozen=True)
class PlannedTrial:
    source_dir: Path
    output_dir: Path
    material: str
    diameter_key: str
    diameter_mm: float
    depth_mm: float
    test_no: int
    source_material: str = ""
    xy_step_mm: float | None = None

    @property
    def trial_id(self) -> str:
        return (
            f"{self.material}_{self.diameter_key}_"
            f"z{_format_number(self.depth_mm)}_test{self.test_no}"
        )

    def trial_info(self) -> dict:
        info = {
            "trial_id": self.trial_id,
            "material": self.material,
            "indenter_diameter_mm": self.diameter_mm,
            "z_max_indentation_mm": self.depth_mm,
            "experiment_no": self.test_no,
            "source_trial_dir": str(self.source_dir),
        }
        if self.source_material:
            info["source_material"] = self.source_material
        if self.xy_step_mm is not None:
            info["xy_step_mm"] = self.xy_step_mm
        return info


def _format_number(value: float) -> str:
    return f"{value:g}"


def _material_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return key or "material"


def _xy_key(path: Path) -> str:
    match = XY_DIR_RE.match(path.name)
    if match is None:
        raise ValueError(f"xy directory must look like xy_1mm/xy_0.5mm, got {path.name!r}")
    step = _format_number(float(match.group("step"))).replace(".", "p")
    return f"xy{step}"


def _xy_step_mm(path: Path) -> float | None:
    match = XY_DIR_RE.match(path.name)
    return None if match is None else float(match.group("step"))


def _output_material_name(source_material: str, requested_material: str, xy_dir: Path | None) -> str:
    base = source_material if requested_material == AUTO_MATERIAL else requested_material
    key = _material_key(base)
    if xy_dir is None:
        return key
    return f"{key}_{_xy_key(xy_dir)}"


def _source_archive_root(source_root: Path) -> Path:
    sats_root = source_root / "sats"
    return sats_root if sats_root.exists() else source_root


def parse_depth_map(items: list[str] | None) -> dict[str, float]:
    if not items:
        return dict(DEFAULT_DEPTH_MAP)
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"depth-map item must be dN=Z, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip().lower()
        if not D_DIR_RE.match(key):
            raise ValueError(f"depth-map key must look like d5/d10, got {key!r}")
        out[key] = float(raw_value)
    return out


def parse_z_start_map(items: list[str] | None) -> dict[float, float] | None:
    """--z-start d10=13.0 → {10.0: 13.0}. 지정 시 레거시 z_start를 오버라이드한다.

    새 센서는 rig 기하가 달라 인덴터 접촉 시작 z가 바뀔 수 있다. 미지정(None)이면
    레거시 Z_START_BY_DIAMETER_MM을 그대로 쓴다(기존 데이터 동작 불변).
    """
    if not items:
        return None
    out: dict[float, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"z-start item must be dN=Z, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip().lower()
        match = D_DIR_RE.match(key)
        if not match:
            raise ValueError(f"z-start key must look like d5/d10, got {key!r}")
        out[float(match.group("diameter"))] = float(raw_value)
    return out


def resolve_source_material_dir(source_root: Path, source_material: str) -> Path:
    candidates = [
        source_root / source_material,
        source_root / "sats" / source_material,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"source material not found. Tried: {', '.join(str(c) for c in candidates)}"
    )


def _diameter_sort_key(path: Path) -> tuple[float, str]:
    match = D_DIR_RE.match(path.name)
    return (float(match.group("diameter")) if match else float("inf"), path.name)


def _source_trial_sort_key(path: Path) -> tuple[str, int, str]:
    match = SOURCE_TEST_SORT_RE.match(path.name)
    if match is None:
        return ("", 0, path.name)
    date = match.group("date") or ""
    return (date, int(match.group("test_no")), path.name)


def _source_layout_sort_key(path: Path) -> tuple[int, float, str]:
    d_match = D_DIR_RE.match(path.name)
    if d_match is not None:
        return (0, float(d_match.group("diameter")), path.name)
    xy_match = XY_DIR_RE.match(path.name)
    if xy_match is not None:
        return (1, float(xy_match.group("step")), path.name)
    return (2, float("inf"), path.name)


def _iter_source_material_dirs(source_root: Path, source_material: str) -> list[Path]:
    if source_material.lower() in ALL_SOURCE_MATERIALS:
        archive_root = _source_archive_root(source_root)
        if not archive_root.exists():
            raise FileNotFoundError(f"source archive root not found: {archive_root}")
        return sorted((p for p in archive_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    return [resolve_source_material_dir(source_root, source_material)]


def _append_planned_from_diameter_dir(
    *,
    d_dir: Path,
    source_root: Path,
    sensor_root: Path,
    output_material: str,
    source_material: str,
    xy_step_mm: float | None,
    depth_map: dict[str, float],
    registry: dict,
    planned: list[PlannedTrial],
    skipped: list[str],
) -> None:
    d_match = D_DIR_RE.match(d_dir.name)
    if d_match is None:
        return
    diameter_key = d_dir.name.lower()
    if diameter_key not in depth_map:
        skipped.append(f"{d_dir}: no depth-map entry")
        return
    depth_mm = float(depth_map[diameter_key])
    diameter_mm = float(d_match.group("diameter"))

    usable_source_dirs: list[Path] = []
    test_dirs = (
        p for p in d_dir.iterdir()
        if p.is_dir() and SOURCE_TEST_DIR_RE.match(p.name)
    )
    for test_dir in sorted(test_dirs, key=_source_trial_sort_key):
        try:
            find_bin_set(test_dir)
        except Exception as exc:
            skipped.append(f"{test_dir}: {exc}")
            continue
        usable_source_dirs.append(test_dir)

    group_key = f"{output_material}/{diameter_key}"
    group_map = registry.setdefault(group_key, {})
    for source_dir in usable_source_dirs:
        source_key = source_dir.relative_to(source_root).as_posix()
        test_no = _assign_test_no(group_map, source_key)
        output_dir = (
            sensor_root
            / output_material
            / diameter_key
            / f"z_{_format_number(depth_mm)}mm"
            / f"test{test_no}"
        )
        planned.append(
            PlannedTrial(
                source_dir=source_dir,
                output_dir=output_dir,
                material=output_material,
                diameter_key=diameter_key,
                diameter_mm=diameter_mm,
                depth_mm=depth_mm,
                test_no=test_no,
                source_material=source_material,
                xy_step_mm=xy_step_mm,
            )
        )


def discover_planned_trials(
    source_root: Path,
    learning_root: Path,
    *,
    source_material: str,
    material: str,
    depth_map: dict[str, float],
    registry: dict | None = None,
) -> tuple[list[PlannedTrial], list[str]]:
    """raw archive를 스캔해 PlannedTrial 목록을 만든다.

    ``registry``를 주면 test 번호를 영구 고정(append-only)한다. 호출자는 실행 후
    registry를 저장해야 다음 실행에서도 번호가 유지된다. None이면 매 호출 fresh
    번호(날짜 정렬 순서)를 부여한다(테스트/미리보기용).
    """
    if registry is None:
        registry = {}
    sensor_root = learning_root / "sensor_raw_bin"
    planned: list[PlannedTrial] = []
    skipped: list[str] = []

    for source_material_dir in _iter_source_material_dirs(source_root, source_material):
        source_material_name = source_material_dir.name
        for child in sorted((p for p in source_material_dir.iterdir() if p.is_dir()), key=_source_layout_sort_key):
            if D_DIR_RE.match(child.name):
                _append_planned_from_diameter_dir(
                    d_dir=child,
                    source_root=source_root,
                    sensor_root=sensor_root,
                    output_material=_output_material_name(source_material_name, material, None),
                    source_material=source_material_name,
                    xy_step_mm=None,
                    depth_map=depth_map,
                    registry=registry,
                    planned=planned,
                    skipped=skipped,
                )
                continue

            if not XY_DIR_RE.match(child.name):
                continue
            output_material = _output_material_name(source_material_name, material, child)
            for d_dir in sorted((p for p in child.iterdir() if p.is_dir()), key=_diameter_sort_key):
                _append_planned_from_diameter_dir(
                    d_dir=d_dir,
                    source_root=source_root,
                    sensor_root=sensor_root,
                    output_material=output_material,
                    source_material=source_material_name,
                    xy_step_mm=_xy_step_mm(child),
                    depth_map=depth_map,
                    registry=registry,
                    planned=planned,
                    skipped=skipped,
                )

    return planned, skipped


def run_merge_stage(
    planned: list[PlannedTrial],
    source_root: Path,
    *,
    target_hz: float,
    max_dt_ms: float,
    window_ms: float,
    window_agg: str,
    stable_xy_only: bool,
    baseline_fallback_sec: float,
    force_round_dp: int | None,
    preview_csv_rows: int,
    full_csv: bool,
    dry_run: bool,
    z_start_map: dict[float, float] | None = None,
) -> list[dict]:
    summaries: list[dict] = []
    for trial in planned:
        print(f"[merge] {trial.source_dir} -> {trial.output_dir / (trial.trial_id + '_merged.bin')}")
        if dry_run:
            continue
        summary = process_trial_dir(
            trial.source_dir,
            source_root,
            out_dir=trial.output_dir,
            trial_info_override=trial.trial_info(),
            target_hz=target_hz,
            max_dt_ms=max_dt_ms,
            window_ms=window_ms,
            window_agg=window_agg,
            stable_xy_only=stable_xy_only,
            baseline_fallback_sec=baseline_fallback_sec,
            force_round_dp=force_round_dp,
            export_csv="all" if full_csv else "none",
            z_start_map=z_start_map,
        )
        if preview_csv_rows > 0:
            preview_path = trial.output_dir / f"{trial.trial_id}_merged_preview.csv"
            export_merged_bin_csv(
                summary["merged_bin"],
                preview_path,
                u_zero_only=False,
                limit=preview_csv_rows,
            )
            summary["merged_preview_csv"] = str(preview_path)
        summaries.append(summary)
    return summaries


def run_gt_stage(
    learning_root: Path,
    *,
    z_s: float,
    patch_step: float,
    fz_mode: str,
    fz_min_abs: float,
    dry_run: bool,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    generate_gt_py = Path(__file__).resolve().with_name("generate_gt.py")
    raw_dir = learning_root / "sensor_raw_bin"
    out_dir = learning_root / "gt"
    cmd = [
        sys.executable,
        str(generate_gt_py),
        "--raw-dir",
        str(raw_dir),
        "--out-dir",
        str(out_dir),
        "--input-format",
        "bin",
        # u_mm은 node 내부 대기/가상 축이므로 필터 기준으로 쓰지 않는다.
        # dataset(use_u_zero_only=False)과 동일하게 전체 row를 GT로 생성한다.
        "--include-shear-u",
        "--z-s",
        str(z_s),
        "--patch-step",
        str(patch_step),
        "--fz-mode",
        fz_mode,
        "--fz-min-abs",
        str(fz_min_abs),
    ]
    print("[gt] " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=repo_root, check=True)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build learning_data/sensor_raw_bin and learning_data/gt from skin_ws raw BIN files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=repo_root / "skin_ws" / "raw_data")
    parser.add_argument("--source-material", default="all")
    parser.add_argument(
        "--material",
        default=AUTO_MATERIAL,
        help="Output material key. Use 'auto' to derive it from source material and xy resolution.",
    )
    parser.add_argument("--learning-root", type=Path, default=repo_root / "learning_data")
    parser.add_argument(
        "--depth-map",
        action="append",
        default=None,
        help="Diameter depth mapping, e.g. --depth-map d5=2.5 --depth-map d10=3.5",
    )
    parser.add_argument(
        "--z-start",
        action="append",
        default=None,
        help="Override indenter z_start(mm), e.g. --z-start d10=13.0. "
             "새 센서 rig에서 접촉 시작 z가 바뀔 때 사용. 미지정 시 레거시 맵 사용.",
    )
    parser.add_argument("--stage", choices=["all", "merge", "gt"], default="all")
    parser.add_argument("--target-hz", type=float, default=200.0)
    parser.add_argument("--max-dt-ms", type=float, default=10.0)
    parser.add_argument("--window-ms", type=float, default=10.0)
    parser.add_argument("--window-agg", choices=["mean", "median"], default="median")
    parser.add_argument("--no-stable-xy-filter", action="store_true")
    parser.add_argument("--baseline-fallback-sec", type=float, default=2.0)
    parser.add_argument("--force-round-dp", type=int, default=-1)
    parser.add_argument("--preview-csv-rows", type=int, default=10_000)
    parser.add_argument("--full-csv", action="store_true")
    parser.add_argument("--z-s", type=float, default=2.0)
    parser.add_argument("--patch-step", type=float, default=0.1)
    parser.add_argument("--fz-mode", choices=["positive_only", "abs", "signed"], default="abs")
    parser.add_argument("--fz-min-abs", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depth_map = parse_depth_map(args.depth_map)
    z_start_map = parse_z_start_map(args.z_start)

    registry_path = args.learning_root / REGISTRY_FILENAME
    registry = load_trial_registry(registry_path)
    planned, skipped = discover_planned_trials(
        args.source_root,
        args.learning_root,
        source_material=args.source_material,
        material=args.material,
        depth_map=depth_map,
        registry=registry,
    )
    print(f"planned trials: {len(planned)}")
    for trial in planned:
        print(f"  {trial.trial_id}: {trial.source_dir} -> {trial.output_dir}")
    if skipped:
        print("skipped:")
        for item in skipped:
            print(f"  {item}")

    # 부여한 번호를 영구 보존한다(미리보기 모드에서는 저장하지 않음).
    if not args.dry_run:
        save_trial_registry(registry_path, registry)
        print(f"trial registry: {registry_path}")

    if args.stage in {"all", "merge"}:
        run_merge_stage(
            planned,
            args.source_root,
            target_hz=args.target_hz,
            max_dt_ms=args.max_dt_ms,
            window_ms=args.window_ms,
            window_agg=args.window_agg,
            stable_xy_only=not args.no_stable_xy_filter,
            baseline_fallback_sec=args.baseline_fallback_sec,
            force_round_dp=args.force_round_dp if args.force_round_dp >= 0 else None,
            preview_csv_rows=args.preview_csv_rows,
            full_csv=args.full_csv,
            dry_run=args.dry_run,
            z_start_map=z_start_map,
        )
    if args.stage in {"all", "gt"}:
        run_gt_stage(
            args.learning_root,
            z_s=args.z_s,
            patch_step=args.patch_step,
            fz_mode=args.fz_mode,
            fz_min_abs=args.fz_min_abs,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
