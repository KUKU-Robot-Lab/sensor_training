#!/usr/bin/env python3
"""Summarize comparable SATS training runs into a CSV."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from sats.training.config import SATSConfig, filter_trial_ids

SUMMARY_COLUMNS = [
    "material",
    "train_trials",
    "val_trials",
    "best_epoch",
    "best_val_rmse",
    "final_train_loss",
    "grid_step_mm",
    "gt_mode",
    "run_dir",
]


def _valid_config_fields() -> set[str]:
    return {field.name for field in dataclasses.fields(SATSConfig)}


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(run_dir: Path) -> tuple[dict[str, Any], SATSConfig]:
    raw = _load_json(run_dir / "config.json", {})
    if not isinstance(raw, dict):
        raw = {}
    filtered = {key: value for key, value in raw.items() if key in _valid_config_fields()}
    return raw, SATSConfig(**filtered)


def _history_rows(run_dir: Path) -> list[dict[str, Any]]:
    raw = _load_json(run_dir / "history.json", [])
    if isinstance(raw, dict):
        raw = raw.get("history", [])
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def _format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".12g")
    return str(value)


def _cell_list(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, (list, tuple)):
        return ";".join(str(value) for value in values)
    return str(values)


def _discover_trial_ids(cfg: SATSConfig) -> list[str]:
    index_path = Path(cfg.dataset_index_path)
    if index_path.exists():
        raw_index = _load_json(index_path, {})
        trials = raw_index.get("trials", []) if isinstance(raw_index, dict) else []
        return [item["trial_id"] for item in trials if isinstance(item, dict) and "trial_id" in item]

    root = Path(cfg.raw_dir)
    trial_ids: list[str] = []
    for merged in sorted(root.glob("**/*_merged.bin")):
        trial_ids.append(merged.stem.replace("_merged", ""))
    for merged in sorted(root.glob("**/*_merged.csv")):
        trial_id = merged.stem.replace("_merged", "")
        if trial_id not in trial_ids:
            trial_ids.append(trial_id)
    return trial_ids


def _infer_train_trials(raw_config: dict[str, Any], cfg: SATSConfig) -> list[str]:
    explicit = raw_config.get("train_trials")
    if isinstance(explicit, list):
        return [str(trial_id) for trial_id in explicit]

    try:
        trial_ids = _discover_trial_ids(cfg)
    except OSError:
        return []
    trial_ids = filter_trial_ids(
        trial_ids,
        include_materials=cfg.include_materials,
        exclude_diameters=cfg.exclude_diameters,
    )
    val_set = set(cfg.val_trials)
    return [trial_id for trial_id in trial_ids if trial_id not in val_set]


def _material_label(raw_config: dict[str, Any], cfg: SATSConfig) -> str:
    include_materials = raw_config.get("include_materials", cfg.include_materials)
    if include_materials:
        return _cell_list(include_materials)
    return str(raw_config.get("material", cfg.material))


def _best_history_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row for row in rows
        if isinstance(row.get("val_rmse"), (int, float)) and math.isfinite(float(row["val_rmse"]))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["val_rmse"]))


def summarize_run(run_dir: Path) -> dict[str, str]:
    raw_config, cfg = _load_config(run_dir)
    history = _history_rows(run_dir)
    best = _best_history_row(history)
    final = history[-1] if history else {}
    train_trials = _infer_train_trials(raw_config, cfg)

    return {
        "material": _material_label(raw_config, cfg),
        "train_trials": _cell_list(train_trials),
        "val_trials": _cell_list(cfg.val_trials),
        "best_epoch": _format_scalar(best.get("epoch") if best else None),
        "best_val_rmse": _format_scalar(best.get("val_rmse") if best else None),
        "final_train_loss": _format_scalar(final.get("train_loss")),
        "grid_step_mm": _format_scalar(cfg.grid_step_mm),
        "gt_mode": str(cfg.gt_mode),
        "run_dir": str(run_dir),
    }


def summarize_runs(run_dirs: list[Path]) -> list[dict[str, str]]:
    return [summarize_run(Path(run_dir)) for run_dir in run_dirs]


def write_summary_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Summarize SATS training runs into a comparable CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    rows = summarize_runs([Path(run_dir) for run_dir in args.run_dirs])
    out_path = Path(args.out)
    write_summary_csv(rows, out_path)
    print(f"summary saved: {out_path}")


if __name__ == "__main__":
    main()
