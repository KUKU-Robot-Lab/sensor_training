from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineRow:
    group: str
    run_name: str
    source: str
    split_policy: str
    table_role: str
    condition_policy: str
    x_mae: float | None = None
    y_mae: float | None = None
    z_mae: float | None = None
    fz_mae: float | None = None
    note: str = ""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _mean_metric(metrics: dict[str, Any], index: int) -> float | None:
    mae = metrics.get("mae")
    if not isinstance(mae, dict):
        return None
    mean = mae.get("mean")
    if not isinstance(mean, list) or len(mean) <= index:
        return None
    return float(mean[index])


def load_current_comparison_rows(path: Path) -> list[BaselineRow]:
    payload = _load_json(path)
    model_metrics = payload["multi_head_field"]
    return [
        BaselineRow(
            group="current_xyzf",
            run_name="multi_head_field_stage3_cv5",
            source=str(path),
            split_policy="cv5_val_only",
            table_role="official",
            condition_policy="direct_model_output",
            x_mae=_mean_metric(model_metrics, 0),
            y_mae=_mean_metric(model_metrics, 1),
            z_mae=_mean_metric(model_metrics, 2),
            fz_mae=_mean_metric(model_metrics, 3),
            note="trial-aware 5-fold CV aggregate",
        )
    ]


def load_current_zfz_rows(path: Path) -> list[BaselineRow]:
    payload = _load_json(path)
    gt_summary = {"mae": {"mean": [0.0, 0.0]}}
    folds = payload.get("per_fold", [])
    if folds:
        gt_z = [float(fold["gt_xy"]["mae"][0]) for fold in folds]
        gt_fz = [float(fold["gt_xy"]["mae"][1]) for fold in folds]
        gt_summary = {"mae": {"mean": [sum(gt_z) / len(gt_z), sum(gt_fz) / len(gt_fz)]}}

    return [
        BaselineRow(
            group="current_zfz",
            run_name="z_fz_regressor_gt_xy",
            source=str(path),
            split_policy="cv5_val_only",
            table_role="reference_only",
            condition_policy="gt_xy+gt_radius",
            z_mae=_mean_metric(gt_summary, 0),
            fz_mae=_mean_metric(gt_summary, 1),
            note="upper-bound reference; not deployment condition",
        ),
        BaselineRow(
            group="current_zfz",
            run_name="z_fz_regressor_predicted_xy",
            source=str(path),
            split_policy="cv5_val_only",
            table_role="separated_upper_bound",
            condition_policy="pred_xy+gt_radius",
            z_mae=_mean_metric(payload["predicted_xy_summary"], 0),
            fz_mae=_mean_metric(payload["predicted_xy_summary"], 1),
            note="radius still GT; keep separate from main score table",
        ),
    ]


def load_legacy_0409_rows(base_dir: Path) -> list[BaselineRow]:
    rows: list[BaselineRow] = []
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        result_path = subdir / "comparison_results.json"
        if not result_path.exists():
            continue
        payload = _load_json(result_path)
        for model_name, metrics in sorted(payload.items()):
            mae = metrics.get("mae", [])
            rows.append(
                BaselineRow(
                    group="legacy_0409",
                    run_name=f"{subdir.name}_{model_name}",
                    source=str(result_path),
                    split_policy="aggregate_legacy",
                    table_role="official",
                    condition_policy="legacy_direct_regression",
                    x_mae=float(mae[0]) if len(mae) > 0 else None,
                    y_mae=float(mae[1]) if len(mae) > 1 else None,
                    z_mae=float(mae[2]) if len(mae) > 2 else None,
                    note="legacy aggregate without explicit split metadata",
                )
            )
    return rows


def load_legacy_doc_reference_rows(repo_root: Path) -> list[BaselineRow]:
    report_path = _existing_path(
        repo_root / "training/runs/접촉점 기준 학습_0409/md/sensor_learning_report_final_20260406.md",
        repo_root / "hitmap/training/runs/접촉점 기준 학습_0409/md/sensor_learning_report_final_20260406.md",
    )
    if report_path is None:
        return []
    return [
        BaselineRow(
            group="legacy_0409_doc",
            run_name="legacy_sr_ff_report_reference",
            source=str(report_path),
            split_policy="document_reference",
            table_role="reference_only",
            condition_policy="legacy_pipeline_doc",
            x_mae=0.62,
            y_mae=0.31,
            z_mae=0.08,
            fz_mae=0.602,
            note="doc-only reference; not split-aligned json output",
        )
    ]


def _parse_metric_cell(value: str) -> float | None:
    value = value.strip()
    if value == "-":
        return None
    return float(value)


def load_markdown_report_rows(path: Path) -> list[BaselineRow]:
    if not path.exists():
        return []

    rows: list[BaselineRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or line.startswith("| Group "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 11:
            continue
        group, run_name, table_role, split_policy, condition_policy = cells[:5]
        x_mae, y_mae, z_mae, fz_mae = (_parse_metric_cell(cell) for cell in cells[5:9])
        note = cells[9]
        source = cells[10].strip("`")
        rows.append(
            BaselineRow(
                group=group,
                run_name=run_name,
                source=source,
                split_policy=split_policy,
                table_role=table_role,
                condition_policy=condition_policy,
                x_mae=x_mae,
                y_mae=y_mae,
                z_mae=z_mae,
                fz_mae=fz_mae,
                note=note,
            )
        )
    return rows


def _extend_unique(rows: list[BaselineRow], new_rows: list[BaselineRow]) -> None:
    seen = {row.run_name for row in rows}
    for row in new_rows:
        if row.run_name in seen:
            continue
        rows.append(row)
        seen.add(row.run_name)


def collect_baseline_rows(repo_root: Path) -> list[BaselineRow]:
    rows: list[BaselineRow] = []
    comparison_path = _existing_path(
        repo_root / "training/runs/runs_comparison/comparison_results.json",
        repo_root / "hitmap/training/runs/runs_comparison/comparison_results.json",
    )
    if comparison_path is not None:
        rows.extend(load_current_comparison_rows(comparison_path))

    zfz_path = _existing_path(
        repo_root / "training/runs/runs_z_fz/cv_summary_z_fz_regressor.json",
        repo_root / "hitmap/training/runs/runs_z_fz/cv_summary_z_fz_regressor.json",
    )
    if zfz_path is not None:
        rows.extend(load_current_zfz_rows(zfz_path))

    legacy_dir = _existing_path(
        repo_root / "training/runs/접촉점 기준 학습_0409/학습결과",
        repo_root / "hitmap/training/runs/접촉점 기준 학습_0409/학습결과",
    )
    if legacy_dir is not None:
        rows.extend(load_legacy_0409_rows(legacy_dir))

    rows.extend(load_legacy_doc_reference_rows(repo_root))
    _extend_unique(rows, load_markdown_report_rows(repo_root / "hitmap/md/260415_reconstruction_baseline_report.md"))
    return rows


def render_markdown_table(rows: list[BaselineRow]) -> str:
    header = (
        "| Group | Run | Role | Split | Condition | x MAE | y MAE | z MAE | fz MAE | Note | Source |\n"
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n"
    )
    body = []
    for row in rows:
        body.append(
            "| {group} | {run_name} | {table_role} | {split_policy} | {condition_policy} | "
            "{x_mae} | {y_mae} | {z_mae} | {fz_mae} | {note} | `{source}` |".format(
                group=row.group,
                run_name=row.run_name,
                table_role=row.table_role,
                split_policy=row.split_policy,
                condition_policy=row.condition_policy,
                x_mae=_round_or_none(row.x_mae) if row.x_mae is not None else "-",
                y_mae=_round_or_none(row.y_mae) if row.y_mae is not None else "-",
                z_mae=_round_or_none(row.z_mae) if row.z_mae is not None else "-",
                fz_mae=_round_or_none(row.fz_mae) if row.fz_mae is not None else "-",
                note=row.note,
                source=row.source,
            )
        )
    return header + "\n".join(body) + "\n"


def render_reclassification_notes() -> str:
    return "\n".join(
        [
            "- `eval-split all` heatmap 결과는 exploratory로만 분류한다. 현재 공식 baseline 표에는 포함하지 않는다.",
            "- `predicted_xy + GT radius`는 pseudo end-to-end upper-bound이므로 메인 성능 표에서 분리한다.",
            "- `z_fz_regressor_gt_xy`도 deployment condition이 아니므로 reference-only로 유지한다.",
        ]
    )
