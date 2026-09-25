from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRIAL_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_d(?P<diameter>\d+(?:\.\d+)?)_z(?P<depth>\d+(?:\.\d+)?)_test(?P<test>\d+)$"
)
DEFAULT_MIN_SAMPLES_BY_REGIME = {
    "shallow": 200_000,
    "deep": 200_000,
}


@dataclass(frozen=True)
class TrialMetadata:
    trial_id: str
    diameter_mm: float
    depth_mm: float
    test_index: int
    depth_regime: str
    stratify_key: str


def parse_trial_metadata(trial_id: str) -> TrialMetadata | None:
    match = TRIAL_PATTERN.match(str(trial_id))
    if not match:
        return None
    diameter_mm = float(match.group("diameter"))
    depth_mm = float(match.group("depth"))
    depth_regime = "shallow" if depth_mm <= 1.1 else "deep"
    return TrialMetadata(
        trial_id=str(trial_id),
        diameter_mm=diameter_mm,
        depth_mm=depth_mm,
        test_index=int(match.group("test")),
        depth_regime=depth_regime,
        stratify_key=f"d{diameter_mm:g}_{depth_regime}",
    )


def load_loading_trial_counts(zarr_path: Path) -> dict[str, int]:
    compact_index = zarr_path / "dataset_index_compact.npz"
    if not compact_index.exists():
        raise FileNotFoundError(f"Compact index not found: {compact_index}")
    payload = np.load(compact_index, allow_pickle=True)
    trial_codes = payload["trial_codes"].astype(int)
    phase_codes = payload["phase_codes"].astype(int)
    trial_vocab = [str(x) for x in payload["trial_vocab"].tolist()]
    loading_mask = phase_codes == 0
    counts: dict[str, int] = {}
    for code in np.unique(trial_codes[loading_mask]):
        if int(code) < 0:
            continue
        counts[trial_vocab[int(code)]] = int(((trial_codes == int(code)) & loading_mask).sum())
    return counts


def suggest_holdout_trials(trial_ids: list[str]) -> list[str]:
    by_stratum: dict[str, list[TrialMetadata]] = {}
    for trial_id in sorted(trial_ids):
        metadata = parse_trial_metadata(trial_id)
        if metadata is None:
            continue
        by_stratum.setdefault(metadata.stratify_key, []).append(metadata)
    selected: list[str] = []
    for stratum in sorted(by_stratum):
        chosen = max(by_stratum[stratum], key=lambda item: (item.test_index, item.trial_id))
        selected.append(chosen.trial_id)
    return sorted(selected)


def build_stratified_cv_folds(
    trial_ids: list[str],
    cv_folds: int,
    held_out_trials: list[str],
) -> list[list[str]]:
    candidates = [trial_id for trial_id in sorted(trial_ids) if trial_id not in set(held_out_trials)]
    metadata_rows = [parse_trial_metadata(trial_id) for trial_id in candidates]
    if any(row is None for row in metadata_rows):
        raise RuntimeError("Cannot build diameter/depth stratified folds because some trial ids are unparsable.")

    folds: list[list[str]] = [[] for _ in range(max(2, min(cv_folds, len(candidates))))]
    stratum_counts = [dict() for _ in folds]
    metadata_by_stratum: dict[str, list[TrialMetadata]] = {}
    for row in metadata_rows:
        assert row is not None
        metadata_by_stratum.setdefault(row.stratify_key, []).append(row)

    for stratum in sorted(metadata_by_stratum):
        for row in sorted(metadata_by_stratum[stratum], key=lambda item: (item.test_index, item.trial_id)):
            fold_index = min(
                range(len(folds)),
                key=lambda idx: (
                    int(stratum_counts[idx].get(stratum, 0)),
                    len(folds[idx]),
                    idx,
                ),
            )
            folds[fold_index].append(row.trial_id)
            stratum_counts[fold_index][stratum] = int(stratum_counts[fold_index].get(stratum, 0)) + 1
    return [sorted(fold) for fold in folds if fold]


def evaluate_regime_sample_counts(
    trial_counts: dict[str, int],
    held_out_trials: list[str],
) -> dict[str, int]:
    counts = {"shallow": 0, "deep": 0}
    for trial_id in held_out_trials:
        metadata = parse_trial_metadata(trial_id)
        if metadata is None:
            continue
        counts[metadata.depth_regime] += int(trial_counts.get(trial_id, 0))
    return counts


def render_split_policy_report(
    trial_counts: dict[str, int],
    held_out_trials: list[str],
    folds: list[list[str]],
) -> str:
    lines = [
        "# Split Policy Report",
        "",
        "## Held-out Test Trials",
        "",
        "| Trial | Diameter (mm) | Depth (mm) | Regime | Loading Samples |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for trial_id in held_out_trials:
        metadata = parse_trial_metadata(trial_id)
        if metadata is None:
            continue
        lines.append(
            f"| {trial_id} | {metadata.diameter_mm:.1f} | {metadata.depth_mm:.1f} | "
            f"{metadata.depth_regime} | {trial_counts.get(trial_id, 0)} |"
        )

    regime_counts = evaluate_regime_sample_counts(trial_counts, held_out_trials)
    lines.extend(
        [
            "",
            "## Minimum Regime Sample Policy",
            "",
            f"- shallow: >= {DEFAULT_MIN_SAMPLES_BY_REGIME['shallow']:,} samples",
            f"- deep: >= {DEFAULT_MIN_SAMPLES_BY_REGIME['deep']:,} samples",
            f"- current held-out shallow samples: {regime_counts['shallow']:,}",
            f"- current held-out deep samples: {regime_counts['deep']:,}",
            "",
            "## Diameter/Depth Stratified CV Design",
            "",
        ]
    )
    for fold_index, fold_trials in enumerate(folds):
        lines.append(f"- fold_{fold_index}: {', '.join(fold_trials)}")
    return "\n".join(lines) + "\n"
