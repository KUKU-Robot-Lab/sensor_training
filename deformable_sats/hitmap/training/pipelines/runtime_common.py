from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from training.data.dataset_zarr import ZarrDataset


@dataclass(frozen=True)
class TrialSplit:
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    train_trials: list[str]
    val_trials: list[str]
    test_trials: list[str]
    fold_index: int = 0
    num_folds: int = 1
    split_mode: str = "manual"


@dataclass(frozen=True)
class TrialMetadata:
    trial_id: str
    sample_count: int
    diameter_mm: Optional[float]
    min_depth_mm: Optional[float]
    max_depth_mm: Optional[float]
    depth_bin_counts: dict[str, int]
    stratify_label: str


def resolve_zarr_path(data_dir: str, zarr_path: str = "") -> str:
    if zarr_path:
        return str(Path(zarr_path))
    p = Path(data_dir)
    if p.suffix == ".zarr":
        return str(p)
    cands = sorted((p / "zarr_data").glob("*.zarr")) + sorted(p.glob("*.zarr"))
    cands = sorted(set(cands))
    if len(cands) == 1:
        return str(cands[0])
    if len(cands) > 1:
        formatted = ", ".join(str(c) for c in cands)
        raise RuntimeError(
            "Found multiple .zarr datasets; pass --zarr-path explicitly or provide a single integrated zarr. "
            f"Candidates: {formatted}"
        )
    return ""


def parse_trial_list(values: Optional[Sequence[str]]) -> Optional[list[str]]:
    if values is None:
        return None

    parsed: list[str] = []
    for raw in values:
        parsed.extend(token.strip() for token in str(raw).split(",") if token.strip())
    return parsed


def dataset_split_ids(dataset: Dataset) -> list[str]:
    explicit_ids = getattr(dataset, "sample_trial_ids", None)
    if explicit_ids is not None:
        if len(explicit_ids) != len(dataset):
            raise RuntimeError(
                f"Dataset split metadata length mismatch: {len(explicit_ids)} ids for {len(dataset)} samples"
            )
        return [str(x) for x in explicit_ids]

    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise RuntimeError("Dataset does not expose trial split metadata; expected sample_trial_ids or samples")

    split_ids = []
    for sample in samples:
        if isinstance(sample, dict) and "trial_id" in sample:
            split_ids.append(str(sample["trial_id"]))
        elif isinstance(sample, tuple) and sample:
            split_ids.append(str(sample[0]))
        else:
            raise RuntimeError(f"Cannot infer trial id for dataset sample: {sample!r}")
    return split_ids


def _dataset_optional_array(dataset: Dataset, attr_name: str) -> Optional[list[float]]:
    values = getattr(dataset, attr_name, None)
    if values is None:
        return None
    if len(values) != len(dataset):
        raise RuntimeError(
            f"Dataset metadata length mismatch for {attr_name}: {len(values)} values for {len(dataset)} samples"
        )
    if torch.is_tensor(values):
        return [float(x) for x in values.detach().cpu().view(-1).tolist()]
    return [float(x) for x in values]


def dataset_diameter_values(dataset: Dataset) -> Optional[list[float]]:
    for attr_name in ("sample_diameter_mm",):
        values = _dataset_optional_array(dataset, attr_name)
        if values is not None:
            return values

    radius_values = _dataset_optional_array(dataset, "sample_radius_mm")
    if radius_values is not None:
        return [float(x) * 2.0 for x in radius_values]

    samples = getattr(dataset, "samples", None)
    if samples is None:
        return None

    out: list[float] = []
    for sample in samples:
        if isinstance(sample, dict) and "diameter_mm" in sample:
            out.append(float(sample["diameter_mm"]))
        elif isinstance(sample, tuple) and len(sample) >= 3:
            out.append(float(sample[2]))
        else:
            return None
    return out


def dataset_depth_values(dataset: Dataset) -> Optional[list[float]]:
    for attr_name in ("sample_depth_mm",):
        values = _dataset_optional_array(dataset, attr_name)
        if values is not None:
            return values

    samples = getattr(dataset, "samples", None)
    if samples is None:
        return None

    out: list[float] = []
    for sample in samples:
        if isinstance(sample, dict) and "depth_mm" in sample:
            out.append(float(sample["depth_mm"]))
        else:
            return None
    return out


def format_depth_bin_label(lo: float, hi: float) -> str:
    hi_text = "inf" if np.isinf(hi) else f"{hi:g}"
    return f"[{lo:g},{hi_text})"


def normalize_depth_bin_edges(depth_bin_edges: Optional[Sequence[float]]) -> list[float]:
    if not depth_bin_edges:
        return [0.0, float("inf")]

    finite = sorted(float(edge) for edge in depth_bin_edges if not np.isinf(float(edge)))
    if not finite:
        return [0.0, float("inf")]

    if finite[0] > 0.0:
        finite = [0.0] + finite
    if not np.isinf(finite[-1]):
        finite.append(float("inf"))
    return finite


def depth_bin_index(depth_mm: float, depth_bin_edges: Sequence[float]) -> int:
    for idx in range(len(depth_bin_edges) - 1):
        lo = depth_bin_edges[idx]
        hi = depth_bin_edges[idx + 1]
        if depth_mm >= lo and depth_mm < hi:
            return idx
    return max(0, len(depth_bin_edges) - 2)


def collect_trial_metadata(dataset: Dataset, depth_bin_edges: Optional[Sequence[float]] = None) -> dict[str, TrialMetadata]:
    trial_ids = dataset_split_ids(dataset)
    diameter_values = dataset_diameter_values(dataset)
    depth_values = dataset_depth_values(dataset)
    edges = normalize_depth_bin_edges(depth_bin_edges)

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, trial_id in enumerate(trial_ids):
        grouped_indices[str(trial_id)].append(idx)

    metadata: dict[str, TrialMetadata] = {}
    for trial_id, indices in grouped_indices.items():
        diameter_mm = None
        if diameter_values is not None:
            unique_diameters = sorted({round(float(diameter_values[i]), 6) for i in indices})
            diameter_mm = unique_diameters[0] if unique_diameters else None

        min_depth_mm = None
        max_depth_mm = None
        depth_counts: dict[str, int] = {}
        depth_label = "depth-unknown"
        if depth_values is not None:
            trial_depths = [float(depth_values[i]) for i in indices]
            min_depth_mm = min(trial_depths)
            max_depth_mm = max(trial_depths)
            raw_counts = defaultdict(int)
            for depth_mm in trial_depths:
                bin_idx = depth_bin_index(depth_mm, edges)
                label = format_depth_bin_label(edges[bin_idx], edges[bin_idx + 1])
                raw_counts[label] += 1
            depth_counts = dict(sorted(raw_counts.items()))
            dominant_bin = max(depth_counts.items(), key=lambda item: (item[1], item[0]))[0]
            depth_label = f"depth-{dominant_bin}"

        diameter_label = "diameter-unknown" if diameter_mm is None else f"diameter-{diameter_mm:g}"
        metadata[trial_id] = TrialMetadata(
            trial_id=trial_id,
            sample_count=len(indices),
            diameter_mm=diameter_mm,
            min_depth_mm=min_depth_mm,
            max_depth_mm=max_depth_mm,
            depth_bin_counts=depth_counts,
            stratify_label=f"{diameter_label}|{depth_label}",
        )
    return metadata


def split_indices_for_trials(dataset: Dataset, train_trials: Sequence[str], val_trials: Sequence[str], test_trials: Sequence[str]) -> TrialSplit:
    sample_trials = dataset_split_ids(dataset)
    train_set = set(str(t) for t in train_trials)
    val_set = set(str(t) for t in val_trials)
    test_set = set(str(t) for t in test_trials)
    return TrialSplit(
        train_indices=[i for i, trial in enumerate(sample_trials) if trial in train_set],
        val_indices=[i for i, trial in enumerate(sample_trials) if trial in val_set],
        test_indices=[i for i, trial in enumerate(sample_trials) if trial in test_set],
        train_trials=sorted(train_set),
        val_trials=sorted(val_set),
        test_trials=sorted(test_set),
    )


def split_indices_by_trial(
    dataset: Dataset,
    seed: int,
    val_trials: Optional[Sequence[str]] = None,
    test_trials: Optional[Sequence[str]] = None,
    val_ratio: float = 0.2,
) -> TrialSplit:
    sample_trials = dataset_split_ids(dataset)
    trial_ids = sorted(set(sample_trials))
    if len(trial_ids) < 2 and val_trials is None and test_trials is None:
        raise RuntimeError(
            "Trial-level split needs at least 2 distinct trials for train/val. "
            f"Found: {trial_ids}"
        )

    all_trials = set(trial_ids)
    requested_val = set(val_trials or [])
    requested_test = set(test_trials or [])
    unknown = (requested_val | requested_test) - all_trials
    if unknown:
        raise RuntimeError(f"Requested split trials are not present in dataset: {sorted(unknown)}")

    overlap = requested_val & requested_test
    if overlap:
        raise RuntimeError(f"Trials cannot be both val and test: {sorted(overlap)}")

    rng = np.random.default_rng(seed)
    remaining_trials = [t for t in trial_ids if t not in requested_val and t not in requested_test]

    if val_trials is None:
        if len(remaining_trials) < 2:
            raise RuntimeError(
                "Seed-based trial split needs at least 2 train/val candidate trials after test exclusion. "
                f"Candidates: {remaining_trials}"
            )
        shuffled = np.array(remaining_trials, dtype=object)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        n_val = min(n_val, len(shuffled) - 1)
        requested_val = set(str(t) for t in shuffled[:n_val])
        remaining_trials = [str(t) for t in shuffled[n_val:]]

    train_trials = sorted(t for t in remaining_trials if t not in requested_val)
    val_trials_sorted = sorted(requested_val)
    test_trials_sorted = sorted(requested_test)

    if not train_trials:
        raise RuntimeError("Trial split leaves no train trials. Adjust --val-trials/--test-trials.")
    if not val_trials_sorted:
        raise RuntimeError("Trial split leaves no val trials. Provide --val-trials or more trials.")

    base = split_indices_for_trials(dataset, train_trials, val_trials_sorted, test_trials_sorted)
    return TrialSplit(
        train_indices=base.train_indices,
        val_indices=base.val_indices,
        test_indices=base.test_indices,
        train_trials=base.train_trials,
        val_trials=base.val_trials,
        test_trials=base.test_trials,
        fold_index=0,
        num_folds=1,
        split_mode="manual" if val_trials is not None or test_trials is not None else "single_split",
    )


def build_cv_splits(
    dataset: Dataset,
    seed: int,
    cv_folds: int,
    val_trials: Optional[Sequence[str]] = None,
    test_trials: Optional[Sequence[str]] = None,
    depth_bin_edges: Optional[Sequence[float]] = None,
    stratify_diameter_depth: bool = False,
    auto_test_trials: int = 0,
) -> list[TrialSplit]:
    if val_trials is not None or test_trials is not None:
        return [split_indices_by_trial(dataset, seed, val_trials=val_trials, test_trials=test_trials)]

    sample_trials = dataset_split_ids(dataset)
    trial_ids = sorted(set(sample_trials))
    if len(trial_ids) < 2:
        raise RuntimeError(f"Trial-aware CV needs at least 2 distinct trials. Found: {trial_ids}")

    metadata = collect_trial_metadata(dataset, depth_bin_edges)
    rng = np.random.default_rng(seed)
    available_trials = list(trial_ids)

    selected_test_trials: list[str] = []
    max_auto_test_trials = max(0, len(trial_ids) - 2)
    auto_test_trials = max(0, min(int(auto_test_trials), max_auto_test_trials))
    if auto_test_trials > 0:
        grouped_trials: dict[str, list[str]] = defaultdict(list)
        for trial_id in available_trials:
            grouped_trials[metadata[trial_id].stratify_label].append(trial_id)
        for group in grouped_trials.values():
            rng.shuffle(group)
        labels = sorted(grouped_trials)
        rng.shuffle(labels)
        cursor = 0
        while len(selected_test_trials) < auto_test_trials and labels:
            label = labels[cursor % len(labels)]
            if grouped_trials[label]:
                selected_test_trials.append(grouped_trials[label].pop())
            cursor += 1
            labels = [name for name in labels if grouped_trials[name]]
        available_trials = [trial_id for trial_id in trial_ids if trial_id not in set(selected_test_trials)]

    num_folds = max(2, min(cv_folds, len(available_trials)))
    rng = np.random.default_rng(seed)
    trial_folds: list[list[str]]
    if stratify_diameter_depth:
        grouped_trials: dict[str, list[str]] = defaultdict(list)
        for trial_id in available_trials:
            grouped_trials[metadata[trial_id].stratify_label].append(trial_id)
        trial_folds = [[] for _ in range(num_folds)]
        fold_cursor = 0
        for label in sorted(grouped_trials):
            group = grouped_trials[label]
            rng.shuffle(group)
            for trial_id in group:
                trial_folds[fold_cursor % num_folds].append(str(trial_id))
                fold_cursor += 1
        trial_folds = [sorted(fold) for fold in trial_folds if fold]
    else:
        shuffled = np.array(available_trials, dtype=object)
        rng.shuffle(shuffled)
        trial_folds = [sorted(str(t) for t in fold.tolist()) for fold in np.array_split(shuffled, num_folds) if len(fold) > 0]

    splits: list[TrialSplit] = []
    for fold_index, val_fold in enumerate(trial_folds):
        train_trials = sorted(str(t) for t in available_trials if str(t) not in set(val_fold))
        base = split_indices_for_trials(dataset, train_trials, val_fold, selected_test_trials)
        splits.append(
            TrialSplit(
                train_indices=base.train_indices,
                val_indices=base.val_indices,
                test_indices=base.test_indices,
                train_trials=base.train_trials,
                val_trials=base.val_trials,
                test_trials=base.test_trials,
                fold_index=fold_index,
                num_folds=len(trial_folds),
                split_mode="kfold",
            )
        )
    return splits


def _bucket_counts(values: Optional[list[float]]) -> Optional[dict[str, int]]:
    if values is None:
        return None
    counts = defaultdict(int)
    for value in values:
        counts[f"{float(value):g}"] += 1
    return dict(sorted(counts.items()))


def _depth_bin_counts(values: Optional[list[float]], depth_bin_edges: Sequence[float]) -> Optional[dict[str, int]]:
    if values is None:
        return None
    counts = defaultdict(int)
    for value in values:
        idx = depth_bin_index(float(value), depth_bin_edges)
        label = format_depth_bin_label(depth_bin_edges[idx], depth_bin_edges[idx + 1])
        counts[label] += 1
    return dict(sorted(counts.items()))


def _split_metadata_counts(split: TrialSplit, diameter_values: Optional[list[float]], depth_values: Optional[list[float]], depth_bin_edges: Sequence[float]) -> dict:
    def _select(values: Optional[list[float]], indices: list[int]) -> Optional[list[float]]:
        if values is None:
            return None
        return [float(values[idx]) for idx in indices]

    return {
        "train_diameter_counts": _bucket_counts(_select(diameter_values, split.train_indices)),
        "val_diameter_counts": _bucket_counts(_select(diameter_values, split.val_indices)),
        "test_diameter_counts": _bucket_counts(_select(diameter_values, split.test_indices)),
        "train_depth_bin_counts": _depth_bin_counts(_select(depth_values, split.train_indices), depth_bin_edges),
        "val_depth_bin_counts": _depth_bin_counts(_select(depth_values, split.val_indices), depth_bin_edges),
        "test_depth_bin_counts": _depth_bin_counts(_select(depth_values, split.test_indices), depth_bin_edges),
    }


def save_cv_manifest(
    path: Path,
    splits: Sequence[TrialSplit],
    dataset: Optional[Dataset] = None,
    depth_bin_edges: Optional[Sequence[float]] = None,
    min_depth_bin_samples: int = 0,
    stratify_diameter_depth: bool = False,
) -> None:
    normalized_depth_bins = normalize_depth_bin_edges(depth_bin_edges)
    diameter_values = dataset_diameter_values(dataset) if dataset is not None else None
    depth_values = dataset_depth_values(dataset) if dataset is not None else None
    payload = {
        "num_folds": len(splits),
        "split_policy": {
            "train_val_test": True,
            "stratify_diameter_depth": bool(stratify_diameter_depth),
            "depth_bin_edges": normalized_depth_bins,
            "min_depth_bin_samples": int(min_depth_bin_samples),
        },
        "folds": [
            {
                "fold_index": split.fold_index,
                "num_folds": split.num_folds,
                "split_mode": split.split_mode,
                "train_trials": split.train_trials,
                "val_trials": split.val_trials,
                "test_trials": split.test_trials,
                "train_count": len(split.train_indices),
                "val_count": len(split.val_indices),
                "test_count": len(split.test_indices),
                **_split_metadata_counts(split, diameter_values, depth_values, normalized_depth_bins),
            }
            for split in splits
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class ZarrSequenceDataset(Dataset):
    """
    Build sequence samples from preprocessed Zarr data.
    Sequence key: (trial_id, x_mm, y_mm), sorted by depth within the requested phase.
    """

    def __init__(self, zarr_path: str, seq_len: int = 50, stride: int = 5, phase: str = "all"):
        start = time.perf_counter()
        self.seq_len = seq_len
        self.stride = stride

        print(
            f"[INFO] Loading ZarrSequenceDataset from {zarr_path} "
            f"(phase={phase}, seq_len={seq_len}, stride={stride})",
            flush=True,
        )
        zds = ZarrDataset(zarr_path=zarr_path, split="all", phase=phase)
        tactile = zds.tactile_data.float()
        if getattr(zds, "aux_last_field", "diameter_mm") == "contact_radius_mm":
            radius = zds.aux_data[:, 3:4].float()
        else:
            radius = (zds.aux_data[:, 3:4] / 2.0).float()
        cx = zds.cx_data.float()
        cy = zds.cy_data.float()
        depth = zds.depth_data.float()
        fz = zds.fz_data.float()
        trial_ids = zds.trial_ids

        groups = defaultdict(list)
        for i in range(len(trial_ids)):
            key = (str(trial_ids[i]), round(float(cx[i].item()), 3), round(float(cy[i].item()), 3))
            groups[key].append(i)
        print(
            f"[INFO] Grouped {len(trial_ids):,} rows into {len(groups):,} trial/xy sequences",
            flush=True,
        )

        self.samples = []
        self.sample_trial_ids = []
        self.sample_radius_mm = []
        self.sample_diameter_mm = []
        self.sample_depth_mm = []
        for key, idxs in groups.items():
            idxs = sorted(idxs, key=lambda j: float(depth[j].item()))
            trial_id = key[0]
            t = len(idxs)
            if t <= 0:
                continue
            if t <= seq_len:
                self.samples.append(idxs)
                self.sample_trial_ids.append(trial_id)
                last_i = idxs[-1]
                radius_mm = float(radius[last_i].item())
                self.sample_radius_mm.append(radius_mm)
                self.sample_diameter_mm.append(radius_mm * 2.0)
                self.sample_depth_mm.append(float(depth[last_i].item()))
            else:
                max_start = t - seq_len
                for s in range(0, max_start + 1, stride):
                    seq = idxs[s : s + seq_len]
                    self.samples.append(seq)
                    self.sample_trial_ids.append(trial_id)
                    last_i = seq[-1]
                    radius_mm = float(radius[last_i].item())
                    self.sample_radius_mm.append(radius_mm)
                    self.sample_diameter_mm.append(radius_mm * 2.0)
                    self.sample_depth_mm.append(float(depth[last_i].item()))

        self.tactile = tactile
        self.radius = radius
        self.cx = cx
        self.cy = cy
        self.depth = depth
        self.fz = fz
        print(
            f"[INFO] ZarrSequenceDataset ready: {len(self.samples):,} samples built in "
            f"{time.perf_counter() - start:.1f}s",
            flush=True,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idxs = self.samples[idx]
        t = len(idxs)
        s16 = self.tactile[idxs]
        r = self.radius[idxs]

        if t < self.seq_len:
            pad = self.seq_len - t
            s16 = torch.cat([s16, torch.zeros(pad, 16, dtype=s16.dtype)], dim=0)
            r = torch.cat([r, torch.zeros(pad, 1, dtype=r.dtype)], dim=0)

        s16 = s16[: self.seq_len]
        r = r[: self.seq_len]
        grid = s16.reshape(self.seq_len, 1, 4, 4)

        iso = torch.zeros(self.seq_len, 17, dtype=s16.dtype)
        iso[:, :16] = s16
        iso[:, 16:17] = r

        last_i = idxs[-1]
        tgt = torch.zeros(5, dtype=s16.dtype)
        tgt[0] = self.cx[last_i]
        tgt[1] = self.cy[last_i]
        tgt[2] = self.depth[last_i]
        tgt[3] = self.fz[last_i]
        tgt[4] = self.radius[last_i]
        return grid, iso, tgt
