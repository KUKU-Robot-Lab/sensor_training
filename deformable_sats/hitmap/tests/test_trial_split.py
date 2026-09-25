import json
import tempfile
import unittest
from pathlib import Path

from training.pipelines.runtime_common import (
    build_cv_splits,
    collect_trial_metadata,
    parse_trial_list,
    save_cv_manifest,
    split_indices_by_trial,
)


class FakeDataset:
    def __init__(
        self,
        sample_trial_ids: list[str],
        sample_diameter_mm: list[float] | None = None,
        sample_depth_mm: list[float] | None = None,
    ) -> None:
        self.sample_trial_ids = sample_trial_ids
        self.sample_diameter_mm = sample_diameter_mm or [10.0] * len(sample_trial_ids)
        self.sample_depth_mm = sample_depth_mm or [1.0] * len(sample_trial_ids)

    def __len__(self) -> int:
        return len(self.sample_trial_ids)


class TrialSplitTest(unittest.TestCase):
    def test_seed_split_does_not_mix_trials_between_train_and_val(self) -> None:
        dataset = FakeDataset(
            [
                "trial_a",
                "trial_a",
                "trial_a",
                "trial_b",
                "trial_b",
                "trial_c",
                "trial_c",
            ]
        )

        split = split_indices_by_trial(dataset, seed=7)

        train_trials = {dataset.sample_trial_ids[i] for i in split.train_indices}
        val_trials = {dataset.sample_trial_ids[i] for i in split.val_indices}

        self.assertTrue(split.train_indices)
        self.assertTrue(split.val_indices)
        self.assertTrue(train_trials.isdisjoint(val_trials))

    def test_explicit_val_trials_route_only_those_trials_to_val(self) -> None:
        dataset = FakeDataset(
            [
                "trial_a",
                "trial_b",
                "trial_b",
                "trial_c",
                "trial_c",
            ]
        )

        split = split_indices_by_trial(dataset, seed=7, val_trials=["trial_b"])

        self.assertEqual(split.val_trials, ["trial_b"])
        self.assertEqual({dataset.sample_trial_ids[i] for i in split.val_indices}, {"trial_b"})
        self.assertEqual({dataset.sample_trial_ids[i] for i in split.train_indices}, {"trial_a", "trial_c"})

    def test_explicit_test_trials_are_excluded_from_train_and_val(self) -> None:
        dataset = FakeDataset(
            [
                "trial_a",
                "trial_b",
                "trial_c",
                "trial_d",
            ]
        )

        split = split_indices_by_trial(dataset, seed=7, val_trials=["trial_b"], test_trials=["trial_d"])

        self.assertEqual(split.test_trials, ["trial_d"])
        self.assertEqual({dataset.sample_trial_ids[i] for i in split.test_indices}, {"trial_d"})
        self.assertEqual({dataset.sample_trial_ids[i] for i in split.train_indices}, {"trial_a", "trial_c"})
        self.assertEqual({dataset.sample_trial_ids[i] for i in split.val_indices}, {"trial_b"})

    def test_too_few_trials_raise_clear_error(self) -> None:
        dataset = FakeDataset(["trial_a", "trial_a"])

        with self.assertRaisesRegex(RuntimeError, "at least 2 distinct trials"):
            split_indices_by_trial(dataset, seed=7)

    def test_parse_trial_list_accepts_comma_or_space_separated_values(self) -> None:
        self.assertEqual(parse_trial_list(["trial_a,trial_b", "trial_c"]), ["trial_a", "trial_b", "trial_c"])
        self.assertIsNone(parse_trial_list(None))

    def test_collect_trial_metadata_builds_diameter_depth_strata(self) -> None:
        dataset = FakeDataset(
            ["a", "a", "b", "b"],
            sample_diameter_mm=[5.0, 5.0, 10.0, 10.0],
            sample_depth_mm=[0.7, 0.9, 1.6, 1.8],
        )

        metadata = collect_trial_metadata(dataset, depth_bin_edges=[0.8, 1.1, 1.4, 1.7, float("inf")])

        self.assertEqual(metadata["a"].diameter_mm, 5.0)
        self.assertIn("diameter-5", metadata["a"].stratify_label)
        self.assertIn("depth-[0.8,1.1)", metadata["a"].stratify_label)
        self.assertEqual(metadata["b"].depth_bin_counts["[1.7,inf)"], 1)

    def test_build_cv_splits_supports_auto_test_trials_and_stratification(self) -> None:
        dataset = FakeDataset(
            [
                "a",
                "a",
                "b",
                "b",
                "c",
                "c",
                "d",
                "d",
                "e",
                "e",
            ],
            sample_diameter_mm=[5, 5, 5, 5, 10, 10, 10, 10, 5, 5],
            sample_depth_mm=[0.8, 0.9, 1.0, 1.1, 1.5, 1.6, 1.8, 1.9, 1.7, 1.75],
        )

        splits = build_cv_splits(
            dataset,
            seed=7,
            cv_folds=3,
            depth_bin_edges=[0.8, 1.1, 1.4, 1.7, float("inf")],
            stratify_diameter_depth=True,
            auto_test_trials=1,
        )

        self.assertEqual(len(splits), 3)
        self.assertTrue(all(split.test_trials for split in splits))
        shared_test_trials = {tuple(split.test_trials) for split in splits}
        self.assertEqual(len(shared_test_trials), 1)
        for split in splits:
            self.assertTrue(set(split.train_trials).isdisjoint(split.test_trials))
            self.assertTrue(set(split.val_trials).isdisjoint(split.test_trials))

    def test_save_cv_manifest_records_policy_and_split_counts(self) -> None:
        dataset = FakeDataset(
            ["a", "a", "b", "b", "c", "c"],
            sample_diameter_mm=[5, 5, 10, 10, 5, 5],
            sample_depth_mm=[0.8, 0.9, 1.5, 1.6, 1.8, 1.9],
        )
        splits = build_cv_splits(
            dataset,
            seed=3,
            cv_folds=2,
            depth_bin_edges=[0.8, 1.1, 1.4, 1.7, float("inf")],
            stratify_diameter_depth=True,
            auto_test_trials=1,
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "cv_manifest.json"
            save_cv_manifest(
                manifest_path,
                splits,
                dataset=dataset,
                depth_bin_edges=[0.8, 1.1, 1.4, 1.7, float("inf")],
                min_depth_bin_samples=16,
                stratify_diameter_depth=True,
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertTrue(payload["split_policy"]["train_val_test"])
        self.assertTrue(payload["split_policy"]["stratify_diameter_depth"])
        self.assertEqual(payload["split_policy"]["min_depth_bin_samples"], 16)
        self.assertIn("val_depth_bin_counts", payload["folds"][0])


if __name__ == "__main__":
    unittest.main()
