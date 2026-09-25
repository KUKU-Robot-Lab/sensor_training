import unittest

from training.utils.split_policy import (
    DEFAULT_MIN_SAMPLES_BY_REGIME,
    build_stratified_cv_folds,
    evaluate_regime_sample_counts,
    parse_trial_metadata,
    suggest_holdout_trials,
)


class SplitPolicyTest(unittest.TestCase):
    def test_parse_trial_metadata_extracts_diameter_depth_and_regime(self) -> None:
        parsed = parse_trial_metadata("ecomesh_d5_z1.5_test9")

        assert parsed is not None
        self.assertEqual(parsed.diameter_mm, 5.0)
        self.assertEqual(parsed.depth_mm, 1.5)
        self.assertEqual(parsed.depth_regime, "deep")
        self.assertEqual(parsed.test_index, 9)

    def test_holdout_trials_pick_one_last_trial_per_stratum(self) -> None:
        trial_ids = [
            "ecomesh_d10_z1.0_test1",
            "ecomesh_d10_z1.0_test3",
            "ecomesh_d5_z1.0_test2",
            "ecomesh_d5_z1.0_test3",
            "ecomesh_d5_z1.5_test7",
            "ecomesh_d5_z1.5_test9",
        ]

        held_out = suggest_holdout_trials(trial_ids)

        self.assertEqual(
            held_out,
            [
                "ecomesh_d10_z1.0_test3",
                "ecomesh_d5_z1.0_test3",
                "ecomesh_d5_z1.5_test9",
            ],
        )

    def test_stratified_cv_excludes_holdout_trials(self) -> None:
        trial_ids = [
            "ecomesh_d10_z1.0_test1",
            "ecomesh_d10_z1.0_test2",
            "ecomesh_d10_z1.0_test3",
            "ecomesh_d5_z1.0_test1",
            "ecomesh_d5_z1.0_test2",
            "ecomesh_d5_z1.0_test3",
            "ecomesh_d5_z1.5_test1",
            "ecomesh_d5_z1.5_test2",
            "ecomesh_d5_z1.5_test3",
            "ecomesh_d5_z1.5_test4",
            "ecomesh_d5_z1.5_test5",
        ]
        holdout = ["ecomesh_d10_z1.0_test3", "ecomesh_d5_z1.0_test3", "ecomesh_d5_z1.5_test5"]

        folds = build_stratified_cv_folds(trial_ids, cv_folds=3, held_out_trials=holdout)

        assigned = {trial for fold in folds for trial in fold}
        self.assertTrue(assigned.isdisjoint(set(holdout)))
        self.assertEqual(assigned, set(trial_ids) - set(holdout))

    def test_regime_minimums_are_checked_against_holdout_counts(self) -> None:
        trial_counts = {
            "ecomesh_d10_z1.0_test3": 239_399,
            "ecomesh_d5_z1.0_test3": 239_590,
            "ecomesh_d5_z1.5_test9": 239_446,
        }
        holdout = list(trial_counts)

        regime_counts = evaluate_regime_sample_counts(trial_counts, holdout)

        self.assertGreaterEqual(regime_counts["shallow"], DEFAULT_MIN_SAMPLES_BY_REGIME["shallow"])
        self.assertGreaterEqual(regime_counts["deep"], DEFAULT_MIN_SAMPLES_BY_REGIME["deep"])


if __name__ == "__main__":
    unittest.main()
