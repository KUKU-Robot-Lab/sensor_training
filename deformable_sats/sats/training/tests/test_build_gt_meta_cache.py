from sats.training.build_gt_meta_cache import _build_parser, _select_trial_ids_for_cache
from sats.training.config import SATSConfig


def test_cache_builder_cli_parses_include_materials():
    args = _build_parser().parse_args(
        [
            "--include-materials",
            "eco20_xy1",
            "eco50_xy1",
            "--exclude-diameters",
            "10",
        ]
    )

    assert args.include_materials == ["eco20_xy1", "eco50_xy1"]
    assert args.exclude_diameters == [10]


def test_cache_trial_selection_applies_material_and_diameter_filters():
    cfg = SATSConfig(
        include_materials=["eco20_xy1"],
        exclude_diameters=[10],
    )
    trial_ids = [
        "eco20_xy1_d5_z2.5_test1",
        "eco20_xy1_d5_z2.5_test2",
        "eco20_xy1_d10_z3.5_test1",
        "eco50_xy1_d5_z2.5_test1",
    ]

    assert _select_trial_ids_for_cache(cfg, trial_ids) == [
        "eco20_xy1_d5_z2.5_test1",
        "eco20_xy1_d5_z2.5_test2",
    ]
