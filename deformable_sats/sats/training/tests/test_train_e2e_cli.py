from sats.training.train_e2e import _build_parser, _config_from_args


def test_train_e2e_cli_stores_include_materials_in_config():
    args = _build_parser().parse_args(
        [
            "--gt-mode",
            "gpu_on_the_fly",
            "--include-materials",
            "eco20_xy1",
            "--exclude-diameters",
            "10",
            "--val-ratio",
            "0",
            "--val-trials",
            "eco20_xy1_d5_z2.5_test3",
            "--grid-step-mm",
            "0.5",
            "--run-name",
            "xy1_d5_eco20_e2e_g05",
        ]
    )

    cfg = _config_from_args(args)

    assert cfg.include_materials == ["eco20_xy1"]
    assert cfg.exclude_diameters == [10]
    assert cfg.val_ratio == 0
    assert cfg.val_trials == ["eco20_xy1_d5_z2.5_test3"]
    assert cfg.grid_size == 41
