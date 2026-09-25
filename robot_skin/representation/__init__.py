"""Taxel representation: tactile value features, taxel tokens / encoder, masked pretraining.

- :mod:`.encoder` — :func:`tactile_value_features` (the single per-taxel feature function used by
  pretraining, VTLA and online control), :class:`TactileFeatureSpec` / :class:`TactileHistory`
  (temporal stacking), :class:`TaxelEncoder`, ``encoder_state.pt`` save/load.
- :mod:`.tokenizer` — :class:`TaxelTokenizer` (value MLP + pose Fourier MLP → token).
- :mod:`.pretrain` — MAE-style :class:`MaskedTaxelPretrainer`, masks, dataset, metrics
  (stage runner: ``robot_skin.stages.pretrain``).
"""
from .encoder import (ENCODER_STATE_NAME, OBS_MODES, TACTILE_FRAME_DIMS, TactileFeatureSpec,
                      TactileHistory, TaxelEncoder, episode_tactile_arrays, history_indices,
                      load_pretrained_encoder, read_encoder_state, save_pretrained_encoder,
                      stack_history, tactile_value_dim, tactile_value_features, z_feature,
                      z_from_feature)
from .pretrain import (MASK_MODES, MaskedTaxelPretrainer, TaxelPretrainDataset, collate_pretrain,
                       evaluate_reconstruction, layout_group_matrix, level_class_weights,
                       pretrain_loss, random_taxel_mask, sample_taxel_mask)
from .tokenizer import TaxelTokenizer, fourier_features

__all__ = [
    # features
    "OBS_MODES", "TACTILE_FRAME_DIMS", "tactile_value_dim", "tactile_value_features", "z_feature",
    "z_from_feature", "history_indices", "stack_history", "episode_tactile_arrays",
    "TactileFeatureSpec", "TactileHistory",
    # encoder
    "TaxelTokenizer", "fourier_features", "TaxelEncoder", "ENCODER_STATE_NAME",
    "save_pretrained_encoder", "read_encoder_state", "load_pretrained_encoder",
    # pretraining
    "MASK_MODES", "random_taxel_mask", "sample_taxel_mask", "layout_group_matrix",
    "TaxelPretrainDataset", "collate_pretrain", "level_class_weights", "MaskedTaxelPretrainer",
    "pretrain_loss", "evaluate_reconstruction",
]
