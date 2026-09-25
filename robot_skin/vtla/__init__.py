"""VTLA: vision + tactile + language → action chunks (see ``vtla/README.md``).

- :mod:`.adapter` — :class:`TactileTokenAdapter` (Perceiver-style taxel → K tokens) and
  :class:`ContactGate` (no contact ⇒ no tactile signal).
- :mod:`.model` — :class:`VTLAConfig`, :class:`VTLAPolicy` (fusion transformer over language /
  vision / tactile / proprio / readout tokens, modality dropout, obs_mode ablation, aux contact
  head) and the deployable ``policy_bundle.pt`` (:func:`build_policy_from_bundle`).
- :mod:`.heads` — :class:`ChunkRegressionHead` (ACT) and :class:`FlowMatchingHead` (flow
  matching / rectified flow; τ = 0 noise → τ = 1 data, Euler sampling).
- :mod:`.losses` — masked chunk losses, aux contact BCE, the Trainer ``loss_fn`` :func:`vtla_loss`.
- :mod:`.dataset` — :class:`VTLADataset` (policy-tick samples of processed D2 episodes),
  :func:`make_observation` (shared with online control), :func:`collate_vtla`.
- :mod:`.dpo` — :func:`dpo_loss` / :func:`preference_loss` (DPO hook; pair collection is a stub).

Stage runner: :mod:`robot_skin.stages.vtla`.
"""
from .adapter import ContactGate, TactileTokenAdapter
from .dataset import (AUX_TARGETS, BOOTSTRAP_DEFAULTS, CONTACT_RULES, PSEUDO_LABEL_KEY,
                      TACTILE_SOURCES, TASK_PHASES, VTLACollator, VTLADataset,
                      bootstrap_tactile_arrays, collate_vtla, contact_from_level, episode_dead_taxels,
                      episode_tactile, eval_transform_from_dict, eval_transform_to_dict, history_ticks,
                      make_observation, sample_phase_mask)
from .dpo import (PreferencePair, build_preference_pairs, chunk_log_likelihood, dpo_loss,
                  make_reference_policy, preference_loss)
from .heads import (HEAD_TYPES, TAU_DISTS, ChunkRegressionHead, FlowMatchingHead, build_head,
                    draw_on_generator, sinusoidal_embedding)
from .losses import (contact_bce, masked_error, masked_l1, masked_mse, masked_step_sums,
                     vtla_loss, weighted_total)
from .model import (BUNDLE_FORMAT, BUNDLE_VERSION, MODALITIES, POLICY_BUNDLE_NAME, VTLAConfig,
                    VTLAPolicy, build_policy_from_bundle, bundle_components, read_policy_bundle,
                    save_policy_bundle)

__all__ = [
    # adapter
    "ContactGate", "TactileTokenAdapter",
    # model + bundle
    "MODALITIES", "VTLAConfig", "VTLAPolicy", "POLICY_BUNDLE_NAME", "BUNDLE_FORMAT", "BUNDLE_VERSION",
    "save_policy_bundle", "read_policy_bundle", "build_policy_from_bundle", "bundle_components",
    # heads
    "HEAD_TYPES", "TAU_DISTS", "ChunkRegressionHead", "FlowMatchingHead", "build_head",
    "sinusoidal_embedding", "draw_on_generator",
    # losses
    "masked_error", "masked_l1", "masked_mse", "masked_step_sums", "contact_bce", "weighted_total",
    "vtla_loss",
    # dataset
    "TASK_PHASES", "CONTACT_RULES", "AUX_TARGETS", "TACTILE_SOURCES", "BOOTSTRAP_DEFAULTS",
    "PSEUDO_LABEL_KEY", "bootstrap_tactile_arrays", "episode_tactile", "episode_dead_taxels",
    "contact_from_level",
    "sample_phase_mask", "history_ticks", "eval_transform_to_dict", "eval_transform_from_dict",
    "make_observation",
    "VTLADataset", "VTLACollator", "collate_vtla",
    # dpo
    "dpo_loss", "chunk_log_likelihood", "preference_loss", "make_reference_policy",
    "PreferencePair", "build_preference_pairs",
]
