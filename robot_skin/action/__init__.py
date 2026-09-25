"""Action spaces, chunking and hand → robot retargeting.

- :mod:`.space` — canonical 54-D MANO hand action (wrist pos | wrist rot 6D | 15×3 finger
  axis-angle), robot joint actions, relative actions, :class:`ActionNormalizer`.
- :mod:`.chunking` — ACT action chunks on the 200 Hz master clock and :class:`TemporalEnsembler`.
- :mod:`.retarget` — :class:`FingertipRetargeter` (DexPilot / AnyTeleop-style fingertip-vector
  matching against any differentiable FK, e.g. ``pose.urdf.URDFModel``).
"""
from .chunking import TemporalEnsembler, action_chunk, action_chunks, policy_stride, policy_tick_indices
from .retarget import HAND_FINGERS, FingertipRetargeter, hand_action_fingertips, human_fingertips
from .space import (ACTION_KINDS, FINGERS_AA, HAND_MANO_DIM, HAND_MANO_NAMES, HAND_SLICES, MANO_FINGER_JOINTS,
                    NORM_METHODS, REL_MODES, WRIST_POS, WRIST_ROT6D, ActionNormalizer, ActionSpec, HandArrays,
                    actions_from_episode, hand_action_from_arrays, hand_action_from_episode, hand_action_to_arrays,
                    make_absolute, make_relative, robot_action_from_q, wrist_rotation)

__all__ = [
    "ActionSpec", "ACTION_KINDS", "HAND_MANO_DIM", "HAND_MANO_NAMES", "HAND_SLICES", "MANO_FINGER_JOINTS",
    "WRIST_POS", "WRIST_ROT6D", "FINGERS_AA", "REL_MODES", "NORM_METHODS", "HandArrays",
    "hand_action_from_arrays", "hand_action_to_arrays",
    "wrist_rotation", "robot_action_from_q", "hand_action_from_episode", "actions_from_episode",
    "make_relative", "make_absolute", "ActionNormalizer",
    "action_chunk", "action_chunks", "policy_stride", "policy_tick_indices", "TemporalEnsembler",
    "FingertipRetargeter", "HAND_FINGERS", "human_fingertips", "hand_action_fingertips",
]
