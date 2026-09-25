"""Taxel pose providers and the kinematics behind them (MANO skeleton, URDF FK, glove IMUs).

See ``pose/README.md``. Everything resolves to ``transform_taxels(layout, {parent: 4×4})``.
"""
from .provider import (
    StaticPoseProvider, TaxelPoseProvider, TransformPoseProvider, sample_poses, transform_taxels,
)
from .mano import (
    CAPSULE_NAMES, FINGER_JOINTS, FINGERS, MANO_JOINTS, MANO_PARENTS, SEGMENT_TO_JOINT, TIP_NAMES,
    ManoPoseProvider, ManoSkeleton, self_touch_from_hand, taxel_poses_from_hand,
)
from .urdf import URDFJoint, URDFModel
from .robot_fk import RobotFKPoseProvider, taxel_poses_from_joints
from .imu_model import (
    ImuHandPoseNet, apply_imu_offsets, apply_imu_offsets_to_vectors, calibrate_imu_offsets,
    estimate_world_alignment, hand_pose_loss, imu_calibration_from_dict, imu_calibration_to_dict,
    imu_feature_dim, imu_features, imu_reference_rotations, imu_site_quats, imu_windows,
    predict_finger_pose_sequence, rotation_geodesic, synthesize_imu, to_axis_angle,
)
from .glove_imu2mano import GloveImu2ManoPoseProvider, finetune_vifnet_s, load_vifnet_s
from .vision_hand import (
    HaMeREstimator, VisionHandEstimator, estimate_sequence, load_hand_labels, register_hand_labels,
    save_hand_labels, smooth_hand_labels,
)

__all__ = [
    # providers
    "StaticPoseProvider", "TaxelPoseProvider", "TransformPoseProvider", "sample_poses",
    "transform_taxels", "ManoPoseProvider", "RobotFKPoseProvider", "GloveImu2ManoPoseProvider",
    # MANO
    "CAPSULE_NAMES", "FINGER_JOINTS", "FINGERS", "MANO_JOINTS", "MANO_PARENTS", "SEGMENT_TO_JOINT",
    "TIP_NAMES", "ManoSkeleton", "self_touch_from_hand", "taxel_poses_from_hand",
    # URDF
    "URDFJoint", "URDFModel", "taxel_poses_from_joints",
    # IMU
    "ImuHandPoseNet", "apply_imu_offsets", "apply_imu_offsets_to_vectors", "calibrate_imu_offsets",
    "estimate_world_alignment", "hand_pose_loss", "imu_calibration_from_dict",
    "imu_calibration_to_dict", "imu_feature_dim", "imu_features", "imu_reference_rotations",
    "imu_site_quats",
    "imu_windows", "predict_finger_pose_sequence", "rotation_geodesic", "synthesize_imu",
    "to_axis_angle", "finetune_vifnet_s", "load_vifnet_s",
    # vision labels
    "HaMeREstimator", "VisionHandEstimator", "estimate_sequence", "load_hand_labels",
    "register_hand_labels", "save_hand_labels", "smooth_hand_labels",
]
