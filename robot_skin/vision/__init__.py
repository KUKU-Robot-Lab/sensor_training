"""Camera → visual tokens for the VTLA policy: encoders, image transforms, frozen-feature cache.

See ``vision/README.md``. Optional backbones (torchvision ResNet, HuggingFace DINOv2 / SigLIP /
CLIP) are imported lazily; :class:`TinyConvEncoder` needs nothing beyond torch.
"""
from .encoders import (
    POOL_MODES, HFVisionEncoder, ResNetEncoder, SpatialSoftmax, TinyConvEncoder, TokenPool,
    VisionEncoder, build_vision_encoder, sanitize_key, sincos_pos_embed_2d,
)
from .feature_cache import (
    cache_episode_features, cache_features, feature_path, gather_frame_features, has_cached,
    load_cache_info, load_cached,
)
from .transforms import (
    CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, SIGLIP_MEAN, SIGLIP_STD, EvalTransform,
    Normalize, TrainAugment, build_transforms, center_crop, crop_box, crop_resize, resize,
    resize_short, to_float_tensor,
)

__all__ = [
    # encoders
    "VisionEncoder", "TinyConvEncoder", "ResNetEncoder", "HFVisionEncoder", "SpatialSoftmax",
    "TokenPool", "POOL_MODES", "build_vision_encoder", "sanitize_key", "sincos_pos_embed_2d",
    # transforms
    "IMAGENET_MEAN", "IMAGENET_STD", "CLIP_MEAN", "CLIP_STD", "SIGLIP_MEAN", "SIGLIP_STD",
    "to_float_tensor", "Normalize", "resize", "resize_short", "center_crop", "crop_box",
    "crop_resize", "TrainAugment", "EvalTransform", "build_transforms",
    # feature cache
    "cache_episode_features", "cache_features", "feature_path", "gather_frame_features",
    "has_cached", "load_cache_info", "load_cached",
]
