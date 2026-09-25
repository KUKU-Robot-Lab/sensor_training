import json
import math
import pickle
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from robot_skin.datasets import Episode, EpisodeMeta
from robot_skin.datasets import episode as E
from robot_skin.vision import (
    IMAGENET_MEAN, IMAGENET_STD, POOL_MODES, EvalTransform, HFVisionEncoder, Normalize,
    ResNetEncoder, SpatialSoftmax, TinyConvEncoder, TrainAugment, VisionEncoder, build_transforms,
    build_vision_encoder, cache_episode_features, cache_features, center_crop, crop_box,
    crop_resize, feature_path, gather_frame_features, has_cached, load_cache_info, load_cached,
    resize, resize_short, sanitize_key, sincos_pos_embed_2d, to_float_tensor,
)
from robot_skin.vision import feature_cache as FC

SIZES = [(32, 32), (48, 64), (17, 23), (96, 128), (8, 8)]


def _frames(n=5, h=24, w=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, h, w, 3), dtype=np.uint8)


# ── encoders ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("hw", SIZES)
def test_tiny_grid_tokens_fixed_for_any_size(hw):
    torch.manual_seed(0)
    enc = TinyConvEncoder(out_dim=48, grid=(3, 4))
    out = enc(torch.randn(2, 3, *hw))
    assert out.shape == (2, 12, 48) and enc.n_tokens(*hw) == 12 and enc.out_dim == 48
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("pool,expect", [("none", None), ("avg", 1), ("spatial_softmax", 7)])
@pytest.mark.parametrize("hw", SIZES)
def test_tiny_other_pools_token_count_matches_forward(pool, expect, hw):
    torch.manual_seed(0)
    enc = TinyConvEncoder(out_dim=32, pool=pool, n_keypoints=7, channels=(16, 32))
    out = enc(torch.randn(1, 3, *hw))
    assert out.shape == (1, enc.n_tokens(*hw), 32)
    if expect is None:  # 2 stride-2 stages → ceil(h/4)·ceil(w/4)
        assert enc.n_tokens(*hw) == -(-hw[0] // 4) * -(-hw[1] // 4)
    else:
        assert enc.n_tokens(*hw) == expect


def test_tiny_is_trainable_deterministic_and_validates_input():
    torch.manual_seed(0)
    a = TinyConvEncoder(out_dim=16)
    torch.manual_seed(0)
    b = TinyConvEncoder(out_dim=16)
    x = torch.randn(2, 3, 32, 40)
    assert torch.equal(a(x), b(x))
    a(x).pow(2).mean().backward()
    assert all(p.grad is not None for p in a.parameters() if p.requires_grad)
    assert not a.is_frozen and a.freeze().is_frozen and not a.training
    with pytest.raises(ValueError):
        a(torch.randn(2, 32, 40, 3))          # channel-last
    with pytest.raises(TypeError):
        a(torch.zeros(2, 3, 8, 8, dtype=torch.uint8))
    with pytest.raises(ValueError):
        TinyConvEncoder(pool="max")


def test_spatial_softmax_finds_peak():
    ss = SpatialSoftmax(2, None, temperature=0.01, learnable_temperature=False)
    feat = torch.zeros(1, 2, 5, 8)
    feat[0, 0, 0, 7] = 10.0     # top-right cell
    feat[0, 1, 4, 0] = 10.0     # bottom-left cell
    kp = ss(feat)
    assert kp.shape == (1, 2, 2)
    np.testing.assert_allclose(kp[0, 0].numpy(), [1 - 1 / 8, -1 + 1 / 5], atol=1e-4)
    np.testing.assert_allclose(kp[0, 1].numpy(), [-1 + 1 / 8, 1 - 1 / 5], atol=1e-4)


def test_sincos_pos_embed():
    pe = sincos_pos_embed_2d(3, 4, 18)
    assert pe.shape == (12, 18) and torch.all(pe[:, 16:] == 0)
    assert torch.unique(pe[:, :16], dim=0).shape[0] == 12          # every cell distinct
    np.testing.assert_allclose(pe[0, :16].numpy(), [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4, atol=1e-7)


def test_build_vision_encoder_tiny_and_validation():
    enc = build_vision_encoder({"type": "tiny", "out_dim": 24, "grid": [2, 2]})
    assert isinstance(enc, TinyConvEncoder) and enc.n_tokens(40, 40) == 4 and enc.out_dim == 24
    assert isinstance(build_vision_encoder(), TinyConvEncoder)
    with pytest.raises(ValueError, match="unknown vision encoder option"):
        build_vision_encoder({"type": "tiny", "bogus": 1})
    with pytest.raises(ValueError, match="unknown vision encoder type"):
        build_vision_encoder({"type": "vgg"})
    assert sanitize_key("facebook/dinov2-small") == "facebook_dinov2-small"


@pytest.mark.parametrize("cfg,module", [
    ({"type": "resnet18"}, "torchvision"),
    ({"type": "resnet", "name": "resnet34", "pretrained": False}, "torchvision"),
    ({"type": "hf", "model_id": "facebook/dinov2-small"}, "transformers"),
    ({"type": "siglip"}, "transformers"),
])
def test_optional_backbones_raise_clear_import_error(monkeypatch, cfg, module):
    monkeypatch.setitem(sys.modules, module, None)   # import fails even if installed
    with pytest.raises(ImportError) as ei:
        build_vision_encoder(cfg)
    msg = str(ei.value)
    assert f"`{module}`" in msg and "pip install" in msg and "'tiny'" in msg


# ── transforms ────────────────────────────────────────────────────────────────
def test_to_float_tensor_range_shape_and_layouts(tmp_path):
    x = _frames(3, 10, 12)
    x[0, 0, 0] = [0, 128, 255]
    t = to_float_tensor(x)
    assert t.shape == (3, 3, 10, 12) and t.dtype == torch.float32
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0
    np.testing.assert_allclose(t[0, :, 0, 0].numpy(), [0, 128 / 255, 1.0], atol=1e-7)
    assert torch.equal(to_float_tensor(torch.from_numpy(x)), t)
    assert to_float_tensor(x[0]).shape == (3, 10, 12)                      # single image
    assert to_float_tensor(x[None]).shape == (1, 3, 3, 10, 12)              # extra leading dim
    assert torch.equal(to_float_tensor(t), t)                               # float CHW passthrough
    np.save(tmp_path / "x.npy", x)
    ro = np.load(tmp_path / "x.npy", mmap_mode="r")                         # read-only memmap ok
    assert torch.equal(to_float_tensor(ro), t)
    with pytest.raises(ValueError):
        to_float_tensor(np.zeros((2, 8, 8, 4), np.uint8))
    with pytest.raises(TypeError):
        to_float_tensor(np.zeros((2, 8, 8, 3), np.int32))


def test_normalize_roundtrip_and_resize_crop():
    t = to_float_tensor(_frames(2, 20, 30))
    n = Normalize()
    y = n(t)
    np.testing.assert_allclose(y[:, 0].mean().item(), ((t[:, 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]).mean().item(),
                               rtol=1e-5)
    assert torch.allclose(n.inverse(y), t, atol=1e-6)
    assert resize(t, (10, 15)).shape == (2, 3, 10, 15)
    assert resize_short(t, 10).shape == (2, 3, 10, 15)
    assert center_crop(t, 16).shape == (2, 3, 16, 16)
    assert torch.equal(center_crop(t, (20, 30)), t)
    with pytest.raises(ValueError):
        center_crop(t, 40)


def test_crop_resize_identity_and_2x_is_box_filter():
    t = to_float_tensor(_frames(2, 12, 16))
    one, zero = torch.ones(2), torch.zeros(2)
    assert torch.allclose(crop_resize(t, one, one, zero, zero, (12, 16)), t, atol=1e-5)
    ev = EvalTransform((6, 8), mean=None, std=None)          # exact 2× → bilinear = 2×2 average
    assert torch.allclose(ev(_frames(2, 12, 16)), F.avg_pool2d(t, 2), atol=1e-5)


def test_eval_transform_centre_crop_and_normalization():
    x = _frames(4, 24, 32)
    ev = EvalTransform((16, 16))                             # short-side centre crop 24×24 → 16×16
    y = ev(x)
    assert y.shape == (4, 3, 16, 16) and torch.equal(ev(x), y)
    raw = EvalTransform((16, 16), mean=None, std=None)(x)
    assert torch.allclose(Normalize()(raw), y, atol=1e-5)
    assert 0.0 <= float(raw.min()) and float(raw.max()) <= 1.0
    ref = resize(center_crop(to_float_tensor(x), 24), 16, antialias=False)
    assert (raw - ref).abs().mean() < 0.05                   # same view as crop+resize
    big = EvalTransform(64)(_frames(1, 480, 640))            # anti-aliased prefilter path
    assert big.shape == (1, 3, 64, 64) and torch.isfinite(big).all()


def test_train_augment_seeded_determinism_and_range():
    x = _frames(6, 24, 32)
    a1 = TrainAugment((20, 20), seed=0, saturation=0.3)
    a2 = TrainAugment((20, 20), seed=0, saturation=0.3)
    y1, y2 = a1(x), a2(x)
    assert y1.shape == (6, 3, 20, 20) and torch.equal(y1, y2)
    assert not torch.equal(a1(x), y1)                                      # generator advances
    assert not torch.equal(TrainAugment((20, 20), seed=1, saturation=0.3)(x), y1)
    g = torch.Generator().manual_seed(5)
    y3 = a1(x, generator=g)
    assert torch.equal(y3, a2(x, generator=torch.Generator().manual_seed(5)))
    un = Normalize().inverse(y1)
    assert float(un.min()) >= -1e-5 and float(un.max()) <= 1 + 1e-5       # valid image after undoing norm
    raw = TrainAugment((20, 20), seed=0, mean=None, std=None)(x)
    assert 0.0 <= float(raw.min()) and float(raw.max()) <= 1.0
    torch.manual_seed(3)
    r1 = TrainAugment((20, 20))(x)                                         # global RNG path
    torch.manual_seed(3)
    assert torch.equal(TrainAugment((20, 20))(x), r1)
    b = pickle.loads(pickle.dumps(a1))                                     # generator state survives pickling
    assert torch.equal(b(x), a1(x))


def test_train_augment_shares_params_over_history_dim():
    x = _frames(4, 24, 32)
    hist = np.stack([x, x, x], axis=1)                                     # [B=4, T=3, H, W, 3]
    y = TrainAugment((16, 16), seed=0)(hist)
    assert y.shape == (4, 3, 3, 16, 16)
    assert torch.equal(y[:, 0], y[:, 1]) and torch.equal(y[:, 0], y[:, 2])
    assert not torch.equal(y[0, 0], y[1, 0])
    assert TrainAugment(None, seed=0)(x[0]).shape == (3, 24, 32)          # single image, native size


def test_train_augment_validation_and_build_transforms():
    with pytest.raises(ValueError):
        TrainAugment(scale=(0.0, 1.0))
    with pytest.raises(ValueError):
        TrainAugment(brightness=1.5)
    enc = TinyConvEncoder(8)
    tr, ev = build_transforms({"image_size": [16, 20], "seed": 0, "scale": [0.6, 1.0]}, encoder=enc)
    assert isinstance(tr, TrainAugment) and isinstance(ev, EvalTransform)
    assert ev.crop_scale == pytest.approx(0.8) and tr(_frames(2)).shape == (2, 3, 16, 20)
    same, ev2 = build_transforms({"augment": False, "image_size": 16})
    assert same is ev2
    with pytest.raises(ValueError, match="unknown transform option"):
        build_transforms({"imgsize": 3})


# ── feature cache ─────────────────────────────────────────────────────────────
def _episode(tmp_path, n_frames=7, T=40, cams=("ego",), jpg=False):
    meta = EpisodeMeta(episode_id="ep0", dataset="task", kind="glove", layout="glove_template",
                       n_taxels=4, cameras=list(cams))
    t = np.arange(T) / 200.0
    arrays = {E.K_T: t}
    for c in cams:
        arrays[E.cam_idx_key(c)] = np.clip(np.arange(T) // 6 - 1, -1, n_frames - 1).astype(np.int32)
    root = Episode(meta, arrays).save(tmp_path / "task" / "ep0")
    frames = {}
    for i, c in enumerate(cams):
        d = root / f"camera_{c}"
        d.mkdir()
        fr = _frames(n_frames, 24, 32, seed=i)
        frames[c] = fr
        if jpg:
            from PIL import Image
            for k, f in enumerate(fr):
                Image.fromarray(f).save(d / f"{k:06d}.jpg", quality=95)
        else:
            np.save(d / "frames.npy", fr)
        np.save(d / "timestamps.npy", np.arange(n_frames) * 6 / 200.0)
    return Episode.load(root), frames


def test_feature_cache_write_read_matches_encoder(tmp_path):
    torch.manual_seed(0)
    ep, frames = _episode(tmp_path)
    enc = TinyConvEncoder(out_dim=16, grid=(2, 3)).freeze()
    tf = EvalTransform((16, 16), mean=enc.mean, std=enc.std)
    path = cache_episode_features(ep, "ego", enc, tf, batch_size=3, key="tiny_test")
    assert path == feature_path(ep, "ego", "tiny_test") == ep.root / "derived" / "vision_tiny_test_ego.npy"
    feats = load_cached(ep, "ego", "tiny_test")
    assert feats.shape == (7, 6, 16) and feats.dtype == np.float16 and has_cached(ep.root, "ego", "tiny_test")
    with torch.no_grad():
        ref = enc(tf(frames["ego"])).numpy()
    np.testing.assert_allclose(feats.astype(np.float32), ref, rtol=1e-2, atol=1e-2)
    info = load_cache_info(ep, "ego", "tiny_test")
    assert info["n_frames"] == 7 and info["shape"] == [7, 6, 16] and info["encoded_hw"] == [16, 16]
    assert not list((ep.root / "derived").glob("*.tmp"))
    # per frame (not per master tick): master index → frame via cam_idx
    idx = ep[E.cam_idx_key("ego")]
    x, valid = gather_frame_features(feats, idx)
    assert x.shape == (40, 6, 16) and x.dtype == np.float32
    assert not valid[:6].any() and valid[6:].all() and np.all(x[:6] == 0)
    np.testing.assert_array_equal(x[20], feats[idx[20]].astype(np.float32))
    with pytest.raises(IndexError):
        gather_frame_features(feats, [7])


def test_feature_cache_skip_overwrite_stale_and_default_key(tmp_path):
    ep, _ = _episode(tmp_path, cams=("ego", "third"))
    enc = TinyConvEncoder(out_dim=8).freeze()
    p = cache_episode_features(ep.root, "ego", enc)          # path input, default key + transform
    assert p.name == f"vision_{enc.cache_key}_ego.npy"
    assert load_cached(ep, "ego", enc.cache_key).shape == (7, enc.n_tokens(24, 32), 8)
    mtime = p.stat().st_mtime_ns
    cache_episode_features(ep, "ego", enc)                    # cached → skipped
    assert p.stat().st_mtime_ns == mtime
    cache_episode_features(ep, "ego", enc, overwrite=True)
    # stale cache (frame count changed after re-preprocessing) is detected
    np.save(p, np.zeros((3, 16, 8), np.float16))
    assert not has_cached(ep, "ego", enc.cache_key)
    with pytest.raises(ValueError, match="stale"):
        load_cached(ep, "ego", enc.cache_key)
    with pytest.raises(FileNotFoundError):
        load_cached(ep, "third", enc.cache_key)
    paths = cache_features([ep.root], None, enc, overwrite=True)  # both cameras from meta
    assert sorted(q.name for q in paths) == sorted([f"vision_{enc.cache_key}_ego.npy",
                                                   f"vision_{enc.cache_key}_third.npy"])
    with pytest.raises(FileNotFoundError):
        cache_episode_features(ep, "wrist", enc)
    with pytest.raises(ValueError):
        feature_path(ep, "ego", "bad/key")
    with pytest.raises(ValueError):
        cache_episode_features(ep, "ego", enc, key="bad/key")          # explicit keys are validated
    with pytest.raises(ValueError):
        cache_episode_features(Episode(ep.meta, {E.K_T: ep.t}), "ego", enc)   # not on disk
    with pytest.warns(UserWarning, match="trainable"):
        cache_episode_features(ep, "ego", TinyConvEncoder(out_dim=8), key="trainable")


def test_feature_cache_jpg_frames_and_cli(tmp_path, capsys):
    ep, frames = _episode(tmp_path, n_frames=3, jpg=True)
    enc = TinyConvEncoder(out_dim=8, grid=(1, 1)).freeze()
    cache_episode_features(ep, "ego", enc, key="jpg")
    assert load_cached(ep, "ego", "jpg").shape == (3, 1, 8)
    rc = FC.main(["--root", str(tmp_path), "--dataset", "task",
                  "--encoder", json.dumps({"type": "tiny", "out_dim": 4, "grid": [1, 2]}),
                  "--image-size", "16", "16", "--key", "cli"])
    assert rc == 0 and "cached 1 camera streams" in capsys.readouterr().out
    assert load_cached(ep, "ego", "cli").shape == (3, 2, 4)


# ── exact geometry, edge cases, optional backends via fake modules ──────────────
def test_eval_crop_is_pixel_exact_and_scale_is_relative_to_largest_aspect_crop():
    x = _frames(3, 24, 32)
    t = to_float_tensor(x)
    raw = dict(mean=None, std=None, antialias=False)
    # 24×32 → square: largest square crop is 24×24 at cols 4..28 (1:1 sampling → exact)
    assert torch.allclose(EvalTransform((24, 24), **raw)(x), t[..., 4:28], atol=1e-6)
    # crop_scale 0.25 of that square = 12×12 centred box: rows 6..18, cols 10..22
    assert torch.allclose(EvalTransform((12, 12), crop_scale=0.25, **raw)(x), t[..., 6:18, 10:22], atol=1e-6)
    # wide output (aspect 2): full width, 16 rows centred
    assert torch.allclose(EvalTransform((16, 32), **raw)(x), t[..., 4:20, :], atol=1e-6)
    # stretch: whole image squashed, exact 2× → 2×2 box filter
    assert torch.allclose(EvalTransform((12, 16), stretch=True, mean=None, std=None)(x), F.avg_pool2d(t, 2), atol=1e-5)
    # crop_box: 4:3 camera → square output; area fraction is relative to the 480×480 max crop
    s = torch.tensor([1.0, 0.81, 0.25])
    fw, fh = crop_box((480, 640), (224, 224), s, torch.zeros(3))
    np.testing.assert_allclose((fw * 640).numpy(), [480, 432, 240], rtol=1e-6)
    np.testing.assert_allclose((fh * 480).numpy(), [480, 432, 240], rtol=1e-6)


def test_train_augment_zoom_active_for_mismatched_aspect_and_boxes_inside():
    aug = TrainAugment((224, 224), scale=(0.3, 1.0), ratio=(0.5, 2.0), seed=0)
    p = aug.sample_params(4000)
    fw, fh = crop_box((480, 640), (224, 224), p["scale"], p["log_ratio"])
    cx, cy = p["tx"] * (1 - fw), p["ty"] * (1 - fh)
    assert bool(((cx.abs() + fw) <= 1 + 1e-9).all() and ((cy.abs() + fh) <= 1 + 1e-9).all())
    np.testing.assert_allclose(((fw * 640) / (fh * 480)).numpy(), p["log_ratio"].exp().numpy(), rtol=1e-9)
    # default scale (0.8, 1.0) must actually zoom a 4:3 frame cropped to a square
    d = TrainAugment((224, 224), seed=0).sample_params(500)
    fw, fh = crop_box((480, 640), (224, 224), d["scale"], d["log_ratio"])
    rel_area = (fw * 640) * (fh * 480) / 480 ** 2
    assert float(rel_area.min()) < 0.82 and float(rel_area.max()) > 0.98
    np.testing.assert_allclose(rel_area.numpy(), d["scale"].numpy(), rtol=1e-9)


def test_anisotropic_downscale_is_antialiased_and_empty_batches():
    x = _frames(2, 480, 640)
    y = EvalTransform((64, 16), stretch=True, mean=None, std=None)(x)
    ref = F.interpolate(to_float_tensor(x), size=(64, 16), mode="bilinear", antialias=True, align_corners=False)
    assert float((y - ref).abs().mean()) < 1e-3
    empty = np.zeros((0, 24, 32, 3), np.uint8)
    assert TrainAugment((16, 16), seed=0)(empty).shape == (0, 3, 16, 16)
    assert TrainAugment((16, 16), seed=0)(np.zeros((0, 2, 24, 32, 3), np.uint8)).shape == (0, 2, 3, 16, 16)
    assert EvalTransform((16, 16))(empty).shape == (0, 3, 16, 16)
    assert resize(torch.zeros(0, 3, 8, 8), 4).shape == (0, 3, 4, 4)


def test_train_augment_reseeds_per_worker_and_epoch(monkeypatch):
    x = _frames(4, 24, 32)
    info = types.SimpleNamespace(seed=100)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: info)
    w0 = TrainAugment((16, 16), seed=0)(x)
    assert torch.equal(TrainAugment((16, 16), seed=0)(x), w0)             # same worker seed → same
    info.seed = 101
    assert not torch.equal(TrainAugment((16, 16), seed=0)(x), w0)         # other worker
    aug = TrainAugment((16, 16), seed=0)
    info.seed = 100
    a = aug(x)
    info.seed = 200                                                        # next epoch → new stream
    assert not torch.equal(aug(x), a)


def test_sincos_hand_values_and_cpu_cache_is_not_aliased():
    pe = sincos_pos_embed_2d(2, 3, 8)          # d4 = 2 → ω = [1, 1/100]
    cell = pe[1 * 3 + 2]                       # row y=1, col x=2
    expect = [math.sin(2), math.sin(0.02), math.cos(2), math.cos(0.02),
              math.sin(1), math.sin(0.01), math.cos(1), math.cos(0.01)]
    np.testing.assert_allclose(cell.numpy(), expect, rtol=1e-6)
    pe.zero_()                                 # mutating a result must not corrupt the cache
    assert torch.allclose(sincos_pos_embed_2d(2, 3, 8)[5], torch.tensor(expect, dtype=torch.float32))
    assert sincos_pos_embed_2d(2, 3, 8, dtype=torch.float64).dtype == torch.float64


class _Patchify(VisionEncoder):
    """No analytic n_tokens → base-class fallback (one dummy forward)."""

    def __init__(self):
        super().__init__()
        self.out_dim = 5
        self.conv = nn.Conv2d(3, 5, 4, 4)

    def forward(self, images):
        self._check(images)
        return self.conv(images).flatten(2).transpose(1, 2)


def test_base_n_tokens_fallback_and_tiny_out_dim_none():
    enc = _Patchify().double()
    assert enc.n_tokens(17, 33) == (17 // 4) * (33 // 4) and enc.training   # float64 params ok
    tiny = build_vision_encoder({"type": "tiny", "out_dim": None, "channels": [8, 12], "pool": "avg"})
    assert tiny.out_dim == 12 and tiny(torch.randn(1, 3, 16, 16)).shape == (1, 1, 12)


# fake torchvision: same attribute layout as torchvision.models.resnet*
class _FakeResNet(nn.Module):
    def __init__(self, weights=None):
        super().__init__()

        def blk(ci, co, s):
            return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU())
        self.conv1, self.bn1 = nn.Conv2d(3, 8, 7, 2, 3, bias=False), nn.BatchNorm2d(8)
        self.relu, self.maxpool = nn.ReLU(), nn.MaxPool2d(3, 2, 1)
        self.layer1, self.layer2 = blk(8, 8, 1), blk(8, 16, 2)
        self.layer3, self.layer4 = blk(16, 16, 2), blk(16, 32, 2)
        self.fc = nn.Linear(32, 10)


@pytest.fixture
def fake_torchvision(monkeypatch):
    mod = types.SimpleNamespace(models=types.SimpleNamespace(resnet18=_FakeResNet, resnet34=_FakeResNet,
                                                              resnet50=_FakeResNet))
    monkeypatch.setitem(sys.modules, "torchvision", mod)
    return mod


@pytest.mark.parametrize("pool", POOL_MODES)
def test_resnet_wrapper_token_count_matches_forward(fake_torchvision, pool):
    torch.manual_seed(0)
    enc = build_vision_encoder({"type": "resnet18", "pool": pool, "out_dim": 24 if pool != "none" else None,
                                "n_keypoints": 5})
    assert isinstance(enc, ResNetEncoder)
    for hw in [(96, 128), (65, 33), (224, 224)]:
        out = enc(torch.randn(2, 3, *hw))
        assert out.shape == (2, enc.n_tokens(*hw), enc.out_dim)
    if pool == "none":   # stride 32, ceil
        assert enc.n_tokens(65, 33) == 3 * 2 and enc.out_dim == 32 and enc.is_frozen


def test_resnet_wrapper_frozen_bn_unfreeze_and_group_norm(fake_torchvision):
    torch.manual_seed(0)
    enc = build_vision_encoder({"type": "resnet18"})
    assert enc.frozen_backbone and enc.cache_key == "resnet18_none"
    enc.train()
    assert enc.training and not enc.backbone.training                    # frozen trunk: BN in eval
    rm = enc.backbone[1].running_mean.clone()
    out = enc(torch.randn(4, 3, 64, 64))
    assert torch.equal(rm, enc.backbone[1].running_mean) and not out.requires_grad
    for p in enc.backbone.parameters():                                   # manual unfreeze is honoured
        p.requires_grad_(True)
    enc.train()
    assert not enc.frozen_backbone and enc.backbone.training
    enc(torch.randn(2, 3, 64, 64)).sum().backward()
    assert enc.backbone[0].weight.grad is not None
    with pytest.warns(UserWarning, match="GroupNorm"):
        build_vision_encoder({"type": "resnet18", "group_norm": True})    # pretrained + frozen + GN
    gn = build_vision_encoder({"type": "resnet34", "group_norm": True, "frozen": False, "pretrained": False})
    assert not any(isinstance(m, nn.BatchNorm2d) for m in gn.modules())
    assert gn.cache_key == "resnet34_scratch_gn_none"


# fake transformers (vision side): ViT-like model with dropout + a dual encoder
class _FakeViT(nn.Module):
    def __init__(self, hidden=12, patch=4):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden)
        self.patch = nn.Conv2d(3, hidden, patch, patch)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden))
        self.drop = nn.Dropout(0.5)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        t = self.patch(pixel_values).flatten(2).transpose(1, 2)
        t = self.drop(torch.cat([self.cls.expand(len(t), -1, -1), t], 1))
        return types.SimpleNamespace(last_hidden_state=t, pooler_output=t[:, 0] * 2)


class _FakeDual(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace()
        self.vision_model, self.text_model = _FakeViT(), nn.Linear(2, 2)


def _fake_transformers(model_factory):
    auto = types.SimpleNamespace(from_pretrained=lambda mid, **kw: model_factory(),
                                 from_config=lambda cfg: model_factory())
    return types.SimpleNamespace(
        AutoModel=auto, AutoConfig=types.SimpleNamespace(from_pretrained=lambda mid, **kw: None),
        AutoImageProcessor=types.SimpleNamespace(
            from_pretrained=lambda mid, **kw: types.SimpleNamespace(image_mean=[0.5] * 3, image_std=[0.25] * 3)))


def test_hf_vision_wrapper_with_fake_transformers(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(_FakeDual))
    enc = build_vision_encoder({"type": "siglip"})                        # dual → vision tower
    assert isinstance(enc, HFVisionEncoder) and isinstance(enc.backbone, _FakeViT)
    assert enc.mean == (0.5, 0.5, 0.5) and enc.std == (0.25, 0.25, 0.25)
    assert enc.cache_key == "siglip-base-patch16-224" and enc.is_frozen and enc._interp
    x = torch.randn(2, 3, 16, 20)
    enc.train()                                                           # frozen tower: no dropout
    y1, y2 = enc(x), enc(x)
    assert torch.equal(y1, y2) and y1.shape == (2, 1 + 4 * 5, 12) and not y1.requires_grad
    assert enc.n_tokens(16, 20) == 21 and enc.n_tokens(8, 8) == 5
    pooled = build_vision_encoder({"type": "hf", "model_id": "org/fake-vit", "tokens": "pooled", "out_dim": 6,
                                   "pretrained": False, "revision": "abc"})
    assert pooled(x).shape == (2, 1, 6) and not pooled.is_frozen and pooled.frozen_backbone
    assert pooled.cache_key == "fake-vit_rev-abc_scratch_pooled"
    for p in pooled.backbone.parameters():
        p.requires_grad_(True)
    pooled(x).sum().backward()
    assert pooled.backbone.patch.weight.grad is not None


# ── feature cache: staleness via the sidecar, frames not copied ───────────────
def test_feature_cache_signature_mismatch_and_missing_frames(tmp_path):
    torch.manual_seed(0)
    ep, _ = _episode(tmp_path)
    enc = TinyConvEncoder(out_dim=8, grid=(2, 2)).freeze()
    tf16 = EvalTransform((16, 16), mean=enc.mean, std=enc.std)
    cache_episode_features(ep, "ego", enc, tf16, key="k")
    info = load_cache_info(ep, "ego", "k")
    assert info["transform"] == repr(tf16) and "0x" not in info["transform"]
    cache_episode_features(ep, "ego", enc, EvalTransform((16, 16), mean=enc.mean, std=enc.std), key="k")  # same → skip
    with pytest.raises(ValueError, match="built differently"):
        cache_episode_features(ep, "ego", enc, EvalTransform((24, 24), mean=enc.mean, std=enc.std), key="k")
    with pytest.raises(ValueError, match="built differently"):
        cache_episode_features(ep, "ego", TinyConvEncoder(out_dim=4).freeze(), tf16, key="k")
    p = cache_episode_features(ep, "ego", enc, EvalTransform((24, 24), mean=enc.mean, std=enc.std), key="k",
                               overwrite=True)
    assert load_cached(ep, "ego", "k").shape[0] == 7 and load_cache_info(ep, "ego", "k")["encoded_hw"] == [24, 24]
    # a callable without a stable repr is not compared (no false "built differently")
    cache_episode_features(ep, "ego", enc, lambda im: tf16(im), key="lam")
    cache_episode_features(ep, "ego", enc, lambda im: tf16(im), key="lam")
    # training box without frames: timestamps only, then nothing at all → sidecar count
    (ep.root / "camera_ego" / "frames.npy").unlink()
    assert load_cached(ep, "ego", "k").shape[0] == 7 and has_cached(ep, "ego", "k")
    (ep.root / "camera_ego" / "timestamps.npy").unlink()
    (ep.root / "camera_ego").rmdir()
    assert load_cached(ep, "ego", "k").shape[0] == 7 and has_cached(ep, "ego", "k")
    np.save(p, np.zeros((3, 4, 8), np.float16))                          # truncated → sidecar says 7
    with pytest.raises(ValueError, match="stale"):
        load_cached(ep, "ego", "k")
