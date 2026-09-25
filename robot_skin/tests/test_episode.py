import numpy as np
import pytest

from robot_skin.datasets import Episode, EpisodeMeta, list_episodes
from robot_skin.datasets import episode as E


def _ep(T=50, N=4):
    meta = EpisodeMeta(episode_id="e1", dataset="motion", kind="glove", layout="glove_template",
                       n_taxels=N, cameras=["ego"], phase_names=["rest", "move"],
                       task={"task_id": "x", "instruction": "pick up the cup"})
    t = np.arange(T) / 200.0
    arrays = {E.K_T: t, E.K_DELTA: np.zeros((T, N), np.float32),
              E.K_PHASE: np.where(t < 0.1, 0, 1).astype(np.int16),
              E.cam_idx_key("ego"): np.clip(np.arange(T) // 7 - 1, -1, None).astype(np.int32)}
    return Episode(meta, arrays, static={E.S_BASELINE_RAW: np.ones(N)})


def test_roundtrip_mmap_and_derived(tmp_path):
    ep = _ep()
    ep.set_derived(E.D_RESIDUAL, np.ones((50, 4), np.float32), save=False)
    root = ep.save(tmp_path / "motion" / "e1")
    frames = np.random.default_rng(0).integers(0, 255, (8, 6, 5, 3), dtype=np.uint8)
    (root / "camera_ego").mkdir()
    np.save(root / "camera_ego" / "frames.npy", frames)
    np.save(root / "camera_ego" / "timestamps.npy", np.arange(8) * 7 / 200.0)
    ep2 = Episode.load(root)
    assert ep2.T == 50 and ep2.meta.instruction == "pick up the cup"
    assert isinstance(ep2[E.K_DELTA], np.memmap)
    np.testing.assert_array_equal(ep2.static[E.S_BASELINE_RAW], np.ones(4))
    assert ep2.has_derived(E.D_RESIDUAL) and ep2.derived(E.D_RESIDUAL).sum() == 200
    assert ep2.frame_at("ego", 0) is None
    np.testing.assert_array_equal(ep2.frame_at("ego", 20), frames[20 // 7 - 1])
    assert ep2.phase_mask("move").sum() == 30 and not ep2.phase_mask("nope").any()
    assert list_episodes(tmp_path) == [root] and list_episodes(tmp_path, "task") == []
    assert set(Episode.load(root, keys=[E.K_DELTA]).arrays) == {E.K_T, E.K_DELTA}


def test_validation():
    ep = _ep()
    with pytest.raises(ValueError):
        Episode(ep.meta, {**ep.arrays, E.K_DELTA: np.zeros((49, 4))})
    with pytest.raises(ValueError):
        Episode(ep.meta, {**ep.arrays, E.K_DELTA: np.zeros((50, 3))})
    with pytest.raises(ValueError):
        ep.set_derived("x", np.zeros(3))
