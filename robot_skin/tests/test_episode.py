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


def test_derived_writes_are_atomic_and_pseudo_label_key(tmp_path):
    """``set_derived`` replaces the file (temp + os.replace): a reader holding a memmap of the old
    array keeps valid data, and no temp file is left behind. The contact stage's pseudo labels are
    part of the contract (``D_CONTACT_LABEL_PSEUDO``, re-exported by ``contact.pseudo_label``)."""
    from robot_skin.contact import pseudo_label
    from robot_skin.vtla.dataset import PSEUDO_LABEL_KEY

    assert E.D_CONTACT_LABEL_PSEUDO == "contact_label_pseudo"
    assert pseudo_label.D_CONTACT_LABEL_PSEUDO is E.D_CONTACT_LABEL_PSEUDO == PSEUDO_LABEL_KEY
    ep = _ep()
    root = ep.save(tmp_path / "e1")
    ep.set_derived(E.D_CONTACT_LABEL_PSEUDO, np.zeros((50, 4), np.int8))
    reader = Episode.load(root)
    old = reader.derived(E.D_CONTACT_LABEL_PSEUDO)
    assert isinstance(old, np.memmap)
    ep.set_derived(E.D_CONTACT_LABEL_PSEUDO, np.ones((50, 4), np.int8))            # a stage re-run
    assert int(np.asarray(old).sum()) == 0                                          # old view intact
    assert int(Episode.load(root).derived(E.D_CONTACT_LABEL_PSEUDO).sum()) == 200   # new data on disk
    assert sorted(p.name for p in (root / "derived").iterdir()) == [f"{E.D_CONTACT_LABEL_PSEUDO}.npy"]
    with pytest.raises(ValueError, match="rows"):
        ep.set_derived(E.D_RESIDUAL, np.zeros((3, 4)))


def test_datasets_package_is_lazy():
    """``import robot_skin.datasets`` loads only the Episode contract; build / splits / stats / motion
    resolve on first access (PEP 562) and the synthetic generator is never imported eagerly."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys\n"
        "import robot_skin.datasets as D\n"
        "heavy = ['robot_skin.datasets.' + m for m in ('build', 'splits', 'stats', 'motion', 'synthetic')]\n"
        "assert not [m for m in heavy if m in sys.modules], [m for m in heavy if m in sys.modules]\n"
        "assert 'torch' not in sys.modules\n"
        "assert D.make_splits.__module__ == 'robot_skin.datasets.splits'\n"
        "assert 'robot_skin.datasets.build' not in sys.modules and 'torch' not in sys.modules\n"
        "assert D.preprocess_session.__module__ == 'robot_skin.datasets.build'\n"
        "assert 'robot_skin.datasets.synthetic' not in sys.modules\n"
        "assert D.BaselineWindowDataset.__name__ == 'BaselineWindowDataset'\n"
        "assert D.stats.compute_stats is D.compute_stats\n"
        "assert 'synthetic' in dir(D) and 'build_all' in D.__all__\n"
        "try:\n"
        "    D.no_such_name\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('no AttributeError')\n"
        "assert D.synthetic.generate_session\n"
        "print('ok')\n"
    )
    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(repo), os.environ.get("PYTHONPATH", "")])}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=repo, env=env, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]
