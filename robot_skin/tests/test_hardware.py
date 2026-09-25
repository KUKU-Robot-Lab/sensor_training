"""Tests for robot_skin.train.hardware (device / precision / profiles / arch checks) and
robot_skin.train.distributed (single-process no-op paths only — no process groups spawned)."""
from __future__ import annotations

import json
import os
import warnings

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import IterableDataset, TensorDataset

from robot_skin.train import (DistInfo, TrainConfig, all_reduce_mean, all_reduce_sum,
                              apply_hw_profile, apply_profile_env, barrier, check_arch_support,
                              cleanup, describe_environment, detect_hw_profile, enable_tf32,
                              init_distributed, list_hw_profiles, load_hw_profile,
                              make_eval_sampler, make_sampler, maybe_apply_hw_profile,
                              resolve_device, resolve_precision, unwrap_model, wrap_ddp)
from robot_skin.train.distributed import ShardSampler, is_dist_initialized

NO_CUDA = not torch.cuda.is_available()
GPU_PROFILES = ("rtx5090", "rtx4090", "rtx3090", "a100")
STAGES = ("imu_pose", "baseline", "contact", "pretrain", "vtla")


# ───────────────────────────────────────────────────────────────────────────── device

def test_resolve_device_cpu_and_auto():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")).type == "cpu"
    auto = resolve_device("auto")
    if NO_CUDA and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        assert auto == torch.device("cpu")
    assert resolve_device(None).type == auto.type


@pytest.mark.skipif(not NO_CUDA, reason="checks the no-CUDA error path")
def test_resolve_device_cuda_unavailable_raises():
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError):
        resolve_device("cuda:1")


def test_resolve_device_invalid():
    for bad in ("tpu", "cuda:x", "gpu0"):
        with pytest.raises(ValueError):
            resolve_device(bad)


# ───────────────────────────────────────────────────────────────────────────── precision

def test_resolve_precision_cpu_rules():
    p = resolve_precision("auto", "cpu")
    assert (p.name, p.autocast_dtype, p.use_grad_scaler, p.enabled) == ("fp32", None, False, False)
    p = resolve_precision("bf16", "cpu")
    assert p.autocast_dtype is torch.bfloat16 and not p.use_grad_scaler
    with pytest.warns(UserWarning):
        assert resolve_precision("fp16", "cpu").name == "fp32"
    assert resolve_precision("fp32", "cpu").make_scaler() is None
    with pytest.warns(UserWarning):
        assert resolve_precision("bf16", "mps").name == "fp32"
    assert resolve_precision("auto", "mps").name == "fp32"
    with pytest.raises(ValueError, match="precision"):
        resolve_precision("int8", "cpu")


@pytest.mark.parametrize("cap,name,scaler", [
    ((12, 0), "bf16", False),   # RTX 5090 (Blackwell)
    ((8, 9), "bf16", False),    # RTX 4090 (Ada)
    ((8, 6), "bf16", False),    # RTX 3090 (Ampere)
    ((8, 0), "bf16", False),    # A100
    ((7, 5), "fp16", True),     # Turing (T4 / RTX 20xx)
    ((7, 0), "fp16", True),     # Volta (V100)
])
def test_resolve_precision_cuda_auto_by_capability(cap, name, scaler):
    p = resolve_precision("auto", torch.device("cuda", 0), capability=cap)
    assert p.name == name and p.use_grad_scaler is scaler and p.device_type == "cuda"
    assert p.autocast_dtype is (torch.bfloat16 if name == "bf16" else torch.float16)


def test_resolve_precision_cuda_explicit():
    dev = torch.device("cuda")
    with pytest.warns(UserWarning, match="bf16"):
        p = resolve_precision("bf16", dev, capability=(7, 5))
    assert p.name == "fp16" and p.use_grad_scaler
    assert resolve_precision("bfloat16", dev, capability=(12, 0)).name == "bf16"
    assert resolve_precision("fp16", dev, capability=(12, 0)).use_grad_scaler
    assert resolve_precision("fp32", dev, capability=(12, 0)).autocast_dtype is None


@pytest.mark.skipif(not NO_CUDA, reason="checks the no-CUDA error path")
def test_resolve_precision_cuda_without_capability_raises():
    with pytest.raises(RuntimeError, match="capability"):
        resolve_precision("auto", "cuda")


def test_precision_plan_autocast_cpu():
    a, b = torch.randn(4, 4), torch.randn(4, 4)
    with resolve_precision("bf16", "cpu").autocast():
        assert (a @ b).dtype == torch.bfloat16
    with resolve_precision("fp32", "cpu").autocast():
        assert (a @ b).dtype == torch.float32


def test_enable_tf32_roundtrip():
    prev = enable_tf32(True)
    try:
        assert torch.get_float32_matmul_precision() == "high"
        enable_tf32(False)
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(prev if prev != "unknown" else "highest")


# ───────────────────────────────────────────────────────────────────────────── profiles

def test_builtin_profiles_load_and_are_valid():
    names = list_hw_profiles()
    assert set(GPU_PROFILES) | {"cpu"} <= set(names)
    for n in set(GPU_PROFILES) | {"cpu"}:
        prof = load_hw_profile(n)
        assert prof["name"] == n
        with warnings.catch_warnings():
            warnings.simplefilter("error")            # every key must be a TrainConfig field
            TrainConfig.from_dict(prof["train"])
            for stage in STAGES:
                TrainConfig.from_dict(prof["suggest"][stage])
    p5090 = load_hw_profile("rtx5090")
    assert p5090["gpu"]["compute_capability"] == [12, 0] and p5090["gpu"]["memory_gb"] == 32
    assert p5090["train"]["precision"] == "bf16" and p5090["train"]["compile"] is True
    assert p5090["train"]["num_workers"] == 8
    cpu = load_hw_profile("cpu")
    assert cpu["train"]["precision"] == "fp32" and cpu["train"]["compile"] is False
    assert cpu["train"]["num_workers"] == 0
    assert load_hw_profile("rtx4090")["gpu"]["memory_gb"] == 24
    assert load_hw_profile("a100")["gpu"]["memory_gb"] == 80


@pytest.mark.parametrize("stage", STAGES)
def test_gpu_profiles_keep_effective_batch(stage):
    eff = {n: load_hw_profile(n)["suggest"][stage]["batch_size"]
           * load_hw_profile(n)["suggest"][stage]["grad_accum"] for n in GPU_PROFILES}
    assert len(set(eff.values())) == 1, eff


def test_load_hw_profile_path_and_errors(tmp_path):
    p = tmp_path / "mybox.yaml"
    p.write_text(yaml.safe_dump({"train": {"precision": "fp16", "num_workers": 2}}))
    prof = load_hw_profile(p)
    assert prof["name"] == "mybox" and prof["train"]["precision"] == "fp16"
    with pytest.raises(FileNotFoundError, match="rtx5090"):
        load_hw_profile("h100_does_not_exist")
    bad = tmp_path / "bad.yaml"
    bad.write_text("train: 3\n")
    with pytest.raises(ValueError):
        load_hw_profile(bad)


def test_apply_hw_profile_precedence():
    stage_cfg = {"stage": "vtla", "model": {"d": 8},
                 "train": {"batch_size": 999, "lr": 1e-4, "precision": "fp32"}}
    out = apply_hw_profile(stage_cfg, "rtx5090")
    assert out["train"]["batch_size"] == 64 and out["train"]["grad_accum"] == 2   # suggest
    assert out["train"]["precision"] == "bf16" and out["train"]["compile"] is True  # profile
    assert out["train"]["lr"] == 1e-4 and out["model"] == {"d": 8}                  # stage kept
    assert out["hardware"] == "rtx5090"
    assert stage_cfg["train"]["batch_size"] == 999                                   # not mutated
    no_stage = apply_hw_profile({"train": {"batch_size": 7}}, load_hw_profile("cpu"))
    assert no_stage["train"]["batch_size"] == 7 and no_stage["train"]["device"] == "cpu"
    assert apply_hw_profile({}, "cpu", stage="baseline")["train"]["batch_size"] == 256
    # the merged train section is a valid TrainConfig
    TrainConfig.from_dict(out["train"])


def test_maybe_apply_hw_profile():
    cfg = {"train": {"lr": 1.0}}
    same = maybe_apply_hw_profile(cfg)
    assert same == cfg and same is not cfg
    out = maybe_apply_hw_profile({"hardware": "cpu", "stage": "vtla", "train": {}})
    assert out["train"]["batch_size"] == 8 and out["train"]["precision"] == "fp32"


def test_apply_profile_env(monkeypatch):
    monkeypatch.delenv("RS_TEST_VAR", raising=False)
    monkeypatch.setenv("RS_KEEP", "user")
    prof = {"env": {"RS_TEST_VAR": "1", "RS_KEEP": "profile"}}
    applied = apply_profile_env(prof)
    assert applied == {"RS_TEST_VAR": "1"} and os.environ["RS_KEEP"] == "user"
    assert apply_profile_env(prof, override=True)["RS_KEEP"] == "profile"


def test_detect_hw_profile():
    assert detect_hw_profile(["NVIDIA GeForce RTX 5090"]) == "rtx5090"
    assert detect_hw_profile(["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090"]) == "rtx4090"
    assert detect_hw_profile(["NVIDIA GeForce RTX 3090"]) == "rtx3090"
    assert detect_hw_profile(["NVIDIA A100-SXM4-80GB"]) == "a100"
    assert detect_hw_profile(["Tesla T4"]) is None
    assert detect_hw_profile([]) == "cpu"


def test_detect_hw_profile_matches_whole_models_and_checks_memory():
    """Regression: plain substring matching mapped 'RTX A1000' (6-8 GB) to the 80 GB a100 profile
    and laptop 5090s (24 GB) to the 32 GB rtx5090 profile — an immediate OOM with hardware: auto."""
    for name in ("NVIDIA RTX A1000", "NVIDIA RTX A1000 6GB Laptop GPU", "NVIDIA GeForce RTX 5090 Laptop GPU",
                 "NVIDIA GeForce RTX 4090 Laptop GPU", "NVIDIA RTX A100X"):
        assert detect_hw_profile([name]) is None, name
    assert detect_hw_profile(["NVIDIA A100 80GB PCIe"], memory_gb=[79.2]) == "a100"
    assert detect_hw_profile(["NVIDIA A100-SXM4-40GB"], memory_gb=[39.4]) is None     # 80 GB profile
    assert detect_hw_profile(["NVIDIA GeForce RTX 5090"], memory_gb=[31.4]) == "rtx5090"
    assert detect_hw_profile(["NVIDIA GeForce RTX 3090 Ti"], memory_gb=[23.6]) == "rtx3090"


# ───────────────────────────────────────────────────────────────────────────── diagnostics

def test_describe_environment_is_json():
    info = describe_environment()
    json.dumps(info)
    for k in ("torch", "torch_cuda", "cuda_available", "gpus", "arch_list", "warnings", "env",
              "profile_guess"):
        assert k in info
    if NO_CUDA:
        assert info["gpus"] == [] and info["profile_guess"] == "cpu"


OLD_ARCHS = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


def test_check_arch_support_blackwell_needs_cu128():
    w = check_arch_support([(12, 0)], OLD_ARCHS, "12.4", ["NVIDIA GeForce RTX 5090"])
    assert len(w) == 1
    assert "sm_120" in w[0] and "12.8" in w[0] and "cu128" in w[0] and "RTX 5090" in w[0]
    assert check_arch_support([(12, 0)], OLD_ARCHS + ["sm_100", "sm_120"], "12.8") == []
    # unknown arch list but a CUDA < 12.8 build is still flagged for Blackwell
    w = check_arch_support([(12, 0)], [], "12.4")
    assert len(w) == 1 and "12.8" in w[0]
    assert check_arch_support([(12, 0)], [], "12.8") == []


def test_check_arch_support_compatibility_rules():
    assert check_arch_support([(8, 9)], ["sm_80", "sm_86"], "12.1") == []   # same-major cubin
    assert check_arch_support([(8, 9)], ["sm_75", "compute_75"], "11.8") == []  # PTX JIT
    assert check_arch_support([(9, 0)], ["sm_90a"], "12.4") == []           # arch-specific exact
    w = check_arch_support([(6, 1)], ["sm_75", "sm_80"], "12.8", ["GTX 1080"])
    assert len(w) == 1 and "GTX 1080" in w[0] and "cu128" not in w[0]
    w = check_arch_support([(12, 0)], ["sm_90a", "compute_90a"], "12.4")
    assert len(w) == 1                                                       # 'a' PTX: no forward compat
    assert check_arch_support([(8, 6), (12, 0)], OLD_ARCHS, "12.4")[0].startswith("GPU 1")
    # generic PTX could be JIT-ed, but a pre-12.8 toolkit build is still flagged for Blackwell
    w = check_arch_support([(12, 0)], OLD_ARCHS + ["compute_90"], "12.4")
    assert len(w) == 1 and "PTX" in w[0] and "12.8" in w[0]
    assert check_arch_support([(12, 0)], ["sm_90", "compute_90"], "12.8") == []


def test_check_arch_support_live_does_not_crash():
    assert isinstance(check_arch_support(), list)


# ───────────────────────────────────────────────────────────────────────────── distributed

@pytest.fixture
def no_dist_env(monkeypatch):
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_init_distributed_noop_without_env(no_dist_env):
    info = init_distributed()
    assert info == DistInfo(rank=0, world_size=1, local_rank=0, is_main=True, backend=None)
    assert not info.distributed and not is_dist_initialized()
    no_dist_env.setenv("WORLD_SIZE", "1")
    assert init_distributed().world_size == 1 and not is_dist_initialized()


def test_init_distributed_requires_torchrun_env(no_dist_env):
    no_dist_env.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="torchrun"):
        init_distributed()
    assert not is_dist_initialized()
    no_dist_env.setenv("WORLD_SIZE", "two")
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        init_distributed()


def test_distinfo_is_main_defaults_to_rank_zero():
    assert DistInfo().is_main is True and not DistInfo().distributed
    assert DistInfo(rank=1, world_size=2).is_main is False
    assert DistInfo(rank=1, world_size=2, is_main=True).is_main is True   # explicit wins
    with pytest.raises(ValueError, match="rank"):
        DistInfo(rank=2, world_size=2)


def test_wrap_and_unwrap_single_process():
    m = nn.Linear(2, 2)
    assert wrap_ddp(m, DistInfo()) is m
    assert unwrap_model(nn.DataParallel(m)) is m

    class FakeCompiled(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    assert unwrap_model(FakeCompiled(nn.DataParallel(m))) is m
    with pytest.raises(RuntimeError):
        wrap_ddp(m, DistInfo(rank=0, world_size=2, is_main=True))  # no process group


def test_make_sampler_single_process_is_epoch_seeded():
    ds = TensorDataset(torch.arange(20))
    s = make_sampler(ds, DistInfo(), shuffle=True, seed=3)
    s.set_epoch(0)
    e0 = list(s)
    assert sorted(e0) == list(range(20)) and e0 != list(range(20))
    assert list(s) == e0                                   # same epoch → same order
    s.set_epoch(1)
    assert list(s) != e0
    s2 = make_sampler(ds, None, shuffle=True, seed=3)
    s2.set_epoch(0)
    assert list(s2) == e0                                  # reproducible across instances
    assert list(make_sampler(ds, DistInfo(), shuffle=False)) == list(range(20))

    class Stream(IterableDataset):
        def __iter__(self):
            return iter(range(3))

    assert make_sampler(Stream(), DistInfo()) is None and make_eval_sampler(Stream()) is None


def test_shard_sampler_partitions_without_padding():
    ds = TensorDataset(torch.arange(10))
    shards = [list(ShardSampler(ds, DistInfo(rank=r, world_size=3, is_main=r == 0)))
              for r in range(3)]
    assert shards == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert [len(ShardSampler(ds, DistInfo(rank=r, world_size=3))) for r in range(3)] == [4, 3, 3]
    assert list(make_eval_sampler(ds)) == list(range(10))


def test_collectives_are_identity_without_group():
    assert all_reduce_sum({"a": 1.5, "b": 2}) == {"a": 1.5, "b": 2.0}
    assert all_reduce_mean({"a": 1.5}) == {"a": 1.5}
    assert all_reduce_sum({}) == {}
    barrier()
    cleanup()
    assert not is_dist_initialized()


# ───────────────────────────────────────────────────────────────────────────── real DDP (gloo)

def _ddp_data(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    return TensorDataset(x, x @ torch.arange(1.0, 5.0) + 0.5 + 0.1 * torch.randn(n, generator=g))


def _ddp_loss(model, batch):
    x, y = batch
    return {"loss": ((model(x).squeeze(-1) - y) ** 2).mean()}


def _ddp_cfg(out: str, **kw) -> TrainConfig:
    base = dict(max_epochs=3, lr=0.05, weight_decay=0.01, warmup_steps=2, schedule="cosine",
                device="cpu", precision="fp32", log_every=2, seed=0, monitor="val/loss",
                ema_decay=0.9, out_dir=out)
    base.update(kw)
    return TrainConfig(**base)


def _ddp_worker(rank: int, world: int, port: int, out: str, q) -> None:
    """torchrun-like child: env vars → Trainer picks them up via init_distributed()."""
    try:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                          WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
        torch.set_num_threads(1)
        from robot_skin.train import Trainer
        torch.manual_seed(rank)       # different init per rank: DDP must broadcast rank 0's
        tr = Trainer(nn.Linear(4, 1), _ddp_loss, _ddp_cfg(out, batch_size=8, grad_accum=2),
                     _ddp_data(60, 0), _ddp_data(21, 1))
        assert tr.dist.world_size == world and tr.dist.is_main == (rank == 0)
        tr.fit()
        q.put((rank, [p.detach().tolist() for p in tr.model.parameters()],
               [r["val/loss"] for r in tr.history], tr.step,
               {k: v.tolist() for k, v in tr.ema.shadow.items()}, None))
    except BaseException:  # report to the parent instead of hanging it
        import traceback
        q.put((rank, None, None, None, None, traceback.format_exc()))
    finally:
        cleanup()


def _free_loopback_port() -> int | None:
    import socket

    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])
    except OSError:  # sandbox without loopback sockets
        return None


@pytest.mark.skipif(not (torch.distributed.is_available()
                         and torch.distributed.is_gloo_available()), reason="needs gloo")
def test_ddp_two_process_gloo_matches_single_process(tmp_path):
    """2 CPU ranks × (batch 8, grad_accum 2) ≡ 1 process × batch 32: rank-0 weight broadcast
    (also into the EMA), DistributedSampler shards, no_sync accumulation (short last group
    included: 60 samples → 30 per rank), gradient all-reduce, exact sharded validation
    (21 samples → 11 + 10) and rank-0-only files."""
    import multiprocessing as mp

    from robot_skin.train import Trainer

    port = _free_loopback_port()
    if port is None:
        pytest.skip("no loopback TCP sockets available")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_ddp_worker, args=(r, 2, port, str(tmp_path / "ddp"), q))
             for r in range(2)]
    for p in procs:
        p.start()
    try:
        res = sorted((q.get(timeout=90) for _ in procs), key=lambda r: r[0])
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
    for rank, *_, err in res:
        assert err is None, f"rank {rank} failed:\n{err}"

    torch.manual_seed(0)
    single = Trainer(nn.Linear(4, 1), _ddp_loss, _ddp_cfg(str(tmp_path / "single"), batch_size=32),
                     _ddp_data(60, 0), _ddp_data(21, 1), dist_info=DistInfo())
    single.fit()
    ref = torch.cat([p.detach().reshape(-1) for p in single.model.parameters()])
    for _rank, params, val, step, ema, _ in res:
        got = torch.cat([torch.tensor(p).reshape(-1) for p in params])
        torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)
        assert step == single.step == 6
        assert val == pytest.approx([r["val/loss"] for r in single.history], rel=1e-5)
        for k, v in single.ema.shadow.items():   # EMA starts from the broadcast weights
            torch.testing.assert_close(torch.tensor(ema[k]), v, atol=1e-6, rtol=1e-6)
    summary = json.loads((tmp_path / "ddp" / "summary.json").read_text())
    assert summary["world_size"] == 2
    assert json.loads((tmp_path / "ddp" / "config.json").read_text())["effective_batch_size"] == 32


def _spawn_ddp(target, args: tuple, n: int = 2, timeout: float = 90.0) -> list:
    """Run ``target(rank, world, port, *args, q)`` on ``n`` spawned gloo ranks; results by rank."""
    import multiprocessing as mp

    port = _free_loopback_port()
    if port is None:
        pytest.skip("no loopback TCP sockets available")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=target, args=(r, n, port, *args, q)) for r in range(n)]
    for p in procs:
        p.start()
    try:
        return sorted((q.get(timeout=timeout) for _ in procs), key=lambda r: r[0])
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()


def _ddp_env(rank: int, world: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)


class _RankStream(IterableDataset):
    """Rank-sharded stream without ``__len__`` (unknown-length loader): rank r gets r, r+w, …"""

    def __init__(self, ds: TensorDataset, rank: int, world: int):
        self.ds, self.rank, self.world = ds, rank, world

    def __iter__(self):
        x, y = self.ds.tensors
        return iter(zip(x[self.rank::self.world], y[self.rank::self.world]))


def _tail_cfg(out: str) -> TrainConfig:
    return TrainConfig(max_steps=2, batch_size=8, grad_accum=2, lr=0.05, weight_decay=0.0,
                       warmup_steps=0, schedule="constant", optimizer="sgd", grad_clip=None,
                       device="cpu", precision="fp32", log_every=1000, seed=0,
                       monitor="train/loss", out_dir=out)


def _ddp_tail_worker(rank: int, world: int, port: int, out: str, q) -> None:
    try:
        _ddp_env(rank, world, port)
        import warnings as _w

        from robot_skin.train import Trainer
        torch.manual_seed(rank)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            tr = Trainer(nn.Linear(4, 1), _ddp_loss, _tail_cfg(out),
                         _RankStream(_ddp_data(48, 0), rank, world))
            tr.fit()
        q.put((rank, [p.detach().tolist() for p in tr.model.parameters()], tr.step, None))
    except BaseException:  # report to the parent instead of hanging it
        import traceback
        q.put((rank, None, None, traceback.format_exc()))
    finally:
        cleanup()


@pytest.mark.skipif(not (torch.distributed.is_available()
                         and torch.distributed.is_gloo_available()), reason="needs gloo")
def test_ddp_iterable_stream_partial_last_group_is_all_reduced(tmp_path):
    """Regression: an unknown-length stream that ends mid accumulation group (3 batches per rank,
    grad_accum 2) ran the tail group entirely under no_sync and stepped with each rank's *local*
    gradient — the replicas silently diverged. Both steps must use the rank-averaged gradient."""
    res = _spawn_ddp(_ddp_tail_worker, (str(tmp_path / "ddp"),))
    for rank, *_, err in res:
        assert err is None, f"rank {rank} failed:\n{err}"
    torch.manual_seed(0)
    ref = nn.Linear(4, 1)
    opt = torch.optim.SGD(ref.parameters(), lr=0.05, momentum=0.9, nesterov=True)
    x, y = _ddp_data(48, 0).tensors
    for sl in (slice(0, 16), slice(16, 24)):      # group (b0, b1) of every rank, then the tail b2
        opt.zero_grad()
        xb = torch.cat([x[r::2][sl] for r in range(2)])
        yb = torch.cat([y[r::2][sl] for r in range(2)])
        _ddp_loss(ref, (xb, yb))["loss"].backward()
        opt.step()
    want = torch.cat([p.detach().reshape(-1) for p in ref.parameters()])
    for _rank, params, step, _ in res:
        assert step == 2
        got = torch.cat([torch.tensor(p).reshape(-1) for p in params])
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def _ddp_resume_worker(rank: int, world: int, port: int, root: str, q) -> None:
    try:
        _ddp_env(rank, world, port)
        import pathlib
        import warnings as _w

        from robot_skin.train import Trainer
        ds = _ddp_data(32, 0)
        node = str(pathlib.Path(root) / f"node{rank}")         # node-local run directory
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            torch.manual_seed(rank)
            Trainer(nn.Linear(4, 1), _ddp_loss, _ddp_cfg(node, max_epochs=2, batch_size=8, ema_decay=None),
                    ds).fit()                                   # only rank 0 writes a checkpoint
            torch.manual_seed(10 + rank)
            tr = Trainer(nn.Linear(4, 1), _ddp_loss,
                         _ddp_cfg(node, max_epochs=4, batch_size=8, ema_decay=None, resume="auto"), ds)
            try:
                tr.fit()
                local = "no error"
            except RuntimeError as e:
                local = str(e)
            shared = str(pathlib.Path(root) / "node0")           # the same checkpoint on every rank
            tr = Trainer(nn.Linear(4, 1), _ddp_loss,
                         _ddp_cfg(shared, max_epochs=4, batch_size=8, ema_decay=None, resume="auto"), ds)
            tr.fit()
        q.put((rank, local, (tr.resumed_from is not None, tr.step,
                             [p.detach().tolist() for p in tr.model.parameters()]), None))
    except BaseException:  # report to the parent instead of hanging it
        import traceback
        q.put((rank, None, None, traceback.format_exc()))
    finally:
        cleanup()


@pytest.mark.skipif(not (torch.distributed.is_available()
                         and torch.distributed.is_gloo_available()), reason="needs gloo")
def test_ddp_resume_requires_every_rank_to_see_the_checkpoint(tmp_path):
    """Regression: with node-local out_dirs only rank 0 found ckpt_last.pt; it resumed at step 4
    while rank 1 started fresh at step 0 with other weights (DDP had already broadcast at init),
    so the replicas diverged / the collectives mismatched. Now every rank raises; a shared
    out_dir resumes identically on every rank."""
    res = _spawn_ddp(_ddp_resume_worker, (str(tmp_path),))
    for rank, *_, err in res:
        assert err is None, f"rank {rank} failed:\n{err}"
    for _rank, local, _, _ in res:
        assert "ranks disagree" in local and "shared" in local
    (r0, s0, p0), (r1, s1, p1) = (r[2] for r in res)
    assert r0 and r1 and s0 == s1 == 8 and p0 == p1             # 2 steps / epoch × 4 epochs
