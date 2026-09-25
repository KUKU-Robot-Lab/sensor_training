"""Device / precision resolution and per-GPU hardware profiles.

The same stage config must train on an RTX 5090 workstation, an RTX 4090 / 3090 box or an A100
node reached over Tailscale, and on a CPU laptop for tests. This module turns a small set of
knobs (``device``, ``precision``) into concrete torch settings and ships YAML *hardware profiles*
(``robot_skin/configs/hardware/<name>.yaml``) that adapt hardware-dependent training knobs.

Precision rules (:func:`resolve_precision`)::

    auto  → CUDA cc ≥ 8.0 (Ampere, Ada, Hopper, Blackwell) : bf16 autocast, no GradScaler
            older CUDA (Volta/Turing)                       : fp16 autocast + GradScaler
            CPU / MPS                                       : fp32
    bf16  → honoured on CUDA cc ≥ 8.0 and on CPU (CPU autocast); on older CUDA falls back to
            fp16 + GradScaler (warning); on MPS falls back to fp32 (warning)
    fp16  → CUDA: fp16 + GradScaler; elsewhere falls back to fp32 (warning)
    fp32  → no autocast

bf16 has fp32's exponent range, so no loss scaling is needed (the same choice the SATS trainer
in ``deformable_sats/sats/training`` makes with ``use_amp``).

RTX 5090 (Blackwell, compute capability 12.0 = ``sm_120``) needs a torch wheel built for
CUDA ≥ 12.8 — older wheels import fine but fail at the first kernel launch with "no kernel
image is available for execution on the device". :func:`check_arch_support` detects that
up-front from ``torch.cuda.get_arch_list()``. The repo pins ``torch==2.9.0+cu128`` in
``deformable_sats/requirements.txt``.

Profile precedence (:func:`apply_hw_profile`)::

    stage YAML ``train:``  <  profile ``suggest.<stage>``  <  profile ``train:``  (< CLI overrides)

Run ``python -m robot_skin.train.hardware`` on any machine to print the environment report.
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import platform
import re
import shutil
import socket
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from robot_skin.config import deep_merge

PROFILES_DIR = Path(__file__).resolve().parent.parent / "configs" / "hardware"

#: minimum compute capability with native bf16 tensor-core support (Ampere)
BF16_MIN_CAPABILITY = (8, 0)

_PRECISION_ALIASES = {
    "auto": "auto",
    "bf16": "bf16", "bfloat16": "bf16", "bf16-mixed": "bf16",
    "fp16": "fp16", "float16": "fp16", "half": "fp16", "16": "fp16", "16-mixed": "fp16",
    "fp32": "fp32", "float32": "fp32", "32": "fp32", "full": "fp32", "none": "fp32",
}

# whole model token in torch.cuda.get_device_name() → profile name (checked in order): a plain
# substring test would map "RTX A1000" (6-8 GB) to the 80 GB a100 profile
_GPU_NAME_TO_PROFILE: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bRTX\s*5090\b", re.I), "rtx5090"),
    (re.compile(r"\bRTX\s*4090\b", re.I), "rtx4090"),
    (re.compile(r"\bRTX\s*3090\b", re.I), "rtx3090"),
    (re.compile(r"\bA100\b", re.I), "a100"),
)
#: laptop / mobile variants share the desktop model number but not its memory or power budget
_MOBILE_GPU_RE = re.compile(r"\b(laptop|mobile|max-q)\b", re.I)
#: a GPU with less memory than this fraction of the profile's ``gpu.memory_gb`` gets no profile
_MIN_MEMORY_FRAC = 0.9

__all__ = [
    "PROFILES_DIR", "PrecisionPlan", "resolve_device", "resolve_precision", "cuda_capability",
    "enable_tf32", "list_hw_profiles", "load_hw_profile", "apply_hw_profile",
    "maybe_apply_hw_profile", "apply_profile_env", "detect_hw_profile", "describe_environment",
    "check_arch_support",
]


# ───────────────────────────────────────────────────────────────────────────── device

def resolve_device(pref: str | torch.device | None = "auto",
                   local_rank: int | None = None) -> torch.device:
    """``"auto" | "cpu" | "cuda" | "cuda:N" | "mps"`` → :class:`torch.device`.

    ``auto`` picks CUDA, then Apple MPS, then CPU. Under ``torchrun`` pass ``local_rank`` so
    ``auto``/``cuda`` map to ``cuda:<local_rank>`` (one process per GPU). Requesting an
    unavailable accelerator raises ``RuntimeError`` instead of silently training on CPU.
    """
    if isinstance(pref, torch.device):
        pref = str(pref)
    p = (pref or "auto").strip().lower()
    if p == "auto":
        if torch.cuda.is_available():
            p = "cuda"
        elif _mps_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    if p == "cpu":
        return torch.device("cpu")
    if p == "mps":
        if not _mps_available():
            raise RuntimeError("device 'mps' requested but torch.backends.mps is not available")
        return torch.device("mps")
    m = re.fullmatch(r"cuda(?::(\d+))?", p)
    if m is None:
        raise ValueError(f"unknown device {pref!r}; use auto | cpu | cuda | cuda:N | mps")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device {pref!r} requested but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). Install a CUDA "
            "build of torch (RTX 50xx needs CUDA >= 12.8 wheels, e.g. torch==2.9.0+cu128) or "
            "use device: cpu.")
    n = torch.cuda.device_count()
    if m.group(1) is not None:
        idx = int(m.group(1))
    else:
        idx = int(local_rank) if local_rank is not None else 0
    if idx >= n:
        raise RuntimeError(f"cuda:{idx} requested but only {n} CUDA device(s) are visible "
                           f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    return torch.device("cuda", idx)


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    try:
        return bool(mps is not None and mps.is_available())
    except Exception:  # pragma: no cover - platform specific
        return False


def cuda_capability(device: torch.device | str | int | None = None) -> tuple[int, int] | None:
    """Compute capability ``(major, minor)`` of a CUDA device, ``None`` without CUDA."""
    if not torch.cuda.is_available():
        return None
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return None
        device = device.index if device.index is not None else torch.cuda.current_device()
    return tuple(torch.cuda.get_device_capability(device))  # type: ignore[return-value]


# ───────────────────────────────────────────────────────────────────────────── precision

@dataclass(frozen=True)
class PrecisionPlan:
    """Resolved mixed-precision plan.

    ``autocast_dtype`` is ``None`` for fp32; ``use_grad_scaler`` only for fp16 on CUDA.
    """
    name: str                              # "bf16" | "fp16" | "fp32"
    autocast_dtype: torch.dtype | None
    use_grad_scaler: bool
    device_type: str = "cpu"

    @property
    def enabled(self) -> bool:
        return self.autocast_dtype is not None

    def autocast(self):
        """Context manager for the forward pass (``nullcontext`` for fp32)."""
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype)

    def make_scaler(self):
        """``torch.amp.GradScaler`` for fp16, else ``None``."""
        if not self.use_grad_scaler:
            return None
        return torch.amp.GradScaler(self.device_type)


def resolve_precision(pref: str = "auto", device: torch.device | str = "cpu", *,
                      capability: tuple[int, int] | None = None) -> PrecisionPlan:
    """Map a precision preference to a :class:`PrecisionPlan` for ``device`` (rules in the module
    docstring). ``capability`` overrides the queried CUDA compute capability (tests / planning
    for a remote GPU from a CPU machine)."""
    key = _PRECISION_ALIASES.get(str(pref).strip().lower())
    if key is None:
        raise ValueError(f"unknown precision {pref!r}; use one of auto | bf16 | fp16 | fp32")
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    dt = dev.type
    fp32 = PrecisionPlan("fp32", None, False, dt)

    if dt == "cuda":
        cap = capability if capability is not None else cuda_capability(dev)
        if cap is None:
            raise RuntimeError(f"cannot resolve precision for {dev}: CUDA is not available "
                               "(pass capability=(major, minor) to plan for a remote GPU)")
        cap = (int(cap[0]), int(cap[1]))
        bf16_ok = cap >= BF16_MIN_CAPABILITY
        bf16 = PrecisionPlan("bf16", torch.bfloat16, False, "cuda")
        fp16 = PrecisionPlan("fp16", torch.float16, True, "cuda")
        if key == "fp32":
            return fp32
        if key == "fp16":
            return fp16
        if key == "bf16" and not bf16_ok:
            warnings.warn(f"bf16 requested but GPU compute capability {cap} < 8.0 has no native "
                          "bf16; falling back to fp16 + GradScaler", stacklevel=2)
            return fp16
        if key == "bf16":
            return bf16
        return bf16 if bf16_ok else fp16  # auto

    if dt == "cpu":
        if key == "bf16":
            return PrecisionPlan("bf16", torch.bfloat16, False, "cpu")
        if key == "fp16":
            warnings.warn("fp16 autocast is not supported for CPU training here; using fp32",
                          stacklevel=2)
        return fp32

    if key in ("bf16", "fp16"):
        warnings.warn(f"{key} requested on device type {dt!r}; using fp32", stacklevel=2)
    return fp32


def enable_tf32(enabled: bool = True) -> str:
    """Allow TF32 tensor-core matmuls for fp32 ops on Ampere+ GPUs; returns the previous setting.

    Uses only ``torch.set_float32_matmul_precision`` — mixing it with the newer
    ``torch.backends.*.fp32_precision`` API raises in torch ≥ 2.9. cuDNN convolutions already
    default to TF32. Call it only when training on CUDA (on some CPUs ``"high"`` also relaxes
    oneDNN matmul precision).
    """
    try:
        prev = torch.get_float32_matmul_precision()
    except RuntimeError:  # someone mixed the legacy and new TF32 APIs
        prev = "unknown"
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    return prev


# ───────────────────────────────────────────────────────────────────────────── profiles

def list_hw_profiles(profiles_dir: str | Path | None = None) -> list[str]:
    """Names of the built-in hardware profiles."""
    d = Path(profiles_dir) if profiles_dir is not None else PROFILES_DIR
    return sorted(p.stem for p in d.glob("*.yaml"))


def load_hw_profile(name_or_path: str | Path, profiles_dir: str | Path | None = None) -> dict:
    """Load a hardware profile by name (``rtx5090``, ``rtx4090``, ``rtx3090``, ``a100``, ``cpu``)
    or by YAML path. ``"auto"`` resolves via :func:`detect_hw_profile`."""
    s = str(name_or_path)
    if s == "auto":
        detected = detect_hw_profile()
        if detected is None:
            raise ValueError("hardware 'auto': no built-in profile matches the visible GPU; "
                             f"pass one of {list_hw_profiles(profiles_dir)} or a YAML path")
        s = detected
    p = Path(s)
    if p.suffix not in (".yaml", ".yml") or not p.exists():
        d = Path(profiles_dir) if profiles_dir is not None else PROFILES_DIR
        p = d / f"{s}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"hardware profile {name_or_path!r} not found; available: "
                                f"{list_hw_profiles(profiles_dir)} (or pass a .yaml path)")
    prof = yaml.safe_load(p.read_text()) or {}
    if not isinstance(prof, dict):
        raise ValueError(f"{p}: hardware profile must be a mapping")
    for key in ("train", "suggest", "env"):
        if key in prof and prof[key] is not None and not isinstance(prof[key], Mapping):
            raise ValueError(f"{p}: '{key}' must be a mapping")
    prof.setdefault("name", p.stem)
    return prof


def apply_hw_profile(stage_cfg: Mapping[str, Any], profile: str | Path | Mapping[str, Any],
                     stage: str | None = None) -> dict:
    """Return a copy of ``stage_cfg`` with the profile applied to its ``train`` section.

    Precedence: ``stage_cfg["train"]`` < ``profile["suggest"][stage]`` < ``profile["train"]``.
    ``stage`` defaults to ``stage_cfg.get("stage")``. The resolved profile name is recorded in
    ``out["hardware"]``. Apply CLI overrides *after* this so they win.
    """
    prof = load_hw_profile(profile) if not isinstance(profile, Mapping) else dict(profile)
    out = copy.deepcopy(dict(stage_cfg))
    train = dict(out.get("train") or {})
    stage = stage if stage is not None else out.get("stage")
    suggest = (prof.get("suggest") or {}).get(stage) if stage else None
    if suggest:
        train = deep_merge(train, suggest)
    if prof.get("train"):
        train = deep_merge(train, prof["train"])
    out["train"] = train
    out["hardware"] = prof.get("name", "custom")
    return out


def maybe_apply_hw_profile(cfg: Mapping[str, Any], stage: str | None = None) -> dict:
    """Apply ``cfg["hardware"]`` (profile name / path / mapping / ``"auto"``) if set; ``"auto"``
    with no matching GPU leaves the config unchanged (warning)."""
    hw = cfg.get("hardware")
    if not hw:
        return copy.deepcopy(dict(cfg))
    if hw == "auto" and detect_hw_profile() is None:
        warnings.warn("hardware: auto — no built-in profile matches this machine; using the "
                      "stage defaults", stacklevel=2)
        return copy.deepcopy(dict(cfg))
    return apply_hw_profile(cfg, hw, stage=stage)


def apply_profile_env(profile: Mapping[str, Any], override: bool = False) -> dict[str, str]:
    """Export ``profile["env"]`` into ``os.environ`` (``setdefault`` unless ``override``).

    Call before the first CUDA call (e.g. ``PYTORCH_CUDA_ALLOC_CONF`` is read at CUDA init).
    Returns the variables actually set.
    """
    applied: dict[str, str] = {}
    for k, v in (profile.get("env") or {}).items():
        if override or k not in os.environ:
            os.environ[k] = str(v)
            applied[k] = str(v)
    return applied


def _profile_memory_gb(name: str) -> float | None:
    try:
        return float(((load_hw_profile(name).get("gpu") or {}).get("memory_gb")))
    except (OSError, ValueError, TypeError):
        return None


def detect_hw_profile(gpu_names: Sequence[str] | None = None,
                      memory_gb: Sequence[float] | None = None) -> str | None:
    """Pick a built-in profile from the first visible GPU (``"cpu"`` without CUDA); ``None`` for
    an unknown GPU — the stages then keep their defaults (warning) instead of guessing. The model
    must appear as a whole token (``RTX A1000`` is not an ``A100``), laptop / mobile variants never
    match, and a GPU with clearly less memory than the profile's ``gpu.memory_gb`` (e.g. an A100
    40 GB vs the 80 GB profile) gets ``None``. ``gpu_names`` / ``memory_gb`` (GiB) override the
    query (tests; the memory check is skipped when names are given without it)."""
    if gpu_names is None:
        if not torch.cuda.is_available():
            return "cpu"
        n = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n)]
        if memory_gb is None:
            memory_gb = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(n)]
    if not gpu_names:
        return "cpu"
    name = gpu_names[0]
    if _MOBILE_GPU_RE.search(name):
        return None
    prof = next((p for pat, p in _GPU_NAME_TO_PROFILE if pat.search(name)), None)
    if prof is None:
        return None
    if memory_gb:
        need = _profile_memory_gb(prof)
        if need is not None and float(memory_gb[0]) < _MIN_MEMORY_FRAC * need:
            return None
    return prof


# ───────────────────────────────────────────────────────────────────────────── diagnostics

def describe_environment() -> dict:
    """JSON-serialisable report of the software/hardware environment (written to
    ``<out_dir>/env.json`` by the Trainer so runs from different machines stay comparable)."""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "mps_available": _mps_available(),
        "num_cpus": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
    }
    try:
        info["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception:
        info["arch_list"] = []
    try:
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except Exception:
        info["cudnn"] = None
    gpus = []
    if info["cuda_available"]:
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            cap = (props.major, props.minor)
            gpus.append({
                "index": i, "name": props.name, "capability": list(cap),
                "memory_gb": round(props.total_memory / 2**30, 2),
                "multiprocessors": props.multi_processor_count,
                "bf16": cap >= BF16_MIN_CAPABILITY,
            })
        try:
            info["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
        except Exception:
            info["nccl"] = None
    info["gpus"] = gpus
    info["profile_guess"] = detect_hw_profile([g["name"] for g in gpus],
                                              [g["memory_gb"] for g in gpus]) if gpus else (
        "cpu" if not info["cuda_available"] else None)
    keys = ("CUDA_VISIBLE_DEVICES", "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR",
            "MASTER_PORT", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_P2P_DISABLE",
            "NCCL_IB_DISABLE", "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS")
    info["env"] = {k: os.environ[k] for k in keys if k in os.environ}
    info["warnings"] = check_arch_support()
    return info


_ARCH_RE = re.compile(r"^(sm|compute)_(\d+)([a-z]?)$")


def _parse_arch(entry: str) -> tuple[str, int, int, str] | None:
    m = _ARCH_RE.match(entry.strip().lower())
    if m is None:
        return None
    digits = m.group(2)
    if len(digits) < 2:
        return None
    return m.group(1), int(digits[:-1]), int(digits[-1]), m.group(3)


def _kernel_available(cap: tuple[int, int], arch_list: Sequence[str]) -> bool:
    """True if ``arch_list`` has a cubin for ``cap`` (same major, minor ≤ — CUDA binary
    compatibility; ``a``-suffixed arch-specific cubins only on the exact capability) or PTX
    (``compute_XY`` ≤ cap, JIT-compiled forward)."""
    for entry in arch_list:
        parsed = _parse_arch(entry)
        if parsed is None:
            continue
        kind, major, minor, suffix = parsed
        if suffix == "a":
            if (major, minor) == cap:
                return True
            continue
        if kind == "sm" and major == cap[0] and minor <= cap[1]:
            return True
        if kind == "compute" and (major, minor) <= cap:
            return True
    return False


def _version_tuple(v: str | None) -> tuple[int, ...] | None:
    if not v:
        return None
    nums = re.findall(r"\d+", v)
    return tuple(int(x) for x in nums[:2]) if nums else None


def check_arch_support(capabilities: Sequence[tuple[int, int]] | None = None,
                       arch_list: Sequence[str] | None = None,
                       torch_cuda: str | None = None,
                       names: Sequence[str] | None = None) -> list[str]:
    """Human-readable warnings when the installed torch cannot run on the visible GPUs.

    Main case: RTX 5090 / RTX 50xx (Blackwell, capability 12.0) or B200 (10.0) with a torch
    built for CUDA < 12.8 (no ``sm_120`` / ``sm_100`` kernels). All arguments default to the live
    environment; pass them to check a remote machine's ``env.json`` or in tests.
    """
    live = capabilities is None
    warns: list[str] = []
    if live:
        if not torch.cuda.is_available():
            if shutil.which("nvidia-smi") is not None:
                warns.append(
                    "nvidia-smi is present but torch.cuda.is_available() is False: this is a "
                    f"CPU-only torch build ({torch.__version__}) or a driver/CUDA mismatch. "
                    "Install a CUDA wheel matching the GPU (RTX 50xx: CUDA >= 12.8).")
            return warns
        n = torch.cuda.device_count()
        capabilities = [tuple(torch.cuda.get_device_capability(i)) for i in range(n)]
        names = [torch.cuda.get_device_name(i) for i in range(n)]
    if arch_list is None:
        try:
            arch_list = list(torch.cuda.get_arch_list())
        except Exception:
            arch_list = []
    if torch_cuda is None and live:
        torch_cuda = torch.version.cuda
    cuda_v = _version_tuple(torch_cuda)
    names = list(names) if names is not None else [f"GPU{i}" for i in range(len(capabilities))]
    for i, cap in enumerate(capabilities):
        cap = (int(cap[0]), int(cap[1]))
        label = f"GPU {i} ({names[i] if i < len(names) else '?'}, sm_{cap[0]}{cap[1]})"
        blackwell = cap[0] >= 10
        # CUDA 12.8 is the first toolkit with Blackwell support: an older build is flagged even
        # if a PTX entry could in principle be JIT-compiled (its cuBLAS/cuDNN predate sm_100/120)
        old_toolkit = blackwell and cuda_v is not None and cuda_v < (12, 8)
        has_kernel = bool(arch_list) and _kernel_available(cap, arch_list)
        if has_kernel and not old_toolkit:
            continue
        if not arch_list and not old_toolkit:
            continue  # unknown build contents and nothing specific to say
        build = f"{torch.__version__}, CUDA {torch_cuda}" if live else f"CUDA {torch_cuda}"
        if has_kernel:
            msg = (f"{label}: this torch build ({build}) can only reach compute capability "
                   f"{cap[0]}.{cap[1]} through PTX JIT (arch list: {list(arch_list)}); expect "
                   "slow first launches or 'no kernel image is available' errors from its "
                   "bundled CUDA libraries.")
        else:
            msg = (f"{label}: this torch build ({build}) has no kernels for compute capability "
                   f"{cap[0]}.{cap[1]} (arch list: {list(arch_list)}); expect 'no kernel image "
                   "is available for execution on the device'.")
        if blackwell:
            msg += (" Blackwell GPUs (RTX 50xx = sm_120, B200 = sm_100) need torch built with "
                    "CUDA >= 12.8, e.g. pip install torch==2.9.0 --index-url "
                    "https://download.pytorch.org/whl/cu128 (pinned in "
                    "deformable_sats/requirements.txt), and an NVIDIA driver that supports "
                    "Blackwell (R570 or newer).")
            if cuda_v is not None and cuda_v < (12, 8):
                msg += f" Installed torch is built for CUDA {torch_cuda} < 12.8."
        warns.append(msg)
    return warns


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m robot_skin.train.hardware [--json]``: print the environment report."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args(argv)
    info = describe_environment()
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    print(f"host {info['hostname']}  python {info['python']}  torch {info['torch']} "
          f"(CUDA {info['torch_cuda']})")
    print(f"arch list: {info['arch_list']}")
    for g in info["gpus"]:
        print(f"  cuda:{g['index']} {g['name']}  cc {g['capability'][0]}.{g['capability'][1]}  "
              f"{g['memory_gb']} GB  bf16={g['bf16']}")
    if not info["gpus"]:
        print("  no CUDA GPU visible")
    print(f"suggested hardware profile: {info['profile_guess']}  "
          f"(available: {', '.join(list_hw_profiles())})")
    for w in info["warnings"]:
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
