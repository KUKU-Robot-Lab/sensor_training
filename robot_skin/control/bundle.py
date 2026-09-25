"""Load a trained VTLA ``policy_bundle.pt`` for deployment: :func:`load_policy_bundle` → :class:`PolicyBundle`.

The bundle (written by ``stages/vtla.py`` via :func:`robot_skin.vtla.save_policy_bundle`) holds
everything the online side must reproduce from training: the model config + weights, the action
space / normalizer / relative mode, the proprio normalizer and history, the tactile
:class:`~robot_skin.representation.TactileFeatureSpec` and contact rule, references to the stage-1
calibrator (embedded as ``tactile.calibrator_state``) and baseline model, the camera list and image
eval transform, the text tokenizer and the timing (policy rate, source rate, stride, horizon).
:class:`PolicyBundle` wraps :func:`robot_skin.vtla.bundle_components` with typed attributes and the
deployment checks (a policy trained on **bootstrap** tactile levels — no motion-artefact model — is
refused unless explicitly allowed).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = ["PolicyBundle", "load_policy_bundle"]


@dataclass
class PolicyBundle:
    """A deployable VTLA policy with its preprocessing contract (see module docstring)."""

    policy: Any                       # VTLAPolicy (eval mode, on ``device``)
    action_spec: Any                  # action.space.ActionSpec
    action_normalizer: Any            # action.space.ActionNormalizer
    rel_mode: str
    chunk_offset: int
    proprio_normalizer: Any
    feature_spec: Any                 # representation.TactileFeatureSpec
    contact_rule: str
    eval_transform: Any               # vision.EvalTransform | None
    tokenizer: Any
    cameras: tuple[str, ...]
    policy_hz: float
    source_hz: float
    stride: int
    horizon: int
    obs_history: int
    head: str
    flow_steps: int
    tactile: dict = field(default_factory=dict)
    vision: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    path: Path | None = None
    raw: dict = field(default_factory=dict, repr=False)
    device: str = "cpu"

    # ── derived views ─────────────────────────────────────────────────────
    @property
    def action_kind(self) -> str:
        return str(self.action_spec.kind)

    @property
    def action_dim(self) -> int:
        return int(self.action_spec.dim)

    @property
    def ensemble_k(self) -> float:
        """ACT temporal-ensembling constant stored at training time (default 0.01)."""
        return float((self.meta or {}).get("ensemble_k", 0.01))

    @property
    def tactile_source(self) -> str:
        return str((self.tactile or {}).get("source", "derived"))

    @property
    def uses_tactile(self) -> bool:
        return self.feature_spec is not None and int(self.feature_spec.dim) > 0

    def calibrator(self) -> Any:
        """The stage-1 :class:`~robot_skin.contact.calibration.ResidualCalibrator` the policy was
        trained with: the embedded ``tactile.calibrator_state`` or the referenced ``calibrator.json``
        (``None`` if neither is available)."""
        from .online import load_calibrator

        tac = self.tactile or {}
        if tac.get("calibrator_state"):
            return load_calibrator(dict(tac["calibrator_state"]))
        ref = tac.get("calibrator")
        if ref:
            p = Path(ref)
            if p.is_dir():
                p = p / "calibrator.json"
            if p.is_file():
                return load_calibrator(p)
            warnings.warn(f"bundle calibrator {ref!r} not found", stacklevel=2)
        return None

    def baseline_model_path(self) -> Path | None:
        """Existing path of the referenced stage-1 baseline model (``baseline_model.pt`` or run dir)."""
        ref = (self.tactile or {}).get("baseline_model")
        if not ref:
            return None
        p = Path(ref)
        if p.is_dir():
            p = p / "baseline_model.pt"
        return p if p.is_file() else None

    def stage1_matches(self, layout: Any) -> bool:
        """Whether the bundle's stage-1 references (calibrator / baseline model) belong to the skin
        ``layout``: the layouts of the bundle's training episodes (``tactile.layouts`` — built-in names
        or layout files) include one named like it. Bundles that do not record their layouts match
        (only the taxel count is checked downstream). Built-in templates of different skins can have
        the same taxel count (glove_template and robot_hand_template: 9), hence the name check."""
        from common.layouts import load_layout

        lays = [str(x) for x in ((self.tactile or {}).get("layouts") or [])]
        if not lays:
            return True
        name = layout.name if hasattr(layout, "name") else load_layout(layout).name
        names = set()
        for x in lays:
            try:
                names.add(load_layout(x).name)
            except Exception:  # noqa: BLE001 - a moved / foreign layout file: compare its file stem
                names.add(Path(x).stem)
        return name in names

    def check_deployable(self, *, allow_bootstrap: bool = False) -> None:
        """Raise if the bundle must not drive hardware: tactile levels from the **bootstrap**
        stand-in (``tactile.source`` bootstrap / mixed: a static reference without the motion-artefact
        model — motion alone raises false contact) unless ``allow_bootstrap``."""
        src = self.tactile_source
        if self.uses_tactile and src in ("bootstrap", "mixed") and not allow_bootstrap:
            raise ValueError(f"policy bundle {self.path} was trained on {src!r} tactile levels (vtla bootstrap "
                             "fallback, no motion-artefact model): run the baseline + contact stages and retrain "
                             "before deploying (allow_bootstrap=True only for simulation / bring-up)")

    def summary(self) -> dict:
        return {"path": None if self.path is None else str(self.path), "action_kind": self.action_kind,
                "action_dim": self.action_dim, "rel_mode": self.rel_mode, "policy_hz": self.policy_hz,
                "source_hz": self.source_hz, "stride": self.stride, "horizon": self.horizon,
                "obs_history": self.obs_history, "head": self.head, "cameras": list(self.cameras),
                "obs_mode": None if self.feature_spec is None else self.feature_spec.obs_mode,
                "tactile_source": self.tactile_source, "contact_rule": self.contact_rule,
                "ensemble_k": self.ensemble_k}


def load_policy_bundle(bundle: Any, *, device: str | Any = "cpu", allow_bootstrap: bool = True) -> PolicyBundle:
    """Read ``policy_bundle.pt`` (a file, its run directory or an already loaded dict) and rebuild
    the policy on ``device`` (:func:`robot_skin.vtla.bundle_components`). ``allow_bootstrap=False``
    runs :meth:`PolicyBundle.check_deployable`; with the default the bootstrap case only warns."""
    from ..vtla.model import POLICY_BUNDLE_NAME, bundle_components, read_policy_bundle

    path = None
    if isinstance(bundle, (str, Path)):
        path = Path(bundle)
        if path.is_dir():
            path = path / POLICY_BUNDLE_NAME
        raw = read_policy_bundle(path)
    elif isinstance(bundle, Mapping):
        raw = dict(bundle)
    else:
        raise TypeError(f"bundle must be a path or a bundle dict, got {type(bundle).__name__}")
    comp = bundle_components(raw, device=device)
    names = ("policy", "action_spec", "action_normalizer", "rel_mode", "chunk_offset", "proprio_normalizer",
             "feature_spec", "contact_rule", "eval_transform", "tokenizer", "cameras", "policy_hz", "source_hz",
             "stride", "horizon", "obs_history", "head", "flow_steps", "tactile", "vision", "meta")
    pb = PolicyBundle(**{k: comp[k] for k in names}, path=path, raw=raw, device=str(device))
    pb.tactile = dict(pb.tactile or {})
    pb.vision = dict(pb.vision or {})
    pb.meta = dict(pb.meta or {})
    if pb.uses_tactile and pb.tactile_source in ("bootstrap", "mixed"):
        if not allow_bootstrap:
            pb.check_deployable(allow_bootstrap=False)
        warnings.warn(f"policy bundle tactile source is {pb.tactile_source!r} (bootstrap levels): "
                      "simulation / bring-up only", stacklevel=2)
    return pb
