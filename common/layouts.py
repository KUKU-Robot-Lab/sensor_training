"""Taxel layout schema, YAML loader and :func:`grid_layout`.

A layout describes *where* each taxel is and *what it is attached to*::

    name: sats_4x4
    units: mm                  # mm | m — positions are converted to metres on load
    parent_frame: urdf         # sensor | mano | urdf : namespace of the `parent` names
    taxels:
      - {id: S1, channel: 0, parent: sats_pad, position: [-9.75, -9.75, 0], normal: [0, 0, 1],
         groups: [row0, col0]}
    imu_sites:                 # optional (glove)
      - {name: wrist, parent: wrist}

``parent`` is a link/segment name (URDF link for a robot hand, MANO segment for a glove);
``position``/``normal`` are expressed in that parent frame. A pose provider
(``robot_skin.pose``) turns ``parent`` + local pose into world taxel poses at time t.

Built-in layouts live in ``common/layouts/*.yaml`` (the directory has no ``__init__.py`` so
``common.layouts`` resolves to this module).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

LAYOUT_DIR = Path(__file__).resolve().parent / "layouts"
_UNIT_SCALE = {"m": 1.0, "mm": 1e-3}
PARENT_FRAMES = ("sensor", "mano", "urdf")

#: MANO segment names used as `parent` in glove layouts.
MANO_SEGMENTS = (
    "wrist", "palm",
    "thumb1", "thumb2", "thumb3", "index1", "index2", "index3",
    "middle1", "middle2", "middle3", "ring1", "ring2", "ring3",
    "pinky1", "pinky2", "pinky3",
)


@dataclass(frozen=True)
class Taxel:
    id: str
    channel: int
    parent: str
    position: np.ndarray  # [3] metres, parent frame
    normal: np.ndarray    # [3] unit, parent frame
    groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImuSite:
    name: str
    parent: str


@dataclass(frozen=True)
class Layout:
    name: str
    parent_frame: str
    taxels: tuple[Taxel, ...]
    imu_sites: tuple[ImuSite, ...] = ()
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parent_frame not in PARENT_FRAMES:
            raise ValueError(f"parent_frame must be one of {PARENT_FRAMES}, got {self.parent_frame!r}")
        ids = [t.id for t in self.taxels]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate taxel ids in layout {self.name!r}")
        chans = [t.channel for t in self.taxels]
        if len(set(chans)) != len(chans):
            raise ValueError(f"duplicate channels in layout {self.name!r}")
        if self.parent_frame == "mano":
            bad = {t.parent for t in self.taxels} - set(MANO_SEGMENTS)
            bad |= {s.parent for s in self.imu_sites} - set(MANO_SEGMENTS)
            if bad:
                raise ValueError(f"unknown MANO segments: {sorted(bad)}")

    # ── views ──────────────────────────────────────────────────────────────
    @property
    def n(self) -> int:
        return len(self.taxels)

    @property
    def channels(self) -> np.ndarray:
        return np.array([t.channel for t in self.taxels], dtype=np.int64)

    @property
    def positions(self) -> np.ndarray:
        """[N, 3] metres, parent frames."""
        return np.stack([t.position for t in self.taxels]).astype(np.float64)

    @property
    def normals(self) -> np.ndarray:
        return np.stack([t.normal for t in self.taxels]).astype(np.float64)

    @property
    def parents(self) -> tuple[str, ...]:
        return tuple(t.parent for t in self.taxels)

    @property
    def groups(self) -> dict[str, list[int]]:
        """group name → taxel indices (layout order)."""
        out: dict[str, list[int]] = {}
        for i, t in enumerate(self.taxels):
            for g in t.groups:
                out.setdefault(g, []).append(i)
        return out

    def by_channel(self, raw: np.ndarray) -> np.ndarray:
        """Reorder ``[..., C]`` channel-major data into layout order ``[..., N]``."""
        return np.asarray(raw)[..., self.channels]

    def to_dict(self, units: str = "m") -> dict:
        s = 1.0 / _UNIT_SCALE[units]
        return {
            "name": self.name, "units": units, "parent_frame": self.parent_frame,
            "meta": dict(self.meta),
            "taxels": [{"id": t.id, "channel": t.channel, "parent": t.parent,
                        "position": [round(float(v) * s, 9) for v in t.position],
                        "normal": [float(v) for v in t.normal], "groups": list(t.groups)}
                       for t in self.taxels],
            "imu_sites": [{"name": m.name, "parent": m.parent} for m in self.imu_sites],
        }


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    nrm = np.linalg.norm(a)
    if nrm == 0:
        raise ValueError("normal must be non-zero")
    return a / nrm


def layout_from_dict(d: dict) -> Layout:
    units = d.get("units", "m")
    if units not in _UNIT_SCALE:
        raise ValueError(f"units must be one of {list(_UNIT_SCALE)}, got {units!r}")
    s = _UNIT_SCALE[units]
    taxels = tuple(
        Taxel(id=str(t["id"]), channel=int(t["channel"]), parent=str(t["parent"]),
              position=np.asarray(t["position"], dtype=np.float64).reshape(3) * s,
              normal=_unit(t.get("normal", (0.0, 0.0, 1.0))),
              groups=tuple(t.get("groups", ())))
        for t in d["taxels"])
    imus = tuple(ImuSite(name=str(m["name"]), parent=str(m["parent"])) for m in d.get("imu_sites", ()))
    return Layout(name=str(d["name"]), parent_frame=str(d.get("parent_frame", "sensor")),
                  taxels=taxels, imu_sites=imus, meta=dict(d.get("meta", {})))


def load_layout(name_or_path: str | Path) -> Layout:
    """Load a built-in layout by name (``"sats_4x4"``) or any YAML path."""
    p = Path(name_or_path)
    if p.suffix not in (".yaml", ".yml"):
        p = LAYOUT_DIR / f"{name_or_path}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"layout not found: {p} (built-ins: {available_layouts()})")
    return layout_from_dict(yaml.safe_load(p.read_text()))


def available_layouts() -> list[str]:
    return sorted(p.stem for p in LAYOUT_DIR.glob("*.yaml"))


def grid_layout(
    rows: int,
    cols: int,
    pitch: float,
    *,
    name: str = "grid",
    parent: str = "sensor",
    parent_frame: str = "sensor",
    units: str = "mm",
    id_prefix: str = "S",
) -> Layout:
    """Planar ``rows × cols`` grid centred on the parent origin, normals +z.

    Channel order is row-major with x fastest, matching SATS
    (``deformable_sats/sats/training/local_map_module.py``: S1 = (-x, -y), S4 = (+x, -y)).
    Groups: ``row{r}``, ``col{c}`` and ``all``.
    """
    s = _UNIT_SCALE[units]
    xs = (np.arange(cols) - (cols - 1) / 2.0) * pitch
    ys = (np.arange(rows) - (rows - 1) / 2.0) * pitch
    taxels = []
    for r in range(rows):
        for c in range(cols):
            k = r * cols + c
            taxels.append(Taxel(id=f"{id_prefix}{k + 1}", channel=k, parent=parent,
                                position=np.array([xs[c], ys[r], 0.0]) * s,
                                normal=np.array([0.0, 0.0, 1.0]),
                                groups=(f"row{r}", f"col{c}", "all")))
    return Layout(name=name, parent_frame=parent_frame, taxels=tuple(taxels),
                  meta={"rows": rows, "cols": cols, "pitch": pitch, "pitch_units": units})
