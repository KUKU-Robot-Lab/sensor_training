"""EtherMotion node profile parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NodeProfile:
    """Summary of the numeric X/Y/Z/U commands in an EtherMotion node file."""

    path: Path
    row_count: int
    xy_count: int
    grid_size_x: int
    grid_size_y: int
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    xy_step_mm: float
    z_min_mm: float
    z_max_mm: float
    z_step_mm: float
    z_depth_mm: float
    u_values_mm: tuple[float, ...]
    first_u_cycle_mm: tuple[float, ...]


def _parse_float(value: str) -> float:
    return float(value.strip())


def _positive_min_step(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    diffs = [
        round(values[i + 1] - values[i], 10)
        for i in range(len(values) - 1)
        if values[i + 1] > values[i]
    ]
    return min(diffs) if diffs else 0.0


def _rounded_unique(values: list[float]) -> tuple[float, ...]:
    return tuple(sorted({round(v, 10) for v in values}))


def _first_u_cycle(records: list[tuple[float, float, float, float]]) -> tuple[float, ...]:
    if not records:
        return ()
    first_x, first_y, first_z, _ = records[0]
    cycle = [
        u
        for x, y, z, u in records
        if x == first_x and y == first_y and z == first_z
    ]
    return tuple(round(u, 10) for u in cycle[:3])


def parse_node_profile(path: Path | str) -> NodeProfile:
    """Parse an EtherMotion node file and summarize numeric motion commands."""

    node_path = Path(path)
    records: list[tuple[float, float, float, float]] = []

    with open(node_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if len(parts) < 6 or not parts[0].isdigit():
                continue
            try:
                x = _parse_float(parts[2])
                y = _parse_float(parts[3])
                z = _parse_float(parts[4])
                u = _parse_float(parts[5])
            except ValueError:
                continue
            records.append((x, y, z, u))

    if not records:
        raise ValueError(f"No numeric node records found: {node_path}")

    scan_records = [(x, y, z, u) for x, y, z, u in records if z > 0.0]
    if not scan_records:
        raise ValueError(f"No positive-Z scan records found: {node_path}")

    xs = _rounded_unique([r[0] for r in scan_records])
    ys = _rounded_unique([r[1] for r in scan_records])
    zs = _rounded_unique([r[2] for r in scan_records])
    us = _rounded_unique([r[3] for r in scan_records])
    xy = {(r[0], r[1]) for r in scan_records}
    xy_steps = [s for s in (_positive_min_step(xs), _positive_min_step(ys)) if s > 0.0]

    return NodeProfile(
        path=node_path,
        row_count=len(scan_records),
        xy_count=len(xy),
        grid_size_x=len(xs),
        grid_size_y=len(ys),
        x_min_mm=min(xs),
        x_max_mm=max(xs),
        y_min_mm=min(ys),
        y_max_mm=max(ys),
        xy_step_mm=min(xy_steps) if xy_steps else 0.0,
        z_min_mm=min(zs),
        z_max_mm=max(zs),
        z_step_mm=_positive_min_step(zs),
        z_depth_mm=round(max(zs) - min(zs), 10),
        u_values_mm=us,
        first_u_cycle_mm=_first_u_cycle(scan_records),
    )
