"""Touch-grid environment (stub).

Plan: a hand (URDF, or MANO for glove transfer) in a physics sim (MuJoCo / Isaac) with a grid
of touch targets. Per-taxel contact force → clean ΔS via a per-taxel linear+saturating
response, then :class:`~robot_skin.sim.domain_rand.TaxelDomainRandomizer`; observations built
by ``robot_skin.policy.ObservationBuilder`` so the ablation modes are identical in sim/real.
Task: reach and press the lit grid cell with a target ordinal level.
"""
from __future__ import annotations


class TouchGridEnv:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TouchGridEnv not implemented yet (see module docstring)")
