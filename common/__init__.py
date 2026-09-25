"""Shared, dependency-light building blocks for ``robot_skin`` and ``deformable_sats``.

Dependency direction (one way only)::

    robot_skin  ──▶  common  ◀──  deformable_sats

``common`` must never import ``robot_skin`` or ``sats``/``hitmap``; it depends only on
numpy (+ PyYAML for layout files). ``common/tests/test_dependency_direction.py`` enforces this.
"""
