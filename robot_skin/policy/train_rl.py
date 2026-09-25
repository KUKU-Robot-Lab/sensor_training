"""RL training entry point (stub).

Plan: PPO on ``sim.TouchGridEnv`` for each ``policy.OBS_MODES`` ablation × seeds, same reward;
evaluate on hardware with ``eval.metrics`` (hallucination rate, recovery time) and task success.
"""
from __future__ import annotations

import argparse
import sys

from .observation import OBS_MODES


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="train_rl", description="Tactile-ablation RL training (stub).")
    p.add_argument("--obs-mode", choices=OBS_MODES, default="ordinal")
    p.add_argument("--seed", type=int, default=0)
    p.parse_args(argv)
    raise NotImplementedError("RL training not implemented yet (needs sim.TouchGridEnv)")


if __name__ == "__main__":
    sys.exit(main())
