"""robot_skin — whole-hand tactile skin (glove ⇄ robot hand) on top of ``common``.

Pipeline::

    acquisition → pose (taxel poses at t) → baseline (motion-induced ΔS prediction)
      → contact (residual → ordinal / FSM / masks) → representation (taxel tokens)
      → policy / vtla ; sim + transfer feed training ; eval scores all of it.

Depends on ``common`` only — never on ``deformable_sats`` (``sats``/``hitmap``).
"""
