"""Optional first-tile peeling for an ordinary CPU reduction loop.

``first`` is None in the incumbent loop, True for the peeled tile, and
False for the retained remainder loop. It is a compiler-time marker, never
an operand-value predicate. The body owns the real load/compute commands.
"""
from pathlib import Path
from merlin.compile.tuning_knobs import knob


def emit_reduction(f, k: int, body):
    """Emit positive-K reduction; call ``body(kk, first)`` for each tile.

    Disabled emission retains the exact incumbent 0:K:16 loop. Enabled
    emission executes the offset-zero tile once and retains a 16:K:16 remainder,
    avoiding Python unrolling and any remainder when K <= 16.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError('reduction K must be a positive static integer')
    peel = knob('peel_reduction', default=False, values=(False, True),
                root=Path(__file__).resolve().parents[1])
    if not peel:
        f.for_loop(0, k, 16, lambda kk: body(kk, None), retain_loop=True)
        return
    body(f.const(0), True)
    if k > 16:
        f.for_loop(16, k, 16, lambda kk: body(kk, False), retain_loop=True)
