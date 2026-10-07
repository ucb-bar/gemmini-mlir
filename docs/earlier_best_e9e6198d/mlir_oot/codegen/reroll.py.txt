"""Loop RE-ROLLING over the emitted command stream.

A tiled contraction emits the same few commands once per tile, with every operand field advancing by
a constant step. Emitted straight-line, the PROGRAM then grows with the payload -- which is the
defect the route-quality gate names as "the host still works at element scale". Rolling those
repeats back into a loop keeps the issued command sequence identical while the emitted code stops
scaling: the host computes addresses, issues commands and branches, which is exactly the division of
labour the contract permits.

The transform is verified by RE-EXPANSION: :func:`reroll` returns a node list only when expanding it
reproduces the input stream instruction for instruction. There is no case where it can silently
change the program.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..target import isa

#: Loop bodies longer than this are not searched for; a tiled nest's inner body is small.
MAX_PERIOD = 96
#: Below this trip count a loop costs more code than it saves.
MIN_TRIP = 4
#: A rolled loop pays per-iteration address arithmetic and a branch. Unrolling the body amortises
#: that over several commands, which matters because the cycle-accurate tier measures the SCALAR
#: core too -- the point of rolling is to stop the CODE growing with the payload, not to hand the
#: core more work per command.
UNROLL = 4
#: Only unroll when there are enough iterations left for the loop to still be worth having.
MIN_UNROLL_TRIP = 32


@dataclass
class Rolled:
    """``trip`` repeats of ``body``, where repeat ``i`` adds ``i * step`` to each operand."""

    trip: int
    body: list[isa.Instr]
    steps: list[tuple[int, int]]

    @property
    def saved(self) -> int:
        return len(self.body) * (self.trip - 1)


Node = isa.Instr | str | Rolled


def _operand_step(a, b) -> int | None:
    """The constant difference from operand ``a`` to ``b``, or None if they are not comparable."""
    if isinstance(a, isa.Imm) and isinstance(b, isa.Imm):
        return b.value - a.value
    if isinstance(a, isa.ArgAddr) and isinstance(b, isa.ArgAddr) and a.tensor == b.tensor:
        return b.byte_offset - a.byte_offset
    return None


def _advance(op, step: int):
    if isinstance(op, isa.Imm):
        return isa.Imm(op.value + step)
    return isa.ArgAddr(op.tensor, op.byte_offset + step)


def _advance_instr(base: isa.Instr, i: int, step: tuple[int, int]) -> isa.Instr:
    return isa.Instr(
        base.cls, _advance(base.rs1, i * step[0]), _advance(base.rs2, i * step[1]), base.note
    )


def _emit_rolled(nodes: list, body: list, steps: list[tuple[int, int]], trip: int) -> None:
    """Append the loop for ``trip`` repeats of ``body``, unrolled where that is profitable."""
    factor = UNROLL if trip >= MIN_UNROLL_TRIP else 1
    iterations = trip // factor
    wide_body: list[isa.Instr] = []
    wide_steps: list[tuple[int, int]] = []
    for i in range(factor):
        for j, base in enumerate(body):
            wide_body.append(_advance_instr(base, i, steps[j]))
            wide_steps.append((steps[j][0] * factor, steps[j][1] * factor))
    nodes.append(Rolled(trip=iterations, body=wide_body, steps=wide_steps))
    for i in range(iterations * factor, trip):
        for j, base in enumerate(body):
            nodes.append(_advance_instr(base, i, steps[j]))


def _period_trip(stream: list, start: int, period: int) -> tuple[int, list[tuple[int, int]]] | None:
    """How many times the window at ``start`` repeats with a CONSTANT per-operand step."""
    n = len(stream)
    if start + 2 * period > n:
        return None
    first = stream[start : start + period]
    if any(isinstance(x, str) for x in first):
        return None
    steps: list[tuple[int, int]] = []
    for j in range(period):
        a, b = stream[start + j], stream[start + period + j]
        if isinstance(b, str) or a.cls != b.cls:
            return None
        s1, s2 = _operand_step(a.rs1, b.rs1), _operand_step(a.rs2, b.rs2)
        if s1 is None or s2 is None:
            return None
        steps.append((s1, s2))
    trip = 2
    while start + (trip + 1) * period <= n:
        ok = True
        for j in range(period):
            a, b = stream[start + j], stream[start + trip * period + j]
            if isinstance(b, str) or a.cls != b.cls:
                ok = False
                break
            if _operand_step(a.rs1, b.rs1) != steps[j][0] * trip:
                ok = False
                break
            if _operand_step(a.rs2, b.rs2) != steps[j][1] * trip:
                ok = False
                break
        if not ok:
            break
        trip += 1
    return (trip, steps) if trip >= MIN_TRIP else None


def expand(nodes: list[Node]) -> list:
    out: list = []
    for n in nodes:
        if isinstance(n, Rolled):
            for i in range(n.trip):
                for j, base in enumerate(n.body):
                    s1, s2 = n.steps[j]
                    out.append(
                        isa.Instr(base.cls, _advance(base.rs1, i * s1),
                                  _advance(base.rs2, i * s2), base.note)
                    )
        else:
            out.append(n)
    return out


def _same(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return a.cls == b.cls and a.rs1 == b.rs1 and a.rs2 == b.rs2


def reroll(stream: list) -> list[Node]:
    """Roll every profitable repeat in ``stream``; verified by re-expansion before returning."""
    nodes: list[Node] = []
    i = 0
    n = len(stream)
    while i < n:
        if isinstance(stream[i], str):
            nodes.append(stream[i])
            i += 1
            continue
        best: tuple[int, int, list[tuple[int, int]]] | None = None
        for period in range(1, min(MAX_PERIOD, (n - i) // 2) + 1):
            found = _period_trip(stream, i, period)
            if found is None:
                continue
            trip, steps = found
            saved = period * (trip - 1)
            if best is None or saved > best[0]:
                best = (saved, period, steps, trip)
        if best is not None and best[0] > 0:
            _, period, steps, trip = best
            _emit_rolled(nodes, list(stream[i : i + period]), steps, trip)
            i += period * trip
        else:
            nodes.append(stream[i])
            i += 1
    if len(expand(nodes)) != n or not all(_same(a, b) for a, b in zip(expand(nodes), stream)):
        return list(stream)  # re-expansion disagreed: ship the stream as it was
    return nodes
