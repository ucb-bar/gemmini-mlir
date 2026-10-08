"""What this target can have IN FLIGHT at once, read off its loop unrollers' own Chisel.

Sister to :mod:`gemmini_accumulator_ports`, which answers what can meet in the accumulator. This
answers the other half of the same family of questions: whether one group's DMA can be issued while a
neighbouring group still computes. That is not an ISA property either -- the ISA has no fence operand
and no "issue early" bit -- it is three structural facts about the unrollers:

- **how many descriptors a unroller holds at once** (``concurrent_loops``). This is the run-ahead
  budget: with two slots, descriptor N+1's loads can start while descriptor N still computes, because
  each stage independently advances to the tail loop once the head loop's stage has started;
- **whether a mode is per-descriptor or per-module.** A mode register declared at MODULE scope is
  read by every stage on behalf of whichever descriptor that stage is currently serving -- not the
  descriptor that set it. Two descriptors that disagree about such a mode therefore cannot be in
  flight together, whatever the slot count says;
- **what a unroller admits while it has a loop configured** (its ``cmd.ready``). The two unrollers are
  chained, so one holding the command stream serializes the other's groups regardless of any software
  fence.

Every value here is the source's own text, reported verbatim; nothing is normalised into a verdict and
an expression this cannot find once is a refusal rather than a default.
"""

from __future__ import annotations

from merlin.targetgen.capability_discovery import _balanced_end

from .gemmini_accumulator_ports import PortsError, _blocks, definition, strip_comments


def loop_slots(source: str) -> int:
    """``concurrent_loops`` -- the number of descriptors this unroller holds at once.

    The run-ahead budget, and also the divisor that cuts the scratchpad and accumulator into per-slot
    regions. Stated once per unroller; found zero times or more than once is a refusal, because a
    budget read from the wrong declaration is worse than no budget.
    """
    text = strip_comments(source)
    hits = [line for line in text.splitlines() if line.strip().startswith("val concurrent_loops =")]
    if len(hits) != 1:
        raise PortsError(f"source states concurrent_loops {len(hits)} times; one declaration was expected")
    value = hits[0].split("=", 1)[1].strip()
    if not value.isdigit():
        raise PortsError(f"concurrent_loops is `{value}`, which this reader does not evaluate")
    return int(value)


def register_scope(source: str, name: str) -> str:
    """The class that declares ``val <name> = RegInit(...)``.

    The whole question a mode register raises: is it a field of the PER-DESCRIPTOR state, or a
    register of the module every stage reads? The answer is which class encloses the declaration.
    """
    text = strip_comments(source)
    needle, blocks, found = f"val {name} = RegInit(", _blocks(text), []
    cursor = 0
    while True:
        at = text.find(needle, cursor)
        if at < 0:
            break
        found.append(next(n for n, s, e in blocks if s <= at < e))
        cursor = at + len(needle)
    if len(found) != 1:
        raise PortsError(f"`{name}` is declared as a register {len(found)} times; one declaration was expected")
    return found[0]


def cleared_by_reset(source: str, state_class: str, name: str) -> bool:
    """Whether ``<state_class>.reset()`` assigns ``name``.

    Comments are stripped first, so a cleared-then-commented-out assignment reads as NOT cleared --
    which is the fact, and is exactly the case this exists to catch. A register a descriptor's reset
    does not clear carries the previous descriptor's value into the next one.
    """
    text = strip_comments(source)
    block = next((text[s:e] for n, s, e in _blocks(text) if n == state_class), None)
    if block is None:
        raise PortsError(f"source declares no class {state_class}")
    at = block.find("def reset(")
    if at < 0:
        raise PortsError(f"{state_class} declares no reset()")
    brace = block.find("{", at)
    if brace < 0:
        raise PortsError(f"{state_class}.reset() has no body this reader can bound")
    return f"{name} :=" in block[brace : _balanced_end(block, brace)]


def nonloop_admission(source: str, unroller: str) -> str:
    """The unroller's ``cmd.ready`` -- what it accepts from upstream, and when.

    A unroller that gates non-loop commands on ``!loop_configured`` holds the entire upstream stream
    while it has a descriptor live. The two unrollers are chained (the convolution sequencer's output
    is the matmul unroller's input), so each one's live descriptor serializes the other's groups.
    """
    return definition(source, unroller, "cmd.ready")
