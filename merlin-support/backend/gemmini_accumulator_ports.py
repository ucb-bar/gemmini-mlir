"""Every site where this target's loop unrollers hand the accumulator a local address.

The accumulator is the only place two tensors can meet without a DRAM round trip, so the question
"can operation X be folded into operation Y's readout?" is always the same question about this unit:
**how many DRAM operands can reach one accumulator region, and which of them ADD rather than
REPLACE?** The answer is not in the ISA header. The header says an accumulator address carries an
``accumulate`` bit (``LocalAddr``), so every fold looks encodable; what decides is which *literal*
each unroller passes for that bit at each of its own mvin/preload/mvout sites, and those literals are
in the Chisel, not in any macro.

So this reads them. For each ``cast_to_acc_addr(...)`` call in an unroller source it records the
enclosing Chisel class, what kind of command the address is being built for, and the ``accumulate``
and ``read_full`` arguments verbatim. Nothing is interpreted and no value is assumed: a site this
cannot classify is reported as :data:`UNCLASSIFIED` rather than dropped, because a silently dropped
site is an accumulate port that a caller would conclude does not exist.

Read as data, not prose: :func:`accumulating_loads` answers "which DRAM loads ADD to the
accumulator", which is the whole question a fusion asks.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.targetgen.capability_discovery import _balanced_end, _split_top_level

#: The role of a site whose command wire this module does not recognise. Fail closed: a renamed wire
#: must surface here, never vanish from the table.
UNCLASSIFIED = "UNCLASSIFIED"

_CAST = "cast_to_acc_addr("
#: Command-wire name token -> what the address is for. These are the unrollers' own wire names
#: (``mvin_cmd_rs2``, ``pre_cmd_rs2``, ``mvout_cmd_rs2``, ``pool_mvout_cmd_rs2``, ``ln_mvout_cmd_rs2``),
#: and a store is recognised before a load because a store wire may carry either token.
_STORE, _LOAD, _PRELOAD = "store", "load", "preload"


class PortsError(ValueError):
    """A source this cannot read structurally. Never softened into a default."""


@dataclass(frozen=True)
class AccPort:
    """One accumulator-addressing site, exactly as the source spells it."""

    #: The Chisel class the site is in -- the unroller stage that owns the port.
    module: str
    #: ``load`` (DRAM -> accumulator mvin), ``store`` (accumulator -> DRAM mvout), ``preload`` (the
    #: mesh's own output address), or :data:`UNCLASSIFIED`.
    role: str
    #: The ``accumulate =`` argument, verbatim. ``false.B`` REPLACES the accumulator contents;
    #: ``true.B`` adds to them; anything else is a condition, carried through as written.
    accumulate: str
    #: The ``read_full =`` argument, verbatim.
    read_full: str
    #: The address expression the site casts.
    address: str


def strip_comments(source: str) -> str:
    """``source`` with ``//`` and ``/* */`` comments blanked out, string literals preserved.

    Line-for-line: a comment becomes spaces rather than disappearing, so every offset and line number
    still points where it did. Necessary before any bracket walk -- these sources carry commented-out
    code (a disabled ``when`` guard on the bias pointer, among others) and apostrophes inside prose
    comments, either of which would derail a scan that took the file at face value.
    """
    if '"""' in source:
        raise PortsError("triple-quoted string in source; this reader does not model one")
    out, i, n = [], 0, len(source)
    while i < n:
        ch = source[i]
        if ch == '"':
            out.append(ch)
            i += 1
            while i < n:
                out.append(source[i])
                if source[i] == "\\" and i + 1 < n:
                    out.append(source[i + 1])
                    i += 2
                    continue
                if source[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if source.startswith("/*", i):
            while i < n and not source.startswith("*/", i):
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _blocks(source: str) -> list[tuple[str, int, int]]:
    """``(class name, start offset, end offset)`` for every top-level Chisel class in ``source``."""
    lines, found, offset = source.splitlines(keepends=True), [], 0
    for line in lines:
        for keyword in ("class ", "object ", "abstract class "):
            if line.startswith(keyword):
                rest = line[len(keyword) :]
                name = rest.split("(", 1)[0].split("[", 1)[0].split(" ", 1)[0].strip()
                found.append((name, offset))
                break
        offset += len(line)
    if not found:
        raise PortsError("source declares no top-level class")
    return [(n, s, found[i + 1][1] if i + 1 < len(found) else len(source)) for i, (n, s) in enumerate(found)]


def _role(receiver: str) -> str:
    """What kind of command a ``<wire>.local_addr`` assignment is building, from the wire's own name."""
    tokens = receiver.split(".")[0].split("_")
    if "mvout" in tokens:
        return _STORE
    if "mvin" in tokens:
        return _LOAD
    if "pre" in tokens:
        return _PRELOAD
    return UNCLASSIFIED


def _named(args: list[str], name: str, where: str) -> str:
    """The value of the ``name = value`` argument among ``args``. A positional call is a refusal."""
    for arg in args:
        key, sep, value = arg.partition("=")
        if sep and key.strip() == name:
            return value.strip()
    raise PortsError(f"{where}: cast_to_acc_addr has no `{name} =` argument; this reader does not read it positionally")


def accumulator_ports(source: str) -> tuple[AccPort, ...]:
    """Every ``cast_to_acc_addr`` site in one unroller source, in source order."""
    text = strip_comments(source)
    blocks = _blocks(text)

    def enclosing(offset: int) -> str:
        for name, start, end in blocks:
            if start <= offset < end:
                return name
        raise PortsError(f"offset {offset} is in no class block")

    ports, cursor = [], 0
    while True:
        at = text.find(_CAST, cursor)
        if at < 0:
            return tuple(ports)
        open_paren = at + len(_CAST) - 1
        close = _balanced_end(text, open_paren)
        args = _split_top_level(text[open_paren + 1 : close - 1], ",")
        module = enclosing(at)
        # The assignment's left-hand side is the command wire the address is being built for.
        line_start = text.rfind("\n", 0, at) + 1
        receiver = text[line_start:at].split(":=", 1)[0].strip()
        if len(args) < 2:
            raise PortsError(f"{module}: cast_to_acc_addr takes {len(args)} argument(s)")
        ports.append(
            AccPort(
                module=module,
                role=_role(receiver),
                accumulate=_named(args, "accumulate", module),
                read_full=_named(args, "read_full", module),
                address=" ".join(args[1].split()),
            )
        )
        cursor = close


def accumulating_loads(source: str) -> tuple[str, ...]:
    """The modules whose DRAM -> accumulator mvin ADDS to the accumulator instead of replacing it.

    This is the question a fusion asks. An operand can join something already in the accumulator only
    through a load port whose ``accumulate`` argument is not the literal ``false.B``; a port that is
    unconditionally ``false.B`` can carry exactly one tensor into a region, and a second load through
    it discards the first.
    """
    return tuple(p.module for p in accumulator_ports(source) if p.role == _LOAD and p.accumulate != "false.B")


def unclassified(source: str) -> tuple[AccPort, ...]:
    """Sites whose command wire this reader did not recognise. Non-empty means the table is incomplete."""
    return tuple(p for p in accumulator_ports(source) if p.role == UNCLASSIFIED)


def definition(source: str, module: str, name: str) -> str:
    """The right-hand side of ``val <name> = ...`` or ``<name> := ...`` inside ``module``.

    Whitespace-normalised and continued across lines until its brackets balance, so a multi-line
    ``Mux`` comes back whole. Not found, or found twice, is a refusal: a caller asking what a field is
    wired to must not be answered with one of two answers.
    """
    text = strip_comments(source)
    block = next((text[s:e] for n, s, e in _blocks(text) if n == module), None)
    if block is None:
        raise PortsError(f"source declares no class {module}")

    def depth(piece: str) -> int:
        return sum(piece.count(c) for c in "([{") - sum(piece.count(c) for c in ")]}")

    lines, hits = block.splitlines(), []
    for index, line in enumerate(lines):
        stripped = line.strip()
        for opener in (f"val {name} =", f"{name} :=", f"{name} ="):
            if stripped.startswith(opener) and not stripped.startswith(f"{opener}="):
                rest, taken = stripped[len(opener) :], index
                open_brackets = depth(rest)
                while open_brackets > 0 and taken + 1 < len(lines):
                    taken += 1
                    rest += " " + lines[taken].strip()
                    open_brackets += depth(lines[taken])
                hits.append(" ".join(rest.split()))
                break
    if len(hits) != 1:
        raise PortsError(f"{module}.{name} is defined {len(hits)} times; a single wiring was expected")
    return hits[0]
