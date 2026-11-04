"""RV64 instruction boundaries and conservative register facts from actual words.

This is a CPU frontend, not an accelerator emulator. Accelerator meaning comes
from the caller's selected operand observer. Unknown writes, calls and joins
discard facts; a static instruction site never supplies execution multiplicity.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, replace
from itertools import product

MASK = (1 << 64) - 1
_REGIONS = OrderedDict()


def signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


@dataclass(frozen=True)
class CPUInstruction:
    kind: str
    size: int = 4
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0
    target: int | None = None
    bits: int = 64


def decode_cpu(row):
    """Decode standard RV64GC scalar effects without relying on objdump aliases."""
    if row.relocations:
        return CPUInstruction("opaque", 2 if len(row.hexcode) == 4 else 4)
    if not row.hexcode:
        return CPUInstruction("opaque")
    w = int(row.hexcode, 16)
    pc = row.addr
    if w & 3 != 3:
        if w >= 1 << 16:
            return CPUInstruction("opaque", 2)
        q, f = w & 3, w >> 13
        rd, r1, r2 = (w >> 7) & 31, 8 + ((w >> 7) & 7), 8 + ((w >> 2) & 7)
        ci = signed(((w >> 2) & 31) | ((w >> 12) & 1) << 5, 6)
        if q == 0:
            if f == 0:
                imm = ((w >> 7) & 15) << 6 | ((w >> 11) & 3) << 4 | ((w >> 5) & 1) << 3 | ((w >> 6) & 1) << 2
                return CPUInstruction("addi", 2, r2, 2, imm=imm) if imm else CPUInstruction("opaque", 2)
            if f in (2, 6):
                imm = ((w >> 5) & 1) << 6 | ((w >> 10) & 7) << 3 | ((w >> 6) & 1) << 2
                return CPUInstruction("load" if f == 2 else "store", 2, r2 if f == 2 else 0, r1, r2, imm, bits=32)
            if f in (3, 7):
                imm = ((w >> 5) & 3) << 6 | ((w >> 10) & 7) << 3
                return CPUInstruction("load" if f == 3 else "store", 2, r2 if f == 3 else 0, r1, r2, imm)
            return CPUInstruction("fp_memory" if f in (1, 5) else "opaque", 2)
        if q == 1:
            if f in (0, 1, 2):
                if f == 1 and rd == 0:
                    return CPUInstruction("opaque", 2)
                return CPUInstruction("addi", 2, rd, 0 if f == 2 else rd, imm=ci, bits=32 if f == 1 else 64)
            if f == 3:
                if rd == 2:
                    imm = signed(
                        ((w >> 12) & 1) << 9
                        | ((w >> 3) & 3) << 7
                        | ((w >> 5) & 1) << 6
                        | ((w >> 2) & 1) << 5
                        | ((w >> 6) & 1) << 4,
                        10,
                    )
                    return CPUInstruction("addi", 2, 2, 2, imm=imm) if imm else CPUInstruction("opaque", 2)
                if rd == 0 or ci == 0:
                    return CPUInstruction("opaque", 2)
                return CPUInstruction("lui", 2, rd, imm=ci << 12)
            if f == 4:
                mode = (w >> 10) & 3
                if mode != 3:
                    return CPUInstruction(
                        ("srli", "srai", "andi")[mode],
                        2,
                        r1,
                        r1,
                        imm=ci if mode == 2 else ((w >> 2) & 31) | ((w >> 12) & 1) << 5,
                    )
                op = (w >> 5) & 3
                if w & (1 << 12):
                    return CPUInstruction(("sub", "add", "unknown", "unknown")[op], 2, r1, r1, r2, bits=32)
                return CPUInstruction(("sub", "xor", "or", "and")[op], 2, r1, r1, r2)
            if f == 5:
                imm = signed(
                    ((w >> 12) & 1) << 11
                    | ((w >> 8) & 1) << 10
                    | ((w >> 9) & 3) << 8
                    | ((w >> 6) & 1) << 7
                    | ((w >> 7) & 1) << 6
                    | ((w >> 2) & 1) << 5
                    | ((w >> 11) & 1) << 4
                    | ((w >> 3) & 7) << 1,
                    12,
                )
                return CPUInstruction("jump", 2, target=pc + imm)
            if f in (6, 7):
                imm = signed(
                    ((w >> 12) & 1) << 8
                    | ((w >> 5) & 3) << 6
                    | ((w >> 2) & 1) << 5
                    | ((w >> 10) & 3) << 3
                    | ((w >> 3) & 3) << 1,
                    9,
                )
                return CPUInstruction("beq" if f == 6 else "bne", 2, rs1=r1, target=pc + imm)
        if q == 2:
            if f == 0:
                return CPUInstruction("slli", 2, rd, rd, imm=((w >> 2) & 31) | ((w >> 12) & 1) << 5)
            if f in (2, 3):
                if rd == 0:
                    return CPUInstruction("opaque", 2)
                imm = (
                    (((w >> 2) & 3) << 6 | ((w >> 12) & 1) << 5 | ((w >> 4) & 7) << 2)
                    if f == 2
                    else (((w >> 2) & 7) << 6 | ((w >> 12) & 1) << 5 | ((w >> 5) & 3) << 3)
                )
                return CPUInstruction("load", 2, rd, 2, imm=imm, bits=32 if f == 2 else 64)
            if f in (6, 7):
                imm = (
                    (((w >> 7) & 3) << 6 | ((w >> 9) & 15) << 2)
                    if f == 6
                    else (((w >> 7) & 7) << 6 | ((w >> 10) & 7) << 3)
                )
                return CPUInstruction("store", 2, rs1=2, rs2=(w >> 2) & 31, imm=imm, bits=32 if f == 6 else 64)
            if f == 4:
                rs = (w >> 2) & 31
                if not rs:
                    if rd == 0:
                        return CPUInstruction("opaque", 2)
                    return CPUInstruction(
                        "call_indirect" if w & (1 << 12) else "return" if rd in (1, 5) else "indirect",
                        2,
                        1 if w & (1 << 12) else 0,
                        rd,
                    )
                return CPUInstruction("add", 2, rd, rd if w & (1 << 12) else 0, rs)
            return CPUInstruction("fp_memory" if f in (1, 5) else "opaque", 2)
        return CPUInstruction("opaque", 2)
    if w & 31 == 31:
        return CPUInstruction("opaque")
    opcode, rd, f, r1, r2, f7 = w & 127, (w >> 7) & 31, (w >> 12) & 7, (w >> 15) & 31, (w >> 20) & 31, w >> 25
    imm = signed(w >> 20, 12)
    if opcode in (0x37, 0x17):
        return CPUInstruction("lui" if opcode == 0x37 else "auipc", rd=rd, imm=signed(w & 0xFFFFF000, 32))
    if opcode in (0x13, 0x1B):
        if opcode == 0x1B and f not in (0, 1, 5):
            return CPUInstruction("opaque")
        kind = {0: "addi", 2: "slti", 3: "sltiu", 4: "xori", 6: "ori", 7: "andi"}.get(f, "unknown")
        if f == 1:
            kind, imm = (
                ("slli" if w >> (25 if opcode == 0x1B else 26) == 0 else "unknown"),
                (w >> 20) & (31 if opcode == 0x1B else 63),
            )
        elif f == 5:
            top = w >> (25 if opcode == 0x1B else 26)
            kind, imm = (
                {0: "srli", 0x20 if opcode == 0x1B else 0x10: "srai"}.get(top, "unknown"),
                (w >> 20) & (31 if opcode == 0x1B else 63),
            )
        return CPUInstruction(kind, rd=rd, rs1=r1, imm=imm, bits=32 if opcode == 0x1B else 64)
    if opcode in (0x33, 0x3B):
        if opcode == 0x3B and f not in {0: (0, 1, 5), 0x20: (0, 5), 1: (0, 4, 5, 6, 7)}.get(f7, ()):
            return CPUInstruction("opaque")
        kind = (
            {0: "add", 1: "sll", 2: "slt", 3: "sltu", 4: "xor", 5: "srl", 6: "or", 7: "and"}.get(f)
            if f7 == 0
            else {0: "sub", 5: "sra"}.get(f)
            if f7 == 0x20
            else {0: "mul", 4: "div", 5: "divu", 6: "rem", 7: "remu"}.get(f)
            if f7 == 1
            else None
        )
        return CPUInstruction(kind or "unknown", rd=rd, rs1=r1, rs2=r2, bits=32 if opcode == 0x3B else 64)
    if opcode == 0x03:
        if f == 7:
            return CPUInstruction("opaque")
        return CPUInstruction("load_unsigned" if f >= 4 else "load", rd=rd, rs1=r1, imm=imm, bits=8 << (f & 3))
    if opcode == 0x23:
        if f > 3:
            return CPUInstruction("opaque")
        return CPUInstruction("store", rs1=r1, rs2=r2, imm=signed((w >> 25) << 5 | (w >> 7) & 31, 12), bits=8 << f)
    if opcode == 0x63:
        imm = signed((w >> 31) << 12 | ((w >> 7) & 1) << 11 | ((w >> 25) & 63) << 5 | ((w >> 8) & 15) << 1, 13)
        return CPUInstruction(
            {0: "beq", 1: "bne", 4: "blt", 5: "bge", 6: "bltu", 7: "bgeu"}.get(f, "opaque"),
            rs1=r1,
            rs2=r2,
            target=pc + imm,
        )
    if opcode == 0x6F:
        imm = signed((w >> 31) << 20 | ((w >> 12) & 255) << 12 | ((w >> 20) & 1) << 11 | ((w >> 21) & 1023) << 1, 21)
        return CPUInstruction("call" if rd in (1, 5) else "jump", rd=rd, target=pc + imm)
    if opcode == 0x67:
        if f != 0:
            return CPUInstruction("opaque")
        return CPUInstruction(
            "call_indirect" if rd in (1, 5) else "return" if r1 in (1, 5) and not imm and not rd else "indirect",
            rd=rd,
            rs1=r1,
            imm=imm,
        )
    if opcode == 0x0F:
        return CPUInstruction("fence" if f in (0, 1) else "opaque")
    if opcode == 0x73 and f == 0:
        return CPUInstruction("opaque")
    if opcode in (0x07, 0x27):
        return CPUInstruction("fp_memory")
    if opcode in (0x43, 0x47, 0x4B, 0x4F):
        return CPUInstruction("fp")
    if opcode == 0x53 and f7 not in (0x50, 0x51, 0x60, 0x61, 0x70, 0x71):
        return CPUInstruction("fp")
    # Includes custom opcodes: their result register is unknown until observed.
    return CPUInstruction("unknown", rd=rd)


BRANCHES = frozenset(("beq", "bne", "blt", "bge", "bltu", "bgeu"))
CALLS = frozenset(("call", "call_indirect"))


def section_of(row):
    return getattr(row, "section_name", "") or row.section or "unlabelled"


def _ends_segment(d):
    """An unconditional transfer: control never falls through to the next word.

    Like a return, a direct or register jump ends the current segment, so a
    tail call (``jal zero``/``c.j``/``jalr zero``) no longer runs into the next
    function. A jump that stays inside a function (a loop back edge placed last,
    a jump over a block) rejoins its code through the branch-reach pass below.
    """
    return d.kind in ("return", "jump", "indirect")


def region_index(rows, *, artifact_sha256, custom_opcode=None):
    """Partition complete executable sections; preserve every selected custom site.

    Direct targets, returns and unconditional jumps suggest boundaries. Branch-crossed intervals are
    joined. These are static code spans, not claimed source function extents.
    """
    cache_key = (
        artifact_sha256,
        custom_opcode,
        len(rows),
        rows[0].addr if rows else None,
        rows[-1].addr if rows else None,
    )
    if cache_key in _REGIONS:
        _REGIONS.move_to_end(cache_key)
        return _REGIONS[cache_key]
    sections = {}
    # A startup/driver section can call a kernel in another executable
    # section. Its target is also a boundary for tail transfers in that code.
    call_targets = {d.target for row in rows if (d := decode_cpu(row)).kind == "call"}
    for first, second in zip(rows, rows[1:]):
        high, low = decode_cpu(first), decode_cpu(second)
        if (
            high.kind == "auipc"
            and high.rd
            and low.kind == "call_indirect"
            and low.rs1 == high.rd
            and first.addr + high.size == second.addr
            and section_of(first) == section_of(second)
        ):
            call_targets.add((first.addr + high.imm + low.imm) & MASK & ~1)
    for row in rows:
        sections.setdefault(section_of(row), []).append(row)
    regions = []
    for section, body in sections.items():
        body = sorted(body, key=lambda r: r.addr)
        decoded = [decode_cpu(r) for r in body]
        starts = [0]
        for i in range(1, len(body)):
            if (
                body[i].addr in call_targets
                or _ends_segment(decoded[i - 1])
                or body[i].addr != body[i - 1].addr + decoded[i - 1].size
            ):
                starts.append(i)
        bounds = starts + [len(body)]
        owner = [0] * len(body)
        for k, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            owner[lo:hi] = [k] * (hi - lo)
        addresses = {r.addr: i for i, r in enumerate(body)}
        reach = list(range(len(starts)))
        for i, d in enumerate(decoded):
            if d.kind not in BRANCHES | {"jump"} or d.target not in addresses:
                continue
            if d.kind == "jump" and d.target in call_targets:
                continue
            lo, hi = sorted((owner[i], owner[addresses[d.target]]))
            reach[lo] = max(reach[lo], hi)
        k = 0
        while k < len(starts):
            end, cursor = reach[k], k
            while cursor <= end:
                end = max(end, reach[cursor])
                cursor += 1
            selected = body[bounds[k] : bounds[end + 1]]
            start, stop = selected[0].addr, selected[-1].addr + decode_cpu(selected[-1]).size
            counts = Counter()
            for r in selected:
                if r.hexcode and len(r.hexcode) == 8 and int(r.hexcode, 16) & 127 == custom_opcode:
                    counts[int(r.hexcode, 16) >> 25] += 1
            import hashlib

            identity = hashlib.sha256(f"{artifact_sha256}:{section}:{start:x}:{stop:x}".encode()).hexdigest()
            regions.append(
                (
                    dict(
                        region_id="region-" + identity[:24],
                        section=section,
                        entry_pc=start,
                        address_range=[start, stop],
                        boundary_status="inferred_static_span",
                        accelerator_sites=sum(counts.values()),
                        funct_sites=dict(sorted(counts.items())),
                    ),
                    selected,
                )
            )
            k = end + 1
    _REGIONS[cache_key] = regions
    while len(_REGIONS) > 3:
        _REGIONS.popitem(last=False)
    return regions


@dataclass(frozen=True)
class Bits:
    mask: int = 0
    value: int = 0
    base: str | None = None
    offset: int = 0
    # Post-fixpoint loop annotation (never part of the fixed point itself):
    # (constant, pointer_base, ((header_pc, step, count), ...)) meaning
    # value = [pointer_base +] constant + sum(step * k_header), 0 <= k < count.
    affine: tuple | None = None

    @classmethod
    def const(cls, value):
        return cls(MASK, value & MASK)

    @property
    def exact(self):
        return self.value if self.mask == MASK else None

    def join(self, other):
        mask = self.mask & other.mask & ~(self.value ^ other.value) & MASK
        pointer = self.base is not None and self.base == other.base and self.offset == other.offset
        return Bits(mask, self.value & mask, self.base if pointer else None, self.offset if pointer else 0)

    def record(self):
        return dict(
            kind="const" if self.exact is not None else "unknown",
            raw=self.exact,
            **(
                dict(known_bits=dict(mask=self.mask, value=self.value, width=64))
                if self.exact is None and self.mask
                else {}
            ),
            **(dict(pointer_base=self.base, byte_offset=self.offset) if self.base else {}),
            **(_affine_record(self.affine) if self.affine else {}),
        )


_VALUE_SET_LIMIT = 4096


def _affine_record(affine):
    if isinstance(affine, _Table):
        return dict(
            affine=dict(
                terms=[dict(header_pc=h, count=n) for h, n in affine.dims],
                basis="proved_trip_counts_pointwise_cpu_semantics",
            ),
            value_set=sorted(set(affine.values)),
        )
    constant, pointer, terms = affine
    out = dict(
        affine=dict(
            base=constant & MASK if pointer is None else None,
            **(dict(pointer_base=pointer, byte_offset=constant) if pointer else {}),
            terms=[dict(header_pc=h, step=step, count=count) for h, step, count in terms],
            basis="proved_induction_and_trip_counts",
        )
    )
    total = 1
    for _, _, count in terms:
        total *= count
    if pointer is None and total <= _VALUE_SET_LIMIT:
        values = {constant}
        for _, step, count in terms:
            values = {v + step * k for v in values for k in range(count)}
        out["value_set"] = sorted(v & MASK for v in values)
    return out


UNKNOWN = Bits()
ZERO = Bits.const(0)


def _sext(value, bits):
    low = (1 << bits) - 1
    mask, result = value.mask & low, value.value & low
    if mask & (1 << (bits - 1)):
        mask |= MASK ^ low
        if result & (1 << (bits - 1)):
            result |= MASK ^ low
    return Bits(mask, result)


def arithmetic(kind, a, b):
    av, bv = a.exact, b.exact
    if kind in ("add", "sub"):
        if bv is not None and a.base:
            delta = signed(bv, 64) * (1 if kind == "add" else -1)
            return Bits(base=a.base, offset=a.offset + delta)
        if av is not None and b.base and kind == "add":
            return Bits(base=b.base, offset=b.offset + signed(av, 64))
        if av is not None and bv is not None:
            return Bits.const(av + bv if kind == "add" else av - bv)
        # Standard known-bits addition with an explicit carry (a - b == a + ~b + 1).
        # A bit is known when both operand bits and its incoming carry are known.
        a_zero, a_one = a.mask & ~a.value & MASK, a.mask & a.value
        b_zero, b_one = b.mask & ~b.value & MASK, b.mask & b.value
        carry = 0
        if kind == "sub":
            b_zero, b_one, carry = b_one, b_zero, 1
        largest = ((~a_zero & MASK) + (~b_zero & MASK) + carry) & MASK
        smallest = (a_one + b_one + carry) & MASK
        carry_zero = ~(largest ^ a_zero ^ b_zero) & MASK
        carry_one = (smallest ^ a_one ^ b_one) & MASK
        known = (a_zero | a_one) & (b_zero | b_one) & (carry_zero | carry_one) & MASK
        return Bits(known, smallest & known)
    if kind == "and":
        zero = (a.mask & ~a.value) | (b.mask & ~b.value)
        ones = a.mask & a.value & b.mask & b.value
        return Bits((zero | ones) & MASK, ones)
    if kind == "or":
        ones = (a.mask & a.value) | (b.mask & b.value)
        zero = a.mask & ~a.value & b.mask & ~b.value
        return Bits((zero | ones) & MASK, ones)
    if kind == "xor":
        return Bits(a.mask & b.mask, (a.value ^ b.value) & a.mask & b.mask)
    if kind in ("sll", "srl", "sra") and bv is not None:
        amount = bv & 63
        if not amount:
            return a
        if kind == "sll":
            return Bits(((a.mask << amount) | ((1 << amount) - 1)) & MASK, (a.value << amount) & MASK)
        high = MASK ^ ((1 << (64 - amount)) - 1)
        sign_known = bool(a.mask & (1 << 63))
        fill_mask = high if kind == "srl" or sign_known else 0
        fill_value = high if kind == "sra" and sign_known and a.value >> 63 else 0
        return Bits((a.mask >> amount) | fill_mask, (a.value >> amount) | fill_value)
    if av is not None and bv is not None:
        if kind == "mul":
            return Bits.const(av * bv)
        if kind in ("slt", "sltu"):
            return Bits.const(int((signed(av, 64) < signed(bv, 64)) if kind == "slt" else av < bv))
    if kind in ("slt", "sltu"):
        return Bits(MASK ^ 1, 0)
    return UNKNOWN


# Stack-model key present while the region's own frame is private: no value
# derived from the entry stack pointer has left the tracked registers and
# frame slots. Joins intersect keys, so privacy survives a join only when every
# incoming path is private. A private frame lies below the entry stack pointer,
# which no other code owned at region entry, so only an address this region
# hands out can reach it. Writes by code outside this view (a word the base ISA
# does not define, a store through an untracked pointer) therefore keep slots
# below the entry stack pointer unless the frame has escaped. Calls still drop
# every slot, since the psABI lets a callee write its stack arguments. A
# dangling pointer kept from before region entry is outside this model.
_PRIVATE_FRAME = None
_FRAME = "entry_sp"


def _forget_unprivate(stack):
    """Forget slots that other code may address: all of them once escaped."""
    if _PRIVATE_FRAME not in stack:
        stack.clear()
        return
    for key in [key for key in stack if key is not _PRIVATE_FRAME]:
        if key[1] + key[2] // 8 > 0:
            del stack[key]


def _frame_handed_out(registers, *operands):
    """True when a frame address is visible to code outside this CPU view."""
    return any(registers[r].base == _FRAME for r in range(1, 32) if r != 2 or r in operands)


def _coarse_effect(word, d):
    """Refine words decode_cpu keeps coarse, using only base RV64GC encodings.

    Returns ("none", int_sources), ("store", base, imm, bits) or
    ("foreign", operands). "foreign" covers every word whose effects the base
    ISA does not define (custom or reserved opcodes, atomics, CSRs, vectors).
    """
    if not word:
        return ("foreign", (2,))
    w = int(word, 16)
    if len(word) == 4:
        q, f = w & 3, w >> 13
        if d.kind == "fp_memory" and f in (1, 5):
            if f == 1:
                return ("none", ())  # c.fld / c.fldsp write only an FP register
            if q == 0:
                return ("store", 8 + ((w >> 7) & 7), ((w >> 5) & 3) << 6 | ((w >> 10) & 7) << 3, 64)
            return ("store", 2, ((w >> 7) & 7) << 6 | ((w >> 10) & 7) << 3, 64)
        return ("foreign", (2,))
    opcode, funct3, rs1 = w & 127, (w >> 12) & 7, (w >> 15) & 31
    if d.kind == "fp":
        # Integer-to-FP converts (funct7 0b11010xx) and moves (0b11110xx) read
        # integer rs1; every other FP computation reads FP registers only.
        return ("none", (rs1,) if opcode == 0x53 and w >> 27 in (0x1A, 0x1E) else ())
    if opcode == 0x07:
        return ("none", ())  # LOAD-FP (scalar or vector) writes no memory
    if opcode == 0x27 and 1 <= funct3 <= 4:
        return ("store", rs1, signed((w >> 25) << 5 | (w >> 7) & 31, 12), 8 << funct3)
    if opcode == 0x53:
        return ("none", ())  # OP-FP compare/convert/move to an integer register
    return ("foreign", (rs1, (w >> 20) & 31))


# Absolute-address memory slots exist only when a caller seeds them through
# ``entry_state`` (asserting ordinary initialized memory, e.g. ``.data``). An
# untracked exact address may be a device register, so a store to it never
# creates a slot; a store through an unknown pointer forgets every slot.
_ABSOLUTE = "abs"


def _store(stack, address, bits, value):
    if value.base == _FRAME:
        stack.pop(_PRIVATE_FRAME, None)
    if address.base != _FRAME:
        exact = address.exact
        kept = {}
        if exact is not None:
            lo, hi = exact, exact + bits // 8
            kept = {
                key: slot
                for key, slot in stack.items()
                if key is not _PRIVATE_FRAME and key[0] == _ABSOLUTE and not (lo < key[1] + key[2] // 8 and key[1] < hi)
            }
            tracked = (_ABSOLUTE, exact, bits) in stack
        _forget_unprivate(stack)
        stack.update(kept)
        if exact is not None and tracked:
            stack[(_ABSOLUTE, exact, bits)] = value
        return
    lo, hi = address.offset, address.offset + bits // 8
    for old in [key for key in stack if key is not _PRIVATE_FRAME]:
        if old[0] == _FRAME and lo < old[1] + old[2] // 8 and old[1] < hi:
            del stack[old]
    if len(stack) < 256:
        stack[(address.base, address.offset, bits)] = value


_INTEGER_MAJOR_OPCODES = frozenset((0x03, 0x0F, 0x13, 0x17, 0x1B, 0x23, 0x33, 0x37, 0x3B, 0x63, 0x67, 0x6F))
_CALLER_SAVED = (1, 5, 6, 7, *range(10, 18), *range(28, 32))
_FP_CALLER_SAVED = (*range(0, 8), *range(10, 18), *range(28, 32))
_NAN_BOX = Bits.const(MASK ^ ((1 << 32) - 1))


def _fp_effect(word, d, registers, fp):
    """Apply a word's FP-register effect; return an exact FP-to-integer bit copy.

    ``fmv.x.w``/``fmv.x.d`` copy FP bits to an integer register and
    ``fmv.w.x``/``fmv.d.x`` copy integer bits into an FP register (single
    precision NaN-boxed). Every other word that may write an FP register makes
    it unknown; FP loads stay unknown because the rows carry no loaded bytes.
    """
    if d.kind == "opaque":
        fp[:] = [UNKNOWN] * 32
        return None
    if not word:
        return None
    w = int(word, 16)
    if len(word) == 4:
        q, f = w & 3, w >> 13
        if f == 1 and q == 0:
            fp[8 + ((w >> 2) & 7)] = UNKNOWN
        elif f == 1 and q == 2:
            fp[(w >> 7) & 31] = UNKNOWN
        return None
    opcode, rd, f3, r1, r2, f7 = w & 127, (w >> 7) & 31, (w >> 12) & 7, (w >> 15) & 31, (w >> 20) & 31, w >> 25
    if opcode in _INTEGER_MAJOR_OPCODES or opcode == 0x27:
        return None
    if opcode == 0x53 and f3 == 0 and r2 == 0 and f7 in (0x70, 0x71):
        return _sext(fp[r1], 32) if f7 == 0x70 else fp[r1]
    if opcode == 0x53 and f3 == 0 and r2 == 0 and f7 in (0x78, 0x79):
        source = registers[r1]
        fp[rd] = (
            arithmetic("or", arithmetic("and", source, Bits.const((1 << 32) - 1)), _NAN_BOX) if f7 == 0x78 else source
        )
        return None
    if opcode == 0x53 and f7 in (0x50, 0x51, 0x60, 0x61, 0x70, 0x71):
        return None  # writes an integer register only
    fp[rd] = UNKNOWN
    return None


def _call_target(d, registers, pc):
    if d.kind == "call":
        return d.target
    base = registers[d.rs1].exact
    return None if base is None else (base + d.imm) & MASK & ~1


def _register_index(name):
    """ABI or architectural register name -> ("x"|"f", index)."""
    abi = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1"]
    abi += [f"a{i}" for i in range(8)] + [f"s{i}" for i in range(2, 12)] + [f"t{i}" for i in range(3, 7)]
    fabi = [f"ft{i}" for i in range(8)] + ["fs0", "fs1"] + [f"fa{i}" for i in range(8)]
    fabi += [f"fs{i}" for i in range(2, 12)] + [f"ft{i}" for i in range(8, 12)]
    if isinstance(name, int):
        return "x", name
    if name == "fp":
        return "x", 8
    if name in abi:
        return "x", abi.index(name)
    if name in fabi:
        return "f", fabi.index(name)
    head, digits = name[:1], name[1:]
    if head in ("x", "f") and digits.isdigit() and int(digits) < 32:
        return head, int(digits)
    raise ValueError(f"unknown register name {name!r}")


def _seed(value):
    return value if isinstance(value, Bits) else Bits.const(int(value))


def _initial_state(entry_state):
    registers = [ZERO] + [UNKNOWN] * 31
    registers[2] = Bits(base=_FRAME)
    for i in range(10, 18):
        registers[i] = Bits(base=f"arg{i - 10}")
    fp, stack = [UNKNOWN] * 32, {_PRIVATE_FRAME: ZERO}
    for name, value in (entry_state or {}).items():
        if isinstance(name, tuple):
            kind, address, width = name
            if kind != "memory" or width not in (8, 16, 32, 64):
                raise ValueError(f"unsupported entry_state memory key {name!r}")
            stack[(_ABSOLUTE, address & MASK, width)] = _seed(value)
            continue
        kind, index = _register_index(name)
        if kind == "x" and index in (0, 2):
            raise ValueError("entry_state cannot seed zero or the entry stack pointer")
        (registers if kind == "x" else fp)[index] = _seed(value)
    return registers, stack, fp


def _summary_applies(summary, registers, stack):
    """True when the caller's state implies every value the summary assumed."""
    if not summary or summary.get("status") != "proved":
        return False
    for name, value in summary.get("assumptions", ()):
        if isinstance(name, tuple):
            held = stack.get((_ABSOLUTE, name[1] & MASK, name[2]))
        else:
            kind, index = _register_index(name)
            if kind != "x":
                return False
            held = registers[index]
        if held is None or held.exact is None or held.exact != value & MASK:
            return False
    return True


def _apply_summary(summary, registers, stack, fp):
    for r in _CALLER_SAVED:
        registers[r] = UNKNOWN
    for r in _FP_CALLER_SAVED:
        fp[r] = UNKNOWN
    if summary.get("writes_memory", True):
        private = _PRIVATE_FRAME in stack
        stack.clear()
        if private:
            stack[_PRIVATE_FRAME] = ZERO
        for (_, address, width), value in summary.get("memory", ()):
            if value is not None:
                stack[(_ABSOLUTE, address, width)] = Bits.const(value)
    for name, r in (("a0", 10), ("a1", 11)):
        value = summary.get("returns", {}).get(name)
        if value is not None:
            registers[r] = Bits.const(value)


def _step(d, registers, stack, pc, word=None, fp=None, summaries=None):
    fp = [UNKNOWN] * 32 if fp is None else fp
    copied = _fp_effect(word, d, registers, fp)
    a, b = registers[d.rs1], registers[d.rs2]
    # A frame address that loses its symbolic base is no longer tracked.
    computes_from_frame = d.kind not in ("load", "load_unsigned", "store") and _FRAME in (a.base, b.base)
    value = UNKNOWN
    if d.kind in ("lui", "auipc"):
        value = Bits.const(d.imm + (pc if d.kind == "auipc" else 0))
    elif d.kind.endswith("i") and d.kind[:-1] in ("add", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu"):
        if d.bits == 32 and d.kind in ("srli", "srai"):
            a = arithmetic("and", a, Bits.const((1 << 32) - 1)) if d.kind == "srli" else _sext(a, 32)
        value = arithmetic(d.kind[:-1], a, Bits.const(d.imm))
    elif d.kind in ("slti", "sltiu"):
        value = arithmetic("slt" if d.kind == "slti" else "sltu", a, Bits.const(d.imm))
    elif d.kind in ("add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu", "mul"):
        if d.bits == 32 and d.kind in ("sll", "srl", "sra"):
            b = arithmetic("and", b, Bits.const(31))
            if d.kind == "srl":
                a = arithmetic("and", a, Bits.const((1 << 32) - 1))
            elif d.kind == "sra":
                a = _sext(a, 32)
        value = arithmetic(d.kind, a, b)
    elif d.kind in CALLS:
        # The psABI lets a callee clobber caller-saved registers and its
        # stack-argument area, so no slot value survives the call.
        if _frame_handed_out(registers, d.rs1):
            stack.pop(_PRIVATE_FRAME, None)
        summary = (summaries or {}).get(_call_target(d, registers, pc))
        if _summary_applies(summary, registers, stack):
            _apply_summary(summary, registers, stack, fp)
        else:
            for r in _CALLER_SAVED:
                registers[r] = UNKNOWN
            for r in _FP_CALLER_SAVED:
                fp[r] = UNKNOWN
            private = _PRIVATE_FRAME in stack
            stack.clear()
            if private:
                stack[_PRIVATE_FRAME] = ZERO
        value = Bits.const(pc + d.size)
    elif d.kind in ("load", "load_unsigned", "store"):
        address = arithmetic("add", a, Bits.const(d.imm))
        key = (address.base, address.offset, d.bits)
        if d.kind == "store":
            _store(stack, address, d.bits, b)
        else:
            if address.base == _FRAME:
                value = stack.get(key, UNKNOWN)
            elif address.exact is not None:
                value = stack.get((_ABSOLUTE, address.exact, d.bits), UNKNOWN)
            if d.kind == "load_unsigned":
                value = arithmetic("and", value, Bits.const((1 << d.bits) - 1))
            elif d.bits < 64:
                value = _sext(value, d.bits)
    elif d.kind == "opaque":
        registers[:] = [ZERO] + [UNKNOWN] * 31
        stack.clear()
    elif d.kind in ("fp_memory", "unknown", "fp"):
        effect = _coarse_effect(word, d)
        if effect[0] == "none":
            if any(registers[r].base == _FRAME for r in effect[1]):
                stack.pop(_PRIVATE_FRAME, None)
        elif effect[0] == "store":
            _, base, imm, bits = effect
            _store(stack, arithmetic("add", registers[base], Bits.const(imm)), bits, UNKNOWN)
        else:
            if _frame_handed_out(registers, *effect[1]):
                stack.pop(_PRIVATE_FRAME, None)
            _forget_unprivate(stack)
    if copied is not None:
        value = copied
    if d.rd:
        registers[d.rd] = _sext(value, 32) if d.bits == 32 and d.kind not in ("load_unsigned", "store") else value
        if computes_from_frame and registers[d.rd].base != _FRAME:
            stack.pop(_PRIVATE_FRAME, None)
    registers[0] = ZERO


_WIDEN_AFTER = 8


def _analyze(rows, max_steps, entry_state, call_summaries):
    if not rows:
        return {}, dict(status="unknown", reason="empty_machine_region"), {}
    if any(row.relocations for row in rows):
        return {}, dict(status="unknown", reason="unresolved_object_relocations_require_linked_bytes"), {}
    decoded = {r.addr: decode_cpu(r) for r in rows}
    words = {r.addr: r.hexcode for r in rows}
    if len(decoded) != len(rows):
        return {}, dict(status="unknown", reason="repeated_pc_requires_execution_view"), {}
    order = sorted(decoded)
    leaders = {rows[0].addr}
    for pc, d in decoded.items():
        following = pc + d.size
        if d.target in decoded and d.kind in BRANCHES | {"jump"}:
            leaders.add(d.target)
        if d.kind in BRANCHES | CALLS | {"jump", "return", "indirect"} and following in decoded:
            leaders.add(following)
    for left, right in zip(order, order[1:]):
        if left + decoded[left].size != right:
            leaders.add(right)
    blocks, owner = {}, {}
    for pc in order:
        if pc in leaders or not blocks:
            current = pc
            blocks[current] = []
        blocks[current].append(pc)
        owner[pc] = current
    entry = rows[0].addr
    initial, initial_stack, initial_fp = _initial_state(entry_state)
    incoming = {entry: (tuple(initial), initial_stack, tuple(initial_fp))}
    queue, queued, observations = deque([entry]), {entry}, {}
    outgoing, successors_of, call_states, return_states, private_after = {}, {}, {}, {}, {}
    widenings = Counter()
    gaps, steps = set(), 0
    while queue:
        block = queue.popleft()
        queued.discard(block)
        registers, stack, fp = list(incoming[block][0]), dict(incoming[block][1]), list(incoming[block][2])
        for pc in blocks[block]:
            steps += 1
            if steps > max_steps:
                return {}, dict(status="unknown", reason="machine_dataflow_budget_exhausted", steps=steps), {}
            observations[pc] = tuple(registers)
            if decoded[pc].kind in CALLS:
                call_states[pc] = tuple(registers)
            elif decoded[pc].kind == "return":
                return_states[pc] = (tuple(registers), dict(stack))
            _step(decoded[pc], registers, stack, pc, words[pc], fp, call_summaries)
            private_after[pc] = _PRIVATE_FRAME in stack
        last = blocks[block][-1]
        d = decoded[last]
        following = last + d.size
        successors = []
        if d.kind in BRANCHES | {"jump"}:
            if d.target in decoded:
                successors.append(d.target)
            else:
                gaps.add("branch_leaves_selected_span")
        if d.kind not in {"jump", "return", "indirect"} and following in decoded:
            successors.append(following)
        if d.kind == "indirect":
            gaps.add("unresolved_indirect_transfer")
        outgoing[block] = (tuple(registers), dict(stack))
        successors_of[block] = [owner[t] for t in successors]
        for target in successors:
            target = owner[target]
            old = incoming.get(target)
            new = (tuple(registers), dict(stack), tuple(fp))
            if old:
                new = (
                    tuple(a.join(b) for a, b in zip(old[0], registers)),
                    {key: value.join(stack[key]) for key, value in old[1].items() if key in stack},
                    tuple(a.join(b) for a, b in zip(old[2], fp)),
                )
                widenings[target] += 1
                if widenings[target] > _WIDEN_AFTER and old != new:
                    # Widening: a fact still changing after repeated joins
                    # (e.g. a counter losing one carry bit per iteration) is
                    # dropped, so the fixed point is reached in bounded steps.
                    new = (
                        tuple(o if o == n else UNKNOWN for o, n in zip(old[0], new[0])),
                        {key: value for key, value in new[1].items() if old[1].get(key) == value},
                        tuple(o if o == n else UNKNOWN for o, n in zip(old[2], new[2])),
                    )
            if old != new:
                incoming[target] = new
                if target not in queued:
                    queued.add(target)
                    queue.append(target)
    cfg = _Cfg(entry, blocks, successors_of, decoded)
    cfg.words, cfg.private_after, cfg.observations = words, private_after, observations
    loops = _natural_loops(cfg, observations, outgoing)
    _annotate_affine(cfg, loops, observations, outgoing)
    call_sites = []
    for pc in sorted(call_states):
        registers = call_states[pc]
        call_sites.append(
            dict(
                pc=pc,
                target_pc=_call_target(decoded[pc], registers, pc),
                argument_state={f"a{i}": registers[10 + i].exact for i in range(8)},
                argument_records={f"a{i}": registers[10 + i].record() for i in range(8)},
            )
        )
    analysis = dict(
        status="analyzed",
        blocks=len(blocks),
        reachable_blocks=len(incoming),
        reachable_instructions=len(observations),
        steps=steps,
        gaps=sorted(gaps),
        multiplicity="unknown",
        loops=[loop["record"] for loop in loops],
        call_sites=call_sites,
    )
    return observations, analysis, dict(returns=return_states, loops=loops, cfg=cfg)


def register_facts(rows, *, max_steps=2_000_000, entry_state=None, call_summaries=None):
    """Fixed-point known bits at each instruction, including loops and joins.

    Only reachable blocks receive observations. An unresolved control transfer
    is retained as a gap. No register is initialized to an invented zero.
    ``entry_state`` seeds named registers (and ``("memory", address, bits)``
    slots of initialized memory); ``call_summaries`` maps a callee entry pc to
    a ``summarize_function`` result, applied only when its assumptions hold.
    The analysis also lists natural loops (induction variables, trip counts)
    and the argument registers at every reachable call.
    """
    observations, analysis, _ = _analyze(rows, max_steps, entry_state, call_summaries)
    return observations, analysis


class _Cfg:
    def __init__(self, entry, blocks, successors, decoded):
        self.entry, self.blocks, self.succ, self.decoded = entry, blocks, successors, decoded
        self.pred = {b: [] for b in successors}
        for b, targets in successors.items():
            for t in targets:
                if t in self.pred:
                    self.pred[t].append(b)
        order, seen, stack = [], {entry}, [(entry, iter(successors.get(entry, ())))]
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                order.append(node)
                stack.pop()
            elif child not in seen and child in successors:
                seen.add(child)
                stack.append((child, iter(successors.get(child, ()))))
        self.rpo = order[::-1]
        self.index = {b: i for i, b in enumerate(self.rpo)}
        idom = {entry: entry}
        changed = True
        while changed:
            changed = False
            for b in self.rpo[1:]:
                done = [p for p in self.pred[b] if p in idom]
                if not done:
                    continue
                new = done[0]
                for p in done[1:]:
                    new = self._intersect(idom, p, new)
                if idom.get(b) != new:
                    idom[b] = new
                    changed = True
        self.idom = idom

    def _intersect(self, idom, a, b):
        while a != b:
            while self.index[a] > self.index[b]:
                a = idom[a]
            while self.index[b] > self.index[a]:
                b = idom[b]
        return a

    def dominates(self, a, b):
        while True:
            if a == b:
                return True
            if b == self.entry or b not in self.idom:
                return False
            b = self.idom[b]


def _defined_registers(d):
    if d.kind == "opaque":
        return set(range(1, 32))
    if d.kind in CALLS:
        return set(_CALLER_SAVED) | ({d.rd} if d.rd else set())
    return {d.rd} if d.rd else set()


def _reenters_without(cfg, start, body, header):
    """True when ``start`` lies on a cycle inside ``body`` that avoids ``header``."""
    seen, work = set(), [t for t in cfg.succ.get(start, ()) if t in body and t != header]
    while work:
        node = work.pop()
        if node == start:
            return True
        if node in seen:
            continue
        seen.add(node)
        work.extend(t for t in cfg.succ.get(node, ()) if t in body and t != header)
    return False


_CONDITIONS = {
    "beq": lambda x, y: x == y,
    "bne": lambda x, y: x != y,
    "blt": lambda x, y: x < y,
    "bge": lambda x, y: x >= y,
    "bltu": lambda x, y: x < y,
    "bgeu": lambda x, y: x >= y,
}


def _register_name(r):
    return f"x{r}"


def _natural_loops(cfg, observations, outgoing):
    """Natural loops of the reachable CFG with proved induction and trip counts."""
    latches = {}
    for b, targets in cfg.succ.items():
        for h in targets:
            if h in cfg.idom and b in cfg.idom and cfg.dominates(h, b):
                latches.setdefault(h, []).append(b)
    loops = []
    for header, ends in latches.items():
        body, work = {header}, list(ends)
        while work:
            node = work.pop()
            if node not in body:
                body.add(node)
                work.extend(p for p in cfg.pred.get(node, ()) if p in cfg.idom)
        loops.append(dict(header=header, latches=sorted(set(ends)), body=body))
    loops.sort(key=lambda loop: (len(loop["body"]), loop["header"]))
    for loop in loops:
        parents = [o for o in loops if o is not loop and loop["header"] in o["body"] and loop["body"] <= o["body"]]
        loop["parent"] = min(parents, key=lambda o: len(o["body"])) if parents else None
    for loop in loops:
        depth, parent = 1, loop["parent"]
        while parent is not None:
            depth, parent = depth + 1, parent["parent"]
        loop["depth"] = depth
    for loop in sorted(loops, key=lambda o: -o["depth"]):  # inner exits feed outer loops
        _loop_facts(cfg, loop, loops, observations, outgoing)
    loops.sort(key=lambda loop: loop["header"])
    return loops


# Affine forms are (constant, symbol, terms): ``[symbol +] constant +
# sum(step * k_header)`` over ``terms = ((header_pc, step, count), ...)``.
# A symbol is a pointer base of the fixed point (``entry_sp``, ``argN``) or,
# inside one loop iteration, the value a register/frame slot held at the
# header. Frame slots are keyed ``(offset_from_entry_sp, bits)``.


def _form_add(f, g, sign=1):
    if f is None or g is None or isinstance(f, _Table) or isinstance(g, _Table):
        return None
    (fc, fp, ft), (gc, gp, gt) = f, g
    if gp is not None:
        if sign < 0 and gp == fp:
            fp = None
        elif sign > 0 and fp is None:
            fp = gp
        else:
            return None
    terms = {h: (s, n) for h, s, n in ft}
    for h, s, n in gt:
        terms[h] = (terms.get(h, (0, n))[0] + sign * s, n)
    return (fc + sign * gc, fp, tuple(sorted((h, s, n) for h, (s, n) in terms.items() if s)))


def _form_scale(f, c):
    if f is None or isinstance(f, _Table) or (f[1] is not None and c != 1):
        return None
    return (f[0] * c, f[1], tuple((h, s * c, n) for h, s, n in f[2] if s * c))


def _form_fits32(f):
    if f is None or f[1] is not None:
        return None
    low = high = signed(f[0] & MASK, 64)
    for _, s, n in f[2]:
        low, high = low + min(0, s * (n - 1)), high + max(0, s * (n - 1))
    return (signed(f[0] & MASK, 64), None, f[2]) if -(1 << 31) <= low and high < (1 << 31) else None


class _Slots:
    """Frame-slot forms within one iteration; unwritten slots read lazily."""

    def __init__(self, known=None, lazy=True, written=()):
        self.known, self.lazy, self.written = dict(known or {}), lazy, list(written)

    def copy(self):
        return _Slots(self.known, self.lazy, self.written)

    def read(self, key, seed):
        if key in self.known:
            return self.known[key]
        lo, hi = key[0], key[0] + key[1] // 8
        if not self.lazy or hi > 0 or any(lo < h and w < hi for w, h in self.written):
            return None
        return seed(key)

    def write(self, key, form):
        lo, hi = key[0], key[0] + key[1] // 8
        for old in [k for k in self.known if lo < k[0] + k[1] // 8 and k[0] < hi]:
            del self.known[old]
        self.written.append((lo, hi))
        if form is not None:
            self.known[key] = form

    def clear(self, private):
        if private:
            for old in [k for k in self.known if k[0] + k[1] // 8 > 0]:
                del self.known[old]
        else:
            self.known.clear()
            self.lazy = False

    def join(self, other):
        return _Slots(
            {k: f for k, f in self.known.items() if k in other.known and other.known[k] == f},
            self.lazy and other.lazy,
            self.written + [w for w in other.written if w not in self.written],
        )


def _join_states(states):
    regs, slots = dict(states[0][0]), states[0][1].copy()
    for other_regs, other_slots in states[1:]:
        regs = {r: f if other_regs.get(r) == f else None for r, f in regs.items()}
        slots = slots.join(other_slots)
    return regs, slots


def _child_effect(cfg, child, regs, slots):
    defined = {r for b in child["body"] for pc in cfg.blocks[b] for r in _defined_registers(cfg.decoded[pc])}
    regs = {r: (None if r in defined else f) for r, f in regs.items()}
    slots = slots.copy()
    for b in child["body"]:
        for pc in cfg.blocks[b]:
            d = cfg.decoded[pc]
            if d.kind in CALLS or d.kind == "opaque":
                slots.clear(False)
            elif d.kind == "store" or (
                d.kind in ("fp_memory", "unknown", "fp") and _coarse_effect(cfg.words[pc], d)[0] != "none"
            ):
                key = _frame_key(cfg, pc, d)
                if key is not None:
                    slots.write(key, None)
                else:
                    slots.clear(cfg.private_after.get(pc, False))
    return regs, slots


def _frame_key(cfg, pc, d):
    """Frame slot (offset, bits) accessed by a load/store at ``pc``, else None."""
    if d.kind in ("load", "load_unsigned", "store"):
        base, imm, bits = d.rs1, d.imm, d.bits
    else:
        effect = _coarse_effect(cfg.words[pc], d)
        if effect[0] != "store":
            return None
        _, base, imm, bits = effect
    value = cfg.observations[pc][base]
    if value.base != _FRAME:
        return None
    return (value.offset + imm, bits)


def _run_iteration(cfg, loop, loops, seed_regs, seed_slot, narrow=None, visit=None, child_seed=None):
    """Forms of one iteration of ``loop`` from its header to every own block.

    Child loops are summarized by what they may write. ``narrow`` collects the
    symbols that passed through a 32-bit operation; ``visit(pc, regs, slots)``
    sees the in-state of every own instruction.
    """
    header, body = loop["header"], loop["body"]
    children = [o for o in loops if o["parent"] is loop]
    child_of = {b: child for child in children for b in child["body"]}
    block_out, child_in = {}, {}
    for block in cfg.rpo:
        if block not in body or block in child_of:
            continue
        if block == header:
            regs, slots = dict(seed_regs), _Slots()
        else:
            incoming = []
            for p in cfg.pred.get(block, ()):
                child = child_of.get(p)
                if child is not None:
                    if id(child) not in child_in:
                        incoming = None
                        break
                    regs, slots = _child_effect(cfg, child, *child_in[id(child)])
                    exit_value = child.get("exit_value")
                    if exit_value is not None and p == exit_value[0]:
                        regs[exit_value[1]] = (exit_value[2], None, ())
                    incoming.append(_refine_edge(cfg, p, block, regs, slots))
                elif p in block_out and p in body:
                    incoming.append(block_out[p])
                else:
                    incoming = None  # entered other than through the header
                    break
            if not incoming:
                regs, slots = {r: None for r in range(1, 32)}, _Slots(lazy=False)
            else:
                regs, slots = _join_states(incoming)
        for pc in cfg.blocks[block]:
            state = cfg.observations[pc]
            if visit is not None:
                visit(pc, regs, slots)

            def operand(r, state=state, regs=regs):
                if r == 0:
                    return (0, None, ())
                if state[r].exact is not None:
                    return (state[r].exact, None, ())
                form = regs.get(r)
                if form is None and state[r].base is not None:
                    return (state[r].offset, state[r].base, ())
                return form

            d = cfg.decoded[pc]
            result = None
            if d.kind in ("load", "load_unsigned"):
                key = _frame_key(cfg, pc, d)
                if key is not None and d.kind == "load" and d.bits in (32, 64):
                    result = slots.read(key, seed_slot)
                    if result is not None and d.bits == 32:
                        result = _narrow(result, narrow)
            elif d.kind == "store":
                key = _frame_key(cfg, pc, d)
                if key is None:
                    slots.clear(cfg.private_after.get(pc, False))
                else:
                    value = operand(d.rs2) if d.bits in (32, 64) else None
                    slots.write(key, _narrow(value, narrow) if value is not None and d.bits == 32 else value)
            elif d.kind in CALLS or d.kind == "opaque":
                slots.clear(False)
            elif d.kind in ("fp_memory", "unknown", "fp"):
                if _coarse_effect(cfg.words[pc], d)[0] != "none":
                    key = _frame_key(cfg, pc, d)
                    if key is not None:
                        slots.write(key, None)
                    else:
                        slots.clear(cfg.private_after.get(pc, False))
            elif d.rd:
                result = _form_transfer(d, operand, narrow)
            for r in _defined_registers(d):
                regs[r] = None
            if d.rd and result is not None:
                regs[d.rd] = result
        block_out[block] = (regs, slots)
        for child in children:
            if child["header"] in cfg.succ.get(block, ()):
                state = (dict(regs), slots.copy())
                previous = child_in.get(id(child))
                child_in[id(child)] = state if previous is None else _join_states([previous, state])
    if child_seed is not None:
        child_seed.update(child_in)
    return block_out


def _refine_edge(cfg, p, block, regs, slots):
    """Equality facts of a ``beq``/``bne`` edge from ``p`` into ``block``.

    On the equal edge the register equals the constant. On the unequal edge a
    register with exactly one unknown bit, whose known bits match the
    constant, holds the constant with that bit flipped.
    """
    pc = cfg.blocks[p][-1]
    d = cfg.decoded[pc]
    if d.kind not in ("beq", "bne") or d.target == pc + d.size:
        return regs, slots
    equal = (block == d.target) == (d.kind == "beq")
    state = cfg.observations[pc]
    for r, other in ((d.rs1, d.rs2), (d.rs2, d.rs1)):
        constant = 0 if other == 0 else state[other].exact
        if r == 0 or constant is None or state[r].exact is not None:
            continue
        unknown = ~state[r].mask & MASK
        value = None
        if equal:
            value = constant
        elif unknown & (unknown - 1) == 0 and constant & state[r].mask == state[r].value:
            value = constant ^ unknown
        if value is not None:
            regs = dict(regs)
            regs[r] = (value, None, ())
    return regs, slots


def _narrow(form, narrow):
    """A 32-bit result: exact range check for terms, recorded for symbols."""
    if form is None:
        return None
    if isinstance(form, _Table):
        return _Table(form.dims, tuple(_sext(Bits.const(v), 32).exact for v in form.values))
    if form[1] is None:
        return _form_fits32(form)
    if narrow is None or form[2]:
        return None
    narrow.add(form[1])
    return form


_TABLE_LIMIT = 4096
_POINTWISE = frozenset(
    ("addi", "andi", "ori", "xori", "slli", "srli", "srai", "slti", "sltiu")
    + ("add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu", "mul")
)


@dataclass(frozen=True)
class _Table:
    """A register value at every point of its loops' joint iteration space.

    ``dims`` is ((header_pc, count), ...); ``values`` enumerates the points in
    row-major order (last header fastest), each reduced modulo 2**64.
    """

    dims: tuple
    values: tuple


def _form_dims(form):
    if isinstance(form, _Table):
        return form.dims
    return tuple((h, n) for h, _, n in form[2])


def _form_at(form, point):
    if isinstance(form, _Table):
        index = 0
        for h, n in form.dims:
            index = index * n + point[h]
        return form.values[index]
    return (form[0] + sum(s * point[h] for h, s, _ in form[2])) & MASK


def _pointwise(d, a, b):
    """Apply the CPU's own transfer at every joint iteration point."""
    if d.kind not in _POINTWISE or a is None or b is None:
        return None
    if any(not isinstance(f, _Table) and f[1] is not None for f in (a, b)):
        return None
    dims = dict(_form_dims(a))
    for h, n in _form_dims(b):
        if dims.setdefault(h, n) != n:
            return None
    dims = tuple(sorted(dims.items()))
    total = 1
    for _, n in dims:
        total *= n
    if not dims or total > _TABLE_LIMIT:
        return None
    values = []
    registers = [ZERO] + [UNKNOWN] * 31
    scratch = 31 if d.rd != 31 else 30
    for point in product(*(range(n) for _, n in dims)):
        at = dict(zip((h for h, _ in dims), point))
        registers[d.rs1] = Bits.const(_form_at(a, at)) if d.rs1 else ZERO
        if d.rs2 != d.rs1:
            registers[d.rs2] = Bits.const(_form_at(b, at)) if d.rs2 else ZERO
        registers[scratch] = UNKNOWN
        _step(d, registers, {}, 0)
        value = registers[d.rd].exact
        if value is None:
            return None
        values.append(value)
    if len(set(values)) == 1:
        return (values[0], None, ())
    return _Table(dims, tuple(values))


def _form_transfer(d, operand, narrow=None):
    a, b = operand(d.rs1), operand(d.rs2)
    if isinstance(a, _Table) or isinstance(b, _Table):
        return None if narrow is not None else _pointwise(d, a, b)
    if d.kind == "addi":
        result = _form_add(a, (d.imm, None, ()))
    elif d.kind in ("add", "sub"):
        result = _form_add(a, b, 1 if d.kind == "add" else -1)
    elif d.kind == "slli":
        result = _form_scale(a, 1 << (d.imm & 63))
    elif d.kind in ("sll", "mul") and b is not None and not b[2] and b[1] is None:
        result = _form_scale(a, (1 << (b[0] & 63)) if d.kind == "sll" else signed(b[0] & MASK, 64))
    elif d.kind == "mul" and a is not None and not a[2] and a[1] is None:
        result = _form_scale(b, signed(a[0] & MASK, 64))
    else:
        result = None
    if result is not None and result[1] is None and not result[2]:
        result = (result[0] & MASK, None, ())
    if result is not None:
        result = _narrow(result, narrow) if d.bits == 32 else result
    if result is None and narrow is None and (a is not None and a[2] or b is not None and b[2]):
        return _pointwise(d, a, b)
    return result


def _variable_name(var):
    return _register_name(var[1]) if var[0] == "x" else f"frame[{var[1][0]}]:{var[1][1]}"


def _symbol(var):
    return f"header:x{var[1]}" if var[0] == "x" else f"header:frame:{var[1][0]}:{var[1][1]}"


def _loop_facts(cfg, loop, loops, observations, outgoing):
    header, body = loop["header"], loop["body"]
    pcs = [pc for b in sorted(body) for pc in cfg.blocks[b]]
    defs = {}
    for pc in pcs:
        for r in _defined_registers(cfg.decoded[pc]):
            defs.setdefault(r, []).append(pc)
    loop["defs"] = defs
    preheaders = [p for p in cfg.pred.get(header, ()) if p not in body]
    narrow, at_exit = set(), {}
    exiting = sorted(b for b in body if any(t not in body for t in cfg.succ.get(b, ())))
    exit_pc = cfg.blocks[exiting[0]][-1] if len(exiting) == 1 else None

    def visit(pc, regs, slots):
        if pc == exit_pc:
            at_exit["state"] = (dict(regs), slots.copy())

    seed = {r: (0, _symbol(("x", r)), ()) for r in range(1, 32)}
    out = _run_iteration(cfg, loop, loops, seed, lambda key: (0, _symbol(("s", key)), ()), narrow, visit)
    latch_states = [out.get(b) for b in loop["latches"]]

    def transfer(var):
        """('step', s) when every latch leaves var = header value + s; ('const', c)."""
        kinds = set()
        for state in latch_states:
            if state is None:
                return None
            regs, slots = state
            form = regs.get(var[1]) if var[0] == "x" else slots.read(var[1], lambda key: (0, _symbol(("s", key)), ()))
            if form is None or form[2]:
                return None
            if form[1] == _symbol(var):
                kinds.add(("step", form[0]))
            elif form[1] is None:
                kinds.add(("const", form[0] & MASK))
            else:
                return None
        return kinds.pop() if len(kinds) == 1 else None

    def initial(var):
        joined = None
        for p in preheaders:
            registers, stack = outgoing[p]
            value = registers[var[1]] if var[0] == "x" else stack.get((_FRAME, var[1][0], var[1][1]), UNKNOWN)
            joined = value if joined is None else joined.join(value)
        return None if joined is None else joined.exact

    induction, records = {}, []
    variables = [("x", r) for r in sorted(defs) if r]
    for state in latch_states:
        if state is not None:
            variables += [("s", key) for key in state[1].known if ("s", key) not in variables]
    for var in variables:
        kind = transfer(var)
        if kind is None or kind[0] != "step" or kind[1] == 0:
            continue
        init = initial(var)
        item = dict(reg=_variable_name(var), init=init, step=signed(kind[1] & MASK, 64), status="proved")
        if init is None:
            item.update(status="unknown", reason="initial_value_not_constant")
        else:
            induction[var] = dict(init=init, step=item["step"], narrow=_symbol(var) in narrow)
        records.append(item)
    loop["induction"] = induction
    unsupported = None
    if any(cfg.decoded[pc].kind in ("return", "indirect", "opaque") for pc in pcs):
        unsupported = "loop_contains_return_indirect_or_opaque_transfer"
    loop["trip"] = _trip_count(cfg, loop, exiting, at_exit.get("state"), transfer, initial, narrow, unsupported)
    exit_value = loop["trip"].pop("_exit_value", None)
    if exit_value is not None:
        loop["exit_value"] = (exiting[0], *exit_value)
    parent = loop["parent"]
    loop["record"] = dict(
        header_pc=header,
        latch_pcs=[cfg.blocks[b][-1] for b in loop["latches"]],
        blocks=sorted(body),
        induction=records,
        trip_count=loop["trip"],
        nest_depth=loop["depth"],
        parent_header_pc=parent["header"] if parent else None,
    )


def _trip_count(cfg, loop, exiting, exit_state, transfer, initial, narrow, unsupported):
    def unknown(reason, **extra):
        return dict(status="unknown", reason=reason, **extra)

    if unsupported:
        return unknown(unsupported)
    header, body = loop["header"], loop["body"]
    for b in body:
        d = cfg.decoded[cfg.blocks[b][-1]]
        if d.kind in BRANCHES | {"jump"} and d.target not in cfg.decoded:
            return unknown("exit_leaves_selected_span")
    if len(exiting) != 1:
        return unknown("loop_has_multiple_or_no_exits", exits=len(exiting))
    exit_block = exiting[0]
    branch_pc = cfg.blocks[exit_block][-1]
    d = cfg.decoded[branch_pc]
    if d.kind not in BRANCHES:
        return unknown("exit_is_not_a_conditional_branch")
    if not all(cfg.dominates(exit_block, latch) for latch in loop["latches"]):
        return unknown("exit_not_on_every_iteration")
    if _reenters_without(cfg, exit_block, body, header):
        return unknown("exit_on_cycle_without_header")
    if exit_state is None:
        return unknown("exit_state_unavailable")
    target_in, fall_in = d.target in body, branch_pc + d.size in body
    if target_in == fall_in:
        return unknown("exit_branch_direction_unresolved")
    state = cfg.observations[branch_pc]
    regs = exit_state[0]

    def operand(r):
        if r == 0:
            return (0, None, ())
        if state[r].exact is not None:
            return (state[r].exact, None, ())
        return regs.get(r)

    forms = [operand(d.rs1), operand(d.rs2)]
    candidates = []
    for side, form in enumerate(forms):
        if form is None or form[2] or form[1] is None or not form[1].startswith("header:"):
            continue
        parts = form[1].split(":")
        var = ("x", int(parts[1][1:])) if parts[1] != "frame" else ("s", (int(parts[2]), int(parts[3])))
        kind = transfer(var)
        if kind is not None and not (kind[0] == "step" and kind[1] == 0):
            candidates.append((side, var, kind, form[0]))
    if len(candidates) != 1:
        return unknown("exit_compare_has_no_single_induction_variable")
    side, var, kind, offset = candidates[0]
    bound = forms[1 - side]
    if bound is None or bound[2]:
        return unknown("exit_bound_varies_in_loop")
    if bound[1] is not None:
        return unknown("exit_bound_invariant_but_not_constant", bound_symbol=bound[1])
    init = initial(var)
    if init is None:
        return unknown("induction_initial_value_not_constant", induction=_variable_name(var))
    unsigned = d.kind in ("bltu", "bgeu")
    lo, hi = (0, MASK) if unsigned else (-(1 << 63), (1 << 63) - 1)
    if _symbol(var) in narrow:
        lo, hi = max(lo, -(1 << 31)), min(hi, (1 << 31) - 1)

    def domain(value):
        return value & MASK if unsigned else signed(value & MASK, 64)

    limit = domain(bound[0])
    test = _CONDITIONS[d.kind]

    def exits_at(value):
        holds = test(value, limit) if side == 0 else test(limit, value)
        return holds != target_in

    if kind[0] == "const":
        # The variable is init on the first iteration and a constant after.
        values = [domain(init + offset), domain(kind[1] + offset)]
        if any(not lo <= v <= hi for v in values):
            return unknown("induction_out_of_comparison_range")
        if exits_at(values[0]):
            k = 0
        elif exits_at(values[1]):
            k = 1
        else:
            return unknown("loop_does_not_exit_through_compared_variable")
        return _proved_trip(k, var, branch_pc, bound[0], (d.rs1 if side == 0 else d.rs2, values[k]))
    step = signed(kind[1] & MASK, 64)
    start = (init & MASK if unsigned else signed(init & MASK, 64)) + signed(offset & MASK, 64)
    if not lo <= start <= hi:
        return unknown("induction_out_of_comparison_range")
    k_max = (hi - start) // step if step > 0 else (start - lo) // -step

    def exits(k):
        return exits_at(start + k * step)

    if exits(0):
        k = 0
    elif d.kind in ("beq", "bne"):
        if k_max >= 1 and exits(1):
            k = 1
        elif (limit - start) % step == 0 and 0 < (limit - start) // step <= k_max:
            k = (limit - start) // step
            if not exits(k):
                return unknown("exit_value_not_reached")
        else:
            return unknown("exit_value_not_reached_without_wraparound")
    else:
        if not exits(k_max):
            return unknown("induction_wraps_before_exit")
        low, high = 0, k_max
        while high - low > 1:
            middle = (low + high) // 2
            if exits(middle):
                high = middle
            else:
                low = middle
        k = high
    return _proved_trip(k, var, branch_pc, bound[0], (d.rs1 if side == 0 else d.rs2, start + k * step))


def _proved_trip(k, var, branch_pc, bound, exit_value):
    register, value = exit_value
    return dict(
        _exit_value=(register, value & MASK),
        status="proved",
        value=k + 1,
        backedges_taken=k,
        counts="header_executions_per_loop_entry",
        induction=_variable_name(var),
        exit_pc=branch_pc,
        bound=bound & MASK,
    )


def _annotate_affine(cfg, loops, observations, outgoing):
    """Per-site ``base + sum(step * k)`` forms for registers inside proved loops.

    Runs after the fixed point and never changes it: a form is attached only
    where the fixed point lost the value, every term's loop has a proved trip
    count, and every intra-iteration path to the site yields the same form.
    """
    entries = {}  # id(loop) -> (regs, slots) at entry, from the proved parent
    for loop in sorted(loops, key=lambda o: o["depth"]):
        if loop["trip"]["status"] != "proved":
            continue
        header, body = loop["header"], loop["body"]
        count = loop["trip"]["value"]
        preheaders = [p for p in cfg.pred.get(header, ()) if p not in body]
        outer = entries.get(id(loop))
        written = {key for b in body for pc in cfg.blocks[b] if (key := _store_key(cfg, pc)) is not None}
        clears = any(_store_key(cfg, pc) is None and _writes_memory(cfg, pc) for b in body for pc in cfg.blocks[b])

        def outside(var, outer=outer):
            if outer is not None:
                form = outer[0].get(var[1]) if var[0] == "x" else outer[1].read(var[1], outer[2])
                if form is not None:
                    return form
            joined = None
            for p in preheaders:
                registers, stack = outgoing[p]
                value = registers[var[1]] if var[0] == "x" else stack.get((_FRAME, var[1][0], var[1][1]), UNKNOWN)
                joined = value if joined is None else joined.join(value)
            if joined is None:
                return None
            if joined.exact is not None:
                return (joined.exact, None, ())
            return (joined.offset, joined.base, ()) if joined.base else None

        def seeded(var):
            iv = loop["induction"].get(var)
            if iv is not None:
                start = outside(var)
                if start is None or isinstance(start, _Table) or start[1] is not None:
                    start = (iv["init"], None, ())
                return _form_add(start, (0, None, ((header, iv["step"], count),)))
            if var[0] == "x":
                return None if var[1] in loop["defs"] else outside(var)
            lo, hi = var[1][0], var[1][0] + var[1][1] // 8
            if clears or any(lo < k[0] + k[1] // 8 and k[0] < hi for k in written):
                return None
            return outside(var)

        def visit(pc, regs, slots):
            state = observations[pc]
            annotated = list(state)
            for r, form in regs.items():
                if form is not None and (isinstance(form, _Table) or form[2]) and state[r].exact is None:
                    annotated[r] = replace(state[r], affine=form)
            if annotated != list(state):
                observations[pc] = tuple(annotated)

        seed = {r: seeded(("x", r)) for r in range(1, 32)}
        child_in = {}

        def seed_slot(key, seeded=seeded):
            return seeded(("s", key))

        _run_iteration(cfg, loop, loops, seed, seed_slot, None, visit, child_seed=child_in)
        for child in loops:
            if child["parent"] is loop and id(child) in child_in:
                entries[id(child)] = (*child_in[id(child)], seed_slot)


def _store_key(cfg, pc):
    d = cfg.decoded[pc]
    if not _writes_memory(cfg, pc) or d.kind in CALLS or d.kind == "opaque":
        return None
    return _frame_key(cfg, pc, d)


def _writes_memory(cfg, pc):
    d = cfg.decoded[pc]
    return (
        d.kind in CALLS
        or d.kind in ("store", "opaque")
        or (d.kind in ("fp_memory", "unknown", "fp") and _coarse_effect(cfg.words[pc], d)[0] != "none")
    )


def summarize_function(rows, *, entry_state=None, call_summaries=None, max_steps=2_000_000):
    """Return-register values of one function body under its entry state.

    A return is a known value only when every reachable return proves it from
    the entry state (e.g. allocator state the caller read from initialized
    ``.data`` and supplied as a ``("memory", address, bits)`` entry); otherwise
    it is None. ``assumptions`` lists the entry values the summary relies on:
    a call site applies the summary only when its own state proves them.
    """
    unknown = dict(returns=dict(a0=None, a1=None), status="unknown")
    assumptions = []
    for name, value in (entry_state or {}).items():
        seeded = _seed(value)
        if seeded.exact is None:
            return dict(unknown, reason="summary_assumptions_must_be_exact_values")
        assumptions.append((name, seeded.exact))
    _, analysis, extra = _analyze(rows, max_steps, entry_state, call_summaries)
    major = sorted({int(r.hexcode, 16) & 127 for r in rows if r.hexcode and len(r.hexcode) == 8})
    base = dict(assumptions=assumptions, major_opcodes=major)
    if analysis.get("status") != "analyzed":
        return dict(unknown, reason=analysis.get("reason"), **base)
    if analysis["gaps"]:
        return dict(unknown, reason="callee_control_gaps:" + ",".join(analysis["gaps"]), **base)
    if not extra["returns"]:
        return dict(unknown, reason="no_reachable_return", **base)
    joined_registers = joined_stack = None
    for registers, stack in extra["returns"].values():
        if joined_registers is None:
            joined_registers, joined_stack = list(registers), dict(stack)
        else:
            joined_registers = [a.join(b) for a, b in zip(joined_registers, registers)]
            joined_stack = {k: v.join(stack[k]) for k, v in joined_stack.items() if k in stack}
    writes = False
    for row in rows:
        d = decode_cpu(row)
        if d.kind in CALLS | {"store", "opaque"} or (
            d.kind in ("fp_memory", "unknown", "fp") and _coarse_effect(row.hexcode, d)[0] != "none"
        ):
            writes = True
    memory = [
        (key, value.exact) for key, value in joined_stack.items() if key is not _PRIVATE_FRAME and key[0] == _ABSOLUTE
    ]
    calls_finite = all(
        (call_summaries or {}).get(site["target_pc"], {}).get("terminates") == "proved"
        for site in analysis["call_sites"]
    )
    terminates = "unknown"
    if calls_finite and not analysis["loops"]:
        terminates = "proved"  # acyclic body whose calls all terminate
    elif calls_finite and all(loop["trip_count"]["status"] == "proved" for loop in analysis["loops"]):
        counted = finite_control_counts(
            rows, custom_opcode=None, max_steps=max_steps, entry_state=entry_state, call_summaries=call_summaries
        )
        terminates = "proved" if counted["status"] == "proved" else "unknown"
    return dict(
        returns={name: joined_registers[r].exact for name, r in (("a0", 10), ("a1", 11))},
        status="proved",
        reason=None,
        writes_memory=writes,
        memory=sorted(memory, key=lambda item: (item[0][1], item[0][2])),
        terminates=terminates,
        **base,
    )


def finite_control_counts(
    rows,
    *,
    custom_opcode,
    max_steps=2_000_000,
    max_commands=1_000_000,
    entry_state=None,
    call_summaries=None,
    record_commands=False,
    max_recorded=200_000,
    assume_returning_calls=False,
):
    """Count a complete original body only when every scalar branch is known.

    Symbolic argument pointers preserve identity and offsets. Tensor loads and
    accelerator results stay unknown. Calls, data-dependent control, unsupported
    effects and exhausted budgets refuse the *entire* observation. A call into
    a summarized callee that provably terminates and issues no selected-opcode
    word is stepped over with its summary instead of refusing.

    ``record_commands`` adds, for a proved body only, ``command_stream``: every
    selected-opcode issue in execution order as ``{seq, pc, funct, rs1, rs2}``
    (an ``int`` when the register is exact at issue, else its compact record:
    pointer base and offset, or known bits) and every ``fence`` as
    ``{seq, pc, kind: "fence"}``. Each entry carries ``host_instructions_since_prev``:
    scalar instructions executed since the previous selected-opcode issue.

    The record is kept as periodic affine runs (``command_stream_runs``; see
    :func:`expand_command_stream`), at most ``max_recorded`` runs, so long kernels
    stay bounded. ``command_stream`` expands at most ``max_recorded`` entries;
    ``command_stream_truncated`` is true when it is shorter than the issue stream and
    ``command_stream_runs_complete`` says whether the runs cover all of it. An
    unproved body gains ``command_stream_status`` with the refusal reason; an
    exhausted step budget is ``step_budget_exhausted`` with ``steps_reached``.

    ``assume_returning_calls`` also steps over a call whose callee summary shows it
    issues no selected-opcode word although its termination is not proved: the
    callee is assumed to return, its caller-saved registers and memory become
    unknown, and the proved result lists it under ``assumed_returning_calls``.
    """
    recorder = _StreamRecorder(max_recorded) if record_commands else None
    result = _finite_control_counts(
        rows, custom_opcode, max_steps, max_commands, entry_state, call_summaries, recorder, assume_returning_calls
    )
    if result["status"] == "proved":
        result["dynamic_command_count"] = dict(
            status="proved", value=result["commands"], basis="bounded_scalar_execution_of_proved_control"
        )
    else:
        result["dynamic_command_count"] = dict(status="unknown", reason=result["reason"])
    if recorder is not None and result["status"] == "proved":
        result["command_stream"] = list(expand_command_stream(recorder.runs, limit=max_recorded))
        result["command_stream_runs"] = recorder.runs
        result["command_stream_entries"] = recorder.entries
        result["command_stream_truncated"] = len(result["command_stream"]) < recorder.entries
        result["command_stream_runs_complete"] = not recorder.overflow
    elif recorder is not None:
        budget = result["reason"] == "finite_instruction_budget_exhausted"
        result["command_stream_status"] = dict(
            status="unknown",
            reason="step_budget_exhausted" if budget else result["reason"],
            **(dict(steps_reached=max_steps) if budget else {}),
        )
    return result


_MAX_PERIOD = 8


ET_REL = 1  # ELF e_type of a relocatable object (System V gABI)


def linked_input_status(path, *, linked_names=("package_kernel.elf",)):
    """``None`` for a linked ELF; for a relocatable object an explicit refusal.

    Scalar stepping needs resolved control transfers, so an ``ET_REL`` input is
    reported as ``unlinked_object_use_linked_elf``; a linked file of one of
    ``linked_names`` beside it is named in the record, never substituted.
    """
    from pathlib import Path

    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(20)
    if header[:4] != b"\x7fELF" or len(header) < 18:
        return None
    e_type = int.from_bytes(header[16:18], "little" if header[5] == 1 else "big")
    if e_type != ET_REL:
        return None
    beside = [str(path.parent / name) for name in linked_names if (path.parent / name).is_file()]
    return dict(status="unknown", reason="unlinked_object_use_linked_elf", input=str(path),
                **(dict(linked_elf_beside=beside) if beside else {}))


def _run_step(first, value):
    """The constant step from ``first`` to ``value`` (an int, or a pointer's byte offset), or ``None``."""
    if type(first) is int and type(value) is int:
        return (value - first) & MASK
    if isinstance(first, dict) and isinstance(value, dict) and first.get("pointer_base"):
        same = {k: v for k, v in first.items() if k != "byte_offset"} == {
            k: v for k, v in value.items() if k != "byte_offset"
        }
        return value["byte_offset"] - first["byte_offset"] if same else None
    return 0 if first == value else None


def _advance(value, step, k):
    if not step:
        return value
    if type(value) is int:
        return (value + step * k) & MASK
    return dict(value, byte_offset=value["byte_offset"] + step * k)


def _slot(entry):
    return dict(entry, rs1_step=0, rs2_step=0, next_host=entry["host"])


def _identity(entry):
    return entry["pc"], entry.get("kind"), entry.get("funct")


class _StreamRecorder:
    """Issue-order record as periodic affine runs.

    A run is a period of up to ``_MAX_PERIOD`` issue slots repeated ``count`` times
    (plus ``partial`` leading slots of one more period); each slot's operands advance
    by a constant step per period, and every repeat has the same host gaps. At most
    ``max_runs`` runs are kept; beyond that ``overflow`` is set and recording stops.
    """

    def __init__(self, max_runs):
        self.runs, self.max_runs, self.entries, self.overflow = [], max_runs, 0, False

    def add(self, pc, host, funct=None, rs1=None, rs2=None, kind=None):
        if self.overflow:
            self.entries += 1
            return
        entry = dict(pc=pc, host=host)
        entry.update(dict(kind=kind) if kind is not None else dict(funct=funct, rs1=rs1, rs2=rs2))
        seq, self.entries = self.entries, self.entries + 1
        if self._extend(entry):
            return
        self._split_partial()
        if self._extend(entry):
            return
        self._push(dict(seq=seq, slots=[_slot(entry)], count=1, partial=0))
        self._fold()

    def _expected(self, run):
        slot = run["slots"][run["partial"]]
        period = run["count"]
        return slot, period

    def _extend(self, entry):
        if not self.runs:
            return False
        run = self.runs[-1]
        if run["count"] < 2:
            return False
        slot, period = self._expected(run)
        if _identity(slot) != _identity(entry) or slot["next_host"] != entry["host"]:
            return False
        if "kind" not in slot and (
            _advance(slot["rs1"], slot["rs1_step"], period) != entry["rs1"]
            or _advance(slot["rs2"], slot["rs2_step"], period) != entry["rs2"]
        ):
            return False
        run["partial"] += 1
        if run["partial"] == len(run["slots"]):
            run["count"], run["partial"] = run["count"] + 1, 0
        return True

    def _split_partial(self):
        """A period broken midway leaves its issued slots as single-issue runs."""
        if not self.runs or not self.runs[-1]["partial"]:
            return
        run = self.runs[-1]
        width, period = len(run["slots"]), run["count"]
        seq = run["seq"] + width * period
        tail = []
        for index, slot in enumerate(run["slots"][: run["partial"]]):
            entry = dict(pc=slot["pc"], host=slot["next_host"])
            if "kind" in slot:
                entry["kind"] = slot["kind"]
            else:
                entry.update(
                    funct=slot["funct"],
                    rs1=_advance(slot["rs1"], slot["rs1_step"], period),
                    rs2=_advance(slot["rs2"], slot["rs2_step"], period),
                )
            tail.append(dict(seq=seq + index, slots=[_slot(entry)], count=1, partial=0))
        run["partial"] = 0
        for item in tail:
            self._push(item)

    def _push(self, run):
        if len(self.runs) >= self.max_runs:
            self.overflow = True
            return
        self.runs.append(run)

    def _fold(self):
        """Merge the last ``2w`` single-issue runs into one run of period ``w`` when they repeat."""
        runs = self.runs
        for width in range(1, _MAX_PERIOD + 1):
            if len(runs) < 2 * width:
                return
            tail = runs[-2 * width :]
            if any(r["count"] != 1 or len(r["slots"]) != 1 or r["partial"] for r in tail):
                return
            first, second = [r["slots"][0] for r in tail[:width]], [r["slots"][0] for r in tail[width:]]
            if any(_identity(a) != _identity(b) for a, b in zip(first, second, strict=True)):
                continue
            if any(a["host"] != b["host"] for a, b in zip(first[1:], second[1:], strict=True)):
                continue
            steps = []
            for a, b in zip(first, second, strict=True):
                pair = (0, 0) if "kind" in a else (_run_step(a["rs1"], b["rs1"]), _run_step(a["rs2"], b["rs2"]))
                if None in pair:
                    break
                steps.append(pair)
            if len(steps) != width:
                continue
            slots = [
                dict(a, rs1_step=s1, rs2_step=s2, next_host=b["host"])
                for a, b, (s1, s2) in zip(first, second, steps, strict=True)
            ]
            del runs[-2 * width :]
            runs.append(dict(seq=tail[0]["seq"], slots=slots, count=2, partial=0))
            return


def expand_command_stream(runs, *, limit=None):
    """Entries of a ``command_stream_runs`` record in issue order (at most ``limit``)."""
    produced = 0
    for run in runs:
        width = len(run["slots"])
        for k in range(run["count"] * width + run["partial"]):
            if limit is not None and produced >= limit:
                return
            period, slot = divmod(k, width)
            slot = run["slots"][slot]
            entry = dict(seq=run["seq"] + k, pc=slot["pc"])
            if "kind" in slot:
                entry["kind"] = slot["kind"]
            else:
                entry.update(
                    funct=slot["funct"],
                    rs1=_advance(slot["rs1"], slot["rs1_step"], period),
                    rs2=_advance(slot["rs2"], slot["rs2_step"], period),
                )
            entry["host_instructions_since_prev"] = slot["host"] if period == 0 else slot["next_host"]
            yield entry
            produced += 1


def _issue_operand(bits):
    """An issue-time register: its exact integer, else its bounded symbolic record."""
    if bits.exact is not None:
        return bits.exact
    record = dict(kind="unknown", raw=None)
    if bits.base:
        record.update(pointer_base=bits.base, byte_offset=bits.offset)
    elif bits.mask:
        record["known_bits"] = dict(mask=bits.mask, value=bits.value, width=64)
    return record


def _finite_control_counts(
    rows, custom_opcode, max_steps, max_commands, entry_state, call_summaries, recorder=None, assume_calls=False
):
    if not rows:
        return dict(status="unknown", reason="empty_machine_region")
    if any(row.relocations for row in rows):
        return dict(status="unknown", reason="unresolved_object_relocations_require_linked_bytes")
    by_pc = {row.addr: row for row in rows}
    if len(by_pc) != len(rows):
        return dict(status="unknown", reason="repeated_pc_requires_execution_view")
    registers, stack, fp = _initial_state(entry_state)
    pc, executions, command_counts = rows[0].addr, Counter(), Counter()
    commands = fences = host = 0
    summarized, assumed = Counter(), Counter()
    for step in range(max_steps):
        row = by_pc.get(pc)
        if row is None:
            summary = (call_summaries or {}).get(pc) if assume_calls else None
            if summary is not None and custom_opcode not in summary.get("major_opcodes", (custom_opcode,)):
                # A tail transfer to a callee that issues no selected-opcode word ends the body.
                assumed[pc] += 1
                d = CPUInstruction("return")
                row = None
            else:
                return dict(status="unknown", reason="control_leaves_selected_span", pc=pc)
        else:
            d = decode_cpu(row)
        custom = row is not None and len(row.hexcode) == 8 and int(row.hexcode, 16) & 127 == custom_opcode
        stepped = False
        if d.kind in CALLS:
            target = _call_target(d, registers, pc)
            summary = (call_summaries or {}).get(target)
            if (
                _summary_applies(summary, registers, stack)
                and summary.get("terminates") == "proved"
                and custom_opcode not in summary.get("major_opcodes", (custom_opcode,))
            ):
                stepped = True
                summarized[target] += 1
            elif (
                assume_calls
                and summary is not None
                and custom_opcode not in summary.get("major_opcodes", (custom_opcode,))
            ):
                stepped = True
                assumed[target] += 1
        if (
            (d.kind in CALLS and not stepped)
            or d.kind in ("indirect", "opaque")
            or (d.kind == "unknown" and not custom)
        ):
            return dict(status="unknown", reason="opaque_control_requires_capture", pc=pc)
        executions[pc] += 1
        if custom:
            commands += 1
            command_counts[pc] += 1
            if commands > max_commands:
                return dict(status="unknown", reason="finite_command_budget_exhausted")
            if recorder is not None:
                from merlin.compile.analysis.rocc_commands import fields_of

                word = fields_of(int(row.hexcode, 16))
                recorder.add(
                    pc,
                    host,
                    funct=word["funct"],
                    rs1=_issue_operand(registers[word["rs1"]]),
                    rs2=_issue_operand(registers[word["rs2"]]),
                )
            host = 0
        else:
            if d.kind == "fence" and recorder is not None:
                recorder.add(pc, host, kind="fence")
            host += 1
        if d.kind == "fence":
            fences += 1
        if d.kind == "return":
            return dict(
                status="proved",
                count_basis="proved_finite_scalar_control",
                scope="selected_function_body",
                instruction_executions=dict(executions),
                command_executions=dict(command_counts),
                commands=commands,
                fences=fences,
                steps=step + 1,
                hardware_execution="unknown",
                **(dict(summarized_calls=dict(summarized)) if summarized else {}),
                **(dict(assumed_returning_calls=dict(assumed)) if assumed else {}),
            )
        following = pc + d.size
        if recorder is not None and d.kind in (*BRANCHES, "jump") and d.target == pc:
            # A zero-displacement transfer is how an unlinked object's unresolved branch reads.
            return dict(status="unknown", reason="unlinked_object_use_linked_elf", pc=pc)
        if d.kind in BRANCHES:
            a, b = registers[d.rs1], registers[d.rs2]
            av, bv = a.exact, b.exact
            if av is None or bv is None:
                # Equality of offsets is invariant under pointer wraparound.
                # Ordered comparisons need a proven address range instead.
                if d.kind in ("beq", "bne") and a.base is not None and a.base == b.base:
                    av, bv = a.offset, b.offset
                else:
                    return dict(status="unknown", reason="data_dependent_or_unbound_control", pc=pc)
            if d.kind in ("blt", "bge"):
                av, bv = signed(av, 64), signed(bv, 64)
            condition = {
                "beq": av == bv,
                "bne": av != bv,
                "blt": av < bv,
                "bge": av >= bv,
                "bltu": av < bv,
                "bgeu": av >= bv,
            }[d.kind]
            following = d.target if condition else following
        elif d.kind == "jump":
            following = d.target
        _step(d, registers, stack, pc, row.hexcode, fp, call_summaries)
        pc = following
    return dict(status="unknown", reason="finite_instruction_budget_exhausted")
