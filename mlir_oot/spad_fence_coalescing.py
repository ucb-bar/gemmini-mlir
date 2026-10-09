"""Restricted same-tile SPAD execute-chain fence coalescing.

This target proof covers ordered full DIM products only. It does not grant
external-memory coherence, accumulator execute reads, partial aliases, dynamic
addresses, transposition or a different controller implementation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from xdsl.rewriter import Rewriter

from .ir import gemmini_dialect as G
from .tables import isa
from .tables import rtl_facts as F

_SOURCE_HASHES = {
    "Configs.scala": "178c21ac741c89c08158efa71fb34da35d27b75bd50bde5116a5e19178f8391b",
    "GemminiConfigs.scala": "7294a082a1df4ae7aa7d81adc5bde75b7d5118683c083f6a4b998c9ce64d102e",
    "ExecuteController.scala": "59b068a79d6063d283561813f7fb65c3503e94d5923334e553ab5d42e2692e8c",
    "ReservationStation.scala": "d2d71c91d4ff1e26691bbcd7cd809aad16652680fd50177410f99de9bfb255e4",
    "LocalAddr.scala": "cb1d7de3d0f04c77f79044fb53c560409743962cd82d2a86b8ae95898231ac61",
    "MeshWithDelays.scala": "c33f4d576301a902d33cb7ab9f3fd4c95dc9b76f605fb399037c55e03291e6cb",
    "TagQueue.scala": "513c0b8aa5b813207919eee9916b3c27f4a5805f3c5a07f2acab04042671c0ae",
}


@dataclass(frozen=True)
class OrderingContract:
    """Reviewed controller semantics bound to exact immutable source files.

    Source identity is a compiler legality witness, not measured bitstream
    support. Every new execution engine still needs complete output validation.
    """

    source_directory: str

    def require(self):
        directory = Path(self.source_directory)
        pins = []
        for name, expected in _SOURCE_HASHES.items():
            path = directory / name
            if not path.is_file():
                raise ValueError("missing pinned SPAD ordering implementation")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError("unsupported SPAD ordering implementation")
            pins.append({"path": str(path.resolve()), "sha256": digest})
        if F.DIM != 16 or F.OPERAND_DTYPE != "i8" or F.ACCUMULATOR_DTYPE != "i32":
            raise ValueError("unsupported SPAD ordering resource configuration")
        return pins


def _unit_ws(op):
    if not isinstance(op, G.ConfigExOp) or op.operands_:
        return False
    return (
        op.a("dataflow") == isa.WEIGHT_STATIONARY
        and op.a("a_stride", 1) == op.a("c_stride", 1) == 1
        and op.a("a_transpose", 0) == op.a("b_transpose", 0) == 0
        and op.a("sys_shift", 0) == 0
        and op.a("acc_scale", 1.0) == 1.0
        and op.a("set_only_strides", 0) == 0
        and op.a("act", 0) in (isa.RELU, isa.NO_ACTIVATION)
    )


def _interval(address, rows, reserved):
    if type(address) is not int or address < 0 or address & isa.ACC_ADDR_BIT:
        raise ValueError("same-tile coalescing requires static real SPAD addresses")
    if address % F.DIM or rows != F.DIM:
        raise ValueError("same-tile coalescing requires full aligned DIM extents")
    end = address + rows
    if not any(
        start <= address and end <= start + length for start, length in reserved
    ):
        raise ValueError("SPAD tile is outside its full reserved lifetime")
    return address, end


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _product(ops, reserved):
    if len(ops) != 8:
        raise ValueError("coalescing requires a complete four-tile product")
    writes, reads = [], []
    for tile in range(4):
        pre, comp = ops[2 * tile : 2 * tile + 2]
        if not isinstance(pre, G.PreloadOp) or not isinstance(comp, G.ComputeOp):
            raise ValueError("unknown instruction in SPAD product chain")  # noqa: TRY004
        if set(pre.attributes) != {"bd", "c", "bd_rows", "bd_cols", "c_rows", "c_cols"}:
            raise ValueError("unknown preload semantic fields")
        if set(comp.attributes) != {
            "a",
            "bd",
            "a_rows",
            "a_cols",
            "bd_rows",
            "bd_cols",
            "accumulate",
        }:
            raise ValueError("unknown compute semantic fields")
        if (
            pre.operands_
            or comp.operands_
            or pre.res is not None
            or comp.res is not None
        ):
            raise ValueError("dynamic SPAD product address refuses fence coalescing")
        if any(pre.a(k) != F.DIM for k in ("bd_rows", "bd_cols", "c_rows", "c_cols")):
            raise ValueError("partial preload tile refuses fence coalescing")
        if any(comp.a(k) != F.DIM for k in ("a_rows", "a_cols", "bd_rows", "bd_cols")):
            raise ValueError("partial compute tile refuses fence coalescing")
        if bool(comp.a("accumulate", False)) != (tile != 0):
            raise ValueError("stationary B flip/stay order must remain explicit")
        weight = pre.a("bd")
        if tile == 0:
            _interval(weight, F.DIM, reserved)
        elif weight != isa.GARBAGE_ADDR:
            raise ValueError(
                "retained stationary B must use the declared garbage sentinel"
            )
        output = _interval(pre.a("c"), F.DIM, reserved)
        moving = _interval(comp.a("a"), F.DIM, reserved)
        real_d = _interval(comp.a("bd"), F.DIM, reserved)
        if _overlap(output, moving) or _overlap(output, real_d):
            raise ValueError("in-place SPAD product lacks an input lifetime proof")
        if tile and (
            output[0] != writes[0][0] + tile * F.DIM
            or moving[0] != reads[0][0] + tile * F.DIM
            or real_d[0] != reads[1][0] + tile * F.DIM
        ):
            raise ValueError(
                "product tile endpoints disagree with contiguous reservations"
            )
        writes.append(output)
        reads.extend((moving, real_d))
    return writes, reads


def coalesce(module, contract: OrderingContract, reservations):
    """Validate every marked chain completely before removing any fence.

    Markers request consideration only. Actual primitive operands, controller
    state and disjoint reserved full extents establish legality. The chain is
    straight-line, bounded by draining CONFIG_EX operations; DMA/CPU effects
    inside it refuse. All unmarked fences remain unchanged.
    """
    pins = contract.require()
    module.verify()
    reserved = list(reservations.values())
    if any(
        type(start) is not int
        or type(length) is not int
        or start < 0
        or length < 0
        or start + length > F.SPAD_ROWS
        for start, length in reserved
    ):
        raise ValueError("invalid full SPAD reservation")
    if any(
        _overlap((a, a + b), (c, c + d))
        for i, (a, b) in enumerate(reserved)
        for c, d in reserved[i + 1 :]
        if b and d
    ):
        raise ValueError("overlapping full SPAD reservations")
    marked = [
        op
        for op in module.walk()
        if isinstance(op, G.FenceOp) and op.a("internal_spad_stage", 0) == 1
    ]
    chains = {}
    for fence in marked:
        if fence.operands_ or fence.res is not None:
            raise ValueError("unknown fence operands")
        block = fence.parent
        if block is None:
            raise ValueError("detached SPAD fence")
        operations = list(block.ops)
        index = operations.index(fence)
        begin = index - 1
        while begin >= 0 and not isinstance(operations[begin], G.ConfigExOp):
            begin -= 1
        end = index + 1
        while end < len(operations) and not isinstance(operations[end], G.ConfigExOp):
            end += 1
        if begin < 0 or end == len(operations):
            raise ValueError("SPAD chain lacks draining configuration boundaries")
        chains[(block, begin, end)] = operations
    observed, products, dependency_edges = set(), 0, 0
    for (_, begin, end), operations in chains.items():
        if not _unit_ws(operations[begin]) or not _unit_ws(operations[end]):
            raise ValueError("SPAD ordering requires unit WS strides and no transposes")
        body = operations[begin + 1 : end]
        group, history = [], []
        for op in body:
            if isinstance(op, G.FenceOp):
                if op not in marked:
                    raise ValueError("unmarked completion boundary inside SPAD chain")
                writes, reads = _product(group, reserved)
                # The controller compares tile bases, not arbitrary intervals.
                # Exact aligned extents make every overlapping RAW tile equal.
                for older_writes in history:
                    for w in older_writes:
                        for r in reads:
                            if _overlap(w, r):
                                if w != r:
                                    raise ValueError(
                                        "partial RAW tile overlap lacks a mesh hazard witness"
                                    )
                                dependency_edges += 1
                history.append(writes)
                products += 1
                observed.add(op)
                group = []
            else:
                group.append(op)
        if group:
            raise ValueError(
                "unterminated product or unknown side effect in SPAD chain"
            )
    if observed != set(marked):
        raise ValueError("unproved internal SPAD fence")
    for fence in marked:
        Rewriter.erase_op(fence)
    module.verify()
    return {
        "schema": "same_tile_spad_fence_coalescing_v1",
        "source_pins": pins,
        "static_products": products,
        "static_removed_fences": len(marked),
        "exact_same_tile_raw_edges": dependency_edges,
        "external_memory_fences_preserved": True,
        "physical_bitstream_support": "UNKNOWN until complete engine qualification",
    }
