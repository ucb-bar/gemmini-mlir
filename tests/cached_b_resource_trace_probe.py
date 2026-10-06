"""Resolve emitted dense addresses and verify tensor cells and K order."""

from __future__ import annotations

from collections import Counter

from merlin.llvmlower.static_llvm_cfg import (
    StaticInt,
    StaticPointer,
    trace_static_function,
)

from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import isa
from mlir_oot.tables import rtl_facts as F


def prove(generator):
    s = generator.shape
    module = generator.build()
    fn = module.body.block.first_op
    arguments = [StaticPointer(i, StaticInt(0, 64)) for i in range(3)]
    scratch = {}
    accumulator = {}
    written = set()
    stride = {}
    block_stride = {}
    counts = Counter()
    loads = Counter()
    requested = Counter()
    weight = None
    destination = None
    real_weights = 0
    for step in trace_static_function(
        fn,
        arguments,
        observe=lambda op: isinstance(op, G._GemminiOp),
        pointer_index_bits=64,
    ):
        op = step.operation
        counts[op.name] += 1
        if isinstance(op, G.ConfigLdOp):
            state = op.a("load_id")
            stride[state] = op.a("stride")
            block_stride[state] = op.a("block_stride", F.DIM)
        elif isinstance(op, G.MvinOp):
            pointer = step.inputs[0]
            assert isinstance(pointer, StaticPointer) and pointer.base in (0, 1)
            local = op.a("local")
            rows = op.a("rows")
            cols = op.a("cols")
            state = op.a("load_id")
            assert local < F.SPAD_ROWS and rows <= F.DIM and cols <= 4 * F.DIM
            extent = s.m * s.k if pointer.base == 0 else s.k * s.n
            loads[str(pointer.base)] += 1
            requested["load_bytes"] += rows * cols
            for r in range(rows):
                for c in range(cols):
                    cell = (local + r + c // F.DIM * block_stride[state], c % F.DIM)
                    source = pointer.offset.value + r * stride[state] + c
                    assert cell[0] < F.SPAD_ROWS and source < extent
                    scratch[cell] = (pointer.base, source)
        elif isinstance(op, G.PreloadOp):
            if op.a("bd") != isa.GARBAGE_ADDR:
                real_weights += 1
                weight = [
                    [(scratch[(op.a("bd") + r, c)]) for c in range(op.a("bd_cols"))]
                    for r in range(op.a("bd_rows"))
                ]
            destination = (op.a("c") & 0x3FFF, bool(op.a("c") & isa.ACC_ACCUMULATE_BIT))
        elif isinstance(op, G.ComputeOp):
            assert not step.inputs, "This dense probe expects static local addresses"
            assert weight is not None and destination is not None
            a_rows, a_cols = op.a("a_rows"), op.a("a_cols")
            c_row, accumulate = destination
            assert len(weight) == a_cols
            n0 = weight[0][0][1] % s.n
            k0 = weight[0][0][1] // s.n
            for k, row in enumerate(weight):
                for col, source in enumerate(row):
                    assert source == (1, (k0 + k) * s.n + n0 + col)
            for row in range(a_rows):
                a = [scratch[(op.a("a") + row, k)] for k in range(a_cols)]
                m0 = a[0][1] // s.k
                assert all(
                    source == (0, m0 * s.k + k0 + k) for k, source in enumerate(a)
                )
                for col in range(len(weight[0])):
                    cell = (c_row + row, col)
                    assert cell[0] < F.ACC_ROWS
                    if accumulate:
                        assert accumulator[cell] == (m0, n0 + col, k0)
                    else:
                        assert k0 == 0
                    accumulator[cell] = (m0, n0 + col, k0 + a_cols)
        elif isinstance(op, G.MvoutOp):
            pointer = step.inputs[0]
            assert isinstance(pointer, StaticPointer) and pointer.base == 2
            width = 4 if s.output_dtype == "i32" else 1
            assert pointer.offset.value % width == 0
            local = op.a("local") & 0x3FFF
            for row in range(op.a("rows")):
                for col in range(op.a("cols")):
                    offset = pointer.offset.value // width + row * s.n + col
                    logical = (offset // s.n, offset % s.n, s.k)
                    cell = (local + row + col // F.DIM * F.DIM, col % F.DIM)
                    assert accumulator[cell] == logical and offset not in written
                    assert offset < s.m * s.n
                    written.add(offset)
            requested["store_bytes"] += op.a("rows") * op.a("cols") * width
    assert written == set(range(s.m * s.n))
    return {
        "commands": dict(counts),
        "input_load_commands": loads["0"],
        "weight_load_commands": loads["1"],
        "requested_payload": dict(requested),
        "real_B_mesh_preloads": real_weights,
        "all_output_cells_written_once": len(written),
        "increasing_source_K_and_exact_source_addresses": True,
        "full_live_storage_endpoints_checked": True,
        "physical_DRAM_bytes": "UNKNOWN",
    }
