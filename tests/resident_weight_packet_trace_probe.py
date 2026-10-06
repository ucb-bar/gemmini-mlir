"""Prove emitted resident convolution source cells, K order and live extents."""

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
    s = generator.conv
    fn = generator.build().body.block.first_op
    scratch, accumulator, strides, blocks = {}, {}, {}, {}
    counts, requested = Counter(), Counter()
    written, input_written = set(), set()
    weights = destination = None
    for step in trace_static_function(
        fn,
        [StaticPointer(i, StaticInt(0, 64)) for i in range(3)],
        observe=lambda op: isinstance(op, G._GemminiOp),
        pointer_index_bits=64,
    ):
        op = step.operation
        counts[op.name] += 1
        if isinstance(op, G.ConfigLdOp):
            strides[op.a("load_id")] = op.a("stride")
            blocks[op.a("load_id")] = op.a("block_stride", F.DIM)
        elif isinstance(op, G.MvinOp):
            pointer = step.inputs[0]
            assert isinstance(pointer, (StaticPointer, StaticInt))
            zero = isinstance(pointer, StaticInt) or pointer.base is None
            if zero:
                assert (
                    pointer.value
                    if isinstance(pointer, StaticInt)
                    else pointer.offset.value
                ) == 0
            else:
                assert pointer.base in (0, 1)
            state, local = op.a("load_id"), op.a("local")
            if not zero:
                requested["load_bytes"] += op.a("rows") * op.a("cols")
            for row in range(op.a("rows")):
                for col in range(op.a("cols")):
                    cell = (local + row + col // F.DIM * blocks[state], col % F.DIM)
                    assert 0 <= cell[0] < F.SPAD_ROWS
                    if generator.flat_spatial_planes and state == 0:
                        assert cell not in input_written
                        input_written.add(cell)
                    if zero:
                        scratch[cell] = None
                    else:
                        source = pointer.offset.value + row * strides[state] + col
                        assert (
                            0
                            <= source
                            < (
                                s.h * s.w * s.cin
                                if pointer.base == 0
                                else 9 * s.cin * s.cout
                            )
                        )
                        scratch[cell] = (pointer.base, source)
        elif isinstance(op, G.PreloadOp):
            if op.a("bd") != isa.GARBAGE_ADDR:
                weights = [
                    [scratch[(op.a("bd") + k, n)] for n in range(op.a("bd_cols"))]
                    for k in range(op.a("bd_rows"))
                ]
            address = (
                isa.acc_addr(
                    step.inputs[0].value, accumulate=bool(op.a("c_accumulate"))
                )
                if step.inputs
                else op.a("c")
            )
            destination = (address & 0x3FFF, bool(address & isa.ACC_ACCUMULATE_BIT))
        elif isinstance(op, G.ComputeOp):
            assert weights is not None and destination is not None
            a_address = step.inputs[0].value if step.inputs else op.a("a")
            k0, n0 = divmod(weights[0][0][1], s.cout)
            tap, ci = divmod(k0, s.cin)
            kh, kw = divmod(tap, 3)
            for k, row in enumerate(weights):
                for n, source in enumerate(row):
                    assert source == (1, (k0 + k) * s.cout + n0 + n)
            c_row, accumulate = destination
            tile = c_row // F.DIM // s.bn
            output_y = generator.row_tiles[tile][0]
            for row in range(op.a("a_rows")):
                oy, ox = (
                    divmod(output_y + row, s.ow)
                    if generator.flat_spatial_planes
                    else (
                        output_y + row // generator.output_pitch,
                        row % generator.output_pitch,
                    )
                )
                valid = oy < s.oh and ox < s.ow
                for k in range(op.a("a_cols")):
                    cell = (a_address + row * s.stride, k)
                    assert 0 <= cell[0] < s.cin // F.DIM * generator.plane
                    if valid:
                        iy, ix = oy * s.stride + kh - 1, ox * s.stride + kw - 1
                        expected = (
                            (0, (iy * s.w + ix) * s.cin + ci + k)
                            if 0 <= iy < s.h and 0 <= ix < s.w
                            else None
                        )
                        assert scratch[cell] == expected
                for n in range(len(weights[0])):
                    cell = (c_row + row, n)
                    assert 0 <= cell[0] < F.ACC_ROWS
                    if accumulate:
                        assert accumulator[cell] == (oy, ox, n0 + n, k0)
                    else:
                        assert k0 == 0
                    accumulator[cell] = (oy, ox, n0 + n, k0 + op.a("a_cols"))
        elif isinstance(op, G.MvoutOp):
            pointer = step.inputs[0]
            assert isinstance(pointer, StaticPointer) and pointer.base == 2
            width = 4 if s.output_dtype == "i32" else 1
            assert pointer.offset.value % width == 0
            local = op.a("local") & 0x3FFF
            for row in range(op.a("rows")):
                for col in range(op.a("cols")):
                    offset = pointer.offset.value // width + row * s.cout + col
                    pixel, n = divmod(offset, s.cout)
                    oy, ox = divmod(pixel, s.ow)
                    cell = (local + row + col // F.DIM * F.DIM, col % F.DIM)
                    assert accumulator[cell] == (oy, ox, n, 9 * s.cin)
                    assert 0 <= offset < s.oh * s.ow * s.cout and offset not in written
                    written.add(offset)
            requested["store_bytes"] += op.a("rows") * op.a("cols") * width
    assert written == set(range(s.oh * s.ow * s.cout))
    if generator.flat_spatial_planes:
        assert len(input_written) == s.cin * generator.plane
    return {
        "commands": dict(counts),
        "requested_payload": dict(requested),
        "all_output_cells_written_once": len(written),
        "source_operand_and_increasing_K_proved": True,
        "scratch_and_ACC_lifetimes_proved": True,
        "physical_DRAM_bytes": "UNKNOWN",
    }
