"""Flatten a verified ``gemmini``-dialect kernel into the linear ISA command stream it denotes.

One :class:`~mlir_oot.target.isa.Instr` per target op, with each operand resolved to either an
immediate or ``(kernel pointer argument, byte offset)``. This is the form the loop re-roller and the
LLVM emitter both consume, so neither of them has to know the dialect.
"""

from __future__ import annotations

from ..target import dialect as gd
from ..target import isa
from ..target.dialect import _float, _int


class StreamError(Exception):
    pass


def flatten(kernel: gd.KernelOp, arg_tensors: list[str]) -> list[isa.Instr | str]:
    """The kernel's command stream.

    Two string sentinels ride in it: a bare ``"fence"`` is the scalar memory fence, and
    ``"lane:<i>"`` is the position off-mesh stage ``i`` expands into.
    """
    block = kernel.body.block
    addr: dict = {}
    out: list[isa.Instr | str] = []
    for op in block.ops:
        if isinstance(op, gd.DramAddrOp):
            i = list(block.args).index(op.operands[0])
            addr[op.results[0]] = isa.ArgAddr(arg_tensors[i], _int(op, "offset", 0))
        elif isinstance(op, gd.ReturnOp):
            continue
        elif isinstance(op, gd.FenceOp):
            out.append("fence")
        elif isinstance(op, gd.LaneStageOp):
            # A hole, not an instruction: the re-roller carries a `str` node through untouched and
            # never rolls one into a loop, so the marker keeps its exact position in the stream.
            out.append(f"lane:{_int(op, 'index', 0)}")
        else:
            instr = _encode(op)
            if op.operands:
                instr = isa.Instr(instr.cls, addr[op.operands[0]], instr.rs2, instr.note)
            out.append(instr)
    return out


def _encode(op) -> isa.Instr:
    if isinstance(op, gd.FlushOp):
        return isa.flush(_int(op, "skip", 0))
    if isinstance(op, gd.ConfigExOp):
        return isa.config_ex(
            dataflow=_int(op, "dataflow"), sys_act=_int(op, "act", 0),
            sys_shift=_int(op, "sys_shift", 0), sys_acc_scale=_float(op, "acc_scale", 1.0),
            c_stride=_int(op, "c_stride", 1), a_stride=_int(op, "a_stride", 1),
            a_transpose=bool(_int(op, "a_transpose", 0)),
            b_transpose=bool(_int(op, "b_transpose", 0)),
        )
    if isinstance(op, gd.ConfigLdOp):
        return isa.config_ld(_int(op, "stride"), _float(op, "scale", 1.0),
                             bool(_int(op, "shrunk", 0)), _int(op, "id", 0))
    if isinstance(op, gd.ConfigStOp):
        return isa.config_st(
            _int(op, "stride"), _int(op, "act", 0), _float(op, "acc_scale", 1.0),
            pool_stride=_int(op, "pool_stride", 0), pool_size=_int(op, "pool_size", 0),
            pool_out_dim=_int(op, "pool_out_dim", 0), porows=_int(op, "porows", 0),
            pocols=_int(op, "pocols", 0), orows=_int(op, "orows", 0), ocols=_int(op, "ocols", 0),
            upad=_int(op, "upad", 0), lpad=_int(op, "lpad", 0),
        )
    if isinstance(op, (gd.MvinOp, gd.Mvin2Op, gd.Mvin3Op, gd.MvoutOp)):
        builder = {gd.MvinOp: isa.mvin, gd.Mvin2Op: isa.mvin2,
                   gd.Mvin3Op: isa.mvin3, gd.MvoutOp: isa.mvout}[type(op)]
        return builder(isa.Imm(0), _int(op, "spad"), _int(op, "cols"), _int(op, "rows"))
    if isinstance(op, gd.PreloadOp):
        return isa.preload(_int(op, "bd"), _int(op, "c"), _int(op, "bd_cols"),
                           _int(op, "bd_rows"), _int(op, "c_cols"), _int(op, "c_rows"))
    if isinstance(op, gd.ComputeOp):
        return isa.compute(_int(op, "a"), _int(op, "a_cols"), _int(op, "a_rows"),
                           bool(_int(op, "preloaded", 1)),
                           _int(op, "bd", None) if "bd" in op.attributes else None)
    raise StreamError(f"no encoding for target op {op.name!r}")
