"""The `gemmini` target dialect: one op per RoCC command class this backend emits.

The ops are real xDSL IRDL operations with verifiers, so the lowered module is checked against
the RTL-derived machine limits (mesh `DIM`, scratchpad depth, accumulator depth, the legal funct
set) before anything is encoded.  `gemmini.host_*` ops are the compiler-generated CPU-lane
fixups the accelerator store path cannot express; they lower to ordinary LLVM control flow, not
to a library call.
"""
from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr, i64
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_result_def, var_operand_def
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from ..tables import rtl_facts as F
from ..tables import isa

DIM = F.DIM


def _val(attr: Any) -> Any:
    if isinstance(attr, StringAttr):
        return attr.data
    if isinstance(attr, IntegerAttr):
        return int(attr.value.data)
    if isinstance(attr, FloatAttr):
        return float(attr.value.data)
    if isinstance(attr, ArrayAttr):
        return [_val(a) for a in attr.data]
    return attr


class _GemminiOp(IRDLOperation):
    """Shared concrete syntax: `gemmini.<op> %operands {attrs} : (types) -> (types)`."""

    operands_ = var_operand_def()
    res = opt_result_def()

    def print(self, printer: Printer) -> None:
        if self.operands_:
            printer.print_string(" ")
            printer.print_list(self.operands_, printer.print_ssa_value)
        printer.print_op_attributes(self.attributes)
        printer.print_string(" : ")
        printer.print_function_type(
            [v.type for v in self.operands_],
            [self.res.type] if self.res is not None else [],
        )

    @classmethod
    def parse(cls, parser: Parser) -> "_GemminiOp":
        pos = parser.pos
        unresolved = (
            parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_unresolved_operand,
                parser.parse_unresolved_operand,
            )
            or []
        )
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        ftype = parser.parse_function_type()
        op = cls(operands=[parser.resolve_operands(unresolved, ftype.inputs.data, pos)],
                 result_types=[list(ftype.outputs.data)])
        op.attributes |= attrs
        return op

    def a(self, key: str, default: Any = None) -> Any:
        attr = self.attributes.get(key)
        return default if attr is None else _val(attr)

    def _extent(self, key: str) -> None:
        n = self.a(key)
        if not isinstance(n, int) or not (1 <= n <= DIM):
            raise VerifyException(f"{self.name}: `{key}` = {n!r} must be in 1..{DIM} (mesh DIM)")

    def _local(self, key: str) -> None:
        addr = self.a(key)
        if not isinstance(addr, int) or addr < 0:
            raise VerifyException(f"{self.name}: `{key}` must be a non-negative local address")
        if addr == isa.GARBAGE_ADDR:
            return
        if addr & isa.ACC_ADDR_BIT:
            row = addr & 0x3FFF
            if row >= F.ACC_ROWS:
                raise VerifyException(
                    f"{self.name}: accumulator row {row} exceeds the RTL depth {F.ACC_ROWS}")
        elif addr >= F.SPAD_ROWS:
            raise VerifyException(
                f"{self.name}: scratchpad row {addr} exceeds the RTL depth {F.SPAD_ROWS}")


@irdl_op_definition
class FlushOp(_GemminiOp):
    """`gemmini.flush` — drain the accelerator (k_FLUSH)."""

    name = "gemmini.flush"


@irdl_op_definition
class ConfigExOp(_GemminiOp):
    """`gemmini.config_ex` — dataflow / activation / strides (k_CONFIG, CONFIG_EX)."""

    name = "gemmini.config_ex"

    def verify_(self) -> None:
        if self.a("dataflow") not in (0, 1):
            raise VerifyException("gemmini.config_ex: `dataflow` must be 0 or 1")


@irdl_op_definition
class ConfigLdOp(_GemminiOp):
    """`gemmini.config_ld` — DMA load stride / scale / id (k_CONFIG, CONFIG_LD)."""

    name = "gemmini.config_ld"

    def verify_(self) -> None:
        if self.a("load_id") not in (0, 1, 2):
            raise VerifyException("gemmini.config_ld: `load_id` must be 0, 1 or 2")
        if int(self.a("stride", 0)) < 0:
            raise VerifyException("gemmini.config_ld: `stride` must be non-negative")


@irdl_op_definition
class ConfigStOp(_GemminiOp):
    """`gemmini.config_st` — DMA store stride, accumulator activation/scale, pooling geometry."""

    name = "gemmini.config_st"

    def verify_(self) -> None:
        if self.a("acc_act") not in (0, 1):
            raise VerifyException("gemmini.config_st: `acc_act` must be NO_ACTIVATION or RELU")
        for key,bits in (("pool_stride",2),("pool_size",2),("pool_out_dim",8),
                         ("porows",8),("pocols",8),("orows",8),("ocols",8),
                         ("upad",2),("lpad",6)):
            value=int(self.a(key,0))
            if not 0<=value<(1<<bits):
                raise VerifyException(f"gemmini.config_st: {key} exceeds its hardware field")
        if self.a("pool_stride",0) and any(self.a(k,0)<=0 for k in
                ("pool_size","pool_out_dim","porows","pocols","orows","ocols")):
            raise VerifyException("gemmini.config_st: enabled pooling requires complete geometry")


@irdl_op_definition
class MvinOp(_GemminiOp):
    """`gemmini.mvin` — DRAM -> scratchpad/accumulator DMA (k_MVIN / k_MVIN2)."""

    name = "gemmini.mvin"

    def verify_(self) -> None:
        if len(self.operands_) != 1:
            raise VerifyException("gemmini.mvin takes the source DRAM pointer as its operand")
        self._extent("rows")
        cols = self.a("cols")
        # The load controller distributes up to four adjacent DIM-wide
        # blocks down scratchpad rows using CONFIG_LD.block_mvin_stride.
        # q1013 uses 16x64 MVIN/MVIN2 for its 1x1 kernels.
        local = self.a("local")
        max_cols = DIM if isinstance(local, int) and local & isa.ACC_ADDR_BIT else DIM * 4
        if not isinstance(cols, int) or not (1 <= cols <= max_cols):
            raise VerifyException(f"gemmini.mvin: `cols` = {cols!r} must be in 1..{max_cols}")
        self._local("local")


@irdl_op_definition
class MvoutOp(_GemminiOp):
    """`gemmini.mvout` — scratchpad/accumulator -> DRAM DMA (k_MVOUT)."""

    name = "gemmini.mvout"

    def verify_(self) -> None:
        if len(self.operands_) != 1:
            raise VerifyException("gemmini.mvout takes the destination DRAM pointer as its operand")
        self._extent("rows")
        cols = self.a("cols")
        local = self.a("local")
        # The store controller can read four adjacent 16-column accumulator
        # tiles as one 16x64 int8 DMA.  q1013 uses this form dynamically.  A
        # full-width i32 accumulator read keeps the single-tile bound.
        wide_acc_i8 = (isinstance(local, int) and bool(local & isa.ACC_ADDR_BIT)
                       and not bool(local & isa.ACC_FULL_ROW_BIT))
        max_cols = DIM * 4 if wide_acc_i8 else DIM
        if not isinstance(cols, int) or not (1 <= cols <= max_cols):
            raise VerifyException(
                f"gemmini.mvout: `cols` = {cols!r} must be in 1..{max_cols}")
        self._local("local")


@irdl_op_definition
class PreloadOp(_GemminiOp):
    """`gemmini.preload` — load the stationary operand into the mesh (k_PRELOAD)."""

    name = "gemmini.preload"

    def verify_(self) -> None:
        for key in ("bd_cols", "bd_rows", "c_cols", "c_rows"):
            self._extent(key)
        self._local("bd")
        self._local("c")


@irdl_op_definition
class ComputeOp(_GemminiOp):
    """`gemmini.compute` — stream the moving operand through the mesh (k_COMPUTE_*)."""

    name = "gemmini.compute"

    def verify_(self) -> None:
        self._extent("a_cols")
        self._extent("a_rows")
        if len(self.operands_) == 0:
            self._local("a")
        elif len(self.operands_) == 1:
            if self.operands_[0].type != i64 or "a" in self.attributes:
                raise VerifyException("gemmini.compute: dynamic A address must be one i64 operand")
            maximum, reserved = self.a("a_max"), self.a("a_reserved_rows")
            if (not isinstance(maximum, int) or not isinstance(reserved, int)
                    or maximum < 0 or maximum + DIM > reserved
                    or reserved > F.SPAD_ROWS):
                raise VerifyException("gemmini.compute: dynamic A range exceeds reserved scratchpad rows")
        else:
            raise VerifyException("gemmini.compute: at most one dynamic A address is supported")
        if "bd" in self.attributes:
            self._local("bd")


@irdl_op_definition
class FenceOp(_GemminiOp):
    """`gemmini.fence` — wait for every outstanding accelerator command to retire."""

    name = "gemmini.fence"


@irdl_op_definition
class ScratchOp(_GemminiOp):
    """`gemmini.scratch` — a compiler-allocated DRAM staging buffer."""

    name = "gemmini.scratch"

    def verify_(self) -> None:
        if int(self.a("bytes", 0)) <= 0:
            raise VerifyException("gemmini.scratch: `bytes` must be positive")


@irdl_op_definition
class HostEpilogueOp(_GemminiOp):
    """`gemmini.host_epilogue` — compiler-generated CPU-lane readout the store path cannot fuse.

    Operands are (source i32 staging buffer, destination buffer[, bias buffer]).  The `stages`
    attribute is the ordered ABI epilogue; codegen materialises a real loop nest for it.
    """

    name = "gemmini.host_epilogue"

    def verify_(self) -> None:
        if len(self.operands_) < 2:
            raise VerifyException("gemmini.host_epilogue needs a source and a destination")
        if self.a("stages") is None:
            raise VerifyException("gemmini.host_epilogue: `stages` is required")


@irdl_op_definition
class HostTransposeOp(_GemminiOp):
    """`gemmini.host_transpose` — compiler-generated 2-D transpose into a staging buffer."""

    name = "gemmini.host_transpose"

    def verify_(self) -> None:
        if len(self.operands_) != 2:
            raise VerifyException("gemmini.host_transpose needs a source and a destination")


@irdl_op_definition
class HostLaneProgramOp(_GemminiOp):
    """`gemmini.host_lane_program` — a region this datapath admits no lowering for, placed on
    the scalar lane.

    It carries NO accelerator instruction by construction: the operands are the DRAM pointers
    the compiler-generated CPU-lane program reads and writes, and `regions` names the interface
    regions whose placement produced it.  Its presence in the target module is what makes the
    routing decision visible in the IR rather than only in the command buffer.
    """

    name = "gemmini.host_lane_program"

    def verify_(self) -> None:
        if not self.operands_:
            raise VerifyException("gemmini.host_lane_program needs at least one buffer")
        if self.a("regions_placed") is None:
            raise VerifyException("gemmini.host_lane_program: `regions_placed` is required")


GEMMINI_OPS = (
    FlushOp, ConfigExOp, ConfigLdOp, ConfigStOp, MvinOp, MvoutOp, PreloadOp,
    ComputeOp, FenceOp, ScratchOp, HostEpilogueOp, HostTransposeOp,
    HostLaneProgramOp,
)

GEMMINI = Dialect("gemmini", list(GEMMINI_OPS), [])
