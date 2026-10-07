"""The ``gemmini`` TARGET dialect: one xDSL op per RoCC instruction class.

This is the compiler's intermediate target IR. Every op verifies its own fields against the
RTL-derived facts (mesh DIM, scratchpad/accumulator depth, DMA block length, decoder-legal funct),
so an out-of-range scratchpad row or an over-wide DMA burst is a ``verify()`` failure in the
compiler rather than a garbled word on the device.

The region of ``gemmini.kernel`` is the emitted kernel's body; its block arguments are the DRAM
buffers the harness passes in, one per interface tensor, in the ABI's argument order.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr, f32, i64
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, Region, TypeAttribute
from xdsl.traits import IsTerminator, NoTerminator
from xdsl.irdl import (
    IRDLOperation,
    traits_def,
    irdl_attr_definition,
    irdl_op_definition,
    opt_result_def,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from ..tables.funct_table import FUNCT
from . import facts

DIALECT_NAME = "gemmini"


@irdl_attr_definition
class DramType(ParametrizedAttribute, TypeAttribute):
    """``!gemmini.dram<"NAME">`` -- a DRAM buffer the harness allocates and passes as a pointer."""

    name = "gemmini.dram"

    tensor: StringAttr

    @classmethod
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        parser.parse_punctuation("<")
        s = parser.parse_str_literal()
        parser.parse_punctuation(">")
        return (StringAttr(s),)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print_string_literal(self.tensor.data)
        printer.print_string(">")


@irdl_attr_definition
class AddrType(ParametrizedAttribute, TypeAttribute):
    """``!gemmini.addr`` -- a materialised DRAM byte address (a pointer argument plus an offset)."""

    name = "gemmini.addr"


def _int(op: IRDLOperation, key: str, default: int | None = None) -> int:
    a = op.attributes.get(key)
    if isinstance(a, IntegerAttr):
        return int(a.value.data)
    if default is None:
        raise VerifyException(f"{op.name}: missing integer attribute '{key}'")
    return default


def _float(op: IRDLOperation, key: str, default: float = 1.0) -> float:
    a = op.attributes.get(key)
    if isinstance(a, FloatAttr):
        return float(a.value.data)
    if isinstance(a, IntegerAttr):
        return float(a.value.data)
    return default


def iattr(v: int) -> IntegerAttr:
    return IntegerAttr(int(v), i64)


def fattr(v: float) -> FloatAttr:
    return FloatAttr(float(v), f32)


def iarray(vs) -> ArrayAttr:
    return ArrayAttr([iattr(v) for v in vs])


class _GemminiOp(IRDLOperation):
    """Shared assembly format: ``<mnemonic> <operands>? <attr-dict>``."""

    INSTR_CLASS: ClassVar[str] = ""

    @classmethod
    def parse(cls, parser: Parser):
        pos = parser.pos
        unresolved = (
            parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_unresolved_operand, parser.parse_unresolved_operand
            )
            or []
        )
        attrs = parser.parse_optional_attr_dict()
        results: list[Attribute] = []
        in_types: list[Attribute] = []
        if parser.parse_optional_punctuation(":") is not None:
            ftype = parser.parse_optional_function_type()
            if ftype is None:
                results = [parser.parse_type()]
            else:
                in_types = list(ftype.inputs.data)
                results = list(ftype.outputs.data)
        if unresolved and not in_types:
            in_types = [AddrType()] * len(unresolved)
        operands = parser.resolve_operands(unresolved, in_types, pos)
        return cls.build(operands=[operands], result_types=[results], attributes=attrs)

    def print(self, printer: Printer) -> None:
        if self.operands:
            printer.print_string(" ")
            printer.print_list(self.operands, printer.print_ssa_value)
        if self.attributes:
            printer.print_string(" ")
            printer.print_attr_dict(self.attributes)
        if self.operands or self.results:
            printer.print_string(" : (")
            printer.print_list([o.type for o in self.operands], printer.print_attribute)
            printer.print_string(") -> ")
            if len(self.results) == 1:
                printer.print_attribute(self.results[0].type)
            else:
                printer.print_string("()")

    def verify_(self) -> None:
        if self.INSTR_CLASS:
            f = FUNCT.get(self.INSTR_CLASS)
            if f is None or f not in facts.LEGAL_FUNCTS:
                raise VerifyException(
                    f"{self.name}: instruction class {self.INSTR_CLASS!r} is not decoded here"
                )

    # -- shared field checks -------------------------------------------------
    def _check_tile(self, rows_key: str = "rows", cols_key: str = "cols", max_cols: int | None = None) -> None:
        rows, cols = _int(self, rows_key), _int(self, cols_key)
        if not 1 <= rows <= facts.DIM:
            raise VerifyException(f"{self.name}: rows={rows} outside 1..{facts.DIM}")
        limit = max_cols if max_cols is not None else facts.DIM
        if not 1 <= cols <= limit:
            raise VerifyException(f"{self.name}: cols={cols} outside 1..{limit}")

    def _check_local(self, key: str = "spad") -> None:
        a = _int(self, key)
        if a == facts.GARBAGE_ADDR:
            return
        if a & facts.BIT_IS_ACC:
            row = a & 0x3FFF
            if row >= facts.ACC_ROWS:
                raise VerifyException(f"{self.name}: accumulator row {row} >= {facts.ACC_ROWS}")
        elif a >= facts.SP_ROWS:
            raise VerifyException(f"{self.name}: scratchpad row {a} >= {facts.SP_ROWS}")


@irdl_op_definition
class KernelOp(IRDLOperation):
    """``gemmini.kernel`` -- the emitted kernel, one block argument per DRAM buffer."""

    name = "gemmini.kernel"
    body = region_def()

    @classmethod
    def parse(cls, parser: Parser):
        attrs = parser.parse_optional_attr_dict()
        region = parser.parse_region()
        return cls.build(regions=[region], attributes=attrs)

    def print(self, printer: Printer) -> None:
        if self.attributes:
            printer.print_string(" ")
            printer.print_attr_dict(self.attributes)
        printer.print_string(" ")
        printer.print_region(self.body)

    def verify_(self) -> None:
        for a in self.body.block.args:
            if not isinstance(a.type, DramType):
                raise VerifyException("gemmini.kernel arguments must be !gemmini.dram buffers")


@irdl_op_definition
class DramAddrOp(_GemminiOp):
    """``gemmini.dram_addr`` -- ptrtoint of a kernel pointer argument plus a constant byte offset.

    The ONLY way a DRAM address enters the instruction stream. There is no literal-address form.
    """

    name = "gemmini.dram_addr"
    arguments = var_operand_def()
    result = var_result_def()

    def verify_(self) -> None:
        super().verify_()
        if len(self.operands) != 1 or not isinstance(self.operands[0].type, DramType):
            raise VerifyException("gemmini.dram_addr takes exactly one !gemmini.dram buffer")
        if _int(self, "offset", 0) < 0:
            raise VerifyException("gemmini.dram_addr: negative offset")


@irdl_op_definition
class FlushOp(_GemminiOp):
    name = "gemmini.flush"
    INSTR_CLASS: ClassVar[str] = "FLUSH"
    arguments = var_operand_def()
    result = opt_result_def()


@irdl_op_definition
class FenceOp(_GemminiOp):
    """``gemmini.fence`` -- the scalar ``fence`` that retires the accelerator queue."""

    name = "gemmini.fence"
    arguments = var_operand_def()
    result = opt_result_def()


@irdl_op_definition
class LaneStageOp(_GemminiOp):
    """``gemmini.lane_stage`` -- a hole in the command stream where an OFF-MESH stage runs.

    Some programs interleave a region this datapath has no encoding for (a normalisation, a rotary
    map, a row-wise exponential -- none of which the readout implements, because this elaborated
    design instantiates no normalizer) BETWEEN regions that belong on the array. Neither whole-lane
    route answers such a program: running it all on the scalar lane leaves the mesh idle for a
    contraction that belongs on it, and the mesh has no encoding for the off-mesh stage.

    So the rewrite keeps the contraction on the array and marks the stage's POSITION in the command
    stream with this op. It encodes nothing and issues no instruction; the code generator expands it
    in place into scalar blocks inside the SAME ``gemmini_kernel``, between the instructions either
    side of it. ``index`` selects which of the program's off-mesh stages goes here.
    """

    name = "gemmini.lane_stage"
    arguments = var_operand_def()
    result = opt_result_def()


@irdl_op_definition
class ConfigExOp(_GemminiOp):
    name = "gemmini.config_ex"
    INSTR_CLASS: ClassVar[str] = "CONFIG"
    arguments = var_operand_def()
    result = opt_result_def()

    def verify_(self) -> None:
        super().verify_()
        if _int(self, "dataflow") not in (0, 1):
            raise VerifyException("gemmini.config_ex: dataflow must be 0 (OS) or 1 (WS)")


@irdl_op_definition
class ConfigLdOp(_GemminiOp):
    name = "gemmini.config_ld"
    INSTR_CLASS: ClassVar[str] = "CONFIG"
    arguments = var_operand_def()
    result = opt_result_def()

    def verify_(self) -> None:
        super().verify_()
        if _int(self, "id") not in (0, 1, 2):
            raise VerifyException("gemmini.config_ld: id must name one of the three load units")


@irdl_op_definition
class ConfigStOp(_GemminiOp):
    name = "gemmini.config_st"
    INSTR_CLASS: ClassVar[str] = "CONFIG"
    arguments = var_operand_def()
    result = opt_result_def()


@irdl_op_definition
class MvinOp(_GemminiOp):
    name = "gemmini.mvin"
    INSTR_CLASS: ClassVar[str] = "MVIN"
    arguments = var_operand_def()
    result = opt_result_def()

    def verify_(self) -> None:
        super().verify_()
        self._check_tile(max_cols=facts.DIM * facts.MAX_BLOCK_LEN)
        self._check_local()


@irdl_op_definition
class Mvin2Op(_GemminiOp):
    """Load unit 1: the stationary B operand."""

    name = "gemmini.mvin2"
    INSTR_CLASS: ClassVar[str] = "MVIN2"
    arguments = var_operand_def()
    result = opt_result_def()

    def verify_(self) -> None:
        super().verify_()
        self._check_tile(max_cols=facts.DIM * facts.MAX_BLOCK_LEN)
        self._check_local()


@irdl_op_definition
class Mvin3Op(_GemminiOp):
    """Load unit 2: the bias/D operand, moved straight into the accumulator."""

    name = "gemmini.mvin3"
    INSTR_CLASS: ClassVar[str] = "MVIN3"
    arguments = var_operand_def()
    result = opt_result_def()

    def verify_(self) -> None:
        super().verify_()
        self._check_tile(max_cols=facts.DIM * facts.MAX_BLOCK_LEN)
        self._check_local()


@irdl_op_definition
class MvoutOp(_GemminiOp):
    name = "gemmini.mvout"
    INSTR_CLASS: ClassVar[str] = "MVOUT"
    arguments = var_operand_def()
    result = opt_result_def()

    def verify_(self) -> None:
        super().verify_()
        self._check_tile(max_cols=facts.DIM * facts.MAX_BLOCK_LEN)
        self._check_local()


@irdl_op_definition
class PreloadOp(_GemminiOp):
    name = "gemmini.preload"
    INSTR_CLASS: ClassVar[str] = "PRELOAD"
    arguments = var_operand_def()
    result = opt_result_def()

    def verify_(self) -> None:
        super().verify_()
        self._check_tile("bd_rows", "bd_cols")
        self._check_tile("c_rows", "c_cols")
        self._check_local("bd")
        self._check_local("c")


@irdl_op_definition
class ComputeOp(_GemminiOp):
    """``gemmini.compute`` -- ``preloaded`` picks COMPUTE_PRELOADED over COMPUTE_ACCUMULATE."""

    name = "gemmini.compute"
    arguments = var_operand_def()
    result = opt_result_def()

    def verify_(self) -> None:
        cls = "COMPUTE_PRELOADED" if _int(self, "preloaded", 1) else "COMPUTE_ACCUMULATE"
        if FUNCT[cls] not in facts.LEGAL_FUNCTS:
            raise VerifyException(f"gemmini.compute: {cls} is not decoded here")
        self._check_tile("a_rows", "a_cols")
        self._check_local("a")


@irdl_op_definition
class ReturnOp(_GemminiOp):
    """``gemmini.return`` -- the kernel's terminator."""

    name = "gemmini.return"
    arguments = var_operand_def()
    result = opt_result_def()
    traits = traits_def(IsTerminator())


OPS = [
    KernelOp,
    ReturnOp,
    DramAddrOp,
    FlushOp,
    FenceOp,
    ConfigExOp,
    ConfigLdOp,
    ConfigStOp,
    MvinOp,
    Mvin2Op,
    Mvin3Op,
    MvoutOp,
    PreloadOp,
    ComputeOp,
]

Gemmini = Dialect(DIALECT_NAME, OPS, [DramType, AddrType])
