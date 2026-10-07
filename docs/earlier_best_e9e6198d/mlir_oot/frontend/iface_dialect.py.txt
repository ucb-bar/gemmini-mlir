"""The ``merlin_iface`` v0.1 INPUT dialect, defined as real xDSL IRDL ops and types.

Parsing the interface through a typed dialect (rather than scraping text) is what buys the
MLIR-style guarantee: a malformed graph -- an operand of the wrong type, a missing required
attribute, a resident handle used after it was evicted -- fails in ``verify()`` at parse time.

The grammar is regular: ``<mnemonic> <operands>? <attr-dict>? : <function-type-or-type>``. One
shared parser/printer implements it for every op, and each op then adds its own verifier.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FloatAttr,
    IntAttr,
    IntegerAttr,
    StringAttr,
    TensorType,
)
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    opt_result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

DIALECT_NAME = "merlin_iface"
SUPPORTED_VERSION = "0.1"

#: The epilogue vocabulary of the command-buffer ABI (command_buffer_abi.yaml, COMMIT.attributes).
EPILOGUE_STAGES = ("bias_add", "bias", "requant", "acc_scale", "relu", "maxpool")
#: role values the grammar defines for a leaf tensor.
LEAF_ROLES = ("weight", "input", "bias", "scale")


@irdl_attr_definition
class ResidentType(ParametrizedAttribute, TypeAttribute):
    """``!merlin_iface.resident`` -- an opaque handle to a packed, stationary weight."""

    name = "merlin_iface.resident"


@irdl_attr_definition
class AccType(ParametrizedAttribute, TypeAttribute):
    """``!merlin_iface.acc<i32>`` -- an opaque integer accumulator handle."""

    name = "merlin_iface.acc"

    elem: Attribute

    @classmethod
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        parser.parse_punctuation("<")
        t = parser.parse_type()
        parser.parse_punctuation(">")
        return (t,)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print_attribute(self.elem)
        printer.print_string(">")


class _IfaceOp(IRDLOperation):
    """Shared assembly format for the whole dialect.

    ``<operands>? <attr-dict>? : (<in-types>) -> <out-type>``, or ``: <type>`` for a leaf
    declaration that takes no operand.
    """

    #: attribute names this op requires; the verifier reports each missing one by name.
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def parse(cls, parser: Parser):
        pos = parser.pos
        unresolved = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_unresolved_operand, parser.parse_unresolved_operand
        )
        unresolved = unresolved or []
        attrs = parser.parse_optional_attr_dict()
        parser.parse_punctuation(":")
        ftype = parser.parse_optional_function_type()
        if ftype is None:
            result_types: list[Attribute] = [parser.parse_type()]
            input_types: list[Attribute] = []
        else:
            input_types = list(ftype.inputs.data)
            result_types = list(ftype.outputs.data)
        operands = parser.resolve_operands(unresolved, input_types, pos)
        return cls.build(operands=[operands], result_types=[result_types], attributes=attrs)

    def print(self, printer: Printer) -> None:
        if self.operands:
            printer.print_string(" ")
            printer.print_list(self.operands, printer.print_ssa_value)
        if self.attributes:
            printer.print_string(" ")
            printer.print_attr_dict(self.attributes)
        printer.print_string(" : ")
        if not self.operands:
            printer.print_string("")
            printer.print_attribute(self.results[0].type)
            return
        printer.print_string("(")
        printer.print_list([o.type for o in self.operands], printer.print_attribute)
        printer.print_string(") -> ")
        if len(self.results) == 1:
            printer.print_attribute(self.results[0].type)
        else:
            printer.print_string("()")

    # -- shared verification helpers ----------------------------------------
    def _require(self, *names: str) -> None:
        for n in names:
            if n not in self.attributes:
                raise VerifyException(f"{self.name}: required attribute '{n}' is missing")

    def verify_(self) -> None:
        self._require(*self.REQUIRED_ATTRS)
        _verify_epilogue(self)


def _string(op: IRDLOperation, key: str) -> str | None:
    a = op.attributes.get(key)
    return a.data if isinstance(a, StringAttr) else None


def _str_list(op: IRDLOperation, key: str) -> list[str] | None:
    a = op.attributes.get(key)
    if not isinstance(a, ArrayAttr):
        return None
    out: list[str] = []
    for e in a.data:
        if not isinstance(e, StringAttr):
            raise VerifyException(f"{op.name}: '{key}' must be a list of strings")
        out.append(e.data)
    return out


def int_list(op: IRDLOperation, key: str) -> list[int] | None:
    """An unquoted integer-list geometry attribute (``kernel``, ``stride``, ``pool_size``, ...)."""
    a = op.attributes.get(key)
    if a is None:
        return None
    if not isinstance(a, ArrayAttr):
        raise VerifyException(f"{op.name}: '{key}' must be an integer list")
    out: list[int] = []
    for e in a.data:
        if isinstance(e, IntegerAttr):
            out.append(e.value.data)
        elif isinstance(e, IntAttr):
            out.append(e.data)
        else:
            raise VerifyException(
                f"{op.name}: '{key}' must be an UNQUOTED integer list; got {type(e).__name__}"
            )
    return out


def float_attr(op: IRDLOperation, key: str) -> float | None:
    a = op.attributes.get(key)
    if isinstance(a, FloatAttr):
        return a.value.data
    if isinstance(a, IntegerAttr):
        return float(a.value.data)
    return None


def bool_attr(op: IRDLOperation, key: str) -> bool | None:
    """A boolean attribute: xDSL parses `true`/`false` into an i1 IntegerAttr."""
    a = op.attributes.get(key)
    if isinstance(a, IntegerAttr):
        return bool(a.value.data)
    return None


def _verify_epilogue(op: IRDLOperation) -> None:
    """Every epilogue stage must be in the ABI vocabulary AND carry its own parameters."""
    stages = _str_list(op, "epilogue")
    if stages is None:
        return
    for s in stages:
        if s not in EPILOGUE_STAGES:
            raise VerifyException(f"{op.name}: unknown epilogue stage {s!r}")
    if "acc_scale" in stages and float_attr(op, "acc_scale") is None:
        raise VerifyException(f"{op.name}: epilogue 'acc_scale' declares no 'acc_scale' multiplier")
    if "requant" in stages and op.attributes.get("requant_shift") is None:
        raise VerifyException(f"{op.name}: epilogue 'requant' declares no 'requant_shift'")
    if ("bias_add" in stages or "bias" in stages) and _string(op, "bias") is None:
        raise VerifyException(f"{op.name}: a bias epilogue stage names no bias tensor")
    if "maxpool" in stages:
        for k in ("pool_in_dims", "pool_size", "pool_stride"):
            if int_list(op, k) is None:
                raise VerifyException(f"{op.name}: epilogue 'maxpool' requires '{k}'")
        pad = int_list(op, "pool_padding") or [0, 0, 0, 0]
        if any(p for p in pad) and op.attributes.get("pool_pad_value") is None:
            raise VerifyException(
                f"{op.name}: nonzero pool_padding requires 'pool_pad_value' (no identity is assumed)"
            )


@irdl_op_definition
class TensorOp(_IfaceOp):
    """``merlin_iface.tensor`` -- declare a leaf input/weight/bias/scale."""

    name = "merlin_iface.tensor"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name", "role")

    def verify_(self) -> None:
        super().verify_()
        if self.operands:
            raise VerifyException("merlin_iface.tensor takes no operands")
        role = _string(self, "role")
        if role not in LEAF_ROLES:
            raise VerifyException(f"merlin_iface.tensor: role {role!r} is not one of {LEAF_ROLES}")
        if not isinstance(self.results[0].type, TensorType):
            raise VerifyException("merlin_iface.tensor: result must be a ranked tensor")


@irdl_op_definition
class ResidentPackOp(_IfaceOp):
    """``merlin_iface.resident_pack`` -> command-buffer ``RES_PACK``."""

    name = "merlin_iface.resident_pack"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("layout",)

    def verify_(self) -> None:
        super().verify_()
        if len(self.operands) != 1 or not isinstance(self.operands[0].type, TensorType):
            raise VerifyException("merlin_iface.resident_pack takes exactly one tensor operand")
        if not isinstance(self.results[0].type, ResidentType):
            raise VerifyException("merlin_iface.resident_pack must produce !merlin_iface.resident")


@irdl_op_definition
class MatmulOp(_IfaceOp):
    """``merlin_iface.matmul`` -> ``MATMUL_RESIDENT``."""

    name = "merlin_iface.matmul"
    arguments = var_operand_def()
    result = var_result_def()

    def verify_(self) -> None:
        super().verify_()
        if len(self.operands) != 2:
            raise VerifyException("merlin_iface.matmul takes (tensor, resident)")
        if not isinstance(self.operands[1].type, ResidentType):
            raise VerifyException("merlin_iface.matmul: rhs must be !merlin_iface.resident")
        if not isinstance(self.results[0].type, AccType):
            raise VerifyException("merlin_iface.matmul must produce !merlin_iface.acc")


@irdl_op_definition
class CommitOp(_IfaceOp):
    """``merlin_iface.commit`` -> ``COMMIT``."""

    name = "merlin_iface.commit"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name", "epilogue", "output_dtype")

    def verify_(self) -> None:
        super().verify_()
        if len(self.operands) != 1 or not isinstance(self.operands[0].type, AccType):
            raise VerifyException("merlin_iface.commit consumes exactly one accumulator handle")


@irdl_op_definition
class EvictOp(_IfaceOp):
    """``merlin_iface.evict`` -> ``EVICT``."""

    name = "merlin_iface.evict"
    arguments = var_operand_def()
    result = opt_result_def()

    def verify_(self) -> None:
        super().verify_()
        if len(self.operands) != 1 or not isinstance(self.operands[0].type, ResidentType):
            raise VerifyException("merlin_iface.evict takes one !merlin_iface.resident handle")


@irdl_op_definition
class Conv2dOp(_IfaceOp):
    """``merlin_iface.conv2d`` -> ``CONV2D``."""

    name = "merlin_iface.conv2d"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name", "kernel", "output_dtype")

    def verify_(self) -> None:
        super().verify_()
        layout = _string(self, "layout")
        if layout is not None and layout != "nhwc":
            raise VerifyException(f"merlin_iface.conv2d: layout {layout!r} is rejected (nhwc only)")
        k = int_list(self, "kernel")
        if k is None or len(k) != 4:
            raise VerifyException("merlin_iface.conv2d: 'kernel' must be [kh, kw, ci, co]")
        ifm = self.operands[0].type
        if isinstance(ifm, TensorType):
            shape = [d for d in ifm.get_shape()]
            if len(shape) != 4:
                raise VerifyException("merlin_iface.conv2d: ifm must be rank-4 NHWC")
            if shape[3] != k[2]:
                raise VerifyException(
                    f"merlin_iface.conv2d: kernel ci={k[2]} != ifm channel dim {shape[3]}"
                )


@irdl_op_definition
class MovementOp(_IfaceOp):
    """``merlin_iface.movement`` -> ``MOVEMENT`` (identity load/store round trip)."""

    name = "merlin_iface.movement"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name",)


@irdl_op_definition
class MatmulBatchedOp(_IfaceOp):
    """``merlin_iface.matmul_batched`` -> ``BATCHED_MATMUL``."""

    name = "merlin_iface.matmul_batched"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name", "output_dtype")

    def verify_(self) -> None:
        super().verify_()
        types = [o.type for o in self.operands] + [self.results[0].type]
        shapes = [list(t.get_shape()) for t in types if isinstance(t, TensorType)]
        if len(shapes) != 3 or len({len(s) for s in shapes}) != 1 or len(shapes[0]) < 3:
            raise VerifyException("merlin_iface.matmul_batched: a, w and dst share a rank >= 3")
        if shapes[0][:-2] != shapes[1][:-2] or shapes[0][:-2] != shapes[2][:-2]:
            raise VerifyException("merlin_iface.matmul_batched: batch prefixes must be identical")


@irdl_op_definition
class BiasAddOp(_IfaceOp):
    """``merlin_iface.bias_add`` -> ``BIAS_ADD``."""

    name = "merlin_iface.bias_add"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name",)


@irdl_op_definition
class AttentionQkOp(_IfaceOp):
    """``merlin_iface.attention_qk`` -> ``ATTENTION_QK`` (dst = q @ transpose(k))."""

    name = "merlin_iface.attention_qk"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name", "output_dtype")


@irdl_op_definition
class AttentionPvOp(_IfaceOp):
    """``merlin_iface.attention_pv`` -> ``ATTENTION_PV`` (dst = p @ v)."""

    name = "merlin_iface.attention_pv"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name", "output_dtype")


@irdl_op_definition
class KChainOp(_IfaceOp):
    """``merlin_iface.k_chain`` -> ``K_CHAIN`` (dst = (src @ weight) @ weight2)."""

    name = "merlin_iface.k_chain"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name",)


@irdl_op_definition
class ResidualAddOp(_IfaceOp):
    """``merlin_iface.residual_add`` -> ``RESIDUAL_ADD``."""

    name = "merlin_iface.residual_add"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name", "lhs_scale", "rhs_scale", "bound_lsb")


@irdl_op_definition
class DepthwiseConv2dOp(_IfaceOp):
    """``merlin_iface.depthwise_conv2d`` -> ``DEPTHWISE_CONV2D``."""

    name = "merlin_iface.depthwise_conv2d"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name", "kernel")


@irdl_op_definition
class SoftmaxOp(_IfaceOp):
    """``merlin_iface.softmax`` -> ``SOFTMAX``.

    A row-wise normalization over ``axis``. ``scale`` pre-multiplies the operand (attention's
    1/sqrt(d)); both ride as attributes, so the operand list is just the source tensor.
    """

    name = "merlin_iface.softmax"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name",)


@irdl_op_definition
class RmsnormOp(_IfaceOp):
    """``merlin_iface.rmsnorm`` -> ``RMSNORM`` (src scaled by gamma over its row RMS)."""

    name = "merlin_iface.rmsnorm"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name", "eps")


@irdl_op_definition
class RopeOp(_IfaceOp):
    """``merlin_iface.rope`` -> ``ROPE`` (rotary position embedding over the trailing axis)."""

    name = "merlin_iface.rope"
    arguments = var_operand_def()
    result = var_result_def()
    REQUIRED_ATTRS: ClassVar[tuple[str, ...]] = ("name", "theta")


OPS = [
    TensorOp,
    ResidentPackOp,
    MatmulOp,
    CommitOp,
    EvictOp,
    Conv2dOp,
    MovementOp,
    MatmulBatchedOp,
    BiasAddOp,
    AttentionQkOp,
    AttentionPvOp,
    KChainOp,
    ResidualAddOp,
    DepthwiseConv2dOp,
    SoftmaxOp,
    RmsnormOp,
    RopeOp,
]

MerlinIface = Dialect(DIALECT_NAME, OPS, [ResidentType, AccType])
