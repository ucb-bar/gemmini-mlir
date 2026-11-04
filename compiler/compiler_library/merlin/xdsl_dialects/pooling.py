"""The standard NHWC max-pool operation missing from xDSL's linalg inventory.

Reuse xDSL's named structured-operation parser, operand segmentation, printer and
shape verification. The hidden body is the standard signed maximum of the input
and destination accumulator; the shape-only window argument is unused. Target
admission remains a separate source proof, including dtype and initializer.
"""
from xdsl.builder import Builder
from xdsl.dialects import arith
from xdsl.dialects.builtin import AnyFloat, IntegerType, MemRefType, TensorType, Signedness
from xdsl.dialects.linalg.ops import PoolingOperation, YieldOp
from xdsl.ir import Region
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.irdl import irdl_op_definition
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class PoolingNhwcMaxOp(PoolingOperation):
    name = 'linalg.pooling_nhwc_max'

    @classmethod
    def get_hidden_region(cls, inputs, outputs):
        if len(inputs) != 2 or len(outputs) != 1:
            raise ValueError('NHWC max pooling requires two inputs and one output')
        arg_types = cls.body_arg_types((*inputs, *outputs))
        if arg_types[0] != arg_types[2]:
            raise ValueError('NHWC max pooling currently requires matching input/output element types')
        if isinstance(arg_types[0], IntegerType):
            if arg_types[0].signedness.data != Signedness.SIGNLESS:
                raise ValueError('NHWC signed max requires a signless integer element type')
            maximum = arith.MaxSIOp
        elif isinstance(arg_types[0], AnyFloat):
            maximum = arith.MaximumfOp
        else:
            raise ValueError('NHWC max pooling requires integer or floating elements')

        @Builder.implicit_region(arg_types)
        def body(args):
            YieldOp(maximum(args[2], args[0]))
        return body

    def get_default_indexing_maps(self):
        n, oh, ow, c, kh, kw = (AffineDimExpr(i) for i in range(6))
        sh, sw = self.strides.get_values()
        dh, dw = self.dilations.get_values()
        return (
            AffineMap(6, 0, (n, oh * sh + kh * dh, ow * sw + kw * dw, c)),
            AffineMap(6, 0, (kh, kw)),
            AffineMap(6, 0, (n, oh, ow, c)),
        )

    def verify_(self):
        if len(self.inputs) != 2 or len(self.outputs) != 1:
            raise VerifyException('NHWC max pooling requires two inputs and one output')
        for value, rank in zip(self.operands, (4, 2, 4), strict=True):
            if not isinstance(value.type, (TensorType, MemRefType)) or len(value.type.get_shape()) != rank:
                raise VerifyException('NHWC max pooling requires rank-4 input/output and rank-2 window')
        for name, attr in (('strides', self.strides), ('dilations', self.dilations)):
            values = tuple(attr.get_values())
            if attr.get_shape() != (2,) or len(values) != 2 or any(v <= 0 for v in values):
                raise VerifyException(f'NHWC max pooling {name} must contain two positive integers')
        if 'cast' in self.attributes or 'cast_fn' in self.attributes:
            raise VerifyException('explicit NHWC pool cast functions are not supported')
        out_type = self.outputs[0].type
        if isinstance(out_type, TensorType):
            if len(self.res) != 1 or self.res[0].type != out_type:
                raise VerifyException('NHWC pool tensor result must match its destination type')
        elif self.res:
            raise VerifyException('NHWC pool buffer form has no SSA result')
        try:
            expected = self.get_hidden_region(self.inputs, self.outputs)
        except (ValueError, AssertionError) as error:
            raise VerifyException(str(error)) from error
        block = self.body.block
        actual_ops, expected_ops = tuple(block.ops), tuple(expected.block.ops)
        if len(actual_ops) != 2 or len(block.args) != 3:
            raise VerifyException('NHWC max pooling requires its canonical maximum/yield region')
        maximum, end = actual_ops
        if (maximum.name != expected_ops[0].name or maximum.properties != expected_ops[0].properties
            or tuple(maximum.operands) not in (
                ((block.args[2], block.args[0]), (block.args[0], block.args[2]))
                if isinstance(out_type.element_type, IntegerType)
                else ((block.args[2], block.args[0]),))
            or len(maximum.results) != 1 or maximum.results[0].type != out_type.element_type
            or maximum.regions or maximum.successors
            or end.name != 'linalg.yield' or tuple(end.operands) != tuple(maximum.results)
            or end.properties or end.regions or end.successors
            or any(not key.startswith('prov.') for op in actual_ops for key in op.attributes)):
            raise VerifyException('NHWC max pooling scalar region must preserve its named maximum semantics')
        super().verify_()


def register_pooling(ctx):
    """Add exactly the missing operation; retain the caller's existing dialect."""
    ctx.load_op(PoolingNhwcMaxOp)
