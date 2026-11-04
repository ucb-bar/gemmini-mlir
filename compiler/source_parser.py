"""Typed source parser with the standard named structured product declarations.

The selected xDSL release omits these upstream linalg names. Definitions here
reuse NamedOperation's actual custom parser, operand segmentation and region
contract. No source text, provenance tag or capsule identity is rewritten.
"""
from __future__ import annotations
from functools import cache
from typing import ClassVar

from xdsl.dialects import arith
from xdsl.dialects.builtin import ArrayAttr, AffineMapAttr, IntegerType, AnyFloat
from xdsl.dialects.linalg import Linalg
from xdsl.dialects.linalg.abstract_ops import NamedOperation
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap, AffineDimExpr
from xdsl.irdl import irdl_op_definition
from xdsl.utils.exceptions import VerifyException

# Loop coordinates: batch b, output row m, output column n, reduction k.
# Tuple entries map loop coordinates to each operand's ranked dimensions.
PRODUCTS = {
    'linalg.matmul': (3, ((0,2),(2,1),(0,1)), (2,)),
    'linalg.batch_matmul': (4, ((0,1,3),(0,3,2),(0,1,2)), (3,)),
    'linalg.batch_matmul_transpose_a': (4, ((0,3,1),(0,3,2),(0,1,2)), (3,)),
    'linalg.batch_matmul_transpose_b': (4, ((0,1,3),(0,2,3),(0,1,2)), (3,)),
    'linalg.matmul_transpose_a': (3, ((2,0),(2,1),(0,1)), (2,)),
    'linalg.matmul_transpose_b': (3, ((0,2),(1,2),(0,1)), (2,)),
    'linalg.matvec': (2, ((0,1),(1,),(0,)), (1,)),
    'linalg.vecmat': (2, ((1,),(1,0),(0,)), (1,)),
    'linalg.dot': (1, ((0,),(0,),()), (0,)),
    'linalg.batch_matvec': (3, ((0,1,2),(0,2),(0,1)), (2,)),
    'linalg.batch_vecmat': (3, ((0,2),(0,2,1),(0,1)), (2,)),
    'linalg.batch_reduce_matmul': (4, ((0,1,3),(0,3,2),(1,2)), (0,3)),
}


def _declaration(name, geometry):
    rank, maps, reductions = geometry
    class Product(NamedOperation):
        PRINT_ATTRS_IN_FRONT: ClassVar[bool] = True
        @classmethod
        def get_hidden_region(cls, inputs, outputs):
            types = cls.body_arg_types((*inputs,*outputs))
            if len(types) != 3:
                raise ValueError(f'{name} requires two inputs and one destination')
            block = Block(arg_types=types)
            a,b,old = block.args
            dtype = old.type
            if isinstance(dtype,IntegerType):
                widened=[]
                for value in (a,b):
                    if not isinstance(value.type,IntegerType) or value.type.width.data > dtype.width.data:
                        raise ValueError(f'{name} requires integer operands no wider than its accumulator')
                    if value.type != dtype:
                        cast=arith.ExtSIOp(value,dtype);block.add_op(cast);value=cast.result
                    widened.append(value)
                product=arith.MuliOp(*widened)
                addition=arith.AddiOp(product.result,old)
            elif isinstance(dtype,AnyFloat):
                if a.type != dtype or b.type != dtype:
                    raise ValueError(f'{name} requires matching floating operand types')
                product=arith.MulfOp(a,b);addition=arith.AddfOp(product.result,old)
            else:
                raise ValueError(f'{name} has an unsupported accumulator type')
            block.add_ops([product,addition,YieldOp(addition.result)])
            return Region(block)
        def get_default_indexing_maps(self):
            return tuple(AffineMap(rank,0,tuple(AffineDimExpr(i) for i in positions)) for positions in maps)
        def get_iterator_types(self):
            return ArrayAttr(IteratorTypeAttr.reduction() if i in reductions else IteratorTypeAttr.parallel() for i in range(rank))
        def verify_(self):
            for key in ('indexing_maps', 'iterator_types'):
                explicit = self.properties.get(key, self.attributes.get(key))
                expected = self.get_indexing_maps() if key == 'indexing_maps' else self.get_iterator_types()
                if explicit is not None and explicit != expected:
                    raise VerifyException(f'{name} explicit {key} differ from its admitted named geometry')
            cast = self.properties.get('cast', self.attributes.get('cast'))
            if cast is not None and 'cast_signed' not in str(cast):
                raise VerifyException(f'{name} requires signed integer extension')
            if len(self.inputs)!=2 or len(self.outputs)!=1:
                raise VerifyException(f'{name} requires two inputs and one destination')
            extents={}
            for operand,positions in zip((*self.inputs,*self.outputs),maps,strict=True):
                shape=operand.type.get_shape()
                if len(shape)!=len(positions):
                    raise VerifyException(f'{name} operand rank differs from its indexing map')
                for dimension,extent in zip(positions,shape,strict=True):
                    if extent>=0 and dimension in extents and extents[dimension]!=extent:
                        raise VerifyException(f'{name} contraction extent mismatch')
                    if extent>=0:extents[dimension]=extent
            if len(self.results)!=1 or self.results[0].type!=self.outputs[0].type:
                raise VerifyException(f'{name} tensor result must match its destination')
    Product.name=name
    Product.__name__=''.join(part.capitalize() for part in name.split('.')[1].split('_'))+'SourceOp'
    return irdl_op_definition(Product)


@cache
def register_named_products():
    """Register missing standard names for all fresh source/inventory contexts."""
    declarations=[]
    existing={op.name for op in Linalg.operations}
    for name,geometry in PRODUCTS.items():
        if name not in existing or name == 'linalg.matmul':
            declaration=_declaration(name,geometry)
            if name in existing:
                Linalg._operations[:] = [declaration if op.name == name else op for op in Linalg.operations]
            else:
                Linalg._operations.append(declaration)
            declarations.append(declaration)
    # The public contract reader caches its multi-result-aware linalg dialect.
    # Register the same declarations there if it was already initialized.
    from merlin.targetgen.contract import linalg_iface
    dialect=linalg_iface._LINALG_DIALECT
    if dialect is not None:
        replacements = {op.name: op for op in declarations}
        known={op.name for op in dialect.operations}
        dialect._operations[:] = [replacements.get(op.name, op) for op in dialect.operations]
        dialect._operations.extend(op for op in declarations if op.name not in known)
    return tuple(declarations)


def make_source_context():
    register_named_products()
    from merlin.frontends.linalg_mlir import make_context
    return make_context()


def parse_source_module(source: str):
    """Parse the original source with typed named-product operations."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    module = parse_mlir_text(source,ctx=make_source_context())
    # Canonical rank-two matmul enters the existing scalar-DAG dense matcher.
    # One named product becomes one generic product; result positions, values,
    # operation order and provenance remain unchanged before graph grouping.
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.rewriter import Rewriter
    for op in list(module.walk()):
        if op.name != 'linalg.matmul':
            continue
        op.verify()
        generic = GenericOp(inputs=tuple(op.inputs), outputs=tuple(op.outputs),
                            body=op.body.clone(),
                            indexing_maps=op.get_indexing_maps(),
                            iterator_types=op.get_iterator_types(),
                            result_types=tuple(value.type for value in op.results))
        generic.attributes.update(op.attributes)
        generic.location = op.location
        Rewriter.replace_op(op,generic)
    return module
