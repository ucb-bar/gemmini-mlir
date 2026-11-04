"""Borrow closed immutable source copies through counted host descriptors.

The shared preparation admission supplies the source operation domain. This
adapter additionally proves each source coordinate is in range. No writable
pointer escapes: published results and carried source regions continue through
the ordinary materialization boundary. This is not a physical storage or opaque
callee alias contract.
"""
from __future__ import annotations

from math import prod

from xdsl.dialects import builtin
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

from merlin.llvmlower.tensor_provenance_cse import _supported
from merlin.transforms.certificates import ir_digest


def copy_view(operation):
    """Return a readonly projection descriptor, or retain ordinary lowering.

    Only the exact shared all-parallel, complete identity-output copy domain is
    consumed. Integer-to-float preparations admitted by the shared CSE API are
    deliberately outside this descriptor family: their scalar recipes remain
    in the shared host lowering.
    """
    if operation.name != 'linalg.generic' or not _supported(operation):
        return None
    block = operation.regions[0].block
    body = tuple(block.ops)
    if (len(body) != 1 or body[0].name != 'linalg.yield'
            or tuple(body[0].operands) != (block.args[0],)):
        return None
    if any(not key.startswith('prov.')
           for key in (*body[0].attributes, *body[0].properties)):
        return None
    parent = operation.parent_op()
    while parent is not None:
        if any('strictfp' in key or 'strictfp' in str(value)
               for key, value in (*parent.attributes.items(), *parent.properties.items())):
            return None
        parent = parent.parent_op()
    source_type, result_type = operation.operands[0].type, operation.results[0].type
    if (source_type.element_type != result_type.element_type
            or any(not isinstance(t.encoding, builtin.NoneAttr)
                   or any(n <= 0 for n in t.get_shape()) for t in (source_type,result_type))):
        return None
    shape = tuple(result_type.get_shape())
    mapping = operation.indexing_maps.data[0].data
    if mapping.num_symbols or mapping.num_dims != len(shape):
        return None
    coordinates = []
    for expression, extent in zip(mapping.results, source_type.get_shape(), strict=True):
        if isinstance(expression, AffineDimExpr) and shape[expression.position] == extent:
            coordinates.append(('axis',expression.position))
        elif isinstance(expression, AffineConstantExpr) and 0 <= expression.value < extent:
            coordinates.append(('constant',expression.value))
        else:
            return None
    width = {'i1':1,'i8':1,'i16':2,'i32':4,'i64':8,'f32':4,'bf16':2}.get(str(result_type.element_type))
    if width is None:
        return None
    return dict(shape=shape, coordinates=tuple(coordinates), report=dict(
        schema='merlin.gemmini.host_readonly_view.v1',
        operation=operation.name, source_operation_sha256=ir_digest(operation),
        source_shape=list(source_type.get_shape()), result_shape=list(shape),
        source_map=str(mapping), source_region=str(operation.attributes.get('prov.region_id','')),
        admission='merlin.llvmlower.tensor_provenance_cse._supported; yield-only bounded projection',
        elided_result_storage_bytes=prod(shape)*width,
        publication='ordinary store_result copy; source SCF descriptors materialize before carrying'))
