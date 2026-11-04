"""Gather a regular source slice grid directly into its concatenated matrix.

Every admission comes from tensor shapes, view associations, actual permutation,
static slice addresses and concat dimension. Padding remains the original source
tensor, including its exact halo values. No convolution or arithmetic is inferred
from provenance. Device calls and numerical stages stay opaque.
"""
from __future__ import annotations


def _panel(value):
    from .source_tensor_views import _static_tensor, _slice, _permutation

    shape = _static_tensor(value)
    if shape is None or len(shape) != 2:
        return None
    expand = value.owner
    if (getattr(expand, 'name', '') != 'tensor.expand_shape' or len(expand.operands) != 1
            or len(expand.results) != 1):
        return None
    collapse = expand.operands[0].owner
    if (getattr(collapse, 'name', '') != 'tensor.collapse_shape' or len(collapse.operands) != 1
            or len(collapse.results) != 1 or len(_static_tensor(collapse.results[0]) or ()) != 1):
        return None
    transpose = collapse.operands[0].owner
    if getattr(transpose, 'name', '') != 'linalg.transpose' or _permutation(transpose) != (0, 2, 3, 1):
        return None
    input_shape = _static_tensor(transpose.operands[0])
    if input_shape is None or len(input_shape) != 4:
        return None
    batch, channels, height, width = input_shape
    if shape != (batch*height*width, channels):
        return None
    # A collapse to one flat dimension and its ranked expansion preserve
    # row-major order; verification establishes both reassociation contracts.
    if (expand.operands[0].type.get_shape() != (batch*channels*height*width,)
            or any(v.type.element_type != value.type.element_type
                   for op in (expand, collapse, transpose) for v in (*op.operands, *op.results))):
        return None
    allowed = {
        'tensor.expand_shape': {'reassociation', 'static_output_shape'},
        'tensor.collapse_shape': {'reassociation'},
    }
    for op in (expand, collapse):
        if any(k not in allowed[op.name] and not k.startswith('prov.')
               for k in (*op.properties, *op.attributes)):
            return None
    current = transpose.operands[0]
    offsets, strides = [0]*4, [1]*4
    members = [expand, collapse, transpose]
    while (entry := _slice(current)) is not None:
        view, off, step = entry
        if view.parent is not expand.parent:
            return None
        offsets = [a*s+b for a,s,b in zip(offsets, step, off, strict=True)]
        strides = [a*b for a,b in zip(strides, step, strict=True)]
        members.append(view)
        current = view.operands[0]
    if len(members) == 3 or _static_tensor(current) is None:
        return None
    if any(op.parent is not expand.parent for op in members):
        return None
    base_shape = _static_tensor(current)
    if (len(base_shape) != 4 or offsets[:2] != [0,0] or strides[:2] != [1,1]
            or base_shape[:2] != (batch,channels)):
        return None
    return dict(value=value, source=current, shape=shape, image=(batch,channels,height,width),
                offsets=tuple(offsets), strides=tuple(strides), members=tuple(members))


def _grid(op):
    from .source_tensor_views import _static_tensor
    if (op.name != 'tensor.concat' or len(op.results) != 1 or len(op.operands) < 2
            or len(op.operands) > 256):
        return None
    dim = op.properties.get('dim', op.attributes.get('dim'))
    if getattr(getattr(dim,'value',None),'data',None) != 1:
        return None
    if any(k != 'dim' and not k.startswith('prov.') for k in (*op.properties, *op.attributes)):
        return None
    panels = [_panel(v) for v in op.operands]
    if any(p is None for p in panels):
        return None
    first = panels[0]
    if (any(p['source'] is not first['source'] or p['shape'] != first['shape']
            or p['image'] != first['image'] or p['strides'] != first['strides']
            or p['value'].type != first['value'].type for p in panels)
            or any(p['members'][0].parent is not op.parent for p in panels)):
        return None
    coordinates = [p['offsets'][2:] for p in panels]
    rows = sorted(set(y for y,x in coordinates))
    columns = sorted(set(x for y,x in coordinates))
    if coordinates != [(y,x) for y in rows for x in columns]:
        return None
    dy = rows[1]-rows[0] if len(rows)>1 else 1
    dx = columns[1]-columns[0] if len(columns)>1 else 1
    if (rows != [rows[0]+i*dy for i in range(len(rows))]
            or columns != [columns[0]+i*dx for i in range(len(columns))]):
        return None
    m,c = first['shape']
    if (_static_tensor(op.results[0]) != (m,len(panels)*c)
            or op.results[0].type.element_type != first['value'].type.element_type):
        return None
    return panels, rows, columns, dy, dx


def prepare_source_im2col(module):
    from math import prod
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DictionaryAttr
    from xdsl.dialects.linalg.attrs import IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineExpr, AffineMap

    module.verify()
    records = []
    for op in tuple(module.walk()):
        grid = _grid(op)
        if grid is None:
            continue
        panels, rows, columns, dilation_h, dilation_w = grid
        p = panels[0]
        batch, channels, height, width = p['image']
        output_row, output_column = (AffineExpr.dimension(i) for i in range(2))
        kernel_index = output_column // channels
        indices = (output_row // (height*width), output_column % channels,
            ((output_row // width) % height)*p['strides'][2] + rows[0]
                + (kernel_index // len(columns))*dilation_h,
            (output_row % width)*p['strides'][3] + columns[0]
                + (kernel_index % len(columns))*dilation_w)
        destination = tensor.EmptyOp([], op.results[0].type)
        body = Block(arg_types=[op.results[0].type.element_type]*2)
        body.add_op(YieldOp(body.args[0]))
        replacement = GenericOp([p['source']], [destination.results[0]], Region(body),
            [AffineMapAttr(AffineMap(2,0,indices)), AffineMapAttr(AffineMap.identity(2))],
            [IteratorTypeAttr.parallel()]*2, result_types=[op.results[0].type])
        replacement.attributes.update(op.attributes)
        members = {member for panel in panels for member in panel['members']}
        metadata = [DictionaryAttr({k:v for k,v in member.attributes.items() if k.startswith('prov.')})
                    for member in op.parent.ops if member in members]
        if any(metadata):
            replacement.attributes['prov.composed_panel_sources'] = ArrayAttr(metadata)
        block = op.parent
        block.insert_op_before(destination, op)
        block.insert_op_before(replacement, op)
        op.results[0].replace_all_uses_with(replacement.results[0])
        block.erase_op(op)
        removed = []
        for member in reversed(tuple(block.ops)):
            if member in members and all(not result.uses for result in member.results):
                removed.append(dict(operation=member.name,
                    shape=list(member.results[0].type.get_shape())))
                block.erase_op(member)
        records.append(dict(source_shape=list(p['source'].type.get_shape()),
            output_shape=list(replacement.results[0].type.get_shape()), kernel_grid=[len(rows),len(columns)],
            kernel_origin=[rows[0],columns[0]], kernel_dilation=[dilation_h,dilation_w],
            spatial_stride=list(p['strides'][2:]), input_indexing_map=str(replacement.indexing_maps.data[0]),
            original_panel_count=len(panels), copied_elements=prod(replacement.results[0].type.get_shape()),
            removed_operations=removed))
    module.verify()
    return dict(schema='submission.source_im2col.v1', transformed_gathers=len(records), gathers=records,
        semantics='exact original regular slice grid/permutation/reshape/concatenation addresses',
        padding='original padded tensor retained unchanged', public_abi='unchanged')


def rewrite_text(source):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    module = parse_mlir_text(source)
    report = prepare_source_im2col(module)
    return (text(module,generic=True) if report['transformed_gathers'] else source), report
