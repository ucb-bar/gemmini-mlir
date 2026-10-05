"""Exhaustively compare a scalar Q/DQ residual with primitive scaled-load addition.

Both operands are signed int8, so the complete value domain is only 65,536
pairs. A proof is exact for arbitrary runtime tensors with the matched scalar
qparams. A failed proof retains the source arithmetic and records a witness.
"""
import math
import numpy as np
from xdsl.ir import Operation
from xdsl.ir.affine import AffineMap
from xdsl.dialects.builtin import TensorType
from .captured_requant import scalar
from .golden_requant import f32


def op_name(op):
    return op.attributes['op_name__'].data if op.name == 'builtin.unregistered' else op.name


def prove(lhs_scale, rhs_scale, output_scale, *, lhs_load, rhs_load, readout, relu=False):
    scales = tuple(f32(s) for s in (lhs_scale, rhs_scale, output_scale, lhs_load, rhs_load, readout))
    if any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('positive finite f32 qparams required')
    lhs_scale, rhs_scale, output_scale, lhs_load, rhs_load, readout = scales
    a = np.arange(-128, 128, dtype=np.float32)[:, None]
    b = np.arange(-128, 128, dtype=np.float32)[None, :]
    low = 0 if relu else -128
    source = np.add(a * np.float32(lhs_scale), b * np.float32(rhs_scale), dtype=np.float32)
    if relu:
        source = np.maximum(source, np.float32(0))
    source = np.clip(np.rint(source * np.float32(f32(1.0 / output_scale))), low, 127).astype(np.int16)
    load_a = np.clip(np.rint(a * np.float32(lhs_load)), -128, 127).astype(np.int16)
    load_b = np.clip(np.rint(b * np.float32(rhs_load)), -128, 127).astype(np.int16)
    target = np.clip(np.rint((load_a + load_b).astype(np.float32) * np.float32(readout)), low, 127).astype(np.int16)
    delta = np.abs(source - target)
    mismatch = delta != 0
    report = dict(
        proof='exhaustive signed-int8 operand pair comparison with original f32 ordering',
        pairs=65536, exact=not bool(mismatch.any()), mismatched_pairs=int(mismatch.sum()),
        max_output_lsb_error=int(delta.max()),
        source=dict(lhs_scale=lhs_scale, rhs_scale=rhs_scale, output_scale=output_scale, relu=relu),
        primitive=dict(lhs_load=lhs_load, rhs_load=rhs_load, readout=readout),
    )
    if mismatch.any():
        i, j = np.argwhere(mismatch)[0]
        report['witness'] = dict(lhs=int(i)-128, rhs=int(j)-128, source=int(source[i,j]), target=int(target[i,j]))
    return report


def _pointwise(op, scalar_name, arity):
    if not isinstance(op, Operation) or op_name(op) != 'linalg.generic':
        raise ValueError('expected pointwise region')
    if len(op.operands) != arity+1 or len(op.results) != 1 or len(op.regions) != 1:
        raise ValueError('noncanonical region arity')
    shape = op.results[0].type.get_shape()
    identity = AffineMap.identity(len(shape))
    maps = op.properties.get('indexing_maps')
    if maps is None or len(maps.data) != arity+1 or any(m.data != identity for m in maps.data):
        raise ValueError('nonidentity pointwise maps')
    if len(op.properties['iterator_types'].data) != len(shape) or any(it.data.value != 'parallel' for it in op.properties['iterator_types'].data):
        raise ValueError('pointwise reduction is unsupported')
    if any(v.type != op.results[0].type for v in op.operands):
        raise ValueError('pointwise tensor types differ')
    block = op.regions[0].block
    if len(block.args) != arity+1:
        raise ValueError('pointwise scalar arity differs')
    ops = [o for o in block.ops if o.name != 'arith.constant']
    if [o.name for o in ops] != [scalar_name, 'linalg.yield'] or list(ops[1].operands) != list(ops[0].results):
        raise ValueError('noncanonical scalar body')
    if str(ops[0].properties.get('fastmath', '#arith.fastmath<none>')) != '#arith.fastmath<none>':
        raise ValueError('fastmath is unsupported')
    if scalar_name == 'arith.addf' and set(ops[0].operands) != set(block.args[:arity]):
        raise ValueError('add scalar operands differ')
    return ops[0], block


def _qdq(op, *, input_elem, output_elem):
    if len(op.operands) != 3 or len(op.results) != 1:
        raise ValueError('noncanonical Q/DQ arity')
    inp, out = op.operands[0].type, op.results[0].type
    if not isinstance(inp, TensorType) or not isinstance(out, TensorType):
        raise ValueError('Q/DQ requires tensors')
    if (str(inp.get_element_type()) != input_elem or str(out.get_element_type()) != output_elem
            or inp.get_shape() != out.get_shape() or any(d <= 0 for d in inp.get_shape())):
        raise ValueError('Q/DQ tensor shapes or types differ')
    if scalar(op.operands[2]) != 0:
        raise ValueError('requires symmetric Q/DQ')
    if any(k not in op.properties for k in ('quant_min', 'quant_max')) or [op.properties[k].value.data for k in ('quant_min', 'quant_max')] != [-128,127]:
        raise ValueError('noncanonical signed int8 clamp')
    scale = f32(scalar(op.operands[1]))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('positive finite f32 qparams required')
    return scale


def match(quantize):
    if op_name(quantize) != 'quant_ext.quantize_per_tensor':
        raise ValueError('not a scalar quantize')
    output_scale = _qdq(quantize, input_elem='f32', output_elem='i8')
    value = quantize.operands[0]
    relu = False
    owner = value.owner
    if isinstance(owner, Operation) and op_name(owner) == 'linalg.generic':
        try:
            maximum, block = _pointwise(owner, 'arith.maximumf', 1)
        except ValueError:
            pass
        else:
            constants = [v for v in maximum.operands if v is not block.args[0]]
            if len(constants) != 1 or maximum.operands.count(block.args[0]) != 1:
                raise ValueError('noncanonical ReLU')
            constant = constants[0].owner
            if not isinstance(constant, Operation) or constant.name != 'arith.constant' or constant.properties['value'].value.data != 0:
                raise ValueError('ReLU threshold is not zero')
            value = owner.operands[0]
            relu = True
    addition = value.owner
    _pointwise(addition, 'arith.addf', 2)
    scales=[]
    for operand in addition.operands[:2]:
        dq = operand.owner
        if not isinstance(dq, Operation) or op_name(dq) != 'quant_ext.dequantize_per_tensor':
            raise ValueError('both residual operands need direct scalar dequantize producers')
        scales.append(_qdq(dq, input_elem='i8', output_elem='f32'))
    if str(value.type.get_element_type()) != 'f32':
        raise ValueError('requires f32 source addition')
    return dict(lhs_scale=scales[0], rhs_scale=scales[1], output_scale=output_scale, relu=relu)


def census(source):
    from .frontend.parse import parse_module
    module = parse_module(source)
    accepted=[]
    refused=[]
    for op in module.walk():
        if op_name(op) != 'quant_ext.quantize_per_tensor':
            continue
        try:
            qparams = match(op)
        except ValueError:
            continue
        ratios = [f32(qparams[k] / qparams['output_scale']) for k in ('lhs_scale','rhs_scale')]
        factor = max(1.0, *ratios)
        proof = prove(**qparams, lhs_load=f32(ratios[0]/factor), rhs_load=f32(ratios[1]/factor), readout=factor)
        proof['region'] = getattr(op.attributes.get('prov.region_id'), 'data', '')
        proof['shape'] = list(op.results[0].type.get_shape())
        (accepted if proof['exact'] else refused).append(proof)
    return dict(accepted=accepted, refused=refused, scope='Exact pair-domain proof only; no graph rewrite, runtime measurements or release of approximate arithmetic.')
