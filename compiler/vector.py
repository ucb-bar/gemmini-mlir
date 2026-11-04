"""Typed linalg scalar DAG matching for Gemmini accumulator DMA operations.

Patterns use scalar SSA, operand dtypes, indexing and readout proofs. Provenance
labels never select arithmetic. Floating ABI seams are left to the host compiler.
"""
from __future__ import annotations
from math import prod
import re
try:
    from . import target
    from .elementwise import _geometry, _tiles, _address, lower_movement, lower_bias_add
except ImportError:
    import target
    from elementwise import _geometry, _tiles, _address, lower_movement, lower_bias_add
from merlin.llvmlower.requantization import prove_scale


def _map_dims(text):
    m = re.fullmatch(r'affine_map<\((.*?)\) -> \((.*?)\)>', text.strip())
    if not m:
        raise ValueError('device vector path requires explicit projected indexing')
    domain = tuple(x.strip() for x in m[1].split(',') if x.strip())
    result = tuple(x.strip() for x in m[2].split(',') if x.strip())
    return domain, result


def _expression(record):
    body = record.get('scalar_body')
    if not body or body.get('effects') != [] or len(body.get('yields', ())) != 1:
        raise ValueError('device vector path requires one pure scalar result')
    values = {f'arg:{i}': ('arg', i, ty) for i, ty in enumerate(body['arguments'])}
    for i, capture in enumerate(body.get('captures', ())):
        if 'const_value' not in capture:
            raise ValueError('device scalar capture is not an immutable scalar constant')
        values[f'capture:{i}'] = ('const', capture['const_value'], capture['dtype'])
    for i, op in enumerate(body['operations']):
        if op.get('effects') != [] or op.get('regions') or len(op['results']) != 1:
            raise ValueError('device scalar instruction must be pure with one result')
        attrs = op.get('attributes', {})
        if attrs.get('fastmath', '#arith.fastmath<none>') != '#arith.fastmath<none>':
            raise ValueError('device readout does not implement relaxed floating arithmetic')
        if attrs.get('overflowFlags', '#arith.overflow<none>') != '#arith.overflow<none>':
            raise ValueError('device accumulator addition requires modular integer overflow')
        if op['op'] == 'arith.constant':
            literal = attrs['value'].split(':', 1)[0].strip()
            value = float(literal) if op['results'][0] == 'f32' else int(literal)
            expression = ('const', value, op['results'][0])
        else:
            expression = (op['op'], tuple(values[x] for x in op['operands']), op['results'][0])
        values[f'op:{i}:0'] = expression
    return values[body['yields'][0]]


def _unary(expr, kind):
    if expr[0] != kind or len(expr[1]) != 1:
        raise ValueError(f'device readout expects {kind}')
    return expr[1][0]


def _binary_const(expr, kind):
    if expr[0] != kind or len(expr[1]) != 2:
        raise ValueError(f'device readout expects {kind} with scalar constant')
    x, y = expr[1]
    if x[0] == 'const': return y, x[1]
    if y[0] == 'const': return x, y[1]
    raise ValueError('readout clamp/scale is not an immutable scalar constant')


def _readout(expr):
    """Match exact f32 scale(s), half-even round, clamp, signed-i8 conversion."""
    value = _unary(expr, 'arith.fptosi')
    value, high = _binary_const(value, 'arith.minimumf')
    value, low = _binary_const(value, 'arith.maximumf')
    if high != 127. or low not in (-128., 0.):
        raise ValueError('readout requires signed i8 saturation (optionally ReLU)')
    value = _unary(value, 'math.roundeven')
    scales = []
    while value[0] == 'arith.mulf':
        value, scale = _binary_const(value, 'arith.mulf')
        scales.insert(0, scale)
    value = _unary(value, 'arith.sitofp')
    proof = prove_scale(scales or [1.], -(1 << 31), (1 << 31)-1)
    return value, proof['scale'], low == 0.


def lower_readout(b, src, out, shape, scale=1., relu=False, *, column_scales=None,
                  column_scale_pointer=None, bias=None, bias_is_column=True):
    """Standalone i32 accumulator readout; static or runtime per-column scales.

    Runtime scale loads only move f32 bits into the configuration register. The
    target performs every conversion, multiplication, rounding and activation.
    """
    rows, cols = _geometry(shape)
    if column_scales is not None and (column_scale_pointer is not None or len(column_scales) != cols):
        raise ValueError('readout needs exactly one scale source with one value per column')
    target.cfg_ld(b, cols * 4, stream=0, shrunk=False)
    if bias is not None:
        target.cfg_ld(b, 0 if bias_is_column else cols*4, stream=1, shrunk=False)

    def emit(row, col, nr, nc):
        target.mvin(b, _address(b, src, row, col, cols, 4), target.ACC, nr, nc)
        b.fence()
        if bias is not None:
            bp = b.ptr(bias, b.mul_i(col,b.const(4))) if bias_is_column else _address(b,bias,row,col,cols,4)
            target.mvin(b, bp, target.ACC | target.ACCUMULATE, nr, nc, stream=1)
            b.fence()
        target.mvout(b, _address(b, out, row, col, cols, 1), target.ACC, nr, nc)
        b.fence()

    if column_scale_pointer is not None:
        # One column at a time ensures a distinct runtime multiplier is applied
        # only to that column, while retaining up to DIM rows per DMA command.
        def column(col):
            bits = b.load_i64(column_scale_pointer, col, 'i32')
            word = b.or_i(b.const(cols), b.shl_i(bits, 32))
            b.rocc(0, 2 | (int(relu)<<2), word)
            full = rows // target.DIM * target.DIM
            if full: b.for_loop(0,full,target.DIM,lambda row: emit(row,col,target.DIM,1))
            if full != rows: emit(b.const(full),col,rows-full,1)
        b.for_loop(0,cols,1,column)
    elif column_scales is not None:
        from merlin.llvmlower.integer_readout import partition_column_scales
        partition = partition_column_scales(column_scales, cols, relu=relu)
        for region in partition['regions']:
            start, width = region['first_column'], region['columns']
            # The proof's source scale for this equal-scale column run.
            target.cfg_st(b, cols, scale=column_scales[start], activation=int(relu))
            _tiles(b, rows, width, lambda row,col,nr,nc:
                   emit(row,b.add_i(col,b.const(start)),nr,nc))
    else:
        target.cfg_st(b, cols, scale=scale, activation=int(relu))
        _tiles(b,rows,cols,emit)


def lower_vector_map(b, cb, command, pointers):
    """Return device evidence, or raise ValueError before emission if unsupported."""
    rec = command['attributes']['semantic_record']
    args, tensors = command['operands'], cb['tensors']
    if rec.get('reduction_dims') or any(x != 'parallel' for x in rec.get('iterator_types', ())):
        raise ValueError('vector device path accepts only parallel scalar maps')
    if len(rec['results']) != 1:
        raise ValueError('vector device path requires one result')
    dst = args['dst0']; spec = tensors[dst]; shape = spec['shape']
    dims = tuple(f'd{i}' for i in range(len(shape)))
    maps = [_map_dims(x) for x in rec['indexing_maps']]
    if any(d != dims for d,_ in maps) or maps[-1][1] != dims:
        raise ValueError('vector result indexing must preserve row-major order')
    expr = _expression(rec)
    input_names = [args[f'in{i}'] for i in range(len(rec['ins']))]
    def scalar_input(value, dtype):
        if value[0] != 'arg' or value[2] != dtype or value[1] >= len(input_names):
            raise ValueError('device arithmetic source must be a typed tensor input')
        name = input_names[value[1]]
        if tensors[name]['dtype'] != dtype:
            raise ValueError('source scalar type differs from tensor type')
        return name, maps[value[1]][1]
    def check_full(name, mapping):
        if tensors[name]['shape'] != shape or mapping != dims:
            raise ValueError('source is not an identity-indexed result-shaped tensor')

    if expr[0] == 'arg':
        name, mapping = scalar_input(expr, spec['dtype']); check_full(name,mapping)
        lower_movement(b,pointers[name],pointers[dst],shape,spec['dtype'],spec['dtype'])
        return {'form':'accumulator_movement'}

    readout_scale, relu = None, False
    if spec['dtype'] == 'i8':
        expr, readout_scale, relu = _readout(expr)
    elif spec['dtype'] != 'i32':
        raise ValueError('Gemmini has no floating-output readout')

    if expr[0] == 'arith.addi' and expr[2] == 'i32':
        left, right = [scalar_input(x,'i32') for x in expr[1]]
        if left[1] != dims and right[1] == dims: left,right = right,left
        check_full(*left)
        bias_is_column = right[1] == (dims[-1],) and tensors[right[0]]['shape'] == [shape[-1]]
        if not bias_is_column: check_full(*right)
        if readout_scale is None and bias_is_column:
            lower_bias_add(b,pointers[left[0]],pointers[right[0]],pointers[dst],shape)
        elif readout_scale is not None:
            lower_readout(b,pointers[left[0]],pointers[dst],shape,readout_scale,relu,
                          bias=pointers[right[0]],bias_is_column=bias_is_column)
        else:
            rows,cols = _geometry(shape)
            target.cfg_ld(b,cols*4,stream=0,shrunk=False)
            target.cfg_ld(b,cols*4,stream=1,shrunk=False)
            target.cfg_st(b,cols*4)
            def emit(row,col,nr,nc):
                for stream,(name,_) in enumerate((left,right)):
                    target.mvin(b,_address(b,pointers[name],row,col,cols,4),
                                target.ACC | (target.ACCUMULATE if stream else 0),nr,nc,stream=stream)
                    b.fence()
                target.mvout(b,_address(b,pointers[dst],row,col,cols,4),target.ACC|target.FULL,nr,nc)
                b.fence()
            _tiles(b,rows,cols,emit)
        return {'form':'accumulator_add','bias_is_column':bias_is_column,'readout_scale':readout_scale,'relu':relu}
    if readout_scale is not None:
        name,mapping = scalar_input(expr,'i32'); check_full(name,mapping)
        lower_readout(b,pointers[name],pointers[dst],shape,readout_scale,relu)
        return {'form':'accumulator_readout','scale':readout_scale,'relu':relu}
    raise ValueError('scalar DAG has no exact selected Gemmini vector form')


def lower_generic(builder, op, pointers):
    """Raw xDSL graph adapter; return True only after an exact device realization.

    A False result leaves the graph coordinator responsible for placement: a
    floating result is an unsupported device seam, whereas a supported integer
    operation with an unmatched layout is a remaining compiler task.
    """
    from merlin.targetgen.contract import linalg_iface as li
    ins, outs = list(op.inputs), list(op.outputs)
    if len(op.results) != 1:
        return False
    all_values = [*ins, *outs, *op.results]
    names = {value: f'vector{i}' for i,value in enumerate(all_values)}
    def operand(value):
        result = {'shape':li._shape(value.type), 'dtype':li._dtype(value.type)}
        constant = li._scalar_const(value)
        if constant is not None: result['const_value'] = constant
        return result
    rec = {'ins':[operand(v) for v in ins], 'outs':[operand(v) for v in outs],
           'results':[operand(v) for v in op.results],
           'indexing_maps':li._indexing_maps(op, ins, outs),
           'iterator_types':li._iterator_types(op, ins),
           'reduction_dims':li._reduction_dims(op),
           'scalar_body':li._scalar_body(op, operand)}
    cb = {'tensors':{names[value]:operand(value) for value in all_values}}
    command = {'opcode':'VECTOR_MAP', 'attributes':{'semantic_record':rec},
               'operands':{**{f'in{i}':names[value] for i,value in enumerate(ins)},
                           **{f'init{i}':names[value] for i,value in enumerate(outs)},
                           'dst0':names[op.results[0]]}}
    ptrs = {names[v]:pointers[v] for v in [*ins,*op.results]}
    try:
        lower_vector_map(builder,cb,command,ptrs)
    except ValueError:
        return False
    return True
