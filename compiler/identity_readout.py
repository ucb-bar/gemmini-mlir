"""Exact doubled-integer representation for an identity-scale byte residual.

Optional target realization only: the source command and its scale boundary stay
unchanged. Four consumed loads form 2*(lhs+rhs), and a genuine half-scale readout
writes the original result. This adds two loads per output DMA rectangle.
"""
from functools import lru_cache
from math import prod


def _target_path():
    from source_epilogue import declared_path
    path = declared_path()
    stages = [s for s in path['readout'] if s['kind'] != 'pool']
    if (path['accumulator']['dtype'] != 'i32' or not path['exits'].get('requant')
            or [(s['kind'], s.get('function', s.get('to', s.get('dtype')))) for s in stages]
                != [('activation','relu'), ('convert','f32'), ('scale','f32'), ('requant','i8')]
            or any(s.get('rounding') != 'half_even' for s in stages[1:])
            or stages[0].get('domain') != 'accumulator'
            or stages[-1].get('clamp') != [-128,127]):
        raise ValueError('selected readout does not declare exact f32 RNE signed-i8 narrowing')
    return path


@lru_cache(maxsize=2)
def _prove(relu):
    """Exhaust every source input using Merlin's actual binary32 arithmetic."""
    from merlin.llvmlower.requantization import f32, quantized
    _target_path()
    checked = 0
    for lhs in range(-128,128):
        for rhs in range(-128,128):
            source_sum = f32(f32(lhs * f32(1.0)) + f32(rhs * f32(1.0)))
            source = max(0 if relu else -128, min(127, round(source_sum)))
            encoded = lhs + rhs + lhs + rhs
            target = quantized(encoded, [0.5], relu=relu)
            if source != target:
                raise ValueError('identity representation changes residual result at ' + str((lhs,rhs)))
            checked += 1
    return {'schema':'exact_identity_operand_sum_v1', 'proof':'exhaustive signed-i8 Cartesian domain using Merlin binary32 arithmetic',
            'pairs_checked':checked, 'max_error_lsb':0,
            'source_scales':[1.0,1.0], 'operand_load_scales':[1.0,1.0],
            'accumulator_representation':'lhs + rhs + lhs + rhs',
            'accumulator_min':-512, 'accumulator_max':508,
            'readout_scale':0.5, 'rounding':'half_even',
            'clamp':[0 if relu else -128,127], 'relu':bool(relu),
            'extra_effects_per_rectangle':{'accumulating_i8_loads':2,'completion_fences':2},
            'scope':'exact source residual boundary; no source tensor or quantization parameter changed'}


def plan(command, tensors):
    """Return a source-derived admission/refusal. No policy is enabled here."""
    refusal = lambda reason: {'admitted':False, 'reason':reason}
    if command.get('opcode') != 'RESIDUAL_ADD':
        return refusal('requires a residual-add source command')
    attrs, ops = command.get('attributes', {}), command.get('operands', {})
    if attrs.get('lhs_scale') != 1.0 or attrs.get('rhs_scale') != 1.0:
        return refusal('requires both original source scales to equal binary32 one')
    bound = attrs.get('bound_lsb')
    if type(bound) is not int or bound < 0:
        return refusal('requires the original nonnegative integer error bound')
    stages = attrs.get('epilogue', [])
    if not isinstance(stages,list) or stages not in ([], ['relu']):
        return refusal('requires an optional single source ReLU after the sum')
    specs = [tensors.get(ops.get(k)) for k in ('lhs','rhs','dst')]
    if any(not isinstance(s,dict) or s.get('dtype') != 'i8' for s in specs):
        return refusal('requires three declared signed-i8 tensors')
    shape = specs[0].get('shape')
    if (not isinstance(shape,list) or any(type(x) is not int or x <= 0 for x in shape)
            or any(s.get('shape') != shape for s in specs[1:])
            or attrs.get('output_dtype', 'i8') != 'i8'):
        return refusal('requires equal positive static shapes and signed-i8 output')
    try:
        proof = _prove('relu' in stages)
    except ValueError as exc:
        return refusal(str(exc))
    rectangles = prod(shape) // 256 + bool(prod(shape) % 256 // 16) + bool(prod(shape) % 16)
    return {'admitted':True, 'proof':dict(proof), 'elements':prod(shape),
            'dma_rectangles':rectangles, 'extra_accumulating_i8_loads':2*rectangles,
            'extra_completion_fences':2*rectangles,
            'performance':'adds consumed DMA work; whole-call timing is not established'}


def emit(builder, command, tensors, pointers):
    """Emit the admitted representation, or return a refusal without effects."""
    admission = plan(command,tensors)
    if not admission['admitted']:
        return admission
    from isa import ACC, ACCUMULATE, DIM
    from elementwise import _tiles
    ops = command['operands']
    relu = admission['proof']['relu']
    builder.config_ld(DIM, 1.0, shrink=True, id=0)
    builder.config_ld(DIM, 1.0, shrink=True, id=1)
    builder.config_ex(activation=int(relu), scale=0.5)
    builder.config_st(DIM, activation=int(relu), scale=0.5)
    def rectangle(offset, cols, rows):
        for key, overwrite, load_id in (('lhs',True,0),('rhs',False,1),('lhs',False,0),('rhs',False,1)):
            builder.mvin(builder.ptr_offset(pointers[ops[key]],offset),
                         ACC if overwrite else ACC | ACCUMULATE,cols,rows,id=load_id)
            builder.fence()
        builder.mvout(builder.ptr_offset(pointers[ops['dst']],offset),ACC,cols,rows)
        builder.fence()
    _tiles(builder,admission['elements'],rectangle)
    builder.fence()
    return admission
