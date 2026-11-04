"""Exact source epilogue admission and concrete dense readout realization.

The original SSA remains the semantic artifact. Admitted groups are a partition
rewrite: their virtual stages become one device readout, with original direct
operation ordinals retained for source ownership. Unknown runtime numbers refuse.
"""
from dataclasses import dataclass
from pathlib import Path
from math import prod


@dataclass(frozen=True)
class SourceEpiloguePreparation:
    module: object
    groups: tuple[dict, ...]
    refusals: tuple[dict, ...]


def enabled():
    from merlin.compile.tuning_knobs import knob
    return knob('source_epilogue_fusion', default='off', values=('off', 'exact'), root=Path(__file__).parent) == 'exact'


def declared_path():
    import yaml
    from merlin.targetgen.output_path import validate
    contract = yaml.safe_load((Path(__file__).parent / 'contracts' / 'target_contract.yaml').read_text())
    return validate(contract['output_epilogue']['output_path'])


def _binary32_decision(chain, path, extents, original):
    """Use the library's dedicated IEEE binary32 theorem for its exact grammar.

    This is independent of the optional extensible format registry: no format
    aliases or proof internals are changed. Unsupported grammars still refuse.
    """
    import numpy as np
    from merlin.llvmlower.requantization import f32, synthesize_bias
    from merlin.xdsl_dialects.lowering.output_fusion import Decision
    kinds = [s.kind for s in chain.stages]
    if chain.unread or chain.float_dtype != 'f32' or not kinds or kinds[0] != 'convert' or kinds[-1] != 'quantize':
        raise ValueError('binary32 theorem requires a closed explicit f32 conversion/quantizer chain')
    expected = [('activation', 'relu'), ('convert', 'f32'), ('scale', 'f32'), ('requant', 'i8')]
    stages = [s for s in path['readout'] if s['kind'] != 'pool']
    observed = [(s['kind'], s.get('function') if s['kind']=='activation' else s.get('to', s.get('dtype'))) for s in stages]
    if (observed != expected or path['accumulator']['dtype'] != 'i32' or not path['exits']['requant']
            or any(s.get('rounding') != 'half_even' for s in stages[1:])
            or stages[-1].get('clamp') != [-128,127]
            or stages[2].get('granularity') != 'per_command'):
        raise ValueError('selected readout is outside dedicated binary32 theorem')
    scales, biases, relu, phase = [], np.zeros(chain.channels), False, 'scale'
    for stage in chain.stages[1:-1]:
        if stage.kind == 'scale' and phase == 'scale':
            numbers = np.asarray(stage.numbers).reshape(-1)
            if numbers.size != 1:
                raise ValueError('dedicated theorem requires scalar source scales')
            scales.append(float(numbers[0]))
        elif stage.kind == 'bias' and phase == 'scale':
            numbers = np.asarray(stage.numbers).reshape(-1)
            if numbers.size not in (1, chain.channels):
                raise ValueError('source bias has incompatible channels')
            biases = np.broadcast_to(numbers, (chain.channels,))
            phase = 'activation'
        elif stage.kind == 'relu' and phase in ('scale', 'activation'):
            relu, phase = True, 'quantize'
        else:
            raise ValueError('source stage order is outside dedicated binary32 theorem')
    quant = chain.stages[-1]
    if quant.detail != {'zero_point':0,'min':-128,'max':127,'dtype':'i8'}:
        raise ValueError('dedicated theorem requires zero-point-zero signed-i8 RNE clamp')
    qscale = np.asarray(quant.numbers).reshape(-1)
    if qscale.size != 1:
        raise ValueError('dedicated theorem requires scalar output scale')
    extent = extents.get(id(chain.root))
    if not extent or extent[1] is None:
        raise ValueError('contraction reduction extent is unknown')
    bound = int(extent[1]) * 128 * 128
    if bound >= (1 << 31):
        raise ValueError('source i32 product can overflow the exact theorem domain')
    proof = synthesize_bias(scales or [1.0], biases, f32(1.0 / float(qscale[0])), -bound, bound, relu=relu)
    if proof['accepted_channels'] != chain.channels:
        raise ValueError('no exact integer readout bias for every channel: ' + str(proof['refused_channels']))
    proof.update(offsets=proof['integer_bias'], device_scale=[proof['scale']] * chain.channels,
                 accumulator_bound=[-bound,bound], theorem='merlin.llvmlower.requantization.synthesize_bias')
    return Decision('requant', list(chain.ops), kinds, original.refusals, proof)


def prepare_source_epilogues(module, *, entry='forward'):
    """Prove private i8 product→f32 stages→signed-i8 quantize groups.

    ``groups`` carries actual operation objects as well as original ordinals;
    the host splitter routes those members together and the tensor lowerer
    consumes ``readout`` to replace the complete chain by its dense COMMIT.
    Unsupported chains keep their incumbent source operations and a reason.
    """
    from merlin.xdsl_dialects.lowering import output_fusion as OF, compute_groups as CG
    from merlin.compile.analysis.contraction_shapes import contraction_extents
    from merlin.targetgen.contract import linalg_iface as LI
    from merlin.llvmlower.requantization import prove_scale
    owners = [op for op in module.walk() if op.name == 'func.func' and LI._fn_name(op) == entry and op.body.blocks]
    if len(owners) != 1:
        raise ValueError('epilogue fusion requires one identified tensor entry')
    operations = list(owners[0].body.block.ops)
    ordinal = {op: i for i, op in enumerate(operations)}
    values, extents, path = OF.Values(), contraction_extents(module), declared_path()
    groups, refusals = [], []
    def literal(value):
        op = value.owner
        if getattr(op, 'name', '') in ('tensor.splat', 'linalg.fill'):
            return literal(op.operands[0])
        if getattr(op, 'name', '') == 'arith.constant':
            attr = {**op.attributes, **op.properties}.get('value')
            return getattr(getattr(attr, 'value', None), 'data', None)
        return None
    for root in operations:
        stage = CG.classify(root)
        if stage is None or stage.kind != CG.CONTRACTION:
            continue
        inputs, outputs = LI._ins_outs(root)
        if (len(root.results[0].type.get_shape()) != 2 or root.name == 'linalg.matmul_transpose_b'
                or len(inputs) != 2 or len(outputs) != 1
                or any(str(v.type.element_type) != 'i8' for v in inputs)
                or str(root.results[0].type.element_type) != 'i32'):
            continue
        if root.name == 'linalg.generic':
            maps = getattr(root, 'indexing_maps', None)
            if maps is None:
                continue
            from xdsl.ir.affine import AffineDimExpr
            dims = [tuple(e.position for e in m.data.results) for m in maps
                    if all(isinstance(e, AffineDimExpr) for e in m.data.results)]
            if len(dims) != 3 or len(dims[0]) != 2 or len(dims[1]) != 2 or dims[0][-1] != dims[1][-2]:
                continue
        chain = OF.source_chain(root, values)
        # Recognize the already decomposed quantizer by its actual scalar
        # program, rather than its provenance spelling. OF's source reader
        # handles the preceding cast/scale/bias/activation stages.
        if chain.unread is None:
            current = chain.ops[-1].results[0] if chain.ops else root.results[0]
            uses = list(current.uses)
            if len(uses) == 1 and uses[0].operation.name == 'linalg.generic':
                from linalg_lowering import _body, _maps
                activation = uses[0].operation
                ai, ao = LI._ins_outs(activation)
                identity = tuple(range(len(current.type.get_shape())))
                try:
                    activation_expr = _body(activation)
                except ValueError:
                    activation_expr = None
                if (len(ai) == 1 and ai[0] is current and len(ao) == 1
                        and _maps(activation) == [identity, identity]
                        and activation_expr in (
                            ('arith.maximumf', ('arg', 0), ('const', 0.0)),
                            ('arith.maximumf', ('const', 0.0), ('arg', 0)))):
                    chain.stages.append(OF.Stage('relu', activation))
                    chain.ops.append(activation)
                    current = activation.results[0]
                    uses = list(current.uses)
            if len(uses) == 1 and uses[0].operation.name == 'linalg.generic':
                quant = uses[0].operation
                qi, qo = LI._ins_outs(quant)
                if len(qi) == 3 and len(qo) == 1 and qi[0] is current:
                    from linalg_lowering import _body, _maps, _constant
                    arg = ('arg', 0)
                    rounded = ('math.roundeven', ('arith.mulf', arg, ('arg', 1)))
                    shifted = ('arith.addf', rounded, ('arith.sitofp', ('arg', 2)))
                    expected = ('arith.fptosi', ('arith.minimumf', ('arith.maximumf', shifted, ('const', -128.0)), ('const', 127.0)))
                    try:
                        matches = _body(quant) == expected
                    except ValueError:
                        matches = False
                    maps = _maps(quant)
                    rank = len(current.type.get_shape())
                    identity = tuple(range(rank))
                    reciprocal = values.array(qi[1])
                    if reciprocal is None and getattr(qi[1].owner, 'name', '') == 'linalg.generic':
                        reciprocal_op = qi[1].owner
                        ri, ro = LI._ins_outs(reciprocal_op)
                        try:
                            reciprocal_expr = _body(reciprocal_op)
                        except ValueError:
                            reciprocal_expr = None
                        if len(ri) == 1 and reciprocal_expr == ('arith.divf', ('const', 1.0), ('arg', 0)):
                            denominator = values.array(ri[0])
                            if denominator is not None and denominator.size == 1:
                                from merlin.llvmlower.requantization import f32
                                reciprocal = f32(1.0 / f32(float(denominator.reshape(-1)[0])))
                    if matches and maps == [identity, (), (), identity] and _constant(qi[2]) == 0 and reciprocal is not None:
                        import numpy as np
                        from merlin.llvmlower.requantization import f32
                        numbers = np.asarray(reciprocal).reshape(-1)
                        if numbers.size == 1 and float(numbers[0]) > 0:
                            inverse = f32(float(numbers[0]))
                            scale = f32(1.0 / inverse)
                            if f32(1.0 / scale) == inverse:
                                chain.stages.append(OF.Stage('quantize', quant, scale,
                                    {'zero_point': 0, 'min': -128, 'max': 127, 'dtype': 'i8'}))
                                chain.ops.append(quant)
        decision = OF.decide(chain, path, values, extents)
        theorem_refusal = None
        if decision.cut != 'requant':
            try:
                decision = _binary32_decision(chain, path, extents, decision)
            except ValueError as exc:
                theorem_refusal = str(exc)
        reason = None
        if literal(outputs[0]) != 0:
            reason = 'source initializer is not a literal zero; proof bound excludes arbitrary initializer'
        elif not chain.stages or chain.stages[0].kind != 'convert':
            reason = 'adapter requires explicit integer-to-f32 conversion before floating epilogue'
        elif decision.cut != 'requant':
            reason = '; '.join(r['code'] + ': ' + r['reason'] for r in decision.refusals) + '; binary32 theorem: ' + str(theorem_refusal)
        elif any(op.name == 'linalg.transpose' for op in decision.members):
            reason = 'readout cannot permute an externally distinct layout'
        elif not decision.members or any(op not in ordinal for op in decision.members):
            reason = 'epilogue is not a direct entry chain'
        else:
            scales = decision.proof.get('device_scale', [])
            offsets = decision.proof.get('offsets', [])
            final = decision.members[-1].results[0]
            if (not scales or any(float(s) != float(scales[0]) for s in scales)
                    or str(final.type.element_type) != 'i8'
                    or prod(final.type.get_shape()) != prod(root.results[0].type.get_shape())):
                reason = 'adapter requires one scalar scale and a contiguous equal-element signed-i8 result'
            elif len(offsets) != chain.channels:
                reason = 'proof did not provide one exact accumulator offset per channel'
            else:
                # Independently check scalar scale realization across the full
                # i32 domain; the library decision proves the complete source
                # chain including bias, activation order, and quantizer clamp.
                scale_proof = prove_scale([float(scales[0])], relu='relu' in decision.stages)
                attrs = {'output_dtype': 'i8', 'epilogue': ['acc_scale'], 'acc_scale': float(scales[0])}
                if any(offsets):
                    attrs['epilogue'].insert(0, 'bias_add')
                    attrs['source_epilogue_bias_values'] = list(offsets)
                if 'relu' in decision.stages:
                    attrs['epilogue'].append('relu')
                groups.append({'root': root, 'members': tuple(decision.members),
                               'root_ordinal': ordinal[root], 'source_op_indices': [ordinal[root], *[ordinal[o] for o in decision.members]],
                               'result': final, 'readout': attrs, 'proof': decision.proof,
                               'scale_proof': scale_proof, 'cut': decision.cut})
        if reason:
            refusals.append({'root_ordinal': ordinal[root], 'reason': reason, 'library_refusals': decision.refusals})
    return SourceEpiloguePreparation(module, tuple(groups), tuple(refusals))


def fuse_integer_bias_readout(command, bias, destination, *, private, bias_dtype, output_dtype):
    """Mutate a real full-width contraction COMMIT into exact i32 bias preload.

    Addition in the declared i32 accumulator is modular, as is source arith.addi;
    changing its position relative to the integer sum cannot change any bit.
    """
    path = declared_path()
    preload = next((s for s in path.get('accumulator_stages', ()) if s['kind'] == 'preload'), None)
    if not private:
        return False, 'contraction intermediate is externally observed'
    if (preload is None or preload.get('dtype') != 'i32' or preload.get('rounding') != 'none'
            or not path.get('exits', {}).get('accumulator')):
        return False, 'selected output path does not declare exact i32 preload and full-width exit'
    attrs = command['attributes']
    if bias_dtype != 'i32' or output_dtype != 'i32' or attrs.get('output_dtype') != 'i32' or attrs.get('epilogue'):
        return False, 'bias fusion requires bare i32 product, i32 bias, and full-width i32 result'
    command['operands'].update(bias=bias, dst=destination)
    attrs['epilogue'] = ['bias_add']
    return True, 'exact modular i32 bias preload before source integer reduction'
