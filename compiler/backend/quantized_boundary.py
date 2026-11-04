"""Typed obligations at captured per-tensor quantization boundaries.

The selected DMA accepts integer payloads, while its narrowing readout consumes
an i32 accumulator. An f32 tensor therefore has no standalone device quantizer.
That fact does not rule out an exact integer-producer/readout composition. This
module checks such compositions against actual scalar SSA and binary32 output
transitions; it neither changes placement nor grants producer/effect ownership.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Operation
from xdsl.ir.affine import AffineMap
from merlin.common.mlir_query import op_name
from merlin.llvmlower.requantization import f32, prove_scale
from merlin.xdsl_dialects.lowering.group_numerics import _constant_of, _int_attr


class QuantizedBoundaryError(ValueError):
    pass


def _type(value):
    typ = value.type
    if not isinstance(typ, TensorType):
        return {'type':str(typ), 'dtype':str(typ), 'shape':None, 'dense':False}
    from xdsl.dialects.builtin import NoneAttr
    return {'type':str(typ), 'dtype':str(typ.element_type), 'shape':list(typ.get_shape()),
            'dense':isinstance(typ.encoding, NoneAttr)}


def _identity(op):
    return {'operation':op_name(op),
            'source_ids':{k:str(v) for k,v in op.attributes.items()
                          if k.startswith(('prov.', 'merlin.source'))}}


def selected_boundary_forms():
    """Forms implemented by this backend's selected integer DMA/readout path."""
    return {'standalone_dma_input_dtypes':['i8', 'i32'], 'accumulator_dtype':'i32',
            'readout_input_dtype':'i32', 'output_dtypes':['i8', 'i32'],
            'f32_tensor_input':False, 'readout_rounding':'half_even',
            'readout_order':['optional_relu_i32', 'convert_f32', 'multiply_f32',
                             'round_half_even', 'saturate_signed_i8'],
            'clamp':[-128,127], 'scale_granularity':'per_command',
            'basis':['backend.isa integer operand/accumulator DMA formats',
                     'backend.elementwise.lower_readout typed i32 input',
                     'selected output_path_v1 convert/scale/requant stages'],
            'standalone_form':'ACCUMULATOR_READOUT',
            'fused_form':'integer producer accumulator narrowing readout'}


def _quantizer(op):
    if op_name(op) != 'quant_ext.quantize_per_tensor':
        raise QuantizedBoundaryError('operation is not the captured per-tensor quantization contract')
    if len(op.operands) != 3 or len(op.results) != 1:
        raise QuantizedBoundaryError('quantizer needs exactly value, scalar scale, scalar zero point and one result')
    value, scale, zero = op.operands
    src, dst, st, zt = map(_type, (value, op.results[0], scale, zero))
    info = {'input':src, 'output':dst, 'scale':{'operand_type':st, 'value':_constant_of(scale)},
            'zero_point':{'operand_type':zt, 'value':_constant_of(zero)},
            'clamp':[_int_attr(op,'quant_min'), _int_attr(op,'quant_max')],
            'rounding':'half_even',
            'rounding_evidence':'selected lower_quant_ext: reciprocal, multiply, math.roundeven, zero-point add, clamp, fptosi',
            'source_order':['reciprocal_in_scale_dtype', 'multiply', 'round_half_even',
                            'zero_point_add', 'clamp', 'fptosi'],
            'source':_identity(op)}
    # Keep unsupported signatures separate from numerical folding refusals.
    if (src['shape'] is None or dst['shape'] is None or src['shape'] != dst['shape']
            or st['shape'] != [] or zt['shape'] != []):
        raise QuantizedBoundaryError('quantizer does not have matching tensor shapes and rank-zero scale/zero point')
    return info


def _cmp_select_relu_slot(op):
    """Prove captured ordered comparison/select ReLU from scalar SSA.

    This recognizes the source's branch topology, including its explicit
    integer-zero-to-f32 conversion. It never treats an arbitrary select or
    predicate as an activation. The enclosing readout proof establishes a
    non-NaN integer-derived domain and preserves quantized zero semantics.
    """
    if op.name!='linalg.generic' or len(op.regions)!=1:return None
    body=op.regions[0].block
    scalar=[o for o in body.ops if o.name!='arith.constant']
    if len(scalar) not in (3,4):return None
    casts=[]
    if len(scalar)==4:
        cast=scalar.pop(0)
        if (cast.name!='arith.sitofp' or len(cast.operands)!=1
                or _constant_of(cast.operands[0])!=0 or str(cast.results[0].type)!='f32'):
            return None
        casts=[cast]
    compare,select,yld=scalar
    if ([o.name for o in scalar]!=['arith.cmpf','arith.select','linalg.yield']
            or list(yld.operands)!=list(select.results)
            or select.operands[0] is not compare.results[0]):return None
    if any(str(o.properties.get('fastmath','#arith.fastmath<none>'))!='#arith.fastmath<none>'
           for o in [*casts,compare,select]):return None
    predicate=getattr(getattr(compare.properties.get('predicate'),'value',None),'data',None)
    if predicate not in (2,3,4,5):return None  # ordered >, >=, <, <=
    for slot,value in enumerate(getattr(op,'inputs',())):
        if _type(value)['shape']!=_type(op.results[0])['shape'] or _type(value)['dtype']!='f32':continue
        data=body.args[slot]
        if data not in compare.operands:continue
        zero=compare.operands[1] if compare.operands[0] is data else compare.operands[0]
        if str(zero.type)!='f32':continue
        literal=zero
        if zero in body.args[:len(op.inputs)]:literal=op.inputs[list(body.args).index(zero)]
        is_zero=(_constant_of(literal)==0 or bool(casts and zero is casts[0].results[0]))
        if not is_zero:continue
        data_is_left=compare.operands[0] is data
        true_keeps_data=(predicate in (2,3))==data_is_left
        branches=[data,zero] if true_keeps_data else [zero,data]
        if list(select.operands[1:])==branches:return slot
    return None


def _pointwise(op, result):
    if (op.name != 'linalg.generic' or len(op.results) != 1 or op.results[0] is not result
            or len(op.outputs) != 1 or len(op.regions) != 1):
        raise QuantizedBoundaryError('integer readout path crosses a noncanonical pointwise or layout operation')
    typ = _type(result)
    shape = typ['shape']
    if shape is None or not shape or any(n <= 0 for n in shape) or not typ['dense']:
        raise QuantizedBoundaryError('readout path needs static non-scalar dense tensor boundaries')
    rank = len(shape)
    identity = AffineMap.identity(rank)
    scalar = AffineMap(rank, 0, ())
    if (len(op.indexing_maps) != len(op.operands)
            or tuple(op.indexing_maps)[-1].data != identity
            or tuple(it.data.value for it in op.iterator_types) != ('parallel',)*rank):
        raise QuantizedBoundaryError('readout path changes element order or reduction axes')
    for value, mapping in zip(op.inputs, op.indexing_maps, strict=False):
        s = _type(value)
        if (s['shape'] == shape and mapping.data == identity and s['dense']):
            continue
        if s['shape'] == [] and mapping.data == scalar:
            continue
        raise QuantizedBoundaryError('readout input has an unproved broadcast or address projection')
    body = op.regions[0].block
    if (len(body.args) != len(op.operands)
            or any(arg.type != value.type.element_type
                   for arg, value in zip(body.args, op.operands, strict=True))
            or _type(op.outputs[0]) != typ):
        raise QuantizedBoundaryError('readout scalar arguments/output do not match their tensor element types')
    ops = [o for o in body.ops if o.name != 'arith.constant']
    if _cmp_select_relu_slot(op) is not None:
        return body,ops[-2]
    if (len(ops) != 2 or ops[-1].name != 'linalg.yield'
            or list(ops[-1].operands) != list(ops[0].results)):
        raise QuantizedBoundaryError('readout scalar body contains additional arithmetic or control')
    fm = ops[0].properties.get('fastmath')
    if fm is not None and str(fm) != '#arith.fastmath<none>':
        raise QuantizedBoundaryError('readout scalar fastmath requires a separate equivalence proof')
    return body, ops[0]


def _shape_view(value):
    """Return the source of a verified dense, contiguous tensor reshape."""
    from math import prod
    owner = value.owner
    if not isinstance(owner, Operation) or owner.name not in ('tensor.expand_shape','tensor.collapse_shape'):
        return None
    if len(owner.results) != 1 or owner.results[0] is not value or len(owner.operands) != 1:
        raise QuantizedBoundaryError('readout reshape needs one static source and result')
    source = owner.operands[0]
    a, b = _type(source), _type(value)
    if (not a['dense'] or not b['dense'] or not a['shape'] or not b['shape']
            or any(n <= 0 for n in (*a['shape'],*b['shape']))
            or a['dtype'] != b['dtype'] or prod(a['shape']) != prod(b['shape'])):
        raise QuantizedBoundaryError('readout reshape does not preserve dense static element order')
    try:
        owner.verify()
    except Exception as exc:
        raise QuantizedBoundaryError('readout reshape has invalid reassociation') from exc
    return source


def _trace(value, limit, *, allow_shape_views=False):
    """Read only identity-mapped i32->f32, positive scalar multiplies, ReLU."""
    scales, operations, relu = [], [], False
    current = value
    for _ in range(limit):
        op = current.owner
        if not isinstance(op, Operation):
            raise QuantizedBoundaryError('floating boundary has no in-graph integer-to-f32 producer')
        if allow_shape_views:
            source = _shape_view(current)
            if source is not None:
                operations.append(op)
                current = source
                continue
        body, scalar = _pointwise(op,current)
        if scalar.name == 'arith.sitofp':
            if (len(op.inputs) != 1 or list(scalar.operands) != [body.args[0]]
                    or _type(op.inputs[0])['dtype'] != 'i32'
                    or _type(current)['dtype'] != 'f32' or str(scalar.results[0].type) != 'f32'):
                raise QuantizedBoundaryError('readout cast is not exact source i32-to-f32 conversion')
            operations.append(op)
            source = op.inputs[0]
            if allow_shape_views:
                while (underlying := _shape_view(source)) is not None:
                    if len(operations) >= limit:
                        raise QuantizedBoundaryError('readout source chain exceeds the explicit analysis budget')
                    operations.append(source.owner)
                    source = underlying
            return source, tuple(reversed(operations)), list(reversed(scales)), relu
        if _type(current)['dtype'] != 'f32' or str(scalar.results[0].type) != 'f32':
            raise QuantizedBoundaryError('readout path does not use binary32 arithmetic')
        if scalar.name == 'arith.mulf' and len(op.inputs) == 2:
            if set(scalar.operands) != set(body.args[:2]):
                raise QuantizedBoundaryError('source multiplication is not its two declared tensor inputs')
            constants = [(i,_constant_of(v)) for i,v in enumerate(op.inputs) if _type(v)['shape'] == []]
            if len(constants) != 1 or constants[0][1] is None:
                raise QuantizedBoundaryError('source multiplication needs a known scalar scale, not stored/per-column data')
            i, scale = constants[0]
            scale = f32(float(scale))
            if not isfinite(scale) or scale <= 0:
                raise QuantizedBoundaryError('source scale is not finite positive binary32')
            scales.append(scale)
            current = op.inputs[1-i]
        elif scalar.name=='arith.select' and (slot:=_cmp_select_relu_slot(op)) is not None:
            relu=True
            current=op.inputs[slot]
        elif scalar.name in ('arith.maximumf','arith.maxnumf') and len(op.inputs) in (1,2):
            slot = next((i for i,v in enumerate(op.inputs) if _type(v)['shape'] == _type(current)['shape']),None)
            if slot is None or body.args[slot] not in scalar.operands:
                raise QuantizedBoundaryError('rectifier is disconnected from its source tensor')
            other = next((v for v in scalar.operands if v is not body.args[slot]),None)
            if other in body.args[:len(op.inputs)]:
                other = op.inputs[list(body.args).index(other)]
            if other is None or _constant_of(other) != 0:
                raise QuantizedBoundaryError('readout rectifier does not have an exact zero bound')
            relu = True
            current = op.inputs[slot]
        else:
            raise QuantizedBoundaryError('readout path requires a proof for scalar operation '+scalar.name)
        operations.append(op)
    raise QuantizedBoundaryError('readout source chain exceeds the explicit analysis budget')


@dataclass(frozen=True)
class AccumulatorQuantizeProof:
    quantizer: object
    source: object
    operations: tuple
    attributes: dict
    numeric_proof: dict
    source_scales: tuple
    allow_shape_views: bool = False

    def revalidate(self):
        fresh = prove_accumulator_quantize(self.quantizer,allow_shape_views=self.allow_shape_views)
        if (fresh.source is not self.source or fresh.operations != self.operations
                or fresh.attributes != self.attributes or fresh.numeric_proof != self.numeric_proof
                or fresh.source_scales != self.source_scales):
            raise QuantizedBoundaryError('quantizer source SSA, attributes or numerical proof changed')
        return self.record()

    def record(self):
        members = set(self.operations)
        observers = [{**_identity(op), 'type':str(result.type)}
                     for op in self.operations[:-1] for result in op.results
                     if any(use.operation not in members for use in result.uses)]
        return {'schema':'submission.accumulator_quantize_proof.v1',
                'source':_type(self.source), 'output':_type(self.quantizer.results[0]),
                'source_chain':[_identity(op) for op in self.operations],
                'source_scales_f32':list(self.source_scales), 'numeric_proof':self.numeric_proof,
                'readout_attributes':dict(self.attributes), 'live_intermediate_outputs':observers,
                'dense_shape_views':self.allow_shape_views,
                'standalone_form':'ACCUMULATOR_READOUT',
                'fusion_obligations':['retain every live intermediate output',
                    'independently prove source integer producer membership, ownership and completion'],
                'scope':'exact source arithmetic over the complete signed-i32 domain; no placement/effect grant'}


def prove_accumulator_quantize(op, *, max_chain_ops=64, allow_shape_views=False):
    """Return a source-bound exact readout candidate; refuse every unproved fold."""
    if type(max_chain_ops) is not int or not 1 <= max_chain_ops <= 4096:
        raise QuantizedBoundaryError('quantizer analysis needs an explicit bounded source-chain budget')
    q = _quantizer(op)
    if (q['input']['dtype'] != 'f32' or q['output']['dtype'] != 'i8'
            or not q['input']['dense'] or not q['output']['dense']
            or q['scale']['operand_type']['dtype'] != 'f32'
            or not q['zero_point']['operand_type']['dtype'].startswith('i')):
        raise QuantizedBoundaryError('exact selected quantizer readout needs f32 data/scale and signed-i8 result')
    if q['clamp'] not in ([-128,127], [0,127]):
        raise QuantizedBoundaryError('quantizer clamp differs from signed-i8 saturation/optional ReLU')
    if type(q['zero_point']['value']) is not int or q['zero_point']['value'] != 0:
        raise QuantizedBoundaryError('post-rounding zero-point addition has no selected readout implementation')
    scale = q['scale']['value']
    if scale is None or not isfinite(float(scale)) or float(scale) <= 0:
        raise QuantizedBoundaryError('constant finite positive quantizer scale is required')
    scale = f32(float(scale))
    if not isfinite(scale) or scale <= 0:
        raise QuantizedBoundaryError('quantizer scale is not finite positive after binary32 conversion')
    try:
        reciprocal = f32(1.0/scale)
    except OverflowError as exc:
        raise QuantizedBoundaryError('binary32 reciprocal overflows the selected finite readout scale') from exc
    if not isfinite(reciprocal) or reciprocal <= 0:
        raise QuantizedBoundaryError('binary32 reciprocal is not finite positive')
    source, operations, scales, relu = _trace(op.operands[0],max_chain_ops,allow_shape_views=allow_shape_views)
    if _type(source)['shape'] != q['output']['shape']:
        from math import prod
        if not allow_shape_views or prod(_type(source)['shape']) != prod(q['output']['shape']):
            raise QuantizedBoundaryError('integer readout source/output shape changed')
    source_scales = tuple([*scales,reciprocal])
    relu = relu or q['clamp'][0] == 0
    try:
        numeric = prove_scale(source_scales,relu=relu)
    except (ValueError,OverflowError) as exc:
        raise QuantizedBoundaryError(str(exc)) from exc
    attrs = {'output_dtype':'i8','acc_scale':numeric['scale'],
             'epilogue':['acc_scale']+(['relu'] if relu else [])}
    return AccumulatorQuantizeProof(op,source,(*operations,op),attrs,numeric,source_scales,allow_shape_views)


def _producer_candidates(value, budget):
    """Bounded ancestry census; stop at integer tensors and explicit quantizers.

    Candidate ancestry is evidence of a remaining fusion obligation, not evidence
    that arbitrary floating computation can be absorbed by integer readout.
    """
    pending=[value];seen=set();candidates=[];barriers=[]
    while pending and len(seen)<budget:
        v=pending.pop()
        if v in seen:continue
        seen.add(v)
        typ=_type(v);owner=v.owner
        if typ['dtype']=='i32':
            candidates.append({'kind':'i32_tensor', 'type':typ,
                'producer':_identity(owner) if isinstance(owner,Operation) else {'operation':'function_argument'}})
            continue
        if not isinstance(owner,Operation):continue
        name=op_name(owner)
        if name=='quant_ext.dequantize_per_tensor':
            if owner.operands and _type(owner.operands[0])['dtype']=='i8':
                candidates.append({'kind':'dequantized_i8', 'type':_type(owner.operands[0]), 'producer':_identity(owner)})
            continue
        if name.startswith('quant_ext.quantize_'):continue
        if name=='linalg.generic':
            nonlinear=[o.name for r in owner.regions for b in r.blocks for o in b.ops
                       if o.name.startswith('math.')]
            if nonlinear:
                barriers.append({**_identity(owner),'scalar_operations':nonlinear,
                    'reason':'selected accumulator readout has no floating nonlinear stage'})
                continue
        if name=='tensor.empty' or name=='arith.constant':continue
        pending.extend(v for v in owner.operands if _type(v)['dtype'] in ('f32','i32'))
    return candidates,barriers,bool(pending)


def _source_obligations(value, budget):
    """Describe actual source seams without treating ancestry as admission.

    A shape-only view may occur next to a nonlinear or runtime channel operation.
    Recording the view alone would hide the latter obligation. Follow every
    floating tensor input, stop at integer/dequantization boundaries, and retain
    each operation's address maps and scalar body for a target adapter to prove.
    """
    pending=[value];seen=set();barriers=[];views=[];frontiers=[]
    while pending and len(seen)<budget:
        current=pending.pop(0)
        if current in seen:continue
        seen.add(current)
        spec=_type(current);owner=current.owner
        if spec['dtype']=='i32':
            frontiers.append({'kind':'signed_i32_tensor','type':spec,
                'producer':_identity(owner) if isinstance(owner,Operation) else {'operation':'function_argument'}})
            continue
        if not isinstance(owner,Operation):
            if spec['dtype']=='f32':
                barriers.append({'kind':'runtime_f32_tensor','operation':'function_argument',
                    'argument_index':int(current.index),
                    'result':spec,'reason':'stored floating values have no integer accumulator/readout representation proof'})
            continue
        name=op_name(owner)
        if name in ('tensor.empty','arith.constant','tensor.splat'):continue
        if name.startswith('quant_ext.'):
            frontiers.append({'kind':'quantization_boundary',**_identity(owner),
                'result':spec,'inputs':[_type(v) for v in owner.operands],
                'reason':'dequantization zero point, scale and integer-source composition require an independent exact proof'})
            continue
        item={**_identity(owner),'result':spec,
              'inputs':[_type(v) for v in owner.operands]}
        if name in ('tensor.expand_shape','tensor.collapse_shape'):
            item.update(kind='shape_view',
                reassociation=str(owner.properties.get('reassociation',owner.attributes.get('reassociation',''))),
                reason='shape-only view is not the complete boundary proof; retain subsequent arithmetic and source/output address correspondence')
            views.append(item)
        elif name=='linalg.transpose':
            item.update(kind='axis_permutation',
                permutation=str(owner.properties.get('permutation',owner.attributes.get('permutation',''))),
                reason='axis permutation needs an explicit integer-source address projection; flat readout does not implement it')
            barriers.append(item)
        elif name.startswith('linalg.'):
            scalar_ops=[op_name(o) for r in owner.regions for b in r.blocks for o in b.ops
                        if op_name(o) not in ('linalg.yield','arith.constant')]
            item['scalar_operations']=scalar_ops
            item['indexing_maps']=[str(a) for a in getattr(owner,'indexing_maps',())]
            item['iterator_types']=[str(a) for a in getattr(owner,'iterator_types',())]
            nonlinear=[n for n in scalar_ops if n.startswith('math.')]
            if nonlinear:
                item.update(kind='floating_nonlinear',
                    reason='selected accumulator readout has no stage for '+', '.join(nonlinear))
            elif name in ('linalg.reduce','linalg.pooling_nchw_max','linalg.pooling_nhwc_max') or any('reduction' in s for s in item['iterator_types']):
                item.update(kind='floating_reduction',
                    reason='source reduction order, initializer and floating arithmetic have no integer readout equivalence proof')
            elif 'arith.addf' in scalar_ops or 'arith.subf' in scalar_ops:
                item.update(kind='floating_addition',
                    reason='floating bias/residual arithmetic and its runtime operands cannot be replaced by an integer accumulator add without a rounding/overflow proof')
            elif scalar_ops==['arith.mulf']:
                constants=[{'input':i,'value':_constant_of(v),'type':_type(v)}
                           for i,v in enumerate(getattr(owner,'inputs',())) if _constant_of(v) is not None]
                item['constant_inputs']=constants
                item.update(kind='floating_multiplication',reason=
                    'constant scale is present, but broadcast/address form or complete source chain is not admitted' if constants else
                    'both multiplication inputs are source tensors; runtime/channel scaling or tensor products have no constant per-command readout scale proof')
            else:
                item.update(kind='unproved_scalar_body',
                    reason='actual scalar body is not the admitted identity-map cast/constant-scale/zero-ReLU chain')
            # Do not call an already admitted local operation a missing adapter.
            # Its full chain can still fail, and all inputs must remain visible.
            try:
                body,scalar=_pointwise(owner,current)
                admitted=(scalar.name=='arith.sitofp'
                          and len(owner.inputs)==1 and _type(owner.inputs[0])['dtype']=='i32')
                if scalar.name=='arith.select' and _cmp_select_relu_slot(owner) is not None:
                    admitted=True
                if scalar.name=='arith.mulf' and len(owner.inputs)==2:
                    admitted=any(_type(v)['shape']==[] and _constant_of(v) is not None for v in owner.inputs)
                if not admitted:barriers.append(item)
            except QuantizedBoundaryError:
                barriers.append(item)
        else:
            item.update(kind='unproved_tensor_operation',
                reason='source tensor operation has no address/effect and numeric composition proof for accumulator readout')
            barriers.append(item)
        pending.extend(v for v in owner.operands if _type(v)['dtype'] in ('f32','i32')
                       and _type(v)['shape'] not in (None,[]))
    return {'schema':'submission.quantized_source_obligations.v1',
            'barriers':barriers,'shape_views':views,'integer_frontiers':frontiers,
            'analysis_truncated':any(v not in seen for v in pending),'values_visited':len(seen),
            'scope':'source-bound remaining obligations only; no placement, numerical equivalence or ownership admission'}


def analyze_quantized_boundary(op, *, max_chain_ops=64):
    """Return JSON-safe typed eligibility evidence, or None for another op.

    Parent placement accounting must keep ``compiler_adapter_gap`` and
    ``unresolved_proof`` in its unknown/remaining-obligation census. A proved
    readout candidate does not itself mean the source region is on the device.
    """
    if op_name(op) != 'quant_ext.quantize_per_tensor':return None
    if type(max_chain_ops) is not int or not 1 <= max_chain_ops <= 4096:
        raise ValueError('quantizer analysis needs an explicit bounded source-chain budget')
    record={'operation':op_name(op),'source':_identity(op),'selected_forms':selected_boundary_forms()}
    try:
        q=_quantizer(op)
    except QuantizedBoundaryError as exc:
        return {**record,'status':'unresolved_proof','standalone':{'status':'unresolved_proof'},
                'fused':{'status':'unresolved_proof'},'remaining_proof':[str(exc)]}
    record.update(q)
    inp=q['input']['dtype']
    if inp.startswith(('f','bf')):
        standalone={'status':'hardware_unsupported','reason':
            'typed floating input cannot enter selected integer DMA/accumulator readout; f32 bit reinterpretation would change values'}
    else:
        standalone={'status':'unresolved_proof','reason':
            'typed input alone does not prove the source quantizer equals integer accumulator readout'}
    record['standalone']=standalone
    try:
        proof=prove_accumulator_quantize(op,max_chain_ops=max_chain_ops)
    except (QuantizedBoundaryError,ValueError) as exc:
        candidates,barriers,truncated=_producer_candidates(op.operands[0],max_chain_ops)
        fused={'status':'unresolved_proof' if candidates or truncated else 'not_available',
               'integer_producer_candidates':candidates,'floating_barriers':barriers,
               'source_obligations':_source_obligations(op.operands[0],max_chain_ops),
               'analysis_truncated':truncated,'remaining_proof':[str(exc)]}
        record.update(fused=fused,remaining_proof=[str(exc)])
        record['status']='unresolved_proof' if candidates or truncated or standalone['status']!='hardware_unsupported' else 'hardware_unsupported'
    else:
        record.update(status='compiler_adapter_gap',
            fused={'status':'numeric_proved','proof':proof.record(),
                   'remaining_proof':['producer membership/effects and complete live-output preservation']},
            integer_source_standalone={'status':'eligible','device_form':'ACCUMULATOR_READOUT',
                                       'input':_type(proof.source),'attributes':dict(proof.attributes)},
            remaining_proof=['extract exact source chain to the existing integer readout lowerer; retain intermediate observers'])
    return record


def analyze_scalar_boundary(op, *, max_chain_ops=64):
    """Classify actual scalar casts/comparisons against selected device forms.

    A signed floating cast truncates toward zero; it is not the narrowing
    readout's rounded, saturated quantizer. An integer ancestor is retained as
    a possible composition obligation rather than granting or denying fusion.
    """
    if op.name != 'linalg.generic' or len(op.results) != 1 or len(op.regions) != 1:
        return None
    if type(max_chain_ops) is not int or not 1 <= max_chain_ops <= 4096:
        raise ValueError('scalar boundary analysis needs an explicit bounded budget')
    body=op.regions[0].block
    scalar=[o for o in body.ops if o.name!='arith.constant']
    if (len(scalar)!=2 or scalar[-1].name!='linalg.yield'
            or list(scalar[-1].operands)!=list(scalar[0].results)):
        return None
    operation=scalar[0]
    if operation.name not in ('arith.fptosi','arith.fptoui','arith.cmpf','arith.cmpi','arith.trunci'):
        return None
    inputs=tuple(getattr(op,'inputs',()))
    if (not inputs or len(body.args)!=len(op.operands)
            or any(value not in body.args[:len(inputs)] for value in operation.operands)):
        return None
    record={'schema':'submission.scalar_boundary_eligibility.v1','source':_identity(op),
            'scalar_operation':operation.name,'input_types':[_type(v) for v in inputs],
            'output':_type(op.results[0]),
            'indexing_maps':[str(a) for a in getattr(op,'indexing_maps',())],
            'iterator_types':[str(a) for a in getattr(op,'iterator_types',())],
            'selected_forms':selected_boundary_forms()}
    if operation.name in ('arith.cmpf','arith.cmpi'):
        if str(operation.results[0].type)!='i1' or record['output']['dtype']!='i1':return None
        reason=('selected integer DMA/accumulator readout has no '+operation.name+
                ' predicate stage or signed-i1 output ABI')
        record.update(status='hardware_unsupported',standalone={'status':'hardware_unsupported','reason':reason},
            fused={'status':'hardware_unsupported','reason':reason},
            source_semantics={'predicate':str(operation.properties.get('predicate','')),
                              'result_dtype':'i1'},remaining_proof=[])
        return record
    if operation.name=='arith.trunci':
        # This is a modulo bit truncation, distinct from saturating readout.
        reason='source integer truncation preserves low bits; selected narrowing readout saturates signed values'
    else:
        reason=('source floating-to-integer cast truncates toward zero without a signed saturation stage; '
                'selected device readout consumes i32, rounds half-even and saturates to signed i8')
    standalone={'status':'hardware_unsupported','reason':reason}
    candidates=[];barriers=[];truncated=False
    for value in inputs:
        found,blocked,limited=_producer_candidates(value,max_chain_ops)
        candidates.extend(found);barriers.extend(blocked);truncated|=limited
    fused={'status':'unresolved_proof' if candidates or truncated else 'not_available',
           'integer_producer_candidates':candidates,'floating_barriers':barriers,
           'analysis_truncated':truncated,
           'reason':('an integer ancestor alone does not prove truncation equals saturated readout; '
                     'exact source range, rounding and overflow equivalence remain required') if candidates or truncated else
                    'no in-graph integer accumulator producer was found within the complete source ancestry'}
    record.update(status='unresolved_proof' if candidates or truncated else 'hardware_unsupported',
        standalone=standalone,fused=fused,
        source_semantics={'rounding':'toward_zero' if operation.name!='arith.trunci' else 'none',
                          'saturation':False,'cast_result_dtype':str(operation.results[0].type)},
        remaining_proof=[fused['reason']] if candidates or truncated else [])
    return record
