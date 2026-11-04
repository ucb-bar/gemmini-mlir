"""Bounded source-form diagnostic queries. Never decides placement or tolerances."""
from __future__ import annotations
import re
from itertools import islice


def value_record(value):
    from xdsl.dialects.builtin import TensorType, NoneAttr
    t=value.type
    return {'type':str(t),'dtype':str(t.element_type) if isinstance(t,TensorType) else str(t),
            'shape':list(t.get_shape()) if isinstance(t,TensorType) else None,
            'layout':'dense_identity_storage' if isinstance(t,TensorType) and isinstance(t.encoding,NoneAttr) else 'unknown_or_encoded'}


def _bounded_generic(op, budget=64):
    """Clone only a bounded upstream scalar chain; shared Q/DQ lowering is exact.

    Original fanout is checked separately, since a query clone cannot grant
    ownership of external uses. Unknown producers remain explicit tensor inputs.
    """
    from xdsl.ir import Block, Region, Operation
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.dialects.func import FuncOp, ReturnOp
    from merlin.common.mlir_query import op_name
    from merlin.llvmlower.passes_xdsl import lower_quant_ext
    allowed={'linalg.generic','quant_ext.quantize_per_tensor','quant_ext.dequantize_per_tensor','tensor.splat','arith.constant'}
    owned=set();order=[];leaves=[];seen=set();scalar_count=0
    def visit(value):
        nonlocal scalar_count
        if value in seen:return
        if len(seen)>=256:raise ValueError('diagnostic SSA-value query budget exceeded')
        seen.add(value);owner=value.owner
        if (value is not op.results[0] and str(getattr(value.type,'element_type','')) in ('i8','i32')
                or not isinstance(owner,Operation) or owner.parent is not op.parent
                or op_name(owner) not in allowed):
            leaves.append(value);return
        if owner in owned:return
        owned.add(owner)
        nested_ops=tuple(islice(owner.walk(),129));scalar_count+=len(nested_ops)
        if len(owned)>budget or len(nested_ops)>128 or scalar_count>512 or any(len(n.operands)>16 for n in nested_ops):
            raise ValueError('diagnostic source-chain/scalar query budget exceeded')
        for operand in owner.operands:visit(operand)
        internal={v for nested in nested_ops for v in nested.results}
        internal.update(a for nested in nested_ops for region in nested.regions for block in region.blocks for a in block.args)
        for nested in nested_ops:
            for operand in nested.operands:
                if operand not in internal:visit(operand)
        order.append(owner)
    visit(op.results[0])
    block=Block(arg_types=[v.type for v in leaves]);mapping=dict(zip(leaves,block.args))
    for original in order:block.add_op(original.clone(value_mapper=mapping))
    result=mapping[op.results[0]];block.add_op(ReturnOp(result))
    fn=FuncOp('diagnostic_query',([v.type for v in leaves],[result.type]),Region(block))
    module=ModuleOp([fn]);module.verify();lower_quant_ext(module);module.verify()
    root=block.last_op.operands[0].owner
    live=[]
    for original in order:
        if original is op or op_name(original) in ('tensor.splat','arith.constant'):continue
        if any(u.operation not in owned for v in original.results for u in v.uses):
            live.append({k:str(v) for k,v in original.attributes.items() if k.startswith(('prov.','merlin.source'))})
    return root,live


def _capability(form, attributes):
    """Use the existing selected output-path admission, not inferred dtype support."""
    from .source_groups import _path
    path=_path();acc={s['kind'] for s in path.get('accumulator_stages',()) if s.get('dtype')=='i32'}
    readout=path.get('readout',())
    if path.get('accumulator',{}).get('dtype')!='i32' or 'preload' not in acc:
        return False,'selected output path lacks i32 accumulator preload'
    if form=='RESIDUAL_ADD' and 'accumulate' not in acc:
        return False,'selected output path lacks accumulator operand addition'
    if not any(s.get('kind')=='requant' and s.get('to')=='i8' and s.get('rounding')=='half_even' and s.get('clamp')==[-128,127] for s in readout):
        return False,'selected output path lacks signed i8 half-even saturating readout'
    if 'relu' in attributes.get('epilogue',()) and not any(s.get('kind')=='activation' and s.get('function')=='relu' and s.get('domain')=='accumulator' for s in readout):
        return False,'selected output path lacks accumulator ReLU'
    return True,None


def classify_source(op, *, placed_form=None, placed_proof=None, fused=False):
    """Return standalone and fused eligibility separately; UNKNOWN stays unknown.

    A numerical refusal applies to the selected concrete form, not every possible
    implementation of the source. Exceptions are proof/query failures, never
    evidence that hardware cannot implement the source.
    """
    from merlin.common.mlir_query import op_name
    from .quantized_boundary import selected_boundary_forms
    forms=selected_boundary_forms()
    operands=[value_record(v) for v in op.operands];outputs=[value_record(v) for v in op.results]
    record={'operand_types':operands,'result_types':outputs,
            'indexing_maps':[str(m) for m in op.properties.get('indexing_maps',())],
            'iterator_types':[str(i) for i in op.properties.get('iterator_types',())],
            'selected_capabilities':forms,
            'standalone':{'status':'UNKNOWN','reason':'standalone scalar/layout/numeric proof not established'},
            'fused':{'status':'UNKNOWN','reason':'fused eligibility not inspected'},
            'capability_mode':'UNKNOWN','selected_form':None,'status':'unresolved_proof','host_reason':'no proved complete device form'}
    from .generic_quantized_readout import _literal
    payloads=list(op.inputs) if hasattr(op,'inputs') else list(op.operands)
    floating=any(v['dtype'].startswith(('f','bf')) for v in outputs) or any(
        value_record(v)['dtype'].startswith(('f','bf')) and not (value_record(v)['shape']==[] and _literal(v) is not None)
        for v in payloads)
    if floating:
        record['standalone']={'status':'hardware_unsupported','reason':'typed floating tensor boundary cannot enter or exit selected integer DMA/readout as a standalone operation'}
    if placed_form:
        record.update(status='supported',selected_form=placed_form,host_reason=None)
        record['fused' if fused else 'standalone']={'status':'supported','device_form':placed_form,'proof':placed_proof}
        record['capability_mode']='fused_only' if fused and floating else 'fused_proved_standalone_UNKNOWN' if fused else 'standalone'
        return record
    name=op_name(op)
    if name not in ('linalg.generic','quant_ext.quantize_per_tensor') or not outputs or outputs[0]['dtype']!='i8':return record
    if name=='linalg.generic' and not any(n.name=='arith.fptosi' for n in islice(op.walk(),129)):return record
    failures=[]
    source_bound=0
    for key in ('merlin.bound_lsb','bound_lsb'):
        attr=op.attributes.get(key,op.properties.get(key))
        bound=getattr(getattr(attr,'value',None),'data',None)
        if bound is not None:source_bound=int(bound)
    try:root,live=_bounded_generic(op)
    except Exception as exc:
        record['queries']=[{'query':'bounded_generic_chain','status':'unresolved_proof','exception':type(exc).__name__,'reason':str(exc)}]
        return record
    from .source_lowered_residual import prove as prove_residual, ResidualRefusal
    from .generic_quantized_readout import prove_generic_readout, GenericReadoutError
    for form,prove in (('RESIDUAL_ADD',prove_residual),('ACCUMULATOR_READOUT',prove_generic_readout)):
        try:proof=prove(root)
        except Exception as exc:
            reason=str(exc);failure={'query':form,'status':'unresolved_proof','exception':type(exc).__name__,'reason':reason}
            # This message is raised by the complete 65536-pair screen, after
            # source SSA/layout/scale proof, with an actual integer witness.
            witness=re.fullmatch(r'operand-sum load rounding exceeds declared bound (\d+): pair \((-?\d+), (-?\d+)\) error (\d+)',reason)
            if form=='RESIDUAL_ADD' and isinstance(exc,ValueError) and witness:
                bound,a,b,error=map(int,witness.groups())
                failure.update(status='numerically_invalid' if source_bound==0 else 'unresolved_proof',counterexample={'lhs_i8':a,'rhs_i8':b,'error_lsb':error,'bound_lsb':bound,'pairs_screened':65536,'proof_scope':'exact bound-zero selected schedule','source_bound_lsb':source_bound})
            elif source_bound==0 and isinstance(exc,ResidualRefusal) and reason=='original f32 stages differ from device operand rounding':
                failure.update(status='numerically_invalid',evidence='exhaustive original f32-stage comparison differs')
            failures.append(failure);continue
        attrs=proof.attributes
        try:admitted,reason=_capability(form,attrs)
        except Exception as exc:
            failures.append({'query':form,'status':'unresolved_proof','exception':type(exc).__name__,'reason':str(exc)})
            continue
        if not admitted:
            failures.append({'query':form,'status':'hardware_unsupported','reason':reason});continue
        record.update(selected_form=form,queries=failures)
        if live:
            record.update(status='unresolved_proof',host_reason='numeric form proved; live intermediate outputs require a separate output ABI')
            record['fused']={'status':'unresolved_proof','device_form':form,'live_intermediates':live}
            return record
        info={'status':'supported','device_form':form,'attributes':attrs,
              'proof':proof.proof if form=='RESIDUAL_ADD' else proof.record()}
        # Whole-chain extraction crosses the f32 seam; it is a fused source
        # form even though its selected schedule is an unfused DMA readout.
        record['fused']=info
        record['capability_mode']='fused_only' if floating else 'standalone'
        record.update(status='compiler_unimplemented',host_reason='exact selected source form proved but absent from actual device plan')
        if not floating:record['standalone']=info.copy()
        return record
    record['queries']=failures
    invalid=next((q for q in failures if q['status']=='numerically_invalid'),None)
    if invalid:
        record.update(status='numerically_invalid',selected_form=invalid['query'],host_reason=invalid['reason'])
        record['fused']={'status':'UNKNOWN','reason':'selected candidate is numerically invalid; other fused forms not inspected','selected_candidate':invalid}
    return record
