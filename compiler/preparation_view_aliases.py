"""Elide checked whole-buffer collapse/expand round trips in private preparation.

This is logical value substitution, not physical alias admission. Only the exact
entry-concat -> collapse -> expand -> one LHS consumer chain qualifies. Original
source, checked candidate, all public pointers, and final output copies survive.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from math import prod
import json
from pathlib import Path


def setting():
    from merlin.compile.tuning_knobs import knob
    return knob('preparation_view_elimination',default='off',values=('off','whole_buffer_roundtrip'),root=Path(__file__).parent)


def _core(cb):
    params=cb.get('params',{})
    return {**{k:cb.get(k) for k in ('tensors','commands','kernel_abi')},
        'params':{k:params.get(k) for k in ('source_sha256','compiler_source_sha256','source_op_count',
            'tensor_source_map','source_task_definitions','source_transport_tasks','compiler_temporaries',
            'source_preparation_storage')}}


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _derive(cb):
    from xdsl.dialects import tensor
    from xdsl.dialects.linalg.ops import GenericOp
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.transforms.input_row_packing import verify_preparation_storage,_owner
    from merlin.transforms.dense_row_batching_check import shape,boundary_attributes
    from merlin.targetgen.oot_starterkit.plan import source_operation_inventory
    p=cb.get('params',{});cert=p.get('source_preparation_storage')
    if cert is None:return deepcopy(cb),[]
    if any(k in p for k in ('global_program_plan','dense_schedule_receipts')):
        raise ValueError('view refinement must precede scheduling/emission; stale products cannot be reused')
    inventory=source_operation_inventory(cert['original_source'])
    allocations=verify_preparation_storage(inventory,cb)
    module=parse_mlir_text(cert['candidate_source']);module.verify();block=_owner(module).body.block
    ops=tuple(block.ops);names={}
    for name,origin in cert['candidate_tensor_sources'].items():
        key=tuple(sorted(origin.items()))
        if key in names:raise ValueError('view refinement requires unique actual candidate SSA bindings')
        names[key]=name
    def name(value):
        if value.owner not in ops:raise ValueError('view refinement requires invocation-private source results')
        return names.get(tuple(sorted({'op_index':ops.index(value.owner),'result_index':value.index}.items())))
    transports=p['source_transport_tasks'];events=[]
    # Observe every actual read/write in emission order, including unrelated
    # copies. Unknown effects refuse; source purity alone cannot bless a mutated
    # physical command graph. Handles never count as materialized tensor writes.
    for i in range(len(cb['commands'])+1):
        for ti,t in enumerate(transports):
            if t['before_command_index']!=i:continue
            if 'fill' in t:raise ValueError('view refinement does not admit transport fills')
            for ci,c in enumerate(t.get('copies',[])):
                events.append(('copy',ti,ci,(c['src'],),(c['dst'],)))
        if i==len(cb['commands']):continue
        c=cb['commands'][i];o=c['operands'];kind=c['opcode']
        if kind=='RES_PACK':reads=(o['src'],);writes=()
        elif kind=='MATMUL_RESIDENT':reads=(o['lhs'],);writes=()
        elif kind=='COMMIT':reads=();writes=(o['dst'],)
        elif kind=='EVICT':reads=();writes=()
        else:raise ValueError('view refinement lacks complete command effect authority')
        events.append(('command',i,None,reads,writes))
    if any(type(t['before_command_index']) is not int or not 0<=t['before_command_index']<=len(cb['commands']) for t in transports):
        raise ValueError('transport event outside actual command lifetime')
    def accesses(n,which):return [j for j,e in enumerate(events) if n in e[which]]
    def exact_copy(ti,src,dst,elements):
        t=transports[ti]
        expected={'src':src,'dst':dst,'shape':[elements],'source_strides':[1],
            'destination_strides':[1],'source_offset':0,'destination_offset':0}
        return t.get('copies')==[expected]
    rows=[];removed=set();drop=set();substitute={}
    for expand in ops:
        if type(expand) is not tensor.ExpandShapeOp:continue
        collapse=expand.operands[0].owner
        if type(collapse) is not tensor.CollapseShapeOp:continue
        if (len(expand.operands)!=1 or len(collapse.operands)!=1
                or not boundary_attributes(expand) or not boundary_attributes(collapse)
                or set(expand.properties)!={'reassociation','static_output_shape'}
                or set(collapse.properties)!={'reassociation'}
                or expand.properties['reassociation']!=collapse.properties['reassociation']):
            raise ValueError('view round trip lacks exact static inverse reassociation')
        root=collapse.operands[0];flat=collapse.results[0];final=expand.results[0]
        shape(root);shape(flat);shape(final)
        if (type(root.owner) is not tensor.ConcatOp or root.type!=final.type
                or root.type.element_type!=flat.type.element_type
                or prod(shape(root))!=prod(shape(flat))):
            raise ValueError('view round trip changes complete tensor type or element order')
        root_uses=tuple(root.uses);flat_uses=tuple(flat.uses);uses=tuple(final.uses)
        if (len(root_uses)!=1 or root_uses[0].operation is not collapse
                or len(flat_uses)!=1 or flat_uses[0].operation is not expand
                or len(uses)!=1 or type(uses[0].operation) is not GenericOp
                or uses[0].index!=0):
            raise ValueError('view round trip has unaccounted whole-source observers')
        r,f,x=map(name,(root,flat,final))
        if any(n not in allocations for n in (r,f,x)) or len({r,f,x})!=3:
            raise ValueError('view round trip requires three distinct owned private SSA allocations')
        elements=prod(shape(root));width=int(cb['tensors'][r]['dtype'][1:])//8
        rr,rw=accesses(r,3),accesses(r,4);fr,fw=accesses(f,3),accesses(f,4);xr,xw=accesses(x,3),accesses(x,4)
        if len(rr)!=1 or len(fr)!=1 or len(fw)!=1 or len(xr)!=1 or len(xw)!=1 or not rw:
            raise ValueError('view buffers have extra physical writers/readers')
        a,b,k=rr[0],fr[0],xr[0]
        if not (fw==[a] and xw==[b] and max(rw)<a<b<k
                and events[a][0]==events[b][0]=='copy' and events[k][0]=='command'):
            raise ValueError('view definitions/consumers violate completed invocation order')
        ta,tb=events[a][1],events[b][1];command=events[k][1]
        if (not exact_copy(ta,r,f,elements) or not exact_copy(tb,f,x,elements)
                or cb['commands'][command]['opcode']!='MATMUL_RESIDENT'
                or cb['commands'][command]['operands']['lhs']!=x):
            raise ValueError('actual view transport is not the complete zero-offset unit-stride source round trip')
        # Source concat is a full, disjoint row-major definition of its private
        # root, with no other writers or observations before its one consumer.
        concat=root.owner;expected=[];offset=0
        for arg in concat.inputs:
            if arg.owner is not block:raise ValueError('packed root must concatenate original immutable entry arguments')
            arg_name=names.get((('arg_index',arg.index),));dims=shape(arg)
            strides=[prod(dims[j+1:]) for j in range(len(dims))]
            expected.append({'src':arg_name,'dst':r,'shape':list(dims),'source_strides':strides,
                'destination_strides':strides,'source_offset':0,'destination_offset':offset})
            offset+=prod(dims)
        if (concat.dim.value.data!=0 or offset!=elements or any(events[j][0]!='copy' for j in rw)
                or len({events[j][1] for j in rw})!=1
                or transports[events[rw[0]][1]].get('copies')!=expected):
            raise ValueError('packed root lacks complete original concat definition')
        removed.update((f,x));drop.update((ta,tb));substitute[command]=(x,r)
        rows.append({'root':r,'removed':[f,x],'consumer_command':command,'elements':elements,
            'dtype':cb['tensors'][r]['dtype'],'removed_copy_bytes':2*elements*width,
            'removed_private_bytes':allocations[f]['storage_bytes']+allocations[x]['storage_bytes']})
    out=deepcopy(cb)
    if not rows:return out,rows
    q=out['params'];storage=q['source_preparation_storage']
    for i,(old,new) in substitute.items():out['commands'][i]['operands']['lhs']=new
    q['source_transport_tasks']=[t for i,t in enumerate(q['source_transport_tasks']) if i not in drop]
    for n in removed:
        del out['tensors'][n];del storage['candidate_tensor_sources'][n]
    storage['allocations']=[a for a in storage['allocations'] if a['tensor'] not in removed]
    q['compiler_temporaries']=[a for a in q['compiler_temporaries'] if a['tensor'] not in removed]
    for task in q['source_task_definitions']:
        task['writes']=[n for n in task['writes'] if n not in removed]
    out['kernel_abi']['args']=[a for a in out['kernel_abi']['args'] if a['tensor'] not in removed]
    # Existing storage/source checker still sees the exact original source and
    # candidate; it independently admits only the remaining materialized subset.
    verify_preparation_storage(inventory,out)
    return out,rows


@dataclass
class PreparationViewElimination:
    original:dict
    candidate:dict
    original_digest:str
    rows:list
    def revalidate(self):
        if _digest(_core(self.original))!=self.original_digest:
            raise ValueError('original preparation graph changed; replan view elimination')
        expected,rows=_derive(self.original)
        if _core(expected)!=_core(self.candidate) or rows!=self.rows:
            raise ValueError('emitted preparation differs from checked exact view elimination')
        return self
    def record(self):
        self.revalidate()
        return {'schema':'gemmini.preparation_view_elimination.v1','original_graph_sha256':self.original_digest,
            'candidate_graph_sha256':_digest(_core(self.candidate)),'round_trips':deepcopy(self.rows),
            'scope':'one invocation; exact value substitution, no physical alias grant'}


def eliminate_preparation_views(command_buffer):
    original=deepcopy(command_buffer);candidate,rows=_derive(original)
    plan=PreparationViewElimination(original,candidate,_digest(_core(original)),rows)
    plan.revalidate();return candidate,plan
