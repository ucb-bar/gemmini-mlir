"""Opt-in region-local input packing consumed by checked source row batching.

No retained device state or cross-call cache is introduced. The source is pure
static integer tensor algebra; original entry/output identities remain the ABI.
The existing row-batching owner proves the actual combined recurrence and slices.
"""
from dataclasses import dataclass
from hashlib import sha256
from xdsl.dialects import builtin,func,tensor
from xdsl.dialects.linalg.ops import GenericOp,FillOp,YieldOp
from xdsl.rewriter import Rewriter
from merlin.transforms import dense_row_batching_check as check
from merlin.transforms.dense_row_batching import DenseRowBatchingBudgets,plan_source_dense_row_batching,_state
from merlin.transforms.pipeline_state import PipelineContext
from io import StringIO
from merlin.common.dense_float_ir import DenseFloatPrinter

def _owner(module):
    owners=[op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data=='forward' and len(op.body.blocks)==1]
    if len(owners)!=1:raise ValueError('input row packing requires one original forward block')
    return owners[0]

def _text(module):
    stream=StringIO();DenseFloatPrinter(stream=stream,print_generic_format=True).print_op(module)
    return stream.getvalue()

_IO=('merlin.logical_io','merlin.logical_inputs','merlin.logical_outputs','merlin.logical_input_roles')

def _plain(module):
    copy=module.clone();owner=_owner(copy)
    abi_names=[]
    for key,count in ((_IO[1],len(owner.body.block.args)),(_IO[2],len(owner.body.block.last_op.operands))):
        if key not in owner.attributes:continue
        names=owner.attributes[key]
        if (not isinstance(names,builtin.ArrayAttr) or len(names)!=count
                or any(not isinstance(n,builtin.StringAttr) or not n.data for n in names)):
            raise ValueError('input packing requires exact typed logical ABI names')
        abi_names.extend(n.data for n in names)
    if len(abi_names)!=len(set(abi_names)):
        raise ValueError('input packing requires distinct logical ABI names')
    if _IO[0] in owner.attributes and owner.attributes[_IO[0]]!=builtin.StringAttr('merlin.logical_io.v1'):
        raise ValueError('input packing requires the existing logical IO contract')
    # Selected logical-v2 ABI classifies these initialized leaf tensors as
    # input/weight/bias/scale. All remain original read-only entry pointers;
    # this metadata grants no new effect, immutability across calls or alias.
    roles=owner.attributes.get('merlin.logical_input_roles')
    if roles is not None and (not isinstance(roles,builtin.ArrayAttr)
            or len(roles)!=len(owner.body.block.args)
            or any(not isinstance(role,builtin.StringAttr)
                or role.data not in ('input','weight','bias','scale') for role in roles)):
        raise ValueError('input packing requires exact logical-v2 input role arity and declared leaf roles')
    for key in _IO:owner.attributes.pop(key,None)
    # Named tensor fill is an exact complete scalar splat, independent of its
    # destination initializer. Validate its actual scalar body before replacing.
    for op in tuple(owner.body.block.ops):
        if type(op) is not FillOp:continue
        ops=tuple(op.body.block.ops)
        if (len(op.inputs)!=1 or len(op.outputs)!=1 or len(op.results)!=1
            or not isinstance(op.results[0].type,builtin.TensorType)
            or op.outputs[0].type!=op.results[0].type or len(op.body.block.args)!=2
            or len(ops)!=1 or type(ops[0]) is not YieldOp
            or tuple(ops[0].operands)!=(op.body.block.args[0],)
            or ops[0].attributes or ops[0].properties or op.successors
            or set(op.properties)!={'operandSegmentSizes'}
            or not check.boundary_attributes(op)):
            raise ValueError('tensor fill lacks exact complete scalar splat semantics')
        check.shape(op.results[0])
        splat=tensor.SplatOp(op.inputs[0],[],op.results[0].type)
        splat.attributes.update(op.attributes);Rewriter.replace_op(op,splat)
    return copy

def _prune(module):
    # Existing effect-aware predicate: unused tensor results have no observers.
    # Restrict cleanup to the pure tensor setup created/obsoleted by batching.
    from xdsl.transforms.dead_code_elimination import is_trivially_dead
    copy=module.clone();block=_owner(copy).body.block
    for op in reversed(tuple(block.ops)):
        if type(op) in (tensor.EmptyOp,tensor.SplatOp,tensor.ExtractSliceOp) and is_trivially_dead(op):
            block.erase_op(op)
    copy.verify()
    return copy

@dataclass
class SharedRHSRowPacking:
    source:object
    prepared:object
    candidate:object
    row_plan:object
    groups:tuple
    original_state:tuple
    prepared_state:tuple
    candidate_state:tuple
    records:tuple
    def revalidate(self):
        if (_state(self.source)!=self.original_state or _state(self.prepared)!=self.prepared_state
                or _state(self.candidate)!=self.candidate_state):
            raise ValueError('shared RHS actual source/preparation/candidate changed; replan')
        self.row_plan.revalidate()
        if _text(_plain(self.candidate))!=_text(_prune(self.row_plan.candidate)):
            raise ValueError('actual emitted row-batched source differs from checked product')
        # Invert only the exact constructed concat/slices. Every original source
        # operation, operand, attribute, result observer and return must survive.
        restored=self.prepared.clone();block=_owner(restored).body.block
        operations=list(block.ops);remove=[]
        for record in self.records:
            concat=operations[record['concat_ordinal']]
            if (type(concat) is not tensor.ConcatOp or concat.attributes
                    or concat.dim.value.data!=0 or tuple(concat.inputs)!=tuple(block.args[i] for i in record['lhs_args'])):
                raise ValueError('actual input packing no longer concatenates exact original arguments')
            remove.append(concat)
            for ordinal,member,arg,start,m,k in record['views']:
                view=operations[ordinal];op=operations[member]
                if (type(view) is not tensor.ExtractSliceOp or view.attributes or view.source is not concat.result
                    or tuple(view.static_offsets.get_values())!=(start,0)
                    or tuple(view.static_sizes.get_values())!=(m,k)
                    or tuple(view.static_strides.get_values())!=(1,1)
                    or len(tuple(view.result.uses))!=1
                    or op.inputs[0] is not view.result):
                    raise ValueError('actual packed member view changed source coordinates/observer')
                op.operands=(block.args[arg],*op.operands[1:]);remove.append(view)
        for operation in reversed(list(block.ops)):
            if operation in remove:block.erase_op(operation)
        restored.verify()
        if _text(restored)!=_text(_plain(self.source)):
            raise ValueError('packing inversion did not recover complete original source')
        return self
    def record(self):
        self.revalidate()
        return {'schema':'gemmini.shared_rhs_row_packing.v1','original_source_sha256':sha256(_text(self.source).encode()).hexdigest(),
            'prepared_sha256':sha256(_text(self.prepared).encode()).hexdigest(),
            'candidate_sha256':sha256(_text(self.candidate).encode()).hexdigest(),
            'groups':list(self.groups),'checked_row_batching':self.row_plan.record(),
            'scope':'one invocation; exact signed integer tensor concat/slices, no physical alias or retention grant'}

def plan_shared_rhs_grouping(module,*,max_members=8,max_rows=16):
    if type(max_members) is not int or not 2<=max_members<=64 or type(max_rows) is not int or max_rows<=0:
        raise ValueError('shared RHS packing requires bounded positive member/row budgets')
    module.verify();source_owner=_owner(module)
    # Only a complete pure static block qualifies. Unknown calls/memrefs/effects
    # refuse the whole region; matching tensor names are never evidence.
    plain=_plain(module);block=_owner(plain).body.block;body=list(block.ops)
    if not check.safe_span(body[:-1]):return None
    partitions={}
    for op in body:
        if type(op) is not GenericOp or not check.supported_scope(op):continue
        try:(m,k,n),dtypes=check.geometry(op,index_bits=64)
        except check.RowBatchingUnavailable:continue
        lhs,rhs=op.inputs
        if (lhs.owner is not block or rhs.owner is not block or lhs.type.element_type!=builtin.i8
                or rhs.type.element_type!=builtin.i8 or op.results[0].type.element_type!=builtin.i32):continue
        partitions.setdefault((rhs,k,n),[]).append((op,m,k,n,lhs.index))
    selected=[]
    for rows in partitions.values():
        # Never split a source family based on runtime values or a case name.
        if 2<=len(rows)<=max_members and sum(row[1] for row in rows)<=max_rows:selected.append(rows)
    if not selected:return None
    prepared=plain.clone();cloned=dict(zip(plain.walk(),prepared.walk(),strict=True));pblock=_owner(prepared).body.block
    constructions=[];groups=[]
    for rows in selected:
        anchor=cloned[rows[0][0]];args=[pblock.args[row[4]] for row in rows]
        concat=tensor.ConcatOp(args,0,builtin.TensorType(builtin.i8,(sum(row[1] for row in rows),rows[0][2])));pblock.insert_op_before(concat,anchor);start=0;views=[]
        for original,m,k,n,arg in rows:
            member=cloned[original];view=tensor.ExtractSliceOp.from_static_parameters(concat.result,(start,0),(m,k))
            pblock.insert_op_before(view,anchor);member.operands=(view.result,*member.operands[1:]);views.append((view,member,arg,start,m,k));start+=m
        constructions.append((concat,args,views));groups.append({'members':len(rows),'rows':start,'k':rows[0][2],'n':rows[0][3],
            'rhs_arg':rows[0][0].inputs[1].index,'lhs_args':[row[4] for row in rows]})
    prepared.verify();positions={op:i for i,op in enumerate(pblock.ops)}
    records=tuple({'concat_ordinal':positions[c],'lhs_args':[a.index for a in args],
        'views':[(positions[v],positions[m],a,start,rows,k) for v,m,a,start,rows,k in views]} for c,args,views in constructions)
    row_plan=plan_source_dense_row_batching(prepared,PipelineContext(),DenseRowBatchingBudgets(max_members=max_members,max_batches=len(selected),max_candidates=16384))
    if row_plan is None:raise ValueError('exact packing has no admitted row-batching consumer')
    candidate=_prune(row_plan.candidate)
    for key in _IO:
        if key in source_owner.attributes:_owner(candidate).attributes[key]=source_owner.attributes[key]
    result=SharedRHSRowPacking(module,prepared,candidate,row_plan,tuple(groups),_state(module),_state(prepared),_state(candidate),records)
    return result.revalidate()



def verify_preparation_storage(inventory,command_buffer):
    """Reconstruct exact original SSA preparation; return derived allocations.

    Serialized claims never authorize arbitrary dimensions. The source bytes
    must match the original inventory; the actual reconstructed candidate must
    match exactly. Existing source row-batching and packing inversion run again.
    This admits private intermediate allocations only, never views/aliases,
    persistent caches, new arithmetic, asynchronous completion or permissions.
    """
    from math import prod
    from merlin.frontends.linalg_mlir import parse_mlir_text
    record=command_buffer.get('params',{}).get('source_preparation_storage')
    if record is None:return {}
    if (type(record) is not dict or set(record)!={'schema','original_source','candidate_source','max_members','max_rows','candidate_tensor_sources','allocations'}
            or record['schema']!='merlin.input_row_packing_storage.v1'):
        raise ValueError('invalid typed input row packing storage record')
    original,candidate=record['original_source'],record['candidate_source']
    if (type(original) is not str or type(candidate) is not str or len(original.encode())>2<<20 or len(candidate.encode())>4<<20
            or sha256(original.encode()).hexdigest()!=inventory['source_sha256']):
        raise ValueError('preparation storage must bind exact original source bytes')
    owner=plan_shared_rhs_grouping(parse_mlir_text(original),max_members=record['max_members'],max_rows=record['max_rows'])
    if owner is None or _text(owner.candidate)!=candidate:
        raise ValueError('preparation storage lacks exact constructive source/candidate correspondence')
    owner.revalidate();block=_owner(owner.candidate).body.block;ops=tuple(block.ops)
    sources=record['candidate_tensor_sources'];allocations=record['allocations'];tensors=command_buffer.get('tensors',{})
    if (type(sources) is not dict or set(sources)!=set(tensors) or type(allocations) is not list
            or any(type(a) is not dict for a in allocations)):
        raise ValueError('preparation storage requires complete typed SSA bindings/allocations')
    original_names=set(command_buffer['params']['tensor_source_map'])
    new_names=set(tensors)-original_names
    if len(allocations)!=len(new_names) or {a.get('tensor') for a in allocations}!=new_names:
        raise ValueError('preparation storage private allocation roster changed')
    tasks=command_buffer.get('params',{}).get('source_task_definitions',[])
    if len(tasks)!=1 or tasks[0].get('task_index')!=0 or tasks[0].get('source_op_indices')!=list(range(inventory['source_op_count'])):
        raise ValueError('preparation storage requires the complete checked original region lifetime')
    outputs=_owner(owner.candidate).body.block.last_op.operands
    approved={}
    for allocation in allocations:
        name=allocation['tensor'];origin=sources.get(name)
        if type(origin) is not dict or set(origin)!={'op_index','result_index'}:
            raise ValueError('prepared allocation lacks exact candidate SSA result')
        i,j=origin['op_index'],origin['result_index']
        if type(i) is not int or type(j) is not int or not 0<=i<len(ops) or not 0<=j<len(ops[i].results):
            raise ValueError('prepared allocation names an absent candidate result')
        value=ops[i].results[j];ty=value.type
        if not isinstance(ty,builtin.TensorType) or value in outputs or ty.element_type not in (builtin.i8,builtin.i32):
            raise ValueError('prepared allocation is not a private signed integer tensor')
        shape=list(ty.get_shape());dtype=str(ty.element_type);size=prod(shape)*(1 if dtype=='i8' else 4)
        expected={'tensor':name,'candidate_source':origin,'shape':shape,'dtype':dtype,'storage_bytes':size,'lifetime_task_index':0,'allocation':'distinct_invocation_scratch'}
        if allocation!=expected or tensors.get(name)!={'shape':shape,'dtype':dtype,'role':'intermediate'}:
            raise ValueError('prepared storage shape/type/bytes/lifetime differs from actual constructive result')
        approved[name]=expected
    # Actual logical outputs are exactly the checked original return family.
    for name,origin in command_buffer['params']['tensor_source_map'].items():
        if origin.get('arg_index') is not None:
            actual=sources.get(name)
            if actual!=origin:raise ValueError('preparation changed original input identity')
    original_returns=inventory['returns'];abi_outputs=command_buffer.get('kernel_abi',{}).get('outputs',[])
    if len(abi_outputs)!=len(outputs) or len(outputs)!=len(original_returns):
        raise ValueError('preparation changed complete original live output count')
    for name,value,returned in zip(abi_outputs,outputs,original_returns,strict=True):
        if (command_buffer['params']['tensor_source_map'].get(name)!=returned['source']
                or sources.get(name)!={'op_index':ops.index(value.owner),'result_index':value.index}):
            raise ValueError('preparation changed original live output inversion/binding')
    return approved
