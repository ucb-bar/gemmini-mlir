"""Executable connected tensor graphs with source-bound device/host tasks."""
from io import StringIO
from math import prod
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects import llvm
from xdsl.dialects.builtin import TensorType, i8
from xdsl.ir import Block, Region
from merlin.targetgen.contract import linalg_iface as li
from merlin.llvmlower.host_builder import PTR
from llvm_emitter import LLVMEmitter
from host_lowering import HostTensorLowerer, TensorValue


def pointwise_fusion_groups(operations):
    """Return tail-indexed chains, crossing only immutable initializer operations."""
    candidates={op:i for i,op in enumerate(operations) if HostTensorLowerer.pointwise_fusion_candidate(op)}
    grouped=set();groups={}
    for head,index in candidates.items():
        if head in grouped:continue
        chain=[(index,head)];current=head
        while True:
            uses=list(current.results[0].uses)
            if len(uses)!=1:break
            use=uses[0];consumer=use.operation
            if consumer not in candidates or consumer in grouped:break
            next_index=candidates[consumer]
            if next_index<=chain[-1][0]:break
            gaps=operations[chain[-1][0]+1:next_index]
            if any(li._op_name(op) not in ('tensor.empty','arith.constant','tensor.splat') for op in gaps):break
            if tuple(consumer.results[0].type.get_shape())!=tuple(head.results[0].type.get_shape()):break
            ins,_=li._ins_outs(consumer)
            if use.index>=len(ins):break
            edge_map=list(consumer.indexing_maps)[use.index].data
            if [getattr(expr,'position',None) for expr in edge_map.results]!=list(range(edge_map.num_dims)):break
            chain.append((next_index,consumer));current=consumer
        if len(chain)>1:
            groups[chain[-1][0]]=chain
            grouped.update(op for _,op in chain)
    return groups


def graph_storage_lifetimes(operations, returned_values, delayed_consumers):
    """Inclusive lifetimes in the selected serial schedule.

    Nested scalar captures belong to their enclosing source operation. Fusion
    postpones those reads to the chain tail. Views are excluded at every edge,
    so a view cannot hide a live reference to a pooled root allocation.
    """
    indices={op:i for i,op in enumerate(operations)}
    views=set()
    for op in operations:
        if li._op_name(op) in ('tensor.collapse_shape','tensor.expand_shape','tensor.cast'):
            views.update(op.operands);views.update(op.results)
    returned=set(returned_values)
    result={}
    for index,op in enumerate(operations):
        for j,value in enumerate(op.results):
            if not isinstance(value.type,TensorType):continue
            last=index
            for use in value.uses:
                consumer=use.operation
                while consumer is not None and consumer not in indices:
                    consumer=consumer.parent_op()
                if consumer is None:raise ValueError('graph tensor use escapes source entry')
                consumer_index=indices[consumer]
                last=max(last,delayed_consumers.get(consumer_index,consumer_index))
            exclusions=[]
            if value in returned:exclusions.append('caller output')
            if value in views:exclusions.append('alias view participates in source lifetime')
            if li._op_name(op)=='tensor.empty':exclusions.append('uninitialized source storage')
            result[value]={'source_op_index':index,'source_result_index':j,
                           'live_start':index,'live_end':last,
                           'reuse_eligible':not exclusions,'exclusions':exclusions}
    return result



def graph_lane_placement(operations, diagnostics):
    """Declare actual source dispatches through the public lane-report boundary.

    This is a compiler declaration, not an execution receipt. Shared provenance
    IDs are grouped only when family and actual emitted lane also agree. Source
    indices retain distinct operations even when a region ID is repeated.
    """
    groups={}
    for op,row in zip(operations,diagnostics,strict=True):
        prov=li._prov(op)
        region=prov.get('region_id')
        family=prov.get('family') or li._op_name(op)
        lane='unit' if row['placement']=='device' else 'host'
        reason=row['host_reason'] if lane=='host' else 'typed source operation emitted by selected device adapter'
        key=(region,family,lane,reason)
        if key not in groups:
            group={'lane':lane,'family':family,'reason':reason,
                   'source_op_indices':[],'global_task_indices':[0]}
            if region is not None:group['region_id']=region
            groups[key]=group
        groups[key]['source_op_indices'].extend(row['source_op_indices'])
    return list(groups.values())


def device_epilogue_fusion_groups(operations):
    """Prove zero-initialized signed dot -> exact signed saturating truncation.

    The conservative K bound excludes i32 overflow in every partial sum. Unit
    f32 readout preserves all in-range integers and cannot move an out-of-range
    signed integer across either exactly representable saturation threshold.
    """
    from xdsl.ir.affine import AffineDimExpr
    indices={op:i for i,op in enumerate(operations)}
    groups={}
    for index,op in enumerate(operations):
        kind=li._op_name(op)
        if kind!='linalg.generic' or len(op.results)!=1:continue
        ins,outs=li._ins_outs(op)
        if len(ins)!=2 or len(outs)!=1:continue
        if any(not isinstance(v.type,TensorType) or len(v.type.get_shape())!=2 for v in [*ins,*outs]):continue
        if [str(v.type.element_type) for v in ins]!=['i8','i8'] or str(op.results[0].type.element_type)!='i32':continue
        m,k=ins[0].type.get_shape();rk,n=ins[1].type.get_shape()
        if min(m,k,n)<=0 or k!=rk or tuple(op.results[0].type.get_shape())!=(m,n):continue
        if k*16384>(1<<31)-1:continue
        init=outs[0].owner
        if getattr(init,'name',None) not in ('linalg.fill','tensor.splat') or li._scalar_const(init.operands[0])!=0:continue
        maps=[str(x).replace(' ','') for x in op.indexing_maps]
        if maps!=['affine_map<(d0,d1,d2)->(d0,d2)>','affine_map<(d0,d1,d2)->(d2,d1)>','affine_map<(d0,d1,d2)->(d0,d1)>']:continue
        if [getattr(x.data,'value',None) for x in op.iterator_types]!=['parallel','parallel','reduction']:continue
        body=op.regions[0].blocks[0];scalar=list(body.ops)
        if [li._op_name(x) for x in scalar]!=['arith.extsi','arith.extsi','arith.muli','arith.addi','linalg.yield']:continue
        x,y,mul,add,yieldop=scalar
        if list(x.operands)!=[body.args[0]] or list(y.operands)!=[body.args[1]]:continue
        if any(str(v.type)!='i32' for v in [x.results[0],y.results[0],mul.results[0],add.results[0]]):continue
        if list(mul.operands)!=[x.results[0],y.results[0]]:continue
        if list(add.operands) not in ([body.args[2],mul.results[0]],[mul.results[0],body.args[2]]):continue
        if list(yieldop.operands)!=[add.results[0]]:continue
        uses=list(op.results[0].uses)
        if len(uses)!=1:continue
        consumer=uses[0].operation;tail=indices.get(consumer)
        if tail is None or tail<=index or uses[0].index!=0:continue
        if any(li._op_name(gap) not in ('tensor.empty','arith.constant') for gap in operations[index+1:tail]):continue
        if li._op_name(consumer)!='linalg.generic' or len(consumer.results)!=1:continue
        cin,cout=li._ins_outs(consumer)
        if len(cin)!=1 or len(cout)!=1 or cin[0] is not op.results[0]:continue
        if str(consumer.results[0].type.element_type)!='i8' or tuple(consumer.results[0].type.get_shape())!=(m,n):continue
        if any(getattr(x.data,'value',None)!='parallel' for x in consumer.iterator_types):continue
        if len(consumer.iterator_types)!=2:continue
        def identity(amap):
            return amap.num_dims==2 and amap.num_symbols==0 and len(amap.results)==2 and all(isinstance(x,AffineDimExpr) and x.position==i for i,x in enumerate(amap.results))
        if not all(identity(x.data) for x in consumer.indexing_maps):continue
        body=consumer.regions[0].blocks[0];env={body.args[0]:(-(1<<31),(1<<31)-1)};valid=True;interval=None
        for scalar in body.ops:
            name=li._op_name(scalar)
            if name=='linalg.yield':
                if len(scalar.operands)!=1 or interval is None or scalar.operands[0] is not narrowed:valid=False
                continue
            if name=='arith.constant':continue
            if name in ('arith.maxsi','arith.minsi') and interval is None:
                left,right=scalar.operands;child=env.get(left);constant=li._scalar_const(right)
                if child is None:child=env.get(right);constant=li._scalar_const(left)
                if child is None or type(constant) is not int or str(scalar.results[0].type)!='i32':valid=False;break
                f=max if name=='arith.maxsi' else min
                env[scalar.results[0]]=tuple(f(v,constant) for v in child)
            elif name=='arith.trunci' and interval is None and str(scalar.operands[0].type)=='i32' and str(scalar.results[0].type)=='i8':
                interval=env.get(scalar.operands[0]);narrowed=scalar.results[0]
            else:valid=False;break
        if not valid or interval not in ((-128,127),(0,127)):continue
        groups[tail]={'producer_index':index,'producer':op,'consumer':consumer,'relu':interval[0]==0,'m':m,'k':k,'n':n}
    return groups


def emit_fused_device_epilogue(e,host,group):
    from matmul_lowering import emit_matmul
    ins,_=li._ins_outs(group['producer'])
    a,b=(host.values[v] for v in ins)
    result=host.result(group['consumer'].results[0])
    emit_matmul(e,a.ptr,b.ptr,result.ptr,group['m'],group['k'],group['n'],
                attrs={'output_dtype':'i8','acc_scale':1.0,'epilogue':['relu'] if group['relu'] else []},
                compute_provenance={'task_index':0,'source_op_index':group['producer_index'],
                                    'region_id':li._prov(group['producer']).get('region_id')})
    e.fence()


def compile_graph(cb, *, device_callback=None):
    """Lower the exact source graph. Callback(e, op, host)->bool handles device ops.

    The default callback lowers proven signed i8 matrix products; unsupported
    integer compute is rejected rather than silently assigned to the CPU.
    """
    module=Parser(li.make_linalg_context(),cb['source_text']).parse_module()
    fn=next(op for op in module.walk() if li._op_name(op)=='func.func')
    block=fn.regions[0].blocks[0]
    inputs=cb['inputs'];outputs=cb['outputs']
    e=LLVMEmitter(inputs+outputs)
    globals=[];scratch_bytes=0
    storage_slots=[];logical_storage_bytes=0;lifetimes={}
    reuse_mode='retained'
    def allocate(count,dtype,*,source_value=None):
        nonlocal scratch_bytes,logical_storage_bytes
        sizes={'i1':1,'i8':1,'i16':2,'i32':4,'i64':8,'f32':4,'bf16':2}
        if dtype not in sizes:raise ValueError('unsupported graph storage dtype '+dtype)
        n=count*sizes[dtype];logical_storage_bytes+=n
        life=lifetimes.get(source_value)
        eligible=(reuse_mode=='reuse' and life is not None and life['reuse_eligible'])
        slot=None
        if eligible:
            # Strict inequality prevents input/output aliasing inside an op.
            # Default device adapters fence before returning to this schedule.
            slot=next((candidate for candidate in storage_slots
                       if candidate['reusable'] and candidate['dtype']==dtype
                       and candidate['element_count']==count and candidate['alignment']==64
                       and candidate['live_end']<life['live_start']),None)
        if slot is None:
            scratch_bytes+=n
            symbol='__merlin_graph_tmp_'+str(len(globals))
            ty=llvm.LLVMArrayType(n,i8)
            init=Block();zero=llvm.ZeroOp(result_types=[ty]);init.add_op(zero);init.add_op(llvm.ReturnOp(zero.results[0]))
            globals.append(llvm.GlobalOp(ty,symbol,'internal',alignment=64,body=Region(init)))
            slot={'symbol':symbol,'dtype':dtype,'element_count':count,'byte_size':n,
                  'alignment':64,'reusable':eligible,'live_end':life['live_end'] if life else None,
                  'allocations':[]}
            storage_slots.append(slot)
        if life is not None:
            slot['allocations'].append(dict(life))
            slot['live_end']=life['live_end']
        else:
            slot['allocations'].append({'purpose':'private lowering temporary; reuse excluded'})
        return e.b.add(llvm.AddressOfOp(slot['symbol'],PTR)).results[0]
    from merlin.compile.tuning_knobs import knob
    from pathlib import Path
    init_copy=knob('host_init_copy',default='retained',values=('retained','elide'),root=Path(__file__).parent)
    cb['params']['host_init_copy']=init_copy
    host=HostTensorLowerer(e,allocate,elide_overwritten_init=(init_copy=='elide'))
    for arg,name in zip(block.args,inputs):
        host.values[arg]=TensorValue(e.args[name],tuple(arg.type.get_shape()),str(arg.type.element_type))
    returned=next(op for op in block.ops if li._op_name(op)=='func.return')
    host.bound_outputs={value:e.args[name] for value,name in zip(returned.operands,outputs)}
    fusion=knob('host_pointwise_fusion',default='retained',values=('retained','fuse'),root=Path(__file__).parent)
    cb['params']['host_pointwise_fusion']=fusion
    operations=list(block.ops)
    fusion_groups=pointwise_fusion_groups(operations) if fusion=='fuse' else {}
    fusion_heads={i:chain[-1][0] for chain in fusion_groups.values() for i,_ in chain[:-1]}
    device_fusion=knob('graph_device_fusion',default='retained',values=('retained','fuse'),root=Path(__file__).parent)
    cb['params']['graph_device_fusion']=device_fusion
    if device_fusion=='fuse' and device_callback is not None:
        raise ValueError('device fusion requires the selected typed contraction emitter')
    device_groups=device_epilogue_fusion_groups(operations) if device_fusion=='fuse' else {}
    device_heads={g['producer_index']:tail for tail,g in device_groups.items()}
    fusion_heads.update(device_heads)
    sibling_mode=knob('sibling_order',default='retained',values=('retained','pack','pack_rows'),root=Path(__file__).parent)
    cb['params']['sibling_order']=sibling_mode
    sibling_groups={}
    if sibling_mode in ('pack','pack_rows'):
        if device_callback is not None:
            raise ValueError('sibling packing requires the selected typed contraction emitter')
        from graph_sibling_lowering import sibling_groups as find_sibling_groups
        forbidden=set(device_heads)|set(device_groups)
        sibling_groups=find_sibling_groups(operations,forbidden_indices=forbidden,
                                          max_rows=16 if sibling_mode=='pack_rows' else 1)
    sibling_heads={i:tail for tail,g in sibling_groups.items() for i,_ in g['members'] if i!=tail}
    fusion_heads.update(sibling_heads)
    sibling_receipts=[]
    reuse_mode=knob('graph_storage_reuse',default='retained',values=('retained','reuse'),root=Path(__file__).parent)
    cb['params']['graph_storage_reuse']=reuse_mode
    if reuse_mode=='reuse' and device_callback is not None:
        raise ValueError('storage reuse requires the selected serialized device completion path')
    lifetimes=graph_storage_lifetimes(operations,returned.operands,fusion_heads)
    host.result_allocator=lambda value,count,dtype:allocate(count,dtype,source_value=value)
    tasks=[]
    for index,op in enumerate(block.ops):
        kind=li._op_name(op)
        prov=li._prov(op);region=prov.get('region_id')
        host.source_op_index=index
        with e.task(0,source_op_index=index if kind!='func.return' else None,region_id=region):
            if index in sibling_heads:
                placement='device';reason='sibling contraction emitted with source '+str(sibling_heads[index])
            elif index in sibling_groups:
                from graph_sibling_lowering import emit_sibling_group
                sibling_receipts.append(emit_sibling_group(e,host,sibling_groups[index]));placement='device';reason='shared RHS sibling rows packed into matrix'
            elif index in device_heads:
                placement='device';reason='contraction emitted with saturating consumer '+str(device_heads[index])
            elif index in device_groups:
                emit_fused_device_epilogue(e,host,device_groups[index]);placement='device';reason='fused exact signed saturation readout'
            elif index in fusion_heads:
                placement='host';reason='fused pointwise source emitted with consumer '+str(fusion_heads[index])
            elif index in fusion_groups:
                host.fused_generics(fusion_groups[index]);placement='host';reason='fused single-use f32 pointwise chain'
            elif kind=='func.return':
                for src,name in zip(op.operands,outputs):
                    value=host.values[src]
                    if value.ptr is not e.args[name]:host.copy(value,TensorValue(e.args[name],value.shape,value.dtype))
                placement='host';reason='logical output ABI copy'
            else:
                handled=(device_callback or emit_device)(e,op,host)
                if handled:placement='device';reason='typed supported device form'
                else:
                    # Integer MACs are mandatory accelerator domains even if a
                    # missing adapter prevented default recognition.
                    if prov.get('family')=='contraction' and kind.startswith('linalg.') and any(isinstance(v.type,TensorType) and str(v.type.element_type)=='i8' for v in op.operands):raise ValueError('device contraction adapter missing: '+kind)
                    if kind=='linalg.generic' and any(li._op_name(x)=='arith.muli' for x in op.regions[0].blocks[0].ops):
                        raise ValueError('unrecognized integer MAC source remains a device compiler gap')
                    host.lower(op);placement='host';reason='floating arithmetic or tensor ABI preparation'
            tasks.append({'id':index,'source_op_indices':[index],'kind':kind,'placement':placement,'host_reason':reason if placement=='host' else None,'region_id':region})
    from hashlib import sha256
    operations=[op for op in block.ops if li._op_name(op)!='func.return']
    owner={value:(i,j) for i,op in enumerate(operations) for j,value in enumerate(op.results)}
    values=[]
    for value,name in zip(returned.operands,outputs):
        if value not in owner:raise ValueError('whole program cannot alias an input return')
        i,j=owner[value];values.append({'op_index':i,'result_index':j,'tensor':name})
    plan={'schema':'mixed_program_plan_v1','source_sha256':sha256(cb['source_text'].encode()).hexdigest(),
          'source_op_count':len(operations),'tasks':[{'task_index':0,'kind':'mixed_graph',
          'source_op_indices':list(range(len(operations))),'instruction_start':0,'instruction_end':1,
          'reads':list(inputs),'writes':list(outputs)}],'schedule_instruction_count':1,
          'prologue_instruction_range':[0,0],'epilogue_instruction_range':[1,1],
          'entry_bindings':list(inputs),'source_values':values,'output_bindings':list(outputs),
          'placement_diagnostics':tasks,'static_scratch_bytes':scratch_bytes,
          'preparation_optimizations':host.preparation_optimizations,
          'sibling_packing':sibling_receipts,
          'device_epilogue_fusions':[{'producer_source_op_index':g['producer_index'],
              'consumer_source_op_index':tail,'source_op_indices':[g['producer_index'],tail],
              'output_dtype':'i8','acc_scale':1.0,'relu':g['relu'],
              'proof':'zero i32 init; exact signed scalar MAC; K*16384 <= INT32_MAX; single-use identity signed clamp/trunci',
              'removed_intermediate_bytes':g['m']*g['n']*4}
              for tail,g in device_groups.items()],
          'private_storage_plan':{'mode':reuse_mode,
              'completion_order':'serialized host operations and fenced selected device operations',
              'interval_unit':'full source operation indices; inclusive endpoints',
              'alias_policy':'caller buffers, returned values, views and tensor.empty never pooled',
              'logical_allocation_bytes':logical_storage_bytes,
              'physical_allocation_bytes':scratch_bytes,
              'reused_allocation_count':sum(max(0,len(slot['allocations'])-1) for slot in storage_slots),
              'slots':storage_slots}}

    cb['tensors']={name:cb['tensors'][name] for name in inputs+outputs}
    cb['commands']=[]
    cb['kernel_abi']={'kind':'whole_program','args':[{'tensor':n,'access':'write' if n in outputs else 'read'} for n in inputs+outputs],'outputs':list(outputs)}
    cb['params']['global_program_plan']=plan
    cb['params']['lane_placement']=graph_lane_placement(operations,tasks[:-1])
    cb['compiler_pointer_abi']={'version':1,'arguments':inputs+outputs}
    e._scope={'global_task':0}
    e._globals.extend(globals)
    e._globals.extend(host.declarations.values())
    result=e.module()
    result.verify()
    stream=StringIO();Printer(stream=stream).print_op(result)
    return stream.getvalue()+'\n'


def emit_device(e,op,host):
    """Signed i8 contraction lowering selected from typed scalar semantics."""
    from matmul_lowering import emit_matmul
    from elementwise_lowering import emit_graph_elementwise
    if emit_graph_elementwise(e,op,host):return True
    from graph_movement_lowering import try_emit_graph_movement
    if try_emit_graph_movement(e,op,host):return True
    kind=li._op_name(op)
    if kind not in ('linalg.matmul','linalg.batch_matmul','linalg.generic'):return False
    ins,outs=li._ins_outs(op)
    if len(ins)!=2 or len(outs)!=1 or len(op.results)!=1:return False
    if [str(v.type.element_type) for v in ins]!=['i8','i8'] or str(op.results[0].type.element_type)!='i32':return False
    if kind=='linalg.generic':
        maps=[str(x).replace(' ','') for x in op.indexing_maps]
        expected=['affine_map<(d0,d1,d2)->(d0,d2)>','affine_map<(d0,d1,d2)->(d2,d1)>','affine_map<(d0,d1,d2)->(d0,d1)>']
        if maps!=expected:return False
        names=[li._op_name(x) for x in op.regions[0].blocks[0].ops]
        if names!=['arith.extsi','arith.extsi','arith.muli','arith.addi','linalg.yield']:return False
    initial=outs[0].owner
    zero_init=getattr(initial,'name',None) in ('linalg.fill','tensor.splat') and li._scalar_const(initial.operands[0])==0
    a,b=(host.values[x] for x in ins);c=host.result(op.results[0])
    if a.shape[-1]!=b.shape[-2]:raise ValueError('contraction reduction mismatch')
    m,k,n=a.shape[-2],a.shape[-1],b.shape[-1]
    batches=prod(a.shape[:-2])
    if a.shape[:-2]!=b.shape[:-2]:raise ValueError('batch contraction dimensions mismatch')
    for batch in range(batches):
        ap=e.b.gep(a.ptr,e.b.const(batch*m*k),i8);bp=e.b.gep(b.ptr,e.b.const(batch*k*n),i8)
        from xdsl.dialects.builtin import i32
        cp=e.b.gep(c.ptr,e.b.const(batch*m*n),i32)
        emit_matmul(e,ap,bp,cp,m,k,n,attrs={'output_dtype':'i32'})
    e.fence()
    if not zero_init:
        from elementwise_lowering import emit_elementwise
        old=host.values[outs[0]]
        temp=TensorValue(host.allocate(c.count,c.dtype),c.shape,c.dtype)
        host.copy(c,temp)
        cb={'tensors':{n:{'shape':list(t.shape),'dtype':t.dtype} for n,t in [('lhs',temp),('rhs',old),('dst',c)]}}
        cmd={'opcode':'INTEGER_ADD','operands':{'lhs':'lhs','rhs':'rhs','dst':'dst'},'attributes':{}}
        emit_elementwise(e,cb,cmd,{'lhs':temp.ptr,'rhs':old.ptr,'dst':c.ptr})
        e.fence()
    return True
