"""Pack independent shared-RHS source members into physical operand rows.

Payload moves only through source-bound accelerator DMA. The compact dense
library still owns the complete reduction, clipped tile extents and output
coverage. Disjoint source outputs are scattered from their exact accumulator
row intervals after the shared product completes.
"""
from __future__ import annotations


def lower(ctx,members,*,completion_epochs=None):
    from . import isa
    from .dense import _schedule
    from .resident_rhs import _rewrite, _compute_epoch_plan, _prepare
    from .dense_readout import _pack
    from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
    from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
    if completion_epochs is None:
        from pathlib import Path
        from merlin.compile.tuning_knobs import knob
        completion_epochs=knob('ATTENTION_MEMBER_ROW_EPOCHS',default='off',
            values=('off','on'),root=Path(__file__).resolve().parent.parent)=='on'
    if type(completion_epochs)is not bool:raise ValueError('member row completion choice must be boolean')
    members=tuple(members)
    if not 2<=len(members)<=isa.DIM:return None
    rhs=members[0]['rhs']
    shapes=[]
    for row in members:
        if row['rhs']!=rhs:return None
        a,b,c=(ctx.spec(row[key])for key in ('lhs','rhs','dst'))
        attrs=row.get('attributes',{})
        if (a['dtype']!='i8'or b['dtype']!='i8'or c['dtype']!='i32'
                or len(a['shape'])!=2 or len(b['shape'])!=2):return None
        m,k=a['shape'];rk,n=b['shape']
        if k!=rk or c['shape']!=[m,n]or attrs.get('output_dtype','i32')!='i32':return None
        if any(key in row and row[key]!=v for key,v in (('m',m),('k',k),('n',n))):return None
        if (attrs.get('epilogue')or attrs.get('bias')or attrs.get('accumulator_init')
                or attrs.get('relu')or attrs.get('requant_shift') is not None
                or attrs.get('acc_scale_by_column') is not None
                or attrs.get('block_scale') is not None
                or float(attrs.get('acc_scale',1.0))!=1.0):return None
        shapes.append((m,k,n))
    if len({(k,n)for m,k,n in shapes})!=1:return None
    total=sum(m for m,k,n in shapes)
    if total>isa.DIM:return None
    _,k,n=shapes[0]
    outputs={row['dst']for row in members}
    inputs={row[key]for row in members for key in ('lhs','rhs')}
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    roots={a.get('source')for a in aliases.values()if isinstance(a,dict)}
    if len(outputs)!=len(members)or outputs&inputs or (outputs|inputs)&(set(aliases)|roots):return None
    try:
        plan=plan_dense_rhs_residency(total,k,n,block=isa.DIM,
            operand_rows=isa.SPAD_ROWS,accumulator_rows=isa.ACC_ROWS,rhs_base=isa.DIM)
        if plan is None:return None
        schedule=_schedule(total,k,n,False,False)
        kernel,proof=_rewrite(schedule,plan)
        # This existing admission proves immutable disjoint operand windows and
        # selected in-order execute progress before removing intermediate waits.
        prepared=[(members[0],{},(total,k,n),plan,schedule,kernel,proof)]
        admitted=_compute_epoch_plan(prepared)
        if admitted is None:return None
        prepared,epoch=admitted
        _,_,_,_,_,kernel,proof=prepared[0]
        completion_proof=None
        if completion_epochs:
            completed=_row_completion_epoch(kernel)
            if completed is None:return None
            kernel,completion_proof=completed
    except (ValueError,KeyError,TypeError,OverflowError):return None
    fb=ctx.fb;d=isa.DIM
    original_index=ctx.current_source_index
    lhs_base=proof['retained_lhs']['lhs_base']
    ctx.fence()
    _prepare(ctx,rhs,plan,drain_each=False)
    ctx.emit(*isa.config_ld(k,channel=0))
    start=0
    intervals=[]
    for row,(m,_,_)in zip(members,shapes):
        ctx.current_source_index=row.get('attributes',{}).get('merlin.source_op_index',original_index)
        row_start=start
        intervals.append({'lhs':row['lhs'],'dst':row['dst'],'start':start,'rows':m})
        def transfer(ki,cols):
            ki=fb.const(ki)if isinstance(ki,int)else ki
            offset=fb.mul_i(ki,fb.const(d))
            addr=fb.add_i(fb.const(lhs_base+row_start),fb.mul_i(ki,fb.const(d)))
            ctx.emit(*isa.mvin(ctx.ptr(row['lhs'],offset),addr,m,cols,channel=0,fb=fb))
        if k//d:fb.for_loop(0,k//d,1,lambda ki:transfer(ki,d),retain_loop=True)
        if k%d:transfer(k//d,k%d)
        start+=m
    ctx.fence()
    ctx.current_source_index=members[0].get('attributes',{}).get('merlin.source_op_index',original_index)
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_st(n*4,scale=1.0))
    descriptor=[None]
    def emit(instruction,arguments):
        a={key:bridge.value(value)for key,value in arguments.items()}
        if instruction=='gemmini_preload':
            descriptor[0]=a['cols']
            acc=fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
            ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),_pack(fb,acc,a['rows'],a['cols']))
        elif instruction=='gemmini_compute_preloaded':
            ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),_pack(fb,isa.GARBAGE,a['depth'],descriptor[0]))
        elif instruction=='gemmini_mvout':
            # Finish all ordered MACs before publishing separate source rows.
            if not completion_epochs:ctx.fence()
            for row,interval in zip(members,intervals):
                ctx.current_source_index=row.get('attributes',{}).get('merlin.source_op_index',original_index)
                addr=fb.add_i(a['acc'],fb.const(interval['start']))
                ctx.emit(*isa.mvout(ctx.ptr(row['dst'],a['dram_col']),
                    isa.acc_addr(addr,full=True,fb=fb),interval['rows'],a['cols'],fb=fb))
                if not completion_epochs:ctx.fence()
        elif instruction=='gemmini_member_panel_drain' and completion_epochs:
            ctx.fence()
        else:raise ValueError('unexpected packed member terminal operation '+instruction)
    bridge=FnBuilderIndexAdapter(fb)
    emission=emit_index_program(kernel,bridge,emit)
    schedule.revalidate();ctx.fence();ctx.current_source_index=original_index
    receipt={'mechanism':'shared_rhs_physical_member_rows','products':len(members),
        'folded_geometry':[total,k,n],'row_intervals':intervals,
        'rhs_source_loads':plan.k_tiles*plan.n_tiles,
        'source_kernel_sha256':schedule.kernel.digest(),'packed_kernel_sha256':kernel.digest(),
        'proof':proof,'epoch':epoch,'emission':emission,
        'numerics':'raw i8*i8 -> i32; same per-row K order; exact disjoint output scatter',
        'completion':'immutable disjoint operand DMA then full drain; ordered execute; full drain before each output readout'}
    if completion_proof is not None:
        receipt['completion_epochs']=completion_proof
        receipt['completion']='immutable input DMA drain; ordered compute; drain before and after each private ACC readout phase; no reuse until complete'
    ctx.metadata.setdefault('attention_physical_member_rows',[]).append(receipt)
    return receipt


def _row_completion_epoch(kernel):
    """Qualify store-only completion phases with selected public ordering.

    This never overlaps compute with readout. Every original operation and loop
    remains in place; full drains bracket store-only phases. Between the drains
    inputs/ACC rows/configuration remain immutable and source outputs disjoint.
    """
    import hashlib,json
    from pathlib import Path
    try:
        record=json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
        order=record['ordering']
        digest=hashlib.sha256(json.dumps(order,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    except(OSError,ValueError,KeyError,TypeError):
        return None
    if(record.get('base_facts_sha256')!='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'
            or digest!='ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'):
        return None
    store=record.get('finite_entries',{}).get('store',{})
    if(order.get('completion_order',{}).get('store',{}).get('order')!='in_order'
            or order.get('same_queue',{}).get('store',{}).get('by_class',{}).get('MVOUT')!='completion'
            or store.get('status')!='derived' or type(store.get('value'))is not int or store['value']<=0):
        return None
    candidate,proof=_batch_panel_readouts(kernel)
    return candidate,{**proof,'facts_sha256':record['base_facts_sha256'],
        'canonical_ordering_sha256':digest,'store_queue_entries':store['value'],
        'configuration':'one unchanged CONFIG_ST/CONFIG_EX pair before all compute/readout; no configuration setter moves or is omitted',
        'progress':'pure store phase; no pending load/execute after opening drain; finite store queue backpressure drains independent immutable ACC rows to disjoint source outputs',
        'hazards':'opening full drain resolves execute-to-store RAW; closing full drain resolves readout-to-ACC-reuse WAR; no cross-queue RAW/WAR inference',
        'stack':'same bounded counted loops and common RoCC encoder; no queue-sized host buffering or source-sized stack objects'}


def _batch_panel_readouts(kernel):
    """Complete each maximal immutable output-readout phase once.

    Compact dense proves distinct accumulator/output rows within its tile
    group. This rewrite changes only completion: every original call remains
    in exactly the same loop and order, and the entire store phase drains
    before any later compute can recycle accumulator rows.
    """
    from dataclasses import replace
    from merlin.sched.ir import Call,Loop
    drain='gemmini_member_panel_drain'
    def pure_store(op):
        return (all(pure_store(x)for x in op.body)if isinstance(op,Loop)
                else op.instr=='gemmini_mvout')
    def walk(body):
        out=[];reading=False
        for op in body:
            store=pure_store(op)
            if store and not reading:out.append(Call(drain,()));reading=True
            if not store and reading:out.append(Call(drain,()));reading=False
            out.append(replace(op,body=walk(op.body))if isinstance(op,Loop)and not store else op)
        if reading:out.append(Call(drain,()))
        return tuple(out)
    candidate=replace(kernel,body=walk(kernel.body))
    def strip(body):
        return tuple(replace(op,body=strip(op.body))if isinstance(op,Loop)else op
            for op in body if isinstance(op,Loop)or op.instr!=drain)
    if strip(candidate.body)!=kernel.body:
        raise ValueError('panel readout completion changed ordered source operations')
    return candidate,{'source_kernel_sha256':kernel.digest(),
        'completed_kernel_sha256':candidate.digest(),
        'original_calls_loops_and_order_preserved':True,
        'ownership':'all output rows in compact tile group remain immutable until complete store-phase drain',
        'queues':'selected ready/valid store backpressure; no following compute issued before full completion'}


def lower_panels(ctx,members,*,full_m=False,literal=False,banked=False,regional=False):
    """Retain complete member inputs; group output columns by ACC capacity.

    This alternative keeps distinct source tensors and source output shapes.
    The library compact dense emitter groups output columns into one ordered
    compute phase. Actual complete operand footprints, including all members,
    must fit the selected scratchpad before any command is emitted.
    """
    from . import isa
    from .dense import _adapter
    from .resident_rhs import _rewrite,_compute_epoch_plan,_prepare,_prepare_lhs
    from .dense_readout import _pack
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage,DenseMemoryView,schedule_compact_dense
    from merlin.transforms.dense_planning import DenseProblem,DenseResources
    from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
    from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
    members=tuple(members)
    if not 2<=len(members)<=32:return None
    rhs=members[0]['rhs'];shapes=[]
    for row in members:
        if row['rhs']!=rhs:return None
        a,b,c=(ctx.spec(row[key])for key in ('lhs','rhs','dst'))
        attrs=row.get('attributes',{})
        if a['dtype']!='i8'or b['dtype']!='i8'or c['dtype']!='i32':return None
        if len(a['shape'])!=2 or len(b['shape'])!=2:return None
        m,k=a['shape'];rk,n=b['shape']
        if k!=rk or c['shape']!=[m,n]or attrs.get('output_dtype','i32')!='i32':return None
        if any(key in row and row[key]!=v for key,v in (('m',m),('k',k),('n',n))):return None
        if (attrs.get('epilogue')or attrs.get('bias')or attrs.get('accumulator_init')
                or attrs.get('relu')or attrs.get('requant_shift') is not None
                or attrs.get('acc_scale_by_column') is not None
                or attrs.get('block_scale') is not None
                or float(attrs.get('acc_scale',1.0))!=1.0):return None
        shapes.append((m,k,n))
    if len({(k,n)for m,k,n in shapes})!=1 or sum(m for m,k,n in shapes)<=isa.DIM:return None
    outputs={row['dst']for row in members};inputs={row[key]for row in members for key in ('lhs','rhs')}
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    roots={a.get('source')for a in aliases.values()if isinstance(a,dict)}
    if len(outputs)!=len(members)or outputs&inputs or (outputs|inputs)&(set(aliases)|roots):return None
    d=isa.DIM;_,k,n=shapes[0]
    regional_receipt=None
    if regional:
        kt,nt=(k+d-1)//d,(n+d-1)//d
        lhs_rows=sum((m+d-1)//d*kt*d for m,_,_ in shapes)
        rhs_rows=kt*nt*d
        acc_rows=[(m+d-1)//d*nt*d for m,_,_ in shapes]
        baseline_bytes=sum(m*k+k*n for m,_,_ in shapes)
        retained_bytes=sum(m*k for m,_,_ in shapes)+k*n
        regional_receipt={'class':'shared_multitile_rhs_complete_member_epochs',
            'members':len(members),'rhs_tiles':kt*nt,
            'minimum_member_rows':min(m for m,_,_ in shapes),
            'array_rows':d,
            'lhs_rows':lhs_rows,'rhs_rows':rhs_rows,
            'partition_rows':isa.SPAD_ROWS//2,'member_accumulator_rows':acc_rows,
            'accumulator_capacity_rows':isa.ACC_ROWS,
            'independent_operand_bytes_lower_bound':baseline_bytes,
            'shared_operand_bytes':retained_bytes,
            'saved_rhs_source_bytes':baseline_bytes-retained_bytes,
            'saved_rhs_tile_loads':(len(members)-1)*kt*nt,
            'profitability':'every member fills at least one array row tile and shares a multitile RHS; structural traffic reduction only; modeled latency panel remains required',
            'applied':False}
        ctx.metadata.setdefault('attention_member_regional',[]).append(regional_receipt)
        # Fit establishes legality, independently of the reuse class's cost
        # admission. Small members pay complete-epoch setup/readout boundaries
        # without filling an array row tile. Keep their existing per-member
        # lowering even when a larger RHS offers source traffic savings. This
        # is a DIM-relative policy; ragged K/N and M tails above DIM remain legal.
        legality_refusal=None
        if not full_m or not banked:legality_refusal='regional class requires checked bank-separated complete member epochs'
        elif lhs_rows>isa.SPAD_ROWS//2 or rhs_rows>isa.SPAD_ROWS//2:legality_refusal='complete inputs exceed disjoint operand partitions'
        elif max(acc_rows)>isa.ACC_ROWS:legality_refusal='a complete member output exceeds accumulator capacity'
        profitability_refusal=None
        if min(m for m,_,_ in shapes)<d:profitability_refusal='sub-array-row member retains incumbent'
        elif kt*nt<=1:profitability_refusal='single primitive RHS retains incumbent'
        elif retained_bytes>=baseline_bytes:profitability_refusal='no strict source operand traffic reduction'
        regional_receipt['resource_legal']=legality_refusal is None
        regional_receipt['profitability_eligible']=profitability_refusal is None
        refusal=legality_refusal or profitability_refusal
        if refusal is not None:
            regional_receipt['refusal']=refusal
            return None
    # Pin selected public scratchpad geometry to the same admitted ISA store.
    # facts.json facts.memory_address_spaces[name=scratchpad]: banks=4,
    # depth=4096, row_elems=16. ISA.SPAD_ROWS is declared as 4*4096.
    bank_count,rows_per_bank=4,4096
    if banked and isa.SPAD_ROWS!=bank_count*rows_per_bank:return None
    rhs_partition_base=(bank_count//2)*rows_per_bank
    bank_geometry={'facts_sha256':'b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af',
        'fact_path':'facts.memory_address_spaces[name=scratchpad]',
        'bank_count':bank_count,'rows_per_bank':rows_per_bank,
        'lhs_row_partition':[0,rhs_partition_base],
        'rhs_row_partition':[rhs_partition_base,isa.SPAD_ROWS],
        'mapping_basis':'same admitted contiguous row partitions as ISA and incumbent dense storage',
        'performance':'bank placement hypothesis; no concurrent-read speedup established'}
    column_blocks=min((n+d-1)//d,isa.ACC_ROWS//d,(isa.SPAD_ROWS//2)//d)
    prepared=[]
    row_blocks=[]
    for m,_,_ in shapes:
        row_blocks.append(min((m+d-1)//d,isa.ACC_ROWS//(column_blocks*d))if full_m else 1)
    try:
        for row,(m,k,n),m_blocks in zip(members,shapes,row_blocks):
            plan=plan_dense_rhs_residency(m,k,n,block=d,operand_rows=isa.SPAD_ROWS,
                accumulator_rows=isa.ACC_ROWS,rhs_base=rhs_partition_base if banked else max(row_blocks)*d)
            if plan is None:return None
            problem=DenseProblem(m,k,n,d,d,d,1,1,4)
            storage=DenseLoopStorage(DenseMemoryView('lhs','spad',d,0,m_blocks*d),
                DenseMemoryView('rhs','spad',d,isa.SPAD_ROWS//2,column_blocks*d),
                DenseMemoryView('acc','acc',4*d,0,m_blocks*column_blocks*d))
            resources=DenseResources((m_blocks+column_blocks)*d*d,m_blocks*column_blocks*d*d*4,
                max_m_blocks=m_blocks,max_n_blocks=column_blocks,max_k_blocks=1,
                partial_output_tiles=True,partial_output_blocks=True)
            schedule=schedule_compact_dense(problem,resources,storage,_adapter(),
                name='shared_rhs_member_column_panels',dtypes=('i8','i8','i32'),
                static_budget=128,external_buffers_disjoint=True,verification_timeout_ms=2000)
            kernel,proof=_rewrite(schedule,plan)
            prepared.append((row,dict(row.get('attributes',{})),(m,k,n),plan,schedule,kernel,proof))
        admitted=_compute_epoch_plan(prepared)
        if admitted is None:return None
        prepared,epoch=admitted
        if banked:
            # Preserve the independently checked immutable-input epoch, but put
            # LHS below the half-store split and RHS above it. Both actual row
            # intervals are checked before replacing any retained addresses.
            from .resident_rhs import _retain_lhs
            cursor=0;separated=[]
            for row,attrs,shape,plan,schedule,old_kernel,proof in prepared:
                rows=((shape[0]+d-1)//d)*plan.k_tiles*d
                if cursor+rows>rhs_partition_base:return None
                if plan.rhs_base+plan.k_tiles*plan.n_tiles*d>isa.SPAD_ROWS:return None
                original,_=_rewrite(schedule,plan)
                kernel,retention=_retain_lhs(original,shape,cursor)
                separated.append((row,attrs,shape,plan,schedule,kernel,{**proof,'retained_lhs':retention}))
                cursor+=rows
            prepared=separated
            epoch={**epoch,'lhs_windows':[[x[-1]['retained_lhs']['lhs_base'],
                x[-1]['retained_lhs']['lhs_base']+x[-1]['retained_lhs']['lhs_rows']]for x in prepared],
                'storage_partition':'LHS lower half; RHS upper half; actual ranges disjoint',
                'operand_rows_used':prepared[0][3].rhs_base+prepared[0][3].k_tiles*prepared[0][3].n_tiles*d}
        if full_m:
            completed=[]
            for row,attrs,shape,plan,schedule,kernel,proof in prepared:
                kernel,completion=_batch_panel_readouts(kernel)
                completed.append((row,attrs,shape,plan,schedule,kernel,{**proof,'panel_readout_completion':completion}))
            prepared=completed
        if literal or regional:
            from .resident_rhs import _literal_epoch
            specialized=[]
            # Budget each source member independently. The source group itself
            # already has a 32-member limit; the emitted static group additionally
            # stays below 4096 calls. No source-dependent startup profiling.
            static_calls=0
            literal_refusal=None
            for member in prepared:
                candidate,applied=_literal_epoch([member],max_calls=512,max_rhs_tiles=64)
                if not applied:
                    literal_refusal='member exceeds 512-call or 64-RHS-tile literal budget'
                    break
                static_calls+=candidate[0][-1]['literal_epoch']['ordered_instruction_instances_checked']
                if static_calls>4096:
                    literal_refusal='group exceeds 4096-call literal budget'
                    break
                specialized.extend(candidate)
            if literal_refusal is None:
                prepared=specialized
                literal=True
            elif regional:
                # Bank separation and exact source traffic reduction remain
                # valid beyond the code expansion budget. Keep counted panels.
                regional_receipt['literal_refusal']=literal_refusal
                literal=False
            else:return None
    except (ValueError,KeyError,TypeError,OverflowError):return None
    fb=ctx.fb;original_index=ctx.current_source_index
    ctx.fence();_prepare(ctx,rhs,prepared[0][3],drain_each=False)
    ctx.emit(*isa.config_ld(k,channel=0))
    for row,attrs,(m,k,n),plan,schedule,kernel,proof in prepared:
        ctx.current_source_index=attrs.get('merlin.source_op_index',original_index)
        _prepare_lhs(ctx,row['lhs'],m,k,proof['retained_lhs']['lhs_base'])
    ctx.fence();ctx.emit(*isa.config_ex(dataflow=1));ctx.emit(*isa.config_st(n*4,scale=1.0))
    emitted=[]
    for row,attrs,(m,k,n),plan,schedule,kernel,proof in prepared:
        ctx.current_source_index=attrs.get('merlin.source_op_index',original_index)
        descriptor=[None]
        def emit(instruction,arguments):
            a={key:bridge.value(value)for key,value in arguments.items()}
            if instruction=='gemmini_preload':
                descriptor[0]=a['cols']
                acc=fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
                ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),_pack(fb,acc,a['rows'],a['cols']))
            elif instruction=='gemmini_compute_preloaded':
                ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),_pack(fb,isa.GARBAGE,a['depth'],descriptor[0]))
            elif instruction=='gemmini_member_panel_drain':
                ctx.fence()
            elif instruction=='gemmini_mvout':
                if not full_m:ctx.fence()
                offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(n)),a['dram_col'])
                ctx.emit(*isa.mvout(ctx.ptr(row['dst'],offset),isa.acc_addr(a['acc'],full=True,fb=fb),a['rows'],a['cols'],fb=fb))
                if not full_m:ctx.fence()
            else:raise ValueError('unexpected member panel terminal operation '+instruction)
        bridge=FnBuilderIndexAdapter(fb)
        emitted.append(emit_index_program(kernel,bridge,emit));schedule.revalidate()
    ctx.fence();ctx.current_source_index=original_index
    receipt={'mechanism':'shared_rhs_retained_member_column_panels','products':len(members),
        'member_geometries':[list(x)for x in shapes],'column_blocks':column_blocks,
        'selected_blocks':[[x[4].choice.block.m,x[4].choice.block.n,x[4].choice.block.k]for x in prepared],
        'literal_members':literal,'bank_separated_operands':banked,
        'bank_geometry':bank_geometry if banked else None,
        'row_blocks':row_blocks,'panel_geometry':'capacity_bounded_MxN'if full_m else 'single_M_by_N',
        'rhs_source_loads':prepared[0][3].k_tiles*prepared[0][3].n_tiles,
        'members':[item[-1]for item in prepared],'epoch':epoch,'emission':emitted,
        'numeric':'raw i8*i8 -> i32; source rows and reduction order unchanged',
        'completion':('all disjoint DMA before ordered MxN execute; full completion around maximal immutable output phases'if full_m else 'all disjoint DMA before ordered column-panel execute; full completion around each raw output DMA')}
    if regional_receipt is not None:
        regional_receipt['applied']=True
        regional_receipt['selected']='literal_banked'if literal else'banked'
        receipt['regional_selection']=regional_receipt
    ctx.metadata.setdefault('attention_member_panels',[]).append(receipt)
    return receipt
