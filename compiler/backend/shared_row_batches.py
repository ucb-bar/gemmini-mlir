"""Checked short-row products: separate source tensors, one private array tile.

The caller supplies a proven uninterrupted independent-product interval. The
library schedules the concatenated *logical* rows; this adapter partitions only
its DMA/readout effects onto actual buffers. No concatenated host allocation or
physical source adjacency is required. Full completion closes every transition.
"""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from merlin.compile.tuning_knobs import knob

from merlin.sched.ir import Call, Loop, as_expr
from merlin.sched.ir.kernel import instances
from merlin.sched.ir.expr import evaluate
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from . import isa
from .dense import _adapter
from .dense_pipeline import _retained_schedule
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
from .dense_readout import _pack
from .dense_ready import uniform_store_command


def checked_schedule(specs, *, rhs_transfers='primitive', schedule_selection='incumbent'):
    """Return library schedule, partitioned kernel, receipt; refuse before emit.

    Each spec is (M,K,N). Only a single combined M tile is admitted, so all
    member subrows remain in the same primitive operand/accumulator tile.
    """
    if rhs_transfers not in ('primitive','block64'):
        raise ValueError('unsupported row-batch RHS transfer selection')
    if schedule_selection not in ('incumbent','ready'):
        raise ValueError('unsupported row-batch schedule selection')
    specs = tuple(tuple(s) for s in specs)
    if not 2 <= len(specs) <= isa.DIM or any(len(s) != 3 or any(type(v) is not int or v <= 0 for v in s) for s in specs):
        return None
    total = sum(s[0] for s in specs)
    k, n = specs[0][1:]
    if total > isa.DIM or any(s[1:] != (k,n) for s in specs):
        return None
    # Bound verification effort independently of shape spelling. Emitted loops
    # remain counted, regardless of the concrete verification enumeration.
    kt, nt = (k+isa.DIM-1)//isa.DIM, (n+isa.DIM-1)//isa.DIM
    if kt*nt*(2*len(specs)+4) > 65536:
        return None
    # Reuse the shared full-K retained-LHS pipeline on the combined source.
    # Reuse the caller-selected checked transfer transformation and a bounded
    # existing geometry. This adds no new scheduling/ordering law.
    schedule, refusal = _retained_schedule(total,k,n,False,False,False,'balanced',
        'half',rhs_transfers,'lhs','compute_then_prefetch','compact',schedule_selection=='ready')
    if schedule is None:
        return None
    schedule.revalidate()
    starts=[]; cursor=0
    for m,_,_ in specs:
        starts.append(cursor);cursor+=m
    adapter = _adapter()
    def transform(body):
        output=[]
        for operation in body:
            if isinstance(operation,Loop):
                output.append(replace(operation,body=transform(operation.body)))
                continue
            if operation.instr not in ('gemmini_mvin_lhs','gemmini_mvout'):
                output.append(operation);continue
            original=dict(operation.args)
            for i,((m,_,_),start) in enumerate(zip(specs,starts)):
                a=dict(original);a['rows']=as_expr(m)
                if operation.instr=='gemmini_mvin_lhs':
                    a['addr']=a['addr']+start
                    # Local source row zero, not the combined logical row.
                    a['dram_row']=as_expr(0)
                    instruction=f'row_lhs_{i}'
                else:
                    a['acc']=a['acc']+start
                    a['dram_row']=as_expr(0)
                    instruction=f'row_store_{i}'
                output.append(replace(operation,instr=instruction,args=tuple(a.items())))
        return tuple(output)
    kernel=replace(schedule.kernel,body=transform(schedule.kernel.body))
    # Decode the actual rewritten call fields; verify exact ordered row effects
    # against the checked combined source. All other effects stay identical.
    expected=[]
    load_sites=store_sites=rhs_loads=0
    for call,env in instances(schedule.kernel):
        a={name:evaluate(value,env) for name,value in call.args}
        rhs_loads+=int(call.instr=='gemmini_mvin_weight')
        if call.instr != DRAIN:
            adapter.project(Call(call.instr,tuple((name,as_expr(value)) for name,value in a.items())))
        if call.instr in ('gemmini_mvin_lhs','gemmini_mvout'):
            if a['dram_row'] != 0 or a['rows'] != total:
                raise ValueError('row batch needs one complete primitive M tile')
            lhs=call.instr=='gemmini_mvin_lhs'
            load_sites+=int(lhs);store_sites+=int(not lhs)
            for i,((m,_,_),start) in enumerate(zip(specs,starts)):
                b=dict(a);b['rows']=m;b['dram_row']=0
                b['addr' if lhs else 'acc']+=start
                expected.append((f'row_lhs_{i}' if lhs else f'row_store_{i}',b))
        else:expected.append((call.instr,a))
    actual=[(call.instr,{key:evaluate(value,env) for key,value in call.args}) for call,env in instances(kernel)]
    if actual != expected:
        raise ValueError('row batch changed checked source/effect partition')
    # Check every partition's actual DMA origin, physical subrow, and complete
    # destination ownership rather than relying on pointer naming conventions.
    for instruction,a in actual:
        if instruction.startswith('row_lhs_'):
            member=int(instruction[len('row_lhs_'):]);start=starts[member]
            if not (a['dram_row']==0 and a['rows']==specs[member][0] and a['addr']%isa.DIM==start
                    and 0<=a['dram_col']<k and a['dram_col']+a['cols']<=k):
                raise ValueError('invalid actual LHS row partition')
        elif instruction.startswith('row_store_'):
            member=int(instruction[len('row_store_'):]);start=starts[member]
            if not (a['dram_row']==0 and a['rows']==specs[member][0] and a['acc']%isa.DIM==start
                    and 0<=a['dram_col']<n and a['dram_col']+a['cols']<=n):
                raise ValueError('invalid actual output row partition')
    receipt={'combined_shape':[total,k,n],'member_shapes':[list(s) for s in specs],
        'row_starts':starts,'source_kernel_sha256':schedule.kernel.digest(),
        'partitioned_kernel_sha256':kernel.digest(),'checked_ordered_effects':len(actual),
        'lhs_member_tile_loads':load_sites*len(specs),'rhs_dma_loads':rhs_loads,
        'rhs_primitive_bands':kt*nt,'rhs_transfer_selection':rhs_transfers,
        'rhs_transfer_projection':schedule.realization.get('rhs_block_transfers'),
        'member_tile_stores':store_sites*len(specs),'retained_lhs':True,
        'combined_compute_tiles':kt*nt,'independent_compute_tiles':kt*nt*len(specs),
        'row_projection':'ordered source rows biject onto separate source buffers and disjoint private tile rows',
        'verification':'actual call fields projected against checked combined DenseProblem; no elapsed timing'}
    receipt['schedule_selection']=schedule_selection
    if schedule_selection=='ready':
        from .shared_row_ready import check_ready_partition
        receipt['ready_partition']=check_ready_partition(schedule,kernel,specs,starts,actual)
    return schedule,kernel,receipt


def lower(ctx,members):
    """Emit complete raw products or return None before any context mutation.

    Caller must prove independence and immutable shared RHS for this interval;
    this adapter additionally excludes aliases, output dependencies and all
    arithmetic/readout overrides. Uses each actual LHS/output pointer directly.
    """
    members=tuple(members)
    if not 2<=len(members)<=isa.DIM:return None
    try:
        rhs=members[0]['rhs']
        outputs=[row['dst'] for row in members]
        inputs={row[key] for row in members for key in ('lhs','rhs')}
        aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
        if aliases or len(set(outputs))!=len(outputs) or set(outputs)&inputs or any(row['rhs']!=rhs for row in members):return None
        specs=[]
        for row in members:
            attrs=row.get('attributes',{})
            if (attrs.get('epilogue') or attrs.get('bias') is not None or attrs.get('accumulator_init') is not None
                    or attrs.get('relu') or attrs.get('acc_scale_by_column') is not None
                    or float(attrs.get('acc_scale',1))!=1 or attrs.get('output_dtype','i32')!='i32'
                    or attrs.get('requant_shift') is not None):return None
            a,b,c=(ctx.spec(row[key]) for key in ('lhs','rhs','dst'))
            if (a['dtype'],b['dtype'],c['dtype'])!=('i8','i8','i32'):return None
            if len(a['shape'])!=2 or len(b['shape'])!=2:return None
            m,k=a['shape'];bk,n=b['shape']
            if k!=bk or c['shape']!=[m,n] or any(key in row and row[key]!=value for key,value in (('m',m),('k',k),('n',n))):return None
            specs.append((m,k,n))
        transfers=knob('SHARED_RHS_TRANSFERS',default='primitive',values=('primitive','block64'),
            root=Path(__file__).resolve().parents[1])
        selection=knob('SHARED_ROW_BATCH_SCHEDULE',default='incumbent',values=('incumbent','ready'),
            root=Path(__file__).resolve().parents[1])
        prepared=checked_schedule(specs,rhs_transfers=transfers,schedule_selection=selection)
        if prepared is None:return None
        schedule,kernel,proof=prepared
        identity=sha256(json.dumps({'members':members,'specs':specs},sort_keys=True).encode()).hexdigest()
    except (ValueError,KeyError,TypeError,OverflowError):return None
    fb=ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(specs[0][1],channel=0))
    ctx.emit(*isa.config_ld(specs[0][2],channel=1))
    # All raw outputs share this immutable full-width readout configuration.
    # Configure once while drained; never reconfigure between split stores.
    ctx.emit(*uniform_store_command({},specs[0][2],'i32'))
    columns=[None]
    def emit(instruction,arguments):
        a={key:bridge.value(value) for key,value in arguments.items()}
        if instruction == DRAIN:
            ctx.fence();return
        if instruction.startswith('row_lhs_'):
            member=int(instruction[len('row_lhs_'):]);row=members[member]
            ctx.current_source_index=row.get('attributes',{}).get('merlin.source_op_index')
            offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(specs[member][1])),a['dram_col'])
            ctx.emit(isa.FUNCT['mvin'],ctx.ptr(row['lhs'],offset),_pack(fb,a['addr'],a['rows'],a['cols']))
        elif instruction=='gemmini_mvin_weight':
            offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(specs[0][2])),a['dram_col'])
            ctx.emit(isa.FUNCT['mvin2'],ctx.ptr(rhs,offset),_pack(fb,a['addr'],a['rows'],a['cols']))
        elif instruction=='gemmini_preload':
            columns[0]=a['cols']
            acc=fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
            ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),_pack(fb,acc,a['rows'],a['cols']))
        elif instruction=='gemmini_compute_preloaded':
            ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),_pack(fb,isa.GARBAGE,a['depth'],columns[0]))
        elif instruction.startswith('row_store_'):
            member=int(instruction[len('row_store_'):]);row=members[member]
            ctx.current_source_index=row.get('attributes',{}).get('merlin.source_op_index')
            offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(specs[member][2])),a['dram_col'])
            ctx.emit(*isa.mvout(ctx.ptr(row['dst'],offset),isa.acc_addr(a['acc'],full=True,fb=fb),
                a['rows'],a['cols'],fb=fb))
        else:raise ValueError('unknown row batch terminal instruction '+instruction)
    bridge=FnBuilderIndexAdapter(fb)
    emission=emit_index_program(kernel,bridge,emit)
    schedule.revalidate()
    ctx.fence()
    receipt={'mechanism':'checked_shared_rhs_row_batches','source_identity':identity,
        'products':len(members),'rhs':rhs,'proof':proof,'emission':emission,
        'completion':'all checked retained-pipeline drains preserved; split DMA and stores touch disjoint rows within the original complete effect; immutable raw-i32 store configuration',
        'scope':'source/effect checked raw i32 row batching; independent target timing required'}
    ctx.metadata.setdefault('shared_row_batches',[]).append(receipt)
    return receipt
