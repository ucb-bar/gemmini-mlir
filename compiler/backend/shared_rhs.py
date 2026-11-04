"""Complete shared-RHS products with actual checked full-K panel schedules.

The shared scheduler owns every source tile, retained RHS address and member
block decision. Its plan is consumed by scheduling, and its resulting effects
are the effects emitted below. This zero-seeded raw-i32 route retains the
existing source grouping, output ABI and conservative completion fallback.
"""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json

from merlin.compile.scheduling.block_schedule import Load, Store
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_shared_rhs import schedule_dense_shared_rhs
from merlin.compile.scheduling.dense_shared_rhs_check import DenseRHSReuseLaw
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from merlin.transforms.dense_planning import DenseProblem, DenseResources
from merlin.sched.ir import Call, Loop, Ptr, as_expr
from . import isa
from .dense import _adapter
from .dense_readout import store_tile, _pack
from .shared_products import _epoch_index_kernel, _literal_epoch


def _ready_admissions(problems,blocks,adapter):
    """Bind causal readout to the same selected ACC waits as dense ready."""
    from merlin.compile.scheduling.dense_panel_pipeline import (
        DenseReadyReadoutAdmission,DenseRetainedReductionAdmission)
    record=json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    ordering=record['ordering']
    digest=sha256(json.dumps(ordering,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (digest!='ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'
            or record['base_facts_sha256']!='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'
            or ordering['status']!='derived'):
        raise ValueError('shared ready lacks selected ordering facts')
    cross={(x['older'],x['younger'],x['space']):x for x in ordering['cross_queue']}
    for pair in (('PRELOAD.c_addr','MVOUT.acc_addr'),('MVOUT.acc_addr','PRELOAD.c_addr')):
        wait=cross.get((*pair,'accumulator'),{})
        if wait.get('overlap')!='completion' or wait.get('tracked') is not True:
            raise ValueError('shared ready lacks tracked producer/store completion')
    if (ordering['completion_order']['execute']['order']!='in_order'
            or ordering['completion_order']['store']['order']!='in_order'
            or ordering['same_queue']['store']['release']!='completion'):
        raise ValueError('shared ready lacks ordered ACC commits and store progress')
    for lane in ('execute','load','store'):
        entry=record['finite_entries'][lane]
        if entry.get('status')!='derived' or type(entry.get('value')) is not int or entry['value']<1:
            raise ValueError('shared ready lacks selected finite queue progress')
    for name in ('PRELOAD.c_addr','MVOUT.acc_addr'):
        if ordering['extents'].get(name,{}).get('status')!='derived':
            raise ValueError('shared ready lacks actual ACC descriptor extents')
    identity='selected-shared-causal-acc-readout:'+digest
    epochs=[];readouts=[]
    for p,b in zip(problems,blocks,strict=True):
        epochs.append(DenseRetainedReductionAdmission(adapter.configuration_identity,
            adapter.numerical_policy_identity,identity,max(1,b.m*b.k*b.n),
            max(1,p.tiles[1]*p.tiles[2],b.m*b.k),2,8,
            max_issued_readouts=b.m*b.n,readout_configuration='ordered_snapshot_before_store_issue'))
        readouts.append(DenseReadyReadoutAdmission(adapter.configuration_identity,
            adapter.numerical_policy_identity,identity,identity+':uniform-CONFIG_ST-through-final-drain',
            b.m*b.n,1))
    return tuple(epochs),tuple(readouts)


def checked_schedule(specs, source_identity, *, ready=False):
    specs = tuple(tuple(map(int, spec)) for spec in specs)
    if not 2 <= len(specs) <= 32 or any(len(s) != 3 or min(s) <= 0 for s in specs):
        return None
    d = isa.DIM
    k,n = specs[0][1:]
    if any(s[1:] != (k,n) for s in specs):
        return None
    kt,nt = (k+d-1)//d,(n+d-1)//d
    rhs_rows = kt*nt*d
    bm = min(4,max((m+d-1)//d for m,_,_ in specs))
    lhs_rows = bm*kt*d
    if lhs_rows+rhs_rows > isa.SPAD_ROWS:
        return None
    order = json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if order.get('completion_order',{}).get('execute',{}).get('order') != 'in_order':
        return None
    adapter = _adapter()
    identity = 'shared-rhs-selected-ordered-private-member:' + sha256(
        Path(__file__).read_bytes()+Path(__file__).with_name('dense_ordering.json').read_bytes()).hexdigest()
    adapter = replace(adapter, qualification_identity=identity)
    law = DenseRHSReuseLaw(adapter.configuration_identity,adapter.numerical_policy_identity,
        identity,identity,rhs_unchanged_by_member_operations=True,
        independent_member_interleaving=True,suppressed_loads_no_faults=True,
        suppressed_loads_no_extra_observations=True)
    shared = DenseMemoryView('rhs','spad',d,lhs_rows,rhs_rows)
    storages = tuple(DenseLoopStorage(DenseMemoryView(f'lhs{i}','spad',d,0,lhs_rows),
        shared,DenseMemoryView(f'acc{i}','acc',d*4,0,isa.ACC_ROWS)) for i in range(len(specs)))
    problems = tuple(DenseProblem(m,k,n,d,d,d,1,1,4) for m,_,_ in specs)
    resources = DenseResources(isa.SPAD_ROWS*d,isa.ACC_ROWS*d*4,
        max_m_blocks=bm,max_n_blocks=4,max_k_blocks=kt,
        partial_output_tiles=True,partial_output_blocks=True)
    def encode(member, operation):
        call = adapter.encode(operation)
        args = call.args
        if isinstance(operation,Load):
            name,stride = ('rhs',n) if operation.role == 'weight' else (f'lhs{member}',k)
            args += (('external_pointer',Ptr(name,as_expr(operation.dram_row)*stride+operation.dram_col)),)
        elif isinstance(operation,Store):
            args += (('external_pointer',Ptr(f'output{member}',
                (as_expr(operation.dram_row)*n+operation.dram_col)*4)),)
        return Call(('shared_common_' if member is None else f'shared_member{member}_')+call.instr,args)
    def project(call):
        if call.instr.startswith('shared_common_'):
            member,mnemonic = None,call.instr[len('shared_common_'):]
        else:
            index,mnemonic = call.instr[len('shared_member'):].split('_',1)
            member = int(index)
        return member,adapter.project(Call(mnemonic,call.args))
    result = schedule_dense_shared_rhs(problems,resources,storages,adapter,
        encode_member=encode,project_member=project,source_identity=source_identity,
        name='shared_rhs_panels',dtypes=('i8','i8','i32'),static_budget=512,
        external_buffers_disjoint=True,retention_law=law,
        verification_timeout_ms=5000,max_members=32)
    if ready and result is not None:
        from merlin.compile.scheduling.dense_shared_rhs_ready import schedule_dense_shared_rhs_ready
        epochs,readouts=_ready_admissions(problems,result.blocks,adapter)
        result=schedule_dense_shared_rhs_ready(result,problems,resources,storages,adapter,
            encode_member=encode,project_member=project,retention_law=law,
            epoch_admissions=epochs,readout_admissions=readouts)
    return result


def lower(ctx,members):
    """Return None before mutation for every unsupported source/numeric form."""
    members = tuple(members)
    if not 2 <= len(members) <= 32:
        return None
    rhs = members[0]['rhs']
    names = {row[key] for row in members for key in ('lhs','rhs','dst')}
    outputs = [row['dst'] for row in members]
    inputs = {row[key] for row in members for key in ('lhs','rhs')}
    aliases = ctx.cb.get('params',{}).get('tensor_aliases',{})
    sources = {a.get('source') for a in aliases.values() if isinstance(a,dict)}
    if (len(set(outputs)) != len(outputs) or set(outputs)&inputs or names&(set(aliases)|sources)
            or any(row['rhs'] != rhs for row in members)):
        return None
    specs = []
    for row in members:
        attrs = row.get('attributes',{})
        a,b,c = (ctx.spec(row[key]) for key in ('lhs','rhs','dst'))
        if a['dtype'] != 'i8' or b['dtype'] != 'i8' or c['dtype'] != 'i32':
            return None
        if (attrs.get('epilogue') or attrs.get('bias') or attrs.get('accumulator_init')
                or attrs.get('relu') or attrs.get('acc_scale_by_column') is not None
                or float(attrs.get('acc_scale',1)) != 1
                or attrs.get('output_dtype','i32') != 'i32' or attrs.get('requant_shift') is not None):
            return None
        if len(a['shape']) != 2 or len(b['shape']) != 2:
            return None
        m,k = a['shape'];bk,n = b['shape']
        if (k != bk or c['shape'] != [m,n]
                or any(key in row and row[key] != value for key,value in (('m',m),('k',k),('n',n)))):
            return None
        specs.append((m,k,n))
    source_identity = sha256(json.dumps({'members':members,'specs':specs},sort_keys=True).encode()).hexdigest()
    try:
        from merlin.compile.tuning_knobs import knob
        ready_selection=knob('SHARED_READY_READOUT',default='off',values=('off','ready'),
            root=Path(__file__).resolve().parents[1])
        ready_receipt={'selection':ready_selection,'applied':False}
        try:
            schedule = checked_schedule(specs,source_identity,ready=ready_selection=='ready')
        except ValueError as refusal:
            if ready_selection!='ready':raise
            ready_receipt['reason']=str(refusal)
            schedule=checked_schedule(specs,source_identity)
        if schedule is None:
            return None
        schedule.revalidate()
        # The checked library synchronous stream carries full per-effect
        # drains. The same selected-order epoch mapper used for shared LHS
        # groups replaces only those edges; every data effect is retained.
        def data(body):
            return tuple(replace(op,body=data(op.body)) if isinstance(op,Loop) else op
                for op in body if isinstance(op,Loop) or op.instr != DRAIN)
        ready = hasattr(schedule,'original')
        if ready:
            ready_receipt={'selection':'ready','applied':True,**schedule.realization}
            def index_body(body):
                return tuple(replace(op,body=index_body(op.body)) if isinstance(op,Loop)
                    else Call('shared_epoch_drain',()) if op.instr==DRAIN
                    else replace(op,args=tuple((k,v) for k,v in op.args if not isinstance(v,Ptr)))
                    for op in body)
            kernel=replace(schedule.kernel,body=index_body(schedule.kernel.body))
            epochs={'mechanism':'checked shared causal ready epochs','inter_member_drain':True}
        else:
            source_kernel = replace(schedule.kernel,body=data(schedule.kernel.body))
            kernel,epochs = _epoch_index_kernel(source_kernel,readout_epochs=True)
        transfers = None
        transfer_receipt = {'selection':'primitive','applied':False}
        if knob('SHARED_RHS_TRANSFERS',default='primitive',values=('primitive','block64'),
                root=Path(__file__).resolve().parents[1]) == 'block64':
            from .shared_rhs_transfers import select_rhs_block_transfers
            try:
                transfers,transfer_receipt = select_rhs_block_transfers(schedule,kernel,specs[0][2])
            except ValueError as refusal:
                # Unsupported mapping keeps the identical checked shared
                # schedule, rather than falling back to independent products.
                transfer_receipt = {'applied':False,'reason':str(refusal)}
            transfer_receipt = {'selection':'block64',**transfer_receipt}
            if transfers is not None:
                kernel = transfers.kernel
        kernel,literal = _literal_epoch(kernel)
        readout_configuration = None
        readout_receipt = {'selection':'explicit','applied':False}
        if ready:
            readout_receipt={'selection':'uniform_causal','applied':True,
                'actual_config_st_words':isa.config_st(specs[0][2]*4,scale=1.0,activation=0),
                'entry':'after full device drain','lifetime':'unchanged through final full drain',
                'terminal':'same store_tile; exact one CONFIG_ST suppressed and one actual MVOUT required'}
        if not ready and knob('SHARED_READOUT_CONFIG',default='explicit',values=('explicit','counted'),
                root=Path(__file__).resolve().parents[1]) == 'counted':
            from .shared_readout_configs import select_raw_readout_configuration
            try:
                readout_configuration = select_raw_readout_configuration(kernel,specs[0][2])
            except ValueError as refusal:
                readout_receipt = {'selection':'counted','applied':False,'reason':str(refusal)}
            if readout_configuration is not None:
                kernel = readout_configuration.kernel
                readout_receipt = {'selection':'counted',**readout_configuration.report}
    except (ValueError,KeyError,TypeError,OverflowError):
        return None
    fb = ctx.fb
    ctx.fence()
    ctx.emit(*isa.config_ex(dataflow=1))
    ctx.emit(*isa.config_ld(specs[0][1],channel=0))
    ctx.emit(*isa.config_ld(specs[0][2],channel=1))
    if ready:
        # The dedicated causal consumer assumes this exact uniform state.
        # Every real store still runs through the checked common terminal.
        ctx.emit(*isa.config_st(specs[0][2]*4,scale=1.0,activation=0))
    columns = {}
    def emit(instruction,arguments):
        if instruction == 'shared_epoch_drain':
            ctx.fence();return
        if instruction == 'shared_raw_readout_config':
            ctx.emit(isa.FUNCT['config'],bridge.value(arguments['rs1']),bridge.value(arguments['rs2']))
            return
        if instruction.startswith('shared_config_rhs_'):
            ctx.emit(*isa.config_ld(specs[0][2],channel=1))
            return
        if instruction.startswith('shared_common_'):
            member,mnemonic = None,instruction[len('shared_common_'):]
        else:
            index,mnemonic = instruction[len('shared_member'):].split('_',1)
            member = int(index)
        row = members[0 if member is None else member]
        ctx.current_source_index = row.get('attributes',{}).get('merlin.source_op_index')
        m,k,n = specs[0 if member is None else member]
        a = {key:bridge.value(value) for key,value in arguments.items()}
        if mnemonic.startswith('gemmini_mvin_'):
            left = mnemonic == 'gemmini_mvin_lhs'
            stride = k if left else n
            offset = fb.add_i(fb.mul_i(a['dram_row'],fb.const(stride)),a['dram_col'])
            ctx.emit(isa.FUNCT['mvin'] if left else isa.FUNCT['mvin2'],
                ctx.ptr(row['lhs'] if left else rhs,offset),_pack(fb,a['addr'],a['rows'],a['cols']))
        elif mnemonic == 'gemmini_preload':
            columns[member] = a['cols']
            acc = fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
            ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),_pack(fb,acc,a['rows'],a['cols']))
        elif mnemonic == 'gemmini_compute_preloaded':
            ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),
                _pack(fb,isa.GARBAGE,a['depth'],columns[member]))
        elif mnemonic == 'gemmini_mvout':
            # Every member has the identical raw-i32 store configuration.
            # The checked source schedule supplies the full distinct output
            # coverage, and the completion mapper closes this pure readout
            # nest before any subsequent accumulator overwrite. No pending
            # store can observe changed scale, activation or row stride.
            terminal = store_tile
            store_context=ctx
            if ready:
                from .shared_readout_configs import _ExactStoreTerminal
                store_context=_ExactStoreTerminal(ctx,isa.config_st(n*4,scale=1.0,activation=0))
            elif readout_configuration is not None:
                from .shared_readout_configs import store_tile as configured_store
                terminal = lambda *args,**kwargs: configured_store(readout_configuration,*args,**kwargs)
            terminal(store_context,row['dst'],a['acc'],a['rows'],a['cols'],n,a['dram_row'],a['dram_col'],
                row.get('attributes',{}),completion=lambda phase: None)
            if ready:store_context.finish()
        else:
            raise ValueError('unsupported checked shared RHS operation '+mnemonic)
    bridge = FnBuilderIndexAdapter(fb)
    emission = emit_index_program(kernel,bridge,emit)
    schedule.revalidate()
    if transfers is not None:
        transfers.revalidate()
    if readout_configuration is not None:
        readout_configuration.revalidate()
    ctx.fence()
    receipt = {'mechanism':'checked_shared_rhs_panels','rhs':rhs,'products':len(members),
        'schedule_consumed':True,'schedule_sha256':sha256(schedule.plan_text.encode()).hexdigest(),
        'checked_kernel_sha256':schedule.kernel.digest(),'emitted_kernel_sha256':kernel.digest(),
        'source_identity':source_identity,'blocks':[{'m':b.m,'k':b.k,'n':b.n} for b in schedule.blocks],
        'epochs':epochs,'literal_epoch':literal,'emission':emission,'rhs_transfers':transfer_receipt,
        'readout_configuration':readout_receipt,
        'ready_readout':ready_receipt,
        'rhs_source_loads':((specs[0][1]+isa.DIM-1)//isa.DIM)*((specs[0][2]+isa.DIM-1)//isa.DIM),
        'readout_completion':('causal final-K producer/store frontier; full drain before every member/window reuse; uniform raw CONFIG_ST'
            if ready else 'full drain after every pure checked store nest; immutable accumulator rows and identical raw-i32 configuration within nest'),
        'scope':'checked full-K shared source schedule; independent target execution required'}
    ctx.metadata.setdefault('shared_rhs_panels',[]).append(receipt)
    return receipt
