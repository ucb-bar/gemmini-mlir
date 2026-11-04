"""Bounded residual DMA epochs over disjoint accumulator row windows.

The source numeric screen remains owned by elementwise.lower_residual. This
adapter changes only ordering and physical row ownership, using the same ISA
helpers and shared counted schedule emitter as the other device lowerers.
"""
from __future__ import annotations

from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from merlin.sched.ir import Call, Loop, Var, as_expr, check_structure
from merlin.sched.ir.kernel import Kernel
from . import isa


PHASES=('residual_epoch_seed','residual_epoch_add','residual_epoch_store')


def _check_sequence(kernel):
    """Check actual loop/call ordering, including every completion edge."""
    def has_drain(items):
        return any(has_drain(x.body) if isinstance(x,Loop) else x.instr=='residual_epoch_drain' for x in items)
    def scan(items,state=0,active=False):
        for item in items:
            if isinstance(item,Loop):
                if item.extent<=0 or item.iter_args or item.yields or item.results:
                    raise ValueError('unsupported residual epoch loop domain')
                if has_drain(item.body):
                    if state or active:raise ValueError('residual epoch loop crosses a phase boundary')
                    end,unfinished=scan(item.body)
                    if end or unfinished:raise ValueError('residual epoch loop lacks closing drain')
                else:
                    end,unfinished=scan(item.body,state,active)
                    if end!=state or not unfinished:raise ValueError('residual data loop crosses a completion edge')
                    active=True
            elif item.instr=='residual_epoch_drain':
                if item.args or not active:raise ValueError('residual drain has no completed phase')
                state=(state+1)%3;active=False
            else:
                if item.instr!=PHASES[state] or len(item.args)!=4 or {k for k,_ in item.args}!={'offset','acc','rows','cols'}:
                    raise ValueError('residual phase order or call operands changed')
                active=True
        return state,active
    if scan(kernel.body)!=(0,False):raise ValueError('residual schedule lacks a closing phase drain')


def _check_partition_finite(kernel,count,*,max_instances=16384):
    """Exact bounded proof when the optional SMT package is unavailable.

    The budget bounds compiler proof work, not source geometry or scheduling.
    Larger or unsupported domains decline before any device emission.
    """
    from merlin.sched.ir.kernel import instances
    from merlin.sched.ir.expr import evaluate
    def size(items):
        return sum(x.extent*size(x.body) if isinstance(x,Loop) else 1 for x in items)
    actual_size=size(kernel.body)
    if actual_size>max_instances:raise ValueError('residual finite proof command budget exceeded')
    state=0;phases=[[],[],[]];cursor=0;epochs=0
    for call,env in instances(kernel):
        if call.instr=='residual_epoch_drain':
            if not phases[state]:raise ValueError('empty residual phase')
            if state==2:
                if not phases[0]==phases[1]==phases[2]:raise ValueError('residual phase operands differ')
                windows=[]
                for a in phases[2]:
                    offset,acc,rows,cols=(a[key] for key in ('offset','acc','rows','cols'))
                    if not (0<rows<=isa.DIM and 0<cols<=isa.DIM and (cols==isa.DIM or rows==1)
                            and 0<=acc and acc+rows<=isa.ACC_ROWS and offset==cursor):
                        raise ValueError('residual finite source or ACC bound failed')
                    if any(acc<hi and lo<acc+rows for lo,hi in windows):
                        raise ValueError('residual finite ACC windows overlap')
                    windows.append((acc,acc+rows));cursor+=rows*cols
                phases=[[],[],[]];epochs+=1
            state=(state+1)%3
        else:
            if call.instr!=PHASES[state]:raise ValueError('residual finite phase order changed')
            phases[state].append({key:evaluate(value,env) for key,value in call.args})
    if state or any(phases) or cursor!=count:raise ValueError('residual finite source coverage is incomplete')
    return {'proof_kind':'exact_bounded_finite','checked_instruction_instances':actual_size,
        'max_instruction_instances':max_instances,'checked_epochs':epochs,
        'proof':'actual ordered calls: seed/drain/add/drain/store/drain; identical phase operands; adjacent exact source coverage; bounded disjoint ACC windows per epoch'}


def _check_partition(kernel,count,capacity):
    """Prove coverage/ownership from the actual counted call operands."""
    _check_sequence(kernel)
    try:
        import z3
    except ModuleNotFoundError:
        return _check_partition_finite(kernel,count)
    from merlin.sched.ir import Const, Sum, Scaled
    def projection(items,instruction):
        result=[]
        for item in items:
            if isinstance(item,Loop):
                body=projection(item.body,instruction)
                if body:result.append((item.var,item.extent,body))
            elif item.instr==instruction:result.append(item.args)
        return tuple(result)
    phases=[projection(kernel.body,instruction) for instruction in PHASES]
    if not phases[0] or not phases[0]==phases[1]==phases[2]:
        raise ValueError('residual phases changed source/local projection')
    sites=[]
    def collect(items,scopes=(),multiplicity=1):
        total=0
        for item in items:
            if isinstance(item,Loop):
                total+=collect(item.body,(*scopes,(item.var,item.extent)),multiplicity*item.extent)
            elif item.instr=='residual_epoch_store':
                a=dict(item.args)
                if not isinstance(a['rows'],Const) or not isinstance(a['cols'],Const):
                    raise ValueError('residual tile extents must be constant')
                total+=multiplicity*a['rows'].value*a['cols'].value
                sites.append((a,scopes))
        return total
    if collect(kernel.body)!=count:raise ValueError('residual source element count changed')
    def expr(value,env):
        if isinstance(value,Const):return z3.IntVal(value.value)
        if isinstance(value,Var):return env[value.name]
        if isinstance(value,Sum):return sum(expr(x,env) for x in value.terms)
        if isinstance(value,Scaled):return expr(value.expr,env)*value.factor
        raise ValueError('unsupported residual address expression')
    def instantiate(site,label):
        args,scopes=site
        env={name:z3.Int(label+name) for name,_ in scopes}
        domain=[z3.And(env[name]>=0,env[name]<extent) for name,extent in scopes]
        return {key:expr(value,env) for key,value in args.items()},env,domain
    queries=0
    def prove(constraints):
        nonlocal queries
        solver=z3.Solver();solver.set(timeout=2000);solver.add(*constraints);queries+=1
        if solver.check()!=z3.unsat:raise ValueError('residual source coverage or ACC ownership is unproved')
    for i,site in enumerate(sites):
        a,ea,da=instantiate(site,'a_')
        length=a['rows']*a['cols']
        prove([*da,z3.Or(a['offset']<0,a['offset']+length>count,
            a['rows']<=0,a['rows']>isa.DIM,a['cols']<=0,a['cols']>isa.DIM,
            z3.And(a['cols']!=isa.DIM,a['rows']!=1),
            a['acc']<0,a['acc']+a['rows']>isa.ACC_ROWS,
            a['offset']%capacity!=a['acc']*isa.DIM)])
        for j in range(i,len(sites)):
            b,eb,db=instantiate(sites[j],'b_')
            distinct=z3.Or(*(ea[name]!=eb[name] for name,_ in site[1])) if i==j else z3.BoolVal(True)
            prove([*da,*db,distinct,a['offset']<b['offset']+b['rows']*b['cols'],
                b['offset']<a['offset']+length])
            prove([*da,*db,distinct,a['offset']/capacity==b['offset']/capacity,
                a['acc']<b['acc']+b['rows'],b['acc']<a['acc']+a['rows']])
    return {'proof_kind':'symbolic_smt','solver_queries':queries,'phase_order':'seed/drain/add/drain/store/drain checked on actual loop/call sequence',
        'proof':'actual counted call operands: identical phases, exact element count, bounded disjoint source intervals and disjoint ACC rows inside every epoch'}


def residual_epoch_schedule(count):
    """Build a source-exact flattened partition with three completed phases."""
    if type(count) is not int or count <= isa.DIM*isa.DIM:
        return None
    d=isa.DIM
    slots=isa.ACC_ROWS//d
    if slots<2:return None
    width=d*d
    capacity=slots*width
    full,tail=divmod(count,capacity)
    def epoch(base,elements):
        tiles,remainder=divmod(elements,width)
        rows,cols=divmod(remainder,d)
        def phase(instruction):
            result=[]
            def tile(offset,acc,height,width):
                return Call(instruction,tuple((key,as_expr(value)) for key,value in
                    (('offset',base+offset),('acc',acc),('rows',height),('cols',width))))
            if tiles:
                i=Var('tile')
                result.append(Loop('tile',tiles,(tile(i*width,i*d,d,d),)))
            if rows:result.append(tile(tiles*width,tiles*d,rows,d))
            if cols:result.append(tile(tiles*width+rows*d,tiles*d+rows,1,cols))
            result.append(Call('residual_epoch_drain',()))
            return tuple(result)
        return tuple(x for op in ('residual_epoch_seed','residual_epoch_add','residual_epoch_store') for x in phase(op))
    body=[]
    if full:body.append(Loop('epoch',full,epoch(Var('epoch')*capacity,capacity)))
    if tail:body.extend(epoch(as_expr(full*capacity),tail))
    kernel=Kernel('residual_epochs',(),tuple(body))
    if check_structure(kernel):raise ValueError('invalid residual epoch counted schedule')
    # Every phase has identical source and local partitions. The regular tiles
    # occupy [tile*DIM,(tile+1)*DIM); the two tail regions follow those tiles.
    # Their maximum row is ceil(elements/DIM), bounded by slots*DIM. Source
    # epochs occupy adjacent [epoch*capacity,(epoch+1)*capacity) intervals.
    checked=_check_partition(kernel,count,capacity)
    return kernel,{'elements':count,'accumulator_rows':isa.ACC_ROWS,'checked_partition':checked,
        'tile_elements':width,'tiles_per_full_epoch':slots,'epoch_elements':capacity,
        'full_epochs':full,'tail_elements':tail,'epochs':full+int(bool(tail)),
        'source_partition':'adjacent flattened epochs; identical seed/add/store element intervals',
        'local_ownership':'tile j owns [j*DIM,(j+1)*DIM); partial rows and final partial row follow full tiles',
        'kernel_sha256':kernel.digest()}


def lower_residual_epochs(ctx,lhs,rhs,dst,count,proof,activation):
    """Return an application receipt, or decline before any emission."""
    selection=knob('RESIDUAL_EPOCHS',default='off',values=('off','on'),
                   root=Path(__file__).resolve().parent.parent)
    if selection!='on' or ctx.cb.get('params',{}).get('tensor_aliases'):
        return None
    try:
        result=residual_epoch_schedule(count)
    except ValueError as exc:
        ctx.metadata.setdefault('residual_epoch_refusals',[]).append(str(exc))
        return None
    if result is None:return None
    kernel,partition=result
    ctx.fence()
    ctx.emit(*isa.config_ld(isa.DIM,channel=0,scale=proof['lhs_load_f32'],shrink=True))
    ctx.emit(*isa.config_ld(isa.DIM,channel=1,scale=proof['rhs_load_f32'],shrink=True))
    ctx.emit(*isa.config_st(isa.DIM,scale=proof['readout_scale_f32'],activation=activation))
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(instruction,arguments):
        if instruction=='residual_epoch_drain':
            ctx.fence();return
        a={key:bridge.value(value) for key,value in arguments.items()}
        if instruction in ('residual_epoch_seed','residual_epoch_add'):
            add=instruction=='residual_epoch_add'
            address=isa.acc_addr(a['acc'],accumulate=add,fb=ctx.fb)
            ctx.emit(*isa.mvin(ctx.ptr(rhs if add else lhs,a['offset']),address,
                rows=a['rows'],cols=a['cols'],channel=int(add),fb=ctx.fb))
        elif instruction=='residual_epoch_store':
            ctx.emit(*isa.mvout(ctx.ptr(dst,a['offset']),isa.acc_addr(a['acc'],fb=ctx.fb),
                rows=a['rows'],cols=a['cols'],fb=ctx.fb))
        else:raise ValueError('unknown residual epoch operation '+instruction)
    emission=emit_index_program(kernel,bridge,emit)
    if residual_epoch_schedule(count)[0]!=kernel:
        raise ValueError('residual source partition changed during emission')
    receipt={'mechanism':'disjoint_accumulator_residual_epochs','applied':True,
        'lhs':lhs,'rhs':rhs,'dst':dst,'partition':partition,'numeric_screen':proof,
        'emission':emission,'completion':'all seed DMA; full drain; all accumulating RHS DMA; full drain; all readouts; full drain before any ACC reuse',
        'alias_contract':'explicit partial aliases refused; exact same-name in-place output safe because both reads finish before writes to that epoch',
        'resource_contract':'ACC_ROWS/DIM disjoint tile windows; full completion closes every cross-channel dependency and ownership transition; finite DMA queues apply backpressure',
        'facts_sha256':'b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'}
    ctx.metadata.setdefault('residual_epochs',[]).append(receipt)
    return receipt
