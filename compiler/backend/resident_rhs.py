"""Complete RHS residency for a caller-owned, uninterrupted resident epoch.

Capacity is planned by Merlin. The authored rewrite consumes checked compact
source schedules, preserves every MAC/initializer/store, and changes only RHS
load placement and the corresponding weight addresses. Full completion edges
close every ownership transition; no load-queue completion order is assumed.
"""
from __future__ import annotations

from dataclasses import replace, asdict, dataclass
from math import isfinite

from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program
from merlin.sched.ir import Call, Loop, as_expr
from . import isa
from .dense import _schedule
from .dense_readout import store_tile, _pack


@dataclass(frozen=True)
class ResidentPhysicalPitches:
    """Caller-owned rectangular subviews; logical schedules retain their N."""
    rhs_row_elements: int
    output_row_elements: int

    def validate(self, logical_n):
        if any(type(x) is not int or x < logical_n or x >= 1 << 63
               for x in (self.rhs_row_elements, self.output_row_elements)):
            raise ValueError('resident panel pitches must contain each logical row')


def _proof_dependency_available():
    """Optional symbolic optimization proof; absence grants no proof result."""
    try:
        import z3
    except ModuleNotFoundError as exc:
        if exc.name == 'z3':
            return False
        raise
    return True


def resident_groups(cb):
    """Map first COMMIT indices to complete, uninterrupted resident epochs.

    Discovery preserves every committed value. Actual shape, readout, capacity
    and alias admission remains in ``lower_resident_products`` before emission.
    No unrecognized operation, nested packing or unrelated eviction may cross
    this local-storage ownership interval.
    """
    commands=cb.get('commands',())
    groups={}
    for start,command in enumerate(commands):
        if command.get('opcode')!='RES_PACK':continue
        pack=command.get('operands',{})
        handle,rhs=pack.get('dst'),pack.get('src')
        if handle is None or rhs is None:continue
        products={}
        committed=set()
        members=[]
        closed=False
        valid=True
        for index in range(start+1,len(commands)):
            command=commands[index]
            code=command.get('opcode')
            operands=command.get('operands',{})
            attrs=command.get('attributes',{})
            if code=='EVICT':
                closed=operands.get('handle')==handle
                valid=valid and closed
                break
            if code=='MATMUL_RESIDENT':
                acc=operands.get('dst')
                lhs=operands.get('lhs')
                if operands.get('rhs')!=handle or acc is None or lhs is None or acc in products:
                    valid=False;break
                products[acc]=(lhs,dict(attrs))
            elif code=='COMMIT':
                acc,dst=operands.get('src'),operands.get('dst')
                if acc not in products or acc in committed or dst is None:
                    valid=False;break
                lhs,producer_attrs=products[acc]
                merged={**producer_attrs,**attrs}
                if 'maxpool' in merged.get('epilogue',()):
                    valid=False;break
                committed.add(acc)
                members.append({'command_index':index,'lhs':lhs,'rhs':rhs,'dst':dst,'attributes':merged})
            else:
                valid=False;break
        if valid and closed and len(members)>=2 and committed==set(products):
            groups[members[0]['command_index']]=tuple(members)
    return groups


def rhs_product_groups(cb):
    """Propose complete RHS reuse across independent consecutive COMMITs.

    Tensor names here are canonical source value identities, never layout or
    spelling guesses. Resident handles are resolved at their producer only,
    and only a balanced, uninterrupted pack/evict epoch can supply a product.
    All commits in a run must commute, including members of different RHS
    buckets: executing a bucket at its first commit can reorder those members.
    Admission/emission remains atomic in ``lower_resident_products``.
    """
    commands = cb.get('commands', ())
    # Resolve handles once, at the original MATMUL position. A later reuse of
    # the handle spelling must never change an earlier product's source value.
    active, resolved = {}, {}
    malformed = False
    handles = {c.get('operands', {}).get('dst') for c in commands if c.get('opcode') == 'RES_PACK'}
    bookkeeping = {'RES_PACK', 'EVICT', 'MATMUL', 'MATMUL_RESIDENT', 'COMMIT'}
    for index, command in enumerate(commands):
        code = command.get('opcode')
        a = command.get('operands', {})
        if code == 'RES_PACK':
            handle, source = a.get('dst'), a.get('src')
            if handle is None or source is None or handle in active or source in active:
                # Nested aliases and redefinitions have no proven lifetime.
                malformed = True
                for epoch in active.values(): epoch['valid'] = False
                if handle in active: active[handle]['valid'] = False
                continue
            active[handle] = {'source': source, 'valid': True, 'products': []}
        elif code == 'MATMUL_RESIDENT':
            epoch = active.get(a.get('rhs'))
            if epoch is not None:
                epoch['products'].append(index)
        elif code == 'EVICT':
            epoch = active.pop(a.get('handle'), None)
            if epoch is None:
                malformed = True
                for live in active.values(): live['valid'] = False
            elif epoch['valid']:
                for product in epoch['products']: resolved[product] = epoch['source']
        elif code not in bookkeeping:
            for epoch in active.values(): epoch['valid'] = False
    # Malformed handles cannot establish an ownership interval.
    if malformed or active:
        return {}
    aliases = cb.get('params', {}).get('tensor_aliases', {})
    # Alias relation formats differ across source routes. The emitter also
    # refuses explicit aliases; discovery conservatively declines such modules.
    if aliases:
        return {}
    groups, products, used = {}, {}, set()
    run, reads, writes = [], set(), set()

    def finish():
        nonlocal run, reads, writes
        buckets = {}
        for row in run:
            buckets.setdefault(row['rhs'], []).append(row)
        for members in buckets.values():
            if len(members) >= 2:
                groups[members[0]['command_index']] = tuple(members)
        run, reads, writes = [], set(), set()

    for index, command in enumerate(commands):
        code = command.get('opcode')
        a, attrs = command.get('operands', {}), command.get('attributes', {})
        if code in ('RES_PACK', 'EVICT'):
            continue
        if code in ('MATMUL', 'MATMUL_RESIDENT'):
            acc, lhs = a.get('dst'), a.get('lhs')
            rhs = a.get('rhs') if code == 'MATMUL' else resolved.get(index)
            if (acc is None or lhs is None or rhs is None or lhs in handles or rhs in handles
                    or acc in products or acc in used):
                finish()
                if acc is not None:
                    products.pop(acc, None)
                    used.add(acc)
                continue
            products[acc] = (lhs, rhs, dict(attrs))
            continue
        if code != 'COMMIT':
            finish()
            products.clear()
            continue
        acc, dst = a.get('src'), a.get('dst')
        product = products.pop(acc, None)
        if product is None or acc in used or dst is None:
            finish()
            continue
        used.add(acc)
        lhs, rhs, producer_attrs = product
        merged = {**producer_attrs, **attrs}
        if (any(e not in ('bias_add', 'acc_scale', 'relu') for e in merged.get('epilogue', ()))
                or merged.get('requant_shift') is not None
                or any(key.startswith('pool') for key in merged)):
            finish()
            continue
        row_reads = {lhs, rhs}
        row_reads.update(merged[key] for key in ('bias', 'accumulator_init') if merged.get(key) is not None)
        # A self-dependent write is valid standalone but cannot be moved before
        # other products. A dependency closes the previous independent run.
        if dst in row_reads:
            finish()
            continue
        if row_reads & writes or dst in reads or dst in writes:
            finish()
        run.append({'command_index': index, 'lhs': lhs, 'rhs': rhs,
                    'dst': dst, 'attributes': merged})
        reads.update(row_reads)
        writes.add(dst)
    finish()
    return groups


def _rewrite(schedule, plan):
    """Checked source MACs survive verbatim, except their resident weight row.

    The original compact verifier proves each weight block matches its source
    load and every MAC is present. We additionally prove all removed loads are
    exactly full-RHS-cache tiles, and all continuing writes avoid that cache.
    """
    import z3
    from merlin.sched.ir import Const, Var, Sum, Scaled, Select
    queries=0
    removed=0
    def expr(e,env):
        if isinstance(e,Const):return z3.IntVal(e.value)
        if isinstance(e,Var):return env[e.name]
        if isinstance(e,Sum):return sum(expr(x,env) for x in e.terms)
        if isinstance(e,Scaled):return expr(e.expr,env)*e.factor
        if isinstance(e,Select):return z3.If(env[e.var]==e.equals,expr(e.then,env),expr(e.other,env))
        raise ValueError('unsupported checked address expression')
    def prove(scopes,condition):
        nonlocal queries
        env={key:z3.Int(key) for key,_ in scopes}
        solver=z3.Solver();solver.set(timeout=2000)
        solver.add(*(z3.And(env[key]>=0,env[key]<extent) for key,extent in scopes))
        solver.add(z3.Not(condition(env)))
        queries+=1
        if solver.check()!=z3.unsat:raise ValueError('unproved resident RHS source/address relation')
    def walk(items,scopes=()):
        nonlocal removed
        result=[]
        for item in items:
            if isinstance(item,Loop):
                body=walk(item.body,(*scopes,(item.var,item.extent)))
                if body:result.append(replace(item,body=body))
                continue
            a=dict(item.args)
            if item.instr=='gemmini_mvin_weight':
                def check(env):
                    b0,b1=expr(a['block0'],env),expr(a['block1'],env)
                    dr,dc=expr(a['dram_row'],env),expr(a['dram_col'],env)
                    return z3.And(b0>=0,b0<plan.k_tiles,b1>=0,b1<plan.n_tiles,
                        dr==b0*plan.block,dc==b1*plan.block,
                        expr(a['rows'],env)==z3.If(dr+plan.block<=plan.k,plan.block,plan.k-dr),
                        expr(a['cols'],env)==z3.If(dc+plan.block<=plan.n,plan.block,plan.n-dc))
                prove(scopes,check);removed+=1
                continue
            if item.instr=='gemmini_mvin_lhs':
                prove(scopes,lambda env:z3.And(expr(a['addr'],env)>=0,
                    expr(a['addr'],env)+expr(a['rows'],env)<=plan.rhs_base))
            elif item.instr=='gemmini_preload':
                prove(scopes,lambda env:z3.And(expr(a['block0'],env)>=0,
                    expr(a['block0'],env)<plan.k_tiles,expr(a['block1'],env)>=0,
                    expr(a['block1'],env)<plan.n_tiles))
                a['weight']=as_expr(plan.rhs_base)+(a['block0']*plan.n_tiles+a['block1'])*plan.block
                item=replace(item,args=tuple(a.items()))
            elif item.instr not in ('gemmini_compute_preloaded','gemmini_mvout','gemmini_bias_mvin'):
                raise ValueError('unknown source schedule operation')
            result.append(item)
        return tuple(result)
    schedule.revalidate()
    kernel=replace(schedule.kernel,body=walk(schedule.kernel.body))
    if not removed:raise ValueError('resident transformation removed no repeated weight transfers')
    # Check the exact transformed projection: every initializer, MAC and output
    # remains in its original nest and order; only weight loads/addresses differ.
    def projection(items):
        out=[]
        for item in items:
            if isinstance(item,Loop):
                body=projection(item.body)
                if body:out.append(replace(item,body=body))
            elif item.instr!='gemmini_mvin_weight':
                args=tuple((k,v) for k,v in item.args if not (item.instr=='gemmini_preload' and k=='weight'))
                out.append(replace(item,args=args))
        return tuple(out)
    if projection(kernel.body)!=projection(schedule.kernel.body):
        raise ValueError('resident rewrite changed source MAC/output coverage')
    return kernel,{'source_kernel_sha256':schedule.kernel.digest(),
        'resident_kernel_sha256':kernel.digest(),'symbolic_address_queries':queries,
        'complete_mac_projection_preserved':True,'removed_static_rhs_load_sites':removed}


def _retain_lhs(kernel, shape, base):
    """Move checked source LHS tiles to a complete immutable member window."""
    import z3
    from merlin.sched.ir import Const, Var, Sum, Scaled, Select
    m, k, _ = shape
    d = isa.DIM
    mt, kt = (m+d-1)//d, (k+d-1)//d
    queries = 0
    def expr(e, env):
        if isinstance(e, Const): return z3.IntVal(e.value)
        if isinstance(e, Var): return env[e.name]
        if isinstance(e, Sum): return sum(expr(t, env) for t in e.terms)
        if isinstance(e, Scaled): return expr(e.expr, env)*e.factor
        if isinstance(e, Select): return z3.If(env[e.var] == e.equals, expr(e.then, env), expr(e.other, env))
        raise ValueError('unknown retained LHS address expression')
    def prove(scopes, condition):
        nonlocal queries
        env = {name:z3.Int(name) for name, _ in scopes}
        solver = z3.Solver(); solver.set(timeout=2000)
        solver.add(*(z3.And(env[name]>=0, env[name]<extent) for name, extent in scopes))
        solver.add(z3.Not(condition(env)))
        queries += 1
        if solver.check() != z3.unsat:
            raise ValueError('unproved retained LHS tile/source relation')
    def walk(items, scopes=()):
        result = []
        for item in items:
            if isinstance(item, Loop):
                body = walk(item.body, (*scopes, (item.var, item.extent)))
                if body: result.append(replace(item, body=body))
                continue
            a = dict(item.args)
            if item.instr == 'gemmini_mvin_lhs':
                def condition(env):
                    mi, ki = expr(a['block0'],env), expr(a['block1'],env)
                    dr, dc = expr(a['dram_row'],env), expr(a['dram_col'],env)
                    return z3.And(mi>=0, mi<mt, ki>=0, ki<kt, dr==mi*d, dc==ki*d,
                        expr(a['rows'],env)==z3.If(dr+d<=m,d,m-dr),
                        expr(a['cols'],env)==z3.If(dc+d<=k,d,k-dc))
                prove(scopes, condition)
                continue
            if item.instr == 'gemmini_compute_preloaded':
                prove(scopes, lambda env:z3.And(expr(a['block0'],env)>=0,
                    expr(a['block0'],env)<mt, expr(a['block1'],env)>=0,
                    expr(a['block1'],env)<kt))
                a['lhs'] = as_expr(base)+(a['block0']*kt+a['block1'])*d
                item = replace(item, args=tuple(a.items()))
            result.append(item)
        return tuple(result)
    retained = replace(kernel, body=walk(kernel.body))
    def projection(items):
        out = []
        for item in items:
            if isinstance(item, Loop):
                body = projection(item.body)
                if body: out.append(replace(item, body=body))
            elif item.instr != 'gemmini_mvin_lhs':
                args = tuple((key,value) for key,value in item.args
                    if not (item.instr=='gemmini_compute_preloaded' and key=='lhs'))
                out.append(replace(item, args=args))
        return tuple(out)
    if projection(kernel.body) != projection(retained.body):
        raise ValueError('retained LHS rewrite changed source MAC/init/readout projection')
    return retained, {'source_kernel_sha256':kernel.digest(),
        'retained_kernel_sha256':retained.digest(), 'symbolic_address_queries':queries,
        'complete_mac_projection_preserved':True, 'lhs_base':base,
        'lhs_rows':mt*kt*d, 'lhs_source_loads':mt*kt}


def _compute_epoch_plan(prepared):
    """Capacity + selected-order admission; None retains compact completion."""
    import json
    from pathlib import Path
    order = json.loads(Path(__file__).with_name('dense_ordering.json').read_text())
    if order.get('completion_order',{}).get('execute',{}).get('order') != 'in_order':
        return None
    # Selected facts.json b8ff28ed... timing_structure.command_classes.classes:
    # execute.entries=16 and load.entries=8, both status=derived. Finite ready/
    # valid command queues apply backpressure; no program premise requires a
    # whole epoch to fit in those queues. A pure execute epoch has no pending
    # DMA or store, and its immutable operand windows cannot form a wait cycle.
    progress = {'facts_sha256':'b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af',
        'fact_path':'facts.timing_structure.command_classes.classes',
        'execute_entries':16, 'load_entries':8,
        'basis':'selected finite ready/valid queues with backpressure; pure execute has no pending DMA/store dependency'}
    rhs_plan = prepared[0][3]
    cursor = rhs_plan.rhs_base + rhs_plan.k_tiles*rhs_plan.n_tiles*isa.DIM
    result = []
    try:
        for row, attrs, shape, plan, schedule, kernel, proof in prepared:
            m,k,n = shape
            if (k,n) != (rhs_plan.k,rhs_plan.n):
                return None
            rows = ((m+isa.DIM-1)//isa.DIM)*plan.k_tiles*isa.DIM
            if cursor+rows > isa.SPAD_ROWS:
                return None
            retained, checked = _retain_lhs(kernel, shape, cursor)
            proof = {**proof, 'retained_lhs':checked}
            result.append((row,attrs,shape,plan,schedule,retained,proof))
            cursor += rows
    except (ValueError, KeyError, TypeError):
        return None
    return result, {'operand_rows_used':cursor,
        'rhs_window':[rhs_plan.rhs_base,rhs_plan.rhs_base+rhs_plan.k_tiles*rhs_plan.n_tiles*isa.DIM],
        'lhs_windows':[[r[-1]['retained_lhs']['lhs_base'],
            r[-1]['retained_lhs']['lhs_base']+r[-1]['retained_lhs']['lhs_rows']] for r in result],
        'execute_order_identity':order['identity'], 'finite_queue_progress':progress,
        'lhs_source_loads':sum(r[-1]['retained_lhs']['lhs_source_loads'] for r in result),
        'readout_epochs':sum(((r[2][0]+isa.DIM-1)//isa.DIM)*r[3].n_tiles for r in result),
        'removed_intercompute_drains':sum(((r[2][0]+isa.DIM-1)//isa.DIM)*r[3].n_tiles*r[3].k_tiles for r in result),
        'completion':'all unique disjoint input DMA writes then full drain; ordered pure execute per output tile then readout drains'}


def _prepare_lhs(ctx, name, m, k, base, *, literal=False):
    """Stage every (M,K) source tile exactly once into a disjoint window."""
    fb = ctx.fb; d = isa.DIM; kt = (k+d-1)//d
    def tile(mi, ki, rows, cols):
        mi = fb.const(mi) if isinstance(mi,int) else mi
        ki = fb.const(ki) if isinstance(ki,int) else ki
        offset = fb.add_i(fb.mul_i(mi,fb.const(d*k)),fb.mul_i(ki,fb.const(d)))
        addr = fb.add_i(fb.const(base),fb.mul_i(fb.add_i(fb.mul_i(mi,fb.const(kt)),ki),fb.const(d)))
        ctx.emit(*isa.mvin(ctx.ptr(name,offset),addr,rows,cols,channel=0,fb=fb))
    if literal:
        for mi in range((m+d-1)//d):
            for ki in range(kt):tile(mi,ki,min(d,m-mi*d),min(d,k-ki*d))
    else:
        def ktiles(mi,rows):
            if k//d:fb.for_loop(0,k//d,1,lambda ki:tile(mi,ki,rows,d),retain_loop=True)
            if k%d:tile(mi,k//d,rows,k%d)
        if m//d:fb.for_loop(0,m//d,1,lambda mi:ktiles(mi,d),retain_loop=True)
        if m%d:ktiles(m//d,m%d)


def _literal_epoch(prepared, *, max_calls=256, max_rhs_tiles=64):
    """Specialize bounded complete epochs with the shared ordered unroll pass.

    This removes induction-address arithmetic for small storage epochs. It
    issues the same dynamic operations, in the same order, with identical
    operands. Larger epochs retain their counted form and bounded code size.
    """
    from merlin.sched.primitives import unroll, Cursor
    from merlin.sched.ir.kernel import instances
    from merlin.sched.ir.expr import evaluate
    def size(items):
        return sum(item.extent*size(item.body) if isinstance(item,Loop) else 1 for item in items)
    if prepared[0][3].k_tiles*prepared[0][3].n_tiles>max_rhs_tiles or sum(size(item[5].body) for item in prepared)>max_calls:
        return prepared,False
    def concrete(kernel):
        return tuple((call.instr,tuple((key,evaluate(value,env)) for key,value in call.args))
                     for call,env in instances(kernel))
    def first_loop(items,path=()):
        for item in items:
            if isinstance(item,Loop):
                nested=first_loop(item.body,(*path,item.var))
                return nested or (*path,item.var)
        return None
    result=[]
    for row,attrs,shape,plan,schedule,kernel,proof in prepared:
        original=concrete(kernel)
        specialized=kernel
        while (path:=first_loop(specialized.body)) is not None:
            specialized=unroll(specialized,Cursor(path))
        if concrete(specialized)!=original:
            raise ValueError('bounded resident unroll changed ordered instruction instances')
        proof={**proof,'literal_epoch':{'pass':'merlin.sched.primitives.unroll',
            'ordered_instruction_instances_checked':len(original),
            'source_sha256':kernel.digest(),'specialized_sha256':specialized.digest(),
            'max_epoch_calls':max_calls,'max_rhs_tiles':max_rhs_tiles}}
        result.append((row,attrs,shape,plan,schedule,specialized,proof))
    return result,True


def _prepare(ctx,rhs,plan,*,literal=False,drain_each=True,row_stride=None):
    """Transfer every source RHS tile once into its immutable row window."""
    fb=ctx.fb;d=plan.block
    pitch=plan.n if row_stride is None else row_stride
    ctx.emit(*isa.config_ld(pitch,channel=1))
    def tile(ki,nj,rows,cols):
        ki=fb.const(ki) if isinstance(ki,int) else ki
        nj=fb.const(nj) if isinstance(nj,int) else nj
        offset=fb.add_i(fb.mul_i(ki,fb.const(d*pitch)),fb.mul_i(nj,fb.const(d)))
        row=fb.add_i(fb.const(plan.rhs_base),fb.mul_i(fb.add_i(fb.mul_i(ki,fb.const(plan.n_tiles)),nj),fb.const(d)))
        ctx.emit(*isa.mvin(ctx.ptr(rhs,offset),row,rows,cols,channel=1,fb=fb))
        if drain_each:ctx.fence()
    if literal:
        for ki in range(plan.k_tiles):
            for nj in range(plan.n_tiles):
                tile(ki,nj,min(d,plan.k-ki*d),min(d,plan.n-nj*d))
        if drain_each:ctx.fence()
        return
    def ntiles(ki,rows):
        if plan.n//d:fb.for_loop(0,plan.n//d,1,lambda nj:tile(ki,nj,rows,d),retain_loop=True)
        if plan.n%d:tile(ki,plan.n//d,rows,plan.n%d)
    if plan.k//d:fb.for_loop(0,plan.k//d,1,lambda ki:ntiles(ki,d),retain_loop=True)
    if plan.k%d:ntiles(plan.k//d,plan.k%d)
    if drain_each:ctx.fence()


def lower_resident_products(ctx,members,*,physical_pitches=None):
    """Return a receipt, or None before mutation when a complete epoch is refused.

    Members are mappings lhs/rhs/dst/attributes, with optional checked m/k/n.
    The caller owns an uninterrupted device interval: either an explicit
    resident epoch or a proven independent product run. It must exclude
    intervening storage-clobbering operations. ABI buffers must be disjoint;
    this adapter excludes explicit aliases, output dependencies and RHS writes.
    """
    members=tuple(members)
    if len(members)<2:return None
    if not _proof_dependency_available():return None
    rhs=members[0]['rhs']
    if any(row['rhs']!=rhs for row in members):return None
    outputs=[row['dst'] for row in members]
    inputs={row[key] for row in members for key in ('lhs','rhs')}
    auxiliaries={row.get('attributes',{}).get(key) for row in members for key in ('bias','accumulator_init')}-{None}
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    if len(set(outputs))!=len(outputs) or set(outputs)&(inputs|auxiliaries) or any(name in aliases for name in inputs|set(outputs)|auxiliaries):return None
    prepared=[]
    try:
        for row in members:
            attrs=dict(row.get('attributes',{}))
            lhs,weight,dst=(ctx.spec(row[key]) for key in ('lhs','rhs','dst'))
            if lhs['dtype']!='i8' or weight['dtype']!='i8' or dst['dtype'] not in ('i8','i32'):return None
            a,b,c=lhs['shape'],weight['shape'],dst['shape']
            if len(a)!=2 or len(b)!=2 or a[1]!=b[0] or c!=[a[0],b[1]]:return None
            m,k,n=int(a[0]),int(a[1]),int(b[1])
            if physical_pitches is not None:
                if type(physical_pitches) is not ResidentPhysicalPitches:return None
                physical_pitches.validate(n)
                if attrs.get("bias") or attrs.get("accumulator_init"):return None
            if any(key in row and row[key]!=value for key,value in (('m',m),('k',k),('n',n))):return None
            if attrs.get('output_dtype',dst['dtype'])!=dst['dtype']:return None
            if any(e not in ('bias_add','acc_scale','relu') for e in attrs.get('epilogue',())):return None
            if attrs.get('requant_shift') is not None:return None
            scales=attrs.get('acc_scale_by_column')
            if scales is not None and (len(scales)!=n or len(set(map(float,scales)))>1):return None
            scale=float(scales[0] if scales else attrs.get('acc_scale',1.0))
            if not isfinite(scale) or scale<0:return None
            if dst['dtype']=='i32' and (scale!=1 or scales is not None or attrs.get('relu') or 'relu' in attrs.get('epilogue',())):return None
            bias,init=attrs.get('bias'),attrs.get('accumulator_init')
            if bias and (ctx.spec(bias)['dtype']!='i32' or ctx.spec(bias)['shape']!=[n]):return None
            if init and (ctx.spec(init)['dtype']!='i32' or ctx.spec(init)['shape']!=[m,n]):return None
            plan=plan_dense_rhs_residency(m,k,n,block=isa.DIM,operand_rows=isa.SPAD_ROWS,
                accumulator_rows=isa.ACC_ROWS,rhs_base=isa.DIM)
            if plan is None:return None
            schedule=_schedule(m,k,n,bool(bias or init),False)
            kernel,proof=_rewrite(schedule,plan)
            prepared.append((row,attrs,(m,k,n),plan,schedule,kernel,proof))
    except (ValueError,KeyError,TypeError,OverflowError):
        return None
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    epoch_receipt = None
    if knob('RESIDENT_COMPUTE_EPOCHS',default='off',values=('off','on'),root=Path(__file__).resolve().parent.parent) == 'on':
        admitted = _compute_epoch_plan(prepared)
        if admitted is not None:
            prepared, epoch_receipt = admitted
    prepared,literal=_literal_epoch(prepared)
    fb=ctx.fb
    ctx.fence()
    ctx.current_source_index=prepared[0][1].get('merlin.source_op_index')
    _prepare(ctx,rhs,prepared[0][3],literal=literal,drain_each=epoch_receipt is None,
        row_stride=physical_pitches.rhs_row_elements if physical_pitches else None)
    if epoch_receipt is not None:
        ctx.emit(*isa.config_ld(prepared[0][2][1],channel=0))
        for row, attrs, (m,k,n), plan, schedule, kernel, proof in prepared:
            ctx.current_source_index=attrs.get('merlin.source_op_index')
            _prepare_lhs(ctx,row['lhs'],m,k,proof['retained_lhs']['lhs_base'],literal=literal)
        ctx.fence()
    for row,attrs,(m,k,n),plan,schedule,kernel,proof in prepared:
        ctx.current_source_index=attrs.get('merlin.source_op_index')
        ctx.emit(*isa.config_ex(dataflow=1))
        if epoch_receipt is None:ctx.emit(*isa.config_ld(k,channel=0))
        descriptor=[None]
        def emit(instruction,arguments):
            a={key:bridge.value(value) for key,value in arguments.items()}
            if instruction=='gemmini_mvin_lhs':
                offset=fb.add_i(fb.mul_i(a['dram_row'],fb.const(k)),a['dram_col'])
                ctx.emit(isa.FUNCT['mvin'],ctx.ptr(row['lhs'],offset),_pack(fb,a['addr'],a['rows'],a['cols']))
                ctx.fence()
            elif instruction=='gemmini_bias_mvin':
                init,bias=attrs.get('accumulator_init'),attrs.get('bias')
                if init:
                    offset=fb.add_i(fb.mul_i(fb.mul_i(a['block0'],fb.const(isa.DIM)),fb.const(n)),a['dram_col'])
                    ctx.emit(*isa.config_ld(n*4,channel=2,shrink=False))
                    ctx.emit(*isa.mvin(ctx.ptr(init,offset),isa.acc_addr(a['addr'],fb=fb),a['rows'],a['cols'],channel=2,fb=fb));ctx.fence()
                if bias:
                    ctx.emit(*isa.config_ld(0,channel=2,shrink=False))
                    ctx.emit(*isa.mvin(ctx.ptr(bias,a['dram_col']),isa.acc_addr(a['addr'],accumulate=bool(init),fb=fb),a['rows'],a['cols'],channel=2,fb=fb));ctx.fence()
            elif instruction=='gemmini_preload':
                descriptor[0]=a['cols']
                acc=fb.or_i(fb.or_i(a['acc'],fb.const(1<<31)),fb.shl_i(a['accumulate'],30))
                ctx.emit(isa.FUNCT['preload'],_pack(fb,a['weight'],a['depth'],a['cols']),_pack(fb,acc,a['rows'],a['cols']))
            elif instruction=='gemmini_compute_preloaded':
                ctx.emit(isa.FUNCT['compute_preloaded'],_pack(fb,a['lhs'],a['rows'],a['depth']),_pack(fb,isa.GARBAGE,a['depth'],descriptor[0]))
                if epoch_receipt is None:ctx.fence()
            elif instruction=='gemmini_mvout':
                store_tile(ctx,row['dst'],a['acc'],a['rows'],a['cols'],
                    physical_pitches.output_row_elements if physical_pitches else n,
                    a['dram_row'],a['dram_col'],attrs)
            else:raise ValueError('unknown resident terminal operation '+instruction)
        bridge=FnBuilderIndexAdapter(fb)
        proof['emission']=emit_index_program(kernel,bridge,emit)
        schedule.revalidate()
        ctx.fence()
    receipt={'mechanism':'complete_rhs_residency','compute_epochs':epoch_receipt,'rhs':rhs,'products':len(prepared),'literal_epoch':literal,
        'capacity':asdict(prepared[0][3]),'rhs_source_loads':prepared[0][3].k_tiles*prepared[0][3].n_tiles,
        'members':[item[-1] for item in prepared],
        'completion':epoch_receipt['completion'] if epoch_receipt else 'full fences after RHS/LHS DMA, execute, and every readout; disjoint immutable RHS rows',
        'physical_pitches':asdict(physical_pitches) if physical_pitches else None,
        'scope':'source-checked schedule rewrite and terminal emission; independent execution required'}
    ctx.metadata.setdefault('resident_rhs',[]).append(receipt)
    return receipt


def lower_resident_panel_products(ctx,members):
    """Optional caller-selected N panels for an uninterrupted source RHS group."""
    if not _proof_dependency_available():return None
    from .resident_rhs_panels import lower
    return lower(ctx,members)
