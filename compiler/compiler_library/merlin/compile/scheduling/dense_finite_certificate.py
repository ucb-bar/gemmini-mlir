"""Exact finite-domain source/ownership certificate for dense products.

This checker executes no arithmetic contraction. It interprets every actual
projected command over its full counted integer domain and tracks source row
identities, reduction prefixes and physical ownership. No sampled iterations,
numeric tensor values, SMT assumptions or cached attestations participate.

Serial mode uses the adapter's synchronous completion premise. Retained mode
checks pure drained staging/initialization/ordered-compute/readout epochs;
segmented mode additionally checks disjoint operand writes across compute.
Only square primitive i8/i8/i32 products and optional contiguous i32 column
bias are covered. Typed readout boundaries and larger domains retain SMT.
"""
from dataclasses import replace
import time
from merlin.sched.ir import Call, Loop, evaluate, instances, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from .block_schedule import Load, Preload, Compute, Store, InitializeAccumulator

MAX_COMMANDS = 2_000_000
MAX_OUTPUT_ROW_SPANS = 100_000


def verify_finite_dense(kernel, problem, resources, storage, adapter, admission=None, *, static_budget,
                        output_readout=None, bias_source=None, timeout_ms=5000, mode='serial'):
    if mode not in ('serial','retained','segmented'):
        raise ValueError('unknown finite dense completion contract')
    if mode!='serial' and admission is None:
        raise ValueError('finite dense epochs require selected hardware admission')
    # Unsupported semantic boundaries keep the original symbolic verifier.
    if (output_readout is not None
            or tuple(a.dtype for a in kernel.args)!=
                (('i8','i8','i32','i32') if bias_source is not None else ('i8','i8','i32'))):
        return None
    if (problem.tile_n != problem.tile_k or problem.tile_m != problem.tile_k
            or (problem.lhs_bytes,problem.rhs_bytes,problem.accumulator_bytes)!=(1,1,4)):
        return None
    def count(body):
        return sum(s.extent*count(s.body) if type(s) is Loop else 1 for s in body)
    dynamic=count(kernel.body)
    if (dynamic>MAX_COMMANDS or
            problem.m*((problem.n+problem.tile_n-1)//problem.tile_n)>MAX_OUTPUT_ROW_SPANS):
        return None
    if (check_structure(kernel) or kernel.attrs or type(static_budget) is not int or static_budget<=0
            or type(timeout_ms) is not int or timeout_ms<=0
            or not adapter.synchronous_completion or not adapter.exact_accumulator_continuation
            or storage.accumulator is None):
        raise ValueError('finite dense certificate lacks exact source structure/adapter law')
    from .compact_dense import DenseBiasSource
    if problem.bias!=(bias_source is not None):
        raise ValueError('finite dense bias presence differs from source declaration')
    if bias_source is not None:
        if type(bias_source) is not DenseBiasSource:
            raise ValueError('finite dense bias requires exact typed source')
        bias_source.validate(problem,('i8','i8','i32'))
    expected=(((problem.m,problem.k),'i8','read'),((problem.k,problem.n),'i8','read'),
              *((((problem.n,),'i32','read'),) if bias_source is not None else ()),
              ((problem.m,problem.n),'i32','write'))
    if (tuple(a.name for a in kernel.args)!=(('lhs','rhs','bias','output') if bias_source is not None else ('lhs','rhs','output')) or len(kernel.args)!=len(expected)
            or any((a.shape,a.dtype,a.access)!=e for a,e in zip(kernel.args,expected))):
        raise ValueError('finite dense certificate source boundary changed')
    dim=problem.tile_k
    if not resources.partial_output_tiles and (problem.m%dim or problem.n%dim):
        raise ValueError('finite dense output tails not admitted')
    check_integer_domain(kernel,adapter.index_bits)
    # Preserve original epoch structure rule: one extent-one owner with a final
    # full drain, no nested epoch, no effect outside an epoch.
    def structure(body,inside=False):
        for s in body:
            if type(s) is Loop:
                if s.iter_args or s.yields or s.results:
                    raise ValueError('finite dense cannot erase loop completion effects')
                drains=[x for x in s.body if type(x) is Call and x.instr=='merlin_dense_panel_drain']
                if drains:
                    if mode=='serial' or inside or s.extent!=1 or len(drains)!=1 or s.body[-1]!=Call('merlin_dense_panel_drain',()):
                        raise ValueError('finite dense epoch boundary changed')
                    structure(s.body[:-1],True)
                else:structure(s.body,inside)
            else:
                if s.produces is not None or s.awaits or s.unit or s.stages or s.sets or s.assumes:
                    raise ValueError('finite dense cannot erase command effects')
                if mode!='serial' and not inside:raise ValueError('finite dense effect outside drained epoch')
    structure(kernel.body)
    def static_count(body):
        return sum(static_count(x.body) if type(x) is Loop else 1 for x in body)
    if static_count(kernel.body)>static_budget:raise ValueError('finite dense static command budget exceeded')
    from .compact_verification import _project
    from .dense_panel_pipeline import _serial_projection
    sites=_project(kernel if mode=='serial' else _serial_projection(kernel),adapter,fold_singleton_loops=False)
    descriptor=None
    for site in sites:
        if type(site.operation) is Preload:
            if descriptor is not None:raise ValueError('finite dense static descriptor overwritten')
            descriptor=site
        elif type(site.operation) is Compute:
            if descriptor is None or descriptor.scopes!=site.scopes:
                raise ValueError('finite dense descriptor/compute counted domains differ')
            descriptor=None
        elif descriptor is not None:
            raise ValueError('finite dense static descriptor pair split')
    if descriptor is not None:raise ValueError('finite dense dangling static descriptor')
    start=time.monotonic(); memory={'lhs':{},'weight':{}};acc={};outputs={};pending=None
    reads={'lhs':set(),'weight':set()};writes={'lhs':set(),'weight':set()}
    ar=set();aw=set();epochs=[];counts={'lhs_loads':0,'rhs_loads':0,'compute_pairs':0,'stores':0,'initializations':0}
    total_bytes={'lhs':0,'weight':0,'store':0,'bias':0};ec={'loads':0,'compute_pairs':0,'stores':0,'initializations':0};kinds=set()
    views={'lhs':storage.lhs,'weight':storage.rhs,'acc':storage.accumulator}
    if views['lhs'].backing==views['acc'].backing or views['weight'].backing==views['acc'].backing:
        raise ValueError('finite dense certificate requires distinct operand/ACC backing')
    if views['lhs'].backing==views['weight'].backing:
        l,r=views['lhs'],views['weight']
        if l.base*l.unit_bytes<(r.base+r.capacity)*r.unit_bytes and r.base*r.unit_bytes<(l.base+l.capacity)*l.unit_bytes:
            raise ValueError('finite dense operand backing windows overlap')
    high_water={role:0 for role in views}
    def interval(role,row,rows):
        v=views[role]
        if type(row) is not int or type(rows) is not int or rows<=0 or row<v.base or row%v.alignment or row+rows>v.base+v.capacity:
            raise ValueError('finite dense local row interval escapes assigned window')
        high_water[role]=max(high_water[role],(row+rows-v.base)*v.unit_bytes)
        return range(row,row+rows)
    def access(role,rows,write=False):
        if mode=='serial':return
        own,other=(writes[role],reads[role]) if write else (reads[role],writes[role])
        if any(r in other or (write and r in own) for r in rows):raise ValueError('finite dense epoch load/consume hazard')
        own.update(rows)
    def positive(*values):
        if any(type(x) is not int or x<=0 or x>dim for x in values):
            raise ValueError('finite dense primitive extent or tail changed')
    for call,env in instances(kernel):
        if call.instr=='merlin_dense_panel_drain':
            if call.args or pending is not None:raise ValueError('finite dense drain with unfinished descriptor')
            if mode=='retained' and kinds not in ({'load'},{'compute'},{'store'},{'initialize'}):
                raise ValueError('finite retained epoch mixes original effect classes')
            if ec['stores'] and (ec['loads'] or ec['compute_pairs'] or ec['initializations']):
                raise ValueError('finite dense separate readout epoch law changed')
            if (ec['loads']>admission.max_issued_loads or ec['compute_pairs']>admission.max_issued_compute_pairs
                    or ec['stores']>admission.max_issued_readouts
                    or ec['initializations']>getattr(admission,'max_issued_initializations',0)):
                raise ValueError('finite dense epoch exceeds selected execution law')
            epochs.append(dict(ec));ec={'loads':0,'compute_pairs':0,'stores':0,'initializations':0};kinds.clear()
            for x in (*reads.values(),*writes.values(),ar,aw):x.clear()
            continue
        actual=replace(call,args=tuple((k,evaluate(v,env)) for k,v in call.args))
        op=adapter.project(actual)
        if mode=='serial':
            for x in (*reads.values(),*writes.values(),ar,aw):x.clear()
        if type(op) is InitializeAccumulator:
            if pending is not None or bias_source is None:
                raise ValueError('finite dense undeclared or unpaired accumulator initialization')
            mi,ni=op.block;ms,ns=mi*dim,ni*dim
            if (type(mi) is not int or type(ni) is not int or mi<0 or ni<0
                    or ms>=problem.m or ns>=problem.n or op.dram_col!=ns
                    or op.source_row_stride!=0 or op.source_col_stride!=1
                    or op.rows!=min(dim,problem.m-ms) or op.cols!=min(dim,problem.n-ns)
                    or op.cols*4>views['acc'].unit_bytes):
                raise ValueError('finite dense initializer source coordinates/strides/tail changed')
            for i,row in enumerate(interval('acc',op.row,op.rows)):
                if row in acc or row in ar or row in aw:
                    raise ValueError('finite dense initializer overwrites live sum or epoch destination')
                acc[row]=(ms+i,ns,op.cols,0);aw.add(row)
            ec['initializations']+=1;counts['initializations']+=1
            total_bytes['bias']+=op.cols*4;kinds.add('initialize')
        elif type(op) is Load:
            if pending is not None or op.role not in memory or op.gather is not None or op.step!=1:
                raise ValueError('finite dense unsupported source movement')
            positive(op.rows,op.cols)
            if op.cols>views[op.role].unit_bytes:raise ValueError('finite dense operand row width exceeded')
            h,w=(problem.m,problem.k) if op.role=='lhs' else (problem.k,problem.n)
            if op.dram_row<0 or op.dram_col<0 or op.dram_row+op.rows>h or op.dram_col+op.cols>w:
                raise ValueError('finite dense load source rectangle escapes tensor')
            if (op.dram_row%dim or op.dram_col%dim
                    or op.rows!=min(dim,h-op.dram_row) or op.cols!=min(dim,w-op.dram_col)):
                raise ValueError('finite dense load primitive source tail/coordinates changed')
            if op.block!=(op.dram_row//dim,op.dram_col//dim):
                raise ValueError('finite dense load source block identity changed')
            rows=interval(op.role,op.row,op.rows);access(op.role,rows,True)
            for i,r in enumerate(rows):memory[op.role][r]=(op.dram_row+i,op.dram_col,op.cols)
            counts['lhs_loads' if op.role=='lhs' else 'rhs_loads']+=1
            total_bytes[op.role]+=op.rows*op.cols;ec['loads']+=1;kinds.add('load')
        elif type(op) is Preload:
            if pending is not None or op.weight_row is None:raise ValueError('finite dense descriptor pairing changed')
            positive(op.rows,op.cols,op.weight_rows)
            if op.cols*4>views['acc'].unit_bytes or op.cols>views['weight'].unit_bytes:
                raise ValueError('finite dense descriptor physical row width exceeded')
            if type(op.accumulate) not in (int,bool) or op.accumulate not in (0,1):
                raise ValueError('finite dense continuation flag is not Boolean')
            access('weight',interval('weight',op.weight_row,op.weight_rows))
            interval('acc',op.accumulator_row,op.rows)
            pending=op
        elif type(op) is Compute:
            if pending is None or op.fresh_weights is not True or op.input_gather is not None or op.rows!=pending.rows or op.cols!=pending.weight_rows:
                raise ValueError('finite dense compute/descriptor pairing changed')
            positive(op.rows,op.cols)
            if op.cols>views['lhs'].unit_bytes:raise ValueError('finite dense LHS physical row width exceeded')
            access('lhs',interval('lhs',op.input_row,op.rows))
            first=memory['weight'].get(pending.weight_row)
            if first is None:raise ValueError('finite dense weight read before completed load')
            ks,ns,nwidth=first
            if pending.weight_block!=(ks//dim,ns//dim):
                raise ValueError('finite dense descriptor source identity changed')
            if nwidth<pending.cols:raise ValueError('finite dense weight tail exceeds loaded columns')
            for d in range(op.cols):
                if memory['weight'].get(pending.weight_row+d)!=(ks+d,ns,nwidth):
                    raise ValueError('finite dense weight source rows do not form exact K/N tile')
            if ks%dim or ns%dim or pending.cols!=min(dim,problem.n-ns) or op.cols!=min(dim,problem.k-ks):
                raise ValueError('finite dense source K/N coordinates or ragged extent changed')
            if bool(pending.accumulate)!=(bias_source is not None or ks>0):raise ValueError('finite dense accumulation initialization changed')
            for i in range(op.rows):
                token=memory['lhs'].get(op.input_row+i)
                if token is None or token[1]!=ks or token[2]<op.cols:
                    raise ValueError('finite dense LHS residency/source reduction changed')
                mi=token[0];row=pending.accumulator_row+i
                if op.input_block!=(mi//dim,ks//dim):
                    raise ValueError('finite dense compute source identity changed')
                if i==0 and (mi%dim or op.rows!=min(dim,problem.m-mi)):
                    raise ValueError('finite dense M coordinate or ragged extent changed')
                if token[0]!=(memory['lhs'][op.input_row][0]+i):
                    raise ValueError('finite dense LHS source rows are not contiguous')
                if row in ar:raise ValueError('finite dense store/compute ACC epoch hazard')
                if (ks or bias_source is not None) and acc.get(row)!=(mi,ns,pending.cols,ks):
                    raise ValueError('finite dense ACC ownership or ordered K prefix changed')
                # Starting a new sum cannot discard an unobserved previous sum.
                if not ks and bias_source is None and row in acc:raise ValueError('finite dense ACC overwrite before final observer')
                acc[row]=(mi,ns,pending.cols,ks+op.cols);aw.add(row)
            counts['compute_pairs']+=1;ec['compute_pairs']+=1;pending=None;kinds.add('compute')
        elif type(op) is Store:
            if pending is not None or op.readout is not None:raise ValueError('finite dense unsupported output boundary')
            positive(op.rows,op.cols)
            if op.dram_row<0 or op.dram_col<0 or op.dram_row+op.rows>problem.m or op.dram_col+op.cols>problem.n:
                raise ValueError('finite dense output rectangle escapes tensor')
            if (op.dram_row%dim or op.dram_col%dim or op.rows!=min(dim,problem.m-op.dram_row)
                    or op.cols!=min(dim,problem.n-op.dram_col)):
                raise ValueError('finite dense output primitive coordinates/tail changed')
            for i,row in enumerate(interval('acc',op.accumulator_row,op.rows)):
                if row in aw:raise ValueError('finite dense compute/store ACC epoch hazard')
                expected=(op.dram_row+i,op.dram_col,op.cols,problem.k)
                if acc.get(row)!=expected:raise ValueError('finite dense store lacks complete exact reduction')
                spans=outputs.setdefault(op.dram_row+i,{});span=(op.dram_col,op.dram_col+op.cols)
                # Full primitive tails above are aligned; equal column starts
                # are the only possible overlaps. Keep bounded linear state.
                if span[0] in spans:raise ValueError('finite dense duplicate output observer')
                spans[span[0]]=span[1];del acc[row];ar.add(row)
            counts['stores']+=1;ec['stores']+=1;total_bytes['store']+=op.rows*op.cols*4;kinds.add('store')
        else:raise ValueError('finite dense unsupported actual projected effect')
    if pending is not None or (mode!='serial' and any(ec.values())) or acc:raise ValueError('finite dense incomplete final effects or observers')
    if len(outputs)!=problem.m:raise ValueError('finite dense missing output source rows')
    for spans in outputs.values():
        end=0
        for a,b in sorted(spans.items()):
            if a!=end:raise ValueError('finite dense output source cover has gaps')
            end=b
        if end!=problem.n:raise ValueError('finite dense incomplete output columns')
    if (high_water['lhs']+high_water['weight']>resources.operand_bytes
            or high_water['acc']>resources.accumulator_bytes):
        raise ValueError('finite dense actual storage high-water exceeds hardware budget')
    return dict(schema='merlin.dense_finite_source_epochs.v1',completion_mode=mode,kernel_sha256=kernel.digest(),
        certificate='all counted commands; exact source row maps and optional bias, ordered K prefixes, complete unique output cover and original completion/ownership laws',
        executed_instruction_calls=dynamic,static_instruction_calls=static_count(kernel.body),epochs=epochs,
        counts=counts,padded_high_water_bytes={views[role].name:value for role,value in high_water.items()},
        lhs_loads=counts['lhs_loads'],rhs_loads=counts['rhs_loads'],bias_initializations=counts['initializations'],
        output_tiles=counts['stores'],
        bias_source=None if bias_source is None else dict(columns=bias_source.columns,dtype=bias_source.dtype,
            arithmetic=bias_source.arithmetic,row_stride=0,column_stride=1),requested_lhs_bytes=total_bytes['lhs'],requested_rhs_bytes=total_bytes['weight'],
        requested_bias_load_bytes=total_bytes['bias'],
        requested_load_bytes=total_bytes['lhs']+total_bytes['weight']+total_bytes['bias'],requested_store_bytes=total_bytes['store'],
        solver_queries=0,enumeration_bound=MAX_COMMANDS,output_row_span_bound=MAX_OUTPUT_ROW_SPANS,checking_seconds=time.monotonic()-start,
        configuration_identity=adapter.configuration_identity,numerical_policy_identity=adapter.numerical_policy_identity,
        c_index_bits=adapter.index_bits,machine_emission='not_established',timing='UNKNOWN')


def verify_finite_segmented(kernel, problem, resources, storage, adapter, admission, **options):
    # The existing segmented family has no initializer contract.
    if problem.bias or options.get('bias_source') is not None:
        return None
    return verify_finite_dense(kernel,problem,resources,storage,adapter,admission,mode='segmented',**options)
