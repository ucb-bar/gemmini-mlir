"""Completion-only composition of checked dense schedules.

No constructor is called here. The proposal erases only internal completion
wrappers in a contiguous, store-free K stream, and retains its original closing
drain. Exact counted data trees, storage and pair issue order are invariant.
Selected RAW/WAR/execute-order laws supply the asynchronous dependency edges.
Since a load queue can complete out of order, every overlapping pair of load
instances must have an intervening consuming read. A quantified byte-interval
proof checks this bridge, including across iterations, without expanding K.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import inspect
import time

from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Select, Sum, Var, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from .block_schedule import Compute, Load, Preload, Store, InitializeAccumulator
from .compact_dense import _digest
from .compact_verification import _project
from .dense_contracts import TiledIntegerStore
from .dense_panel_pipeline import DRAIN, _calls, _remap, _serial_projection

LAWS = ("local_read_after_write", "local_write_after_read",
        "ordered_accumulator_writes", "accumulator_result_visibility",
        "ordered_configuration_capture", "finite_queue_progress",
        "complete_device_drain", "exclusive_private_state")


class CompletionRefusal(ValueError):
    """A valid original route remains available; this proof does not apply."""

    def __init__(self,message,category='proof_refusal'):
        super().__init__(message)
        self.category=category


def _epoch(s):
    return (type(s) is Loop and s.extent == 1 and s.body
            and s.body[-1] == Call(DRAIN, ()))


def _count(body, instruction):
    return sum(s.extent * _count(s.body, instruction) if type(s) is Loop
               else int(s.instr == instruction) for s in body)


def data_tree(body):
    """Canonical exact data program, substituting erased singleton indices."""
    out = []
    for s in body:
        if s == Call(DRAIN, ()):
            continue
        if _epoch(s):
            out.extend(data_tree(_remap(s.body[:-1], s.var, None)))
        elif type(s) is Loop:
            out.append(replace(s, body=data_tree(s.body)))
        else:
            out.append(s)
    return tuple(out)


def _check_epochs(kernel):
    epochs = []
    def walk(body, path=(), inside=False, multiplicity=1):
        for s in body:
            if type(s) is Loop:
                if s.iter_args or s.yields or s.results:
                    raise CompletionRefusal('carried completion state needs its own composition proof','unsupported_effects')
                direct = [c for c in s.body if c == Call(DRAIN, ())]
                if direct:
                    if inside or not _epoch(s) or len(direct) != 1:
                        raise CompletionRefusal("malformed or nested completion epoch",'malformed_ir')
                    epochs.append(((*path, s.var), s, multiplicity))
                    walk(s.body[:-1], (*path, s.var), True, multiplicity)
                else:
                    walk(s.body, (*path, s.var), inside, multiplicity*s.extent)
            elif not inside:
                raise CompletionRefusal("missing closing completion: effect outside a drained epoch",'malformed_ir')
            elif s.instr == DRAIN:
                raise CompletionRefusal("malformed completion call",'malformed_ir')
    if type(kernel) is not Kernel or check_structure(kernel):
        raise CompletionRefusal("invalid original scheduling IR",'malformed_ir')
    walk(kernel.body)
    if not epochs:
        raise CompletionRefusal("missing closing completion",'malformed_ir')
    return epochs


def _output_completion_epochs(kernel,adapter,problem):
    """Conservatively identify every original final-K completion by operands.

    Stores need not be adjacent to a completed output. A final contribution
    therefore protects its epoch even in a delayed-store or fused program.
    A may-contain result retains the whole original scope, never guesses which
    iteration is a boundary. No output-block label or loop name is trusted.
    """
    from .compact_dense_group_check import _range
    terminal=set();records=[]
    def walk(body,scopes=(),epoch=None,path=(),multiplicity=1):
        for s in body:
            if type(s) is Loop:
                scope=(id(s),s.var,s.extent)
                walk(s.body,(*scopes,scope),s if _epoch(s) else epoch,
                     (*path,s.var),multiplicity*s.extent)
            elif s.instr!=DRAIN:
                op=adapter.project(s)
                if type(op) is Compute:
                    low,high=_range(op.input_block[1],scopes)
                    if low<=problem.tiles[1]-1<=high:
                        if epoch is None:
                            raise CompletionRefusal('final K contribution lacks a closing completion','malformed_ir')
                        terminal.add(id(epoch))
    walk(kernel.body)
    for path,epoch,multiplicity in _check_epochs(kernel):
        if id(epoch) in terminal:
            records.append(dict(path=path,dynamic_completions=multiplicity,
                                basis='actual compute K coordinate may equal final source K tile'))
    return terminal,tuple(records)


def propose_completion(kernel, adapter, problem):
    """Keep every readout/init epoch and the last original drain of each run."""
    _check_epochs(kernel)
    groups = []
    terminal,_=_output_completion_epochs(kernel,adapter,problem)
    def contains_terminal(s):
        return type(s) is Loop and (id(s) in terminal or any(contains_terminal(c) for c in s.body))
    def stream(s):
        if type(s) is not Loop:
            return False
        if _epoch(s):
            ops = tuple(adapter.project(c) for c in _calls(s.body[:-1]))
            return bool(ops) and all(type(op) in (Load, Preload, Compute) for op in ops)
        if contains_terminal(s):
            return False
        return bool(s.body) and all(stream(c) for c in s.body)

    def rewrite(body, outer=1, path=()):
        out, i = [], 0
        while i < len(body):
            if not stream(body[i]):
                s = body[i]
                out.append(replace(s, body=rewrite(s.body, outer*s.extent, (*path, s.var)))
                           if type(s) is Loop and not _epoch(s) else s)
                i += 1
                continue
            end = i
            while end < len(body) and stream(body[end]):
                end += 1
                if id(body[end-1]) in terminal:
                    break
            run = body[i:end]
            drains = _count(run, DRAIN)
            compute = any(type(adapter.project(c)) is Compute for c in _calls(run) if c.instr != DRAIN)
            # A closing singleton is deliberately required. Do not peel an
            # output loop or move its final completion to manufacture support.
            if drains > 1 and compute and _epoch(run[-1]):
                closing = run[-1]
                stripped = data_tree(run)
                out.append(replace(closing, body=(*stripped, Call(DRAIN, ()))))
                groups.append(dict(path=(*path, closing.var), dynamic_groups=outer,
                                   original_drains_per_group=drains,
                                   retained_closing_epoch=closing.var))
            else:
                # A full-K retained constructor may enclose its staging and
                # computation epochs in a counted K-panel loop. Compose at
                # that inner scope instead of moving its closing completion
                # out of the loop or reconstructing a two-window program.
                for s in run:
                    out.append(replace(s,body=rewrite(s.body,outer*s.extent,(*path,s.var)))
                               if type(s) is Loop and not _epoch(s) else s)
            i = end
        return tuple(out)
    proposal = replace(kernel, body=rewrite(kernel.body))
    _check_epochs(proposal)
    if data_tree(kernel.body) != data_tree(proposal.body):
        raise CompletionRefusal("completion proposal changed exact ordered data operations")
    return proposal, tuple(groups)


def issued_requirements(kernel, adapter):
    """Exact finite counts per actual drained scope; pending capacity is separate."""
    requirements = dict(max_issued_compute_pairs=0, max_issued_loads=0, max_issued_readouts=0)
    epochs = []
    def counts(body):
        total = dict(pairs=0, loads=0, readouts=0, initialization=0)
        pending = None
        for s in body:
            if type(s) is Loop:
                if pending is not None:
                    raise CompletionRefusal("loop splits descriptor/compute pair")
                c = counts(s.body)
                for key in total:
                    total[key] += s.extent*c[key]
                continue
            if s == Call(DRAIN, ()):
                continue
            op = adapter.project(s)
            if type(op) is Preload:
                if pending is not None:
                    raise CompletionRefusal("unconsumed preload descriptor")
                pending = op
            elif type(op) is Compute:
                if pending is None:
                    raise CompletionRefusal("unpaired compute")
                total['pairs'] += 1
                pending = None
            else:
                if pending is not None:
                    raise CompletionRefusal("effect splits descriptor/compute pair")
                if type(op) is Load:
                    total['loads'] += 1
                elif type(op) in (Store, TiledIntegerStore):
                    total['readouts'] += 1
                elif type(op) is InitializeAccumulator:
                    total['initialization'] += 1
                else:
                    raise CompletionRefusal("unknown projected effect")
        if pending is not None:
            raise CompletionRefusal("unconsumed final descriptor")
        return total
    for path, epoch, multiplicity in _check_epochs(kernel):
        count = counts(epoch.body[:-1])
        if count['readouts'] and (count['pairs'] or count['loads'] or count['initialization']):
            raise CompletionRefusal("mixed readout stream has no completion composition proof")
        if count['initialization'] and (count['pairs'] or count['loads']):
            raise CompletionRefusal("mixed initialization stream has no completion composition proof")
        for field, key in [('max_issued_compute_pairs','pairs'),('max_issued_loads','loads'),
                           ('max_issued_readouts','readouts')]:
            requirements[field] = max(requirements[field], count[key])
        epochs.append(dict(path=path, dynamic_epochs=multiplicity, **count))
    return requirements, tuple(epochs)


def _bounds(admission):
    if hasattr(admission, 'overlap') and hasattr(admission, 'reduction'):
        return {key:min(getattr(admission.overlap,key),getattr(admission.reduction,key))
                for key in ('max_issued_compute_pairs','max_issued_loads','max_issued_readouts')}
    return {key:getattr(admission,key) for key in
            ('max_issued_compute_pairs','max_issued_loads','max_issued_readouts')}


def _prove_load_bridges(kernel, storage, adapter, *, timeout_ms):
    """Every possible overlapping ordered WAW has a RAW -> WAR bridge.

    No same-queue load completion order is assumed. UNSAT is required; timeout,
    unknown and a genuine counterexample all decline with distinct details.
    The existing source checker proves values; this proves completion edges.
    """
    import z3
    from .compact_verification import _defined_exists
    deadline = time.monotonic()+timeout_ms/1000
    sites = _project(_serial_projection(kernel), adapter, fold_singleton_loops=False)
    # Scope numbering matches _project's traversal even though drains vanish.
    markers, serial = [], 0
    def walk(body, ancestry=()):
        nonlocal serial
        for s in body:
            if type(s) is Loop:
                serial += 1
                scope = (serial,s.var,s.extent)
                if _epoch(s): markers.append((*ancestry,scope))
                walk(s.body,(*ancestry,scope))
    walk(kernel.body)
    def expr(x, env):
        if type(x) is int:return z3.IntVal(x)
        if type(x) is Const:return z3.IntVal(x.value)
        if type(x) is Var:return env[x.name]
        if type(x) is Sum:return z3.Sum(*(expr(t,env) for t in x.terms))
        if type(x) is Scaled:return expr(x.expr,env)*x.factor
        if type(x) is Select:return z3.If(env[x.var]==x.equals,expr(x.then,env),expr(x.other,env))
        raise CompletionRefusal("unknown operand expression in dependency proof")
    def domain(site,tag):
        env={name:z3.Int(f'{tag}:{scope}:{name}') for scope,name,_ in site.scopes}
        return env,[z3.And(env[name]>=0,env[name]<extent) for _,name,extent in site.scopes]
    def position(site,env):
        return tuple(z3.IntVal(v) if type(v) is int else env[v[1]] for v in site.path)
    def before(a,ae,b,be):
        ap,bp=position(a,ae),position(b,be)
        return z3.Or(*(z3.And(*(ap[j]==bp[j] for j in range(i)),ap[i]<bp[i])
                       for i in range(min(len(ap),len(bp)))))
    def interval(view,row,rows,env):
        lo=expr(row,env)*view.unit_bytes
        return lo,lo+expr(rows,env)*view.unit_bytes
    queries=0
    for outer in markers:
        marker=outer[-1][0]
        selected=[s for s in sites if marker in {x[0] for x in s.scopes}]
        loads=[];reads=[]
        for site in selected:
            op=site.operation
            if type(op) is Load:
                view=storage.lhs if op.role=='lhs' else storage.rhs
                loads.append((site,view,op.row,op.rows))
            elif type(op) is Preload:
                reads.append((site,storage.rhs,op.weight_row,op.weight_rows,'weight',op.weight_block))
            elif type(op) is Compute:
                reads.append((site,storage.lhs,op.input_row,op.rows,'lhs',op.input_block))
        for a,av,ar,an in loads:
            for b,bv,br,bn in loads:
                if av.backing!=bv.backing:continue
                ae,ac=domain(a,'older');be,bc=domain(b,'younger')
                al,ah=interval(av,ar,an,ae);bl,bh=interval(bv,br,bn,be)
                point=z3.Int('overlap_byte')
                same=[ae[name]==be[name] for _,name,_ in outer]
                base=[*ac,*bc,*same,before(a,ae,b,be),point>=al,point<ah,point>=bl,point<bh]
                remaining=int((deadline-time.monotonic())*1000)
                if remaining<=0:raise CompletionRefusal("load reuse proof timeout",'proof_unavailable')
                solver=z3.Solver();solver.set(timeout=remaining,random_seed=0);solver.add(*base)
                queries+=1
                answer=solver.check()
                if answer==z3.unsat:continue
                if answer!=z3.sat:raise CompletionRefusal("load reuse overlap proof unknown: "+solver.reason_unknown(),'proof_unavailable')
                bridges=[]
                for read,rv,rr,rn,role,coordinates in reads:
                    if rv.backing!=av.backing or role!=a.operation.role:continue
                    re,rc=domain(read,'reader'+str(read.ordinal))
                    rl,rh=interval(rv,rr,rn,re)
                    from merlin.sched.ir.expr import free_vars
                    coordinate_vars=frozenset().union(*(free_vars(v) for v in coordinates))
                    claim=z3.And(*rc,*(ae[name]==re[name] for _,name,_ in outer),
                                 before(a,ae,read,re),before(read,re,b,be),point>=rl,point<rh,
                                 *(expr(x,re)==expr(y,ae) for x,y in zip(coordinates,a.operation.block)),
                                 *(re[name]==0 for _,name,extent in read.scopes
                                   if extent==1 or name not in coordinate_vars
                                   and name not in {v[1] for v in outer}))
                    bridge,_=_defined_exists(list(re.values()),claim)
                    # Eliminate remaining bounded affine coordinates BEFORE
                    # asking the universal no-counterexample question. Generic
                    # SMT quantifier instantiation can return "incomplete"
                    # depending on prior solver activity; that must not make
                    # production emission nondeterministic.
                    remaining=int((deadline-time.monotonic())*1000)
                    if remaining<=0:raise CompletionRefusal('load reuse witness elimination timeout','proof_unavailable')
                    try:
                        bridge=z3.TryFor(z3.Tactic('qe'),remaining)(bridge).as_expr()
                    except z3.Z3Exception as exc:
                        raise CompletionRefusal('load reuse witness elimination unavailable: '+str(exc),'proof_unavailable') from exc
                    def quantified(value):
                        return z3.is_quantifier(value) or any(quantified(c) for c in value.children())
                    if quantified(bridge):
                        raise CompletionRefusal('load reuse witness elimination unavailable','proof_unavailable')
                    bridges.append(bridge)
                solver.add(z3.Not(z3.Or(*bridges)))
                remaining=int((deadline-time.monotonic())*1000)
                if remaining<=0:raise CompletionRefusal("load reuse bridge proof timeout",'proof_unavailable')
                solver.set(timeout=remaining);queries+=1
                answer=solver.check()
                if answer==z3.sat:
                    raise CompletionRefusal(f"overlapping load reuse lacks a consuming-read bridge: sites {a.ordinal}, {b.ordinal}",'dependence_counterexample')
                if answer!=z3.unsat:
                    raise CompletionRefusal("load reuse bridge proof unknown: "+solver.reason_unknown(),'proof_unavailable')
    return dict(solver_queries=queries, result='unsat for every unbridged overlapping load pair',
                law='local RAW then local WAR; never load-queue completion order')


def context_identity(context):
    a=context['adapter']
    return _digest((asdict(context['problem']),asdict(context['block']),asdict(context['resources']),
                    asdict(context['storage']),asdict(context['admission']),context['premises'],
                    a.configuration_identity,a.numerical_policy_identity,a.index_bits,
                    a.synchronous_completion,a.exact_accumulator_continuation,
                    inspect.getsource(a.project),inspect.getsource(a.encode)))


def make_completion_plan(original,context,*,policy='coalesce_k_stream'):
    """A separate post-construction schedule decision; the route stays intact."""
    from xdsl.dialects.builtin import DenseArrayBase, ModuleOp, StringAttr, i64
    from xdsl.ir import Block, Region
    from merlin.xdsl_dialects.schedule import BindOp, CompletionPlacementOp, HandleType, PlanOp
    bounds=_bounds(context['admission'])
    bind=BindOp(result_types=[HandleType()],properties={'target':StringAttr(
        'dense-completion:'+original.kernel.digest()+':'+context_identity(context))})
    decision=CompletionPlacementOp(operands=[bind.handle],properties={
        'policy':StringAttr(policy),'issued_bounds':DenseArrayBase.from_list(i64,list(bounds.values()))})
    plan=ModuleOp([PlanOp(properties={'sym_name':StringAttr('dense_completion')},
                         regions=[Region([Block([bind,decision])])])])
    from xdsl.utils.exceptions import VerifyException
    try:
        plan.verify()
    except VerifyException as exc:
        raise CompletionRefusal('invalid completion decision: '+str(exc),'malformed_decision') from exc
    return plan


def read_completion_plan(original,context):
    """Reject ignored/foreign decisions; return the policy actually executed."""
    from xdsl.dialects.builtin import ModuleOp
    from merlin.xdsl_dialects.schedule import BindOp, CompletionPlacementOp, PlanOp
    from .schedule_ir import dense_schedule_text
    plan=context.get('completion_plan')
    if type(plan) is not ModuleOp:
        raise CompletionRefusal('completion placement requires a typed scheduling plan')
    from xdsl.utils.exceptions import VerifyException
    plan=plan.clone()
    try:
        plan.verify()
    except VerifyException as exc:
        raise CompletionRefusal('invalid completion decision: '+str(exc),'malformed_decision') from exc
    members=list(plan.body.block.ops)
    if (plan.attributes or plan.properties or len(members)!=1 or type(members[0]) is not PlanOp
            or members[0].attributes or set(members[0].properties)!={'sym_name'}
            or len(members[0].body.blocks)!=1 or members[0].body.block.args):
        raise CompletionRefusal('unsupported completion plan effects or structure')
    ops=list(members[0].body.block.ops)
    if len(ops)!=2 or type(ops[0]) is not BindOp or type(ops[1]) is not CompletionPlacementOp:
        raise CompletionRefusal('completion consumer needs exactly bind then completion_placement')
    bind,decision=ops
    if (bind.attributes or set(bind.properties)!={'target'} or bind.operands or bind.regions
            or bind.target.data!='dense-completion:'+original.kernel.digest()+':'+context_identity(context)
            or decision.target is not bind.handle or decision.attributes or decision.results or decision.regions
            or set(decision.properties)!={'policy','issued_bounds'}):
        raise CompletionRefusal('completion decision has unknown effects or a stale source/context binding')
    bounds=tuple(decision.issued_bounds.get_values())
    if bounds!=tuple(_bounds(context['admission']).values()):
        raise CompletionRefusal('completion decision differs from selected issued budgets')
    return decision.policy.data,dense_schedule_text(plan)


def checked_context_identity(original,context):
    from .compact_dense_group_check import _range
    _,text=read_completion_plan(original,context)
    range_owner=sha256(Path(inspect.getsourcefile(_range)).read_bytes()).hexdigest()
    return _digest((context_identity(context),text,range_owner))


@dataclass(frozen=True)
class DenseCompletionResult:
    original: object
    kernel: Kernel
    realization: dict
    context: dict
    checked_context: str
    checked_kernel: str
    checked_owner: str

    @property
    def admission(self):return self.context['admission']

    def revalidate(self):
        self.original.revalidate()
        if (self.kernel.digest()!=self.checked_kernel
                or checked_context_identity(self.original,self.context)!=self.checked_context
                or sha256(Path(__file__).read_bytes()).hexdigest()!=self.checked_owner):
            raise ValueError('completion program, source owner or selected proof context changed')
        proposal,_=propose_completion(self.original.kernel,self.context['adapter'],self.context['problem'])
        if proposal!=self.kernel:
            raise ValueError('completion emission differs from exact checked transformation')

    def emit_index_program(self,control,lower_effect,*,complete_epoch):
        from .llvm_program import emit_index_effect_program
        self.revalidate()
        receipt=emit_index_effect_program(self.kernel,control,lower_effect,
            complete_epoch=complete_epoch,completion_instruction=DRAIN)
        self.revalidate()
        return receipt

    def materialize_scf(self):
        from .scf_commands import materialize_scf_commands
        self.revalidate(); result=materialize_scf_commands(self.kernel);self.revalidate()
        return result

    def emit_llvm(self,builder,lower_call):
        from .llvm_program import FnBuilderIndexAdapter, emit_index_program
        self.revalidate()
        result=emit_index_program(self.kernel,FnBuilderIndexAdapter(builder),lower_call)
        self.revalidate()
        return result


def coalesce_dense_completion(original,context,*,timeout_ms=15000):
    original.revalidate()
    for name in LAWS:
        if context['premises'].get(name) is not True:
            raise CompletionRefusal('missing ordering/program premise: '+name,'missing_premise')
    p,b,r,s,a=(context[key] for key in ('problem','block','resources','storage','adapter'))
    if not s.admits(p,b,r):
        raise CompletionRefusal('unknown or overlapping original live storage ownership','ownership_refusal')
    if not a.synchronous_completion or not a.exact_accumulator_continuation:
        raise CompletionRefusal('missing original full completion or exact continuation')
    proposal,groups=propose_completion(original.kernel,a,p)
    if not groups:raise CompletionRefusal('no redundant K-stream completion with an original closing singleton','no_opportunity')
    check_integer_domain(proposal,a.index_bits)
    requirements,epochs=issued_requirements(proposal,a)
    bounds=_bounds(context['admission'])
    if any(type(bounds[k]) is not int or bounds[k]<v for k,v in requirements.items()):
        raise CompletionRefusal('insufficient issued command budget for actual completion scopes','issued_budget_refusal')
    policy,plan_text=read_completion_plan(original,context)
    if policy!='coalesce_k_stream':
        raise CompletionRefusal('schedule explicitly selects original completion placement','selected_original')
    # Explicit access projection rejects all unmodeled fields/annotations.
    _project(_serial_projection(proposal),a)
    bridges=_prove_load_bridges(proposal,s,a,timeout_ms=timeout_ms)
    before=_count(original.kernel.body,DRAIN);after=_count(proposal.body,DRAIN)
    report=dict(original.realization)
    report.update(kernel_sha256=proposal.digest(),dynamic_epoch_drains=after)
    report['completion_placement']=dict(schema='merlin.dense_completion.v1',status='optimized',
        consumed_schedule_ir=plan_text,
        source_kernel_sha256=original.kernel.digest(),actual_kernel_sha256=proposal.digest(),
        exact_ordered_data_tree_sha256=replace(proposal,body=data_tree(proposal.body)).digest(),groups=groups,epochs=epochs,
        original_dynamic_drains=before,retained_dynamic_drains=after,removed_dynamic_drains=before-after,
        issued_requirements=requirements,issued_bounds=bounds,load_reuse_proof=bridges,
        premises=dict(context['premises']),storage_sha256=_digest(asdict(s)),
        original_validation=context.get('original_validation'),
        preserved_output_completion_scopes=_output_completion_epochs(original.kernel,a,p)[1],
        preserve='all original data calls, readout epochs and original closing drains; unchanged storage',
        target_execution='conditional selected ordering facts; numerical execution checked separately')
    result=DenseCompletionResult(original,proposal,report,dict(context),checked_context_identity(original,context),
        proposal.digest(),sha256(Path(__file__).read_bytes()).hexdigest())
    result.revalidate()
    return result
