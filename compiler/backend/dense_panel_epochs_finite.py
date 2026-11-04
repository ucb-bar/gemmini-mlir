"""Exact bounded certification of the shared dense panel epoch proposal.

The original drained program is checked against every source tile. Batched
readouts inherit that proof through an exact pure-readout tree refinement.
Each removed pipeline boundary is checked over its complete finite
outer/chain/local domain, proving its earlier load bytes are read by the next
step. This is an authored finite certificate, never an SMT certificate.
"""
from dataclasses import asdict,dataclass,replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import monotonic
from merlin.sched.ir import Call,Loop,check_structure,evaluate
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.compile.scheduling.block_schedule import Load,Preload,Compute,Store
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN,_serial_projection,_calls
from merlin.compile.scheduling.dense_panel_readout import _propose
from merlin.compile.scheduling.dense_panel_epochs import _plan,_boundaries,_chain_extents,_sites,_record,_drains
from .dense_finite_verification import FiniteRetainedResult,verify_retained_commands

MAX_CONTEXTS=2000000
MAX_ACCESSES=20000000
TIMEOUT_SECONDS=90


def _owners():
    from merlin.compile.scheduling import dense_panel_pipeline,dense_panel_epochs,dense_panel_readout
    from . import dense_finite_verification
    return tuple((str(Path(m.__file__).resolve()),sha256(Path(m.__file__).read_bytes()).hexdigest())
        for m in (dense_panel_pipeline,dense_panel_epochs,dense_panel_readout,dense_finite_verification))


def _identity(original,batched,actual,law,admission,adapter,storage,completion_epochs):
    return sha256(repr((original.digest(),batched.digest(),actual.digest(),asdict(law),asdict(admission),
        repr(adapter),asdict(storage),completion_epochs)).encode()).hexdigest()


def prove_consumption(streams,adapter,storage,*,deadline):
    """Exhaust all bindings of every boundary; cover full physical byte intervals."""
    if monotonic()>deadline:raise ValueError('finite panel epoch time bound exceeded')
    contexts=accesses=proved=0
    # Per-proof templates only. Keep the immutable calls alive; never share a
    # projection across source/kernel/law identities or compilation requests.
    templates={}
    sites={}
    def template(call):
        key=id(call)
        if key not in templates:
            op=adapter.project(call)
            if type(op) is Load:
                if op.gather is not None or op.role not in ('lhs','weight'):
                    raise ValueError('finite panel epochs require contiguous original operand loads')
                item=(storage.lhs if op.role=='lhs' else storage.rhs,'write',op.row,op.rows)
            elif type(op) is Compute:item=(storage.lhs,'read',op.input_row,op.rows)
            elif type(op) is Preload:
                item=None if op.weight_row is None else (storage.rhs,'read',op.weight_row,op.weight_rows)
            else:raise ValueError('finite panel stream has unknown effect')
            templates[key]=(call,item)
        original,item=templates[key]
        if original is not call:raise ValueError('finite panel epoch projection identity changed')
        return item
    def element_sites(element):
        key=id(element)
        if key not in sites:sites[key]=(element,tuple((template(call),local) for call,local in _sites((element,))))
        original,result=sites[key]
        if original is not element:raise ValueError('finite panel epoch site identity changed')
        return result
    def domains(scopes):
        return product(*(range(extent) for _,extent in scopes))
    def access(item,env):
        nonlocal accesses
        accesses+=1
        if accesses>MAX_ACCESSES:raise ValueError('finite panel epoch access bound exceeded')
        if item is None:return None
        view,kind,row_expr,rows_expr=item
        row,rows=evaluate(row_expr,env),evaluate(rows_expr,env)
        if not (view.base<=row and rows>0 and row+rows<=view.base+view.capacity):
            raise ValueError('finite panel epoch access escapes original physical window')
        return kind,view.backing,row*view.unit_bytes,(row+rows)*view.unit_bytes
    def effects(element,base):
        for item,local in element_sites(element):
            for values in domains(local):
                effect=access(item,{**base,**dict(zip((name for name,_ in local),values))})
                if effect is not None:yield effect
    def covered(backing,lo,hi,reads):
        for low,high in sorted((low,high) for space,low,high in reads if space==backing):
            if low>lo:break
            if high>lo:lo=high
            if lo>=hi:return True
        return False
    for stream in streams:
        extents=_chain_extents(stream['elements']);outer=stream['ancestry']
        for left,lb,right,rb,guards in _boundaries(stream['elements']):
            # A boundary template without writes imposes no rewrite obligation.
            if not any(item is not None and item[1]=='write' for item,_ in element_sites(left)):continue
            symbols=sorted({name for binding in (lb,rb) for name,(kind,_) in binding.items() if kind!='const'})
            scopes=(*outer,*((name,extents[name]-(name in guards)) for name in symbols))
            for values in domains(scopes):
                contexts+=1
                if contexts>MAX_CONTEXTS or monotonic()>deadline:
                    raise ValueError('finite panel epoch domain/time bound exceeded')
                env=dict(zip((name for name,_ in scopes),values))
                def binding(b):return {**env,**{name:(value if kind=='const' else env[name]+value) for name,(kind,value) in b.items()}}
                reads=[(space,low,high) for kind,space,low,high in effects(right,binding(rb)) if kind=='read']
                for kind,space,low,high in effects(left,binding(lb)):
                    if kind=='write':
                        if not covered(space,low,high,reads):
                            raise ValueError('finite panel epoch load is not fully read before its later rewrite')
                        proved+=1
    if monotonic()>deadline:raise ValueError('finite panel epoch time bound exceeded')
    return dict(boundary_contexts=contexts,load_intervals_proved=proved,access_instances=accesses,
        proof='every original outer/chain/local binding; full physical load byte interval coverage',
        bounds=dict(max_contexts=MAX_CONTEXTS,max_accesses=MAX_ACCESSES,timeout_seconds=TIMEOUT_SECONDS))


def verify_readout_refinement(original,batched,adapter,admission,source_report,*,deadline):
    """Reuse exact source proof only for a checked pure-readout refinement.

    Independently compare actual statement trees. Only full drains inside a
    Store-only subtree may disappear, and one final full drain must replace
    them. All data operands, counted domains and surrounding phases stay exact.
    Thus the original exhaustive source/codec/ACC proof transfers unchanged;
    issued readouts alone increase, bounded by the selected snapshot admission.
    """
    if monotonic()>deadline:raise ValueError('finite panel epoch time bound exceeded')
    drain=Call(DRAIN,())
    names=set()
    def collect(body):
        for node in body:
            if type(node) is Loop:
                names.add(node.var);collect(node.body)
    collect(original.body)
    def pure(node):
        if type(node) is Call:
            if node==drain:return (),0
            if type(adapter.project(node)) is not Store:return None
            return (node,),1
        if type(node) is not Loop:return None
        pieces=[pure(child) for child in node.body]
        if any(piece is None for piece in pieces):return None
        body=tuple(child for children,_ in pieces for child in children)
        count=node.extent*sum(count for _,count in pieces)
        if node.extent==1 and node.body and node.body[-1]==drain:return body,count
        return (replace(node,body=body),),count
    groups=[]
    def compare(before,after,outer=1):
        if len(before)!=len(after):raise ValueError('finite readout refinement changed statement count')
        for old,new in zip(before,after):
            if old==new:continue
            stripped=pure(old)
            if (stripped is not None and 1<stripped[1]<=admission.max_issued_readouts
                    and type(new) is Loop and new.extent==1 and new.var not in names
                    and new.body==(*stripped[0],drain)):
                names.add(new.var)
                groups.append(dict(readouts_per_epoch=stripped[1],dynamic_epochs=outer))
                continue
            if (type(old) is not Loop or type(new) is not Loop
                    or replace(old,body=())!=replace(new,body=())):
                raise ValueError('finite readout refinement changed data, source scope or completion phase')
            compare(old.body,new.body,outer*old.extent)
    # Preserve every kernel-level source/ABI attribute as well as its body law.
    if replace(original,body=())!=replace(batched,body=()):
        raise ValueError('finite readout refinement changed source ABI')
    compare(original.body,batched.body)
    if check_structure(batched) or len(tuple(_calls(batched.body)))>256:
        raise ValueError('finite readout refinement exceeds structural/static bounds')
    check_integer_domain(batched,adapter.index_bits)
    if monotonic()>deadline:raise ValueError('finite panel epoch time bound exceeded')
    return dict(kernel_sha256=batched.digest(),source_kernel_sha256=original.digest(),
        proof='exact pure-Store subtree refinement of exhaustively source-checked kernel',
        source_verification_reused=True,source_verification_sha256=sha256(repr(source_report).encode()).hexdigest(),
        groups=groups,max_issued_readouts=admission.max_issued_readouts,
        dynamic_epoch_drains=_drains(batched.body),static_commands=len(tuple(_calls(batched.body))))


@dataclass(frozen=True)
class FinitePanelEpochResult:
    kernel:object
    realization:dict
    original:object
    batched:object
    law:object
    admission:object
    adapter:object
    storage:object
    completion_epochs:bool
    checked_identity:str
    owner_sha256:str
    library_owners:tuple

    def revalidate(self):
        self.original.revalidate();self.law.validate_for(self.adapter,self.admission)
        if (sha256(Path(__file__).read_bytes()).hexdigest()!=self.owner_sha256 or _owners()!=self.library_owners
                or _identity(self.original.kernel,self.batched,self.kernel,self.law,self.admission,self.adapter,self.storage,self.completion_epochs)!=self.checked_identity
                or self.kernel.digest()!=self.realization['kernel_sha256']):
            raise ValueError('finite panel epoch original, actual commands, law or owner changed')
        batched,_=_propose(self.original.kernel,self.adapter,self.admission.max_issued_readouts)
        actual,_=_plan(batched,self.adapter,self.law,self.admission) if self.completion_epochs else (batched,[])
        if batched!=self.batched or actual!=self.kernel:
            raise ValueError('finite panel epoch proposal differs from checked shared transformation')
    def emit_llvm(self,builder,lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
        self.revalidate();result=emit_index_program(self.kernel,FnBuilderIndexAdapter(builder),lower_call);self.revalidate();return result
    def materialize_scf(self):
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands
        self.revalidate();return materialize_scf_commands(self.kernel)


@dataclass(frozen=True)
class _SourceEpochBudget:
    """One original initializer per full-drained pure phase, never batching.

    The overlap admission has no initializer field. This extra verification
    bound accepts only the builder's original singleton drained bias epochs;
    it does not add initialization overlap to the selected execution law.
    """
    admission:object
    max_issued_initializations:int=1
    def __post_init__(self):self.admission.__post_init__()
    def __getattr__(self,name):return getattr(self.admission,name)


def finite_panel_epoch_schedule(p,block,r,s,a,admission,law,plan,source,*,completion_epochs=True):
    """Certify source, batched readout schedule and shared epoch transformation."""
    deadline=monotonic()+TIMEOUT_SECONDS
    admission.__post_init__();law.validate_for(a,admission)
    if block.k!=1 or law.unknown():raise ValueError('finite panel epochs require one-K block and complete selected law')
    from . import dense_finite_verification
    from merlin.compile.scheduling.compact_dense import DenseBiasSource
    from merlin.compile.scheduling.schedule_ir import read_dense_schedule_plan,check_dense_schedule_commands
    from merlin.compile.scheduling.dense_panel_pipeline import build_dense_panel_pipeline
    bias=DenseBiasSource(p.n,'i32') if p.bias else None
    decision=read_dense_schedule_plan(plan,p,r,s,a,source_identity=source,dtypes=('i8','i8','i32'),
        static_budget=256,bias_source=bias)
    if (decision.block!=block or decision.storage!=s or decision.pipeline_kind!='two_window'
            or not decision.pipeline_prefetch or decision.loop_transforms
            or decision.pipeline_issue_order!='compute_then_prefetch'):
        raise ValueError('finite panel epoch original plan differs from supported checked source realization')
    source_kernel=build_dense_panel_pipeline(p,block,r,s,a,name='dense_panel_epochs',dtypes=('i8','i8','i32'),
        prefetch=True,outer_order=decision.outer_order,window_offsets=decision.pipeline_window_offsets,
        issue_order=decision.pipeline_issue_order,bias_source=bias)
    source_budget=_SourceEpochBudget(admission)
    source_report=verify_retained_commands(source_kernel,p,r,s,a,source_budget,static_budget=256,allow_disjoint_mixed=True)
    check_dense_schedule_commands(decision,source_report['static_commands'],256)
    source_report['schedule_ir']=dict(sha256=decision.plan_sha256,text=decision.plan_text,source_identity=source,
        pipeline_kind='two_window',prefetch=True,pipeline_issue_order=decision.pipeline_issue_order,
        decisions_consumed=['block_tiles','outer_order','row_bases','panel_pipeline','pipeline_window_offsets','pipeline_issue_order'])
    original=FiniteRetainedResult(source_kernel,source_report,source_kernel.digest(),
        sha256(Path(dense_finite_verification.__file__).read_bytes()).hexdigest())
    batched,groups=_propose(original.kernel,a,admission.max_issued_readouts)
    batching=verify_readout_refinement(original.kernel,batched,a,admission,source_report,deadline=deadline)
    if batching['groups']!=groups:raise ValueError('finite readout refinement differs from shared group accounting')
    actual,streams=_plan(batched,a,law,admission) if completion_epochs else (batched,[])
    if check_structure(actual) or _serial_projection(actual)!=_serial_projection(batched):
        raise ValueError('finite panel epoch changed source effect, scope or issue order')
    if len(tuple(_calls(actual.body)))>256:raise ValueError('finite panel epoch static command bound exceeded')
    check_integer_domain(actual,a.index_bits)
    if s.accumulator.backing in (s.lhs.backing,s.rhs.backing):raise ValueError('finite panel epoch accumulator aliases operands')
    consumption=prove_consumption(streams,a,s,deadline=deadline)
    before,after=_drains(batched.body),_drains(actual.body)
    report=dict(schema='authored_dense_finite_panel_epochs.v1',kernel_sha256=actual.digest(),
        source=source,schedule_ir=original.realization['schedule_ir'],input_schedule=original.realization,
        readout_batching=dict(groups=groups,finite_verification=batching),admission=asdict(admission),
        epoch_law=asdict(law),unknown_premises=[],dynamic_epoch_drains=after,completion_epochs=completion_epochs,
        panel_epochs=dict(input_dynamic_drains=before,dynamic_drains=after,removed_dynamic_drains=before-after,
            streams=[_record(stream) for stream in streams],consumption_proofs=consumption),
        qualification='exact bounded source/effect/consumption verification under selected original ordering law',
        target_execution='not_established',missing_dependency='z3')
    result=FinitePanelEpochResult(actual,report,original,batched,law,admission,a,s,completion_epochs,
        _identity(original.kernel,batched,actual,law,admission,a,s,completion_epochs),sha256(Path(__file__).read_bytes()).hexdigest(),_owners())
    result.revalidate();return result
