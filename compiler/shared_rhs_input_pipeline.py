"""Source-custodied overlap of independent shared-RHS members.

The checked ready owner proves each original contraction and pointer roster.
This exact transport moves only the next member's complete LHS staging, into a
separately bounded slot, before current M,N,K work. External pointers, RHS,
ACC/output effects and the original ready frontier remain unchanged.
"""
from dataclasses import asdict,dataclass,replace
from hashlib import sha256
from math import prod
from pathlib import Path
from merlin.sched.ir import Call,Loop,Const,check_structure
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.compile.scheduling.block_schedule import Load,Preload,Compute,Store
from merlin.compile.scheduling.compact_verification import _project
from merlin.compile.scheduling.compact_dense_group_check import _range,_span
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN,_serial_projection,_remap
from merlin.compile.scheduling.dense_shared_rhs import DenseSharedRHSResult
from dense_readout import resident_readout_laws


@dataclass(frozen=True)
class SharedRHSInputPipelineResult:
    original: object
    kernel: object
    realization: dict
    _guard: object
    @property
    def blocks(self):return self.original.blocks
    def revalidate(self):self._guard(self)
    emit_index_program=DenseSharedRHSResult.emit_index_program


def _sites(kernel,adapter,project_member):
    owners=[]
    def project(call):
        member,operation=project_member(call)
        owners.append(member)
        return operation
    sites=_project(_serial_projection(kernel),replace(adapter,project=project),fold_singleton_loops=False)
    return tuple(zip(sites,owners,strict=True))


def _transport(original,problems,storages,adapter,encode_member,project_member,reduction,epoch,*,slot_rows=None):
    """Derive the complete actual product and prove only its changed effects."""
    original.revalidate()
    if len(problems)<2 or len(original.kernel.body)!=1+2*len(problems):
        raise ValueError('input pipeline requires one original LHS stage and ready epoch per independent member')
    if epoch.unknown():raise ValueError('input pipeline has unknown selected load/compute/configuration/progress premises')
    epoch.validate_for(adapter,reduction)
    if any(s.lhs!=storages[0].lhs or s.rhs!=storages[0].rhs or s.accumulator!=storages[0].accumulator for s in storages):
        raise ValueError('input pipeline requires identical explicitly owned private views')
    # Source ownership remains the original exact independent-pointer proof.
    sites=_sites(original.kernel,adapter,project_member)
    lhs=storages[0].lhs
    high=lhs.base
    for site,_ in sites:
        op=site.operation
        if type(op)is Load and op.role=='lhs' or type(op)is Compute:
            row=op.row if type(op)is Load else op.input_row
            high=max(high,_range(row,site.scopes)[1]+_range(op.rows,site.scopes)[1])
    alignment=lhs.alignment
    required=((high-lhs.base+alignment-1)//alignment)*alignment
    if slot_rows is None:slot_rows=required
    if type(slot_rows)is not int or slot_rows<required or slot_rows%alignment:
        raise ValueError('input pipeline slot does not cover the complete original LHS footprint')
    if slot_rows*2>lhs.capacity:
        raise ValueError('input pipeline has no capacity for two disjoint complete LHS slots')
    slots=tuple(replace(lhs,base=lhs.base+i*slot_rows,capacity=slot_rows)for i in range(2))
    maximum=(1<<(adapter.index_bits-1))-1
    phases=[];laws=[]
    for member,(p,b) in enumerate(zip(problems,original.blocks,strict=True)):
        stage,work=original.kernel.body[1+2*member:3+2*member]
        for s in (stage,work):
            if type(s)is not Loop or s.extent!=1 or not s.body or s.body[-1]!=Call(DRAIN,()):
                raise ValueError('input pipeline needs original singleton fully drained phases')
        def relocate(body,kind):
            out=[]
            for s in body:
                if type(s)is Loop:out.append(replace(s,body=relocate(s.body,kind)));continue
                if s.instr==DRAIN:raise ValueError('input pipeline cannot cross an inner completion')
                if s!=Call(s.instr,s.args):raise ValueError('input pipeline encountered additional unknown call effects')
                owner,op=project_member(s)
                if owner!=member or encode_member(member,op)!=s:
                    raise ValueError('input pipeline changed original member/pointer/effect ownership')
                if kind=='stage':
                    if type(op)is not Load or op.role!='lhs':raise ValueError('input pipeline stage is not pure original LHS loading')
                    shifted=replace(op,row=op.row+(member%2)*slot_rows)
                else:
                    if type(op)not in (Preload,Compute,Store):raise ValueError('input pipeline requires original M,N,K ready work')
                    shifted=replace(op,input_row=op.input_row+(member%2)*slot_rows)if type(op)is Compute else op
                actual=encode_member(member,shifted)
                if project_member(actual)!=(member,shifted):raise ValueError('input pipeline relocation does not roundtrip actual operands')
                # encode_member derives the same external pointer from unchanged
                # source coordinates. Neither tensor identity nor offset moves.
                before=tuple((k,v)for k,v in s.args if type(v).__name__=='Ptr')
                after=tuple((k,v)for k,v in actual.args if type(v).__name__=='Ptr')
                if before!=after:raise ValueError('input pipeline relocated an external source pointer')
                out.append(actual)
            return tuple(out)
        stage=replace(stage,body=(*relocate(_remap(stage.body[:-1],stage.var,None),'stage'),Call(DRAIN,())))
        work=replace(work,body=(*relocate(_remap(work.body[:-1],work.var,None),'work'),Call(DRAIN,())))
        bound,admission=resident_readout_laws(adapter,b,reduction,epoch,input_pipeline=True)
        if admission.disjoint_load_qualification_identity is None:
            raise ValueError('input pipeline lacks selected disjoint LD/compute/readout conjunction')
        admission.validate_for(adapter,bound)
        laws.append((bound,admission))
        phases.append((stage,work))
    body=[original.kernel.body[0],phases[0][0]]
    for member in range(len(phases)-1):
        stage,work=phases[member+1][0],phases[member][1]
        body.append(Loop('shared_input_overlap_'+str(member),1,
                         (*stage.body[:-1],*work.body[:-1],Call(DRAIN,()))))
    body.append(phases[-1][1])
    actual=replace(original.kernel,body=tuple(body))
    if check_structure(actual):raise ValueError('input pipeline changed counted binder custody')
    check_integer_domain(actual,adapter.index_bits)
    # Check real decoded addresses against disjoint slots and unchanged RHS/ACC
    # views; these conservative counted interval bounds cover every instance.
    counts=[dict(loads=0,pairs=0,stores=0)for _ in problems]
    for site,member in _sites(actual,adapter,project_member):
        op=site.operation
        # Owner is decoded from the same actual call projected by _project;
        # no synthetic canonical tensor or source alias supplies membership.
        if type(op)is Load and op.role=='weight':
            _span(storages[0].rhs,op.row,op.rows,op.cols,problems[0].rhs_bytes,site.scopes,maximum)
            continue
        multiplicity=prod(extent for _,_,extent in site.scopes)
        if type(op)is Load:
            _span(slots[member%2],op.row,op.rows,op.cols,problems[member].lhs_bytes,site.scopes,maximum)
            counts[member]['loads']+=multiplicity
        elif type(op)is Compute:
            _span(slots[member%2],op.input_row,op.rows,op.cols,problems[member].lhs_bytes,site.scopes,maximum)
            counts[member]['pairs']+=multiplicity
        elif type(op)is Preload:
            _span(storages[member].rhs,op.weight_row,op.weight_rows,op.cols,problems[member].rhs_bytes,site.scopes,maximum)
            _span(storages[member].accumulator,op.accumulator_row,op.rows,op.cols,problems[member].accumulator_bytes,site.scopes,maximum)
        elif type(op)is Store:
            # A selected multi-block MVOUT advances ACC rows by one local
            # block per 16 columns. Consume all of its decoded bands, while
            # counting the physical issue once for the selected queue bound.
            if type(op.cols) is Const and op.cols.value>problems[member].tile_n:
                from shared_rhs_readout_bands import expand_ready_store_bands
                bands=expand_ready_store_bands(op,problems[member],storages[member])
            else:bands=(op,)
            for band in bands:
                _span(storages[member].accumulator,band.accumulator_row,band.rows,band.cols,problems[member].accumulator_bytes,site.scopes,maximum)
            counts[member]['stores']+=multiplicity
        else:raise ValueError('input pipeline has an unknown actual memory effect')
    for member,count in enumerate(counts):
        bound,admission=laws[member]
        if (count['loads']>bound.max_issued_loads or count['pairs']>bound.max_issued_compute_pairs
                or count['stores']>admission.max_issued_readouts):
            raise ValueError('input pipeline exceeds selected issued-work bounds')
    # Only next-member load moves across current work. Its slot is disjoint
    # from current LHS, all RHS, ACC and external outputs. Stage i precedes
    # compute i; final full drain precedes any later reuse of either slot or ACC.
    record=dict(selected=True,slot_rows=slot_rows,slot_views=[asdict(s)for s in slots],member_counts=counts,
        original_kernel_sha256=original.kernel.digest(),actual_kernel_sha256=actual.digest(),
        selected_overlap_laws=[dict(bound=asdict(b),admission=asdict(a))for b,a in laws],
        selected_epoch_law=asdict(epoch),
        source_transport='exact original pointers and complete data calls; only private LHS load/consume rows relocated',
        epoch_transport='prologue stage0; next stage plus current complete ready work; final current work; each fully drained',
        proof_boundary='original complete source/pointer proof plus exact deterministic transport and conservative actual interval bounds; no canonical alias',
        dynamic_epoch_drains=len(problems)+2)
    return actual,record


def pipeline_ready_members(original,problems,storages,adapter,encode_member,project_member,reduction,epoch):
    try:
        actual,record=_transport(original,problems,storages,adapter,encode_member,project_member,reduction,epoch)
    except ValueError as error:
        original.revalidate()
        return replace(original,realization={**original.realization,'input_pipeline':dict(selected=False,reason=str(error))})
    root=Path(__file__).parent
    paths=('shared_rhs_input_pipeline.py','shared_rhs_ready.py','shared_rhs_readout_bands.py','resident_chain.py',
        'dense_readout.py','encoding.py','dense.py','data/command_ordering.json','data/facts.json',
        'compiler_library/merlin/compile/scheduling/dense_shared_rhs.py',
        'compiler_library/merlin/compile/scheduling/dense_panel_pipeline.py',
        'compiler_library/merlin/compile/scheduling/llvm_program.py')
    owners={name:sha256((root/name).read_bytes()).hexdigest()for name in paths}
    owner=owners['shared_rhs_input_pipeline.py']
    def guard(result):
        if any(sha256((root/name).read_bytes()).hexdigest()!=identity for name,identity in owners.items()):
            raise ValueError('input pipeline proof or selected emission owner changed')
        rebuilt,proof=_transport(original,problems,storages,adapter,encode_member,project_member,reduction,epoch)
        if (result.kernel!=actual or rebuilt!=actual or result.realization['input_pipeline']!=proof
                or result.realization['input_pipeline_source_owners']!=owners):
            raise ValueError('input pipeline actual effects or selected premises changed')
    result=SharedRHSInputPipelineResult(original,actual,{**original.realization,
        'schema':'merlin.shared_rhs_member_input_pipeline.v1','kernel_sha256':actual.digest(),
        'dynamic_epoch_drains':record['dynamic_epoch_drains'],'input_pipeline':record,
        'owner_sha256':owner,'input_pipeline_source_owners':owners},guard)
    result.revalidate()
    return result
