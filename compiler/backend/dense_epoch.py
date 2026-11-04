"""Source-qualified completion epochs for checked segmented reductions.

Only compute-containing chains merge. Pure staging, initialization, stores and
their boundaries remain intact. The shared epoch consumer proves that every
future RHS load is consumed before its slot is reused, including backedges.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import time

from merlin.sched.ir import Loop, check_structure
from merlin.compile.scheduling.block_schedule import Compute, Load, Preload
from merlin.compile.scheduling.dense_panel_epochs import (
    DensePanelEpochLaw, _is_chain, _is_epoch, _kinds, _count, _epochs,
    _strip, _drains, _prove_stream, _record,
)
from merlin.compile.scheduling.dense_panel_pipeline import _serial_projection


def _facts(adapter):
    document=json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    ordering=document['ordering']
    digest=sha256(json.dumps(ordering,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (digest!='ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780'
            or document['base_facts_sha256']!='b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'
            or ordering.get('status')!='derived'):
        raise ValueError('segmented epochs lack selected ordering facts')
    waits=[]
    for older,younger in (('MVIN2.spad_addr','PRELOAD.weight_spad'),
                          ('PRELOAD.weight_spad','MVIN2.spad_addr')):
        rows=[row for row in ordering['cross_queue'] if row['older']==older
            and row['younger']==younger and row['space']=='scratchpad']
        if len(rows)!=1 or rows[0]['overlap']!='completion' or rows[0].get('tracked') is not True:
            raise ValueError('segmented epochs lack tracked RHS load/descriptor completion')
        waits.append(rows[0])
    if ordering['completion_order']['execute']['order']!='in_order':
        raise ValueError('segmented epochs lack ordered accumulator completion')
    for lane,minimum in (('load',8),('execute',4)):
        entry=document['finite_entries'][lane]
        if entry.get('status')!='derived' or entry.get('value',0)<minimum:
            raise ValueError('segmented epochs lack finite queue progress')
    # Configuration does not occur inside the transformed counted kernel.
    # The caller establishes it behind its original opening full drain.
    law=DensePanelEpochLaw(adapter.configuration_identity,adapter.numerical_policy_identity,
        'source-qualified-segmented-rhs:'+digest,
        local_read_after_write=True,local_write_after_read=True,
        ordered_accumulator_writes=True,ordered_configuration_capture=True,
        finite_queue_progress=True,complete_device_drain=True,exclusive_private_state=True)
    return law,dict(ordering_sha256=digest,base_facts_sha256=document['base_facts_sha256'],
        tracked_rhs_waits=waits,finite_entries={k:v['value'] for k,v in document['finite_entries'].items()},
        scope='RHS loads and descriptor reads only; original checked LHS/RHS views disjoint; pure staging drains retained',
        configuration='no configuration effect in merged streams; original caller configuration boundary unchanged')


def _plan(kernel,adapter,max_pairs,max_loads):
    streams=[]
    operands={Load,Preload,Compute}
    def chain(statement):
        kinds=_kinds(statement,adapter) if _is_chain(statement) else set()
        return bool(kinds) and kinds<=operands and Compute in kinds
    def walk(body,ancestry):
        output=[]
        index=0
        while index<len(body):
            end=index
            while end<len(body) and chain(body[end]):
                end+=1
            last=end-1
            while last>=index and not _is_epoch(body[last]):
                last-=1
            elements=tuple(body[index:last+1])
            pairs=sum(_count(e,adapter,{Compute}) for e in elements)
            loads=sum(_count(e,adapter,{Load}) for e in elements)
            if elements and sum(_epochs(e) for e in elements)>1 and pairs<=max_pairs and loads<=max_loads:
                output.extend((*(_strip(e) for e in elements[:-1]),elements[-1]))
                streams.append(dict(ancestry=ancestry,elements=elements,readout=None,
                    compute_pairs=pairs,loads=loads,readouts=0,removed_drains=_drains(elements)-1))
                index=last+1
                continue
            statement=body[index]
            if type(statement) is Loop and not _is_epoch(statement):
                statement=replace(statement,body=walk(statement.body,(*ancestry,(statement.var,statement.extent))))
            output.append(statement)
            index+=1
        return tuple(output)
    return replace(kernel,body=walk(kernel.body,())),streams


def _owner():
    return sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class DenseSegmentEpochResult:
    original: object
    kernel: object
    realization: dict
    checked_kernel_sha256: str
    source_kernel_sha256: str
    owner_sha256: str
    adapter: object
    max_pairs: int
    max_loads: int
    facts: dict

    @property
    def source_identity(self):
        return self.original.source_identity

    @property
    def admission(self):
        return self.original.admission

    def revalidate(self):
        self.original.revalidate()
        _,facts=_facts(self.adapter)
        actual,_=_plan(self.original.kernel,self.adapter,self.max_pairs,self.max_loads)
        if (self.original.kernel.digest()!=self.source_kernel_sha256
                or actual!=self.kernel or self.kernel.digest()!=self.checked_kernel_sha256
                or self.owner_sha256!=_owner() or facts!=self.facts):
            raise ValueError('segmented epoch source, facts, proof owner or actual effects changed')

    def emit_llvm(self,builder,lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter,emit_index_program
        self.revalidate()
        result=emit_index_program(self.kernel,FnBuilderIndexAdapter(builder),lower_call)
        self.revalidate()
        return result

    def materialize_scf(self):
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands
        self.revalidate()
        result=materialize_scf_commands(self.kernel)
        self.revalidate()
        return result

    def emit_index_program(self,control,lower_effect,*,complete_epoch):
        from merlin.compile.scheduling.llvm_program import emit_index_effect_program
        from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
        self.revalidate()
        result=emit_index_effect_program(self.kernel,control,lower_effect,
            complete_epoch=complete_epoch,completion_instruction=DRAIN)
        self.revalidate()
        return result


def select_segment_epochs(original,problem,block,storage,adapter,selection):
    """Separate the supported performance class from epoch legality proofs."""
    if selection not in ('off','on','capacity_guarded'):
        raise ValueError('unsupported segmented epoch selection')
    original.revalidate()
    if selection=='off' or (selection=='capacity_guarded' and problem.tiles[0]!=1):
        return original,dict(applied=False,selection=selection,m_tiles=problem.tiles[0],
            refusal=('segmented epochs disabled' if selection=='off' else
                     'additional M tiles lack matched complete timing support'))
    result,report=schedule_segment_epochs(original,problem,block,storage,adapter)
    return result,dict(report,selection=selection,m_tiles=problem.tiles[0])


def schedule_segment_epochs(original,problem,block,storage,adapter):
    """Return an actual source-checked alternative, or unchanged plus refusal."""
    original.revalidate()
    try:
        law,facts=_facts(adapter)
        law.validate_for(adapter,original.admission.overlap)
        # These are exact finite command counts, not queue capacity claims.
        max_pairs=block.m*problem.tiles[1]*block.n
        max_loads=problem.tiles[1]*block.n
        actual,streams=_plan(original.kernel,adapter,max_pairs,max_loads)
        if not streams:
            return original,dict(applied=False,refusal='no mergeable checked compute chains')
        if check_structure(actual) or _serial_projection(actual)!=_serial_projection(original.kernel):
            raise ValueError('segmented epochs changed original data effects or counted scopes')
        from merlin.compile.scheduling.dense_panel_pipeline import _calls
        for stream in streams:
            for element in stream['elements']:
                for call in _calls((element,)):
                    if call.instr=='merlin_dense_panel_drain':
                        continue
                    operation=adapter.project(call)
                    if type(operation) is Load and operation.role!='weight':
                        raise ValueError('segmented epochs must keep LHS staging completion')
        # Original admission proves private disjoint operand assignment. Only
        # original RHS loads occur here; accumulator partial sums and their
        # descriptor/compute issue order are identical to that checked source.
        deadline=time.monotonic()+30
        proofs=sum(_prove_stream(stream,adapter,storage,deadline,4096) for stream in streams)
    except ValueError as error:
        return original,dict(applied=False,refusal=str(error))
    report=dict(schema='authored_dense_segment_epochs.v1',applied=True,
        source_kernel_sha256=original.kernel.digest(),kernel_sha256=actual.digest(),
        source=original.realization['source'],law=asdict(law),facts=facts,
        max_issued_compute_pairs=max_pairs,max_issued_loads=max_loads,
        pending_compute_pairs=original.admission.overlap.pending_compute_pairs,
        pending_loads=original.admission.overlap.pending_loads,
        consumption_proofs=proofs,streams=[_record(s) for s in streams],
        original_dynamic_drains=_drains(original.kernel.body),dynamic_drains=_drains(actual.body),
        preserved='pure staging, initialization, all readouts, output group boundaries and final completion',
        data_projection='identical original calls, arguments, loop scopes and issue order')
    result=DenseSegmentEpochResult(original,actual,dict(original.realization,
        kernel_sha256=actual.digest(),segment_epochs=report),actual.digest(),
        original.kernel.digest(),_owner(),adapter,max_pairs,max_loads,facts)
    result.revalidate()
    return result,report
