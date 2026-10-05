"""Source-bound exports through Merlin's existing inventory and plan seams.

This is an operation export, not a whole-model compiler or timing solver.
The selected primitive schedule is compiled by the ordinary OOT lowerer and
accounted for by GlobalPlanEmission. Surrounding source operations remain
outside its declared scope. Unobserved occupancy is never filled with zeros.
"""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from xdsl.dialects.builtin import StringAttr

from .b_slot_placement import select_remaining_b_slots
from .dense_schedule import select_kernel
from .golden_contraction_upstream import select
from .golden_device_compile import compile_module


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_inventory(package):
    """Resolve real AST/component owners and use the shared edit-contract checks."""
    from merlin.perf.agent_guidance import inspect_compiler_package,build_compiler_edit_contract
    from merlin.perf.phase2_edit_contract import load,validate_against_package
    package=Path(package).resolve(strict=True)
    inventory=inspect_compiler_package(package)
    contract=build_compiler_edit_contract(inventory)
    contract['required_decisions']=[dict(decision=s.id,owner=f'{s.path}:{s.symbol}')
                                    for s in inventory.surfaces]
    manifest=package/'manifest.yaml'
    import yaml
    metadata=yaml.safe_load(manifest.read_text())
    contract=load(metadata['target'],metadata['package_id'],body=contract)
    validate_against_package(contract,package)
    return dict(schema='golden_package_inventory_export_v1',
                manifest_sha256=_sha(manifest),package=str(package),
                package_inventory=inventory.to_dict(),compiler_edit_contract=contract,
                authority='Descriptive package export; the host must freeze and enforce the existing contract before edits',
                shared_solver_selected=False)


class SourceContractionEmitter:
    """Compile one bound alternative and return the shared executable accounting."""

    def __init__(self,source_path,logical,alternative,generator,llvm_bin,workdir):
        from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest
        self.source_path=Path(source_path)
        self.source_sha256=dict(alternative.metadata)['source_sha256']
        if _sha(source_path)!=self.source_sha256:
            raise ValueError('contraction source changed before emission binding')
        self.logical_digest=dispatch_digest(logical)
        self.alternative=alternative
        self.generator=generator
        self.schedule_binding=(asdict(generator.shape),generator.prefetch_b_rows)
        self.llvm_bin=Path(llvm_bin)
        self.workdir=Path(workdir)
        self.compilation=None

    def emit_global_plan(self,program,plan):
        from merlin.xdsl_dialects.lowering.dispatch_program import DispatchProgram,Node
        from merlin.xdsl_dialects.lowering.global_plan_emission import (
            BoundaryMapping,EmittedComponent,GlobalPlanEmission,dispatch_digest,
        )
        if _sha(self.source_path)!=self.source_sha256 or dispatch_digest(program)!=self.logical_digest:
            raise ValueError('contraction emission source or logical dispatch binding changed')
        if plan.transitions or plan.selected!=(self.alternative,):
            raise ValueError('contraction export requires its exact bound singleton alternative')
        if (asdict(self.generator.shape),self.generator.prefetch_b_rows)!=self.schedule_binding:
            raise ValueError('selected generator schedule changed after binding')
        module=self.generator.build()
        module.body.block.first_op.properties['sym_name']=StringAttr(self.alternative.implementation)
        self.compilation=compile_module(module,self.llvm_bin,self.workdir)
        if _sha(self.source_path)!=self.source_sha256:
            raise ValueError('contraction source changed during emission')
        dispatch=DispatchProgram(program.entry,list(program.args),dict(program.buffers),
            [Node('dispatch',self.alternative.implementation,['lhs','rhs'],['output'],
                  prov=dict(program.nodes[0].prov),captures=[])],list(program.results))
        return GlobalPlanEmission(dispatch,plan.digest,self.logical_digest,
            (EmittedComponent(self.alternative.id,(0,),
                (('lhs','lhs'),('rhs','rhs')),(('output','output'),)),),(),
            (BoundaryMapping('input','lhs','lhs'),BoundaryMapping('input','rhs','rhs'),
             BoundaryMapping('output','output','output')),
            (f'source_sha256:{self.source_sha256}',
             f'target_ir_sha256:{self.compilation["target_ir_sha256"]}',
             f'object_sha256:{self.compilation["object_sha256"]}',
             'scope:one exact source contraction; surrounding model work is unmodified and unaccounted here'))


def select_contraction_export(source_path,region_id,*,dense_input_policy=None,
                              dense_b_slot_policy=None,large_n=False,prefetch_b=False):
    """Derive one legal source alternative without compiling or pricing it."""
    from merlin.xdsl_dialects.lowering.dispatch_program import Buffer,DispatchProgram,Node
    from merlin.xdsl_dialects.lowering.global_plan import (
        BufferRepresentation,CycleInterval,RegionAlternative,ValueRepresentation,
    )
    if dense_input_policy not in (None,'banked_command_cost','resident_a_command_cost','transfer_command_cost'):
        raise ValueError('unknown dense compiler policy')
    if dense_b_slot_policy not in (None,'remaining_rows'):
        raise ValueError('unknown B slot compiler policy')
    source_path=Path(source_path)
    source_bytes=source_path.read_bytes()
    source_sha=hashlib.sha256(source_bytes).hexdigest()
    dims,shape,binding=select(source_bytes.decode(),region_id,large_n=large_n,prefetch_b=prefetch_b)
    if binding['abi']!='gemmini_golden_gemm':
        raise ValueError('operation export currently admits unbatched integer GEMM only')
    generator,kind=select_kernel(shape,
        banked_command_policy=dense_input_policy=='banked_command_cost',
        resident_a_command_policy=dense_input_policy=='resident_a_command_cost',
        transfer_command_policy=dense_input_policy=='transfer_command_cost')
    b_decision=None
    if dense_b_slot_policy is not None:
        generator,b_decision=select_remaining_b_slots(generator)
        if b_decision['applied']:kind+=',remaining_rows_b_prefetch'
    key=dict(source_sha256=source_sha,binding=binding,schedule=asdict(generator.shape),
             generator=type(generator).__qualname__,schedule_kind=kind,
             prefetch_b_rows=generator.prefetch_b_rows)
    digest=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    symbol='gemmini_export_'+digest[:16]
    unknown=CycleInterval.unknown('No source-bound timing calibration was supplied')
    logical=DispatchProgram('contraction_export',[0,1],{
        'lhs':Buffer('lhs',[dims.m,dims.k],'i8','arg',0),
        'rhs':Buffer('rhs',[dims.k,dims.n],'i8','arg',1),
        'output':Buffer('output',[dims.m,dims.n],'i32','intermediate')},
        [Node('dispatch',binding['source_operation'],['lhs','rhs'],['output'],
              prov={'source_sha256':source_sha,'region_id':region_id,
                    'source_operation_ordinal':str(binding['source_operation_ordinal'])},captures=[])],['output'])
    rep=lambda name:ValueRepresentation('memory','dense_row_major',logical.buffers[name].dtype)
    alternative=RegionAlternative('contraction_'+digest[:16],(0,),symbol,'gemmini',unknown,
        inputs=tuple(BufferRepresentation(name,rep(name))for name in ('lhs','rhs')),
        outputs=(BufferRepresentation('output',rep('output')),),
        metadata=(('source_sha256',source_sha),))
    return dict(source_sha256=source_sha,dimensions=asdict(dims),binding=binding,
        generator=generator,schedule_kind=kind,b_slot_decision=b_decision,
        kernel_symbol=symbol,logical=logical,alternative=alternative)


def export_contraction(source_path,region_id,llvm_bin,workdir,*,
                       dense_input_policy=None,dense_b_slot_policy=None,
                       large_n=False,prefetch_b=False,activity_timeline=None,
                       activity_artifact_sha256=None):
    """Select, compile and verify the actual exported operation implementation."""
    from merlin.xdsl_dialects.lowering.global_plan import GlobalPlan
    from merlin.xdsl_dialects.lowering.global_plan_emission import emit_global_plan
    if activity_timeline is None and activity_artifact_sha256 is not None:
        raise ValueError('activity artifact identity requires a timeline')
    options=dict(dense_input_policy=dense_input_policy,dense_b_slot_policy=dense_b_slot_policy,
                 large_n=large_n,prefetch_b=prefetch_b)
    selection=select_contraction_export(source_path,region_id,**options)
    logical,alternative,generator=(selection[name]for name in ('logical','alternative','generator'))
    unknown=alternative.cycles
    plan=GlobalPlan((alternative,),(),unknown,
                    notes=('Explicit source compiler decision; shared cycle-ranking solver is not invoked',))
    workdir=Path(workdir)
    emitter=SourceContractionEmitter(source_path,logical,alternative,generator,llvm_bin,workdir)
    emission=emit_global_plan(logical,plan,emitter)
    activity=None
    if activity_timeline is not None:
        from merlin.perf.activity_schedule import ActivityTimeline,schedule_activity
        if not isinstance(activity_timeline,ActivityTimeline):
            raise TypeError('activity evidence must use the shared ActivityTimeline')
        if activity_artifact_sha256!=emitter.compilation['object_sha256']:
            raise ValueError('activity evidence is not bound to the emitted object')
        if (not activity_timeline.events or not math.isfinite(activity_timeline.total_cycles) or
                not math.isfinite(activity_timeline.critical_path_cycles) or
                any(not row.event.provenance or not math.isfinite(row.event.cycles)
                    for row in activity_timeline.events)):
            raise ValueError('activity evidence requires finite durations and explicit provenance')
        if schedule_activity([row.event for row in activity_timeline.events])!=activity_timeline:
            raise ValueError('activity timeline differs from the shared scheduler result')
        event_provenance={row.event.id:row.event.provenance for row in activity_timeline.events}
        activity=dict(scope='Caller-supplied duration evidence, bound to the emitted object; no simulator observation is inferred',
                      object_sha256=activity_artifact_sha256,timeline=activity_timeline.to_dict(),
                      event_provenance=event_provenance,
                      occupancy=asdict(activity_timeline.occupancy(
                          provenance=(f'object_sha256:{activity_artifact_sha256}',
                                      *event_provenance.values()))))
    result=dict(schema='golden_source_contraction_export_v1',source_sha256=selection['source_sha256'],
        dimensions=selection['dimensions'],binding=selection['binding'],schedule=asdict(generator.shape),
        schedule_kind=selection['schedule_kind'],prefetch_b_rows=generator.prefetch_b_rows,
        compiler_options=options,
        b_slot_decision=selection['b_slot_decision'],kernel_symbol=selection['kernel_symbol'],compilation=emitter.compilation,
        logical_dispatch=logical.to_dict(),global_plan=plan.to_dict(),
        global_plan_emission=emission.receipt(),emitted_dispatch=emission.dispatch.to_dict(),
        activity=activity,occupancy_status='unknown' if activity is None else 'caller_supplied',
        source_scope='Exactly one matched contraction; surrounding model operations are outside this export',
        shared_solver_selected=False,selected_plan_controls_emitted_code=True,
        whole_model_memory_bound=False,whole_model_correctness_verified=False)
    (workdir/'golden_export.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def export_capture(source_path,llvm_bin,workdir,**options):
    """Delegate to the existing exact capture binder; preserve its actual scope."""
    from .captured_requant_bundle import build
    source_path=Path(source_path)
    if source_path.name!='model.mlir':
        raise ValueError('capture export requires the pinned capture model.mlir')
    result=build(source_path.parent,Path(llvm_bin),Path(workdir),**options)
    return dict(schema='golden_capture_export_v1',bundle=result,
                source_scope='Only source-proven captured epilogue routes in requant.json',
                shared_solver_selected=False,whole_model_plan_emitted=False,
                whole_model_correctness_verified=False)
