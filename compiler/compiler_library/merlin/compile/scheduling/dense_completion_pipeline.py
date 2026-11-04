"""Registered post-storage completion alternatives on the actual counted kernel."""
from dataclasses import asdict
from merlin.sched.ir import check_structure
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession
from merlin.transforms.pipeline_policy import (
    RegisteredPass, PipelineVariant, PipelinePolicy, PolicyFacts,
    PropertyPredicate, ResourcePredicate, execute_policy,
)
from .compact_dense import _digest
from .dense_completion import bind_dense_completion_scope, _owner
from .dense_panel_epochs import STREAM_FLAGS


def prepare_dense_completion_pipeline(original,context,law,placement,*,enabled=True):
    """Create the same registered pass contracts for drained and ordered-K choices.

    placement is the caller's production source-bound entry. The selected
    checked result is the artifact handed directly to the ordinary emitter.
    """
    def verify(kernel):
        errors = check_structure(kernel)
        if errors: raise ValueError('; '.join(errors))
    p = context['problem']
    session = PipelineSession(original.kernel,ir_level='dense_commands',
        fingerprint=lambda kernel:kernel.digest(),verify_ir=verify,
        context=PipelineContext(hardware=law.hardware_identity,
            numerical_policy=law.numerical_policy_identity,
            resources=_digest((asdict(context['resources']),asdict(context['storage']))),
            execution=_digest(asdict(law)),library=_digest(_owner())))
    source = PassSpec('completion_source',3,produces=('completion_input',),
        mutates_ir=False,ir_level='dense_commands')
    scope = PassSpec('completion_issued_scope',4,requires=('completion_input',),
        produces=('scoped_completion_input',),mutates_ir=False,ir_level='dense_commands')
    ordered = PassSpec('completion_ordered_k',4,requires=('scoped_completion_input',),
        produces=('completed_schedule',),mutates_ir=True,ir_level='dense_commands')
    drained = PassSpec('completion_drained',4,requires=('completion_input',),
        produces=('completed_schedule',),mutates_ir=False,ir_level='dense_commands')
    final = PassSpec('completion_emission_input',5,requires=('completed_schedule',),
        produces=('checked_completion_result',),mutates_ir=False,ir_level='dense_commands')
    def source_action(s):
        original.revalidate()
        if s.module != original.kernel: raise ValueError('completion source changed before selection')
        return {'completion_input':original}
    def scope_action(s):
        current = s.analysis('completion_input')
        scoped = bind_dense_completion_scope(current,context,law)
        selected = dict(context,admission=getattr(scoped,'admission',context['admission']))
        return {'scoped_completion_input':(scoped,selected)}
    def ordered_action(s):
        current,selected = s.analysis('scoped_completion_input')
        result = placement(current,selected)
        s.module = result.kernel
        return {'completed_schedule':result}
    def drained_action(s):
        return {'completed_schedule':s.analysis('completion_input')}
    def final_action(s):
        result = s.analysis('completed_schedule')
        result.revalidate()
        if result.kernel != s.module: raise ValueError('completion result disconnected from actual emission input')
        return {'checked_completion_result':result}
    registry = {spec.name:RegisteredPass(spec,action) for spec,action in (
        (source,source_action),(scope,scope_action),(ordered,ordered_action),
        (drained,drained_action),(final,final_action))}
    policy = PipelinePolicy((
        PipelineVariant('ordered_k',(source.name,scope.name,ordered.name,final.name),(
            PropertyPredicate('capability','completion_selected'),
            PropertyPredicate('capability','known_stream_ordering'),ResourcePredicate('k_tiles',2))),
        PipelineVariant('drained',(source.name,drained.name,final.name))),fallback='drained')
    facts = PolicyFacts(session.stamp(),shapes=((p.m,p.k),(p.k,p.n)),
        element_types=('i8','i8'),operations=frozenset(('matmul',)),
        capabilities=(('completion_selected',enabled),('known_stream_ordering',None if law.unknown(STREAM_FLAGS) else True)),
        resources=(('k_tiles',p.tiles[1]),))
    return dict(session=session,policy=policy,registry=registry,facts=facts,output='checked_completion_result')


def run_dense_completion_pipeline(original,context,law,placement,*,enabled=True):
    product = execute_policy(**prepare_dense_completion_pipeline(original,context,law,placement,enabled=enabled))
    result = product.artifact
    result.realization['completion_pipeline'] = product.receipt
    return result
