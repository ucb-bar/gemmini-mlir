"""Conservative convolution route adapter for the shared empirical selector.

This file does not promote settings, read live timing feedback, or infer a win
from command counts. Frozen typed feedback and validation partitions go through
assess_shape_policy. Missing evidence/domain pins retain synchronous lowering.
The emit_route callback must emit the current plan with the requested storage
and completion options through conv_policy and the common terminal encoder.
"""
from dataclasses import dataclass
from pathlib import Path
from math import prod
from hashlib import sha256
import json

from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan
from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext, PipelineSession, PassSpec
from merlin.transforms.pipeline_policy import (
    PipelineVariant, PipelinePolicy, PolicyFacts, PropertyPredicate, RegisteredPass, execute_policy,
)
from merlin.transforms.shape_policy import (
    ShapeScheduleClass, ShapeCaseFeedback, PerformanceDomain, assess_shape_policy,
)
from merlin.transforms.evaluated_exploration import ExplorationIdentity, CompletedObservation
from merlin.transforms.pass_exploration import ExplorationGeometry, ValidationPartition


VALUES=('off','validated_epochs','supported_geometry')
DECLARATION={
    'values':list(VALUES),
    'mechanism':'shared paired-evidence convolution shape policy',
    'note':'Off preserves explicit schedules. Strict validated_epochs needs typed pins; supported_geometry uses frozen completed pairs and reviewed geometry guards. Both emit shared routes.',
}
INCUMBENT='conv_synchronous'
CANDIDATE='conv_retained_input_pairs16'
OUTPUT='convolution_emission'
EVIDENCE_FILE='conv_selection_evidence.json'


@dataclass(frozen=True)
class Route:
    schedule: str
    completion_epochs: str


ROUTES={INCUMBENT:Route('synchronous','off'),
        CANDIDATE:Route('retained_input','pairs16')}


def selected():
    return knob('conv_selection',default='off',values=VALUES,root=Path(__file__).parent)


def geometry_resources(plan,*,pooling=False,per_column=False,source_bias=False,
                       ordering_supported=False):
    """Original geometry and selected capacities, independent of workload names.

    These admission conditions describe the deliberately narrow first panel,
    not unsupported capabilities of the general synchronous implementation.
    plan.bias also includes explicit source-zero initialization and therefore
    remains separate from source_bias in the public interface.
    """
    if type(plan) is not CountedConvolutionPlan:
        raise ValueError('selection requires the original counted convolution plan')
    plan.__post_init__()
    w=plan.window
    ci=w.input_shape[-1]
    oh,ow=w.output_spatial
    edge=plan.primitive
    tiles=(plan.output_channels+edge-1)//edge
    reduction=prod(w.kernel)*ci
    acc_needed=tiles*plan.row_tile
    aligned=ci%edge==0 and plan.output_channels%edge==0
    ragged=ow%edge!=0
    resources=dict(source_reduction_elements=reduction,array_edge=edge,
        source_single_slice=int(reduction<=edge),column_tiles=tiles,
        spatial_row_tail=ow%edge,input_channel_tail=ci%edge,
        output_channel_tail=plan.output_channels%edge,
        accumulator_rows_needed=acc_needed,accumulator_rows=plan.accumulator_rows,
        accumulator_rows_available=plan.accumulator_rows-plan.accumulator_row,
        output_spatial_elements=oh*ow*w.input_shape[0])
    conditions=dict(
        multi_slice=reduction>edge,
        multiple_output_tiles=tiles>=2,
        bounded_columns=tiles<=32,
        retained_accumulator_fit=plan.accumulator_row+acc_needed<=plan.accumulator_rows,
        primitive_columns=plan.column_tile==edge,
        aligned_channels=aligned,
        no_per_column_readout=not per_column,
        no_pooling=not pooling,
        source_bias_or_overwrite=bool(source_bias) or not plan.bias,
        spatial_kernel=w.kernel==(3,3),
        unit_dilation=w.dilation==(1,1),
        bounded_stride=w.stride in ((1,1),(2,2)),
        bounded_padding=w.padding in ((0,0,0,0),(1,1,1,1)),
        ordering_supported=ordering_supported is True)
    admitted=all(conditions.values())
    capabilities=dict(conv_retained_epoch_admitted=admitted,
        conv_aligned_spatial_panel=admitted and not ragged,
        conv_ragged_spatial_panel=admitted and ragged)
    return dict(resources=resources,capabilities=capabilities,conditions=conditions,
                exclusions=tuple(k for k,v in conditions.items() if not v),
                geometry_class='ragged_spatial_rows' if ragged else 'aligned_spatial_rows')


def alternatives():
    """The real route callback is registered once for each exact option pair."""
    return (
        PipelineVariant(CANDIDATE,(CANDIDATE,),
            (PropertyPredicate('capability','conv_retained_epoch_admitted'),)),
        PipelineVariant(INCUMBENT,(INCUMBENT,)),
    )


def assess(*,cases=(),designs=None):
    """Use original typed completed observations; JSON summaries are not authority.

    designs maps each geometry class to its frozen ValidationPartition. An
    absent partition, missing pairs, stale implementation domain or held-out
    regression retains the incumbent in the shared evaluator.
    """
    designs={} if designs is None else designs
    classes=tuple(ShapeScheduleClass(name,
        (PropertyPredicate('capability',capability),),INCUMBENT,designs.get(name))
        for name,capability in (
            ('aligned_spatial_rows','conv_aligned_spatial_panel'),
            ('ragged_spatial_rows','conv_ragged_spatial_panel')))
    return assess_shape_policy(classes,alternatives(),tuple(cases),fallback=INCUMBENT)


def implementation_identity(root=None):
    """Pin authored compiler sources, excluding settings and the evidence itself."""
    root=Path(__file__).parent if root is None else Path(root)
    rows=[(p.name,sha256(p.read_bytes()).hexdigest()) for p in sorted(root.glob('*.py'))]
    return sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def library_identity(root=None):
    root=(Path(__file__).parent if root is None else Path(root))/'compiler_library'
    rows=[(str(p.relative_to(root)),sha256(p.read_bytes()).hexdigest())
          for p in sorted(root.rglob('*.py'))]
    return sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def load_frozen(root=None):
    """Load only an authored, pinned typed panel; never read qa or run a probe.

    JSON uses dataclasses.asdict for context/domain/case records and designs.
    Each case has geometry, facts, observations (name/CompletedObservation
    pairs), and implementation_sha256, matching ShapeCaseFeedback fields.
    Missing or stale pins are a normal conservative fallback. Malformed authored
    evidence raises rather than silently claiming a validated route.
    """
    root=Path(__file__).parent if root is None else Path(root)
    facts_pin=sha256((root/'facts.json').read_bytes()).hexdigest()
    fallback=(assess(),PipelineContext(hardware=facts_pin,resources=facts_pin),None,
              dict(status='missing',file=EVIDENCE_FILE))
    path=root/EVIDENCE_FILE
    if not path.exists(): return fallback
    data=json.loads(path.read_text())
    if data.get('schema')!='authored.conv_selection_evidence.v1':
        raise ValueError('unknown frozen convolution evidence schema')
    if data.get('evidence_status')=='pending_identity_pins':
        return (*fallback[:3],dict(status='pending_identity_pins',file=EVIDENCE_FILE,
            panel_sha256=sha256(path.read_bytes()).hexdigest(),
            missing=tuple(data.get('missing',()))))
    context=PipelineContext(**data['context'])
    domain=PerformanceDomain(**data['domain'])
    if (domain.facts_sha256!=facts_pin or context.hardware!=facts_pin
            or context.resources!=facts_pin
            or domain.implementation_sha256!=implementation_identity(root)
            or domain.library_sha256!=library_identity(root)
            or context.library!=domain.library_sha256):
        return (*fallback[:3],dict(status='stale_pins',file=EVIDENCE_FILE))
    def geometry(row):
        return ExplorationGeometry(row['family'],tuple(tuple(s) for s in row['operand_shapes']))
    def policy_facts(row):
        stamp=row['stamp']
        stamp=AnalysisStamp(stamp['source_sha256'],PipelineContext(**stamp['context']),stamp['ir_level'])
        return PolicyFacts(stamp,shapes=tuple(tuple(s) for s in row.get('shapes',())),
            element_types=tuple(row.get('element_types',())),layouts=tuple(row.get('layouts',())),
            operations=None if row.get('operations') is None else frozenset(row['operations']),
            numerical_semantics=None if row.get('numerical_semantics') is None else frozenset(row['numerical_semantics']),
            capabilities=tuple(tuple(r) for r in row.get('capabilities',())),
            resources=tuple(tuple(r) for r in row.get('resources',())),
            graph=tuple(tuple(r) for r in row.get('graph',())))
    cases=[]
    for row in data.get('cases',()):
        observations=[]
        for name,record in row['observations']:
            record=dict(record);record['identity']=ExplorationIdentity(**record['identity'])
            observations.append((name,CompletedObservation(**record)))
        cases.append(ShapeCaseFeedback(geometry(row['geometry']),policy_facts(row['facts']),
            tuple(observations),row['implementation_sha256']))
    designs={name:ValidationPartition(tuple(geometry(g) for g in row['training']),
        tuple(geometry(g) for g in row['held_out']),tuple(row.get('withheld_families',())))
        for name,row in data.get('designs',{}).items()}
    return assess(cases=tuple(cases),designs=designs),context,domain,dict(
        status='loaded',file=EVIDENCE_FILE,panel_sha256=sha256(path.read_bytes()).hexdigest())


def load_supported_geometry(root=None):
    """Factual paired provenance plus reviewed construction, not typed authority.

    This finite authored compiler route does not fabricate CompletedObservation
    pins. Original and held-out timing remains modeled and provisional.
    Compatibility pins refer to current adapter construction, never
    relabel the original measured compiler or grant physical qualification.
    """
    root=Path(__file__).parent if root is None else Path(root)
    facts_pin=sha256((root/'facts.json').read_bytes()).hexdigest()
    context=PipelineContext(hardware=facts_pin,resources=facts_pin)
    path=root/EVIDENCE_FILE
    def declined(reason):return context,dict(status=reason,file=EVIDENCE_FILE)
    if not path.exists():return declined('missing')
    data=json.loads(path.read_text())
    support=data.get('supported_geometry',{})
    if (data.get('schema')!='authored.conv_selection_evidence.v1'
            or support.get('geometry_class')!='ragged_spatial_rows'
            or support.get('schedule')!='retained_input' or support.get('completion_epochs')!='pairs16'):
        return declined('no_supported_geometry_declaration')
    pins=support.get('current_adapter_compatibility_sha256',{})
    required=('convolution.py','conv_policy.py','conv_addressing.py','conv_selection.py','isa.py')
    if set(pins)!=set(required) or any(sha256((root/name).read_bytes()).hexdigest()!=pins[name] for name in required):
        return declined('stale_adapter_compatibility')
    groups={}
    for row in data.get('witnesses',()):
        p=row['performance']
        if (row.get('numerical_status')!='pass' or not p.get('complete_event_accounting')
                or p.get('facts_sha256')!=facts_pin
                or type(p.get('estimated_whole_soc_cycles')) is not int
                or p['estimated_whole_soc_cycles']<=0):return declined('incomplete_original_pair')
        groups.setdefault(p['capsule_sha256'],[]).append(row)
    if len(groups)<2:return declined('insufficient_original_pairs')
    domains=set()
    for rows in groups.values():
        if len(rows)!=2:return declined('incomplete_original_pair')
        for row in rows:
            p=row['performance'];domains.add(tuple(p[k] for k in (
                'model_sha256','scope_sha256','comparison_domain_sha256','facts_sha256')))
        before=[r for r in rows if r['route']==INCUMBENT]
        after=[r for r in rows if r['route']==CANDIDATE]
        if len(before)!=1 or len(after)!=1:return declined('unknown_original_route')
        if after[0]['performance']['estimated_whole_soc_cycles']>=before[0]['performance']['estimated_whole_soc_cycles']:
            return declined('original_pair_regressed')
    if len(domains)!=1:return declined('incomparable_original_domains')
    return context,dict(status='supported_geometry',file=EVIDENCE_FILE,
        panel_sha256=sha256(path.read_bytes()).hexdigest(),original_domain=next(iter(domains)),
        pairs=len(groups),geometry_class='ragged_spatial_rows',
        uncertainty='provisional modeled complete original and held-out pairs; no physical qualification',
        compatibility=support.get('compatibility'))


def execute(plan,emit_route,*,context,policy=None,domain=None,mode=None,
            explicit_options=None,pooling=False,per_column=False,source_bias=False,
            ordering_supported=False,geometry_support=None):
    """Emit one actual route on the current typed source plan.

    explicit_options, if present, always wins, including synchronous/off.
    Otherwise existing nondefault conv_schedule/conv_completion_epochs settings
    win before the automatic policy. Callers that distinguish an intentional
    explicit default must pass it in explicit_options. The callback signature is
    emit_route(plan, schedule=..., completion_epochs=...) -> actual receipt.
    It must run the existing shared schedule emitter, with no retry after issue.
    """
    if not callable(emit_route) or type(context) is not PipelineContext:
        raise ValueError('selection requires an actual route emitter and pinned context')
    mode=selected() if mode is None else mode
    if mode not in VALUES: raise ValueError('unknown convolution selection mode')
    import conv_policy
    current=Route(conv_policy.selected(),conv_policy.epoch_selection())
    explicit=(Route(*explicit_options) if explicit_options is not None else
              current if current!=ROUTES[INCUMBENT] else None)
    geometry=geometry_resources(plan,pooling=pooling,per_column=per_column,
        source_bias=source_bias,ordering_supported=ordering_supported)
    if explicit is not None or mode=='off':
        route=explicit or current
        if route.schedule not in conv_policy.VALUES or route.completion_epochs not in conv_policy.EPOCH_VALUES:
            raise ValueError('unknown explicit convolution options')
        actual=emit_route(plan,schedule=route.schedule,completion_epochs=route.completion_epochs)
        return dict(selection_mode=mode,selection_reason='explicit_options' if explicit else 'disabled',
                    requested_route=vars(route),geometry=geometry,actual_route=actual)
    session=PipelineSession(plan,context=context,ir_level='counted-convolution',
        fingerprint=lambda value:value.identity,verify_ir=lambda value:value.__post_init__())
    facts=PolicyFacts(session.stamp(),
        shapes=(plan.window.input_shape,(*plan.window.kernel,plan.window.input_shape[-1],plan.output_channels)),
        element_types=('i8','i8'),layouts=('NHWC','HWIO'),
        operations=frozenset(('convolution',)),
        capabilities=tuple(sorted(dict(geometry['capabilities'],
            conv_frozen_geometry_supported=geometry_support is not None and
                geometry_support.get('status')=='supported_geometry').items())),
        resources=tuple(sorted(geometry['resources'].items())))
    def action(name):
        route=ROUTES[name]
        return lambda current_session:{OUTPUT:emit_route(current_session.module,
            schedule=route.schedule,completion_epochs=route.completion_epochs)}
    registry={name:RegisteredPass(PassSpec(name,0,produces=(OUTPUT,),
        mutates_ir=False,ir_level='counted-convolution'),action(name)) for name in ROUTES}
    if mode=='supported_geometry':
        path=PipelineVariant(CANDIDATE,(CANDIDATE,),(
            PropertyPredicate('capability','conv_ragged_spatial_panel'),
            PropertyPredicate('capability','conv_frozen_geometry_supported')))
        result=execute_policy(session,PipelinePolicy((path,alternatives()[1]),INCUMBENT),
            registry,facts,output=OUTPUT)
        report=dict(construction='finite authored supported geometry',provenance=geometry_support,
                    qualification='reviewed route applicability; original modeled pairs only')
    else:
        policy=assess() if policy is None else policy
        result=policy.execute(session,registry,facts,output=OUTPUT,domain=domain)
        report=policy.report
    name=result.receipt['selected_variant']
    return dict(selection_mode=mode,selection_reason='shared_shape_policy',
        requested_route=vars(ROUTES[name]),geometry=geometry,actual_route=result.artifact,
        policy_report=report,policy_receipt=result.receipt)
