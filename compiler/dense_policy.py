"""Frozen hardware-relative dense policy design, using shared shape_policy.

The auto route selects a checked epoch schedule when the logical output
needs more than one accumulator store. Its declared panel has original paired
whole-call feedback; small outputs retain compact. No capsule-name dispatch or
simulation occurs during compilation; ordinary current-source pass checks apply.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json

from merlin.transforms.pass_exploration import ExplorationGeometry,ValidationPartition
from merlin.transforms.pipeline_policy import (DimensionRef,ResourceRef,TileFeature,
    FeaturePredicate,ResourcePredicate,PipelineVariant,PolicyFacts,RegisteredPass)
from merlin.transforms.pipeline_state import PassSpec,PipelineContext,PipelineSession
from merlin.transforms.shape_policy import ShapeScheduleClass,assess_shape_policy


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def operand_geometry(m,k,n):
    return ExplorationGeometry('dense',((m,k),(k,n)))


def partition(training,held_out):
    return ValidationPartition(tuple(operand_geometry(*s) for s in training),
        tuple(operand_geometry(*s) for s in held_out))


def alternatives():
    return (PipelineVariant('compact',('compact',)),
        PipelineVariant('panel_epochs',('panel_epochs',),(ResourcePredicate('epochs_admitted',at_least=1),)),
        PipelineVariant('ready_readout',('ready_readout',),(ResourcePredicate('ready_admitted',at_least=1),)))


def classes(design=None):
    """One reduction tile versus multiple tiles; M/tail are panel interactions.

    Guards use integer ceiling tile counts, so the boundary neighbors are 1/2
    tiles on the selected array rather than an empirical fixed tensor size.
    Alignment and source legality stay with the checked scheduling consumers.
    """
    k_tiles=TileFeature(DimensionRef(0,1),ResourceRef('array_edge'))
    return (ShapeScheduleClass('single_reduction_tile',
            (FeaturePredicate(k_tiles,at_most=(1,1)),),'compact',design),
        ShapeScheduleClass('multiple_reduction_tiles',
            (FeaturePredicate(k_tiles,at_least=(2,1)),),'compact',design))


def assess(design=None,feedback=()):
    """Feedback is typed ShapeCaseFeedback from exact ordinary owner identities."""
    return assess_shape_policy(classes(design),alternatives(),tuple(feedback),fallback='compact',
        max_regression_ratio=1.0)


def execution_owner(incumbent,adapter,problem,geometry,facts_path,*,bias=False,initial=False,
                    source_sha256=None,pipeline_context=None,include_ready=True):
    """Bind registered real passes over the same source and checked dense artifact.

    Produces current PolicyFacts and registry for AssessedShapePolicy.execute;
    callers must provide its exact current PerformanceDomain to activate any
    assessed non-incumbent path. The returned artifact uses the ordinary dense
    terminal encoder via consume_checked_dense, without any policy-specific ISA.
    """
    from gemmini_dense_pipeline import plan_pipeline
    from gemmini_dense_readout import plan_ready_readout,ReadoutUnavailable
    source={'incumbent':incumbent,'problem':problem}
    context=pipeline_context or PipelineContext(hardware=adapter.configuration_identity,
        resources=_digest([asdict(geometry[1]),asdict(geometry[3])]),
        numerical_policy=adapter.numerical_policy_identity,
        execution='ordinary-source-bound-dense',library='current-shared-dense-scheduling')
    original_effects=incumbent.kernel.digest()
    def verify_source(state):
        state['incumbent'].revalidate()
        if state['incumbent'].kernel.digest()!=original_effects or state['problem']!=problem:
            raise ValueError('policy source artifact changed across registered analysis passes')
    session=PipelineSession(source,context=context,ir_level='checked_dense',
        fingerprint=lambda state:source_sha256 or _digest({'problem':asdict(state['problem']),
            'source_effects':state['incumbent'].kernel.digest()}),
        verify_ir=verify_source)
    # Legality discovery consumes the same shared checks. It never predicts speed.
    checked={'compact':incumbent};receipts={};refusals={}
    try:checked['panel_epochs'],receipts['panel_epochs']=plan_pipeline(incumbent,adapter,problem,
        geometry,facts_path,mode='panel_epochs',bias=bias,initial=initial)
    except ValueError as error:refusals['panel_epochs']=str(error)
    if include_ready:
        try:checked['ready_readout'],receipts['ready_readout']=plan_ready_readout(incumbent,adapter,problem,
            geometry,facts_path,bias=bias,initial=initial)
        except ReadoutUnavailable as error:refusals['ready_readout']=str(error)
    registry={}
    for mode,result in checked.items():
        def action(sess,result=result,mode=mode):
            result.revalidate()
            return {'schedule':result}
        registry[mode]=RegisteredPass(PassSpec(mode,0,produces=('schedule',),
            mutates_ir=False,ir_level='checked_dense'),action)
    facts=PolicyFacts(session.stamp(),shapes=((problem.m,problem.k),(problem.k,problem.n)),
        operations=frozenset({'dense'}),resources=(('array_edge',geometry[0]),
            ('output_accumulator_stores',output_stores(problem.m,problem.n,geometry)),
            *ready_resources(incumbent,problem,geometry).items(),
            ('epochs_admitted',int('panel_epochs' in checked)),('ready_admitted',int('ready_readout' in checked))))
    return session,registry,facts,refusals


def output_stores(m,n,geometry):
    """Ceiling physical tile rows over actual selected accumulator row capacity."""
    edge,_,_,_,rows=geometry
    tiles=((m+edge-1)//edge)*((n+edge-1)//edge)
    return (tiles*edge+rows-1)//rows


def capacity_class(design=None):
    return ShapeScheduleClass('output_exceeds_accumulator',
        (ResourcePredicate('output_accumulator_stores',at_least=2),),'compact',design)


def capacity_alternatives():
    # Ready is an independent explicit mechanism. It regresses against epochs
    # on the reserved large-K reuse geometry and is not enabled by this rule.
    return alternatives()[:2]


def assess_capacity(design,feedback):
    return assess_shape_policy((capacity_class(design),),capacity_alternatives(),
        tuple(feedback),fallback='compact',max_regression_ratio=1.0)


def ready_resources(incumbent,problem,geometry):
    """Full-K staging pressure of the existing M/N block in RTL store fractions.

    This is an empirical class feature. Storage legality and ordering remain
    separate checks in the actual shared ready-readout consumer.
    """
    edge,_,operand_rows,_,accumulator_rows=geometry
    block=incumbent.choice.block
    rows=(block.m+block.n)*problem.tiles[1]*edge
    return {'full_k_staging_quarter_stores':(4*rows+operand_rows-1)//operand_rows,
            'full_k_quarter_reduction_extent':(operand_rows//(4*(block.m+block.n)*edge))*edge,
            'full_k_staging_eighth_stores':(8*rows+operand_rows-1)//operand_rows,
            'full_k_eighth_reduction_extent':(operand_rows//(8*(block.m+block.n)*edge))*edge,
            'accumulator_block_halves':(2*block.m*block.n*edge)//accumulator_rows}


def ready_class(design=None,*,staging_divisor=4):
    if staging_divisor not in (4,8):raise ValueError('undeclared staging region')
    extent={4:'full_k_quarter_reduction_extent',8:'full_k_eighth_reduction_extent'}[staging_divisor]
    return ShapeScheduleClass('light_full_k_staging',
        (FeaturePredicate(TileFeature(DimensionRef(0,1),
            ResourceRef(extent)),at_most=(1,1)),),'panel_epochs',design)


def ready_alternatives():
    return (PipelineVariant('panel_epochs',('panel_epochs',)),
            PipelineVariant('ready_readout',('ready_readout',),
                (ResourcePredicate('ready_admitted',at_least=1),
                 ResourcePredicate('accumulator_block_halves',at_least=1),
                 ResourcePredicate('output_accumulator_stores',at_least=2))))


def assess_ready(design,feedback,*,staging_divisor=4):
    return assess_shape_policy((ready_class(design,staging_divisor=staging_divisor),),ready_alternatives(),tuple(feedback),
        fallback='panel_epochs',max_regression_ratio=1.0)


def ready_design_identity(*,staging_divisor=8):
    """Bind empirical feedback to the exact reusable region and admission guards."""
    return _digest({'shape_class':asdict(ready_class(staging_divisor=staging_divisor)),
                    'alternatives':[asdict(v) for v in ready_alternatives()]})


def ready_auto_schedule(incumbent,adapter,problem,geometry,facts_path,*,bias=False,initial=False):
    """Explicit candidate: assessed ready region, with the existing auto fallback.

    Missing/unsupported evidence retains the current auto result. This function
    does not alter the default auto policy or infer speed from storage fitting.
    """
    from pathlib import Path
    from merlin.transforms.pipeline_policy import PipelinePolicy,execute_policy
    evidence_path=Path(__file__).resolve().parent/'data/dense_ready_policy.json'
    feature=ready_resources(incumbent,problem,geometry)
    if (not evidence_path.is_file() or output_stores(problem.m,problem.n,geometry)<2 or
            feature['full_k_staging_eighth_stores']>1 or feature['accumulator_block_halves']<1):
        return auto_schedule(incumbent,adapter,problem,geometry,facts_path,bias=bias,initial=initial)
    evidence=json.loads(evidence_path.read_text())
    assessed=evidence['assessment']['classes'][0]
    if (evidence.get('design_sha256')!=ready_design_identity() or
            assessed['selected']!='ready_readout' or
            assessed['status']!='empirically_supported_declared_panel' or
            _digest(json.loads(Path(facts_path).read_text()))!=evidence['selected_facts_sha256']):
        return auto_schedule(incumbent,adapter,problem,geometry,facts_path,bias=bias,initial=initial)
    session,registry,facts,refusals=execution_owner(incumbent,adapter,problem,geometry,facts_path,
        bias=bias,initial=initial,include_ready=True)
    ready=ready_alternatives()[1]
    candidate=PipelineVariant('ready_readout',ready.passes,
        (*ready_class(staging_divisor=8).predicates,*ready.predicates))
    epoch=PipelineVariant('panel_epochs',('panel_epochs',),
        (ResourcePredicate('epochs_admitted',at_least=1),))
    paths=tuple(p for p in (candidate,epoch,alternatives()[0])
                if all(name in registry for name in p.passes))
    selected=execute_policy(session,PipelinePolicy(paths,'compact'),registry,facts,output='schedule')
    receipt=dict(selected.receipt,evidence_sha256=_digest(evidence),refusals=refusals,
        empirical_qualification='explicit candidate; provisional modeled panel; no physical qualification',
        implementation_custody='current checked shared passes and ordinary terminal encoder',
        original_assessment=evidence['assessment']['policy_sha256'])
    return selected.artifact,selected.receipt['selected_variant'],receipt


def auto_schedule(incumbent,adapter,problem,geometry,facts_path,*,bias=False,initial=False):
    """Execute a frozen empirical rule via existing registered shared passes.

    Empirical support belongs to the immutable original panel, not to arbitrary
    future compiler trees. This rule migration rechecks source legality;
    changing the default still requires affected ordinary execution checks.
    """
    from pathlib import Path
    from merlin.transforms.pipeline_policy import PipelinePolicy,execute_policy
    evidence_path=Path(__file__).resolve().parent/'data/dense_policy.json'
    evidence=json.loads(evidence_path.read_text())
    if evidence['assessment']['classes'][0]['selected']!='panel_epochs':
        return incumbent,'compact',{'policy':'unsupported empirical region; incumbent retained'}
    if _digest(json.loads(Path(facts_path).read_text()))!=evidence['selected_facts_sha256']:
        return incumbent,'compact',{'policy':'selected facts changed; incumbent retained'}
    stores=output_stores(problem.m,problem.n,geometry)
    if stores<2:
        return incumbent,'compact',{'policy':'output fits accumulator; incumbent retained',
            'output_accumulator_stores':stores,'evidence_sha256':_digest(evidence)}
    session,registry,facts,refusals=execution_owner(incumbent,adapter,problem,geometry,facts_path,
        bias=bias,initial=initial,include_ready=False)
    epoch=PipelineVariant('panel_epochs',('panel_epochs',),
        (*capacity_class().predicates,ResourcePredicate('epochs_admitted',at_least=1)))
    paths=(epoch,capacity_alternatives()[0]) if 'panel_epochs' in registry else (capacity_alternatives()[0],)
    selected=execute_policy(session,PipelinePolicy(paths,'compact'),registry,facts,output='schedule')
    receipt=dict(selected.receipt,evidence_sha256=_digest(evidence),refusals=refusals,
        empirical_qualification='provisional modeled panel; no physical timing qualification',
        implementation_custody='empirical rule migration; current checked pass and ordinary encoder',
        original_assessment=evidence['assessment']['policy_sha256'])
    return selected.artifact,selected.receipt['selected_variant'],receipt
