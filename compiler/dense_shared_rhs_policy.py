"""Opt-in coverage policy over the existing source-owned shared-RHS emitter.

Explicit off/drained/epochs retain their meaning. supported_geometry offers the
same checked epochs route for two to eight raw independent products whose M is
at most one selected array edge and K/N at most four edges. These are coverage
guards around the paired M16K64N16 and M16K16N64 source regions, not inferred
profitable crossovers. Tails, member-count neighbors and other K/N combinations
remain experimental pending complete paired holdouts. No default is changed.

ShapeScheduleClass/ShapePredicate describe source tensor extents. ResourcePredicate
checks necessary private-row fit separately. Neither grants legality: the actual
build_rhs_epochs kernel still proves immutable source membership, independence,
exact readout, selected ordering, complete physical placement and byte effects.
SelectedRHS.emit consumes that kernel through the existing common ISA encoder.
No profiling, case identity, source name or timing estimate participates in selection.
"""
from dataclasses import dataclass, asdict
import hashlib
import json

from merlin.transforms.pipeline_policy import PolicyFacts, ShapePredicate, ResourcePredicate
from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
from merlin.transforms.shape_policy import ShapeScheduleClass
from dense_rhs_epochs import RHSProduct, build_rhs_epochs, emit_rhs_epochs

KNOB_DECLARATION = {'dense_shared_rhs_schedule': {
    'values': ['off', 'drained', 'epochs', 'supported_geometry'],
    'mechanism': 'source-owned shared RHS with checked member epochs and conservative source geometry coverage',
    'note': 'off/drained/epochs unchanged; supported_geometry is an opt-in M<=edge,K/N<=4edge raw independent group experiment, still subject to all original scheduling refusals. Default remains off; complete paired holdouts required.'}}

EVIDENCE = {
    'status': 'two complete modeled training regions; no held-out promotion',
    'artifact': 'artifacts/experiments/rhs_epochs023/evidence.json',
    'baseline_request': '6_1793277750385102498_38283fa978b14ac4d3177b48ffbb99db',
    'candidate_request': '8_1793277750385629613_46b766440b4b88612f42d9847f2411f8',
    'baseline_source_sha256': 'cb455020226b773ba8b15a871752bf58a4c2a6aca5630b9189952c47ca057420',
    'candidate_source_sha256': 'cea7bf67e564c598c15b64cd6d3dd3cd58587877aaae657a9338dce40133d2c0',
    'observations': [
        {'m':16,'k':64,'n':16,'members':8,'before_complete_cycles':22895,'after_complete_cycles':21077},
        {'m':16,'k':16,'n':64,'members':8,'before_complete_cycles':48558,'after_complete_cycles':32077}],
    'qualification': 'matched complete Spike model with numeric pass; no RTL timing claim; guards bound coverage rather than learned crossover'}


def _facts(products, storage, source_identity):
    edge = storage.lhs.unit_bytes
    if type(edge) is not int or edge <= 0:
        raise ValueError('shared RHS policy requires selected positive array edge')
    shapes = tuple((p.m, p.k) for p in products) + ((products[0].k, products[0].n),)
    required_rhs_rows = ((products[0].k+edge-1)//edge)*((products[0].n+edge-1)//edge)*edge
    required_lhs_rows = ((products[0].k+edge-1)//edge)*edge
    resources = (('rhs_private_rows', storage.rhs.capacity),
                 ('lhs_private_rows', storage.lhs.capacity),
                 ('acc_private_rows', storage.accumulator.capacity))
    resource_identity = hashlib.sha256(json.dumps(resources).encode()).hexdigest()
    facts = PolicyFacts(AnalysisStamp(source_identity, PipelineContext(resources=resource_identity),
        'source-shared-rhs-geometry'), shapes=shapes, resources=resources)
    return edge, facts, required_rhs_rows, required_lhs_rows


def _classes(products, edge, rhs_rows, lhs_rows):
    rhs_operand = len(products)
    coverage = tuple(ShapePredicate(i, 0, at_least=1, at_most=edge) for i in range(len(products)))
    coverage += (ShapePredicate(rhs_operand, 0, at_least=1, at_most=4*edge),
                 ResourcePredicate('rhs_private_rows',rhs_rows),
                 ResourcePredicate('lhs_private_rows',lhs_rows),
                 ResourcePredicate('acc_private_rows',edge))
    return (
        ShapeScheduleClass('shared_rhs_one_output_tile', coverage + (
            ShapePredicate(rhs_operand,1,at_least=1,at_most=edge),), 'off'),
        ShapeScheduleClass('shared_rhs_multiple_output_tiles', coverage + (
            ShapePredicate(rhs_operand,1,at_least=edge+1,at_most=4*edge),), 'off'))


def select_rhs_schedule(mode, products, storage, *, source_identity):
    """Return a route preference; actual scheduling remains independently checked."""
    if mode not in ('off','drained','epochs','supported_geometry'):
        raise ValueError('unknown shared RHS policy mode')
    products = tuple(products)
    if not products or any(type(p) is not RHSProduct for p in products):
        raise ValueError('shared RHS policy requires typed actual source products')
    if any(type(v) is not int or v<=0 for p in products for v in (p.m,p.k,p.n)):
        raise ValueError('shared RHS policy requires positive static source extents')
    if (not isinstance(source_identity,str) or len(source_identity)!=64
            or any(c not in '0123456789abcdef' for c in source_identity)):
        raise ValueError('shared RHS policy requires current source digest')
    edge, facts, rhs_rows, lhs_rows = _facts(products,storage,source_identity)
    classes = _classes(products,edge,rhs_rows,lhs_rows) if 2<=len(products)<=8 else ()
    matches = [(c,tuple(p.match(facts) for p in c.predicates)) for c in classes]
    matched = next((c.name for c, values in matches if all(v is True for v in values)),None)
    raw = all(p.attrs.get('output_dtype','i32')=='i32' and not p.attrs.get('epilogue',[])
              and p.bias is None and not p.rhs_transposed and p.zero_seed
              and p.attrs.get('acc_scale_by_column') is None for p in products)
    same_rhs = all((p.rhs,p.k,p.n)==(products[0].rhs,products[0].k,products[0].n) for p in products)
    eligible = matched is not None and raw and same_rhs
    selected = ('epochs' if eligible else 'off') if mode=='supported_geometry' else mode
    return {'schema':'gemmini.shared_rhs_geometry_policy.v1','requested':mode,
        'selected':selected,'geometry_class':matched,'coverage_candidate':eligible,
        'array_edge':edge,'member_shapes':[[p.m,p.k,p.n] for p in products],
        'members':len(products),'source_identity':source_identity,
        'necessary_private_rows':{'rhs':rhs_rows,'lhs':lhs_rows,'accumulator':edge},
        'class_checks':[{'class':c.name,'incumbent':c.incumbent,
            'predicates':[asdict(p) for p in c.predicates],'matches':list(values)} for c,values in matches],
        'source_numerical_guard':raw,'same_original_rhs_and_reduction_extent':same_rhs,
        'status':'explicit experimental coverage policy; paired heldouts pending',
        'legality':'actual original source, alias, dependency, layout/readout, complete capacity and selected ordering checks remain authoritative',
        'refusal':None if selected!='off' or mode=='off' else 'outside conservative coverage, raw source grammar, or necessary resource fit'}


@dataclass
class SelectedRHS:
    plan: object
    policy: dict

    @property
    def products(self): return self.plan.products

    @property
    def kernel(self): return self.plan.kernel

    def revalidate(self):
        self.plan.revalidate()
        return self

    def emit(self,builder,pointers,*,index_unroll='off'):
        self.revalidate()
        receipt=emit_rhs_epochs(builder,self.plan,pointers,index_unroll=index_unroll)
        result={'policy':{**self.policy,'actual_schedule_applied':True},'consumed_route':receipt,
                'checked_kernel_sha256':self.plan.kernel.digest()}
        if not hasattr(builder,'dense_shared_rhs_policy_receipts'):builder.dense_shared_rhs_policy_receipts=[]
        builder.dense_shared_rhs_policy_receipts.append(result)
        return result


def build_selected_rhs(mode, products, resources, storage, adapter, *, source_identity,
                       external_buffers_disjoint=False, **schedule_kwargs):
    """Return (actual owner or None, decision); call owner.emit to consume route.

    Call after actual source-task coalescing. A pre-coalescing policy decision
    can select candidate geometry, but never authorizes source task relabeling.
    Explicit modes bypass coverage preference only, never scheduler legality.
    """
    decision=select_rhs_schedule(mode,products,storage,source_identity=source_identity)
    if decision['selected']=='off':return None,decision
    try:
        plan=build_rhs_epochs(tuple(products),resources,storage,adapter,
            source_identity=source_identity,external_buffers_disjoint=external_buffers_disjoint,
            mode=decision['selected'],**schedule_kwargs)
    except ValueError as error:
        return None,{**decision,'selected':'off','schedule_refusal':str(error),
                     'actual_schedule_applied':False}
    decision={**decision,'checked_schedule_built':True,'actual_schedule_applied':False,
              'checked_kernel_sha256':plan.kernel.digest()}
    return SelectedRHS(plan,decision),decision
