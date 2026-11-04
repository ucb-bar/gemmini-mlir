"""Hardware-relative opt-in region over the existing checked segmented owner."""
from dataclasses import replace,asdict
from pathlib import Path
from hashlib import sha256
from math import isqrt


def selection(m,k,n,*,bias=False):
    from dense import _capacities
    from merlin.transforms.pipeline_policy import PolicyFacts, ResourcePredicate
    from merlin.transforms.pipeline_state import AnalysisStamp, PipelineContext
    if any(type(x)is not int or x<=0 for x in (m,k,n)):
        raise ValueError('frontier region requires positive static dimensions')
    kr,edge,_,orows,_,_,arows,_=_capacities()
    mt=(m+edge-1)//edge;kt=(k+kr-1)//kr;nt=(n+edge-1)//edge
    balanced=isqrt(arows//edge);lhs=orows//2;rhs=orows-lhs
    values=dict(row_tiles=mt,reduction_tiles=kt,column_tiles=nt,array_edge=edge,
        balanced_panel_tiles=balanced,lhs_window_rows=lhs,rhs_window_rows=rhs,
        minimum_retained_lhs_rows=2*kt*edge,
        balanced_full_k_rhs_rows=kt*balanced*kr,complete_rhs_rows=kt*nt*kr,
        multirow=int(mt>=2),two_full_k_lhs_rows_fit=int(2*kt*edge<=lhs),
        full_balanced_rhs_panel_fits_one_slot=int(2*kt*balanced*kr<=rhs),
        repeated_n=int(nt>balanced),rhs_does_not_fit=int(kt*nt*kr>rhs),zero_seed=int(not bias))
    guards=tuple(ResourcePredicate(key,at_least=1)for key in
        ('multirow','two_full_k_lhs_rows_fit','full_balanced_rhs_panel_fits_one_slot','repeated_n','rhs_does_not_fit','zero_seed'))
    identity=sha256(repr((m,k,n,bias,values)).encode()).hexdigest()
    facts=PolicyFacts(AnalysisStamp(identity,PipelineContext(
        hardware=sha256((Path(__file__).parent/'data/facts.json').read_bytes()).hexdigest(),resources=identity),'dense-multirow-geometry'),resources=tuple(sorted(values.items())))
    return dict(schema='authored.frontier_multirow_region.v4',selected=all(g.match(facts)is True for g in guards),resources=values,
        predicates=[dict(resource=g.resource,at_least=g.at_least)for g in guards],
        profitability='prefrozen coarse hypothesis; full balanced RHS K panel fits one slot; multiple M tiles amortize narrow K segments; independent validation required',
        legality='actual shared dense_segmented_blocks and checked source/physical schedule admission remain authoritative',fallback='unchanged regional_ready_transfers')


def readout_proof(dtype,n,k):
    """Bind existing source exit to selected target readout, no new hardware law.

    The schedule proves exact i32 producer and logical output rectangles. The
    unchanged terminal Encoder applies this original pure per-cell exit once
    at each final store; all columns use the same original coefficient.
    """
    from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout
    from merlin.llvmlower.integer_producer_range import IntegerSumProductsRange
    from merlin.llvmlower.integer_readout import derive,evaluate
    from merlin.targetgen import output_path as OP
    import json
    if dtype=='i32':return dict(dtype='i32',readout='raw original accumulator exit')
    if dtype!='i8':raise ValueError('only original raw or identity narrowing exit')
    target=Path(__file__).parent/'data/frontier_output_path.json'
    selected=json.loads(target.read_text())
    raw=selected['output_path']
    path=OP.validate(raw)
    # Explicit original selected sequence: activation disabled, f32 conversion,
    # f32 scale and nearest-even signed-i8 saturation. Source scale stays 1.0.
    stages=raw['readout']
    convert=next(x for x in stages if x['kind']=='convert')
    scale=next(x for x in stages if x['kind']=='scale')
    quant=next(x for x in stages if x['kind']=='requant')
    if convert!={'kind':'convert','rounding':'half_even','to':'f32'} or scale['dtype']!='f32' or scale['rounding']!='half_even' or quant['to']!='i8' or quant['rounding']!='half_even' or quant['clamp']!=[-128,127]:
        raise ValueError('selected narrowing path does not establish identity i8 source exit')
    _,edge,*_=_capacities_for_readout()
    schema=DenseTiledIntegerReadout.from_source_scales(1.0,columns=n,tile_columns=edge,relu=False)
    bounds=IntegerSumProductsRange(k,-128,127,-128,127).interval()
    if max(abs(x)for x in bounds)>2**24:
        raise ValueError('identity narrowing admission requires exact i32 to binary32 producer interval')
    proof=derive([1.0],*bounds)
    for value in (bounds[0],-129,-128,-127,-1,0,1,126,127,128,bounds[1]):
        if evaluate(value,proof)!=max(-128,min(127,value)):
            raise ValueError('source identity narrowing proof differs from selected exit')
    return dict(dtype=dtype,source_readout=asdict(schema),producer_interval=bounds,
        selected_output_path_sha256=sha256(target.read_bytes()).hexdigest(),
        integer_readout_proof_sha256=sha256(repr(proof).encode()).hexdigest(),
        proof='exact i32 producer range lies within binary32 integer domain; original scale1 and nearest-even saturation; original Encoder per-cell store',
        encoder_sha256=sha256((Path(__file__).parent/'encoding.py').read_bytes()).hexdigest())


def _capacities_for_readout():
    from dense import _capacities
    return _capacities()


def plan(m,k,n,*,bias=False,panel_shape='balanced'):
    from dense import _plan_dense
    region=selection(m,k,n,bias=bias)
    if not region['selected']:
        return _plan_dense(m,k,n,bias=bias,alternative='regional_ready_transfers',panel_shape=panel_shape)
    p,original=_plan_dense(m,k,n,bias=bias,alternative='segmented_frontier_dma',panel_shape='rectangular')
    original.revalidate();schedule=original.realization['schedule_ir']
    if schedule.get('pipeline_kind')!='retained_lhs_segmented_k':
        raise ValueError('region did not consume original segmented schedule owner')
    record=dict(region,actual_kernel_sha256=original.kernel.digest(),actual_schedule_sha256=sha256(schedule['text'].encode()).hexdigest(),
        owner_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),shared_route='segmented_frontier_dma',panel_geometry='rectangular',
        pass_order=['dense_segmented_blocks','make_dense_schedule_plan','schedule_dense_pipeline:retained_lhs_segmented_k:ready','coalesce_dense_epochs:all_bands'])
    result=replace(original,realization=dict(original.realization,frontier_multirow=record));result.revalidate()
    return p,result
