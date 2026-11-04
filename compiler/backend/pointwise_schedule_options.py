"""Optional checked RHS panels for original scanline-qualified pointwise rows."""
from dataclasses import replace


def rhs_panels(problem,resources,storage,adapter,*,source_identity,qualification_identity,bias_source,n_tiles):
    from merlin.compile.scheduling.dense_panel_pipeline import (
        DenseRetainedReductionAdmission,dense_m_panel_capacity,schedule_dense_m_panels)
    # Four is a compiler expansion budget, not a storage/hardware fact. The
    # shared capacity/choice functions independently admit the actual M extent.
    bounded=replace(resources,max_m_blocks=4)
    capacity=dense_m_panel_capacity(problem,bounded,storage,n_tiles=n_tiles)
    maximum=capacity['maximum_m_tiles']
    if maximum<1:return None,dict(selection='rhs_panels',applied=False,capacity=capacity)
    # Same selected retained-reduction law/pending capacities as the incumbent.
    # Issued work is the authored full K*N*M envelope, independently of pending
    # queue capacity and its existing qualified ready/valid backpressure.
    law=DenseRetainedReductionAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        qualification_identity,max(1,maximum*problem.tiles[1]*n_tiles),
        max(1,(maximum+n_tiles)*problem.tiles[1]),2,8)
    attempt=schedule_dense_m_panels(problem,bounded,storage,adapter,law,n_tiles=n_tiles,
        name='pointwise_stride2_rhs_panels',dtypes=('i8','i8','i32'),source_identity=source_identity,
        external_buffers_disjoint=True,static_budget=128,timeout_ms=20000,bias_source=bias_source)
    report=dict(attempt.feedback,selection='rhs_panels',applied=attempt.result is not None,
        authored_max_compute_pairs=law.max_issued_compute_pairs,
        authored_max_loads=law.max_issued_loads,pending_compute_pairs=2,pending_loads=8,
        law='same selected retained-reduction admission as incumbent; no new target premise',
        storage='shared dense_m_panel_capacity and actual schedule storage/epoch verification')
    return attempt.result,report


def selected_capacity_n_limit():
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    value=knob('POINTWISE_STRIDED_CAPACITY_N_LIMIT',default='auto',
        values=('auto','8','16'),root=Path(__file__).resolve().parents[1])
    if value not in ('auto','8','16'):
        raise ValueError('POINTWISE_STRIDED_CAPACITY_N_LIMIT must be auto, 8 or 16')
    return value


def rhs_capacity(problem,resources,storage,adapter,*,source_identity,qualification_identity,bias_source,n_tiles,n_limit=None):
    """Bounded shared construction search over M/N and disjoint private rows.

    This ranks requested transfers, not time. Every candidate uses the original
    source, total assigned interval, accumulator view and selected target law.
    Only the final shared checked schedule reaches the terminal consumer.
    """
    from merlin.compile.scheduling.dense_panel_pipeline import (
        DenseRetainedReductionAdmission,dense_retained_choices,
        dense_m_panel_capacity,schedule_dense_m_panels)
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage
    n_limit=selected_capacity_n_limit() if n_limit is None else n_limit
    if n_limit not in ('auto','8','16'):
        raise ValueError('POINTWISE_STRIDED_CAPACITY_N_LIMIT must be auto, 8 or 16')
    left,right=storage.lhs,storage.rhs
    # Repartition only the original contiguous, identically addressed backing.
    # A gap, distinct backing, or different row units needs another allocator.
    if (left.backing!=right.backing or left.unit_bytes!=right.unit_bytes
            or left.base+left.capacity!=right.base):
        return None,dict(selection='rhs_capacity',applied=False,
            refusal='original_contiguous_equal_row_operand_assignment')
    end=right.base+right.capacity
    accumulator=storage.accumulator
    if accumulator is None:
        return None,dict(selection='rhs_capacity',applied=False,refusal='original_accumulator_assignment')
    # Eight M tiles is a bounded code-construction budget, not a capacity fact.
    # N is separately bounded by the original ACC rows and shared candidate cap.
    m_budget=min(8,problem.tiles[0],resources.candidate_budget)
    n_bound=min(problem.tiles[2],accumulator.capacity//problem.tile_m,
                resources.candidate_budget//max(1,m_budget))
    if n_limit!='auto':n_bound=min(n_bound,int(n_limit))
    if n_bound<1:
        return None,dict(selection='rhs_capacity',applied=False,refusal='original_ACC_or_candidate_bound')
    candidates=[]
    for reserve_m in range(1,m_budget+1):
        rows=problem.tiles[1]*problem.tile_m*reserve_m
        rhs_base=left.base+rows
        rhs_base+=(-rhs_base)%right.alignment
        if rhs_base>=end:continue
        proposed=DenseLoopStorage(replace(left,capacity=rhs_base-left.base),
            replace(right,base=rhs_base,capacity=end-rhs_base),accumulator)
        bounded=replace(resources,max_m_blocks=reserve_m,max_n_blocks=max(1,n_bound))
        choices=dense_retained_choices(problem,bounded,proposed,roles=('weight',))
        for choice in choices:
            block=choice.block
            # Same transfer objective/order as the shared retained catalogue.
            key=(choice.requested_operand_bytes,choice.lhs_loads+choice.rhs_loads,
                 -block.m*block.n,block.m,block.n,reserve_m)
            candidates.append((key,choice,bounded,proposed))
    if not candidates:
        return None,dict(selection='rhs_capacity',applied=False,
            refusal='no_shared_complete_K_choice_in_original_storage',
            m_expansion_budget=m_budget,n_expansion_bound=n_bound)
    _,choice,bounded,proposed=min(candidates,key=lambda item:item[0])
    block=choice.block
    capacity=dense_m_panel_capacity(problem,bounded,proposed,n_tiles=block.n)
    law=DenseRetainedReductionAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        qualification_identity,max(1,block.m*problem.tiles[1]*block.n),
        max(1,(block.m+block.n)*problem.tiles[1]),2,8)
    attempt=schedule_dense_m_panels(problem,bounded,proposed,adapter,law,n_tiles=block.n,m_tiles=block.m,
        name='pointwise_stride2_rhs_capacity',dtypes=('i8','i8','i32'),source_identity=source_identity,
        external_buffers_disjoint=True,static_budget=128,timeout_ms=20000,bias_source=bias_source)
    report=dict(attempt.feedback,selection='rhs_capacity',applied=attempt.result is not None,
        construction_objective='shared requested_operand_bytes, load calls, negative M*N; timing UNKNOWN',
        candidates_considered=len(candidates),m_expansion_budget=m_budget,n_expansion_bound=n_bound,
        selected_n_limit=n_limit,selected_n_tiles=block.n,authored_max_compute_pairs=law.max_issued_compute_pairs,
        authored_max_loads=law.max_issued_loads,pending_compute_pairs=2,pending_loads=8,
        original_operand_interval=[left.base,end],
        private_rows=dict(lhs=[proposed.lhs.base,proposed.lhs.base+proposed.lhs.capacity],
                          rhs=[proposed.rhs.base,proposed.rhs.base+proposed.rhs.capacity],
                          acc=[accumulator.base,accumulator.base+accumulator.capacity]),
        selected_capacity=capacity,
        law='same selected retained-reduction admission as incumbent; no new target premise',
        storage='shared dense_retained_choices, dense_m_panel_capacity and actual source/storage/epoch verification')
    return attempt.result,report


class PreparedStridedInvocation:
    """One context's checked product and exact original binding, never global."""
    def __init__(self,key,structural_key,pointers,geometry,product):
        self.key,self.structural_key,self.pointers=key,structural_key,pointers
        self.geometry,self.product=geometry,product


def prepare_invocation(ctx,ifm,rhs,dst,attrs,geometry,route,construct):
    """Prepare without commands/LLVM/allocation; reuse only exact live bindings."""
    import hashlib,json
    from pathlib import Path
    window,n,tile_m,zero_seed=geometry
    names=[ifm,rhs,dst]+[attrs[k]for k in ('bias','accumulator_init')if attrs.get(k)]
    pointers=tuple(ctx.buffer(name)for name in names)
    ordering=Path(__file__).with_name('dense_ordering.json').read_bytes()
    mixed=attrs.get('acc_scale_by_column') is not None and len(set(map(float,attrs['acc_scale_by_column'])))>1
    raw_reload=mixed and route=='raw_reload' and not attrs.get('strided_pointwise_raw_producer',False)
    n_limit=selected_capacity_n_limit()
    readout_binding={}
    if raw_reload:
        from merlin.compile.tuning_knobs import knob
        selection=attrs.get('column_readout_batching',knob('COLUMN_READOUT_BATCHING',
            default='off',values=('off','on','palette'),root=Path(__file__).resolve().parents[1]))
        if selection not in ('off','on','palette'):
            raise ValueError('COLUMN_READOUT_BATCHING must be off, on or palette')
        readout_binding['batching']=selection
        if selection!='off':
            from .column_batches import _admission
            readout_binding['selected_admission']=_admission()
    bound=dict(names=names,views=[ctx.spec(name)for name in names],attributes=attrs,
        producer='rhs_capacity',capacity_n_limit=n_limit,column_readout=route,readout=readout_binding,
        selected_ordering_sha256=hashlib.sha256(ordering).hexdigest(),
        source_index=getattr(ctx,'current_source_index',None))
    structural=hashlib.sha256(json.dumps(bound,sort_keys=True).encode()).hexdigest()
    key=(structural,tuple(map(id,pointers)))
    previous=getattr(ctx,'_strided_pointwise_prepared',None)
    if previous is not None and previous.key==key and all(a is b for a,b in zip(previous.pointers,pointers)):
        return previous
    ctx._strided_pointwise_prepared=None
    product=construct(window,n,tile_m,bool(attrs.get('bias')or attrs.get('accumulator_init')or zero_seed),
        False if raw_reload else mixed,producer='rhs_capacity',capacity_n_limit=n_limit)
    schedule=product[0]
    if schedule is None:return None
    if not schedule.realization.get('strided_producer',{}).get('applied'):
        raise ValueError('prepared capacity plan did not consume its requested producer')
    prepared=PreparedStridedInvocation(key,structural,pointers,geometry,product)
    ctx._strided_pointwise_prepared=prepared
    return prepared
