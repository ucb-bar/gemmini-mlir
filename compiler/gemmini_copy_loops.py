"""Bounded affine copy traversal over the ordinary exact host scalar lowering.

No hooks or policy changes live here. The caller supplies a checked column
bijection and disjoint storage; the same StoredTensor/host memory operations
retain element types and source values. Only the bounded copy traversal changes.
"""


def copy_affine_columns(host, source, destination, *, rows, plan, dtype,
                        factor=1, external_buffers_disjoint=False, max_regions=16,
                        short_inner=False, outer_lanes=False):
    """Emit one exact affine copy, strip-mined by a bounded constant factor.

    ``external_buffers_disjoint=True`` is the caller's proof obligation: source
    and destination must be distinct, nonoverlapping complete allocations for
    ``rows * len(plan.permutation)`` elements. Distinct pointer SSA values alone
    do not prove this. Refusals precede builder mutation. Factor one reproduces
    the incumbent StoredTensor copy traversal, including its loop structure.

    For each region, ``full = floor(inner/factor)*factor``. The main loop walks
    [0, full) by factor; its statically emitted lanes cover [j, j+factor).
    A scalar loop covers [full, inner). These are a disjoint exact partition of
    each original inner interval, including inner < factor and negative source
    strides. ColumnCopyPlan.verify proves every affine address equals the
    original bounded permutation. The row offset preserves those bounds.

    Opt-in ``short_inner`` uses the largest power of two no greater than both
    the requested factor and each region's inner extent. It changes only that
    region's traversal quantum; the same partition proof and scalar tail apply.
    The default retains the original traversal and receipt fields exactly.

    Opt-in ``outer_lanes`` fills the requested lane count from adjacent outer
    groups when short_inner is enabled and inner divides a larger factor. Each
    full chunk contains factor/inner groups with statically emitted inner
    coordinates. Remaining outer groups use the same exact scalar addresses.
    """
    if type(short_inner) is not bool:
        raise ValueError('affine copy short-inner policy must be an explicit boolean')
    if type(outer_lanes) is not bool:
        raise ValueError('affine copy outer-lanes policy must be an explicit boolean')
    from gemmini_column_copy import ColumnCopyPlan
    if type(plan) is not ColumnCopyPlan:
        raise ValueError('affine copy requires an original ColumnCopyPlan')
    plan.verify()
    if type(factor) is not int or factor not in (1,2,4,8):
        raise ValueError('affine copy factor must be one of 1, 2, 4, 8')
    if type(rows) is not int or not 0 < rows < 2**63:
        raise ValueError('affine copy requires positive bounded static rows')
    if dtype not in ('i8','i32'):
        raise ValueError('affine copy preserves only original i8 or i32 elements')
    if type(max_regions) is not int or not 0 < max_regions <= 16 or len(plan.regions)>max_regions:
        raise ValueError('affine copy requires at most 16 explicitly bounded regions')
    if type(external_buffers_disjoint) is not bool or not external_buffers_disjoint or source is destination:
        raise ValueError('affine copy requires caller-proven disjoint complete buffers')
    columns=len(plan.permutation)
    if rows*columns*{'i8':1,'i32':4}[dtype]>=2**63:
        raise ValueError('affine copy exceeds signed byte address bounds')
    from gemmini_host import StoredTensor
    fn=host.fb
    target=host._stored((rows,columns),dtype,destination)
    effective_factors=[]
    outer_grouping=[]
    for r in plan.regions:
        effective=factor
        if short_inner:
            effective=1 << (min(factor,r.inner).bit_length()-1)
        effective_factors.append(effective)
        if outer_lanes and short_inner and r.inner < factor and factor % r.inner == 0:
            qstep=factor//r.inner
            qfull=r.outer//qstep*qstep
            outer_grouping.append(dict(region=len(effective_factors)-1,
                inner=r.inner,outer=r.outer,outer_step=qstep,static_lanes=factor,
                full_groups=rows*(r.outer//qstep),tail_groups=rows*(r.outer-qfull),
                tail_elements=rows*(r.outer-qfull)*r.inner))
            def row_body(row,r=r,qstep=qstep,qfull=qfull):
                base=fn.mul_i(row,fn.const(columns))
                def group(q,groups):
                    source_base=fn.add_i(base,fn.add_i(fn.const(r.source),fn.mul_i(q,fn.const(r.outer_stride))))
                    destination_base=fn.add_i(base,fn.add_i(fn.const(r.first),fn.mul_i(q,fn.const(r.inner))))
                    for group_lane in range(groups):
                        for j in range(r.inner):
                            src=fn.add_i(source_base,fn.const(group_lane*r.outer_stride+j*r.inner_stride))
                            dst=fn.add_i(destination_base,fn.const(group_lane*r.inner+j))
                            host._write(target.pointer,dst,host._read(source,src,dtype),target.ety)
                fn.for_loop(0,qfull,qstep,lambda q:group(q,qstep),retain_loop=True)
                fn.for_loop(qfull,r.outer,1,lambda q:group(q,1),retain_loop=True)
            fn.for_loop(0,rows,1,row_body,retain_loop=True)
            continue
        def coordinates(indices,r=r):
            row,q,j=indices
            return row,fn.add_i(fn.const(r.first),fn.add_i(fn.mul_i(q,fn.const(r.inner)),j))
        def read(indices,r=r):
            row,q,j=indices
            col=fn.add_i(fn.const(r.source),fn.add_i(fn.mul_i(q,fn.const(r.outer_stride)),fn.mul_i(j,fn.const(r.inner_stride))))
            offset=fn.add_i(fn.mul_i(row,fn.const(columns)),col)
            return host._read(source,offset,dtype)
        logical=StoredTensor(host,(rows,columns),dtype,None,
            copy_traversal=((rows,r.outer,r.inner),coordinates,read))
        if effective==1:
            host._copy(logical,target)
            continue
        full=r.inner//effective*effective
        def outer(indices,r=r,full=full,effective=effective):
            row,q=indices
            # Hoist the common row/region bases outside the strip-mined loop.
            base=fn.mul_i(row,fn.const(columns))
            source_base=fn.add_i(base,fn.add_i(fn.const(r.source),fn.mul_i(q,fn.const(r.outer_stride))))
            destination_base=fn.add_i(base,fn.add_i(fn.const(r.first),fn.mul_i(q,fn.const(r.inner))))
            def element(j):
                src=fn.add_i(source_base,fn.mul_i(j,fn.const(r.inner_stride)))
                dst=fn.add_i(destination_base,j)
                host._write(target.pointer,dst,host._read(source,src,dtype),target.ety)
            def chunk(j):
                for lane in range(effective):
                    element(j if lane==0 else fn.add_i(j,fn.const(lane)))
            fn.for_loop(0,full,effective,chunk,retain_loop=True)
            fn.for_loop(full,r.inner,1,element,retain_loop=True)
        host._loops((rows,r.outer),outer)
    stats=dict(affine_regions=len(plan.regions),index_table_loads=0,factor=factor,
        copied_elements=rows*columns,
        full_chunks=sum(rows*r.outer*(r.inner//factor) for r in plan.regions),
        tail_elements=0 if factor==1 else sum(rows*r.outer*(r.inner%factor) for r in plan.regions),
        max_static_lanes=factor,disjoint_buffers='explicit caller obligation')
    if short_inner:
        stats.update(requested_factor=factor,effective_factors=effective_factors,
            full_chunks=sum(rows*r.outer*(r.inner//f) for r,f in zip(plan.regions,effective_factors)),
            tail_elements=sum(rows*r.outer*(r.inner%f) for r,f in zip(plan.regions,effective_factors)),
            max_static_lanes=max(effective_factors))
    if outer_grouping:
        selected={g['region'] for g in outer_grouping}
        stats.update(outer_grouping=outer_grouping,
            outer_full_groups=sum(g['full_groups'] for g in outer_grouping),
            outer_tail_groups=sum(g['tail_groups'] for g in outer_grouping),
            outer_tail_elements=sum(g['tail_elements'] for g in outer_grouping),
            full_chunks=sum(g['full_groups'] for g in outer_grouping)+sum(
                rows*r.outer*(r.inner//f) for i,(r,f) in enumerate(zip(plan.regions,effective_factors)) if i not in selected),
            tail_elements=sum(g['tail_elements'] for g in outer_grouping)+sum(
                rows*r.outer*(r.inner%f) for i,(r,f) in enumerate(zip(plan.regions,effective_factors)) if i not in selected),
            max_static_lanes=factor)
    return stats
