"""Explicit source grouping and selected-law adapter for shared immutable RHS.

These helpers do not install hooks, read tuning settings, capture a graph
function, or replace a scheduler. The graph owner supplies already recognized
source tasks, selected storage geometry, policy admission and ordinary emission
callbacks. Original parsing, ABI, task attribution and terminal remain owned by
the caller. Performance selection remains outside this module.
"""


def group_tasks(tasks, geometry, layouts, fused_readout, skipped, block, returned,
                source_identity, *, max_members, tile_edge, rhs_rows, zero_seed,
                enabled=True):
    """Group complete independent tasks with a capacity-fitting immutable RHS.

    ``tile_edge`` and ``rhs_rows`` describe the selected square primitive and
    assigned RHS store of the ordinary shared-RHS owner, after reservations.
    The caller applies tuning/override policy through ``enabled``. No source
    dimensions or model identities select a schedule here.
    """
    if not enabled:return tasks,{}
    if (type(max_members) is not int or not 2<=max_members<=8
            or type(tile_edge) is not int or tile_edge<=0
            or type(rhs_rows) is not int or rhs_rows<=0 or not callable(zero_seed)):
        raise ValueError('shared RHS groups need explicit bounded members, selected storage and zero-seed recognition')
    if (type(source_identity) is not str or len(source_identity)!=64
            or any(c not in '0123456789abcdef' for c in source_identity)):
        raise ValueError('shared RHS groups need the original source SHA256')
    # Keep the original whole-list ownership guard: partial grouping must not
    # conceal an operation assigned to more than one source task.
    owned_operations=[op for _,owned in tasks for op in owned]
    if len(set(owned_operations))!=len(owned_operations):return tasks,{}

    def eligible(task):
        kind,owned=task
        roots=[op for op in owned if geometry.get(op) is not None]
        if kind!='device' or len(roots)!=1:return None
        root=roots[0]
        if (root in layouts or root in fused_readout or not zero_seed(root.operands[-1])
                or any(op is not root and op not in skipped for op in owned)
                or any(value not in block.args for value in root.operands[:2])
                or root.results[0] not in returned.operands
                or str(root.results[0].type.element_type)!='i32'):
            return None
        return root

    def group_region(region,records):
        # Only these independent tasks may cross one another. Every read is
        # an immutable ABI input; every write is a distinct returned tensor.
        if len({root.results[0] for root,_ in records})!=len(records):return region,{}
        buckets={}
        for root,owned in records:
            buckets.setdefault((root.operands[1],geometry[root]),[]).append((root,owned))
        groups={};new_tasks=[]
        for bucket in buckets.values():
            for start in range(0,len(bucket),max_members):
                chunk=bucket[start:start+max_members]
                root=chunk[0][0];m,k,n=geometry[root]
                full_rhs_rows=((k+tile_edge-1)//tile_edge)*((n+tile_edge-1)//tile_edge)*tile_edge
                if len(chunk)<2 or full_rhs_rows>rhs_rows:
                    new_tasks.extend(('device',owned) for _,owned in chunk);continue
                groups[root]=dict(roots=[r for r,_ in chunk],source_identity=source_identity,
                    source_geometry=(m,k,n),source_independence='ABI-input reads; distinct returned fresh tensor writes')
                new_tasks.append(('device',[op for _,owned in chunk for op in owned]))
        return (new_tasks,groups) if groups else (region,{})

    # A host task or an ineligible device task is an ordering barrier. No
    # assumptions about its reads, writes or observers justify crossing it.
    new_tasks=[];groups={};region=[];records=[]
    for task in tasks:
        root=eligible(task)
        if root is not None:
            region.append(task);records.append((root,task[1]));continue
        grouped,admitted=group_region(region,records)
        new_tasks.extend(grouped);groups.update(admitted)
        new_tasks.append(task);region=[];records=[]
    grouped,admitted=group_region(region,records)
    new_tasks.extend(grouped);groups.update(admitted)
    return (new_tasks,groups) if groups else (tasks,{})


def emit_group(fb, group, ptrs, names, boundary_name, *, emit_resident_group,
               emit_dot, fence):
    """Use ordinary callbacks and return explicit task reads, writes, receipt.

    ``emit_resident_group`` must decline before builder mutation. Its optional
    scheduler callback can be bound by the caller with ``functools.partial``.
    A decline emits each original product through the ordinary fallback.
    """
    roots=group['roots'];m,k,n=group['source_geometry']
    receipt=emit_resident_group(fb,[ptrs[r.operands[0]] for r in roots],
        ptrs[roots[0].operands[1]],[ptrs[r.results[0]] for r in roots],m,k,n,
        attrs={'output_dtype':'i32'},source_identity=group['source_identity'])
    if receipt is None:
        for root in roots:
            emit_dot(fb,ptrs[root.operands[0]],ptrs[root.operands[1]],ptrs[root.results[0]],
                (m,k,n),None,output_dtype='i32')
            fence(fb)
        receipt={'fallback':'existing shared RHS owner refused before emission'}
    receipt={**receipt,'source_independence':group['source_independence']}
    reads=list(dict.fromkeys(boundary_name(value) for root in roots for value in root.operands[:2]))
    return reads,[names[root.results[0]] for root in roots],receipt


def schedule_shared_rhs(ordinary_scheduler, *args, facts_path, ordering_path=None,
                        logical_issue_bound=4096, record=None, **kwargs):
    """Inject selected retained-reduction/epoch laws into the ordinary owner.

    ``logical_issue_bound`` bounds finite compiler-issued work; it is NOT queue
    capacity. Pending compute-pair capacity is half the selected execute slots
    (PRELOAD + COMPUTE). Load and readout pending capacities are selected queue
    slots. The readout issue bound is the ordinary resource block's M*N bound.
    The library still checks actual storage, ranges, source order and effects.
    A checked-library refusal returns None before target emission.
    """
    from gemmini_dense_pipeline import ordering_premises
    from merlin.compile.scheduling.dense_panel_pipeline import DenseRetainedReductionAdmission
    from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw
    if type(logical_issue_bound) is not int or logical_issue_bound<=0:
        raise ValueError('shared RHS logical issue bound must be a positive compiler bound')
    if 'epoch_admission' in kwargs or 'epoch_law' in kwargs:
        raise ValueError('shared RHS selected-law injection cannot replace caller-supplied laws')
    problems,resources,_,adapter=args[:4]
    flags,pending,ordering=ordering_premises(facts_path,ordering_path)
    max_readouts=resources.max_m_blocks*resources.max_n_blocks
    identity='selected-command-ordering:'+ordering['ordering_sha256']
    admission=DenseRetainedReductionAdmission(adapter.configuration_identity,adapter.numerical_policy_identity,
        identity+':gemmini-encoder-v1',logical_issue_bound,logical_issue_bound,
        pending['execute']//2,pending['load'],max_issued_readouts=max_readouts,
        pending_readouts=pending['store'],readout_configuration='ordered_snapshot_before_store_issue')
    law=DensePanelEpochLaw(adapter.configuration_identity,adapter.numerical_policy_identity,identity,**flags)
    refusal=None
    try:result=ordinary_scheduler(*args,**kwargs,epoch_admission=admission,epoch_law=law)
    except ValueError as error:result=None;refusal=str(error)
    if record is not None:
        record(dict(source_identity=kwargs['source_identity'],members=len(problems),
            problem=dict(problems[0].__dict__),checked_kernel_sha256=None if result is None else result.kernel.digest(),
            ordering=ordering,library='schedule_dense_shared_rhs',epoch_law=dict(law.__dict__),
            logical_issue_bounds=dict(compute_pairs=logical_issue_bound,loads=logical_issue_bound,readouts=max_readouts),
            physical_queue_slots=dict(pending),pending_compute_pairs=pending['execute']//2,
            terminal='existing emit_resident_group -> dense_terminal',refusal=refusal,timing='UNKNOWN'))
    return result
