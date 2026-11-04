"""Source-required resident reuse, independent of optional timing preferences.

RES_PACK makes an explicit packed/stationary handle; ordinary MATMUL does not.
When one immutable handle lifetime has multiple consumers and the complete RHS
fits, preserving its live physical cells is a source lowering obligation. An
optional resident-off setting or faster completion schedule cannot remove that
obligation. The existing dense_residency owner proves cold/warm schedules and
invalidates them on EVICT, repack, or private scratchpad clobber.

No numerical attribute or M/K/N crossover distinguishes a repeated explicit
resident handle from this obligation. Thus this policy can also select the
slower synchronous retained route on larger K. Performance must be recovered by
a checked retained-epoch implementation, rather than by silently dropping reuse.
The predicate does not emit, establish alias facts, or suppress invalidations.
"""
from dense_residency import retention_legality, retention_admission


def source_resident_admission(problem, storage, *, resident_key, consumer_count,
                              explicit_res_pack, immutable_rhs,
                              lifetime_safe, external_buffers_disjoint,
                              requested_mode='off', rhs_transposed=False,
                              per_column=False, options=None):
    """Choose the existing actual owner after explicit source/lifetime proofs.

    ``consumer_count`` is the count in this RES_PACK-to-EVICT/repack lifetime,
    not a count across equal names or pointer values. ``lifetime_safe`` means
    the selected repeated-consumer interval has no unaccounted foreign write to
    its reserved RHS cells, and the backend maintains/invalidate-checks the live
    owner. Invalidation alone does not prove reuse across an intervening clobber;
    such intervals require a different checked placement or a narrower lifetime.
    ``external_buffers_disjoint`` excludes writes aliasing immutable RHS data.
    ``explicit_res_pack`` must come from a real source handle definition.

    Parent uses ``selected_mode`` as lower_dense's resident_reuse override and
    records this decision. A required-but-unavailable route is explicit via
    ``requires_refusal``; it must not be represented as satisfied resident reuse.
    Actual prepare/revalidate/commit and physical ownership remain mandatory.
    """
    if type(consumer_count) is not int or consumer_count < 0:
        raise ValueError('resident consumer_count must be a nonnegative lifetime-local integer')
    for label, value in [('explicit_res_pack', explicit_res_pack), ('immutable_rhs', immutable_rhs),
                         ('lifetime_safe', lifetime_safe), ('external_buffers_disjoint', external_buffers_disjoint)]:
        if type(value) is not bool:
            raise ValueError(label + ' must be an explicit source/ownership proof boolean')
    if requested_mode not in ('off', 'retained', 'retained_tiny'):
        raise ValueError('unknown optional resident mode')
    required = explicit_res_pack and resident_key is not None and consumer_count > 1
    source_refusals = []
    if not explicit_res_pack or resident_key is None:
        source_refusals.append('no explicit source RES_PACK handle')
    if not immutable_rhs: source_refusals.append('immutable original RHS not established')
    if not lifetime_safe: source_refusals.append('unclobbered resident interval and invalidation custody not established')
    if not external_buffers_disjoint: source_refusals.append('original RHS/output nonaliasing not established')
    legality = retention_legality(problem, storage, resident_key=resident_key,
        rhs_transposed=rhs_transposed, per_column=per_column)
    eligible = not source_refusals and legality['eligible']
    # tiny uses exactly the same checked owner and bounded literal axis emission;
    # it exposes physical cells to the trace while preserving source effect order.
    if required:
        selected = 'retained_tiny' if eligible else 'off'
        admitted = eligible
        optional = None
    else:
        optional = retention_admission(problem, storage, mode=requested_mode,
            resident_key=resident_key, rhs_transposed=rhs_transposed,
            per_column=per_column, options=options)
        admitted = eligible and optional['admitted']
        selected = requested_mode if admitted else 'off'
    reason = '; '.join(source_refusals) or legality['reason']
    if not admitted and reason is None and optional is not None:
        reason = optional['refusal']
    return {'schema': 'gemmini.source_resident_admission.v1',
        'required_by_source': required, 'consumer_count': consumer_count,
        'explicit_source_handle': explicit_res_pack and resident_key is not None,
        'requested_optional_mode': requested_mode, 'selected_mode': selected,
        'admitted': admitted, 'requires_refusal': required and not admitted,
        'reason': reason, 'legality': legality, 'optional_performance': optional,
        'source_basis': 'explicit packed/stationary handle reused within one immutable lifetime',
        'precedence': 'source-required physical reuse before optional completion/readout performance preferences',
        'emission_requirement': 'existing checked prepare/revalidate/commit; retain invalidations; no dummy loads or physical evidence probes',
        'performance': 'not a speed claim; required synchronous ownership may cost complete-call cycles'}


def resident_lifetime_use_counts(command_buffer):
    """Count actual MATMUL_RESIDENT consumers per explicit source pack lifetime.

    Returns records keyed by pack command index, avoiding a name-based count
    across EVICT/repack. This is source discovery only: pointer nonaliasing and
    private-memory custody remain separate required adapter facts.
    """
    records, active = {}, {}
    for index, command in enumerate(command_buffer.get('commands', ())):
        opcode = command['opcode']
        operands = command.get('operands') or {}
        if opcode == 'RES_PACK':
            handle = operands['dst']
            if handle in active:
                records[active[handle]]['end_command_index'] = index
            record = {'handle': handle, 'source': operands['src'], 'pack_command_index': index,
                      'consumer_count': 0, 'consumer_command_indices': [], 'end_command_index': None}
            records[index], active[handle] = record, index
        elif opcode == 'EVICT':
            packed = active.pop(operands.get('handle'), None)
            if packed is not None: records[packed]['end_command_index'] = index
        elif opcode == 'MATMUL_RESIDENT':
            packed = active.get(operands.get('rhs'))
            if packed is not None:
                records[packed]['consumer_count'] += 1
                records[packed]['consumer_command_indices'].append(index)
    return records


def resident_source_contracts(command_buffer, *, external_buffers_disjoint=False):
    """Return MATMUL-indexed source proofs for exact explicit handle lifetimes.

    Each record's ``kwargs`` is accepted by source_resident_admission together
    with the backend's actual resident_key, problem and storage. No physical fit
    is asserted here. Logical entry input/weight identity plus absence of writes
    proves RHS immutability; actual ABI nonaliasing is a separate caller fact.
    Same-lifetime resident matmuls and their commits are the only compute effects
    allowed between the first consumer and last committed consumer. Transports,
    foreign matmuls and every other accelerator operation conservatively refuse
    the physical interval. Administrative pack/evict commands do not compute.
    """
    if type(external_buffers_disjoint) is not bool:
        raise ValueError('external_buffers_disjoint must be an explicit ABI proof boolean')
    commands = command_buffer.get('commands', ())
    tensors = command_buffer.get('tensors', {})
    if not isinstance(tensors, dict):
        raise ValueError('resident source proof requires canonical named tensor declarations')
    transports = command_buffer.get('params', {}).get('source_transport_tasks', ())
    lifetimes = resident_lifetime_use_counts(command_buffer)
    # Resolve source handle aliases at their actual definition, never by final
    # spelling. A later repack cannot retroactively change an older lifetime.
    active, accumulator_owner, definitions, commits_by_pack = {}, {}, {}, {}
    for index, command in enumerate(commands):
        opcode, operands = command['opcode'], command.get('operands') or {}
        if opcode == 'RES_PACK':
            source = operands['src']
            prior = active.get(source)
            definitions[index] = definitions[prior] if prior is not None else source
            active[operands['dst']] = index
        elif opcode == 'EVICT':
            active.pop(operands.get('handle'), None)
        elif opcode in ('MATMUL_RESIDENT', 'MATMUL'):
            pack = active.get(operands.get('rhs')) if opcode == 'MATMUL_RESIDENT' else None
            accumulator_owner[operands.get('dst')] = pack
        elif opcode == 'COMMIT':
            pack = accumulator_owner.get(operands.get('src'))
            if pack is not None:
                commits_by_pack.setdefault(pack, []).append(index)
    result = {}
    for pack, lifetime in lifetimes.items():
        consumers = lifetime['consumer_command_indices']
        if not consumers:
            continue
        source = definitions[pack]
        declaration = tensors.get(source, {})
        entry_rhs = declaration.get('role') in ('input', 'weight') and declaration.get('dtype') == 'i8'
        mutations = []
        for index, command in enumerate(commands):
            if command['opcode'] in ('RES_PACK', 'EVICT'):
                continue
            if (command.get('operands') or {}).get('dst') == source:
                mutations.append({'command_index': index, 'opcode': command['opcode']})
        for index, record in enumerate(transports):
            writes = [copy.get('dst') for copy in record.get('copies', ())]
            if 'fill' in record: writes.append(record['fill'].get('dst'))
            if source in writes:
                mutations.append({'transport_record_index': index,
                    'before_command_index': record.get('before_command_index')})
        immutable = entry_rhs and not mutations
        # Our actual dense encoder executes products on COMMIT, so include
        # complete owned commits, not merely the source MATMUL markers.
        commits = commits_by_pack.get(pack, [])
        start, stop = min(consumers), max(consumers + commits)
        allowed = set(consumers + commits)
        barriers = []
        for index in range(start, stop + 1):
            opcode = commands[index]['opcode']
            if index not in allowed and opcode not in ('RES_PACK', 'EVICT'):
                barriers.append({'command_index': index, 'opcode': opcode,
                    'reason': 'foreign accelerator operation may overwrite retained private cells'})
        for index, record in enumerate(transports):
            position = record.get('before_command_index')
            if type(position) is not int or start <= position <= stop:
                barriers.append({'transport_record_index': index,
                    'before_command_index': position,
                    'reason': 'source transport may overwrite retained private cells'})
        # A commit outside its actual handle lifetime is not an admitted live
        # physical interval, even if a stale source accumulator name survives.
        end = lifetime['end_command_index']
        if end is not None and stop >= end:
            barriers.append({'command_index': end, 'opcode': commands[end]['opcode'],
                'reason': 'consumer completion crosses the explicit handle lifetime end'})
        safe = not barriers
        kwargs = dict(consumer_count=len(consumers), explicit_res_pack=True,
                      immutable_rhs=immutable, lifetime_safe=safe,
                      external_buffers_disjoint=external_buffers_disjoint)
        reasons = []
        if not entry_rhs: reasons.append('resolved RHS is not an original signed-i8 entry input/weight')
        if mutations: reasons.append('original RHS has a source write')
        if barriers: reasons.append('physical interval has an unproved foreign scratchpad writer or lifetime crossing')
        if not external_buffers_disjoint: reasons.append('actual ABI nonaliasing not supplied')
        for index in consumers:
            result[index] = {'schema': 'gemmini.resident_source_contract.v1',
                'matmul_command_index': index, 'pack_command_index': pack,
                'handle': lifetime['handle'], 'resolved_rhs': source,
                'consumer_command_indices': list(consumers), 'commit_command_indices': commits,
                'lifetime_end_command_index': end, 'physical_interval': [start, stop],
                'kwargs': dict(kwargs), 'required_by_source': len(consumers) > 1,
                'source_proven': not reasons, 'rhs_source_writes': mutations,
                'physical_barriers': barriers, 'refusals': reasons,
                'proof_scope': 'source identity and effects only; actual retention_legality and checked cold/warm emission remain required'}
    return result
