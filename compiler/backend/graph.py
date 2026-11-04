"""Conservative source-task rewrites for the common command graph."""
from __future__ import annotations
import copy


def _candidate(cb, task):
    if task.get('kind') != 'device':
        return None
    commands = cb['commands'][task['instruction_start']:task['instruction_end']]
    if [c['opcode'] for c in commands] != ['RES_PACK', 'MATMUL_RESIDENT', 'COMMIT', 'EVICT']:
        return None
    pack, matmul, commit, evict = commands
    p, mm, out, end = [c.get('operands', {}) for c in commands]
    a = {**matmul.get('attributes', {}), **commit.get('attributes', {})}
    if (set(p) != {'src', 'dst'} or set(mm) != {'lhs', 'rhs', 'dst'}
            or set(out) != {'src', 'dst'} or set(end) != {'handle'}
            or p.get('dst') != mm.get('rhs') or mm.get('dst') != out.get('src')
            or end.get('handle') != p.get('dst') or a.get('output_dtype') != 'i32'
            or a.get('epilogue', []) or a.get('bias') is not None
            or a.get('acc_scale', 1.0) != 1.0 or a.get('acc_scale_by_column') is not None
            or a.get('relu', False)):
        return None
    lhs, rhs, dst = mm.get('lhs'), p.get('src'), out.get('dst')
    if len({lhs, rhs, dst}) != 3 or None in (lhs, rhs, dst):
        return None
    try:
        x, w, y = [cb['tensors'][n] for n in (lhs, rhs, dst)]
        m, k = x['shape']; kb, n = w['shape']
    except (KeyError, TypeError, ValueError):
        return None
    if ([x['dtype'], w['dtype'], y['dtype']] != ['i8', 'i8', 'i32']
            or k != kb or y['shape'] != [m, n] or min(m, k, n) <= 0
            or set(task.get('reads', [])) != {lhs, rhs} or task.get('writes') != [dst]):
        return None
    boundary = a.get('source_numeric_boundary', {})
    if boundary.get('destination_constants') != [0]:
        return None
    if [v.get('dtype') for v in boundary.get('inputs', [])] != ['i8', 'i8']:
        return None
    return {'lhs': lhs, 'rhs': rhs, 'dst': dst, 'm': m, 'k': k, 'n': n,
            'attributes': a}


def fuse_sibling_matmuls(command_buffer, *, enabled=None):
    """Merge adjacent independent exact signed-i8 source matmul tasks.

    Unrecognized command groups and graphs without a source plan retain the
    incumbent route. Source bindings and all materialized output tensors stay
    unchanged; only owned source groups and their command ranges are rewritten.
    """
    if enabled is None:
        from pathlib import Path
        from merlin.compile.tuning_knobs import knob
        enabled = knob('sibling_fusion', default=False, values=(False, True, 'auto', 'rhs', 'rhs_auto', 'rhs_rows'),
                       root=Path(__file__).resolve().parents[1])
    if enabled in ('rhs', 'rhs_auto', 'rhs_rows'):
        return fuse_shared_rhs_matmuls(command_buffer, automatic=enabled == 'rhs_auto')
    if not enabled:
        return command_buffer
    plan = command_buffer.get('params', {}).get('global_program_plan')
    if not isinstance(plan, dict) or not plan.get('tasks'):
        return command_buffer
    tasks = plan['tasks']
    # This rewrite currently owns neither wrapper effects nor gaps in ranges.
    if (plan.get('prologue_instruction_range') != [0, 0]
            or plan.get('epilogue_instruction_range') != [len(command_buffer['commands'])] * 2
            or tasks[0]['instruction_start'] != 0
            or tasks[-1]['instruction_end'] != len(command_buffer['commands'])
            or any(a['instruction_end'] != b['instruction_start'] for a, b in zip(tasks, tasks[1:]))):
        return command_buffer
    cb = copy.deepcopy(command_buffer)
    commands, rewritten, evidence = [], [], []
    index = 0
    while index < len(tasks):
        first = _candidate(command_buffer, tasks[index])
        members = [first] if first else []
        group = [tasks[index]]
        if first:
            for next_index in range(index + 1, min(len(tasks), index + 4)):
                member = _candidate(command_buffer, tasks[next_index])
                if member is None or (member['lhs'], member['m'], member['k']) != (first['lhs'], first['m'], first['k']):
                    break
                new_reads = {member['lhs'], member['rhs']}
                old_reads = {n for old in members for n in (old['lhs'], old['rhs'])}
                old_writes = {old['dst'] for old in members}
                if (member['rhs'] in {old['rhs'] for old in members}
                        or member['dst'] in old_writes | old_reads
                        or new_reads & old_writes):
                    break
                members.append(member); group.append(tasks[next_index])
        if enabled == 'auto':
            from .sibling_policy import fusion_profitable
            if not fusion_profitable(members):
                members = [first] if first else []
                group = [tasks[index]]
        tid, start = len(rewritten), len(commands)
        if len(members) > 1:
            ownership = sorted({s for task in group for s in task['source_op_indices']})
            regions = sorted({r for task in group for r in task.get('source_region_ids', [])})
            operands = {'lhs': first['lhs']}
            for i, member in enumerate(members):
                member['attributes']['task_index'] = tid
                operands[f'rhs{i}'], operands[f'dst{i}'] = member['rhs'], member['dst']
            commands.append({'opcode': 'VECTOR_MAP', 'operands': operands,
                             'attributes': {'op': 'sibling_matmul', 'task_index': tid,
                                            'source_op_index': members[0]['attributes']['source_op_index'],
                                            'members': members, 'output_dtype': 'i32',
                                            'fusion_proof': 'adjacent independent exact signed-i8 zero-i32 contractions; shared lhs SSA storage'}})
            task = {'task_index': tid, 'kind': 'device', 'source_op_indices': ownership,
                    'source_region_ids': regions, 'instruction_start': start, 'instruction_end': len(commands),
                    'reads': list(dict.fromkeys(n for m in members for n in (m['lhs'], m['rhs']))),
                    'writes': [m['dst'] for m in members]}
            evidence.append({'task_index': tid, 'source_op_indices': ownership,
                             'source_task_indices': [t['task_index'] for t in group],
                             'shared_lhs': first['lhs'], 'outputs': task['writes'],
                             'transformation': 'shared_lhs_sibling_matmul_v1',
                             'source_effects_preserved': True})
        else:
            task = copy.deepcopy(tasks[index])
            for command in copy.deepcopy(command_buffer['commands'][task['instruction_start']:task['instruction_end']]):
                command.setdefault('attributes', {})['task_index'] = tid
                commands.append(command)
            task.update(task_index=tid, instruction_start=start, instruction_end=len(commands))
        rewritten.append(task)
        index += len(group)
    if not evidence:
        return command_buffer
    cb['commands'] = commands
    p = cb['params']['global_program_plan']
    p.update(tasks=rewritten, schedule_instruction_count=len(commands),
             epilogue_instruction_range=[len(commands)] * 2)
    cb['params']['sibling_fusion_evidence'] = evidence
    return cb


def fuse_shared_rhs_matmuls(command_buffer, *, automatic=False):
    """Group immutable RHS consumers across proven independent raw task runs.

    Only complete canonical contraction tasks are crossed. Every pair in each
    maximal candidate run must be read/write disjoint (shared reads allowed).
    Unknown tasks, wrappers, missing effects, aliases and dependencies are hard
    boundaries; this pass neither crosses them nor guesses their semantics.
    """
    plan = command_buffer.get('params', {}).get('global_program_plan')
    if not isinstance(plan, dict) or not plan.get('tasks'):
        return command_buffer
    tasks = plan['tasks']
    if (plan.get('prologue_instruction_range') != [0, 0]
            or plan.get('epilogue_instruction_range') != [len(command_buffer['commands'])] * 2
            or tasks[0]['instruction_start'] != 0
            or tasks[-1]['instruction_end'] != len(command_buffer['commands'])
            or any(a['instruction_end'] != b['instruction_start'] for a, b in zip(tasks, tasks[1:]))):
        return command_buffer
    def candidate(task):
        member = _candidate(command_buffer, task)
        if (member is None or not 1 <= member['m'] <= 16
                or any(str(key).startswith('_') for key in member['attributes'])):
            return None
        boundary = member['attributes']['source_numeric_boundary']
        if (boundary.get('inputs') != [{'shape': [member['m'], member['k']], 'dtype': 'i8'},
                                      {'shape': [member['k'], member['n']], 'dtype': 'i8'}]
                or boundary.get('outputs') != [{'shape': [member['m'], member['n']], 'dtype': 'i32'}]
                or member['attributes'].get('source_effects') not in (None, {})):
            return None
        # These canonical logical tensors have distinct materialized storage.
        # An explicit view/alias descriptor needs separate alias analysis.
        for name in (member['lhs'], member['rhs'], member['dst']):
            spec = command_buffer['tensors'][name]
            if any(key in spec for key in ('alias', 'alias_of', 'base', 'view_of',
                                           'storage', 'offset', 'strides')):
                return None
        return member
    units, evidence = [], []
    cursor = 0
    while cursor < len(tasks):
        run, reads, writes = [], set(), set()
        end = cursor
        while end < len(tasks):
            member = candidate(tasks[end])
            if member is None:
                break
            r, w = {member['lhs'], member['rhs']}, {member['dst']}
            if w & (reads | writes) or r & writes:
                break
            run.append((tasks[end], member));reads |= r;writes |= w;end += 1
        if not run:
            units.append(([tasks[cursor]], None));cursor += 1;continue
        # First-occurrence order is stable. Pairwise disjoint effects above
        # prove every moved task commutes with all tasks it passes over.
        groups = {}
        for task, member in run:
            key = (member['rhs'], member['m'], member['k'], member['n'])
            slots = groups.setdefault(key, [])
            if not slots or len(slots[-1]) == 8:
                slots.append([])
            slots[-1].append((task, member))
        if automatic:
            from .shared_rhs_policy import select_columns
            accepted, consumed = {}, set()
            for slots in groups.values():
                for group in slots:
                    members = [m for _, m in group]
                    if select_columns(members) is not None:
                        accepted[id(group[0][0])] = group
                        consumed.update(id(t) for t, _ in group[1:])
            # Keep all rejected tasks in their original relative order. Only
            # accepted consumers move, under the entire run's effect proof.
            for task, member in run:
                if id(task) in accepted:
                    group = accepted[id(task)]
                    units.append(([t for t, _ in group], [m for _, m in group]))
                elif id(task) not in consumed:
                    units.append(([task], None))
        else:
            for slots in groups.values():
                for group in slots:
                    units.append(([t for t, _ in group], [m for _, m in group] if len(group) > 1 else None))
        cursor = end
    if not any(members for _, members in units):
        return command_buffer
    cb = copy.deepcopy(command_buffer)
    commands, rewritten = [], []
    for group, members in units:
        tid, start = len(rewritten), len(commands)
        if members:
            members = copy.deepcopy(members)
            ownership = sorted({s for task in group for s in task['source_op_indices']})
            regions = sorted({r for task in group for r in task.get('source_region_ids', [])})
            operands = {'rhs': members[0]['rhs']}
            for i, member in enumerate(members):
                member['attributes']['task_index'] = tid
                operands['lhs' + str(i)] = member['lhs']
                operands['dst' + str(i)] = member['dst']
            proof = 'canonical exact signed-i8 zero-i32 one-tile-M tasks; pairwise disjoint read/write effects across entire reordered run; immutable shared RHS storage'
            commands.append({'opcode': 'VECTOR_MAP', 'operands': operands,
                             'attributes': {'op': 'shared_rhs_matmul', 'task_index': tid,
                                            'source_op_index': members[0]['attributes']['source_op_index'],
                                            'members': members, 'output_dtype': 'i32', 'fusion_proof': proof}})
            if automatic:
                from .shared_rhs_policy import select_columns, combine_rows
                commands[-1]['attributes']['_shared_rhs_columns'] = select_columns(members)
                if combine_rows(members):
                    commands[-1]['attributes']['_shared_rhs_combine_rows'] = True
            task = {'task_index': tid, 'kind': 'device', 'source_op_indices': ownership,
                    'source_region_ids': regions, 'instruction_start': start, 'instruction_end': len(commands),
                    'reads': list(dict.fromkeys(n for m in members for n in (m['lhs'], m['rhs']))),
                    'writes': [m['dst'] for m in members]}
            evidence.append({'task_index': tid, 'source_task_indices': [t['task_index'] for t in group],
                             'source_op_indices': ownership, 'shared_rhs': members[0]['rhs'],
                             'outputs': task['writes'], 'transformation': 'shared_rhs_single_m_tile_v1',
                             'source_effects_preserved': True, 'proof': proof})
        else:
            task = copy.deepcopy(group[0])
            for command in copy.deepcopy(command_buffer['commands'][task['instruction_start']:task['instruction_end']]):
                command.setdefault('attributes', {})['task_index'] = tid
                commands.append(command)
            task.update(task_index=tid, instruction_start=start, instruction_end=len(commands))
        rewritten.append(task)
    cb['commands'] = commands
    cb['params']['global_program_plan'].update(tasks=rewritten, schedule_instruction_count=len(commands),
                                             epilogue_instruction_range=[len(commands)] * 2)
    cb['params']['sibling_fusion_evidence'] = evidence
    return cb
