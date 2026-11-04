"""Whole-source family/shape work profiles joined to actual compiler placement.

Source operations, including host seams and operations with unknown costs, stay
in the denominator. Work ranks are static logical counts, never time estimates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from math import prod


def build_model_profile(source, command_buffer, *, source_profile=None,
                        placement_report=None, entry_invocations=1):
    """Return a JSON-safe model profile for the exact original source and plan.

    Existing ``source_work_profile`` and ``source_placement_diagnostics`` are
    consumed when present. Otherwise the reusable source profiler supplies work
    and explicit task ownership. No operation-name rule chooses placement or
    hardware eligibility. Counts default to one entry evaluation; an unknown
    runtime entry count must be passed as ``entry_invocations=None``.
    """
    from merlin.targetgen.oot_starterkit.plan import source_operation_inventory
    from source_profile import build_source_profile
    from merlin.frontends.linalg_mlir import parse_mlir_text
    source = source.decode() if isinstance(source, bytes) else source
    inventory = source_operation_inventory(source)
    digest = inventory['source_sha256']
    params = command_buffer.get('params', {})
    placement = placement_report if placement_report is not None else params.get('source_placement_diagnostics')
    work = source_profile if source_profile is not None else params.get('source_work_profile')
    for label, report in (('work', work), ('placement', placement)):
        if report is not None and report.get('source_sha256') != digest:
            raise ValueError(label + ' profile digest differs from exact original source')
    if entry_invocations is not None and (type(entry_invocations) is not int or entry_invocations < 0):
        raise ValueError('entry_invocations must be a static nonnegative integer or None')
    if work is None or work.get('entry_invocations') != entry_invocations:
        work = build_source_profile(source, command_buffer, entry_invocations=entry_invocations,
                                    placement_report=placement)
    for key in ('source_sha256', 'compiler_source_sha256'):
        declared = params.get(key)
        if declared is not None and declared != digest:
            raise ValueError('command plan digest differs from exact original source')
    nops = inventory['source_op_count']
    if work.get('counts', {}).get('source_operations') != nops:
        raise ValueError('work profile operation count differs from original source')
    placement_rows = {row['source_op_index']: row for row in (placement or {}).get('operations', [])}
    # Exact node identities are source observations, independently of whether
    # an eligibility/placement report was available to the caller.
    module = parse_mlir_text(source)
    owner = next(op for op in module.body.block.ops if op.name == 'func.func'
                 and op.sym_name.data == inventory['entry'])
    direct = tuple(op for op in owner.body.block.ops if op.name != 'func.return')
    def ids(index, name):
        attr = direct[index].attributes.get(name)
        return [getattr(value, 'data', str(value)) for value in attr] if attr is not None else []
    identities = [{'source_op_index': i, 'source_region_id': row.get('region_id'),
                   'source_node_ids': ids(i, 'prov.source_node_ids'),
                   'origin_node_ids': ids(i, 'prov.origin_node_ids')}
                  for i, row in enumerate(inventory['operations'])]
    actual_placements = defaultdict(set)
    for region in work.get('regions', []):
        observed = region.get('placement')
        if observed in ('host', 'accelerator', 'mixed'):
            for index in region['source_op_indices']:
                actual_placements[index].add(observed)
    def placed(index):
        reported = placement_rows.get(index, {}).get('placement')
        if reported in ('host', 'accelerator', 'mixed', 'unassigned'):
            return reported
        observed = actual_placements[index]
        return (next(iter(observed)) if len(observed) == 1 else 'mixed' if len(observed) > 1
                else 'UNKNOWN')
    for identity in identities:
        index = identity['source_op_index']
        evidence = placement_rows.get(index, {})
        identity.update(operation=inventory['operations'][index].get('operation', direct[index].name),
                        source_scope={'entry': inventory['entry'], 'top_level_ordinal': index},
                        placement=placed(index),
                        placement_evidence='native placement report' if evidence else 'explicit task ownership' if actual_placements[index] else 'UNKNOWN',
                        task_indices=sorted({region['task_index'] for region in work.get('regions', [])
                            if index in region['source_op_indices']}),
                        selected_device_forms=deepcopy(evidence.get('selected_device_forms', [])),
                        host_reason=evidence.get('host_reason'),
                        hardware_eligible=evidence.get('eligible'),
                        eligibility_gap=evidence.get('gap'))
    nested_counts = Counter()
    nested_scopes = []
    for index, op in enumerate(direct):
        children = list(op.walk())[1:]
        for child_index, child in enumerate(children):
            opname = getattr(getattr(child, 'op_name', None), 'data', child.name)
            nested_counts[opname] += 1
            def child_ids(name):
                attr = child.attributes.get(name)
                return [getattr(value, 'data', str(value)) for value in attr] if attr is not None else []
            nested_scopes.append({'entry': inventory['entry'], 'enclosing_source_op_index': index,
                'nested_preorder_index': child_index, 'operation': opname,
                'source_node_ids': child_ids('prov.source_node_ids'),
                'origin_node_ids': child_ids('prov.origin_node_ids'),
                'enclosing_task_placement': placed(index)})
    unknown_work = {row['source_op_index']: row['reasons'] for row in work.get('unknown_work', [])}
    boundaries = {row['source_op_index']: row for row in work.get('quantization_and_conversion_boundaries', [])}
    variants = []
    families = {}
    for group_index, group in enumerate(work.get('operator_shape_groups', [])):
        indices = group['source_op_indices']
        if any(type(i) is not int or not 0 <= i < nops for i in indices):
            raise ValueError('shape group contains an absent original source ordinal')
        observed_families = {placement_rows.get(i, {}).get('family') for i in indices}
        # A shape group is indivisible for its already-proved work total. Never
        # distribute its subtotal among semantic families by occurrence ratios.
        native_family = next(iter(observed_families)) if len(observed_families) == 1 and None not in observed_families else None
        family_key = native_family if native_family is not None else 'operator:' + group['operation']
        if len(observed_families - {None}) > 1:
            family_key = 'mixed_native_families:' + group['operation']
        placement_counts = Counter(placed(i) for i in indices)
        row = {'shape_group_index': group_index, 'family': family_key,
               'family_basis': 'native source descriptor' if native_family else 'exact source operation; semantic family unavailable or mixed',
               'operation': group['operation'], 'source_op_indices': list(indices),
               'inputs': deepcopy(group['inputs']), 'outputs': deepcopy(group['outputs']),
               'layout_properties': deepcopy(group['layout_properties']),
               'scalar_body_operator_counts': deepcopy(group['scalar_body']),
               'iteration_extents': deepcopy(group.get('iteration_extents')),
               'per_entry_static_occurrences': len(indices),
               'declared_invocations': len(indices) * entry_invocations if entry_invocations is not None else None,
               'invocation_count_kind': 'static per entry; declared entry multiplier' if entry_invocations is not None else 'per-entry static only; runtime entry count UNKNOWN',
               'known_macs_per_entry': group.get('known_macs_per_entry', 0),
               'known_scalar_evaluations_per_entry': group.get('known_scalar_evaluations_per_entry', 0),
               'numeric_work_unknown_indices': [i for i in indices if i in unknown_work],
               'placement_counts': dict(placement_counts),
               'numeric_boundary_indices': [i for i in indices if i in boundaries],
               'exact_numeric_boundaries': [deepcopy(boundaries[i]) for i in indices if i in boundaries],
               'elapsed_cost': {'status': 'UNKNOWN', 'reason': 'no whole-source elapsed measurement supplied'},
               'logical_output_elements_status': 'static' if all(value.get('shape') is not None
                    and all(type(d) is int and d >= 0 for d in value['shape'])
                    for value in group['outputs']) else 'partial; dynamic or nontensor outputs UNKNOWN',
               'logical_output_elements_per_entry': sum(prod(value['shape']) * len(indices)
                    for value in group['outputs'] if value.get('shape') is not None
                    and all(type(d) is int and d >= 0 for d in value['shape']))}
        variants.append(row)
        family = families.setdefault(family_key, {'family': family_key, 'source_op_indices': [],
            'shape_group_indices': [], 'operator_counts': Counter(), 'placement_counts': Counter(),
            'known_macs_per_entry': 0, 'known_scalar_evaluations_per_entry': 0,
            'numeric_work_unknown_indices': [], 'numeric_boundary_indices': [],
            'elapsed_cost_status': 'UNKNOWN'})
        family['source_op_indices'].extend(indices)
        family['shape_group_indices'].append(group_index)
        family['operator_counts'][group['operation']] += len(indices)
        family['placement_counts'].update(placement_counts)
        for key in ('known_macs_per_entry', 'known_scalar_evaluations_per_entry'):
            family[key] += row[key]
        family['numeric_work_unknown_indices'].extend(row['numeric_work_unknown_indices'])
        family['numeric_boundary_indices'].extend(row['numeric_boundary_indices'])
    observed_indices = [i for row in variants for i in row['source_op_indices']]
    if sorted(observed_indices) != list(range(nops)):
        raise ValueError('source shape groups do not partition all original operations exactly once')
    ranked = sorted(families.values(), key=lambda row:
        (row['known_scalar_evaluations_per_entry'], row['known_macs_per_entry']), reverse=True)
    for row in ranked:
        row['source_op_indices'].sort()
        row['numeric_work_unknown_indices'].sort()
        row['numeric_boundary_indices'].sort()
        row['operator_counts'] = dict(row['operator_counts'])
        row['placement_counts'] = dict(row['placement_counts'])
        row['per_entry_static_occurrences'] = len(row['source_op_indices'])
    host_regions = sorted((deepcopy(row) for row in work.get('regions', []) if row.get('placement') == 'host'),
                          key=lambda row: (row['known_scalar_evaluations_per_entry'], row['known_macs_per_entry']), reverse=True)
    counts = Counter(placed(i) for i in range(nops))
    return {'schema': 'whole_source_model_profile_v1', 'source_sha256': digest,
            'entry': inventory['entry'], 'scope': 'original whole-source operations and planned placement; static logical work only',
            'count_scope': 'all original operations, including host seams and unknown costs; one source entry evaluation unless multiplied explicitly',
            'entry_invocations': entry_invocations,
            'entry_invocation_kind': 'caller-declared static multiplier' if entry_invocations is not None else 'runtime count UNKNOWN',
            'source_identities': identities,
            'identity_scope': 'canonical entry source ordinals; nested operator identities use enclosing ordinal and local preorder index; nested placement refers only to enclosing task ownership',
            'nested_source_scopes': nested_scopes,
            'counts': {'source_operations': nops, 'shape_groups': len(variants),
                'nested_source_operations': len(nested_scopes),
                'nested_source_operator_counts': dict(nested_counts),
                'families': len(ranked), 'placement': dict(counts),
                'known_macs_per_entry': work['counts']['known_macs_per_entry'],
                'known_macs_declared_invocations': work['counts']['known_macs_per_entry'] * entry_invocations if entry_invocations is not None else None,
                'known_scalar_evaluations_per_entry': sum(row['known_scalar_evaluations_per_entry'] for row in variants),
                'numeric_work_unknown_operations': len(unknown_work),
                'elapsed_cost_unknown_operations': nops,
                'eligibility': deepcopy((placement or {}).get('counts'))},
            'ranked_source_families': ranked, 'shape_layout_numeric_variants': variants,
            'ranked_host_regions': host_regions,
            'unknown_numeric_work': deepcopy(work.get('unknown_work', [])),
            'control_and_call_multiplicities': deepcopy(work.get('control_and_call_multiplicities', [])),
            'numeric_boundaries': [deepcopy(row) for row in boundaries.values()],
            'task_graph_edges': deepcopy(work.get('task_graph_edges', [])),
            'contraction_interstage_edges': deepcopy(work.get('contraction_interstage_edges', [])),
            'ranking_scope': 'proved static scalar evaluations, then MACs; UNKNOWN work is not free, and this is not a latency ranking',
            'whole_chain_elapsed': None, 'measured_host_elapsed': None, 'physical_transfer_bytes': None,
            'gaps': deepcopy(work.get('gaps', [])) + deepcopy((placement or {}).get('gaps', []))}


def attach_model_profile(source, command_buffer, **options):
    profile = build_model_profile(source, command_buffer, **options)
    command_buffer.setdefault('params', {})['model_profile'] = profile
    return profile
