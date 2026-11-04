"""Source-bound placement accounting, separate from execution qualification."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from math import prod
from pathlib import Path


def _attr(op, name):
    value = op.attributes.get(name)
    if value is None:
        return None
    return getattr(value, 'data', str(value))


def _ids(op, name):
    value = op.attributes.get(name)
    return [getattr(item, 'data', str(item)) for item in value] if value is not None else []


def _name(op):
    name = getattr(op, 'op_name', None)
    return name.data if op.name == 'builtin.unregistered' and name is not None else op.name


def _size(spec):
    shape, dtype = spec.get('shape'), spec.get('dtype', '')
    if not isinstance(shape, (list, tuple)) or any(type(d) is not int or d < 0 for d in shape):
        return None
    # These are actual MLIR scalar storage widths, not capability declarations.
    spelling = dtype[1:] if dtype.startswith(('i', 'f')) else '16' if dtype == 'bf16' else ''
    if not spelling.isdigit():
        return None
    return (prod(shape) * int(spelling) + 7) // 8


def _contract(value):
    if value is None or isinstance(value, dict):
        return value
    import yaml
    return yaml.safe_load(Path(value).read_text())


def build_placement_report(source, command_buffer, *, target_contract=None):
    """Join exact original source operations to final planned device/host work.

    Supply the selected public target contract, never a compiler-invented map.
    Eligibility is the native declarative predicate, not emitted-code presence.
    Missing source classifications, data files and numerical fusion proofs remain
    explicit unknowns. Planned coverage is not a measured execution certificate.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.oot_starterkit.plan import source_operation_inventory
    from merlin.targetgen.model_coverage import (regions_from_module, _is_region_op,
        region_family, _captured_input_formats, _ELEM_DTYPE)
    from merlin.targetgen.eligibility import RegionDescriptor
    from merlin.targetgen.eligibility import (
        capability_map_from_contract, providers_from_contract, undetermined_families_from_contract,
        is_eligible,
    )
    from source_attribution import task_definitions, materialized_task_accesses

    if target_contract is None:
        selected = Path(__file__).resolve().parent / 'contracts' / 'target_contract.yaml'
        target_contract = selected if selected.is_file() else None
    source = source.decode() if isinstance(source, bytes) else source
    inventory = source_operation_inventory(source)
    module = parse_mlir_text(source)
    owner = next(op for op in module.body.block.ops if op.name == 'func.func'
                 and op.sym_name.data == inventory['entry'])
    direct = [op for op in owner.body.block.ops if op.name != 'func.return']
    params = command_buffer.get('params', {})
    gaps = []
    try:
        tasks = task_definitions(command_buffer)
    except ValueError as exc:
        tasks = []
        gaps.append({'stage': 'task_ownership', 'detail': str(exc)})
    task_by_op = {}
    for task in tasks:
        for index in task.get('source_op_indices', []):
            task_by_op.setdefault(index, []).append(task)
    commands = command_buffer.get('commands', [])
    transports = params.get('source_transport_tasks', [])
    contract = _contract(target_contract)
    caps = providers = undetermined = None
    if contract is not None:
        try:
            caps = capability_map_from_contract(contract)
            providers = providers_from_contract(contract)
            undetermined = undetermined_families_from_contract(contract)
        except Exception as exc:
            gaps.append({'stage': 'selected_capability_map', 'detail': str(exc)})
    else:
        gaps.append({'stage': 'selected_capability_map', 'detail': 'selected target contract not supplied'})
    descriptors = {}
    try:
        carriers = [op for op in module.walk() if _is_region_op(op)]
        descriptors = dict(zip(carriers, regions_from_module(module), strict=True))
    except Exception as exc:
        gaps.append({'stage': 'native_region_inventory', 'detail': str(exc)})
        # The richer shape observer can require an absent packaged registry.
        # Retain native family/operand observations directly, without replacing
        # that registry or guessing dtype aliases. Shape-dependent proof remains
        # absent; the declarative predicate must report its own result.
        for op in module.walk():
            if not _is_region_op(op):
                continue
            observed = _captured_input_formats(op)
            family = region_family(op)
            rank = next((len(v.type.get_shape()) for v in op.results
                         if hasattr(v.type, 'get_shape')), None)
            descriptors[op] = RegionDescriptor(source=op.name, op=op.name, family=family,
                in_dtype=observed[0] if observed else None,
                weight_dtype=observed[1] if family == 'contraction' and len(observed) > 1 else None,
                captured_input_formats=observed, rank=rank)
    # Reopen the complete native demand reader separately; it can diagnose
    # carriers beyond the linalg-region reader. Its absence never shrinks counts.
    demand_status = {'status': 'unknown', 'count': None}
    try:
        from merlin.targetgen.source_demand_inventory import model_op_demands_checked
        demands = model_op_demands_checked(source, 'int8', 'int8')
        demand_status = {'status': 'complete', 'count': len(demands)}
    except Exception as exc:
        demand_status['reason'] = str(exc)
    rows = []
    for index, op in enumerate(direct):
        owned = task_by_op.get(index, [])
        host_tasks = [task for task in owned if task.get('kind') == 'host_source']
        device_tasks = [task for task in owned if task.get('kind') != 'host_source']
        task_indices = {task['task_index'] for task in device_tasks}
        command_indices = sorted({i for task in device_tasks for i in task.get('command_indices', [])
                                  if type(i) is int and 0 <= i < len(commands)})
        transport_rows = [row for row in transports if row.get('source_task_index') in task_indices]
        forms = sorted({commands[i]['opcode'] for i in command_indices}
                       | {str(row.get('kind', row.get('operation', 'source_tensor_transport')))
                          for row in transport_rows})
        placement = ('mixed' if host_tasks and device_tasks else 'host' if host_tasks
                     else 'accelerator' if device_tasks else 'unassigned')
        eligibility = None
        eligibility_reason = 'native source descriptor unavailable for this operation'
        standalone = fused = None
        descriptor = descriptors.get(op)
        if descriptor is not None and caps is not None:
            try:
                output_types = {str(value.type.element_type) for value in op.results
                                if hasattr(value.type, 'element_type')}
                from merlin.targetgen.model_coverage import _ELEM_DTYPE
                out = next(iter(output_types)) if len(output_types) == 1 else None
                # Native descriptors may spell f32 as the registry name fp32.
                # Retain the actual parsed machine spelling instead when source
                # roles are available; no quantization metadata is inferred.
                inputs = tuple(getattr(op, 'inputs', ()))
                observed = tuple(str(v.type.element_type) if hasattr(v.type, 'element_type') else None
                                 for v in inputs)
                descriptor = replace(descriptor, out_dtype=out,
                    captured_input_formats=observed if inputs else descriptor.captured_input_formats)
                verdict = is_eligible(descriptor, caps, providers=providers, undetermined=undetermined)
                standalone = {'eligible': verdict.eligible, 'unknown': verdict.undetermined,
                              'reason': verdict.reason, 'refusal': verdict.refusal}
                eligibility = None if verdict.undetermined else verdict.eligible
                eligibility_reason = verdict.reason
                producer_families = {descriptors[v.owner].resolved_family() for v in op.operands
                                     if v.owner in descriptors}
                producer_families.discard(None)
                if producer_families:
                    attached = is_eligible(descriptor, caps, providers=providers, undetermined=undetermined,
                                           fused_with=frozenset(producer_families))
                    fused = {'eligible': attached.eligible, 'unknown': attached.undetermined,
                             'reason': attached.reason, 'producer_families': sorted(producer_families)}
                    if attached.eligible and not verdict.eligible:
                        # Producer adjacency alone does not establish the numeric
                        # order or that an actual compatible fusion was emitted.
                        eligibility = None
                        eligibility_reason = 'fused-only declared capability; exact source numeric/placement proof unresolved'
            except Exception as exc:
                eligibility = None
                eligibility_reason = 'native eligibility unavailable: ' + str(exc)
        elif caps is None:
            eligibility_reason = 'selected target capability map unavailable'
        host_reason = None
        if host_tasks:
            explicit = _attr(op, 'prov.placement')
            host_reason = ('explicit source host_required' if explicit == 'host_required'
                           else 'original typed host seam selected by emitted host task; hardware eligibility evaluated separately')
        covered = bool(device_tasks and (forms or any(task.get('kind') == 'source_tensor_transport' for task in device_tasks)))
        gap = ('eligible source work remains host' if eligibility is True and host_tasks else
               'eligible source operation lacks device emission' if eligibility is True and not covered else
               eligibility_reason if eligibility is None else None)
        rows.append({'source_op_index': index, 'operation': _name(op),
                     'source_region_id': _attr(op, 'prov.region_id'),
                     'source_node_ids': _ids(op, 'prov.source_node_ids'),
                     'origin_node_ids': _ids(op, 'prov.origin_node_ids'),
                     'family': descriptor.resolved_family() if descriptor is not None else None,
                     'operand_types': [str(v.type) for v in op.operands],
                     'result_types': [str(v.type) for v in op.results],
                     'placement': placement, 'task_indices': [task['task_index'] for task in owned],
                     'selected_device_forms': forms, 'host_reason': host_reason,
                     'emission_gap': 'device task declared without command/transport form evidence' if device_tasks and not forms else None,
                     'eligible': eligibility, 'required': eligibility is True,
                     'covered_by_planned_device_work': covered,
                     'standalone_capability': standalone, 'fused_capability': fused,
                     'gap': gap,
                     'nested_operations': [{'operation': _name(child),
                          'source_region_id': _attr(child, 'prov.region_id'),
                          'operand_types': [str(v.type) for v in child.operands],
                          'result_types': [str(v.type) for v in child.results],
                          'source_node_ids': _ids(child, 'prov.source_node_ids'),
                          'origin_node_ids': _ids(child, 'prov.origin_node_ids')}
                         for child in op.walk() if child is not op]})
    accesses = {}
    for task in tasks:
        try:
            reads, writes = materialized_task_accesses(inventory, command_buffer, task['source_op_indices'])
            accesses[task['task_index']] = (set(task.get('reads', reads)), set(task.get('writes', writes)))
        except Exception as exc:
            gaps.append({'stage': 'static_boundary', 'task_index': task.get('task_index'), 'detail': str(exc)})
    boundaries = []
    for producer in tasks:
        for consumer in tasks:
            p, c = producer['task_index'], consumer['task_index']
            if p >= c or p not in accesses or c not in accesses:
                continue
            if (producer.get('kind') == 'host_source') == (consumer.get('kind') == 'host_source'):
                continue
            for tensor in sorted(accesses[p][1] & accesses[c][0]):
                boundaries.append({'tensor': tensor, 'producer_task': p, 'consumer_task': c,
                                   'logical_bytes': _size(command_buffer.get('tensors', {}).get(tensor, {}))})
    return {'schema': 'source_bound_placement_diagnostics_v1',
            'source_sha256': inventory['source_sha256'], 'target': command_buffer.get('target'),
            'capability_contract_sha256': sha256(__import__('json').dumps(contract, sort_keys=True).encode()).hexdigest() if contract is not None else None,
            'scope': 'planned source ownership and declarative capability; no execution or numeric qualification',
            'native_complete_demand_inventory': demand_status, 'operations': rows,
            'eligibility_complete': all(row['eligible'] is not None for row in rows),
            'count_scope': 'required and covered count only positively established eligible operations; unknown operations remain outside that denominator',
            'counts': {'source_operations': len(rows), 'required': sum(row['required'] for row in rows),
                       'covered': sum(row['required'] and row['covered_by_planned_device_work'] for row in rows),
                       'planned_device_operations': sum(row['covered_by_planned_device_work'] for row in rows),
                       'planned_host_operations': sum(row['placement'] in ('host', 'mixed') for row in rows),
                       'unassigned_operations': sum(row['placement'] == 'unassigned' for row in rows),
                       'eligible_but_host': sum(row['eligible'] is True and row['placement'] in ('host', 'mixed') for row in rows),
                       'unknown': sum(row['eligible'] is None for row in rows)},
            'static_boundary': {'logical_bytes': sum(row['logical_bytes'] for row in boundaries) if all(row['logical_bytes'] is not None for row in boundaries) else None,
                                'edges': boundaries, 'scope': 'logical tensor bytes per cross-placement edge; excludes execution frequency and physical transfer traffic'},
            'measured_host_time': None, 'measured_transferred_bytes': None, 'gaps': gaps}


def attach_placement_report(source, command_buffer, *, target_contract=None):
    report = build_placement_report(source, command_buffer, target_contract=target_contract)
    command_buffer.setdefault('params', {})['source_placement_diagnostics'] = report
    return report
