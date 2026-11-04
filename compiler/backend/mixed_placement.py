"""Source-bound declarations for an already compiled mixed tensor program.

No placement is selected here. Device execution comes from retained callback
artifacts; eligibility diagnostics reuse Merlin's selected capability oracle.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from math import prod
from pathlib import Path


class MixedPlacementError(ValueError):
    pass


class MixedPlacementBindingGap(MixedPlacementError):
    """A verified graph preparation lacks original-to-prepared op bindings."""


# These exact IR operations create storage/values or change only tensor metadata.
# Slice/copy/transpose operations intentionally do not belong to this list.
_INITIALIZERS = frozenset(('arith.constant', 'tensor.empty', 'tensor.splat', 'linalg.fill'))
_VIEWS = frozenset(('tensor.collapse_shape', 'tensor.expand_shape', 'tensor.cast'))
_SHAPE = frozenset(('tensor.dim', 'tensor.rank'))


def mixed_lane_placement(result, inventory, *, capability_map=None) -> list:
    """Declare actual device groups and every remaining original source operation.

    Fail rather than join different source versions or silently discard callback
    ownership. This function reads products and returns metadata only; it never
    rewrites source, commands, artifacts, or numerical policy. A caller may pass
    an already selected shared ``SemanticCapability`` map; this is descriptive,
    and grants no scalar equivalence proof or fallback permission. Missing shared
    resources remain explicit unknown eligibility, independently of device proof.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.oot_starterkit.plan import source_operation_inventory
    from merlin.targetgen.model_coverage import region_ops, regions_from_module, _ELEM_DTYPE
    from merlin.targetgen import eligibility as E
    from merlin.perf.lane_cost import dtype_bits
    from .placement import annotate_placement

    raw = Path(result.stats['adapter_original_source']).read_bytes()
    actual = source_operation_inventory(raw)
    if actual != inventory:
        raise MixedPlacementError('mixed placement inventory differs from exact original source')
    report = json.loads(Path(result.report_path).read_text())
    if report.get('source_sha256') != actual['source_sha256']:
        preparation = result.stats.get('adapter_host_graph_preparation')
        # Only an actual retained pre-discovery transform can explain a new
        # ordinal domain. A differing report hash alone is corrupt metadata.
        if isinstance(preparation, dict) and preparation.get('binding_stage') == 'before_device_region_discovery':
            try:
                source, output = preparation['source'], preparation['output']
                retained_source = Path(source['path']).read_bytes()
                retained_output = Path(output['path']).read_bytes()
                source_hash = hashlib.sha256(retained_source).hexdigest()
                output_hash = hashlib.sha256(retained_output).hexdigest()
            except (KeyError, TypeError, OSError) as exc:
                raise MixedPlacementError('prepared-source binding receipt lacks retained source products') from exc
            if (retained_source != raw or source_hash != source['sha256']
                    or output_hash != output['sha256'] or output_hash != report.get('source_sha256')):
                raise MixedPlacementError('prepared-source ordinal domain has stale or corrupt source identities')
            if retained_output != raw and preparation.get('changed') is True:
                # Check actual callback products in their real prepared domain
                # before allowing the caller to downgrade only the missing join.
                # Corrupt device/source records must remain fatal even when a
                # legitimate transformation also changed operation ordinals.
                prepared_result = copy(result)
                prepared_result.stats = dict(result.stats, adapter_original_source=output['path'])
                mixed_lane_placement(prepared_result, source_operation_inventory(retained_output),
                    capability_map=capability_map)
                raise MixedPlacementBindingGap('verified prepared-source ordinal domain differs from original source; original-to-prepared operation binding required')
        raise MixedPlacementError('device ordinal domain differs from original source without a verified preparation binding')
    module = parse_mlir_text(raw.decode('utf-8'))
    entry = [op for op in module.body.block.ops if op.name == 'func.func' and op.body.blocks]
    if len(entry) != 1:
        raise MixedPlacementError('placement requires one original tensor entry')
    direct = [op for op in entry[0].body.blocks[0].ops if op.name != 'func.return']
    indices = {op: i for i, op in enumerate(direct)}
    ordinals = dict(enumerate(module.walk()))
    operation_ordinals = {op: n for n, op in ordinals.items()}
    descriptors = dict(zip(region_ops(module), regions_from_module(module), strict=True))
    from .source_region_diagnostics import source_region_descriptors
    additional_regions = source_region_descriptors(module)
    for op, (descriptor, proof) in additional_regions.items():
        descriptors.setdefault(op, descriptor)
    from merlin.common.yaml import safe_load_text
    contract_path = Path(__file__).with_name('data') / 'target_contract.yaml'
    contract_raw = contract_path.read_bytes()
    contract = safe_load_text(contract_raw.decode('utf-8'))
    if contract.get('name') != 'gemmini':
        raise MixedPlacementError('selected capability contract names a different target')
    contract_sha256 = hashlib.sha256(contract_raw).hexdigest()

    def types(values):
        rows = []
        for value in values:
            ty = value.type
            if hasattr(ty, 'get_shape') and hasattr(ty, 'get_element_type'):
                shape, dtype = list(ty.get_shape()), str(ty.get_element_type())
                bits = dtype_bits(dtype)
                count = prod(shape) * (bits // 8) if bits and bits % 8 == 0 and all(d >= 0 for d in shape) else None
                rows.append(dict(shape=shape, dtype=dtype, bytes=count))
            else:
                rows.append(dict(type=str(ty), bytes=None))
        return rows

    def source_row(i):
        op = direct[i]
        inputs, outputs = types(op.operands), types(op.results)
        def total(rows):
            return sum(r['bytes'] for r in rows) if all(r['bytes'] is not None for r in rows) else None
        return dict(source_op_indices=[i], source_ordinals=[operation_ordinals[op]],
            source_sha256=actual['source_sha256'], capability_contract_sha256=contract_sha256, operation=inventory['operations'][i]['operation'],
            source_inventory_operation=deepcopy(inventory['operations'][i]),
            source_ids={k: str(v) for k, v in op.attributes.items() if k.startswith(('prov.', 'merlin.source'))},
            inputs=inputs, outputs=outputs, attributes={str(k):str(v) for k,v in op.attributes.items()},
            properties={str(k):str(v) for k,v in op.properties.items()},
            boundary_bytes=dict(inputs=total(inputs), outputs=total(outputs)),
            byte_basis='static logical source boundaries; not measured transfers',
            measured_transfer_bytes=None, measured_host_time=None)

    devices = report.get('device_regions')
    if not isinstance(devices, list) or len(devices) != len(result.device_artifacts):
        raise MixedPlacementError('device report does not cover actual callback artifacts')
    owned = set()
    rows = []
    for declaration, artifact in zip(devices, result.device_artifacts, strict=True):
        if declaration.get('kernel_symbol') != artifact.symbol or not Path(artifact.path).is_file():
            raise MixedPlacementError('device report differs from actual compiled symbol/artifact')
        source_path = Path(result.report_path).parent / declaration['symbol'] / 'source.mlir'
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != declaration.get('source_sha256'):
            raise MixedPlacementError('retained callback source identity changed')
        cb = annotate_placement(artifact.command_buffer)
        placements = cb['params']['lane_placement']
        group = declaration.get('source_ordinals')
        if not isinstance(group, list) or not group or len(group) != len(set(group)):
            raise MixedPlacementError('device region lacks unique source ordinals')
        bound = []
        auxiliaries = []
        for ordinal in group:
            op = ordinals.get(ordinal)
            if op not in indices:
                raise MixedPlacementError('device source ordinal is not an original direct entry operation')
            i = indices[op]
            if op.name in _INITIALIZERS:
                auxiliaries.append(i)
                continue
            if i in owned:
                raise MixedPlacementError('source compute operation is owned by multiple device callbacks')
            owned.add(i)
            bound.append(i)
        if not bound:
            raise MixedPlacementError('device callback owns no original source compute')
        # Recreate the original callback source with the shared extraction helper.
        # A valid ordinal that points to a different operation must not inherit
        # another kernel's declaration merely because its type happens to fit.
        from .source_groups import _dependencies
        from .host_runtime import DeviceRegion, _source_function
        members = tuple(ordinals[n] for n in group)
        root = direct[bound[-1]]
        reconstructed, external = _dependencies(root, members)
        if reconstructed != members or len(root.results) != 1:
            raise MixedPlacementError('device ordinals do not reconstruct the original closed source region')
        reconstructed_source = _source_function(DeviceRegion(members, external, (root.results[0],)), artifact.symbol)
        if hashlib.sha256(reconstructed_source.encode()).hexdigest() != declaration['source_sha256']:
            raise MixedPlacementError('device source ordinals differ from actual callback source semantics or ABI')
        callback_inventory = source_operation_inventory(source_path.read_bytes())
        def boundary_record(typ):
            shape, dtype = typ.get('shape'), typ.get('dtype')
            bits = dtype_bits(dtype)
            count = prod(shape) * (bits // 8) if shape is not None and bits and bits % 8 == 0 and all(d >= 0 for d in shape) else None
            return dict(**typ, bytes=count)
        inputs = [boundary_record(t) for t in callback_inventory['arguments']]
        outputs = [boundary_record({k:v for k,v in t.items() if k != 'source'}) for t in callback_inventory['returns']]
        rows.append(dict(lane='unit:gemmini', unit='gemmini', family='compiled_device_region',
            status='compiled_device', source_op_indices=bound,
            source_ordinals=[n for n in group if indices[ordinals[n]] in bound],
            requested_source_ordinals=list(group),
            source_auxiliary_op_indices=auxiliaries, source_sha256=actual['source_sha256'],
            capability_contract_sha256=contract_sha256,
            symbol=artifact.symbol, artifact=str(artifact.path),
            artifact_sha256=hashlib.sha256(Path(artifact.path).read_bytes()).hexdigest(),
            device_source_sha256=declaration['source_sha256'], device_form='compiled_callback',
            source_group_form='fused' if len(bound) > 1 else 'standalone',
            inputs=inputs, outputs=outputs, boundary_bytes=dict(
                inputs=sum(t['bytes'] for t in inputs) if all(t['bytes'] is not None for t in inputs) else None,
                outputs=sum(t['bytes'] for t in outputs) if all(t['bytes'] is not None for t in outputs) else None),
            source_operations=[source_row(i) for i in bound],
            selected_device_forms=deepcopy(placements), proof=declaration.get('reason'),
            byte_basis='nested static logical source/device boundaries; not measured transfers',
            measured_transfer_bytes=None, measured_host_time=None))

    resource_failure = None
    try:
        caps = E.capability_map_from_contract(contract) if capability_map is None else capability_map
        undetermined = E.undetermined_families_from_contract(contract)
        providers = E.providers_from_contract(contract) if capability_map is None else None
    except FileNotFoundError as exc:
        caps = None
        resource_failure = dict(status='unresolved_capability_resource',
            dependency_path=str(exc.filename), error=str(exc),
            capability_contract_sha256=contract_sha256)
    from .source_groups import placement_diagnostics
    diagnostics = {r['source_index']: r for r in placement_diagnostics(module)['regions']}
    for i, op in enumerate(direct):
        if i in owned:
            continue
        row = source_row(i)
        if op in additional_regions:
            row['source_descriptor_proof'] = deepcopy(additional_regions[op][1])
        if op.name in _INITIALIZERS | _VIEWS | _SHAPE:
            kind = 'initialization' if op.name in _INITIALIZERS else 'view' if op.name in _VIEWS else 'shape'
            row.update(lane='host', family=kind, status='noncompute', reason='original typed '+kind+' operation')
        elif caps is None:
            row.update(lane='host', family=descriptors[op].resolved_family() if op in descriptors else 'unknown',
                status='unresolved_capability_resource', eligibility=None,
                reason='selected shared capability oracle resource unavailable; eligibility is unknown',
                capability_resolution=deepcopy(resource_failure))
        elif op in descriptors:
            descriptor = descriptors[op]
            output = types(op.results)
            out = _ELEM_DTYPE.get(output[0].get('dtype')) if output else None
            # The original tensor type independently identifies a plain machine
            # float. Preserve that evidence instead of introducing a registry
            # alias (fp32) into a decision about a literal primitive contract.
            def machine_float(dtype):
                return isinstance(dtype, str) and dtype.startswith('f') and dtype[1:].isdigit()
            if output and machine_float(output[0].get('dtype')):
                out = output[0]['dtype']
            changes = dict(out_dtype=out)
            roles = types(getattr(op, 'inputs', ()))
            observed = descriptor.captured_input_formats
            if observed is not None and len(roles) == len(observed):
                original = tuple(_ELEM_DTYPE.get(role.get('dtype')) for role in roles)
                if original == observed:
                    selected = tuple(role['dtype'] if machine_float(role.get('dtype')) else fmt
                        for role, fmt in zip(roles, observed, strict=True))
                    changes['captured_input_formats'] = selected
                    if original and descriptor.in_dtype == original[0]:
                        changes['in_dtype'] = selected[0]
                    if len(original) > 1 and descriptor.weight_dtype == original[1]:
                        changes['weight_dtype'] = selected[1]
            descriptor = replace(descriptor, **changes)
            try:
                verdict = E.is_eligible(descriptor, caps, undetermined=undetermined, providers=providers)
                parents = frozenset(descriptors[v.owner].resolved_family() for v in op.operands
                    if v.owner in descriptors and descriptors[v.owner].resolved_family())
                fused = E.is_eligible(descriptor, caps, undetermined=undetermined,
                    providers=providers, fused_with=parents)
            except FileNotFoundError as exc:
                row.update(lane='host', family=descriptor.resolved_family() or 'unknown',
                    status='unresolved_capability_resource', eligibility=None,
                    reason='selected shared format resource unavailable for this operation; eligibility is unknown',
                    capability_resolution=dict(status='unresolved_capability_resource',
                        dependency_path=str(exc.filename), error=str(exc),
                        capability_contract_sha256=contract_sha256))
            else:
                # Coarse family eligibility is not a scalar-body/numeric equivalence proof.
                status = 'unresolved_proof'
                if not verdict.undetermined and verdict.refusal in ('input_dtype','weight_dtype','operand_pair','result_dtype','undeclared_family'):
                    status = 'hardware_unsupported'
                if verdict.eligible or fused.eligible:
                    status = 'compiler_adapter_gap'
                row.update(lane='host', family=descriptor.resolved_family() or 'unknown', status=status,
                    reason=verdict.reason, standalone_capability=asdict(verdict),
                    fused_capability=asdict(fused), fused_with=sorted(parents),
                    remaining_proof=None if status == 'hardware_unsupported' else 'exact scalar-body, layout, numeric order and admitted device-form realization')
        else:
            row.update(lane='host', family='unknown', status='unresolved_proof',
                reason='no shared computation descriptor for this original operation; hardware support is unknown')
        ordinal = row['source_ordinals'][0]
        if ordinal in diagnostics:
            row['source_numeric_diagnostic'] = deepcopy(diagnostics[ordinal])
        row['scalar_body_operations'] = [nested.name for nested in op.walk() if nested is not op]
        rows.append(row)
    covered = [i for row in rows for i in row['source_op_indices']]
    if sorted(covered) != list(range(len(direct))):
        raise MixedPlacementError('placement does not partition original source operations')
    return rows
