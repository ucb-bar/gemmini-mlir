"""Original-index transport for Merlin's exact immutable preparation CSE.

This is invocation-local source reuse, not a persistent value cache or an arena
allocator. All arithmetic and matching decisions belong to the existing library.
The original module, ABI, returned writers and every source ordinal survive as
actual retained operations or explicit aliases to the dominating preparation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from io import StringIO
from pathlib import Path

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntegerAttr, i64
from xdsl.printer import Printer
from source_preparation import _owner
from merlin.llvmlower import tensor_provenance_cse


@dataclass(frozen=True)
class HostRegionPreparation:
    module: object
    witnesses: tuple[dict, ...]
    source_ordinals: tuple[int, ...]
    aliases: tuple[dict, ...]
    report: dict


def _text(module):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def _remove_marker(attribute, marker):
    if isinstance(attribute, DictionaryAttr):
        return DictionaryAttr({key: _remove_marker(value, marker)
                               for key, value in attribute.data.items() if key != marker})
    if isinstance(attribute, ArrayAttr):
        return ArrayAttr([_remove_marker(value, marker) for value in attribute])
    return attribute


def _division_refusals(ops, block):
    """Describe distinct actual divisors; this census never grants equivalence."""
    candidates = {}
    for ordinal, operation in enumerate(ops):
        if (operation.name != 'linalg.generic' or len(operation.regions) != 1
                or len(operation.regions[0].blocks) != 1 or len(operation.operands) != 3):
            continue
        body = tuple(operation.regions[0].block.ops)
        args = tuple(operation.regions[0].block.args)
        if (len(args) != 3 or len(body) != 2 or body[0].name != 'arith.divf'
                or body[1].name != 'linalg.yield'
                or tuple(body[0].operands) != args[:2]
                or tuple(body[1].operands) != tuple(body[0].results)):
            continue
        candidates.setdefault(operation.operands[0], []).append((ordinal, operation.operands[1]))
    return tuple({'reason': 'shared_input_has_distinct_source_divisors',
                  'source_op_indices': [ordinal for ordinal, _ in rows],
                  'divisor_argument_indices': [value.index if value.owner is block else None
                                               for _, value in rows]}
                 for rows in candidates.values()
                 if len(rows) > 1 and len({value for _, value in rows}) > 1)


def prepare_host_regions(module, *, enabled=False, entry='forward'):
    """Consume the existing preparation contract before source region cutting.

    The returned fields match LayoutPreparation. For integration, use this product
    before other source rewrites, or compose every later direct ordinal and alias
    through ``source_ordinals``; never apply these original indices to new source.
    Only an explicit Boolean enables the transform. No loop placement, float
    reassociation, mutable storage alias or completion permission is inferred.
    """
    if type(enabled) is not bool:
        raise ValueError('exact host preparation reuse requires an explicit Boolean')
    module.verify()
    source = _text(module)
    owner = _owner(module, entry)
    original_ops = tuple(owner.body.block.ops)
    digest = hashlib.sha256(source.encode()).hexdigest()
    report = {'schema': 'merlin.host_region_reuse.v1', 'requested': enabled,
              'source_sha256': digest, 'output_sha256': digest,
              'status': 'off', 'removed_operations': {},
              'scope': 'one immutable source SSA invocation',
              'temporary_storage': {'status': 'refused',
                  'reason': 'requires_original_storage_policy_and_live_call_lifetime'},
              'numerics': 'not_established', 'timing': 'UNKNOWN'}
    identity = tuple(range(len(original_ops)))
    if not enabled:
        return HostRegionPreparation(module, (), identity, (), report)
    report['refusals'] = list(_division_refusals(original_ops, owner.body.block))
    definitions = [op for op in module.body.block.ops if op.name == 'func.func'
                   and op.regions and op.regions[0].blocks]
    if definitions != [owner]:
        report.update(status='unchanged', reason='requires_single_original_source_entry')
        return HostRegionPreparation(module, (), identity, (), report)
    if any('strictfp' in key or 'strictfp' in str(value)
           for key, value in (*owner.attributes.items(), *owner.properties.items())):
        report.update(status='unchanged', reason='strict_floating_environment')
        return HostRegionPreparation(module, (), identity, (), report)
    prepared = module.clone()
    prepared_owner = _owner(prepared, entry)
    block = prepared_owner.body.block
    cloned_ops = tuple(block.ops)
    ordinals = dict(zip(cloned_ops, identity, strict=True))
    marker = 'prov.host_region_reuse_original_ordinal'
    while any(marker in op.attributes for op in prepared.walk()):
        marker += '_'
    # The library deliberately strips provenance for matching and retains all
    # merged provenance. This temporary lineage field is therefore descriptive,
    # never an extra semantic matching permission.
    for operation, ordinal in ordinals.items():
        operation.attributes[marker] = IntegerAttr(ordinal, i64)
    counts = tensor_provenance_cse.deduplicate_tensor_preparation(prepared)
    retained = tuple(block.ops)
    retained_ordinals = tuple(ordinals[op] for op in retained)
    alias_rows = []
    for operation in retained:
        records = operation.attributes.get('prov.cse_sources')
        if not isinstance(records, ArrayAttr):
            continue
        for record in records:
            if not isinstance(record, DictionaryAttr):
                continue
            source_index = record.data.get(marker)
            if not isinstance(source_index, IntegerAttr):
                continue
            source_index = source_index.value.data
            if source_index == ordinals[operation]:
                continue
            alias_rows.append({'source_ordinal': source_index,
                               'retained_ordinal': ordinals[operation],
                               'operation': original_ops[source_index].name,
                               'mechanism': 'merlin.llvmlower.tensor_provenance_cse'})
    aliases = tuple(sorted(alias_rows, key=lambda row: row['source_ordinal']))
    removed = set(identity) - set(retained_ordinals)
    if len(aliases) != len(removed) or {row['source_ordinal'] for row in aliases} != removed:
        raise ValueError('library preparation CSE lost exact original source ownership')
    if Counter(original_ops[index].name for index in removed) != Counter(counts):
        raise ValueError('library preparation CSE count differs from actual transformed graph')
    original_return = next(op for op in original_ops if op.name == 'func.return')
    returned_writers = {identity[original_ops.index(value.owner)] for value in original_return.operands
                        if value.owner in original_ops}
    if returned_writers - set(retained_ordinals):
        raise ValueError('host region reuse removed an original source output writer')
    if owner.function_type != prepared_owner.function_type:
        raise ValueError('host region reuse changed original logical input/output ABI')
    for operation in prepared.walk():
        for fields in (operation.attributes, operation.properties):
            fields.pop(marker, None)
            for key in tuple(fields):
                fields[key] = _remove_marker(fields[key], marker)
    prepared.verify()
    if _text(module) != source:
        raise ValueError('host region reuse mutated original source')
    transformed = _text(prepared)
    report.update(status='applied' if removed else 'unchanged',
                  reason=None if removed else 'no_exact_duplicate_preparations',
                  output_sha256=hashlib.sha256(transformed.encode()).hexdigest(),
                  removed_operations=counts,
                  original_pointer_abi=str(owner.function_type),
                  original_returned_writer_ordinals=sorted(returned_writers),
                  library_sha256=hashlib.sha256(Path(tensor_provenance_cse.__file__).read_bytes()).hexdigest())
    return HostRegionPreparation(prepared, aliases, retained_ordinals, aliases, report)
