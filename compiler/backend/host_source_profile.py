"""Source-bound host graph diagnostics; static work is never runtime timing.

The ordinary Merlin parser and linalg iteration-domain query own the typed
source interpretation. Unknown domains and dynamic invocation counts stay
unknown. This module emits no arithmetic and makes no placement decision.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from math import prod
from pathlib import Path


def profile_source(source_text: str, *, placement: dict | None = None) -> dict:
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.op_profile import classify_generic_body
    from xdsl.dialects.builtin import ArrayAttr, StringAttr, TensorType

    source_sha256 = hashlib.sha256(source_text.encode()).hexdigest()
    if placement is not None and placement.get('source_sha256') != source_sha256:
        raise ValueError('placement and profile must describe the exact same source bytes')
    membership = {}
    for region in (placement or {}).get('device_regions', []):
        for ordinal in region['source_ordinals']:
            membership.setdefault(ordinal, []).append(region['symbol'])
    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops if op.name == 'func.func' and op.body.blocks]
    candidates = [op for op in functions if op.sym_name.data == 'forward'] or functions
    if len(candidates) != 1:
        raise ValueError('source profile needs one unambiguous public tensor entry')
    entry = candidates[0]
    ordinals = {op: i for i, op in enumerate(module.walk())}

    def strings(value):
        if isinstance(value, StringAttr):
            return [value.data]
        if isinstance(value, ArrayAttr):
            return [item.data for item in value if isinstance(item, StringAttr)]
        return []

    def metadata(operation, name):
        return strings(operation.attributes.get('prov.' + name))

    rows = []
    for index, operation in enumerate(entry.body.block.ops):
        if operation.name == 'func.return':
            continue
        body = Counter(op.name for op in operation.walk() if op is not operation
                       and op.name.startswith(('arith.', 'math.')))
        domain, scalar_evaluations, domain_reason = None, None, None
        if operation.name == 'linalg.generic':
            try:
                domain = list(operation.get_static_loop_ranges())
                if any(d < 0 for d in domain):
                    raise ValueError('dynamic iteration extent')
                scalar_evaluations = {name: count * prod(domain) for name, count in body.items()}
            except (ValueError, NotImplementedError) as exc:
                domain_reason = str(exc)
        elif body:
            domain_reason = 'no exact generic iteration-domain query for this operation'
        family = metadata(operation, 'family')
        inferred = classify_generic_body(sorted(body)) if operation.name == 'linalg.generic' else None
        rows.append(dict(source_index=index, source_ordinal=ordinals[operation], mlir_op=operation.name,
            source_node_ids=metadata(operation, 'source_node_ids'),
            origin_node_ids=metadata(operation, 'origin_node_ids'),
            region_ids=metadata(operation, 'region_id'),
            family=family[0] if family else inferred[0] if inferred else 'unresolved',
            family_basis='source provenance' if family else 'Merlin scalar-body classifier' if inferred else 'unknown',
            inputs=[str(v.type) for v in operation.operands], outputs=[str(v.type) for v in operation.results],
            output_elements=[prod(v.type.get_shape()) if all(d >= 0 for d in v.type.get_shape()) else None
                             for v in operation.results if isinstance(v.type, TensorType)],
            input_producers=[ordinals.get(v.owner) for v in operation.operands],
            selected_device_regions=membership.get(ordinals[operation], []) if placement is not None else None,
            placement_status=('selected device region member' if ordinals[operation] in membership
                              else 'remaining source; eligibility unresolved') if placement is not None else 'UNKNOWN',
            scalar_body_sites=dict(body), iteration_domain=domain,
            scalar_evaluations_per_source_operation=scalar_evaluations,
            unknown_iteration_domain_reason=domain_reason,
            dynamic_invocations=None, measured_host_cycles=None))
    totals = Counter()
    for row in rows:
        totals.update(row['scalar_evaluations_per_source_operation'] or {})
    return dict(schema='submission.host_source_profile.v1',
        source_sha256=source_sha256, entry=entry.sym_name.data,
        source_operation_count=len(rows), families=dict(Counter(row['family'] for row in rows)),
        family_count_scope='all top-level source operations, including initializers carrying family provenance',
        generic_family_counts=dict(Counter(row['family'] for row in rows if row['mlir_op'] == 'linalg.generic')),
        operation_kinds=dict(Counter(row['mlir_op'] for row in rows)),
        scalar_evaluations_per_source_operation_sum=dict(totals), operations=rows,
        dynamic_invocation_coverage=dict(status='UNKNOWN', observed_operations=None,
            source_operations=len(rows), reason='no source-bound execution hit table supplied'),
        static_boundary_bytes=(placement or {}).get('static_boundary_bytes'),
        scope='static typed source graph and literal generic iteration domains; no runtime cost or compiled instruction count')


def main():
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--placement', type=Path)
    args = parser.parse_args()
    placement = json.loads(args.placement.read_text()) if args.placement else None
    args.out.write_text(json.dumps(profile_source(args.source.read_text(), placement=placement), indent=2))


if __name__ == '__main__':
    main()
