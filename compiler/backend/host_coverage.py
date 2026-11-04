"""Join typed source operations to original native source-demand evidence.

The native semantic certificate owns eligibility. This adapter never upgrades
an absent, ambiguous or unrecognized capability proof to hardware unsupported.
Source operation coverage and dynamic invocation coverage have separate scopes.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


def source_coverage(profile: dict, receipt: dict | None = None) -> dict:
    if receipt is None:
        return dict(schema='submission.source_region_coverage.v1',
            source_sha256=profile['source_sha256'],
            source_operation_count=profile['source_operation_count'],
            counts=dict(eligible_covered=0, eligible_host=0, hardware_unsupported=0,
                        numerically_invalid=0, proof_unresolved=profile['source_operation_count']),
            required_source_device_operations=None, completed_required_source_device_operations=None,
            native_certificate_region_count=None, native_denominator_completeness=None,
            static_boundary_bytes=profile.get('static_boundary_bytes'),
            measured_host_cycles=None, measured_transferred_bytes=None,
            dynamic_invocation_coverage=dict(status='UNKNOWN', reason='no native source-demand receipt supplied'),
            scope='all typed top-level source operations; native eligibility unavailable',
            operations=[dict(op, status='proof_unresolved',
                             reason='no native source-demand receipt supplied',
                             native_certificate_region_indices=[], native_evidence=[])
                        for op in profile['operations']])
    placement = receipt['source_placement']
    certificate = placement['certificate']
    source_sha = profile['source_sha256']
    for report, key in ((placement, 'source_sha256'),
                        (certificate, 'source_mlir_sha256')):
        if report.get(key) != source_sha:
            raise ValueError('coverage evidence must bind the exact original source')
    if placement.get('n_source_operations') != profile['source_operation_count']:
        raise ValueError('native and typed source operation denominators differ')
    demand = receipt.get('candidate_source_coverage', {})
    if demand and demand.get('source_sha256') != source_sha:
        raise ValueError('completed source demand belongs to different source bytes')
    eligible = set(placement.get('eligible_source_op_indices', []))
    completed = (set(demand.get('completed_eligible_source_op_indices', []))
                 if demand.get('status') == 'verified' else set())
    regions = certificate.get('regions', [])
    rows = []
    for op in profile['operations']:
        matches = [i for i, region in enumerate(regions)
                   if region.get('region_id') in op['region_ids']
                   and region.get('carrier_op') == op['mlir_op']]
        evidence = [regions[i] for i in matches]
        index = op['source_index']
        status, reason = 'proof_unresolved', 'no unique source-bound native capability proof'
        if index in eligible:
            status = ('eligible_covered' if index in completed else 'eligible_host'
                      if len(evidence) == 1 and evidence[0].get('decision') == 'cpu_fallback'
                      else 'proof_unresolved')
            reason = ('mandatory source device demand completed' if index in completed
                      else 'eligible source device demand has no verified completion')
        elif len(evidence) == 1:
            region = evidence[0]
            reason = region.get('eligibility_reason') or reason
            if region.get('target_eligible'):
                status = ('eligible_host' if region.get('decision') == 'cpu_fallback'
                          else 'proof_unresolved')
            elif region.get('eligibility_undetermined'):
                status = 'proof_unresolved'
            elif region.get('numerically_invalid') is True:
                status = 'numerically_invalid'
            elif (not region.get('target_eligible') and
                  any(token in reason for token in ('not in ', 'unsupported dtype',
                                                    'unsupported format'))):
                status = 'hardware_unsupported'
            # An unrecognized semantic family is a missing proof, not a hardware limit.
        rows.append(dict(source_index=index, source_ordinal=op['source_ordinal'],
            source_node_ids=op['source_node_ids'], origin_node_ids=op['origin_node_ids'],
            region_ids=op['region_ids'], mlir_op=op['mlir_op'], family=op['family'],
            inputs=op['inputs'], outputs=op['outputs'], status=status, reason=reason,
            native_certificate_region_indices=matches,
            native_evidence=evidence,
            selected_device_regions=op.get('selected_device_regions'),
            dynamic_invocations=None, measured_host_cycles=None))
    counts = dict.fromkeys(('eligible_covered', 'eligible_host', 'hardware_unsupported',
                           'numerically_invalid', 'proof_unresolved'), 0)
    counts.update(Counter(row['status'] for row in rows))
    return dict(schema='submission.source_region_coverage.v1', source_sha256=source_sha,
        source_operation_count=len(rows), counts=counts,
        required_source_device_operations=len(eligible),
        completed_required_source_device_operations=len(eligible & completed),
        native_certificate_region_count=len(regions),
        native_denominator_completeness=certificate.get('denominator_completeness'),
        completed_dispatch_elf=receipt.get('completed_dispatch', {}).get('linked_elf'),
        static_boundary_bytes=profile.get('static_boundary_bytes'),
        measured_host_cycles=None, measured_transferred_bytes=None,
        dynamic_invocation_coverage=dict(status='UNKNOWN',
            reason='completed mandatory device demands do not establish all host node invocation counts'),
        scope='all typed top-level source operations; unresolved rows remain in the denominator',
        operations=rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('receipts', type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = source_coverage(json.loads(args.profile.read_text()),
                             json.loads(args.receipts.read_text())[args.case])
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('source_sha256', 'counts',
        'required_source_device_operations', 'completed_required_source_device_operations')}))


if __name__ == '__main__':
    main()
