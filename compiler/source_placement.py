"""Source-bound static lane declarations; never fabricate executable commands or costs."""
from __future__ import annotations


def source_capabilities(module, operations):
    """Snapshot the shared standalone oracle on the original, unrewritten SSA."""
    from merlin.targetgen import eligibility as E, model_coverage as MC
    result=[{'status':'unknown','reason':'shared region descriptor does not cover this source operation',
             'family':None,'scope':'standalone source descriptor'} for _ in operations]
    positions={op:i for i,op in enumerate(operations)}
    descriptors=[(op,descriptor) for op,descriptor in zip(MC.region_ops(module),MC.regions_from_module(module),strict=True) if op in positions]
    for op,descriptor in descriptors:
        result[positions[op]].update(family=descriptor.resolved_family(),captured_input_formats=descriptor.captured_input_formats)
    try:
        import hashlib,json
        from pathlib import Path
        import yaml
        root=Path(__file__).with_name('data')
        manifest=json.loads((root/'source_placement_contract.json').read_text())
        data=(root/'source_placement_contract.yaml').read_bytes()
        if hashlib.sha256(data).hexdigest()!=manifest['packaged_sha256'] or manifest['packaged_sha256']!=manifest['source_sha256']:
            raise ValueError('selected placement contract hash differs')
        contract=yaml.safe_load(data)
        caps=E.capability_map_from_contract(contract)
        undetermined=E.undetermined_families_from_contract(contract)
        for op,descriptor in descriptors:
            verdict=E.is_eligible(descriptor,caps,undetermined=undetermined)
            result[positions[op]]={'status':('unknown' if verdict.undetermined else
                'eligible' if verdict.eligible else 'ineligible'),
                'family':verdict.family,'reason':verdict.reason,'refusal':verdict.refusal,
                'scope':'standalone source descriptor; selected exact lowering proof recorded separately',
                'captured_input_formats':descriptor.captured_input_formats,
                'contract_sha256':manifest['source_sha256']}
    except (OSError,ValueError,KeyError,TypeError) as error:
        # Missing capability evidence must not silently become hardware unsupported.
        for row in result:
            row.update(status='unknown',reason='shared capability analysis unavailable: '+str(error))
            if 'manifest' in locals():row['contract_sha256']=manifest.get('source_sha256')
    return result


def lane_placements(placements, receipts, capabilities, target_sites, *, source_sha256):
    """Join actual taken branches to owned schedule receipts / emitted target sites.

    A lane declaration is static emission evidence, not execution, coverage credit,
    numerical qualification, or a cycle/byte measurement. Storage-only source ops
    keep their own identity and never inherit accelerator credit from a consumer.
    """
    result=[]
    for index,source in enumerate(placements):
        assert source['source_op_index']==index
        capability=capabilities[index]
        events=source['lowerings']
        owned_receipts=[]
        for event in events:
            begin,end=event['schedule_receipt_range']
            if not 0<=begin<=end<=len(receipts):raise ValueError('invalid source receipt range')
            for j in range(begin,end):
                receipt=receipts[j]
                owners=receipt.get('source_op_indices',[receipt.get('source_op_index')])
                if index in owners and j not in owned_receipts:owned_receipts.append(j)
        device_events=[event for event in events if event['kind']=='device']
        sites=target_sites.get(index,[])
        device=bool(device_events and (owned_receipts or sites))
        kinds={event['kind'] for event in events}
        storage=bool(kinds and kinds<= {'storage'})
        preparation=bool(kinds and kinds<= {'preparation'})
        floats=any(boundary.get('dtype','').startswith(('f','bf'))
            for boundary in (*source['inputs'],*source['outputs']))
        if device:
            reason='selected exact source lowering emitted target work'
            gap=None
        elif device_events:
            reason='device branch selected but no owned schedule receipt or target emission joined'
            gap='emission_attribution_gap'
        elif storage:
            reason='source storage/view preparation; no separate computation emitted'
            gap=None
        elif preparation:
            reason='source constant preparation emitted through shared host lowering'
            gap=None
        elif capability['status']=='eligible':
            reason='shared host lowering selected for an eligible source region; exact device adapter/proof remains unresolved'
            gap='compiler_or_proof_gap'
        elif floats:
            reason='shared host lowering preserves typed floating source arithmetic/order; equivalent standalone device realization is not established'
            gap='numerical_or_device_proof_boundary'
        else:
            reason='shared host source lowering selected; see standalone eligibility refusal/unknown and source lowering events'
            gap='compiler_or_proof_gap' if capability['status']=='unknown' else 'source_capability_refusal'
        family=capability.get('family') or ('preparation' if source['kind'] in ('storage','preparation') else 'unknown')
        row={**source,'lane':'systolic_mesh' if device else 'host','family':family,
            'reason':reason,'remaining_gap':gap,'device_eligibility':capability,
            'selected_device_forms':[event['lowering'] for event in device_events],
            'schedule_receipt_indices':owned_receipts,'target_instruction_indices':sites,
            'selected_lowering_proof':'admitted_exact_device_branch' if device else None,
            'host_lowerings':[event['lowering'] for event in events if event['kind']=='host'],
            'source_sha256':source_sha256,'evidence':'static original-source / actual emission join',
            'measurements':{'host_time':None,'transferred_bytes':None,'cycles':None},
            'execution_status':'not_measured',
            'computation_status':'storage_only' if storage else 'preparation_only' if preparation else 'emitted' if events else 'unattributed'}
        result.append(row)
    return result
