"""Fact-backed K streams over the shared checked dense construction.

No elapsed-time assumption: local hazards and accumulator visibility are
selected ordering facts; finite hardware queues provide backpressure.  The
library proves source coverage, next-reader consumption and output-block scope.
"""
from hashlib import sha256
import json
from pathlib import Path

from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw, schedule_dense_panel_epochs


def selected_epoch_law(adapter):
    root = Path(__file__).resolve().parent
    facts_bytes = (root / 'rtl_facts/facts.json').read_bytes()
    selection_bytes = (root / 'command_ordering.json').read_bytes()
    selection = json.loads(selection_bytes)
    record = selection['ordering']
    digest = sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    if (selection['base_facts']['sha256'] != sha256(facts_bytes).hexdigest()
            or selection['ordering_sha256'] != digest or record['status'] != 'derived'):
        raise ValueError('dense epochs require selected ordering bound to current RTL facts')
    cross = {(r['older'], r['younger'], r['space']): r.get('overlap') == 'completion'
             for r in record['cross_queue']}
    loads = ('MVIN.spad_addr', 'MVIN2.spad_addr')
    readers = ('COMPUTE_PRELOADED.a_spad', 'PRELOAD.weight_spad')
    raw = all(cross.get((w, r, 'scratchpad'), False) for w in loads for r in readers)
    war = all(cross.get((r, w, 'scratchpad'), False) for w in loads for r in readers)
    waw = record['completion_order']['execute']['order'] == 'in_order'
    visible = cross.get(('PRELOAD.c_addr', 'MVOUT.acc_addr', 'accumulator'), False)
    queues = record['same_queue']
    codes = next(r['by_code'] for r in record['funct_queue'] if r['funct'] == 0)
    config = all(codes[str(i)] == q and queues[q]['release'] in ('issue', 'completion')
                 for i, q in ((0, 'execute'), (1, 'load'), (2, 'store')))
    classes = json.loads(facts_bytes)['facts']['timing_structure']['command_classes']['classes']
    progress = all(classes[q]['entries']['status'] == 'derived' and
                   classes[q]['entries']['value'] >= minimum
                   for q, minimum in (('load', 1), ('execute', 2), ('store', 1)))
    return DensePanelEpochLaw(adapter.configuration_identity, adapter.numerical_policy_identity,
        'selected-dense-ordering:' + sha256(selection_bytes).hexdigest(),
        local_read_after_write=raw, local_write_after_read=war,
        ordered_accumulator_writes=waw, accumulator_result_visibility=visible,
        ordered_configuration_capture=config, finite_queue_progress=progress,
        complete_device_drain=True, exclusive_private_state=True)


def apply_epochs(result, problem, block, resources, storage, adapter):
    source = getattr(result, 'checked', result)
    try:
        law = selected_epoch_law(adapter)
        return schedule_dense_panel_epochs(result, problem, block, resources, storage,
            adapter, source.admission, law, checked_source=source if source is not result else None,
            static_budget=128, timeout_ms=15000, max_read_instances=65536)
    except ValueError as exc:
        # Refused/UNKNOWN proof preserves the already checked schedule, including
        # every original drain.  Never rewrite a refused stream ourselves.
        import sys
        print('projection epochs fallback: ' + str(exc), file=sys.stderr)
        return result
