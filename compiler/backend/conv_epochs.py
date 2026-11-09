"""Source-bound admissions for synchronous-boundary convolution epochs.

The packaged record is the selected public ReservationStation ordering result.
No capacity is inferred from an issue budget. Finite queues backpressure the
stream, and the separately selected in-order descriptor/data lane preserves
pair progress. All original store and final ownership drains remain real.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def epoch_admissions(source, hardware, numeric, qualification, *, complete_reduction=False):
    from merlin.compiler.schedule.window_transfer import (
        ConvolutionLoadIssueLaw, ConvolutionComputeEpochLaw,
    )
    record = json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    ordering = record['ordering']
    actual = hashlib.sha256(json.dumps(ordering, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (actual != 'ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780' or ordering.get('status') != 'derived'
            or record['base_facts_sha256'] != 'b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af'):
        raise ValueError('convolution epoch ordering does not match selected public facts')
    evidence = {}
    cross = {(r['older'], r['younger'], r['space']): r for r in ordering['cross_queue']}
    def waits(older, younger, space):
        row = cross.get((older, younger, space), {})
        if row.get('overlap') != 'completion':
            raise ValueError('convolution epoch lacks completion wait: '+repr((older, younger, space)))
        # A disjoint completion wait with tracked=None is conservative: this
        # queue waits even without an address-overlap discrimination. It never
        # authorizes removing an actual overlap wait.
        evidence[' -> '.join((older, younger, space))] = row
    loads = ['MVIN.spad_addr', 'MVIN2.spad_addr', 'MVIN3.spad_addr']
    readers = ['COMPUTE_PRELOADED.a_spad', 'PRELOAD.weight_spad']
    for load in loads:
        for reader in readers:
            waits(load, reader, 'scratchpad')
            waits(reader, load, 'scratchpad')
        waits(load, 'PRELOAD.c_addr', 'accumulator')
    waits('PRELOAD.c_addr', 'MVOUT.acc_addr', 'accumulator')
    waits('MVOUT.acc_addr', 'PRELOAD.c_addr', 'accumulator')
    same = ordering['same_queue']
    if (same['execute']['release'] not in ('issue','completion')
            or same['load']['release'] not in ('issue','completion')
            or ordering['completion_order']['execute']['order'] != 'in_order'):
        raise ValueError('convolution epochs require ordered configuration and descriptor/compute issue')
    queues = {}
    for row in ordering['funct_queue']:
        if 'by_code' in row:
            queues.update({('config',int(k)):v for k,v in row['by_code'].items()})
        elif 'funct' in row:
            queues[row['funct']] = row['queue']
    if any(queues.get(f) != 'execute' for f in (4,6)) or queues.get(('config',0)) != 'execute' or queues.get(('config',1)) != 'load':
        raise ValueError('convolution epoch configuration/pairs enter different lanes')
    entries = record['finite_entries']
    for lane, minimum in [('load',1),('execute',2)]:
        entry = entries[lane]
        if entry.get('status') != 'derived' or type(entry.get('value')) is not int or entry['value'] < minimum:
            raise ValueError('convolution epochs lack finite queue pair progress')
    pending_pairs = entries['execute']['value']//2
    # Two complete original N traversals per epoch, independent of K/spatial
    # size. The issued bound may exceed pending pairs: ready/valid backpressure
    # is separately admitted by the selected finite queues and lane order.
    p = source.plan
    if type(complete_reduction) is not bool:
        raise ValueError('convolution reduction epoch choice must be Boolean')
    pairs = max(2, min(64, 2*len(source.column_intervals)))
    reduction_pairs = None
    if complete_reduction:
        # The shared planner gives each live N tile exactly row_tile ACC rows.
        # This bounds issued work by the source reduction and the actual private
        # ACC allocation, independently of finite queue *pending* capacity.
        live_columns = min(len(source.column_intervals),
            (p.accumulator_rows-p.accumulator_row)//p.row_tile)
        chunks = (p.window.input_shape[-1]+p.channel_tile-1)//p.channel_tile
        reduction_pairs = p.window.kernel[0]*p.window.kernel[1]*chunks*live_columns
        # No modulo boundary falls within this complete reduction. The original
        # terminal readout's before-config full drain remains, as do all stores,
        # preparation and final drains. No ownership crosses a column block.
        # A power-of-two bound also keeps the shared emitter's runtime modulo
        # an inexpensive mask. It is an issued-work limit, not queue capacity.
        pairs = max(2, min(2**31-1, 1 << reduction_pairs.bit_length()))
    ids = (hardware, numeric, source.identity, qualification)
    issue = ConvolutionLoadIssueLaw(*ids,
        load_issue_budget=p.row_tile+1+len(source.column_intervals)*int(p.bias),
        local_range_dependencies=True, ordered_load_configuration=True,
        paired_descriptor_compute=True, finite_queue_backpressure=True,
        complete_device_drain=True, immutable_mapped_disjoint_buffers=True)
    epoch = ConvolutionComputeEpochLaw(*ids, max_compute_pairs=pairs,
        local_read_after_write=True, local_write_after_read=True,
        local_write_after_write=True, accumulator_result_visibility=True,
        operand_consumption_retirement=True, ordered_configuration_capture=True,
        adjacent_descriptor_compute=True, finite_queue_pair_progress=True,
        complete_device_drain=True, exclusive_private_state=True,
        pending_compute_pairs=pending_pairs)
    receipt = dict(ordering_sha256=actual, dependency_waits=len(evidence),
        max_issued_compute_pairs=pairs, pending_compute_pairs=pending_pairs,
        completion='real epoch/store/preparation/final drains',
        load_rewrite='only after its consuming compute; no zero/data overwrite')
    if complete_reduction:
        receipt.update(epoch_scope='one complete original reduction per private column block',
            source_compute_pairs_per_block=reduction_pairs,
            no_intermediate_compute_boundary=pairs>reduction_pairs,
            live_columns_from_acc_capacity=live_columns,
            ownership_boundary='original before-config/readout/store full drains retained')
    return issue, epoch, pairs, receipt
