"""Two bounded private spatial windows with actual completion before reuse.

The original selected dispatch overlap waits provide producer visibility and
protect conflicting accesses. No command configuration snapshot is assumed:
CONFIG_ST stays invariant from the initial full drain to the final full drain.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from merlin.compile.scheduling.convolution_output_groups import plan_convolution_output_group
from .conv_readout_epochs import prepare_uniform_readout


@dataclass(frozen=True)
class SpatialPipeline:
    group: object
    readout: object
    ordering_sha256: str

    def record(self):
        return dict(mechanism='two_spatial_slots',slots=2,
            lhs_rows=[self.group.lhs_row(i) for i in range(2)],
            accumulator_rows_per_slot=self.group.panel.accumulator_rows,
            ordering_sha256=self.ordering_sha256,
            completion='full drain before every two-slot reuse and phase end',
            readiness='selected tracked accumulator RAW completion wait',
            configuration='invariant CONFIG_ST for entire invocation',
            reduction_order=['kernel_y','kernel_x','input_channel'])


def prepare_spatial_pipeline(panel,pixels,dtype,pitch,attrs):
    """Reopen exact selected dependencies and shared capacity-checked mapping."""
    group=plan_convolution_output_group(panel,pixels,max_row_blocks=2)
    if group.row_blocks != 2:
        raise ValueError('two spatial slots do not fit private LHS/ACC capacity')
    record=json.loads(Path(__file__).with_name('conv_ordering.json').read_text())
    ordering=record['ordering']
    digest=hashlib.sha256(json.dumps(ordering,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!='ce5fade2537a00f0fca7a56a3e80b19391da5b35c87621dd878773195a540780':
        raise ValueError('spatial pipeline ordering source changed')
    cross={(r['older'],r['younger'],r['space']):r for r in ordering['cross_queue']}
    for older,younger in [('PRELOAD.c_addr','MVOUT.acc_addr'),('MVOUT.acc_addr','PRELOAD.c_addr')]:
        r=cross.get((older,younger,'accumulator'),{})
        if r.get('tracked') is not True or r.get('overlap')!='completion':
            raise ValueError('spatial pipeline lacks tracked ACC producer/consumer completion')
    # Initialization uses the actual original MVIN3 ACC address and execute
    # consumers; preserve this ordering even when no bias is needed.
    r=cross.get(('MVIN3.spad_addr','PRELOAD.c_addr','accumulator'),{})
    if r.get('tracked') is not True or r.get('overlap')!='completion':
        raise ValueError('spatial pipeline lacks initializer visibility')
    if ordering['completion_order']['execute']['order']!='in_order':
        raise ValueError('spatial pipeline lacks ordered original reductions')
    readout=prepare_uniform_readout(dtype,pitch,attrs,group)
    return SpatialPipeline(group,readout,digest)
