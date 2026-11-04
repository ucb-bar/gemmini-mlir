"""Source-bound readout completion for the existing spatial panel consumer.

Only one actual M tile/phase is live. Original gather and RHS preparation are
unchanged; the real output-last-use fence retires every readout before private
LHS/ACC/RHS reuse. No pending capacity is inferred from an issued bound.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from . import conv_epochs

@dataclass(frozen=True)
class SpatialReadoutEpoch:
    source_identity: str
    group_identity: str
    law: object
    evidence: dict

    def require(self,prepared,rows,widths):
        from merlin.compile.scheduling.window_transfer import ConvolutionOutputTile
        prepared.group.__post_init__()
        if self.source_identity!=prepared.source.identity or self.group_identity!=repr(prepared.group):
            raise ValueError('spatial readout changed its checked source or storage group')
        # The unchanged phase loops embed these relative original N intervals
        # at phase*panel.column_blocks*DIM. ACC cells use the same checked map.
        tiles=tuple(ConvolutionOutputTile(n*prepared.group.panel.tile,columns,
            prepared.group.accumulator_row(0,n,active_column_blocks=len(widths)))
            for n,columns in enumerate(widths))
        self.law.require(prepared.source.plan,rows,tiles)

    def receipt(self):
        return dict(law=asdict(self.law),admission=self.evidence,
            admission_sha256=hashlib.sha256(json.dumps(self.evidence,sort_keys=True).encode()).hexdigest(),
            completion='real spatial_output_last_use before private LHS/ACC/RHS reuse')


def admit(prepared):
    from merlin.compile.scheduling.window_transfer import ConvolutionReadoutEpochLaw
    p=prepared.source.plan;group=prepared.group
    # The shared law checks primitive M rows against the original counted plan.
    # Narrow source row plans have a different row bound, so keep them intact.
    if p.row_tile!=p.primitive or group.row_blocks!=1:return None
    prepared.gather.__post_init__();group.__post_init__()
    issue,epoch,_,report=conv_epochs.epoch_admissions(prepared.source,
        prepared.epoch.hardware_identity,prepared.epoch.numerical_policy_identity,
        prepared.epoch.qualification_identity,complete_reduction=True)
    if epoch!=prepared.epoch:raise ValueError('spatial readout changed selected compute ordering')
    record=json.loads(Path(conv_epochs.__file__).with_name('conv_ordering.json').read_text())
    ordering=record['ordering']
    cross={(r['older'],r['younger'],r['space']):r for r in ordering['cross_queue']}
    raw=cross['PRELOAD.c_addr','MVOUT.acc_addr','accumulator']
    war=cross['MVOUT.acc_addr','PRELOAD.c_addr','accumulator']
    if raw['overlap']!='completion' or war['overlap']!='completion':
        raise ValueError('spatial readout lacks exact ACC RAW/result visibility or WAR retirement')
    configs=next(r['by_code']for r in ordering['funct_queue']if 'by_code'in r)
    if (configs.get('2')!='store'
            or next(r['queue']for r in ordering['funct_queue']if r.get('funct')==3)!='store'
            or ordering['same_queue']['store']['release']!='completion'
            or ordering['completion_order']['store']['order']!='in_order'):
        raise ValueError('spatial readout lacks captured ordered store configuration')
    pending=record['finite_entries']['store']
    if pending['status']!='derived' or type(pending['value'])is not int or pending['value']<1:
        raise ValueError('spatial readout lacks finite store queue progress')
    law=ConvolutionReadoutEpochLaw(epoch.hardware_identity,epoch.numerical_policy_identity,
        prepared.source.identity,'original-spatial-M-tile-and-N-phase-store-epoch-v1',
        max_issued_readouts=group.panel.column_blocks,pending_readouts=pending['value'],
        local_read_after_write=True,accumulator_result_visibility=True,
        ordered_configuration_capture=True,finite_queue_progress=True,
        complete_device_drain=True,exclusive_private_state=True)
    evidence=dict(ordering_sha256=report['ordering_sha256'],producer_to_store=raw,
        store_to_next_producer=war,store_queue_release='completion',
        configuration_and_readout_queue='store',pending_readouts=pending['value'],
        private_group=repr(group),unchanged_gather=repr(prepared.gather),
        retirement='real output-last-use drain before any input/ACC replacement or RHS phase change')
    result=SpatialReadoutEpoch(prepared.source.identity,repr(group),law,evidence)
    # Revalidate every admitted phase/tail width; the actual consumer also
    # reopens its precise active widths and M extent before suppressing drains.
    full,tail=divmod(p.output_channels,p.primitive)
    phase,rest=divmod(full,group.panel.column_blocks)
    if phase:result.require(prepared,p.primitive,(p.primitive,)*group.panel.column_blocks)
    widths=(p.primitive,)*rest+((tail,)if tail else())
    if widths:result.require(prepared,p.primitive,widths)
    return result
