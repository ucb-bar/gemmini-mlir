"""Bounded two-M completion groups with original fresh weight consumers.

No array-weight retention or rolling transport is introduced. Separate LHS and
ACC cells come from ConvolutionOutputGroup; source K order is unchanged. The
existing selected load/execute/store dependencies and finite queues apply to
both members, with one real ownership drain at group end.
"""
from dataclasses import asdict, dataclass, replace
from .conv_spatial_readout_epochs import admit as admit_single
from merlin.compile.scheduling.convolution_output_groups import plan_convolution_output_group
from merlin.compile.scheduling.window_transfer import (
    ConvolutionInputRetentionLaw, ConvolutionOutputTile,
)


@dataclass(frozen=True)
class GroupedSpatialReadoutEpoch:
    source_identity: str
    group_identity: str
    law: object
    independence: object
    evidence: dict
    consumer: object

    def require(self, prepared, rows, widths, *, row_block):
        if (prepared.source.identity != self.source_identity
                or repr(prepared.group) != self.group_identity):
            raise ValueError('spatial M group changed its original source or storage')
        prepared.group.__post_init__()
        self.independence.__post_init__()
        if not all((self.independence.independent_outputs,
                    self.independence.independent_output_rows,
                    self.independence.no_intermediate_observers,
                    self.independence.immutable_mapped_disjoint_buffers)):
            raise ValueError('spatial M group lacks independent source output ownership')
        tiles=tuple(ConvolutionOutputTile(n*prepared.group.panel.tile,columns,
            prepared.group.accumulator_row(row_block,n,active_column_blocks=len(widths)))
            for n,columns in enumerate(widths))
        self.law.require(prepared.source.plan,rows,tiles)

    def require_group(self, prepared, counts, widths):
        self.consumer.require(counts,widths)

    def member_issue(self,builder,issue,*,row_block,rows,widths):
        return self.consumer.member_issue(builder,issue,row_block=row_block,rows=rows,widths=widths)

    def emit_group(self,*args,**kwargs):
        return self.consumer.emit_group(*args,**kwargs)

    def receipt(self):
        return dict(law=asdict(self.law),source_independence=asdict(self.independence),
            composition='checked two-member actual-effect rebasing; original member issued and pending limits unchanged',
            admission=self.evidence,
            outer_source_bound=dict(
                max_issued_readouts=self.consumer.group.row_blocks*self.consumer.group.panel.column_blocks,
                max_compute_pairs=self.consumer.group.row_blocks*self.consumer.group.panel.column_blocks*self.consumer.group.panel.reduction_tiles,
                qualification='maximum full-phase source work; actual active widths/tails rechecked by consumer; not new target queue capacity'),
            completion='one real spatial_output_last_use closes all group LHS/ACC rows before reuse')


def admit(prepared):
    # Reuse every original selected RAW/WAR/configuration/progress check. The
    # ordinary consumer still declines narrow original row partitions.
    single=admit_single(prepared)
    if single is None:return None
    p=prepared.source.plan;panel=prepared.group.panel
    pixels=p.window.input_shape[0]*p.window.output_spatial[0]*p.window.output_spatial[1]
    group=plan_convolution_output_group(panel,pixels,max_row_blocks=2)
    if group.row_blocks!=2:return None
    # These are source permissions, not hardware assertions: the already
    # admitted original operation is a pure convolution into disjoint storage.
    # Its scalar reductions never read output/ACC cells from another M row;
    # there is no intermediate observation inside this invocation's producer.
    ids=(prepared.epoch.hardware_identity,prepared.epoch.numerical_policy_identity,
         prepared.source.identity,'original-pure-convolution-disjoint-M-group-v1')
    independence=ConvolutionInputRetentionLaw(*ids,max_live_tiles=group.row_blocks*panel.column_blocks,
        input_preserved=True,independent_outputs=True,immutable_mapped_disjoint_buffers=True,
        independent_output_rows=True,no_intermediate_observers=True)
    independence.__post_init__()
    # Retain each original member's issued and pending bounds verbatim. The
    # separate checked consumer composes their actual disjoint mapped effects.
    from merlin.compile.scheduling.convolution_disjoint_m_groups import DisjointConvolutionMGroup
    consumer=DisjointConvolutionMGroup(prepared.source,prepared.gather,group,
        single.law,prepared.epoch,independence)
    grouped=replace(prepared,group=group)
    law=single.law
    evidence=dict(single.evidence,private_group=repr(group),
        original_single_group=repr(prepared.group),
        source_permissions='pure original convolution; each M/N cell has its own source-ordered reduction; no observer before original invocation completion',
        no_array_retention='every product still PRELOADs its resident RHS descriptor fresh',
        retirement='one real group-end drain before either LHS/ACC row block or RHS phase is reused')
    result=GroupedSpatialReadoutEpoch(prepared.source.identity,repr(group),law,independence,evidence,consumer)
    full,tail=divmod(p.output_channels,p.primitive)
    phases,rest=divmod(full,panel.column_blocks)
    if phases:result.require_group(grouped,(p.primitive,)*group.row_blocks,(p.primitive,)*panel.column_blocks)
    widths=(p.primitive,)*rest+((tail,)if tail else())
    if widths:result.require_group(grouped,(p.primitive,)*group.row_blocks,widths)
    return grouped,result
