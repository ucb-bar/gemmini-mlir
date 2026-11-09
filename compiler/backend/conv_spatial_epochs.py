"""Bounded disjoint-ACC M groups under the selected convolution compute law.

This extends compute ownership only. No asynchronous store-config snapshot or
cross-readout reuse is assumed; every original terminal store fence remains.
"""
from dataclasses import asdict,dataclass,replace
from pathlib import Path
from merlin.compiler.selection.knobs import knob
from merlin.compiler.schedule.convolution_output_groups import plan_convolution_output_group

@dataclass(frozen=True)
class SpatialComputeGroup:
    group: object
    epoch: object
    realization: dict


def plan_spatial_compute_group(prepared, *, max_row_blocks=None):
    # This is a compiler issue/unroll bound, never a hardware queue depth.
    # The shared planner independently limits it by actual SP and ACC capacity.
    if max_row_blocks is None:
        max_row_blocks=knob('CONV_SPATIAL_GROUP_ROWS',default=2,values=(2,4),
            root=Path(__file__).resolve().parent.parent)
    if type(max_row_blocks) is not int or max_row_blocks not in (2,4):
        raise ValueError('spatial compute group supports original two or four row bound')
    source=prepared.source;p=source.plan
    batch,_,_,_=p.window.input_shape;oh,ow=p.window.output_spatial
    pixels=batch*oh*ow
    group=plan_convolution_output_group(prepared.group.panel,pixels,max_row_blocks=max_row_blocks)
    if group.row_blocks<2:return None
    panel=group.panel
    # Every actual full/tail phase gets the same explicit disjoint ACC proof.
    # Source N rows are immutable and output cells have no intermediate observer
    # inside the original operation; only input/control addresses move on host.
    checked=0
    for active in range(1,panel.column_blocks+1):
        spans=[]
        for row in range(group.row_blocks):
            for col in range(active):
                start=group.accumulator_row(row,col,active_column_blocks=active)
                end=start+panel.tile
                if end>panel.accumulator_capacity_rows or any(start<hi and lo<end for lo,hi in spans):
                    raise ValueError('spatial compute group overlaps a live ACC tile')
                spans.append((start,end));checked+=1
    pairs=panel.reduction_tiles*panel.column_blocks*group.row_blocks
    # The original selected law establishes RAW/WAR, ordered execute, and finite
    # queue progress independently of this issued work budget. Preserve its
    # pending-pair count; no queue capacity or additional hardware is inferred.
    epoch=replace(prepared.epoch,max_compute_pairs=max(2,1<<pairs.bit_length()))
    epoch.require(max(2,pairs))
    return SpatialComputeGroup(group,epoch,dict(
        mechanism='shared_convolution_output_group_compute_epoch',row_blocks=group.row_blocks,
        requested_max_row_blocks=max_row_blocks,
        shared_output_group=asdict(group),selected_compute_law=asdict(epoch),
        source_sha256=source.identity,panel_rows=panel.rhs_rows,rhs_base=panel.rhs_base,
        checked_disjoint_acc_intervals=checked,max_issued_compute_pairs=pairs,
        pending_compute_pairs=epoch.pending_compute_pairs,
        ordering=prepared.ordering,
        input_ownership='original LHS reuse after original consuming compute via selected RAW/WAR waits',
        accumulator_ownership='disjoint group M/N cells live until real group completion and original stores',
        completion='entry/preparation/bias/group-compute/per-store/output-last-use/phase/final drains retained',
        readout='all original before-config and after-store full drains; no asynchronous config law assumed'))
