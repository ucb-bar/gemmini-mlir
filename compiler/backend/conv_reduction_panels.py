"""Bounded one-K RHS stages with complete source outputs retained in ACC.

The shared dense capacity search owns M/N extents. Original affine convolution
runs own input addresses and halos; no im2col tensor or host payload work exists.
Only K residency changes: each source ky/kx/channel stage prepares one RHS slice
for every live N consumer, then reads every live M input before replacement.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from merlin.compile.tuning_knobs import knob
from . import isa


@dataclass(frozen=True)
class ReductionPanel:
    block: object
    rhs_base: int
    source_identity: str
    chunks: int
    epoch_pairs: int
    epoch: object
    geometry: dict


def plan(prepared,geometry=None):
    p=prepared.source.plan;w=p.window
    pixels=w.input_shape[0]*w.output_spatial[0]*w.output_spatial[1]
    ci=w.input_shape[-1];kh,kw=w.kernel;chunks=(ci+isa.DIM-1)//isa.DIM
    if geometry is None:
        geometry=prepared.attributes.get('conv_reduction_geometry',knob('CONV_REDUCTION_GEOMETRY',
            default='n_wide',values=('n_wide','balanced','m_tall'),root=Path(__file__).resolve().parents[1]))
    from .conv_reduction_geometry import plan_geometry
    block,geometry_receipt=plan_geometry(prepared,geometry)
    if block is None:return None
    # Preserve the incumbent N-wide admission exactly. Explicit joint-M/N
    # alternatives also explore M reuse when complete-K residency owns more N.
    if geometry=='n_wide'and block.n<=prepared.group.panel.column_blocks:return None
    rhs_base=block.m*isa.DIM
    if rhs_base+block.n*isa.DIM>isa.SPAD_ROWS:return None
    if block.m*block.n*isa.DIM>isa.ACC_ROWS:return None
    # A real completion after each original ky bounds issued work, independent
    # of finite queue capacity. Every individual ACC keeps original K order.
    epoch_pairs=kw*chunks*block.m*block.n
    if epoch_pairs>=1<<30:return None
    epoch=prepared.epoch
    if epoch_pairs>epoch.max_compute_pairs:
        # Widen only the compiler-issued source-work budget. ACC allocation
        # already proves disjoint private live outputs. Original dependency,
        # configuration capture and finite ready/valid pair progress premises
        # remain identical; pending queue capacity is never changed.
        epoch=replace(epoch,max_compute_pairs=max(2,1<<epoch_pairs.bit_length()),
            qualification_identity=epoch.qualification_identity+'-joint-mn-private-group')
    epoch.require(max(2,epoch_pairs))
    return ReductionPanel(block,rhs_base,prepared.source.identity,chunks,epoch_pairs,epoch,geometry_receipt)


def emit(prepared,stage,builder,lower,complete,*,hoist_gather=False):
    from .conv_spatial_panels import (
        _gather_coordinates,_gather_bands,_emit_gather_bands,_input_fragments,_predicate_and,
    )
    if stage.source_identity!=prepared.source.identity:raise ValueError('changed convolution source')
    stage.epoch.require(max(2,stage.epoch_pairs))
    p=prepared.source.plan;w=p.window;kh,kw=w.kernel;ci=w.input_shape[-1]
    pixels=w.input_shape[0]*w.output_spatial[0]*w.output_spatial[1]
    bm,bn=stage.block.m,stage.block.n
    def phase(column,widths):
        active=len(widths)
        def group(first,row_counts):
            outputs=tuple(tuple(dict(output_row=builder.op('add',first,m*isa.DIM),
                output_column=builder.op('add',column,n*isa.DIM),rows=rows,columns=columns,
                accumulator_row=(m*active+n)*isa.DIM)
                for n,columns in enumerate(widths))for m,rows in enumerate(row_counts))
            if p.bias:
                for row in outputs:
                    for out in row:lower('initialize_bias',out)
                complete('reduction_panel_initializer_ready',{})
            coordinates=(tuple(_gather_coordinates(prepared,builder,builder.op('add',first,m*isa.DIM),rows)
                for m,rows in enumerate(row_counts))if hoist_gather else None)
            def kernel_y(ky):
                def kernel_x(kx):
                    bands=(tuple(_gather_bands(prepared,builder,c,ky,kx)for c in coordinates)
                        if hoist_gather else None)
                    def channels(chunk,depth):
                        tap=builder.op('add',builder.op('mul',ky,kw),kx)
                        start=builder.op('add',builder.op('mul',tap,ci),builder.op('mul',chunk,isa.DIM))
                        for n,columns in enumerate(widths):
                            lower('load_weight',dict(reduction_row=start,
                                output_column=builder.op('add',column,n*isa.DIM),
                                local_row=stage.rhs_base+n*isa.DIM,rows=depth,columns=columns))
                        channel=builder.op('mul',chunk,isa.DIM)
                        initial=_predicate_and(builder,builder.cmp('eq',ky,0),builder.cmp('eq',kx,0))
                        initial=_predicate_and(builder,initial,builder.cmp('eq',chunk,0))
                        for m,rows in enumerate(row_counts):
                            def transfer(f,m=m):
                                lower('load_input',dict(f,local_row=builder.op('add',f['local_row'],m*isa.DIM)))
                            if bands is not None:
                                _emit_gather_bands(prepared,builder,bands[m],channel,depth,transfer)
                            else:
                                _input_fragments(prepared,builder,builder.op('add',first,m*isa.DIM),rows,
                                    ky,kx,channel,depth,transfer)
                            for n,out in enumerate(outputs[m]):
                                lower('compute',dict(out,lhs_row=m*isa.DIM,
                                    rhs_row=stage.rhs_base+n*isa.DIM,depth=depth,
                                    first_reduction=initial,bias=p.bias,fresh_weights=True))
                    full,tail=divmod(ci,isa.DIM)
                    if full:builder.loop(0,full,1,lambda c:channels(c,isa.DIM))
                    if tail:channels(full,tail)
                builder.loop(0,kw,1,kernel_x)
                complete('reduction_panel_kernel_y_last_use',{})
            builder.loop(0,kh,1,kernel_y)
            for row in outputs:
                for out in row:lower('store',out)
            complete('reduction_panel_output_last_use',{})
        full,remaining=divmod(pixels,bm*isa.DIM)
        if full:builder.loop(0,full,1,lambda m:group(builder.op('mul',m,bm*isa.DIM),(isa.DIM,)*bm))
        if remaining:
            rows,tail=divmod(remaining,isa.DIM)
            group(full*bm*isa.DIM,(isa.DIM,)*rows+((tail,)if tail else ()))
    columns,tail=divmod(p.output_channels,isa.DIM)
    phases,rest=divmod(columns,bn)
    if phases:builder.loop(0,phases,1,lambda n:phase(builder.op('mul',n,bn*isa.DIM),(isa.DIM,)*bn))
    widths=(isa.DIM,)*rest+((tail,)if tail else ())
    if widths:phase(phases*bn*isa.DIM,widths)
    return dict(mechanism='convolution_one_k_rhs_stages',source_sha256=stage.source_identity,
        capacity_block=asdict(stage.block),rhs_base=stage.rhs_base,
        source_channel_tiles=stage.chunks,gather_hoisted=hoist_gather,
        completion='original kernel-Y stage boundary; common readout; output group last-use',
        max_actual_compute_pairs_per_epoch=stage.epoch_pairs,
        compute_epoch=asdict(stage.epoch),reduction_geometry=stage.geometry,
        reduction_order=['kernel_y','kernel_x','input_channel'],
        physical_mapping_verified=False,target_execution=False,timing=None)
