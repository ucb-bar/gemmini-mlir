"""Resident horizontal reduction panels for a private pooled producer.

The shared planner owns the complete immutable flattened RHS and disjoint LHS
slots. Every DMA remains primitive width. Horizontal halo tiles retain the
canonical original-tap path; no padded host image or payload arithmetic exists.
"""
from dataclasses import dataclass, replace
from merlin.compile.scheduling.horizontal_convolution import plan_horizontal_convolution_panels
from merlin.compile.scheduling.convolution_exploration import ConvolutionExplorationSource
from .conv_epochs import epoch_admissions
from . import isa


@dataclass(frozen=True)
class Pipeline:
    block: object
    panels: object
    receipt: dict


def prepare(block):
    p = block.source.producer
    w = p.window
    ci = w.input_shape[-1]
    kh, kw = w.kernel
    if (ci >= isa.DIM or w.dilation[1] != 1 or p.row_tile != isa.DIM
            or kw <= 1 or (kw*ci+isa.DIM-1)//isa.DIM >= kw):
        return None
    # The complete immutable horizontal RHS must stay below the separate
    # original-tap fallback RHS. Both share only LHS rows after a real drain.
    source = replace(p, operand_rows=p.rhs_row, row_tile=isa.DIM,
                     rhs_row=isa.DIM)
    panels = plan_horizontal_convolution_panels(source, max_row_blocks=1,
        maximum_transfer_columns=4*isa.DIM, private_rows_owned=True)
    if panels is None or panels.reduction_tiles > 32 or len(panels.column_intervals)>8:
        return None
    _, epoch, _, report = epoch_admissions(
        ConvolutionExplorationSource(p, panels.column_intervals, p.identity),
        'gemmini-fire-aligned-b8ff28edc096392c',
        'signed-i8-products-i32-exact-spatial-readout',
        'private-pooled-horizontal-panels-v1', complete_reduction=True)
    epoch.require(max(2,panels.reduction_tiles))
    return Pipeline(block,panels,dict(
        mechanism='shared_horizontal_panels_private_pool',
        source_sha256=p.identity, panel_plan=panels.summary(),
        lhs_rows=panels.lhs_rows, rhs_rows=panels.rhs_rows,
        operand_rows=panels.operand_rows, original_fallback_rhs_base=p.rhs_row,
        original_reduction_tiles=kh*kw*((ci+isa.DIM-1)//isa.DIM),
        horizontal_reduction_tiles=panels.reduction_tiles, ordering=report,
        ownership='immutable full RHS; LHS preparation/product/pool have real drains'))


def emit_pipeline(pipeline, builder, lower, complete):
    panels = pipeline.panels
    p = pipeline.block.source.producer
    w = p.window
    _, height, width, ci = w.input_shape
    kh, kw = w.kernel
    oh, ow = w.output_spatial
    sh, sw = w.stride
    dh, _ = w.dilation
    pt, pl, _, _ = w.padding
    op = builder.op
    chunks = panels.horizontal_chunks
    # Consume the shared storage transform's actual slot/column assignment.
    for ni,(column,columns) in enumerate(panels.column_intervals):
        for ky in range(kh):
            for chunk in range(chunks):
                slot=ky*chunks+chunk
                lower('load_weight',dict(reduction_row=ky*kw*ci+chunk*isa.DIM,
                    output_column=column,local_row=panels.rhs_base+(ni*panels.reduction_tiles+slot)*isa.DIM,
                    rows=min(isa.DIM,kw*ci-chunk*isa.DIM),columns=columns))
    complete('horizontal_rhs_ready',{})

    def interior_tile(f):
        row=f['output_row']; rows=f['rows']; column=f['output_column']
        x=op('urem',row,ow)
        def horizontal():
            # All input slots become ready before any consuming product. One
            # branch selects actual original rows or device-zero rows per ky.
            for ky in range(kh):
                # The original typed window emits the actual source address
                # and validity. Horizontal partitioning proves the remaining
                # KW*CI lanes and row-run span are within this original row.
                run=w.emit_row_run(builder,row,1,ky,0,0,ci)
                def load(zero):
                    base=0 if zero else run['first_element']
                    for chunk in range(chunks):
                        lower('load_input',dict(zero=zero,
                            first_element=0 if zero else op('add',base,chunk*isa.DIM),
                            row_stride_elements=run['row_stride_elements'],
                            local_row=(ky*chunks+chunk)*isa.DIM,
                            rows=rows,columns=min(isa.DIM,kw*ci-chunk*isa.DIM)))
                builder.branch(run['predicate'],lambda:load(False),lambda:load(True))
            if p.bias:
                lower('initialize_bias',f)
            complete('horizontal_inputs_ready',f)
            ni=op('udiv',column,p.column_tile)
            for ky in range(kh):
                for chunk in range(chunks):
                    slot=ky*chunks+chunk
                    lower('compute',dict(f,lhs_row=slot*isa.DIM,
                        rhs_row=op('add',panels.rhs_base+slot*isa.DIM,
                            op('mul',ni,panels.reduction_tiles*isa.DIM)),
                        depth=min(isa.DIM,kw*ci-chunk*isa.DIM),
                        first_reduction=builder.cmp('eq',slot,0)))
            complete('horizontal_raw_tile_complete',f)
        horizontal()

    def tile(f, fallback):
        # Partition at original horizontal halo bounds before vector tiling.
        # Only the at-most-kernel-width border uses original per-tap products.
        first=min(ow,(pl+sw-1)//sw)
        last=max(first,min(ow,(width+pl-kw)//sw+1))
        x=op('urem',f['output_row'],ow)
        def clip_count(bound):
            distance=builder.select(builder.cmp('slt',x,bound),op('sub',bound,x),0)
            return builder.select(builder.cmp('ult',distance,f['rows']),distance,f['rows'])
        prefix=clip_count(first)
        end=clip_count(last)
        def fields(start,count):
            return dict(f,output_row=op('add',f['output_row'],start),rows=count,
                        accumulator_row=op('add',f['accumulator_row'],start))
        def border(i):
            inside=op('and',builder.cmp('ule',prefix,i),builder.cmp('ult',i,end),'i1')
            builder.branch(inside,lambda:None,lambda:fallback(fields(i,1)))
        builder.loop(0,f['rows'],1,border)
        count=op('sub',end,prefix)
        builder.branch(builder.cmp('ult',0,count),
                       lambda:interior_tile(fields(prefix,count)),lambda:None)
    return tile
