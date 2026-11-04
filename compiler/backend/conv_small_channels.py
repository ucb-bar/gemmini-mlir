"""Row-contained horizontal reduction panels for small NHWC channels.

The library's checked horizontal reduction artifact maps q=KX*CI+C into
primitive K chunks. This consumer keeps M in one original output row, making
all interior source transfers affine rectangles. No im2col or host payload
copy exists. Original horizontal halo columns retain the shared exact vertical
boundary schedule; all narrowing/readout uses the incumbent terminal encoder.
"""
from __future__ import annotations

from . import isa
from .conv_flattened_rows import _prepare, _lower_prepared, _emit_boundary


def _emit_row_bands(prepared,builder,lower,complete):
    panels=prepared.panels
    panels.__post_init__()
    p=prepared.original;w=p.window
    batch,height,width,ci=w.input_shape
    kh,kw=w.kernel;oh,ow=w.output_spatial
    interior=panels.source.window.output_spatial[1]
    chunks=panels.horizontal_chunks
    d=isa.DIM
    if panels.row_blocks!=1:raise ValueError('row bands require one private M tile')
    # The shared panel artifact supplies injective complete-K weight storage.
    # Every original reduction row appears once, in KH/KW/CI source order.
    for ky in range(kh):
        for chunk in range(chunks):
            slot=ky*chunks+chunk
            depth=min(d,panels.horizontal_columns-chunk*d)
            for ni,(column,columns)in enumerate(panels.column_intervals):
                lower('load_weight',dict(reduction_row=ky*kw*ci+chunk*d,
                    output_column=column,local_row=panels.rhs_base+(ni*panels.reduction_tiles+slot)*d,
                    rows=depth,columns=columns))
    complete('small_channel_rhs_ready',{})
    def image(image_index):
        def row(output_y):
            sy=builder.op('mul',output_y,w.stride[0])
            original_line=builder.op('add',builder.op('mul',image_index,oh),output_y)
            output_base=builder.op('add',builder.op('mul',original_line,ow),prepared.first_column)
            image_base=builder.op('mul',image_index,height*width*ci)
            def tile(x,rows):
                output=builder.op('add',output_base,x)
                ix=builder.op('add',builder.op('mul',x,w.stride[1]),prepared.input_x)
                xbase=builder.op('add',image_base,builder.op('mul',ix,ci))
                # Each rectangle writes its destination cells once, from the
                # actual original row or the device zero source. Vertical halo
                # selection never races a zero fill with a data overwrite.
                for ky in range(kh):
                    vy=builder.op('add',sy,ky*w.dilation[0])
                    valid=builder.op('and',builder.cmp('ule',w.padding[0],vy),
                        builder.cmp('ult',vy,w.padding[0]+height),'i1')
                    def stage(zero,ky=ky,vy=vy):
                        base=0 if zero else builder.op('add',xbase,
                            builder.op('mul',builder.op('sub',vy,w.padding[0]),width*ci))
                        for chunk in range(chunks):
                            lower('load_input',dict(zero=zero,
                                first_element=0 if zero else builder.op('add',base,chunk*d),
                                local_row=(ky*chunks+chunk)*d,rows=rows,
                                columns=min(d,panels.horizontal_columns-chunk*d),
                                row_stride_elements=w.stride[1]*ci))
                    if not w.padding[0]and not w.padding[2]:stage(False)
                    else:builder.branch(valid,lambda:stage(False),lambda:stage(True))
                outputs=tuple(dict(output_row=output,output_column=column,rows=rows,
                    columns=columns,accumulator_row=ni*d)
                    for ni,(column,columns)in enumerate(panels.column_intervals))
                if p.bias:
                    for out in outputs:lower('initialize_bias',out)
                complete('small_channel_band_ready',{})
                for ky in range(kh):
                    for chunk in range(chunks):
                        slot=ky*chunks+chunk
                        for ni,out in enumerate(outputs):
                            lower('compute',dict(out,lhs_row=slot*d,
                                rhs_row=panels.rhs_base+(ni*panels.reduction_tiles+slot)*d,
                                depth=min(d,panels.horizontal_columns-chunk*d),
                                first_reduction=slot==0,bias=p.bias,fresh_weights=True))
                for out in outputs:lower('store',out)
                complete('small_channel_band_last_use',{})
            full,tail=divmod(interior,d)
            if full:builder.loop(0,full,1,lambda x:tile(builder.op('mul',x,d),d))
            if tail:tile(full*d,tail)
        builder.loop(0,oh,1,row)
    builder.loop(0,batch,1,image)
    _emit_boundary(prepared,builder,lower,complete)
    return dict(panels.summary(),mechanism='row_contained_small_channel_horizontal_panels',
        original_source_sha256=p.identity,original_output_hw=[oh,ow],
        interior_output_columns=[prepared.first_column,prepared.first_column+interior],
        source_partition='interior original-row M bands; exact incumbent vertical boundary columns',
        input_embedding='((image*H+output_y*SH-pad_top+KY*DH)*W+(first+band_x)*SW-pad_left)*CI+chunk*DIM',
        reduction_embedding='KY*(KW*CI)+chunk*DIM+lane',
        boundary_transport='unchanged shared original-output-Y gather; disjoint zero/valid/zero bands',
        spatial_tiles=batch*oh*((interior+d-1)//d),
        original_flattened_interior_tiles=(batch*oh*interior+d-1)//d,
        host_payload_arithmetic=False,payload_materialization=False)


def lower(ctx,ifm,rhs,dst,attrs):
    """Safe opt-in source/resource guard; refusal precedes all target effects."""
    prepared=_prepare(ctx,ifm,rhs,dst,attrs)
    if prepared is None:return False
    # All original shared packing/layout/quantization/alias/resource guards
    # remain in force. Narrow or exotic sources keep their incumbent route.
    return _lower_prepared(ctx,ifm,rhs,dst,prepared,
        schedule_emitter=_emit_row_bands,metadata_key='convolution_small_channels')
