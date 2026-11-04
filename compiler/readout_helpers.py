"""Compute-once, partitioned device readout of materialized i32 accumulators.

The source i32 values stay unchanged. Gemmini reloads them, adds any exact i32
column bias in the accumulator, then applies the declared f32 multiplier, RNE,
activation and i8 saturation. The host computes addresses and loop bounds only.
"""
from __future__ import annotations
import math
try:
    from . import target
    from .elementwise import _tiles
except ImportError:
    import target
    from elementwise import _tiles
from merlin.llvmlower.integer_readout import partition_column_scales
from merlin.llvmlower.requantization import f32


def lower_partitioned_readout(b, source, output, rows, columns, scales, *,
                              relu=False, bias=None, source_stride=None,
                              output_stride=None):
    """Emit exact i32+bias -> scaled i8 readout with one scalar per column."""
    if min(rows,columns)<=0 or len(scales)!=columns:
        raise ValueError('partitioned readout needs positive dimensions and N scales')
    scales=[f32(float(value)) for value in scales]
    if any(not math.isfinite(value) for value in scales):
        raise ValueError('partitioned readout scale must be finite f32')
    if relu and any(value<0 for value in scales):
        raise ValueError('negative scale cannot commute accumulator ReLU')
    if all(value > 0 for value in scales):
        partition=partition_column_scales(scales,columns,relu=relu,
                                          max_partitions=max(128,columns))
        regions=partition['regions']
    else:
        # Zero and signed scalar gains retain exactly the same hardware f32
        # operation. The positive-gain transition-proof API excludes them, so
        # group only adjacent equal gains without changing arithmetic.
        regions=[]
        for column,scale in enumerate(scales):
            if regions and regions[-1]['scale']==scale:
                regions[-1]['columns']+=1
            else:
                regions.append(dict(first_column=column,columns=1,scale=scale))
    source_stride=columns*4 if source_stride is None else source_stride
    output_stride=columns if output_stride is None else output_stride
    if any(type(value) is not int or value<=0 for value in (source_stride,output_stride)):
        raise ValueError('partitioned readout strides must be positive byte strides')
    target.cfg_ld(b,source_stride,stream=0,scale=1.,shrunk=False)
    if bias is not None:target.cfg_ld(b,0,stream=1,scale=1.,shrunk=False)
    def pointer(base,row,column,stride,item_bytes):
        offset=b.add_i(b.mul_i(row,b.const(stride)),b.mul_i(column,b.const(item_bytes)))
        return b.ptr(base,offset)
    for region in regions:
        first,width=region['first_column'],region['columns']
        target.cfg_st(b,output_stride,scale=scales[first],activation=int(relu))
        def tile(row,column,nr,nc):
            column=b.add_i(column,b.const(first))
            target.mvin(b,pointer(source,row,column,source_stride,4),target.ACC,nr,nc,stream=0)
            b.fence()
            if bias is not None:
                target.mvin(b,b.ptr(bias,b.mul_i(column,b.const(4))),
                            target.ACC|target.ACCUMULATE,nr,nc,stream=1)
                b.fence()
            target.mvout(b,pointer(output,row,column,output_stride,1),target.ACC,nr,nc)
            b.fence()
        _tiles(b,rows,width,tile)
    return {'schema':'gemmini.partitioned_accumulator_readout.v1',
            'scale_regions':len(regions),'columns':columns,
            'source_dtype':'i32','output_dtype':'i8','bias_in_accumulator':bias is not None,
            'numeric_order':'i32 add bias; f32 scale; RNE; ReLU; signed i8 saturation'}
