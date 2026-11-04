"""Terminal Gemmini accumulator readout shared by dense and spatial schedules."""
from __future__ import annotations
import struct
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr,i64
from xdsl.ir import Block
from . import isa


def _c(fb,v):
    return fb.const(v) if isinstance(v,int) else v


def _pack(fb,addr,rows,cols):
    return fb.or_i(fb.or_i(_c(fb,addr),fb.shl_i(_c(fb,cols),32)),fb.shl_i(_c(fb,rows),48))


def _dtype(ctx,name):
    spec=ctx.spec(name)
    return spec.get('dtype','i32') if isinstance(spec,dict) else spec.dtype


def _scale_bits(value):
    return struct.unpack('<I',struct.pack('<f',float(value)))[0]


def _emit_column_config(ctx,column,funct,rs1,words):
    """Issue a literal configuration through bounded column selection.

    The balanced tree executes at most ceil(log2(N)) column comparisons and
    never loads configuration memory or transforms tensor values on the CPU.
    Equal words share a command leaf, including nonadjacent scale runs. Keeping
    the command in that leaf preserves its complete constant encoding for
    instruction inspection; merging words first would obscure scale bits.
    """
    fb=ctx.fb
    merge=Block()
    leaves={}
    def choose(lo,hi):
        if all(word==words[lo] for word in words[lo:hi]):
            word=words[lo]
            if word not in leaves:
                leaves[word]=Block()
                fb.region.add_block(leaves[word])
            fb.add(llvm.BrOp(leaves[word]))
            return
        mid=(lo+hi)//2
        left,right=Block(),Block()
        fb.region.add_block(left)
        fb.region.add_block(right)
        condition=fb.add(llvm.ICmpOp(column,fb.const(mid),IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(),i64))).results[0]
        fb.add(llvm.CondBrOp(condition,left,[],right,[]))
        fb.blk=left
        choose(lo,mid)
        fb.blk=right
        choose(mid,hi)
    choose(0,len(words))
    for word,leaf in leaves.items():
        fb.blk=leaf
        ctx.emit(funct,rs1,word)
        fb.add(llvm.BrOp(merge))
    # Keep the printed instruction order consistent with the dataflow: every
    # literal configuration precedes the common store in the merge block.
    fb.region.add_block(merge)
    fb.blk=merge


def store_tile(ctx,dst,acc_row,rows,cols,n,output_row,output_col,attributes,*,completion=None,uniform_columns=None):
    """Emit exact hardware scale, rectifier and narrowing for one accumulator tile.

    Coordinates and extents may be i64 SSA values. Per-column scale parameters
    select literal complete command words solely to configure the device.
    No output arithmetic executes on the host.
    A supplied completion callback receives ``before_config`` and ``after_store``;
    its caller owns the corresponding readiness and storage-reuse proof. The
    default retains a real full fence at both boundaries.
    """
    fb=ctx.fb
    attrs=attributes or {}
    dtype=attrs.get('output_dtype',_dtype(ctx,dst))
    ep=attrs.get('epilogue',[])
    if 'maxpool' in ep:
        raise ValueError('maxpool requires a spatial pooling schedule before terminal tile readout')
    relu=bool(attrs.get('relu',False) or 'relu' in ep)
    if attrs.get('requant_shift') is not None and 'requant' in ep:
        raise ValueError('half-up requantization has no exact Gemmini readout contract')
    scales=attrs.get('acc_scale_by_column')
    scale=attrs.get('acc_scale',1.0)
    full=dtype=='i32'
    if full and (scales is not None or float(scale)!=1.0 or relu):
        raise ValueError('full-width accumulator readout cannot apply a narrowing epilogue')
    def complete(phase):
        if completion is None:
            ctx.fence()
        else:
            completion(phase)
    def emit_store(col_offset,width,scale_value=None,config_words=None):
        complete('before_config')
        if config_words is None:
            ctx.emit(*isa.config_st(n*(4 if full else 1),scale=scale_value if scale_value is not None else scale,activation=int(relu)))
        else:
            funct,r1,_=isa.config_st(n,scale=1.0,activation=int(relu))
            _emit_column_config(ctx,_c(fb,output_col),funct,r1,config_words)
        col=fb.add_i(_c(fb,output_col),_c(fb,col_offset))
        offset=fb.add_i(fb.mul_i(_c(fb,output_row),fb.const(n)),col)
        ctx.emit(isa.FUNCT['mvout'],ctx.ptr(dst,offset),_pack(fb,fb.or_i(_c(fb,acc_row),fb.const((1<<31)|((1<<29) if full else 0))),rows,width))
        complete('after_store')
    if uniform_columns is not None:
        from .dense_uniform_columns import DenseUniformColumnPanels
        if type(uniform_columns) is not DenseUniformColumnPanels or full:
            raise ValueError('direct mixed-scale readout requires its original i8 panel proof')
        uniform_columns.validate_source(n,attrs)
        if isinstance(output_col,int) and isinstance(cols,int):
            if (output_col,cols) not in tuple((start,width) for start,width,_ in uniform_columns.tiles):
                raise ValueError('direct dense readout requires a complete original N tile')
    if scales is None or len(set(map(float,scales)))==1:
        emit_store(0,cols,scale_value=scales[0] if scales else scale)
        return
    if uniform_columns is not None:
        # The actual checked schedule has already proved output_col aligned
        # and cols equal to the complete original tile, including the N tail.
        # Retain the original real config/store completion boundaries.
        emit_store(0,cols,config_words=uniform_columns.words)
        return
    # Mixed scales use a one-column producer tile, so the value is always in
    # accumulator lane zero. Configuration selection uses no memory loads.
    cache=getattr(ctx,'_dense_scale_tables',None)
    if cache is None:
        cache={}
        ctx._dense_scale_tables=cache
    bits=tuple(_scale_bits(x) for x in scales)
    key=(n,dtype,bits)
    if key not in cache:
        words=tuple(isa.config_st(n,scale_bits=value)[2] for value in bits)
        cache[key]=words
    emit_store(0,1,config_words=cache[key])
