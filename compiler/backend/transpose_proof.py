"""Constructive proof of the actual canonical baseline transpose DMA stream.

Exact Kernel equality binds every counted domain, affine descriptor, argument,
attribute, configuration and completion site to the pure constructor. Separate
mixed-radix and interval arguments below prove that construction implements the
original permutation, including unit axes and peeled row/column tails. Work is
bounded by rank and static descriptor sites, never by tensor element count.
"""
from __future__ import annotations
from math import prod
from merlin.sched.ir import Call,Const,Loop,Scaled,Sum,Var
from . import isa


def _affine(expr):
    """Normalize an actual emitted address to its integer affine coefficients."""
    if isinstance(expr,Const):
        if type(expr.value)is not int:raise ValueError('transpose affine constant is not an integer')
        return expr.value,{}
    if isinstance(expr,Var):return 0,{expr.name:1}
    if isinstance(expr,Scaled):
        if type(expr.factor)is not int:raise ValueError('transpose affine coefficient is not an integer')
        constant,terms=_affine(expr.expr)
        return constant*expr.factor,{k:v*expr.factor for k,v in terms.items()if v*expr.factor}
    if isinstance(expr,Sum):
        constant=0;terms={}
        for term in expr.terms:
            c,t=_affine(term);constant+=c
            for name,value in t.items():terms[name]=terms.get(name,0)+value
        return constant,{k:v for k,v in terms.items()if v}
    raise ValueError('transpose address is not a supported integer affine expression')


def check_baseline_projection(kernel,shape,permutation,dtype):
    from .transpose import _baseline_kernel,TransposeError
    shape,permutation=tuple(shape),tuple(permutation)
    canonical=_baseline_kernel(shape,permutation,dtype)
    if kernel!=canonical or kernel.digest()!=canonical.digest():
        raise TransposeError('transpose counted offsets, tails, arguments, attributes, configuration or completion changed')
    rank=len(shape);width=1 if dtype=='i8'else 4
    output=tuple(shape[i]for i in permutation)
    source_strides=tuple(prod(shape[i+1:])for i in range(rank))
    output_strides=tuple(prod(output[i+1:])for i in range(rank))
    count=prod(shape)
    # General finite-rank mixed-radix law: at each axis the trailing domain
    # has exactly its stride cardinality. This includes singleton digits and
    # the empty trailing domain at rank one; no tensor points are enumerated.
    for extents, strides in ((shape, source_strides), (output, output_strides)):
        trailing=1
        for axis in reversed(range(rank)):
            if strides[axis]!=trailing or type(extents[axis])is not int or extents[axis]<=0:
                raise TransposeError('transpose axis domain does not form an exact positive mixed radix')
            trailing*=extents[axis]
        if trailing!=count:
            raise TransposeError('transpose source/output axis domains have different cardinality')
    # Prove the contiguous suffix independently of constructor control flow.
    # Every varying suffix digit has exactly its mixed-radix coefficient;
    # a unit digit is zero and contributes nothing regardless of its pitch.
    start=rank;suffix=1
    while start:
        axis=start-1;extent=output[axis]
        if extent!=1 and source_strides[permutation[axis]]!=suffix:break
        suffix*=extent;start-=1
    if suffix!=prod(output[start:]):raise TransposeError('transpose suffix radix product changed')
    for axis in range(start,rank):
        if output[axis]!=1 and source_strides[permutation[axis]]!=output_strides[axis]:
            raise TransposeError('transpose suffix does not map contiguous source columns')
    row_axis=start-1
    row_extent=output[row_axis]if row_axis>=0 else 1
    source_pitch=source_strides[permutation[row_axis]]if row_axis>=0 else suffix
    destination_pitch=suffix
    if (not 0<source_pitch*width<1<<32 or not 0<destination_pitch*width<1<<32
            or not 0<count*width<1<<63):
        raise TransposeError('transpose address or configuration field bound failed')
    # Reordering a complete axis permutation retains the exact maximum source
    # coordinate. This proves every source point is in the original buffer.
    source_last=sum((output[i]-1)*source_strides[permutation[i]]for i in range(rank))
    output_last=sum((output[i]-1)*output_strides[i]for i in range(rank))
    if source_last!=count-1 or output_last!=count-1:
        raise TransposeError('transpose permutation does not retain original mixed-radix bounds')
    prefix=tuple(range(max(0,row_axis)))
    prefix_domains=tuple((f'p{i}',output[i])for i in prefix)
    row_full,row_tail=divmod(row_extent,isa.DIM)
    col_full,col_tail=divmod(suffix,isa.DIM)
    if row_full*isa.DIM+row_tail!=row_extent or col_full*isa.DIM+col_tail!=suffix:
        raise TransposeError('transpose full/tail partition is not exhaustive')
    sites=[]
    def integer(call,key):
        value=call.arg(key)
        if not isinstance(value,Const)or type(value.value)is not int:
            raise TransposeError('transpose descriptor is not a static integer')
        return value.value
    def walk(body,scope):
        at=0
        while at<len(body):
            op=body[at]
            if isinstance(op,Loop):
                if (op.var in dict(scope)or type(op.extent)is not int or op.extent<=0
                        or op.iter_args or op.yields or op.results):
                    raise TransposeError('transpose counted domain is invalid')
                walk(op.body,scope+((op.var,op.extent),));at+=1;continue
            group=body[at:at+4]
            if len(group)!=4 or any(not isinstance(x,Call)for x in group):
                raise TransposeError('transpose descriptor completion partition changed')
            load,drain,store,last=group
            if [x.instr for x in group]!=['transpose_mvin','transpose_drain','transpose_mvout','transpose_drain']:
                raise TransposeError('transpose accumulator completion edges changed')
            rows,cols=integer(load,'rows'),integer(load,'cols')
            if (not 0<rows<=min(isa.DIM,isa.ACC_ROWS)or not 0<cols<=isa.DIM
                    or integer(store,'rows')!=rows or integer(store,'cols')!=cols
                    or integer(load,'addr')!=isa.acc_addr(0)
                    or integer(store,'addr')!=isa.acc_addr(0,full=dtype=='i32')):
                raise TransposeError('transpose DMA or accumulator capacity bound failed')
            domains=dict(scope)
            expected_domains=prefix_domains+((('r',row_full),)if 'r'in domains else ())+((('c',col_full),)if 'c'in domains else ())
            if scope!=expected_domains:
                raise TransposeError('transpose prefix or full-tile counted domains changed')
            r0=0 if 'r'in domains else row_full*isa.DIM
            c0=0 if 'c'in domains else col_full*isa.DIM
            if rows!=(isa.DIM if 'r'in domains else row_tail)or cols!=(isa.DIM if 'c'in domains else col_tail):
                raise TransposeError('transpose peeled descriptor extent is not its exact remainder')
            def address(call,pitch,coefficients):
                terms={f'p{i}':coefficients[i]for i in prefix}
                if 'r'in domains:terms['r']=isa.DIM*pitch
                if 'c'in domains:terms['c']=isa.DIM
                constant=r0*pitch+c0
                try:actual=_affine(call.arg('offset'))
                except ValueError as exc:raise TransposeError(str(exc))from exc
                if actual!=(constant,terms):
                    raise TransposeError('transpose actual DMA address differs from source axis projection')
                upper=constant+sum((extent-1)*terms[name]for name,extent in scope)+(rows-1)*pitch+cols-1
                if not 0<=constant<=upper<count:
                    raise TransposeError('transpose actual affine DMA endpoint exceeds its original boundary')
            address(load,source_pitch,tuple(source_strides[i]for i in permutation))
            address(store,destination_pitch,output_strides)
            sites.append((scope,rows,cols));at+=4
    walk(kernel.body[3:],())
    cardinality=sum(prod(extent for _,extent in scope)*rows*cols for scope,rows,cols in sites)
    if not sites or cardinality!=count:
        raise TransposeError('transpose counted DMA cardinality differs from original tensor')
    # Prefix coordinates, unique q*DIM+t on the row axis, and unique q*DIM+t
    # in the flattened contiguous suffix form a Cartesian partition. Mixed
    # radix decomposition of the suffix is unique, so each output coordinate
    # appears once. Actual affine coefficients above map it to the original
    # input coordinate with axes permutation[i], proving the bijection.
    return dict(schema='submission.transpose_address_projection.v1',kernel_sha256=kernel.digest(),
        solver_queries=0,proof_method='exact canonical Kernel conformance; affine source-axis equality; Cartesian full/tail partition',
        source_shape=list(shape),output_shape=list(output),permutation=list(permutation),
        output_elements=count,actual_dma_sites=len(sites),
        dynamic_dma_pairs=sum(prod(n for _,n in scope)for scope,_,_ in sites),
        requested_load_bytes=count*width,requested_store_bytes=count*width,
        actual_tile_columns=sorted({cols for _,_,cols in sites}),
        source_row_stride_bytes=source_pitch*width,destination_row_stride_bytes=destination_pitch*width,
        contiguous_suffix_start=start,contiguous_suffix_elements=suffix,
        source_address_bounds=[0,source_last],destination_address_bounds=[0,output_last],
        coverage='in-bounds injective output addresses with exact source cardinality',
        semantics='signed i8 identity narrowing'if dtype=='i8'else'raw full-width i32 readout',
        ownership='MVIN then full drain then MVOUT then full drain before row reuse',
        alias_contract='distinct ordinary ABI source/output buffers')
