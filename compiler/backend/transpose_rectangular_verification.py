"""Exact algebraic proof for the baseline rectangular transpose AST.

This deliberately admits a narrow counted grammar. It consumes the actual
loops, descriptors, offsets, configurations and drains; it does not regenerate
an expected kernel. Source strides derive an independent Cartesian partition
whose mixed-radix coordinates prove bounds, injectivity and complete coverage.
"""
from __future__ import annotations
from math import prod
from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Sum, TensorArg, Var
from . import isa


def check_rectangular_projection(kernel, shape, permutation, dtype):
    from .transpose import TransposeError
    def fail(message):
        raise TransposeError('rectangular transpose proof: '+message)
    shape,permutation=tuple(shape),tuple(permutation)
    rank=len(shape)
    if (not 2<=rank<=4 or any(type(n) is not int or n<=0 for n in shape)
        or any(type(i) is not int for i in permutation)
        or sorted(permutation)!=list(range(rank)) or dtype not in ('i8','i32')):
        fail('invalid complete integer source/permutation')
    width=1 if dtype=='i8' else 4
    count=prod(shape)
    if count*width>=1<<63:fail('source byte footprint exceeds signed index range')
    output=tuple(shape[i] for i in permutation)
    ss=tuple(prod(shape[i+1:]) for i in range(rank))
    ds=tuple(prod(output[i+1:]) for i in range(rank))
    expected=(TensorArg('src',shape,dtype,'read'),TensorArg('dst',output,dtype,'write'))
    if (type(kernel) is not Kernel or type(kernel.args) is not tuple
        or any(type(arg) is not TensorArg or type(arg.shape) is not tuple
               or any(type(n) is not int or n<=0 for n in arg.shape)
               or any(type(value) is not str for value in (arg.name,arg.dtype,arg.access))
               for arg in kernel.args)
        or kernel.args!=expected or kernel.attrs!=() or type(kernel.body) is not tuple):
        fail('typed baseline source boundary or attributes changed')
    # Derive the contiguous suffix from source layout, independently of AST.
    run=1;start=rank
    for axis in range(rank-1,-1,-1):
        if output[axis]!=1 and ss[permutation[axis]]!=run:break
        run*=output[axis];start=axis
    row_axis=start-1
    row_extent=output[row_axis] if row_axis>=0 else 1
    sp=ss[permutation[row_axis]] if row_axis>=0 else run
    dp=run
    if max(sp,dp)*width>=1<<32:fail('DMA pitch exceeds configuration field')
    scope={}
    def call(op,name,keys):
        if (type(op) is not Call or op.instr!=name or type(op.args) is not tuple
            or len(op.args)!=len(keys) or {key for key,_ in op.args}!=set(keys)
            or op.unit is not None or op.produces is not None or op.awaits!=()
            or op.stages!=() or op.sets!=() or op.assumes!=()):
            fail('instruction fields, effects or completion order changed')
        return dict(op.args)
    def integer(value):
        if type(value) is not Const or type(value.value) is not int:
            fail('descriptor requires an exact integer constant')
        return value.value
    def loop(op,name,extent):
        if (type(op) is not Loop or op.var!=name or type(op.extent) is not int
            or op.extent!=extent or extent<=0 or name in scope
            or type(op.body) is not tuple or op.iter_args!=() or op.yields!=() or op.results!=()):
            fail('counted Cartesian loop domain changed')
        return op.body
    def affine(expr):
        # Return (constant,{binder:coefficient}), checking every actual
        # subexpression's integer domain, so index construction cannot wrap.
        if type(expr) is Const:
            if type(expr.value) is not int:fail('noninteger address constant')
            result=(expr.value,{})
        elif type(expr) is Var:
            if type(expr.name) is not str or expr.name not in scope:fail('unbound address variable')
            result=(0,{expr.name:1})
        elif type(expr) is Sum:
            if type(expr.terms) is not tuple or not expr.terms:fail('invalid address sum')
            constant=0;terms={}
            for term in expr.terms:
                c,ts=affine(term);constant+=c
                for key,value in ts.items():terms[key]=terms.get(key,0)+value
            result=(constant,{key:value for key,value in terms.items() if value})
        elif type(expr) is Scaled:
            if type(expr.factor) is not int:fail('noninteger address multiplier')
            c,ts=affine(expr.expr)
            result=(c*expr.factor,{key:value*expr.factor for key,value in ts.items() if value*expr.factor})
        else:fail('address is outside the affine grammar')
        constant,terms=result
        lo=constant+sum(min(0,value*(scope[key]-1)) for key,value in terms.items())
        hi=constant+sum(max(0,value*(scope[key]-1)) for key,value in terms.items())
        if not -(1<<63)<=lo<=hi<(1<<63):fail('address subexpression exceeds signed index range')
        return result
    if len(kernel.body)<3:fail('missing configuration prologue')
    call(kernel.body[0],'transpose_drain',())
    ld=call(kernel.body[1],'transpose_config_load',('stride',))
    st=call(kernel.body[2],'transpose_config_store',('stride',))
    if integer(ld['stride'])!=sp*width or integer(st['stride'])!=dp*width:
        fail('configuration pitches differ from source-derived rectangle')
    body=kernel.body[3:]
    src_terms={};dst_terms={}
    # Each prefix loop is exactly one output coordinate. It contributes its
    # source and destination stride, preserving axes even when their extent1
    # gives no physical displacement.
    for axis in range(max(0,row_axis)):
        name=f'p{axis}'
        if len(body)!=1:fail('prefix does not cover a single Cartesian product')
        body=loop(body[0],name,output[axis]);scope[name]=output[axis]
        src_terms[name]=ss[permutation[axis]];dst_terms[name]=ds[axis]
    sites=0;dynamic=0;columns=set();covered=0
    def tile(ops,row,rows,col,cols):
        nonlocal sites,dynamic,covered
        if len(ops)!=4:fail('tile must own exactly one load/readout completion pair')
        load=call(ops[0],'transpose_mvin',('offset','addr','rows','cols'))
        call(ops[1],'transpose_drain',())
        store=call(ops[2],'transpose_mvout',('offset','addr','rows','cols'))
        call(ops[3],'transpose_drain',())
        if (not 0<rows<=min(isa.DIM,isa.ACC_ROWS) or not 0<cols<=isa.DIM
            or integer(load['rows'])!=rows or integer(store['rows'])!=rows
            or integer(load['cols'])!=cols or integer(store['cols'])!=cols
            or integer(load['addr'])!=isa.acc_addr(0)
            or integer(store['addr'])!=isa.acc_addr(0,full=dtype=='i32')):
            fail('DMA dimensions or local accumulator footprint changed')
        for actual,pitch,terms in [(load,sp,src_terms),(store,dp,dst_terms)]:
            # row/col encode independently proved quotient/remainder tiles.
            rc,rt=row;cc,ct=col
            coefficients=dict(terms)
            for key,value in rt.items():coefficients[key]=coefficients.get(key,0)+value*pitch
            for key,value in ct.items():coefficients[key]=coefficients.get(key,0)+value
            expected_offset=(rc*pitch+cc,{key:value for key,value in coefficients.items() if value})
            if affine(actual['offset'])!=expected_offset:fail('actual DMA offset differs from original source coordinates')
            # Include the last row and column, not just the tile origin.
            constant,coeffs=expected_offset
            low=constant+sum(min(0,v*(scope[key]-1)) for key,v in coeffs.items())
            high=constant+sum(max(0,v*(scope[key]-1)) for key,v in coeffs.items())+(rows-1)*pitch+cols-1
            if not 0<=low<=high<count or (high+1)*width>count*width:
                fail('actual DMA rectangle leaves its original allocation')
        multiplicity=prod(scope.values());sites+=1;dynamic+=multiplicity
        covered+=multiplicity*rows*cols;columns.add(cols)
    def column_partition(items,row,rows):
        full,tail=divmod(run,isa.DIM);at=0
        if full:
            if not items:fail('missing full column partition')
            inside=loop(items[0],'c',full);scope['c']=full
            tile(inside,row,rows,(0,{'c':isa.DIM}),isa.DIM)
            del scope['c'];at=1
        if tail:
            tile(items[at:at+4],row,rows,(full*isa.DIM,{}),tail);at+=4
        if len(items)!=at:fail('extra or duplicated column partition')
        # [j*DIM,j*DIM+DIM) for j<full, followed by the exact
        # remainder, partitions [0,run) into disjoint contiguous intervals.
        if full*isa.DIM+tail!=run:fail('column partition is incomplete')
    full,tail=divmod(row_extent,isa.DIM);at=0
    if full:
        if not body:fail('missing full row partition')
        inside=loop(body[0],'r',full);scope['r']=full
        column_partition(inside,(0,{'r':isa.DIM}),isa.DIM)
        del scope['r'];at=1
    if tail:
        column_partition(body[at:],(full*isa.DIM,{}),tail);at=len(body)
    if len(body)!=at or full*isa.DIM+tail!=row_extent:fail('row partition changed')
    # Prefix coordinates, row intervals, and contiguous suffix intervals form
    # one disjoint Cartesian partition of the output. Mixed-radix row-major
    # coordinates uniquely represent every output element. The suffix's
    # source-stride equalities identify the same original source coordinates;
    # permutation bijectivity makes source coverage injective as well.
    if not sites or covered!=count:fail('complete injective Cartesian coverage cardinality changed')
    return dict(schema='submission.transpose_address_projection.v1',kernel_sha256=kernel.digest(),
        solver_queries=0,proof='exact actual-AST affine rectangles and disjoint mixed-radix Cartesian partitions',
        source_shape=list(shape),output_shape=list(output),permutation=list(permutation),
        output_elements=count,actual_dma_sites=sites,dynamic_dma_pairs=dynamic,
        requested_load_bytes=count*width,requested_store_bytes=count*width,
        actual_tile_columns=sorted(columns),source_row_stride_bytes=sp*width,destination_row_stride_bytes=dp*width,
        coverage='in-bounds injective output addresses with exact source cardinality',
        semantics='signed i8 identity narrowing' if dtype=='i8' else 'raw full-width i32 readout',
        ownership='MVIN then full drain then MVOUT then full drain before row reuse',
        alias_contract='distinct ordinary ABI source/output buffers')
