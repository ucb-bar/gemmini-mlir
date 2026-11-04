"""Direct dense-to-spatial-max lowering without an external intermediate ABI.

The shared spatial gather planner owns output order and clamped source indices.
One complete window becomes the M dimension of a small ordinary contraction;
RHS tiles are reused across its rows. All tensor arithmetic executes on Gemmini.
"""
from dataclasses import replace
from math import isfinite

from merlin.compile.logical_abi import pool_params
from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.spatial_gather import plan_gathered_spatial_max
from merlin.compile.scheduling.llvm_program import emit_index_program

from dense import _capacities
from elementwise import _output

ACC=1<<31


def emit_dense_pool(enc,a,w,out,m,k,n,**attrs):
    params=pool_params(attrs,op='dense pooled readout')
    h,iw=params['pool_in_dims']
    if min(m,k,n,h,iw)<=0 or m%(h*iw):
        raise ValueError('dense pool source rows must exactly cover pool_in_dims planes')
    batch=m//(h*iw)
    source=SpatialMaxReadout(tuple(params['pool_in_dims']),tuple(params['pool_size']),
        tuple(params['pool_stride']),tuple(params['pool_padding']),params['pad_value'])
    kh,kw=source.window
    sh,sw=source.stride
    pt,pl,pb,pr=source.padding
    oh,ow=source.output_hw
    if kh!=kw or kh not in (1,2,3):
        raise ValueError('native pooled dense readout requires square 1..3 window')
    if any(source.padding) and source.pad_value not in (0,-128):
        raise ValueError('native pooled dense supports explicit zero or minimum-i8 padding')
    # Clamping is the library-proved min-padding coordinate map. For zero
    # padding we additionally clear the corresponding final accumulator row.
    gather_source=replace(source,pad_value=-128) if any(source.padding) else source
    plan=plan_gathered_spatial_max(gather_source,batches=batch,channels=n,channel_block=16)
    edge,columns,_,operand_rows,operand_bytes,_,acc_rows,acc_bytes=_capacities()
    if edge!=16 or columns!=16 or kh*kw>edge or operand_rows<edge*2 or acc_rows<kh*kw:
        raise ValueError('selected storage/array geometry cannot hold gathered dense window')
    if operand_bytes<16 or acc_bytes<64:
        raise ValueError('selected physical rows do not hold primitive dense pooled tiles')
    normalized=dict(attrs)
    if 'scale' in attrs:
        if isinstance(attrs['scale'],(list,tuple)):
            normalized['acc_scale_by_column']=attrs['scale']
        else:
            normalized['acc_scale']=attrs['scale']
    normalized.setdefault('output_dtype',attrs.get('out_dtype','i8'))
    dtype,scalar_scale,relu=_output(normalized)
    if dtype!='i8':
        raise ValueError('native spatial pooling requires signed-i8 narrowing readout')
    bias=attrs.get('bias')
    scales=attrs.get('acc_scale_by_column',attrs.get('scale',scalar_scale))
    if isinstance(scales,(list,tuple)):
        if len(scales)!=n:
            raise ValueError('dense pooled column scales must match output channels')
        # Native gather blocks use sixteen columns per setting. Arbitrary
        # equal-scale runs need a different channel gather partition.
        if any(len(set(map(float,scales[c:c+16])))!=1 for c in range(0,n,16)):
            raise ValueError('pooled dense needs equal scales within each primitive channel tile')
    elif not isfinite(float(scales)):
        raise ValueError('dense pooled readout needs finite scale')
    staged=[]
    control=enc.index_builder

    def finish_window(f):
        if len(staged)!=kh*kw:
            raise ValueError('shared spatial planner did not emit one complete source window')
        channel=f['first_column']; width=f['columns']
        if bias is not None:
            enc.load(enc.ptr_offset(bias,enc.mul(channel,4)),ACC,kh*kw,width,0,
                     stream=2,shrunk=False)
            enc.fence()
        def reduction(offset,depth,continue_sum):
            # Every source row belongs to the original A matrix; no im2col
            # tensor or CPU packing is created.
            for row in staged:
                index=enc.add(enc.mul(row['source_row'],k),offset)
                enc.load(enc.ptr_offset(a,index),row['window_row'],1,depth,k,stream=0)
            index=enc.add(enc.mul(offset,n),channel)
            enc.load(enc.ptr_offset(w,index),edge,depth,width,n,stream=1)
            enc.fence()
            enc.preload(edge,0,depth,width,kh*kw,accumulate=continue_sum)
            enc.compute(0,depth,kh*kw)
            enc.fence()
        reduction(0,min(edge,k),bias is not None)
        full=k//edge*edge
        if full>edge:
            enc.fn.for_loop(edge,full,edge,lambda offset:reduction(offset,edge,True),retain_loop=True)
        if k>edge and k%edge:
            reduction(full,k%edge,True)
        if any(source.padding) and source.pad_value==0:
            output_row=f['output_row']
            pos=enc.fn.urem_i(output_row,enc.val(oh*ow))
            oy=enc.fn.udiv_i(pos,enc.val(ow)); ox=enc.fn.urem_i(pos,enc.val(ow))
            for r in range(kh*kw):
                yy=enc.add(enc.mul(oy,sh),r//kw-pt)
                xx=enc.add(enc.mul(ox,sw),r%kw-pl)
                valid=control.op('and',control.cmp('ult',yy,h),control.cmp('ult',xx,iw),dtype='i1')
                def clear(r=r):
                    enc.load(enc.ptr_zero(),ACC|r,1,width,0,stream=2,shrunk=False)
                    enc.fence()
                control.branch(valid,lambda:None,clear)
        dest=enc.add(enc.mul(f['output_row'],n),channel)
        def store(scale):
            enc.store(enc.ptr_offset(out,dest),0,1,width,n,dtype='i8',scale=scale,relu=relu,
                pool_out_dim=1,pool_size=kh,pool_stride=1,
                porows=1,pocols=1,orows=kh,ocols=kw,upad=0,lpad=0)
            enc.fence()
        if isinstance(scales,(list,tuple)):
            # Channel-loop position is dynamic; emit configuration selection
            # using address/control predicates only, never loading tensor data.
            runs=[]
            for c in range(0,n,16):
                value=float(scales[c])
                if not isfinite(value):raise ValueError('nonfinite pooled channel scale')
                if runs and runs[-1][2]==value:
                    first,stop,v=runs[-1];runs[-1]=(first,min(c+16,n),v)
                else:runs.append((c,min(c+16,n),value))
            def select_run(index):
                if index==len(runs)-1:
                    store(runs[index][2]);return
                _,stop,value=runs[index]
                control.branch(control.cmp('ult',channel,stop),lambda:store(value),lambda:select_run(index+1))
            select_run(0)
        else:
            store(float(scales))
        staged.clear()

    def lower(kind,fields):
        if kind=='gather_window_row':staged.append(fields)
        elif kind=='store_window_max':finish_window(fields)
        else:raise ValueError(f'unknown shared spatial instruction {kind}')
    emit_index_program(plan.kernel,enc.index_builder,lower)
    return {'route':'direct_dense_gathered_pool','output_shape':[batch*oh*ow,n],
            'window_rows':kh*kw,'shared_schedule_sha256':plan.kernel.digest()}
