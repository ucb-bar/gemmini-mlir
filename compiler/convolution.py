"""Source NHWC convolution through Merlin's counted affine window transfers.

Only coordinates and command descriptors run on the host. Every reduction and
integer readout runs on Gemmini. Interior windows use a strided multi-row DMA;
borders load either source bytes or a mapped immutable zero row exactly once.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

from merlin.compile.scheduling.window_transfer import (
    Window2D, CountedConvolutionPlan, emit_counted_convolution, emit_convolution_tile,
)
from isa import ACC, ACCUMULATE, FULL, GARBAGE, DIM, f32_bits


def _name(operands, *keys):
    for key in keys:
        if key in operands:
            return operands[key]
    raise ValueError('convolution operand missing: '+ '/'.join(keys))


def _capacities(facts=None):
    """Read the selected public storage geometry, including private row bounds."""
    if facts is None:
        for path in (Path(__file__).parent/'facts.json', Path('rtl_facts/facts.json')):
            if path.is_file():
                facts=json.loads(path.read_text()); break
    elif isinstance(facts,(str,Path)):
        facts=json.loads(Path(facts).read_text())
    if facts is None:
        raise ValueError('convolution requires selected RTL facts')
    from merlin.targetgen.address_space import derive_address_space, operand_store, accumulator_store
    space=derive_address_space('gemmini',facts=facts)
    op,acc=operand_store(space),accumulator_store(space)
    edge=min(space.array_rows,space.array_cols)
    if edge != DIM or op.store.row_bytes < edge or acc.store.row_bytes < edge*4:
        raise ValueError('convolution primitive disagrees with selected integer stores')
    return edge,op.capacity_rows(),acc.capacity_rows()


def lower_conv(fb, command, tensors, pointers, *, facts=None, residents=None, schedule_options=None):
    """Lower a complete CONV/CONV2D command; return its checked source coverage.

    ``tensors`` is the command-buffer declaration map; ``pointers`` maps logical
    tensor names to LLVM pointers. ``residents`` resolves RES_PACK handles back
    to their immutable logical source names (or dictionaries with a src key).
    All private rows are invocation-local; every callback completes before reuse.
    """
    from conv_addressing import ConvolutionAddressBuilder
    fb=ConvolutionAddressBuilder(fb)
    a=command.get('attributes',{})
    operands=command.get('operands',{})
    src=_name(operands,'ifm','input','A','lhs','src')
    weight=_name(operands,'weight','W','B','rhs')
    dst=_name(operands,'dst','output','C','Y')
    residents=residents or {}
    seen=set()
    while weight in residents:
        if weight in seen: raise ValueError('cyclic convolution resident handle')
        seen.add(weight)
        value=residents[weight]
        weight=value.get('src',value.get('source')) if isinstance(value,dict) else value
    if weight not in tensors or weight not in pointers:
        raise ValueError('convolution weight must resolve to a mapped immutable source')
    inp=tensors[src]; out=tensors[dst]; wt=tensors[weight]
    shape=tuple(inp['shape'])
    if len(shape)!=4 or str(a.get('layout','nhwc')).lower()!='nhwc':
        raise ValueError('convolution requires an explicit NHWC input')
    if inp['dtype']!='i8' or wt['dtype']!='i8':
        raise ValueError('convolution contractions require i8 operands')
    kernel=tuple(a['kernel'])
    if len(kernel) not in (2,4): raise ValueError('convolution kernel must be KH,KW[,CI,CO]')
    ci=shape[-1]
    co=kernel[3] if len(kernel)==4 else int(out['shape'][-1])
    if len(kernel)==4 and kernel[2]!=ci: raise ValueError('convolution channel mismatch')
    if math.prod(wt['shape']) != kernel[0]*kernel[1]*ci*co:
        raise ValueError('convolution weights must be contiguous flattened HWIO')
    window=Window2D(shape,kernel[:2],tuple(a.get('stride',(1,1))),
                    tuple(a.get('dilation',(1,1))),tuple(a.get('padding',(0,0,0,0))))
    oh,ow=window.output_spatial
    pooling='maxpool' in a.get('epilogue',[])
    pool_geometry=None
    if pooling:
        ps=tuple(a['pool_size']); st=tuple(a['pool_stride']); pad=tuple(a.get('pool_padding',(0,0,0,0)))
        if len(ps)!=2 or len(st)!=2 or len(pad)!=4 or ps[0]!=ps[1] or st[0]!=st[1] or not 1<=ps[0]<=3 or not 1<=st[0]<=3:
            raise ValueError('pool controller requires a square window and equal bounded strides')
        ph=(oh+pad[0]+pad[2]-ps[0])//st[0]+1
        pw=(ow+pad[1]+pad[3]-ps[1])//st[1]+1
        if min(ph,pw)<=0 or pw>=256: raise ValueError('pool controller output geometry out of range')
        if any(pad) and ('relu' not in a.get('epilogue',[]) or int(a.get('pool_pad_value',-128))>0):
            raise ValueError('padded pool requires nonnegative activated values and nonpositive source padding')
        if a.get('pool_in_dims') and tuple(a['pool_in_dims'])!=(oh,ow):
            raise ValueError('pool source spatial dimensions disagree with convolution')
        pool_geometry=(ps[0],st[0],pad,ph,pw)
    else:
        ph,pw=oh,ow
    if math.prod(out['shape']) != shape[0]*ph*pw*co:
        raise ValueError('convolution output extent disagrees with its source window/readout')
    dtype=a.get('output_dtype',out['dtype'])
    if dtype not in ('i8','i32') or dtype != out['dtype']:
        raise ValueError('convolution output must be source i8 or raw i32')
    epilogue=a.get('epilogue',[])
    unsupported=set(epilogue)-{'bias','bias_add','acc_scale','relu','maxpool'}
    if unsupported: raise ValueError('unsupported convolution readout stages: '+str(sorted(unsupported)))
    if dtype=='i32' and any(x in epilogue for x in ('acc_scale','relu','maxpool')):
        raise ValueError('raw accumulator exit cannot implement scaled/activated readout')
    if 'requant_shift' in a: raise ValueError('shift requantization differs from Gemmini half-even readout')
    bias=a.get('bias',operands.get('bias'))
    if ('bias' in epilogue or 'bias_add' in epilogue) and not bias:
        raise ValueError('convolution bias stage lacks source tensor')
    if bias and (bias not in pointers or tensors[bias]['dtype']!='i32' or math.prod(tensors[bias]['shape'])!=co):
        raise ValueError('convolution bias must be an i32 output-channel vector')
    scales=a.get('acc_scale_by_column')
    if scales is not None and (len(scales)!=co or 'acc_scale' in a):
        raise ValueError('convolution per-column scales must cover original columns exactly')
    if scales is None: scales=[float(a.get('acc_scale',1.0))]*co
    if not all(math.isfinite(float(x)) for x in scales): raise ValueError('non-finite convolution scale')
    if dtype=='i32' and any(float(x)!=1.0 for x in scales):
        raise ValueError('raw convolution exit cannot apply a readout scale')
    edge,operand_rows,accumulator_rows=_capacities(facts)
    from conv_policy import accumulator_init_selection
    requested_init=accumulator_init_selection()
    reduction_elements=kernel[0]*kernel[1]*ci
    zero_accumulate=not bias and (requested_init=='zero_accumulate' or
        requested_init=='zero_single_slice' and reduction_elements<=edge)
    explicit_initializer=bool(bias) or zero_accumulate
    column_tile=edge
    uniform_scale=all(float(x)==float(scales[0]) for x in scales)
    if not uniform_scale:
        for col in range(1,co):
            if float(scales[col])!=float(scales[col-1]): column_tile=math.gcd(column_tile,col)
    plan=CountedConvolutionPlan(window,co,edge,min(edge,ow),column_tile,min(edge,ci),
                                operand_rows,accumulator_rows,0,edge,0,explicit_initializer)
    fb.fence()
    fb.config_ex(dataflow=1)
    # LOAD3 broadcasts the original i32 bias, or source reduction identity, to
    # the active ACC rows. The shared plan orders initialization before products.
    if explicit_initializer: fb.config_ld(0,id=2)
    activation=int('relu' in epilogue)
    width=4 if dtype=='i32' else 1
    default_scale=float(scales[0])
    fb.config_st(co*width,activation=activation,scale=default_scale)

    scale_encoding='scalar_immediate'
    scale_bits=None
    if not uniform_scale:
        # Immutable source constants become command bits through immediate CFG
        # selection. This reads no tensor/table payload and preserves each
        # original F32 gain without duplicating its complete convolution.
        scale_encoding='immediate_cfg'
        scale_bits=[f32_bits(float(scale)) for scale in scales]
    def scale_config(column,**pool):
        if uniform_scale:
            fb.config_st(co*width,activation=activation,scale=default_scale,**pool)
        else:
            fb.constant_i32_dispatch(scale_bits,column,lambda bits:
                fb.config_st(co*width,activation=activation,scale_bits=bits,**pool))
    # Padding is a mapped immutable integer-zero row. A real source transfer
    # overwrites every active input lane, independent of residual local state
    # and of simulator-specific treatment of a null DRAM source.
    # The counted window emitter materializes both sides of its runtime bounds
    # branch even for a source with no padding. Its unreachable zero-source arm
    # must therefore still contain a valid mapped pointer in the generated IR.
    zero_row=fb.constant_i32_table([0]*(edge if zero_accumulate else (edge+3)//4))
    suppress_store=False
    def lower(kind,f,*,drain=True):
        if kind=='store' and suppress_store: return
        if kind=='load_input':
            fb.config_ld(f['row_stride_elements'],id=0)
            ptr=zero_row if f['zero'] else fb.ptr_offset(pointers[src],f['first_element'])
            fb.mvin(ptr,f['local_row'],cols=f['columns'],rows=f['rows'],id=0)
        elif kind=='load_weight':
            fb.config_ld(co,id=1)
            offset=fb.add(fb.mul(f['reduction_row'],co),f['output_column'])
            fb.mvin(fb.ptr_offset(pointers[weight],offset),f['local_row'],
                    cols=f['columns'],rows=f['rows'],id=1)
        elif kind=='initialize_bias':
            ptr=fb.ptr_offset(pointers[bias],fb.mul(f['output_column'],4)) if bias else zero_row
            fb.mvin(ptr,fb.or_(ACC,f['accumulator_row']),cols=f['columns'],rows=f['rows'],id=2)
        elif kind=='compute':
            def product(continuation):
                acc=fb.or_(ACC | (ACCUMULATE if continuation else 0),f['accumulator_row'])
                fb.preload(f['rhs_row'],acc,b_cols=f['columns'],b_rows=f['depth'],
                           c_cols=f['columns'],c_rows=f['rows'])
                fb.compute(f['lhs_row'],GARBAGE,a_cols=f['depth'],a_rows=f['rows'],
                           bd_cols=f['columns'],bd_rows=f['rows'])
            if explicit_initializer:product(True)
            else:
                # Each real branch owns its complete preload/compute pair.
                # Exactly one pair executes before the shared completion law.
                fb.branch(f['first_reduction'],lambda:product(False),lambda:product(True))
        elif kind=='store':
            # Counted column tiles retain every original scalar-scale boundary.
            column=f['output_column']
            if not uniform_scale: scale_config(column)
            offset=fb.mul(fb.add(fb.mul(f['output_row'],co),column),width)
            acc=fb.or_(ACC | (FULL if dtype=='i32' else 0),f['accumulator_row'])
            fb.mvout(fb.ptr_offset(pointers[dst],offset),acc,cols=f['columns'],rows=f['rows'])
        else:
            raise ValueError('unhandled source convolution effect '+kind)
        if drain: fb.fence()

    if pooling:
        # Each original pooling observation owns its complete live convolution
        # patch. The patch stays in ACC until StoreController performs the max;
        # overlapping source patches are recomputed, never reduced on the host.
        # Clipped border patches pass explicit upad/lpad to the controller.
        size,stride,pad,ph,pw=pool_geometry
        if size>plan.row_tile or size*size>accumulator_rows:
            raise ValueError('pool patch exceeds private accumulator/primitive capacity')
        def regions(extent,outputs,before):
            result=[]
            for i in range(outputs):
                first=i*stride-before
                skip=max(0,-first)
                valid=min(extent,first+size)-max(0,first)
                if valid<=0: raise ValueError('pool window has no source values')
                key=(skip,valid)
                if result and result[-1][2:]==key:
                    result[-1]=(result[-1][0],i+1,*key)
                else: result.append((i,i+1,*key))
            return result
        from conv_policy import pool_retained_tiles, emit_pool_retained_row, emit_pool_epoch_tile, epoch_selection
        from conv_pool_policy import options as pool_options
        pool_storage,pool_completion,pool_policy=pool_options(plan,pool_size=size,
            per_column=a.get('acc_scale_by_column') is not None,
            request=(schedule_options or {}).get('conv_pool_schedule'))
        retained_tiles=pool_retained_tiles(plan,patch_rows=size*size,selection=pool_storage)
        pool_epochs=pool_completion!='off' and (co+column_tile-1)//column_tile<=32
        suppress_store=True
        def image_body(image):
            for yl,yh,up,ny in regions(oh,ph,pad[0]):
                def y_body(py,up=up,ny=ny):
                    cy=fb.add(fb.mul(py,stride),up-pad[0])
                    for xl,xh,left,nx in regions(ow,pw,pad[1]):
                        def x_body(px,left=left,nx=nx):
                            cx=fb.add(fb.mul(px,stride),left-pad[1])
                            def column_body(col,nc):
                                def patch_row(iy):
                                    row=fb.add(fb.mul(fb.add(fb.mul(image,oh),fb.add(cy,iy)),ow),cx)
                                    args=dict(output_row=row,output_column=col,rows=nx,columns=nc,
                                              accumulator_row=fb.mul(iy,nx))
                                    applied=emit_pool_epoch_tile(plan,fb,lower,**args,
                                        issue=lambda kind,f:lower(kind,f,drain=False),complete=lambda kind,f:fb.fence(),
                                        completion_mode=pool_completion)
                                    if applied is None:
                                        emit_convolution_tile(plan,fb,lower,**args,synchronous_completion=True)
                                if pool_epochs:
                                    for iy in range(ny): patch_row(iy)
                                else: fb.loop(0,ny,1,patch_row)
                                scale_config(col,pool_stride=stride,pool_size=size,pool_out_dim=pw,
                                    porows=1,pocols=1,orows=ny,ocols=nx,upad=up,lpad=left)
                                offset=fb.add(fb.mul(fb.add(fb.mul(fb.add(fb.mul(image,ph),py),pw),px),co),col)
                                fb.mvout(fb.ptr_offset(pointers[dst],offset),ACC,cols=nc,rows=0)
                                fb.fence()
                            if retained_tiles is None:
                                full,tail=divmod(co,column_tile)
                                if pool_epochs:
                                    for col in range(0,co,column_tile):column_body(col,min(column_tile,co-col))
                                elif full:
                                    fb.loop(0,full,1,lambda nc: column_body(fb.mul(nc,column_tile),column_tile))
                                if tail and not pool_epochs: column_body(full*column_tile,tail)
                            else:
                                # At most three original patch rows are a
                                # static controller extent, independent of the
                                # spatial image or reduction dimensions.
                                for iy in range(ny):
                                    row=fb.add(fb.mul(fb.add(fb.mul(image,oh),fb.add(cy,iy)),ow),cx)
                                    emit_pool_retained_row(plan,fb,lower,tiles=retained_tiles,
                                        patch_offset=iy*nx,output_row=row,rows=nx,
                                        issue=lambda kind,f:lower(kind,f,drain=False),complete=lambda kind,f:fb.fence(),
                                        completion_mode=pool_completion)
                                for tile in retained_tiles:
                                    scale_config(tile.output_column,pool_stride=stride,pool_size=size,pool_out_dim=pw,
                                        porows=1,pocols=1,orows=ny,ocols=nx,upad=up,lpad=left)
                                    offset=fb.add(fb.mul(fb.add(fb.mul(fb.add(fb.mul(image,ph),py),pw),px),co),tile.output_column)
                                    fb.mvout(fb.ptr_offset(pointers[dst],offset),ACC|tile.accumulator_row,
                                        cols=tile.columns,rows=0)
                                    fb.fence()
                        fb.loop(xl,xh,1,x_body)
                fb.loop(yl,yh,1,y_body)
        fb.loop(0,shape[0],1,image_body)
        receipt=dict(plan_sha256=plan.identity,**plan.coverage,
                     pooling='source_patches_in_accumulator_store_controller_max',
                     pool_input_retention_applied=retained_tiles is not None,
                     completion_epochs=pool_completion,
                     pooled_output_elements=shape[0]*ph*pw*co)
        if pool_policy['requested']!='off':receipt['pool_schedule_policy']=pool_policy
    else:
        from conv_policy import emit_alternative
        from conv_selection import selected as selection_mode
        mode=selection_mode()
        from raw_convolution_schedule import emit_selected as emit_raw_selected
        raw_selected = emit_raw_selected(plan,fb,lower,command=command,
            issue=lambda kind,fields:lower(kind,fields,drain=False),
            complete=lambda kind,fields:fb.fence())
        if raw_selected is not None:
            receipt = raw_selected
        elif mode=='off':
            # Keep the incumbent path byte-for-byte unchanged when disabled.
            receipt=emit_alternative(plan,fb,lower,issue=lambda kind,fields:lower(kind,fields,drain=False),
                                     complete=lambda kind,fields:fb.fence())
            if receipt is None:
                receipt=emit_counted_convolution(plan,fb,lower,synchronous_completion=True)
        else:
            from conv_selection import execute,load_frozen,load_supported_geometry
            from conv_policy import completion_laws
            if mode=='supported_geometry':
                context,evidence=load_supported_geometry()
                policy,domain=None,None
            else:
                policy,context,domain,evidence=load_frozen()
            live_tiles=(co+edge-1)//edge
            ordering=completion_laws(plan,plan.identity,live_tiles,mode='pairs16') is not None
            def emit_route(current_plan,*,schedule,completion_epochs):
                actual=emit_alternative(current_plan,fb,lower,
                    issue=lambda kind,fields:lower(kind,fields,drain=False),
                    complete=lambda kind,fields:fb.fence(),selection=schedule,
                    completion_mode=completion_epochs)
                if actual is None:
                    actual=emit_counted_convolution(current_plan,fb,lower,synchronous_completion=True)
                return actual
            selection=execute(plan,emit_route,context=context,policy=policy,domain=domain,
                mode=mode,source_bias=bool(bias),geometry_support=evidence,
                per_column=a.get('acc_scale_by_column') is not None,ordering_supported=ordering)
            receipt=dict(selection.pop('actual_route'),conv_selection=selection,
                         conv_selection_evidence=evidence)
    fb.fence()
    return dict(receipt,lowering='counted_nhwc_window_row_runs',output=dst,
                command_scale_encoding=scale_encoding,
                accumulator_init='source_bias' if bias else ('zero_accumulate' if zero_accumulate else 'overwrite'),
                requested_accumulator_init=requested_init,
                source_reduction_elements=reduction_elements,selected_array_edge=edge)
