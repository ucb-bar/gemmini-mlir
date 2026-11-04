"""Contiguous raw-ACC epochs retaining vertical producer halos for pooling.

The packed spatial readout-cache library uses this same bounded contiguous
epoch lifetime, but its i8 reload owner cannot consume raw convolution results.
This adapter instead transforms a checked StripedSpatialMax source program.
It retains the library index IR, original clamped source cells and synchronous
last-use boundaries; no intermediate tensor or target opcode is introduced.
"""
from dataclasses import dataclass

from merlin.compile.scheduling.spatial_max_stripes import StripedSpatialMax, _horizontal
from merlin.compile.scheduling.spatial_gather import _axis_index
from merlin.sched.ir import Call, Const, Kernel, Loop, Var, select
from merlin.sched.ir.expr import evaluate


def _kernel(original, output_rows):
    s=original.source
    h,w=s.input_hw;kh,kw=s.window;sh,sw=s.stride;oh,ow=s.output_hw
    pt,pl,_,_=s.padding
    width=(ow-1)*sw+kw
    left,middle,right=_horizontal(w,width,pl)

    def group(count,epochs,first):
        height=kh+(count-1)*sh
        raw=Var('epoch')*(count*sh)+Var('local_y')+(first*sh-pt)
        # Peel only the first/last epoch's clipped rows. Interior rows retain
        # their original affine source index and counted row loop.
        y=select('epoch',0,_axis_index('local_y',h,height,first*sh,1,pt),raw)
        y=select('epoch',epochs-1,
            _axis_index('local_y',h,height,(first+(epochs-1)*count)*sh,1,pt),y)
        base=Var('batch')*(h*w)+y*w

        def load(x,local_x,rows):
            return Call('stage_spatial_rows',(
                ('source_row',base+x),('local_row',Var('local_y')*width+local_x),
                ('rows',Const(rows))))
        stages=[]
        if left:stages.append(Loop('left',left,(load(Const(0),Var('left'),1),)))
        full,tail=divmod(middle,original.transfer_rows)
        if full:stages.append(Loop('middle',full,(
            load(Var('middle')*original.transfer_rows,Var('middle')*original.transfer_rows+left,original.transfer_rows),)))
        if tail:stages.append(load(Const(full*original.transfer_rows),Const(left+full*original.transfer_rows),tail))
        if right:stages.append(Loop('right',right,(load(Const(w-1),Var('right')+left+middle,1),)))
        reads=Call('read_spatial_strip',(
            ('output_row',Var('batch')*(oh*ow)+(Var('epoch')*count+Var('pool_y')+first)*ow),
            ('local_row',Var('pool_y')*(sh*width))))
        return Loop('epoch',epochs,(Loop('local_y',height,tuple(stages)),Loop('pool_y',count,(reads,))))
    full,tail=divmod(oh,output_rows)
    groups=[]
    if full:groups.append(group(output_rows,full,0))
    if tail:groups.append(group(tail,1,full*output_rows))
    return Kernel('retained_producer_spatial_epochs',original.kernel.args,
        (Loop('batch',original.batches,tuple(groups)),))


@dataclass(frozen=True)
class ProducerPoolEpochs:
    original: StripedSpatialMax
    output_rows: int
    kernel: Kernel

    @property
    def temporary_rows(self):
        kh,_=self.original.stripe_hw
        return (kh+(self.output_rows-1)*self.original.source.stride[0])*self.original.stripe_hw[1]

    def revalidate(self):
        o=self.original;o.revalidate()
        if o.channels!=o.channel_block:
            raise ValueError('raw producer epochs require one caller-owned output channel interval')
        s=o.source;kh,kw=s.window;sh,sw=s.stride;oh,ow=s.output_hw
        width=o.stripe_hw[1]
        maximum=min(oh,1+(o.capacity_rows//width-kh)//sh)
        if self.output_rows!=maximum or self.output_rows<1 or self.kernel!=_kernel(o,self.output_rows):
            raise ValueError('producer pool epoch differs from its capacity-derived original source')
        if self.temporary_rows>o.capacity_rows:
            raise ValueError('producer epoch exceeds private accumulator ownership')
        # Source-only certificate: every actually consumed local row resolves
        # to exactly the same clamped original pixel as the canonical program.
        # No payload arithmetic, approximate hashes or unchecked label joins.
        h,w=s.input_hw;pt,pl,_,_=s.padding
        cells={};outputs=set();staged=0
        def visit(body,env):
            nonlocal staged
            for node in body:
                if isinstance(node,Loop):
                    for i in range(node.extent):visit(node.body,{**env,node.var:i})
                else:
                    f={k:evaluate(v,env) for k,v in node.args}
                    if node.instr=='stage_spatial_rows':
                        if not 0<=f['local_row']<f['local_row']+f['rows']<=self.temporary_rows:
                            raise ValueError('actual producer fragment exceeds assigned epoch rows')
                        for i in range(f['rows']):
                            cells[f['local_row']+i]=f['source_row']+i
                        staged+=f['rows']
                    elif node.instr=='read_spatial_strip':
                        row=f['output_row'];batch,y=divmod(row//ow,oh)
                        if row in outputs:raise ValueError('producer epoch writes original output twice')
                        outputs.add(row)
                        for x in range(ow):
                            for ky in range(kh):
                                for kx in range(kw):
                                    expected=batch*h*w+max(0,min(h-1,y*sh+ky-pt))*w+max(0,min(w-1,x*sw+kx-pl))
                                    local=f['local_row']+ky*width+x*sw+kx
                                    if cells.get(local)!=expected:raise ValueError('retained halo source correspondence failed')
                    else:raise ValueError('unknown producer epoch effect')
        visit(self.kernel.body,{})
        if len(outputs)!=o.batches*oh:raise ValueError('producer epoch misses original pooled outputs')
        return dict(schema='merlin.raw_producer_pool_epochs.v1',
            original_kernel_sha256=o.kernel.digest(),kernel_sha256=self.kernel.digest(),
            output_rows_per_epoch=self.output_rows,temporary_rows=self.temporary_rows,
            staged_producer_pixels=staged,original_staged_producer_pixels=o.batches*oh*kh*width,
            output_strips=len(outputs),source_correspondence='every original clamped pool operand checked',
            completion='every product and pooled store completes before ACC reuse',
            timing=None,target_execution=False)


def plan_producer_pool_epochs(original):
    if type(original) is not StripedSpatialMax:
        raise ValueError('producer epochs require a checked canonical spatial strip')
    original.revalidate()
    kh,width=original.stripe_hw;sh=original.source.stride[0]
    rows=min(original.source.output_hw[0],1+(original.capacity_rows//width-kh)//sh)
    result=ProducerPoolEpochs(original,rows,_kernel(original,rows))
    result.revalidate()
    return result


def emit_epoch_producer_tile(schedule, builder, lower, complete, *, output_row,
                             output_column, rows, columns, accumulator_row):
    """Consume the library compute epoch under a bounded ACC translation.

    The source checker above owns the translated physical stage interval. All
    ACC effects share one affine translation, and a full leaf completion occurs
    before its interval changes. Scratchpad addresses/configuration and original
    source K coordinates are unchanged. No ordering premise is inferred here.
    """
    from merlin.compile.scheduling.window_transfer import (
        ConvolutionOutputTile, emit_convolution_compute_epoch_column_block)
    schedule.verify_current()
    def mapped(kind,fields):
        if 'accumulator_row' in fields:
            fields=dict(fields,accumulator_row=builder.op('add',accumulator_row,fields['accumulator_row']))
        lower('finish_raw_tile' if kind=='store' else kind,fields)
    receipt=emit_convolution_compute_epoch_column_block(schedule.source.plan,builder,
        mapped,complete,output_row=output_row,rows=rows,
        output_tiles=(ConvolutionOutputTile(output_column,columns,0),),
        law=schedule.retention_law,issue_law=schedule.issue_law,
        epoch_law=schedule.epoch_law,compute_pairs=schedule.choice.compute_pairs)
    return dict(receipt,accumulator_translation='same bounded stage offset for all ACC effects; full fragment completion',
        checked_schedule_sha256=schedule.identity)
