"""Checked source-domain specialization for convolution input transfers.

First-reduction peeling is deliberately absent: biased products already encode
constant accumulation, and their unused first-step predicate is removed by LLVM.
The useful transformation here consumes the shared exact valid-window gather
and emits its one nonzero source band directly, retaining its fragment guard.
"""
from dataclasses import asdict,dataclass
import hashlib
import json
from merlin.compile.scheduling.affine_gather_runs import ConvolutionGatherRuns,emit_convolution_gather_runs
from merlin.compile.scheduling.window_transfer import Window2D


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class ValidConvolutionGather:
    window: Window2D
    gather: ConvolutionGatherRuns
    lhs_row: int

    def validate(self):
        if type(self.window)is not Window2D or type(self.gather)is not ConvolutionGatherRuns:
            raise ValueError('valid gather requires original shared window/gather artifacts')
        self.window.__post_init__();self.gather.__post_init__()
        w=self.window;g=self.gather
        if (any(w.padding) or g.input_shape!=w.input_shape
                or g.output_spatial!=(w.input_shape[0],*w.output_spatial)
                or (g.stride,g.dilation,g.kernel)!=(w.stride,w.dilation,w.kernel)
                or type(self.lhs_row)is not int or self.lhs_row<0):
            raise ValueError('valid gather must describe the complete original unpadded window')
        # For each axis, the final output coordinate plus final dilated tap
        # remains inside the original image. Shared gather fragments split at
        # every original output row/image boundary and guard inactive lengths.
        for size,outputs,stride,dilation,kernel in zip(w.input_shape[1:3],w.output_spatial,w.stride,w.dilation,w.kernel):
            if (outputs-1)*stride+(kernel-1)*dilation>=size:
                raise ValueError('original output/tap domain exceeds source image')

    def programs(self):
        self.validate()
        source=dict(window=asdict(self.window),gather=asdict(self.gather),lhs_row=self.lhs_row,
            bands=['zero_prefix','valid_source','zero_suffix'],row_guard='0 <= tap_y < input_height',
            fragment_guard='rows > 0',source_order=['kernel_y','kernel_x','input_channel'])
        transformed=dict(window=source['window'],gather=source['gather'],lhs_row=self.lhs_row,
            bands=[dict(kind='valid_source',start_row=0,rows='fragment_rows',
                source='original fragment origin plus original dilated tap/channel')],
            fragment_guard='rows > 0',source_order=source['source_order'])
        return source,transformed

    def emit(self,builder,first,count,ky,kx,channel,depth,transfer,*,coordinates=None):
        """Consume the shared fragment product and specialized single-band map."""
        self.validate();w=self.window;_,height,width,ci=w.input_shape
        dh,dw=w.dilation
        if coordinates is None:
            tap=builder.op('add',builder.op('mul',ky,dh*width*ci),
                builder.op('add',builder.op('mul',kx,dw*ci),channel))
            fragments=((builder.op('add',origin,tap),local,rows)
                for origin,local,rows in emit_convolution_gather_runs(self.gather,builder,first,count))
        else:
            fragments=((builder.op('add',builder.op('mul',
                builder.op('add',builder.op('mul',image,height),builder.op('add',y,builder.op('mul',ky,dh))),width*ci),
                builder.op('add',builder.op('mul',builder.op('add',x,builder.op('mul',kx,dw)),ci),channel)),local,rows)
                for image,y,x,local,rows in coordinates)
        for origin,local,rows in fragments:
            def active(origin=origin,local=local,rows=rows):
                transfer(dict(zero=False,first_element=origin,
                    local_row=builder.op('add',self.lhs_row,local),rows=rows,columns=depth,
                    row_stride_elements=self.gather.source_row_stride))
            builder.branch(builder.cmp('ult',0,rows),active,lambda:None)

    def receipt(self):
        source,transformed=self.programs()
        return dict(mechanism='exact_valid_window_single_dma_band',source_program=source,
            transformed_program=transformed,source_program_sha256=_digest(source),
            transformed_program_sha256=_digest(transformed),
            pass_order=['shared.ConvolutionGatherRuns','window.valid_domain_specialize','backend.isa'],
            proof='original padding zero; exact shared output shape; final stride/dilation tap inside image; original row/image fragment boundaries',
            effects='same original nonempty valid DMA; zero prefix/suffix proven empty; original fragment guard',
            completion='unchanged all preparation/reduction/readout/last-use completion boundaries')


def plan_valid_convolution_gather(prepared):
    if any(prepared.source.plan.window.padding):return None
    plan=ValidConvolutionGather(prepared.source.plan.window,prepared.gather,prepared.group.lhs_row(0))
    plan.validate()
    return plan
