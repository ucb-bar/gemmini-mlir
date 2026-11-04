"""Checked whole-panel specialization of padded convolution source gathers.

The first shared fragment must cover the entire nonempty M panel. Original
output-row/image boundaries then prove all rows share its image and y; source
window extrema prove every original dilated tap is valid. Border panels retain
all original zero/valid/zero effects through the incumbent emitter.
"""
from dataclasses import asdict, dataclass
from .conv_reduction_peeling import _digest
from merlin.compile.scheduling.affine_gather_runs import ConvolutionGatherRuns
from merlin.compile.scheduling.window_transfer import Window2D


@dataclass(frozen=True)
class InteriorConvolutionGather:
    window: Window2D
    gather: ConvolutionGatherRuns
    lhs_row: int

    def validate(self):
        if type(self.window)is not Window2D or type(self.gather)is not ConvolutionGatherRuns:
            raise ValueError('interior specialization requires original shared source artifacts')
        w=self.window;g=self.gather
        w.__post_init__();g.__post_init__()
        b,h,width,ci=w.input_shape;pt,pl,pb,pr=w.padding
        if (not any(w.padding) or g.input_shape!=(b,h+pt+pb,width+pl+pr,ci)
                or g.output_spatial!=(b,*w.output_spatial)
                or (g.stride,g.dilation,g.kernel)!=(w.stride,w.dilation,w.kernel)
                or type(self.lhs_row)is not int or self.lhs_row<0):
            raise ValueError('interior specialization must bind the exact padded source window')

    def predicate(self,builder,count,coordinates):
        """Evaluate once outside K using the actual shared first fragment."""
        self.validate()
        _,y,x,_,rows=coordinates[0]
        _,h,width,_=self.window.input_shape;pt,pl,_,_=self.window.padding
        kh,kw=self.window.kernel;dh,dw=self.window.dilation;_,sw=self.window.stride
        tests=(builder.cmp('eq',rows,count),builder.cmp('ule',pt,y),
            builder.cmp('ult',builder.op('add',y,(kh-1)*dh),pt+h),
            builder.cmp('ule',pl,x),
            builder.cmp('ult',builder.op('add',x,(count-1)*sw+(kw-1)*dw),pl+width))
        result=tests[0]
        for test in tests[1:]:result=builder.op('and',result,test,'i1')
        return result

    def emit(self,builder,first,count,ky,kx,channel,depth,transfer,*,coordinates):
        """Consume the proved first fragment as one valid source DMA band."""
        self.validate()
        image,y,x,local,_=coordinates[0]
        _,h,width,ci=self.window.input_shape;pt,pl,_,_=self.window.padding
        dh,dw=self.window.dilation;_,sw=self.window.stride
        vy=builder.op('sub',builder.op('add',y,builder.op('mul',ky,dh)),pt)
        vx=builder.op('sub',builder.op('add',x,builder.op('mul',kx,dw)),pl)
        physical=builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',image,h),vy),width*ci),
            builder.op('add',builder.op('mul',vx,ci),channel))
        transfer(dict(zero=False,first_element=physical,
            local_row=builder.op('add',self.lhs_row,local),rows=count,
            columns=depth,row_stride_elements=sw*ci))

    def receipt(self):
        self.validate()
        source=dict(window=asdict(self.window),gather=asdict(self.gather),lhs_row=self.lhs_row,
            bands=['zero_prefix','valid_source','zero_suffix'],source_order=['kernel_y','kernel_x','input_channel'])
        transformed=dict(source=source,predicate='first fragment covers nonempty panel; original first/last dilated taps inside original image',
            interior_bands=['one valid source band covering panel'],border_bands=source['bands'],
            predicate_scope='outside complete K traversal',completion='unchanged')
        return dict(mechanism='checked_interior_panel_single_dma_band',source_program=source,
            transformed_program=transformed,source_program_sha256=_digest(source),
            transformed_program_sha256=_digest(transformed),
            pass_order=['shared.ConvolutionGatherRuns','window.interior_panel_specialize','backend.isa'])


def plan_interior_convolution_gather(prepared):
    w=prepared.source.plan.window
    if not any(w.padding):return None
    # Whole primitive panels cannot be interior within a narrower output row.
    # Avoid adding a never-taken producer alternative to those geometries.
    if w.output_spatial[1]<prepared.gather.max_rows:return None
    plan=InteriorConvolutionGather(w,prepared.gather,prepared.group.lhs_row(0))
    plan.validate()
    return plan
