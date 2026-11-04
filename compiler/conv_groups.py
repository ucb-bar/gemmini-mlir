"""Exact compact convolution gathers through a virtual integer-zero halo.

The shared linear gather splits output-row boundaries. This adapter maps each
resulting virtual input rectangle back to original NHWC runs, splitting only
at the two horizontal halo boundaries. It emits no host payload arithmetic or
whole-image buffer. Original and zero-source segments write disjoint LHS rows.
"""
from dataclasses import dataclass
from merlin.compile.scheduling.window_transfer import Window2D


@dataclass(frozen=True)
class CompactConvolutionRuns:
    original: Window2D
    edge: int

    def __post_init__(self):
        if type(self.original) is not Window2D or type(self.edge) is not int or self.edge <= 0:
            raise ValueError('compact convolution requires its original typed window and array edge')
        self.original.__post_init__()
        if self.original.stride[1] != 1:
            raise ValueError('compact halo adapter currently requires unit horizontal stride')
        if self.virtual.output_spatial != self.original.output_spatial:
            raise ValueError('compact virtual halo changed original output coverage')

    @property
    def virtual(self):
        b,h,w,c = self.original.input_shape
        t,l,d,r = self.original.padding
        return Window2D((b,h+t+d,w+l+r,c), self.original.kernel,
            self.original.stride,self.original.dilation)

    def project(self, first, rows, columns):
        """Independent integer projection of one actual virtual DMA rectangle."""
        b,h,w,ci=self.original.input_shape
        pt,pl,pb,pr=self.original.padding
        ph,pw=h+pt+pb,w+pl+pr
        image,rem=divmod(first,ph*pw*ci)
        y,rem=divmod(rem,pw*ci)
        x,c=divmod(rem,ci)
        if not (0<=image<b and 0<rows<=self.edge and x+rows<=pw
                and 0<columns<=ci-c):
            raise ValueError('virtual source run exceeds exact original padded view')
        result=[]
        for used in range(rows):
            iy,ix=y-pt,x+used-pl
            source=(((image*h+iy)*w+ix)*ci+c if 0<=iy<h and 0<=ix<w else None)
            if result and ((source is None and result[-1][0] is None)
                    or (source is not None and result[-1][0] is not None
                        and source==result[-1][0]+result[-1][2]*ci)):
                base,local,count=result[-1];result[-1]=(base,local,count+1)
            else:
                result.append((source,used,1))
        if len(result)>3:
            raise ValueError('source rectangle is not an exact three-segment halo run')
        return tuple(result)

    def emit_load(self, builder, fields, issue):
        """Lower the shared gather's actual virtual source address and length."""
        b,h,w,ci=self.original.input_shape
        pt,pl,pb,pr=self.original.padding
        ph,pw=h+pt+pb,w+pl+pr
        if fields['zero'] or fields['row_stride_elements'] != ci:
            raise ValueError('compact adapter requires exact shared valid unit-stride gathers')
        op=builder.op
        choose=builder.select
        def minimum(a,b):
            return choose(builder.cmp('ult',a,b),a,b)
        address=fields['first_element']
        image=op('udiv',address,ph*pw*ci)
        y=op('urem',op('udiv',address,pw*ci),ph)
        x=op('urem',op('udiv',address,ci),pw)
        c=op('urem',address,ci)
        rows=fields['rows']
        def segment(first,local,count,zero):
            def emit():
                issue('load_input',dict(fields,first_element=first,
                    local_row=op('add',fields['local_row'],local),rows=count,
                    row_stride_elements=0 if zero else ci,zero=zero))
            builder.branch(builder.cmp('ult',0,count),emit,lambda:None)
        def valid_y():
            left=choose(builder.cmp('ult',x,pl),minimum(rows,op('sub',pl,x)),0)
            begin=op('add',x,left)
            remaining=op('sub',rows,left)
            valid=choose(builder.cmp('ult',begin,pl+w),
                minimum(remaining,op('sub',pl+w,begin)),0)
            source=op('add',op('mul',op('add',op('mul',image,h),op('sub',y,pt)),w),
                op('sub',begin,pl))
            source=op('add',op('mul',source,ci),c)
            segment(0,0,left,True)
            segment(source,left,valid,False)
            segment(0,op('add',left,valid),op('sub',remaining,valid),True)
        in_y=op('and',builder.cmp('ule',pt,y),builder.cmp('ult',y,pt+h),'i1')
        builder.branch(in_y,valid_y,lambda:segment(0,0,rows,True))

    def record(self):
        return dict(schema='authored.compact_convolution_runs.v1',
            original_window=repr(self.original),virtual_window=repr(self.virtual),
            mapping='virtual (image,y,x,c) -> original (image,y-top,x-left,c) or integer zero',
            physical_staging_rows=self.edge,max_segments_per_shared_fragment=3,
            payload_host_operations=0,whole_image_scratch_bytes=0,
            completion='unchanged shared linear gather RAW/WAR dependencies and output-block last-use drain')
