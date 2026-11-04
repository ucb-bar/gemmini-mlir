"""Original convolution taps sharing stride-residue device row panels.

Each panel row contains one original CI-vector, loaded or zeroed once. For
kx=r+q*sx, product lane i reads panel[r][q+i]. Thus the target's ordinary
unit-stride A addressing implements the original dilated-free convolution.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from .window_transfer import CountedConvolutionPlan, ConvolutionComputeEpochLaw, _validate_convolution_load_issue

@dataclass(frozen=True)
class SpatialResidueRuns:
    original: CountedConvolutionPlan
    base: int

    def __post_init__(self):
        p = self.original
        if type(p) is not CountedConvolutionPlan:
            raise ValueError('residue panels require the original typed convolution')
        p.__post_init__()
        ci = p.window.input_shape[-1]
        if (p.linear_rows or p.window.dilation[1] != 1 or not 1 <= ci <= p.channel_tile
                or not 1 <= p.window.stride[1] <= p.window.kernel[1]
                or type(self.base) is not int or self.base < p.lhs_row+p.row_tile
                or self.base < p.rhs_row+p.channel_tile
                or self.base+self.required_rows > p.operand_rows):
            raise ValueError('residue panels require bounded disjoint original CI rows')

    @property
    def phase_rows(self):
        kw, sx = self.original.window.kernel[1], self.original.window.stride[1]
        return tuple(self.original.row_tile+(kw-1-r)//sx for r in range(sx))

    @property
    def required_rows(self):
        return sum(self.phase_rows)

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

    def emit(self, builder, issue, complete, *, output_row, output_column,
             rows, columns, accumulator_row, law, epoch_law, compute_pairs):
        self.__post_init__()
        if type(epoch_law) is not ConvolutionComputeEpochLaw or not callable(getattr(builder,'select',None)):
            raise ValueError('residue panels require a typed epoch law and select builder')
        p, window = self.original, self.original.window
        if (any(type(v) is not int or v < 1 for v in (rows, columns))
                or rows > p.row_tile or columns > p.column_tile):
            raise ValueError("residue rectangle exceeds the original primitive tile")
        before = _validate_convolution_load_issue(p,builder,issue,complete,rows,law,int(p.bias))
        if law.load_issue_budget < self.required_rows+1+int(p.bias):
            raise ValueError("residue panels exceed selected pending load budget")
        epoch_law.require(compute_pairs)
        if any(getattr(law,k)!=getattr(epoch_law,k) for k in
            ('hardware_identity','numerical_policy_identity','source_identity','qualification_identity')):
            raise ValueError('residue panels need the exact original epoch law')
        epoch_before = asdict(epoch_law)
        identity = self.identity
        op=builder.op
        ci=window.input_shape[-1]
        kh,kw=window.kernel
        sx=window.stride[1]
        common=dict(output_row=output_row,output_column=output_column,rows=rows,
                    columns=columns,accumulator_row=accumulator_row)
        if p.bias: issue('initialize_bias',common)
        def run_at(index,n,ky,r):
            batch,height,width,channels=window.input_shape
            oh,ow=window.output_spatial
            sh,sw=window.stride
            dh,_=window.dilation
            top,left,_,_=window.padding
            image=op('udiv',output_row,oh*ow)
            y=op('urem',op('udiv',output_row,ow),oh)
            x=op('urem',output_row,ow)
            iy=op('add',op('sub',op('mul',y,sh),top),op('mul',ky,dh))
            ix=op('add',op('sub',op('mul',op('add',x,index),sw),left),r)
            pred=op('and',builder.cmp('ult',iy,height),builder.cmp('ult',ix,width),'i1')
            pred=op('and',pred,builder.cmp('ult',op('add',ix,(n-1)*sw),width),'i1')
            first=op('mul',op('add',op('mul',op('add',op('mul',image,height),iy),width),ix),ci)
            return dict(predicate=pred,first_element=first,row_stride_elements=sw*ci)
        def ky_body(ky):
            start=self.base
            for r,capacity in enumerate(self.phase_rows):
                count=rows+(kw-1-r)//sx
                def transfer(index,n,local):
                    # The panel extends past the last output start, while
                    # source bounds stay those of the original input image.
                    run=run_at(index,n,ky,r)
                    def load():
                        issue('load_input',dict(first_element=run['first_element'],
                            row_stride_elements=run['row_stride_elements'],local_row=local,
                            rows=n,columns=ci,zero=False))
                    def boundary():
                        def one(i):
                            item=run_at(op('add',index,i),1,ky,r)
                            def effect(zero):
                                issue('load_input',dict(first_element=item['first_element'] if not zero else 0,
                                    row_stride_elements=item['row_stride_elements'] if not zero else 0,
                                    local_row=op('add',local,i),rows=1,columns=ci,zero=zero))
                            builder.branch(item['predicate'],lambda:effect(False),lambda:effect(True))
                        builder.loop(0,n,1,one)
                    builder.branch(run['predicate'],load,boundary)
                # Native transfers are primitive-bounded, including ragged runs.
                whole,tail=divmod(count,p.primitive)
                if whole: builder.loop(0,whole,1,lambda i:transfer(op('mul',i,p.primitive),p.primitive,op('add',start,op('mul',i,p.primitive))))
                if tail: transfer(whole*p.primitive,tail,start+whole*p.primitive)
                start+=capacity
            def kx_body(kx):
                reduction=op('mul',op('add',op('mul',ky,kw),kx),ci)
                phase=op('urem',kx,sx)
                base=self.base
                local=base
                for r,capacity in enumerate(self.phase_rows):
                    if r: local=builder.select(builder.cmp('eq',phase,r),base,local)
                    base+=capacity
                lhs=op('add',local,op('udiv',kx,sx))
                issue('load_weight',dict(reduction_row=reduction,output_column=output_column,
                    local_row=p.rhs_row,rows=ci,columns=columns))
                first=op('and',builder.cmp('eq',ky,0),builder.cmp('eq',kx,0),'i1')
                fields=dict(common,lhs_row=lhs,rhs_row=p.rhs_row,depth=ci,
                            first_reduction=first,bias=p.bias)
                issue('compute',fields)
                pair=op('add',op('add',op('mul',ky,kw),kx),1)
                boundary=op('and',builder.cmp('eq',op('urem',pair,compute_pairs),0),
                    builder.cmp('ult',pair,kh*kw),'i1')
                builder.branch(boundary,lambda:complete('compute',fields),lambda:None)
            builder.loop(0,kw,1,kx_body)
        builder.loop(0,kh,1,ky_body)
        issue('store',common)
        complete('store',common)
        self.__post_init__()
        if identity!=self.identity or before!=(p.identity,asdict(law)) or epoch_before!=asdict(epoch_law):
            raise ValueError('residue source or target contract changed during emission')
        return dict(schema='merlin.spatial_residue_runs.v1',identity=identity,
            source_identity=p.identity,private_operand_rows=self.required_rows,
            mapping='kx=r+q*sx: lane i reads phase[r][q+i]',host_payload_access=False)
