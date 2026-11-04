"""Device-only contiguous horizontal reduction with source-exact edge fallback.

The HWCF row index remains ky*(KW*CI)+chunk. Each NHWC load row begins
SX*CI bytes after the preceding output lane; no payload is read by the host.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from .spatial_residue_runs import SpatialResidueRuns
from .window_transfer import ConvolutionComputeEpochLaw, _validate_convolution_load_issue


@dataclass(frozen=True)
class SpatialHorizontalFlat:
    fallback: SpatialResidueRuns

    def __post_init__(self):
        if type(self.fallback) is not SpatialResidueRuns:
            raise ValueError('horizontal flattening requires checked original boundary runs')
        self.fallback.__post_init__()
        p = self.original
        if (p.window.dilation[1] != 1 or p.channel_tile > p.primitive
                or p.lhs_row+p.row_tile > p.rhs_row
                or p.rhs_row+p.primitive > self.fallback.base):
            raise ValueError('horizontal flattening requires disjoint primitive LHS/RHS rows')

    @property
    def original(self):
        return self.fallback.original

    @property
    def required_rows(self):
        return self.fallback.required_rows

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

    def emit(self, builder, issue, complete, *, output_row, output_column,
             rows, columns, accumulator_row, law, epoch_law, compute_pairs):
        self.__post_init__()
        p, w = self.original, self.original.window
        if type(epoch_law) is not ConvolutionComputeEpochLaw:
            raise ValueError('horizontal flattening requires checked original epochs')
        before = _validate_convolution_load_issue(p,builder,issue,complete,rows,law,int(p.bias))
        epoch_law.require(compute_pairs)
        if any(getattr(law,k)!=getattr(epoch_law,k) for k in
            ('hardware_identity','numerical_policy_identity','source_identity','qualification_identity')):
            raise ValueError('horizontal flattening requires the exact original epoch law')
        epoch_before, identity = asdict(epoch_law), self.identity
        if not 1 <= columns <= p.column_tile:
            raise ValueError('horizontal flattening exceeds original column tile')
        op = builder.op
        _,height,width,ci = w.input_shape
        oh,ow = w.output_spatial
        kh,kw = w.kernel
        sy,sx = w.stride
        top,left,_,_ = w.padding
        image = op('udiv',output_row,oh*ow)
        y = op('urem',op('udiv',output_row,ow),oh)
        x = op('urem',output_row,ow)
        ix = op('sub',op('mul',x,sx),left)
        last = op('add',ix,(rows-1)*sx+kw-1)
        interior = op('and',builder.cmp('ult',ix,width),builder.cmp('ult',last,width),'i1')
        common=dict(output_row=output_row,output_column=output_column,rows=rows,
                    columns=columns,accumulator_row=accumulator_row)
        span=kw*ci
        chunks=(span+p.primitive-1)//p.primitive
        total=kh*chunks
        def flat():
            if p.bias: issue('initialize_bias',common)
            def ky_body(ky):
                iy=op('add',op('sub',op('mul',y,sy),top),op('mul',ky,w.dilation[0]))
                valid=builder.cmp('ult',iy,height)
                first=op('mul',op('add',op('mul',op('add',op('mul',image,height),iy),width),ix),ci)
                def chunk(c,depth):
                    delta=op('mul',c,p.primitive)
                    def load(zero):
                        issue('load_input',dict(first_element=0 if zero else op('add',first,delta),
                            row_stride_elements=0 if zero else sx*ci,local_row=p.lhs_row,
                            rows=rows,columns=depth,zero=zero))
                    builder.branch(valid,lambda:load(False),lambda:load(True))
                    reduction=op('add',op('mul',ky,span),delta)
                    issue('load_weight',dict(reduction_row=reduction,output_column=output_column,
                        local_row=p.rhs_row,rows=depth,columns=columns))
                    fields=dict(common,lhs_row=p.lhs_row,rhs_row=p.rhs_row,depth=depth,
                        first_reduction=op('and',builder.cmp('eq',ky,0),builder.cmp('eq',c,0),'i1'),bias=p.bias)
                    issue('compute',fields)
                    pair=op('add',op('add',op('mul',ky,chunks),c),1)
                    boundary=op('and',builder.cmp('eq',op('urem',pair,compute_pairs),0),builder.cmp('ult',pair,total),'i1')
                    builder.branch(boundary,lambda:complete('compute',fields),lambda:None)
                whole,tail=divmod(span,p.primitive)
                if whole: builder.loop(0,whole,1,lambda c:chunk(c,p.primitive))
                if tail: chunk(whole,tail)
            builder.loop(0,kh,1,ky_body)
            issue('store',common)
            complete('store',common)
        builder.branch(interior,flat,lambda:self.fallback.emit(builder,issue,complete,
            **common,law=law,epoch_law=epoch_law,compute_pairs=compute_pairs))
        self.__post_init__()
        if identity != self.identity or before != (p.identity,asdict(law)) or epoch_before != asdict(epoch_law):
            raise ValueError('horizontal flattening source or contracts changed during emission')
        return dict(schema='merlin.spatial_horizontal_flat.v1',identity=identity,
            source_identity=p.identity,host_payload_access=False,
            mapping='interior: original ky,kx,ci flattened within each ky; boundary: original residue runs',
            interior_pairs=total,boundary_pairs=kh*kw)
