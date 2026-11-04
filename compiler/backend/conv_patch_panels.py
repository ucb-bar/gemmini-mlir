"""Non-overlapping NHWC patches through checked contiguous row fragments.

Only the source index space changes. Each original KH by KW patch becomes KH
rows of KW*CI channels; neither input nor weights are packed or copied. The
shared resident-panel owner, epoch law and terminal encoder emit the device
contraction and its original readout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import prod

from merlin.compile.scheduling.affine_gather_runs import emit_convolution_gather_runs
from . import conv_spatial_panels


@dataclass(frozen=True)
class PatchSource:
    input_shape: tuple
    kernel: tuple
    output_hw: tuple

    def __post_init__(self):
        b,h,w,ci=self.input_shape
        kh,kw,kci,co=self.kernel
        if (any(type(x)is not int or x<=0 for x in (*self.input_shape,*self.kernel))
                or ci!=kci or self.output_hw!=(h//kh,w//kw)
                or min(self.output_hw)<=0):
            raise ValueError('invalid original non-overlapping patch source')

    @property
    def virtual_shape(self):
        b,h,_,ci=self.input_shape
        return (b,h,self.output_hw[1],self.kernel[1]*ci)

    @property
    def virtual_kernel(self):
        kh,kw,ci,co=self.kernel
        return (kh,1,kw*ci,co)

    @property
    def identity(self):
        return sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

    def source_element(self,m,k):
        """Exact original flattened source coordinate, also usable by checks."""
        self.__post_init__()
        b,h,w,ci=self.input_shape;kh,kw,_,co=self.kernel;oh,ow=self.output_hw
        if not 0<=m<b*oh*ow or not 0<=k<kh*kw*ci:
            raise ValueError('patch coordinate outside original contraction')
        image,pixel=divmod(m,oh*ow);y,x=divmod(pixel,ow)
        ky,remainder=divmod(k,kw*ci)
        return ((image*h+y*kh+ky)*w+x*kw)*ci+remainder


class _SourceContext:
    def __init__(self,ctx,ifm,source):
        self._ctx,self._ifm,self._source=ctx,ifm,source
    def spec(self,name):
        spec=self._ctx.spec(name)
        return dict(spec,shape=self._source.virtual_shape) if name==self._ifm else spec
    def __getattr__(self,name):
        return getattr(self._ctx,name)


def _prepare(ctx,ifm,rhs,dst,attributes):
    """Pure admission, including storage, aliases and exact terminal contract."""
    try:
        attrs=dict(attributes or {})
        shape=tuple(ctx.spec(ifm)['shape']);kernel=tuple(attrs['kernel'])
        if len(shape)!=4 or len(kernel)!=4:return None
        kh,kw,ci,co=kernel
        padding=tuple(attrs.get('padding',(0,0,0,0)))
        if len(padding)==2:padding=(*padding,*padding)
        if (len(padding)!=4 or any(padding)
                or tuple(attrs.get('stride',(1,1)))!=(kh,kw)
                or tuple(attrs.get('dilation',(1,1)))!=(1,1)
                or kh*kw<=1):return None
        source=PatchSource(shape,kernel,(shape[1]//kh,shape[2]//kw))
        # Flattened K/N order is identical; the original RHS remains untouched.
        if tuple(ctx.spec(rhs)['shape'])!=(kh*kw*ci,co):return None
        if prod(ctx.spec(dst)['shape'])!=shape[0]*prod(source.output_hw)*co:return None
        view=_SourceContext(ctx,ifm,source)
        transformed=dict(attrs,kernel=source.virtual_kernel,stride=(kh,1),padding=(0,0,0,0))
        prepared=conv_spatial_panels._prepare(view,ifm,rhs,dst,transformed,allow_wide=True)
        if prepared is None:return None
        return source,view,transformed,prepared
    except(KeyError,TypeError,ValueError,OverflowError,ZeroDivisionError):
        return None


def can_lower(ctx,ifm,rhs,dst,attributes):
    return _prepare(ctx,ifm,rhs,dst,attributes)is not None


def _input_fragments(source,prepared,builder,first,count,ky,kx,channel,depth,transfer):
    """Project each same-original-row gather fragment without halo branches."""
    source.__post_init__()
    _,_,width,ci=source.input_shape
    virtual_pitch=source.output_hw[1]*source.kernel[1]*ci
    physical_pitch=width*ci
    for origin,local,rows in emit_convolution_gather_runs(prepared.gather,builder,first,count):
        def active(origin=origin,local=local,rows=rows):
            row=builder.op('udiv',origin,virtual_pitch)
            within=builder.op('urem',origin,virtual_pitch)
            offset=builder.op('add',builder.op('mul',builder.op('add',row,ky),physical_pitch),
                builder.op('add',within,channel))
            transfer(dict(zero=False,first_element=offset,
                local_row=builder.op('add',prepared.group.lhs_row(0),local),
                rows=rows,columns=depth,row_stride_elements=source.kernel[1]*ci))
        builder.branch(builder.cmp('ult',0,rows),active,lambda:None)


def lower(ctx,ifm,rhs,dst,attributes):
    """Return False before effects if source, hardware or epilogue refuses."""
    admission=_prepare(ctx,ifm,rhs,dst,attributes)
    if admission is None:return False
    source,view,transformed,prepared=admission
    def fragments(*args):
        return _input_fragments(source,*args)
    result=conv_spatial_panels.lower(view,ifm,rhs,dst,transformed,
        allow_wide=True,input_fragments=fragments)
    if result is False:raise RuntimeError('patch source admission changed during emission')
    receipt=dict(result,mechanism='nonoverlapping_patch_row_panels',
        original_patch_source=asdict(source),original_source_sha256=source.identity,
        virtual_shape=list(source.virtual_shape),virtual_kernel=list(source.virtual_kernel),
        source_mapping='original row pitch; contiguous KW*CI channel chunks; same-row M fragments',
        payload_copies=0)
    ctx.metadata.setdefault('convolution_patch_panels',[]).append(receipt)
    return receipt


lower_conv_patch_panels=lower
