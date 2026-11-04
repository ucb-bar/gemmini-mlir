"""Explicit source-bound tap descriptor preparation before the CI loop.

Only index/control arithmetic moves. Original zero/data bands, private LHS
ownership, channel-tail widths and complete-K/N ordering retain their meaning.
The common emitter must consume prepare_tap outside the input-channel loop.
"""
from dataclasses import dataclass,asdict
from hashlib import sha256
import json
from . import isa
from .conv_spatial_panels import _Prepared,_prepare,_emit_prepared
from .conv_transfer_runs import plan_transfer_runs,_binding
from merlin.compile.scheduling.affine_gather_runs import emit_convolution_gather_runs


def _source_binding(prepared):
    return sha256(json.dumps(dict(index_binding=_binding(prepared),
        attributes=prepared.attributes,rhs_pitch=prepared.rhs_pitch,output_pitch=prepared.output_pitch,
        zero_seed=prepared.zero_seed,epoch=asdict(prepared.epoch),ordering=prepared.ordering),
        sort_keys=True,separators=(',',':')).encode()).hexdigest()

def _same(a,b):return a is b or type(a)is int and type(b)is int and a==b

def _fingerprint(descriptors):
    def identity(v):return ('int',v)if type(v)in(int,bool)else('ssa',id(v))
    return tuple(tuple((k,identity(v))for k,v in d)for d in descriptors)

@dataclass(frozen=True)
class TapFragments:
    product:object
    coordinates:tuple
    descriptors:tuple
    fingerprint:tuple

    def __call__(self,prepared,builder,first,count,ky,kx,channel,depth,transfer,*,origins=None):
        self.product.revalidate(prepared)
        if (len(self.coordinates)!=6 or not all(_same(a,b)for a,b in zip(self.coordinates,(builder,first,count,ky,kx,origins)))
                or _fingerprint(self.descriptors)!=self.fingerprint):
            raise ValueError('prepared tap descriptor or invocation changed')
        ci=prepared.source.plan.window.input_shape[-1]
        if type(depth)is not int or not 0<depth<=isa.DIM:
            raise ValueError('prepared tap requires original primitive channel width')
        if type(channel)is int and not 0<=channel<=ci-depth:
            raise ValueError('prepared tap leaves original channel domain')
        def issue_group(descriptors):
            for descriptor in descriptors:
                d=dict(descriptor);d.pop('predicate')
                def issue(d=d):
                    payload=dict(d,columns=depth)
                    if not d['zero']:payload['first_element']=builder.op('add',d['first_element'],channel)
                    transfer(payload)
                builder.branch(builder.cmp('ult',0,d['rows']),issue,lambda:None)
        if self.product.transfers:
            separate=dict(self.descriptors[0])['predicate']
            normal=tuple(d for d in self.descriptors if _same(dict(d)['predicate'],separate))
            joined=tuple(d for d in self.descriptors if not _same(dict(d)['predicate'],separate))
            builder.branch(dict(joined[0])['predicate'],lambda:issue_group(joined),lambda:issue_group(normal))
        else:issue_group(self.descriptors)

@dataclass(frozen=True)
class ConvRowRecurrence:
    prepared:object
    binding_sha256:str
    transfers:object

    def revalidate(self,prepared):
        if type(prepared)is not _Prepared or prepared is not self.prepared or _source_binding(prepared)!=self.binding_sha256:
            raise ValueError('tap preparation changed its exact prepared source/allocation')
        if self.transfers != plan_transfer_runs(prepared):
            raise ValueError('tap preparation changed its joined transfer product')
        if self.transfers is not None:self.transfers.revalidate(prepared)

    def __call__(self,*args,**kwargs):
        raise ValueError('tap product requires common prepare_tap protocol before CI emission')

    def prepare_tap(self,prepared,builder,first,count,ky,kx,origins):
        self.revalidate(prepared)
        w=prepared.source.plan.window
        batch,height,width,ci=w.input_shape;pt,pl,pb,pr=w.padding
        ph,pw=height+pt+pb,width+pl+pr;sh,sw=w.stride;dh,dw=w.dilation
        if type(count)is not int or not 0<count<=isa.DIM:raise ValueError('tap preparation row bound changed')
        pixels=batch*w.output_spatial[0]*w.output_spatial[1]
        if type(first)is int and not 0<=first<=pixels-count:raise ValueError('tap preparation leaves output domain')
        if any(type(i)is int and not 0<=i<n for i,n in ((ky,w.kernel[0]),(kx,w.kernel[1]))):
            raise ValueError('tap preparation leaves kernel domain')
        original_origins=origins
        if origins is None:
            origins=tuple((o,l,r,builder.op('udiv',o,ph*pw*ci),
                builder.op('urem',builder.op('udiv',o,pw*ci),ph),
                builder.op('urem',builder.op('udiv',o,ci),pw))
                for o,l,r in emit_convolution_gather_runs(prepared.gather,builder,first,count))
        descriptors=[]
        joined=builder.cmp('eq',kx,self.transfers.horizontal_identity_tap)if self.transfers else builder.cmp('eq',0,1)
        separate=builder.cmp('eq',builder.select(joined,0,1),1)
        def add(zero,physical,local,rows,stride,predicate):
            descriptors.append(tuple(dict(zero=zero,first_element=physical,
                local_row=builder.op('add',prepared.group.lhs_row(0),local),rows=rows,
                row_stride_elements=stride,predicate=predicate).items()))
        # Same original row fragments. Invalid vertical rows are a single zero
        # prefix; an unused wrapped address never reaches pointer construction.
        for origin,local,rows,image,y,x in origins:
            vy=builder.op('add',y,builder.op('mul',ky,dh));vx=builder.op('add',x,builder.op('mul',kx,dw))
            valid=builder.op('and',builder.cmp('ule',pt,vy),builder.cmp('ult',vy,pt+height),'i1')
            def before(limit):
                delta=builder.select(builder.cmp('ult',vx,limit),builder.op('sub',limit,vx),0)
                n=builder.op('udiv',builder.op('add',delta,sw-1),sw)
                return builder.select(builder.cmp('ult',n,rows),n,rows)
            prefix=builder.select(valid,before(pl),rows);end=builder.select(valid,before(pl+width),rows)
            ix=builder.op('sub',builder.op('add',vx,builder.op('mul',prefix,sw)),pl)
            physical=builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',image,height),
                builder.op('sub',vy,pt)),width*ci),builder.op('mul',ix,ci))
            add(True,0,local,prefix,sw*ci,separate)
            add(False,physical,builder.op('add',local,prefix),builder.op('sub',end,prefix),sw*ci,separate)
            add(True,0,builder.op('add',local,end),builder.op('sub',rows,end),sw*ci,separate)
        if self.transfers:
            # Reproduce existing joined identity-tap map, split only at images.
            tap_y=builder.op('mul',ky,dh)
            def before(limit):
                delta=builder.select(builder.cmp('ult',tap_y,limit),builder.op('sub',limit,tap_y),0)
                return builder.select(builder.cmp('ult',delta,height),delta,height)
            valid_start=builder.op('mul',before(pt),width);valid_end=builder.op('mul',before(height+pt),width)
            fragments=((first,0,count),)if batch==1 else emit_convolution_gather_runs(self.transfers.image_gather,builder,first,count)
            for position,local,rows in fragments:
                image=0 if batch==1 else builder.op('udiv',position,height*width)
                spatial=position if batch==1 else builder.op('urem',position,height*width)
                def clip(endpoint):
                    delta=builder.select(builder.cmp('ult',spatial,endpoint),builder.op('sub',endpoint,spatial),0)
                    return builder.select(builder.cmp('ult',delta,rows),delta,rows)
                prefix,end=clip(valid_start),clip(valid_end)
                pixel=builder.op('sub',builder.op('add',builder.op('add',spatial,prefix),builder.op('mul',tap_y,width)),pt*width)
                physical=builder.op('mul',builder.op('add',builder.op('mul',image,height*width),pixel),ci)
                add(True,0,local,prefix,ci,joined)
                add(False,physical,builder.op('add',local,prefix),builder.op('sub',end,prefix),ci,joined)
                add(True,0,builder.op('add',local,end),builder.op('sub',rows,end),ci,joined)
        frozen=tuple(descriptors)
        return TapFragments(self,(builder,first,count,ky,kx,original_origins),frozen,_fingerprint(frozen))

    def receipt(self):
        return dict(schema='authored_conv_tap_preparation.v1',source_sha256=self.prepared.source.identity,
            binding_sha256=self.binding_sha256,joined_transfer_product=self.transfers.receipt()if self.transfers else None,
            transformation='prepare exact zero/data band descriptors once per KH/KW before CI loop',
            completion='unchanged original epoch and LHS/ACC ownership',
            authority='explicit experimental index preparation only; no performance qualification')


def plan_row_recurrence(prepared):
    if type(prepared)is not _Prepared:return None
    prepared.gather.__post_init__();prepared.group.__post_init__()
    return ConvRowRecurrence(prepared,_source_binding(prepared),plan_transfer_runs(prepared))


def lower_row_recurrence(ctx,ifm,rhs,dst,attributes):
    prepared=_prepare(ctx,ifm,rhs,dst,attributes,allow_wide=True)
    product=plan_row_recurrence(prepared)
    if product is None:return None
    product.revalidate(prepared)
    emitted=_emit_prepared(ctx,ifm,rhs,dst,prepared,trial_mode='affine_uniform_stores',input_fragment_emitter=product)
    if not emitted or emitted['source_sha256']!=prepared.source.identity:raise ValueError('tap product not consumed')
    receipt=dict(product.receipt(),emitted=emitted)
    ctx.metadata.setdefault('convolution_row_recurrence',[]).append(receipt)
    return receipt
