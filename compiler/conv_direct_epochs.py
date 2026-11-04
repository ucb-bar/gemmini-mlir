"""Bounded direct-source compute epochs; no drain motion across rolling ownership.

This is a new target owner, not a ConvolutionResidentInputRows/DensePanel result.
Its actual counted Kernel drives paired common-encoder effects. All source DMA,
input-ready, last-use, initializer and readout drains remain unchanged.
"""
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
from merlin.compile.tuning_knobs import knob
from merlin.sched.ir import Kernel,Loop,Call,Const,Var
from merlin.compile.scheduling.llvm_program import emit_index_program
from conv_policy import completion_laws
from merlin.compile.scheduling.rolling_window import emit_rolling_window
import conv_rolling_binding as direct
from isa import ACC,ACCUMULATE,FULL,GARBAGE,DIM
VALUES=('off','pairs16')
DECLARATION=dict(values=list(VALUES),mechanism='selected-fact bounded compute epochs over actual direct-source retained planes',note='Default off; source/load/ring/readout drains retained; custom actual Kernel owner, not a borrowed convolution-resident-input owner. Timing untested.')
def selected():return knob('conv_direct_epochs',default='off',values=VALUES,root=Path(__file__).parent)
def _kernel(p):
 w=p.source.window;co=p.source.channels
 effects=[]
 for ni,column in enumerate(range(0,co,DIM)):
  args=tuple(zip(('ky','kx','ni','column','count'),(Var('ky'),Var('kx'),Const(ni),Const(column),Const(min(DIM,co-column)))))
  effects.extend((Call('pair',args),Call('boundary',args)))
 return Kernel('direct_original_source_compute_epochs',(),(Loop('ky',w.kernel[0],(Loop('kx',w.kernel[1],tuple(effects)),)),))
@dataclass(frozen=True)
class EpochPlan:
 direct:object
 kernel:object
 law:object
 evidence:object
 def revalidate(self):
  self.direct.revalidate();self.law.require(16)
  selected=completion_laws(self.direct.original,self.direct.original.identity,self.direct.rhs.n_tiles,mode='pairs16')
  if selected is None or self.law!=selected[1] or self.evidence!=selected[3]:raise ValueError('epoch owner differs from actual selected ordering facts')
  if self.law.source_identity!=self.direct.original.identity or self.kernel!=_kernel(self.direct):raise ValueError('actual direct-source kernel or source binding changed')
  # No descriptor/operand/readout mutation is authorized by a completion pass.
  return self
 def emit(self,builder,lower):
  self.revalidate()
  result=emit_index_program(self.kernel,builder,lower)
  self.revalidate();return result
 def report(self):
  return dict(route='direct_source_retained_kx_planes_pairs16',schema='target.direct_source_compute_epoch.v1',kernel_sha256=self.kernel.digest(),source_sha256=self.direct.original.identity,compute_pairs_bound=16,actual_source_mapping=self.direct.report(),ordering_evidence=self.evidence,consumer='own checked actual counted Kernel; no declaration of compatibility with ConvolutionResidentInputRows',preserved_drains='every DMA, initializer, rolling ready/last-use, each original readout, final whole-call',changed_drains='only consecutive original ky/kx/N paired computes; modulo16 plus full pre-readout completion',timing='UNKNOWN')
def prepare(command,tensors,pointers,*,residents=None):
 p=direct.prepare(command,tensors,pointers,residents=residents)
 laws=completion_laws(p.original,p.original.identity,p.rhs.n_tiles,mode='pairs16')
 if laws is None or laws[1] is None:raise ValueError('selected direct-source compute ordering/progress facts unavailable')
 owner=EpochPlan(p,_kernel(p),laws[1],laws[3]);return owner.revalidate()

def _product(builder,epoch,pointers,output):
 p=epoch.direct;epoch.revalidate()
 p.revalidate();s=p.source;w=s.window;b,h,width,ci=w.input_shape;co=s.channels;oh,ow=w.output_spatial;kh,kw=w.kernel;sy,sx=w.stride;top,left,_,_=w.padding;dtype=s.attrs['output_dtype'];nbytes=4 if dtype=='i32' else 1
 builder.fence();builder.config_ex(dataflow=1);builder.config_st(co*nbytes,activation=int('relu' in s.attrs.get('epilogue',())),scale=float(s.attrs.get('acc_scale',1.)))
 if p.base.initializer:builder.config_ld(0,id=2)
 zero=builder.constant_i32_table([0]*DIM)
 def columns(body):
  full,tail=divmod(co,DIM)
  if full:builder.loop(0,full,1,lambda ni:body(ni,builder.op('mul',ni,DIM),DIM))
  if tail:body(full,full*DIM,tail)
 def rhs_ky(ky):
  def rhs_kx(kx):
   tap=builder.op('add',builder.op('mul',ky,kw),kx)
   def load(ni,column,count):
    local=builder.op('add',p.rhs.rhs_base,builder.op('mul',builder.op('add',builder.op('mul',ni,kh*kw),tap),DIM));off=builder.op('add',builder.op('mul',tap,ci*co),column)
    builder.config_ld(co,id=1);builder.mvin(builder.ptr_offset(pointers[s.rhs],off),local,cols=count,rows=ci,id=1);builder.fence()
   columns(load)
  builder.loop(0,kw,1,rhs_kx)
 builder.loop(0,kh,1,rhs_ky)
 receipts=[]
 def stripe(origin,rows):
  _,rolling,ir=next(t for t in p.templates if t[0]==rows);token=object()
  def lower(kind,f):
   if kind=='begin_transport':return token
   if kind=='stage_row':
    iy=builder.op('sub',f['input_row'],top);image=f['image'];slot=builder.op('udiv',f['cache_byte'],p.slot_pitch*DIM)
    yvalid=builder.cmp('ult',iy,h)
    def tap(kx):
     base=builder.op('add',builder.op('mul',slot,p.slot_pitch),builder.op('mul',kx,DIM));ix=builder.op('sub',builder.op('add',builder.op('mul',origin,sx),kx),left)
     def transfer(x,local,count,valid):
      def load():
       off=builder.op('mul',builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',image,h),iy),width),x),ci)
       builder.config_ld(sx*ci,id=0);builder.mvin(builder.ptr_offset(pointers[s.lhs],off),local,cols=ci,rows=count,id=0);builder.fence()
      def clear():
       builder.config_ld(0,id=0);builder.mvin(zero,local,cols=ci,rows=count,id=0);builder.fence()
      builder.branch(valid,load,clear)
     last=builder.op('add',ix,(rows-1)*sx);valid=builder.op('and',yvalid,builder.op('and',builder.cmp('ult',ix,width),builder.cmp('ult',last,width),'i1'),'i1')
     def boundary():
      def one(r):
       x=builder.op('add',ix,builder.op('mul',r,sx));ok=builder.op('and',yvalid,builder.cmp('ult',x,width),'i1');transfer(x,builder.op('add',base,r),1,ok)
      builder.loop(0,rows,1,one)
     builder.branch(valid,lambda:transfer(ix,base,rows,True),boundary)
    builder.loop(0,kw,1,tap)
   elif kind=='await_transport':builder.fence();return token
   elif kind=='consume':
    y=f['output_y'];image=f['image'];outrow=builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',image,oh),y),ow),origin)
    if p.base.initializer:
     def init(ni,column,count):
      ptr=builder.ptr_offset(pointers[s.bias],builder.op('mul',column,4)) if s.bias else zero
      builder.mvin(ptr,builder.op('or',ACC,builder.op('mul',ni,DIM)),cols=count,rows=rows,id=2);builder.fence()
     columns(init)
    def pair_effect(call,fields):
     ky,kx,ni,column,count=(fields[k] for k in ('ky','kx','ni','column','count'))
     if call=='boundary':
      # The original pair index is tap*N+ni+1. Constant N/ni prove some
      # residues cannot be divisible by16, without issuing host arithmetic.
      from math import gcd
      if (ni+1)%gcd(p.rhs.n_tiles,16):return
      tap=builder.op('add',builder.op('mul',ky,kw),kx)
      index=builder.op('add',builder.op('mul',tap,p.rhs.n_tiles),ni+1)
      builder.branch(builder.cmp('eq',builder.op('urem',index,16),0),builder.fence,lambda:None)
      return
     if call!='pair':raise ValueError('unknown checked actual epoch effect')
     slot=builder.op('urem',builder.op('add',builder.op('mul',y,sy),ky),kh)
     tap=builder.op('add',builder.op('mul',ky,kw),kx)
     lhs=builder.op('add',builder.op('mul',slot,p.slot_pitch),builder.op('mul',kx,DIM))
     rhs=builder.op('add',p.rhs.rhs_base,builder.op('mul',builder.op('add',builder.op('mul',ni,kh*kw),tap),DIM));acc=builder.op('or',ACC,builder.op('mul',ni,DIM))
     def pair(accumulate):
      addr=builder.op('or',acc,ACCUMULATE) if accumulate else acc
      builder.preload(rhs,addr,b_cols=count,b_rows=ci,c_cols=count,c_rows=rows)
      builder.compute(lhs,GARBAGE,a_cols=ci,a_rows=rows,bd_cols=count,bd_rows=rows)
     if p.base.initializer:pair(True)
     else:builder.branch(builder.cmp('eq',tap,0),lambda:pair(False),lambda:pair(True))
    epoch.emit(builder,pair_effect)
    # Complete the final partial epoch before any source readout or ring reuse.
    builder.fence()
    def store(ni,column,count):
     off=builder.op('mul',builder.op('add',builder.op('mul',outrow,co),column),nbytes);addr=builder.op('or',ACC|(FULL if dtype=='i32' else 0),builder.op('mul',ni,DIM))
     builder.mvout(builder.ptr_offset(output,off),addr,cols=count,rows=rows);builder.fence()
    columns(store);return token
   elif kind=='complete':builder.fence()
   else:raise ValueError('unknown actual rolling effect '+kind)
  receipts.append(emit_rolling_window(ir,rolling,builder,lower))
 full,tail=divmod(ow,DIM)
 if full:builder.loop(0,full,1,lambda i:stripe(builder.op('mul',i,DIM),DIM))
 if tail:stripe(full*DIM,tail)
 builder.fence();return dict(rolling_emission=receipts,source_binding='actual MVIN pointer uses original NHWC allocation or zero allocation; actual SP operand uses checked private row-plane map',all_original_outputs=True,completion='each DMA/initializer, rolling ready/last-use and original readout fully fenced; compute pairs complete at modulo16 boundaries plus full pre-readout drain; final whole-call completion')


def emit(builder,command,tensors,pointers,*,residents=None,mode=None):
 mode=selected() if mode is None else mode
 if mode=='off':return None
 if mode not in VALUES:raise ValueError('unknown direct-source epoch mode')
 try:epoch=prepare(command,tensors,pointers,residents=residents)
 except (ValueError,KeyError,TypeError,IndexError) as error:
  if not hasattr(builder,'conv_direct_epochs_refusals'):builder.conv_direct_epochs_refusals=[]
  builder.conv_direct_epochs_refusals.append(str(error));return None
 p=epoch.direct
 if p.base.pool is None:actual=_product(builder,epoch,pointers,pointers[p.source.output])
 else:
  from conv_pool_policy import emit_quantized_once
  size=p.source.output_rows*p.source.channels;q1=builder.global_buffer(size,'i8')
  actual=emit_quantized_once(builder,p.base.pool,intermediate=q1,output=pointers[p.source.output],workspace_bytes=size,emit_producer=lambda intermediate:_product(builder,epoch,pointers,intermediate))
 builder.fence();receipt=dict(epoch.report(),actual=actual)
 owner=builder
 while 'builder' in getattr(owner,'__dict__',{}):owner=owner.builder
 if not hasattr(owner,'conv_pool_panel_receipts'):owner.conv_pool_panel_receipts=[]
 owner.conv_pool_panel_receipts.append(dict(mechanism='conv_direct_epochs',actual=receipt));return receipt
