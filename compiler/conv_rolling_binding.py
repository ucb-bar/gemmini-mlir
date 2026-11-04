"""Default-off original-source DMA binding for retained convolution row planes.

066 already consumed its expanded map in real commands. This alternative uses
one CI-wide kx plane, stages directly from original NHWC or mapped zero, and
eliminates whole-input halo materialization. It grants no overlap or stay law.
"""
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from pathlib import Path
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan
from merlin.compile.scheduling.rolling_window import RollingWindowPlan,materialize_rolling_window,verify_rolling_window,emit_rolling_window
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
from isa import ACC,ACCUMULATE,FULL,GARBAGE,DIM

VALUES=('off','direct_source_planes')
DECLARATION=dict(values=list(VALUES),mechanism='source-bound original NHWC DMA into checked retained ky-slot/kx row planes; zero-source at halo, no materialized input halo',note='Default off. CI<=array edge, dilation1 and vertical overlap; capacity checked. Original ky/kx/ci order, fresh products, quantize once then pool and all callback completions retained. Extra plane DMA/array padding costs unmeasured.')
def selected():return knob('conv_rolling_binding',default='off',values=VALUES,root=Path(__file__).parent)

@dataclass(frozen=True)
class DirectPlan:
 base:object
 rhs:object
 templates:tuple
 plane_pitch:int
 slot_pitch:int
 proof:dict
 @property
 def source(self):return self.base.source
 @property
 def original(self):return self.base.original
 @property
 def ring_rows(self):return self.source.window.kernel[0]*self.slot_pitch
 def revalidate(self):
  self.original.__post_init__();w=self.source.window;ci=w.input_shape[-1];kh,kw=w.kernel
  if self.plane_pitch!=DIM or self.slot_pitch!=kw*DIM or self.rhs.rhs_base!=kh*self.slot_pitch:raise ValueError('private packet mapping differs from checked row planes')
  if self.original.window!=w or self.original.output_channels!=self.source.channels:raise ValueError('original counted source changed')
  if self.rhs.rhs_base+self.rhs.rhs_rows>self.original.operand_rows or self.rhs.n_tiles*DIM>self.original.accumulator_rows:raise ValueError('source-bound rows exceed original selected capacity')
  for rows,rolling,ir in self.templates:
   if rolling.source_identity!=self.original.identity or rolling.window!=self.base.padded or rolling.row_stride_bytes!=self.slot_pitch*DIM or rolling.slots!=kh:raise ValueError('rolling source identity/cache mapping changed')
   verify_rolling_window(ir,rolling)
  return self
 def report(self):
  return dict(route='direct_source_retained_kx_planes',source_identity=self.original.identity,source_attributes=self.source.attrs,source_window=repr(self.source.window),ring_operand_rows=self.ring_rows,rhs_rows=self.rhs.rhs_rows,full_ld0='CI*source_stride_x for source, zero stride for mapped zero row',actual_packet='SP row=(unbiased_input_y%KH)*KW*16+kx*16+output_lane, lane=ci; every original-source transfer encodes this row',binding_proof=self.proof,halo_materialization=False,host_payload_operations=0,timing='UNKNOWN')

def _binding_proof(base,rhs,plane_pitch,slot_pitch):
 import z3
 w=base.source.window;b,h,width,ci=w.input_shape;kh,kw=w.kernel;sy,sx=w.stride;top,left,_,_=w.padding;oh,ow=w.output_spatial
 slot,kx,r,c,slot2,kx2,r2,c2,y,x,image=z3.Ints('direct_slot direct_kx direct_r direct_c direct_slot2 direct_kx2 direct_r2 direct_c2 direct_y direct_x direct_image')
 queries=0
 def prove(claim,domain,reason):
  nonlocal queries
  queries+=1;s=z3.Solver();s.set(timeout=2000);s.add(*domain,z3.Not(claim))
  if s.check()!=z3.unsat:raise ValueError('direct packet proof failed: '+reason)
 local=slot*slot_pitch+kx*plane_pitch+r;local2=slot2*slot_pitch+kx2*plane_pitch+r2
 dom=[slot>=0,slot<kh,kx>=0,kx<kw,r>=0,r<DIM,c>=0,c<ci]
 dom2=[slot2>=0,slot2<kh,kx2>=0,kx2<kw,r2>=0,r2<DIM,c2>=0,c2<ci]
 prove(z3.And(local>=0,local<rhs.rhs_base,c>=0,c<DIM),dom,'actual source load intersects RHS/outside local lane')
 prove(z3.Implies(z3.And(local==local2,c==c2),z3.And(slot==slot2,kx==kx2,r==r2,c==c2)),dom+dom2,'distinct pending row planes alias')
 iy=y*sy-top+slot;ix=(x+r)*sx-left+kx
 source=((image*h+iy)*width+ix)*ci+c
 valid=[image>=0,image<b,y>=0,y<oh,slot>=0,slot<kh,x>=0,x<ow,r>=0,r<DIM,x+r<ow,kx>=0,kx<kw,c>=0,c<ci,iy>=0,iy<h,ix>=0,ix<width]
 prove(z3.And(source>=0,source<prod(w.input_shape)),valid,'valid source DMA escapes original tensor')
 prove(source==((image*h+(y*sy+slot-top))*width+((x+r)*sx+kx-left))*ci+c,valid,'absolute original source coordinate changed')
 # Padded rolling input row is exactly unbiased y*SY+ky. Modulo slot
 # identity comes from the checked rolling induction; source offset removes
 # original padding once. No halo buffer address is used by target loads.
 prove(((y*sy+slot)%kh)*slot_pitch+kx*plane_pitch+r==((y*sy+slot)%kh)*kw*DIM+kx*DIM+r,valid,'consumer row differs from actual staged packet')
 if plane_pitch!=DIM or slot_pitch!=kw*DIM:raise ValueError('unsupported row-plane pitch')
 return dict(schema='original_nhwc_direct_row_plane_binding.v1',symbolic_queries=queries,source_byte='((image*H+unbiased_y-top)*W+(origin+output_lane)*SW+kx-left)*CI+ci',local_row='(unbiased_y%KH)*KW*16+kx*16+output_lane',lane='original ci, with no padding lanes consumed',guard='source read iff original iy,ix in bounds; otherwise mapped zero DMA to the same assigned row, never zero then overwrite',row_lifetime='shared rolling initial/frontier/modulo induction; full await before consume, full complete before any slot replacement',selected_law=base.law)

def _storage(base,rhs):
 from merlin.compile.scheduling.temporary_storage import StorageAdapter,StorageEffects,TemporaryObject,verify_temporary_storage
 from merlin.compile.scheduling.program_limits import EncodedAccess
 from merlin.semantic_compiler.allocation_types import StorageBank
 from merlin.sched.ir import Kernel,Call
 w=base.source.window;ci=w.input_shape[-1];kw=w.kernel[1];ow=w.output_spatial[1];reports=[]
 for rows in sorted({min(DIM,ow),ow%DIM}-{0}):
  effects={};body=[];objects=[];writes=[];reads=[]
  def add(name,access=(),completion='synchronous',token=None,waits=()):
   body.append(Call(name,()));effects[name]=StorageEffects(name,tuple(access),completion,token,tuple(waits))
  for kx in range(kw):
   for row in range(rows):
    byte=(kx*DIM+row)*DIM;objects.append(TemporaryObject('kx%drow%d'%(kx,row),'sp',byte,ci));writes.append(EncodedAccess('sp',byte,ci,True));reads.append(EncodedAccess('sp',byte,ci,False))
  for generation in range(2):
   load='load'+str(generation);consume='consume'+str(generation)
   add(load,writes,'asynchronous',load);add('ready'+str(generation),completion='wait',waits=(load,));add(consume,reads,'asynchronous',consume);add('complete'+str(generation),completion='wait',waits=(consume,))
  kernel=Kernel('direct_source_row_plane_slot',(),tuple(body));adapter=StorageAdapter(lambda c:effects[c.instr],lambda c,a:c,base.law['facts_sha256'],'original-nhwc-signed-i8-row-planes',base.law['ordering_sha256'])
  banks=(StorageBank('sp','selected-operand-scratchpad',rhs.rhs_base*DIM,'bytes',1),)
  def check(k):return verify_temporary_storage(k,k,tuple(objects),banks,adapter,private_ownership=True,max_calls=32,max_objects=1024)
  proof=check(kernel);rejected=[]
  for missing in ('ready0','complete0'):
   broken=Kernel(kernel.name,(),tuple(c for c in body if c.instr!=missing))
   try:check(broken)
   except ValueError:rejected.append(missing)
   else:raise ValueError('direct row-plane storage accepted missing '+missing)
  reports.append(dict(rows=rows,proof=proof,invalid_orders_rejected=rejected))
 return reports

def prepare(command,tensors,pointers,*,residents=None):
 from conv_window_reuse import prepare as prepare_source
 base=prepare_source(command,tensors,pointers,residents=residents)
 if base is None:raise ValueError('original source/refinement unavailable')
 w=base.source.window;kh,kw=w.kernel;oh,ow=w.output_spatial;co=base.source.channels;plane=DIM;slot=kw*plane;ring=kh*slot
 rhs=plan_dense_rhs_residency(min(DIM,ow),kh*kw*DIM,co,block=DIM,operand_rows=base.original.operand_rows,accumulator_rows=base.original.accumulator_rows,rhs_base=ring)
 if rhs is None or rhs.n_tiles*DIM>base.original.accumulator_rows:raise ValueError('complete RHS plus direct source row planes exceeds selected capacity')
 proof=_binding_proof(base,rhs,plane,slot);proof['physical_storage']=_storage(base,rhs)
 templates=[]
 for rows in sorted({min(DIM,ow),ow%DIM}-{0}):
  rolling=RollingWindowPlan(base.padded,base.original.identity,0,rows,0,oh,AddressWindow('direct_nhwc_row_planes',0,ring*DIM),kh,slot*DIM,DIM)
  ir=materialize_rolling_window(rolling);verify_rolling_window(ir,rolling);templates.append((rows,rolling,ir))
 return DirectPlan(base,rhs,tuple(templates),plane,slot,proof).revalidate()

def _product(builder,p,pointers,output):
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
    def ky_body(ky):
     slot=builder.op('urem',builder.op('add',builder.op('mul',y,sy),ky),kh)
     def kx_body(kx):
      tap=builder.op('add',builder.op('mul',ky,kw),kx);lhs=builder.op('add',builder.op('mul',slot,p.slot_pitch),builder.op('mul',kx,DIM))
      def product(ni,column,count):
       rhs=builder.op('add',p.rhs.rhs_base,builder.op('mul',builder.op('add',builder.op('mul',ni,kh*kw),tap),DIM));acc=builder.op('or',ACC,builder.op('mul',ni,DIM))
       def pair(accumulate):
        addr=builder.op('or',acc,ACCUMULATE) if accumulate else acc
        builder.preload(rhs,addr,b_cols=count,b_rows=ci,c_cols=count,c_rows=rows);builder.compute(lhs,GARBAGE,a_cols=ci,a_rows=rows,bd_cols=count,bd_rows=rows);builder.fence()
       if p.base.initializer:pair(True)
       else:builder.branch(builder.cmp('eq',tap,0),lambda:pair(False),lambda:pair(True))
      columns(product)
     builder.loop(0,kw,1,kx_body)
    builder.loop(0,kh,1,ky_body)
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
 builder.fence();return dict(rolling_emission=receipts,source_binding='actual MVIN pointer uses original NHWC allocation or zero allocation; actual SP operand uses checked private row-plane map',all_original_outputs=True,completion='every transfer/product/readout and rolling await/complete fully fenced')

def emit(builder,command,tensors,pointers,*,residents=None,mode=None):
 mode=selected() if mode is None else mode
 if mode=='off':return None
 if mode not in VALUES:raise ValueError('unknown direct rolling binding mode')
 try:p=prepare(command,tensors,pointers,residents=residents)
 except (ValueError,KeyError,TypeError,IndexError) as error:
  if not hasattr(builder,'conv_rolling_binding_refusals'):builder.conv_rolling_binding_refusals=[]
  builder.conv_rolling_binding_refusals.append(str(error));return None
 if p.base.pool is None:actual=_product(builder,p,pointers,pointers[p.source.output])
 else:
  from conv_pool_policy import emit_quantized_once
  size=p.source.output_rows*p.source.channels;q1=builder.global_buffer(size,'i8')
  actual=emit_quantized_once(builder,p.base.pool,intermediate=q1,output=pointers[p.source.output],workspace_bytes=size,emit_producer=lambda intermediate:_product(builder,p,pointers,intermediate))
 builder.fence();receipt=dict(p.report(),actual=actual)
 owner=builder
 while 'builder' in getattr(owner,'__dict__',{}):owner=owner.builder
 if not hasattr(owner,'conv_rolling_binding_receipts'):owner.conv_rolling_binding_receipts=[]
 owner.conv_rolling_binding_receipts.append(receipt)
 if not hasattr(owner,'conv_pool_panel_receipts'):owner.conv_pool_panel_receipts=[]
 owner.conv_pool_panel_receipts.append(dict(mechanism='conv_rolling_binding',actual=receipt));return receipt
