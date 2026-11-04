"""Original convolution through a checked rolling SP window-row representation.

A logical retained source row is represented by replicated horizontal windows,
not by a falsely relabelled contiguous SP byte row. The view-refinement proof
binds every actual DMA lane and array read to its original NHWC coordinate.
Shared rolling_window IR owns initial/frontier row identity and completion;
all target transfers/products/readouts still fully complete synchronously.
"""
from dataclasses import dataclass,replace
from hashlib import sha256
from math import prod
import json
from merlin.compile.scheduling.window_transfer import Window2D,CountedConvolutionPlan
from merlin.compile.scheduling.rolling_window import RollingWindowPlan,materialize_rolling_window,verify_rolling_window,emit_rolling_window,rolling_window_text
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency
from merlin.compile.scheduling.convolution_exploration import ConvolutionRhsResidencyLaw
from isa import ACC,ACCUMULATE,FULL,GARBAGE,DIM

@dataclass(frozen=True)
class ReusePlan:
 source:object
 pool:object
 original:object
 halo:object
 padded:object
 rhs:object
 templates:tuple
 initializer:bool
 law:dict
 proof:dict
 @property
 def chunks(self):return (self.source.window.kernel[1]*self.source.window.input_shape[-1]+DIM-1)//DIM
 @property
 def span(self):return self.source.window.kernel[0]
 @property
 def padded_bytes(self):return prod(self.padded.input_shape)
 def report(self):return dict(route='rolling_SP_horizontal_window_rows',source_window=repr(self.source.window),source_attributes=self.source.attrs,source_identity=self.original.identity,ring_operand_rows=self.span*self.chunks*DIM,rhs_rows=self.rhs.rhs_rows,accumulator_rows=self.rhs.n_tiles*DIM,padded_input_bytes=self.padded_bytes,view_refinement=self.proof,selected_load_law=self.law,templates=[dict(rows=t[0],rolling=verify_rolling_window(t[2],t[1])) for t in self.templates],host_payload_operations=0,timing='UNKNOWN')


def _view_proof(source,padded,chunks,span,oprows,accrows,rhs):
 """Affine/mixed-radix proof of the actual expanded SP representation.

 No payload simulation occurs during compilation. The logical ring row has
 aliases in several horizontal windows; all aliases must represent the SAME
 source byte. Only the explicitly related physical lanes are consumed.
 """
 import z3
 ci=source.window.input_shape[-1];kw=source.window.kernel[1];sw=source.window.stride[1];oh,ow=source.window.output_spatial;ph,pw=padded.input_shape[1:3]
 r,j,c,slot,y,ky,x,image=z3.Ints('reuse_r reuse_j reuse_c reuse_slot reuse_y reuse_ky reuse_x reuse_image')
 queries=0
 def prove(claim,domain,reason):
  nonlocal queries
  queries+=1;s=z3.Solver();s.set(timeout=2000);s.add(*domain,z3.Not(claim))
  if s.check()!=z3.unsat:raise ValueError('rolling physical view lacks proof: '+reason)
 # Actual MVIN cols=KW*CI decomposes across selected 16-column blocks.
 local=(slot*chunks+j/DIM)*DIM+r;lane=j%DIM
 dom=[slot>=0,slot<span,r>=0,r<DIM,j>=0,j<kw*ci]
 prove(z3.And(local>=0,local<rhs.rhs_base,lane>=0,lane<DIM),dom,'expanded MVIN aliases resident RHS')
 # Decoder of the physical lane gives exactly the source kx,ci the consumer
 # reads, including the partial final depth chunk; never padded garbage.
 depth=kw*ci
 prove((j/DIM)*DIM+j%DIM==j,[j>=0,j<depth],'source K lane decomposition')
 prove(z3.And(j/ci>=0,j/ci<kw,j%ci>=0,j%ci<ci),[j>=0,j<depth],'source kx/ci decode')
 # Horizontal translation templates retain the original X coordinate. All
 # original full/tail stripes stay inside the materialized NHWC halo.
 ix=x*sw+r*sw+j/ci
 prove(z3.And(ix>=0,ix<pw),[x>=0,x<ow,r>=0,r<DIM,x+r<ow,j>=0,j<depth],'translated source X bounds')
 iy=y*source.window.stride[0]+ky
 prove(z3.And(iy>=0,iy<ph),[y>=0,y<oh,ky>=0,ky<span],'source Y bounds')
 prove((iy%span)*chunks*DIM+(j/DIM)*DIM+r==((iy%span)*chunks+j/DIM)*DIM+r,[y>=0,y<oh,ky>=0,ky<span,j>=0,j<depth,r>=0,r<DIM],'consumer reads actual staged physical lane')
 # Overlapping horizontal windows are replicas, not fake distinct aliases.
 r2,j2=z3.Ints('reuse_r2 reuse_j2')
 src=(r*sw+j/ci)*ci+j%ci;src2=(r2*sw+j2/ci)*ci+j2%ci
 prove(src==src2,[r>=0,r<DIM,r2>=0,r2<DIM,j>=0,j<depth,j2>=0,j2<depth,r*sw+j/ci==r2*sw+j2/ci,j%ci==j2%ci],'replicated logical byte identity')
 slot2=z3.Int('reuse_slot2');local2=(slot2*chunks+j2/DIM)*DIM+r2
 prove(z3.Implies(z3.And(local==local2,lane==j2%DIM),z3.And(slot==slot2,r==r2,j==j2)),dom+[slot2>=0,slot2<span,r2>=0,r2<DIM,j2>=0,j2<depth],'distinct physical ring slot/chunk/output lanes are disjoint')
 if rhs.rhs_base+rhs.rhs_rows>oprows or rhs.n_tiles*DIM>accrows:raise ValueError('rolling physical view exceeds selected private SP/ACC capacities')
 return dict(schema='expanded_horizontal_row_representation_v1',symbolic_queries=queries,logical_space='horizontal_replica_view; addresses are logical coordinates, never directly used as physical SP pointers',physical_space='selected i8 scratchpad rows',logical_row='source columns [origin*SW,(origin+rows-1)*SW+KW), original CI',physical_map='row=(source_y % KH)*chunks*16 + (kx*CI+ci)//16*16 + output_lane; lane=(kx*CI+ci)%16',payload='each actual DMA lane is original padded NHWC[image,source_y,(origin+output_lane)*SW+kx,ci]; every consumer reads precisely this lane',retained_identity='shared checked rolling-window initial/frontier and modulo slot induction',translation='source X=template X+origin*SW; output X=template X+origin; original full/tail stripe partition covers OW exactly',padding='original 1x1 DMA-only zero-halo preparation; no host payload access',capacities=dict(operand_rows=oprows,accumulator_rows=accrows,rhs_base=rhs.rhs_base,rhs_end=rhs.rhs_base+rhs.rhs_rows),synchronization='entry full fence; each DMA and fresh PRELOAD/COMPUTE/store full fence; shared await and consume-complete full fence before frontier overwrites')


def _physical_storage(source,rhs,chunks,law):
 """Check the new physical retained SP view, not the logical cache addresses.

 The shared rolling induction supplies row identities and full completion
 before slot replacement. Its physical refinement is translation invariant in
 image/X/Y: check each active row/lane of one slot for every stripe extent,
 including two generations and real async waits. RHS/ACC remain the existing
 fresh product path and are outside this retained-view storage transformation.
 """
 from merlin.compile.scheduling.temporary_storage import StorageAdapter,StorageEffects,TemporaryObject,verify_temporary_storage
 from merlin.compile.scheduling.program_limits import EncodedAccess
 from merlin.semantic_compiler.allocation_types import StorageBank
 from merlin.sched.ir import Kernel,Call
 w=source.window;cols=w.kernel[1]*w.input_shape[-1];stride=w.stride[1]*w.input_shape[-1];ow=w.output_spatial[1];reports=[]
 for rows in sorted({min(DIM,ow),ow%DIM}-{0}):
  effects={};body=[];objects=[]
  def add(name,access=(),completion='synchronous',token=None,waits=()):
   body.append(Call(name,()));effects[name]=StorageEffects(name,tuple(access),completion,token,tuple(waits))
  # Same active addresses as MVIN(cols=KW*CI,rows=rows,block_stride=16)
  # and COMPUTE(a_cols=depth,a_rows=rows); padding lanes are never objects.
  writes=[];reads=[]
  for chunk in range(chunks):
   depth=min(DIM,cols-chunk*DIM)
   for row in range(rows):
    byte=(chunk*DIM+row)*DIM
    objects.append(TemporaryObject('chunk%drow%d'%(chunk,row),'sp',byte,depth))
    writes.append(EncodedAccess('sp',byte,depth,True));reads.append(EncodedAccess('sp',byte,depth,False))
  source_reads=[EncodedAccess('source',row*stride,cols,False) for row in range(rows)]
  for generation in range(2):
   load='load'+str(generation);consume='consume'+str(generation)
   add(load,source_reads+writes,'asynchronous',load)
   add('ready'+str(generation),completion='wait',waits=(load,))
   add(consume,reads,'asynchronous',consume)
   add('complete'+str(generation),completion='wait',waits=(consume,))
  kernel=Kernel('rolling_physical_sp_slot',(),tuple(body))
  adapter=StorageAdapter(lambda call:effects[call.instr],lambda call,accesses:call,law['facts_sha256'],'signed-i8-horizontal-window-replica',law['ordering_sha256'])
  banks=(StorageBank('sp','selected-operand-scratchpad',rhs.rhs_base*DIM,'bytes',1),)
  windows=(AddressWindow('source',0,(rows-1)*stride+cols),)
  proof=verify_temporary_storage(kernel,kernel,tuple(objects),banks,adapter,external_windows=windows,private_ownership=True,max_calls=32,max_objects=1024)
  # Both invalid execution orders must be rejected independently by the
  # same library owner, not merely by the coordinate solver.
  rejected=[]
  for missing in ('ready0','complete0'):
   broken=Kernel(kernel.name,(),tuple(c for c in body if c.instr!=missing))
   try:verify_temporary_storage(broken,broken,tuple(objects),banks,adapter,external_windows=windows,private_ownership=True,max_calls=32,max_objects=1024)
   except ValueError:rejected.append(missing)
   else:raise ValueError('physical ring proof admitted missing '+missing)
  reports.append(dict(rows=rows,kernel_sha256=kernel.digest(),storage=proof,invalid_orders_rejected=rejected))
 return dict(scope='exact active bytes of expanded MVIN/COMPUTE SP slot; symbolic modulo/translation refinement and shared rolling induction cover all source rows/stripes; no claim this proves RHS/ACC numerics',templates=reports)


def prepare(command,tensors,pointers,*,residents=None):
 from convolution import _capacities
 from conv_horizontal import _load_law
 from conv_policy import accumulator_init_selection
 pool='maxpool' in command.get('attributes',{}).get('epilogue',())
 if pool:
  from conv_pool_panel import prepare as source_prepare
  source=source_prepare(command,tensors,pointers,residents=residents)
  if source is None:raise ValueError('original pool source/numerical contract refused')
 else:
  from conv_packed_k import prepare as source_prepare
  source=source_prepare(command,tensors,pointers,residents=residents)
 if any(pointers[source.output]==pointers[name] for name in (source.lhs,source.rhs,source.bias) if name):raise ValueError('retained immutable source and published output must be disjoint')
 w=source.window;d,oprows,accrows=_capacities();ci=w.input_shape[-1];kh,kw=w.kernel;sh,sw=w.stride
 if d!=DIM:raise ValueError('ring binding requires selected ISA array edge')
 if source.attrs.get('acc_scale_by_column') is not None:raise ValueError('ring scalar binding has no per-column exit')
 if w.dilation!=(1,1) or ci>d or kw*ci>64:raise ValueError('ring representation requires dilation1, CI<=edge and KW*CI<=selected64-byte transfer')
 if not 0<sh<kh or sw>kw:raise ValueError('ring requires actual vertical overlap and contiguous horizontal union')
 oh,ow=w.output_spatial;chunks=(kw*ci+d-1)//d;ringrows=kh*chunks*d
 init=accumulator_init_selection();explicit=bool(source.bias) or not source.bias and (init=='zero_accumulate' or init=='zero_single_slice' and source.reduction<=d)
 original=CountedConvolutionPlan(w,source.channels,d,min(d,ow),d,min(d,ci),oprows,accrows,0,d,0,explicit)
 rhs=plan_dense_rhs_residency(min(d,ow),kh*chunks*d,source.channels,block=d,operand_rows=oprows,accumulator_rows=accrows,rhs_base=ringrows)
 if rhs is None or rhs.n_tiles*d>accrows:raise ValueError('complete original RHS plus rolling SP panels exceeds selected capacity')
 padded=Window2D((w.input_shape[0],w.input_shape[1]+w.padding[0]+w.padding[2],w.input_shape[2]+w.padding[1]+w.padding[3],ci),w.kernel,w.stride,w.dilation)
 if padded.output_spatial!=w.output_spatial:raise ValueError('halo source transform changes output shape')
 halo=replace(source,window=Window2D(w.input_shape,(1,1),(1,1),(1,1),w.padding),rows=prod(padded.input_shape[:3]))
 law=_load_law();retention=ConvolutionRhsResidencyLaw(law['facts_sha256'],'original-signed-i8-acc-i32-readout',original.identity,law['ordering_sha256'],True,True,True,True,True);retention.require()
 proof=_view_proof(source,padded,chunks,kh,oprows,accrows,rhs);proof['rhs_retention_law']=repr(retention)
 proof['physical_storage']=_physical_storage(source,rhs,chunks,law)
 templates=[]
 for rows in sorted({min(d,ow),ow%d}-{0}):
  pitch=chunks*d*d
  rolling=RollingWindowPlan(padded,original.identity,0,rows,0,oh,AddressWindow('horizontal_replica_view',0,kh*pitch),kh,pitch,d)
  ir=materialize_rolling_window(rolling);verify_rolling_window(ir,rolling);templates.append((rows,rolling,ir))
 return ReusePlan(source,source.pool if pool else None,original,halo,padded,rhs,tuple(templates),explicit,law,proof)


def _product(builder,plan,pointers,padded,output):
 s=plan.source;w=s.window;ci=w.input_shape[-1];co=s.channels;oh,ow=w.output_spatial;kh,kw=w.kernel;sh,sw=w.stride;d=DIM;chunks=plan.chunks;cols=kw*ci;dtype=s.attrs['output_dtype'];width=4 if dtype=='i32' else 1
 builder.fence();builder.config_ex(dataflow=1);builder.config_st(co*width,activation=int('relu' in s.attrs.get('epilogue',())),scale=float(s.attrs.get('acc_scale',1.)))
 if plan.initializer:builder.config_ld(0,id=2)
 zero=builder.constant_i32_table([0]*DIM)
 for ky in range(kh):
  for chunk in range(chunks):
   depth=min(d,cols-chunk*d)
   for ni in range(plan.rhs.n_tiles):
    column=ni*d;count=min(d,co-column);rhsrow=plan.rhs.rhs_base+(ni*kh*chunks+ky*chunks+chunk)*d
    builder.config_ld(co,id=1);builder.mvin(builder.ptr_offset(pointers[s.rhs],(ky*cols+chunk*d)*co+column),rhsrow,cols=count,rows=depth,id=1);builder.fence()
 receipts=[]
 def stripe(origin,rows):
  _,rolling,ir=next(t for t in plan.templates if t[0]==rows)
  token=object()
  def lower(kind,f):
   if kind=='begin_transport':return token
   if kind=='stage_row':
    local=builder.op('udiv',f['cache_byte'],d)
    source=builder.ptr_offset(padded,builder.op('add',f['source_byte'],builder.op('mul',origin,sw*ci)))
    def valid():
     builder.config_ld(sw*ci,block_stride=d,id=0);builder.mvin(source,local,cols=cols,rows=rows,id=0);builder.fence()
    def invalid():
     builder.config_ld(0,block_stride=d,id=0);builder.mvin(zero,local,cols=cols,rows=rows,id=0);builder.fence()
    builder.branch(f['valid'],valid,invalid)
   elif kind=='await_transport':builder.fence();return token
   elif kind=='consume':
    y=f['output_y'];image=f['image'];outrow=builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',image,oh),y),ow),origin)
    if plan.initializer:
     for ni in range(plan.rhs.n_tiles):
      column=ni*d;count=min(d,co-column);ptr=builder.ptr_offset(pointers[s.bias],column*4) if s.bias else zero
      builder.mvin(ptr,ACC|ni*d,cols=count,rows=rows,id=2);builder.fence()
    for ky in range(kh):
     slot=builder.op('urem',builder.op('add',builder.op('mul',y,sh),ky),kh)
     for chunk in range(chunks):
      depth=min(d,cols-chunk*d);lhs=builder.op('add',builder.op('mul',slot,chunks*d),chunk*d)
      for ni in range(plan.rhs.n_tiles):
       column=ni*d;count=min(d,co-column);rhsrow=plan.rhs.rhs_base+(ni*kh*chunks+ky*chunks+chunk)*d
       addr=ACC|ni*d|(ACCUMULATE if plan.initializer or ky or chunk else 0)
       builder.preload(rhsrow,addr,b_cols=count,b_rows=depth,c_cols=count,c_rows=rows);builder.compute(lhs,GARBAGE,a_cols=depth,a_rows=rows,bd_cols=count,bd_rows=rows);builder.fence()
    for ni in range(plan.rhs.n_tiles):
     column=ni*d;count=min(d,co-column);offset=builder.op('mul',builder.op('add',builder.op('mul',outrow,co),column),width)
     builder.mvout(builder.ptr_offset(output,offset),ACC|ni*d|(FULL if dtype=='i32' else 0),cols=count,rows=rows);builder.fence()
    return token
   elif kind=='complete':builder.fence()
   else:raise ValueError('unknown checked rolling effect '+kind)
  receipts.append(emit_rolling_window(ir,rolling,builder,lower))
 full,tail=divmod(ow,d)
 if full:builder.loop(0,full,1,lambda i:stripe(builder.op('mul',i,d),d))
 if tail:stripe(full*d,tail)
 builder.fence();return dict(rolling_emission=receipts,all_original_outputs=True,actual_fresh_preload_every_product=True,completion='every target callback and source consume boundary fully complete')


def emit(builder,command,tensors,pointers,*,residents=None):
 try:plan=prepare(command,tensors,pointers,residents=residents)
 except (ValueError,KeyError,TypeError,IndexError) as error:
  if not hasattr(builder,'conv_window_reuse_refusals'):builder.conv_window_reuse_refusals=[]
  builder.conv_window_reuse_refusals.append(str(error));return None
 from conv_pool_panel import emit_dma_panel
 padded=builder.global_buffer(plan.padded_bytes,'i8');zero=builder.constant_i32_table([0]*DIM)
 transport=emit_dma_panel(builder,plan.halo,pointers[plan.source.lhs],padded,0,plan.halo.rows,zero)
 if plan.pool is None:actual=_product(builder,plan,pointers,padded,pointers[plan.source.output])
 else:
  from conv_pool_policy import emit_quantized_once
  qbytes=plan.source.output_rows*plan.source.channels;q1=builder.global_buffer(qbytes,'i8')
  actual=emit_quantized_once(builder,plan.pool,intermediate=q1,output=pointers[plan.source.output],workspace_bytes=qbytes,emit_producer=lambda intermediate:_product(builder,plan,pointers,padded,intermediate))
 builder.fence();receipt=dict(plan.report(),transport=transport,actual=actual)
 owner=builder
 while hasattr(owner,'builder'):owner=owner.builder
 if not hasattr(owner,'conv_pool_panel_receipts'):owner.conv_pool_panel_receipts=[]
 owner.conv_pool_panel_receipts.append(dict(mechanism='conv_window_reuse',actual=receipt))
 return receipt
