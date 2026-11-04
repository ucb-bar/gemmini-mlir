"""Original-NHWC horizontal K packets with separately owned halo stripes.

This authored schedule owns two private layouts with disjoint lifetimes. Complete
horizontal windows use the selected multiblock MVIN and contiguous original
KY/KX/CI reduction; halo stripes retain the original CI-wide source-plane path.
No source halo, destination-column injection, or array stay operation is used.
"""
from dataclasses import dataclass
from pathlib import Path
from math import prod, gcd
from merlin.compile.tuning_knobs import knob
from merlin.compile.scheduling.rolling_window import RollingWindowPlan, materialize_rolling_window, verify_rolling_window, emit_rolling_window
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.sched.ir import Kernel, Loop, Call, Const, Var
from merlin.compile.scheduling.llvm_program import emit_index_program
from conv_policy import completion_laws
from isa import ACC, ACCUMULATE, FULL, GARBAGE, DIM

VALUES = ('off', 'flat_horizontal', 'reduced_padding')
DECLARATION = dict(values=list(VALUES), mechanism='original NHWC wide horizontal K packets for in-bounds stripes; exact direct-source boundary planes', note='Default off. reduced_padding is a structural candidate guard, not a latency prediction: interior>=32 and fewer source packet/stripe groups. New private packet/storage owner, selected multiblock DMA and pairs16 laws; full transport, ring ready/last-use, initializer and readout completion. Original scalar numeric/Q1-once pooling retained.')
def selected(): return knob('conv_flat_horizontal', default='off', values=VALUES, root=Path(__file__).parent)


def _partition(window):
    _, _, width, _ = window.input_shape
    _, kw = window.kernel
    sx = window.stride[1]; left = window.padding[1]; ow = window.output_spatial[1]
    first = min(ow, max(0, (left + sx - 1) // sx))
    last = min(ow, max(first, (width + left - kw) // sx + 1))
    return ((0, first), (first, last), (last, ow))


def _extents(intervals):
    result = set()
    for start, end in intervals:
        full, tail = divmod(end-start, DIM)
        if full: result.add(DIM)
        if tail: result.add(tail)
    return tuple(sorted(result))


def _kernel(source, chunks, flat):
    co = source.channels; count = chunks if flat else source.window.kernel[1]
    effects = []
    for ni, column in enumerate(range(0, co, DIM)):
        args = tuple(zip(('ky','packet','ni','column','count'), (Var('ky'),Var('packet'),Const(ni),Const(column),Const(min(DIM,co-column)))))
        effects.extend((Call('pair',args),Call('boundary',args)))
    return Kernel('original_nhwc_flat_horizontal_epochs' if flat else 'original_nhwc_boundary_plane_epochs', (), (Loop('ky',source.window.kernel[0],(Loop('packet',count,tuple(effects)),)),))


@dataclass(frozen=True)
class PacketLayout:
    flat: bool
    intervals: tuple
    pitch: int
    rhs: object
    templates: tuple
    kernel: object
    storage: object


@dataclass(frozen=True)
class FlatPlan:
    base: object
    direct: object
    partition: tuple
    layouts: tuple
    law: object
    ordering: object
    proof: dict
    @property
    def source(self): return self.base.source
    @property
    def chunks(self): return self.base.chunks
    def revalidate(self):
        self.direct.revalidate(); self.base.original.__post_init__(); self.law.require(16)
        if self.partition != _partition(self.source.window): raise ValueError('original horizontal partition changed')
        laws = completion_laws(self.base.original,self.base.original.identity,self.base.rhs.n_tiles,mode='pairs16')
        if laws is None or self.law != laws[1] or self.ordering != laws[3]: raise ValueError('horizontal compute completion differs from selected facts')
        if self.direct.original != self.base.original or self.direct.source != self.source: raise ValueError('boundary and flat layouts have different source ownership')
        kh,kw = self.source.window.kernel
        expected_kinds = ((False,) if any(a<b for a,b in (self.partition[0],self.partition[2])) else ()) + (True,)
        if tuple(layout.flat for layout in self.layouts) != expected_kinds: raise ValueError('missing, repeated or reordered physical phase')
        for layout in self.layouts:
            packets = self.chunks if layout.flat else kw
            expected_intervals = (self.partition[1],) if layout.flat else (self.partition[0],self.partition[2])
            if layout.intervals != tuple(pair for pair in expected_intervals if pair[0] < pair[1]): raise ValueError('source stripe intervals changed')
            if layout.rhs != (self.base.rhs if layout.flat else self.direct.rhs): raise ValueError('physical RHS differs from source-qualified residency layout')
            if tuple(t[0] for t in layout.templates) != _extents(layout.intervals): raise ValueError('missing, repeated or reordered stripe templates')
            if layout.pitch != packets*DIM or layout.rhs.rhs_base != kh*layout.pitch: raise ValueError('horizontal physical packet mapping changed')
            if layout.rhs.rhs_base+layout.rhs.rhs_rows > self.base.original.operand_rows or layout.rhs.n_tiles*DIM > self.base.original.accumulator_rows: raise ValueError('horizontal packets exceed selected SP/ACC capacity')
            if layout.kernel != _kernel(self.source,self.chunks,layout.flat): raise ValueError('actual source compute kernel changed')
            for rows,rolling,ir in layout.templates:
                if rows not in _extents(layout.intervals) or rolling.source_identity != self.base.original.identity or rolling.window != self.base.padded or rolling.row_stride_bytes != layout.pitch*DIM or rolling.slots != kh or rolling.output_x_first != 0 or rolling.output_x_count != rows or rolling.output_y_first != 0 or rolling.output_y_count != self.source.window.output_spatial[0] or rolling.cache.start != 0 or rolling.cache.size != kh*layout.pitch*DIM: raise ValueError('actual rolling horizontal packet template changed')
                verify_rolling_window(ir,rolling)
        return self
    def report(self):
        return dict(route='original_nhwc_interior_flat_horizontal_pairs16',schema='target.original_nhwc_flat_horizontal.v1',source_identity=self.base.original.identity,source_attributes=self.source.attrs,source_window=repr(self.source.window),partition=self.partition,layouts=[dict(flat=p.flat,intervals=p.intervals,slot_pitch=p.pitch,rhs_base=p.rhs.rhs_base,rhs_rows=p.rhs.rhs_rows,kernel_sha256=p.kernel.digest(),storage=p.storage) for p in self.layouts],source_binding=self.proof,ordering_evidence=self.ordering,halo_materialization=False,host_payload_operations=0,completion='every DMA/initializer, rolling await/last-use, output readout, phase switch and final call fully complete; actual checked compute pairs drain every16 plus before readout',timing='UNKNOWN')


def _source_proof(base, partition):
    import z3
    w = base.source.window; batch,h,width,ci = w.input_shape; kh,kw = w.kernel; sy,sx = w.stride; top,left,_,_ = w.padding; oh,ow = w.output_spatial
    image,y,ky,x,r,j,slot,slot2,r2,j2 = z3.Ints('flat_image flat_y flat_ky flat_x flat_r flat_j flat_slot flat_slot2 flat_r2 flat_j2')
    queries = 0
    def prove(claim,domain,reason):
        nonlocal queries
        queries += 1; solver=z3.Solver();solver.set(timeout=2000);solver.add(*domain,z3.Not(claim))
        if solver.check() != z3.unsat: raise ValueError('original horizontal packet proof failed: '+reason)
    start,end = partition[1]; depth=kw*ci; chunks=base.chunks
    domain=[image>=0,image<batch,y>=0,y<oh,ky>=0,ky<kh,x>=start,x<end,r>=0,r<DIM,x+r<end,j>=0,j<depth]
    iy=y*sy-top+ky; ix=(x+r)*sx-left+j/ci
    prove(z3.And(ix>=0,ix<width),domain,'interior source packet crosses original horizontal allocation')
    offset=((image*h+iy)*width+(x+r)*sx-left)*ci+j
    original=((image*h+iy)*width+ix)*ci+j%ci
    prove(offset==original,domain,'flattened byte differs from original ky/kx/ci')
    prove(z3.And(offset>=0,offset<prod(w.input_shape)),domain+[iy>=0,iy<h],'valid source rectangle escapes original NHWC')
    prove(z3.And(j/ci>=0,j/ci<kw,j%ci>=0,j%ci<ci,(j/DIM)*DIM+j%DIM==j),domain,'flat slices do not cover original K exactly')
    local=(slot*chunks+j/DIM)*DIM+r; local2=(slot2*chunks+j2/DIM)*DIM+r2
    storage=[slot>=0,slot<kh,r>=0,r<DIM,j>=0,j<depth]
    prove(z3.And(local>=0,local<base.rhs.rhs_base,j%DIM>=0,j%DIM<DIM),storage,'flat packet aliases resident weights')
    prove(z3.Implies(z3.And(local==local2,j%DIM==j2%DIM),z3.And(slot==slot2,r==r2,j==j2)),storage+[slot2>=0,slot2<kh,r2>=0,r2<DIM,j2>=0,j2<depth],'distinct flat packet lanes alias')
    prove(((y*sy+ky)%kh)*chunks*DIM+(j/DIM)*DIM+r==(((y*sy+ky)%kh)*chunks+j/DIM)*DIM+r,domain,'actual consumer differs from staged flat packet')
    if partition[0][0]!=0 or partition[0][1]!=start or partition[2][0]!=end or partition[2][1]!=ow: raise ValueError('horizontal stripes do not partition every original output')
    return dict(schema='original_nhwc_flat_horizontal_binding.v1',symbolic_queries=queries,source_byte='((image*H+output_y*SY+ky-top)*W+(origin+output_lane)*SX-left)*CI+j',original_coordinate='ky, kx=j//CI, ci=j%CI in increasing source reduction order; j split into consecutive <=16 depths',physical_row='(unbiased_y%KH)*chunks*16+(j//16)*16+output_lane',physical_lane='j%16',source_stride_bytes=sx*ci,wide_columns=depth,horizontal_guard='only statically proven complete original-X windows use wide DMA; boundary stripes use separately owned original CI-wide planes',vertical_guard='original iy valid -> one wide source MVIN; invalid -> one zero-source MVIN; never zero then overwrite',phase_ownership='all boundary products/readouts/ring generations complete before flat layout/RHS replacement; each phase reloads its exact original RHS once',rolling='shared checked initial/frontier/modulo induction; explicit ready and last-use callbacks retained',flat_storage_reference=base.proof['physical_storage'])


def _storage(base, flat, rows_set, rhs):
    from merlin.compile.scheduling.temporary_storage import StorageAdapter,StorageEffects,TemporaryObject,verify_temporary_storage
    from merlin.compile.scheduling.program_limits import EncodedAccess
    from merlin.semantic_compiler.allocation_types import StorageBank
    w=base.source.window;ci=w.input_shape[-1];kw=w.kernel[1];packets=base.chunks if flat else kw;reports=[]
    for rows in rows_set:
        objects=[];writes=[];reads=[];effects={};body=[]
        for packet in range(packets):
            depth=min(DIM,kw*ci-packet*DIM) if flat else ci
            for row in range(rows):
                byte=(packet*DIM+row)*DIM
                objects.append(TemporaryObject('packet%drow%d'%(packet,row),'sp',byte,depth));writes.append(EncodedAccess('sp',byte,depth,True));reads.append(EncodedAccess('sp',byte,depth,False))
        def add(name,access=(),completion='synchronous',token=None,waits=()):
            body.append(Call(name,()));effects[name]=StorageEffects(name,tuple(access),completion,token,tuple(waits))
        for generation in range(2):
            load='load'+str(generation);consume='consume'+str(generation)
            add(load,writes,'asynchronous',load);add('ready'+str(generation),completion='wait',waits=(load,));add(consume,reads,'asynchronous',consume);add('complete'+str(generation),completion='wait',waits=(consume,))
        kernel=Kernel('original_horizontal_private_slot',(),tuple(body));adapter=StorageAdapter(lambda c:effects[c.instr],lambda c,a:c,base.law['facts_sha256'],'original-nhwc-horizontal-flat' if flat else 'original-nhwc-boundary-planes',base.law['ordering_sha256']);banks=(StorageBank('sp','selected-operand-scratchpad',rhs.rhs_base*DIM,'bytes',1),)
        def check(k):return verify_temporary_storage(k,k,tuple(objects),banks,adapter,private_ownership=True,max_calls=32,max_objects=1024)
        proof=check(kernel);rejected=[]
        for missing in ('ready0','complete0'):
            try:check(Kernel(kernel.name,(),tuple(c for c in body if c.instr!=missing)))
            except ValueError:rejected.append(missing)
            else:raise ValueError('horizontal owner accepted missing '+missing)
        reports.append(dict(rows=rows,proof=proof,invalid_orders_rejected=rejected))
    return reports


def prepare(command,tensors,pointers,*,residents=None):
    from conv_window_reuse import prepare as prepare_flat
    from conv_rolling_binding import prepare as prepare_boundary
    base=prepare_flat(command,tensors,pointers,residents=residents);direct=prepare_boundary(command,tensors,pointers,residents=residents)
    w=base.source.window;ci=w.input_shape[-1];kh,kw=w.kernel
    if kw*ci<=ci: raise ValueError('no horizontal K flattening opportunity')
    partition=_partition(w)
    if partition[1][0]>=partition[1][1]: raise ValueError('no complete original-X window')
    laws=completion_laws(base.original,base.original.identity,base.rhs.n_tiles,mode='pairs16')
    if laws is None or laws[1] is None: raise ValueError('selected horizontal completion/progress law unavailable')
    layouts=[]
    for flat,intervals in ((False,(partition[0],partition[2])),(True,(partition[1],))):
        intervals=tuple(pair for pair in intervals if pair[0]<pair[1])
        if not intervals:continue
        pitch=(base.chunks if flat else kw)*DIM;rhs=base.rhs if flat else direct.rhs;templates=[];rows_set=_extents(intervals)
        for rows in rows_set:
            rolling=RollingWindowPlan(base.padded,base.original.identity,0,rows,0,w.output_spatial[0],AddressWindow('original_horizontal_private_packets',0,kh*pitch*DIM),kh,pitch*DIM,DIM)
            ir=materialize_rolling_window(rolling);verify_rolling_window(ir,rolling);templates.append((rows,rolling,ir))
        layouts.append(PacketLayout(flat,intervals,pitch,rhs,tuple(templates),_kernel(base.source,base.chunks,flat),_storage(base,flat,rows_set,rhs)))
    proof=_source_proof(base,partition);proof['original_boundary_binding']=direct.proof
    return FlatPlan(base,direct,partition,tuple(layouts),laws[1],laws[3],proof).revalidate()


def _product(builder, plan, pointers, output):
    plan.revalidate();s=plan.source;w=s.window;batch,h,width,ci=w.input_shape;co=s.channels;oh,ow=w.output_spatial;kh,kw=w.kernel;sy,sx=w.stride;top,left,_,_=w.padding;dtype=s.attrs['output_dtype'];nbytes=4 if dtype=='i32' else 1;horizontal=kw*ci
    builder.fence();builder.config_ex(dataflow=1);builder.config_st(co*nbytes,activation=int('relu' in s.attrs.get('epilogue',())),scale=float(s.attrs.get('acc_scale',1.)))
    if plan.base.initializer:builder.config_ld(0,id=2)
    zero=builder.constant_i32_table([0]*DIM)
    def columns(body):
        full,tail=divmod(co,DIM)
        if full:builder.loop(0,full,1,lambda ni:body(ni,builder.op('mul',ni,DIM),DIM))
        if tail:body(full,full*DIM,tail)
    receipts=[]
    for layout in plan.layouts:
        # This drain closes every previous layout's ring/RHS/readout lifetime.
        builder.fence();packets=plan.chunks if layout.flat else kw
        def depth(packet):
            if not layout.flat:return ci
            tail=horizontal-(plan.chunks-1)*DIM
            return DIM if tail==DIM else builder.select(builder.cmp('eq',packet,plan.chunks-1),tail,DIM)
        def rhs_ky(ky):
            def rhs_packet(packet):
                tap=builder.op('add',builder.op('mul',ky,packets),packet)
                def load(ni,column,count):
                    local=builder.op('add',layout.rhs.rhs_base,builder.op('mul',builder.op('add',builder.op('mul',ni,kh*packets),tap),DIM))
                    source_k=builder.op('add',builder.op('mul',ky,horizontal),builder.op('mul',packet,DIM if layout.flat else ci))
                    offset=builder.op('add',builder.op('mul',source_k,co),column)
                    builder.config_ld(co,id=1);builder.mvin(builder.ptr_offset(pointers[s.rhs],offset),local,cols=count,rows=depth(packet),id=1);builder.fence()
                columns(load)
            builder.loop(0,packets,1,rhs_packet)
        builder.loop(0,kh,1,rhs_ky)
        def stripe(origin,rows):
            _,rolling,ir=next(t for t in layout.templates if t[0]==rows);token=object()
            def lower(kind,f):
                if kind=='begin_transport':return token
                if kind=='stage_row':
                    iy=builder.op('sub',f['input_row'],top);image=f['image'];slot=builder.op('udiv',f['cache_byte'],layout.pitch*DIM);yvalid=builder.cmp('ult',iy,h)
                    if layout.flat:
                        local=builder.op('mul',slot,layout.pitch);ix=builder.op('sub',builder.op('mul',origin,sx),left)
                        def load():
                            offset=builder.op('mul',builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',image,h),iy),width),ix),ci)
                            builder.config_ld(sx*ci,block_stride=DIM,id=0);builder.mvin(builder.ptr_offset(pointers[s.lhs],offset),local,cols=horizontal,rows=rows,id=0);builder.fence()
                        def clear():
                            builder.config_ld(0,block_stride=DIM,id=0);builder.mvin(zero,local,cols=horizontal,rows=rows,id=0);builder.fence()
                        builder.branch(yvalid,load,clear)
                    else:
                        def tap(kx):
                            base=builder.op('add',builder.op('mul',slot,layout.pitch),builder.op('mul',kx,DIM));ix=builder.op('sub',builder.op('add',builder.op('mul',origin,sx),kx),left)
                            def transfer(x,local,count,valid):
                                def load():
                                    offset=builder.op('mul',builder.op('add',builder.op('mul',builder.op('add',builder.op('mul',image,h),iy),width),x),ci)
                                    builder.config_ld(sx*ci,id=0);builder.mvin(builder.ptr_offset(pointers[s.lhs],offset),local,cols=ci,rows=count,id=0);builder.fence()
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
                    if plan.base.initializer:
                        def init(ni,column,count):
                            pointer=builder.ptr_offset(pointers[s.bias],builder.op('mul',column,4)) if s.bias else zero
                            builder.mvin(pointer,builder.op('or',ACC,builder.op('mul',ni,DIM)),cols=count,rows=rows,id=2);builder.fence()
                        columns(init)
                    def effect(call,f):
                        ky,packet,ni,column,count=(f[k] for k in ('ky','packet','ni','column','count'))
                        tap=builder.op('add',builder.op('mul',ky,packets),packet)
                        if call=='boundary':
                            if (ni+1)%gcd(layout.rhs.n_tiles,16):return
                            index=builder.op('add',builder.op('mul',tap,layout.rhs.n_tiles),ni+1)
                            builder.branch(builder.cmp('eq',builder.op('urem',index,16),0),builder.fence,lambda:None);return
                        if call!='pair':raise ValueError('unknown checked horizontal compute effect '+call)
                        slot=builder.op('urem',builder.op('add',builder.op('mul',y,sy),ky),kh)
                        lhs=builder.op('add',builder.op('mul',slot,layout.pitch),builder.op('mul',packet,DIM));rhs=builder.op('add',layout.rhs.rhs_base,builder.op('mul',builder.op('add',builder.op('mul',ni,kh*packets),tap),DIM));acc=builder.op('or',ACC,builder.op('mul',ni,DIM))
                        def pair(accumulate):
                            address=builder.op('or',acc,ACCUMULATE) if accumulate else acc
                            builder.preload(rhs,address,b_cols=count,b_rows=depth(packet),c_cols=count,c_rows=rows);builder.compute(lhs,GARBAGE,a_cols=depth(packet),a_rows=rows,bd_cols=count,bd_rows=rows)
                        if plan.base.initializer:pair(True)
                        else:builder.branch(builder.cmp('eq',tap,0),lambda:pair(False),lambda:pair(True))
                    emit_index_program(layout.kernel,builder,effect);builder.fence()
                    def store(ni,column,count):
                        offset=builder.op('mul',builder.op('add',builder.op('mul',outrow,co),column),nbytes);addr=builder.op('or',ACC|(FULL if dtype=='i32' else 0),builder.op('mul',ni,DIM))
                        builder.mvout(builder.ptr_offset(output,offset),addr,cols=count,rows=rows);builder.fence()
                    columns(store);return token
                elif kind=='complete':builder.fence()
                else:raise ValueError('unknown actual horizontal rolling effect '+kind)
            receipts.append(dict(flat=layout.flat,rows=rows,rolling=emit_rolling_window(ir,rolling,builder,lower)))
        for start,end in layout.intervals:
            full,tail=divmod(end-start,DIM)
            if full:builder.loop(0,full,1,lambda i:stripe(builder.op('add',start,builder.op('mul',i,DIM)),DIM))
            if tail:stripe(start+full*DIM,tail)
        builder.fence()
    plan.revalidate();builder.fence();return dict(rolling_emission=receipts,all_original_outputs=True,actual_wide_source_columns=horizontal,phase_switch_completion='full fence before RHS remap and after every output/readout/rolling consume')


def geometry_selection(plan):
    """Describe the source-only candidate predicate; this is not a cycle score."""
    plan.revalidate()
    w=plan.source.window;kw=w.kernel[1];ow=w.output_spatial[1]
    def stripes(width):return (width+DIM-1)//DIM
    left,interior,right=plan.partition
    interior_width=interior[1]-interior[0]
    spatial_rows=w.input_shape[0]*w.output_spatial[0]
    original_stripes=spatial_rows*stripes(ow)
    boundary_stripes=spatial_rows*(stripes(left[1]-left[0])+stripes(right[1]-right[0]))
    interior_stripes=spatial_rows*stripes(interior_width)
    original_groups=original_stripes*kw
    transformed_groups=boundary_stripes*kw+interior_stripes*plan.chunks
    enough_interior=interior_width>=2*DIM
    fewer_groups=original_groups>transformed_groups
    return dict(schema='target.conv_flat_source_geometry_selection.v1',
        hypothesis='strictly fewer horizontal source packet/stripe groups with at least two full interior row tiles',
        status='candidate' if enough_interior and fewer_groups else 'fallback',
        source_only=True,latency_prediction=False,
        batch_output_row_count=spatial_rows,output_width=ow,input_channels=w.input_shape[-1],kernel_width=kw,chunks=plan.chunks,
        interior_width=interior_width,minimum_interior_width=2*DIM,
        original_stripes=original_stripes,boundary_stripes=boundary_stripes,interior_stripes=interior_stripes,
        original_packet_stripe_groups=original_groups,transformed_packet_stripe_groups=transformed_groups,
        enough_interior=enough_interior,strictly_fewer_groups=fewer_groups)


def emit(builder,command,tensors,pointers,*,residents=None,mode=None):
    mode=selected() if mode is None else mode
    if mode=='off':return None
    if mode not in VALUES:raise ValueError('unknown flat horizontal mode')
    try:plan=prepare(command,tensors,pointers,residents=residents)
    except (ValueError,KeyError,TypeError,IndexError) as error:
        if not hasattr(builder,'conv_flat_horizontal_refusals'):builder.conv_flat_horizontal_refusals=[]
        builder.conv_flat_horizontal_refusals.append(str(error));return None
    selection=geometry_selection(plan) if mode=='reduced_padding' else None
    if selection is not None and selection['status']!='candidate':
        if not hasattr(builder,'conv_flat_horizontal_selections'):builder.conv_flat_horizontal_selections=[]
        builder.conv_flat_horizontal_selections.append(dict(mode=mode,**selection))
        return None
    if plan.base.pool is None:actual=_product(builder,plan,pointers,pointers[plan.source.output])
    else:
        from conv_pool_policy import emit_quantized_once
        size=plan.source.output_rows*plan.source.channels;q1=builder.global_buffer(size,'i8')
        actual=emit_quantized_once(builder,plan.base.pool,intermediate=q1,output=pointers[plan.source.output],workspace_bytes=size,emit_producer=lambda intermediate:_product(builder,plan,pointers,intermediate))
    builder.fence();receipt=dict(plan.report(),actual=actual)
    if selection is not None:receipt['source_geometry_selection']=dict(mode=mode,**selection)
    owner=builder
    while 'builder' in getattr(owner,'__dict__',{}):owner=owner.builder
    if not hasattr(owner,'conv_flat_horizontal_receipts'):owner.conv_flat_horizontal_receipts=[]
    owner.conv_flat_horizontal_receipts.append(receipt)
    if not hasattr(owner,'conv_pool_panel_receipts'):owner.conv_pool_panel_receipts=[]
    owner.conv_pool_panel_receipts.append(dict(mechanism='conv_flat_horizontal',actual=receipt));return receipt
