"""Source-owned tensor programs using Merlin host codegen and Gemmini contractions."""
from __future__ import annotations
import hashlib
from math import prod
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, TensorType, IntegerAttr, i64
from xdsl.ir import Block
from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.host_builder import FnBuilder, PTR
from merlin.llvmlower.host_scalar_builder import FpBuilder
from merlin.llvmlower.host_tensor_codegen import HostLinalg, TensorVal, elem_name, tensor_shape, LoweringDeclined
from merlin.llvmlower.passes_xdsl import lower_quant_ext
from encoding import Encoder, kernel_module
from dense import emit_dense


def _is_device(op):
    return (op.name in ('linalg.generic','linalg.matmul','linalg.batch_matmul')
        and len(op.operands) == 3 and len(op.results)==1
        and all(isinstance(v.type,TensorType) and str(v.type.element_type)=='i8' for v in op.operands[:2])
        and str(op.results[0].type.element_type)=='i32'
        and (op.name!='linalg.generic' or any('reduction' in str(i) for i in op.properties['iterator_types'])))


class _SourceFnBuilder(FnBuilder):
    """Expose statically nonempty schedule loops as mandatory executable CFG."""
    def attribute(self, op):
        op.attributes['merlin.global_task']=IntegerAttr(0,i64)
        index=getattr(self,'current_source_index',None)
        if index is not None:
            op.attributes['merlin.source_op_index']=IntegerAttr(index,i64)
        return op

    def add(self, op):
        result=super().add(self.attribute(op))
        capture=getattr(self,'storage_capture',None)
        if capture is not None:capture.record(op)
        return result

    def const(self, value, ty=i64):
        result=super().const(value,ty)
        if 'merlin.global_task' not in result.owner.attributes:
            self.attribute(result.owner)
        return result

    def for_loop(self, start, stop, step, body, *, retain_loop=False):
        if type(retain_loop) is not bool or step <= 0 or start < 0 or stop < start:
            raise ValueError('invalid static source loop bounds')
        if start == stop:
            return
        active=Block(arg_types=[i64]); after=Block()
        self.region.add_block(active)
        self.region.add_block(after)
        self.add(llvm.BrOp(active,self.const(start)))
        self.blk=active
        index=active.args[0]
        body(index)
        nxt=self.add_i(index,self.const(step))
        condition=self.add(llvm.ICmpOp(nxt,self.const(stop),
            IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(),i64))).results[0]
        latch=self.add(llvm.CondBrOp(condition,active,[nxt],after,[]))
        if retain_loop:
            from merlin.llvmlower.llvm_loop_metadata import disable_loop_unroll
            disable_loop_unroll(latch)
        self.blk=after


class _SourceFpBuilder(FpBuilder):
    """Keep shared floating constants dominating counted device schedule blocks."""
    def fconst(self,value):
        key=self._fconst_key(value)
        if key in self._fconsts:
            return self._fconsts[key]
        result=super().fconst(value)
        op=result.owner; entry=self.fb.entry
        self.fb.attribute(op)
        if op.parent is entry:
            first=next((x for x in entry.ops if x.has_trait(llvm.IsTerminator)),None)
            if first is not None:
                ordered=list(entry.ops)
                if ordered.index(op)>ordered.index(first):
                    entry.detach_op(op)
                    entry.insert_op_before(op,first)
        return result


class _LoadedElements:
    """Defer shared scalar loads until a host consumer actually requests them."""
    def __init__(self, host, ptr, count, ety):
        self.host,self.ptr,self.count,self.ety=host,ptr,count,ety
        self.cache={}
    def __len__(self): return self.count
    def __getitem__(self,index):
        if isinstance(index,slice): return [self[i] for i in range(*index.indices(self.count))]
        if index<0: index+=self.count
        if not 0<=index<self.count: raise IndexError(index)
        if index not in self.cache:
            host=self.host; offset=host.fp.fb.const(index)
            self.cache[index]=(host.fp.fb.load_i64(self.ptr,offset,self.ety) if self.ety.startswith('i') else host.fp.load(self.ptr,offset,self.ety))
        return self.cache[index]


class MixedHost(HostLinalg):
    """Reuse the shared host semantics, replacing admitted contraction handlers."""
    def __init__(self,*args,enc,ptrs,**kwargs):
        super().__init__(*args,**kwargs)
        self.enc,self.ptrs=enc,ptrs
        self.materialized_results=set()
        self.device_receipts=[]

    def record_full_writer(self, value, implementation):
        """Bind an actual complete counted writer to its emitted boundary."""
        capture=getattr(self.fp.fb,'storage_capture',None)
        if capture is not None:
            capture.full_writer(value,implementation)

    def load_arg(self, ptr, ty):
        shape=tensor_shape(ty); ety=elem_name(ty.element_type)
        return TensorVal(shape,_LoadedElements(self,ptr,prod(shape),ety),ety)

    def _parallel_generic(self,op):
        """Loop the shared scalar region over original memory-backed tensor maps."""
        if (op.name!='linalg.generic' or len(op.results)!=1
            or any('parallel' not in str(x) for x in op.properties['iterator_types'])
            or any(v not in self.ptrs for v in (*op.operands,*op.results))
            or any(x.name=='linalg.index' for x in op.regions[0].block.ops)):
            return False
        maps=[x.data for x in op.properties['indexing_maps']]
        shape=tensor_shape(op.results[0].type)
        if (len(shape)!=maps[-1].num_dims
            or any(not isinstance(e,AffineDimExpr) or e.position!=i for i,e in enumerate(maps[-1].results))
            or any(not isinstance(e,(AffineDimExpr,AffineConstantExpr)) for m in maps for e in m.results)):
            return False
        fn=self.fp.fb
        def body(flat):
            ivs=[]
            for i,size in enumerate(shape):
                val=flat
                pitch=prod(shape[i+1:])
                if pitch!=1: val=fn.udiv_i(val,fn.const(pitch))
                if i: val=fn.urem_i(val,fn.const(size))
                ivs.append(val)
            args=[]
            for operand,m,arg in zip(op.operands,maps,op.regions[0].block.args):
                ety=elem_name(operand.type.element_type)
                if not arg.uses:
                    args.append(self.zero(ety));continue
                offset=fn.const(0)
                operand_shape=tensor_shape(operand.type)
                for i,e in enumerate(m.results):
                    idx=ivs[e.position] if isinstance(e,AffineDimExpr) else fn.const(e.value)
                    pitch=prod(operand_shape[i+1:])
                    if pitch!=1: idx=fn.mul_i(idx,fn.const(pitch))
                    offset=fn.add_i(offset,idx)
                ptr=self.ptrs[operand]
                args.append(fn.load_i64(ptr,offset,ety) if ety.startswith('i') else self.fp.load(ptr,offset,ety))
            value=self._scalar_block(op.regions[0].block,args)
            ety=elem_name(op.results[0].type.element_type);ptr=self.ptrs[op.results[0]]
            if ety.startswith('i'):fn.store_i64(value,ptr,flat,ety)
            else:self.fp.store(value,ptr,flat,ety)
        fn.for_loop(0,prod(shape),1,body,retain_loop=True)
        self.vals[op.results[0]]=self.load_arg(self.ptrs[op.results[0]],op.results[0].type)
        self.materialized_results.add(op.results[0])
        self.record_full_writer(op.results[0],MixedHost._parallel_generic)
        return True

    def _conv(self,op):
        from convolution import emit_conv
        from merlin.llvmlower.host_tensor_codegen import attr_of
        if (len(op.results)!=1 or len(op.operands)!=3
            or any(str(v.type.element_type)!='i8' for v in op.operands[:2])
            or str(op.results[0].type.element_type)!='i32'):
            raise LoweringDeclined('source convolution requires signed i8 operands and exact i32 output',op=op.name)
        inp,weights,seed=op.operands
        owner=seed.owner
        initializer=(owner.operands[0].owner if owner.name in ('tensor.splat','linalg.fill') else None)
        if (initializer is None or initializer.name!='arith.constant'
            or int(initializer.properties['value'].value.data)!=0):
            raise LoweringDeclined('source convolution accumulator initializer is not proven zero',op=op.name)
        def pair(key):
            value=attr_of(op,key)
            if value is None: raise LoweringDeclined('convolution missing '+key,op=op.name)
            return tuple(map(int,value.get_values()))
        ishape=tensor_shape(inp.type); wshape=tensor_shape(weights.type)
        strides,dilation=pair('strides'),pair('dilations')
        if len(ishape)!=4 or len(wshape)!=4:
            raise LoweringDeclined('convolution requires rank-four NHWC and HWCF source tensors',op=op.name)
        n,h,w,ci=ishape;kh,kw,wci,co=wshape
        expected=(n,(h-(kh-1)*dilation[0]-1)//strides[0]+1,
            (w-(kw-1)*dilation[1]-1)//strides[1]+1,co)
        if ci!=wci or tensor_shape(op.results[0].type)!=expected:
            raise LoweringDeclined('convolution declared source geometry is inconsistent',op=op.name)
        receipt=emit_conv(self.enc,self.ptrs[inp],self.ptrs[weights],self.ptrs[op.results[0]],
            ishape,wshape,strides=strides,dilation=dilation,padding=(0,0),out_dtype='i32')
        self.device_receipts.append({'source_op_index':self.current_source_op_index,'emission':receipt})
        capture=getattr(self.fp.fb,'storage_capture',None)
        if capture is not None:
            capture.convolution_full_writer(op.results[0],inputs=(inp,weights),
                output_pointer=self.ptrs[op.results[0]],
                input_pointers=(self.ptrs[inp],self.ptrs[weights]),
                input_shape=ishape,weight_shape=wshape,strides=strides,dilation=dilation,
                emitter=emit_conv,receipt=receipt)
        self.vals[op.results[0]]=self.load_arg(self.ptrs[op.results[0]],op.results[0].type)

    def _clamp_narrow(self,op):
        """Use the exact saturated accumulator readout for signed i32-to-i8."""
        from linalg_device import _constant, _identity
        from elementwise import emit_movement
        if (op.name!='linalg.generic' or len(op.operands)!=2 or len(op.results)!=1
            or any('parallel' not in str(x) for x in op.properties['iterator_types'])
            or any(v not in self.ptrs for v in (*op.operands,*op.results))
            or str(op.operands[0].type.element_type)!='i32'
            or str(op.results[0].type.element_type)!='i8'):
            return False
        shape=tensor_shape(op.results[0].type)
        if (tensor_shape(op.operands[0].type)!=shape
            or any(not _identity(m.data,len(shape)) for m in op.properties['indexing_maps'])):
            return False
        body=list(op.regions[0].block.ops);args=op.regions[0].block.args
        if (len(args)!=2 or args[1].uses or [x.name for x in body]!=
            ['arith.maxsi','arith.minsi','arith.trunci','linalg.yield']):return False
        def bound(operation,input_value,bound_value):
            return any(operation.operands[i]==input_value and _constant(operation.operands[1-i])==bound_value for i in (0,1))
        if (not bound(body[0],args[0],-128) or not bound(body[1],body[0].results[0],127)
            or body[2].operands[0]!=body[1].results[0]
            or body[3].operands[0]!=body[2].results[0]):return False
        emit_movement(self.enc,self.ptrs[op.operands[0]],self.ptrs[op.results[0]],shape,
            src_dtype='i32',output_dtype='i8')
        result=op.results[0]
        self.vals[result]=self.load_arg(self.ptrs[result],result.type)
        self.materialized_results.add(result)
        return True

    def _scalar_op(self, op, ins):
        from linalg_dynamic_extract import try_lower_scalar
        handled, value = try_lower_scalar(self, op, ins)
        if handled:
            return value
        from linalg_transcendentals import try_lower_scalar as try_host_math
        handled, value = try_host_math(self, op, ins)
        if handled:
            return value
        if op.name == "math.roundeven":
            return self.fp.round_near_even(ins[0])
        if op.name in ('arith.cmpf','arith.cmpi'):
            from merlin.llvmlower.host_tensor_codegen import attr_of
            predicate=int(attr_of(op,'predicate').value.data)
            fn=self.fp.fb
            if op.name=='arith.cmpf':
                comparison=llvm.FCmpOp(ins[0],ins[1],IntegerAttr(predicate,i64))
            else:
                flags=('EQ','NE','SLT','SLE','SGT','SGE','ULT','ULE','UGT','UGE')
                if not 0<=predicate<len(flags): raise LoweringDeclined('invalid integer predicate',op=op.name)
                left,right=ins
                if predicate>=6:
                    width=op.operands[0].type.width.data
                    if width<64:
                        mask=fn.const((1<<width)-1)
                        left,right=fn.and_i(left,mask),fn.and_i(right,mask)
                comparison=llvm.ICmpOp(left,right,IntegerAttr(getattr(llvm.ICmpPredicateFlag,flags[predicate]).to_int(),i64))
            value=fn.add(comparison).results[0]
            return fn.add(llvm.ZExtOp(value,i64)).results[0]
        if op.name=='arith.select':
            fn=self.fp.fb
            condition=fn.add(llvm.ICmpOp(ins[0],fn.const(0),
                IntegerAttr(llvm.ICmpPredicateFlag.NE.to_int(),i64))).results[0]
            return fn.add(llvm.SelectOp(condition,ins[1],ins[2])).results[0]
        return super()._scalar_op(op, ins)

    def _tensor_op(self, op):
        self.last_lowering=('host','shared_host_tensor','compiler_or_proof_gap')
        from linalg_dynamic_insert import try_lower as try_dynamic_insert
        if try_dynamic_insert(self, op):
            return
        from linalg_pool import try_lower as try_pool
        if try_pool(self, op):
            return
        from linalg_device_pad import try_lower as try_pad
        if try_pad(self, op):
            return
        if op in getattr(self,'elided_rhs_transposes',()):
            self.last_lowering=('storage','fused_transposed_rhs_view',None)
            return
        if op.name=='tensor.empty' and len(op.results)==1 and op.results[0] in self.ptrs:
            # tensor.empty introduces undefined elements, requiring no initialization.
            result=op.results[0]
            self.vals[result]=self.load_arg(self.ptrs[result],result.type)
            self.materialized_results.add(result)
            self.last_lowering=('storage','undefined_tensor_allocation',None)
            return
        if op.name == 'linalg.conv_2d_nhwc_hwcf':
            self.last_lowering=('device','integer_convolution',None)
            return self._conv(op)
        if not _is_device(op):
            if self._clamp_narrow(op):
                self.last_lowering=('device','saturated_accumulator_readout',None);return
            from linalg_activation import try_lower as try_activation
            if try_activation(self,op):
                self.last_lowering=('device','exact_signed_i8_relu',None);return
            from linalg_views import try_lower as try_view
            if try_view(self,op):
                self.last_lowering=('storage','immutable_tensor_alias',None);return
            from linalg_device_zero import try_lower as try_zero
            if try_zero(self,op):
                return
            from linalg_device_word_broadcast import try_lower as try_word_broadcast
            if try_word_broadcast(self,op):
                return
            from linalg_device_broadcast import try_lower as try_broadcast
            if try_broadcast(self,op):
                return
            from linalg_device_concat import try_lower as try_concat
            if try_concat(self,op):
                return
            from linalg_device_word_insert import try_lower as try_word_insert
            if try_word_insert(self,op):
                return
            from linalg_device_insert import try_lower as try_insert
            if try_insert(self,op):
                return
            from linalg_device_gather import try_lower as try_gather
            if try_gather(self,op):
                return
            from linalg_device_word_slice import try_lower as try_word_slice
            if try_word_slice(self,op):
                return
            from linalg_device_slice import try_lower as try_slice
            if try_slice(self,op):
                return
            from linalg_device_word_layout import try_lower as try_word_layout
            if try_word_layout(self,op):
                return
            from linalg_device_layout import try_lower as try_layout
            if try_layout(self,op):
                return
            from linalg_device_projected_copy import try_lower as try_projected_copy
            if try_projected_copy(self,op):
                return
            from linalg_device import try_lower as try_device
            if try_device(self,op):
                self.last_lowering=('device','proven_integer_pointwise',None);return
            from linalg_host import try_lower
            if try_lower(self,op):
                self.last_lowering=('host','shared_counted_tensor','compiler_or_proof_gap');return
            if self._parallel_generic(op):
                self.last_lowering=('host','shared_counted_scalar_region','compiler_or_proof_gap');return
            result=super()._tensor_op(op)
            from source_bool_empty import classify_private_empty
            if classify_private_empty(self,op):
                self.last_lowering=('storage','private_ssa_empty_full_writer',None)
            if op.name=='arith.constant':
                self.last_lowering=('preparation','shared_scalar_constant',None)
            return result
        self.last_lowering=('device','signed_integer_dense',None)
        lhs,rhs,seed=op.operands[:3]
        a,b=tensor_shape(lhs.type),tensor_shape(rhs.type)
        c=tensor_shape(op.results[0].type)
        if len(a)<2 or len(b)<2 or len(c)<2 or a[-1]!=b[-2] or c[-1]!=b[-1] or c[:-1]!=a[:-1]:
            raise LoweringDeclined('integer contraction requires a legal dense layout adapter',op=op.name)
        if op.name=='linalg.generic':
            # The complete scalar region proves exact signed modular i32 recurrence.
            body=list(op.regions[0].block.ops)
            if [x.name for x in body] != ['arith.extsi','arith.extsi','arith.muli','arith.addi','linalg.yield']:
                raise LoweringDeclined('integer generic has no exact multiply-add recurrence',op=op.name)
            args=op.regions[0].block.args
            if len(args)!=3:
                raise LoweringDeclined('integer generic requires three scalar operands',op=op.name)
            if (body[0].operands[0]!=args[0] or body[1].operands[0]!=args[1]
                or set(body[2].operands)!={body[0].results[0],body[1].results[0]}
                or set(body[3].operands)!={args[2],body[2].results[0]}
                or body[4].operands[0]!=body[3].results[0]
                or any(str(x.results[0].type)!='i32' for x in body[:4])):
                raise LoweringDeclined('integer generic recurrence operands differ',op=op.name)
            maps=[x.data for x in op.properties['indexing_maps']]
            # These exact affine maps prove contiguous row-major dense ownership.
            am,bm,cm=maps
            if not all(isinstance(x,AffineDimExpr) for z in maps for x in z.results):
                raise LoweringDeclined('nonprojected integer contraction needs gather adapter',op=op.name)
            ad,bd,cd=[[x.position for x in z.results] for z in maps]
            iterators=list(op.properties['iterator_types'])
            if (any(m.num_symbols or m.num_dims!=len(iterators) for m in maps)
                or set(ad+bd+cd)!=set(range(len(iterators)))
                or any(('reduction' if i==ad[-1] else 'parallel') not in str(kind)
                    for i,kind in enumerate(iterators))):
                raise LoweringDeclined('integer generic iteration domain differs from dense recurrence',op=op.name)
            prefix_ok=(cd[:-1]==ad[:-1] and (len(bd)==2 or ad[:-2]==bd[:-2]==cd[:-2]))
            unique_ok=(len(set(ad))==len(ad) and len(set(bd))==len(bd) and len(set(cd))==len(cd)
                and ad[-1] not in cd and cd[-1] not in ad)
            if not prefix_ok or not unique_ok or ad[-1]!=bd[-2] or cd[-1]!=bd[-1] or ad[-2]!=cd[-2]:
                raise LoweringDeclined('permuted integer contraction needs layout adapter',op=op.name)
        # A fresh hardware product followed by exact accumulator-memory addition
        # preserves arbitrary source i32 initializers without host contraction work.
        owner=seed.owner
        owner_name=getattr(owner,'name',None)
        zero=(owner_name=='tensor.splat' and owner.operands[0].owner.name=='arith.constant'
              and int(owner.operands[0].owner.properties['value'].value.data)==0)
        if owner_name=='linalg.fill':
            v=owner.operands[0].owner
            zero=v.name=='arith.constant' and int(v.properties['value'].value.data)==0
        virtual_zero=getattr(self,'virtual_zero_roots',{}).get(op)
        if virtual_zero is not None and not zero:
            raise LoweringDeclined('virtual initializer lacks the selected dense zero path',op=op.name)
        batch=prod(b[:-2]) if len(b)>2 else 1
        m=prod(a[:-1])//batch
        k,n=a[-1],b[-1]
        fused=getattr(self,'integer_epilogues',{}).get(op)
        output=op.results[0] if fused is None else fused.output
        options=({} if fused is None else dict(out_dtype='i8' if fused.narrowed else 'i32',
            scale=1.0,relu=fused.relu,bias=None if fused.bias is None else self.ptrs[fused.bias]))
        for bi in range(batch):
            transpose=getattr(self,'transposed_rhs_views',{}).get(op)
            if transpose is not None:
                from attention import emit_qk
                receipt={'selected_schedule':'transposed_rhs_tiled',
                    'source_view_op_index':self.transposed_rhs_source_indices[op],
                    'emissions':emit_qk(self.enc,self.ptrs[lhs],self.ptrs[transpose.operands[0]],
                        self.ptrs[op.results[0]],a,tensor_shape(transpose.operands[0].type),schedule='tiled')}
                self.last_lowering=('device','transposed_rhs_tiled',None)
            else:
                receipt=emit_dense(self.enc,self.enc.offset(self.ptrs[lhs],bi*m*k),
                    self.enc.offset(self.ptrs[rhs],bi*k*n),
                    self.enc.offset(self.ptrs[output],bi*m*n*(1 if fused is not None and fused.narrowed else 4)),m,k,n,**options)
            if fused is not None:
                receipt={'selected_schedule':'fused_source_integer_epilogue',
                    'source_integer_epilogue':fused.receipt(self.source_positions),'dense_emission':receipt}
                self.last_lowering=('device','fused_source_integer_epilogue',None)
            if virtual_zero is not None:
                receipt={**receipt,'virtual_zero_initializer':virtual_zero.record()}
            self.device_receipts.append({'source_op_index':self.current_source_op_index,
                'batch_index':bi,'emission':receipt})
        if not zero:
            from linalg_device import _integer_sum
            _integer_sum(self,op.results[0],seed,op.results[0],c,'i32','i32')
        capture=getattr(self.fp.fb,'storage_capture',None)
        if (capture is not None and batch==1 and zero and fused is None
            and getattr(self,'transposed_rhs_views',{}).get(op) is None):
            # emit_dense just consumed its checked complete MxKxN schedule,
            # with default packed output pitch and its real final fence.
            capture.device_full_writer(output,inputs=(lhs,rhs),
                output_pointer=self.ptrs[output],input_pointers=(self.ptrs[lhs],self.ptrs[rhs]),
                shape=(m,n),emitter=emit_dense,receipt=receipt)
        self.vals[output]=self.load_arg(self.ptrs[output],output.type)
        if fused is not None:self.materialized_results.add(output)


def compile_linalg(source: str):
    module=parse_mlir_text(source)
    funcs=[f for f in module.body.block.ops if f.name=='func.func' and f.body.blocks]
    if len(funcs)!=1 or len(funcs[0].body.blocks)!=1:
        raise ValueError('one closed single-block tensor function required')
    f=funcs[0]; block=f.body.block
    original=[o for o in block.ops if o.name!='func.return']
    original_operands={op:tuple(op.operands) for op in original}
    original_results={op:tuple(op.results) for op in original}
    import source_placement as placement_metadata
    original_capabilities=placement_metadata.source_capabilities(module,original)
    returned=list(block.last_op.operands)
    owners={o:i for i,o in enumerate(original)}
    value_owner={v:(i,j) for i,o in enumerate(original) for j,v in enumerate(o.results)}
    import linalg_zero_initializers
    zero_candidates=linalg_zero_initializers.proposals(module,original)
    import source_layout_preparation
    layout_preparation=source_layout_preparation.prepare(module,
        source_sha256=hashlib.sha256(source.encode()).hexdigest())
    layout_bindings=dict(layout_preparation.pop('operation_bindings'))
    import source_preparation
    preparation_plan=(source_preparation.plan_tensor_preparations(module,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        operations=tuple(op for op in block.ops if op.name!='func.return'),
        source_positions={op:owners[op] for op in block.ops if op.name!='func.return'})
        if source_preparation.enabled() else None)
    # Keep a source correspondence through the shared exact quantization rewrite.
    generated={}
    replacements={}
    quant_groups={}
    def rewrite(old,new,replacement):
        index=owners[old]
        for op in new: generated[op]=index
        replacements[old]=replacement
        quant_groups[index]=tuple(new)
    lower_quant_ext(module,on_rewrite=rewrite)
    import linalg_epilogue
    integer_epilogues=linalg_epilogue.plan(block,_is_device)
    fused_private_results={value for fusion in integer_epilogues.values()
        for operation in (fusion.root,*fusion.operations[:-1]) for value in operation.results}
    def emitted_value(value):
        owner=value_owner.get(value)
        if owner is None:return value
        index,result_index=owner
        return replacements.get(original[index],original[index]).results[result_index]
    preparation_aliases=({emitted_value(duplicate):emitted_value(retained)
        for duplicate,retained in preparation_plan.value_aliases.items()}
        if preparation_plan is not None else {})
    preparation_sites=({alias.source_op_index:alias for alias in preparation_plan.aliases}
        if preparation_plan is not None else {})
    from attention_schedule import transposed_rhs_views, complete_rhs_views
    rhs_views=transposed_rhs_views(block,_is_device)
    rhs_views={consumer:transpose for consumer,transpose in rhs_views.items() if consumer not in integer_epilogues}
    if preparation_plan is not None:
        shared_preparations={op for alias in preparation_plan.aliases
            for op in (alias.operation,alias.retained_operation)}
        # A shared preparation must retain its value for every alias observer.
        rhs_views={consumer:transpose for consumer,transpose in rhs_views.items()
            if transpose not in shared_preparations}
    rhs_views=complete_rhs_views(rhs_views)
    elided_rhs_transposes=set(rhs_views.values())
    elided_rhs_results={op.results[0] for op in elided_rhs_transposes}
    import linalg_shared_input
    sibling_groups=(tuple(group for group in linalg_shared_input.find_source_groups(module)
        if not any(root in rhs_views or root in integer_epilogues for root in group)) if linalg_shared_input.enabled() else ())
    sibling_starts={group[0]:group for group in sibling_groups}
    import attention_region
    head_groups=(attention_region.find_head_row_groups(module,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        excluded=set(rhs_views)|set(integer_epilogues)|
            {root for group in sibling_groups for root in group})
        if attention_region.enabled() else ())
    if head_groups:
        from dataclasses import replace
        head_groups=tuple(replace(group,source_indices=tuple(owners[root] for root in group.members))
            for group in head_groups)
    head_starts={group.members[0]:group for group in head_groups}
    zero_initializers=linalg_zero_initializers.select(zero_candidates,is_device=_is_device,
        # Head groups use the same exact zero-seeded dense realization. Their
        # source proof rejects alternate seeds and observers; ordinary dense
        # fallback also consumes VirtualInitializer without reading its pointer.
        excluded_roots=set(rhs_views)|set(integer_epilogues)|
            {root for group in sibling_groups for root in group},
        aliased_values=set(preparation_aliases)|set(preparation_aliases.values()))
    virtual_zero_ops={op:item for item in zero_initializers for _,op in item.storage_operations}
    virtual_zero_values={op.results[0] for op in virtual_zero_ops}
    import attention_destinations
    destination_plan=(attention_destinations.plan_attention_destinations(module,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),source_positions=owners,
        excluded_roots={op for op in block.ops if not _is_device(op)}|
            set(rhs_views)|set(integer_epilogues)|
            {root for group in sibling_groups for root in group}|
            {root for group in head_groups for root in group.members})
        if attention_destinations.ENABLED else None)
    destination_bindings=({binding.root:binding for binding in destination_plan.bindings}
        if destination_plan is not None else {})
    destination_values=(destination_plan.private_values if destination_plan is not None else frozenset())
    from linalg_views import is_aliasable
    aliasable_results={op.results[0] for op in block.ops if is_aliasable(op)}
    tensors={}; names={}; source_values=[]
    def logical_names(key,count,fallback):
        attr=f.attributes.get(key,f.properties.get(key))
        if attr is None: return fallback
        values=[item.data for item in attr]
        if len(values)!=count or any(not isinstance(x,str) or not x for x in values):
            raise ValueError('source logical I/O attribute has invalid arity or names')
        return values
    input_names=logical_names('merlin.logical_inputs',len(block.args),
        ['arg'+str(i) for i in range(len(block.args))])
    output_names=logical_names('merlin.logical_outputs',len(returned),None)
    logical_outputs=dict(zip(returned,output_names)) if output_names is not None else {}

    def declare(v,name,role):
        if not isinstance(v.type,TensorType): return
        tensors[name]={'shape':list(tensor_shape(v.type)),'dtype':elem_name(v.type.element_type),'role':role}
        names[v]=name
    for arg,name in zip(block.args,input_names): declare(arg,name,'input')
    for i,op in enumerate(original):
        if op in layout_bindings:continue
        replacement=replacements.get(op,op)
        for j,v in enumerate(replacement.results):
            if (not isinstance(v.type,TensorType) or v in aliasable_results
                or v in elided_rhs_results or v in preparation_aliases): continue
            if v in fused_private_results or v in virtual_zero_values or v in destination_values:continue
            if str(v.type.element_type)=='i1' and op.results[j] not in returned:
                # Predicate tensors stay in the shared host SSA representation;
                # the public tensor ABI has no byte-storage contract for i1.
                continue
            old=op.results[j]
            name=logical_outputs.get(old,'v'+str(i)+'_'+str(j))
            declare(v,name,'output' if old in returned else 'intermediate')
            source_values.append({'op_index':i,'result_index':j,'tensor':name})
    for duplicate,retained in preparation_aliases.items():
        if retained in names:names[duplicate]=names[retained]
    # Quantization helper tensors are compiler-private and never cross the ABI.
    # The HostLinalg scalar SSA owns them; contractions require only named source operands.
    retvals=list(block.last_op.operands)
    outputs=[names[v] for v in retvals]
    order=list(tensors)
    fn=_SourceFnBuilder([PTR]*len(order)); ptr_by_name=dict(zip(order,fn.entry.args))
    import source_storage_reuse
    storage_capture=None
    if source_storage_reuse.ENABLED:
        from source_storage_adapter import StorageEmissionCapture
        storage_capture=StorageEmissionCapture(fn,hashlib.sha256(source.encode()).hexdigest())
        fn.storage_capture=storage_capture
    ptrs={v:ptr_by_name[n] for v,n in names.items()}
    enc=Encoder(fn)
    if any(_is_device(op) or op.name == "linalg.conv_2d_nhwc_hwcf" for op in block.ops):
        enc.configure_execute()
    host=MixedHost(_SourceFpBuilder(fn),[ptrs[a] for a in block.args],
        [ptr_by_name[n] for n in outputs],enc=enc,ptrs=ptrs,element_budget=4_000_000)
    import json
    from pathlib import Path
    settings_path=Path(__file__).with_name('DENSE_SCHEDULE.json')
    settings=json.loads(settings_path.read_text()) if settings_path.exists() else {}
    host.reduction_carried=settings.get('host_reduction_carried',False)
    if type(host.reduction_carried) is not bool:
        raise ValueError('host_reduction_carried must be Boolean')
    host.aliasable_results=aliasable_results
    host.integer_epilogues=integer_epilogues
    host.virtual_zero_roots={item.root:item for item in zero_initializers}
    host.source_positions=owners
    host.transposed_rhs_views=rhs_views
    host.transposed_rhs_source_indices={consumer:owners[transpose] for consumer,transpose in rhs_views.items()}
    host.elided_rhs_transposes=elided_rhs_transposes
    # Track actual generated operations without rebuilding shared arithmetic rules.
    for a in block.args: host.vals[a]=host.load_arg(ptrs[a],a.type)
    skipped=set()
    placement_events={i:[] for i in range(len(original))}
    def record_placement(index,kind,form,reason,receipt_start,receipt_end,lowered_ops):
        selected=[]
        for receipt in host.device_receipts[receipt_start:receipt_end]:
            emission=receipt.get('emission') or {}
            selected.append({key:emission[key] for key in
                ('selected_schedule','completion','completion_mapping','schedule_plan_sha256') if key in emission})
        placement_events[index].append({'kind':kind,'lowering':form,
            'host_reason':reason,'schedule_receipt_range':[receipt_start,receipt_end],
            'selected_device_forms':selected,'lowered_operations':lowered_ops})
    for removed in layout_bindings:
        record_placement(owners[removed],'storage','composed_exact_static_slice',None,
            0,0,[removed.name])
    for op in list(block.ops):
        if op.name=='func.return': break
        if op in skipped: continue
        index=owners.get(op,generated.get(op))
        host.current_source_op_index=index
        fn.current_source_index=index
        if storage_capture is not None:storage_capture.begin(index)
        receipt_start=len(host.device_receipts)
        handled=False
        if op in destination_bindings:
            binding=destination_bindings[op]
            ptrs[binding.value]=enc.offset(ptrs[binding.destination],binding.element_offset*4)
        if destination_plan is not None and op in destination_plan.concats:
            result=op.results[0]
            host.vals[result]=host.load_arg(ptrs[result],result.type)
            host.materialized_results.add(result)
            record_placement(index,'storage','device_readout_to_exact_concat_slices',None,
                receipt_start,len(host.device_receipts),[op.name])
            if storage_capture is not None:storage_capture.finish()
            continue
        if op in virtual_zero_ops:
            record_placement(index,'storage','virtual_dense_zero_initializer',None,
                receipt_start,len(host.device_receipts),[op.name])
            if storage_capture is not None:storage_capture.finish()
            continue
        if index in preparation_sites:
            alias=preparation_sites[index]
            for duplicate,retained in alias.result_pairs:
                duplicate,retained=emitted_value(duplicate),emitted_value(retained)
                host.vals[duplicate]=host.get(retained)
                if retained in ptrs:ptrs[duplicate]=ptrs[retained]
                if retained in host.materialized_results:host.materialized_results.add(duplicate)
            if index in quant_groups:skipped.update(quant_groups[index])
            record_placement(index,'storage','shared_exact_preparation_alias',None,
                receipt_start,len(host.device_receipts),[alias.operation.name])
            if storage_capture is not None:storage_capture.finish()
            continue
        if op in head_starts:
            group=head_starts[op]
            receipt=attention_region.emit_head_row_group(enc,group,ptrs)
            if receipt is not None:
                for root in group.members:
                    result=root.results[0]
                    host.vals[result]=host.load_arg(ptrs[result],result.type)
                    host.materialized_results.add(result)
                host.device_receipts.append({'source_op_indices':[owners[root] for root in group.members],
                    'emission':receipt})
                for root in group.members:
                    record_placement(owners[root],'device','shared_rhs_head_rows',None,
                        receipt_start,len(host.device_receipts),[root.name])
                skipped.update(group.members[1:]);handled=True
        if not handled and op in sibling_starts:
            siblings=sibling_starts[op]
            shape=tensor_shape(op.operands[0].type)
            widths=[tensor_shape(root.results[0].type)[1] for root in siblings]
            receipt=linalg_shared_input.emit_shared_input(enc,ptrs[op.operands[0]],
                [(ptrs[root.operands[1]],ptrs[root.results[0]]) for root in siblings],
                shape[0],shape[1],widths,source_indices=[owners[root] for root in siblings],
                completion="epochs" if linalg_shared_input.epochs_enabled() else "synchronous")
            if receipt is not None:
                for root in siblings:
                    result=root.results[0]
                    host.vals[result]=host.load_arg(ptrs[result],result.type)
                    host.materialized_results.add(result)
                host.device_receipts.append({'source_op_indices':[owners[root] for root in siblings],'emission':receipt})
                for root in siblings:
                    record_placement(owners[root],'device','shared_input_products',None,
                        receipt_start,len(host.device_receipts),[root.name])
                skipped.update(siblings[1:]);handled=True
        if not handled and op in generated and index in quant_groups and op is quant_groups[index][0]:
            import linalg_host
            lower_group=getattr(linalg_host,'try_lower_quant_group',None)
            if lower_group is not None:
                handled=lower_group(host,quant_groups[index])
                if handled:
                    skipped.update(quant_groups[index])
                    record_placement(index,'host','shared_counted_quantization',
                        'compiler_or_proof_gap',receipt_start,len(host.device_receipts),
                        [x.name for x in quant_groups[index]])
        if not handled:
            host._tensor_op(op)
            kind,form,reason=host.last_lowering
            record_placement(index,kind,form,reason,receipt_start,len(host.device_receipts),[op.name])
            if op in integer_epilogues:
                for tail in integer_epilogues[op].operations:
                    record_placement(owners[tail],'device','fused_source_integer_epilogue',None,
                        receipt_start,len(host.device_receipts),[tail.name])
                    skipped.add(tail)
        for v in op.results:
            if v in ptrs and v not in host.materialized_results and not _is_device(op) and op.name != "linalg.conv_2d_nhwc_hwcf":
                host.store_result(ptrs[v],host.get(v))
                if storage_capture is not None:storage_capture.full_writer(v)
        if storage_capture is not None:storage_capture.finish()
    for b in fn.region.blocks:
        for x in b.ops:
            x.attributes.setdefault('merlin.global_task',IntegerAttr(0,i64))
    kernel=llvm.FuncOp('gemmini_kernel',llvm.LLVMFunctionType([PTR]*len(order)),body=fn.finish())
    storage_receipt=(storage_capture.apply(kernel,names,tensors,order,source_values)
        if storage_capture is not None else None)
    # The return belongs to the same source task as the final writer.
    fn.blk.last_op.attributes['merlin.global_task']=IntegerAttr(0,i64)
    nins=sum(len(tuple(b.ops)) for b in fn.region.blocks)
    source_instruction_indices={i:[] for i in range(len(original))}
    source_target_sites={i:[] for i in range(len(original))}
    for instruction_index,instruction in enumerate(x for b in fn.region.blocks for x in b.ops):
        attr=instruction.attributes.get('merlin.source_op_index')
        if attr is not None:
            owner=int(attr.value.data)
            if owner in source_instruction_indices:
                source_instruction_indices[owner].append(instruction_index)
                asm=instruction.properties.get('asm_string')
                if instruction.name=='llvm.inline_asm' and getattr(asm,'data','').startswith('.insn r '):
                    source_target_sites[owner].append(instruction_index)
    def ranges(indices):
        result=[]
        for i in indices:
            if result and result[-1][1]==i:result[-1][1]=i+1
            else:result.append([i,i+1])
        return result
    def boundary(value):
        item={'type':str(value.type)}
        if isinstance(value.type,TensorType):
            item.update(shape=list(tensor_shape(value.type)),dtype=elem_name(value.type.element_type))
        if value in names:item['tensor']=names[value]
        return item
    source_placement=[]
    for index,op in enumerate(original):
        events=placement_events[index]
        kinds=list(dict.fromkeys(event['kind'] for event in events))
        provenance={key:(value.data if isinstance(getattr(value,'data',None),(str,int,float,bool)) else str(value))
            for key,value in {**op.properties,**op.attributes}.items()
            if key.startswith('prov.') or key.startswith('merlin.source') or key in ('region','region_id')}
        source_placement.append({'source_op_index':index,'operation':op.name,
            'source_provenance':provenance,'task_index':0,
            'scalar_operations':[inner.name for region in op.regions for block in region.blocks for inner in block.ops],
            'inputs':[boundary(v) for v in original_operands[op]],
            'outputs':[boundary(v) for v in original_results[op]],
            'kind':kinds[0] if len(kinds)==1 else 'mixed' if kinds else 'unattributed',
            'lowerings':events,'instruction_ranges':ranges(source_instruction_indices[index])})
        if op in getattr(host,'source_empty_storage_proofs',{}):
            source_placement[-1]['source_storage_proof']=host.source_empty_storage_proofs[op]
        if op in elided_rhs_transposes:
            source_placement[-1]['fused_into_source_op_index']=owners[next(
                consumer for consumer,transpose in rhs_views.items() if transpose is op)]
        if op in virtual_zero_ops:
            source_placement[-1]['fused_into_source_op_index']=virtual_zero_ops[op].root_source_op_index
        if op in layout_bindings:
            source_placement[-1]['fused_into_source_op_index']=owners[layout_bindings[op]]
        if op in destination_bindings:
            source_placement[-1]['destination_source_op_index']=destination_bindings[op].concat_source_op_index
        if destination_plan is not None and op in destination_plan.concats:
            source_placement[-1]['fused_from_source_op_indices']=[binding.source_op_index
                for binding in destination_plan.bindings if binding.destination is op.results[0]]
        if index in preparation_sites:
            source_placement[-1]['retained_source_op_index']=preparation_sites[index].retained_source_op_index
        for root,fusion in integer_epilogues.items():
            if op in fusion.operations:source_placement[-1]['fused_into_source_op_index']=owners[root]
    plan={'schema':'mixed_program_plan_v1','source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'source_op_count':len(original),'tasks':[{'task_index':0,'kind':'mixed_program',
        'source_op_indices':list(range(len(original))),'instruction_start':0,'instruction_end':nins,
        'reads':input_names,
        'writes':[n for n in order if tensors[n]['role']!='input']}],
        'schedule_instruction_count':nins,'prologue_instruction_range':[0,0],
        'epilogue_instruction_range':[nins,nins],'entry_bindings':input_names,
        'source_values':source_values,'output_bindings':outputs}
    cb={'abi_version':'0.1','target':'gemmini','commands':[], 'tensors':tensors,
        'kernel_abi':{'kind':'whole_program','args':[{'tensor':n,'access':'read' if tensors[n]['role']=='input' else 'write'} for n in order],'outputs':outputs},
        'params':{'global_program_plan':plan,'source_schedule_emissions':host.device_receipts,
            'source_placement':source_placement}}
    cb['params']['lane_placement']=placement_metadata.lane_placements(
        source_placement,host.device_receipts,original_capabilities,source_target_sites,
        source_sha256=plan['source_sha256'])
    if preparation_plan is not None:
        cb['params']['source_preparation']=preparation_plan.receipt()
    if layout_preparation['enabled']:
        cb['params']['source_layout_preparation']=layout_preparation
    if destination_plan is not None:
        cb['params']['attention_destinations']=destination_plan.receipt()
    if storage_receipt is not None:
        cb['params']['source_storage_reuse']=storage_receipt
    if zero_initializers:
        cb['params']['virtual_zero_initializers']=[item.record() for item in zero_initializers]
    if getattr(host, 'required_host_math_symbols', None):
        from linalg_transcendentals import dependency_contract
        cb['params']['required_host_math_symbols'] = dependency_contract(host)
    from merlin.xdsl_dialects._common import text
    return cb,text(kernel_module(kernel),generic=True)
