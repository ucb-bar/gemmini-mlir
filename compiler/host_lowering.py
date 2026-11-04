"""Typed tensor adapters over Merlin's admitted host scalar builders.

These adapters retain source arithmetic order. Integer contractions are deliberately
excluded: graph_lowering dispatches them through the device scheduler.
"""
from dataclasses import dataclass
from math import prod
from xdsl.dialects import llvm
from xdsl.dialects.builtin import FloatAttr, Float32Type, TensorType, i64, i1
from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr, AffineBinaryOpKind
from merlin.llvmlower.host_scalar_builder import FpBuilder
from merlin.targetgen.contract import linalg_iface as li


@dataclass
class TensorValue:
    ptr: object
    shape: tuple
    dtype: str

    @property
    def count(self): return prod(self.shape)


class DominatingFpBuilder(FpBuilder):
    def fconst(self, value):
        key = float(value)
        if key in self._fconsts: return self._fconsts[key]
        op = llvm.ConstantOp(FloatAttr(key, Float32Type()), Float32Type())
        term = self.fb.entry.last_op
        if term is not None and term.has_trait(llvm.IsTerminator):
            self.fb.entry.insert_op_before(op, term)
        else: self.fb.entry.add_op(op)
        self._fconsts[key] = op.results[0]
        return op.results[0]


class HostTensorLowerer:
    def __init__(self, emitter, allocate, *, elide_overwritten_init=False):
        self.e, self.b, self.allocate = emitter, emitter.b, allocate
        self.fp = DominatingFpBuilder(self.b)
        self.values = {}
        self.bound_outputs = {}
        self.declarations = {}
        self.active_coords = []
        self.elide_overwritten_init = elide_overwritten_init
        self.preparation_optimizations = []
        self.source_op_index = None
        # Graph code may supply source-aware lifetime allocation for results.
        # Anonymous lowering temporaries keep their separate allocator contract.
        self.result_allocator = None

    def const(self, x): return self.b.const(int(x))
    def index(self, coords, shape):
        value = self.const(0)
        for c, n in zip(coords, shape): value = self.b.add_i(self.b.mul_i(value, self.const(n)), c)
        return value
    def coords(self, iv, shape):
        result = []
        for i, n in enumerate(shape):
            stride = prod(shape[i+1:])
            result.append(self.b.urem_i(self.b.udiv_i(iv, self.const(stride)), self.const(n)))
        return result
    def expr(self, expr, dims):
        if isinstance(expr, AffineDimExpr): return dims[expr.position]
        if isinstance(expr, AffineConstantExpr): return self.const(expr.value)
        a, b = self.expr(expr.lhs, dims), self.expr(expr.rhs, dims)
        funcs = {AffineBinaryOpKind.Add:self.b.add_i, AffineBinaryOpKind.Mul:self.b.mul_i,
                 AffineBinaryOpKind.Mod:self.b.srem_i, AffineBinaryOpKind.FloorDiv:self.b.sdiv_i}
        if expr.kind not in funcs: raise ValueError('unsupported affine map operation')
        return funcs[expr.kind](a,b)
    def load(self, tensor, idx):
        if tensor.dtype in ('f32','bf16'): return self.fp.load(tensor.ptr, idx, 'i8' if tensor.dtype=='i1' else tensor.dtype)
        return self.b.load_i64(tensor.ptr, idx, 'i8' if tensor.dtype=='i1' else tensor.dtype)
    def store(self, value, tensor, idx):
        if tensor.dtype in ('f32','bf16'): return self.fp.store(value, tensor.ptr, idx, 'i8' if tensor.dtype=='i1' else tensor.dtype)
        return self.b.store_i64(value, tensor.ptr, idx, 'i8' if tensor.dtype=='i1' else tensor.dtype)
    def copy(self, src, dst):
        if src.count != dst.count: raise ValueError('copy tensor extent mismatch')
        self.e.for_loop(0,src.count,1,lambda i:self.store(self.load(src,i),dst,i))
    def result(self, value):
        typ=value.type
        if not isinstance(typ,TensorType): raise ValueError('tensor result required')
        count=prod(typ.get_shape());dtype=str(typ.element_type)
        if value in self.bound_outputs:ptr=self.bound_outputs[value]
        elif self.result_allocator is not None:ptr=self.result_allocator(value,count,dtype)
        else:ptr=self.allocate(count,dtype)
        result=TensorValue(ptr,tuple(typ.get_shape()),dtype)
        self.values[value]=result
        return result
    def scalar(self, op, args):
        kind=li._op_name(op); fp=self.fp; b=self.b
        binary={'arith.addf':fp.fadd,'arith.subf':fp.fsub,'arith.mulf':fp.fmul,'arith.divf':fp.fdiv,
                'arith.maximumf':fp.fmax,'arith.maxnumf':fp.fmax,'arith.minimumf':fp.fmin,'arith.minnumf':fp.fmin,
                'arith.addi':b.add_i,'arith.subi':b.sub_i,'arith.muli':b.mul_i,'arith.divsi':b.sdiv_i,
                'arith.maxsi':b.smax,'arith.minsi':b.smin,'arith.xori':b.xor_i,'arith.ori':b.or_i,'arith.andi':b.and_i}
        unary={'math.exp':fp.exp,'math.erf':fp.erf,'math.rsqrt':fp.rsqrt,'math.sqrt':fp.sqrt,
               'math.log':fp.log,'math.tanh':fp.tanh,'math.absf':fp.fabs,'arith.negf':fp.fneg,
               'math.roundeven':fp.round_near_even,'arith.sitofp':fp.sitofp,'arith.fptosi':fp.fptosi}
        if kind=='linalg.index':return self.active_coords[op.dim.value.data]
        if kind=='tensor.extract':return self.load(args[0],self.index(args[1:],args[0].shape))
        if kind in ('math.sin','math.cos'):
            symbol='sinf' if kind=='math.sin' else 'cosf'
            self.declarations[symbol]=llvm.FuncOp(symbol,llvm.LLVMFunctionType([Float32Type()],Float32Type()))
            return b.add(llvm.CallOp(symbol,*args,return_type=Float32Type())).results[0]
        if kind=='arith.cmpf':
            comparison=b.add(llvm.FCmpOp(args[0],args[1],op.predicate)).results[0]
            return b.add(llvm.ZExtOp(comparison,i64)).results[0]
        if kind=='arith.select':
            cond=b.add(llvm.TruncOp(args[0],i1)).results[0]
            return b.add(llvm.SelectOp(cond,args[1],args[2])).results[0]
        if kind in binary:return binary[kind](*args)
        if kind in unary:return unary[kind](*args)
        if kind=='arith.constant':
            attr=op.properties.get('value',op.attributes.get('value'))
            v=attr.value.data
            return fp.fconst(v) if str(op.results[0].type) in ('f32','bf16') else self.const(v)
        if kind in ('arith.extsi','arith.extui','arith.index_cast'):return args[0]
        if kind=='arith.trunci':return b.wrap_int(args[0],int(str(op.results[0].type)[1:]))
        if kind=='arith.extf':return args[0]
        if kind=='arith.truncf' and str(op.results[0].type)=='bf16':return fp.round_bf16(args[0])
        raise ValueError('host scalar adapter unimplemented: '+kind)

    @staticmethod
    def overwritten_destinations(op):
        """Prove that each selected destination's old value is unobservable.

        A permutation of all parallel loop dimensions covers every destination
        element exactly once. Requiring the corresponding block argument to have
        no uses also excludes reductions and any arithmetic depending on the old
        destination. The tensor SSA adapter allocates distinct result storage.
        """
        ins,outs=li._ins_outs(op)
        iterators=li._iterator_types(op,ins)
        maps=[value.data for value in op.indexing_maps]
        block=op.regions[0].blocks[0]
        if not iterators or any(it!='parallel' for it in iterators):
            return [False]*len(outs)
        rank=len(iterators)
        result=[]
        for j,amap in enumerate(maps[len(ins):]):
            dims=[expr.position for expr in amap.results if isinstance(expr,AffineDimExpr)]
            complete=(len(amap.results)==rank and len(dims)==rank and sorted(dims)==list(range(rank)))
            unused=not any(block.args[len(ins)+j].uses)
            result.append(complete and unused)
        return result

    @staticmethod
    def pointwise_fusion_candidate(op):
        """Conservative pure f32 source domain for register-only intermediates."""
        if li._op_name(op)!='linalg.generic' or len(op.results)!=1:
            return False
        ins,outs=li._ins_outs(op)
        if len(outs)!=1 or str(op.results[0].type.element_type)!='f32':
            return False
        if HostTensorLowerer.overwritten_destinations(op)!=[True]:
            return False
        amap=list(op.indexing_maps)[-1].data
        if [getattr(expr,'position',None) for expr in amap.results]!=list(range(amap.num_dims)):
            return False
        safe={'arith.constant','arith.addf','arith.subf','arith.mulf','arith.divf',
              'arith.maximumf','arith.maxnumf','arith.minimumf','arith.minnumf',
              'arith.negf','arith.sitofp','arith.cmpf','arith.select',
              'math.exp','math.erf','math.rsqrt','math.sqrt','math.log','math.tanh',
              'math.absf','math.roundeven','linalg.yield'}
        for scalar in op.regions[0].blocks[0].ops:
            if li._op_name(scalar) not in safe or scalar.regions:
                return False
        return True

    def fused_generics(self, indexed_ops):
        """Emit a proven chain into one loop, retaining every scalar source DAG."""
        tail=indexed_ops[-1][1]
        dst=self.result(tail.results[0])
        shape=dst.shape
        def body(iv):
            coords=self.coords(iv,shape)
            self.active_coords=coords
            intermediates={}
            for source_index,op in indexed_ops:
                with self.e.task(0,source_op_index=source_index,region_id=li._prov(op).get('region_id')):
                    ins,_=li._ins_outs(op);block=op.regions[0].blocks[0];env={}
                    for arg,source,amap in zip(block.args,ins,[m.data for m in op.indexing_maps]):
                        if source in intermediates:
                            env[arg]=intermediates[source]
                        elif any(arg.uses):
                            tensor=self.values[source]
                            env[arg]=self.load(tensor,self.index([self.expr(x,coords) for x in amap.results],tensor.shape))
                    for scalar in block.ops:
                        if li._op_name(scalar)=='linalg.yield':
                            value=scalar.operands[0]
                            intermediates[op.results[0]]=env[value] if value in env else self.values[value]
                        else:
                            args=[env[x] if x in env else self.values[x] for x in scalar.operands]
                            env[scalar.results[0]]=self.scalar(scalar,args)
            self.store(intermediates[tail.results[0]],dst,iv)
        self.e.for_loop(0,dst.count,1,body)
        self.preparation_optimizations.append({
            'kind':'fuse_single_use_f32_pointwise_chain',
            'source_op_indices':[i for i,_ in indexed_ops],
            'source_region_ids':[li._prov(op).get('region_id') for _,op in indexed_ops],
            'shape':list(shape),'dtype':'f32',
            'proof':'pure parallel generics; identity output/internal maps; unique internal use; unused destination arguments',
            'elided_intermediate_store_bytes':(len(indexed_ops)-1)*dst.count*4,
            'elided_intermediate_reload_bytes':(len(indexed_ops)-1)*dst.count*4,
            'elided_intermediate_storage_bytes':(len(indexed_ops)-1)*dst.count*4,
            'loop_count_before':len(indexed_ops),'loop_count_after':1})

    def generic(self, op):
        ins,outs=li._ins_outs(op)
        dests=[self.result(v) for v in op.results]
        maps=[x.data for x in op.indexing_maps]
        overwritten = self.overwritten_destinations(op) if self.elide_overwritten_init else [False] * len(outs)
        for j,(initial,dst) in enumerate(zip(outs,dests)):
            if not overwritten[j]:
                self.copy(self.values[initial],dst)
            else:
                sizes={'i1':1,'i8':1,'i16':2,'i32':4,'i64':8,'f32':4,'bf16':2}
                self.preparation_optimizations.append({
                    'source_op_index':self.source_op_index,'result_index':j,
                    'region_id':li._prov(op).get('region_id'),
                    'kind':'elide_fully_overwritten_destination',
                    'proof':'parallel bijective output map and unused destination block argument',
                    'shape':list(dst.shape),'dtype':dst.dtype,
                    'elided_copy_elements':dst.count,
                    'elided_copy_bytes_read':dst.count*sizes[dst.dtype],
                    'elided_copy_bytes_written':dst.count*sizes[dst.dtype],
                    'elided_body_destination_loads':dst.count})
        tensors=[self.values[v] for v in ins]+dests
        rank=maps[0].num_dims
        bounds=[None]*rank
        for amap,tensor in zip(maps,tensors):
            for expr,size in zip(amap.results,tensor.shape):
                if isinstance(expr,AffineDimExpr):
                    old=bounds[expr.position]
                    if old is not None and old!=size:raise ValueError('inconsistent generic loop bound')
                    bounds[expr.position]=size
        if any(v is None for v in bounds):raise ValueError('cannot infer generic loop bounds')
        block=op.regions[0].blocks[0]
        def body(coords):
            self.active_coords=coords
            env={}
            for arg_index,(arg,tensor,amap) in enumerate(zip(block.args,tensors,maps)):
                if arg_index >= len(ins) and overwritten[arg_index-len(ins)]:
                    continue
                idx=self.index([self.expr(x,coords) for x in amap.results],tensor.shape)
                env[arg]=self.load(tensor,idx)
            for scalar in block.ops:
                if li._op_name(scalar)=='linalg.yield':
                    for value,dst,amap in zip(scalar.operands,dests,maps[len(ins):]):
                        self.store((env[value] if value in env else self.values[value]),dst,self.index([self.expr(x,coords) for x in amap.results],dst.shape))
                else:
                    args=[env[x] if x in env else self.values[x] for x in scalar.operands]
                    result=self.scalar(scalar,args)
                    if len(scalar.results)!=1:raise ValueError('multi-result scalar operation unsupported')
                    env[scalar.results[0]]=result
        def loop(axis,coords):
            if axis==rank:return body(coords)
            self.e.for_loop(0,bounds[axis],1,lambda i:loop(axis+1,coords+[i]))
        loop(0,[])

    def lower(self, op):
        kind=li._op_name(op)
        if kind=='arith.constant' and isinstance(op.results[0].type,TensorType):
            dst=self.result(op.results[0]);attr=op.properties.get('value',op.attributes.get('value'))
            values=list(attr.get_values())
            if len(values)==1:
                v=self.fp.fconst(values[0]) if dst.dtype in ('f32','bf16') else self.const(values[0])
                self.e.for_loop(0,dst.count,1,lambda i:self.store(v,dst,i))
            else:
                for i,value in enumerate(values):self.store(self.fp.fconst(value) if dst.dtype in ('f32','bf16') else self.const(value),dst,self.const(i))
            return
        if kind=='arith.constant' and not isinstance(op.results[0].type,TensorType):
            self.values[op.results[0]]=self.scalar(op,[]);return
        if kind=='tensor.empty':self.result(op.results[0]);return
        if kind=='tensor.splat':
            dst=self.result(op.results[0]); value=self.values[op.operands[0]]
            self.e.for_loop(0,dst.count,1,lambda i:self.store(value,dst,i));return
        if kind=='linalg.fill':
            dst=self.result(op.results[0]); value=self.values[op.operands[0]]
            self.e.for_loop(0,dst.count,1,lambda i:self.store(value,dst,i));return
        if kind in ('tensor.collapse_shape','tensor.expand_shape','tensor.cast'):
            src=self.values[op.operands[0]];typ=op.results[0].type
            if prod(typ.get_shape())!=src.count:raise ValueError('reshape extent mismatch')
            dst=TensorValue(src.ptr,tuple(typ.get_shape()),str(typ.element_type))
            if op.results[0] in self.bound_outputs:
                target=self.result(op.results[0]);self.copy(dst,target)
            else:self.values[op.results[0]]=dst
            return
        if kind=='linalg.transpose':
            src=self.values[op.operands[0]];dst=self.result(op.results[0]);perm=list(op.permutation.get_values())
            def body(i):
                outcoords=self.coords(i,dst.shape);coords=[None]*len(perm)
                for j,p in enumerate(perm):coords[p]=outcoords[j]
                self.store(self.load(src,self.index(coords,src.shape)),dst,i)
            self.e.for_loop(0,dst.count,1,body);return
        if kind=='tensor.extract_slice':
            src=self.values[op.operands[0]];dst=self.result(op.results[0]); offsets=list(op.static_offsets.get_values());strides=list(op.static_strides.get_values())
            if any(x<0 for x in offsets+strides):raise ValueError('dynamic slice unsupported')
            def body(i):
                coords=self.coords(i,dst.shape)
                if len(coords)!=len(offsets):raise ValueError('rank-reducing slice unsupported')
                coords=[self.b.add_i(self.const(o),self.b.mul_i(c,self.const(s))) for c,o,s in zip(coords,offsets,strides)]
                self.store(self.load(src,self.index(coords,src.shape)),dst,i)
            self.e.for_loop(0,dst.count,1,body);return
        if kind=='tensor.insert_slice':
            src=self.values[op.operands[0]];base=self.values[op.operands[1]];dst=self.result(op.results[0]);self.copy(base,dst)
            offsets=list(op.static_offsets.get_values());strides=list(op.static_strides.get_values())
            def body(i):
                coords=self.coords(i,src.shape)
                if len(coords)!=len(offsets):raise ValueError('rank-expanding insert slice unsupported')
                coords=[self.b.add_i(self.const(o),self.b.mul_i(c,self.const(s))) for c,o,s in zip(coords,offsets,strides)]
                self.store(self.load(src,i),dst,self.index(coords,dst.shape))
            self.e.for_loop(0,src.count,1,body);return
        if kind=='tensor.concat':
            dst=self.result(op.results[0]);axis=int(op.dim.value.data);offset=0
            for operand in op.operands:
                src=self.values[operand];base=offset
                def body(i):
                    coords=self.coords(i,src.shape);coords[axis]=self.b.add_i(coords[axis],self.const(base))
                    self.store(self.load(src,i),dst,self.index(coords,dst.shape))
                self.e.for_loop(0,src.count,1,body);offset+=src.shape[axis]
            return
        if kind in ('linalg.matmul','linalg.batch_matmul'):
            ins,outs=li._ins_outs(op);a,b=(self.values[v] for v in ins);dst=self.result(op.results[0]);self.copy(self.values[outs[0]],dst)
            if a.dtype not in ('f32','bf16') or b.dtype not in ('f32','bf16'):raise ValueError('integer named matmul requires device adapter')
            m,k,n=a.shape[-2],a.shape[-1],b.shape[-1]
            if a.shape[:-2]!=b.shape[:-2] or a.shape[-1]!=b.shape[-2]:raise ValueError('named host matmul shape mismatch')
            def batch_body(batch):
                def row_body(row):
                    def col_body(col):
                        di=self.b.add_i(self.b.mul_i(batch,self.const(m*n)),self.b.add_i(self.b.mul_i(row,self.const(n)),col))
                        def k_body(kk):
                            ai=self.b.add_i(self.b.mul_i(batch,self.const(m*k)),self.b.add_i(self.b.mul_i(row,self.const(k)),kk))
                            bi=self.b.add_i(self.b.mul_i(batch,self.const(k*n)),self.b.add_i(self.b.mul_i(kk,self.const(n)),col))
                            self.store(self.fp.fadd(self.load(dst,di),self.fp.fmul(self.load(a,ai),self.load(b,bi))),dst,di)
                        self.e.for_loop(0,k,1,k_body)
                    self.e.for_loop(0,n,1,col_body)
                self.e.for_loop(0,m,1,row_body)
            self.e.for_loop(0,prod(a.shape[:-2]),1,batch_body);return
        if kind=='linalg.generic':self.generic(op);return
        if kind=='linalg.reduce':
            src=self.values[op.operands[0]];dst=self.result(op.results[0]);self.copy(self.values[op.operands[1]],dst)
            axes=set(op.dimensions.get_values());block=op.regions[0].blocks[0]
            def body(i):
                coords=self.coords(i,src.shape);outcoords=[c for j,c in enumerate(coords) if j not in axes]
                di=self.index(outcoords,dst.shape);env={block.args[0]:self.load(src,i),block.args[1]:self.load(dst,di)}
                for scalar in block.ops:
                    if li._op_name(scalar)=='linalg.yield':self.store(env[scalar.operands[0]],dst,di)
                    else:env[scalar.results[0]]=self.scalar(scalar,[env[x] if x in env else self.values[x] for x in scalar.operands])
            self.e.for_loop(0,src.count,1,body);return
        if kind.startswith('quant_ext.quantize_per_tensor') or kind.startswith('quant_ext.dequantize_per_tensor'):
            src=self.values[op.operands[0]];scale=self.values[op.operands[1]];zero=self.values[op.operands[2]];dst=self.result(op.results[0])
            def body(i):
                sc=self.load(scale,self.const(0)) if isinstance(scale,TensorValue) else scale
                zp=self.load(zero,self.const(0)) if isinstance(zero,TensorValue) else zero
                x=self.load(src,i)
                if kind.startswith('quant_ext.quantize'):
                    x=self.fp.round_near_even(self.fp.fdiv(x,sc));x=self.fp.fadd(x,self.fp.sitofp(zp));x=self.fp.fclamp(x,-128,127);x=self.fp.fptosi(x)
                else:x=self.fp.fmul(self.fp.sitofp(self.b.sub_i(x,zp)),sc)
                self.store(x,dst,i)
            self.e.for_loop(0,src.count,1,body);return
        raise ValueError('host tensor adapter unimplemented: '+kind)
