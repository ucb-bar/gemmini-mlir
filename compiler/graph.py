"""Sequential original-SSA mixed graph lowering.

Floating arithmetic remains in its captured type and iteration order. Signed i8
contractions always enter the accelerator adapter; an unsupported contraction
raises instead of gaining a host implementation.
"""
from __future__ import annotations
import math
from xdsl.dialects import llvm
from xdsl.dialects.builtin import (TensorType, IntegerType, IndexType, FloatAttr,
    IntegerAttr, DenseIntOrFPElementsAttr, UnregisteredOp, i64)
from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr, AffineBinaryOpExpr
from merlin.targetgen.contract.linalg_iface import make_linalg_context
from xdsl.parser import Parser
from merlin.targetgen.oot_starterkit.plan import source_operation_inventory
from target import Builder


def _name(op):
    return getattr(getattr(op, 'op_name', None), 'data', getattr(op,'name','block_argument'))


def _attr(op, key):
    return op.properties.get(key, op.attributes.get(key))


def _array(op, key):
    value = _attr(op, key)
    return list(value.get_values()) if hasattr(value, 'get_values') else [x.value.data for x in value]


class GraphLowering:
    def __init__(self, source):
        self.source = source
        from source_parser import parse_source_module
        self.module = parse_source_module(source)
        self.module.verify()
        fs = [x for x in self.module.body.block.ops if x.name == 'func.func' and x.body.blocks]
        if len(fs) != 1 or len(fs[0].body.blocks) != 1:
            raise ValueError('mixed graph requires one static single-block source entry')
        self.fn = fs[0]
        self.ops = [op for op in self.fn.body.block.ops if op.name != 'func.return']
        self.ret = self.fn.body.block.last_op
        self.inventory = source_operation_inventory(source)
        self.names, self.scalars, self.tensors = {}, {}, {}
        self._dynamic_dims={}
        dynamic_storage=[]
        self.source_values = []
        def declare(value, name, role):
            typ = value.type
            if not isinstance(typ, TensorType):
                return
            shape = list(typ.get_shape())
            if any(x < 0 for x in shape):
                if role in ('input','output'):
                    raise ValueError('dynamic logical ABI tensor requires runtime extent bindings')
                from graph_control import bounded_shape
                shape=list(bounded_shape(value))
                dynamic_storage.append({'tensor':name,'declared_shape':list(typ.get_shape()),
                    'capacity_shape':shape,'extent_semantics':'source runtime SSA descriptor'})
                # Dynamic control values are private bounded allocations with
                # carried extents. They are not static logical ABI tensors.
                return
            self.names[value] = name
            self.tensors[name] = dict(shape=shape, dtype=str(typ.element_type), role=role)
        ins = _attr(self.fn, 'merlin.logical_inputs')
        outs = _attr(self.fn, 'merlin.logical_outputs')
        self.inputs = [x.data for x in ins] if ins is not None else [f'arg{i}' for i in range(len(self.fn.body.block.args))]
        self.outputs = [x.data for x in outs] if outs is not None else [f'output{i}' for i in range(len(self.ret.operands))]
        if len(self.inputs) != len(self.fn.body.block.args) or len(self.outputs) != len(self.ret.operands):
            raise ValueError('logical graph ABI arity mismatch')
        if len(set(self.inputs + self.outputs)) != len(self.inputs + self.outputs):
            raise ValueError('logical graph ABI requires distinct buffer names')
        returned = dict(zip(self.ret.operands, self.outputs))
        if len(returned) != len(self.outputs):
            raise ValueError('repeated source result aliases need an explicit output-copy ABI')
        for value, name in zip(self.fn.body.block.args, self.inputs):
            if value in returned:
                raise ValueError('source input/output aliases need an explicit alias ABI')
            declare(value, name, 'input')
        for i, op in enumerate(self.ops):
            for j, value in enumerate(op.results):
                declare(value, returned.get(value, f'value_{i}_{j}'), 'output' if value in returned else 'intermediate')
                if value in self.names:
                    self.source_values.append(dict(op_index=i, result_index=j, tensor=self.names[value]))
        self.cb = {'target':'gemmini', 'abi_version':'0.1', 'tensors':self.tensors,
                   'commands':[], 'params':{}}
        if dynamic_storage: self.cb['params']['dynamic_tensor_storage']=dynamic_storage
        self.cb['kernel_abi'] = {'kind':'whole_program', 'args':[{'tensor':name,'access':'read' if name in self.inputs else 'write' if name in self.outputs else 'readwrite'} for name in self.tensors], 'outputs':self.outputs}
        self.b = Builder(list(self.tensors))
        self.ptrs = {value:self.b.args[name] for value,name in self.names.items()}
        self.device_count = 0
        self.b.fence()
        self.b.blk.last_op.attributes['merlin.global_task']=IntegerAttr(-1,i64)
        self.b.rocc(7,0,0)
        self.b.blk.last_op.attributes['merlin.global_task']=IntegerAttr(-1,i64)

    def emit(self, name, args, types, properties=None):
        # A few LLVM conversion/intrinsic operations have no xDSL class in the
        # selected library; their registered stock LLVM spelling is retained.
        cls = UnregisteredOp.with_name(name)
        return self.b.add(cls.create(operands=args, result_types=types, properties=properties or {})).results[0]

    def scalar(self, op, operands):
        n = _name(op)
        typ = op.results[0].type if op.results else None
        if isinstance(typ, IndexType): typ = i64
        if n=='tensor.extract':
            return self.load(op.operands[0],operands[1:])
        if n == 'arith.constant':
            value = _attr(op,'value')
            if isinstance(value,IntegerAttr) and value.type!=typ:
                value=IntegerAttr(value.value.data,typ)
            return self.b.add(llvm.ConstantOp(value, typ)).results[0]
        binary = {'arith.addf':llvm.FAddOp,'arith.subf':llvm.FSubOp,'arith.mulf':llvm.FMulOp,'arith.divf':llvm.FDivOp,
                  'arith.addi':llvm.AddOp,'arith.subi':llvm.SubOp,'arith.muli':llvm.MulOp,'arith.divsi':llvm.SDivOp,
                  'arith.divui':llvm.UDivOp,'arith.remsi':llvm.SRemOp,'arith.remui':llvm.URemOp,
                  'arith.andi':llvm.AndOp,'arith.ori':llvm.OrOp,'arith.xori':llvm.XOrOp,
                  'arith.shli':llvm.ShlOp,'arith.shrsi':llvm.AShrOp,'arith.shrui':llvm.LShrOp}
        if n in binary: return self.b.add(binary[n](*operands)).results[0]
        if n == 'arith.select': return self.b.add(llvm.SelectOp(*operands)).results[0]
        if n == 'arith.cmpi': return self.b.add(llvm.ICmpOp(*operands, _attr(op,'predicate'))).results[0]
        if n == 'arith.cmpf': return self.b.add(llvm.FCmpOp(*operands, _attr(op,'predicate'))).results[0]
        casts = {'arith.extsi':llvm.SExtOp,'arith.extui':llvm.ZExtOp,'arith.trunci':llvm.TruncOp,
                 'arith.sitofp':llvm.SIToFPOp,'arith.extf':llvm.FPExtOp,'arith.truncf':llvm.FPTruncOp}
        if n in casts: return self.b.add(casts[n](operands[0],typ)).results[0]
        if n == 'arith.index_cast':
            src = operands[0]
            if src.type == typ: return src
            cls = llvm.SExtOp if src.type.width.data < typ.width.data else llvm.TruncOp
            return self.b.add(cls(src,typ)).results[0]
        if n in ('arith.fptosi','arith.fptoui','arith.uitofp','arith.bitcast'):
            return self.emit('llvm.'+n.split('.')[1],operands,[typ])
        if n == 'arith.negf': return self.b.add(llvm.FNegOp(operands[0])).results[0]
        intrinsics = {'math.sqrt':'sqrt','math.rsqrt':'sqrt','math.exp':'exp','math.exp2':'exp2','math.log':'log',
                      'math.log2':'log2','math.cos':'cos','math.sin':'sin','math.tanh':'tanh',
                      'math.absf':'fabs','math.floor':'floor','math.ceil':'ceil','math.roundeven':'roundeven',
                      'arith.maximumf':'maximum','arith.minimumf':'minimum','arith.maxnumf':'maxnum','arith.minnumf':'minnum',
                      'arith.maxsi':'smax','arith.minsi':'smin','arith.maxui':'umax','arith.minui':'umin'}
        if n in intrinsics:
            val = self.b.add(llvm.CallIntrinsicOp('llvm.'+intrinsics[n]+'.'+str(typ),operands,[typ])).results[0]
            if n == 'math.rsqrt':
                one = self.b.add(llvm.ConstantOp(FloatAttr(1.0,typ),typ)).results[0]
                val = self.b.add(llvm.FDivOp(one,val)).results[0]
            return val
        if n == 'math.powf': return self.b.add(llvm.CallIntrinsicOp('llvm.pow.'+str(typ),operands,[typ])).results[0]
        if n in ('func.call','llvm.call'):
            callee = _attr(op,'callee').root_reference.data
            # Closed math functions remain source-owned host calls.
            if callee not in ('roundf','roundevenf','sqrtf','expf','tanhf'):
                raise ValueError(f'unresolved source host callee {callee}')
            intrin = {'roundf':'round','roundevenf':'roundeven','sqrtf':'sqrt','expf':'exp','tanhf':'tanh'}[callee]
            return self.b.add(llvm.CallIntrinsicOp('llvm.'+intrin+'.'+str(typ),operands,[typ])).results[0]
        raise ValueError(f'unimplemented source scalar operation {n}')

    def flat(self, shape, indices):
        acc = self.b.const(0)
        for dim, ix in zip(shape,indices):
            extent=self.b.const(dim) if isinstance(dim,int) else dim
            acc = self.b.add_i(self.b.mul_i(acc,extent),ix)
        return acc

    def affine(self, expr, dims):
        if isinstance(expr,AffineDimExpr): return dims[expr.position]
        if isinstance(expr,AffineConstantExpr): return self.b.const(expr.value)
        if isinstance(expr,AffineBinaryOpExpr):
            a,b = self.affine(expr.lhs,dims),self.affine(expr.rhs,dims)
            kind = expr.kind.name.lower()
            fn = {'add':self.b.add_i,'mul':self.b.mul_i,'floordiv':self.b.sdiv_i,'mod':self.b.srem_i}.get(kind)
            if fn: return fn(a,b)
        raise ValueError(f'unsupported source affine expression {expr}')

    def loops(self, shape, body):
        def rec(i,indices):
            if i == len(shape): return body(indices)
            self.b.for_loop(0,shape[i],1,lambda iv:rec(i+1,[*indices,iv]),retain_loop=True)
        rec(0,[])

    def load(self, value, indices):
        typ = value.type
        shape=typ.get_shape()
        if any(x<0 for x in shape):
            from graph_control import dimensions
            shape=dimensions(self,value)
        return self.b.add(llvm.LoadOp(self.b.gep(self.ptrs[value], self.flat(shape,indices),typ.element_type),typ.element_type)).results[0]

    def store(self, value, indices, scalar):
        typ = value.type
        shape=typ.get_shape()
        if any(x<0 for x in shape):
            from graph_control import dimensions
            shape=dimensions(self,value)
        self.b.add(llvm.StoreOp(scalar,self.b.gep(self.ptrs[value],self.flat(shape,indices),typ.element_type)))

    def copy(self, src, dst):
        st,dt = src.type,dst.type
        if math.prod(st.get_shape()) != math.prod(dt.get_shape()) or st.element_type != dt.element_type:
            raise ValueError('source view changes tensor elements')
        ty = st.element_type
        def body(i):
            x=self.b.add(llvm.LoadOp(self.b.gep(self.ptrs[src],i,ty),ty)).results[0]
            self.b.add(llvm.StoreOp(x,self.b.gep(self.ptrs[dst],i,ty)))
        self.b.for_loop(0,math.prod(st.get_shape()),1,body,retain_loop=True)

    def dense(self,op):
        from dense import lower_matmul
        from merlin.transforms.dense_ir import recognize_signed_integer_matmul
        from merlin.compile.scheduling.dense_source import read_zero_seeded_integer_matmul
        if op.name=='linalg.matmul':
            if len(op.inputs)==2 and all(str(v.type.element_type).startswith('f') for v in (*op.inputs,*op.outputs)):
                lhs,rhs=op.inputs;out=op.results[0]
                m,k=lhs.type.get_shape();rk,n=rhs.type.get_shape()
                if k!=rk: raise ValueError('floating matmul shape mismatch')
                self.copy(op.outputs[0],out)
                def body(ix):
                    i,j,q=ix
                    a=self.load(lhs,[i,q]);bv=self.load(rhs,[q,j]);acc=self.load(out,[i,j])
                    product=self.b.add(llvm.FMulOp(a,bv)).results[0]
                    self.store(out,[i,j],self.b.add(llvm.FAddOp(acc,product)).results[0])
                self.loops((m,n,k),body)
                return
            if len(op.inputs)!=2 or len(op.outputs)!=1 or len(op.results)!=1 or [str(v.type.element_type) for v in (*op.inputs,*op.outputs)]!=['i8','i8','i32']:
                raise ValueError('unsupported named contraction numerical contract')
            lhs,rhs=op.inputs;seed=op.outputs[0]
        else:
            clone=op.clone()
            for inner in clone.walk():
                for key in list(inner.attributes):
                    if key.startswith('prov.'): del inner.attributes[key]
            lhs,rhs,seed=recognize_signed_integer_matmul(clone)
        m,k=lhs.type.get_shape();rk,n=rhs.type.get_shape()
        if k!=rk or tuple(seed.type.get_shape())!=(m,n): raise ValueError('source contraction shape mismatch')
        seed_owner=op.outputs[0].owner
        zero=False
        if _name(seed_owner) in ('tensor.splat','linalg.fill'):
            constant=seed_owner.operands[0].owner
            if _name(constant)=='arith.constant':
                attr=_attr(constant,'value')
                zero=isinstance(attr,IntegerAttr) and attr.value.data==0 and attr.type==seed.type.element_type
        if not zero and _name(seed_owner)=='tensor.empty':
            raise ValueError('contraction reads an uninitialized source accumulator')
        out=op.results[0]
        lower_matmul(self.b,*(self.b.ptr(self.ptrs[v]) for v in (*op.inputs,out)),m,k,n)
        if not zero:
            import target
            from elementwise import _tiles,_address
            b=self.b
            target.cfg_ld(b,n*4,stream=0,shrunk=False)
            target.cfg_ld(b,n*4,stream=1,shrunk=False)
            target.cfg_st(b,n*4)
            def add(row,col,nr,nc):
                target.mvin(b,_address(b,self.ptrs[out],row,col,n,4),target.ACC,nr,nc)
                b.fence()
                target.mvin(b,_address(b,self.ptrs[op.outputs[0]],row,col,n,4),target.ACC|target.ACCUMULATE,nr,nc,stream=1)
                b.fence()
                target.mvout(b,_address(b,self.ptrs[out],row,col,n,4),target.ACC|target.FULL,nr,nc)
                b.fence()
            _tiles(b,m,n,add)
        self.device_count+=1

    def generic(self, op):
        ins,outs = tuple(op.inputs),tuple(op.outputs)
        from graph_products import lower_graph_product
        if lower_graph_product(self,op): return
        from vector import lower_generic
        if lower_generic(self.b,op,self.ptrs):
            self.device_count+=1
            self._last_device=True
            return
        maps = [a.data for a in op.indexing_maps]
        rank = maps[0].num_dims
        extents = [None]*rank
        for val, amap in zip((*ins,*outs),maps):
            for expr,extent in zip(amap.results,val.type.get_shape()):
                if isinstance(expr,AffineDimExpr):
                    old=extents[expr.position]
                    if old is not None and old != extent: raise ValueError('inconsistent linalg iteration domain')
                    extents[expr.position]=extent
        if any(v is None for v in extents): raise ValueError('unresolved linalg loop extent')
        reductions=[i for i,it in enumerate(op.iterator_types) if 'reduction' in str(it)]
        if reductions and len(ins)==2 and str(ins[0].type.element_type)==str(ins[1].type.element_type)=='i8':
            self.dense(op)
            return
        if reductions and any(str(v.type.element_type)=='i8' for v in ins):
            raise ValueError('eligible integer reduction lacks target lowering')
        for init,result in zip(outs,op.results):
            # Parallel destinations are never read when their block argument is
            # unused. Reduction recurrence starts from the exact source seed.
            if reductions: self.copy(init,result)
        block=op.body.block
        def body(indices):
            local={}
            for arg,value,amap in zip(block.args,(*ins,*op.results),maps):
                if arg.uses:
                    local[arg]=self.load(value,[self.affine(e,indices) for e in amap.results])
            for scalar_op in block.ops:
                if scalar_op.name=='linalg.yield':
                    for result,amap,y in zip(op.results,maps[len(ins):],scalar_op.operands):
                        self.store(result,[self.affine(e,indices) for e in amap.results],local[y])
                elif scalar_op.name=='linalg.index':
                    local[scalar_op.results[0]]=indices[_attr(scalar_op,'dim').value.data]
                else:
                    local[scalar_op.results[0]]=self.scalar(scalar_op,[local.get(v,self.scalars.get(v)) for v in scalar_op.operands])
        self.loops(extents,body)

    def operation(self, op):
        n=_name(op)
        from graph_control import lower_control
        if lower_control(self,op): return
        from graph_device import lower_device
        if lower_device(self,op): return
        if n=='tensor.empty': return
        if n=='arith.constant':
            value=_attr(op,'value')
            if isinstance(op.results[0].type,TensorType):
                vals=list(value.get_values())
                # Literal payloads are original source constants. A loop per
                # distinct value avoids expanding large splats into IR.
                dst=op.results[0];ty=dst.type.element_type
                if len(set(vals))==1:
                    attr=FloatAttr(vals[0],ty) if str(ty).startswith('f') else IntegerAttr(vals[0],ty)
                    scalar=self.b.add(llvm.ConstantOp(attr,ty)).results[0]
                    self.loops(dst.type.get_shape(),lambda ix:self.store(dst,ix,scalar))
                else:
                    for i,v in enumerate(vals):
                        attr=FloatAttr(v,ty) if str(ty).startswith('f') else IntegerAttr(v,ty)
                        scalar=self.b.add(llvm.ConstantOp(attr,ty)).results[0]
                        self.b.add(llvm.StoreOp(scalar,self.b.gep(self.ptrs[dst],self.b.const(i),ty)))
            else: self.scalars[op.results[0]]=self.scalar(op,[])
            return
        if (n.startswith(('arith.','math.')) or n=='func.call') and len(op.results)==1 and not isinstance(op.results[0].type,TensorType):
            self.scalars[op.results[0]]=self.scalar(op,[self.scalars[v] for v in op.operands])
            return
        if n=='tensor.splat':
            self.loops(op.results[0].type.get_shape(),lambda ix:self.store(op.results[0],ix,self.scalars[op.operands[0]]));return
        if n in ('tensor.collapse_shape','tensor.expand_shape','tensor.cast'):
            self.copy(op.operands[0],op.results[0]);return
        if n=='linalg.fill':
            scalar=self.scalars[op.operands[0]]
            self.loops(op.results[0].type.get_shape(),lambda ix:self.store(op.results[0],ix,scalar));return
        if n=='tensor.concat':
            dim=_attr(op,'dim').value.data;dst=op.results[0];offset=0
            for src in op.operands:
                start=offset
                def body(ix):
                    out=list(ix);out[dim]=self.b.add_i(out[dim],self.b.const(start))
                    self.store(dst,out,self.load(src,ix))
                self.loops(src.type.get_shape(),body)
                offset+=src.type.get_shape()[dim]
            return
        if n=='tensor.extract':
            self.scalars[op.results[0]]=self.load(op.operands[0],[self.scalars[v] for v in op.operands[1:]])
            return
        if n=='tensor.from_elements':
            dst=op.results[0];ty=dst.type.element_type
            for i,value in enumerate(op.operands):
                self.b.add(llvm.StoreOp(self.scalars[value],self.b.gep(self.ptrs[dst],self.b.const(i),ty)))
            return
        if n=='linalg.transpose':
            perm=_array(op,'permutation');src=op.inputs[0];dst=op.results[0]
            def body(ix):
                original=[None]*len(perm)
                for i,d in enumerate(perm): original[d]=ix[i]
                self.store(dst,ix,self.load(src,original))
            self.loops(dst.type.get_shape(),body);return
        if n=='linalg.generic': self.generic(op);return
        if n=='linalg.matmul':
            self.dense(op);return
        if n.startswith('linalg.'):
            from graph_products import lower_graph_product
            if lower_graph_product(self,op): return
        if n=='linalg.reduce':
            # Keep the declared source reduction dimension order; no tree or
            # reassociation is introduced.
            inp,seed=op.operands;dst=op.results[0]
            self.copy(seed,dst)
            reduced=_array(op,'dimensions');parallel=[i for i in range(len(inp.type.get_shape())) if i not in reduced]
            block=op.regions[0].block
            def body(ix):
                local={block.args[0]:self.load(inp,ix),block.args[1]:self.load(dst,[ix[d] for d in parallel])}
                for scalar_op in block.ops:
                    if scalar_op.name=='linalg.yield': self.store(dst,[ix[d] for d in parallel],local[scalar_op.operands[0]])
                    else: local[scalar_op.results[0]]=self.scalar(scalar_op,[local[v] for v in scalar_op.operands])
            self.loops(inp.type.get_shape(),body);return
        if n in ('tensor.extract_slice','tensor.insert_slice'):
            insert=n=='tensor.insert_slice'
            if insert: self.copy(op.operands[1],op.results[0])
            offsets=_array(op,'static_offsets');strides=_array(op,'static_strides');sizes=_array(op,'static_sizes')
            if any(x<0 for x in offsets+sizes+strides): raise ValueError('dynamic tensor slice not implemented')
            src=op.operands[0];dst=op.results[0]
            slice_type=src.type if insert else dst.type
            # Source static slices may drop unit axes.
            kept=[];j=0
            for i,size in enumerate(sizes):
                if j<len(slice_type.get_shape()) and size==slice_type.get_shape()[j]: kept.append(i);j+=1
                elif size!=1: raise ValueError('unresolved rank-reducing source slice')
            def body(ix):
                expanded=[self.b.const(x) for x in offsets]
                for dim,v in zip(kept,ix): expanded[dim]=self.b.add_i(expanded[dim],self.b.mul_i(v,self.b.const(strides[dim])))
                self.store(dst,expanded if insert else ix,self.load(src,ix if insert else expanded))
            self.loops(slice_type.get_shape(),body);return
        from graph_quant import lower_quant
        if lower_quant(self,op): return
        raise ValueError(f'unimplemented mixed graph source operation {n}')

    def lower(self):
        def lane(op):
            from graph_device import can_lower_device
            if can_lower_device(op): return 'accelerator'
            ins=tuple(getattr(op,'inputs',()))
            its=tuple(getattr(op,'iterator_types',()))
            contraction=(op.name in ('linalg.matmul','linalg.batch_matmul','linalg.matmul_transpose_a','linalg.matmul_transpose_b','linalg.batch_matmul_transpose_a','linalg.batch_matmul_transpose_b') or op.name=='linalg.generic' and any('reduction' in str(x) for x in its)) and len(ins)==2 and all(str(x.type.element_type)=='i8' for x in ins)
            if contraction: return 'accelerator'
            if op.name=='linalg.generic' and not any('reduction' in str(x) for x in its):
                from vector import lower_generic
                probe=Builder(list(self.tensors))
                if lower_generic(probe,op,{v:probe.args[n] for v,n in self.names.items()}):
                    return 'accelerator'
            return 'host'
        groups=[]
        def administrative(op):
            return op.name=='tensor.empty' or op.name=='arith.constant' and all(not isinstance(v.type,TensorType) for v in op.results)
        for index,op in enumerate(self.ops):
            if administrative(op):
                # Empty destinations and scalar literals do not execute a
                # region. Attribute them to adjacent real work rather than
                # introducing a task with no executable body.
                placement=groups[-1][0] if groups else next((lane(x) for x in self.ops[index+1:] if not administrative(x)),'host')
            else: placement=lane(op)
            if not groups or groups[-1][0]!=placement: groups.append((placement,[]))
            groups[-1][1].append((index,op))
        tasks=[]
        for task_index,(placement,members) in enumerate(groups):
            before={op for block in self.b.region.blocks for op in block.ops}
            if placement=='accelerator':
                self.b.rocc(7,0,0)
                self.b.fence()
            for index,op in members: self.operation(op)
            after=[op for block in self.b.region.blocks for op in block.ops if op not in before]
            for op in after: op.attributes['merlin.global_task']=IntegerAttr(task_index,i64)
            indices=[index for index,op in members]
            rows=[self.inventory['operations'][i] for i in indices]
            from merlin.xdsl_dialects.lowering.outline import _free_values
            # Nested regions can capture tensors absent from top-level operands.
            # Share the exact free-value analysis used by source-region extraction.
            reads=sorted({self.names[v] for v in _free_values(tuple(op for i,op in members)) if v in self.names})
            writes=[self.names[v] for i,op in members for v in op.results if v in self.names]
            command={'opcode':'VECTOR_MAP',
                     'attributes':{'source_op_indices':indices,'placement':placement}}
            if placement=='accelerator' and len(members)==1 and len(getattr(members[0][1],'inputs',()))==2:
                root=members[0][1]
                if len(root.inputs)==2 and any('reduction' in str(x) for x in getattr(root,'iterator_types',())): command['opcode']='MATMUL'
                command['operands']={'lhs':self.names[root.inputs[0]],'rhs':self.names[root.inputs[1]],'dst':self.names[root.results[0]]}
            self.cb['commands'].append(command)
            tasks.append({'task_index':task_index,'kind':placement,'source_op_indices':indices,
                'source_region_ids':sorted({r['region_id'] for r in rows if r['region_id']}),
                'instruction_start':task_index,'instruction_end':task_index+1,'reads':reads,'writes':writes})
        self.b.fence()
        self.b.blk.last_op.attributes['merlin.global_task']=IntegerAttr(-2,i64)
        count=len(tasks)
        plan={'schema':'mixed_program_plan_v1','source_sha256':self.inventory['source_sha256'],
            'source_op_count':len(self.ops),'schedule_instruction_count':count,
            'prologue_instruction_range':[0,0],'epilogue_instruction_range':[count,count],
            'entry_bindings':self.inputs,'output_bindings':self.outputs,'source_values':self.source_values,'tasks':tasks}
        self.cb['params'].update(global_program_plan=plan,source_format='linalg-on-tensors',
            input_order=self.inputs,output_order=self.outputs,accelerator_contractions=self.device_count)
        self.b.module_text()
        for op in self.b.entry.ops:
            if op.name=='llvm.mlir.constant': op.attributes['merlin.global_task']=IntegerAttr(-1,i64)
        self.b.blk.last_op.attributes['merlin.global_task']=IntegerAttr(-2,i64)
        import io
        from xdsl.printer import Printer
        stream=io.StringIO()
        Printer(stream=stream,print_generic_format=True).print_op(self.b._module)
        return self.cb,stream.getvalue()+'\n'


def lower_graph(source_text):
    return GraphLowering(source_text).lower()
