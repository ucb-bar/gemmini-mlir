"""Counted host tensor storage around the shared exact scalar source lowering.

This adapter changes storage and iteration construction, not scalar recipes or
placement. The mixed graph owner must remove every admitted integer product
before invoking it. Fresh private tensors keep source SSA immutability; retained
row-major loops keep the original reduction order and bound artifact growth.
"""
from __future__ import annotations

from math import prod

from xdsl.dialects import builtin, llvm
from xdsl.ir import Block
from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr
from merlin.llvmlower.host_builder import PTR
from merlin.llvmlower.host_tensor_codegen import (
    HostLinalg, TensorVal, LoweringDeclined, attr_of, elem_name, tensor_shape, is_int,
)
from gemmini_support import scratch_buffer


class _Elements:
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return prod(self.tensor.shape)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        if isinstance(index, int) and (index < 0 or index >= len(self)):
            raise IndexError(index)
        return self.tensor.flat(index)


class StoredTensor(TensorVal):
    """A source tensor with a dynamic scalar reader, in the shared scalar domain."""
    __slots__ = ("host", "reader", "pointer", "capacity", "coordinate_reader", "traversal", "copy_traversal")

    def __init__(self, host, shape, ety, reader, pointer=None, capacity=None,
                 coordinate_reader=None, traversal=None, copy_traversal=None):
        super().__init__(tuple(shape), [], ety)
        self.host, self.reader, self.pointer = host, reader, pointer
        self.capacity = tuple(shape) if capacity is None else tuple(capacity)
        self.coordinate_reader = coordinate_reader
        self.traversal = traversal
        self.copy_traversal = copy_traversal
        self.elems = _Elements(self)

    def flat(self, index):
        return self.reader(self.host._v(index))

    def at(self, index):
        if len(index) != len(self.shape):
            raise LoweringDeclined("host tensor access rank mismatch", op="host_lane")
        if self.coordinate_reader is not None:
            return self.coordinate_reader(tuple(index))
        return self.flat(self.host._flat(index, self.shape))


class CompactHostLinalg(HostLinalg):
    """HostLinalg-compatible counted tensor adapter, sharing scalar semantics.

    ``run_segment`` and ``vals`` keep the original source SSA keys. The graph
    owner may add task/source attributes to all emitted operations as before.
    Module assembly must include ``fp.fb._gemmini_globals``.
    """
    def run_segment(self, ops, inputs, outputs):
        from pathlib import Path
        from merlin.compile.tuning_knobs import knob
        enabled = getattr(self, 'host_pointwise_fusion', None)
        if enabled is None:
            enabled = knob('host_pointwise_fusion', default='off', values=('off', 'on'),
                           root=Path(__file__).resolve().parent) == 'on'
        if type(enabled) is not bool:
            raise ValueError('host_pointwise_fusion requires a Boolean override')
        if not enabled:
            return super().run_segment(ops, inputs, outputs)
        from gemmini_host_fusion import prepare_segment
        executable, external, results, report = prepare_segment(ops, outputs)
        if not hasattr(self, 'host_fusion_reports'):
            self.host_fusion_reports = []
        self.host_fusion_reports.append(report)
        if not report['rewrites']:
            return super().run_segment(ops, inputs, outputs)
        for value, ptr in inputs:
            if value not in self.vals:
                self.vals[value] = self.load_arg(ptr, value.type)
        old_keys = set(self.vals)
        for original, argument in external:
            self.vals[argument] = self.get(original)
        for operation in executable:
            self._tensor_op(operation)
        published = [(original, self.get(cloned)) for original, cloned in results]
        # Retain source values between segments, never the private clone's keys.
        for key in set(self.vals) - old_keys:
            del self.vals[key]
        self.vals.update(published)
        for value, ptr in outputs:
            self.store_result(ptr, self.get(value))

    @property
    def fb(self):
        return self.fp.fb

    def _v(self, value):
        return self.fb.const(value) if isinstance(value, int) else value

    def _add(self, a, b):
        return self.fb.add_i(self._v(a), self._v(b))

    def _mul(self, a, b):
        return self.fb.mul_i(self._v(a), self._v(b))

    def _flat(self, indices, shape):
        if len(indices) != len(shape):
            raise LoweringDeclined("host tensor access rank mismatch", op="host_lane")
        flat = self._v(0)
        for index, extent in zip(indices, shape):
            flat = self._add(self._mul(flat, extent), index)
        return flat

    def _coordinates(self, flat, shape):
        coordinates = []
        stride = prod(shape)
        for extent in shape:
            stride //= extent
            axis = self.fb.udiv_i(self._v(flat), self._v(stride)) if stride != 1 else self._v(flat)
            coordinates.append(self.fb.urem_i(axis, self._v(extent)))
        return tuple(coordinates)

    @staticmethod
    def _merged_axes(source_shape, target_shape):
        """Prove a row-major reshape expressible by merging source axes only.

        Unit axes need no coordinate. Splitting a nonunit source axis would
        require division, so leave that case to the ordinary indexed reader.
        """
        if any(type(n) is not int or n <= 0 for n in (*source_shape,*target_shape)):
            return None
        axes = [i for i,n in enumerate(source_shape) if n != 1]
        cursor = 0
        groups = []
        for extent in target_shape:
            group = []
            size = 1
            while size < extent and cursor < len(axes):
                axis = axes[cursor]
                cursor += 1
                group.append(axis)
                size *= source_shape[axis]
            if size != extent:
                return None
            groups.append(tuple(group))
        return tuple(groups) if cursor == len(axes) else None

    def _merged_coordinates(self, indices, shape, groups):
        return tuple(self._flat(tuple(indices[a] for a in group),
                                tuple(shape[a] for a in group)) for group in groups)

    def _loops(self, shape, body):
        if any(type(n) is not int or n <= 0 for n in shape):
            raise LoweringDeclined("host loops require positive static extents", op="host_lane")
        def axis(dim, indices):
            if dim == len(shape):
                body(tuple(indices))
            else:
                self.fb.for_loop(0, shape[dim], 1,
                    lambda iv: axis(dim+1, [*indices, iv]), retain_loop=True)
        axis(0, [])

    def _read(self, ptr, index, ety):
        if ety == "i1":
            value=self.fb.add(llvm.LoadOp(self.fb.gep(ptr,self._v(index),builtin.i1),builtin.i1)).results[0]
            return self.fb.add(llvm.SExtOp(value,builtin.i64)).results[0]
        return self.fb.load_i64(ptr, self._v(index), ety) if is_int(ety) else self.fp.load(ptr, self._v(index), ety)

    def _write(self, ptr, index, value, ety):
        if ety == "i1":
            bit=self.fb.add(llvm.TruncOp(value,builtin.i1)).results[0]
            self.fb.add(llvm.StoreOp(bit,self.fb.gep(ptr,self._v(index),builtin.i1)))
        elif is_int(ety):
            self.fb.store_i64(value, ptr, self._v(index), ety)
        else:
            self.fp.store(value, ptr, self._v(index), ety)

    def _stored(self, shape, ety, pointer, capacity=None):
        return StoredTensor(self, shape, ety, lambda index: self._read(pointer, index, ety), pointer, capacity)

    def _new(self, shape, ety, capacity=None):
        width = {"i1":1,"i8":1,"i16":2,"i32":4,"i64":8,"f32":4,"bf16":2}.get(ety)
        if width is None:
            raise LoweringDeclined(f"compact host storage has no format {ety}", op="host_lane")
        capacity = tuple(shape) if capacity is None else tuple(capacity)
        if any(type(n) is not int or n < 0 for n in capacity):
            raise LoweringDeclined("host tensor allocation lacks a proven finite capacity", op="host_lane")
        return self._stored(shape, ety, scratch_buffer(self.fb, max(1,prod(capacity)*width), name="host_tensor"),capacity)

    def _count(self, shape):
        result = self._v(1)
        for extent in shape:
            result = self._mul(result,extent)
        return result

    def _runtime_loop(self, start, stop, step, body):
        header, work, done = Block(arg_types=[builtin.i64]), Block(), Block()
        for block in (header,work,done): self.fb.region.add_block(block)
        self.fb.add(llvm.BrOp(header,self._v(start)))
        self.fb.blk=header
        condition=self.fb.add(llvm.ICmpOp(header.args[0],self._v(stop),builtin.IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(),builtin.i64))).results[0]
        self.fb.add(llvm.CondBrOp(condition,work,[],done,[]));self.fb.blk=work
        body(header.args[0])
        latch=self.fb.add(llvm.BrOp(header,self._add(header.args[0],step)))
        from merlin.llvmlower.llvm_loop_metadata import disable_loop_unroll
        disable_loop_unroll(latch);self.fb.blk=done

    def _copy(self, source, destination):
        if source.shape != destination.shape or source.ety != destination.ety:
            raise LoweringDeclined("host tensor copy type mismatch", op="host_lane")
        if all(type(n) is int for n in source.shape):
            if prod(source.shape) == 0:
                return
            if isinstance(source,StoredTensor) and source.pointer is None:
                if source.copy_traversal is not None:
                    shape,coordinates,read = source.copy_traversal
                else:
                    shape,read = source.traversal or (source.shape,source.at)
                    coordinates = lambda indices:indices
                # A reshape retains the same row-major element sequence. Walk
                # its underlying coordinate domain rather than flattening and
                # recovering every axis with div/rem for each copied element.
                self._loops(shape,lambda indices:self._write(destination.pointer,
                    self._flat(coordinates(indices),source.shape if source.copy_traversal is not None else shape),
                    read(indices),destination.ety))
            else:
                self.fb.for_loop(0, prod(source.shape), 1,
                    lambda iv: self._write(destination.pointer, iv, source.elems[iv], destination.ety), retain_loop=True)
        else:
            self._runtime_loop(0,self._count(source.shape),1,
                lambda iv:self._write(destination.pointer,iv,source.elems[iv],destination.ety))

    def load_arg(self, ptr, ty):
        if self.row_alignment != 1:
            raise LoweringDeclined("compact source ABI requires contiguous tensors", op="host_lane")
        return self._stored(tensor_shape(ty), elem_name(ty.element_type), ptr)

    def store_result(self, ptr, value):
        if isinstance(value, StoredTensor) and value.pointer is ptr:
            return
        self._copy(value, self._stored(value.shape, value.ety, ptr))

    def _affine(self, expr, ivs):
        if isinstance(expr, AffineDimExpr):
            return self._v(ivs[expr.position])
        if isinstance(expr, AffineConstantExpr):
            return self._v(int(expr.value))
        if isinstance(expr, AffineBinaryOpExpr):
            lhs, rhs = self._affine(expr.lhs, ivs), self._affine(expr.rhs, ivs)
            if expr.kind is AffineBinaryOpKind.Add:
                return self._add(lhs, rhs)
            if expr.kind is AffineBinaryOpKind.Mul:
                return self._mul(lhs, rhs)
            if not isinstance(expr.rhs, AffineConstantExpr) or expr.rhs.value <= 0:
                raise LoweringDeclined("affine quotient requires a positive constant divisor", op="host_lane")
            def floor(value):
                quotient = self.fb.sdiv_i(value, rhs)
                remainder = self.fb.srem_i(value, rhs)
                correction = self.fb.and_i(self.fb.ashr_i(remainder, 63), self._v(1))
                return self.fb.sub_i(quotient, correction)
            if expr.kind is AffineBinaryOpKind.FloorDiv:
                return floor(lhs)
            if expr.kind is AffineBinaryOpKind.CeilDiv:
                return self.fb.sub_i(self._v(0), floor(self.fb.sub_i(self._v(0), lhs)))
            if expr.kind is AffineBinaryOpKind.Mod:
                return self.fb.sub_i(lhs, self._mul(floor(lhs), rhs))
        raise LoweringDeclined(f"unsupported counted affine expression {expr}", op="host_lane")

    def _scalar_op(self, op, ins):
        if (op.name in ('arith.maximumf','arith.minimumf','arith.maxnumf','arith.minnumf')
                and hasattr(self.fp,'source_extremum')):
            return self._round(self.fp.source_extremum(op.name,*ins),
                               elem_name(op.results[0].type))
        if op.name in ("math.sin","math.cos"):
            # Preserve the source operation's native math semantics.  Runtime
            # recipes (including portable_trig) require the original platform
            # selection; their availability alone does not select a policy.
            operation = self.fp.sin if op.name == "math.sin" else self.fp.cos
            value = operation(ins[0])
            return self._round(value,elem_name(op.results[0].type))
        if op.name == "tensor.extract":
            source=ins[0]
            if not isinstance(source,TensorVal):
                raise LoweringDeclined("tensor.extract requires a tensor source",op=op.name)
            return source.at(tuple(ins[1:]))
        if op.name == "linalg.index":
            dim = attr_of(op, "dim")
            return self._v(self.ivs[int(dim.value.data)])
        if op.name == "math.roundeven":
            return self.fp.round_near_even(ins[0])
        if op.name in ("arith.cmpf", "arith.cmpi"):
            predicate = int(attr_of(op, "predicate").value.data)
            if op.name == "arith.cmpf":
                operation = llvm.FCmpOp(ins[0], ins[1], builtin.IntegerAttr(predicate,builtin.i64))
            else:
                operation = llvm.ICmpOp(ins[0],ins[1],builtin.IntegerAttr(predicate,builtin.i64))
            bit=self.fb.add(operation).results[0]
            widened=self.fb.add(llvm.ZExtOp(bit,builtin.i64)).results[0]
            return self.fb.wrap_int(widened,1)
        if op.name == "arith.select":
            bit=self.fb.add(llvm.TruncOp(ins[0],builtin.i1)).results[0]
            value=self.fb.add(llvm.SelectOp(bit,ins[1],ins[2])).results[0]
            return self._round(value,elem_name(op.results[0].type))
        return super()._scalar_op(op, ins)

    def _body(self, block, arguments):
        env = dict(zip(block.args, arguments, strict=True))
        def read(value):
            return env[value] if value in env else self.get(value)
        for operation in block.ops:
            if operation.name == "linalg.yield":
                return [read(value) for value in operation.operands]
            if len(operation.results) != 1:
                raise LoweringDeclined("scalar body op must have one result", op=operation.name)
            env[operation.results[0]] = self._scalar_op(operation, [read(v) for v in operation.operands])
        raise LoweringDeclined("host scalar body lacks linalg.yield", op="host_lane")

    def _tensor_op(self, op):
        if op.name in ('tensor.extract_slice', 'linalg.transpose'):
            from pathlib import Path
            from merlin.compile.tuning_knobs import knob
            enabled = getattr(self, 'host_tensor_views', None)
            if enabled is None:
                enabled = knob('host_tensor_views', default='off', values=('off', 'on'),
                               root=Path(__file__).resolve().parent) == 'on'
            if type(enabled) is not bool:
                raise ValueError('host_tensor_views requires a Boolean override')
            if enabled:
                from gemmini_preparation_identity import try_apply
                if try_apply(self, op):
                    return
        # Scalar source producers also occur between tensor regions (shape
        # arithmetic and values extracted from rank-zero tensors).
        if (op.name != "arith.constant" and op.name.startswith(("arith.", "math."))
                and len(op.results) == 1
                and not isinstance(op.results[0].type, builtin.TensorType)):
            self.vals[op.results[0]] = self._scalar_op(op, [self.get(v) for v in op.operands])
            return
        super()._tensor_op(op)

    def _t_arith_constant(self, op):
        typ, attr = op.results[0].type, attr_of(op, "value")
        if not isinstance(typ, builtin.TensorType):
            return super()._t_arith_constant(op)
        if not isinstance(attr, builtin.DenseIntOrFPElementsAttr):
            raise LoweringDeclined("tensor constant requires dense elements", op=op.name)
        shape, ety = tensor_shape(typ), elem_name(typ.element_type)
        globals_ = getattr(self.fb, "_gemmini_globals", None)
        if globals_ is None:
            globals_ = self.fb._gemmini_globals = []
        symbol = f"gemmini_host_constant_{len(globals_)}"
        flattened = builtin.DenseIntOrFPElementsAttr.from_list(
            builtin.TensorType(typ.element_type, (prod(shape),)), list(attr.get_values()))
        glob = llvm.GlobalOp(llvm.LLVMArrayType(prod(shape), typ.element_type), symbol,
            "internal", constant=True, value=flattened, alignment=64)
        globals_.append(glob)
        pointer = self.fb.add(llvm.AddressOfOp(symbol, PTR)).results[0]
        self.vals[op.results[0]] = self._stored(shape, ety, pointer)

    def _splat(self, op, scalar):
        typ = op.results[0].type
        self.vals[op.results[0]] = StoredTensor(self, tensor_shape(typ), elem_name(typ.element_type), lambda _: scalar)

    def _t_tensor_empty(self, op):
        declared=op.results[0].type.get_shape()
        if any(n < 0 for n in declared):
            sizes=iter(op.operands);shape=[];capacity=[]
            for extent in declared:
                if extent>=0:shape.append(extent);capacity.append(extent);continue
                source=next(sizes);interval=self._interval(source)
                if interval is None or interval[0]<0:
                    raise LoweringDeclined("dynamic host tensor allocation requires a proven nonnegative finite capacity",op=op.name)
                shape.append(self.get(source));capacity.append(interval[1])
            out=self._new(shape,elem_name(op.results[0].type.element_type),capacity)
            self._runtime_loop(0,self._count(shape),1,lambda iv:self._write(out.pointer,iv,self.zero(out.ety),out.ety))
            self.vals[op.results[0]]=out;return
        self._splat(op, self.zero(elem_name(op.results[0].type.element_type)))

    def _interval(self, value, env=None):
        """Conservative source integer interval, used only for storage capacity."""
        env={} if env is None else env
        if value in env:return env[value]
        owner=value.owner
        if not hasattr(owner,"name"):
            typ=value.type.element_type if isinstance(value.type,builtin.TensorType) else value.type
            if isinstance(typ,builtin.IntegerType):
                width=typ.width.data;return (-2**(width-1),2**(width-1)-1)
            return None
        name=owner.name
        if name=="arith.constant":
            attr=attr_of(owner,"value")
            if isinstance(attr,builtin.IntegerAttr):return (int(attr.value.data),)*2
            if isinstance(attr,builtin.DenseIntOrFPElementsAttr) and isinstance(value.type.element_type,builtin.IntegerType):
                values=list(attr.get_values());return (min(values),max(values))
        if name in ("tensor.extract","tensor.splat","arith.index_cast","arith.extsi"):
            return self._interval(owner.operands[0],env)
        if name=="arith.extui":
            width=owner.operands[0].type.width.data
            destination=owner.results[0].type.width.data
            return (0,2**width-1) if width<destination else None
        if name=="tensor.dim":
            axis=self.static_int(owner.operands[1]);source=self.vals.get(owner.operands[0])
            if axis is not None and isinstance(source,StoredTensor):return (0,source.capacity[axis])
            return None
        if name in ("arith.addi","arith.subi","arith.muli"):
            a,b=[self._interval(v,env) for v in owner.operands]
            if a is None or b is None:return None
            candidates=[x+y if name=="arith.addi" else x-y if name=="arith.subi" else x*y for x in a for y in b]
            lo,hi=min(candidates),max(candidates)
            width=64 if isinstance(value.type,builtin.IndexType) else value.type.width.data
            return (lo,hi) if -2**(width-1)<=lo<=hi<2**(width-1) else None
        if name=="linalg.generic":
            if any("reduction" in str(kind.data).lower() for kind in owner.iterator_types):return None
            block=owner.regions[0].block
            arguments={arg:self._interval(operand,env) for arg,operand in zip(block.args,owner.operands)}
            for inner in block.ops:
                if inner.name=="linalg.yield":
                    return self._interval(inner.operands[list(owner.results).index(value)],arguments)
                if len(inner.results)==1:arguments[inner.results[0]]=self._interval(inner.results[0],arguments)
        if name=="linalg.reduce":
            count=len(owner.operands)//2;index=list(owner.results).index(value)
            block=owner.regions[0].block;yielded=block.last_op.operands[index];addition=yielded.owner
            if addition.name!="arith.addi" or set(addition.operands)!={block.args[index],block.args[count+index]}:return None
            source=self._interval(owner.operands[index],env);initial=self._interval(owner.operands[count+index],env)
            if source is None or initial is None:return None
            dims=self._static(owner,"dimensions");shape=tensor_shape(owner.operands[index].type)
            size=prod(shape[d] for d in dims)
            lo,hi=initial[0]+size*source[0],initial[1]+size*source[1]
            width=value.type.element_type.width.data
            return (lo,hi) if -2**(width-1)<=lo<=hi<2**(width-1) else None
        return None

    def _t_tensor_splat(self, op):
        self._splat(op, self.get(op.operands[0]))

    def _t_tensor_extract(self, op):
        self.vals[op.results[0]]=self._scalar_op(op,[self.get(v) for v in op.operands])

    def _t_tensor_concat(self, op):
        inputs=[self.get(v) for v in op.operands]
        typ=op.results[0].type;shape=tensor_shape(typ)
        axis=int(attr_of(op,"dim").value.data)
        out=self._new(shape,elem_name(typ.element_type));offset=0
        for source in inputs:
            base=offset
            traversal = source.traversal if isinstance(source,StoredTensor) else None
            groups = self._merged_axes(traversal[0],source.shape) if traversal else None
            domain,read = traversal if groups is not None else (source.shape,source.at)
            def body(indices):
                coordinates = (self._merged_coordinates(indices,domain,groups)
                               if groups is not None else indices)
                target=list(coordinates);target[axis]=self._add(target[axis],base)
                self._write(out.pointer,self._flat(target,out.shape),read(indices),out.ety)
            self._loops(domain,body);offset+=source.shape[axis]
        if offset!=shape[axis]:raise LoweringDeclined("concat dimension mismatch",op=op.name)
        self.vals[op.results[0]]=out

    def _materialize(self, value):
        if not isinstance(value,TensorVal):return value
        if isinstance(value,StoredTensor) and value.pointer is not None:return value
        out=self._new(value.shape,value.ety,getattr(value,"capacity",value.shape));self._copy(value,out)
        return out

    def _pack(self, values):
        flat=[];templates=[]
        for value in values:
            value=self._materialize(value)
            templates.append(value)
            if isinstance(value,TensorVal):
                flat.append(value.pointer)
                flat.extend(self._v(n) for n in value.shape if type(n) is not int)
            else:flat.append(value)
        return flat,templates

    def _unpack(self, templates, flat):
        values=[];flat=iter(flat)
        for template in templates:
            if isinstance(template,TensorVal):
                pointer=next(flat)
                shape=[n if type(n) is int else next(flat) for n in template.shape]
                values.append(self._stored(shape,template.ety,pointer,template.capacity))
            else:values.append(next(flat))
        return values

    def _source_region(self, block, arguments=()):
        self.vals.update(zip(block.args,arguments,strict=True))
        for operation in block.ops:
            if operation.name=="scf.yield":return [self.get(v) for v in operation.operands]
            self._tensor_op(operation)
        raise LoweringDeclined("source SCF region lacks yield",op="scf")

    def _t_scf_for(self, op):
        start,stop,step=[self.get(v) for v in op.operands[:3]]
        step_interval=self._interval(op.operands[2])
        if step_interval is None or step_interval[0]<=0:
            raise LoweringDeclined("source scf.for requires a proven positive step",op=op.name)
        initial,templates=self._pack([self.get(v) for v in op.operands[3:]])
        types=[value.type for value in initial]
        header,work,done=Block(arg_types=[builtin.i64,*types]),Block(),Block(arg_types=types)
        for block in (header,work,done):self.fb.region.add_block(block)
        self.fb.add(llvm.BrOp(header,start,*initial));self.fb.blk=header
        condition=self.fb.add(llvm.ICmpOp(header.args[0],stop,builtin.IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(),builtin.i64))).results[0]
        self.fb.add(llvm.CondBrOp(condition,work,[],done,list(header.args[1:])));self.fb.blk=work
        carried=self._unpack(templates,header.args[1:])
        # Separate loop-entry snapshots from scratch sites rewritten by later
        # iterations, retaining tensor SSA semantics inside each iteration.
        for index,value in enumerate(carried):
            if isinstance(value,TensorVal):
                copied=self._new(value.shape,value.ety,value.capacity);self._copy(value,copied);carried[index]=copied
        yielded=self._source_region(op.regions[0].block,[header.args[0],*carried])
        packed,_=self._pack(yielded)
        latch=self.fb.add(llvm.BrOp(header,self._add(header.args[0],step),*packed))
        from merlin.llvmlower.llvm_loop_metadata import disable_loop_unroll
        disable_loop_unroll(latch);self.fb.blk=done
        self.vals.update(zip(op.results,self._unpack(templates,done.args),strict=True))

    def _t_scf_if(self, op):
        bit=self.fb.add(llvm.TruncOp(self.get(op.operands[0]),builtin.i1)).results[0]
        yes,no=Block(),Block()
        for block in (yes,no):self.fb.region.add_block(block)
        self.fb.add(llvm.CondBrOp(bit,yes,[],no,[]))
        self.fb.blk=yes;yes_values=self._source_region(op.regions[0].block)
        packed,templates=self._pack(yes_values)
        done=Block(arg_types=[v.type for v in packed]);self.fb.region.add_block(done)
        self.fb.add(llvm.BrOp(done,*packed))
        self.fb.blk=no;no_values=self._source_region(op.regions[1].block)
        packed_no,no_templates=self._pack(no_values)
        if len(packed_no)!=len(packed):raise LoweringDeclined("source if descriptor mismatch",op=op.name)
        for yes_template,no_template in zip(templates,no_templates):
            if isinstance(yes_template,TensorVal) and yes_template.capacity!=no_template.capacity:
                raise LoweringDeclined("source if tensor capacities differ",op=op.name)
        self.fb.add(llvm.BrOp(done,*packed_no));self.fb.blk=done
        self.vals.update(zip(op.results,self._unpack(templates,done.args),strict=True))

    def _t_tensor_insert(self, op):
        value=self.get(op.operands[0]);source=self.get(op.operands[1])
        out=self._new(source.shape,source.ety,source.capacity);self._copy(source,out)
        self._write(out.pointer,self._flat([self.get(v) for v in op.operands[2:]],out.shape),value,out.ety)
        self.vals[op.results[0]]=out

    def _t_tensor_dim(self, op):
        source=self.get(op.operands[0]);axis=self.static_int(op.operands[1])
        if axis is None:raise LoweringDeclined("tensor.dim requires a static axis",op=op.name)
        self.vals[op.results[0]]=self._v(source.shape[axis])

    def _t_cf_assert(self, op):
        bit=self.fb.add(llvm.TruncOp(self.get(op.operands[0]),builtin.i1)).results[0]
        success,failure=Block(),Block()
        self.fb.region.add_block(success);self.fb.region.add_block(failure)
        self.fb.add(llvm.CondBrOp(bit,success,[],failure,[]));self.fb.blk=failure
        self.fb.add(llvm.CallIntrinsicOp("llvm.trap",[],[]));self.fb.add(llvm.UnreachableOp())
        self.fb.blk=success

    _t_linalg_fill = _t_tensor_splat

    def _reshape(self, op):
        source = self.get(op.operands[0]); typ = op.results[0].type
        shape, ety = tensor_shape(typ), elem_name(typ.element_type)
        if prod(shape) != prod(source.shape) or source.ety != ety:
            raise LoweringDeclined("reshape changes element count or scalar type", op=op.name)
        traversal = ((source.traversal or (source.shape,source.at))
                     if isinstance(source,StoredTensor) else None)
        self.vals[op.results[0]] = StoredTensor(self, shape, ety, source.elems.__getitem__,
            getattr(source,"pointer",None),traversal=traversal)

    _t_tensor_expand_shape = _reshape
    _t_tensor_collapse_shape = _reshape
    _t_tensor_reshape = _reshape
    _t_tensor_bitcast = _reshape

    def _t_linalg_generic(self, op):
        from pathlib import Path
        from merlin.compile.tuning_knobs import knob
        borrowed = getattr(self, 'host_tensor_views', None)
        if borrowed is None:
            borrowed = knob('host_tensor_views', default='off', values=('off','on'),
                            root=Path(__file__).resolve().parent) == 'on'
        if type(borrowed) is not bool:
            raise ValueError('host_tensor_views requires a Boolean override')
        if borrowed:
            from gemmini_host_views import copy_view
            descriptor = copy_view(op)
            if descriptor is not None:
                source = self.get(op.operands[0])
                shape = descriptor['shape']
                def coordinates(indices):
                    return source.at(tuple(indices[value] if kind == 'axis' else self._v(value)
                        for kind,value in descriptor['coordinates']))
                def read(flat):
                    return coordinates(self._coordinates(flat,shape))
                # No physical pointer is exposed for a noncontiguous view.
                # _pack and store_result retain their ordinary full copies.
                self.vals[op.results[0]] = StoredTensor(self,shape,source.ety,read,
                    coordinate_reader=coordinates)
                if not hasattr(self,'host_view_reports'):
                    self.host_view_reports = []
                self.host_view_reports.append(descriptor['report'])
                return
        # The generic form must not become an accidental scalar implementation
        # of the same signed i8 dot the graph owner is obliged to accelerate.
        if (any("reduction" in str(kind.data).lower() for kind in op.iterator_types)
                and len(op.inputs) >= 2
                and all(str(v.type.element_type) == "i8" for v in op.inputs[:2])
                and any(inner.name == "arith.muli" for inner in op.regions[0].walk())):
            raise LoweringDeclined("integer product requires accelerator graph lowering", op=op.name)
        n_in = len(op.inputs)
        ins = [self.get(v) for v in op.inputs]
        initial = [self.get(v) for v in op.outputs]
        maps = [m.data for m in op.indexing_maps]
        bounds = [0]*maps[0].num_dims
        for mapping, tensor in zip(maps, ins+initial, strict=True):
            for expression, extent in zip(mapping.results, tensor.shape, strict=True):
                if isinstance(expression, AffineDimExpr):
                    old = bounds[expression.position]
                    if old not in (0, extent):
                        raise LoweringDeclined("inconsistent generic dimension extent", op=op.name)
                    bounds[expression.position] = extent
        results = [self._new(tensor_shape(v.type),elem_name(v.type.element_type)) for v in op.results]
        if len(results) != len(initial):
            raise LoweringDeclined("generic result/init arity mismatch", op=op.name)
        # A fused pointwise destination is fully overwritten exactly once and
        # its old value is unobserved. Avoid initializing or loading that value.
        full_write = ('prov.host_pointwise_fused' in op.attributes
                      and all('parallel' in str(kind.data).lower() for kind in op.iterator_types)
                      and all(mapping.num_dims == len(mapping.results)
                              and sorted(e.position if isinstance(e, AffineDimExpr) else -1
                                         for e in mapping.results) == list(range(mapping.num_dims))
                              for mapping in maps[n_in:]))
        unused = [full_write and not argument.uses for argument in op.body.block.args[n_in:]]
        for init, output, skip in zip(initial, results, unused, strict=True):
            if not skip:
                self._copy(init, output)
        def body(indices):
            self.ivs = indices
            arguments = [tensor.at(tuple(self._affine(e, indices) for e in mapping.results))
                for mapping,tensor in zip(maps[:n_in], ins, strict=True)]
            arguments.extend(self.zero(tensor.ety) if skip else
                tensor.at(tuple(self._affine(e, indices) for e in mapping.results))
                for mapping,tensor,skip in zip(maps[n_in:],results,unused,strict=True))
            values = self._body(op.regions[0].block, arguments)
            if len(values) != len(results):
                raise LoweringDeclined("generic yield/result arity mismatch", op=op.name)
            for mapping, output, value in zip(maps[n_in:],results,values,strict=True):
                coordinates = tuple(self._affine(e, indices) for e in mapping.results)
                self._write(output.pointer,self._flat(coordinates,output.shape),self._round(value,output.ety),output.ety)
        self._loops(bounds,body); self.ivs=()
        self.vals.update(zip(op.results,results,strict=True))

    def _t_linalg_map(self, op):
        inputs=[self.get(v) for v in op.operands[:-1]]
        results=[self._new(tensor_shape(v.type),elem_name(v.type.element_type)) for v in op.results]
        def body(indices):
            self.ivs=indices
            values=self._body(op.regions[0].block,[t.at(indices) for t in inputs])
            for out,value in zip(results,values,strict=True):
                self._write(out.pointer,self._flat(indices,out.shape),self._round(value,out.ety),out.ety)
        self._loops(results[0].shape,body);self.ivs=()
        self.vals.update(zip(op.results,results,strict=True))

    def _t_linalg_reduce(self, op):
        n_in=len(op.operands)//2
        inputs=[self.get(v) for v in op.operands[:n_in]]
        inits=[self.get(v) for v in op.operands[n_in:]]
        dims=set(self._static(op,"dimensions"))
        results=[self._new(tensor_shape(v.type),elem_name(v.type.element_type)) for v in op.results]
        for init,out in zip(inits,results,strict=True):self._copy(init,out)
        def body(indices):
            self.ivs=indices
            keep=tuple(value for dim,value in enumerate(indices) if dim not in dims)
            values=self._body(op.regions[0].block,[t.at(indices) for t in inputs]+[t.at(keep) for t in results])
            for out,value in zip(results,values,strict=True):
                self._write(out.pointer,self._flat(keep,out.shape),self._round(value,out.ety),out.ety)
        self._loops(inputs[0].shape,body);self.ivs=()
        self.vals.update(zip(op.results,results,strict=True))

    def _t_linalg_transpose(self, op):
        source=self.get(op.operands[0]); perm=self._static(op,"permutation")
        typ=op.results[0].type;shape=tensor_shape(typ)
        def coordinates(output_indices):
            indices=[None]*len(perm)
            for dim,axis in enumerate(perm):indices[axis]=output_indices[dim]
            return source.at(tuple(indices))
        def read(flat):
            return coordinates(self._coordinates(flat,shape))
        traversal = source.traversal if isinstance(source,StoredTensor) else None
        groups = self._merged_axes(traversal[0],source.shape) if traversal else None
        copy_traversal = None
        if groups is not None:
            domain,source_read = traversal
            def target_coordinates(indices):
                source_coordinates = self._merged_coordinates(indices,domain,groups)
                return tuple(source_coordinates[axis] for axis in perm)
            # This domain is bijective but no longer visits output row-major.
            # Keep it separate from reshape's row-major traversal descriptor.
            copy_traversal = (domain,target_coordinates,source_read)
        self.vals[op.results[0]]=StoredTensor(self,shape,elem_name(typ.element_type),read,
            coordinate_reader=coordinates,copy_traversal=copy_traversal)

    def _t_linalg_broadcast(self, op):
        source=self.get(op.operands[0]); dims=set(self._static(op,"dimensions"))
        typ=op.results[0].type;shape=tensor_shape(typ)
        def coordinates(indices):
            return source.at(tuple(value for dim,value in enumerate(indices) if dim not in dims))
        self.vals[op.results[0]]=StoredTensor(self,shape,elem_name(typ.element_type),
            lambda flat:coordinates(self._coordinates(flat,shape)),coordinate_reader=coordinates)

    def _t_tensor_extract_slice(self, op):
        source=self.get(op.operands[0]);typ=op.results[0].type;shape=tensor_shape(typ)
        offsets=self._static(op,"static_offsets");sizes=self._static(op,"static_sizes");strides=self._static(op,"static_strides")
        if any(v<0 for v in offsets+sizes+strides) or prod(sizes)!=prod(shape):
            raise LoweringDeclined("slice requires complete static extents",op=op.name)
        def coordinates(indices):
            return source.at(tuple(self._add(off,self._mul(index,stride)) for off,index,stride in zip(offsets,indices,strides,strict=True)))
        def read(flat):
            return coordinates(self._coordinates(flat,sizes))
        # Rank-reduced slices retain their full static source iteration domain.
        self.vals[op.results[0]]=StoredTensor(self,shape,elem_name(typ.element_type),read,
            coordinate_reader=coordinates if tuple(sizes)==shape else None,
            traversal=(tuple(sizes),coordinates))

    def _t_tensor_insert_slice(self, op):
        source,destination=[self.get(v) for v in op.operands[:2]]
        typ=op.results[0].type;out=self._new(tensor_shape(typ),elem_name(typ.element_type))
        offsets=self._static(op,"static_offsets");sizes=self._static(op,"static_sizes");strides=self._static(op,"static_strides")
        if len(source.shape)!=len(sizes) or any(v<0 for v in offsets+sizes+strides):
            raise LoweringDeclined("insert slice needs rank-preserving static extents",op=op.name)
        self._copy(destination,out)
        def body(indices):
            position=tuple(self._add(off,self._mul(index,stride)) for off,index,stride in zip(offsets,indices,strides,strict=True))
            self._write(out.pointer,self._flat(position,out.shape),source.at(indices),out.ety)
        self._loops(sizes,body);self.vals[op.results[0]]=out

    def _t_linalg_matmul(self, op):
        lhs,rhs,initial=[self.get(v) for v in op.operands]
        if is_int(lhs.ety) or is_int(rhs.ety) or is_int(initial.ety):
            raise LoweringDeclined("integer products must be removed by the accelerator graph owner",op=op.name)
        if len(lhs.shape)!=2 or len(rhs.shape)!=2:
            raise LoweringDeclined("named host float matmul requires rank two",op=op.name)
        m,k=lhs.shape;rk,n=rhs.shape
        if k!=rk or initial.shape!=(m,n):
            raise LoweringDeclined("host float matmul has incompatible extents",op=op.name)
        out=self._new(initial.shape,initial.ety);self._copy(initial,out)
        def body(indices):
            row,column,depth=indices
            value=self._round(self.fp.fadd(out.at((row,column)),
                self.fp.fmul(lhs.at((row,depth)),rhs.at((depth,column)))),out.ety)
            self._write(out.pointer,self._flat((row,column),out.shape),value,out.ety)
        self._loops((m,n,k),body);self.vals[op.results[0]]=out

    _t_linalg_batch_matmul = _t_linalg_matmul
