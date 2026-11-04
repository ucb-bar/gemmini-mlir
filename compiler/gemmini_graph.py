"""Source-bound mixed tensor graph lowering for Gemmini logical kernels."""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

_LIBRARY = Path(__file__).resolve().parent / 'compiler_library'
if str(_LIBRARY) not in sys.path:
    sys.path.insert(0, str(_LIBRARY))

from xdsl.dialects import builtin, llvm
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.host_builder import FnBuilder, PTR
from merlin.llvmlower.host_scalar_builder import FpBuilder
from merlin.llvmlower.host_tensor_codegen import attr_of
from gemmini_host import CompactHostLinalg
from gemmini_strict_math import StrictSourceMath
from merlin.targetgen.oot_starterkit.plan import source_operation_inventory
from merlin.transforms.dense_ir import static_zero_integer_matmul_geometry

def _print_lowered(module):
    import io
    from merlin.common.dense_float_ir import DenseFloatPrinter
    class GraphPrinter(DenseFloatPrinter):
        def print_op(self, op):
            previous = self.print_generic_format
            if op.name in ("llvm.inline_asm", "llvm.mlir.constant") and op.attributes:
                self.print_generic_format = True
            try:
                super().print_op(op)
            finally:
                self.print_generic_format = previous
    stream = io.StringIO()
    GraphPrinter(stream=stream).print_op(module)
    return stream.getvalue()


def _operation_name(op):
    return getattr(getattr(op, 'op_name', None), 'data', op.name)


def _zero(value):
    op = value.owner
    if not hasattr(op, 'name') or op.name not in ('tensor.splat', 'linalg.fill'):
        return False
    if (not op.operands or len(op.results) != 1 or op.results[0] is not value
            or any(not key.startswith('prov.') for key in op.attributes)):
        return False
    scalar = op.operands[0].owner
    if (getattr(scalar, 'name', None) != 'arith.constant'
            or scalar.operands or len(scalar.results) != 1
            or any(not key.startswith('prov.') for key in scalar.attributes)):
        return False
    attr = attr_of(scalar, 'value')
    return (isinstance(attr, builtin.IntegerAttr) and attr.value.data == 0
            and attr.type == value.type.element_type)


def _dense_geometry(op):
    types = [v.type for v in op.operands]
    if op.name == 'linalg.generic':
        try:
            return static_zero_integer_matmul_geometry(op)
        except ValueError:
            return None
    if op.name == 'linalg.matmul' and len(types) == 3 and len(op.results) == 1 and not (set(op.properties) - {'operandSegmentSizes'}):
        if all(isinstance(t, builtin.TensorType) for t in types) and [str(t.element_type) for t in types] == ['i8', 'i8', 'i32'] and op.results[0].type == types[2] and all(isinstance(t.encoding, builtin.NoneAttr) for t in types) and all(key.startswith('prov.') for key in op.attributes):
            a, b, c = [tuple(t.get_shape()) for t in types]
            if len(a) == len(b) == len(c) == 2 and min(*a, *b, *c) > 0 and a[1] == b[0] and c == (a[0], b[1]):
                return a[0], a[1], b[1]
    return None


def _general_dense_layout(op):
    """Read a projected-permutation dot with contiguous matrix suffixes.

    Prefix axes may be shared, independently broadcast, or absent from an
    operand. Multiple adjacent free axes form one M/N dimension without a copy.
    The scalar recurrence is checked by the same signed-dot library reader.
    """
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from merlin.transforms.dense_ir import recognize_signed_integer_matmul
    if _operation_name(op) not in ('linalg.generic', 'linalg.batch_matmul') or len(op.operands) != 3 or len(op.results) != 1:
        return None
    vals = list(op.operands) + list(op.results)
    if any(not isinstance(v.type, builtin.TensorType) or not isinstance(v.type.encoding, builtin.NoneAttr)
           or any(d <= 0 for d in v.type.get_shape()) for v in vals):
        return None
    if [str(v.type.element_type) for v in vals] != ['i8', 'i8', 'i32', 'i32'] or op.results[0].type != op.operands[2].type:
        return None
    if _operation_name(op) == 'linalg.batch_matmul':
        if (any(not key.startswith('prov.') and key != 'op_name__' for key in op.attributes)
                or set(op.properties) - {'operandSegmentSizes'}):
            return None
        shapes = [tuple(v.type.get_shape()) for v in op.operands]
        if any(len(shape) != 3 for shape in shapes):
            return None
        a, b, c = shapes
        if a[0] != b[0] or c != (a[0], a[1], b[2]) or a[2] != b[1]:
            return None
        return dict(m=a[1], k=a[2], n=b[2], batch_shape=(a[0],),
                    lhs_batch_axes=(0,), rhs_batch_axes=(0,), out_batch_axes=(0,))
    if len(op.inputs) != 2 or len(op.outputs) != 1 or set(op.properties) - {'indexing_maps', 'iterator_types', 'operandSegmentSizes'}:
        return None
    maps = [attr.data for attr in op.indexing_maps]
    if len(maps) != 3 or any(mp.num_symbols or any(not isinstance(x, AffineDimExpr) for x in mp.results) for mp in maps):
        return None
    axes = [tuple(x.position for x in mp.results) for mp in maps]
    if any(len(set(ax)) != len(ax) for ax in axes):
        return None
    a, b, c = axes
    iters = tuple(attr.data for attr in op.iterator_types)
    red = [i for i, kind in enumerate(iters) if kind == IteratorType.REDUCTION]
    par = [i for i, kind in enumerate(iters) if kind == IteratorType.PARALLEL]
    if len(red) != 1 or sorted(c) != par or any(x not in set(par) | set(red) for ax in axes for x in ax):
        return None
    kdim = red[0]
    if kdim not in a or kdim not in b or set(a) | set(b) != set(c) | {kdim}:
        return None
    extents = {}
    for value, ax in zip(op.operands, axes):
        shape = tuple(value.type.get_shape())
        if len(ax) != len(shape):
            return None
        for dim, extent in zip(ax, shape):
            if dim in extents and extents[dim] != extent:
                return None
            extents[dim] = extent
    # M and N occur on exactly one input; shared free dimensions are batches.
    n_axes = tuple(b[b.index(kdim) + 1:])
    rhs_prefix = tuple(b[:b.index(kdim)])
    m_axes = tuple(x for x in a[:-1] if x not in rhs_prefix)
    batch = tuple(x for x in c if x not in m_axes and x not in n_axes)
    if any(x in a for x in n_axes) or any(x in b for x in m_axes):
        return None
    if c != batch + m_axes + n_axes:
        return None
    if a != tuple(x for x in batch if x in a) + m_axes + (kdim,) or b != tuple(x for x in batch if x in b) + (kdim,) + n_axes:
        return None
    m, k, n = math.prod(extents[x] for x in m_axes), extents[kdim], math.prod(extents[x] for x in n_axes)
    # Canonicalize only a detached shell to ask the established reader to check
    # ordered signed widening, modular multiplication/addition and exact yield.
    fake = Block(arg_types=[builtin.TensorType(builtin.i8, (m, k)), builtin.TensorType(builtin.i8, (k, n)), builtin.TensorType(builtin.i32, (m, n))])
    from xdsl.ir.affine import AffineDimExpr as D
    d0, d1, d2 = D(0), D(1), D(2)
    canonical = GenericOp(inputs=fake.args[:2], outputs=fake.args[2:], body=op.regions[0].clone(),
        indexing_maps=builtin.ArrayAttr(builtin.AffineMapAttr(mp) for mp in (
            AffineMap(3, 0, (d0, d2)), AffineMap(3, 0, (d2, d1)), AffineMap(3, 0, (d0, d1)))),
        iterator_types=builtin.ArrayAttr(IteratorTypeAttr(kind) for kind in (IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION)),
        result_types=[fake.args[2].type])
    canonical.attributes = dict(op.attributes)
    try:
        recognize_signed_integer_matmul(canonical)
    except ValueError:
        return None
    return dict(m=m, k=k, n=n, batch_shape=tuple(extents[x] for x in batch),
                lhs_batch_axes=tuple(i for i, dim in enumerate(batch) if dim in a),
                rhs_batch_axes=tuple(i for i, dim in enumerate(batch) if dim in b),
                out_batch_axes=tuple(range(len(batch))))


def _emit_dot(fn, lhs, rhs, dst, geometry, layout, *, output_dtype, initial=None, attrs=None):
    if layout is not None and layout.get('permutation_preparation'):
        if attrs:
            raise ValueError('scalar readout requires an ordinary output layout')
        from gemmini_graph_layouts import emit
        return emit(fn, lhs, rhs, dst, geometry, layout,
                    output_dtype=output_dtype, initial=initial)
    from gemmini_dense import emit_dense
    from gemmini_encode import pointer_offset
    if layout is None or not layout['batch_shape']:
        return emit_dense(fn, lhs, rhs, dst, *geometry, output_dtype=output_dtype, attrs=attrs, **({"initial": initial} if initial is not None else {}))
    m, k, n = geometry
    batch_shape = layout['batch_shape']
    def offset(iv, axes, elements, element_bytes):
        flat = fn.const(0)
        for axis in axes:
            divisor = math.prod(batch_shape[axis + 1:])
            coordinate = fn.udiv_i(iv, fn.const(divisor)) if divisor != 1 else iv
            coordinate = fn.urem_i(coordinate, fn.const(batch_shape[axis]))
            flat = fn.add_i(fn.mul_i(flat, fn.const(batch_shape[axis])), coordinate)
        return fn.mul_i(flat, fn.const(elements * element_bytes))
    def body(iv):
        ap = pointer_offset(fn, lhs, offset(iv, layout['lhs_batch_axes'], m * k, 1))
        bp = pointer_offset(fn, rhs, offset(iv, layout['rhs_batch_axes'], k * n, 1))
        cp = pointer_offset(fn, dst, offset(iv, layout['out_batch_axes'], m * n, 1 if output_dtype == 'i8' else 4))
        seed = None if initial is None else pointer_offset(fn, initial, offset(iv, layout['out_batch_axes'], m * n, 4))
        emit_dense(fn, ap, bp, cp, m, k, n, output_dtype=output_dtype, attrs=attrs, **({"initial": seed} if seed is not None else {}))
    fn.for_loop(0, math.prod(batch_shape), 1, body, retain_loop=True)


def _identity_readout(readout):
    """The first graph consumer retains the producer's row-major output mapping.

    Numeric admission does not authorize deleting a transpose or changing a
    generic indexing map. More general destination mappings need their own
    consumer, so preserve the original host chain outside this scope.
    """
    from xdsl.ir.affine import AffineMap
    current = readout.root.results[0]
    shape = tuple(current.type.get_shape())
    identity = AffineMap.identity(len(shape))
    for op in readout.members:
        if len(op.results) != 1 or tuple(op.results[0].type.get_shape()) != shape:
            return False
        if op.name == 'linalg.generic':
            inputs = list(op.inputs)
            if current not in inputs or len(op.outputs) != 1:
                return False
            maps = [m.data for m in op.indexing_maps]
            if (maps[inputs.index(current)] != identity or maps[-1] != identity
                    or any('parallel' not in str(i.data).lower() for i in op.iterator_types)):
                return False
        elif _operation_name(op) != 'quant_ext.quantize_per_tensor' or op.operands[0] is not current:
            return False
        current = op.results[0]
    return current is readout.destination


def _saturating_i8(op, root):
    """Prove the closed max(-128), min(127), trunc signed readout."""
    if op.name != 'linalg.generic' or len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
        return False
    if op.inputs[0] is not root.results[0] or str(op.results[0].type.element_type) != 'i8':
        return False
    if tuple(op.results[0].type.get_shape()) != tuple(root.results[0].type.get_shape()):
        return False
    from xdsl.ir.affine import AffineMap
    rank = len(op.results[0].type.get_shape())
    if any(m.data != AffineMap.identity(rank) for m in op.indexing_maps):
        return False
    if any('parallel' not in str(i.data).lower() for i in op.iterator_types):
        return False
    body = op.regions[0].block
    inner = list(body.ops)
    if [x.name for x in inner] != ['arith.maxsi', 'arith.minsi', 'arith.trunci', 'linalg.yield']:
        return False
    lower, upper, cast, yld = inner
    def bound(operation, value, number):
        if value not in operation.operands:
            return False
        others = [v for v in operation.operands if v is not value]
        if len(others) != 1:
            return False
        owner = others[0].owner
        if not hasattr(owner, 'name') or owner.name != 'arith.constant':
            return False
        attr = attr_of(owner, 'value')
        return isinstance(attr, builtin.IntegerAttr) and attr.value.data == number
    return (bound(lower, body.args[0], -128) and bound(upper, lower.results[0], 127)
            and list(cast.operands) == [upper.results[0]] and str(cast.results[0].type) == 'i8'
            and list(yld.operands) == [cast.results[0]])


def _integer_contraction(op):
    if _operation_name(op) not in ('linalg.generic', 'linalg.matmul', 'linalg.batch_matmul'):
        return False
    inputs = list(op.operands)[:2]
    return len(inputs) == 2 and all(isinstance(v.type, builtin.TensorType) and str(v.type.element_type) == 'i8' for v in inputs) and (
        _operation_name(op) != 'linalg.generic' or any('reduction' in str(a.data).lower() for a in op.iterator_types))


class GraphFp(StrictSourceMath, FpBuilder):
    def sqrt(self, value):
        # Source math.sqrt has no approximation permission. Native LLVM sqrt
        # preserves the rounding boundary consumed by subsequent quantization.
        return self.fb.add(llvm.FSqrtOp(value)).results[0]

    def rsqrt(self, value):
        # A Newton approximation can differ by one f32 ULP and change the exact
        # integer selected by round-even quantization after normalization.
        return self.fdiv(self.fconst(1.0), self.sqrt(value))

    def fconst(self, value):
        terminator = self.fb.entry.last_op
        result = super().fconst(value)
        owner = result.owner
        if (terminator is not None and terminator.has_trait(llvm.IsTerminator)
                and owner.parent is self.fb.entry
                and self.fb.entry.last_op is owner):
            self.fb.entry.detach_op(owner)
            self.fb.entry.insert_op_before(owner, terminator)
        return result


class GraphHost(CompactHostLinalg):
    """Shared scalar host lane with ordinary dense source constant support."""
    def _scalar_op(self, op, ins):
        if op.name == 'math.roundeven':
            return self.fp.round_near_even(ins[0])
        return super()._scalar_op(op, ins)

    def _tensor_op(self, op):
        name = getattr(getattr(op, 'op_name', None), 'data', op.name)
        if name.startswith('quant_ext.'):
            # Expand the exact shared source recipe privately. Original plan
            # indices and source tensor identities remain those of the caller.
            from merlin.llvmlower.passes_xdsl import lower_quant_ext
            from xdsl.dialects.func import ReturnOp
            copied = op.clone(value_mapper={v: v for v in op.operands})
            ret = ReturnOp(*copied.results)
            local = builtin.ModuleOp([copied, ret])
            if lower_quant_ext(local) != 1:
                raise ValueError('unsupported source quantization operation ' + name)
            for expanded in tuple(local.body.block.ops):
                if expanded is not ret:
                    super()._tensor_op(expanded)
            for old, expanded in zip(op.results, ret.operands):
                self.vals[old] = self.get(expanded)
            return
        super()._tensor_op(op)



def compile_linalg(source: str):
    """Compile one statically shaped tensor entry with explicit source ownership.

    Every supported signed integer dot uses Gemmini. Unimplemented integer
    contraction layouts decline before the shared host tensor lane can see them.
    """
    from gemmini_dense import emit_dense
    from gemmini_encode import fence

    module = parse_mlir_text(source)
    module.verify()
    functions = [op for op in module.body.block.ops if op.name == 'func.func' and op.regions and op.regions[0].blocks]
    if len(functions) != 1 or len(functions[0].regions[0].blocks) != 1:
        raise ValueError('graph lowering requires one single-block tensor function')
    fn = functions[0]
    block = fn.regions[0].block
    ops = [op for op in block.ops if op.name != 'func.return']
    returned = block.last_op
    if returned is None or returned.name != 'func.return' or not returned.operands:
        raise ValueError('graph requires explicit nonempty source returns')
    inventory = source_operation_inventory(source)
    index = {op: i for i, op in enumerate(ops)}
    from graph_device import recognize as pointwise_recipe
    pointwise = {op: recipe for op in ops if (recipe := pointwise_recipe(op)) is not None}
    geometry = {op: _dense_geometry(op) for op in ops}
    layouts = {}
    for op in ops:
        if geometry[op] is None:
            layout = _general_dense_layout(op)
            if layout is None:
                from gemmini_graph_layouts import recognize
                layout = recognize(op)
                if layout is None:
                    from gemmini_graph_initial import recognize as ordered_reductions
                    layout = ordered_reductions(op)
            if layout is not None:
                layouts[op] = layout
                geometry[op] = layout['m'], layout['k'], layout['n']
    fused_readout = {}
    fused_members = {}
    scalar_readouts = {}
    for root in ops:
        if geometry[root] is None:
            continue
        users = [u.operation for u in root.results[0].uses]
        if len(users) != 1 or not _saturating_i8(users[0], root):
            continue
        tail = users[0]
        between = ops[index[root] + 1:index[tail]]
        if all(op.name in ('tensor.empty', 'arith.constant') and all(
                use.operation in set(tail.walk()) for result in op.results for use in result.uses) for op in between):
            fused_readout[root] = tail
            fused_members[root] = between + [tail]
    from gemmini_source_readout import plan as plan_source_readout
    for root in ops:
        if (geometry[root] is None or root in fused_readout
                or layouts.get(root, {}).get('permutation_preparation')):
            continue
        readout, _ = plan_source_readout(root, module)
        if readout is None or not _identity_readout(readout):
            continue
        # Constants and empty destination shells have no runtime arithmetic.
        # Absorb one only when every observer belongs to this proven chain.
        members = set(readout.members)
        changed = True
        while changed:
            changed = False
            for helper in reversed(ops):
                if helper in members or helper is root or helper.name not in (
                        'arith.constant', 'tensor.empty', 'tensor.splat', 'linalg.fill'):
                    continue
                uses = [use.operation for result in helper.results for use in result.uses]
                if uses and all(user in members for user in uses):
                    members.add(helper)
                    changed = True
        scalar_readouts[root] = readout
        fused_readout[root] = readout.destination.owner
        fused_members[root] = sorted(members, key=index.__getitem__)
    fused_owned = {op for members in fused_members.values() for op in members}
    for op in ops:
        if _integer_contraction(op) and geometry[op] is None:
            raise ValueError('integer contraction needs a Gemmini lowering for its exact maps/initialization: ' + op.name)

    # Initializers solely consumed by a supported dot are owned by that dot.
    skipped = set()
    for op in ops:
        if geometry[op] is None and op not in pointwise:
            continue
        source_users = set(op.walk())
        todo = [op.operands[-1].owner] if op in pointwise or _zero(op.operands[-1]) else []
        if op in pointwise:
            todo.extend(v.owner for inner in op.walk() for v in inner.operands
                        if not isinstance(v.type, builtin.TensorType) and getattr(v.owner, 'name', None) == 'arith.constant')
        while todo:
            init = todo.pop()
            if init not in index or init.name not in ('arith.constant', 'tensor.empty', 'tensor.splat', 'linalg.fill'):
                continue
            if any(use.operation not in skipped and use.operation not in source_users for result in init.results for use in result.uses):
                # Revisit ancestors only after their user is accepted below.
                if init is not op.operands[-1].owner:
                    continue
                if any(use.operation not in source_users for result in init.results for use in result.uses):
                    continue
            skipped.add(init)
            todo.extend(value.owner for value in init.operands)

    tasks = []
    pending = []
    for op in ops:
        if op in fused_owned:
            continue
        if geometry[op] is not None or op in pointwise:
            prefix = []
            while pending and pending[-1] in skipped:
                prefix.insert(0, pending.pop())
            if pending:
                tasks.append(('host', pending))
                pending = []
            tasks.append(('device', prefix + [op] + fused_members.get(op, [])))
        else:
            pending.append(op)
    if pending:
        if all(op in skipped for op in pending) and tasks:
            tasks[-1][1].extend(pending)
        else:
            tasks.append(('host', pending))

    # Only actual memory boundaries become pointer ABI members.
    needed = set(block.args) | set(returned.operands)
    for op in ops:
        if op in fused_owned:
            continue
        if geometry[op] is not None:
            needed.update(list(op.operands)[:2])
            if not _zero(op.operands[-1]):
                needed.add(op.operands[-1])
            needed.update(fused_readout.get(op, op).results)
        elif op in pointwise:
            needed.update(pointwise[op]['inputs'])
            needed.add(pointwise[op]['output'])
    # Source views remain owned by their original tasks. Only their private
    # materialization boundaries may share a dominating immutable preparation.
    from merlin.compile.tuning_knobs import knob
    boundary_aliases, preparation_receipt = {}, None
    if knob('graph_preparation_reuse', default='off', values=('off', 'on'), root=Path(__file__).resolve().parent) == 'on':
        from gemmini_graph_reuse import plan_boundary_view_reuse
        boundary_aliases, preparation_receipt = plan_boundary_view_reuse(module, needed, returned.operands)
        # Retain a concrete boundary result in every affected host task. Scalar
        # constants can be interned, so source operations alone do not prove
        # that an otherwise eliminated task will emit an instruction.
        for kind, owned in tasks:
            if kind == 'host':
                remaining = [value for op in owned for value in op.results if value in needed and value not in boundary_aliases]
                if not remaining:
                    for op in owned:
                        for value in op.results:
                            boundary_aliases.pop(value, None)
        if preparation_receipt is not None:
            retained_indices = {index[value.owner] for value in boundary_aliases}
            preparation_receipt['rewrites'] = [row for row in preparation_receipt['rewrites'] if row['source_op_index'] in retained_indices]
            if not preparation_receipt['rewrites']:
                preparation_receipt = None
        needed.difference_update(boundary_aliases)
        needed.update(boundary_aliases.values())
    if any(not isinstance(value.type, builtin.TensorType) for value in needed):
        raise ValueError('graph ABI boundary must carry ranked tensors')
    attrs = fn.attributes
    input_names = [a.data for a in attrs.get('merlin.logical_inputs', [])]
    output_names = [a.data for a in attrs.get('merlin.logical_outputs', [])]
    roles = [a.data for a in attrs.get('merlin.logical_input_roles', [])]
    if input_names and len(input_names) != len(block.args):
        raise ValueError('logical input metadata has wrong arity')
    if output_names and len(output_names) != len(returned.operands):
        raise ValueError('logical output metadata has wrong arity')
    input_names = input_names or ['arg' + str(i) for i in range(len(block.args))]
    output_names = output_names or ['output' + str(i) for i in range(len(returned.operands))]
    names = dict(zip(block.args, input_names))
    if len(set(returned.operands)) != len(returned.operands) or any(v in names for v in returned.operands):
        raise ValueError('returned aliases require an explicit source copy contract')
    names.update(zip(returned.operands, output_names))
    values = []
    for i, op in enumerate(ops):
        for j, result in enumerate(op.results):
            if result in needed:
                names.setdefault(result, 'value_' + str(i) + '_' + str(j))
                values.append({'op_index': i, 'result_index': j, 'tensor': names[result]})
    if len(set(names.values())) != len(names):
        raise ValueError('source logical buffer names collide')
    def boundary_name(value):
        return names[boundary_aliases.get(value, value)]

    tensors = {}
    for value, name in names.items():
        role = 'output' if value in returned.operands else 'intermediate'
        if value in block.args:
            n = list(block.args).index(value)
            role = roles[n] if roles else 'input'
        tensors[name] = {'shape': list(value.type.get_shape()), 'dtype': str(value.type.element_type), 'role': role}
    args = [{'tensor': name, 'access': 'read' if value in block.args else 'write'} for value, name in names.items()]
    cb = {'abi_version': '0.1', 'target': 'gemmini', 'tensors': tensors, 'commands': [], 'outputs': output_names,
          'kernel_abi': {'kind': 'whole_program', 'args': args, 'outputs': output_names}, 'params': {}}
    if preparation_receipt is not None:
        preparation_receipt['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        cb['params']['graph_preparation_reuse'] = preparation_receipt
    reuses = {}
    from merlin.compile.tuning_knobs import knob
    shared_input_mode = knob('graph_shared_inputs', default='off', values=('off', 'on', 'counted'), root=Path(__file__).resolve().parent)
    if shared_input_mode != 'off':
        from gemmini_graph_reuse import plan_shared_products
        from gemmini_shared_counted import plan_counted_shared_products
        # The common group reader accepts tensor.splat zero seeds. This adapter
        # also admits the exact zero linalg.fill shell already checked by the
        # source contraction reader, retaining every original initializer index.
        position = 0
        while position < len(tasks):
            candidates = []
            cursor_position = position
            common_lhs = None
            while cursor_position < len(tasks):
                kind, members = tasks[cursor_position]
                roots = [op for op in members if geometry.get(op) is not None]
                if kind != 'device' or len(roots) != 1:
                    break
                root = roots[0]
                if (root in fused_readout or root in layouts or not _zero(root.operands[-1])
                        or any(op is not root and op not in skipped for op in members)):
                    break
                if common_lhs is None:
                    common_lhs = root.operands[0]
                if root.operands[0] is not common_lhs:
                    break
                candidates.append((root, members))
                cursor_position += 1
                if len(candidates) == 16:
                    break
            if len(candidates) < 2:
                position += 1
                continue
            products = [dict(lhs=boundary_name(root.operands[0]), rhs=boundary_name(root.operands[1]),
                             dst=names[root.results[0]], attrs={'output_dtype': 'i32'},
                             source_command_indices=[index[op] for op in members])
                        for root, members in candidates]
            shared_planner = plan_counted_shared_products if shared_input_mode == 'counted' else plan_shared_products
            reuse = shared_planner(tensors, products, source_identity=inventory['source_sha256'])
            if reuse is None:
                position += 1
                continue
            covered = {op for _, members in candidates for op in members}
            tasks[position:cursor_position] = [('device', [op for op in ops if op in covered])]
            reuses[candidates[0][0]] = reuse
            position += 1
    from gemmini_graph_rhs_groups import group_tasks as group_rhs_tasks
    from gemmini_dense import geometry as rhs_geometry
    rhs_edge, _, rhs_operand_rows, _, _ = rhs_geometry()
    rhs_enabled = knob('graph_shared_rhs', default='off', values=('off', 8), root=Path(__file__).resolve().parent) == 8
    rhs_enabled = rhs_enabled and knob('dense_schedule', default='compact', values=('compact','panel_drained','panel_epochs','ready_readout','segmented_ready','auto','auto_ready','array_epochs'), root=Path(__file__).resolve().parent) in ('auto','auto_ready')
    rhs_enabled = rhs_enabled and knob('gemv_column_blocks', default='off', values=('off',16,32,64), root=Path(__file__).resolve().parent) == 'off'
    tasks, rhs_groups = group_rhs_tasks(tasks, geometry, layouts, fused_readout, skipped, block, returned,
        inventory['source_sha256'], max_members=8, tile_edge=rhs_edge,
        rhs_rows=rhs_operand_rows-rhs_operand_rows//2, zero_seed=_zero, enabled=rhs_enabled)
    fb = FnBuilder([PTR] * len(args))
    from gemmini_support import ensure_entry_context
    for instruction in ensure_entry_context(fb):
        instruction.attributes['merlin.global_task'] = builtin.IntegerAttr(-1, builtin.i64)
    ptrs = dict(zip(names, fb.entry.args))
    ptrs.update((value, ptrs[target]) for value, target in boundary_aliases.items())
    host = GraphHost(GraphFp(fb), [], [])
    plan_tasks = []
    cursor = 2
    for task_id, (kind, owned) in enumerate(tasks):
        before = set(fb.region.walk())
        reads, writes = [], []
        if kind == 'device':
            root = next(op for op in owned if geometry.get(op) is not None or op in pointwise)
            if root in rhs_groups:
                from functools import partial
                from gemmini_graph_rhs_groups import emit_group as emit_rhs_group, schedule_shared_rhs
                if knob('graph_row_batch', default='off', values=('off', 8), root=Path(__file__).resolve().parent) == 8:
                    from gemmini_graph_row_batch import emit_group as emit_rhs_group
                from gemmini_dense import emit_resident_group, FACTS_PATH
                from merlin.compile.scheduling.dense_shared_rhs import schedule_dense_shared_rhs
                from gemmini_encode import fence as rhs_fence
                rhs_scheduler = partial(schedule_shared_rhs, schedule_dense_shared_rhs, facts_path=FACTS_PATH)
                reads, writes, rhs_receipt = emit_rhs_group(fb, rhs_groups[root], ptrs, names, boundary_name,
                    emit_resident_group=partial(emit_resident_group, scheduler=rhs_scheduler),
                    emit_dot=_emit_dot, fence=rhs_fence)
                cb['params'].setdefault('graph_shared_rhs', []).append(rhs_receipt)
            elif root in reuses:
                from gemmini_graph_reuse import emit_shared_products
                from gemmini_shared_counted import CountedSharedProducts, emit_counted_shared_products
                reuse = reuses[root]
                emit_reuse = emit_counted_shared_products if isinstance(reuse, CountedSharedProducts) else emit_shared_products
                receipt = emit_reuse(fb, reuse, {name: ptrs[value] for value, name in names.items()})
                cb['params'].setdefault('graph_shared_inputs', []).append(receipt)
                reads = list(dict.fromkeys([p['lhs'] for p in reuse.products] + [p['rhs'] for p in reuse.products]))
                writes = [p['dst'] for p in reuse.products]
            elif geometry.get(root) is None:
                from graph_device import emit as emit_pointwise
                record = pointwise[root]
                emit_pointwise(fb, record, ptrs)
                reads = [boundary_name(v) for v in record['inputs']]
                writes = [names[record['output']]]
            else:
                lhs, rhs = list(root.operands)[:2]
                for val in (lhs, rhs):
                    if val not in block.args and val in host.vals:
                        # Its host producer writes the boundary in its own task.
                        pass
                destination = fused_readout.get(root, root).results[0]
                readout = scalar_readouts.get(root)
                readout_attrs = None
                if readout is not None:
                    readout.revalidate()
                    readout_attrs = {'acc_scale': readout.scale, 'relu': readout.relu}
                    cb['params'].setdefault('source_scalar_readout', []).append(readout.receipt)
                _emit_dot(fb, ptrs[lhs], ptrs[rhs], ptrs[destination], geometry[root], layouts.get(root),
                          output_dtype=str(destination.type.element_type),
                          initial=None if _zero(root.operands[-1]) else ptrs[root.operands[-1]],
                          attrs=readout_attrs)
                fence(fb)
                reads = [boundary_name(lhs), boundary_name(rhs)]
                if not _zero(root.operands[-1]):
                    reads.append(boundary_name(root.operands[-1]))
                writes = [names[destination]]
        else:
            executable = [op for op in owned if op not in skipped]
            inputs = []
            for op in executable:
                for nested in op.walk():
                    for value in nested.operands:
                        if value in names and value not in host.vals and value.owner not in executable:
                            inputs.append((value, ptrs[value]))
            inputs = list(dict(inputs).items())
            outputs = [(value, ptrs[value]) for op in executable for value in op.results if value in needed]
            host.run_segment(executable, inputs, outputs)
            reads = list(dict.fromkeys(names[value] for value, _ in inputs))
            writes = list(dict.fromkeys(names[value] for value, _ in outputs))
        emitted = [op for op in fb.region.walk() if op not in before]
        if not emitted:
            raise ValueError('source task has no executable lowering')
        for lower in emitted:
            lower.attributes['merlin.global_task'] = builtin.IntegerAttr(task_id, builtin.i64)
            # A fused host task owns the complete original source list.
            if len(owned) == 1:
                lower.attributes['merlin.source_op_index'] = builtin.IntegerAttr(index[owned[0]], builtin.i64)
        regions = sorted({inventory['operations'][index[op]]['region_id'] for op in owned if inventory['operations'][index[op]]['region_id'] is not None})
        count = len(emitted)
        plan_tasks.append({'task_index': task_id, 'kind': kind, 'source_op_indices': [index[op] for op in owned],
                           'source_region_ids': regions, 'instruction_start': cursor, 'instruction_end': cursor + count,
                           'reads': reads, 'writes': writes})
        cursor += count
    fence(fb).attributes['merlin.global_task'] = builtin.IntegerAttr(-2, builtin.i64)
    plan = {'schema': 'mixed_program_plan_v1', 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'source_op_count': len(ops), 'tasks': plan_tasks, 'schedule_instruction_count': cursor + 1,
            'prologue_instruction_range': [0, 2], 'epilogue_instruction_range': [cursor, cursor + 1],
            'entry_bindings': input_names, 'source_values': values, 'output_bindings': output_names}
    cb['params']['global_program_plan'] = plan
    fusion_reports = [report for report in getattr(host, 'host_fusion_reports', []) if report.get('rewrites')]
    if fusion_reports:
        cb['params']['host_pointwise_fusion'] = fusion_reports
    view_reports = getattr(host, 'host_view_reports', [])
    if view_reports:
        cb['params']['host_tensor_views'] = view_reports
    from gemmini_support import selected_entry_symbol, completion_function
    lowered = builtin.ModuleOp([*getattr(fb, '_gemmini_globals', []), llvm.FuncOp(selected_entry_symbol(), llvm.LLVMFunctionType([PTR] * len(args), llvm.LLVMVoidType()),
                              linkage=llvm.LinkageAttr('external'), body=fb.finish()), completion_function()])
    from gemmini_command_expr import simplify_command_expressions
    simplify_command_expressions(lowered)
    # Instruction ranges describe the final representation, while the original
    # source operation ownership and task dependencies remain unchanged.
    from collections import Counter
    counts = Counter(
        op.attributes['merlin.global_task'].value.data
        for op in lowered.walk() if 'merlin.global_task' in op.attributes)
    cursor = counts[-1]
    plan['prologue_instruction_range'] = [0, cursor]
    for task in plan_tasks:
        task['instruction_start'] = cursor
        cursor += counts[task['task_index']]
        task['instruction_end'] = cursor
    plan['epilogue_instruction_range'] = [cursor, cursor + counts[-2]]
    plan['schedule_instruction_count'] = cursor + counts[-2]
    lowered.verify()
    return cb, _print_lowered(lowered)
