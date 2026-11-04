"""Exact SSA adapter for distributed generic accumulator requantization.

No provenance string establishes semantics. Only static identity-indexed scalar
SSA is joined; scale folding uses the shared complete-i32 transition proof.
The output is the existing internal ACCUMULATOR_READOUT or resident-GEMM ABI.
This module performs no placement, host computation, or completion inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from xdsl.dialects.builtin import TensorType, NoneAttr
from xdsl.ir import Operation, BlockArgument
from xdsl.ir.affine import AffineMap, AffineExpr, AffineDimExpr, AffineConstantExpr
from merlin.llvmlower.requantization import f32, prove_scale
from merlin.llvmlower.integer_gemm import match_integer_gemm


class GenericReadoutError(ValueError):
    pass


def _tensor(value):
    t = value.type
    if not isinstance(t, TensorType) or not isinstance(t.encoding, NoneAttr):
        raise GenericReadoutError('ordinary ranked dense tensors required')
    shape = tuple(t.get_shape())
    if any(d <= 0 for d in shape):
        raise GenericReadoutError('dynamic/empty tensor domain is unsupported')
    return shape, str(t.element_type)


def _literal(value):
    op = value.owner
    if not isinstance(op, Operation):
        return None
    if op.name == 'tensor.splat':
        _safe_scalar(op)
        if len(op.operands) != 1 or len(op.results) != 1 or op.regions:
            raise GenericReadoutError('unproved immutable tensor splat')
        _tensor(value)
        if isinstance(op.operands[0].type, TensorType) or str(op.operands[0].type) != str(value.type.element_type):
            raise GenericReadoutError('tensor splat scalar type mismatch')
        return _literal(op.operands[0])
    if op.name != 'arith.constant':
        return None
    _safe_scalar(op)
    if op.operands or len(op.results) != 1 or op.regions:
        raise GenericReadoutError('unproved immutable literal operation')
    attr = op.properties.get('value', op.attributes.get('value'))
    scalar = getattr(getattr(attr, 'value', None), 'data', None)
    if scalar is not None:
        return scalar
    if hasattr(attr, 'iter_values'):
        vals = iter(attr.iter_values())
        first = next(vals, None)
        if first is not None and all(v == first for v in vals):
            return first
    return None


def _safe_scalar(op):
    if any(not k.startswith('prov.') for k in op.attributes):
        raise GenericReadoutError('unproved scalar attributes')
    for k, v in op.properties.items():
        if k == 'fastmath' and str(v) == '#arith.fastmath<none>':
            continue
        if k == 'overflowFlags' and str(v) == '#arith.overflow<none>':
            continue
        if k == 'value' and op.name == 'arith.constant':
            continue
        raise GenericReadoutError('unproved scalar property '+k)


def _uniform_literal_map(value, amap, output_shape):
    """Prove immutable uniform tensor loads independent of broadcast axes.

    Tensor SSA constants/splats cannot be updated through storage aliases. A
    stored function argument or load is never a constant, even if its shape is
    singleton. Projections and fixed indices are checked against the entire
    static iteration domain; unsupported affine arithmetic is refused.
    """
    shape, dtype = _tensor(value)
    if dtype != 'f32' or _literal(value) is None:
        return False
    if (amap.num_dims != len(output_shape) or amap.num_symbols
            or len(amap.results) != len(shape)):
        return False
    for expr, size in zip(amap.results, shape):
        if isinstance(expr, AffineConstantExpr):
            if not 0 <= expr.value < size:
                return False
        elif isinstance(expr, AffineDimExpr):
            if not 0 <= expr.position < len(output_shape) or output_shape[expr.position] > size:
                return False
        else:
            return False
    return True


@dataclass(frozen=True)
class _Expr:
    kind: str
    args: tuple = ()
    value: object = None
    dtype: str = ''


class _Trace:
    def __init__(self, shape, budget, parent):
        self.shape, self.budget, self.parent = shape, budget, parent
        self.members, self.seen = [], set()
        self.scalar_visits = 0

    def tensor(self, value):
        shape, dtype = _tensor(value)
        literal = _literal(value)
        if literal is not None:
            return _Expr('constant', value=literal, dtype=dtype)
        if shape != self.shape:
            raise GenericReadoutError('scale/zero point must be immutable uniform constants; layouts must agree')
        # The source i32 is an explicit readout boundary, never silently a CPU
        # recomputation. Its integer producer is independently matched below.
        if dtype == 'i32':
            return _Expr('accumulator', value=value, dtype=dtype)
        op = value.owner
        if not isinstance(op, Operation) or op.name != 'linalg.generic':
            raise GenericReadoutError('non-generic floating producer or dynamic quantization parameter')
        if op.parent is not self.parent:
            raise GenericReadoutError('pointwise chain crosses its lexical control scope')
        if op in self.seen:
            raise GenericReadoutError('shared/recurrent pointwise expression needs an explicit multi-output ABI')
        self.seen.add(op)
        if len(self.seen) > self.budget:
            raise GenericReadoutError('source-chain budget exceeded')
        if (len(op.results) != 1 or op.results[0] is not value or len(op.outputs) != 1
                or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
            raise GenericReadoutError('one-result single-block pointwise generic required')
        rank = len(shape)
        maps = tuple(m.data for m in op.indexing_maps)
        if (len(maps) != len(op.operands) or maps[-1] != AffineMap.identity(rank)
                or tuple(i.data.value for i in op.iterator_types) != ('parallel',)*rank):
            raise GenericReadoutError('unproved iteration/output projection')
        for v, amap in zip(op.inputs, maps):
            sh, _ = _tensor(v)
            if sh == shape and amap == AffineMap.identity(rank):
                continue
            if sh == () and amap == AffineMap(rank, 0, ()) and _literal(v) is not None:
                continue
            if _uniform_literal_map(v, amap, shape):
                continue
            from .source_column_scale import column_literal,ColumnScaleError
            try:column=column_literal(v,amap,shape,_safe_scalar)
            except ColumnScaleError as exc:raise GenericReadoutError(str(exc))from exc
            if column is not None:
                continue
            raise GenericReadoutError('nonidentity/per-column/dynamic broadcast requires a separate proof')
        allowed = {'indexing_maps', 'iterator_types', 'operandSegmentSizes'}
        if any(k not in allowed for k in op.properties) or any(not k.startswith('prov.') for k in op.attributes):
            raise GenericReadoutError('unproved generic attributes/properties')
        block = op.regions[0].block
        if len(tuple(block.ops)) > 64:
            raise GenericReadoutError('pointwise scalar-region budget exceeded')
        if (len(block.args) != len(op.operands) or block.args[-1].uses
                or op.outputs[0].type != value.type
                or any(a.type != v.type.element_type for a, v in zip(block.args, op.operands))):
            raise GenericReadoutError('destination is observed or scalar/tensor types disagree')
        end = block.last_op
        if end.name != 'linalg.yield' or len(end.operands) != 1:
            raise GenericReadoutError('one scalar yield required')
        used = set()
        def scalar(v):
            self.scalar_visits += 1
            if self.scalar_visits > 128:
                raise GenericReadoutError('pointwise scalar-expression budget exceeded')
            if isinstance(v, BlockArgument):
                if v.block is not block or v.index >= len(op.inputs):
                    raise GenericReadoutError('unknown scalar capture')
                from .source_column_scale import column_literal,ColumnScaleError
                try:column=column_literal(op.inputs[v.index],maps[v.index],shape,_safe_scalar)
                except ColumnScaleError as exc:raise GenericReadoutError(str(exc))from exc
                if column is not None and _literal(op.inputs[v.index])is None:
                    return _Expr('column_constant',value=column,dtype='f32')
                return self.tensor(op.inputs[v.index])
            owner = v.owner
            if not isinstance(owner, Operation):
                raise GenericReadoutError('unknown scalar owner')
            if owner.parent is not block:
                lit = _literal(v)
                if lit is None:
                    raise GenericReadoutError('nonconstant scalar capture')
                return _Expr('constant', value=lit, dtype=str(v.type))
            used.add(owner)
            if owner.name == 'arith.select':
                from .source_select_readout import trace_select,SelectReadoutError
                try:return trace_select(owner,scalar,used,_Expr,_safe_scalar)
                except SelectReadoutError as exc:
                    raise GenericReadoutError(str(exc)) from exc
            _safe_scalar(owner)
            if owner.name == 'arith.constant':
                return _Expr('constant', value=_literal(v), dtype=str(v.type))
            if len(owner.results) != 1 or owner.regions:
                raise GenericReadoutError('unproved scalar operation')
            allowed = {'arith.sitofp', 'arith.mulf', 'arith.divf', 'arith.addf',
                       'math.roundeven', 'arith.maximumf', 'arith.minimumf', 'arith.fptosi', 'arith.maxsi'}
            if owner.name not in allowed:
                raise GenericReadoutError('unsupported scalar '+owner.name)
            if owner.name == 'arith.maxsi' and str(v.type) != 'i32':
                raise GenericReadoutError('only signed-i32 max source admitted')
            return _Expr(owner.name, tuple(scalar(a) for a in owner.operands), dtype=str(v.type))
        expr = scalar(end.operands[0])
        if any(o not in used and o.name not in ('arith.constant','linalg.yield') for o in block.ops):
            raise GenericReadoutError('unused scalar work/effects cannot be dropped')
        self.members.append(op)
        return expr


def _bound(expr, kind, bound):
    if expr.kind != kind or expr.dtype != 'f32' or len(expr.args) != 2:
        raise GenericReadoutError('missing explicit '+kind+' saturation')
    a, b = expr.args
    if a.kind == 'constant' and a.value == bound:
        return b
    if b.kind == 'constant' and b.value == bound:
        return a
    raise GenericReadoutError('saturation bounds differ from signed i8')


def _strip_clamp(expr):
    expr = _bound(expr, 'arith.minimumf', 127)
    try:
        return _bound(expr, 'arith.maximumf', -128), False
    except GenericReadoutError:
        return _bound(expr, 'arith.maximumf', 0), True


def _zero(expr):
    return (expr.kind == 'constant' and expr.value == 0
            or expr.kind == 'arith.sitofp' and len(expr.args) == 1
            and expr.args[0].kind == 'constant' and expr.args[0].value == 0)


@dataclass(frozen=True)
class GenericReadoutProof:
    final: object
    source: object
    operations: tuple
    scales: tuple
    attributes: dict
    numeric_proof: dict
    producer: object = None
    bias: object = None
    bias_operation: object = None

    def revalidate(self):
        fresh = prove_generic_readout(self.final)
        if fresh != self:
            raise GenericReadoutError('source SSA/numerical proof changed')
        return self

    def record(self):
        return {'schema':'submission.generic_readout.v1', 'shape':list(_tensor(self.source)[0]),
                'source_dtype':'i32', 'output_dtype':'i8', 'scale_granularity':('per_column'if 'acc_scale_by_column'in self.attributes else 'uniform'),
                'zero_point':0, 'attributes':dict(self.attributes), 'numeric_proof':self.numeric_proof,
                'generic_regions':len(self.operations), 'integer_product_proved':self.producer is not None,
                'integer_column_bias':self.bias is not None,
                'scope':'complete i32 domain, i8 output bits; no placement/ownership/completion grant'}

    def readout_buffer(self):
        """Existing internal device callback ABI; source includes any i32 bias."""
        self.revalidate()
        self.require_closed(self.operations)
        if 'acc_scale_by_column'in self.attributes:
            raise GenericReadoutError('standalone ACCUMULATOR_READOUT supports only scalar scales; column scales require resident-product COMMIT')
        shape = list(_tensor(self.source)[0])
        return {'abi_version':'0.1','target':'gemmini',
                'params':{'source_internal_opcodes':['ACCUMULATOR_READOUT']},
                'kernel_abi':{'kind':'whole_program','args':[{'tensor':'src','access':'read'},
                             {'tensor':'dst','access':'write'}],'outputs':['dst']},
                'tensors':{'src':{'shape':shape,'dtype':'i32','role':'input'},
                           'dst':{'shape':shape,'dtype':'i8','role':'output'}},
                'commands':[{'opcode':'ACCUMULATOR_READOUT','operands':{'src':'src','dst':'dst'},
                             'attributes':dict(self.attributes)}],
                'source_group_proof':self.record()}

    def require_closed(self, operations):
        members = set(operations)
        for op in members:
            for value in op.results:
                if value is not self.final.results[0] and any(u.operation not in members for u in value.uses):
                    raise GenericReadoutError('live intermediate requires a separate output ABI')

    def product_buffer(self):
        """Consume a closed zero-initialized signed GEMM and optional i32 bias."""
        self.revalidate()
        if self.producer is None:
            raise GenericReadoutError('no exact signed i8 product source proved')
        root = self.producer
        if any(o.parent is not self.final.parent for o in (root, self.bias_operation) if o is not None):
            raise GenericReadoutError('integer producer crosses lexical control scope')
        dims = match_integer_gemm(root)
        if dims is None or dims.batch != 1:
            raise GenericReadoutError('resident interface requires rank-two integer GEMM')
        members = set(self.operations) | {root}
        if self.bias_operation is not None:
            members.add(self.bias_operation)
        self.require_closed(members)
        tensors = {'lhs':{'shape':[dims.m,dims.k],'dtype':'i8','role':'input'},
                   'rhs':{'shape':[dims.k,dims.n],'dtype':'i8','role':'input'},
                   'dst':{'shape':[dims.m,dims.n],'dtype':'i8','role':'output'}}
        attrs = dict(self.attributes)
        if self.bias is not None:
            tensors['bias'] = {'shape':[dims.n],'dtype':'i32','role':'input'}
            attrs['bias'] = 'bias'
            attrs['epilogue'] = ['bias_add',*attrs['epilogue']]
        return {'abi_version':'0.1','target':'gemmini','tensors':tensors,
                'commands':[{'opcode':'RES_PACK','operands':{'src':'rhs','dst':'w_res'},'attributes':{'layout':'packed_rhs'}},
                            {'opcode':'MATMUL_RESIDENT','operands':{'lhs':'lhs','rhs':'w_res','dst':'acc'}},
                            {'opcode':'COMMIT','operands':{'src':'acc','dst':'dst'},'attributes':attrs},
                            {'opcode':'EVICT','operands':{'handle':'w_res'}}],
                'source_group_proof':self.record()}



def _safe_integer_product(root):
    # The shared shape/body matcher does not validate arithmetic promises or
    # extension attributes. They cannot silently enter this stronger proof.
    allowed = {'operandSegmentSizes', 'indexing_maps'}
    if root.name == 'linalg.generic':
        allowed.add('iterator_types')
    if (any(not k.startswith('prov.') for k in root.attributes)
            or any(k not in allowed for k in root.properties)):
        raise GenericReadoutError('unproved integer producer attributes/properties')
    for value in (*root.operands, *root.results):
        _tensor(value)
    if root.name == 'linalg.matmul' and 'indexing_maps' in root.properties:
        expected = [AffineMap(3,0,(AffineExpr.dimension(0),AffineExpr.dimension(2))),
                    AffineMap(3,0,(AffineExpr.dimension(2),AffineExpr.dimension(1))),
                    AffineMap(3,0,(AffineExpr.dimension(0),AffineExpr.dimension(1)))]
        if [m.data for m in root.properties['indexing_maps']] != expected:
            raise GenericReadoutError('unproved named integer producer indexing maps')
    for region in root.regions:
        for block in region.blocks:
            for scalar in block.ops:
                if scalar.name not in {'arith.extsi','arith.muli','arith.addi','linalg.yield'}:
                    raise GenericReadoutError('unproved integer producer scalar operation')
                _safe_scalar(scalar)
    # Check the actual zero literal as well: its value is part of the shared
    # match, while unknown attributes/properties are outside that contract.
    initializer = root.operands[2].owner
    _safe_scalar(initializer.operands[0].owner)


def _product(source):
    root, bias, bop = source.owner, None, None
    if isinstance(root, Operation) and root.name == 'linalg.generic' and len(root.operands) == 3:
        shape, dtype = _tensor(source)
        block = root.regions[0].block if len(root.regions) == 1 and len(root.regions[0].blocks) == 1 else None
        ops = list(block.ops) if block else []
        expected = [AffineMap.identity(2),AffineMap(2,0,(AffineExpr.dimension(1),)),AffineMap.identity(2)]
        if (len(shape) == 2 and dtype == 'i32' and len(ops) == 2
                and [o.name for o in ops] == ['arith.addi','linalg.yield']
                and len(block.args) == 3 and set(ops[0].operands) == set(block.args[:2])
                and tuple(ops[1].operands) == tuple(ops[0].results)
                and [m.data for m in root.indexing_maps] == expected
                and tuple(i.data.value for i in root.iterator_types) == ('parallel','parallel')
                and _tensor(root.operands[0]) == (shape,'i32')
                and _tensor(root.operands[1]) == ((shape[-1],),'i32') and not block.args[-1].uses
                and root.operands[-1].type == source.type
                and all(str(a.type) == 'i32' for a in block.args)
                and str(ops[0].results[0].type) == 'i32'):
            if any(not k.startswith('prov.') for k in root.attributes) or any(
                    k not in {'indexing_maps','iterator_types','operandSegmentSizes'} for k in root.properties):
                raise GenericReadoutError('unproved integer bias generic properties')
            for scalar in ops:
                _safe_scalar(scalar)
            bop, bias, root = root, root.operands[1], root.operands[0].owner
    if not isinstance(root, Operation) or match_integer_gemm(root) is None:
        return None, None, None
    _safe_integer_product(root)
    return root, bias, bop


def prove_generic_readout(op, *, max_chain_ops=64):
    """Prove one source-generic i32→f32 scales→RNE/saturate→i8 chain.

    Immutable uniform f32 tensor constants may use in-bounds projected or fixed
    broadcast indices. A nonzero zero point, f32 bias, arbitrary division,
    dynamic scale or non-final-axis column mapping,
    layout change, live partial expression, or unproved scale fold is refused.
    """
    if type(max_chain_ops) is not int or not 1 <= max_chain_ops <= 4096:
        raise GenericReadoutError('explicit bounded source-chain budget required')
    if op.name != 'linalg.generic' or len(op.results) != 1:
        raise GenericReadoutError('final one-result generic required')
    shape, dtype = _tensor(op.results[0])
    if not shape or dtype != 'i8':
        raise GenericReadoutError('static nonscalar signed-i8 output required')
    trace = _Trace(shape,max_chain_ops,op.parent)
    expr = trace.tensor(op.results[0])
    if expr.kind != 'arith.fptosi' or expr.dtype != 'i8' or len(expr.args) != 1:
        raise GenericReadoutError('explicit f32-to-signed-i8 conversion required')
    expr = expr.args[0]
    if expr.kind == 'math.roundeven':
        if expr.dtype != 'f32' or len(expr.args) != 1:
            raise GenericReadoutError('binary32 rounding required')
        expr, relu = _strip_clamp(expr.args[0])
    else:
        expr, relu = _strip_clamp(expr)
        if expr.kind == 'arith.addf':
            a,b = expr.args
            expr = b if _zero(a) else a if _zero(b) else None
            if expr is None:
                raise GenericReadoutError('nonzero/unproved post-rounding zero point')
        if expr.kind != 'math.roundeven' or expr.dtype != 'f32' or len(expr.args) != 1:
            raise GenericReadoutError('exact math.roundeven required')
        expr = expr.args[0]
    from .source_select_readout import strip_relu
    expr, source_relu = strip_relu(expr)
    relu = relu or source_relu
    scales = []
    while expr.kind in ('arith.mulf','arith.divf'):
        if expr.dtype != 'f32' or len(expr.args) != 2:
            raise GenericReadoutError('scale arithmetic must be binary32')
        from .source_column_scale import peel_scale,ColumnScaleError
        try:expr,scale=peel_scale(expr)
        except ColumnScaleError as exc:raise GenericReadoutError(str(exc))from exc
        scales.append(scale)
    expr, source_relu = strip_relu(expr)
    relu = relu or source_relu
    if expr.kind != 'arith.sitofp' or expr.dtype != 'f32' or len(expr.args) != 1:
        raise GenericReadoutError('scale source is not signed-i32-to-f32 conversion')
    expr = expr.args[0]
    if expr.kind != 'accumulator' or expr.dtype != 'i32':
        raise GenericReadoutError('no signed-i32 accumulator boundary')
    scales = tuple(reversed(scales)) or (1.0,)
    root,bias,bop = _product(expr.value)
    if any(isinstance(s,tuple)for s in scales):
        if root is None:raise GenericReadoutError('column readout requires exact resident integer product; standalone callback is scalar-only')
        members=set((*trace.members,root,*((bop,)if bop is not None else ())))
        for member in members:
            if any(value is not op.results[0]and any(use.operation not in members for use in value.uses)for value in member.results):
                raise GenericReadoutError('live intermediate requires a separate output ABI')
        from .source_column_scale import prove_columns,ColumnScaleError
        try:attrs,proof=prove_columns(scales,shape[-1],relu)
        except (ColumnScaleError,ValueError)as exc:raise GenericReadoutError(str(exc))from exc
    else:
        try:
            proof = prove_scale(scales,relu=relu)
        except ValueError as exc:
            raise GenericReadoutError(str(exc)) from exc
        attrs = {'output_dtype':'i8','acc_scale':proof['scale'],
                 'epilogue':['acc_scale']+(['relu'] if relu else [])}
    result=GenericReadoutProof(op,expr.value,tuple(trace.members),scales,attrs,proof,root,bias,bop)
    if 'acc_scale_by_column'in attrs:
        result.require_closed((*trace.members,root,*((bop,)if bop is not None else ())))
    return result
