"""Source-stable region preparation for mixed W8A8 tensor programs.

Preparation only transports original SSA. It never infers device eligibility
from provenance and never replaces a refused integer operation with host work.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.ir import Block, Region, Operation
from merlin.common.dense_float_ir import DenseFloatPrinter
from xdsl.traits import Pure


def _text(op):
    stream = StringIO()
    DenseFloatPrinter(stream=stream, print_generic_format=True).print_op(op)
    return stream.getvalue()


def _tensor(value):
    return isinstance(value.type, TensorType)


def _scaffold(op):
    if op.name in ('arith.constant', 'tensor.empty', 'tensor.splat', 'linalg.fill'):
        return True
    # Scalar expressions can be shared across source regions without introducing
    # a pointer ABI for a scalar. Recompute the original pure expression exactly.
    return (op.name.startswith('arith.') and not op.regions
            and not any(_tensor(v) for v in (*op.operands, *op.results))
            and op.has_trait(Pure, value_if_unregistered=False))


def _members(operations):
    return {nested for op in operations for nested in op.walk()}


def outline_region(group, available):
    """Clone an original group with its exact scaffold and scalar closure.

    Returns ``(module, inputs, outputs, selected_original_operations)``. The
    order of inputs is their original first use; outputs are every tensor result
    observed outside the selected group. Original source attributes survive on
    all clones, including operations in scalar regions. ``available`` contains
    original SSA values already materialized by the calling program.
    """
    group = tuple(group)
    if not group or len(set(group)) != len(group):
        raise ValueError('region needs distinct original operations')
    original_block = group[0].parent_block()
    if original_block is None or any(op.parent_block() is not original_block for op in group):
        raise ValueError('region operations must have one original block')
    position = {op: i for i, op in enumerate(original_block.ops)}
    if list(group) != sorted(group, key=position.__getitem__):
        raise ValueError('region operations must retain source order')
    selected = set(group)
    available = set(available)
    external = []

    def internal(value):
        owner = value.owner
        if isinstance(owner, Operation):
            if owner in selected:
                return True
            parent = owner.parent_op()
        else:
            parent = owner.parent_op()
        while parent is not None:
            if parent in selected:
                return True
            parent = parent.parent_op()
        return False

    def require(value):
        if internal(value):
            return
        if value in available:
            if not _tensor(value):
                raise ValueError('materialized region boundary must be tensor typed')
            if value not in external:
                external.append(value)
            return
        producer = value.owner
        if not isinstance(producer, Operation) or producer.parent_block() is not original_block or not _scaffold(producer):
            raise ValueError('unavailable original SSA dependency at region boundary: ' + str(value.type))
        selected.add(producer)
        for nested in producer.walk():
            for operand in nested.operands:
                require(operand)

    for op in group:
        for nested in op.walk():
            for operand in nested.operands:
                require(operand)
    ordered = [op for op in original_block.ops if op in selected]
    members = _members(ordered)
    outputs = [v for op in group for v in op.results
               if _tensor(v) and any(use.operation not in members for use in v.uses)]
    body = Block(arg_types=[v.type for v in external])
    mapping = dict(zip(external, body.args))
    for op in ordered:
        cloned = op.clone(value_mapper=mapping)
        body.add_op(cloned)
        mapping.update(zip(op.results, cloned.results))
    body.add_op(func.ReturnOp(*(mapping[v] for v in outputs)))
    owner = func.FuncOp('forward', ([v.type for v in external], [v.type for v in outputs]), Region(body))
    module = ModuleOp([owner])
    module.verify()
    return module, external, outputs, ordered


@dataclass(frozen=True)
class IntegerContraction:
    operation: object
    source_ordinal: int
    source_region_id: str | None
    source_node_ids: tuple[str, ...]
    operand_types: tuple[str, ...]
    result_types: tuple[str, ...]


def integer_contractions(module):
    """Find actual typed integer reductions, retaining original source identity.

    This is a preparation census, not a device admission certificate. Scalar body
    and affine map legality remain the device lowerer's responsibility.
    """
    result = []
    owners = [op for op in module.body.block.ops if isinstance(op, func.FuncOp)
              and op.sym_name.data == 'forward' and len(op.body.blocks) == 1]
    if len(owners) != 1:
        raise ValueError('one closed original forward body required')
    for ordinal, op in enumerate(owners[0].body.block.ops):
        inputs = tuple(getattr(op, 'inputs', ()))
        if len(inputs) < 2 or len(op.results) != 1:
            continue
        if not all(_tensor(v) for v in (*inputs[:2], *op.results)):
            continue
        if tuple(str(v.type.element_type) for v in (*inputs[:2], *op.results)) != ('i8', 'i8', 'i32'):
            continue
        scalar_names = {nested.name for nested in op.walk()}
        if not ({'arith.muli', 'arith.addi'} <= scalar_names or op.name in ('linalg.matmul', 'linalg.batch_matmul')):
            continue
        region = getattr(op.attributes.get('prov.region_id'), 'data', None)
        ids = op.attributes.get('prov.source_node_ids')
        result.append(IntegerContraction(op, ordinal, region,
            tuple(getattr(item, 'data', str(item)) for item in ids) if ids is not None else (),
            tuple(str(v.type) for v in inputs), tuple(str(v.type) for v in op.results)))
    return tuple(result)


def prepare_integer_regions(source):
    """Expose independent original contraction projections with public APIs.

    The returned ``(module, projections)`` retains original SSA input/output
    bindings in every SourceSSARegion. Only literal accumulator seeds and their
    scalar constants are captured; layout preparation and quantizers remain
    explicit boundaries, preserving their exact rounding and source semantics.
    This does not schedule mutually dependent contractions together.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.source_region import extract_source_region

    module = parse_mlir_text(source)
    before = _text(module)
    projections = tuple(extract_source_region(
        module, (root.operation,), capture_constants=True,
        capture_literal_splats=True, max_literal_splat_elements=64 << 20,
    ) for root in integer_contractions(module))
    if _text(module) != before:
        raise ValueError('integer region preparation changed original source')
    return module, projections
