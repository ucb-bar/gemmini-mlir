"""Stock-tool transport for Merlin's broadcast-safe host fusion contract.

The library's fusion_guard runner uses native Python MLIR bindings. This
adapter supplies the same producer-use control to the selected mlir-opt tool:
pins live only across its ordinary fusion pass, before bufferization. Scalar
programs, their floating flags, and reduction order are never rewritten here.
"""
from __future__ import annotations

import hashlib
from math import prod
from pathlib import Path

from xdsl.dialects.builtin import TensorType, UnregisteredOp
from xdsl.ir import Operation
from xdsl.ir.affine import AffineDimExpr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import fusion_guard


_PIN = 'merlin.fusion_pin'
_RESHAPES = ('tensor.expand_shape', 'tensor.collapse_shape', 'tensor.cast')


def selection():
    from merlin.compile.tuning_knobs import knob
    return knob('host_optimization', default='off', values=('off', 'guarded_fusion'),
                root=Path(__file__).resolve().parent)


def _elements(value):
    typ = value.type
    if not isinstance(typ, TensorType) or any(d < 0 for d in typ.get_shape()):
        return None
    return prod(typ.get_shape())


def _iterations(operation):
    """Static domain read through projected operand axes, as fusion_guard does."""
    maps = getattr(operation, 'indexing_maps', None)
    if maps is None or not maps or len(maps) != len(operation.operands):
        return None
    maps = tuple(maps)
    ranges = [None] * maps[0].data.num_dims
    for value, attribute in zip(operation.operands, maps, strict=True):
        if not isinstance(value.type, TensorType):
            continue
        shape, mapping = value.type.get_shape(), attribute.data
        if len(mapping.results) != len(shape):
            return None
        for expression, extent in zip(mapping.results, shape, strict=True):
            if isinstance(expression, AffineDimExpr) and extent >= 0:
                previous = ranges[expression.position]
                if previous is not None and previous != extent:
                    return None
                ranges[expression.position] = extent
    return prod(ranges) if all(value is not None for value in ranges) else None


def _permutation(mapping):
    return (not mapping.num_symbols and len(mapping.results) == mapping.num_dims
            and all(isinstance(e, AffineDimExpr) for e in mapping.results)
            and sorted(e.position for e in mapping.results) == list(range(mapping.num_dims)))


def _refused(module):
    """Binding adapter for fusion_guard._fg_refused's existing control rule."""
    rows = []
    for consumer in module.walk():
        if consumer.name != 'linalg.generic':
            continue
        maps = tuple(getattr(consumer, 'indexing_maps', ()))
        if len(maps) != len(consumer.operands):
            continue
        for index, value in enumerate(consumer.inputs):
            source = value
            while (isinstance(source.owner, Operation) and source.owner.name in _RESHAPES
                   and len(tuple(source.uses)) == 1):
                source = source.owner.operands[0]
            producer = source.owner
            if (not isinstance(producer, Operation) or producer.name != 'linalg.generic'
                    or len(producer.results) != 1 or len(tuple(source.uses)) != 1
                    or not all(it.data.value == 'parallel' for it in producer.iterator_types)
                    or not any(op.name not in ('arith.constant', 'linalg.yield')
                               for op in producer.body.block.ops)):
                continue
            iterations, elements = _iterations(consumer), _elements(value)
            multiplies = (iterations > elements if iterations is not None and elements is not None
                          else not _permutation(maps[index].data))
            if multiplies:
                rows.append((source, consumer, {'producer_shape': list(source.type.get_shape()),
                             'producer_elements': _elements(source), 'consumer_iterations': iterations,
                             'scalar_operations': [op.name for op in producer.body.block.ops
                                                   if op.name != 'linalg.yield']}))
    return rows


def _work(module):
    known, unknown, generics = 0, 0, 0
    for operation in module.walk():
        if operation.name != 'linalg.generic':
            continue
        generics += 1
        count = _iterations(operation)
        scalar = sum(op.name.startswith(('arith.', 'math.')) for op in operation.body.block.ops)
        if count is None:
            unknown += 1
        else:
            known += count * scalar
    return {'generic_count': generics, 'known_scalar_evaluations': known,
            'unknown_iteration_domains': unknown}


def run_stock_pipeline(source, lowered, passes, *, tool, invoke, print_module, report=None):
    """Run the actual stock pipeline with optional existing fusion control.

    Off is the incumbent single invocation. On splits only at the library's
    fusion stage, provides its temporary second-use barrier, and removes every
    barrier before continuing the original buffer/LLVM pipeline.
    """
    selected = selection()
    def run(input_path, output_path, stages, stage, *, pins=False):
        arguments = [tool, input_path, '--pass-pipeline=builtin.module(' + ','.join(stages) + ')',
                     '--mlir-print-op-generic', '-o', output_path]
        if pins:
            arguments.append('--allow-unregistered-dialect')
        invoke(arguments, stage=stage)
    if selected == 'off':
        run(source, lowered, passes, 'tensor lowering')
        return
    if fusion_guard.FUSE_PASS not in passes:
        raise ValueError('guarded host fusion requires an actual library fusion stage')
    current, pending, stages = source, [], []
    before = source.read_bytes()
    for index, name in enumerate(passes):
        if name != fusion_guard.FUSE_PASS:
            pending.append(name)
            continue
        checkpoint = source.parent / ('guard_input_' + str(index) + '.mlir')
        run(current, checkpoint, pending, 'pre-fusion lowering')
        pending = []
        module = parse_mlir_text(checkpoint.read_text())
        rows = _refused(module)
        original_work = _work(module)
        for value, consumer, _ in rows:
            pin = UnregisteredOp.with_name(_PIN).create(operands=[value])
            consumer.parent_block().insert_op_before(pin, consumer)
        module.verify()
        pinned = source.parent / ('guard_pinned_' + str(index) + '.mlir')
        fused = source.parent / ('guard_fused_' + str(index) + '.mlir')
        pinned.write_text(print_module(module))
        run(pinned, fused, [name], 'guarded fusion', pins=True)
        module = parse_mlir_text(fused.read_text())
        pins = [op for op in module.walk() if getattr(getattr(op, 'op_name', None), 'data', op.name) == _PIN]
        if (len(pins) != len(rows) or any(len(pin.operands) != 1 for pin in pins)
                or sorted(str(pin.operands[0].type) for pin in pins)
                != sorted(str(value.type) for value, _, _ in rows)):
            raise ValueError('stock fusion did not retain every typed producer barrier')
        for pin in pins:
            pin.parent_block().erase_op(pin)
        module.verify()
        current = source.parent / ('guard_selected_' + str(index) + '.mlir')
        current.write_text(print_module(module))
        stages.append({'pass': name, 'pinned_producers': len(rows),
                       'regions': [row for _, _, row in rows],
                       'before': original_work, 'after': _work(module),
                       'selected_tensor_sha256': hashlib.sha256(current.read_bytes()).hexdigest()})
    run(current, lowered, pending, 'post-fusion lowering')
    if source.read_bytes() != before:
        raise ValueError('host optimization modified its source artifact')
    if report is not None:
        report.update({'schema': 'merlin.stock_host_fusion_guard.v1', 'selection': selected,
                       'mechanism': 'merlin.llvmlower.fusion_guard',
                       'library_sha256': hashlib.sha256(Path(fusion_guard.__file__).read_bytes()).hexdigest(),
                       'source_sha256': hashlib.sha256(before).hexdigest(),
                       'lowered_sha256': hashlib.sha256(lowered.read_bytes()).hexdigest(),
                       'stages': stages, 'arithmetic_rewritten': False,
                       'work_scope': 'static arith/math evaluations in projected linalg.generic domains; '
                                     'not machine instructions, runtime cycles, or named-reduction work'})
