"""Optional complete static signed-integer concat via native raw transfers."""
from dataclasses import asdict
from math import prod
from pathlib import Path
from hashlib import sha256
from xdsl.dialects.builtin import TensorType, NoneAttr, IntegerAttr
from merlin.common.mlir_query import op_name
from merlin.llvmlower.host_tensor_codegen import tensor_shape
from merlin.compile.scheduling.source_concat_transfer import SourceConcatTransfer
from merlin.compile.scheduling.layout_transfer import LayoutTransfer
from linalg_device_layout import resource_admission as layout_resource_admission

from raw_storage_transfer import emit_raw_run, storage_receipt

ENABLED = True
F32_ENABLED = True


def resource_admission(schedule, facts=None):
    # Reuse the selected raw accumulator/row/column/native-stride field proof.
    # Each run occupies only one accumulator row and <=16 columns. Original
    # large concat pitches appear in pointer arithmetic, never native strides.
    resource = layout_resource_admission(LayoutTransfer((16,), (0,), schedule.dtype, schedule.tile), facts)
    if resource is not None:
        resource['maximum_stride_bytes'] = schedule.tile*schedule.itemsize
        resource['route'] = 'counted_source_concat_transfer'
    return resource


def plan(op):
    """Prove the original same-type full-tensor intervals before any emission."""
    if (op_name(op) != 'tensor.concat' or not op.operands or len(op.results) != 1
        or op.regions or op.successors
        or any(not k.startswith('prov.') for k in op.attributes)
        or set(op.properties)-{'dim'}):
        return None
    values = (*op.operands, *op.results)
    if any(not isinstance(v.type, TensorType) or not isinstance(v.type.encoding, NoneAttr) for v in values):
        return None
    dtype = str(values[0].type.element_type)
    if dtype == 'f32' and not F32_ENABLED:
        return None
    if dtype not in ('i8', 'i32', 'f32') or any(str(v.type.element_type) != dtype for v in values):
        return None
    axis = op.properties.get('dim')
    if not isinstance(axis, IntegerAttr) or str(axis.type) != 'i64':
        return None
    try:
        schedule = SourceConcatTransfer(tuple(tuple(tensor_shape(v.type)) for v in op.operands),
            axis.value.data, tuple(tensor_shape(op.results[0].type)), dtype, 16)
    except (ValueError, TypeError):
        return None
    return schedule if resource_admission(schedule) is not None else None


def try_lower(host, op):
    if not ENABLED:
        return False
    schedule = plan(op)
    if schedule is None:
        return False
    result = op.results[0]
    if (any(v not in host.ptrs for v in (*op.operands, result))
        or result in getattr(host, 'aliasable_results', ())
        or result in getattr(host, 'storage_aliases', {})
        or any(host.ptrs[result] is host.ptrs[v] for v in op.operands)):
        return False
    enc = host.enc
    enc.fence()
    def emit(index, run):
        emit_raw_run(enc,host.ptrs[op.operands[index]],host.ptrs[result],run,schedule.dtype)
    schedule.emit(enc.index_builder, emit)
    enc.fence()
    host.vals[result] = host.load_arg(host.ptrs[result], result.type)
    host.materialized_results.add(result)
    if prod(schedule.result_shape):
        host.record_full_writer(result, try_lower)
    host.last_lowering = ('device', ('exact_static_raw32_concat' if schedule.dtype=='f32' else 'exact_static_integer_concat'), None)
    schedule_path = Path(__file__).parent/'compiler_library/merlin/compile/scheduling/source_concat_transfer.py'
    host.device_receipts.append(dict(source_op_index=getattr(host, 'current_source_op_index', None),
        emission=dict(schema='merlin.source_concat_transfer.v1', plan=asdict(schedule), storage_view=storage_receipt(schedule.dtype),
            resources=resource_admission(schedule),
            adapter_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            schedule_sha256=sha256(schedule_path.read_bytes()).hexdigest(),
            encoder_sha256=sha256(Path(__file__).with_name('encoding.py').read_bytes()).hexdigest(),
            observers='all complete original inputs unmodified; complete distinct result written',
            external_buffers='original full named ABI buffers; result disjoint from every input',
            completion='fence after each load and store; final continuation fence', timing='UNKNOWN')))
    return True
