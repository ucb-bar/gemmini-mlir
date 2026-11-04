"""Exact native i64 transpose through adjacent raw32 word pairs."""
from dataclasses import asdict
from math import prod
from xdsl.dialects.builtin import TensorType, NoneAttr, DenseArrayBase
from merlin.common.mlir_query import op_name
from merlin.compile.scheduling.source_word_layout_transfer import SourceWordLayoutTransfer
from merlin.compile.scheduling.layout_transfer import LayoutTransfer
from linalg_device_layout import resource_admission as base_admission
from merlin.llvmlower.host_tensor_codegen import tensor_shape

from raw_storage_transfer import emit_raw_run, storage_receipt, raw_storage_admission

ENABLED = True


def resource_admission(schedule, facts=None):
    if facts is None:
        from encoding import FACTS
        facts=FACTS
    raw=raw_storage_admission('f32',facts)
    # Reuse all existing selected geometry checks on a conservative full-edge
    # surrogate, then bound the actual native 8-byte pitch separately.
    resources=base_admission(LayoutTransfer(schedule.source_shape,schedule.permutation,'i32',16),facts)
    if raw is None or resources is None:
        return None
    stride=max(16,prod(schedule.source_shape[schedule.permutation[-1]+1:]))*8
    if stride >= 1 << resources['conservative_stride_bits']:
        return None
    return dict(resources,maximum_stride_bytes=stride,raw_word_preservation=raw,
        logical_tile=8,maximum_raw_columns=16,words_per_element=2,
        word_order='same increasing byte-address order; endian-independent',payload_arithmetic=False)


def plan(op):
    """Prove the complete original transpose, including its otherwise-unused seed."""
    if (op_name(op) != 'linalg.transpose' or len(op.operands) != 2
        or len(op.results) != 1 or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1 or op.successors
        or any(not k.startswith('prov.') for k in op.attributes)
        or set(op.properties)-{'permutation','operandSegmentSizes'}):
        return None
    values = (*op.operands, *op.results)
    if any(not isinstance(v.type, TensorType) or not isinstance(v.type.encoding, NoneAttr) for v in values):
        return None
    source, seed, result = values
    dtype = str(source.type.element_type)
    if dtype != 'i64' or any(str(v.type.element_type) != dtype for v in values):
        return None
    perm = op.properties.get('permutation')
    if not isinstance(perm, DenseArrayBase) or str(perm.elt_type) != 'i64':
        return None
    segment = op.properties.get('operandSegmentSizes')
    if segment is not None and (not isinstance(segment, DenseArrayBase) or str(segment.elt_type) != 'i32' or tuple(segment.iter_values()) != (1,1)):
        return None
    block = op.regions[0].block
    body = tuple(block.ops)
    if (len(block.args) != 2 or any(str(a.type) != dtype for a in block.args)
        or len(body) != 1 or op_name(body[0]) != 'linalg.yield'
        or tuple(body[0].operands) != (block.args[0],) or block.args[1].uses
        or body[0].properties or body[0].regions or body[0].successors
        or any(not k.startswith('prov.') for k in body[0].attributes)):
        return None
    try:
        result_plan = SourceWordLayoutTransfer(tuple(tensor_shape(source.type)), tuple(perm.iter_values()), dtype, 8)
    except (ValueError, TypeError):
        return None
    if tuple(tensor_shape(seed.type)) != result_plan.output_shape or tuple(tensor_shape(result.type)) != result_plan.output_shape:
        return None
    if resource_admission(result_plan) is None:
        return None
    return result_plan


def try_lower(host, op):
    if not ENABLED:
        return False
    schedule = plan(op)
    if schedule is None or any(v not in host.ptrs for v in (*op.operands, *op.results)):
        return False
    source, seed = op.operands
    result = op.results[0]
    # Existing logical ABI gives each named full tensor its own buffer. Views
    # may reuse pointers: never write through either original observer's pointer.
    if (result in getattr(host,'aliasable_results',()) or result in getattr(host,'storage_aliases',{})
        or any(v is not result and p is host.ptrs[result] for v,p in host.ptrs.items())):
        return False
    enc = host.enc
    def emit(run):
        emit_raw_run(enc,host.ptrs[source],host.ptrs[result],run,'i32')
    schedule.emit(enc.index_builder, emit)
    host.vals[result] = host.load_arg(host.ptrs[result], result.type)
    host.materialized_results.add(result)
    host.record_full_writer(result, try_lower)
    host.last_lowering = ('device','exact_static_i64_word_transpose',None)
    host.device_receipts.append(dict(source_op_index=getattr(host,'current_source_op_index',None),
        emission=dict(schema='merlin.source_word_layout_transfer.v1', plan=asdict(schedule), storage_view=storage_receipt('i32'), resources=resource_admission(schedule),
            observers='input and seed unmodified; full result written',
            external_buffers='original full named ABI buffers; output disjoint from all known pointers and alias records',
            proof='original element permutation bijection expanded to two adjacent words with the same word order',
            completion='fence after each load and store', timing='UNKNOWN')))
    return True
