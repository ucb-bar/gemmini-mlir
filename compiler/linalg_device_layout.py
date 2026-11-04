"""Optional exact source transpose through synchronous accelerator transfers."""
from dataclasses import asdict
from math import prod
import hashlib
import json
from xdsl.dialects.builtin import TensorType, NoneAttr, DenseArrayBase
from merlin.common.mlir_query import op_name
from merlin.compile.scheduling.layout_transfer import LayoutTransfer
from merlin.llvmlower.host_tensor_codegen import tensor_shape

from raw_storage_transfer import emit_raw_run, storage_receipt, raw_storage_admission

ENABLED = True
F32_ENABLED = True


def resource_admission(schedule, facts=None):
    """Pin the selected edge16/raw ACC encoding and conservatively bound pitches.

    The min stride slot is a conservative bound shared by the selected load
    configuration and output configuration. It is not a claim that the load
    external rs2 register has only sixteen bits. Unknown geometry refuses.
    """
    if facts is None:
        from encoding import FACTS
        facts = FACTS
    raw_law = raw_storage_admission(schedule.dtype,facts)
    if raw_law is None:
        return None
    try:
        mesh = next(x for x in facts['arrays'] if x['name'] == 'mesh')
        acc = next(x for x in facts['memory_address_spaces'] if x['name'] == 'accumulator')
        bundles = next(x['bundles'] for x in facts['interfaces'] if x['name'] == 'register_bundle_layouts')
        def bits(bundle, field):
            spec = bundles[bundle]['fields'][field]
            width = spec.get('width') or spec.get('slot_width')
            if type(width) is not int or width <= 0:
                raise ValueError('unknown selected field width')
            return width
        if (mesh['rows'] != 16 or mesh['cols'] != 16
            or acc['lanes'] != 16 or acc['lane_bits'] != 32
            or acc['row_bytes'] != 64 or acc['elem_bits'] != 32
            or acc['banks'] != 2 or acc['depth'] != 512
            or acc['bytes'] != acc['banks']*acc['depth']*acc['row_bytes']
            or schedule.tile != 16):
            return None
        for bundle in ('MvinRs2','MvoutRs2'):
            if (bits(bundle,'local_addr') != 32
                or 16 >= 1 << bits(bundle,'num_rows')
                or 16 >= 1 << bits(bundle,'num_cols')):
                return None
        stride_bits = min(bits('ConfigMvinRs1','stride'), bits('ConfigMvoutRs2','stride'))
        max_stride = max(schedule.tile, prod(schedule.source_shape[schedule.permutation[-1]+1:]))*schedule.itemsize
        if max_stride >= 1 << stride_bits:
            return None
        return dict(raw_storage_law=raw_law, edge=16, accumulator_rows=acc['banks']*acc['depth'],
            reserved_accumulator_rows=[0,16], raw_accumulator_bits=32,
            conservative_stride_bits=stride_bits, maximum_stride_bytes=max_stride,
            encoding_facts_sha256=hashlib.sha256(json.dumps(facts,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    except (KeyError, StopIteration, TypeError, ValueError):
        return None


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
    if dtype == 'f32' and not F32_ENABLED:
        return None
    if dtype not in ('i8','i32','f32') or any(str(v.type.element_type) != dtype for v in values):
        return None
    perm = op.properties.get('permutation')
    if not isinstance(perm, DenseArrayBase) or str(perm.elt_type) != 'i64':
        return None
    segment = op.properties.get('operandSegmentSizes')
    if segment is not None and (not isinstance(segment, DenseArrayBase) or tuple(segment.iter_values()) != (1,1)):
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
        result_plan = LayoutTransfer(tuple(tensor_shape(source.type)), tuple(perm.iter_values()), dtype, 16)
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
    if host.ptrs[result] is host.ptrs[source] or host.ptrs[result] is host.ptrs[seed]:
        return False
    enc = host.enc
    def emit(run):
        emit_raw_run(enc,host.ptrs[source],host.ptrs[result],run,schedule.dtype)
    schedule.emit(enc.index_builder, emit)
    host.vals[result] = host.load_arg(host.ptrs[result], result.type)
    host.materialized_results.add(result)
    host.record_full_writer(result, try_lower)
    host.last_lowering = ('device',('exact_static_raw32_transpose' if schedule.dtype=='f32' else 'exact_static_integer_transpose'),None)
    host.device_receipts.append(dict(source_op_index=getattr(host,'current_source_op_index',None),
        emission=dict(schema='merlin.layout_transfer.v1', plan=asdict(schedule), storage_view=storage_receipt(schedule.dtype), resources=resource_admission(schedule),
            observers='input and seed unmodified; full result written',
            external_buffers='original full named ABI buffers; output disjoint from input and seed',
            completion='fence after each load and store', timing='UNKNOWN')))
    return True
