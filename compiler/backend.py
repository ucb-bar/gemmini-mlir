"""General logical-ABI orchestration for the authored Gemmini backend.

The command graph retains source tensors and ordered readout semantics. Family
adapters share one terminal encoder and never delegate device arithmetic to CPU.
"""
from __future__ import annotations

import copy
import hashlib
import json
from math import prod
from pathlib import Path

from frontend import parse_source
from isa import FunctionBuilder
from merlin.compile.logical_abi import logical_abi, logical_interface

ROOT = Path(__file__).resolve().parent


def read_program(source: str) -> dict:
    cb = parse_source(source)
    if cb.get("level") == "linalg-on-tensors":
        from host_lowering import needs_host, lower_host_source
        from linalg_lowering import lower_linalg
        from source_attribution import prepare_source_bindings
        if needs_host(source):
            def device_lower(text, *, symbol):
                part = lower_linalg(text)
                prepare_source_bindings(text, part)
                return part, emit_llvm(part, symbol=symbol)
            cb, _ = lower_host_source(source, device_lower=device_lower)
        else:
            from shared_rhs_grouping import setting, lower_owned_source
            packed = lower_owned_source(source) if setting() == 'region_local' else None
            if packed is None:
                cb = lower_linalg(source, cb)
                prepare_source_bindings(source, cb)
            else:
                cb, packing_owner = packed
                packing_owner.revalidate()
                from preparation_view_aliases import setting as view_setting, eliminate_preparation_views
                if view_setting() == 'whole_buffer_roundtrip':
                    cb, view_owner = eliminate_preparation_views(cb)
                    view_owner.revalidate()
                    cb['params']['preparation_view_elimination'] = view_owner.record()
    cb = copy.deepcopy(cb)
    if cb.get("target") not in ("gemmini", ""):
        raise ValueError("this compiler targets gemmini")
    cb["target"] = "gemmini"
    cb["backend"] = "authored-explicit-gemmini"
    cb.setdefault("params", {})["compiler_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    from interface_abi import declare_interface_abi
    cb = declare_interface_abi(cb)
    return cb


def boundary(cb):
    return logical_interface(cb, logical_abi(ROOT / "contracts" / "mlir_oot_backend_contract.yaml"))


def _attrs(command):
    return dict(command.get("attributes") or {})


def _prepare_sibling_schedules(cb):
    """Admit actual independent groups, then construct their shared source task."""
    from dense_siblings import setting, build_siblings
    from merlin.compile.tuning_knobs import knob
    materialized = knob('dense_sibling_materialized_readout', default='off',
        values=('off','raw_then_grouped_readout'), root=ROOT) != 'off'
    if setting() == 'off' and not materialized:
        return {}, set()
    from source_product_groups import discover_product_groups, coalesce_product_tasks
    from dense import _geometry, _adapter
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
    from merlin.transforms.dense_planning import DenseResources
    edge, bank_rows, operand_bytes, accumulator_bytes = _geometry()
    storage = DenseLoopStorage(DenseMemoryView('a','spad',edge,0,bank_rows),
        DenseMemoryView('b','spad',edge,bank_rows,bank_rows),
        DenseMemoryView('c','acc',edge*4,0,accumulator_bytes//(edge*4)))
    resources = DenseResources(operand_bytes,accumulator_bytes,max_m_blocks=1,
        max_k_blocks=1,max_n_blocks=1,partial_output_tiles=True,partial_output_blocks=True)
    original = discover_product_groups(cb, allow_readout=materialized)
    current = cb
    schedules, skipped, reports = {}, set(), []
    for original_group in original.groups:
        signature = original_group.commit_indices
        group = next((g for g in discover_product_groups(current, allow_readout=materialized).groups
                      if g.commit_indices == signature), None)
        if group is None:
            continue
        try:
            # Coalescing operates on a copy. Rejected capacity/alias/numeric
            # candidates leave the original graph and original task order intact.
            candidate, discovery, grouping = coalesce_product_tasks(current, (group,), allow_readout=materialized)
            refreshed = next(g for g in discovery.groups if g.commit_indices == signature)
            if materialized and all(p.attrs.get('output_dtype') == 'i8' for p in refreshed.products):
                from dense_group_readout import build_group_readout
                build = build_group_readout
            elif setting() != 'off':
                build = build_siblings
            else:
                continue
            plan = build(refreshed.products,resources,storage,_adapter(),
                source_identity=(cb['params'].get('compiler_source_sha256') or cb['params']['source_sha256']),
                external_buffers_disjoint=True)
        except ValueError as error:
            reports.append({'commits':list(signature),'applied':False,'refusal':str(error)})
            continue
        current = candidate
        schedules[refreshed.first_commit_index] = (refreshed, plan)
        skipped.update(refreshed.commit_indices)
        reports.append({'commits':list(signature),'applied':True,'task_grouping':grouping,
                        'checked_kernel_sha256':plan.kernel.digest()})
    if current is not cb:
        cb.clear()
        cb.update(current)
    cb.setdefault('params', {})['dense_sibling_admission'] = {
        'groups':reports,
        'discovery_refusals':[vars(row) for row in original.refusals],
    }
    return schedules, skipped


def _prepare_rhs_schedules(cb):
    """Group original immutable RHS consumers before target emission."""
    from merlin.compile.tuning_knobs import knob
    mode = knob('dense_shared_rhs_schedule', default='off',
                values=('off','drained','epochs','supported_geometry'), root=ROOT)
    if mode == 'off':
        return {}, set()
    from dense_siblings import setting as lhs_setting
    materialized = knob('dense_sibling_materialized_readout',default='off',
        values=('off','raw_then_grouped_readout'),root=ROOT)
    if lhs_setting() != 'off' or materialized != 'off':
        cb.setdefault('params', {})['dense_rhs_admission'] = {
            'refusal':'select one source product grouping strategy per compilation'}
        return {}, set()
    from source_product_groups import discover_rhs_product_groups, coalesce_rhs_product_tasks
    from dense_shared_rhs_policy import build_selected_rhs
    from dense import _geometry, _adapter
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
    from merlin.transforms.dense_planning import DenseResources
    edge, bank_rows, operand_bytes, accumulator_bytes = _geometry()
    storage = DenseLoopStorage(DenseMemoryView('a','spad',edge,0,bank_rows),
        DenseMemoryView('b','spad',edge,bank_rows,bank_rows),
        DenseMemoryView('c','acc',edge*4,0,accumulator_bytes//(edge*4)))
    resources = DenseResources(operand_bytes,accumulator_bytes,max_m_blocks=1,
        max_k_blocks=1,max_n_blocks=1,partial_output_tiles=True,partial_output_blocks=True)
    original = discover_rhs_product_groups(cb)
    current, schedules, skipped, reports = cb, {}, set(), []
    for original_group in original.groups:
        signature = original_group.commit_indices
        group = next((g for g in discover_rhs_product_groups(current).groups
                      if g.commit_indices == signature), None)
        if group is None:
            continue
        try:
            candidate, discovery, grouping = coalesce_rhs_product_tasks(current, (group,))
            refreshed = next(g for g in discovery.groups if g.commit_indices == signature)
            plan, decision = build_selected_rhs(mode,refreshed.products,resources,storage,_adapter(),
                source_identity=(cb['params'].get('compiler_source_sha256') or cb['params']['source_sha256']),
                external_buffers_disjoint=True)
            if plan is None:
                reports.append({'commits':list(signature),'applied':False,'policy':decision})
                continue
        except ValueError as error:
            reports.append({'commits':list(signature),'applied':False,'refusal':str(error)})
            continue
        current = candidate
        schedules[refreshed.first_commit_index] = (refreshed, plan)
        skipped.update(refreshed.commit_indices)
        reports.append({'commits':list(signature),'applied':True,'task_grouping':grouping,
                        'checked_kernel_sha256':plan.kernel.digest()})
    if current is not cb:
        cb.clear()
        cb.update(current)
    cb.setdefault('params', {})['dense_rhs_admission'] = {
        'groups':reports,'discovery_refusals':[vars(row) for row in original.refusals]}
    return schedules, skipped


def emit_llvm(cb: dict, *, symbol="kernel_entry", source=None) -> str:
    from tool_runtime import ensure_z3
    ensure_z3()
    if cb.get("params", {}).get("_host_llvm"):
        product = cb["params"]["_host_llvm"]
        if source is not None:
            from attributed_emit import attach_emission_plan
            attach_emission_plan(source, cb, product)
            from placement_diagnostics import attach_placement_report
            attach_placement_report(source, cb)
            from source_profile import attach_source_profile
            attach_source_profile(source, cb, placement_report=cb["params"].get("source_placement_diagnostics"))
            from model_profile import attach_model_profile
            attach_model_profile(source, cb)
        return product
    original_rows_owner = None
    from original_row_batching import setting as original_rows_setting
    from original_row_policy import prepare as prepare_original_rows
    if source is not None and original_rows_setting() == 'separate_rows':
        packed_rows = prepare_original_rows(source, cb)
        if packed_rows is not None:
            rows_cb, original_rows_owner = packed_rows
            cb.clear(); cb.update(rows_cb)
            original_rows_owner.cb = cb
            cb['params']['original_row_batching'] = original_rows_owner.record()
    original_rows_first = {} if original_rows_owner is None else {g['commits'][0]: g for g in original_rows_owner.groups}
    original_rows_all = set() if original_rows_owner is None else {i for g in original_rows_owner.groups for i in g['commits']}
    from dense import lower_dense, lower_batched_dense
    from dense_residency import invalidate_residency
    sibling_schedules, sibling_commits = _prepare_sibling_schedules(cb)
    rhs_schedules, rhs_commits = _prepare_rhs_schedules(cb)
    from dense_resident_policy import resident_source_contracts
    resident_contracts = resident_source_contracts(cb, external_buffers_disjoint=True)
    arguments = boundary(cb)
    b = FunctionBuilder([f"arg{i}" for i in range(len(arguments))], symbol=symbol)
    ptrs = {arg.name: b.args[f"arg{i}"] for i, arg in enumerate(arguments)}
    tensors = cb.get("tensors", {})
    for arg in arguments:
        tensors.setdefault(arg.name, {"shape": list(arg.shape), "dtype": arg.dtype,
                                      "role": "output" if arg.kind == "output" else "input"})

    def pointer(name):
        if name not in ptrs:
            spec = tensors.get(name)
            if spec is None:
                raise ValueError(f"no tensor declaration for {name!r}")
            ty = spec["dtype"]
            if ty not in ("i8", "i16", "i32", "i64", "f32"):
                raise ValueError(f"unsupported temporary dtype {ty}")
            # Private intermediate storage is allocation, never a logical input.
            ptrs[name] = b.global_buffer(prod(spec["shape"]), ty=ty)
        return ptrs[name]

    resident_use_counts = {}
    for command in cb.get("commands", []):
        if command["opcode"] in ("MATMUL_RESIDENT", "MATMUL"):
            rhs = (command.get("operands") or {}).get("rhs")
            resident_use_counts[rhs] = resident_use_counts.get(rhs, 0) + 1
    residents = {}
    accumulators = {}
    transports = cb.get("params", {}).get("source_transport_tasks", [])

    def emit_transports(before_command_index):
        from linalg_lowering import emit_source_transport
        for record in transports:
            if record["before_command_index"] != before_command_index:
                continue
            invalidate_residency(b, reason="source transport may write private scratchpad")
            b.task_index = record["source_task_index"]
            owned = record["source_op_indices"]
            b.source_op_index = record.get("source_primary_op_index", owned[0])
            b.region_id = record.get("prov.region_id")
            for recipe in record.get("copies", []):
                pointer(recipe["src"])
                pointer(recipe["dst"])
            if "fill" in record:
                pointer(record["fill"]["dst"])
            emit_source_transport(b, cb, record, ptrs)

    b.task_index = -1 if cb.get("params", {}).get("tensor_source_map") else None
    b.fence()
    b.flush()
    for command_index, command in enumerate(cb.get("commands", [])):
        emit_transports(command_index)
        op = command["opcode"]
        operands = command.get("operands") or {}
        attrs = _attrs(command)
        attrs.update(cb.get("params", {}).get("command_source_attribution", {}).get(str(command_index), {}))
        b.task_index = attrs.get("source_task_index")
        owned = attrs.get("source_op_indices") or []
        b.source_op_index = attrs.get("source_primary_op_index", owned[0] if owned else None)
        b.region_id = attrs.get("prov.region_id")
        if op == "RES_PACK":
            invalidate_residency(b, key=operands["dst"], reason="new resident lifetime")
            residents[operands["dst"]] = residents.get(operands["src"], operands["src"])
        elif op == "EVICT":
            invalidate_residency(b, key=operands.get("handle"), reason="source EVICT")
            residents.pop(operands.get("handle"), None)
        elif op in ("MATMUL_RESIDENT", "MATMUL"):
            if command_index in resident_contracts:
                attrs['resident_source_contract'] = resident_contracts[command_index]
            accumulators[operands["dst"]] = (operands["lhs"], residents.get(operands["rhs"], operands["rhs"]), attrs,
                operands["rhs"] if operands["rhs"] in residents and resident_use_counts.get(operands["rhs"], 0) > 1 else None)
        elif op == "COMMIT":
            if command_index in original_rows_first:
                from original_row_transport import emit as emit_original_rows
                group = original_rows_first[command_index]
                emit_original_rows(b, original_rows_owner, group, {name:pointer(name) for name in group['lhs']+[group['rhs']]+group['outputs']})
                continue
            if command_index in original_rows_all:
                continue
            if command_index in rhs_schedules:
                group, plan = rhs_schedules[command_index]
                group_pointers = {'rhs':pointer(group.members[0].rhs)}
                for i, member in enumerate(group.members):
                    group_pointers['lhs'+str(i)] = pointer(member.lhs)
                    group_pointers['output'+str(i)] = pointer(member.output)
                plan.emit(b,group_pointers)
                continue
            if command_index in rhs_commits:
                continue
            if command_index in sibling_schedules:
                from dense_siblings import emit_siblings
                group, plan = sibling_schedules[command_index]
                group_pointers = {'lhs':pointer(group.members[0].lhs)}
                for i, member in enumerate(group.members):
                    group_pointers['rhs'+str(i)] = pointer(member.rhs)
                    group_pointers['output'+str(i)] = pointer(member.output)
                if hasattr(plan, 'raw_plan'):
                    from dense_group_readout import emit_group_readout
                    emit_group_readout(b,plan,group_pointers)
                else:
                    emit_siblings(b,plan,group_pointers)
                continue
            if command_index in sibling_commits:
                continue
            lhs, rhs, product_attrs, resident_key = accumulators[operands["src"]]
            shape_a, shape_w = tensors[lhs]["shape"], tensors[rhs]["shape"]
            m, k = shape_a[-2:]
            transpose = bool(product_attrs.get("rhs_transposed", False))
            n = shape_w[-2] if transpose else shape_w[-1]
            expected_k = shape_w[-1] if transpose else shape_w[-2]
            if k != expected_k:
                raise ValueError("contraction dimensions disagree")
            options = {**product_attrs, **attrs}
            bias = operands.get("bias") or attrs.get("bias")
            bias_pointer = pointer(bias) if bias else None
            if "source_epilogue_bias_values" in options:
                if bias_pointer is not None:
                    raise ValueError("source epilogue declares two bias initializers")
                values = options["source_epilogue_bias_values"]
                if not isinstance(values, list) or len(values) != n or any(type(v) is not int or not -(1 << 31) <= v < (1 << 31) for v in values):
                    raise ValueError("source epilogue bias must contain one signed i32 per output column")
                bias_pointer = b.constant_i32_table(values)
            lower_dense(b, pointer(lhs), pointer(rhs), pointer(operands["dst"]), m, k, n,
                        attrs=options, bias_ptr=bias_pointer,
                        rhs_transposed=transpose, resident_key=resident_key)
        elif op == "CONV2D":
            invalidate_residency(b, reason="convolution scratchpad lifetime")
            from column_producer import lower_convolution
            for name in operands.values():
                if name in tensors:
                    pointer(name)
            for name in residents.values():
                pointer(name)
            if attrs.get("bias"):
                pointer(attrs["bias"])
            lower_convolution(b, command, tensors, ptrs, residents=residents)
        elif op in ("BATCHED_MATMUL", "ATTENTION_QK", "ATTENTION_PV"):
            lhs = operands.get("a") or operands.get("q") or operands.get("p")
            rhs = operands.get("w") or operands.get("k") or operands.get("v")
            rhs = residents.get(rhs, rhs)
            sa, sw = tensors[lhs]["shape"], tensors[rhs]["shape"]
            transpose = op == "ATTENTION_QK" or attrs.get("rhs_transposed", False)
            m, k = sa[-2:]
            n = sw[-2] if transpose else sw[-1]
            batch = prod(sa[:-2])
            lower_batched_dense(b, pointer(lhs), pointer(rhs), pointer(operands["dst"]), batch,m,k,n,
                                attrs=attrs, rhs_transposed=transpose)
        elif op == "ROW_MAJOR_VIEW":
            if operands["src"] != operands["dst"]:
                raise ValueError("ROW_MAJOR_VIEW must retain its physical buffer")
        else:
            invalidate_residency(b, reason="elementwise scratchpad lifetime")
            from elementwise import lower_elementwise
            for name in operands.values():
                if name in tensors:
                    pointer(name)
            if op == 'RESIDUAL_ADD':
                from merlin.compile.tuning_knobs import knob
                representation = knob('residual_identity_readout',default='off',
                    values=('off','double_accumulator'),root=ROOT)
                if representation == 'double_accumulator':
                    from identity_readout import emit as emit_identity_readout
                    realization = emit_identity_readout(b,command,tensors,ptrs)
                    if realization['admitted']:
                        cb.setdefault('params',{}).setdefault('identity_readout_receipts',[]).append(realization)
                        continue
            if op == 'MOVEMENT':
                from movement_tiles import setting as movement_setting, lower_movement_tiled
                if movement_setting() == 'contiguous_bytes' and lower_movement_tiled(b, cb, command, ptrs):
                    continue
            from merlin.compile.tuning_knobs import knob
            elementwise_tiles = knob('elementwise_dma_tiles',default='off',values=('off','wide64','small_residual'),root=ROOT)
            if elementwise_tiles != 'off':
                from elementwise_tiles import lower_elementwise_tiled,lower_small_residual
                lower_tiles = lower_small_residual if elementwise_tiles == 'small_residual' else lower_elementwise_tiled
                if lower_tiles(b, cb, command, ptrs):
                    continue
            if not lower_elementwise(b, cb, command, ptrs):
                raise ValueError(f"missing target adapter for {op}")
    emit_transports(len(cb.get("commands", [])))
    b.task_index = -2 if cb.get("params", {}).get("tensor_source_map") else None
    b.source_op_index = None
    b.fence()
    product = b.finish(verify=True)
    receipts = getattr(b, "dense_schedule_receipts", None)
    if receipts:
        cb.setdefault("params", {})["dense_schedule_receipts"] = receipts
    configuration_receipt = getattr(b, "configuration_cse_receipt", None)
    if configuration_receipt and configuration_receipt.get("policy") != "off":
        cb.setdefault("params", {})["configuration_cse_receipt"] = configuration_receipt
    dispatch_receipts = getattr(b, "literal_dispatch_receipts", None)
    if dispatch_receipts:
        cb.setdefault("params", {})["literal_dispatch_receipts"] = dispatch_receipts
    column_receipts = getattr(b, "column_producer_receipts", None)
    if column_receipts:
        cb.setdefault("params", {})["column_producer_receipts"] = column_receipts
    sibling_receipts = getattr(b, "dense_sibling_receipts", None)
    if sibling_receipts:
        cb.setdefault("params", {})["dense_sibling_receipts"] = sibling_receipts
    rhs_receipts = getattr(b, "dense_rhs_epoch_receipts", None)
    if rhs_receipts:
        cb.setdefault("params", {})["dense_rhs_epoch_receipts"] = rhs_receipts
    pool_receipts = getattr(b, "conv_pool_materialization_receipts", None)
    if pool_receipts:
        cb.setdefault("params", {})["conv_pool_materialization_receipts"] = pool_receipts
    for name in ('conv_rolling_binding_receipts','conv_rolling_binding_refusals','conv_configuration_receipts','conv_configuration_refusals','conv_lhs_configuration_receipts','conv_lhs_configuration_refusals','conv_channel_pipeline_receipts','pointwise_convolution_receipts','dense_group_readout_receipts','conv_pool_selection_receipts','dense_resident_source_receipts','dense_shared_rhs_policy_receipts','conv_pool_panel_receipts','target_storage_receipts','elementwise_tile_receipts','movement_tile_receipts'):
        if getattr(b,name,None):
            cb.setdefault('params',{})[name] = getattr(b,name)
    if source is not None and cb.get("params", {}).get("tensor_source_map"):
        from attributed_emit import attach_emission_plan
        attach_emission_plan(source, cb, product)
        from placement_diagnostics import attach_placement_report
        attach_placement_report(source, cb)
        from source_profile import attach_source_profile
        attach_source_profile(source, cb, placement_report=cb["params"].get("source_placement_diagnostics"))
        from model_profile import attach_model_profile
        attach_model_profile(source, cb)
    return product


def target_ir(cb: dict) -> str:
    """A typed xDSL schedule inventory preserves the graph before emission."""
    from target_dialect import schedule_module
    from xdsl.printer import Printer
    from io import StringIO
    module = schedule_module(cb)
    module.verify()
    out = StringIO()
    Printer(stream=out).print_op(module)
    return out.getvalue() + "\n"


def validate_cb(cb):
    import jsonschema
    schema = json.loads((ROOT / "contracts" / "command_buffer.schema.json").read_text())
    jsonschema.Draft7Validator(schema).validate(cb)
    boundary(cb)
