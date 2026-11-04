"""Semantic dispatch from the logical program into shared target schedules."""
from __future__ import annotations

from math import prod
from .frontend import infer_tensors, parse_source
from .context import EmissionContext
from .placement import annotate_placement


def compile_buffer(buffer, *, symbol="kernel_entry"):
    cb = infer_tensors(buffer)
    for command in cb["commands"]:
        if "requant" in command.get("attributes", {}).get("epilogue", []):
            # The selected readout supports f32 scale with half-even rounding;
            # the declared integer shift stage has a different numeric contract.
            declined = {**cb, "commands": [], "declined": {
                "op": "requant", "reason": "selected target has no exact integer-shift requant readout or contraction-scoped route"}}
            return annotate_placement(declined), '"builtin.module"() ({}) : () -> ()\n'
    if cb.get("params", {}).get("frontend_entry_bindings") is not None:
        from .frontend import prepare_direct_program_abi
        cb = prepare_direct_program_abi(cb)
    elif not cb.get("kernel_abi"):
        from .frontend import declare_logical_pointer_abi
        cb = declare_logical_pointer_abi(cb)
    ctx = EmissionContext(cb, symbol)
    if cb.get("params", {}).get("address_space_boundary"):
        from .isa import flush
        ctx.fence()
        ctx.emit(*flush())
    residents, products = {}, {}
    from pathlib import Path
    from merlin.compile.tuning_knobs import knob
    shared = knob('SHARED_PRODUCTS', default='off', values=('off', 'on', 'single_panel', 'equal_partitions'), root=Path(__file__).resolve().parent.parent)
    groups, consumed = {}, set()
    from .resident_rhs import resident_groups
    resident_epochs = resident_groups(cb)
    rhs_reuse = knob('RHS_PRODUCTS', default='off', values=('off', 'on', 'row_bundle', 'row_bundle_retained', 'row_bundle_epochs', 'matrix_bundle'), root=Path(__file__).resolve().parent.parent)
    def bundled_products(members):
        if rhs_reuse == 'matrix_bundle':
            from .rhs_row_bundle_epochs import lower_rhs_row_bundle_epochs
            from .rhs_matrix_bundle import lower_matrix_bundle
            return lower_rhs_row_bundle_epochs(ctx,members) or lower_matrix_bundle(ctx,members)
        if rhs_reuse == 'row_bundle':
            from .rhs_row_bundle import lower_rhs_row_bundle
            return lower_rhs_row_bundle(ctx,members)
        if rhs_reuse == 'row_bundle_retained':
            from .rhs_row_bundle_retained import lower_rhs_row_bundle_retained
            return lower_rhs_row_bundle_retained(ctx,members)
        if rhs_reuse == 'row_bundle_epochs':
            from .rhs_row_bundle_epochs import lower_rhs_row_bundle_epochs
            return lower_rhs_row_bundle_epochs(ctx,members)
        return None
    rhs_groups = {}
    if rhs_reuse != 'off':
        from .resident_rhs import rhs_product_groups
        rhs_groups = rhs_product_groups(cb)
    if shared != 'off':
        from .shared_products import sibling_groups
        groups = {rows[0]['command_index']: rows for rows in sibling_groups(cb)}
    for command_index, command in enumerate(cb["commands"]):
        if command_index in consumed:
            continue
        op = command["opcode"]
        a = command.get("operands", {})
        attrs = command.get("attributes", {})
        ctx.current_source_index = attrs.get("merlin.source_op_index")
        if op == "RES_PACK":
            residents[a["dst"]] = residents.get(a["src"], a["src"])
        elif op == "EVICT":
            pass  # Each schedule owns its physical operand lifetimes.
        elif op in ("MATMUL", "MATMUL_RESIDENT"):
            products[a["dst"]] = (a["lhs"], residents.get(a["rhs"], a["rhs"]), attrs)
        elif op == "COMMIT":
            if command_index in resident_epochs and not any(row['command_index'] in consumed for row in resident_epochs[command_index]):
                from .resident_rhs import lower_resident_products
                receipt = bundled_products(resident_epochs[command_index])
                if receipt is None:
                    receipt = lower_resident_products(ctx, resident_epochs[command_index])
                if receipt:
                    consumed.update(row['command_index'] for row in resident_epochs[command_index])
                    ctx.metadata.setdefault('resident_rhs', []).append(receipt)
                    continue
            if command_index in rhs_groups and not any(row['command_index'] in consumed for row in rhs_groups[command_index]):
                from .resident_rhs import lower_resident_products
                receipt = bundled_products(rhs_groups[command_index])
                if receipt is None and rhs_reuse == 'on':
                    receipt = lower_resident_products(ctx, rhs_groups[command_index])
                if receipt:
                    consumed.update(row['command_index'] for row in rhs_groups[command_index])
                    ctx.metadata.setdefault('rhs_products', []).append(receipt)
                    continue
            if command_index in groups and not any(row['command_index'] in consumed for row in groups[command_index]):
                from .shared_products import lower_shared_products
                receipt = lower_shared_products(ctx, groups[command_index], single_panel_only=shared == 'single_panel', equal_operand_partitions=shared == 'equal_partitions')
                if receipt:
                    consumed.update(row['command_index'] for row in groups[command_index])
                    ctx.metadata.setdefault('shared_products', []).append(receipt)
                    continue
            if a["src"] not in products:
                raise ValueError(f"commit reads unlowered accumulator {a['src']}")
            lhs, rhs, product_attrs = products[a["src"]]
            from .dense import lower_dense
            ash, bsh = ctx.spec(lhs)["shape"], ctx.spec(rhs)["shape"]
            combined = {**product_attrs, **attrs}
            dest, producer_attrs = _pool_target(ctx, a["dst"], [ash[-2], bsh[-1]], combined)
            lower_dense(ctx, lhs, rhs, dest, m=ash[-2], k=ash[-1], n=bsh[-1], attributes=producer_attrs)
            if dest != a["dst"]:
                from .pooling import lower_pool
                lower_pool(ctx, dest, a["dst"], combined)
        elif op == "CONV2D":
            from .conv import lower_conv
            fused_pool = knob('CONV_POOL_FUSION', default='off', values=('off', 'on'), root=Path(__file__).resolve().parent.parent)
            if fused_pool == 'on' and 'maxpool' in attrs.get('epilogue', []):
                from .conv_pool import lower_conv_pool
                pool_schedule = knob('CONV_POOL_SCHEDULE', default='seed', values=('seed', 'resident_clipped'), root=Path(__file__).resolve().parent.parent)
                pool_attrs = dict(attrs, conv_pool_schedule=pool_schedule)
                if lower_conv_pool(ctx, a['ifm'], residents.get(a['weight'], a['weight']), a['dst'], pool_attrs):
                    continue
            shape = ctx.spec(a["ifm"])["shape"]
            dest, producer_attrs = _pool_target(ctx, a["dst"], [shape[0] * prod(attrs.get("pool_in_dims", [1,1])), attrs["kernel"][-1]], attrs)
            rhs = residents.get(a["weight"], a["weight"])
            spatial = knob('CONV_SPATIAL_PANELS', default='off', values=('off', 'on', 'full_channels', 'full_spatial', 'full_spatial_stores', 'regional_affine', 'transfer_runs', 'transfer_regional', 'transfer_affine_all', 'transfer_affine_no_stores', 'prepared_taps'), root=Path(__file__).resolve().parent.parent)
            # Complete-call comparisons favor flattened rows for 3x3 kernels
            # with at least one full input-channel tile. Short reductions and
            # large windows keep their separately tested incumbent schedules.
            from .isa import DIM
            spatial_family = (tuple(producer_attrs['kernel'][:2]) == (3, 3)
                              and producer_attrs['kernel'][2] >= DIM)
            m_groups = knob('CONV_M_GROUPS', default='off', values=('off', 'on', 'regional', 'regional_stride1'), root=Path(__file__).resolve().parent.parent)
            if m_groups in ('regional', 'regional_stride1') and spatial_family and spatial == 'transfer_regional':
                from .conv_m_group_policy import lower_regional_m_groups
                group_option='stride1_resource' if m_groups=='regional_stride1' else 'resource'
                if lower_regional_m_groups(ctx, a['ifm'], rhs, dest, producer_attrs, option=group_option):
                    if dest != a['dst']:
                        from .pooling import lower_pool
                        lower_pool(ctx, dest, a['dst'], attrs)
                    continue
            if m_groups == 'on' and spatial_family:
                from .conv_weight_panels import lower_weight_panels
                if lower_weight_panels(ctx, a['ifm'], rhs, dest, producer_attrs):
                    if dest != a['dst']:
                        from .pooling import lower_pool
                        lower_pool(ctx, dest, a['dst'], attrs)
                    continue
            if spatial == 'prepared_taps' and spatial_family:
                from .conv_row_recurrence import lower_row_recurrence
                if lower_row_recurrence(ctx,a['ifm'],rhs,dest,producer_attrs):
                    if dest != a['dst']:
                        from .pooling import lower_pool
                        lower_pool(ctx,dest,a['dst'],attrs)
                    continue
                spatial = 'full_spatial_stores'
            if spatial in ('transfer_regional','transfer_affine_all','transfer_affine_no_stores') and spatial_family:
                from .conv_transfer_regional import lower_transfer_regional
                option={'transfer_regional':'affine_uniform_stores',
                        'transfer_affine_all':'all_affine_uniform_stores',
                        'transfer_affine_no_stores':'affine_origins'}[spatial]
                if lower_transfer_regional(ctx,a['ifm'],rhs,dest,producer_attrs,option=option):
                    if dest != a['dst']:
                        from .pooling import lower_pool
                        lower_pool(ctx,dest,a['dst'],attrs)
                    continue
                spatial = 'full_spatial_stores'
            if spatial == 'transfer_runs' and spatial_family:
                from .conv_transfer_runs import lower_transfer_runs
                if lower_transfer_runs(ctx,a['ifm'],rhs,dest,producer_attrs):
                    if dest != a['dst']:
                        from .pooling import lower_pool
                        lower_pool(ctx,dest,a['dst'],attrs)
                    continue
                spatial = 'full_spatial_stores'
            if spatial == 'regional_affine' and spatial_family:
                from .conv_regional import lower_conv_regional
                if lower_conv_regional(ctx,a['ifm'],rhs,dest,producer_attrs,
                        option='provisional_affine_fragmented',allow_wide=True):
                    if dest != a['dst']:
                        from .pooling import lower_pool
                        lower_pool(ctx,dest,a['dst'],attrs)
                    continue
            if spatial == 'on' or spatial in ('full_channels', 'full_spatial', 'full_spatial_stores') and spatial_family:
                from .conv_spatial_panels import lower as lower_spatial_panels
                if lower_spatial_panels(ctx, a["ifm"], rhs, dest, producer_attrs, allow_wide=spatial in ('full_spatial', 'full_spatial_stores'),
                        trial_mode='uniform_stores' if spatial == 'full_spatial_stores' else None):
                    if dest != a["dst"]:
                        from .pooling import lower_pool
                        lower_pool(ctx, dest, a["dst"], attrs)
                    continue
            channel_phases = knob('CONV_CHANNEL_PHASES', default='off', values=('off', 'on'), root=Path(__file__).resolve().parent.parent)
            if channel_phases == 'on':
                from .conv_channel_phases import lower as lower_channel_phases
                if lower_channel_phases(ctx, a["ifm"], rhs, dest, producer_attrs):
                    if dest != a["dst"]:
                        from .pooling import lower_pool
                        lower_pool(ctx, dest, a["dst"], attrs)
                    continue
            horizontal = knob('CONV_HORIZONTAL_REDUCTION', default='off', values=('off', 'on'), root=Path(__file__).resolve().parent.parent)
            if horizontal == 'on':
                from .conv_flattened_rows import can_lower as can_lower_horizontal, lower as lower_horizontal
                if can_lower_horizontal(ctx, a["ifm"], rhs, dest, producer_attrs):
                    receipt = lower_horizontal(ctx, a["ifm"], rhs, dest, producer_attrs)
                    if receipt:
                        ctx.metadata.setdefault('horizontal_convolution', []).append(receipt)
                        if dest != a["dst"]:
                            from .pooling import lower_pool
                            lower_pool(ctx, dest, a["dst"], attrs)
                        continue
            pointwise = knob('POINTWISE_DENSE', default='off', values=('off', 'on'), root=Path(__file__).resolve().parent.parent)
            strided_pointwise = knob('POINTWISE_STRIDED_DENSE', default='off', values=('off', 'on'), root=Path(__file__).resolve().parent.parent)
            if strided_pointwise == 'on':
                from .pointwise import can_lower_strided_pointwise, lower_strided_pointwise
                if can_lower_strided_pointwise(ctx, a["ifm"], rhs, dest, producer_attrs):
                    producer_attrs = dict(producer_attrs)
                    producer_attrs['strided_pointwise_store_epochs'] = knob('STRIDED_POINTWISE_STORE_EPOCHS', default='off', values=('off', 'on'), root=Path(__file__).resolve().parent.parent)
                    lower_strided_pointwise(ctx, a["ifm"], rhs, dest, producer_attrs)
                    if dest != a["dst"]:
                        from .pooling import lower_pool
                        lower_pool(ctx, dest, a["dst"], attrs)
                    continue
            if pointwise == 'on':
                from .pointwise import can_lower_pointwise, lower_pointwise
                if can_lower_pointwise(ctx, a["ifm"], rhs, dest, producer_attrs):
                    lower_pointwise(ctx, a["ifm"], rhs, dest, producer_attrs)
                else:
                    lower_conv(ctx, a["ifm"], rhs, dest, producer_attrs)
            else:
                lower_conv(ctx, a["ifm"], rhs, dest, producer_attrs)
            if dest != a["dst"]:
                from .pooling import lower_pool
                lower_pool(ctx, dest, a["dst"], attrs)
        elif op == "MOVEMENT":
            from .elementwise import lower_movement
            lower_movement(ctx, a, attrs)
        elif op == "ACCUMULATOR_READOUT":
            from .elementwise import lower_readout
            lower_readout(ctx, a, attrs)
        elif op == "INSERT_COPY":
            from .source_insert_copy import lower_insert_copy
            ctx.metadata.setdefault('insert_copies', []).append(lower_insert_copy(ctx, a, attrs))
        elif op == "CONCAT_COPY":
            from .source_concat_copy import lower_concat_copy
            ctx.metadata.setdefault('concat_copies', []).append(lower_concat_copy(ctx, a, attrs))
        elif op == "SLICE_COPY":
            from .source_slice_copy import lower_slice_copy
            ctx.metadata.setdefault('slice_copies', []).append(lower_slice_copy(ctx, a, attrs))
        elif op == "STRIDED_MOVEMENT":
            from .transpose import lower_transpose
            lower_transpose(ctx, a, attrs)
        elif op == "RESIDUAL_ADD":
            from .elementwise import lower_residual
            lower_residual(ctx, a, attrs)
        elif op == "MAXPOOL2D":
            from .pooling import lower_pool
            lower_pool(ctx, a["src"], a["dst"], attrs)
        elif op == "BIAS_ADD":
            from .elementwise import lower_bias
            lower_bias(ctx, a, attrs)
        elif op in ("ATTENTION_QK", "ATTENTION_PV"):
            from .attention import lower_qk, lower_pv
            (lower_qk if op == "ATTENTION_QK" else lower_pv)(ctx, a, attrs)
        elif op == "BATCHED_MATMUL":
            from .source_head_batches import lower_head_batches
            lower_head_batches(ctx, a, attrs)
        else:
            raise ValueError(f"target lowering not implemented for {op}")
    artifact = ctx.finish()
    if ctx.metadata:
        from .emission_metadata import deterministic_compiler_metadata
        cb.setdefault('params', {})['compiler_lowering'] = deterministic_compiler_metadata(ctx.metadata)
    if cb.get("params", {}).get("frontend_entry_bindings") is not None:
        from .frontend import build_direct_program_plan
        cb = build_direct_program_plan(cb["params"].get("frontend_source", ""), cb, schedule_instruction_count=len(ctx.emitted)) if cb["params"].get("frontend_source") else cb
    return annotate_placement(cb), artifact


def compile_source(text, *, symbol="kernel_entry"):
    return compile_buffer(parse_source(text), symbol=symbol)


def _pool_target(ctx, dst, shape, attributes):
    if "maxpool" not in attributes.get("epilogue", []):
        return dst, attributes
    if ctx.spec(dst)["dtype"] != "i8":
        raise ValueError("selected device pooling requires narrowed i8 readout")
    name = "_unpooled_" + dst + "_" + str(len(ctx.tensors))
    ctx.alloc(name, shape, "i8")
    attrs = dict(attributes)
    attrs["epilogue"] = [e for e in attributes["epilogue"] if e != "maxpool"]
    return name, attrs
