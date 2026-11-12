"""Compiler-derived source placement census; not native coverage certification.

Every direct function-block operation has one identity. Scalar implementation
bodies belong to their enclosing tensor operation and never add census rows.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from math import prod

_SCAFFOLD = {'arith.constant', 'tensor.empty', 'tensor.splat', 'linalg.fill'}
_VIEWS = {'tensor.collapse_shape', 'tensor.expand_shape', 'tensor.cast'}
_MOVEMENT = _VIEWS | {'linalg.copy', 'linalg.transpose', 'tensor.extract_slice',
                     'tensor.insert_slice', 'tensor.pad', 'tensor.extract', 'tensor.insert',
                     'tensor.concat'}


def _attribute(value):
    if hasattr(value, 'data') and isinstance(value.data, (str, int, float, bool)):
        return value.data
    if hasattr(value, 'data') and isinstance(value.data, tuple):
        return [_attribute(item) for item in value.data]
    return str(value)


def _type(value):
    from xdsl.dialects.builtin import TensorType
    typ = value.type
    if isinstance(typ, TensorType):
        shape = list(typ.get_shape())
        dtype = str(typ.element_type)
        widths = {'i1': 1, 'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8,
                  'f16': 2, 'bf16': 2, 'f32': 4, 'f64': 8}
        # i1 storage is ABI-specific, rather than inferred as one byte.
        size = prod(shape) * widths[dtype] if dtype in widths and dtype != 'i1' and all(d >= 0 for d in shape) else None
        return {'type': str(typ), 'shape': shape, 'dtype': dtype, 'static_bytes': size}
    return {'type': str(typ), 'shape': None, 'dtype': str(typ), 'static_bytes': None}


def _family(op):
    from merlin.common.mlir_query import op_name
    name = op_name(op)
    if name in _SCAFFOLD:
        return 'initialization_or_literal'
    if name.startswith(('scf.', 'cf.')) or name == 'func.call':
        return 'control_or_call'
    if name in _MOVEMENT:
        return 'movement_or_view'
    body = {inner.name for inner in op.walk() if inner is not op}
    if any(n.startswith(('math.exp', 'math.log', 'math.sqrt', 'math.rsqrt', 'math.tanh', 'math.sin', 'math.cos')) for n in body | {name}):
        return 'nonlinear'
    iters = op.properties.get('iterator_types')
    reductions = iters is not None and any(getattr(getattr(v, 'data', None), 'value', None) == 'reduction' for v in iters)
    if name.startswith(('linalg.matmul', 'linalg.batch_matmul', 'linalg.conv')) or (reductions and ({'arith.muli', 'arith.addi'} <= body or {'arith.mulf', 'arith.addf'} <= body)):
        return 'contraction'
    if 'pool' in name or reductions and ('arith.maxsi' in body or 'arith.maximumf' in body or 'arith.cmpf' in body):
        return 'pooling_or_reduction'
    if name.startswith('quant_ext.') or body & {'arith.fptosi', 'arith.sitofp', 'math.roundeven'}:
        return 'readout_or_quantization'
    if reductions:
        return 'reduction'
    if name == 'linalg.generic' or name.startswith(('arith.', 'math.')):
        return 'elementwise'
    return 'other_tensor_or_source_work'


def _admission(op, *, transpose_transfers):
    """Ask actual typed adapters, never treating a pattern miss as unsupported."""
    from .source_groups import _root_inputs, _standalone_plans, _path, _spec
    from .frontend import _batched_matmul
    from merlin.common.mlir_query import op_name
    name = op_name(op)
    if name in ('linalg.generic', 'linalg.matmul'):
        try:
            operands = _batched_matmul(op) or _root_inputs(op)
            shapes = [_spec(v) for v in operands]
            path = _path()
            if (all(s['dtype'] == t for s, t in zip(shapes, ('i8', 'i8', 'i32'), strict=True))
                    and 2 <= len(shapes[0]['shape']) <= 8
                    and path.get('accumulator', {}).get('dtype') == 'i32'
                    and path.get('exits', {}).get('accumulator')):
                return {'eligibility': 'proved_by_current_adapter', 'device_form': 'raw_batched_dense' if len(shapes[0]['shape']) > 2 else 'raw_dense',
                        'capability_scope': 'standalone', 'evidence': 'exact signed scalar recurrence, tensor geometry and selected i32 accumulator exit'}
        except (ValueError, TypeError, AttributeError):
            pass
    class OneOperation:
        def walk(self):
            return iter((op,))
    try:
        plans = _standalone_plans(OneOperation(), set(), transpose_transfers=transpose_transfers)
    except (ValueError, TypeError, AttributeError) as exc:
        return {'eligibility': 'unknown', 'capability_scope': 'unresolved',
                'proof_gap': str(exc), 'reason': 'source adapter could not establish complete standalone admission'}
    if plans:
        return {'eligibility': 'proved_by_current_adapter', 'device_form': plans[0].opcode,
                'capability_scope': 'standalone_or_exact_source_composition', 'evidence': plans[0].diagnostic}
    return None


def placement_report(module, device_regions, *, transpose_transfers=False):
    """Inventory actual source identities, selected groups and unresolved work.

    ``device_regions`` must be the exact regions passed to source extraction on
    this same module, not corresponding operations in a reparsed clone. Required
    counts mean adapter-proved supported compute operations, not native policy
    certification. Unknown admission, composition and byte sizes remain unknown.
    """
    from merlin.common.mlir_query import op_name
    from .quantized_boundary import selected_boundary_forms
    regions = tuple(device_regions)
    ordinals = {op: index for index, op in enumerate(module.walk())}
    if any(member not in ordinals for region in regions for member in region.operations):
        raise ValueError('placement regions must reference operations in this exact source module')
    rows = []
    functions = [op for op in module.walk() if op.name == 'func.func' and op.body.blocks]
    top = []
    refs = {}
    for fn_index, fn in enumerate(functions):
        for block_index, block in enumerate(fn.body.blocks):
            for arg_index, value in enumerate(block.args):
                refs[value] = f'f{fn_index}.b{block_index}.arg{arg_index}'
            for op_index, op in enumerate(block.ops):
                for result_index, value in enumerate(op.results):
                    refs[value] = f'f{fn_index}.b{block_index}.op{op_index}.result{result_index}'
                if op.name != 'func.return':
                    top.append((fn_index, block_index, op_index, fn, op))
    for op, ordinal in ordinals.items():
        for region_index, region in enumerate(op.regions):
            for block_index, block in enumerate(region.blocks):
                for arg_index, value in enumerate(block.args):
                    refs.setdefault(value, f'op{ordinal}.region{region_index}.block{block_index}.arg{arg_index}')
    def value_record(value):
        ref = refs.get(value)
        if ref is None:
            owner = value.owner
            ref = f'nested_op{ordinals[owner]}.result{tuple(owner.results).index(value)}' if owner in ordinals else 'external_or_nested_block_argument'
        return {'ssa': ref, **_type(value)}
    membership = {}
    group_rows = []
    for index, region in enumerate(regions):
        for member in region.operations:
            membership.setdefault(member, []).append(index)
        try:
            proof = json.loads(region.reason)
        except (ValueError, TypeError):
            proof = {'reason': region.reason}
        if not isinstance(proof, dict):
            proof = {'reason': proof}
        member_work = [op for op in region.operations if op_name(op) not in _SCAFFOLD]
        realization = ('plural_independent_products' if proof.get('form') == 'independent_raw_shared_ssa_products'
                       else 'fused_source_composition' if len(member_work) > 1 else 'standalone_source_form')
        group_rows.append({'group_index': index, 'realization': realization, 'source_operation_indices': [ordinals.get(op) for op in region.operations],
                           'inputs': [value_record(v) for v in region.inputs], 'outputs': [value_record(v) for v in region.outputs],
                           'selected_form': proof, 'actual_selected_source_region': True})
    for fn_index, block_index, op_index, fn, op in top:
        name, family = op_name(op), _family(op)
        groups = membership.get(op, [])
        scaffold = name in _SCAFFOLD
        row = {'source_index': ordinals[op], 'source_identity': f'f{fn_index}.b{block_index}.op{op_index}',
               'function': fn.sym_name.data, 'operation': name, 'family': family,
               'census_role': 'scaffolding' if scaffold else 'compute_movement_or_control',
               'source_provenance': {k: _attribute(v) for k, v in op.attributes.items() if k.startswith(('prov.', 'merlin.source'))},
               'inputs': [value_record(v) for v in op.operands], 'outputs': [value_record(v) for v in op.results],
               'device_group_indices': groups}
        dtypes = {_type(v)['dtype'] for v in (*op.operands, *op.results)}
        has_float = any(t.startswith(('f', 'bf')) for t in dtypes)
        has_integer = any(t.startswith('i') and t[1:].isdigit() for t in dtypes)
        row['numerical_domain'] = 'mixed_float_integer' if has_float and has_integer else 'floating' if has_float else 'integer' if has_integer else 'other_or_control'
        row['scalar_body_operation_names'] = sorted({inner.name for inner in op.walk() if inner is not op})
        if name == 'arith.constant':
            row['source_literal'] = _attribute(op.properties.get('value', op.attributes.get('value')))
        if groups:
            row.update(placement='device', status='selected_source_region', eligibility='actual_selected_source_region',
                       required_acceleration=not scaffold,
                       capability_scope='actual_group_membership',
                       device_forms=[group_rows[i]['selected_form'] for i in groups],
                       member_role='initializer_or_literal' if scaffold else ('group_output_producer' if any(v in regions[i].outputs for i in groups for v in op.results) else 'fused_internal_member'))
            if not scaffold:
                standalone = _admission(op, transpose_transfers=transpose_transfers)
                row['standalone_admission'] = standalone if standalone is not None else {'eligibility': 'unknown', 'proof_gap': 'actual group source composition does not by itself prove a standalone form'}
                row['realized_device_forms'] = [group_rows[i]['realization'] for i in groups]
            if len(groups) > 1 and not scaffold:
                row.update(status='overlapping_device_compute_regions', eligibility='unknown', required_acceleration=None,
                           proof_gap='one source compute operation appears in multiple selected device regions')
        elif scaffold:
            row.update(placement='host', status='source_materialization', eligibility='not_applicable', required_acceleration=False,
                       reason='literal/allocation/initializer scaffolding; excluded from compute obligation denominator')
        else:
            admission = _admission(op, transpose_transfers=transpose_transfers)
            if admission is not None and admission['eligibility'] == 'proved_by_current_adapter':
                row.update(admission, placement='host', status='eligible_but_host', required_acceleration=True,
                           proof_gap='proved source form has no membership in the actual selected device regions',
                           reason='eligible placement/compiler adapter gap; no hardware-unsupported claim')
            else:
                types = [_type(v)['dtype'] for v in (*op.operands, *op.results)]
                floating = any(t.startswith(('f', 'bf')) for t in types)
                seam = 'host_control_or_call' if family == 'control_or_call' else 'host_floating_or_nonlinear_seam' if floating or family == 'nonlinear' else 'unresolved_integer_or_movement_form'
                row.update(placement='host', status=seam, eligibility='unknown', required_acceleration=None,
                           capability_scope='standalone_and_composition_unresolved',
                           reason='typed floating/nonlinear/control source retained by host lowering; no complete selected device composition proof' if floating or family in ('nonlinear','control_or_call') else 'typed source has no complete current standalone or fused admission proof; a matcher miss does not establish hardware unsupported',
                           proof_gap='source admission/composition remains unresolved')
                if admission is not None:
                    row['adapter_refusal'] = admission
        if family == 'control_or_call':
            row['nested_source_operation_count'] = sum(1 for inner in op.walk() if inner is not op)
            row['nested_scope'] = 'owned by this control/call row; detailed nested admission is not established'
        rows.append(row)
    work = [row for row in rows if row['census_role'] != 'scaffolding']
    counts = {'required': sum(row['required_acceleration'] is True for row in work),
              'covered': sum(row['placement'] == 'device' and row['eligibility'] != 'unknown' for row in work),
              'eligible_but_host': sum(row['status'] == 'eligible_but_host' for row in work),
              'unknown': sum(row['eligibility'] == 'unknown' for row in work),
              'total_compute_movement_control_operations': len(work), 'scaffolding_operations': len(rows)-len(work),
              'total_top_level_operations': len(rows), 'device_groups': len(regions)}
    boundaries = [value for region in group_rows for value in (*region['inputs'], *region['outputs'])]
    known = [value['static_bytes'] for value in boundaries if value['static_bytes'] is not None]
    from merlin.compiler.dialects._common import text
    from .source_groups import _path
    source_digest = hashlib.sha256(text(module, generic=True).encode()).hexdigest()
    path_digest = hashlib.sha256(json.dumps(_path(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'schema': 'submission.source_placement.v1',
            'source_ir_sha256': source_digest, 'source_digest_format': 'canonical generic MLIR text of the unchanged input module',
            'selected_output_path_sha256': path_digest, 'authority': 'compiler-derived diagnostic; noncertifying',
            'native_complete_source_demand_report': 'UNKNOWN; this census does not replace the native all-region coverage gate',
            'scope': 'each direct function-block MLIR operation except func.return exactly once; scalar implementation bodies excluded; nested control owned by enclosing row',
            'required_count_is_lower_bound': True,
            'source_identity_basis': 'top-level MLIR operation identity; provenance IDs are preserved labels and never used to merge distinct source operations',
            'required_count_basis': 'actual selected source compute or current adapter-proved standalone/source composition; unresolved source obligations remain unknown',
            'counts': counts, 'families': dict(Counter(row['family'] for row in work)),
            'numerical_domains': dict(Counter(row['numerical_domain'] for row in work)),
            'operations': rows, 'device_regions': group_rows, 'selected_boundary_forms': selected_boundary_forms(),
            'static_device_boundary_bytes': {'total': sum(known) if len(known) == len(boundaries) else None,
                'known_subtotal': sum(known), 'known_boundary_values': len(known), 'unknown_boundary_values': len(boundaries)-len(known),
                'denominator': 'sum of ordered input and output SSA tensor extents at each actual selected device callback; repeated values count once per boundary occurrence; storage aliases and physical transfers are not inferred'},
            'measured_host_time': None, 'measured_transfer_bytes': None}
