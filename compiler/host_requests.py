"""Source-bound selections for the public, independently admitted host route.

A request is neither an implementation grant nor evidence of an applied rewrite.
In particular, this release supplies no fusion-group index convention or original
host-capability implementation. Those groups remain empty. The original runtime
must independently accept every requested layout/storage premise.
"""
from hashlib import sha256
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from xdsl.dialects.builtin import TensorType
from xdsl.parser import Parser

from merlin.compile.tuning_knobs import knob
from merlin.targetgen.contract import linalg_iface as li


def build_host_request(program) -> dict | None:
    """Select a schema-v1 request, or None when disabled/ineligible/unavailable.

    Diagnostics live in ``program['params']['host_request_diagnostic']``. Schema
    or typed-source failures decline this optional selection; they never cause
    the compiler to substitute invented source indices or runtime guarantees.
    Exact ``source_text`` bytes, including whitespace, bind the request digest.
    """
    root = Path(__file__).resolve().parent
    mode = knob('host_request', default='retained',
                values=('retained', 'source_safe'), root=root)
    diagnostic = {'mode': mode, 'status': 'disabled',
                  'application_status': 'unknown',
                  'scope': 'request selection only; original runtime independently admits and applies choices'}
    program.setdefault('params', {})['host_request_diagnostic'] = diagnostic
    if mode == 'retained':
        return None
    text = program.get('source_text')
    if program.get('source_format') != 'linalg-on-tensors' or not isinstance(text, str):
        diagnostic.update(status='ineligible', reason='exact typed source is required')
        return None
    digest = sha256(text.encode('utf-8')).hexdigest()
    diagnostic['source_sha256'] = digest
    try:
        module = Parser(li.make_linalg_context(), text).parse_module()
        functions = [op for op in module.walk() if li._op_name(op) == 'func.func']
        if len(functions) != 1 or len(functions[0].regions[0].blocks) != 1:
            diagnostic.update(status='ineligible', reason='requires one source entry with one block')
            return None
        block = functions[0].regions[0].blocks[0]
        operations = list(block.ops)
        if not operations or li._op_name(operations[-1]) != 'func.return':
            diagnostic.update(status='ineligible', reason='source entry must terminate with func.return')
            return None
        # Static tensor SSA exposes possible layout and storage work. It does
        # not establish runtime aliasing, ownership, allocation or lifetime
        # premises: the selected original host route must check those itself.
        for value in [*block.args, *(v for op in operations for v in op.results)]:
            typ = value.type
            if isinstance(typ, TensorType):
                if any(int(n) <= 0 for n in typ.get_shape()):
                    diagnostic.update(status='ineligible', reason='dynamic or empty tensor storage is not selected')
                    return None
            elif str(typ) not in ('index', 'i1', 'i8', 'i16', 'i32', 'i64', 'f32', 'bf16'):
                diagnostic.update(status='ineligible', reason='non-tensor memory or unsupported source value type')
                return None
        # Never request storage/layout rewrites across calls, memory operations
        # or control flow whose effects/ownership are absent from this source.
        prefixes = ('arith.', 'math.', 'tensor.', 'linalg.', 'quant_ext.')
        for op in operations[:-1]:
            if any(not li._op_name(nested).startswith(prefixes) for nested in op.walk()):
                diagnostic.update(status='ineligible', reason='source contains an unselected effect or control boundary')
                return None
        layout_ops = [i for i, op in enumerate(operations[:-1])
                      if li._op_name(op) in ('linalg.transpose', 'tensor.collapse_shape',
                                             'tensor.expand_shape', 'tensor.cast')]
        # Views do not establish independent allocation. Count only tensor
        # producers with an owning result candidate, excluding immutable splats
        # and constants. Runtime remains responsible for actual bufferization.
        materializers = [i for i, op in enumerate(operations[:-1])
                         if li._op_name(op) not in ('arith.constant', 'tensor.splat',
                             'tensor.collapse_shape', 'tensor.expand_shape', 'tensor.cast')
                         and any(isinstance(v.type, TensorType) for v in op.results)]
        layout = bool(layout_ops)
        arena = len(materializers) >= 2
        diagnostic.update(layout_source_op_indices=layout_ops,
                          static_tensor_producer_source_op_indices=materializers,
                          source_index_space='direct entry operations including initializers; diagnostics only',
                          fusion_status='unavailable: public request schema does not define fusion-group index semantics')
        if not layout and not arena:
            diagnostic.update(status='ineligible', reason='no static tensor layout or multi-buffer work')
            return None
        request = {
            'schema': 'merlin.host_optimization_request.v1',
            'source_sha256': digest,
            'propagate_layout': layout,
            'fusion_groups': [],
            'hoist_static_allocs': None,
            'sink_deallocs': None,
            'static_arena': arena,
            # Profitability floors, not asserted savings or measured costs.
            'minimum_transpose_bytes_saved': 64,
            'minimum_dispatches_saved': 1,
            'maximum_added_operations': 16,
        }
        schema = json.loads((root / 'data/schemas/command_buffer.schema.json').read_text())
        Draft202012Validator(schema['properties']['host_optimization_request']).validate(request)
    except Exception as error:
        diagnostic.update(status='unavailable', reason='optional source/schema selection failed',
                          error_type=type(error).__name__, detail=str(error))
        return None
    diagnostic.update(status='requested', requested_features={
        'propagate_layout': layout, 'static_arena': arena,
        'fusion_groups': False, 'hoist_static_allocs': None, 'sink_deallocs': None})
    return request
