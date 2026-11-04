"""Reusable physical preparation for row-per-key attention operands.

The cache belongs to one generated kernel, not to the process executing it:
each invocation executes its first preparation again. Writes to a logical K
buffer between consumers force a fresh copy into the same bounded scratch.
"""
from __future__ import annotations

from pathlib import Path

from xdsl.dialects.builtin import i8, StringAttr
from merlin.compile.tuning_knobs import knob

from . import isa


def _transpose_module(rows, channels):
    """Exact typed internal layout operation consumed by the shared host compiler."""
    if any(type(d) is not int or d <= 0 for d in (rows, channels)):
        raise ValueError('attention key preparation requires positive static dimensions')
    return f'''builtin.module {{
  func.func @host_region(%key: tensor<{rows}x{channels}xi8>) -> tensor<{channels}x{rows}xi8> {{
    %empty = tensor.empty() : tensor<{channels}x{rows}xi8>
    %transpose = linalg.generic {{
      indexing_maps = [affine_map<(d0,d1)->(d1,d0)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    }} ins(%key: tensor<{rows}x{channels}xi8>) outs(%empty: tensor<{channels}x{rows}xi8>) {{
      ^bb0(%value: i8, %old: i8):
        linalg.yield %value : i8
    }} -> tensor<{channels}x{rows}xi8>
    func.return %transpose : tensor<{channels}x{rows}xi8>
  }}
}}'''


def _storage_version(ctx, command, name):
    """Last explicit producer preceding this consumer, or an uncacheable token."""
    version = -1
    for index, previous in enumerate(ctx.cb.get('commands', ())):
        if previous is command:
            return version
        operands = previous.get('operands', {})
        attrs = previous.get('attributes', {})
        outputs = [operands.get('dst'), *attrs.get('output_tensors', ())]
        outputs += [value for key, value in operands.items() if key.startswith('output')]
        if name in outputs:
            version = index
    # A standalone caller has not supplied enough ordering information to prove
    # reuse. A new object guarantees it receives an emitted copy every time.
    return object()


def prepare_key_transpose(ctx, command, name, rows, channels):
    """Return physical `[channels,rows]` i8 storage for logical K[rows,channels]."""
    source = ctx.ptr(name)
    key = (source, int(rows), int(channels))
    cache = getattr(ctx, '_attention_preparations', None)
    if cache is None:
        cache = ctx._attention_preparations = {}
    version = _storage_version(ctx, command, name)
    record = cache.get(key)
    if record is not None and record[0] == version:
        return record[1]
    if record is None:
        scratch_name = '__attention_keys_' + str(len(cache))
        while scratch_name in ctx.cb['tensors']:
            scratch_name += '_'
        target = ctx.alloc(scratch_name, [channels, rows], 'i8')
    else:
        # All prior consumers have completed through matmul's completion fence;
        # a new source generation can overwrite the existing physical scratch.
        target = record[1]
    f = ctx.f
    mode = knob('attention_key_prepare', default='cpu', values=('cpu', 'device'),
                root=Path(__file__).resolve().parents[1])
    if mode == 'device':
        # Reuse the exact bounded layout-DMA implementation used by explicit
        # source transpose operations. Iteration [d,key] reads K[key,d] and
        # writes packed KT[d,key]. Tensor values stay in the DMA/accumulator
        # path; the CPU computes only addresses and loop control.
        from .layout_movement import _indexed
        _indexed(f, source, target, 0, 0, (channels, rows),
                 (1, channels), (rows, 1), i8, 1)
        isa.fence(f)
        cache[key] = (version, target)
        return target
    from .host import lower_host_task
    attrs = command.get('attributes', {})
    preparation = {'attributes': {
        'host_module_ir': _transpose_module(rows, channels),
        'input_tensors': ['key'], 'output_tensors': ['transposed_key'],
        'host_reason': 'internal typed attention key layout preparation',
        **{k: attrs.get(k, getattr(f, k, None))
           for k in ('source_op_index', 'task_index')
           if attrs.get(k, getattr(f, k, None)) is not None},
    }}
    # The shared adapter compiles and appends a closed, synchronous CPU region.
    # These pointers retain the same nonaliasing ownership and completion
    # boundaries as the preceding scalar preparation route.
    prior_blocks = set(f.region.blocks)
    lower_host_task(f, preparation, {'key': source, 'transposed_key': target}, ctx.alloc)
    region = attrs.get('source_region_id', getattr(f, 'region_id', None))
    if region:
        for block in f.region.blocks:
            if block not in prior_blocks:
                for op in block.ops:
                    op.attributes['prov.region_id'] = StringAttr(str(region))
    cache[key] = (version, target)
    return target
