"""Semantic command-buffer scheduling for the selected Gemmini target."""
from llvm_emitter import LLVMEmitter
from matmul_lowering import emit_matmul, load_resident_rhs
from merlin.compile.logical_abi import kernel_arg_order
from merlin.compile.tuning_knobs import knob
from pathlib import Path
from contextlib import nullcontext


def compile_kernel(cb):
    """Compile a semantic command buffer into bounded-size LLVM dialect IR."""
    names = kernel_arg_order(cb)
    e = LLVMEmitter(names)
    tensors = cb.get('tensors', {})
    pointers = dict(e.args)
    for name, tensor in tensors.items():
        if name not in pointers:
            count = 1
            for dim in tensor['shape']:
                count *= dim
            pointers[name] = e.scratch(count, tensor['dtype'])
    # Specialize tiny reused resident lifetimes without expanding large loops.
    resident_uses = {}
    for command in cb['commands']:
        if command.get('opcode', command.get('op')) == 'MATMUL_RESIDENT':
            handle = command.get('operands', {}).get('rhs')
            resident_uses[handle] = resident_uses.get(handle, 0) + 1
    tiny_resident_sources = set()
    residents = {}
    pending = {}
    sp_next = 32
    residency = knob('rhs_residency', default='full', values=('full', 'stream'),
                     root=Path(__file__).parent)
    for task_index, cmd in enumerate(cb['commands']):
        op = cmd.get('opcode', cmd.get('op'))
        args = cmd.get('operands', {})
        attrs = cmd.get('attributes', {})
        src_index = cmd.get('source_index', task_index)
        if not isinstance(src_index, int):
            src_index = task_index
        with e.task(task_index, source_op_index=src_index):
            if op == 'RES_PACK':
                src = args['src']
                shape = tensors[src]['shape']
                cached = None
                if len(shape) == 2:
                    K, N = shape
                    rows = ((K + 15) // 16) * ((N + 15) // 16) * 16
                    if residency == 'full' and sp_next + rows <= 16384:
                        tiny_reuse = (resident_uses.get(args['dst'], 0) > 1
                                      and K <= 32 and N <= 32)
                        if tiny_reuse:
                            tiny_resident_sources.add(src)
                        with (e.bounded_unroll(max_trip_count=2, budget=64)
                              if tiny_reuse else nullcontext()):
                            cached = load_resident_rhs(e, pointers[src], K, N, sp_next)
                        sp_next += rows
                        e._gemmini_resident_end = sp_next
                residents[args['dst']] = (src, cached)
                pointers[args['dst']] = pointers[src]
            elif op in ('MATMUL', 'MATMUL_RESIDENT'):
                lhs = args['lhs']
                rhs, resident = residents[args['rhs']] if op == 'MATMUL_RESIDENT' else (args['rhs'], None)
                pending[args['dst']] = (lhs, rhs, resident, attrs, task_index, src_index)
            elif op == 'COMMIT':
                lhs, rhs, resident, mat_attrs, mat_task, mat_source = pending[args['src']]
                M, K = tensors[lhs]['shape']
                rk, N = tensors[rhs]['shape']
                if rk != K:
                    raise ValueError('matrix reduction dimensions disagree')
                merged = dict(mat_attrs)
                merged.update(attrs)
                tiny_reuse = (rhs in tiny_resident_sources and resident is not None
                              and M <= 32 and K <= 32 and N <= 32)
                with (e.bounded_unroll(max_trip_count=2, budget=64)
                      if tiny_reuse else nullcontext()):
                    emit_matmul(e, pointers[lhs], pointers[rhs], pointers[args['dst']], M, K, N,
                                attrs=merged, bias=pointers.get(args.get('bias', attrs.get('bias'))),
                                rhs_resident=resident, compute_provenance={
                                    "task_index": mat_task, "source_op_index": mat_source})
            elif op == 'EVICT':
                residents.pop(args['handle'], None)
                if not residents:
                    sp_next = 32
                    e._gemmini_resident_end = sp_next
            elif op == 'ROW_MAJOR_VIEW':
                if args['src'] != args['dst']:
                    raise ValueError('row-major view must preserve the physical buffer')
            elif op == 'CONV2D':
                from conv_lowering import lower_conv
                lower_conv(e, cb, cmd, pointers)
            elif op == 'BATCHED_MATMUL':
                from conv_lowering import lower_batched
                lower_batched(e, cb, cmd, pointers)
            elif op in ('ATTENTION_QK', 'ATTENTION_PV'):
                from conv_lowering import lower_attention
                lower_attention(e, cb, cmd, pointers)
            else:
                from elementwise_lowering import emit_elementwise
                if not emit_elementwise(e, cb, cmd, pointers):
                    raise ValueError(f'unsupported semantic command {op}')
    return e.finish()
