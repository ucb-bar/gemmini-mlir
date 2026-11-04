"""Per-invocation entry storage for compiler-private transpose transport.

The backing allocation has byte alignment, so LLVM need not realign the RISC-V
stack. Its interior pointer retains the original 64-byte tile alignment. This
changes neither explicit source alignments nor any tensor allocation.
"""


def entry_transpose_tile(builder, edge):
    """Allocate one distinct static slot for a target-generated RHS tile.

    Use only for lower_dense's private byte transpose tile. Each dynamic use
    writes every byte read by that tile's mvin, the address never escapes, and
    lower_dense completes device effects before its enclosing batch iterates.
    No two call sites share a slot. Reentrant invocations retain separate frames.
    """
    if type(edge) is not int or edge <= 0 or edge * edge > 1024:
        raise ValueError('private transpose requires a positive bounded static tile')
    size, alignment = edge * edge, 64
    # For any backing address p, align_up(p,A)-p is in [0,A-1].
    # Therefore size+A-1 backing bytes contain all size bytes of the aligned tile.
    backing_size = size + alignment - 1
    previous_lines, previous_block = builder.lines, builder.current_block
    serial = len(getattr(builder, 'target_storage_receipts', ()))
    builder.lines = []
    builder.current_block = '__target_private_entry_' + str(serial)
    try:
        raw = builder.alloca(backing_size, ty='i8', alignment=1)
        allocation_text = builder.lines[-1]
        tail = 'alignment = 1 : i64}'
        if allocation_text.count(tail) != 1:
            raise ValueError('private allocation builder lost explicit byte alignment')
        builder.lines[-1] = allocation_text.replace(
            tail, 'alignment = 1 : i64, merlin.target_private_transpose_backing}')
        address = builder.ptrtoint(raw)
        aligned = builder.and_(builder.add(address, alignment - 1), -alignment)
        result = builder.inttoptr(aligned)
        prologue = builder.lines
    finally:
        builder.lines, builder.current_block = previous_lines, previous_block
    # All operands above are newly generated constants or private values. The
    # prologue dominates every original block and keeps original source tags.
    builder.lines[0:0] = prologue
    if not hasattr(builder, 'target_storage_receipts'):
        builder.target_storage_receipts = []
    builder.target_storage_receipts.append({
        'schema':'target_private_entry_tile_v1', 'purpose':'RHS byte transpose transport',
        'bytes':size, 'backing_bytes':backing_size, 'backing_alignment':1,
        'effective_alignment':alignment, 'allocation':'distinct constant entry slot per call site',
        'lifetime':'one invocation; every tile read follows complete overwrite; batch reuse follows dense completion',
        'global_task':builder.task_index, 'source_op_index':builder.source_op_index,
        'region_id':builder.region_id,
        'source_tensor_changes':False, 'shared_global_storage':False,
        'numeric_changes':False,
    })
    return result


def normalize_target_storage(text, *, report=None):
    """Re-establish static entry placement after whole-program LLVM inlining.

    Only slots emitted by entry_transpose_tile carry the marker. Explicit source
    allocations are never candidates. Alignment arithmetic, writes, DMA reads,
    fences and lifetime intrinsics remain at their original program points.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.common.dense_float_ir import DenseFloatPrinter
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from io import StringIO
    module = parse_mlir_text(text, ctx=make_llvm_context())
    changed, slots = False, []
    for function in module.body.block.ops:
        if function.name != 'llvm.func' or not function.body.blocks:
            continue
        entry = function.body.blocks.first
        for op in tuple(function.walk()):
            if 'merlin.target_private_transpose_backing' not in op.attributes:
                continue
            if op.name != 'llvm.alloca' or len(op.operands) != 1:
                raise ValueError('private target storage marker requires one primitive allocation')
            count = op.operands[0].owner
            try:
                number = count.properties['value'].value.data
                valid = (count.name == 'llvm.mlir.constant' and type(number) is int
                         and 64 <= number <= 1087 and str(op.properties['elem_type']) == 'i8'
                         and op.properties['alignment'].value.data == 1)
            except (AttributeError,KeyError):
                valid = False
            if not valid:
                raise ValueError('private transpose backing lost its static byte-storage contract')
            moved = op.parent is not entry
            if moved:
                bound = count.clone()
                op.parent.detach_op(op)
                op.operands = (bound.results[0],)
                entry.insert_ops_before([bound,op],entry.first_op)
                changed = True
            slots.append({'function':function.sym_name.data,'backing_bytes':number,
                          'backing_alignment':1,'effective_tile_alignment':64,'moved_to_entry':moved})
    module.verify()
    if report is not None:
        report.update({'schema':'target_private_static_entry_v1','slots':slots,
                       'source_allocations_changed':False,
                       'user_operations_moved':False,'shared_global_storage':False})
    if not changed:
        return text
    stream = StringIO(); DenseFloatPrinter(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()+'\n'
