"""Source-bound permutation preparations for ordinary dense consumers.

A permutation is not an admitted strided input ABI.  This module materializes
its exact output into a distinct dense rank-two buffer.  The checked transpose
index program is the executed product; reshaping its output changes no address.
The caller allocates the destination, keeps it live through every consumer, and
may release it only after the consumer's ordinary completion boundary.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import prod

from .transpose import TransposePlan, TransposeError, plan_transpose
from . import isa
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter, emit_index_program


class PermutedDenseViewError(ValueError):
    pass


@dataclass(frozen=True)
class MaterializedDenseView:
    source_shape: tuple[int, ...]
    permutation: tuple[int, ...]
    dense_shape: tuple[int, int]
    dtype: str
    movement: TransposePlan

    def revalidate(self):
        output = tuple(self.source_shape[i] for i in self.permutation)
        if (self.movement.shape != self.source_shape
                or self.movement.permutation != self.permutation
                or self.movement.dtype != self.dtype
                or len(self.dense_shape) != 2
                or any(type(x) is not int or x <= 0 for x in self.dense_shape)
                or prod(output) != prod(self.dense_shape)):
            raise PermutedDenseViewError('permutation preparation storage boundary changed')
        proof = self.movement.revalidate()
        return dict(kind='materialized_permuted_dense_view',
            source_shape=list(self.source_shape), permutation=list(self.permutation),
            materialized_shape=list(output), dense_shape=list(self.dense_shape),
            address_projection='dense[row,col] = permutation_output.linear[row*K+col]',
            source_access='read only; original independently live values retained',
            destination_access='distinct fresh dense allocation; exact complete overwrite',
            lifetime='allocation before preparation; completed before consumer; release after last completed consumer',
            movement=proof, kernel_sha256=self.movement.kernel.digest())

    def source_index(self, row: int, column: int) -> int:
        """Exact scalar source map, also useful for independent test oracles."""
        m, k = self.dense_shape
        if not (0 <= row < m and 0 <= column < k):
            raise PermutedDenseViewError('dense coordinate is outside preparation')
        index = row*k + column
        output = tuple(self.source_shape[i] for i in self.permutation)
        coords = [0]*len(output)
        for axis in range(len(output)-1, -1, -1):
            index, coords[axis] = divmod(index, output[axis])
        src = [0]*len(output)
        for axis, original in enumerate(self.permutation):
            src[original] = coords[axis]
        offset = 0
        for extent, coordinate in zip(self.source_shape, src, strict=True):
            offset = offset*extent + coordinate
        return offset


def plan_materialized_dense_view(source_shape, permutation, dense_shape, dtype='i8', *, schedule='batched'):
    """Construct an executed DMA plan; never grant a new external view ABI."""
    source_shape, permutation, dense_shape = tuple(source_shape), tuple(permutation), tuple(dense_shape)
    if (len(dense_shape) != 2 or any(type(x) is not int or x <= 0 for x in dense_shape)
            or prod(source_shape) != prod(dense_shape)):
        raise PermutedDenseViewError('preparation needs an equal-size positive static rank-two destination')
    movement = plan_transpose(source_shape, permutation, dtype, schedule=schedule)
    result = MaterializedDenseView(source_shape, permutation, dense_shape, dtype, movement)
    result.revalidate()
    return result


@dataclass(frozen=True)
class SourcePermutationPreparation:
    """Original SSA custody and the movement product, with no source mutation."""
    source: object
    result: object
    operations: tuple
    plan: MaterializedDenseView


def match_source_preparation(value, *, schedule='batched'):
    """Match explicit transpose followed only by contiguous static reshapes.

    The returned operations may be absorbed into a source device group only
    after its caller preserves any other live intermediate outputs.  This
    matcher refuses such fanout rather than silently dropping an original use.
    The transpose's input remains an ordinary, independently live boundary.
    """
    from xdsl.dialects.builtin import TensorType, NoneAttr
    from .quantized_boundary import _contiguous_view, QuantizedBoundaryError
    from .integer_movement import match as match_integer_movement

    def spec(v):
        typ = v.type
        if (not isinstance(typ, TensorType) or not isinstance(typ.encoding, NoneAttr)
                or any(x <= 0 for x in typ.get_shape()) or str(typ.element_type) not in ('i8','i32')):
            raise PermutedDenseViewError('source preparation requires static unencoded signed integer tensors')
        return tuple(typ.get_shape()), str(typ.element_type)

    dense, dtype = spec(value)
    if len(dense) != 2:
        raise PermutedDenseViewError('dense consumer input must have rank two')
    current=value; reverse=[]
    while getattr(current.owner, 'name', '') in ('tensor.collapse_shape','tensor.expand_shape'):
        op=current.owner
        before, bt = spec(op.operands[0]); after, at = spec(current)
        try:
            contiguous = _contiguous_view(op,current,dtypes=('i8','i32')) is op.operands[0]
        except QuantizedBoundaryError:
            contiguous = False
        if (bt != dtype or at != dtype or prod(before) != prod(after) or not contiguous):
            raise PermutedDenseViewError('reshape lacks exact contiguous source proof')
        reverse.append(op); current=op.operands[0]
    transpose=current.owner
    if getattr(transpose,'name','') != 'linalg.transpose' or len(transpose.operands)!=2 or len(transpose.results)!=1:
        raise PermutedDenseViewError('dense source lacks an explicit source transpose')
    if match_integer_movement(transpose,transpose_transfers=True) is None:
        raise PermutedDenseViewError('transpose lacks the original exact integer scalar-body proof')
    source=transpose.operands[0]
    shape, st = spec(source)
    output, ot = spec(current)
    perm=tuple(transpose.permutation.get_values())
    if st != dtype or ot != dtype or spec(transpose.operands[1]) != (output,dtype):
        raise PermutedDenseViewError('source transpose destination or element type differs')
    plan=plan_materialized_dense_view(shape,perm,dense,dtype,schedule=schedule)
    if output != tuple(shape[i] for i in perm):
        raise PermutedDenseViewError('source transpose output shape differs')
    operations=(transpose,*reversed(reverse))
    for index, op in enumerate(operations[:-1]):
        if any(use.operation is not operations[index+1] for use in op.results[0].uses):
            raise PermutedDenseViewError('permutation intermediate remains independently live')
    return SourcePermutationPreparation(source,value,operations,plan)


def emit_materialized_dense_view(ctx, source: str, destination: str, plan: MaterializedDenseView):
    """Emit the checked product into caller-owned fresh dense storage.

    No tensor aliases, source writes, CPU tensor accesses or external ABI changes
    occur here.  The same Gemmini config/mvin/mvout encoder as integer transpose
    consumes the checked index program.  Its terminal drains complete the write.
    """
    proof=plan.revalidate()
    a,b=ctx.spec(source),ctx.spec(destination)
    aliases=ctx.cb.get('params',{}).get('tensor_aliases',{})
    aliased=set(aliases)|{entry.get('source') for entry in aliases.values() if isinstance(entry,dict)}
    if (source == destination or {source,destination}&aliased
            or tuple(a['shape']) != plan.source_shape or tuple(b['shape']) != plan.dense_shape
            or a['dtype'] != plan.dtype or b['dtype'] != plan.dtype):
        raise PermutedDenseViewError('preparation requires distinct exact original source and fresh dense destination')
    bridge=FnBuilderIndexAdapter(ctx.fb)
    def emit(name,arguments):
        if name=='transpose_drain':ctx.fence();return
        if name=='transpose_config_load':
            ctx.emit(*isa.config_ld(arguments['stride'],shrink=plan.dtype=='i8'));return
        if name=='transpose_config_store':
            ctx.emit(*isa.config_st(arguments['stride']));return
        args={key:bridge.value(value) for key,value in arguments.items()}
        packed=isa.pack_matrix(args['addr'],args['rows'],args['cols'],fb=ctx.fb)
        if name=='transpose_mvin':ctx.emit(isa.FUNCT['mvin'],ctx.ptr(source,args['offset']),packed)
        elif name=='transpose_mvout':ctx.emit(isa.FUNCT['mvout'],ctx.ptr(destination,args['offset']),packed)
        else:raise TransposeError('unknown checked preparation operation')
    emission=emit_index_program(plan.movement.kernel,bridge,emit)
    return dict(proof=proof,emission=emission,kernel_sha256=plan.movement.kernel.digest(),
        consumed_schedule=dict(plan.movement.kernel.attrs).get('transpose_schedule','baseline'))


def allocate_and_emit_dense_view(ctx, source: str, scratch: str, plan: MaterializedDenseView):
    """Create internal storage and prepare it for subsequent ordinary dense loads.

    EmissionContext owns the private allocation for the kernel's entire call, so
    no last-use estimate or borrowed external buffer is needed.  Refuse an
    existing name instead of overwriting another independently live value.
    """
    if scratch in ctx.tensors or scratch in ctx.buffers:
        raise PermutedDenseViewError('dense preparation scratch name is already live')
    # Validate the actual original boundary before allocating any storage.
    plan.revalidate()
    original=ctx.spec(source)
    if tuple(original['shape']) != plan.source_shape or original['dtype'] != plan.dtype:
        raise PermutedDenseViewError('preparation source differs from the checked original boundary')
    ctx.alloc(scratch,plan.dense_shape,plan.dtype)
    receipt=emit_materialized_dense_view(ctx,source,scratch,plan)
    return scratch,{**receipt,'storage':dict(kind='kernel_private_fresh_allocation',name=scratch,
        bytes=prod(plan.dense_shape)*(1 if plan.dtype=='i8' else 4),
        lifetime='entire kernel call; completed preparation precedes ordinary consumer')}
