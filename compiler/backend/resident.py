"""Bounded, generation-scoped scratchpad residency for repeated packed weights.

Selected public rtl_facts/facts.json describes four scratchpad banks, each
4096 rows of sixteen i8 lanes. Reserve only rows128:256, inside a conservative
512-row window. Ordinary matmul uses rows0:16 for A and rows16:32 for B.
Every operation whose scratchpad effects are not known here ends residency.
"""
from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import i8
from . import isa

ATTRIBUTE = '_resident_weight'
BASE_ROW = 128
MAX_K = 128
MAX_N = 16
CONSERVATIVE_ROWS = 512
assert BASE_ROW >= 32 and BASE_ROW + MAX_K <= CONSERVATIVE_ROWS


@dataclass
class ResidentWeight:
    """One packed generation shared by an uninterrupted run of consumers."""
    tensor: str
    handle: str
    generation: int
    k: int
    n: int
    consumers: tuple[int, ...]
    loaded: bool = False

    def address(self, f, kk, nj=None):
        """Local row for a reduction tile; columns always start at zero."""
        return f.add_i(f.const(BASE_ROW), kk)


def plan_resident_weights(cb, *, enabled=False):
    """Return COMMIT-index -> shared descriptor, without changing the buffer.

    Require two or more consecutive consumers of the same RES_PACK generation.
    Other contractions, source writes, repacking, eviction, host work, movement,
    convolution and pooling terminate a run. Fences and logical MATMUL records
    do not overwrite scratchpad. No workload names participate in eligibility.
    """
    if not enabled:
        return {}
    tensors = cb.get('tensors', {})
    handles, products, result = {}, {}, {}
    pending, key = [], None

    def flush():
        nonlocal pending, key
        if len(pending) >= 2:
            handle, generation, tensor, k, n = key
            descriptor = ResidentWeight(tensor, handle, generation, k, n,
                                        tuple(pending))
            result.update((index, descriptor) for index in pending)
        pending, key = [], None

    for index, command in enumerate(cb.get('commands', [])):
        op = command.get('opcode')
        operands = command.get('operands') or {}
        attrs = command.get('attributes') or {}
        if op == 'RES_PACK':
            flush()
            handles[operands.get('dst')] = (operands.get('src'), index)
        elif op in ('MATMUL', 'MATMUL_RESIDENT'):
            products[operands.get('dst')] = command
        elif op in ('BARRIER', 'FENCE', 'SYNC'):
            pass
        elif op == 'COMMIT':
            product = products.get(operands.get('src'), {})
            args = product.get('operands') or {}
            packed = handles.get(args.get('rhs'))
            combined = {**(product.get('attributes') or {}), **attrs}
            # A source write ends the packed generation even if the same
            # logical handle is subsequently read without another RES_PACK.
            written = operands.get('dst')
            invalid = [h for h, (source, _) in handles.items() if source == written]
            if invalid:
                flush()
                for handle in invalid:
                    del handles[handle]
                continue
            epilogue = combined.get('epilogue') or []
            epilogue = [epilogue] if isinstance(epilogue, str) else epilogue
            source = tensors.get(packed[0], {}) if packed else {}
            shape = source.get('shape', [])
            lhs = tensors.get(args.get('lhs'), {})
            ashape = lhs.get('shape', [])
            legal = (packed is not None and len(shape) == len(ashape) == 2
                     and source.get('dtype') == lhs.get('dtype') == 'i8'
                     and 0 < shape[0] <= MAX_K and 0 < shape[1] <= MAX_N
                     and ashape[1] == shape[0] and ashape[0] > 0
                     and combined.get('acc_scale_by_column') is None
                     and 'maxpool' not in epilogue)
            if not legal:
                flush()
                continue
            candidate = (args['rhs'], packed[1], packed[0], *shape)
            if candidate != key:
                flush()
                key = candidate
            pending.append(index)
        else:
            flush()
            # Unknown writers may alter either the reserved device rows or
            # the source storage. Demand a new RES_PACK before optimizing.
            handles.clear()
    flush()
    return result


def prepare(ctx, rhs, k, n, attrs):
    """Load the full RHS once, after the caller configured DMA state1.

    Matmul calls this after config_ld(stride=n,state=1). When a descriptor is
    returned, skip ordinary B loads and use descriptor.address(f,kk) for
    PRELOAD. Both scalar-row and grouped-row encoders must honor this contract.
    The surrounding matmul completion fence also protects later consumers.
    """
    descriptor = (attrs or {}).get(ATTRIBUTE)
    if descriptor is None:
        return None
    if not isinstance(descriptor, ResidentWeight):
        raise ValueError('invalid resident-weight lowering descriptor')
    if (k, n) != (descriptor.k, descriptor.n):
        raise ValueError('resident-weight dimensions disagree with matmul')
    pointer = ctx.ptr(rhs) if isinstance(rhs, str) else rhs
    if pointer is not ctx.ptr(descriptor.tensor):
        raise ValueError('resident-weight source disagrees with matmul')
    if (attrs or {}).get('acc_scale_by_column') is not None:
        raise ValueError('resident-weight lowering requires full-width columns')
    if not descriptor.loaded:
        f = ctx.f
        for kk in range(0, k, isa.DIM):
            isa.mvin(f, f.gep(pointer, f.const(kk * n), i8), BASE_ROW + kk,
                     n, min(isa.DIM, k - kk), state=1)
        descriptor.loaded = True
    return descriptor
