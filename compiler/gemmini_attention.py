"""Attention and equal-prefix batch wrappers around accelerator dense lowering.

QK's layout preparation uses the common host transpose emitter with retained CFG
loops. Products and numerical readout always belong to the dense accelerator
path. Scratch is reused only after that path's synchronous completion.
"""
from __future__ import annotations

from math import prod
from types import SimpleNamespace

from merlin.llvmlower.host_epilogue import HostEpilogueEmitter

try:
    from .gemmini_dense import emit_dense
    from .gemmini_support import scratch_buffer
    from . import gemmini_encode as enc
except ImportError:
    from gemmini_dense import emit_dense
    from gemmini_support import scratch_buffer
    import gemmini_encode as enc


# A caller may supply a longer-lived private buffer under this key to avoid any
# kernel frame growth. It must contain at least N*D bytes and remain live until
# emit_attention returns. Sequential attention invocations may share it.
SCRATCH_KEY = "__attention_scratch__"


class _CountedBuilder:
    """Adapt the shared host emitter's range interface to ordinary CFG loops."""
    def __init__(self, builder):
        self.builder = builder

    def __getattr__(self, name):
        return getattr(self.builder, name)

    def for_range(self, count, body):
        self.builder.for_loop(0, count, 1, body, retain_loop=True)


def _shape(spec, name):
    shape = spec.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) < 2 or any(
        type(extent) is not int or extent <= 0 for extent in shape
    ):
        raise ValueError(f"{name}: requires a positive static tensor of rank at least two")
    return tuple(shape)


def emit_attention(fn, op: str, ptrs: dict, tensors: dict, operands: dict, attrs: dict, *, preparation='scalar'):
    """Emit QK, PV or independent batched matmul; return a JSON-safe receipt.

    ``ptrs`` contains original contiguous logical tensor pointers. QK may also
    receive ``ptrs[SCRATCH_KEY]`` for a disjoint N*D-byte transpose buffer. Equal
    batch prefixes are flattened in row-major lexicographic order; broadcasting
    is refused. Input and destination dtypes remain source declarations.
    """
    attrs = dict(attrs or {})
    if preparation not in ('scalar','blocked'):
        raise ValueError('attention preparation must be scalar or blocked')
    keys = {"ATTENTION_QK": ("q", "k"), "ATTENTION_PV": ("p", "v"),
            "BATCHED_MATMUL": ("a", "w")}
    if op not in keys:
        raise ValueError(f"attention wrapper cannot lower {op!r}")
    left_key, right_key = keys[op]
    left_name, right_name, dst_name = (operands[left_key], operands[right_key], operands["dst"])
    ls, rs, ds = (_shape(tensors[name], name) for name in (left_name, right_name, dst_name))
    if len(ls) != len(rs) or len(ls) != len(ds) or ls[:-2] != rs[:-2] or ls[:-2] != ds[:-2]:
        raise ValueError(f"{op}: tensor ranks and complete batch prefixes must agree")
    if op == "BATCHED_MATMUL" and len(ls) < 3:
        raise ValueError("BATCHED_MATMUL requires a nonempty batch prefix")
    transpose = op == "ATTENTION_QK"
    m, k = ls[-2:]
    if transpose:
        n, rk = rs[-2:]
        if any(key in attrs for key in ("transpose_rhs", "rhs_transposed", "transpose_b")):
            raise ValueError("ATTENTION_QK has intrinsic transpose semantics, not a transpose flag")
    else:
        rk, n = rs[-2:]
    if rk != k or ds[-2:] != (m, n):
        raise ValueError(f"{op}: incompatible source contraction or destination shapes")
    lhs_dtype, rhs_dtype, output_dtype = (str(tensors[name]["dtype"])
        for name in (left_name, right_name, dst_name))
    if lhs_dtype not in ("i8", "int8") or rhs_dtype not in ("i8", "int8"):
        raise ValueError(f"{op}: dense accelerator operands must be i8")
    if output_dtype not in ("i8", "int8", "i32", "int32"):
        raise ValueError(f"{op}: output dtype must be i8 or i32")
    if attrs.get("output_dtype", output_dtype) != output_dtype:
        raise ValueError(f"{op}: output_dtype attribute differs from typed destination")
    if op == "BATCHED_MATMUL" and output_dtype not in ("i32", "int32"):
        raise ValueError("BATCHED_MATMUL i8 products require an i32 destination")
    batch_count = prod(ls[:-2])
    out_width = 1 if output_dtype in ("i8", "int8") else 4
    scratch = None
    allocation = "none"
    if transpose:
        scratch = ptrs.get(SCRATCH_KEY)
        if scratch is None:
            scratch = scratch_buffer(fn, k*n, name="attention_transpose")
            allocation = "module_private"
        else:
            allocation = "caller_private"
    receipts = []
    preparation_receipts=[]
    if not transpose:preparation='none'

    def one_batch(batch):
        def offset(pointer, stride):
            return enc.pointer_offset(fn, pointer, fn.mul_i(batch, fn.const(stride)))
        lhs = offset(ptrs[left_name], m*k)
        rhs = offset(ptrs[right_name], k*n)
        dst = offset(ptrs[dst_name], m*n*out_width)
        if transpose:
            # Prior device effects complete before CPU layout preparation. The
            # common emitter performs only an exact byte permutation here.
            enc.fence(fn)
            if preparation=='blocked':
                from gemmini_attention_prepare import emit_blocked_transpose
                preparation_receipts.append(emit_blocked_transpose(fn,rhs,scratch,n,k))
            else:
                pointers = {"source": rhs, "transpose": scratch}
                emitter = HostEpilogueEmitter(_CountedBuilder(fn), pointers.__getitem__, None,
                    row_alignment=1, operand_dtype="i8", pool_floor=-128)
                emitter.emit_transpose(SimpleNamespace(bufs=("source", "transpose"),
                    attrs={"rows": n, "cols": k, "dtype": "i8"}))
            rhs = scratch
        receipts.append(emit_dense(fn, lhs, rhs, dst, m, k, n, attrs=attrs,
            lhs_dtype=lhs_dtype, rhs_dtype=rhs_dtype, output_dtype=output_dtype))

    # One emitted body independent of the static batch count, preserving compact
    # artifacts for large arbitrary batch prefixes.
    fn.for_loop(0, batch_count, 1, one_batch, retain_loop=True)
    return {"op": op, "batch_shape": list(ls[:-2]), "batch_count": batch_count,
            "m": m, "k": k, "n": n, "transpose_rhs": transpose,
            "scratch_bytes": k*n if transpose else 0, "scratch_allocation": allocation,
            "preparation":preparation,"preparation_receipts":preparation_receipts,
            "dense": receipts}
