"""Batched and attention contractions through actual Gemmini dense schedules.

Only pointer offsets and counted loop induction are computed on the host. QK
selects either exact key-column views or the shared data-movement transpose
emitter; every contraction and numerical readout runs on the accelerator.
"""
from math import prod
from types import SimpleNamespace

from dense import emit_dense


def _shape(shape):
    result = tuple(int(d) for d in shape)
    if len(result) < 2 or any(d <= 0 for d in result):
        raise ValueError("contraction requires positive static matrix extents")
    return result


def _readout(out_dtype, scale, relu, attrs):
    attrs = dict(attrs)
    stages = attrs.pop('epilogue', None)
    dtype = attrs.pop('output_dtype', out_dtype)
    if stages is not None:
        if set(stages) - {'acc_scale', 'relu'}:
            raise ValueError("attention readout implements acc_scale and relu only")
        relu = 'relu' in stages
        if 'acc_scale' in stages:
            if 'acc_scale' not in attrs:
                raise ValueError("acc_scale stage requires its declared multiplier")
            scale = attrs.pop('acc_scale')
        elif 'acc_scale' in attrs:
            raise ValueError("attention multiplier has no acc_scale stage")
    elif 'acc_scale' in attrs:
        scale = attrs.pop('acc_scale')
    bias = attrs.pop('bias', None)
    if bias is not None:
        raise ValueError("attention source declares no bias operand")
    # These name/provenance annotations do not change the contraction.
    for key in tuple(attrs):
        if key in {'name', 'batch'} or key.startswith('prov.'):
            attrs.pop(key)
    if attrs:
        raise ValueError(f"unsupported attention attributes: {sorted(attrs)}")
    if dtype not in ('i8', 'i32'):
        raise ValueError("attention readout requires i8 or i32")
    if isinstance(scale, (tuple, list)):
        raise ValueError("attention currently requires a scalar readout multiplier")
    if dtype == 'i32' and (float(scale) != 1.0 or relu):
        raise ValueError("full-width i32 readout does not apply scaled or activated epilogues")
    return dict(out_dtype=dtype, scale=float(scale), relu=bool(relu))


def emit_batched(enc, a, w, out, ashape, wshape, *, out_dtype='i32',
                 scale=1.0, relu=False, **attrs):
    """Identical-prefix A[*B,M,K] @ W[*B,K,N], with no broadcasting."""
    ashape, wshape = _shape(ashape), _shape(wshape)
    if len(ashape) < 3 or len(ashape) != len(wshape) or ashape[:-2] != wshape[:-2]:
        raise ValueError("batched matmul requires identical nonempty batch prefixes")
    m, k = ashape[-2:]
    if k != wshape[-2]:
        raise ValueError("batched matmul reduction extents disagree")
    n = wshape[-1]
    options = _readout(out_dtype, scale, relu, attrs)
    if options != dict(out_dtype='i32', scale=1.0, relu=False):
        raise ValueError("BATCHED_MATMUL i8 operands require an exact i32 destination")
    receipts = []

    def batch(index):
        receipts.append(emit_dense(enc,
            enc.offset(a, enc.mul(index, m*k)),
            enc.offset(w, enc.mul(index, k*n)),
            enc.offset(out, enc.mul(index, m*n*4)), m, k, n, **options))

    enc.fn.for_loop(0, prod(ashape[:-2]), 1, batch, retain_loop=True)
    return receipts


class _CountedPreparationBuilder:
    """Use the shared transpose emitter with bounded ordinary CPU loops."""
    def __init__(self, builder):
        self.builder = builder

    def __getattr__(self, name):
        return getattr(self.builder, name)

    def for_range(self, count, body):
        return self.builder.for_loop(0, count, 1, body, retain_loop=True)


def _qk_cell(enc, q, k, out, m, d, n, options):
    width = 4 if options['out_dtype'] == 'i32' else 1
    receipts = []

    def query(i):
        qp = enc.offset(q, enc.mul(i, d))

        def key(j):
            kp = enc.offset(k, enc.mul(j, d))
            op = enc.offset(out, enc.mul(enc.add(enc.mul(i, n), j), width))
            receipts.append(emit_dense(enc, qp, kp, op, 1, d, 1, **options))

        enc.fn.for_loop(0, n, 1, key, retain_loop=True)

    enc.fn.for_loop(0, m, 1, query, retain_loop=True)
    return receipts


def _qk_packed(enc, q, k, out, m, d, n, options, scratch, columns, *, tiled=False):
    from merlin.llvmlower.host_epilogue import HostEpilogueEmitter

    width = 4 if options['out_dtype'] == 'i32' else 1
    receipts = []

    def panel(first, count):
        kp = enc.offset(k, enc.mul(first, d))
        emitter = HostEpilogueEmitter(_CountedPreparationBuilder(enc.fn),
            lambda value: value, None, row_alignment=1, operand_dtype='i8', pool_floor=-128)
        emitter.emit_transpose(SimpleNamespace(bufs=(kp, scratch),
            attrs=dict(rows=count, cols=d, dtype='i8')))
        enc.fence()

        def query(i):
            qp = enc.offset(q, enc.mul(i, d))
            op = enc.offset(out, enc.mul(enc.add(enc.mul(i, n), first), width))
            receipts.append(emit_dense(enc, qp, scratch, op, 1, d, count, **options))

        if tiled:
            # The checked dense schedule owns M tiling and exact readout. Its
            # logical M x count product writes a subview of the original M x N
            # destination, retaining that destination's physical row pitch.
            receipt=emit_dense(enc,q,scratch,
                enc.offset(out,enc.mul(first,width)),m,d,count,out_pitch=n,**options)
            receipts.append({**receipt,'qk_schedule':'tiled',
                'source_view':{'q_shape':[m,d],'key_shape':[n,d],
                    'packed_shape':[d,count],'output_shape':[m,n],
                    'output_row_pitch':n,'key_panel_columns':count,
                    'key_panel_capacity':columns},
                'preparation':{'kind':'shared_host_transpose','dtype':'i8',
                    'bytes_per_panel':d*count,'lifetime':'completed_key_panel'}})
        else:
            enc.fn.for_loop(0, m, 1, query, retain_loop=True)
        enc.fence()

    full, tail = divmod(n, columns)
    enc.fn.for_loop(0, full, 1, lambda j: panel(enc.mul(j, columns), columns), retain_loop=True)
    if tail:
        panel(enc.val(full*columns), tail)
    return receipts


def emit_qk(enc, q, k, out, qshape, kshape, *, out_dtype='i32',
            scale=1.0, relu=False, schedule=None, **attrs):
    """Q[*B,M,D] @ K[*B,N,D]^T with identical explicit batch prefixes.

    QK_SCHEDULE=cell retains the incumbent direct dot products (default).
    QK_SCHEDULE=packed uses the library's exact data-movement transpose in a
    bounded private panel, then wider accelerator contractions. Packing storage
    is reused only after full completion and across invocations of this emitter.
    QK_SCHEDULE=tiled additionally tiles all query rows through the shared dense
    schedule, binding each output panel to the original destination row pitch.
    Neither schedule performs a contraction or numerical epilogue on the host.
    """
    qshape, kshape = _shape(qshape), _shape(kshape)
    if len(qshape) != len(kshape) or qshape[:-2] != kshape[:-2] or qshape[-1] != kshape[-1]:
        raise ValueError("attention QK requires Q[*B,M,D] and K[*B,N,D] with identical prefixes")
    m, d = qshape[-2:]
    n = kshape[-2]
    options = _readout(out_dtype, scale, relu, attrs)
    from attention_schedule import selected_schedule,QK_WORKSPACE_BYTES
    selected=selected_schedule(schedule)
    # Half of the selected 8192-byte kernel stack preflight leaves space for
    # scalar spills. A single invocation-private panel is reused across calls.
    workspace_bytes = QK_WORKSPACE_BYTES
    columns = min(n, workspace_bytes // d)
    if selected=='packed':columns=min(16,columns)
    scratch = None
    if selected in ('packed','tiled') and columns >= (1 if selected=='tiled' else 2):
        from xdsl.dialects import llvm
        from xdsl.dialects.builtin import i8
        scratch = getattr(enc, '_qk_workspace', None)
        if scratch is None:
            scratch = enc.fn.add(llvm.AllocaOp(enc.val(workspace_bytes), i8, alignment=64)).results[0]
            enc._qk_workspace = scratch
    width = 4 if options['out_dtype'] == 'i32' else 1
    receipts = []

    def batch(index):
        qp = enc.offset(q, enc.mul(index, m*d))
        kp = enc.offset(k, enc.mul(index, n*d))
        op = enc.offset(out, enc.mul(index, m*n*width))
        if scratch is not None:
            receipts.extend(_qk_packed(enc, qp, kp, op, m, d, n, options, scratch, columns,
                tiled=selected=='tiled'))
        else:
            receipts.extend(_qk_cell(enc, qp, kp, op, m, d, n, options))

    if qshape[:-2]:
        enc.fn.for_loop(0, prod(qshape[:-2]), 1, batch, retain_loop=True)
    else:
        batch(enc.val(0))
    return receipts


def emit_pv(enc, p, v, out, pshape, vshape, *, out_dtype='i32',
            scale=1.0, relu=False, **attrs):
    """P[*B,M,S] @ V[*B,S,D] with identical explicit batch prefixes."""
    pshape, vshape = _shape(pshape), _shape(vshape)
    if (len(pshape) != len(vshape) or pshape[:-2] != vshape[:-2]
            or pshape[-1] != vshape[-2]):
        raise ValueError("attention PV requires P[*B,M,S] and V[*B,S,D] with identical prefixes")
    m, s = pshape[-2:]
    d = vshape[-1]
    options = _readout(out_dtype, scale, relu, attrs)
    if not pshape[:-2]:
        return emit_dense(enc, p, v, out, m, s, d, **options)
    width = 4 if options['out_dtype'] == 'i32' else 1
    receipts = []

    def batch(index):
        receipts.append(emit_dense(enc,
            enc.offset(p, enc.mul(index, m*s)),
            enc.offset(v, enc.mul(index, s*d)),
            enc.offset(out, enc.mul(index, m*d*width)), m, s, d, **options))

    enc.fn.for_loop(0, prod(pshape[:-2]), 1, batch, retain_loop=True)
    return receipts
