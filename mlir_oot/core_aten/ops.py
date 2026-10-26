"""Core ATen target schedules: exact int8/i32 math and full-width readout.

Quantized outputs use an exact rational CPU epilogue, fenced after the device
stores. Distinct source scales remain distinct; coefficients are never rounded
into approximate integer weights. These schedules require unity data-path scales
and full-width accumulator stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .exactness import rational as Q, require, accumulator_bound, code_offset
from .numerics import code, parts127  # noqa: F401  (iface reads ops.parts127)
from .recipes import movement as mv
from .recipes import reduce as rd
from .recipes.conv import ConvGeom, conv2d
from .recipes.gemm import blocked_matmul
from .stream import ACC_BIT, FULL_BIT, Arg, Ins, Stream


@dataclass
class Kernel:
    op: str
    stream: Stream
    meta: dict = field(default_factory=dict)

    @property
    def args(self) -> list[Arg]:
        return [a for a in self.stream.args if a.role != "const"]


def _numel(shape) -> int:
    require(all(d > 0 for d in shape), "empty or dynamic tensor shape")
    return int(np.prod(shape)) if len(shape) else 1


def _begin(args, constants=None) -> Stream:
    from .match import Decline
    if any(any(d <= 0 for d in a.shape) for a in args):
        raise Decline("empty or dynamic tensor shape")
    s = Stream(list(args))
    for name, data in (constants or {}).items():
        s.bind_constant(name,data)
    s.fence()
    s.flush()
    return s


def _end(op, s, **meta) -> Kernel:
    s.fence()
    meta.update(device_arithmetic='exact_integer',quantize_rounding='host_exact_rational_RNE' if s.readouts else 'none',
                device_readout='full_i32' if s.readouts else 'exact_integer_storage')
    from .stream import CompactInstructions
    if isinstance(s.ins,CompactInstructions):
        require(len(s.ins) <= 1 << 20, 'DMA schedule exceeds bounded compiler instruction capacity')
        require(sum(a.nbytes for a in s.args) <= 64 << 20, 'DMA schedule exceeds bounded compiler byte capacity')
    for instruction in (s.ins.templates() if isinstance(s.ins,CompactInstructions) else s.ins):
        if instruction.kind == 'mvout' and instruction.a('local') & ACC_BIT:
            require(instruction.a('local') & FULL_BIT, 'accumulator readout requires full int32 width')
        if instruction.kind not in ('config_ld','config_st'): continue
        factor=instruction.a('scale') if instruction.kind=='config_ld' else instruction.a('acc_scale')
        require(factor == 1, 'nonunit device scale requires an exact integer schedule and host requantization')
    meta['fp32_product_proofs']=[]
    return Kernel(op, s, meta)


def _readout(s, name, scale, relu=False):
    from dataclasses import replace
    from .requant import Readout
    index = s._index[name]
    arg = s.args[index]
    raw = name + '__i32'
    s.args[index] = replace(arg,name=raw,dtype='i32',role='scratch')
    s._index[raw] = index
    s.args.append(replace(arg,dtype='i8',role='output'))
    # Instructions emitted using the original name address the raw tile.
    s.readouts.append(Readout(name,[(raw,Q(scale))],relu=relu))


# ---------------------------------------------------------------------------------------------
# Contractions
# ---------------------------------------------------------------------------------------------

def matmul(op: str, M: int, K: int, N: int, *, batch: int = 0, s_a: float = 1.0, s_b: float = 1.0,
           s_out: float | None = None, alpha: float = 1.0, bias: bool = False, relu: bool = False,
           bias_const: np.ndarray | None = None, bias_rows: bool = False,
           bias_range: tuple[int, int] | None = None, recipe=None, inputs=()) -> Kernel:
    """mm / bmm (batch > 0) / addmm (bias: int32 [N] in accumulator units, i.e. at scale alpha s_a s_b).

    ``bias_const`` makes the (compile-time quantized) bias a private constant instead of an argument.
    s_out None selects the raw int32 readout of the exact integer contraction.
    ``recipe`` replaces the blocked weight-stationary schedule with an argument-compatible one,
    and ``inputs`` declares the extra operands it reads; both come from an admitted composition.
    """
    require(min(M, K, N) > 0 and batch >= 0, "empty or dynamic contraction")
    require(s_out is not None or not relu,'raw i32 contraction cannot silently apply byte-readout ReLU')
    bound = K * 16384
    lo, hi = -bound, bound
    if bias:
        if bias_const is not None:
            literals=np.asarray(bias_const,dtype=object)
            require(all(int(v)==v and -(1<<31)<=v<(1<<31) for v in literals.flat),'seed is not exactly int32')
            values = np.asarray(bias_const, dtype=np.int64)
            lo += min(0, int(values.min()))
            hi += max(0, int(values.max()))
        else:
            require(bias_range is not None, "runtime i32 seed has no nonoverflow range proof")
            require(-(1<<31)<=bias_range[0]<=bias_range[1]<(1<<31),'invalid runtime seed range')
            lo += min(0, bias_range[0])
            hi += max(0, bias_range[1])
    require(-(1 << 31) <= lo <= hi < (1 << 31), "unproved i32 accumulator overflow bound")

    B = max(batch, 1)
    lead = (batch,) if batch else ()
    out_i32 = True
    args = [Arg("a", "i8", lead + (M, K), "input"), Arg("b", "i8", lead + (K, N), "input"), *inputs]
    if bias and bias_const is None:
        args.append(Arg("bias", "i32", lead + (M, N) if bias_rows else (N,), "input"))
    args.append(Arg("out", "i32" if out_i32 else "i8", lead + (M, N), "output"))
    s = _begin(args)
    if s_out is not None:
        _readout(s, "out", Q(alpha)*Q(s_a)*Q(s_b)/Q(s_out), relu)
    bname = None
    if bias:
        bname = "bias"
        if bias_const is not None:
            bname = "bias_c"
            s.const(bname, np.asarray(bias_const, np.int32).reshape(lead + (M, N) if bias_rows else (N,)))
    scale = 1.0
    for bi in range(B):
        if bi:
            # Each batch reuses the same local seed and operand tiles.
            s.fence()
        (recipe or blocked_matmul)(s, ("a", bi * M * K), K, ("b", bi * K * N), N, ("out", bi * M * N * (4 if out_i32 else 1)), N,
                       M, K, N, bias=(bname, bi * M * N * 4 if bias_rows else 0) if bias else None, scale=scale, relu=False, out_i32=out_i32,
                       configure=bi == 0, bias_rows=bias_rows)
    return _end(op, s, readout_scale=scale, accumulator_min=lo, accumulator_max=hi)


def _conv_body(s: Stream, g: ConvGeom, x, w, y, bname, scale, relu, out_i32):
    if (g.kh, g.kw, g.sh, g.sw, g.ph, g.pw, g.groups) == (1, 1, 1, 1, 0, 0, 1):
        M = g.n * g.h * g.w
        blocked_matmul(s, (x, 0), g.cin, (w, 0), g.cout, (y, 0), g.cout, M, g.cin, g.cout,
                       bias=(bname, 0) if bname else None, scale=scale, relu=relu, out_i32=out_i32)
    else:
        conv2d(s, g, x, w, y, bias=bname, scale=scale, relu=relu, out_i32=out_i32)


def convolution(g: ConvGeom, *, s_x: float = 1.0, s_w: float = 1.0, s_out: float | None = None,
                bias: bool = False, relu: bool = False, alpha: float = 1.0) -> Kernel:
    """Channels-last: NHWC x HWIO -> NHWC; bias int32 [Cout] at scale s_x s_w (PT2E)."""
    g.validate()
    require(s_out is not None or not relu,"raw i32 convolution cannot silently apply byte-readout ReLU")
    require(not bias, "runtime convolution seed has no nonoverflow range proof")
    bound = g.kh * g.kw * (g.cin // g.groups) * 16384
    accumulator_bound(bound)
    out_i32 = True
    args = [Arg("x", "i8", (g.n, g.h, g.w, g.cin), "input"),
            Arg("w", "i8", (g.kh, g.kw, g.cin // g.groups, g.cout), "input")]
    if bias:
        args.append(Arg("bias", "i32", (g.cout,), "input"))
    args.append(Arg("y", "i32" if out_i32 else "i8", (g.n, g.oh, g.ow, g.cout), "output"))
    s = _begin(args)
    if s_out is not None:
        _readout(s,"y",Q(alpha)*Q(s_x)*Q(s_w)/Q(s_out),relu)
    scale = 1.0
    _conv_body(s, g, "x", "w", "y", "bias" if bias else None, scale, False, out_i32)
    return _end("aten.convolution.default", s, readout_scale=scale, accumulator_min=-bound, accumulator_max=bound, layout="NHWC,HWIO->NHWC")


def convolution_nchw(g: ConvGeom, *, s_x: float, s_w: float, s_out: float, alpha: float = 1.0,
                     relu: bool = False, constants=None) -> Kernel:
    """The source layout (NCHW x OIHW -> NCHW) with on-device transposes around the channels-last
    convolution: x and w are transposed into caller-provided scratch, fenced (DRAM RAW), convolved,
    fenced, and the NHWC result transposed into y.  Merlin's layout propagation can bind the
    channels-last kernel (``convolution``) instead and drop the transposes."""
    g.validate()
    bound = g.kh * g.kw * g.cin * 16384
    accumulator_bound(bound)
    if g.groups != 1:
        raise ValueError("grouped convolution needs the channels-last kernel")
    args = [Arg("x", "i8", (g.n, g.cin, g.h, g.w), "input"), Arg("w", "i8", (g.cout, g.cin, g.kh, g.kw), "input"),
             Arg("y", "i8", (g.n, g.cout, g.oh, g.ow), "output"),
             Arg("ws_x", "i8", (g.n, g.h, g.w, g.cin), "scratch"),
             Arg("ws_w", "i8", (g.kh, g.kw, g.cin, g.cout), "scratch"),
             Arg("ws_y", "i32", (g.n, g.oh, g.ow, g.cout), "scratch")]
    s = _begin(args,constants)
    _permute_into(s, "x", [g.n, g.cin, g.h, g.w], "ws_x", [0, 2, 3, 1])
    _permute_into(s, "w", [g.cout, g.cin, g.kh, g.kw], "ws_w", [2, 3, 1, 0])
    s.fence()
    scale = Q(alpha) * Q(s_x) * Q(s_w) / Q(s_out)
    _conv_body(s, g, "ws_x", "ws_w", "ws_y", None, 1.0, False, True)
    from .requant import Readout
    s.readouts.append(Readout("y",[("ws_y",scale)],relu=relu,
        permutations={"ws_y":((g.n,g.oh,g.ow,g.cout),(0,3,1,2))}))
    return _end("aten.convolution.default", s, readout_scale=scale, accumulator_min=-bound, accumulator_max=bound, layout="NCHW,OIHW->NCHW via NHWC")


def conv_backward_input(g, *, s_g, s_w, s_out, constants=None):
    from .recipes.backward import conv_transpose
    from .requant import Readout
    g.validate()
    bound=g.kh*g.kw*g.cout*16384
    accumulator_bound(bound)
    args = [Arg('g','i8',(g.n,g.cout,g.oh,g.ow),'input'),
            Arg('w','i8',(g.cout,g.cin,g.kh,g.kw),'input'),
            Arg('y','i8',(g.n,g.cin,g.h,g.w),'output'),
            Arg('ws_g','i8',(g.n,g.oh,g.ow,g.cout),'scratch'),
            Arg('ws_w','i8',(g.kh,g.kw,g.cout,g.cin),'scratch'),
            Arg('acc','i32',(g.n,g.h,g.w,g.cin),'scratch')]
    s = _begin(args,constants)
    _permute_into(s,'g',args[0].shape,'ws_g',[0,2,3,1])
    _permute_into(s,'w',args[1].shape,'ws_w',[2,3,0,1])
    s.fence()
    conv_transpose(s,g,'ws_g','ws_w','acc',1,out_i32=True)
    s.readouts.append(Readout('y',[('acc',Q(s_g)*Q(s_w)/Q(s_out))],
       permutations={'acc':((g.n,g.h,g.w,g.cin),(0,3,1,2))}))
    return _end('aten.convolution_backward.default',s,output='grad_input',accumulator_min=-bound,accumulator_max=bound)


def conv_backward_weight(g, *, s_g, s_x, s_out, constants=None):
    from .recipes.backward import conv_wgrad
    from .requant import Readout
    g.validate()
    bound=g.n*g.oh*g.ow*16384
    accumulator_bound(bound)
    args = [Arg('g','i8',(g.n,g.cout,g.oh,g.ow),'input'),
            Arg('x','i8',(g.n,g.cin,g.h,g.w),'input'),
            Arg('y','i8',(g.cout,g.cin,g.kh,g.kw),'output'),
            Arg('ws_x','i8',(g.n,g.h,g.w,g.cin),'scratch'),
            Arg('acc','i32',(g.kh,g.kw,g.cout,g.cin),'scratch')]
    s = _begin(args,constants)
    _permute_into(s,'x',args[1].shape,'ws_x',[0,2,3,1]); s.fence()
    conv_wgrad(s,g,'g','ws_x','acc',1,out_i32=True)
    s.readouts.append(Readout('y',[('acc',Q(s_g)*Q(s_x)/Q(s_out))],
       permutations={'acc':((g.kh,g.kw,g.cout,g.cin),(2,3,0,1))}))
    return _end('aten.convolution_backward.default',s,output='grad_weight',accumulator_min=-bound,accumulator_max=bound)


# ---------------------------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------------------------

def _dense_linear(op, out_shape, inputs, constant=0, relu=False, *, code_units=None):
    from .linmap import LinearMap
    from .exact_linear import Src, kernel
    n = _numel(out_shape)
    require(n <= 1 << 16, 'dense arithmetic map exceeds bounded compiler capacity; use a structural schedule')
    outs, ids, indices, coefficients, srcs = [], [], [], [], {}
    for sid, (name, shp, index, coefficient) in enumerate(inputs):
        srcs[sid] = Src(name, tuple(shp), (code_units or {}).get(name, 1))
        outs.append(np.arange(n,dtype=np.int64))
        ids.append(np.full(n,sid,np.int64))
        indices.append(np.asarray(index,np.int64).reshape(-1))
        coefficients.append(np.broadcast_to(np.asarray(coefficient,dtype=object).reshape(-1),(n,)))
    lm = LinearMap(tuple(out_shape),np.concatenate(outs),np.concatenate(ids),np.concatenate(indices),
                   np.concatenate(coefficients),np.broadcast_to(np.asarray(constant,dtype=object).reshape(-1),(n,)),{})
    result = kernel(op,lm,srcs,1,relu=relu)
    return result


def scale_map(op, shape, c, *, s_x, s_out):
    return _dense_linear(op,shape,[('x',shape,np.arange(_numel(shape)),Q(c)*Q(s_x)/Q(s_out))], code_units={'x':Q(s_x)/Q(s_out)})


def relu(shape, *, s_x, s_out):
    return _dense_linear('aten.relu.default',shape,[('x',shape,np.arange(_numel(shape)),Q(s_x)/Q(s_out))],relu=True,code_units={'x':abs(Q(s_x)/Q(s_out))})


def add_tensor(op, shape, b_shape, *, alpha=1, s_a, s_b, s_out, wa=1, c=0, relu=False, b_lead=None):
    shape,b_shape = tuple(shape),tuple(b_shape)
    bidx = np.broadcast_to(np.arange(_numel(b_shape)).reshape(b_shape),shape).reshape(-1)
    return _dense_linear(op,shape,[('a',shape,np.arange(_numel(shape)),Q(wa)*Q(s_a)/Q(s_out)),
                                   ('b',b_shape,bidx,Q(alpha)*Q(s_b)/Q(s_out))],Q(c)/Q(s_out),relu, code_units={'a':Q(s_a)/Q(s_out),'b':Q(s_b)/Q(s_out)})


def add_scalar(op, shape, c, *, s_x, s_out, w=1, relu=False):
    return _dense_linear(op,shape,[('x',shape,np.arange(_numel(shape)),Q(w)*Q(s_x)/Q(s_out))],Q(c)/Q(s_out),relu, code_units={'x':Q(s_x)/Q(s_out)})


# ---------------------------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------------------------

def mean_dim(shape, dims, keepdim: bool, *, s_x: float, s_out: float, w: float | None = None,
             op: str = "aten.mean.dim") -> Kernel:
    """sum (w = 1) or mean (w = 1 / L, the default) over dims, one readout scale w s_x / s_out."""
    from .match import Decline
    if any(d <= 0 for d in shape):
        raise Decline("empty or dynamic reduction shape")
    r = len(shape)
    if not r or not dims or any(d < -r or d >= r for d in dims) or len(set(d % r for d in dims)) != len(dims):
        raise Decline("reduction needs nonempty, unique, in-range dimensions")
    dims = sorted(d % r for d in dims)
    # size-1 dims never break contiguity of the reduced block
    eff = [d for d in dims if shape[d] != 1] or dims[:1]
    if any(d not in dims and shape[d] != 1 for d in range(eff[0], eff[-1] + 1)):
        raise ValueError("mean over non-contiguous dims")
    O = _numel([shape[d] for d in range(eff[0]) if d not in dims])
    L = _numel([shape[d] for d in dims])
    I = _numel([shape[d] for d in range(eff[-1] + 1, r) if d not in dims])
    oshape = [1 if d in dims else shape[d] for d in range(r)] if keepdim else \
        [shape[d] for d in range(r) if d not in dims]
    s = _begin([Arg("x", "i8", tuple(shape), "input"), Arg("y", "i32", tuple(oshape), "output")])
    exact_scale = (Q(1) / L if w is None else Q(w)) * Q(s_x) / Q(s_out)
    accumulator_bound(L * 128)
    _readout(s,"y",exact_scale)
    sc = 1.0
    if I == 1:
        rd.mean_last(s, "x", "y", O, L, sc, out_i32=True)
    else:
        rd.mean_middle(s, "x", "y", O, L, I, sc, out_i32=True)
    return _end(op, s, readout_scale=sc, reduced=L, accumulator_min=-L*128, accumulator_max=L*127)


# ---------------------------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------------------------

def _plan_kernel(op, args, plan: mv.Plan, **meta) -> Kernel:
    s = _begin(args)
    mv.emit_plan(s, plan)
    return _end(op, s, **meta)


def clone(shape, *, s_x=1, s_out=1):
    from . import layout
    _numel(shape)
    return layout.quantized('aten.clone.default',[shape],shape,
        [(shape,layout.dense_strides(shape),0,layout.dense_strides(shape))],[Q(s_x)/Q(s_out)],names=['x'])


def permute(shape, perm, *, s_x=1, s_out=1):
    from . import layout
    _numel(shape)
    require(sorted(perm)==list(range(len(shape))), 'invalid movement permutation')
    out = tuple(shape[p] for p in perm)
    ss = layout.dense_strides(shape)
    return layout.quantized('aten.permute.default',[shape],out,
        [(out,tuple(ss[p] for p in perm),0,layout.dense_strides(out))],[Q(s_x)/Q(s_out)],names=['x'])


def _permute_into(s: Stream, x: str, shape, out: str, perm) -> None:
    """Emit a byte permutation using strided row or column DMA."""
    from . import layout
    require(x != out, 'internal layout permutation requires distinct buffers')
    require(sorted(perm)==list(range(len(shape))), 'invalid internal layout permutation')
    strides = layout.dense_strides(shape)
    layout.move(s,(x,0),(out,0),tuple(shape[p] for p in perm),tuple(strides[p] for p in perm))


def cat(shapes, dim, *, s_in=None, s_out=1):
    from . import layout
    scales = [1]*len(shapes) if s_in is None else s_in
    dim %= len(shapes[0])
    out_shape = list(shapes[0]); out_shape[dim] = sum(sh[dim] for sh in shapes)
    maps, placements = [], []
    offset = 0
    for sid,(sh,sc) in enumerate(zip(shapes,scales)):
        require(all(d==dim or sh[d]==out_shape[d] for d in range(len(sh))), 'concat non-axis shape mismatch')
        ds = layout.dense_strides(out_shape)
        maps.append((sh,layout.dense_strides(sh),offset*ds[dim],ds))
        placements.append((_numel(sh),sh[dim]*ds[dim],offset*ds[dim],out_shape[dim]*ds[dim]))
        offset += sh[dim]
    return layout.quantized('aten.cat.default',shapes,out_shape,maps,[Q(sc)/Q(s_out) for sc in scales],placements=placements)


def constant_pad_nd(shape, pad, value, *, s_x=1, s_out=1):
    from . import layout
    lohi = [(0,0)]*len(shape)
    for i in range(len(pad)//2): lohi[-1-i] = (pad[2*i],pad[2*i+1])
    out = tuple(d+lo+hi for d,(lo,hi) in zip(shape,lohi))
    _numel(shape);_numel(out)
    ss,ds = layout.dense_strides(shape),layout.dense_strides(out)
    start = [max(0,-lo) for lo,_ in lohi]
    extents = tuple(min(d,d+hi)-a for d,(_,hi),a in zip(shape,lohi,start))
    require(all(d>0 for d in extents),'padding removes the entire input')
    source_offset = sum(a*s for a,s in zip(start,ss))
    offset = sum((a+lo)*s for a,(lo,_),s in zip(start,lohi,ds))
    placement = dict(shape=extents,strides=ds,offset=offset)
    return layout.quantized('aten.constant_pad_nd.default',[shape],out,
        [(extents,ss,offset,ds,source_offset)],[Q(s_x)/Q(s_out)],names=['x'],placements=[placement],
        fill_code=code(value,s_out) if any(lo>0 or hi>0 for lo,hi in lohi) else None)


def slice_scatter(shape, src_shape, dim, start, end, step, *, s_self=1, s_src=1, s_out=1):
    from . import layout
    dim %= len(shape)
    require(step>0,'scatter requires positive static stride')
    start,end,step = slice(start,end,step).indices(shape[dim])
    size = len(range(start,end,step))
    copied = list(shape);copied[dim]=size
    require(_numel(src_shape)==_numel(copied),'scatter source extent mismatch')
    ds = layout.dense_strides(shape)
    target = list(ds);target[dim]*=step
    placement = dict(shape=tuple(copied),strides=tuple(target),offset=start*ds[dim])
    return layout.quantized('aten.slice_scatter.default',[shape,src_shape],shape,
        [(shape,ds,0,ds),(copied,layout.dense_strides(copied),start*ds[dim],target)],
        [Q(s_self)/Q(s_out),Q(s_src)/Q(s_out)],names=['self','src'],staged=True,
        placements=[None,placement])


def full(op: str, shape, value: float, *, s_out: float, like: bool = False) -> Kernel:
    n = _numel(shape)
    plan = mv.Plan()
    mv.full(plan, "y", n, code(value, s_out))
    args = ([Arg("x", "i8", tuple(shape), "input")] if like else []) + [Arg("y", "i8", tuple(shape), "output")]
    return _plan_kernel(op, args, plan, fill_code=code(value, s_out))


def to_copy_widen(shape) -> Kernel:
    """int8 -> int32 (the quantized codes widen exactly through the accumulator readout)."""
    n = _numel(shape)
    s = _begin([Arg("x", "i8", tuple(shape), "input"), Arg("y", "i32", tuple(shape), "output")])
    mv.emit_widen(s, "x", "y", n)
    return _end("aten._to_copy.default", s)


def contraction_zero_points(kernel, qa, qb, M, K, N, batch=0):
    """Exact centered-byte contraction plus integer row/column corrections.

    q-z = (q-128)+(128-z) for uint8; signed int8 needs only -z.
    No early byte saturation or float conversion participates in the folding.
    """
    s=kernel.stream
    names=[s.retarget_signed(n,s.args[s._index[n]].shape) if q.unsigned else n for n,q in zip('ab',(qa,qb))]
    oa,ob=code_offset(qa),code_offset(qb)
    if not oa and not ob: return
    readout=s.readouts[-1]
    scale=readout.terms[0][1]
    lead=(batch,) if batch else ()
    B=max(batch,1)
    for name,shp in [('row_sum',lead+(M,1)),('column_sum',lead+(1,N))]:
        s._index[name]=len(s.args);s.args.append(Arg(name,'i32',shp,'scratch'))
    # Appended after a completed kernel fence: raw matmul tiles are in DRAM.
    for bi in range(B):
        # The mean recipes address dense bases; use a temporary stream then
        # relocate its operand byte offsets for this independent batch.
        for axis,source,dest in [('row',names[0],'row_sum'),('column',names[1],'column_sum')]:
            # Independent reductions reuse scratchpad rows, including the
            # ones vector whose tail may not be touched by the last compute.
            s.fence()
            temporary=Stream(s.args.copy(),consts=s.consts.copy())
            if axis=='row': rd.mean_last(temporary,source,dest,M,K,1,out_i32=True)
            else: rd.mean_middle(temporary,source,dest,1,K,N,1,out_i32=True)
            for arg in temporary.args[len(s.args):]:
                s._index[arg.name]=len(s.args);s.args.append(arg);s.consts[arg.name]=temporary.consts[arg.name]
            aoff=bi*(M*K if axis=='row' else K*N)
            ooff=bi*(M if axis=='row' else N)*4
            for ins in temporary.ins:
                extra=aoff if ins.arg==s._index[source] else ooff if ins.arg==s._index[dest] else 0
                s.ins.append(Ins(ins.kind,ins.attrs,ins.arg,ins.off+extra))
    s.fence()
    if ob: readout.terms.append(('row_sum',Q(ob)*scale))
    if oa: readout.terms.append(('column_sum',Q(oa)*scale))
    readout.constant=readout.constant+K*oa*ob*scale
    readout.broadcasts={**(readout.broadcasts or {}),'row_sum':lead+(M,1),'column_sum':lead+(1,N)}


def convolution_zero_points(kernel, left, right, g, mode='forward'):
    """Exact zero-point folding through two auxiliary integer contractions.

    (a+oa)*(b+ob) = a*b + ob*sum(a) + oa*sum(b) + oa*ob*count.
    Unit tensors retain the spatial padding mask. Singleton channel/batch
    dimensions share the auxiliary sums through host broadcast; no source
    product is rounded, approximated or redundantly enumerated per channel.
    """
    from dataclasses import replace
    from .recipes.backward import conv_transpose,conv_wgrad
    s=kernel.stream
    qa,qb=left[1],right[1]
    oa,ob=code_offset(qa),code_offset(qb)
    if not oa and not ob:return
    output=next(a for a in kernel.args if a.role=='output')
    readout=s.readouts[-1]
    scale=readout.terms[0][1]
    def fold(offset,ones_name,ones_shape,name,shape,bound,emit,perm=None,broadcast=None):
        """One auxiliary integer contraction with a ones operand: offset * (sum of the other operand)."""
        if not offset: return
        s.const(ones_name,np.ones(ones_shape,dtype=np.int8))
        accumulator_bound(bound)
        s._index[name]=len(s.args)
        s.args.append(Arg(name,'i32',shape,'scratch'))
        readout.terms.append((name,Q(offset)*scale))
        if perm is not None:
            readout.permutations={**(readout.permutations or {}),name:(shape,perm)}
        if broadcast is not None:
            readout.broadcasts={**(readout.broadcasts or {}),name:broadcast}
        kernel.meta.setdefault('zero_point_corrections',[]).append(dict(tile=name,accumulator_min=-bound,accumulator_max=bound))
        emit();s.fence()
    if mode=='forward':
        # ws_x/ws_w are the exact centered NHWC/HWIO operands produced by
        # convolution_nchw, fenced before this auxiliary schedule begins.
        bound=g.kh*g.kw*g.cin*128
        fold(ob,'zp_ones_w',(g.kh,g.kw,g.cin,1),'zp_sum_x',(g.n,g.oh,g.ow,1),bound,broadcast=(g.n,1,g.oh,g.ow),
             emit=lambda:_conv_body(s,replace(g,cout=1),'ws_x','zp_ones_w','zp_sum_x',None,1,False,True))
        fold(oa,'zp_ones_x',(1,g.h,g.w,g.cin),'zp_sum_w',(1,g.oh,g.ow,g.cout),bound,perm=(0,3,1,2),
             emit=lambda:_conv_body(s,replace(g,n=1),'zp_ones_x','ws_w','zp_sum_w',None,1,False,True))
        oh,ow,kh,kw=np.indices((g.oh,g.ow,g.kh,g.kw))
        ih=oh*g.sh-g.ph+kh*g.dh;iw=ow*g.sw-g.pw+kw*g.dw
        counts=((ih>=0)&(ih<g.h)&(iw>=0)&(iw<g.w)).sum(axis=(2,3))*g.cin
        counts=counts.reshape((1,1,g.oh,g.ow))
    elif mode=='input':
        bound=g.kh*g.kw*g.cout*128
        fold(ob,'zp_ones_w',(g.kh,g.kw,g.cout,1),'zp_sum_g',(g.n,g.h,g.w,1),bound,broadcast=(g.n,1,g.h,g.w),
             emit=lambda:conv_transpose(s,replace(g,cin=1),'ws_g','zp_ones_w','zp_sum_g',1,out_i32=True))
        fold(oa,'zp_ones_g',(1,g.oh,g.ow,g.cout),'zp_sum_w',(1,g.h,g.w,g.cin),bound,perm=(0,3,1,2),
             emit=lambda:conv_transpose(s,replace(g,n=1),'zp_ones_g','ws_w','zp_sum_w',1,out_i32=True))
        ih,iw,kh,kw=np.indices((g.h,g.w,g.kh,g.kw))
        h=ih+g.ph-kh*g.dh;w=iw+g.pw-kw*g.dw
        oh,ow=h//g.sh,w//g.sw
        counts=((h%g.sh==0)&(w%g.sw==0)&(oh>=0)&(oh<g.oh)&(ow>=0)&(ow<g.ow)).sum(axis=(2,3))*g.cout
        counts=counts.reshape((1,1,g.h,g.w))
    else:
        bound=g.n*g.oh*g.ow*128
        fold(ob,'zp_ones_x',(g.n,g.h,g.w,1),'zp_sum_g',(g.kh,g.kw,g.cout,1),bound,perm=(2,3,0,1),
             emit=lambda:conv_wgrad(s,replace(g,cin=1),'g__signed' if qa.unsigned else 'g','zp_ones_x','zp_sum_g',1,out_i32=True))
        fold(oa,'zp_ones_g',(g.n,1,g.oh,g.ow),'zp_sum_x',(g.kh,g.kw,1,g.cin),bound,perm=(2,3,0,1),
             emit=lambda:conv_wgrad(s,replace(g,cout=1),'zp_ones_g','ws_x','zp_sum_x',1,out_i32=True))
        kh,kw,oh,ow=np.indices((g.kh,g.kw,g.oh,g.ow))
        ih=oh*g.sh-g.ph+kh*g.dh;iw=ow*g.sw-g.pw+kw*g.dw
        counts=((ih>=0)&(ih<g.h)&(iw>=0)&(iw<g.w)).sum(axis=(2,3))*g.n
        counts=counts.reshape((1,1,g.kh,g.kw))
    readout.constant=readout.constant+np.broadcast_to(counts,output.shape)*oa*ob*scale
