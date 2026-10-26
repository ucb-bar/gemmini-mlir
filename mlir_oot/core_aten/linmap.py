"""Exact linear-map extraction from linalg-on-tensors regions (abstract interpretation).

model2MLIR emits pools, their backwards, pads, ``repeat``, scatters and ``col2im`` as gather
loops: ``linalg.generic`` bodies computing source indices from ``linalg.index`` with integer
arithmetic, ``scf.for``/``scf.if`` and ``tensor.extract``, sometimes fed by index tensors
computed by other generics.  Such a region is linear in its tensor inputs.  This module
evaluates the region once, vectorized over the whole iteration space: integer/index values
are numpy arrays, and every float value is a symbolic linear form ``sum coef * input[idx] +
const`` whose coefficient and index arrays span the iteration space.  Loops with
point-dependent bounds run to the maximum trip count under a mask.  The result is the
region's exact sparse linear map from its inputs (dequantized tensors or plain integer
tensors) to its output, which ``exact_linear`` schedules on the device.

Anything the interpreter does not model exactly (memory effects, non-linear float ops,
data-dependent indices) raises ``NotLinear``; the region then stays on the host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as Q

import numpy as np
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, FloatAttr, IntegerAttr, TensorType
from xdsl.ir import BlockArgument, Operation, OpResult, SSAValue


class NotLinear(Exception):
    pass


class _UninterpretedBits:
    """A pure bit manipulation which must be discarded by a proved selector."""


def _finite_byte_form(value, dtype):
    """Prove no NaN/Inf for a rational affine form over finite storage bytes."""
    formats = {'f16': (11,15), 'bf16': (8,127), 'f32': (24,127), 'f64': (53,1023)}
    if dtype not in formats or isinstance(value, _UninterpretedBits):
        return False
    if isinstance(value, Sym):
        bound = np.abs(np.asarray(value.const, object))
        # Sym leaves do not retain storage signedness. Cover both byte domains:
        # [-128, 127] and [0, 255], including extui/DQ unsigned leaves.
        byte_magnitude = max(abs(np.iinfo(np.int8).min), np.iinfo(np.uint8).max)
        for _,_,coefficient in value.terms:
            bound = bound + byte_magnitude*np.abs(coefficient)
    else:
        bound = np.abs(np.asarray(value, object))
    precision, exponent = formats[dtype]
    maximum = (Q(2)-Q(1,1 << (precision-1)))*(1 << exponent)
    return all(abs(v) <= maximum for v in np.asarray(bound).flat)


def exact_float_widening(source, target):
    formats = {'f16': (11, -24, 15), 'bf16': (8, -133, 127),
               'f32': (24, -149, 127), 'f64': (53, -1074, 1023)}
    if source not in formats or target not in formats:
        return False
    p, low, high = formats[source]
    q, minimum, maximum = formats[target]
    return q >= p and minimum <= low and maximum >= high


def inverse_float_widening(op):
    """Recognize a literal inverse of an exact widening, including cast generics."""
    value = op.operands[0]
    if isinstance(value, BlockArgument):
        block = value.owner
        parent = block.parent_op()
        if parent is None or parent.name != 'linalg.generic' or len(parent.results) != 1:
            return False
        if (value.index >= len(parent.operands)-1 or
                not all(_identity_like(m.data) for m in _attr(parent, 'indexing_maps')) or
                _shape(parent.operands[value.index]) != _shape(parent.results[0])):
            return False
        value = parent.operands[value.index]
    if not isinstance(value, OpResult):
        return False
    producer = value.owner
    if producer.name == 'linalg.generic' and len(producer.operands) == 2:
        maps = _attr(producer, 'indexing_maps')
        body = list(producer.regions[0].blocks[0].ops)
        if not all(_identity_like(m.data) for m in maps) or len(body) != 2:
            return False
        cast, yielded = body
        if (cast.name != 'arith.extf' or yielded.name != 'linalg.yield'
                or list(cast.operands) != [producer.regions[0].blocks[0].args[0]]
                or list(yielded.operands) != [cast.results[0]]):
            return False
    elif producer.name == 'arith.extf':
        cast = producer
    else:
        return False
    source, target = str(cast.operands[0].type), str(cast.results[0].type)
    return exact_float_widening(source, target) and source == str(op.results[0].type)


@dataclass
class Sym:
    """Elementwise symbolic linear form over region inputs; arrays share one shape."""

    terms: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)   # (src, idx, coef)
    const: np.ndarray | float = 0

    @property
    def shape(self):
        if self.terms:
            return self.terms[0][1].shape
        return np.shape(self.const)

    def bshape(self, shp):
        return Sym([(s, np.broadcast_to(i, shp), np.broadcast_to(c, shp)) for s, i, c in self.terms],
                   np.broadcast_to(np.asarray(self.const, object), shp))

    def scale(self, k):
        return Sym([(s, i, c * k) for s, i, c in self.terms], self.const * k)

    def add(self, other, k=1):
        if isinstance(other, Sym):
            return Sym(self.terms + [(s, i, c * k) for s, i, c in other.terms], self.const + k * other.const)
        return Sym(list(self.terms), self.const + k * np.asarray(other, object))

    def take(self, index):
        return Sym([(s, np.asarray(i)[index], np.broadcast_to(c, np.shape(i))[index]) for s, i, c in self.terms], np.asarray(self.const)[index]
                   if np.ndim(self.const) else self.const)

    def reshape(self, shp):
        return Sym([(s, np.asarray(i).reshape(shp), np.broadcast_to(c, np.shape(i)).reshape(shp)) for s, i, c in self.terms],
                   np.asarray(self.const).reshape(shp) if np.ndim(self.const) else np.full(shp, self.const))

    def compact(self):
        """Drop all-zero terms and merge terms of one source with identical index arrays."""
        out = []
        for s, i, c in self.terms:
            if not np.any(c):
                continue
            for k, (s2, i2, c2) in enumerate(out):
                if s2 == s and i2.shape == i.shape and np.array_equal(i2, i):
                    out[k] = (s, i2, c2 + c)
                    break
            else:
                out.append((s, i, c))
        return Sym(out, self.const)


def where(cond, a, b):
    cond = np.asarray(cond, bool)
    if not isinstance(a, Sym) and not isinstance(b, Sym):
        return np.where(cond, a, b)
    sa = a if isinstance(a, Sym) else Sym([], np.asarray(a, object))
    sb = b if isinstance(b, Sym) else Sym([], np.asarray(b, object))
    shp = np.broadcast_shapes(cond.shape, sa.shape, sb.shape)
    sa, sb, cond = sa.bshape(shp), sb.bshape(shp), np.broadcast_to(cond, shp)
    terms = [(s, i, np.where(cond, c, 0)) for s, i, c in sa.terms] + \
            [(s, i, np.where(cond, 0, c)) for s, i, c in sb.terms]
    return Sym(terms, np.where(cond, sa.const, sb.const)).compact()


def selector(value, shape, check=lambda sid: None, compact=False):
    """(which, indices, flat constants) when every output element reads at most one source
    element with a unit coefficient (which -1: none), else None. ``check(sid)`` runs per term."""
    if not isinstance(value, Sym):
        value = Sym([], np.asarray(value, dtype=object))
    value = value.bshape(shape).compact() if compact else value.bshape(shape)
    n = int(np.prod(shape)) if shape else 1
    which, indices = np.full(n, -1, np.int64), np.zeros(n, np.int64)
    for sid, index, coefficient in value.terms:
        c = np.asarray(coefficient).reshape(-1)
        active = c != 0
        if not (np.all(c[active] == 1) and np.all(which[active] == -1)):
            return None
        check(sid)
        which[active], indices[active] = sid, np.asarray(index).reshape(-1)[active]
    return which, indices, np.asarray(value.const, dtype=object).reshape(-1)


@dataclass
class Scatter:
    """A tensor built by an accumulate-scatter loop: base + sum of value[k] at out[k] (flat)."""

    shape: tuple
    base: object                     # Sym or concrete array of ``shape``
    updates: list                    # [(flat out index array, Sym of the same shape)]


def _scatter_to_sym(shape, base, updates) -> Sym:
    """Materialize scatter contributions as dense per-element terms (rank within each output)."""
    n = int(np.prod(shape)) if shape else 1
    out = base if isinstance(base, Sym) else Sym([], np.asarray(base, object))
    out = out.bshape(shape)
    terms = list(out.terms)
    const = np.array(out.const, object).reshape(-1).copy()
    for flat, upd in updates:
        flat = np.asarray(flat, np.int64).reshape(-1)
        np.add.at(const, flat, np.asarray(upd.const, object).reshape(-1) if np.ndim(upd.const) else upd.const)
        for src, idx, coef in upd.terms:
            idx, coef = np.asarray(idx).reshape(-1), np.asarray(coef, object).reshape(-1)
            keep = coef != 0
            f, i, c = flat[keep], idx[keep], coef[keep]
            order = np.argsort(f, kind="stable")
            f, i, c = f[order], i[order], c[order]
            starts = np.r_[0, np.flatnonzero(np.diff(f)) + 1]
            rank = np.arange(f.size) - np.repeat(starts, np.diff(np.r_[starts, f.size]))
            for r in range(int(rank.max()) + 1 if f.size else 0):
                sel = rank == r
                ti = np.zeros(n, np.int64)
                tc = np.zeros(n, object)
                ti[f[sel]] = i[sel]
                tc[f[sel]] = c[sel]
                terms.append((src, ti.reshape(shape), tc.reshape(shape)))
    return Sym(terms, const.reshape(shape))


def _shape(v: SSAValue):
    t = v.type
    return tuple(int(x) for x in t.get_shape()) if isinstance(t, TensorType) else None


def _attr(op, key):
    a = op.properties.get(key)
    return op.attributes.get(key) if a is None else a


def _const(op):
    a = _attr(op, "value")
    if isinstance(a, (IntegerAttr, FloatAttr)):
        v = a.value.data
        return Q(v) if isinstance(a, FloatAttr) else int(v)
    if isinstance(a, DenseIntOrFPElementsAttr):
        vals = np.array([Q(v) if isinstance(v, float) else v for v in a.get_values()], dtype=object)
        return vals.reshape(_shape(op.results[0]))
    raise NotLinear(f"constant {a}")


def _is_float(v: SSAValue) -> bool:
    t = v.type
    if isinstance(t, TensorType):
        t = t.get_element_type()
    return str(t) in ("f32", "f64", "f16", "bf16")


_CMPI = {0: np.equal, 1: np.not_equal, 2: np.less, 3: np.less_equal, 4: np.greater, 5: np.greater_equal,
         6: np.less, 7: np.less_equal, 8: np.greater, 9: np.greater_equal}
_CMPF = {1: np.equal, 2: np.greater, 3: np.greater_equal, 4: np.less, 5: np.less_equal, 6: np.not_equal,
         8: np.equal, 9: np.greater, 10: np.greater_equal, 11: np.less, 12: np.less_equal, 13: np.not_equal}


class Region:
    """Evaluate the producers of ``root`` back to leaves.

    ``leaf(v)`` returns (source id, coefficient) for a value that is a region input (a
    dequantized tensor: coefficient = its scale; a plain integer tensor: 1), or None.
    """

    sum_operations = ('arith.addf',)

    def __init__(self, leaf, max_points: int = 1 << 23):
        self.leaf = leaf
        self.memo: dict[SSAValue, object] = {}
        self.ops: list[Operation] = []
        self.sources: dict[int, SSAValue] = {}
        self.max_points = max_points

    # -- tensor level ------------------------------------------------------------------------
    def tensor(self, v: SSAValue):
        if v in self.memo:
            return self.memo[v]
        lf = self.leaf(v)
        if isinstance(lf, np.ndarray):
            self.memo[v] = lf
            return lf
        if lf is not None:
            sid, coef, *offset = lf
            shp = _shape(v)
            n = int(np.prod(shp)) if shp else 1
            self.sources[sid] = v
            val = Sym([(sid, np.arange(n, dtype=np.int64).reshape(shp), np.full(shp, Q(coef), dtype=object))],
                      np.full(shp,offset[0] if offset else 0,dtype=object))
            self.memo[v] = val
            return val
        if isinstance(v, BlockArgument):
            raise NotLinear("unbound block argument")
        op = v.owner
        val = self._eval_op(op, list(op.results).index(v))
        if isinstance(val, _UninterpretedBits):
            raise NotLinear('unproved floating payload bits affect tensor output')
        self.memo[v] = val
        if op not in self.ops:
            self.ops.append(op)
        return val

    def _eval_op(self, op: Operation, result: int):
        nm = op.name
        if nm == "arith.constant":
            return np.asarray(_const(op))
        if nm == "tensor.splat":
            s = self.tensor(op.operands[0])
            return np.full(_shape(op.results[0]), s)
        if nm == "tensor.empty":
            return np.zeros(_shape(op.results[0]), dtype=object)
        if nm == "tensor.extract":
            return self._scalar(op, [self.tensor(v) for v in op.operands], {}, [], np.asarray(True))
        if nm == "tensor.from_elements":
            shp = _shape(op.results[0])
            values = [self.tensor(v) for v in op.operands]
            if any((v.shape if isinstance(v, Sym) else np.shape(v)) != () for v in values):
                raise NotLinear("from_elements requires scalar elements")
            if not any(isinstance(v, Sym) for v in values):
                return np.asarray(values, dtype=object).reshape(shp)
            # Embed each scalar linear form in its own output element. This
            # covers rank-zero reduction readouts and general scalar assembly.
            terms, constants = [], []
            for position, value in enumerate(values):
                if not isinstance(value, Sym):
                    value = Sym([], value)
                constants.append(np.asarray(value.const).item())
                for sid, index, coefficient in value.terms:
                    indices = np.zeros(len(values), dtype=np.int64)
                    coefficients = np.zeros(len(values), dtype=object)
                    indices[position] = np.asarray(index).item()
                    coefficients[position] = np.asarray(coefficient).item()
                    terms.append((sid, indices.reshape(shp), coefficients.reshape(shp)))
            return Sym(terms, np.asarray(constants, dtype=object).reshape(shp))
        if nm in ("tensor.reshape", "tensor.collapse_shape", "tensor.expand_shape", "tensor.cast"):
            from .views import reindex_shape
            try:
                shp = reindex_shape(op)
            except ValueError as exc:
                raise NotLinear(str(exc)) from exc
            x = self.tensor(op.operands[0])
            if isinstance(x, Scatter):
                return Scatter(shp, x.base.reshape(shp) if isinstance(x.base, Sym) else np.asarray(x.base).reshape(shp),
                               x.updates)
            return x.reshape(shp) if isinstance(x, Sym) else np.asarray(x).reshape(shp)
        if nm == 'linalg.reduce':
            block=op.regions[0].blocks[0]
            body=list(block.ops)
            if len(body)!=2 or body[0].name not in self.sum_operations or body[1].name!='linalg.yield' or set(body[0].operands)!=set(block.args) or list(body[1].operands)!=[body[0].results[0]]:
                raise NotLinear('reduction is not a closed sum')
            value=self.tensor(op.operands[0]);init=self.tensor(op.operands[1])
            dims=tuple(_attr(op,'dimensions').iter_values())
            full_shape=_shape(op.operands[0]);remaining=[i for i in range(len(full_shape)) if i not in dims]
            out_shape=tuple(full_shape[i] for i in remaining)
            def moved(a):
                return np.moveaxis(np.broadcast_to(a,full_shape),dims,list(range(len(remaining),len(full_shape)))).reshape(out_shape+(-1,))
            if isinstance(value,Sym):
                terms=[]
                for sid,idx,coef in value.terms:
                    i,c=moved(idx),moved(coef)
                    terms.extend((sid,i[...,k],c[...,k]) for k in range(i.shape[-1]))
                total=Sym(terms,moved(value.const).sum(-1))
                return total.add(init)
            return moved(value).sum(-1)+init
        if nm == "linalg.generic":
            return self._generic(op)
        if nm == "linalg.broadcast":
            x = self.tensor(op.operands[0])
            bdims = [int(d) for d in _attr(op, "dimensions").iter_values()]
            shp = _shape(op.results[0])
            idx = np.indices(shp)
            src = tuple(idx[d] for d in range(len(shp)) if d not in bdims)
            return x.take(src) if isinstance(x, Sym) else np.asarray(x)[src]
        if nm == "linalg.transpose":
            x = self.tensor(op.operands[0])
            perm = [int(p) for p in _attr(op, "permutation").iter_values()]
            idx = np.indices(_shape(op.results[0]))
            src = tuple(idx[perm.index(d)] for d in range(len(perm)))
            return x.take(src) if isinstance(x, Sym) else np.asarray(x)[src]
        if nm == "tensor.extract_slice" and len(op.operands) == 1:
            x = self.tensor(op.operands[0])
            offs = list(_attr(op, "static_offsets").iter_values())
            strides = list(_attr(op, "static_strides").iter_values())
            idx = np.indices(_shape(op.results[0]))
            src = tuple(o + s * i for o, s, i in zip(offs, strides, idx))
            return x.take(src) if isinstance(x, Sym) else np.asarray(x)[src]
        if nm == "tensor.insert_slice" and len(op.operands) == 2:
            s = self.tensor(op.operands[0])
            d = self.tensor(op.operands[1])
            offs = list(_attr(op, "static_offsets").iter_values())
            strides = list(_attr(op, "static_strides").iter_values())
            shp = _shape(op.results[0])
            idx = np.indices(shp)
            inside = np.ones(shp, bool)
            rel = []
            for k, (o, st) in enumerate(zip(offs, strides)):
                r = idx[k] - o
                inside &= (r >= 0) & (r % st == 0) & (r // st < _shape(op.operands[0])[k])
                rel.append(np.clip(r // st, 0, _shape(op.operands[0])[k] - 1))
            sv = s.take(tuple(rel)) if isinstance(s, Sym) else np.asarray(s)[tuple(rel)]
            return where(inside, sv, d)
        if nm == "tensor.concat":
            dim = int(_attr(op, "dim").value.data)
            parts = [self.tensor(x) for x in op.operands]
            shp = _shape(op.results[0])
            idx = np.indices(shp)
            out = np.zeros(shp, dtype=object)
            start = 0
            for x, p in zip(op.operands, parts):
                n = _shape(x)[dim]
                sel = (idx[dim] >= start) & (idx[dim] < start + n)
                src = list(idx)
                src[dim] = np.clip(idx[dim] - start, 0, n - 1)
                pv = p.take(tuple(src)) if isinstance(p, Sym) else np.asarray(p)[tuple(src)]
                out = where(sel, pv, out)
                start += n
            return out
        if nm == "scf.for":
            return self._scatter_loop(op)
        raise NotLinear(f"unsupported op {nm}")

    def _scatter_loop(self, op: Operation):
        """``for iv: T = insert(extract(T, idx(iv)) + v(iv), T, idx(iv))`` vectorized over iv."""
        if len(op.operands) != 4 or len(op.results) != 1:
            raise NotLinear("tensor loop with several carried values")
        lb, ub, st = (int(np.asarray(self.tensor(v))) for v in op.operands[:3])
        base = self.tensor(op.operands[3])
        shp = _shape(op.results[0])
        body = op.regions[0].blocks[0]
        ivs = np.arange(lb, ub, st, dtype=np.int64)
        if ivs.size > self.max_points:
            raise NotLinear("scatter loop too long to extract")
        carried = body.args[1]
        env = {body.args[0]: ivs, carried: ("carried",)}
        read = None
        upd = None
        mask = np.ones(ivs.shape, bool)
        for o in body.ops:
            nm = o.name
            if nm == "cf.assert":
                condition=env[o.operands[0]] if o.operands[0] in env else self._outer(o.operands[0])
                if isinstance(condition,Sym) or np.any(mask & ~np.asarray(condition,dtype=bool)):
                    raise NotLinear('source assertion is not proved for every active lane')
                continue
            if nm == "scf.yield":
                if upd is None or env.get(o.operands[0]) != ("inserted",):
                    raise NotLinear("loop does not yield its scatter")
                return _scatter_to_sym(shp, base, [upd])
            if nm == "tensor.extract" and o.operands[0] is carried:
                idx = tuple(np.asarray(env[v], np.int64) for v in o.operands[1:])
                read = (tuple(o.operands[1:]), idx)
                env[o.results[0]] = ("read",)
                continue
            if nm == "arith.addf" and ("read",) in [env.get(v) for v in o.operands]:
                other = o.operands[1] if env.get(o.operands[0]) == ("read",) else o.operands[0]
                env[o.results[0]] = ("sum", env[other])
                continue
            if nm == "tensor.insert":
                v, dest = o.operands[0], o.operands[1]
                if dest is not carried or not isinstance(env.get(v), tuple) or env[v][0] != "sum" or read is None \
                        or tuple(o.operands[2:]) != read[0]:
                    raise NotLinear("insert is not an accumulate of the extracted element")
                flat = np.ravel_multi_index(read[1], shp)
                val = env[v][1]
                val = val if isinstance(val, Sym) else Sym([], np.asarray(val, object))
                upd = (flat, val.bshape(ivs.shape))
                env[o.results[0]] = ("inserted",)
                continue
            vals = [env[x] if x in env else self._outer(x) for x in o.operands]
            if any(isinstance(x, tuple) for x in vals):
                raise NotLinear("carried tensor used outside the accumulate idiom")
            out = self._scalar(o, vals, env, [ivs], mask)
            for r, x in zip(o.results, out if isinstance(out, list) else [out]):
                env[r] = x
        raise NotLinear("loop without yield")

    # -- generic bodies ----------------------------------------------------------------------
    def _generic(self, op: Operation):
        its = [str(getattr(x.data, "value", x.data)) for x in _attr(op, "iterator_types").data]
        maps = [m.data for m in _attr(op, "indexing_maps").data]
        n_in = len(op.operands) - len(op.results)
        if len(op.results) != 1:
            raise NotLinear("multi-result generic")
        dims = [None] * maps[-1].num_dims
        for m, x in zip(maps, op.operands):
            for r, e in enumerate(m.results):
                if hasattr(e, "position"):
                    dims[e.position] = _shape(x)[r]
        if any(d is None for d in dims):
            raise NotLinear("unbounded iteration dim")
        if int(np.prod(dims)) > self.max_points:
            raise NotLinear("iteration space too large to extract")
        red = [d for d, it in enumerate(its) if it == "reduction"]
        if any(it not in ("parallel", "reduction") for it in its):
            raise NotLinear("iterator kind")
        ivs = list(np.indices(dims, dtype=np.int64))
        env = {}
        block = op.regions[0].blocks[0]
        out_arg = block.args[n_in]
        for k, (x, m) in enumerate(zip(op.operands, maps)):
            if k >= n_in:
                env[block.args[k]] = ("acc",) if red else None
                continue
            t = self.tensor(x)
            idx = tuple(_affine(m, ivs, dims))
            if any(np.any((i < 0) | (i >= extent)) for i, extent in zip(idx, _shape(x))):
                raise NotLinear('out-of-bounds generic input map')
            env[block.args[k]] = t.take(idx) if isinstance(t, Sym) else np.asarray(t)[idx]
        if not red:
            res = self._block(block, env, ivs, np.ones(dims, bool))
            if not _identity_like(maps[-1]):
                raise NotLinear("non-identity output map")
            return res[0]
        # reduction: the body must be  yield addf(acc, v)  with v independent of acc
        ops_ = list(block.ops)
        y = ops_[-1]
        add = y.operands[0].owner if isinstance(y.operands[0], OpResult) else None
        if add is None or add.name not in ("arith.addf", "arith.addi") or out_arg not in add.operands:
            raise NotLinear("reduction is not a sum")
        v_ssa = add.operands[1] if add.operands[0] is out_arg else add.operands[0]
        for o in ops_[:-1]:
            if o is add:
                continue
            if out_arg in o.operands:
                raise NotLinear("accumulator used in the reduction body")
            vals = [env[w] if w in env else self._outer(w) for w in o.operands]
            out = self._scalar(o, vals, env, ivs, np.ones(dims, bool))
            for r, w in zip(o.results, out if isinstance(out, list) else [out]):
                env[r] = w
        v = env[v_ssa] if v_ssa in env else self._outer(v_ssa)
        init = self.tensor(op.operands[n_in])
        par = [d for d in range(len(dims)) if d not in red]
        omap = maps[-1]
        # the output map must be the identity on the parallel dims
        res = [getattr(e, "position", None) for e in omap.results]
        if res != par:
            raise NotLinear("reduction output map")
        pshape = tuple(dims[d] for d in par)
        moved = (lambda a: np.moveaxis(np.broadcast_to(a, dims), red, list(range(len(par), len(dims))))
                 .reshape(pshape + (-1,)))
        if isinstance(v, Sym):
            terms = []
            for s_, i, c in v.bshape(tuple(dims)).terms:
                ii, cc = moved(i), moved(c)
                for k in range(ii.shape[-1]):
                    terms.append((s_, ii[..., k], cc[..., k]))
            total = Sym(terms, moved(v.bshape(tuple(dims)).const).sum(-1))
        else:
            total = moved(np.asarray(v, object)).sum(-1)
        if isinstance(init, Sym) or isinstance(total, Sym):
            base = init if isinstance(init, Sym) else Sym([], np.asarray(init, object))
            return base.bshape(pshape).add(total if isinstance(total, Sym) else Sym([], total))
        return np.asarray(init, object) + total

    def _block(self, block, env, ivs, mask):
        for op in block.ops:
            nm = op.name
            if nm in ("linalg.yield", "scf.yield"):
                return [env[v] for v in op.operands]
            if nm == "cf.assert":
                condition=env[op.operands[0]] if op.operands[0] in env else self._outer(op.operands[0])
                if isinstance(condition,Sym) or np.any(mask & ~np.asarray(condition,dtype=bool)):
                    raise NotLinear('source assertion is not proved for every active lane')
                continue
            vals = [env[v] if v in env else self._outer(v) for v in op.operands]
            out = self._scalar(op, vals, env, ivs, mask)
            for r, v in zip(op.results, out if isinstance(out, list) else [out]):
                env[r] = v
        raise NotLinear("block without terminator")

    def _outer(self, v):
        """A value defined outside the generic (captured tensor or scalar)."""
        return self.tensor(v)

    def _scalar(self, op, vals, env, ivs, mask):
        # Native frontend add/sub now carries FMA and NaN payload selectors.
        # Under this interpreter's exact-rational quantized contract FMA is a
        # product plus an add, with no extra rounding boundary. Payload bit
        # manipulation may disappear only behind a statically proved choice.
        if op.name == 'arith.select' and not isinstance(vals[0], (Sym, _UninterpretedBits)):
            condition = np.asarray(vals[0], bool)
            if np.all(condition):
                return vals[1]
            if not np.any(condition):
                return vals[2]
        if op.name == 'arith.cmpf' and int(_attr(op, 'predicate').value.data) == 14:
            dtype = str(op.operands[0].type)
            if all(_finite_byte_form(v,dtype) for v in vals):
                return np.zeros(np.broadcast_shapes(*(v.shape if isinstance(v,Sym) else np.shape(v) for v in vals)),bool)
            raise NotLinear('unordered float comparison has no finite byte-domain proof')
        if op.name == 'arith.bitcast':
            if isinstance(vals[0], (Sym, _UninterpretedBits)):
                return _UninterpretedBits()
            import struct
            source, target = str(op.operands[0].type), str(op.results[0].type)
            formats = {'f16': ('e',16), 'f32': ('f',32), 'f64': ('d',64)}
            if source in formats or target in formats:
                fmt,bits = formats.get(source,formats.get(target))
                def cast(v):
                    if source in formats:
                        raw = int.from_bytes(struct.pack('<'+fmt,float(v)), 'little')
                        return raw-(1 << bits) if raw >= (1 << (bits-1)) else raw
                    result = struct.unpack('<'+fmt,(int(v) % (1 << bits)).to_bytes(bits//8,'little'))[0]
                    return Q(result) if np.isfinite(result) else _UninterpretedBits()
                result = np.asarray([cast(v) for v in np.asarray(vals[0]).flat],object).reshape(np.shape(vals[0]))
                return _UninterpretedBits() if any(isinstance(v,_UninterpretedBits) for v in result.flat) else result
        if any(isinstance(v, _UninterpretedBits) for v in vals):
            if op.name == 'arith.ori':
                return _UninterpretedBits()
            raise NotLinear('unproved floating payload bits affect data')
        if op.name == 'math.fma':
            a,b,c = vals
            if isinstance(a,Sym) and isinstance(b,Sym):
                raise NotLinear('product of two inputs')
            product = a.scale(np.asarray(b,object)) if isinstance(a,Sym) else b.scale(np.asarray(a,object)) if isinstance(b,Sym) else np.asarray(a,object)*np.asarray(b,object)
            return (product if isinstance(product,Sym) else Sym([],product)).add(c) if isinstance(product,Sym) or isinstance(c,Sym) else product+c
        # Index arithmetic must not silently wrap numpy int64 or round through
        # floating division. A static cast/wrap we do not prove stays on host.
        integer_ops = {'arith.addi', 'arith.subi', 'arith.muli', 'arith.divsi', 'arith.divui',
                       'arith.floordivsi', 'arith.ceildivsi', 'arith.remsi', 'arith.remui',
                       'arith.extsi', 'arith.extui', 'arith.trunci', 'arith.index_cast', 'arith.index_castui'}
        checked = op.name in integer_ops and not any(isinstance(v, Sym) for v in vals)
        if checked:
            vals = [np.asarray(v, object) for v in vals]
            if op.name in ('arith.extui', 'arith.index_castui') and np.any(vals[0] < 0):
                raise NotLinear('static unsigned cast needs a nonnegative range proof')
            if op.name in ('arith.divsi', 'arith.divui', 'arith.floordivsi', 'arith.ceildivsi', 'arith.remsi', 'arith.remui') and np.any(vals[1] == 0):
                raise NotLinear('zero divisor in static integer index arithmetic')
        value = self._scalar_impl(op, vals, env, ivs, mask)
        if checked and op.results:
            dtype = str(op.results[0].type)
            bits = 64 if dtype == 'index' else int(dtype[1:]) if dtype.startswith('i') and dtype[1:].isdigit() else None
            if bits is not None:
                lower, upper = (0, 2) if bits == 1 else (-(1 << (bits-1)), 1 << (bits-1))
                data = np.asarray(value, object)
                if np.any(data < lower) or np.any(data >= upper):
                    raise NotLinear('static integer cast/arithmetic needs a nonoverflow proof')
        return value

    def _scalar_impl(self, op, vals, env, ivs, mask):
        nm = op.name
        if nm == "arith.constant":
            return np.asarray(_const(op))
        if nm == "linalg.index":
            return ivs[int(_attr(op, "dim").value.data)]
        if nm in ("arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui", "arith.trunci"):
            return vals[0]
        if nm in ("arith.sitofp", "arith.uitofp"):
            return np.asarray(vals[0], object)
        if nm in ("arith.extf", "arith.truncf"):
            if (isinstance(vals[0], Sym)
                    and not exact_float_widening(str(op.operands[0].type), str(op.results[0].type))
                    and not inverse_float_widening(op)):
                raise NotLinear('floating narrowing inside a quantized region needs an exactness proof')
            return vals[0]
        if nm in ("arith.fptosi", "arith.fptoui"):
            if isinstance(vals[0], Sym):
                raise NotLinear("data-dependent integer")
            return np.trunc(vals[0]).astype(np.int64)
        a = vals[0] if vals else None
        b = vals[1] if len(vals) > 1 else None
        if isinstance(a, Sym) or isinstance(b, Sym):
            if nm == "arith.addf":
                return (a if isinstance(a, Sym) else Sym([], a)).add(b)
            if nm == "arith.subf":
                return (a if isinstance(a, Sym) else Sym([], np.asarray(a, object))).add(b, -1)
            if nm == "arith.mulf":
                if isinstance(a, Sym) and not isinstance(b, Sym):
                    return a.scale(np.asarray(b, object))
                if isinstance(b, Sym) and not isinstance(a, Sym):
                    return b.scale(np.asarray(a, object))
                raise NotLinear("product of two inputs")
            if nm == "arith.divf" and isinstance(a, Sym) and not isinstance(b, Sym):
                return a.scale(Q(1) / np.asarray(b, object))
            if nm == "arith.negf":
                return a.scale(-1)
            if nm == "arith.select":
                return where(a, b, vals[2])
            if nm == "tensor.extract":
                pass
            else:
                raise NotLinear(f"{nm} on an input value")
        if nm == "arith.select":
            c, t, e = vals
            if isinstance(t, Sym) or isinstance(e, Sym):
                return where(c, t, e)
            return np.where(np.asarray(c, bool), t, e)
        if nm == "tensor.extract":
            t = vals[0]
            shp = np.shape(t) if not isinstance(t, Sym) else t.shape
            idx = []
            valid = np.ones(np.broadcast_shapes(*[np.shape(i) for i in vals[1:]]) if vals[1:] else (), bool)
            for k, i in enumerate(vals[1:]):
                i = np.asarray(i, np.int64)
                valid = valid & (i >= 0) & (i < shp[k])
                idx.append(np.clip(i, 0, shp[k] - 1))
            if np.any(mask & ~np.broadcast_to(valid, mask.shape)) if np.shape(valid) else not valid:
                raise NotLinear("out-of-bounds extract on an active lane")
            if isinstance(t, Sym):
                return t.take(tuple(np.broadcast_arrays(*idx)))
            return np.asarray(t)[tuple(idx)]
        if nm == "scf.for":
            return self._for(op, vals, env, ivs, mask)
        if nm == "scf.if":
            c = np.asarray(vals[0], bool)
            r1 = self._block(op.regions[0].blocks[0], dict(env), ivs, mask & c)
            r2 = self._block(op.regions[1].blocks[0], dict(env), ivs, mask & ~c) if op.regions[1].blocks else []
            return [where(c, x, y) for x, y in zip(r1, r2)]
        ia, ib = (np.asarray(a) if a is not None else None), (np.asarray(b) if b is not None else None)
        if nm in ("arith.addi", "arith.addf"):
            return ia + ib
        if nm in ("arith.subi", "arith.subf"):
            return ia - ib
        if nm in ("arith.muli", "arith.mulf"):
            return ia * ib
        if nm == "arith.divf":
            return ia / ib
        if nm == "arith.negf":
            return -ia
        if nm in ("arith.divsi",):
            quotient = np.frompyfunc(lambda x, y: 0 if y == 0 else (abs(x)//abs(y)) * (-1 if (x < 0) != (y < 0) else 1), 2, 1)
            return quotient(ia, ib)
        if nm in ("arith.divui", "arith.floordivsi"):
            return np.floor_divide(ia, np.where(ib == 0, 1, ib))
        if nm == "arith.ceildivsi":
            return -np.floor_divide(-ia, np.where(ib == 0, 1, ib))
        if nm in ("arith.remsi",):
            remainder = np.frompyfunc(lambda x, y: 0 if y == 0 else (abs(x) % abs(y)) * (-1 if x < 0 else 1), 2, 1)
            return remainder(ia, ib)
        if nm == "arith.remui":
            return np.mod(ia, np.where(ib == 0, 1, ib))
        if nm in ("arith.minsi", "arith.minui", "arith.minimumf", "arith.minnumf"):
            return np.minimum(ia, ib)
        if nm in ("arith.maxsi", "arith.maxui", "arith.maximumf", "arith.maxnumf"):
            return np.maximum(ia, ib)
        if nm == "arith.andi":
            return ia & ib
        if nm == "arith.ori":
            return ia | ib
        if nm == "arith.xori":
            return ia ^ ib
        if nm == "arith.cmpi":
            predicate = int(_attr(op, "predicate").value.data)
            if predicate >= 6:
                dtype = str(op.operands[0].type)
                bits = 64 if dtype == 'index' else op.operands[0].type.width.data
                unsigned = np.frompyfunc(lambda x: int(x) % (1 << bits),1,1)
                ia,ib = unsigned(np.asarray(ia,object)),unsigned(np.asarray(ib,object))
            return _CMPI[predicate](ia, ib)
        if nm == "arith.cmpf":
            return _CMPF[int(_attr(op, "predicate").value.data)](ia, ib)
        raise NotLinear(f"unsupported body op {nm}")

    def _for(self, op, vals, outer_env, ivs, mask):
        lb, ub, st = (np.asarray(v, np.int64) for v in vals[:3])
        carried = list(vals[3:])
        if np.any(st <= 0):
            raise NotLinear("non-positive loop step")
        trips = np.maximum(0, -np.floor_divide(-(ub - lb), st))
        tmax = int(np.max(np.where(mask, trips, 0))) if np.size(trips) else 0
        body = op.regions[0].blocks[0]
        for t in range(tmax):
            active = mask & (t < trips)
            iv = lb + t * st
            env = dict(outer_env)
            env[body.args[0]] = iv
            for a, c in zip(body.args[1:], carried):
                env[a] = c
            new = self._block(body, env, ivs, active)
            carried = [where(active, n, c) for n, c in zip(new, carried)]
        return carried


def _affine(amap, ivs, dims):
    """Evaluate the actual affine expressions, including floor/mod, exactly."""
    if amap.num_symbols:
        raise NotLinear('unbound affine symbols')
    values = amap.eval([np.asarray(iv, object) for iv in ivs], [])
    out = []
    for value in values:
        data = np.asarray(value, object)
        if np.any(data < -(1 << 63)) or np.any(data >= (1 << 63)):
            raise NotLinear('affine index exceeds the signed 64-bit address range')
        out.append(np.broadcast_to(np.asarray(data, dtype=np.int64), dims))
    return out


def _identity_like(amap) -> bool:
    from xdsl.ir.affine import AffineDimExpr
    return (amap.num_symbols==0 and len(amap.results)==amap.num_dims and
            all(isinstance(e,AffineDimExpr) and e.position==d for d,e in enumerate(amap.results)))


@dataclass
class LinearMap:
    """y.flat[p] = sum_k coef[k] * input_{src[k]}.flat[idx[k]] over k with out[k] == p, + const[p]."""

    out_shape: tuple
    out: np.ndarray
    src: np.ndarray
    idx: np.ndarray
    coef: np.ndarray
    const: np.ndarray
    sources: dict


def extract(root: SSAValue, leaf, *, region_type=Region) -> tuple[LinearMap, list[Operation]]:
    reg = region_type(leaf)
    try:
        val = reg.tensor(root)
    except TypeError as exc:
        # A symbolic form reached integer-only arithmetic (e.g. a float product
        # feeding an integer conversion): not an exactly modeled linear region.
        raise NotLinear(f"symbolic value in unmodeled arithmetic: {exc}") from exc
    shp = _shape(root)
    n = int(np.prod(shp)) if shp else 1
    extra = []
    if isinstance(val, Scatter):
        for flat, upd in val.updates:
            for s_, i, c in upd.compact().terms:
                c = np.asarray(c, object).reshape(-1)
                nz = c != 0
                extra.append((np.asarray(flat).reshape(-1)[nz], s_, np.asarray(i).reshape(-1)[nz], c[nz]))
            if np.any(np.asarray(upd.const)):
                raise NotLinear("scatter of a constant")
        val = val.base
    if not isinstance(val, Sym):
        val = Sym([], np.asarray(val, object))
    val = val.bshape(shp)
    outs, srcs, idxs, coefs = [], [], [], []
    p = np.arange(n, dtype=np.int64)
    for s, i, c in val.compact().terms:
        c = np.asarray(c, object).reshape(-1)
        nz = c != 0
        outs.append(p[nz])
        srcs.append(np.full(int(nz.sum()), s, np.int64))
        idxs.append(np.asarray(i, np.int64).reshape(-1)[nz])
        coefs.append(c[nz])
    for o, s_, i, c in extra:
        outs.append(o)
        srcs.append(np.full(o.size, s_, np.int64))
        idxs.append(i)
        coefs.append(c)
    cat = (lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt))
    lm = LinearMap(shp, cat(outs, np.int64), cat(srcs, np.int64), cat(idxs, np.int64), cat(coefs, object),
                   np.asarray(val.const, object).reshape(-1), reg.sources)
    return lm, reg.ops
