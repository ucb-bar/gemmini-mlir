
import sys

from torch_mlir import ir
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects import llvm

def _erase_self_copies(module):
    """Erase `memref.copy %x, %x` -- copying a buffer onto itself is a no-op. Returns the count."""
    n = 0

    def walk(op):
        nonlocal n
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    walk(inner)
                    if inner.operation.name == "memref.copy" and inner.operands[0] == inner.operands[1]:
                        inner.operation.erase()
                        n += 1

    walk(module.operation)
    return n


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    """Run `pipeline`; when `erase` (or any `mid` rewrite is requested), split it after the
    post-bufferization canonicalize/cse and run the rewrites in between. Splitting on
    buffer-loop-hoisting (not a fixed index) so the hook stays put if the pass list moves.

    `mid` is a sequence of `(label, fn(ctx, module) -> int)` rewrites that need the SAME window as
    the erase: after bufferization has created the buffer ops, before finalize-memref-to-llvm turns
    them into opaque runtime calls. Each reports its count as `OK <label> <n>` so a rewrite that
    matched nothing is visible in the build log instead of passing for applied.

    `late` is the same shape, in a DIFFERENT window: after the forall/linalg -> `scf.parallel`
    conversions and before `convert-scf-to-openmp` turns each `scf.parallel` into a fork. It is a
    separate list rather than more `mid` entries because at the `mid` point no `scf.parallel` exists
    yet -- a grain decision made there would price loops that have not been formed. Empty `late`
    (the default) leaves the pass string split exactly as before, so the lowering is byte-identical.

    `post_openmp` runs immediately AFTER `convert-scf-to-openmp`.  It exists for structural edits
    that need to reason about `scf.parallel` before conversion and then annotate the corresponding
    `omp.parallel` afterwards.  When empty, conversion and its following passes stay in the same
    PassManager invocation as before.
    """
    from torch_mlir.passmanager import PassManager

    def _run(sub):
        if sub:
            PassManager.parse('builtin.module(' + ','.join(sub) + ')', ctx).run(module.operation)

    def _late_split(sub):
        """Run `sub`, pausing before `convert-scf-to-openmp` to run the `late` rewrites."""
        if not late and not post_openmp:
            _run(sub)
            return
        j = next((i for i, p in enumerate(sub) if 'convert-scf-to-openmp' in p), -1)
        if j < 0:
            # No OpenMP conversion in this pass list: run the rewrites at the END, where they still
            # see whatever `scf.parallel` survives, rather than dropping them silently.
            _run(sub)
            for label, fn in late:
                print('OK ' + label, fn(ctx, module))
            for label, fn in post_openmp:
                print('OK ' + label, fn(ctx, module))
            return
        _run(sub[:j])
        for label, fn in late:
            print('OK ' + label, fn(ctx, module))
        if not post_openmp:
            _run(sub[j:])
            return
        _run(sub[j:j + 1])
        for label, fn in post_openmp:
            print('OK ' + label, fn(ctx, module))
        _run(sub[j + 1:])

    passes = [p for p in pipeline.split(',') if p]
    if not passes:
        return
    if pre_generalize:
        marker = '__merlin_targeted_named_broadcast_fold__'
        mark = next((i for i, p in enumerate(passes) if p == marker), -1)
        if mark < 0:
            raise RuntimeError('pre-generalize rewrite requested but its pipeline marker is absent')
        _run(passes[:mark])
        for label, fn in pre_generalize:
            print('OK ' + label, fn(ctx, module))
        passes = passes[mark + 1:]
    elif '__merlin_targeted_named_broadcast_fold__' in passes:
        raise RuntimeError('pre-generalize pipeline marker present without a requested rewrite')
    want_split = bool(erase) or bool(mid)
    k = next((i for i, p in enumerate(passes) if 'buffer-loop-hoisting' in p), -1) if want_split else -1
    if k < 0:
        _late_split(passes)
        return
    # ...hoisting, canonicalize, cse -- but NEVER past the pass that lowers linalg to loops. A mid
    # rewrite may EMIT linalg (expand_memref_copy rewrites a copy to `linalg.copy` and relies on
    # convert-linalg-to-loops to turn it into an scf nest), and in the scalar pipeline that pass sits
    # at k+1, so a fixed k+3 window put the rewrite AFTER its own lowering: the linalg op survived to
    # LLVM conversion as an unrealized_conversion_cast and the whole build failed. Clamping keeps the
    # RVV window (where the pass is at k+7) exactly where it was.
    end = k + 3
    for i, p in enumerate(passes):
        if 'convert-linalg-to-loops' in p or 'convert-linalg-to-parallel-loops' in p:
            end = min(end, i)
            break
    head, tail = passes[:end], passes[end:]
    _run(head)
    if erase:
        print('OK erase_self_copy', _erase_self_copies(module))
    for label, fn in mid:
        print('OK ' + label, fn(ctx, module))
    _late_split(tail)


_ERASE_SELF_COPY = len(sys.argv) > 4 and sys.argv[4] == '1'

def _fuse_transpose_b(module, ctx):
    """Fold `matmul(A, transpose(B, perm))` into a transpose-b `linalg.matmul` (operand + B
    indexing_map rewrite), erasing the now-dead transpose. Returns the number of matmuls fused."""
    from torch_mlir.ir import AffineMap, AffineMapAttr, ArrayAttr

    def _walk(op, fn):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    _walk(inner, fn)
                    fn(inner)

    def _perm(top):
        # linalg.transpose carries `permutation = array<i64: ...>`; parse it structurally.
        try:
            a = top.attributes["permutation"]
        except Exception:
            return None
        s = str(a)
        i = s.find(":")
        if i < 0:
            return None
        try:
            return [int(t.strip()) for t in s[i + 1:s.rfind(">")].split(",")]
        except ValueError:
            return None

    fused = 0
    dead = []
    def _fuse(o):
        nonlocal fused
        op = o.operation
        if op.name != "linalg.matmul" or len(op.operands) < 2:
            return
        b = op.operands[1]
        prod = b.owner
        if not hasattr(prod, "name") or prod.name != "linalg.transpose":
            return
        perm = _perm(prod)
        if perm is None:
            return
        try:
            maps = op.attributes["indexing_maps"]
        except KeyError:
            return                                  # need explicit maps to rewrite B's access
        m_b = maps[1].value                         # AffineMapAttr -> AffineMap
        results = list(m_b.results)
        if len(perm) != len(results):
            return                                  # perm must match B's map arity
        with ctx:
            new_b = AffineMap.get(m_b.n_dims, m_b.n_symbols,
                                  [results[perm[j]] for j in range(len(perm))])
            new_maps = ArrayAttr.get([maps[0], AffineMapAttr.get(new_b), maps[2]])
        op.attributes["indexing_maps"] = new_maps
        op.operands[1] = prod.operands[0]           # read the un-transposed source weight
        dead.append(prod)
        fused += 1

    _walk(module.operation, _fuse)
    for t in dead:
        if len(list(t.results[0].uses)) == 0:
            t.operation.erase()
    return fused


_FUSE_TRANSPOSE_B = len(sys.argv) > 5 and sys.argv[5] == '1'

def _wt_permutation(top):
    """The `permutation = array<i64: ...>` of a `linalg.transpose`, parsed structurally.

    Returns None (never a guess) when the attribute is missing or not a list of ints."""
    try:
        attr = top.attributes["permutation"]
    except (KeyError, IndexError, TypeError):
        return None
    text = str(attr)
    colon = text.find(":")
    close = text.rfind(">")
    if colon < 0 or close < colon:
        return None
    body = text[colon + 1:close].strip()
    if not body:
        return []
    perm = []
    for token in body.split(","):
        token = token.strip()
        try:
            perm.append(int(token))
        except ValueError:
            return None
    if sorted(perm) != list(range(len(perm))):
        return None                             # not a permutation -- refuse rather than reinterpret
    return perm


def _wt_op_name(op):
    """The OPERATION name of `op`, whether it arrives as an Operation or as a typed OpView.

    An OpView's own `.name` is the symbol name (`FuncOp.name` is "forward"), not "func.func", so
    reading `.name` off whatever the bindings hand back silently classifies every function argument
    as non-invariant -- measured: 15 weight transposes skipped, 0 folded."""
    if op is None:
        return None
    return getattr(getattr(op, "operation", op), "name", None)


def _wt_loop_invariant(value):
    """Is `value` stored data rather than something the function computes?

    True for a `func.func` entry-block argument (a weight/parameter the caller hands in) and for a
    materialized constant. Anything this cannot positively identify is False."""
    from torch_mlir import ir as _wtir
    try:
        arg = _wtir.BlockArgument(value)
    except (ValueError, TypeError):
        arg = None
    if arg is not None:
        try:
            parent = arg.owner.owner              # Block -> the op owning the block's region
            return _wt_op_name(parent) == "func.func"
        except (AttributeError, ValueError, TypeError):
            return False
    return _wt_op_name(getattr(value, "owner", None)) in ("arith.constant", "memref.get_global")


def _wt_dim_position(expr):
    """The loop-dim index of `expr` when it is a bare dim (`d3`), else None.

    A map result that is anything else -- a constant (`(d0, 0, d2)`), a sum, a floordiv -- is not a
    dimension this pass can price a stride along, so it reports None and the caller fails closed."""
    from torch_mlir import ir as _wtir
    try:
        return _wtir.AffineDimExpr(expr).position
    except (ValueError, TypeError):
        return None


def _wt_static_shape(value):
    """The operand's static shape, or None if any extent is dynamic/unranked (fail closed)."""
    from torch_mlir import ir as _wtir
    try:
        st = _wtir.ShapedType(value.type)
        if not st.has_static_shape:
            return None
        return list(st.shape)
    except (ValueError, TypeError):
        return None


def _wt_stride(results, shape, dim):
    """Element stride of a row-major operand of `shape`, read through `results`, along loop `dim`.

    Varying loop dim `dim` by one moves the linear offset by the row-major stride of whichever
    operand axis that dim indexes. A dim the map never mentions leaves the operand invariant -> 0.
    Returns None when the access is not a plain permutation of bare dims (fail closed)."""
    for j, e in enumerate(results):
        pos = _wt_dim_position(e)
        if pos is None:
            return None
        if pos == dim:
            stride = 1
            for extent in shape[j + 1:]:
                stride *= int(extent)
            return stride
    return 0


def _wt_hot_dim(op, maps):
    """The loop dim the vectorizer will make contiguous: the FASTEST-VARYING axis of the output.

    MEASURED, and the reason this guard exists. The frozen RVV schedule tiles the contraction's
    parallel dims and vectorizes the innermost OUTPUT dim -- on small_llama int8 the B operand is
    read as `tensor<1x16xi8>`, a row of 16 consecutive n. Folding a `[1, 0]` weight transpose into
    that map turns the same read into `tensor<16x1xi8>`: 16 elements 128 bytes apart, a strided
    read on exactly the axis being vectorized. Statically that is invisible -- same op count, same
    vector shape count, fewer instructions -- and on the K1 it cost 1.09x.

    Returns None when the output map's last result is not a bare dim, so the caller refuses."""
    n_outs = len(op.results)
    if n_outs < 1 or len(maps) <= n_outs:
        return None
    out = maps[len(maps) - n_outs].value          # first output operand's map
    results = list(out.results)
    if not results:
        return None
    return _wt_dim_position(results[-1])


def _wt_indexing_maps(op):
    """The op's `indexing_maps` array when it has one map per operand, else None."""
    try:
        maps = op.attributes["indexing_maps"]
    except (KeyError, IndexError, TypeError):
        return None
    try:
        if len(maps) != len(op.operands):
            return None                         # one map per operand, or this pass does not know
    except TypeError:
        return None
    return maps


def _fold_weight_transposes(module, ctx):
    """Fold every loop-invariant `linalg.transpose` into its consumers' indexing maps.

    Returns (folded, report) where `report` is a list of (kind, detail) lines: one `fold` per
    transpose removed and one `skip` per transpose left in place WITH the reason it was left."""
    from torch_mlir.ir import AffineMap, AffineMapAttr, ArrayAttr

    transposes = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    walk(inner)
                    if inner.operation.name == "linalg.transpose":
                        transposes.append(inner.operation)

    walk(module.operation)

    folded = 0
    report = []
    for top in transposes:
        if not top.operands or not top.results:
            report.append(("skip", "transpose with no operand/result"))
            continue
        src = top.operands[0]
        shape = str(top.results[0].type)
        if not _wt_loop_invariant(src):
            report.append(("skip", shape + ": source is computed, not stored (not loop-invariant)"))
            continue
        perm = _wt_permutation(top)
        if perm is None:
            report.append(("skip", shape + ": permutation attribute not a static permutation"))
            continue
        uses = list(top.results[0].uses)
        if not uses:
            report.append(("skip", shape + ": result is dead; left for DCE"))
            continue

        # PLAN first, rewrite only if EVERY use can be rewritten: a partial fold leaves the
        # transpose alive and buys nothing.
        plan = []
        reason = ""
        for use in uses:
            owner = getattr(use.owner, "operation", use.owner)
            idx = use.operand_number
            maps = _wt_indexing_maps(owner)
            if maps is None:
                reason = "consumer %s states no per-operand indexing_maps" % owner.name
                break
            n_outs = len(owner.results)
            if n_outs and idx >= len(maps) - n_outs:
                reason = "used as an `outs` operand of %s (written, not read)" % owner.name
                break
            m = maps[idx].value
            old_results = list(m.results)
            if len(old_results) != len(perm):
                reason = ("consumer %s map has %d results, permutation has %d"
                          % (owner.name, len(old_results), len(perm)))
                break
            hot = _wt_hot_dim(owner, maps)
            if hot is None:
                reason = ("consumer %s has no bare fastest-varying output dim, so the axis the "
                          "vectorizer makes contiguous cannot be derived" % owner.name)
                break
            t_shape = _wt_static_shape(top.results[0])
            w_shape = _wt_static_shape(src)
            if t_shape is None or w_shape is None:
                reason = "operand shape is dynamic or unranked, so the access stride is unpriceable"
                break
            new_results = list(old_results)
            for t, dst in enumerate(perm):
                new_results[dst] = old_results[t]
            before = _wt_stride(old_results, t_shape, hot)
            after = _wt_stride(new_results, w_shape, hot)
            if before is None or after is None:
                reason = ("consumer %s reads a non-permutation access this pass cannot price"
                          % owner.name)
                break
            if after > before:
                reason = ("folding would move the vectorized axis d%d of %s from stride %d to "
                          "stride %d -- the fold removes a pass over the weight but makes the "
                          "contiguous vector read a strided one, which MEASURED 1.09x SLOWER on "
                          "the K1" % (hot, owner.name, before, after))
                break
            plan.append((owner, idx))
        if reason:
            report.append(("skip", shape + ": " + reason))
            continue

        with ctx:
            for owner, idx in plan:
                maps = owner.attributes["indexing_maps"]     # re-read: an op may hold several uses
                m = maps[idx].value
                old = list(m.results)
                new = list(old)
                for t, dst in enumerate(perm):               # T[i] = W[f] with f[perm[t]] = i[t]
                    new[dst] = old[t]
                entries = [maps[j] for j in range(len(maps))]
                entries[idx] = AffineMapAttr.get(
                    AffineMap.get(m.n_dims, m.n_symbols, new))
                owner.attributes["indexing_maps"] = ArrayAttr.get(entries)
                owner.operands[idx] = src
        top.erase()
        folded += 1
        report.append(("fold", "%s into %d consumer operand(s)" % (shape, len(plan))))

    return folded, report


_FOLD_WEIGHT_TRANSPOSE = len(sys.argv) > 7 and sys.argv[7] == '1'

def _bf_walk(op):
    for region in op.regions:
        for block in region.blocks:
            for inner in list(block.operations):
                yield inner.operation
                yield from _bf_walk(inner.operation)


def _bf_rank(value):
    try:
        shaped = ir.ShapedType(value.type)
        return shaped.rank if shaped.has_rank else None
    except Exception:
        return None


def _bf_maps(op):
    try:
        maps = op.attributes['indexing_maps']
        if len(maps) != len(op.operands):
            return None
        return maps
    except (KeyError, TypeError, ValueError):
        return None


def _bf_all_parallel(op):
    try:
        iters = op.attributes['iterator_types']
    except KeyError:
        return False
    return bool(iters) and all('parallel' in str(iterator) for iterator in iters)


def _fold_broadcasts(module, ctx, required_attr=None):
    """Return ``(folded, report)`` after applying only mechanically proven map folds."""
    from torch_mlir.ir import AffineMap, AffineMapAttr, ArrayAttr

    broadcasts = [op for op in _bf_walk(module.operation) if op.name == 'linalg.broadcast']
    folded = 0
    report = []
    for broadcast in broadcasts:
        if required_attr is not None and required_attr not in broadcast.attributes:
            continue
        shape = str(broadcast.results[0].type) if broadcast.results else '<no result>'
        if len(broadcast.operands) != 2 or len(broadcast.results) != 1:
            report.append(('skip', shape + ': expected one input, one init and one result'))
            continue
        src, init = broadcast.operands
        result = broadcast.results[0]
        uses = list(result.uses)
        if len(uses) != 1:
            report.append(('skip', shape + ': broadcast has %d readers, not one' % len(uses)))
            continue
        use = uses[0]
        consumer = use.owner
        operand_index = int(use.operand_number)
        if consumer.name != 'linalg.generic':
            report.append(('skip', shape + ': sole reader is %s, not linalg.generic'
                           % consumer.name))
            continue
        # This guard is the boundary from the refuted blanket fusion lever: a contraction carries a
        # reduction iterator.  Do not change contraction input maps or its scheduled access pattern.
        if not _bf_all_parallel(consumer):
            report.append(('skip', shape + ': consumer has a non-parallel iterator'))
            continue
        maps = _bf_maps(consumer)
        if maps is None:
            report.append(('skip', shape + ': consumer states no map for every operand'))
            continue
        n_outs = len(consumer.results)
        n_inputs = len(consumer.operands) - n_outs
        if n_outs < 1 or operand_index >= n_inputs:
            report.append(('skip', shape + ': broadcast is not a tensor input of its consumer'))
            continue
        try:
            dimensions = [int(dim) for dim in broadcast.attributes['dimensions']]
        except (KeyError, TypeError, ValueError):
            report.append(('skip', shape + ': dimensions are not a static integer list'))
            continue
        src_rank = _bf_rank(src)
        out_rank = _bf_rank(result)
        if src_rank is None or out_rank is None:
            report.append(('skip', shape + ': source/result is unranked'))
            continue
        if (dimensions != sorted(set(dimensions))
                or any(dim < 0 or dim >= out_rank for dim in dimensions)
                or src_rank + len(dimensions) != out_rank):
            report.append(('skip', shape + ': dimensions do not define a rank projection'))
            continue
        old_map = maps[operand_index].value
        old_results = list(old_map.results)
        if len(old_results) != out_rank:
            report.append(('skip', shape + ': consumer map/result rank disagree'))
            continue
        keep = [position for position in range(out_rank) if position not in dimensions]
        new_results = [old_results[position] for position in keep]

        # Changing the shaped operand must not change the scalar type of the corresponding generic
        # block argument.  The linalg verifier checks this too, but refusing before mutation keeps
        # the transformation atomic and makes the reason visible.
        try:
            if ir.ShapedType(src.type).element_type != ir.ShapedType(result.type).element_type:
                report.append(('skip', shape + ': source/result element types disagree'))
                continue
        except Exception:
            report.append(('skip', shape + ': source/result is not shaped'))
            continue

        with ctx:
            entries = [maps[index] for index in range(len(maps))]
            entries[operand_index] = AffineMapAttr.get(
                AffineMap.get(old_map.n_dims, old_map.n_symbols, new_results))
            consumer.attributes['indexing_maps'] = ArrayAttr.get(entries)
            consumer.operands[operand_index] = src

        # Result is now dead by construction.  Its destination is normally a tensor.empty used only
        # by the broadcast; erase it too, otherwise bufferization can still leave a pointless alloc.
        init_owner = init.owner
        broadcast.erase()
        if (init_owner is not None and init_owner.name == 'tensor.empty'
                and all(not list(value.uses) for value in init_owner.results)):
            init_owner.erase()
        folded += 1
        report.append(('fold', '%s operand %d -> rank-%d projection'
                       % (shape, operand_index, src_rank)))
    return folded, report


_FOLD_BROADCAST = len(sys.argv) > 14 and sys.argv[14] == '1'

def _fold_targeted_generalized_broadcasts(module, ctx, target_attr):
    """Fold only generic broadcasts carrying ``target_attr`` into parallel generics."""
    from torch_mlir.ir import AffineMap, AffineMapAttr, ArrayAttr

    producers = [op for op in _bf_walk(module.operation)
                 if op.name == 'linalg.generic' and target_attr in op.attributes]
    folded = 0
    report = []
    for producer in producers:
        shape = str(producer.results[0].type) if producer.results else '<no result>'
        # Stock named-op generalization turns a tensor broadcast into one source, one init, one
        # result and two maps.  The marker proves its origin, but keep the full structural check so
        # an upstream representation change fails closed.
        maps = _bf_maps(producer)
        if (len(producer.operands) != 2 or len(producer.results) != 1
                or maps is None or len(maps) != 2 or not _bf_all_parallel(producer)):
            report.append(('skip', shape + ': marked producer is not a generalized broadcast'))
            continue
        src, init = producer.operands
        result = producer.results[0]
        uses = list(result.uses)
        if len(uses) != 1:
            report.append(('skip', shape + ': generalized broadcast has %d readers' % len(uses)))
            continue
        use = uses[0]
        consumer = use.owner
        operand_index = int(use.operand_number)
        consumer_maps = _bf_maps(consumer)
        if (consumer.name != 'linalg.generic' or not _bf_all_parallel(consumer)
                or consumer_maps is None):
            report.append(('skip', shape + ': generalized consumer is not all-parallel generic'))
            continue
        n_outs = len(consumer.results)
        n_inputs = len(consumer.operands) - n_outs
        if n_outs != 1 or operand_index >= n_inputs:
            report.append(('skip', shape + ': generalized broadcast is not a consumer input'))
            continue
        source_map = maps[0].value
        output_map = maps[1].value
        consumer_map = consumer_maps[operand_index].value
        src_rank = _bf_rank(src)
        out_rank = _bf_rank(result)
        if (src_rank is None or out_rank is None
                or len(source_map.results) != src_rank
                or len(output_map.results) != out_rank
                or not source_map.is_projected_permutation
                or not output_map.is_permutation
                or len(consumer_map.results) != source_map.n_dims
                or source_map.n_symbols != 0 or consumer_map.n_symbols != 0):
            report.append(('skip', shape + ': generalized maps do not prove broadcast semantics'))
            continue
        try:
            if ir.ShapedType(src.type).element_type != ir.ShapedType(result.type).element_type:
                report.append(('skip', shape + ': source/result element types disagree'))
                continue
        except Exception:
            report.append(('skip', shape + ': source/result is not shaped'))
            continue

        # source_map maps producer points to source indices; consumer_map maps consumer points to
        # producer-result indices. Compose them to address the source directly.
        with ctx:
            composed = [expr.compose(consumer_map) for expr in source_map.results]
            entries = [consumer_maps[index] for index in range(len(consumer_maps))]
            entries[operand_index] = AffineMapAttr.get(
                AffineMap.get(consumer_map.n_dims, 0, composed))
            consumer.attributes['indexing_maps'] = ArrayAttr.get(entries)
            consumer.operands[operand_index] = src

        init_owner = init.owner
        producer.erase()
        if (init_owner is not None and init_owner.name == 'tensor.empty'
                and all(not list(value.uses) for value in init_owner.results)):
            init_owner.erase()
        folded += 1
        report.append(('fold', shape + ' operand %d -> rank-%d projection'
                       % (operand_index, src_rank)))
    return folded, report


def _targeted_named_broadcast_fold(ctx, module):
    """Generalize and fold only sole-use broadcasts feeding named add/mul operations."""
    target_attr = 'merlin.targeted_named_broadcast_fold'
    targets = []
    counts = {'linalg.add': 0, 'linalg.mul': 0}
    for broadcast in _bf_walk(module.operation):
        if broadcast.name != 'linalg.broadcast' or len(broadcast.results) != 1:
            continue
        uses = list(broadcast.results[0].uses)
        if len(uses) != 1 or uses[0].owner.name not in counts:
            continue
        # Stock generalization preserves this discardable attribute when rebuilding the producer,
        # giving the post-pass rewrite a mechanically exact provenance token.
        with ctx:
            broadcast.attributes[target_attr] = ir.UnitAttr.get()
        targets.append(broadcast)
        counts[uses[0].owner.name] += 1

    print('CENSUS targeted_named_broadcast_fold add', counts['linalg.add'],
          'mul', counts['linalg.mul'])
    if not targets:
        return 0

    # This is the same generalization the surrounding pipeline is about to run. Contractions have
    # already been scheduled and lowered before this stage; its later invocation is a no-op.
    PassManager.parse('builtin.module(func.func(linalg-generalize-named-ops))', ctx).run(
        module.operation)
    folded, report = _fold_targeted_generalized_broadcasts(module, ctx, target_attr)
    if folded != len(targets):
        details = '; '.join(kind + ': ' + detail for kind, detail in report)
        raise RuntimeError('targeted named broadcast fold planned %d but folded %d: %s'
                           % (len(targets), folded, details))
    return folded


_TARGETED_NAMED_BROADCAST_FOLD = len(sys.argv) > 15 and sys.argv[15] == '1'
_PRE_GENERALIZE_STAGES = ([('targeted_named_broadcast_fold',
                            _targeted_named_broadcast_fold)]
                          if _TARGETED_NAMED_BROADCAST_FOLD else [])

def _mc_static_memref(t):
    """The `ir.MemRefType` when `t` is a ranked memref of static shape and rank >= 1, else None.

    Fail-closed: anything this cannot prove static (unranked, a dynamic dim, rank 0) returns None
    and the copy is left for the existing lowering rather than expanded on an assumption."""
    from torch_mlir import ir as _mcir
    try:
        mt = _mcir.MemRefType(t)
    except (ValueError, TypeError):
        return None
    if mt.rank < 1:
        return None
    for d in range(mt.rank):
        if mt.is_dynamic_dim(d):
            return None
    return mt


def _expand_memref_copies(ctx, module):
    """Rewrite every static `memref.copy` to a `linalg.copy` so it lowers to an emitted loop nest.

    Returns the number expanded; prints the number SKIPPED (never zero-filled) so a copy that could
    not be proven static is surfaced instead of silently remaining a runtime call."""
    from torch_mlir import ir as _mcir
    from torch_mlir.dialects import linalg as _mclinalg

    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    walk(inner)
                    if inner.operation.name == "memref.copy":
                        todo.append(inner)

    walk(module.operation)
    n = skipped = 0
    with ctx, _mcir.Location.unknown():
        for cp in todo:
            src, dst = cp.operands[0], cp.operands[1]
            s_t, d_t = _mc_static_memref(src.type), _mc_static_memref(dst.type)
            if s_t is None or d_t is None or list(s_t.shape) != list(d_t.shape):
                skipped += 1
                continue
            with _mcir.InsertionPoint(cp):
                op = _mclinalg.CopyOp([], [src], [dst])
                body = op.regions[0].blocks.append(s_t.element_type, d_t.element_type)
                with _mcir.InsertionPoint(body):
                    _mclinalg.YieldOp([body.arguments[0]])
            cp.operation.erase()
            n += 1
    print("OK expand_memref_copy skipped", skipped)
    return n

_CONCAT_DPS = len(sys.argv) > 8 and sys.argv[8] == '1'


def _cd_static(t):
    """The `ir.RankedTensorType` when `t` is ranked, rank >= 1 and fully static, else None.

    Fail-closed: a shape this cannot prove static has no constant slice offsets, so the concat is
    left alone rather than rewritten on an assumption."""
    from torch_mlir import ir as _cdir
    try:
        tt = _cdir.RankedTensorType(t)
    except (ValueError, TypeError):
        return None
    if tt.rank < 1:
        return None
    for d in range(tt.rank):
        if tt.is_dynamic_dim(d):
            return None
    return tt


def _cd_out_operand_index(op, value):
    """Index in ``op.operands`` of the destination-passing ``outs`` entry backing ``value``.

    Structural: a DPS op declares its operand split in ``operandSegmentSizes`` and writes result i
    through outs entry i. Anything without that split, or whose arities disagree, returns None --
    an op that does not say where it writes is not told where to write."""
    try:
        seg = op.operation.attributes["operandSegmentSizes"]
    except KeyError:
        return None
    sizes = [int(seg[i]) for i in range(len(seg))]
    if len(sizes) != 2:
        return None
    n_in, n_out = sizes
    if n_in + n_out != len(op.operands) or n_out != len(op.results):
        return None
    for i, r in enumerate(op.results):
        if r == value:
            return n_in + i
    return None


def _cd_writes_into_empty(op, idx, want_type):
    """True when ``op``'s operand ``idx`` is a `tensor.empty` of ``want_type``.

    The empty may have OTHER users: it is a value with no contents, so several ops may name it as
    their destination and retargeting one use changes nothing for the rest. Requiring a private
    empty instead would make this refuse everything on a module that has run `cse`, which merges
    identical empties."""
    dest = op.operands[idx]
    if dest.type != want_type:
        return False
    try:
        return dest.owner.operation.name == "tensor.empty"
    except AttributeError:              # block argument: not an op at all
        return False


def _cd_uses_only_in(value, op):
    """True when every use of ``value`` is an operand of ``op``."""
    for use in value.uses:
        if use.owner.operation != op.operation:
            return False
    return True


def _cd_next_concat(module, index):
    """The ``index``-th `tensor.concat` in walk order, with its block and that block's ops, or None.

    Re-walked per rewrite ON PURPOSE: erasing an operation invalidates the Python handles to the
    other operations in the context (MEASURED -- the second concat's position lookup raised "the
    operation has been invalidated"), so no handle may be carried across a rewrite."""
    found = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                ops = list(block.operations)
                for inner in ops:
                    walk(inner)
                    if inner.operation.name == "tensor.concat":
                        found.append((inner, ops))

    walk(module.operation)
    return found[index] if index < len(found) else None


def _concat_dps(module, ctx):
    """Rewrite each eligible `tensor.concat` into a destination-passing insert_slice chain.

    Returns ``(n_rewritten, report)``; ``report`` is a list of ``(kind, detail)`` lines the runner
    prints, so a concat left alone is visible in the build log instead of passing for rewritten."""
    from torch_mlir import ir as _cdir
    from torch_mlir.dialects import tensor as _cdt
    from torch_mlir.passmanager import PassManager as _cdpm

    # Canonicalize and CSE FIRST -- the same two passes the pipeline itself opens with, so this is
    # idempotent, not an extra transformation. It is required: run before them, this rewrite gives
    # two identical transcendental generics different destinations and blocks the CSE that
    # `cse_through_provenance` exists to enable (MEASURED: linalg.generic 242 -> 244, transcendental
    # ops 8 -> 10 = 256 libm calls per inference bought back to save 384 copied elements).
    _cdpm.parse("builtin.module(canonicalize,cse)", ctx).run(module.operation)

    n = retargeted = 0
    report = []
    index = 0                       # how many leading concats were REFUSED (and so are still there)
    while True:
        hit = _cd_next_concat(module, index)
        if hit is None:
            break
        cat, ops = hit
        result = _cd_static(cat.results[0].type)
        shapes = [_cd_static(o.type) for o in cat.operands]
        if result is None or any(o is None for o in shapes):
            report.append(("skip_dynamic", str(cat.results[0].type)))
            index += 1
            continue
        pos = {}
        for i, o in enumerate(ops):
            pos[o.operation] = i
        if cat.operation not in pos:
            report.append(("skip_unplaced", str(cat.results[0].type)))
            index += 1
            continue
        dim = int(_cdir.IntegerAttr(cat.operation.attributes["dim"]))

        # Plan, in chain order: which operands can be produced straight into the destination.
        plan = []
        after = -1                  # program position at which the chain value is defined
        placed = []                 # operand values already placed (a repeat is a 2nd placement)
        for value, shape in zip(cat.operands, shapes):
            producer = None
            owner = value.owner
            try:
                owner_op = owner.operation
            except AttributeError:
                owner_op = None
            repeat = any(value == p for p in placed)
            if owner_op is not None and not repeat and pos.get(owner_op, -1) > after:
                oi = _cd_out_operand_index(owner, value)
                if oi is not None and _cd_uses_only_in(value, cat) \
                        and _cd_writes_into_empty(owner, oi, value.type):
                    producer = (owner, oi, pos[owner_op])
                    after = pos[owner_op]
            placed.append(value)
            plan.append((value, shape, producer))
        if not any(p is not None for _, _, p in plan):
            report.append(("skip_no_dps_producer", str(cat.results[0].type)))
            index += 1
            continue

        with ctx, _cdir.Location.unknown():
            first = min(p[2] for _, _, p in plan if p is not None)
            with _cdir.InsertionPoint(ops[first]):
                cur = _cdt.EmptyOp(list(result.shape), result.element_type).result
            offset = 0
            for value, shape, producer in plan:
                offsets = [0] * result.rank
                offsets[dim] = offset
                sizes = list(shape.shape)
                strides = [1] * result.rank
                if producer is not None:
                    owner, oi, k = producer
                    with _cdir.InsertionPoint(ops[k]):
                        sl = _cdt.ExtractSliceOp(value.type, cur, [], [], [],
                                                 offsets, sizes, strides).result
                    ip = _cdir.InsertionPoint(ops[k + 1])   # resolved BEFORE the operand is set:
                    owner.operation.operands[oi] = sl       # setting one invalidates the position
                    retargeted += 1                         # lookup for that operation
                else:
                    ip = _cdir.InsertionPoint(cat)
                with ip:
                    ins = _cdt.InsertSliceOp(value, cur, [], [], [], offsets, sizes, strides)
                for i in range(len(cat.operation.attributes)):
                    a = cat.operation.attributes[i]
                    if a.name != "dim":          # carry provenance onto the replacement ops
                        ins.operation.attributes[a.name] = a.attr
                cur = ins.result
                offset += shape.shape[dim]
        cat.results[0].replace_all_uses_with(cur)
        cat.operation.erase()
        n += 1
    report.append(("retargeted", str(retargeted)))
    return n, report

_PG_TERMINATORS = frozenset((
    "scf.yield", "scf.reduce", "scf.reduce.return", "scf.condition",
    "memref.alloca_scope.return", "omp.yield", "omp.terminator", "func.return", "cf.br"))


def _pg_const_index(value):
    """The integer of an index-typed `arith.constant` defining `value`, else None (fail closed)."""
    from torch_mlir import ir as _pgir
    try:
        owner = value.owner.operation
    except AttributeError:                      # a block argument has no defining operation
        return None
    if owner.name != "arith.constant":
        return None
    try:
        return int(_pgir.IntegerAttr(owner.attributes["value"]))
    except Exception:                           # noqa: BLE001 - not an integer attribute
        return None


def _pg_trip(lb, ub, step):
    """ceil((ub - lb) / step) when all three are index constants and step > 0, else None."""
    l, u, s = _pg_const_index(lb), _pg_const_index(ub), _pg_const_index(step)
    if l is None or u is None or s is None or s <= 0:
        return None
    return max(0, -(-(u - l) // s))


def _pg_lanes(op):
    """Lane count of the widest vector result of `op` (1 for a scalar op).

    A vector operation retires one instruction but does that many lanes of work; pricing it as 1
    would make a vectorized region look 16x cheaper than the scalar spelling of the same work."""
    from torch_mlir import ir as _pgir
    n = 1
    for res in op.results:
        try:
            vt = _pgir.VectorType(res.type)
        except Exception:                       # noqa: BLE001 - not a vector type
            continue
        c = 1
        for d in vt.shape:
            c *= int(d)
        n = max(n, c)
    return n


def _pg_body_cost(block):
    """Static lane-operation cost of `block`, or None when something in it cannot be priced."""
    total = 0
    for handle in block.operations:
        op = handle.operation
        name = op.name
        if name in _PG_TERMINATORS:
            continue
        if name == "scf.parallel":
            return None                          # a nested parallel: leave the outer one alone
        if name == "scf.for":
            trip = _pg_trip(op.operands[0], op.operands[1], op.operands[2])
            if trip is None:
                return None
            inner = _pg_body_cost(op.regions[0].blocks[0])
            if inner is None:
                return None
            total += trip * inner
            continue
        if len(op.regions):
            sub = 0
            for region in op.regions:
                for blk in region.blocks:
                    cost = _pg_body_cost(blk)
                    if cost is None:
                        return None
                    # `scf.if` takes ONE of its regions; everything else (alloca_scope, ...) all.
                    sub = max(sub, cost) if name == "scf.if" else sub + cost
            total += sub + 1
            continue
        total += _pg_lanes(op)
    return total


def _pg_cost(op):
    """Static cost of one `scf.parallel`: parallel trip product * body cost. None => unpriceable."""
    block = op.regions[0].blocks[0]
    rank = len(block.arguments)
    operands = list(op.operands)
    if rank == 0 or len(operands) < 3 * rank:
        return None
    trip = 1
    for i in range(rank):
        t = _pg_trip(operands[i], operands[rank + i], operands[2 * rank + i])
        if t is None:
            return None
        trip *= t
    body = _pg_body_cost(block)
    if body is None:
        return None
    return trip * body


def _pg_serialize(op, ctx):
    """Rewrite one reduction-free `scf.parallel` into an equivalent serial `scf.for` nest."""
    from torch_mlir import ir as _pgir
    from torch_mlir.dialects import scf as _pgscf
    block = op.regions[0].blocks[0]
    rank = len(block.arguments)
    operands = list(op.operands)
    with ctx, _pgir.Location.unknown():
        point = _pgir.InsertionPoint(op)
        ivs = []
        terminator = None
        for i in range(rank):
            with point:
                loop = _pgscf.ForOp(operands[i], operands[rank + i], operands[2 * rank + i])
            ivs.append(loop.induction_variable)
            with _pgir.InsertionPoint(loop.regions[0].blocks[0]):
                terminator = _pgscf.YieldOp([])
            point = _pgir.InsertionPoint(terminator)
    for i in range(rank):
        block.arguments[i].replace_all_uses_with(ivs[i])
    # Everything but the `scf.reduce` terminator moves, in order, into the innermost body.
    for handle in [o.operation for o in block.operations][:-1]:
        handle.move_before(terminator.operation)
    op.erase()


def _parallel_grain(ctx, module):
    """Serialize every `scf.parallel` cheaper than `_PARALLEL_GRAIN` lane-operations.

    Returns the number serialized; prints the kept / unpriceable / reduction-carrying counts so a
    threshold that matched nothing is visible instead of passing for applied."""
    if not _PARALLEL_GRAIN:
        return 0
    found = []

    def _walk(op):
        for region in op.regions:
            for blk in region.blocks:
                for handle in blk.operations:
                    inner = handle.operation
                    if inner.name == "scf.parallel":
                        found.append(inner)
                    _walk(inner)

    _walk(module.operation)
    serialized = kept = unpriceable = reducing = 0
    for op in found:
        if len(op.results):                      # carries a reduction: refuse rather than rewrite
            reducing += 1
            continue
        cost = _pg_cost(op)
        if cost is None:
            unpriceable += 1
            continue
        if cost < _PARALLEL_GRAIN:
            _pg_serialize(op, ctx)
            serialized += 1
        else:
            kept += 1
    print("OK parallel_grain threshold", _PARALLEL_GRAIN, "regions", len(found),
          "serialized", serialized, "kept", kept, "unpriceable", unpriceable,
          "reducing", reducing)
    return serialized

_PT_PLAN = []


def _pt_walk_named(module, wanted):
    found = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for handle in block.operations:
                    inner = handle.operation
                    if inner.name == wanted:
                        found.append(inner)
                    walk(inner)

    walk(module.operation)
    return found


def _pt_width(cost, work, cap):
    needed = max(1, (cost + work - 1) // work)
    width = 1
    while width < needed and width < cap:
        width = min(width * 2, cap)
    return width


def _parallel_team_plan(ctx, module):
    """Record widths in structural order and serialize regions assigned one worker."""
    global _PT_PLAN
    found = _pt_walk_named(module, "scf.parallel")
    _PT_PLAN = []
    serialized = unpriceable = 0
    histogram = {}
    for op in found:
        cost = _pg_cost(op)
        if cost is None or len(op.results):
            width = _PARALLEL_TEAM_CAP
            unpriceable += 1
        else:
            width = _pt_width(cost, _PARALLEL_TEAM_WORK, _PARALLEL_TEAM_CAP)
        if width == 1 and not len(op.results):
            _pg_serialize(op, ctx)
            serialized += 1
            continue
        _PT_PLAN.append(width)
        histogram[width] = histogram.get(width, 0) + 1
    print("OK parallel_team_plan regions", len(found), "serialized", serialized,
          "remaining", len(_PT_PLAN), "unpriceable", unpriceable,
          "widths", ",".join(str(k) + ":" + str(histogram[k]) for k in sorted(histogram)))
    return len(_PT_PLAN)


def _pt_set_num_threads(ctx, old, width):
    """Replace one result-free omp.parallel with an equivalent op carrying num_threads."""
    from torch_mlir import ir as _ptir
    if len(old.results):
        raise RuntimeError("parallel-team policy cannot replace result-carrying omp.parallel")
    block = old.regions[0].blocks[0]
    if len(block.arguments):
        raise RuntimeError("parallel-team policy found omp.parallel block arguments")
    body = [handle.operation for handle in block.operations]
    if not body or body[-1].name != "omp.terminator":
        raise RuntimeError("parallel-team policy found malformed omp.parallel region")
    with ctx, _ptir.Location.unknown(), _ptir.InsertionPoint(old):
        i32 = _ptir.IntegerType.get_signless(32)
        constant = _ptir.Operation.create(
            "arith.constant", results=[i32],
            attributes={"value": _ptir.IntegerAttr.get(i32, int(width))})
        replacement = _ptir.Operation.create(
            "omp.parallel", operands=[constant.results[0]], regions=1,
            attributes={"operandSegmentSizes":
                        _ptir.DenseI32ArrayAttr.get([0, 0, 0, 1, 0, 0])})
        target = replacement.regions[0].blocks.append()
        with _ptir.InsertionPoint(target):
            terminator = _ptir.Operation.create("omp.terminator")
    for inner in body[:-1]:
        inner.move_before(terminator)
    old.erase()


def _parallel_team_apply(ctx, module):
    found = _pt_walk_named(module, "omp.parallel")
    if len(found) != len(_PT_PLAN):
        raise RuntimeError("parallel-team pre/post OpenMP region census differs: planned "
                           + str(len(_PT_PLAN)) + ", converted " + str(len(found)))
    for op, width in zip(found, _PT_PLAN):
        _pt_set_num_threads(ctx, op, width)
    module.operation.verify()
    print("OK parallel_team_apply regions", len(found))
    return len(found)

def _pc_canonical(op):
    """Whether ``op`` is a canonical result/operand-free OpenMP team region."""
    if op.name != "omp.parallel" or len(op.results) or len(op.operands):
        return False
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    block = op.regions[0].blocks[0]
    body = [handle.operation for handle in block.operations]
    return not len(block.arguments) and bool(body) and body[-1].name == "omp.terminator"


def _parallel_coarsen(ctx, module):
    """Merge maximal straight-line runs of canonical ``omp.parallel`` operations."""
    # Conversion materializes duplicate bound constants immediately before individual regions.
    # The normal pipeline removes them during later cleanup, but by then OpenMP has been lowered to
    # LLVM and its team boundaries are no longer editable.  Run that same cleanup here so only real
    # intervening computation separates runs; the pipeline's later canonicalize/CSE is idempotent.
    from torch_mlir.passmanager import PassManager as _pcPassManager
    _pcPassManager.parse("builtin.module(canonicalize,cse)", ctx).run(module.operation)
    merged = groups = original = 0

    def walk(op):
        nonlocal merged, groups, original
        for region in op.regions:
            for block in region.blocks:
                children = [handle.operation for handle in block.operations]
                # Process nested blocks before editing this block, so no recursion observes an
                # operation invalidated by erasing a later member of a run.
                for child in children:
                    walk(child)
                i = 0
                while i < len(children):
                    if not _pc_canonical(children[i]):
                        i += 1
                        continue
                    run = [children[i]]
                    j = i + 1
                    while j < len(children) and _pc_canonical(children[j]):
                        run.append(children[j])
                        j += 1
                    original += len(run)
                    if len(run) > 1:
                        groups += 1
                        target = run[0].regions[0].blocks[0]
                        terminator = [h.operation for h in target.operations][-1]
                        for later in run[1:]:
                            body = [h.operation for h in later.regions[0].blocks[0].operations]
                            for inner in body[:-1]:
                                inner.move_before(terminator)
                            later.erase()
                            merged += 1
                    i = j

    walk(module.operation)
    module.operation.verify()
    print("OK parallel_coarsen original", original, "merged", merged,
          "groups", groups, "remaining", original - merged)
    return merged

_MERLIN_PANEL_MARKER_SYMBOL = "__merlin_parallel_panel_marker"


def _pp_has_nested_parallel(op):
    """Whether ``op`` contains a parallel region below its own body."""
    def walk(inner):
        for region in inner.regions:
            for block in region.blocks:
                for handle in block.operations:
                    child = handle.operation
                    if child.name in ("scf.parallel", "scf.forall", "omp.parallel"):
                        return True
                    if walk(child):
                        return True
        return False
    return walk(op)


def _pp_marker_calls(op):
    """Marker calls below ``op``, preserved through bufferization by their side effect."""
    found = []
    def walk(inner):
        for region in inner.regions:
            for block in region.blocks:
                for handle in block.operations:
                    child = handle.operation
                    if child.name == "func.call":
                        try:
                            if str(child.attributes["callee"]) == "@" + _MERLIN_PANEL_MARKER_SYMBOL:
                                found.append(child)
                        except Exception:
                            pass
                    walk(child)
    walk(op)
    return found


def _parallelize_panel_loops(ctx, module):
    """Rewrite verified bufferized panel carriers and return the number changed."""
    from torch_mlir import ir as _ppir
    from torch_mlir.dialects import scf as _ppscf

    found = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for handle in block.operations:
                    child = handle.operation
                    if child.name == "scf.for" and _pp_marker_calls(child):
                        found.append(child)
                    walk(child)
    walk(module.operation)

    rewritten = refused = nested = 0
    tensor_carrier = nonidentity_carrier = operand_arity = body_arg_arity = 0
    marker_arity = marker_not_direct = bad_yield = 0
    for op in found:
        body = op.regions[0].blocks[0]
        terms = [h.operation for h in body.operations]
        markers = _pp_marker_calls(op)
        # This is the post-bufferization proof boundary.  One-shot bufferization usually erases the
        # tensor carrier completely.  Some accumulator schedules instead leave identity-carried
        # memref/status values: each yield is the unchanged matching block argument.  Those are safe
        # to replace with their initial operands because mutations already happen through the memref.
        # A tensor carrier or any non-identity yield is still refused.
        n_iter = max(0, len(op.operands) - 3)
        _operand_arity = len(op.operands) < 3 or len(op.results) != n_iter
        _body_arg_arity = len(body.arguments) != 1 + n_iter
        _marker_arity = len(markers) != 1
        _marker_not_direct = len(markers) == 1 and markers[0] not in terms
        _bad_yield = not terms or terms[-1].name != "scf.yield"
        yielded = list(terms[-1].operands) if not _bad_yield else []
        iter_args = list(body.arguments)[1:] if not _body_arg_arity else []
        _nonidentity_carrier = (len(yielded) != n_iter or len(iter_args) != n_iter or
                                any(y != arg for y, arg in zip(yielded, iter_args)))
        carrier_types = ([str(x.type) for x in op.results]
                         + [str(x.type) for x in list(op.operands)[3:]]
                         + [str(x.type) for x in iter_args]
                         + [str(x.type) for x in yielded])
        _tensor_carrier = any(t.startswith("tensor<") for t in carrier_types)
        tensor_carrier += int(_tensor_carrier)
        nonidentity_carrier += int(_nonidentity_carrier)
        operand_arity += int(_operand_arity)
        body_arg_arity += int(_body_arg_arity)
        marker_arity += int(_marker_arity)
        marker_not_direct += int(_marker_not_direct)
        bad_yield += int(_bad_yield)
        if (_tensor_carrier or _nonidentity_carrier or _operand_arity or _body_arg_arity or
                _marker_arity or _marker_not_direct or _bad_yield):
            refused += 1
            continue
        if _pp_has_nested_parallel(op):
            refused += 1
            nested += 1
            continue

        operands = list(op.operands)
        lb, ub, step = operands[:3]
        initial = operands[3:]
        old_iv = body.arguments[0]
        with ctx, _ppir.Location.unknown():
            par = _ppscf.ParallelOp([], [lb], [ub], [step], [], ip=_ppir.InsertionPoint(op))
            par.regions[0].blocks.append(lb.type)
            pbody = par.regions[0].blocks[0]
            with _ppir.InsertionPoint(pbody):
                reduce = _ppscf.ReduceOp([], 0)
        old_iv.replace_all_uses_with(pbody.arguments[0])
        for old_arg, init in zip(list(body.arguments)[1:], initial):
            old_arg.replace_all_uses_with(init)
        for result, init in zip(list(op.results), initial):
            result.replace_all_uses_with(init)
        for inner in terms[:-1]:
            if inner == markers[0]:
                inner.erase()
                continue
            inner.move_before(reduce.operation)
        terms[-1].erase()
        op.erase()
        rewritten += 1

    print("OK panel_parallel regions", len(found), "rewritten", rewritten,
          "refused", refused, "nested_parallel", nested,
          "tensor_carrier", tensor_carrier, "nonidentity_carrier", nonidentity_carrier,
          "operand_arity", operand_arity,
          "body_arg_arity", body_arg_arity, "marker_arity", marker_arity,
          "marker_not_direct", marker_not_direct, "bad_yield", bad_yield)
    return rewritten

_EXPAND_MEMREF_COPY = len(sys.argv) > 6 and sys.argv[6] == '1'
_MID_STAGES = []
if _EXPAND_MEMREF_COPY:
    _MID_STAGES.append(("expand_memref_copy", _expand_memref_copies))

_PANEL_PARALLEL = len(sys.argv) > 10 and sys.argv[10] == "1"
if _PANEL_PARALLEL:
    _MID_STAGES.insert(0, ("panel_parallel", _parallelize_panel_loops))

_PARALLEL_GRAIN = int(sys.argv[9]) if len(sys.argv) > 9 else 0
_LATE_STAGES = [("parallel_grain", _parallel_grain)] if _PARALLEL_GRAIN else []

_PARALLEL_TEAM_WORK = int(sys.argv[11]) if len(sys.argv) > 11 else 0
_PARALLEL_TEAM_CAP = int(sys.argv[12]) if len(sys.argv) > 12 else 0
if bool(_PARALLEL_TEAM_WORK) != bool(_PARALLEL_TEAM_CAP):
    raise RuntimeError("parallel-team work and cap must be supplied together")
if _PARALLEL_TEAM_WORK < 0 or _PARALLEL_TEAM_CAP < 0:
    raise RuntimeError("parallel-team work and cap must be non-negative")
if _PARALLEL_TEAM_WORK:
    _LATE_STAGES = [*_LATE_STAGES, ("parallel_team_plan", _parallel_team_plan)]
    _POST_OPENMP_STAGES = [("parallel_team_apply", _parallel_team_apply)]
else:
    _POST_OPENMP_STAGES = []

_PARALLEL_COARSEN = len(sys.argv) > 13 and sys.argv[13] == "1"
if _PARALLEL_COARSEN:
    _POST_OPENMP_STAGES = [*_POST_OPENMP_STAGES,
                           ("parallel_coarsen", _parallel_coarsen)]

def _lower_structured_alloca_scopes(ctx, module):
    """Replace zero-result one-block alloca scopes by explicit LLVM stack lifetime intrinsics."""
    scopes = []

    def visit(op):
        # Post-order makes nested scopes independent: lower the inner lifetime first.
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    visit(child)
        if op.operation.name == 'memref.alloca_scope':
            scopes.append(op.operation)

    for top in list(module.body.operations):
        visit(top)

    lowered = 0
    ptr = ir.Type.parse('!llvm.ptr', ctx)
    for scope in scopes:
        if len(list(scope.results)) != 0 or len(list(scope.regions)) != 1:
            raise RuntimeError('alloca-scope pre-CFG lowering only accepts zero-result scopes')
        blocks = list(scope.regions[0].blocks)
        if len(blocks) != 1:
            raise RuntimeError('alloca-scope reached pre-CFG lowering with ' + str(len(blocks)) +
                               ' blocks; expected exactly one structured block')
        body = blocks[0]
        operations = list(body.operations)
        if not operations or operations[-1].operation.name != 'memref.alloca_scope.return':
            raise RuntimeError('alloca-scope has no recognizable return terminator')
        terminator = operations[-1].operation
        if len(list(terminator.operands)) != 0:
            raise RuntimeError('zero-result alloca-scope return unexpectedly carries operands')

        with ir.InsertionPoint(scope):
            saved = ir.Operation.create(
                'llvm.intr.stacksave', results=[ptr], loc=scope.location).results[0]
        for child in operations[:-1]:
            child.operation.move_before(scope)
        with ir.InsertionPoint(scope):
            ir.Operation.create(
                'llvm.intr.stackrestore', operands=[saved], loc=scope.location)
        terminator.erase()
        scope.erase()
        lowered += 1
    return lowered


_LOWER_ALLOCA_SCOPES = len(sys.argv) > 16 and sys.argv[16] == '1'
if _LOWER_ALLOCA_SCOPES:
    _POST_OPENMP_STAGES = [*_POST_OPENMP_STAGES,
                           ('alloca_scope_pre_cfg', _lower_structured_alloca_scopes)]

def _lower_roundeven_intrinsics(ctx, module):
    """Replace every math.roundeven with llvm.intr.roundeven; return the exact count."""
    from torch_mlir import ir as _ri_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if inner.operation.name == "math.roundeven":
                        todo.append(inner)
                    walk(inner.operation)

    walk(module.operation)
    with ctx, _ri_ir.Location.unknown():
        for old in todo:
            with _ri_ir.InsertionPoint(old):
                new = _ri_ir.Operation.create(
                    "llvm.intr.roundeven", results=[old.results[0].type],
                    operands=[old.operands[0]])
            old.results[0].replace_all_uses_with(new.results[0])
            old.operation.erase()
    module.operation.verify()
    return len(todo)


_RI_MARKER = "__merlin_lower_roundeven_to_intrinsic__"
_RI_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _RI_MARKER not in passes:
        return _RI_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp,
                                   pre_generalize)
    i = passes.index(_RI_MARKER)
    # The marker is after linalg-to-loops: mid/pre-generalize belong to the head, while parallel
    # grain/team/coarsening stages still belong to the tail.  Keeping that routing explicit avoids
    # silently dropping another requested rewrite when this lever is composed with it.
    _RI_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), erase, mid, (), (),
                        pre_generalize)
    print('OK roundeven_intrinsic', _lower_roundeven_intrinsics(ctx, module))
    _RI_ORIG_RUN_STAGES(ctx, module, ','.join(passes[i + 1:]), 0, (), late,
                        post_openmp, ())

def _lower_fma_intrinsics(ctx, module):
    from torch_mlir import ir as _fi_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if inner.operation.name == "math.fma":
                        element = inner.results[0].type
                        if isinstance(element, _fi_ir.VectorType):
                            element = element.element_type
                        if isinstance(element, (_fi_ir.F32Type, _fi_ir.F64Type)):
                            todo.append(inner)
                    walk(inner.operation)

    walk(module.operation)
    with ctx:
        for old in todo:
            with _fi_ir.InsertionPoint(old):
                new = _fi_ir.Operation.create(
                    "llvm.intr.fma", results=[old.results[0].type],
                    operands=list(old.operands), loc=old.location)
                # Retain nonsemantic attribution. Dropping optional fastmath
                # permissions selects the stricter fused operation.
                for name in old.attributes:
                    if name != "fastmath":
                        new.attributes[name] = old.attributes[name]
            old.results[0].replace_all_uses_with(new.results[0])
            old.operation.erase()
    module.operation.verify()
    return len(todo)


_FI_MARKER = "__merlin_lower_fma_to_intrinsic__"
_FI_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _FI_MARKER not in passes:
        return _FI_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late,
                                   post_openmp, pre_generalize)
    i = passes.index(_FI_MARKER)
    _FI_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), erase, mid, (), (),
                        pre_generalize)
    print('OK fma_intrinsic', _lower_fma_intrinsics(ctx, module))
    _FI_ORIG_RUN_STAGES(ctx, module, ','.join(passes[i + 1:]), 0, (), late,
                        post_openmp, ())

def _scalar_contraction_spec(op):
    from torch_mlir import ir as _sc_ir
    if (op.operation.name != "linalg.generic" or len(op.operands) != 3
            or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    types = list(v.type for v in op.operands)
    if any(not isinstance(t, _sc_ir.RankedTensorType) for t in types):
        return None
    types = [_sc_ir.RankedTensorType(t) for t in types]
    if (any(not isinstance(t.element_type, _sc_ir.F32Type) for t in types)
            or op.results[0].type != op.operands[2].type
            or any(d < 0 for t in types for d in t.shape)):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != 3 or len(body) != 3
            or [x.operation.name for x in body] != ["arith.mulf", "arith.addf", "linalg.yield"]
            or any("fastmath" in x.attributes and str(x.attributes["fastmath"]) != "#arith.fastmath<none>" for x in body)
            or any(name != "fastmath" and not name.startswith("prov.")
                   for x in body[:2] for name in x.attributes)):
        return None
    mul, add, yield_op = body
    args = list(block.arguments)
    if list(mul.operands) == args[:2]:
        mul_order = [0, 1]
    elif list(mul.operands) == args[1::-1]:
        mul_order = [1, 0]
    else:
        return None
    if list(add.operands) == [mul.results[0], args[2]]:
        add_order = [0, 1]
    elif list(add.operands) == [args[2], mul.results[0]]:
        add_order = [1, 0]
    else:
        return None
    if list(yield_op.operands) != [add.results[0]]:
        return None
    maps = [_sc_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != 3:
        return None
    dims = maps[0].n_dims
    if dims < 2 or any(m.n_dims != dims or m.n_symbols for m in maps):
        return None
    iters = list(op.attributes["iterator_types"])
    # Enum attribute printer is the upstream binding's stable spelling.
    if ([str(x) for x in iters] !=
            ["#linalg.iterator_type<parallel>"] * (dims - 1) + ["#linalg.iterator_type<reduction>"]):
        return None
    positions = []
    extents = [None] * dims
    for m, t in zip(maps, types):
        if len(m.results) != len(t.shape):
            return None
        pos = []
        for expr, extent in zip(m.results, t.shape):
            if not isinstance(expr, _sc_ir.AffineDimExpr):
                return None
            d = _sc_ir.AffineDimExpr(expr).position
            if d in pos or (extents[d] is not None and extents[d] != extent):
                return None
            pos.append(d)
            extents[d] = extent
        positions.append(pos)
    if (sorted(positions[2]) != list(range(dims - 1))
            or any(dims - 1 not in pos for pos in positions[:2])
            or any(d is None for d in extents)):
        return None
    return positions, extents, mul_order, add_order


def _scalarize_tensor_contractions(ctx, module, outputs=1, reduction_unroll=0, rows=1):
    from torch_mlir import ir as _sc_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _scalar_contraction_spec(inner)
                    eligible = spec is not None and spec[1][-2] % outputs == 0
                    if eligible and rows != 1:
                        positions, extents, _, _ = spec
                        row_axis, col_axis = len(extents) - 3, len(extents) - 2
                        # A rectangular tile is a matrix schedule only when the
                        # exact projected maps prove complementary row/column
                        # reuse. No names, shapes alone or pointer facts suffice.
                        eligible = (row_axis >= 0 and extents[row_axis] % rows == 0
                                    and row_axis in positions[0] and col_axis not in positions[0]
                                    and col_axis in positions[1] and row_axis not in positions[1])
                        owner = inner.operation
                        while eligible and owner is not None:
                            if any(a in owner.attributes for a in ("strictfp", "llvm.strictfp")):
                                eligible = False
                            owner = owner.parent
                    if eligible:
                        todo.append((inner, spec))
                    else:
                        walk(inner.operation)

    walk(module.operation)
    with ctx:
        for old, (positions, extents, mul_order, add_order) in todo:
            with old.location, _sc_ir.InsertionPoint(old):
                index = _sc_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _sc_ir.Operation.create(name, operands=list(operands), results=list(results),
                                                   attributes=attributes or {}, regions=regions)
                def constant(value):
                    return create("arith.constant", results=[index], attributes={
                        "value": _sc_ir.IntegerAttr.get(index, value)}).results[0]
                zero, one = constant(0), constant(1)
                limits = [constant(e) for e in extents]
                step = one if outputs == 1 else constant(outputs)
                offsets = [zero, *[constant(i) for i in range(1, outputs)]]
                row_offsets = [zero, *[constant(i) for i in range(1, rows)]]
                row_step = one if rows == 1 else constant(rows)

                def output_loop(depth, current, indices):
                    if depth < len(extents) - 1:
                        stride = (step if depth == len(extents) - 2 else
                                  row_step if rows != 1 and depth == len(extents) - 3 else one)
                        loop = create("scf.for", [zero, limits[depth], stride, current], [tensor], regions=1)
                        block = _sc_ir.Block.create_at_start(loop.regions[0], [index, tensor])
                        with _sc_ir.InsertionPoint(block):
                            updated = output_loop(depth + 1, block.arguments[1], [*indices, block.arguments[0]])
                            create("scf.yield", [updated])
                        return loop.results[0]
                    if rows == 1:
                        lanes = [indices]
                        for offset in offsets[1:]:
                            column = create("arith.addi", [indices[-1], offset], [index]).results[0]
                            lanes.append([*indices[:-1], column])
                    else:
                        lanes = []
                        for row_offset in row_offsets:
                            row = (indices[-2] if row_offset == zero else
                                   create("arith.addi", [indices[-2], row_offset], [index]).results[0])
                            for offset in offsets:
                                column = (indices[-1] if offset == zero else
                                          create("arith.addi", [indices[-1], offset], [index]).results[0])
                                lanes.append([*indices[:-2], row, column])
                    out_indices = [[lane[d] for d in positions[2]] for lane in lanes]
                    initial = [create("tensor.extract", [current, *idx], [scalar]).results[0]
                               for idx in out_indices]
                    # SCF -> CF forwards LLVM-typed values; CF -> LLVM keeps the key.
                    # Use the final branch property's spelling: a retained namespaced
                    # llvm.loop_annotation is ignored by the installed translator.
                    attributes = ({"loop_annotation": _sc_ir.Attribute.parse(
                        "#llvm.loop_annotation<unroll = <count = " + str(reduction_unroll) + " : i32>>")}
                        if reduction_unroll else {})
                    lane_count = rows * outputs
                    loop = create("scf.for", [zero, limits[-1], one, *initial], [scalar] * lane_count,
                                  attributes=attributes, regions=1)
                    block = _sc_ir.Block.create_at_start(loop.regions[0], [index, *([scalar] * lane_count)])
                    with _sc_ir.InsertionPoint(block):
                        accumulated, shared = [], {}
                        for lane_number, lane in enumerate(lanes):
                            all_indices = [*lane, block.arguments[0]]
                            values = []
                            for i in range(2):
                                if rows == 1:
                                    key = i if len(extents) - 2 not in positions[i] else None
                                else:
                                    key = (i,
                                           lane_number // outputs if len(extents) - 3 in positions[i] else 0,
                                           lane_number % outputs if len(extents) - 2 in positions[i] else 0)
                                if key is not None and key in shared:
                                    value = shared[key]
                                else:
                                    value = create("tensor.extract", [old.operands[i], *[all_indices[d] for d in positions[i]]],
                                                   [scalar]).results[0]
                                    if key is not None:
                                        shared[key] = value
                                values.append(value)
                            product = create("arith.mulf", [values[d] for d in mul_order], [scalar]).results[0]
                            operands = [product, block.arguments[lane_number + 1]]
                            accumulated.append(create("arith.addf", [operands[d] for d in add_order], [scalar]).results[0])
                        create("scf.yield", accumulated)
                    for value, idx in zip(loop.results, out_indices):
                        current = create("tensor.insert", [value, current, *idx], [tensor]).results[0]
                    return current

                result = output_loop(0, old.operands[2], [])
                replacement = result.owner
                for name in old.attributes:
                    if name.startswith("prov."):
                        replacement.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                text = _sc_ir.StringAttr(previous).value + "," if previous is not None else ""
                suffix = ("_2x4_outputs" if rows != 1 else
                          "" if outputs == 1 else "_" + str(outputs) + "_outputs")
                transforms = text + "scalar_contraction_accumulator" + suffix
                if reduction_unroll:
                    transforms += ",unroll_scalar_contraction_reduction_by_" + str(reduction_unroll)
                replacement.attributes["prov.transforms"] = _sc_ir.StringAttr.get(transforms)
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)


_SC_MARKER = "__merlin_scalar_contraction_accumulator__"
_SC2_MARKER = "__merlin_scalar_contraction_accumulator_2_outputs__"
_SC4_MARKER = "__merlin_scalar_contraction_accumulator_4_outputs__"
_SC8_MARKER = "__merlin_scalar_contraction_accumulator_8_outputs__"
_SC2X4_MARKER = "__merlin_scalar_contraction_accumulator_2x4_outputs__"
_SC_MARKERS = (_SC_MARKER, _SC2_MARKER, _SC4_MARKER, _SC8_MARKER, _SC2X4_MARKER)
_SC_UNROLL2_MARKER = "__merlin_scalar_contraction_reduction_unroll_2__"
_SC_UNROLL4_MARKER = "__merlin_scalar_contraction_reduction_unroll_4__"
_SC_UNROLL_MARKERS = {_SC_UNROLL2_MARKER: 2, _SC_UNROLL4_MARKER: 4}
_SC_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    hints = [m for m in _SC_UNROLL_MARKERS if m in passes]
    if len(hints) > 1:
        raise ValueError("scalar contraction reduction unroll counts are alternatives")
    unroll = _SC_UNROLL_MARKERS[hints[0]] if hints else 0
    passes = [p for p in passes if p not in _SC_UNROLL_MARKERS]
    selected = [m for m in _SC_MARKERS if m in passes]
    if not selected:
        return _SC_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if len(selected) != 1:
        raise ValueError("scalar contraction accumulator schedules are alternatives")
    i = passes.index(selected[0])
    head_generalizes = any('linalg-generalize-named-ops' in p for p in passes[:i])
    _SC_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), 0, (), (), (),
                        pre_generalize if head_generalizes else ())
    _SC_ORIG_RUN_STAGES(ctx, module, 'func.func(linalg-generalize-named-ops)', 0, (), (), (),
                        () if head_generalizes else pre_generalize)
    rows = 2 if selected[0] == _SC2X4_MARKER else 1
    outputs = {_SC_MARKER: 1, _SC2_MARKER: 2, _SC4_MARKER: 4, _SC8_MARKER: 8, _SC2X4_MARKER: 4}[selected[0]]
    name = 'scalar_contraction_accumulator' + ('_2x4_outputs' if rows != 1 else '' if outputs == 1 else '_' + str(outputs) + '_outputs')
    count = _scalarize_tensor_contractions(ctx, module, outputs, unroll, rows)
    print('OK', name, count)
    if unroll:
        print('OK unroll_scalar_contraction_reduction_by_' + str(unroll), count)
    _SC_ORIG_RUN_STAGES(ctx, module, ','.join(passes[i + 1:]), erase, mid, late, post_openmp, ())

def _sqs_spec(op):
    from torch_mlir import ir as _sqs_ir
    if (op.operation.name != "linalg.generic" or len(op.operands) != 2
            or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    ancestor = op.operation
    while ancestor is not None:
        if any("strictfp" in name or "strictfp" in str(ancestor.attributes[name])
               for name in ancestor.attributes):
            return None
        ancestor = ancestor.parent
    types = []
    for value in op.operands:
        if not isinstance(value.type, _sqs_ir.RankedTensorType):
            return None
        tensor = _sqs_ir.RankedTensorType(value.type)
        if (not isinstance(tensor.element_type, _sqs_ir.F32Type)
                or any(extent < 0 for extent in tensor.shape)):
            return None
        types.append(tensor)
    if op.results[0].type != op.operands[1].type:
        return None
    allowed_attributes = {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    if any(name not in allowed_attributes and not name.startswith("prov.")
           for name in op.attributes):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != 2 or len(body) != 3
            or [x.operation.name for x in body]
                != ["arith.mulf", "arith.addf", "linalg.yield"]
            or any(x.regions for x in body)
            or any(name != "fastmath" and not name.startswith("prov.")
                   for scalar in body for name in scalar.attributes)
            or any("fastmath" in x.attributes
                   and str(x.attributes["fastmath"]) != "#arith.fastmath<none>"
                   for x in body)):
        return None
    mul, add, terminator = body
    value, seed = block.arguments
    if list(mul.operands) != [value, value]:
        return None
    if list(add.operands) == [mul.results[0], seed]:
        order = [0, 1]
    elif list(add.operands) == [seed, mul.results[0]]:
        order = [1, 0]
    else:
        return None
    if list(terminator.operands) != [add.results[0]]:
        return None
    maps = [_sqs_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != 2:
        return None
    dims = maps[0].n_dims
    if (dims < 1 or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]]
                != ["#linalg.iterator_type<parallel>"] * (dims - 1)
                   + ["#linalg.iterator_type<reduction>"]):
        return None
    extents, positions = [None] * dims, []
    for affine, tensor in zip(maps, types):
        if len(affine.results) != len(tensor.shape):
            return None
        used = []
        for expr, extent in zip(affine.results, tensor.shape):
            if not isinstance(expr, _sqs_ir.AffineDimExpr):
                return None
            position = _sqs_ir.AffineDimExpr(expr).position
            if (position in used or position >= dims
                    or (extents[position] is not None and extents[position] != extent)):
                return None
            used.append(position)
            extents[position] = extent
        positions.append(used)
    if (sorted(positions[0]) != list(range(dims))
            or sorted(positions[1]) != list(range(dims - 1))
            or any(extent is None for extent in extents)):
        return None
    return positions, extents, order


def _scalarize_squared_sums(ctx, module):
    from torch_mlir import ir as _sqs_ir
    todo = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec = _sqs_spec(child)
                    if spec is not None:
                        todo.append((child, spec))
                    else:
                        walk(child.operation)
    walk(module.operation)
    with ctx:
        for old, (positions, extents, add_order) in todo:
            with old.location, _sqs_ir.InsertionPoint(old):
                index = _sqs_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _sqs_ir.Operation.create(name, operands=list(operands),
                        results=list(results), attributes=attributes or {}, regions=regions)
                def constant(value):
                    return create("arith.constant", results=[index], attributes={
                        "value": _sqs_ir.IntegerAttr.get(index, value)}).results[0]
                zero, one = constant(0), constant(1)
                limits = [constant(extent) for extent in extents]
                source_mul, source_add = list(old.regions[0].blocks[0].operations)[:2]
                def output_loop(depth, current, coordinates):
                    if depth < len(extents) - 1:
                        loop = create("scf.for", [zero, limits[depth], one, current],
                                      [tensor], regions=1)
                        block = _sqs_ir.Block.create_at_start(loop.regions[0], [index, tensor])
                        with _sqs_ir.InsertionPoint(block):
                            updated = output_loop(depth + 1, block.arguments[1],
                                                  [*coordinates, block.arguments[0]])
                            create("scf.yield", [updated])
                        return loop.results[0]
                    out_indices = [coordinates[d] for d in positions[1]]
                    initial = create("tensor.extract", [current, *out_indices], [scalar]).results[0]
                    loop = create("scf.for", [zero, limits[-1], one, initial],
                                  [scalar], regions=1)
                    block = _sqs_ir.Block.create_at_start(loop.regions[0], [index, scalar])
                    with _sqs_ir.InsertionPoint(block):
                        all_indices = [*coordinates, block.arguments[0]]
                        value = create("tensor.extract", [old.operands[0],
                            *[all_indices[d] for d in positions[0]]], [scalar]).results[0]
                        product = create("arith.mulf", [value, value], [scalar],
                            {name: source_mul.attributes[name] for name in source_mul.attributes}).results[0]
                        values = [product, block.arguments[1]]
                        total = create("arith.addf", [values[d] for d in add_order], [scalar],
                            {name: source_add.attributes[name] for name in source_add.attributes}).results[0]
                        create("scf.yield", [total])
                    return create("tensor.insert", [loop.results[0], current, *out_indices],
                                  [tensor]).results[0]
                result = output_loop(0, old.operands[1], [])
                for name in old.attributes:
                    if name.startswith("prov."):
                        result.owner.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                prefix = _sqs_ir.StringAttr(previous).value + "," if previous is not None else ""
                result.owner.attributes["prov.transforms"] = _sqs_ir.StringAttr.get(
                    prefix + "scalar_squared_sum_accumulator")
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)

_SQS_MARKER = "__merlin_scalar_squared_sum_accumulator__"
_SQS_ORIG_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _SQS_MARKER not in passes:
        return _SQS_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(_SQS_MARKER) != 1:
        raise ValueError("scalar squared sum marker must occur once")
    at = passes.index(_SQS_MARKER)
    _SQS_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:at]), 0, (), (), (), pre_generalize)
    print('OK scalar_squared_sum_accumulator', _scalarize_squared_sums(ctx, module))
    _SQS_ORIG_RUN_STAGES(ctx, module, ','.join(passes[at + 1:]), erase, mid, late, post_openmp, ())

def _pointwise_unroll_candidate(op):
    from torch_mlir import ir as _pu_ir
    if (op.operation.name != "scf.for" or len(op.operands) != 3
            or len(op.results) or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1
            or "loop_annotation" in op.attributes
            or "llvm.loop_annotation" in op.attributes):
        return False
    bounds = []
    for operand in op.operands:
        owner = operand.owner
        owner = owner.operation if isinstance(owner, _pu_ir.OpView) else owner
        if not isinstance(owner, _pu_ir.Operation) or owner.name != "arith.constant":
            return False
        try:
            bounds.append(_pu_ir.IntegerAttr(owner.attributes["value"]).value)
        except (TypeError, ValueError):
            return False
    lower, upper, step = bounds
    if step != 1 or upper - lower < 2:
        return False
    block = op.regions[0].blocks[0]
    if len(block.arguments) != 1:
        return False
    fmas = 0
    division = False
    load = store = False
    for inner in block.operations:
        name = inner.operation.name
        if inner.regions or any(isinstance(v.type, _pu_ir.VectorType)
                                for v in [*inner.operands, *inner.results]):
            return False
        if "fastmath" in inner.attributes and str(inner.attributes["fastmath"]) not in (
                "#arith.fastmath<none>", "#llvm.fastmath<none>"):
            return False
        if name in ("math.fma", "llvm.intr.fma"):
            if len(inner.results) != 1 or not isinstance(inner.results[0].type, _pu_ir.F32Type):
                return False
            fmas += 1
        elif name == "arith.divf":
            if not isinstance(inner.results[0].type, _pu_ir.F32Type):
                return False
            division = True
        elif name == "memref.load":
            load = True
        elif name == "memref.store":
            store = True
        elif not (name.startswith("arith.") or name == "affine.apply"
                  or (name == "scf.yield" and not inner.operands)):
            return False
    # Existing FMA chains are selected by scalar operation cost, not provenance.
    return fmas >= 4 and division and load and store


def _annotate_pointwise_unroll(ctx, module):
    from torch_mlir import ir as _pu_ir
    todo = []

    def walk(op):
        if any("strictfp" in str(op.attributes[name]) for name in op.attributes):
            return
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if _pointwise_unroll_candidate(inner):
                        todo.append(inner)
                    else:
                        walk(inner.operation)

    walk(module.operation)
    with ctx:
        annotation = _pu_ir.Attribute.parse("#llvm.loop_annotation<unroll = <count = 2 : i32>>")
        for op in todo:
            op.attributes["loop_annotation"] = annotation
    module.operation.verify()
    return len(todo)


_PU_MARKER = "__merlin_scalar_pointwise_fma_division_unroll_2__"
_PU_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _PU_MARKER not in passes:
        return _PU_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(_PU_MARKER) != 1:
        raise ValueError("scalar pointwise unroll stage occurs more than once")
    i = passes.index(_PU_MARKER)
    _PU_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), erase, mid, (), (), pre_generalize)
    print('OK scalar_pointwise_fma_division_unroll', _annotate_pointwise_unroll(ctx, module))
    _PU_ORIG_RUN_STAGES(ctx, module, ','.join(passes[i + 1:]), 0, (), late, post_openmp, ())

def _pointwise_packet_spec(op, multiplication=False, two_products=False):
    from torch_mlir import ir as _pp_ir
    if (op.operation.name != "linalg.generic" or len(op.results) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
        return None
    types = list(v.type for v in op.operands)
    if any(not isinstance(t, _pp_ir.RankedTensorType) or any(d < 0 for d in t.shape) for t in types):
        return None
    if len(op.operands) < 2:
        return None
    maps = [_pp_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    dims = maps[-1].n_dims
    if (dims < 1 or len(maps) != len(types)
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]] != ["#linalg.iterator_type<parallel>"] * dims):
        return None
    positions, extents = [], [None] * dims
    for m, t in zip(maps, types):
        if len(m.results) != len(t.shape):
            return None
        pos = []
        for expr, extent in zip(m.results, t.shape):
            if (multiplication and isinstance(expr, _pp_ir.AffineConstantExpr)
                    and _pp_ir.AffineConstantExpr(expr).value == 0 and extent == 1):
                pos.append(None)
                continue
            if not isinstance(expr, _pp_ir.AffineDimExpr):
                return None
            d = _pp_ir.AffineDimExpr(expr).position
            if d in pos or (extents[d] is not None and extents[d] != extent):
                return None
            pos.append(d)
            extents[d] = extent
        positions.append(pos)
    if (None in positions[-1] or sorted(positions[-1]) != list(range(dims))
            or any(d is None or d <= 0 for d in extents)):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if len(block.arguments) != len(types) or not body or body[-1].operation.name != "linalg.yield" or len(body[-1].operands) != 1:
        return None
    if list(block.arguments[-1].uses):
        return None
    if two_products:
        # A separate closed arithmetic family. The existing >=3 selector keeps
        # its original eligibility and is never silently broadened.
        if (not multiplication or not isinstance(types[-1].element_type, _pp_ir.F32Type)
                or any(inner.operation.name not in {"arith.constant", "arith.sitofp",
                       "arith.uitofp", "arith.mulf", "arith.addf", "linalg.yield"}
                       for inner in body)):
            return None
    allowed = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
               "arith.negf", "arith.maximumf", "arith.minimumf", "arith.cmpf", "arith.cmpi",
               "arith.select", "arith.andi", "arith.ori", "arith.xori", "arith.addi", "arith.subi",
               "arith.muli", "arith.shli", "arith.shrsi", "arith.shrui", "arith.bitcast",
               "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui", "arith.extsi",
               "arith.extui", "arith.trunci", "arith.extf", "arith.truncf", "math.fma"}
    fmas = division = multiplies = 0
    for inner in body[:-1]:
        if inner.operation.name not in allowed or inner.regions or len(inner.results) != 1:
            return None
        if "fastmath" in inner.attributes and str(inner.attributes["fastmath"]) != "#arith.fastmath<none>":
            return None
        if isinstance(inner.results[0].type, _pp_ir.VectorType):
            return None
        if multiplication:
            if any("strictfp" in name or "strictfp" in str(inner.attributes[name])
                   for name in inner.attributes):
                return None
            if any(not isinstance(v.type, (_pp_ir.F32Type, _pp_ir.IntegerType, _pp_ir.IndexType))
                   for v in list(inner.operands) + list(inner.results)):
                return None
            if inner.operation.name in {"math.fma", "arith.divf", "arith.extf", "arith.truncf"}:
                return None
            if inner.operation.name == "arith.mulf" and isinstance(inner.results[0].type, _pp_ir.F32Type):
                multiplies += 1
        if inner.operation.name == "math.fma" and isinstance(inner.results[0].type, _pp_ir.F32Type):
            fmas += 1
        if inner.operation.name == "arith.divf" and isinstance(inner.results[0].type, _pp_ir.F32Type):
            division += 1
    eligible = (multiplies == 2 if two_products else multiplies >= 3) if multiplication else fmas >= 4 and division
    return (positions, extents, body) if eligible else None


def _pointwise_broadcast_axis(op, positions, extents, lanes):
    # Scalar inputs are already loop invariant and do not justify an interchange.
    # Every shared load comes from the exact same immutable tensor and coordinates.
    args = list(op.regions[0].blocks[0].arguments)
    choices = []
    for axis, extent in enumerate(extents):
        if extent < lanes:
            continue
        shared = sum(bool(pos) and axis not in pos and bool(list(arg.uses))
                     for arg, pos in zip(args[:-1], positions[:-1]))
        if shared:
            choices.append((shared, extent, -axis))
    return -max(choices)[2] if choices else None


def _packetize_pointwise(ctx, module, lanes=2, broadcast=False, multiplication=False,
                         two_products=False):
    from torch_mlir import ir as _pp_ir
    todo = []
    def walk(op):
        if any("strictfp" in name or "strictfp" in str(op.attributes[name]) for name in op.attributes):
            return
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _pointwise_packet_spec(inner, multiplication=multiplication,
                                                   two_products=two_products)
                    if spec is not None:
                        axis = _pointwise_broadcast_axis(inner, spec[0], spec[1], lanes) if broadcast else len(spec[1]) - 1
                        if axis is not None:
                            todo.append((inner, spec, axis))
                    else:
                        walk(inner.operation)
    walk(module.operation)
    with ctx:
        for old, (positions, extents, scalar_body), axis in todo:
            with old.location, _pp_ir.InsertionPoint(old):
                index = _pp_ir.IndexType.get()
                tensor_type = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _pp_ir.Operation.create(name, operands=list(operands), results=list(results),
                                                  attributes=attributes or {}, regions=regions)
                constants = {}
                def c(n):
                    if n not in constants:
                        constants[n] = create("arith.constant", results=[index], attributes={"value": _pp_ir.IntegerAttr.get(index, n)}).results[0]
                    return constants[n]
                # Constants dominate all subsequently constructed loops.
                for n in [0, 1, lanes, *extents, *range(lanes), extents[axis] // lanes * lanes]:
                    c(n)
                def packet(current, indices, width):
                    lane_indices = [indices]
                    for lane in range(1, width):
                        coordinate = create("arith.addi", [indices[axis], c(lane)], [index]).results[0]
                        ids = list(indices)
                        ids[axis] = coordinate
                        lane_indices.append(ids)
                    mappings = []
                    shared_extracts = {}
                    args = list(old.regions[0].blocks[0].arguments)
                    for ids in lane_indices:
                        mapped = {}
                        for arg, operand, pos in zip(args[:-1], old.operands[:-1], positions[:-1]):
                            key = (operand, *[c(0) if d is None else ids[d] for d in pos])
                            if broadcast and key in shared_extracts:
                                mapped[arg] = shared_extracts[key]
                            else:
                                mapped[arg] = create("tensor.extract", [operand, *key[1:]], [arg.type]).results[0]
                                if broadcast:
                                    shared_extracts[key] = mapped[arg]
                        mappings.append(mapped)
                    # Breadth-first scalar operation order exposes independent chains.
                    for scalar_op in scalar_body[:-1]:
                        for mapped in mappings:
                            cloned = create(scalar_op.operation.name,
                                [mapped.get(v, v) for v in scalar_op.operands],
                                [r.type for r in scalar_op.results],
                                {name: scalar_op.attributes[name] for name in scalar_op.attributes})
                            mapped[scalar_op.results[0]] = cloned.results[0]
                    for ids, mapped in zip(lane_indices, mappings):
                        value = mapped.get(scalar_body[-1].operands[0], scalar_body[-1].operands[0])
                        current = create("tensor.insert", [value, current, *[ids[d] for d in positions[-1]]], [tensor_type]).results[0]
                    return current
                def nest(depth, current, indices):
                    if depth == len(extents) - 1:
                        end = extents[-1] // lanes * lanes
                        if end:
                            loop = create("scf.for", [c(0), c(end), c(lanes), current], [tensor_type], regions=1)
                            block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                            with _pp_ir.InsertionPoint(block):
                                value = packet(block.arguments[1], [*indices, block.arguments[0]], lanes)
                                create("scf.yield", [value])
                            current = loop.results[0]
                        tail = extents[-1] - end
                        if tail:
                            current = packet(current, [*indices, c(end)], tail)
                        return current
                    loop = create("scf.for", [c(0), c(extents[depth]), c(1), current], [tensor_type], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                    with _pp_ir.InsertionPoint(block):
                        value = nest(depth + 1, block.arguments[1], [*indices, block.arguments[0]])
                        create("scf.yield", [value])
                    return loop.results[0]
                def broadcast_group(depth, current, coordinates, width):
                    order = [d for d in range(len(extents)) if d != axis]
                    if depth == len(order):
                        return packet(current, [coordinates[d] for d in range(len(extents))], width)
                    d = order[depth]
                    loop = create("scf.for", [c(0), c(extents[d]), c(1), current], [tensor_type], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                    with _pp_ir.InsertionPoint(block):
                        value = broadcast_group(depth + 1, block.arguments[1],
                                                {**coordinates, d: block.arguments[0]}, width)
                        create("scf.yield", [value])
                    return loop.results[0]

                if broadcast:
                    current = old.operands[-1]
                    end = extents[axis] // lanes * lanes
                    loop = create("scf.for", [c(0), c(end), c(lanes), current], [tensor_type], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                    with _pp_ir.InsertionPoint(block):
                        value = broadcast_group(0, block.arguments[1], {axis: block.arguments[0]}, lanes)
                        create("scf.yield", [value])
                    result = loop.results[0]
                    tail = extents[axis] - end
                    if tail:
                        result = broadcast_group(0, result, {axis: c(end)}, tail)
                else:
                    result = nest(0, old.operands[-1], [])
                replacement = result.owner
                for name in old.attributes:
                    if name.startswith("prov."):
                        replacement.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                text = _pp_ir.StringAttr(previous).value + "," if previous is not None else ""
                transform = ("scalar_pointwise_two_multiplications_packet_" if two_products else
                             "scalar_pointwise_multiplication_packet_" if multiplication else
                             "scalar_pointwise_broadcast_packet_" if broadcast else "scalar_pointwise_packet_")
                replacement.attributes["prov.transforms"] = _pp_ir.StringAttr.get(text + transform + str(lanes))
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)

_PP_MARKERS = {"__merlin_pointwise_packet_2__": 2, "__merlin_pointwise_packet_4__": 4,
               "__merlin_pointwise_broadcast_packet_2__": 2,
               "__merlin_pointwise_multiplication_packet_4__": 4,
               "__merlin_pointwise_two_multiplications_packet_4__": 4}
_PP_ORIG_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    selected = [m for m in passes if m in _PP_MARKERS]
    if not selected:
        return _PP_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    multiplication_marker = "__merlin_pointwise_multiplication_packet_4__"
    two_products_marker = "__merlin_pointwise_two_multiplications_packet_4__"
    multiplication_markers = {multiplication_marker, two_products_marker}
    if (len([m for m in selected if m not in multiplication_markers]) > 1
            or any(selected.count(m) > 1 for m in multiplication_markers)):
        raise ValueError("pointwise packet schedules are alternatives")
    begin = 0
    for marker in selected:
        i = passes.index(marker, begin)
        _PP_ORIG_RUN_STAGES(ctx, module, ','.join(passes[begin:i]), 0, (), (), (),
                           pre_generalize if begin == 0 else ())
        print('OK scalar_pointwise_packet', _PP_MARKERS[marker], _packetize_pointwise(
            ctx, module, _PP_MARKERS[marker],
            broadcast=marker == "__merlin_pointwise_broadcast_packet_2__",
            multiplication=marker in multiplication_markers,
            two_products=marker == two_products_marker))
        begin = i + 1
    _PP_ORIG_RUN_STAGES(ctx, module, ','.join(passes[begin:]), erase, mid, late, post_openmp, ())

def _broadcast_math_spec(op):
    from torch_mlir import ir as _bh_ir
    if any("strictfp" in name or "strictfp" in str(op.attributes[name])
           for name in op.attributes):
        return None
    if (op.operation.name != "linalg.generic" or len(op.results) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or len(op.operands) < 2):
        return None
    types = [value.type for value in op.operands]
    if any(not isinstance(t, _bh_ir.RankedTensorType)
           or any(extent < 0 for extent in t.shape) for t in types):
        return None
    maps = [_bh_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    dims = maps[-1].n_dims
    if (dims < 1 or len(maps) != len(types)
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]]
               != ["#linalg.iterator_type<parallel>"] * dims):
        return None
    # Every domain dimension is fixed by the complete output permutation.
    out_dims = []
    extents = [None] * dims
    for expr, extent in zip(maps[-1].results, types[-1].shape):
        if not isinstance(expr, _bh_ir.AffineDimExpr):
            return None
        dimension = _bh_ir.AffineDimExpr(expr).position
        if dimension in out_dims:
            return None
        out_dims.append(dimension)
        extents[dimension] = extent
    if (sorted(out_dims) != list(range(dims))
            or any(extent is None or extent <= 0 for extent in extents)):
        # A producer must never evaluate an operation absent from an empty consumer.
        return None
    dependencies = []
    for affine, tensor in zip(maps, types):
        if len(affine.results) != len(tensor.shape):
            return None
        used = set()
        for expr, extent in zip(affine.results, tensor.shape):
            if isinstance(expr, _bh_ir.AffineDimExpr):
                dimension = _bh_ir.AffineDimExpr(expr).position
                if dimension in used or extents[dimension] != extent:
                    return None
                used.add(dimension)
            elif isinstance(expr, _bh_ir.AffineConstantExpr):
                if _bh_ir.AffineConstantExpr(expr).value != 0 or extent != 1:
                    return None
            else:
                return None
        dependencies.append(frozenset(used))
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != len(types) or not body
            or body[-1].operation.name != "linalg.yield"
            or len(body[-1].operands) != 1):
        return None
    # These registered source operations are scalar and have no observable effects.
    # Reject an entire body containing a call/index/memory/unknown operation: moving
    # a pure chain across it could change an observed floating environment.
    allowed = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
               "arith.negf", "arith.maximumf", "arith.minimumf", "arith.cmpf", "arith.cmpi",
               "arith.select", "arith.andi", "arith.ori", "arith.xori", "arith.addi", "arith.subi",
               "arith.muli", "arith.shli", "arith.shrsi", "arith.shrui", "arith.bitcast",
               "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui", "arith.extsi",
               "arith.extui", "arith.trunci", "arith.extf", "arith.truncf", "math.rsqrt"}
    scalar_deps = dict(zip(block.arguments, dependencies))
    defining = {}
    candidates = []
    for scalar in body[:-1]:
        if any("strictfp" in name or "strictfp" in str(scalar.attributes[name])
               for name in scalar.attributes):
            return None
        if (scalar.operation.name not in allowed or scalar.regions
                or len(scalar.results) != 1
                or isinstance(scalar.results[0].type, (_bh_ir.VectorType, _bh_ir.ShapedType))):
            return None
        if ("fastmath" in scalar.attributes
                and str(scalar.attributes["fastmath"]) != "#arith.fastmath<none>"):
            return None
        deps = frozenset().union(*(scalar_deps.get(value, frozenset())
                                  for value in scalar.operands))
        scalar_deps[scalar.results[0]] = deps
        defining[scalar.results[0]] = scalar
        if (scalar.operation.name == "math.rsqrt"
                and isinstance(scalar.results[0].type,
                               (_bh_ir.F16Type, _bh_ir.F32Type, _bh_ir.F64Type))
                and any(d not in deps and extents[d] > 1 for d in range(dims))):
            candidates.append(scalar)
    if not candidates:
        return None
    target = candidates[0]
    needed = set()
    def need(value):
        if value in needed:
            return
        needed.add(value)
        scalar = defining.get(value)
        if scalar is not None:
            for operand in scalar.operands:
                need(operand)
    need(target.results[0])
    # A dependency on an output-init block argument cannot be read by a fresh producer.
    if block.arguments[-1] in needed:
        return None
    inputs = [i for i, value in enumerate(block.arguments[:-1]) if value in needed]
    chain = [scalar for scalar in body[:-1] if scalar.results[0] in needed]
    keep = sorted(scalar_deps[target.results[0]])
    return maps, extents, body, target, inputs, chain, keep


def _hoist_broadcast_source_math(ctx, module):
    from torch_mlir import ir as _bh_ir
    todo = []
    def walk(operation):
        if any("strictfp" in name or "strictfp" in str(operation.attributes[name])
               for name in operation.attributes):
            return
        for region in operation.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _broadcast_math_spec(inner)
                    if spec is not None:
                        todo.append((inner, spec))
                    else:
                        walk(inner.operation)
    walk(module.operation)
    with ctx:
        for old, (maps, extents, body, target, inputs, chain, keep) in todo:
            old_args = list(old.regions[0].blocks[0].arguments)
            with old.location:
                reduced_type = _bh_ir.RankedTensorType.get(
                    [extents[d] for d in keep], target.results[0].type)
            def create(name, operands=(), results=(), attributes=None, regions=0):
                return _bh_ir.Operation.create(name, operands=list(operands), results=list(results),
                                              attributes=attributes or {}, regions=regions)
            def attrs(operation):
                return {name: operation.attributes[name] for name in operation.attributes}
            def clone(scalar, values):
                with scalar.location:
                    copy = create(scalar.operation.name,
                                  [values.get(v, v) for v in scalar.operands],
                                  [v.type for v in scalar.results], attrs(scalar))
                values[scalar.results[0]] = copy.results[0]
            def map_on_reduced(affine):
                results = []
                for expr in affine.results:
                    if isinstance(expr, _bh_ir.AffineDimExpr):
                        position = _bh_ir.AffineDimExpr(expr).position
                        results.append(_bh_ir.AffineDimExpr.get(keep.index(position)))
                    else:
                        results.append(_bh_ir.AffineConstantExpr.get(0))
                return _bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get(len(keep), 0, results))
            with old.location, _bh_ir.InsertionPoint(old):
                empty = create("tensor.empty", results=[reduced_type])
                producer_attrs = {
                    "indexing_maps": _bh_ir.ArrayAttr.get(
                        [map_on_reduced(maps[i]) for i in inputs]
                        + [_bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get_identity(len(keep)))]),
                    "iterator_types": _bh_ir.ArrayAttr.get(
                        [_bh_ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(keep)),
                    "operandSegmentSizes": _bh_ir.DenseI32ArrayAttr.get([len(inputs), 1]),
                }
                producer = create("linalg.generic", [*[old.operands[i] for i in inputs], empty.results[0]],
                                  [reduced_type], producer_attrs, 1)
                producer_block = _bh_ir.Block.create_at_start(
                    producer.regions[0], [*[old_args[i].type for i in inputs], target.results[0].type])
                with _bh_ir.InsertionPoint(producer_block):
                    values = {old_args[i]: producer_block.arguments[j] for j, i in enumerate(inputs)}
                    for scalar in chain:
                        clone(scalar, values)
                    create("linalg.yield", [values[target.results[0]]])
                consumer_attrs = attrs(old)
                consumer_attrs["indexing_maps"] = _bh_ir.ArrayAttr.get(
                    [*[_bh_ir.AffineMapAttr.get(m) for m in maps[:-1]],
                     _bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get(maps[-1].n_dims, 0,
                                               [_bh_ir.AffineDimExpr.get(d) for d in keep])),
                     _bh_ir.AffineMapAttr.get(maps[-1])])
                consumer_attrs["operandSegmentSizes"] = _bh_ir.DenseI32ArrayAttr.get([len(old.operands), 1])
                consumer = create("linalg.generic", [*old.operands[:-1], producer.results[0], old.operands[-1]],
                                  [old.results[0].type], consumer_attrs, 1)
                consumer_block = _bh_ir.Block.create_at_start(
                    consumer.regions[0], [*[v.type for v in old_args[:-1]],
                                         target.results[0].type, old_args[-1].type])
                with _bh_ir.InsertionPoint(consumer_block):
                    values = {old_args[i]: consumer_block.arguments[i] for i in range(len(old_args) - 1)}
                    values[old_args[-1]] = consumer_block.arguments[-1]
                    values[target.results[0]] = consumer_block.arguments[-2]
                    # Backward liveness preserves every other use of a chain intermediate.
                    live = set(body[-1].operands)
                    retained = []
                    for scalar in reversed(body[:-1]):
                        if scalar is target:
                            continue
                        if scalar.results[0] in live:
                            retained.append(scalar)
                            live.update(scalar.operands)
                    for scalar in reversed(retained):
                        clone(scalar, values)
                    create("linalg.yield", [values.get(v, v) for v in body[-1].operands])
            old.results[0].replace_all_uses_with(consumer.results[0])
            old.operation.erase()
    module.operation.verify()
    return len(todo)

_BH_MARKER = "__merlin_broadcast_source_rsqrt_hoist__"
_BH_ORIG_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _BH_MARKER not in passes:
        return _BH_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(_BH_MARKER) != 1:
        raise ValueError("broadcast source math hoist marker must occur once")
    at = passes.index(_BH_MARKER)
    _BH_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:at]), 0, (), (), (), pre_generalize)
    print('OK broadcast_source_rsqrt_hoist', _hoist_broadcast_source_math(ctx, module))
    _BH_ORIG_RUN_STAGES(ctx, module, ','.join(passes[at + 1:]), erase, mid, late, post_openmp, ())

def _memref_typed(v):
    return str(v.type).startswith('memref')


def _op_key(o):
    # OpView and Operation both answer `.operation`, and the bindings hand back equal (hashable)
    # objects for the same op however it was reached -- so this is a stable dict key.
    return o.operation


def _dealloc_placement_violations(module):
    # (violations, n_allocs, n_sunk) for a post-deallocation memref module.
    order, block_of, index_of, owner_of_block = [], {}, {}, {}
    counter = [0]

    def visit(block, owner):
        bid = counter[0]
        counter[0] += 1
        owner_of_block[bid] = owner
        for i, op in enumerate(block.operations):
            k = _op_key(op)
            block_of[k], index_of[k] = bid, i
            order.append(op)
            for region in op.regions:
                for b in region.blocks:
                    visit(b, op)

    for op in module.body.operations:
        for region in op.regions:
            for b in region.blocks:
                visit(b, op)

    alias, roots, uses = {}, {}, {}
    for op in order:
        name = op.operation.name
        if name == 'memref.alloc' and len(op.results) == 1:
            rid = len(roots)
            roots[rid] = op
            alias[op.results[0]] = {rid}
            continue
        hit = set()
        for v in op.operands:
            hit |= alias.get(v, set())
        if not hit:
            continue
        for rid in hit:
            uses.setdefault(rid, []).append(op)
        for r in op.results:
            if _memref_typed(r):
                alias.setdefault(r, set()).update(hit)
        for region in op.regions:
            for b in region.blocks:
                for a in b.arguments:
                    if _memref_typed(a):
                        alias.setdefault(a, set()).update(hit)

    def ancestor_in_block(op, bid):
        o = op
        while True:
            k = _op_key(o)
            if k not in block_of:
                return None
            if block_of[k] == bid:
                return o
            parent = owner_of_block[block_of[k]]
            if parent is None:
                return None
            o = parent

    violations, sunk = [], 0
    for rid, alloc in roots.items():
        abid = block_of[_op_key(alloc)]
        here = [u for u in uses.get(rid, [])
                if u.operation.name == 'memref.dealloc' and block_of[_op_key(u)] == abid]
        if not here:
            continue                      # freed in a conditional/another block: not sunk
        sunk += 1
        d = min(here, key=lambda o: index_of[_op_key(o)])
        di = index_of[_op_key(d)]
        ty = str(alloc.results[0].type)
        for u in uses.get(rid, []):
            if _op_key(u) == _op_key(d):
                continue
            anc = ancestor_in_block(u, abid)
            if anc is None:
                violations.append(ty + ': use by ' + u.operation.name +
                                  ' is outside the block its free was placed in')
            elif index_of[_op_key(anc)] > di:
                violations.append(ty + ': freed at #' + str(di) + ' but used at #' +
                                  str(index_of[_op_key(anc)]) + ' by ' + u.operation.name)
    return violations, len(roots), sunk

_ORIG_RUN_STAGES = _run_stages
_SINK_MARK = 'optimize-allocation-liveness'


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    k = next((i for i, p in enumerate(passes) if _SINK_MARK in p), -1)
    if k < 0:
        return _ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp,
                                pre_generalize)
    head, tail = passes[:k + 1], passes[k + 1:]
    # `erase`/`mid` open their window after buffer-loop-hoisting, and `late` opens its own before
    # convert-scf-to-openmp; hand each to whichever half contains its anchor, or it would be
    # accepted here and silently never run. A `late` whose anchor is in neither half goes to the
    # tail, where the wrapped runner still applies it at the end rather than dropping it.
    hoist_head = any('buffer-loop-hoisting' in p for p in head)
    late_head = any('convert-scf-to-openmp' in p for p in head)
    post_head = late_head
    _ORIG_RUN_STAGES(ctx, module, ','.join(head),
                     erase if hoist_head else 0, mid if hoist_head else (),
                     late if late_head else (), post_openmp if post_head else (),
                     pre_generalize)
    bad, n_alloc, n_sunk = _dealloc_placement_violations(module)
    print('OK dealloc_placement', n_alloc, 'allocations', n_sunk, 'sunk', len(bad), 'violations')
    if bad:
        raise RuntimeError('dealloc placement moved a free before a use of the buffer '
                           '(' + str(len(bad)) + ' violations):\n  ' + '\n  '.join(bad[:20]))
    if tail:
        _ORIG_RUN_STAGES(ctx, module, ','.join(tail),
                         0 if hoist_head else erase, () if hoist_head else mid,
                         () if late_head else late, () if post_head else post_openmp, ())


def _residual_vector_ops(module):
    """Return vector-dialect op names still present at the LLVM translation edge."""
    found = []

    def visit(op):
        name = op.operation.name
        if name.startswith('vector.'):
            found.append(name)
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child)

    for top in module.body.operations:
        visit(top)
    return sorted(set(found))

src_path, out_path, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]
ctx = ir.Context()
with open(src_path) as f:
    module = ir.Module.parse(f.read(), ctx)

# fuse_transpose_b (default-off): fold `matmul(A, transpose(B))` into a transpose-b matmul BEFORE
# the pass manager runs, so the (still-named) linalg.matmul carries the transposed-B indexing map
# and the frozen RVV schedule tiles+vectorizes it while the scalar weight transpose disappears.
if _FUSE_TRANSPOSE_B:
    print("OK fuse_transpose_b", _fuse_transpose_b(module, ctx))
# fold_weight_transpose (default-off): the general form of the fold above -- a loop-invariant weight
# transpose sinks into the indexing_maps of EVERY linalg consumer, not just a linalg.matmul's B
# operand. The quantized datapath emits its contraction as a linalg.generic, so the matmul-only fold
# fires zero times on an int8 model; this one folds all 15 of small_llama's weight transposes.
if _FOLD_WEIGHT_TRANSPOSE:
    _fwt_n, _fwt_report = _fold_weight_transposes(module, ctx)
    for _fwt_kind, _fwt_detail in _fwt_report:
        print("OK fold_weight_transpose", _fwt_kind, _fwt_detail)
    print("OK fold_weight_transpose folded", _fwt_n)
# fold_broadcast_into_generic (default-off): replace a sole-use broadcast intermediate by a
# projected indexing map on an already-all-parallel generic. Reduction consumers are refused.
if _FOLD_BROADCAST:
    _bf_n, _bf_report = _fold_broadcasts(module, ctx)
    for _bf_kind, _bf_detail in _bf_report:
        print("OK fold_broadcast_into_generic", _bf_kind, _bf_detail)
    print("OK fold_broadcast_into_generic folded", _bf_n)
# concat_dps (default-off): a `tensor.concat` operand produced by a destination-passing op is
# retargeted to write STRAIGHT INTO the concatenated buffer, so bufferization has no data movement
# left to emit for it. Must run BEFORE the pass manager: after one-shot-bufferize the destination is
# no longer an operand, only a copy. See llvmlower/concat_dps.py.
if _CONCAT_DPS:
    _cd_n, _cd_report = _concat_dps(module, ctx)
    for _cd_kind, _cd_detail in _cd_report:
        print("OK concat_dps", _cd_kind, _cd_detail)
    print("OK concat_dps rewrote", _cd_n)
_run_stages(ctx, module, pipeline, _ERASE_SELF_COPY, _MID_STAGES, _LATE_STAGES,
            _POST_OPENMP_STAGES, _PRE_GENERALIZE_STAGES)
_vector_residue = _residual_vector_ops(module)
if _vector_residue:
    raise RuntimeError(
        'vector dialect survived the LLVM lowering edge: ' + ', '.join(_vector_residue))

with open(out_path, "w") as f:
    f.write(str(llvm.translate_module_to_llvmir(module.operation)))
print("OK")
