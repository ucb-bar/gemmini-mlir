"""Preserve an original NaN-select window recurrence in scalar tensor SSA.

This changes only recurrence materialization, before bufferization. It clones
all original comparisons, Boolean operations and selects, including attributes.
No maximum intrinsic, domain assumption, reassociation or pointer alias is added.
"""
from io import StringIO
from xdsl.dialects import arith, scf, tensor
from xdsl.dialects.builtin import TensorType, IndexType, NoneAttr, f32
from xdsl.ir import Block
from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr, AffineBinaryOpExpr, AffineBinaryOpKind
from xdsl.rewriter import Rewriter


def _linear(e, rank):
    if isinstance(e, AffineDimExpr):
        a = [0] * rank; a[e.position] = 1; return a, 0
    if isinstance(e, AffineConstantExpr): return [0] * rank, e.value
    if not isinstance(e, AffineBinaryOpExpr): raise ValueError('unsupported affine expression')
    a, x = _linear(e.lhs, rank); b, y = _linear(e.rhs, rank)
    if e.kind == AffineBinaryOpKind.Add: return [u+v for u,v in zip(a,b)], x+y
    if e.kind == AffineBinaryOpKind.Mul:
        if not any(a): return [x*v for v in b], x*y
        if not any(b): return [y*u for u in a], x*y
    raise ValueError('unsupported affine expression')


def _spec(op):
    if op.name != 'linalg.generic' or len(op.results) != 1 or len(op.outputs) != 1 or not op.inputs: return None
    if any(not k.startswith('prov.') for k in op.attributes): return None
    if set(op.properties) - {'indexing_maps', 'iterator_types', 'operandSegmentSizes'}: return None
    ancestor = op
    while ancestor is not None:
        if any('strictfp' in k or 'fenv' in k for k in ancestor.attributes): return None
        ancestor = ancestor.parent_op()
    ts = [v.type for v in op.operands]
    if any(not isinstance(t, TensorType) or not isinstance(t.encoding, NoneAttr) or t.element_type != f32 or any(not 0 < d < 2**63 for d in t.get_shape()) for t in ts): return None
    if op.results[0].type != ts[-1]: return None
    its = [a.data.value for a in op.iterator_types]; rank = len(its)
    if not 2 <= rank <= 8 or 'reduction' not in its: return None
    p = its.index('reduction')
    if p < 1 or its != ['parallel']*p + ['reduction']*(rank-p): return None
    maps = [a.data for a in op.indexing_maps]
    if len(maps) != len(ts) or any(a.num_dims != rank or a.num_symbols for a in maps): return None
    if maps[-1].results != tuple(AffineDimExpr(i) for i in range(p)): return None
    extents = [None]*rank
    for m,t in zip(maps,ts):
        if len(m.results) != len(t.get_shape()): return None
        for e,n in zip(m.results,t.get_shape()):
            if isinstance(e,AffineDimExpr):
                if extents[e.position] not in (None,n): return None
                extents[e.position]=n
    if any(n is None for n in extents): return None
    try:
        forms = [[_linear(e,rank) for e in m.results] for m in maps]
    except ValueError: return None
    for fs,t in zip(forms,ts):
        for (coeff,c),n in zip(fs,t.get_shape()):
            if c < 0 or any(a < 0 for a in coeff): return None
            low=c+sum(min(0,a*(e-1)) for a,e in zip(coeff,extents))
            high=c+sum(max(0,a*(e-1)) for a,e in zip(coeff,extents))
            if low < 0 or high >= n or max(abs(low),abs(high)) >= 2**63: return None
    body=list(op.body.block.ops);args=list(op.body.block.args)
    if len(args)!=len(ts) or [x.name for x in body] != ['arith.cmpf','arith.cmpf','arith.ori','arith.select','linalg.yield']: return None
    uno,gt,orr,sel,yieldop=body
    if uno.predicate.value.data!=14 or gt.predicate.value.data!=2: return None
    if uno.fastmath.data or gt.fastmath.data: return None
    source=uno.operands[0]
    if source not in args[:-1] or tuple(uno.operands)!=(source,source): return None
    seed=args[-1]
    if tuple(gt.operands)!=(source,seed) or tuple(orr.operands)!=(uno.results[0],gt.results[0]): return None
    if tuple(sel.operands)!=(orr.results[0],source,seed) or tuple(yieldop.operands)!=(sel.results[0],): return None
    if any(x.regions or x.successors or any(not k.startswith('prov.') for k in x.attributes) for x in body): return None
    expected=[{'predicate','fastmath'},{'predicate','fastmath'},set(),set(),set()]
    if any(set(x.properties)-allowed for x,allowed in zip(body,expected)):return None
    return extents,p,forms


def rewrite_module(module):
    """Clone source; rewrite only independently matched tensor window reductions."""
    current=module.clone(); receipts=[]
    for old in tuple(current.walk()):
        spec=_spec(old)
        if spec is None: continue
        extents,p,forms=spec; setup=[]
        def const(n,ops):
            c=arith.ConstantOp.from_int_and_width(n,IndexType());ops.append(c);return c.result
        zero=const(0,setup);one=const(1,setup);limits=[const(n,setup) for n in extents]
        def indices(fs,coords,ops):
            result=[]
            for coeff,c in fs:
                pieces=[]
                if c:pieces.append(const(c,ops))
                for a,x in zip(coeff,coords):
                    if not a:continue
                    if a!=1:
                        mul=arith.MuliOp(x,const(a,ops));ops.append(mul);x=mul.result
                    pieces.append(x)
                value=pieces[0] if pieces else zero
                for x in pieces[1:]:
                    add=arith.AddiOp(value,x);ops.append(add);value=add.result
                result.append(value)
            return result
        def reduce(depth,seed,coords,ops):
            if depth==len(extents):
                mapping={old.body.block.args[-1]:seed}
                for arg,operand,fs in zip(old.body.block.args[:-1],old.inputs,forms[:-1]):
                    extract=tensor.ExtractOp(operand,indices(fs,coords,ops),f32);ops.append(extract);mapping[arg]=extract.result
                for bodyop in list(old.body.block.ops)[:-1]:
                    expected_operands=tuple(mapping[v] for v in bodyop.operands)
                    clone=bodyop.clone(value_mapper=mapping)
                    if (clone.name != bodyop.name or clone.attributes != bodyop.attributes
                            or clone.properties != bodyop.properties
                            or tuple(clone.operands) != expected_operands
                            or tuple(v.type for v in clone.results) != tuple(v.type for v in bodyop.results)):
                        raise ValueError('scalar recurrence clone changed original operation')
                    ops.append(clone)
                return mapping[old.body.block.last_op.operands[0]]
            block=Block(arg_types=[IndexType(),f32]);loop=scf.ForOp(zero,limits[depth],one,[seed],block)
            children=[]
            value=reduce(depth+1,block.args[1],[*coords,block.args[0]],children)
            block.add_ops([*children,scf.YieldOp(value)]);ops.append(loop);return loop.results[0]
        def parallel(depth,dest,coords,ops):
            if depth==p:
                seed=tensor.ExtractOp(dest,coords,f32);ops.append(seed)
                value=reduce(depth,seed.result,coords,ops)
                insert=tensor.InsertOp(value,dest,coords);ops.append(insert);return insert.result
            block=Block(arg_types=[IndexType(),dest.type]);children=[]
            loop=scf.ForOp(zero,limits[depth],one,[dest],block)
            result=parallel(depth+1,block.args[1],[*coords,block.args[0]],children)
            block.add_ops([*children,scf.YieldOp(result)]);ops.append(loop);return loop.results[0]
        value=parallel(0,old.outputs[0],[],setup)
        setup[-1].attributes.update(old.attributes)
        receipts.append({'source_region':str(old.attributes.get('prov.region_id','')),'extents':extents,'parallel_rank':p,'reduction_order':'increasing lexicographic suffix axes','scalar_body':'cloned original cmpf uno/cmpf ogt/ori/select with original operand order','bounds':'all affine input/output indices checked over complete static rectangle'})
        Rewriter.replace_op(old,setup,[value])
    current.verify();return current,receipts


def rewrite_text(source):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.common.dense_float_ir import DenseFloatPrinter
    module,receipts=rewrite_module(parse_mlir_text(source))
    if not receipts:return source,receipts
    stream=StringIO();DenseFloatPrinter(stream=stream,print_generic_format=True).print_op(module)
    return stream.getvalue(),receipts


FEATURE = "source_window_reduction_ssa"

def ensure_registered():
    """Register the entry-point callback, which runs before bufferization."""
    from .impr_features import ImprFeature, known, register
    if FEATURE not in known():
        register(ImprFeature(
            name=FEATURE, action_class="PASS",
            description="Original scalar window recurrence in ordered tensor SSA before host bufferization; exact body and iteration order, default off."))
    return FEATURE

def selected_features(features):
    """Explicit feature selection or the candidate tree's declared default-off knob."""
    from merlin.compile.tuning_knobs import knob
    result = set(features)
    if knob("host_window_reduction", default="off", values=("off", "ordered_ssa")) == "ordered_ssa":
        result.add(FEATURE)
    return result

def prepare_owner_source(source, work, *, bound_effects=False):
    """Consume a source-exact transform in the original host lowering invocation.

    Scalar/mask observation proof objects cannot be silently rebound: retain their
    original source unchanged. Borrowed calls use their existing after_preprocessing
    transport separately and retain ordinary native requalification. No CPU/device placement, function ABI, tensor shapes or
    original scalar operations are selected or reconstructed by this callback.
    """
    import hashlib
    import json
    from pathlib import Path
    original_digest = hashlib.sha256(source.encode()).hexdigest()
    report = {"schema": "merlin.window-reduction-owner-callback.v1",
              "requested": True, "selected": True, "applied": False,
              "original_sha256": original_digest, "selected_sha256": original_digest,
              "rewrites": [], "scope": "original host tensor lowering only",
              "timing": "UNKNOWN"}
    selected = source
    if bound_effects:
        report["status"] = "refused_source_bound_scalar_observation_or_mask"
    else:
        from xdsl.utils.exceptions import ParseError
        try:
            selected, rewrites = rewrite_text(source)
        except ParseError:
            report["status"] = "refused_unsupported_source_syntax"
        else:
            report.update(status="applied" if rewrites else "unchanged_no_admitted_window_recurrence",
                          applied=bool(rewrites), rewrites=rewrites,
                          selected_sha256=hashlib.sha256(selected.encode()).hexdigest())
    work = Path(work)
    original = work / "window_reduction_original.mlir"
    receipt = work / "window_reduction_selection.json"
    selected_path = work / "window_reduction_selected.mlir"
    selected_path.write_text(selected)
    original.write_text(source)
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return selected, report, {"window_reduction_original": original,
                              "window_reduction_selected": selected_path,
                              "window_reduction_selection": receipt,
                              "window_reduction_implementation": Path(__file__)}


def bind_stock_recipe(work, sources, result, *, data_layout=None):
    """Bind the consumed source callback to the ordinary stock transport receipt."""
    import hashlib
    import json
    from pathlib import Path
    from .lowering_recipe import FILENAME
    path = Path(work) / FILENAME
    record = json.loads(path.read_text())
    selected = json.loads(sources["window_reduction_selection"].read_text())
    prepared = record["sources"]["prepared_mlir"]
    prepared_path = Path(prepared["path"])
    if prepared["sha256"] != hashlib.sha256(prepared_path.read_bytes()).hexdigest():
        raise ValueError("stock prepared source identity changed")
    expected_digest = selected["selected_sha256"]
    if prepared["sha256"] != expected_digest and data_layout:
        from merlin.frontends.linalg_mlir import parse_mlir_text
        from xdsl.dialects.builtin import StringAttr
        expected = parse_mlir_text(sources["window_reduction_selected"].read_text())
        expected.attributes["llvm.data_layout"] = StringAttr(data_layout)
        actual = parse_mlir_text(prepared_path.read_text())
        if expected.is_structurally_equivalent(actual):
            expected_digest = prepared["sha256"]
    if (record.get("status") != "returned"
            or record["returned_llvm_ir"]["sha256"] != hashlib.sha256(result.encode()).hexdigest()
            or prepared["sha256"] != expected_digest):
        raise ValueError("window reduction callback is not bound to the consumed stock source")
    for name, source in sources.items():
        source = Path(source)
        record["sources"][name] = {"path": str(source.resolve()),
                                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                    "bytes": source.stat().st_size}
    record["features"] = sorted(set(record["features"]) | {FEATURE})
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def transport_borrowed_calls(calls, before, after, report, sources):
    """Use the original selected-call transport without changing its permissions.

    The callback rewrites only pure tensor recurrence materialization. Existing
    original source pins, call roster, ABI/effect metadata and validators survive
    unchanged; the owner subsequently runs its normal native call qualification.
    """
    import json
    from .borrowed_buffer_calls import BorrowedBufferCalls
    if type(calls) is not BorrowedBufferCalls:
        raise ValueError("original selected BorrowedBufferCalls required")
    original = calls.record()
    rebound = calls.after_preprocessing(before, after)
    rebound.revalidate(after)
    current = rebound.record()
    def premises(record):
        return {key: value for key, value in record.items() if key != "current_input_sha256"}
    if premises(original) != premises(current):
        raise ValueError("window reduction changed original borrowed-call premises")
    report["borrowed_calls"] = {
        "transport": "BorrowedBufferCalls.after_preprocessing(before, after)",
        "original_premises_unchanged": True,
        "before_input_sha256": original["current_input_sha256"],
        "after_input_sha256": current["current_input_sha256"],
        "native_qualification": "ordinary owner prepare_native/verify_native remains required"}
    sources["window_reduction_selection"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return rebound
