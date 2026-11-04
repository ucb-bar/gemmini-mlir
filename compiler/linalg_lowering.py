"""Structural integer tensor-graph lowering to the public command-buffer ABI.

Only transformations established from the parsed operations are admitted. Metadata
names and provenance are carried through; provenance never selects arithmetic.
Unsupported scalar programs are left to the source host pipeline by the caller.
"""
from __future__ import annotations

import hashlib
from math import prod
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "compiler_library"))
from merlin.targetgen.contract import linalg_iface as LI
from xdsl.parser import Parser
from xdsl.dialects.builtin import TensorType, StringAttr, IntegerAttr, FloatAttr, ArrayAttr
from xdsl.ir.affine import AffineDimExpr


class UnsupportedLinalg(ValueError):
    pass


def _constant(value):
    owner = value.owner
    if getattr(owner, "name", "") in ("linalg.fill", "tensor.splat"):
        return _constant(owner.operands[0])
    if getattr(owner, "name", "") == "arith.constant":
        v = {**owner.attributes, **owner.properties}.get("value")
        if isinstance(v, (IntegerAttr, FloatAttr)):
            return v.value.data
    return None


def _expr(value, args):
    """Canonical scalar expression with actual SSA operands and constants."""
    if value in args:
        return ("arg", args.index(value))
    constant = _constant(value)
    if constant is not None:
        return ("const", constant)
    owner = value.owner
    if not hasattr(owner, "operands") or owner.regions:
        raise UnsupportedLinalg("unrecognized scalar capture or nested region")
    return (owner.name, *(_expr(v, args) for v in owner.operands))


def _commutes(expr, name, a, b):
    return expr in ((name, a, b), (name, b, a))


def _body(op):
    block = op.body.block
    from xdsl.traits import get_effects
    for nested in block.ops:
        effects = get_effects(nested)
        if nested.name in ("linalg.yield", "arith.extsi"):
            continue  # These scalar dialect operations are pure by definition.
        if effects is None or effects:
            raise UnsupportedLinalg("scalar body purity is not established")
    term = block.last_op
    if term is None or term.name != "linalg.yield" or len(term.operands) != 1:
        raise UnsupportedLinalg("requires a single scalar yield")
    return _expr(term.operands[0], list(block.args))


def _maps(op):
    maps = getattr(op, "indexing_maps", None)
    if maps is None:
        return None
    result = []
    for mapping in maps:
        if any(not isinstance(e, AffineDimExpr) for e in mapping.data.results):
            return None
        result.append(tuple(e.position for e in mapping.data.results))
    return result


def _shape(value):
    return list(value.type.get_shape())


def lower_linalg(text: str, inventory=None) -> dict:
    """Lower exact integer graphs, raising UnsupportedLinalg for other programs.

    The inventory is optional and is only used for provenance transport. Real xDSL
    objects establish operand identity, scalar semantics, initialization and maps.
    """
    inventory = inventory or LI.parse_linalg_mlir(text)
    module = Parser(LI.make_linalg_context(), text).parse_module()
    module.verify()
    definitions = [o for o in module.walk() if o.name == "func.func" and o.body.blocks]
    fn = next((o for o in definitions if LI._fn_name(o) == inventory["entry"]), None)
    if fn is None or len(fn.body.blocks) != 1:
        raise UnsupportedLinalg("requires a single-block tensor entry")
    original_direct = list(fn.body.block.ops)
    original_payload = [o for o in original_direct if o.name not in ("func.return", "linalg.yield") and not LI._initialization_only(o)]
    original_provenance = {o: inventory["ops"][i] for i, o in enumerate(original_payload)}
    original_index = {o: i for i, o in enumerate(original_direct)}
    preparation = None
    from merlin.compile.tuning_knobs import knob
    if knob("source_layout_reuse", default="off", values=("off", "cse"), root=Path(__file__).parent) == "cse":
        from source_preparation import prepare_source_layouts
        candidate = prepare_source_layouts(module, entry=inventory["entry"])
        returned = {original_index[v.owner] for v in original_direct[-1].operands if v.owner in original_index}
        # The whole-program ABI binds a returned original result to its own
        # single writer. Internal immutable layouts can share their storage;
        # returned identities retain the incumbent materialization boundary.
        if not any(a["source_ordinal"] in returned for a in candidate.aliases):
            preparation = candidate
            module = candidate.module
            fn = next(o for o in module.walk() if o.name == "func.func" and LI._fn_name(o) == inventory["entry"])
    block = fn.body.block
    ret = block.last_op
    if ret is None or ret.name != "func.return":
        raise UnsupportedLinalg("entry has no complete return")
    attrs = fn.attributes
    def names(key, fallback):
        value = attrs.get(key)
        return [x.data for x in value] if isinstance(value, ArrayAttr) else fallback
    input_names = names("merlin.logical_inputs", [a.name_hint or f"arg{i}" for i, a in enumerate(block.args)])
    output_names = names("merlin.logical_outputs", [f"Y{i}" for i in range(len(ret.operands))])
    roles = names("merlin.logical_input_roles", ["input"] * len(block.args))
    if len(input_names) != len(block.args) or len(output_names) != len(ret.operands) or len(roles) != len(block.args):
        raise UnsupportedLinalg("logical ABI arity differs from source")
    if len(set(input_names + output_names)) != len(input_names + output_names):
        raise UnsupportedLinalg("logical ABI names must be distinct")
    cb = {"abi_version": "0.1", "target": "gemmini", "tensors": {}, "commands": [],
          "params": {"source_level": "linalg-on-tensors", "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "source_entry": inventory["entry"], "source_ops": inventory["ops"]}}
    tensors, commands = cb["tensors"], cb["commands"]
    import source_epilogue as source_epilogue
    epilogues = source_epilogue.prepare_source_epilogues(module, entry=inventory["entry"]) if source_epilogue.enabled() else None
    epilogue_roots = {g["root"]: g for g in epilogues.groups} if epilogues else {}
    epilogue_members = {member: g["root"] for g in epilogue_roots.values() for member in g["members"]}
    # Compile-time epilogue parameters are evaluated by the exact proof and
    # become readout immediates/tables. Elide only closed private producers;
    # an observed parameter remains a materialized source value.
    epilogue_scaffolds = {}
    for root, group in epilogue_roots.items():
        closed = set(group["members"]) | {root}
        candidates = set()
        def collect(value):
            producer = value.owner
            if producer in closed or producer in candidates or producer.parent_block() is not block:
                return
            candidates.add(producer)
            for operand in producer.operands:
                collect(operand)
        for member in group["members"]:
            for operand in member.operands:
                collect(operand)
        changed = True
        while changed:
            changed = False
            for producer in tuple(candidates):
                if all(use.operation in closed for result in producer.results for use in result.uses):
                    closed.add(producer); candidates.remove(producer)
                    epilogue_scaffolds[producer] = root
                    changed = True
    if epilogues:
        cb["params"]["source_epilogue_refusals"] = list(epilogues.refusals)
    values, commit, view_roots, transposes = {}, {}, {}, {}
    final = {v: output_names[i] for i, v in enumerate(ret.operands)}
    if len(final) != len(ret.operands):
        raise UnsupportedLinalg("duplicate returned value requires an explicit copy")
    payload = [o for o in block.ops if o.name not in ("func.return", "linalg.yield") and not LI._initialization_only(o)]
    direct = list(block.ops)
    ordinals = preparation.source_ordinals if preparation is not None else tuple(range(len(direct)))
    direct_index = {o: i for o, i in zip(direct, ordinals, strict=True)}
    provenance = {o: original_provenance[original_direct[direct_index[o]]] for o in payload}
    removed = set(range(len(original_direct))) - set(ordinals)
    if preparation is not None:
        cb["params"]["source_layout_preparation"] = {"witnesses": list(preparation.witnesses), "aliases": list(preparation.aliases), "source_ordinals": list(ordinals)}
    attributed = set()
    task_sources = {}
    transports = []
    tensor_sources = {}
    cb["params"]["tensor_source_map"] = tensor_sources
    cb["params"]["source_direct_ops"] = [
        {"index": i, "kind": o.name,
         "attributes": {k: str(v) for k, v in {**o.attributes, **o.properties}.items()},
         "location": str(o.location)} for i, o in enumerate(original_direct)]
    def dependencies(owner):
        selected = []
        def visit(current):
            if current not in original_index or current in attributed:
                return
            for nested in current.walk():
                for operand in nested.operands:
                    producer = operand.owner
                    if (producer not in original_provenance
                            or original_index.get(producer) in removed):
                        visit(producer)
            attributed.add(current)
            selected.append(original_index[current])
        visit(original_direct[direct_index[owner]])
        return sorted(selected)
    def declare(value, name, role):
        if not isinstance(value.type, TensorType) or any(d <= 0 for d in _shape(value)):
            raise UnsupportedLinalg("requires positive static ranked tensors")
        tensors[name] = {"shape": _shape(value), "dtype": str(value.type.element_type), "role": role}
        values[value] = name
        if value in block.args:
            tensor_sources[name] = {"arg_index": list(block.args).index(value)}
        elif value.owner in direct_index:
            tensor_sources[name] = {"op_index": direct_index[value.owner], "result_index": list(value.owner.results).index(value)}
        return name
    for value, name, role in zip(block.args, input_names, roles, strict=True):
        declare(value, name, role)
    def output(value):
        if value in values:
            return values[value]
        name = final.get(value, f"tmp{len(values)}")
        if value not in final:
            while name in tensors or name in output_names:
                name += "_"
        return declare(value, name, "output" if value in final else "intermediate")
    def source(value):
        if value not in values:
            raise UnsupportedLinalg(f"unlowered tensor operand from {getattr(value.owner, 'name', 'argument')}")
        return values[value]
    def emit(opcode, operands, attributes=None, owner=None):
        attributes = dict(attributes or {})
        if owner in provenance:
            rec = provenance[owner]
            attributes["source_provenance"] = rec["source_provenance"]
            attributes["source_payload_index"] = rec["id"]
            attributes.update({f"prov.{k}": v for k, v in rec.get("prov", {}).items()})
            task = rec["id"]
            if task not in task_sources:
                task_sources[task] = dependencies(owner)
            attributes["source_task_index"] = task
            attributes["source_op_indices"] = task_sources[task]
            attributes["source_primary_op_index"] = direct_index[owner]
        cmd = {"opcode": opcode, "operands": operands, "attributes": attributes}
        commands.append(cmd)
        return cmd
    def strides(shape):
        return [prod(shape[i + 1:]) for i in range(len(shape))]
    def copy_recipe(src, dst, shape=None, ss=None, ds=None, so=0, do=0):
        return {"src": src, "dst": dst, "shape": tensors[dst]["shape"] if shape is None else shape,
                "source_strides": strides(tensors[src]["shape"]) if ss is None else ss,
                "destination_strides": strides(tensors[dst]["shape"]) if ds is None else ds,
                "source_offset": so, "destination_offset": do}
    def transport(op, copies=None, fill=None):
        task = provenance[op]["id"]
        task_sources[task] = dependencies(op)
        rec = {"before_command_index": len(commands), "source_task_index": task,
               "source_op_indices": task_sources[task], "source_primary_op_index": direct_index[op],
               "copies": copies or []}
        if fill is not None:
            rec["fill"] = fill
        transports.append(rec)
    def integers(op, key):
        attr = {**op.attributes, **op.properties}.get(key)
        return list(attr.get_values()) if hasattr(attr, "get_values") else []
    for op in payload:
        if op in epilogue_scaffolds:
            root = epilogue_scaffolds[op]
            task = provenance[root]["id"]
            if task not in task_sources:
                task_sources[task] = dependencies(root)
            task_sources[task].extend(dependencies(op))
            continue
        if op in epilogue_members:
            root = epilogue_members[op]
            task_sources[provenance[root]["id"]].extend(dependencies(op))
            continue
        if len(op.results) != 1:
            raise UnsupportedLinalg(f"{op.name} requires a multi-result route")
        result = op.results[0]
        ins, outs = LI._ins_outs(op)
        dtype = str(result.type.element_type) if isinstance(result.type, TensorType) else ""
        if op.name == "linalg.transpose" and len(ins) == 1 and dtype in ("i8", "i16", "i32", "i64"):
            permutation = {**op.attributes, **op.properties}.get("permutation")
            perm = list(permutation.get_values()) if hasattr(permutation, "get_values") else []
            uses = list(result.uses)
            consumer_inputs = list(getattr(uses[0].operation, "inputs", ())) if len(uses) == 1 else []
            if (dtype == "i8" and perm == [1, 0] and len(_shape(ins[0])) == 2 and result not in final
                    and len(uses) == 1 and uses[0].operation.name in ("linalg.generic", "linalg.matmul")
                    and len(consumer_inputs) >= 2 and consumer_inputs[1] is result
                    and uses[0].operation not in epilogue_roots):
                # A transposed RHS is exactly the public QK contraction. Keep
                # the original tensor and fold this layout operation into its
                # consumer; no extra buffer or host arithmetic is introduced.
                source(ins[0])
                transposes[result] = ins[0]
                continue
            src, dst = source(ins[0]), output(result)
            if sorted(perm) != list(range(len(perm))):
                raise UnsupportedLinalg("transpose has no static permutation")
            transport(op, [copy_recipe(src, dst, _shape(result),
                          [strides(_shape(ins[0]))[axis] for axis in perm])])
            continue
        if op.name in ("tensor.collapse_shape", "tensor.expand_shape") and len(ins) >= 1:
            if dtype != str(ins[0].type.element_type) or prod(_shape(result)) != prod(_shape(ins[0])):
                raise UnsupportedLinalg("row-major view changes element count or dtype")
            src, dst = source(ins[0]), output(result)
            transport(op, [copy_recipe(src, dst, [prod(_shape(result))], [1], [1])])
            continue
        if op.name in ("tensor.splat", "linalg.fill"):
            value = _constant(op.operands[0])
            if value is None or not dtype.startswith("i"):
                raise UnsupportedLinalg("transport fill requires a literal integer scalar")
            transport(op, fill={"dst": output(result), "value": int(value)})
            continue
        if op.name in ("tensor.extract_slice", "tensor.insert_slice"):
            offsets, sizes, steps = (integers(op, key) for key in ("static_offsets", "static_sizes", "static_strides"))
            base = op.operands[0] if op.name == "tensor.extract_slice" else op.operands[1]
            if (len(offsets) != len(_shape(base)) or any(x < 0 for x in offsets)
                    or any(x <= 0 for x in sizes + steps)):
                raise UnsupportedLinalg("slice requires positive static sizes/strides and static offsets")
            if any(o + (n - 1) * k >= d for o, n, k, d in zip(offsets, sizes, steps, _shape(base))):
                raise UnsupportedLinalg("slice exceeds source tensor")
            dst = output(result)
            bs = strides(_shape(base))
            offset = sum(o * st for o, st in zip(offsets, bs))
            slice_strides = [k * st for k, st in zip(steps, bs)]
            small = result if op.name == "tensor.extract_slice" else op.operands[0]
            # Rank reduction drops only unit-size axes, with exact remaining shape.
            target = _shape(small)
            keep, cursor = [], 0
            for axis, size in enumerate(sizes):
                if cursor < len(target) and size == target[cursor]:
                    keep.append(axis); cursor += 1
                elif size != 1:
                    raise UnsupportedLinalg("slice rank reduction is ambiguous")
            if cursor != len(target):
                raise UnsupportedLinalg("slice result shape differs from static sizes")
            reduced = [slice_strides[i] for i in keep]
            if op.name == "tensor.extract_slice":
                transport(op, [copy_recipe(source(base), dst, target, reduced, strides(target), offset)])
            else:
                transport(op, [copy_recipe(source(base), dst),
                               copy_recipe(source(small), dst, target, strides(target), reduced, 0, offset)])
            continue
        if op.name == "tensor.concat":
            axis_attr = {**op.attributes, **op.properties}.get("dim")
            axis = axis_attr.value.data
            dst = output(result)
            pieces, offset = [], 0
            ds = strides(_shape(result))
            for operand in op.operands:
                shape = _shape(operand)
                pieces.append(copy_recipe(source(operand), dst, shape, strides(shape), ds, 0, offset * ds[axis]))
                offset += shape[axis]
            if offset != _shape(result)[axis]:
                raise UnsupportedLinalg("concat extent differs from its inputs")
            transport(op, pieces)
            continue
        # Establish signed i8 multiplication plus i32 accumulation, including
        # arbitrary iterator numbering and arbitrary nonempty batch prefixes.
        contraction = op.name in ("linalg.matmul", "linalg.batch_matmul", "linalg.matmul_transpose_b")
        maps = _maps(op)
        if op.name == "linalg.generic" and len(ins) == 2 and len(outs) == 1:
            a, b, c = ("arg", 0), ("arg", 1), ("arg", 2)
            product = ("arith.muli", ("arith.extsi", a), ("arith.extsi", b))
            expression = _body(op)
            contraction = _commutes(expression, "arith.addi", c, product) or _commutes(expression, "arith.addi", c, ("arith.muli", product[2], product[1]))
        if contraction:
            if len(ins) != 2 or len(outs) != 1:
                raise UnsupportedLinalg("contraction requires two inputs and one initializer")
            init = outs[0]
            add_initial = _constant(init) != 0
            if add_initial and (init not in values or _shape(init) != _shape(result) or str(init.type.element_type) != "i32"):
                raise UnsupportedLinalg("contraction initializer needs a materialized same-shape i32 tensor")
            if [str(v.type.element_type) for v in ins] != ["i8", "i8"] or dtype != "i32":
                raise UnsupportedLinalg("device contraction requires signed i8 x i8 -> i32")
            a, b, shape = _shape(ins[0]), _shape(ins[1]), _shape(result)
            rank = len(shape)
            if rank < 2 or len(a) != rank or len(b) != rank or a[:-2] != b[:-2] or a[:-2] != shape[:-2]:
                raise UnsupportedLinalg("contraction batch axes differ or broadcast")
            transpose = op.name == "linalg.matmul_transpose_b"
            if maps is not None:
                if len(maps) != 3 or len(maps[2]) != rank:
                    raise UnsupportedLinalg("contraction maps lack a complete output")
                batch, m, n = maps[2][:-2], maps[2][-2], maps[2][-1]
                reduction = [d for d in maps[0] if d not in maps[2]]
                if len(reduction) != 1:
                    raise UnsupportedLinalg("contraction needs one reduction axis")
                k = reduction[0]
                if op.name == "linalg.generic":
                    its = LI._iterator_types(op, ins)
                    if (LI._reduction_dims(op) != [k] or any(it != ("reduction" if i == k else "parallel") for i, it in enumerate(its))):
                        raise UnsupportedLinalg("contraction iterator types disagree with indexing maps")
                if maps[0] != (*batch, m, k):
                    raise UnsupportedLinalg("lhs contraction layout requires a layout route")
                transpose = maps[1] == (*batch, n, k)
                if maps[1] not in ((*batch, k, n), (*batch, n, k)):
                    raise UnsupportedLinalg("rhs contraction layout requires a layout route")
            expected = a[:-2] + [a[-2], b[-2] if transpose else b[-1]]
            if shape != expected or a[-1] != (b[-1] if transpose else b[-2]):
                raise UnsupportedLinalg("contraction shapes disagree with indexing maps")
            dst = output(result)
            final_dst = dst
            if add_initial:
                dst = f"product{len(commands)}"
                tensors[dst] = {"shape": shape, "dtype": dtype, "role": "intermediate"}
                cb["params"].setdefault("compiler_temporaries", []).append(
                    {"tensor": dst, "source_op_index": direct_index[op], "source_result_index": 0,
                     "purpose": "zero-based contraction product before adding source initializer"})
            rhs = transposes.get(ins[1], ins[1])
            storage_transpose = transpose != (ins[1] in transposes)
            if ins[1] in transposes:
                # Attribute the deferred layout producer to the same task as
                # its contraction, including its destination initializer.
                task = provenance[op]["id"]
                task_sources.setdefault(task, dependencies(op))
                task_sources[task].extend(dependencies(ins[1].owner))
            if rank > 2:
                if transpose:
                    raise UnsupportedLinalg("batched transpose needs an explicit layout route")
                emit("BATCHED_MATMUL", {"a": source(ins[0]), "w": source(rhs), "dst": dst}, {"output_dtype": dtype}, op)
            elif storage_transpose:
                emit("ATTENTION_QK", {"q": source(ins[0]), "k": source(rhs), "dst": dst}, {"output_dtype": dtype, "epilogue": []}, op)
            else:
                handle, acc = f"resident{len(commands)}", f"acc{len(commands)}"
                emit("RES_PACK", {"src": source(rhs), "dst": handle}, {"layout": "packed_rhs"}, op)
                emit("MATMUL_RESIDENT", {"lhs": source(ins[0]), "rhs": handle, "dst": acc}, owner=op)
                commit[result] = emit("COMMIT", {"src": acc, "dst": dst}, {"output_dtype": dtype, "epilogue": []}, op)
                emit("EVICT", {"handle": handle}, owner=op)
            if add_initial:
                emit("VECTOR_MAP", {"lhs": dst, "rhs": source(init), "dst": final_dst},
                     {"combine": "add", "activation": [], "output_dtype": "i32"}, op)
                commit.pop(result, None)  # The product readout alone is not the complete source value.
            if op in epilogue_roots:
                if result not in commit:
                    raise UnsupportedLinalg("proved source epilogue requires a dense COMMIT adapter")
                group = epilogue_roots[op]
                old = values.pop(result)
                tensors.pop(old); tensor_sources.pop(old)
                cmd = commit.pop(result)
                cmd["operands"]["dst"] = output(group["result"])
                cmd["attributes"].update(group["readout"])
                cb["params"].setdefault("source_epilogue_fusions", []).append(
                    {"source_op_indices": [direct_index[o] for o in (op, *group["members"])],
                     "cut": group["cut"], "proof": group["proof"], "scale_proof": group["scale_proof"]})
            continue
        if op.name == "linalg.generic" and len(ins) == 2 and len(outs) == 1:
            expr = _body(op)
            if (_commutes(expr, "arith.addi", ("arg", 0), ("arg", 1))
                    and dtype == "i32" and all(str(v.type.element_type) == "i32" for v in ins)
                    and maps == [(0, 1), (1,), (0, 1)]
                    and _shape(ins[0]) == _shape(result) and _shape(ins[1]) == [_shape(result)[1]]):
                bias = source(ins[1])
                if tensors[bias]["role"] == "input":
                    tensors[bias]["role"] = "bias"
                if source_epilogue.enabled() and ins[0] in commit:
                    cmd = commit[ins[0]]
                    old = values[ins[0]]
                    accepted, reason = source_epilogue.fuse_integer_bias_readout(
                        cmd, bias, final.get(result, f"tmp{len(values)}"),
                        private=len(tuple(ins[0].uses)) == 1 and ins[0] not in final,
                        bias_dtype=str(ins[1].type.element_type), output_dtype=dtype)
                    if accepted:
                        values.pop(ins[0]); tensors.pop(old); tensor_sources.pop(old)
                        cmd["operands"]["dst"] = output(result)
                        task_sources[cmd["attributes"]["source_task_index"]].extend(dependencies(op))
                        commit.pop(ins[0]); commit[result] = cmd
                        cb["params"].setdefault("source_epilogue_fusions", []).append(
                            {"source_op_indices": list(task_sources[cmd["attributes"]["source_task_index"]]),
                             "cut": "integer_bias", "proof": reason})
                        continue
                    cb["params"].setdefault("source_epilogue_refusals", []).append(
                        {"root_ordinal": direct_index[op], "reason": reason})
                emit("BIAS_ADD", {"src": source(ins[0]), "bias": bias, "dst": output(result)}, {"output_dtype": dtype, "epilogue": ["bias_add"]}, op)
                continue
        if op.name in ("linalg.generic", "linalg.map") and len(ins) == 1:
            expr = _body(op)
            arg = ("arg", 0)
            if maps is not None and any(m != tuple(range(len(_shape(result)))) for m in maps):
                raise UnsupportedLinalg("elementwise map requires identity indexing")
            # Saturating i32->i8, optionally rectified. Fuse only through a sole
            # producer use, preserving all graph values observable elsewhere.
            lo = ("arith.maxsi", arg, ("const", -128))
            saturated = ("arith.trunci", ("arith.minsi", lo, ("const", 127)))
            relu = ("arith.maxsi", arg, ("const", 0))
            if expr == saturated and dtype == "i8" and ins[0] in commit and len(tuple(ins[0].uses)) == 1:
                cmd = commit[ins[0]]
                old = values.pop(ins[0])
                tensors.pop(old)
                tensor_sources.pop(old)
                cmd["operands"]["dst"] = output(result)
                cmd["attributes"]["output_dtype"] = "i8"
                cmd["attributes"]["epilogue"].append("acc_scale")
                cmd["attributes"]["acc_scale"] = 1.0
                task_sources[cmd["attributes"]["source_task_index"]].extend(dependencies(op))
                commit[result] = cmd
                continue
            if expr == relu and ins[0] in commit and len(tuple(ins[0].uses)) == 1:
                cmd = commit[ins[0]]
                old = values.pop(ins[0]); tensors.pop(old)
                tensor_sources.pop(old)
                cmd["operands"]["dst"] = output(result)
                cmd["attributes"]["epilogue"].append("relu")
                task_sources[cmd["attributes"]["source_task_index"]].extend(dependencies(op))
                commit[result] = cmd
                continue
            if expr in (arg, ("arith.extsi", arg)) and dtype in ("i8", "i32") and _shape(ins[0]) == _shape(result):
                emit("MOVEMENT", {"src": source(ins[0]), "dst": output(result)}, {"output_dtype": dtype, "semantic": "mvin_mvout"}, op)
                continue
        raise UnsupportedLinalg(f"no exact device command route for {op.name} ({LI._prov(op).get('op', '')})")
    for value, name in zip(ret.operands, output_names, strict=True):
        if values.get(value) != name:
            if value not in values:
                raise UnsupportedLinalg("returned value has no producer")
            emit("MOVEMENT", {"src": source(value), "dst": declare(value, name, "output")}, {"output_dtype": str(value.type.element_type), "semantic": "mvin_mvout"})
    cb["params"]["source_input_names"] = input_names
    cb["params"]["source_output_names"] = output_names
    cb["params"]["source_return_op_index"] = direct_index[ret]
    tasks = {}
    for index in range(len(commands) + 1):
        for rec in transports:
            if rec["before_command_index"] == index:
                old = rec["source_task_index"]
                tasks.setdefault(old, len(tasks))
        if index < len(commands):
            old = commands[index]["attributes"].get("source_task_index")
            if old is not None:
                tasks.setdefault(old, len(tasks))
    definitions = []
    for old, new in tasks.items():
        indices = [i for i, cmd in enumerate(commands) if cmd["attributes"].get("source_task_index") == old]
        definition = {"task_index": new, "kind": "source_tensor_transport" if any(r["source_task_index"] == old for r in transports) else "source_tensor_computation",
                      "source_op_indices": sorted(set(task_sources[old])), "command_indices": indices}
        if preparation is not None:
            reads, writes = set(), set()
            owned = set(definition["source_op_indices"])
            for op in direct:
                if direct_index[op] not in owned:
                    continue
                for nested in op.walk():
                    for value in nested.operands:
                        if value in values and (value.owner not in direct_index or direct_index[value.owner] not in owned):
                            reads.add(values[value])
                for value in op.results:
                    if value in values:
                        writes.add(values[value])
            for row in cb["params"].get("compiler_temporaries", []):
                if row["source_op_index"] in owned:
                    writes.add(row["tensor"])
            definition.update(reads=sorted(reads), writes=sorted(writes))
        definitions.append(definition)
    for command in commands:
        attributes = command.get("attributes", {})
        if "source_task_index" in attributes:
            attributes["source_task_index"] = tasks[attributes["source_task_index"]]
    for rec in transports:
        rec["source_task_index"] = tasks[rec["source_task_index"]]
    if transports or preparation is not None:
        cb["params"]["source_transport_tasks"] = transports
        cb["params"]["source_task_definitions"] = definitions
    # The final-view ABI deliberately accepts exactly two semantic fields.
    # Keep compilation attribution outside that closed instruction vocabulary.
    for index, command in enumerate(commands):
        if command["opcode"] == "ROW_MAJOR_VIEW":
            attributes = command["attributes"]
            metadata = {k: v for k, v in attributes.items() if k not in ("source_shape", "shape")}
            cb["params"].setdefault("command_source_attribution", {})[str(index)] = metadata
            command["attributes"] = {k: attributes[k] for k in ("source_shape", "shape")}
    return cb


def emit_source_transport(builder, cb, record, ptrs):
    """Materialize a proven static tensor map using byte-preserving Gemmini DMA.

    Recipes use element strides; LLVM performs address arithmetic only. Scratchpad
    bytes preserve integer bit patterns of every width without readout scaling.
    """
    def width(name):
        dtype = cb['tensors'][name]['dtype']
        if not dtype.startswith('i') or int(dtype[1:]) not in (8, 16, 32, 64):
            raise UnsupportedLinalg('source transport requires byte-addressable integer tensors')
        return int(dtype[1:]) // 8
    builder.config_ex()
    builder.config_ld(16)
    builder.config_st(16)
    def dma(src, dst, n):
        # Full rectangles plus one short last row. Each transfer has a matching
        # fence before scratchpad reuse, including between independent recipes.
        bulk = n // 256 * 256
        if bulk:
            with builder.counted_loop(0, bulk, 256) as i:
                builder.mvin(builder.ptr_offset(src, i), 0, 16, 16)
                builder.fence()
                builder.mvout(builder.ptr_offset(dst, i), 0, 16, 16)
                builder.fence()
        remain = n - bulk
        rows, cols = divmod(remain, 16)
        if rows:
            builder.mvin(builder.ptr_offset(src, bulk), 0, 16, rows)
            builder.fence()
            builder.mvout(builder.ptr_offset(dst, bulk), 0, 16, rows)
            builder.fence()
        if cols:
            pos = bulk + rows * 16
            builder.mvin(builder.ptr_offset(src, pos), 0, cols, 1)
            builder.fence()
            builder.mvout(builder.ptr_offset(dst, pos), 0, cols, 1)
            builder.fence()
    if 'fill' in record:
        fill = record['fill']
        dst, value = fill['dst'], fill['value']
        w = width(dst)
        # A static literal tile is source constant realization, never runtime
        # tensor arithmetic. Replicate its exact little-endian bit pattern.
        raw = (int(value) & ((1 << (w * 8)) - 1)).to_bytes(w, 'little')
        tile = raw * (256 // w)
        words = [int.from_bytes(tile[i:i + 4], 'little') for i in range(0, 256, 4)]
        seed = builder.constant_i32_table(words)
        size = prod(cb['tensors'][dst]['shape']) * w
        bulk = size // 256 * 256
        if bulk:
            with builder.counted_loop(0, bulk, 256) as i:
                dma(seed, builder.ptr_offset(ptrs[dst], i), 256)
        if size - bulk:
            dma(seed, builder.ptr_offset(ptrs[dst], bulk), size - bulk)
    for rec in record.get('copies', []):
        src, dst = rec['src'], rec['dst']
        w = width(src)
        if width(dst) != w:
            raise UnsupportedLinalg('transport cannot change element width')
        shape = list(rec['shape'])
        ss, ds = list(rec['source_strides']), list(rec['destination_strides'])
        run = 1
        while shape and ss[-1] == run and ds[-1] == run:
            run *= shape.pop()
            ss.pop(); ds.pop()
        def move(index):
            so, do = rec['source_offset'], rec['destination_offset']
            for axis, extent in enumerate(shape):
                tail = prod(shape[axis + 1:])
                coord = builder.urem(builder.udiv(index, tail), extent)
                so = builder.add(so, builder.mul(coord, ss[axis]))
                do = builder.add(do, builder.mul(coord, ds[axis]))
            dma(builder.ptr_offset(ptrs[src], builder.mul(so, w)),
                builder.ptr_offset(ptrs[dst], builder.mul(do, w)), run * w)
        if shape:
            with builder.counted_loop(0, prod(shape)) as i:
                move(i)
        else:
            move(0)
    builder.fence()
