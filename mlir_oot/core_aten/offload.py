"""Host ABI glue: replace bound Core ATen regions by calls and adapt MLIR memrefs to the kernels.

``rewrite(source, manifest)`` replaces every catalog binding's region with
``func.call @ca_<symbol>(operands..., out_init) -> result`` (destination passing, the output
initialised by ``tensor.empty``) and declares each callee private with
``llvm.emit_c_interface``; ops the region consumed are erased once they are dead.  After any
standard lowering (one-shot bufferization with function boundaries, memref to LLVM) the
call is ``void _mlir_ciface_ca_<symbol>(result*, arg*..., out*)`` over memref descriptors.

``shim_source(manifest)`` is the C translation unit defining those entry points: it checks
that every descriptor is dense row-major, passes the aligned data pointers in the kernel's
argument order, supplies the kernel's private scratch buffers, and returns the output
descriptor.  This is the same lowered-memref convention Merlin's contraction shims use.
"""

from __future__ import annotations

import json

from xdsl.dialects import func, tensor
from xdsl.dialects.builtin import FunctionType, UnitAttr

from ..frontend.parse import parse_module


def _value(src, ordinals_inv, func_args):
    if "func_arg" in src:
        return func_args[src["func_arg"]]
    return ordinals_inv[src["op_ordinal"]].results[src["result"]]


def rewrite(source: str, manifest: dict) -> str:
    module = parse_module(source)
    ops_ = list(module.walk())
    fn = next(op for op in ops_ if op.name == "func.func")
    func_args = list(fn.body.blocks[0].args)
    kernels = {k["symbol"]: k for k in manifest["kernels"]}
    declared = {}
    doomed = []
    replacements = {}
    for b in manifest["bindings"]:
        k = kernels[b["symbol"]]
        if "result_slices" in b:
            callee, ftype = _rewrite_flat(b, k, ops_, func_args, replacements)
            doomed += [ops_[i] for i in b["replaced_ordinals"]]
            if callee not in declared:
                decl = func.FuncOp.external(callee, ftype.inputs.data, ftype.outputs.data)
                decl.attributes["llvm.emit_c_interface"] = UnitAttr()
                declared[callee] = decl
            continue
        (out_name, out_src), = b["result"].items()
        out_val = _value(out_src, ops_, func_args)
        root = ops_[b["root_ordinal"]]
        ins = [_value(b["operands"][a["name"]], ops_, func_args) for a in k["args"] if a["role"] == "input"]
        ins = [replacements.get(value, value) for value in ins]
        out_ty = out_val.type
        empty = tensor.EmptyOp([], out_ty)
        callee = "ca_" + b["symbol"]
        call = func.CallOp(callee, [*ins, empty.results[0]], [out_ty])
        root.parent_block().insert_ops_before([empty, call], root)
        out_val.replace_all_uses_with(call.results[0])
        replacements[out_val] = call.results[0]
        doomed += [ops_[i] for i in b["replaced_ordinals"]]
        if callee not in declared:
            ftype = FunctionType.from_lists([v.type for v in ins] + [out_ty], [out_ty])
            decl = func.FuncOp.external(callee, ftype.inputs.data, ftype.outputs.data)
            decl.attributes["llvm.emit_c_interface"] = UnitAttr()
            declared[callee] = decl
    # erase replaced ops in reverse program order once they have no remaining users
    changed = True
    alive = [op for op in doomed if op.parent is not None]
    while changed:
        changed = False
        for op in sorted(set(alive), key=lambda o: -ops_.index(o)):
            if op.parent is not None and all(not list(r.uses) for r in op.results):
                op.detach()
                op.erase()
                alive.remove(op)
                changed = True
    # Shared effect-aware cleanup; an unused tensor result does not make a
    # runtime assertion or an unknown operation removable.
    from .liveness import pure_tensor_dce
    pure_tensor_dce(fn)
    for decl in declared.values():
        module.body.blocks[0].add_op(decl)
    fn.attributes["llvm.emit_c_interface"] = UnitAttr()
    module.verify()
    from .storage import module_text
    return module_text(module)


def _rewrite_flat(b, k, ops_, func_args, replacements):
    """One call writes a flat result; each source result is a dense row-major element range."""
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, TensorType, i64

    out, = [a for a in k["args"] if a["role"] == "output"]
    root = ops_[b["root_ordinal"]]
    ins = [_value(b["operands"][a["name"]], ops_, func_args) for a in k["args"] if a["role"] == "input"]
    ins = [replacements.get(value, value) for value in ins]
    values = [_value(s["source"], ops_, func_args) for s in b["result_slices"]]
    element = values[0].type.get_element_type()
    if any(v.type.get_element_type() != element or str(v.type) != s["type"]
           for v, s in zip(values, b["result_slices"])):
        raise ValueError("flat result slices differ from their source result types")
    flat = TensorType(element, out["shape"])
    empty = tensor.EmptyOp([], flat)
    callee = "ca_" + b["symbol"]
    call = func.CallOp(callee, [*ins, empty.results[0]], [flat])
    created = [empty, call]
    for value, piece in zip(values, b["result_slices"]):
        extent = 1
        for d in piece["shape"]:
            extent *= d
        part = tensor.ExtractSliceOp.from_static_parameters(call.results[0], [piece["offset"]], [extent])
        reassociation = ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in range(len(piece["shape"]))])])
        view = tensor.ExpandShapeOp(part.results[0], [], reassociation, piece["shape"], value.type)
        created += [part, view]
        value.replace_all_uses_with(view.results[0])
        replacements[value] = view.results[0]
    root.parent_block().insert_ops_before(created, root)
    return callee, FunctionType.from_lists([v.type for v in ins] + [flat], [flat])


def shim_source(manifest: dict) -> str:
    from .storage import itemsize
    ranks = set()
    lines = []
    body = []
    for k in manifest["kernels"]:
        args = k["args"]
        ins = [a for a in args if a["role"] == "input"]
        outs = [a for a in args if a["role"] == "output"]
        if len(outs) != 1:
            raise ValueError("one output per kernel")
        out = outs[0]
        for a in ins + outs:
            ranks.add(len(a["shape"]))
        sym = k["symbol"]
        params = ", ".join([f"ca_memref_{len(out['shape'])} *result"] +
                           [f"ca_memref_{len(a['shape'])} *{a['name']}" for a in ins] +
                           [f"ca_memref_{len(out['shape'])} *{out['name']}"])
        body.append(f"extern void {sym}({', '.join(['void *'] * len(args))});")
        for a in args:
            if a["role"] == "scratch":
                n = 1
                for d in a["shape"]:
                    n *= d
                body.append(f"static int8_t {sym}__{a['name']}[{n * itemsize(a['dtype'])}] "
                            f"__attribute__((aligned(64)));")
        body.append(f"void _mlir_ciface_ca_{sym}({params}) {{")
        checks = []
        for a in ins + outs:
            r = len(a["shape"])
            checks.append(f"ca_dense_{r}({a['name']}, (const int64_t[]){{{', '.join(map(str, a['shape'])) or '1'}}})")
        body.append(f"  if (!({' && '.join(checks)})) ca_abort(\"{sym}: non-dense or mis-shaped operand\");")
        call = []
        for a in args:
            if a["role"] == "scratch":
                call.append(f"{sym}__{a['name']}")
            else:
                call.append(f"(void *)((uint8_t *){a['name']}->aligned + {a['name']}->offset * {itemsize(a['dtype'])})")
        body.append(f"  {sym}({', '.join(call)});")
        body.append(f"  *result = *{out['name']};")
        body.append("}")
        body.append("")
    lines.append("/* Generated by gemmini-mlir mlir_oot.core_aten.offload: MLIR C-interface entry points. */")
    lines.append("#include <stdint.h>")
    lines.append("#include <stdio.h>")
    lines.append("void ca_abort(const char *why);")
    for r in sorted(ranks):
        if r == 0:
            lines.append("typedef struct { void *allocated; void *aligned; int64_t offset; } ca_memref_0;")
            lines.append("static int ca_dense_0(const ca_memref_0 *m, const int64_t *s) { (void)m; (void)s; return 1; }")
            continue
        lines.append(f"typedef struct {{ void *allocated; void *aligned; int64_t offset; int64_t sizes[{r}]; "
                     f"int64_t strides[{r}]; }} ca_memref_{r};")
        lines.append(f"static int ca_dense_{r}(const ca_memref_{r} *m, const int64_t *s) {{")
        lines.append(f"  int64_t acc = 1;")
        lines.append(f"  for (int i = {r - 1}; i >= 0; i--) {{")
        lines.append(f"    if (m->sizes[i] != s[i] || (s[i] != 1 && m->strides[i] != acc)) return 0;")
        lines.append(f"    acc *= s[i];")
        lines.append(f"  }}")
        lines.append(f"  return 1;")
        lines.append(f"}}")
    return "\n".join(lines + [""] + body) + "\n"


def main() -> int:
    import argparse
    from pathlib import Path

    from .catalog import compile_catalog

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--llvm-bin", type=Path)
    a = ap.parse_args()
    manifest = compile_catalog(a.input, a.workdir, llvm_bin=a.llvm_bin)
    (a.workdir / "offloaded.mlir").write_text(rewrite(a.input.read_text(), manifest) + "\n")
    (a.workdir / "core_aten_shim.c").write_text(shim_source(manifest))
    print(json.dumps({"bindings": len(manifest["bindings"]), "kernels": len(manifest["kernels"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
