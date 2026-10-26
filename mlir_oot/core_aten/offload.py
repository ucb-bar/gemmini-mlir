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
descriptor. Mixed results use independent typed destinations and one aggregate
result descriptor, under an explicit ``ca_buffers_`` symbol.
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
    fn = next(op for op in ops_ if op.name == "func.func" and op.body.blocks)
    func_args = list(fn.body.blocks[0].args)
    kernels = {k["symbol"]: k for k in manifest["kernels"]}
    declared = {}
    doomed = []
    replacements = {}
    for b in manifest["bindings"]:
        k = kernels[b["symbol"]]
        if "result_buffers" in b or "result_slices" in b:
            rewriter = _rewrite_buffers if "result_buffers" in b else _rewrite_flat
            callee, ftype = rewriter(b, k, ops_, func_args, replacements)
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


def _rewrite_buffers(binding, kernel, ops_, func_args, replacements):
    """One destination and returned tensor per independently typed source result."""
    from math import prod
    from .storage import itemsize

    pieces = binding['result_buffers']
    outs = [a for a in kernel['args'] if a['role'] == 'output']
    if 'result_slices' in binding or len(pieces) != len(outs) or not pieces:
        raise ValueError('result buffers differ from the kernel outputs')
    values = [_value(b['source'], ops_, func_args) for b in pieces]
    for b, arg, value in zip(pieces, outs, values):
        if (b['argument'] != arg['name'] or b['shape'] != arg['shape']
                or tuple(value.type.get_shape()) != tuple(b['shape']) or str(value.type) != b['type']
                or str(value.type.get_element_type()) != arg['dtype'] or b['byte_offset'] != 0
                or b['byte_length'] != prod(arg['shape']) * itemsize(arg['dtype'])):
            raise ValueError('result buffer differs from its typed destination')
    ins = [_value(binding['operands'][a['name']], ops_, func_args)
           for a in kernel['args'] if a['role'] == 'input']
    ins = [replacements.get(v, v) for v in ins]
    empties = [tensor.EmptyOp([], v.type) for v in values]
    callee = 'ca_buffers_' + binding['symbol']
    call = func.CallOp(callee, [*ins, *(e.results[0] for e in empties)], [v.type for v in values])
    root = ops_[binding['root_ordinal']]
    root.parent_block().insert_ops_before([*empties, call], root)
    for value, result in zip(values, call.results):
        value.replace_all_uses_with(result)
        replacements[value] = result
    return callee, FunctionType.from_lists([v.type for v in ins + values], [v.type for v in values])


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
    buffer_symbols = {b["symbol"] for b in manifest.get("bindings", []) if "result_buffers" in b}
    for k in manifest["kernels"]:
        if k["symbol"] in buffer_symbols or k.get("runtime_scale_arguments"):
            ranks.update(len(a["shape"]) for a in k["args"] if a["role"] in ("input", "output"))
            body += _buffer_shim(k, buffers=k["symbol"] in buffer_symbols)
            continue
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
    lines.append("#include <string.h>")
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


def _buffer_shim(kernel, *, buffers=True):
    """MLIR returns several descriptors through a pointer to one aggregate."""
    from math import prod
    from .storage import itemsize

    sym, args = kernel['symbol'], kernel['args']
    ins = [a for a in args if a['role'] == 'input']
    outs = [a for a in args if a['role'] == 'output']
    if not outs:
        raise ValueError('buffer ABI requires a destination')
    aggregate = f'ca_results_{sym}'
    fields = ' '.join(f"ca_memref_{len(a['shape'])} result{i};" for i, a in enumerate(outs))
    lines = [f'typedef struct {{ {fields} }} {aggregate};',
             f"extern void {sym}({', '.join(['void *'] * len(args))});"]
    for a in args:
        if a['role'] == 'scratch':
            lines.append(f"static uint8_t {sym}__{a['name']}[{prod(a['shape']) * itemsize(a['dtype'])}] "
                         '__attribute__((aligned(64)));')
    result_type = aggregate if buffers else f"ca_memref_{len(outs[0]['shape'])}"
    callee = ('ca_buffers_' if buffers else 'ca_') + sym
    params = [f'{result_type} *result'] + [f"ca_memref_{len(a['shape'])} *{a['name']}" for a in ins + outs]
    lines.append(f"void _mlir_ciface_{callee}({', '.join(params)}) {{")
    scales = kernel.get('runtime_scale_arguments', [])
    if not set(scales) <= {a['name'] for a in ins}:
        raise ValueError('runtime scale argument has no input descriptor')
    for a in ins + outs:
        lines += _checked_buffer_operand(a, sym, scale=a['name'] in scales)
    lines += _nonaliasing_results(ins + outs, sym)
    call = [f"{sym}__{a['name']}" if a['role'] == 'scratch' else f"p_{a['name']}" for a in args]
    lines.append(f"  {sym}({', '.join(call)});")
    if buffers:
        lines += [f"  result->result{i} = *{a['name']};" for i, a in enumerate(outs)]
    else:
        lines.append(f"  *result = *{outs[0]['name']};")
    return lines + ['}', '']


def _checked_buffer_operand(arg, symbol, *, scale=False):
    from .storage import itemsize

    name, rank, width = arg['name'], len(arg['shape']), itemsize(arg['dtype'])
    shape = ', '.join(map(str, arg['shape'])) or '1'
    lines = [f'  if (!{name} || !{name}->aligned || {name}->offset < 0 || '
             f'!ca_dense_{rank}({name}, (const int64_t[]){{{shape}}})) '
             f'ca_abort("{symbol}: non-dense or mis-shaped operand");',
             f'  uint8_t *p_{name} = (uint8_t *){name}->aligned + {name}->offset * {width};']
    if width > 1:
        lines.append(f'  if ((uintptr_t)p_{name} % {width}) ca_abort("{symbol}: misaligned operand");')
    if scale:
        if arg['dtype'] != 'f32' or tuple(arg['shape']) != (1,):
            raise ValueError('runtime scale must have f32[1] storage')
        lines.append(f'  uint32_t bits_{name}; memcpy(&bits_{name}, p_{name}, 4);')
        lines.append(f'  if ((bits_{name} & 0x7f800000u) == 0x7f800000u) '
                     f'ca_abort("{symbol}: nonfinite runtime scale");')
    return lines


def _nonaliasing_results(args, symbol):
    from math import prod
    from .storage import itemsize

    lines = []
    for i, a in enumerate(args):
        for b in args[i + 1:]:
            if a['role'] != 'output' and b['role'] != 'output':
                continue
            na, nb = a['name'], b['name']
            sa, sb = prod(a['shape']) * itemsize(a['dtype']), prod(b['shape']) * itemsize(b['dtype'])
            lines.append(f'  if ((uintptr_t)p_{na} < (uintptr_t)p_{nb} + {sb} && '
                         f'(uintptr_t)p_{nb} < (uintptr_t)p_{na} + {sa}) '
                         f'ca_abort("{symbol}: aliased result buffer");')
    return lines


def main() -> int:
    import argparse
    from pathlib import Path

    from .catalog import _execution_provider

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--llvm-bin", type=Path)
    ap.add_argument("--numeric-policy", type=Path, help="explicit reviewed Merlin software spec")
    a = ap.parse_args()
    policy = None
    if a.numeric_policy is not None:
        from merlin.targetgen.software_spec import load_software_spec
        policy = load_software_spec(a.numeric_policy, target='gemmini')
    result = _execution_provider().core_aten_offload(a.input, a.workdir,
        package=Path(__file__).resolve().parents[2], llvm_bin=a.llvm_bin, numeric_policy=policy)
    manifest = json.loads(result[3].read_text())
    if result[0] is not None:
        (a.workdir / 'offloaded.mlir').write_text(result[0].read_text())
    print(json.dumps(dict(status=manifest['status'], bindings=len(manifest['bindings']),
                         kernels=len(manifest['kernels']))))
    return 0 if manifest['status'] == 'compiled' else 1


if __name__ == "__main__":
    raise SystemExit(main())
