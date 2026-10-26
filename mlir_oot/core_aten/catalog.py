"""Source-bound Core ATen device catalog: match a linalg-on-tensors module, emit one object.

``build(source, readout_options=...)`` accepts explicitly selected Merlin
path/scheme/facts declarations for source-bound composition planning. Without
a usable readout proof, the integer producer retains its full-width exit.
It matches every supported region (``match.match_all``), lints each kernel
stream against the stock-RTL hazard rules, deduplicates identical kernels by their exact
command stream, and returns the combined device module plus a manifest:

    schema    gemmini_core_aten_catalog_v1
    abi       every argument a pointer to a dense row-major buffer, in ``kernels[i].args``
              order; ``scratch`` arguments are caller-allocated, uninitialized and private to
              the call; no argument aliases another; the kernel fences before returning.
    kernels   symbol, Core ATen op, argument table (name, dtype, shape, role), stream counts
    bindings  per matched region: root op ordinal (whose result the kernel writes), the
              ordinals it replaces, and each kernel argument's source value as a function
              argument index or an (op ordinal, result index) pair
    declines  recognised regions this backend refuses, with the reason

It is opt-in beside the integer GEMM catalog (``golden_device_catalog``) and does not alter
that catalog's bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import BlockArgument

from ..frontend.parse import parse_module
from .emit import build_module, compile_to_object, encode
from .lint import HazardError, lint
from .match import match_all

SCHEMA = "gemmini_core_aten_catalog_v1"


def _stream_digest(kernel) -> str:
    h = hashlib.sha256()
    if kernel.stream.bitwise:
        h.update(b'bitwise')
    if kernel.meta.get('integer_producer') or kernel.meta.get('integer_readout'):
        # A source-retained readout contract belongs to its schedule proof.
        # Equal commands with different exact factors must not share metadata.
        h.update(json.dumps(kernel.meta,sort_keys=True).encode())
    if 'limb_plan' in kernel.meta:
        # A symbol's metadata authenticates one representation plan. Identical
        # instructions with different host weights need distinct records.
        h.update(json.dumps(kernel.meta['limb_plan'], sort_keys=True).encode())
    if 'constant_code_packing' in kernel.meta:
        h.update(json.dumps(kernel.meta['constant_code_packing'],sort_keys=True).encode())
    if 'identical_qparam_proof' in kernel.meta:
        h.update(json.dumps(kernel.meta['identical_qparam_proof'], sort_keys=True).encode())
    h.update(json.dumps([(a.name, a.dtype, list(a.shape), a.role) for a in kernel.stream.args]).encode())
    for name, data in sorted(kernel.stream.consts.items()):
        h.update(name.encode() + data.tobytes())
    from dataclasses import asdict
    h.update(json.dumps(_jsonable([asdict(r) for r in kernel.stream.readouts]),sort_keys=True).encode())
    # Bounds belong to the source-bound integer schedule proof.
    h.update(json.dumps({key: _jsonable(kernel.meta[key]) for key in
                        ('accumulator_min', 'accumulator_max', 'accumulator_bounds',
                         'zero_point_corrections') if key in kernel.meta}, sort_keys=True).encode())
    h.update(repr(kernel.stream.packs).encode())
    from .stream import CompactInstructions
    if isinstance(kernel.stream.ins, CompactInstructions):
        h.update(repr(kernel.stream.ins.nodes).encode())
    else:
        for ins in kernel.stream.ins:
            r = encode(ins)
            h.update(repr((r.key, r.values)).encode())
    return h.hexdigest()


def _source(value, ordinals, func_args):
    if isinstance(value, BlockArgument) and value in func_args:
        return {"func_arg": func_args.index(value)}
    owner = value.owner
    return {"op_ordinal": ordinals[owner], "result": list(owner.results).index(value)}


def build(source: str, *, readout_options=None) -> tuple[ModuleOp, dict, list]:
    from .storage import itemsize
    module = parse_module(source)
    ordinals = {op: i for i, op in enumerate(module.walk())}
    funcs = [op for op in module.walk() if op.name == "func.func"]
    func_args = list(funcs[0].body.blocks[0].args) if funcs else []
    matches, declines = match_all(module, readout_options=readout_options)
    kernels: dict[str, dict] = {}
    device = ModuleOp([])
    bindings = []
    accepted = []
    audits = {}
    for m in matches:
        digest = _stream_digest(m.kernel)
        try:
            report = audits.get(digest)
            if report is None:
                report = lint(m.kernel.stream)
                audits[digest] = report
        except HazardError as exc:
            declines.append(dict(ordinal=ordinals[m.root], op=m.root.name,
                                 reason='unsafe device schedule: ' + str(exc), aten=m.op))
            continue
        accepted.append(m)
        symbol = "gemmini_ca_" + digest[:16]
        if symbol not in kernels:
            km, tree = build_module(m.kernel.stream, symbol, meta=m.kernel.meta)
            for op in list(km.body.blocks[0].ops):
                op.detach()
                device.body.blocks[0].add_op(op)
            kernels[symbol] = dict(symbol=symbol, op=m.op, stream_sha256=digest,
                                   args=[dict(name=a.name, dtype=a.dtype, shape=list(a.shape), role=a.role,
                                              **(dict(element_bytes=itemsize(a.dtype)) if m.kernel.stream.bitwise else {}))
                                         for a in m.kernel.args],
                                   instructions=report["instructions"], fences=report["fences"],
                                   counts=m.kernel.stream.counts(), meta=_jsonable(m.kernel.meta),
                                   requantization=_readout_contracts(m.kernel.stream))
        argmap = {name: _source(v, ordinals, func_args) for name, v in m.operands}
        out_name, out_val = m.result
        provs = [getattr(o.attributes.get("prov.aten"), "data", "") for o in m.consumed if o is not None]
        provs = [p for p in provs if p and not p.startswith("quantized_decomposed.")]
        bindings.append(dict(
            op=m.op, aten=provs[0] if provs else "", symbol=symbol, root_ordinal=ordinals[m.root],
            source_operation_ordinal=ordinals[m.root] if 'limb_plan' in m.numeric else ordinals[m.consumed[0]],
            replaced_ordinals=sorted({ordinals[o] for o in m.consumed if o is not None}),
            **(dict(constant_operands={name:_source(value,ordinals,func_args) for name,value in m.constant_operands}) if m.constant_operands else {}),
            operands=argmap, result={out_name: _source(out_val, ordinals, func_args)},
            result_type=str(out_val.type), operand_types={n: str(v.type) for n, v in m.operands},
            region=getattr(m.root.attributes.get("prov.region_id"), "data", ""),
            layout=m.layout, numeric=_jsonable(m.numeric),
            **(_result_slices(m, ordinals, func_args) if m.outputs else {})))
    device.verify()
    manifest = dict(schema=SCHEMA, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    abi=dict(argument_order="kernels[].args", pointee_layout="dense_row_major",
                             scratch="caller-allocated, uninitialized, private to the call",
                             aliasing="none", completion="kernel fences before returning",
                             quantization="per-binding numeric contract: F1 opaque bytes without quantization; F2/F4 full-i32 components and retained exact host recombination; F3 identical-qparam movement storage codes or exact rational host RNE, zero point and declared clamp"),
                    kernels=list(kernels.values()), bindings=bindings, declines=declines,
                    matched=len(bindings), unique_kernels=len(kernels))
    return device, manifest, accepted


def _result_slices(m, ordinals, func_args):
    """A flat multi-result kernel binds each source result to one dense element range."""
    return dict(result_slices=[dict(source=_source(value, ordinals, func_args), offset=offset, shape=list(extent),
                                    type=str(value.type)) for value, offset, extent, _ in m.outputs],
                source_operation_ordinals=[ordinals[op] for _, _, _, op in m.outputs])


def _readout_contracts(stream):
    from dataclasses import asdict
    return _jsonable([dict(asdict(r),rounding='half_to_even',execution='host',device_readout='full_i32') for r in stream.readouts])


def _jsonable(obj):
    import numpy as np
    if isinstance(obj,np.ndarray): return _jsonable(obj.tolist())
    if isinstance(obj,np.integer): return int(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    from fractions import Fraction
    if isinstance(obj, Fraction):
        return dict(numerator=str(obj.numerator), denominator=str(obj.denominator))
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def compile_catalog(source_path: Path, workdir: Path, *, llvm_bin: Path | None = None, readout_options=None) -> dict:
    source = Path(source_path).read_text()
    device, manifest, _ = build(source, readout_options=readout_options)
    workdir.mkdir(parents=True, exist_ok=True)
    if manifest["kernels"]:
        receipt = compile_to_object(device, workdir, llvm_bin=llvm_bin, stem="core_aten")
        manifest["compilation"] = receipt
        manifest['status'] = 'compiled'
    else:
        manifest.update(status='declined', reason='no exact device kernel produced')
    (workdir / "core_aten_catalog.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def merlin_builder(llvm_bin: Path):
    """Return (manifest, object); an empty catalog returns (refusal manifest, None)."""

    def build_fn(prepared: Path, workdir: Path):
        manifest = compile_catalog(Path(prepared), Path(workdir), llvm_bin=Path(llvm_bin))
        return Path(workdir) / "core_aten_catalog.json", (Path(workdir) / "core_aten.o" if manifest['kernels'] else None)

    return build_fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--llvm-bin", type=Path)
    args = ap.parse_args()
    m = compile_catalog(args.input, args.workdir, llvm_bin=args.llvm_bin)
    print(json.dumps({k: m[k] for k in ("matched", "unique_kernels")} | {"declines": len(m["declines"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
