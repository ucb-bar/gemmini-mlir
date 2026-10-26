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
from xdsl.ir import BlockArgument, OpResult

from ..frontend.parse import parse_module
from .emit import build_module, compile_to_object, encode
from .lint import HazardError, lint
from .match import match_all
from .source import module_of

SCHEMA = "gemmini_core_aten_catalog_v1"


def _stream_digest(kernel) -> str:
    h = hashlib.sha256()
    if kernel.stream.bitwise:
        h.update(b'bitwise')
    if kernel.meta.get('integer_producer') or kernel.meta.get('integer_readout'):
        # A source-retained readout contract belongs to its schedule proof.
        # Equal commands with different exact factors must not share metadata.
        h.update(json.dumps(dict(kernel.meta),sort_keys=True).encode())
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
        # Emission records this contract in metadata. Finalize it before
        # naming the stream so authentication sees the same bytes afterward.
        m.kernel.meta['host_readout_contract'] = 'exact_rational_RNE_v1'
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
            runtime_scales = _runtime_scale_arguments(m.kernel.stream)
            km, tree = build_module(m.kernel.stream, symbol, meta=m.kernel.meta)
            for op in list(km.body.blocks[0].ops):
                op.detach()
                device.body.blocks[0].add_op(op)
            kernels[symbol] = dict(symbol=symbol, op=m.op, stream_sha256=digest,
                                   args=argument_records(m),
                                   instructions=report["instructions"], fences=report["fences"],
                                   counts=m.kernel.stream.counts(), meta=_jsonable(m.kernel.meta),
                                   requantization=_readout_contracts(m.kernel.stream),
                                   **(dict(runtime_scale_arguments=runtime_scales) if runtime_scales else {}),
                                   **(dict(runtime_oracle_obligations=_runtime_oracle_obligations(report))
                                      if report.get('runtime_oracle_obligations') else {}))
        bindings.append(binding_record(m, symbol, ordinals, func_args))
    device.verify()
    manifest = dict(schema=SCHEMA, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    abi=dict(argument_order="kernels[].args", pointee_layout="dense_row_major",
                             scratch="caller-allocated, uninitialized, private to the call",
                             aliasing="none", completion="kernel fences before returning",
                             quantization="per-binding numeric contract: F1 opaque bytes without quantization; F2/F4 full-i32 components and retained exact host recombination; F3 identical-qparam movement storage codes or exact rational host RNE, zero point and declared clamp"),
                    kernels=list(kernels.values()), bindings=bindings, declines=declines,
                    matched=len(bindings), unique_kernels=len(kernels))
    return device, manifest, accepted


def argument_records(match):
    """The ordered physical ABI of a source-bound match, including private scratch.

    Source admission authenticates the match's operands, outputs and physical
    representation. The manifest must reproduce that complete argument list;
    its own names, roles and order cannot serve as evidence for the ABI.
    """
    from .storage import itemsize
    return [dict(name=a.name, dtype=a.dtype, shape=list(a.shape), role=a.role,
                 **(dict(element_bytes=itemsize(a.dtype)) if match.kernel.stream.bitwise else {}))
            for a in match.kernel.args]


def binding_record(m, symbol, ordinals, func_args):
    """The manifest binding ``offload.rewrite`` replays, derived from the match's source values."""
    if m.outputs and m.result_buffers:
        raise ValueError("result buffers and flat slices are mutually exclusive")
    if m.result_buffers:
        _validate_result_buffers(m)
    out_name, out_val = m.result
    provs = [getattr(o.attributes.get("prov.aten"), "data", "") for o in m.consumed if o is not None]
    provs = [p for p in provs if p and not p.startswith("quantized_decomposed.")]
    return dict(
        op=m.op, aten=provs[0] if provs else "", symbol=symbol, root_ordinal=ordinals[m.root],
        source_operation_ordinal=ordinals[m.root] if 'limb_plan' in m.numeric else ordinals[m.consumed[0]],
        replaced_ordinals=sorted({ordinals[o] for o in m.consumed if o is not None}),
        **(dict(constant_operands={name: _source(value, ordinals, func_args) for name, value in m.constant_operands})
           if m.constant_operands else {}),
        operands={name: _source(v, ordinals, func_args) for name, v in m.operands},
        result={out_name: _source(out_val, ordinals, func_args)},
        result_type=str(out_val.type), operand_types={n: str(v.type) for n, v in m.operands},
        region=getattr(m.root.attributes.get("prov.region_id"), "data", ""),
        layout=m.layout, numeric=_jsonable(m.numeric),
        **(_result_slices(m, ordinals, func_args) if m.outputs else {}),
        **(_result_buffers(m, ordinals, func_args) if m.result_buffers else {}))


def authenticate_binding(binding, m):
    """Raise unless ``binding`` is exactly what ``m``'s own source values and kernel bind.

    The manifest binding is what offload rewrites, so every admission path
    authenticates it here: operand/result/slice source references and replaced
    ordinals are re-derived, and the bound results must be the only values
    escaping the replaced ops.
    """
    module = module_of(m.root)
    ordinals = {op: i for i, op in enumerate(module.walk())}
    funcs = [op for op in module.walk() if op.name == "func.func"]
    func_args = list(funcs[0].body.blocks[0].args) if funcs else []
    symbol = 'gemmini_ca_' + _stream_digest(m.kernel)[:16]
    if binding.get('symbol') != symbol:
        raise ValueError('catalog binding symbol differs from its kernel stream digest')
    if _jsonable(binding) != _jsonable(binding_record(m, symbol, ordinals, func_args)):
        raise ValueError('catalog binding differs from the source values its match binds')
    consumed = {id(op) for op in m.consumed if op is not None}
    results = [m.result[1]] if not m.outputs else [value for value, _, _, _ in m.outputs]
    if m.result_buffers:
        results = [buffer.value for buffer in m.result_buffers]
    escaping = {id(value) for op in m.consumed if op is not None for value in op.results
                if any(id(use.operation) not in consumed for use in value.uses)}
    if m.result_buffers and escaping != {id(v) for v in results}:
        raise ValueError('result buffers must bind exactly the escaping source values')
    if (any(not isinstance(v, OpResult) or id(v.owner) not in consumed for v in results)
            or not escaping <= {id(v) for v in results}):
        raise ValueError('bound results are not the values escaping the replaced source operations')
    if any(isinstance(v, OpResult) and id(v.owner) in consumed for _, v in (*m.operands, *m.constant_operands)):
        raise ValueError('bound operand is produced inside the replaced source operations')


def authenticate_catalog(manifest, matches):
    """Authenticate the complete binding table and full stream digests before linking."""
    if len(manifest['bindings']) != len(matches):
        raise ValueError('catalog bindings differ from the admitted matches')
    digests = {}
    arguments = {}
    scales = {}
    obligations = {}
    for binding, match in zip(manifest['bindings'], matches):
        authenticate_binding(binding, match)
        digest = _stream_digest(match.kernel)
        symbol = binding['symbol']
        if symbol in digests and digests[symbol] != digest:
            raise ValueError('catalog binding symbol aliases different kernel stream digests')
        digests[symbol] = digest
        arguments[symbol] = argument_records(match)
        scales[symbol] = _runtime_scale_arguments(match.kernel.stream)
        obligations[symbol] = _runtime_oracle_obligations(match.kernel.stream)
    if (len(manifest['kernels']) != len(digests)
            or {k['symbol']: k.get('stream_sha256') for k in manifest['kernels']} != digests):
        raise ValueError('catalog kernel digests differ from the authenticated binding streams')
    if any(k.get('args') != arguments[k['symbol']] for k in manifest['kernels']):
        raise ValueError('catalog kernel arguments differ from the source-bound kernel ABI')
    if any(k.get('runtime_scale_arguments', []) != scales[k['symbol']] for k in manifest['kernels']):
        raise ValueError('catalog runtime scale arguments differ from the authenticated stream references')
    if any(k.get('runtime_oracle_obligations', []) != obligations[k['symbol']] for k in manifest['kernels']):
        raise ValueError('catalog runtime oracle obligations differ from the authenticated stream lint')
    return digests


def authenticate_proof(manifest, matches, proof):
    """Require the compiler's actual streams and bindings to equal the admitted catalog."""
    digests = authenticate_catalog(manifest, matches)
    if (proof.get('status') != 'proven' or proof.get('source_sha256') != manifest['source_sha256']
            or proof.get('bindings') != manifest['bindings']
            or len(proof.get('kernels', [])) != len(digests)
            or {k['symbol']: k.get('stream_sha256') for k in proof.get('kernels', [])} != digests):
        raise ValueError('compiled catalog differs from the proved numerical schedule or stream digests')


def _validate_result_buffers(m):
    from .match import shape
    from .storage import itemsize

    outputs = [a for a in m.kernel.args if a.role == 'output']
    if (len(outputs) != len(m.result_buffers)
            or len({id(b.value) for b in m.result_buffers}) != len(outputs)
            or len({b.argument for b in m.result_buffers}) != len(outputs)):
        raise ValueError('result buffers do not bind every output exactly once')
    if m.result != (m.result_buffers[0].argument, m.result_buffers[0].value):
        raise ValueError('result compatibility view differs from the first buffer')
    for buffer, arg in zip(m.result_buffers, outputs):
        dtype = str(buffer.value.type.get_element_type())
        if (any(type(d) is not int for d in buffer.shape)
                or buffer.argument != arg.name or tuple(buffer.shape) != shape(buffer.value)
                or tuple(arg.shape) != tuple(buffer.shape) or arg.dtype != dtype
                or itemsize(dtype) not in (1, 4) or buffer.root not in m.consumed):
            raise ValueError('result buffer differs from its source shape, type or output argument')


def _result_buffers(m, ordinals, func_args):
    from math import prod
    from .storage import itemsize

    return dict(result_buffers=[dict(source=_source(b.value, ordinals, func_args), argument=b.argument,
        shape=list(b.shape), type=str(b.value.type), byte_offset=0,
        byte_length=prod(b.shape) * itemsize(str(b.value.type.get_element_type())),
        source_operation_ordinal=ordinals[b.root]) for b in m.result_buffers])


def _result_slices(m, ordinals, func_args):
    """A flat multi-result kernel binds each source result to one dense element range."""
    return dict(result_slices=[dict(source=_source(value, ordinals, func_args), offset=offset, shape=list(extent),
                                    type=str(value.type)) for value, offset, extent, _ in m.outputs],
                source_operation_ordinals=[ordinals[op] for _, _, _, op in m.outputs])


def _runtime_oracle_obligations(source):
    """The lint's runtime oracle obligations, from a lint report or by linting a stream."""
    report = source if isinstance(source, dict) else lint(source)
    return _jsonable(report.get('runtime_oracle_obligations', []))


def _runtime_scale_arguments(stream):
    refs = [ins.a('scale_operand') for ins in stream.ins if ins.a('scale_operand') is not None]
    if not refs:
        return []
    from .runtime_scale import validate_scale_ref

    for ref in refs:
        validate_scale_ref(ref, stream.args)
    return [stream.args[ref.arg_index].name for ref in sorted(set(refs))]


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


def compile_catalog(source_path: Path, workdir: Path, *, llvm_bin: Path | None = None, readout_options=None,
                    numeric_schedule_proof=None) -> dict:
    source = Path(source_path).read_text()
    device, manifest, matches = build(source, readout_options=readout_options)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        if manifest['kernels']:
            from .semantic_gate import compile_admitted
            manifest['compilation'] = compile_admitted(manifest, matches, numeric_schedule_proof,
                lambda: compile_to_object(device, workdir, llvm_bin=llvm_bin, stem="core_aten"),
                readout_options=readout_options)
            manifest['status'] = 'compiled'
    except ValueError as exc:
        manifest.update(status='declined', reason=str(exc),
                        kernels=[], bindings=[], matched=0, unique_kernels=0)
    if numeric_schedule_proof is not None:
        manifest['numeric_schedule_proof'] = numeric_schedule_proof
    if not manifest["kernels"] and manifest.get('status') != 'declined':
        manifest.update(status='declined', reason='no exact device kernel produced')
    (workdir / "core_aten_catalog.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _execution_provider():
    from merlin.targetgen.core_aten_device import load_execution_provider
    package = Path(__file__).resolve().parents[2]
    return load_execution_provider('gemmini', package / 'merlin-support/core_aten/execution_provider.py')


def merlin_builder(llvm_bin: Path, *, numeric_policy=None, readout_options=None):
    """Admit the prepared source before compiling; refusals return no object."""

    def build_fn(prepared: Path, workdir: Path):
        return _execution_provider().core_aten_catalog(Path(prepared), Path(workdir),
            package=Path(__file__).resolve().parents[2], llvm_bin=Path(llvm_bin),
            numeric_policy=numeric_policy, readout_options=readout_options)

    return build_fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--llvm-bin", type=Path)
    ap.add_argument("--numeric-policy", type=Path, help="explicit reviewed Merlin software spec")
    args = ap.parse_args()
    policy = None
    if args.numeric_policy is not None:
        from merlin.targetgen.software_spec import load_software_spec
        policy = load_software_spec(args.numeric_policy, target='gemmini')
    manifest, _ = _execution_provider().core_aten_catalog(args.input, args.workdir,
        package=Path(__file__).resolve().parents[2], llvm_bin=args.llvm_bin, numeric_policy=policy)
    m = json.loads(manifest.read_text())
    print(json.dumps(dict(status=m['status'], matched=m.get('matched', 0),
                         unique_kernels=m.get('unique_kernels', 0), declines=len(m.get('declines', ())))))
    return 0 if m['status'] == 'compiled' else 1


if __name__ == "__main__":
    raise SystemExit(main())
