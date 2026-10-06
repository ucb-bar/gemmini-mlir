"""Opt-in, source-bound joint residuals with explicit private second writers.

Certificates select by the complete source arithmetic relation. Existing route
symbols are provenance, not selection predicates. The generic fresh-writer
rewrite owns allocation and deallocation of both outputs; this provider owns
the ranked descriptor ABI, target producer and scaled stores. The original
exact residual object stays linked under its original names, so unchanged
routes and defaults retain their existing implementations.
"""

import json
import shutil
import subprocess
from pathlib import Path

from merlin.llvmlower.quantized_affine_joint import derive_joint, emit_joint_decoder
from xdsl.dialects import func, llvm, tensor
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IntegerAttr,
    StringAttr,
    TensorType,
    UnitAttr,
    i8,
    i64,
)
from xdsl.ir import Region

from .captured_residual_bundle import binding_attributes, canonical, sha
from .direct_conv_binding import serialize
from .frontend.parse import parse_module
from .golden_device_compile import compile_module
from .golden_joint_resadd import build as build_kernel
from .golden_joint_resadd import tables
from .no_fsm_audit import audit_elf


def _identifier(value):
    if (
        not value
        or not value.isascii()
        or not (value[0].isalpha() or value[0] == "_")
        or any(not (c.isalnum() or c == "_") for c in value)
    ):
        raise ValueError("C identifier required")
    return value


def _proof(proof):
    if proof != derive_joint(**proof["source"], predictors=proof["predictors"]):
        raise ValueError("joint certificate changed or cannot be rederived")
    if not proof["decoder_exact_for_all_pairs"]:
        raise ValueError("complete exact joint relation required")
    return proof


def joint_attributes(route, original):
    attrs = binding_attributes(
        route["original_route"],
        original["source_sha256"],
        original["capture_receipt_sha256"],
    )
    # The retained original proof identifies source semantics. Its old hardware
    # scale/coefficients do not describe the selected two-readout implementation.
    attrs.pop("gemmini.wide_integer_coefficients", None)
    attrs["gemmini.qparams"] = DictionaryAttr(
        {
            k: v
            for k, v in attrs["gemmini.qparams"].data.items()
            if k in ("lhs_scale", "rhs_scale", "output_scale", "relu")
        }
    )
    attrs["gemmini.joint_residual_proof_sha256"] = StringAttr(route["proof_sha256"])
    attrs["gemmini.joint_residual_workspace_bytes"] = IntegerAttr(route["m"] * 64, i64)
    attrs["gemmini.joint_residual_storage"] = StringAttr(
        "private fresh C0/C1; immutable A/B; final fence then exact decoder; "
        "C1 is fully written and live through decoding; no retained/free pointers"
    )
    return attrs


def _validate_record(original, record):
    originals = {r["symbol"]: r for r in original["routes"]}
    names, replacements = set(), set()
    for route in record["routes"]:
        old = originals.get(route["original_symbol"])
        proof = _proof(route["proof"])
        if (
            old is None
            or route["original_route"] != old
            or proof["source"] != old["proof"]["source"]
            or not old["proof"]["exact"]
            or old["numeric_policy"]["max_output_lsb"] != 0
            or route["m"] != old["m"]
            or route["n"] != 64
            or route["result_argument"] != 2
            or route["fully_written_arguments"] != [2, 3]
            or route["symbol"] in names
            or route["original_symbol"] in replacements
        ):
            raise ValueError("joint residual source/shape/writer binding changed")
        _identifier(route["symbol"])
        _identifier(route["kernel"])
        build_kernel(route["m"], proof["predictors"], share_affine=True)
        names.add(route["symbol"])
        replacements.add(route["original_symbol"])


def _declarations(module, original, record, selected):
    replacements = {r["original_symbol"]: r for r in record["routes"]}
    expected = {}
    for old in original["routes"]:
        replacement = replacements.get(old["symbol"]) if selected else None
        name = replacement["symbol"] if replacement else old["symbol"]
        count = 4 if replacement else 3
        typ = TensorType(i8, [old["m"], 64])
        attrs = (
            joint_attributes(replacement, original)
            if replacement
            else binding_attributes(
                old, original["source_sha256"], original["capture_receipt_sha256"]
            )
        )
        expected[name] = count, typ, attrs
    calls = [
        op
        for op in module.walk()
        if isinstance(op, func.CallOp) and op.callee.root_reference.data in expected
    ]
    if sorted(op.callee.root_reference.data for op in calls) != sorted(expected):
        raise ValueError("residual call coverage changed")
    declarations = {
        op.sym_name.data: op
        for op in module.body.block.ops
        if isinstance(op, func.FuncOp) and op.sym_name.data in expected
    }
    if set(declarations) != set(expected):
        raise ValueError("residual declaration coverage changed")
    for name, (count, typ, attributes) in expected.items():
        decl = declarations[name]
        access = decl.properties.get("arg_attrs")
        if (
            decl.body.blocks
            or "llvm.emit_c_interface" not in decl.attributes
            or tuple(decl.function_type.inputs) != (typ,) * count
            or tuple(decl.function_type.outputs) != (typ,)
            or access is None
            or [
                getattr(x.data.get("bufferization.access"), "data", None)
                for x in access
            ]
            != ["read", "read", *(["write"] * (count - 2))]
        ):
            raise ValueError("residual type/effect/ABI contract changed")
        if any(decl.attributes.get(k) != v for k, v in attributes.items()):
            raise ValueError("residual source/numeric/storage contract changed")
        for call in (o for o in calls if o.callee.root_reference.data == name):
            if tuple(v.type for v in call.arguments) != (typ,) * count or tuple(
                v.type for v in call.results
            ) != (typ,):
                raise ValueError("residual call ABI changed")
            for output in call.arguments[2:]:
                if (
                    not isinstance(output.owner, tensor.EmptyOp)
                    or sum(1 for _ in output.uses) != 1
                ):
                    raise ValueError("private sole-use empty output required")
    return declarations, calls


def rewrite(module, original, record):
    """Validate the entire source selection before adding private C1 operands."""
    _validate_record(original, record)
    declarations, calls = _declarations(module, original, record, False)
    replacements = {r["original_symbol"]: r for r in record["routes"]}
    names = {
        o.sym_name.data for o in module.body.block.ops if isinstance(o, func.FuncOp)
    }
    if any(r["symbol"] in names for r in record["routes"]):
        raise ValueError("joint residual symbol collision")
    for call in calls:
        route = replacements.get(call.callee.root_reference.data)
        if route is None:
            continue
        typ = call.results[0].type
        empty = tensor.EmptyOp([], typ)
        replacement = func.CallOp(
            route["symbol"], [*call.arguments, empty.tensor], [typ]
        )
        replacement.attributes.update(call.attributes)
        call.parent.insert_ops_before([empty, replacement], call)
        call.results[0].replace_all_uses_with(replacement.results[0])
        call.parent.erase_op(call)
    for route in record["routes"]:
        old = declarations[route["original_symbol"]]
        typ = old.function_type.outputs.data[0]
        access = ArrayAttr(
            [
                DictionaryAttr({"bufferization.access": StringAttr(x)})
                for x in ("read", "read", "write", "write")
            ]
        )
        new = func.FuncOp(
            route["symbol"],
            ([typ] * 4, [typ]),
            Region(),
            visibility="private",
            arg_attrs=access,
        )
        new.attributes.update(old.attributes)
        new.attributes.pop("gemmini.wide_integer_coefficients", None)
        new.attributes.update(joint_attributes(route, original))
        new.attributes["llvm.emit_c_interface"] = UnitAttr()
        module.body.block.insert_op_before(new, old)
        module.body.block.erase_op(old)
    _declarations(module, original, record, True)
    module.verify()
    return module


def adapter(route, *, first_output_guard):
    """Both outputs are passed by the caller; no model-sized static workspace."""
    symbol, kernel, n = route["symbol"], route["kernel"], route["m"] * 64
    decode = symbol + "_decode"
    code = emit_joint_decoder(
        route["proof"], decode, packed=True, first_output_guard=first_output_guard
    )
    coeff = ",".join(map(str, tables(route["proof"]["predictors"], share_affine=True)))
    return (
        code
        + f"""
#ifndef GEMMINI_JOINT_RESIDUAL_ABI
#define GEMMINI_JOINT_RESIDUAL_ABI
typedef struct {{void *allocated,*aligned; intptr_t offset,sizes[2],strides[2];}} joint_memref2;
#endif
static const int8_t {symbol}_coefficients[768] __attribute__((aligned(64)))={{{coeff}}};
extern void {kernel}(const int8_t*,const int8_t*,int8_t*,int8_t*,const int8_t*);
static int8_t *{symbol}_pointer(joint_memref2 *d) {{
 if(!d->aligned || d->offset<0 || d->sizes[0]!={route["m"]} || d->sizes[1]!=64 || d->strides[0]!=64 || d->strides[1]!=1) __builtin_trap();
 uintptr_t base=(uintptr_t)d->aligned,offset=(uintptr_t)d->offset;
 if(base>UINTPTR_MAX-offset || base+offset>UINTPTR_MAX-{n}) __builtin_trap();
 return (int8_t*)(base+offset);
}}
static int {symbol}_overlap(const int8_t *a,const int8_t *b) {{
 uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
 return x<=y ? y-x<{n} : x-y<{n};
}}
void _mlir_ciface_{symbol}(joint_memref2 *r,joint_memref2 *a,joint_memref2 *b,joint_memref2 *c,joint_memref2 *d) {{
 const int8_t *pa={symbol}_pointer(a),*pb={symbol}_pointer(b);
 int8_t *pc={symbol}_pointer(c),*pd={symbol}_pointer(d);
 if({symbol}_overlap(pc,pd) || {symbol}_overlap(pc,pa) || {symbol}_overlap(pc,pb) || {symbol}_overlap(pd,pa) || {symbol}_overlap(pd,pb)) __builtin_trap();
 {kernel}(pa,pb,pc,pd,{symbol}_coefficients);
 {decode}(pc,pd,{n});
 *r=*c;
}}
"""
    )


def oracle(route):
    body = []
    for index, predictor in enumerate(route["proof"]["predictors"]):
        low = 0 if predictor["relu"] else -128
        body.append(
            f"float z{index}=nearbyintf((float)acc*{float(predictor['scale']).hex()}f);"
            f"if(z{index}<{low})z{index}={low};if(z{index}>127)z{index}=127;"
            f"c{index}[i]=(int8_t)z{index};"
        )
    producer = route["proof"]["predictors"][0]
    return f"""#include <math.h>
void {route["kernel"]}(const int8_t*a,const int8_t*b,int8_t*c0,int8_t*c1,const int8_t*coefficients) {{
 (void)coefficients;
 for(intptr_t i=0;i<{route["m"] * 64};i++) {{
  int32_t acc=(int32_t)a[i]*{producer["p"]}+(int32_t)b[i]*{producer["q"]};
  {"".join(body)}
 }}
}}
"""


def build(original_bundle, certificates, llvm_bin, output, *, first_output_guard=False):
    if type(first_output_guard) is not bool:
        raise ValueError("explicit decoder guard boolean required")
    original_bundle, llvm_bin, output = map(Path, (original_bundle, llvm_bin, output))
    original = json.loads((original_bundle / "residual.json").read_text())
    if original["numeric_policy"]["max_output_lsb"] != 0:
        raise ValueError("original exact source policy required")
    proofs = [_proof(json.loads(Path(p).read_text())) for p in certificates]
    sources = [canonical(proof["source"]) for proof in proofs]
    if len(set(sources)) != len(sources) or not sources:
        raise ValueError(
            "one explicit certificate per selected source relation required"
        )
    selected = dict(zip(sources, proofs, strict=True))
    routes = []
    for old in original["routes"]:
        proof = selected.get(canonical(old["proof"]["source"]))
        if proof is None:
            continue
        if not old["proof"]["exact"] or old["numeric_policy"]["max_output_lsb"] != 0:
            raise ValueError("original complete exact source proof required")
        # Resource verifier also closes shared-coefficient, M and i32 bounds.
        build_kernel(old["m"], proof["predictors"], share_affine=True)
        symbol = _identifier(old["symbol"] + "__joint")
        routes.append(
            {
                "original_symbol": old["symbol"],
                "symbol": symbol,
                "kernel": symbol + "_kernel",
                "m": old["m"],
                "n": 64,
                "original_route": old,
                "proof": proof,
                "proof_sha256": sha(
                    Path(certificates[sources.index(canonical(proof["source"]))])
                ),
                "result_argument": 2,
                "fully_written_arguments": [2, 3],
            }
        )
    if {canonical(r["proof"]["source"]) for r in routes} != set(sources):
        raise ValueError("selected numeric relation absent from bound source")
    output.mkdir(parents=True, exist_ok=False)
    objects, native = [], []
    for route in routes:
        directory = output / route["symbol"]
        module = build_kernel(
            route["m"], route["proof"]["predictors"], share_affine=True
        )
        for op in module.body.block.ops:
            if isinstance(op, llvm.FuncOp):
                op.properties["sym_name"] = StringAttr(route["kernel"])
        route["compilation"] = compile_module(module, llvm_bin, directory)
        code = adapter(route, first_output_guard=first_output_guard)
        source, obj = directory / "adapter.c", directory / "adapter.o"
        source.write_text(code)
        argv = [
            str(llvm_bin / "clang"),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-mcmodel=medany",
            "-O2",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
        subprocess.run(argv, check=True, capture_output=True)
        route["adapter_compilation"] = {
            "argv": argv,
            "source_sha256": sha(source),
            "object_sha256": sha(obj),
        }
        objects.extend([directory / "kernel.o", obj])
        native.append(code + oracle(route))
    linker = llvm_bin / "ld.lld"
    if not linker.is_file():
        linker = Path(shutil.which("ld.lld"))
    argv = [str(linker), "-r", *map(str, objects), "-o", str(output / "joint.o")]
    subprocess.run(argv, check=True, capture_output=True)
    audit = audit_elf((output / "joint.o").read_bytes())
    if audit["status"] != "pass":
        raise ValueError("joint residual object audit failed")
    (output / "native_oracle.c").write_text("\n".join(native))
    record = {
        "schema": "gemmini_source_bound_joint_residual_v1",
        "original_manifest_path": str((original_bundle / "residual.json").resolve()),
        "original_manifest_sha256": sha(original_bundle / "residual.json"),
        "certificates": [
            {"path": str(Path(p).resolve()), "sha256": sha(p)} for p in certificates
        ],
        "first_output_guard": first_output_guard,
        "routes": routes,
        "object_sha256": sha(output / "joint.o"),
        "native_oracle_sha256": sha(output / "native_oracle.c"),
        "nofsm_audit": audit,
        "linker_argv": argv,
        "linker_sha256": sha(linker),
        "target_cycle_cost": "UNKNOWN; requires matched target measurements",
        "workspace": "caller private fresh i8 Mx64 second writer, no retained/free pointers",
        "numeric_policy": original["numeric_policy"],
        "default_selection_enabled": False,
    }
    (output / "joint.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def merlin_callbacks(llvm_bin, original_bundle, joint_bundle, base_callbacks):
    llvm_bin, original_bundle, joint_bundle = map(
        Path, (llvm_bin, original_bundle, joint_bundle)
    )
    original = json.loads((original_bundle / "residual.json").read_text())
    record = json.loads((joint_bundle / "joint.json").read_text())
    original_pin, selected_pin = (
        sha(original_bundle / "residual.json"),
        sha(joint_bundle / "joint.json"),
    )
    if original_pin != record["original_manifest_sha256"]:
        raise ValueError("joint original source context changed")
    _validate_record(original, record)
    prepare_base, build_base = base_callbacks

    def check():
        for path, digest in [
            (original_bundle / "residual.json", original_pin),
            (original_bundle / "residual.o", original["object_sha256"]),
            (original_bundle / "native_oracle.c", original["native_oracle_sha256"]),
            (joint_bundle / "joint.json", selected_pin),
            (joint_bundle / "joint.o", record["object_sha256"]),
            (joint_bundle / "native_oracle.c", record["native_oracle_sha256"]),
            *[(Path(p["path"]), p["sha256"]) for p in record["certificates"]],
        ]:
            if sha(path) != digest:
                raise ValueError("joint residual artifact/source binding changed")

    def prepare(source, work):
        check()
        _declarations(parse_module(Path(source).read_text()), original, record, False)
        prepared = prepare_base(source, work)
        module = parse_module(Path(prepared).read_text())
        rewrite(module, original, record)
        selected = Path(work) / "joint_residual_prepared.mlir"
        selected.write_text(serialize(module, []))
        return selected

    def compile_catalog(source, work):
        check()
        _declarations(parse_module(Path(source).read_text()), original, record, True)
        path, obj = build_base(source, work)
        manifest = json.loads(path.read_text())
        combined = Path(work) / "residual_mixed.o"
        linker = llvm_bin / "ld.lld"
        if not linker.is_file():
            linker = Path(shutil.which("ld.lld"))
        argv = [
            str(linker),
            "-r",
            str(obj),
            str(original_bundle / "residual.o"),
            str(joint_bundle / "joint.o"),
            "-o",
            str(combined),
        ]
        subprocess.run(argv, check=True, capture_output=True)
        if audit_elf(combined.read_bytes())["status"] != "pass":
            raise ValueError("joint mixed model object audit failed")
        manifest["compilation"] = {
            "schema": "gemmini_joint_residual_mixed_compile_v1",
            "object_sha256": sha(combined),
            "object_nofsm_status": "pass",
            "remaining_compilation": manifest["compilation"],
            "residual_manifest_sha256": original_pin,
            "joint_manifest_sha256": selected_pin,
            "linker_sha256": sha(linker),
            "linker_argv": argv,
        }
        replacements = {r["original_symbol"]: r for r in record["routes"]}
        manifest["residual_additions"] = [
            replacements.get(r["symbol"], r) for r in original["routes"]
        ]
        manifest["residual_numeric_policy"] = original["numeric_policy"]
        manifest["total_device_contractions"] += len(original["routes"])
        manifest["native_oracle_sources"].extend(
            [
                str(original_bundle / "native_oracle.c"),
                str(joint_bundle / "native_oracle.c"),
            ]
        )
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return path, combined

    return prepare, compile_catalog
