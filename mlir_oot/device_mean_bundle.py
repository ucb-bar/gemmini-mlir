"""Source-bound exact integer mean reduction with a primitive Gemmini producer.

Merlin owns the complete source certificate and ordered binary32 finishing.
This provider owns integer producer resources, its physical NHWC ABI, and final
device fence. The ordinary tensor call allocates private i32 sums and i8 output;
all original source inputs remain immutable through finishing.
"""

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.llvmlower.guarded_quantized_mean import derive, emit_integer_sum_finish
from xdsl.dialects import func, tensor
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    StringAttr,
    TensorType,
    UnitAttr,
    i8,
    i32,
)
from xdsl.ir import Region

from .captured_residual_bundle import reshape
from .direct_conv_binding import serialize
from .frontend.parse import parse_module
from .golden_device_compile import compile_module
from .golden_gemm import GoldenGemm, Shape
from .golden_resadd_proof import op_name
from .guarded_mean_bundle import inspect, packed_nhwc_route, sha
from .no_fsm_audit import audit_elf
from .tables import rtl_facts as F


@dataclass(frozen=True)
class IntegerSumPlan:
    batches: int
    channels: int
    count: int

    def validate(self):
        if any(
            type(v) is not int or v <= 0
            for v in (self.batches, self.channels, self.count)
        ):
            raise ValueError("positive static mean geometry required")
        if self.count > 128 or 128 * self.count > (1 << 31) - 1:
            raise ValueError("source certificate count or i32 overflow bound exceeded")
        if self.batches * self.channels * self.count >= (1 << 63):
            raise ValueError("physical byte offsets exceed signed64")
        self.shape().validate()

    def shape(self):
        return Shape(
            1,
            self.channels,
            self.count,
            output_dtype="i32",
            bm=1,
            bn=min(F.ACC_ROWS // F.DIM, (self.channels + F.DIM - 1) // F.DIM),
            wide_store=True,
            reuse_b=True,
            cache_a=True,
            wide_b=True,
            prefetch_b=self.count > F.DIM,
        )

    def proof(self):
        self.validate()
        return {
            "input_dtype": "i8",
            "sum_dtype": "i32",
            "initial_value": 0,
            "coefficient": 1,
            "sum_min": -128 * self.count,
            "sum_max": 127 * self.count,
            "overflow": False,
            "input_layout": "dense [B,count,C]",
            "scratch_shape": [self.batches, self.channels],
            "schedule": asdict(self.shape()),
            "input_access": "read_only",
            "sums_access": "complete_write_then_read",
            "output_access": "complete_write",
            "private_outputs": "Fresh disjoint i32 scratch and i8 output; original input immutable through finishing",
            "ordering": "Primitive producer final FENCE precedes every CPU sum read; no streaming overlap",
        }


def rewrite(module, quantize, route, symbol, source_sha):
    if any(
        op.name == "func.func" and op.sym_name.data == symbol
        for op in module.body.block.ops
    ):
        raise ValueError("mean symbol already declared")
    views, value = reshape(route["input"], route["input_matrix_shape"])
    output_type = quantize.results[0].type
    sums_type = TensorType(i32, route["output_shape"])
    sums, output = tensor.EmptyOp([], sums_type), tensor.EmptyOp([], output_type)
    call = func.CallOp(symbol, [value, sums.tensor, output.tensor], [output_type])
    quantize.parent.insert_ops_before([*views, sums, output, call], quantize)
    quantize.results[0].replace_all_uses_with(call.results[0])
    quantize.parent.erase_op(quantize)
    declaration = func.FuncOp(
        symbol,
        ([value.type, sums_type, output_type], [output_type]),
        Region(),
        visibility="private",
        arg_attrs=ArrayAttr(
            [
                DictionaryAttr({"bufferization.access": StringAttr(v)})
                for v in ("read", "write", "write")
            ]
        ),
    )
    declaration.attributes.update(
        {
            "llvm.emit_c_interface": UnitAttr(),
            "merlin.guarded_mean_proof_sha256": StringAttr(
                hashlib.sha256(
                    json.dumps(route["proof"], sort_keys=True).encode()
                ).hexdigest()
            ),
            "merlin.guarded_mean_source_sha256": StringAttr(source_sha),
        }
    )
    module.body.block.add_op(declaration)


def adapter_source(route, symbol):
    proof = route["proof"]
    if proof != derive(proof["count"], proof["input_scale"], proof["output_scale"]):
        raise ValueError("source mean certificate changed")
    plan = IntegerSumPlan(
        route["output_shape"][0], route["output_shape"][1], proof["count"]
    )
    plan.validate()
    if route["input_matrix_shape"] != [plan.batches * plan.count, plan.channels]:
        raise ValueError("physical NHWC mean input extent changed")
    if not symbol.isidentifier():
        raise ValueError("C identifier required")
    finish = emit_integer_sum_finish(
        proof,
        symbol + "_finish",
        plan.batches,
        plan.channels,
        input_layout="channel_minor",
    )
    ones = ",".join(["1"] * plan.count)
    kernel = symbol + "_integer_sum_kernel"
    return (
        finish
        + f"""
_Static_assert(sizeof(int32_t)==4,"i32 sum storage required");
static const int8_t {symbol}_ones[{plan.count}] __attribute__((aligned(64)))={{{ones}}};
void {kernel}(const int8_t*,const int8_t*,int32_t*);
typedef struct {{void *allocated,*aligned; int64_t offset,size[2],stride[2];}} {symbol}_memref2;
void _mlir_ciface_{symbol}({symbol}_memref2 *result,{symbol}_memref2 *a,{symbol}_memref2 *sums,{symbol}_memref2 *out) {{
 if(a->size[0]!={plan.batches * plan.count} || a->size[1]!={plan.channels} || a->stride[0]!={plan.channels} || a->stride[1]!=1 ||
    sums->size[0]!={plan.batches} || sums->size[1]!={plan.channels} || sums->stride[0]!={plan.channels} || sums->stride[1]!=1 ||
    out->size[0]!={plan.batches} || out->size[1]!={plan.channels} || out->stride[0]!={plan.channels} || out->stride[1]!=1) __builtin_trap();
 const int8_t *input=(const int8_t*)a->aligned+a->offset;
 int32_t *totals=(int32_t*)sums->aligned+sums->offset;
 int8_t *output=(int8_t*)out->aligned+out->offset;
 for(int batch=0;batch<{plan.batches};++batch)
  {kernel}({symbol}_ones,input+(int64_t)batch*{plan.count * plan.channels},totals+(int64_t)batch*{plan.channels});
 {symbol}_finish(input,totals,output);
 *result=*out;
}}
"""
    )


def scalar_sum_source(plan, symbol):
    plan.validate()
    return f"""#include <stdint.h>
void {symbol}(const int8_t *ones,const int8_t *input,int32_t *sums) {{
 for(int channel=0;channel<{plan.channels};++channel) {{
  int32_t sum=0;
  for(int k=0;k<{plan.count};++k) sum+=(int32_t)ones[k]*(int32_t)input[(int64_t)k*{plan.channels}+channel];
  sums[channel]=sum;
 }}
}}
"""


def build_and_apply(capture, llvm_bin, directory):
    capture, llvm_bin, directory = map(Path, (capture, llvm_bin, directory))
    directory.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((capture / "capture_receipt.json").read_text())
    for name in (
        "model.mlir",
        "weights.safetensors",
        "weights.safetensors.manifest.json",
    ):
        if sha(capture / name) != receipt["artifacts"][name]["sha256"]:
            raise ValueError("capture identity changed: " + name)
    original, golden = sha(capture / "model.mlir"), sha(capture / "golden.npy")
    module = parse_module((capture / "model.mlir").read_text())
    routes, objects, oracle = [], [], []
    for op in list(module.walk()):
        if op_name(op) != "quant_ext.quantize_per_tensor":
            continue
        try:
            route = packed_nhwc_route(inspect(op), require_word_lanes=False)
        except ValueError:
            continue
        symbol = "gemmini_integer_sum_mean_" + str(len(routes))
        plan = IntegerSumPlan(*route["output_shape"], route["proof"]["count"])
        plan.validate()
        rewrite(module, op, route, symbol, original)
        saved = {k: v for k, v in route.items() if k != "input"} | {
            "symbol": symbol,
            "kernel": symbol + "_integer_sum_kernel",
            "implementation": "gemmini_integer_sum",
            "producer_contract": plan.proof(),
            "result_argument": 2,
            "fully_written_arguments": [1, 2],
        }
        emitted = GoldenGemm(plan.shape()).build()
        emitted.body.block.first_op.properties["sym_name"] = StringAttr(saved["kernel"])
        compiled = compile_module(emitted, llvm_bin, directory / symbol)
        saved["compilation"] = compiled
        objects.append(directory / symbol / "kernel.o")
        routes.append(saved)
        oracle.append(scalar_sum_source(plan, saved["kernel"]))
    if not routes:
        raise ValueError("no canonical static NHWC Q/DQ mean matched")
    module.verify()
    rewritten = directory / "rewritten.mlir"
    rewritten.write_text(serialize(module, []))
    parse_module(rewritten.read_text()).verify()
    code = directory / "adapter.c"
    code.write_text(
        "\n".join(adapter_source(route, route["symbol"]) for route in routes)
    )
    adapter = directory / "mean_adapter.o"
    command = [
        str(llvm_bin / "clang"),
        "--target=riscv64-unknown-elf",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-ffreestanding",
        "-fno-builtin",
        "-c",
        str(code),
        "-o",
        str(adapter),
    ]
    subprocess.run(command, check=True, capture_output=True)
    obj = directory / "adapter.o"
    linker = llvm_bin / "ld.lld"
    if not linker.is_file():
        linker = Path(shutil.which("ld.lld") or "")
    if not linker.is_file():
        raise ValueError("ld.lld required")
    link = [str(linker), "-r", *(str(p) for p in objects), str(adapter), "-o", str(obj)]
    subprocess.run(link, check=True, capture_output=True)
    audit = audit_elf(obj.read_bytes())
    if audit["status"] != "pass":
        raise ValueError("integer mean object audit failed")
    native = directory / "native_oracle.c"
    native.write_text("\n".join(oracle) + code.read_text())
    record = {
        "schema": "device_integer_sum_mean_bundle_v1",
        "source_sha256": original,
        "rewritten_sha256": sha(rewritten),
        "original_golden_sha256": golden,
        "routes": routes,
        "object_sha256": sha(obj),
        "source_code_sha256": sha(code),
        "compiler_argv": command,
        "compiler_sha256": sha(llvm_bin / "clang"),
        "linker_argv": link,
        "linker_sha256": sha(linker),
        "object_audit": audit,
        "native_oracle": str(native),
        "native_oracle_sha256": sha(native),
        "default_enabled": False,
    }
    (directory / "mean.json").write_text(json.dumps(record, indent=2) + "\n")
    shutil.copyfile(rewritten, capture / "model.mlir")
    receipt["artifacts"]["model.mlir"] = {
        "bytes": (capture / "model.mlir").stat().st_size,
        "sha256": record["rewritten_sha256"],
    }
    receipt["guarded_mean_derivation"] = {
        "source_sha256": original,
        "bundle_sha256": sha(directory / "mean.json"),
        "original_golden_sha256": golden,
    }
    (capture / "capture_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert sha(capture / "golden.npy") == golden
    return record


def merlin_callbacks(llvm_bin, directory, base_callbacks):
    llvm_bin, directory = map(Path, (llvm_bin, directory))
    prepare_base, compile_base = base_callbacks
    record = json.loads((directory / "mean.json").read_text())
    manifest_pin = sha(directory / "mean.json")
    if record["schema"] != "device_integer_sum_mean_bundle_v1":
        raise ValueError("integer mean provider schema required")
    symbols = {row["symbol"]: row for row in record["routes"]}

    def check(source):
        if sha(directory / "mean.json") != manifest_pin:
            raise ValueError("integer mean manifest identity changed")
        module = parse_module(Path(source).read_text())
        module.verify()
        calls = [
            op
            for op in module.walk()
            if op.name == "func.call" and op.callee.root_reference.data in symbols
        ]
        declarations = {
            op.sym_name.data: op
            for op in module.walk()
            if op.name == "func.func" and op.sym_name.data in symbols
        }
        if sorted(op.callee.root_reference.data for op in calls) != sorted(
            symbols
        ) or set(declarations) != set(symbols):
            raise ValueError("integer mean call/declaration coverage changed")
        for symbol, declaration in declarations.items():
            route = symbols[symbol]
            plan = IntegerSumPlan(*route["output_shape"], route["proof"]["count"])
            if route["producer_contract"] != plan.proof():
                raise ValueError("integer mean producer resource contract changed")
            output_type = TensorType(i8, route["output_shape"])
            expected_inputs = (
                TensorType(i8, route["input_matrix_shape"]),
                TensorType(i32, route["output_shape"]),
                output_type,
            )
            if tuple(declaration.function_type.inputs) != expected_inputs or tuple(
                declaration.function_type.outputs
            ) != (output_type,):
                raise ValueError("integer mean declared type/extent ABI changed")
            expected = hashlib.sha256(
                json.dumps(route["proof"], sort_keys=True).encode()
            ).hexdigest()
            if (
                getattr(
                    declaration.attributes.get("merlin.guarded_mean_proof_sha256"),
                    "data",
                    None,
                )
                != expected
                or getattr(
                    declaration.attributes.get("merlin.guarded_mean_source_sha256"),
                    "data",
                    None,
                )
                != record["source_sha256"]
            ):
                raise ValueError("integer mean source/certificate binding changed")
            accesses = declaration.properties.get("arg_attrs")
            if accesses is None or [
                getattr(a.data.get("bufferization.access"), "data", None)
                for a in accesses
            ] != ["read", "write", "write"]:
                raise ValueError("integer mean effect ABI changed")
            call = next(op for op in calls if op.callee.root_reference.data == symbol)
            if len(call.operands) != 3 or any(
                not isinstance(v.owner, tensor.EmptyOp) or sum(1 for _ in v.uses) != 1
                for v in call.operands[1:]
            ):
                raise ValueError(
                    "integer mean requires distinct fresh private scratch/output"
                )
            if call.operands[1] is call.operands[2]:
                raise ValueError("integer mean writable operands alias")
            if any(call.operands[0] is value for value in call.operands[1:]):
                raise ValueError("integer mean input aliases a writable operand")

    def prepare(source, work):
        check(source)
        selected = prepare_base(source, work)
        check(selected)
        return selected

    def compile(source, work):
        check(source)
        for filename, digest in [
            ("adapter.o", record["object_sha256"]),
            ("adapter.c", record["source_code_sha256"]),
            ("native_oracle.c", record["native_oracle_sha256"]),
        ]:
            if sha(directory / filename) != digest:
                raise ValueError("integer mean artifact changed")
        manifest_path, old_object = compile_base(source, work)
        output = Path(work) / "integer_sum_mean_mixed.o"
        linker = Path(shutil.which("ld.lld") or "")
        if (llvm_bin / "ld.lld").is_file():
            linker = llvm_bin / "ld.lld"
        if not linker.is_file():
            raise ValueError("ld.lld required")
        command = [
            str(linker),
            "-r",
            str(old_object),
            str(directory / "adapter.o"),
            "-o",
            str(output),
        ]
        subprocess.run(command, check=True, capture_output=True)
        audit = audit_elf(output.read_bytes())
        if audit["status"] != "pass":
            raise ValueError("mixed integer mean object audit failed")
        manifest = json.loads(manifest_path.read_text())
        manifest["compilation"] = {
            "schema": "device_integer_mean_mixed_compile_v1",
            "object_sha256": sha(output),
            "object_nofsm_status": "pass",
            "remaining_compilation": manifest["compilation"],
            "mean_manifest_sha256": manifest_pin,
            "linker_argv": command,
            "linker_sha256": sha(linker),
        }
        manifest["guarded_mean_additions"] = record["routes"]
        manifest["native_oracle_sources"].append(str(directory / "native_oracle.c"))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest_path, output

    return prepare, compile
