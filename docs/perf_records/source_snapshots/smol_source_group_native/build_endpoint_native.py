"""Native-only whole source-bound group policy screen via normal preparation."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.closed_group_writer import (
    ClosedGroupWriterContract, install_closed_group_writers,
    source_function_semantic_sha256,
)
from merlin.llvmlower.device_build import DeviceRouting
from merlin.llvmlower.ordered_bf16_group_binding import SourceExactGroupPreparation
from merlin.llvmlower.ordered_fma_group_outline import outline_ordered_fma_group
from merlin.llvmlower.ordered_fma_groups import analyze_ordered_fma_groups
from merlin.llvmlower.splat_inputs import scalarize_splat_inputs
from merlin.runtime.backends import spike_model
from merlin.xdsl_dialects._common import text
from mlir_oot.golden_device_catalog import merlin_builder, final_elf_audit

w = Path(__file__).resolve().parent
out = w / "bounded_native"
out.mkdir(exist_ok=True)
source = Path("/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/smol_source_sum_fma_bundle")
sha = lambda p: hashlib.file_digest(Path(p).open("rb"), "sha256").hexdigest()
assert sha(source / "model.mlir") == "4814509b8e11a5c819b1f9ae63f01f89f72e9edf9c3d0ec9d5cd0ab6b35de2dc"
native_path = Path("/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/closed_group_endpoint/native_evaluator.py")
spec = importlib.util.spec_from_file_location("native_endpoint_evaluator", native_path)
numeric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numeric)
reference = parse_mlir_text((w / "source_group_exact.mlir").read_text())
reference_function = next(op for op in reference.body.block.ops if op.name == "func.func")
raw_semantic_pin = source_function_semantic_sha256(reference_function)
# Normal preparation keeps typed uniform operands rank zero. Apply its existing
# exact transformation to the source reference, then repartition the same live
# DAG to localize precisely the resulting source constants and unread inits.
uniform_changes = scalarize_splat_inputs(reference)
reference_group, = analyze_ordered_fma_groups(reference)
reference_function = outline_ordered_fma_group(reference, reference_group,
    "normalized_source_group_reference").function
semantic_pin = source_function_semantic_sha256(reference_function)
from xdsl.dialects.builtin import ModuleOp, StringAttr, UnitAttr
standalone = reference_function.clone()
standalone.properties["sym_name"] = StringAttr("source_group_normalized")
standalone.properties["sym_visibility"] = StringAttr("public")
standalone.attributes["llvm.emit_c_interface"] = UnitAttr()
(out / "source_group_normalized.mlir").write_text(text(ModuleOp([standalone])))
header = Path("/scratch/agustin/tmp/merlin-closed-bf16-certificate-20261005/merlin/runtime/c/f32_interval_endpoint.h")
numeric_witness = dict(source_semantic_sha256=semantic_pin,
    raw_source_semantic_sha256=raw_semantic_pin,
    source_reference_sha256=sha(w / "source_group_exact.mlir"),
    normalized_source_reference_sha256=sha(out / "source_group_normalized.mlir"),
    exact_uniform_operand_canonicalization_count=uniform_changes,
    source_contract=numeric.SOURCE_CONTRACT, native_evaluator_sha256=sha(native_path),
    numeric_source_sha256=sha(native_path.parent / "screen_bounded.c"),
    interval_header_sha256=sha(header), max_bf16_steps=1,
    qualification="native functional exact-product stand-in; not a device implementation")
numeric_pin = hashlib.sha256(json.dumps(numeric_witness, sort_keys=True).encode()).hexdigest()
provider = "native_bf16_endpoint_" + numeric_pin[:16]
bridge = """#include <stdint.h>
#include <stdlib.h>
typedef struct { void *allocated,*aligned; int64_t offset,size[4],stride[4]; } desc4;
typedef int (*callback_type)(void **);
static callback_type callback;
void native_endpoint_register(callback_type value) { callback=value; }
"""
args = ",".join("desc4 *a" + str(i) for i in range(12))
pointers = ",".join("a" + str(i) for i in range(12))
bridge += f"void _mlir_ciface_{provider}_borrowed({args}) {{\n"
bridge += f" void *descriptors[12]={{{pointers}}};\n"
bridge += " if(!callback || callback(descriptors)) abort();\n}\n"
(out / "native_bridge.c").write_text(bridge)
contract = ClosedGroupWriterContract(semantic_pin, provider, sha(out / "native_bridge.c"),
    numeric_pin, "bounded_adjacent_bf16_endpoint_max_steps_1", sha(out / "native_bridge.c"),
    64, True, True, True, True, "rne_returned_values", "native_functional_screen")
(out / "numeric_witness.json").write_text(json.dumps(numeric_witness, indent=2) + "\n")

def prepare(source_path, workdir):
    source_control = SourceExactGroupPreparation()
    prior = Path(workdir) / "source_control/source_group_calls.json"
    cached = json.loads(prior.read_text()) if prior.is_file() else None
    if (cached is not None and cached["original_prepared_sha256"] == sha(source_path)
            and Path(cached["selected_path"]).is_file()
            and cached["selected_sha256"] == sha(cached["selected_path"])):
        # An exact-content preparation cache; the live module/body/context/call
        # witnesses are still verified below before any writer installation.
        source_control.receipt = cached
        exact = Path(cached["selected_path"])
        print("REUSED_EXACT_SOURCE_PREPARATION", cached["original_prepared_sha256"], flush=True)
    else:
        exact = source_control(source_path, Path(workdir) / "source_control")
    module = parse_mlir_text(exact.read_text())
    functions = {op.sym_name.data: op for op in module.body.block.ops if op.name == "func.func"}
    records = source_control.receipt["records"]
    selected = {record["symbol"]: contract for record in records
                if source_function_semantic_sha256(functions[record["symbol"]]) == semantic_pin}
    # Scope assertion for this pinned full-model experiment; not a production selector.
    if len(records) != 48 or len(selected) != 48:
        pins = {record["symbol"]: source_function_semantic_sha256(functions[record["symbol"]])
                for record in records}
        (out / "refused_source_semantics.json").write_text(json.dumps(dict(reference=semantic_pin, pins=pins), indent=2))
        raise ValueError("native complete-source grammar does not cover every expected group")
    installation = install_closed_group_writers(module, source_control.receipt, selected)
    module.verify()
    target = Path(workdir) / "bound_endpoint.mlir"
    target.write_text(text(module))
    installation.update(selected_sha256=sha(target), original_prepared_sha256=sha(source_path),
        source_control_receipt=source_control.receipt, contract=asdict(contract),
        native_numeric_witness=numeric_witness, provider_symbol=provider,
        original_source_sha256=sha(source / "model.mlir"),
        scope="Normal source preparation + fresh borrowed native callback; all48groups; target implementation unavailable")
    (out / "installation.json").write_text(json.dumps(installation, indent=2) + "\n")
    print("ENDPOINT_GROUPS_BOUND", len(selected), semantic_pin, flush=True)
    return target

llvm = Path("/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin")
device = DeviceRouting(device="gemmini", package_dir="/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005",
    operand_dtype="i8", accum_dtype="i32", catalog_builder=merlin_builder(llvm),
    final_elf_audit=final_elf_audit, prepared_transform=prepare)
original_run = spike_model._run

def native_only(cmd, **kwargs):
    args = list(map(str, cmd))
    if ("-c" in args and args[args.index("-c") + 1].endswith(".ll")
            and "-o" in args and args[args.index("-o") + 1].endswith("/model.o")):
        (out / "native_ready.json").write_text(json.dumps({"target_command_deferred": args}, indent=2))
        print("NATIVE_READY", flush=True)
        raise SystemExit(0)
    return original_run(cmd, **kwargs)

spike_model._run = native_only
spike_model.build(source, out / "build", arena_mb=8192, stack_bytes=16*1024**2,
    output_dump_cap=1600, dram_bytes=16*1024**3, int8_compute=True,
    features=frozenset({"respect_captured_quantization_scope", "lower_fma_to_intrinsic", "outline_llvm_loops"}),
    cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2",
        "-ffreestanding", "-fno-builtin", "-fno-vectorize", "-fno-slp-vectorize"],
    device=device, host_math_policy="expf_via_double")
