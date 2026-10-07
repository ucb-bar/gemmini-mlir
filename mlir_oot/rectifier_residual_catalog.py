"""Explicit finite-source rectifier alternative in the existing residual route.

Uses the existing joint catalog's source/type/writer closure and callbacks.
Certificates select complete source relations; route names only bind emitted
symbols. No default numerical, routing or profitability policy changes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from merlin.llvmlower.quantized_affine_rectifier import validate
from xdsl.dialects.builtin import StringAttr

from .captured_residual_bundle import canonical, compile_adapter, sha
from .golden_device_compile import compile_module
from .golden_key_rectified_resadd import Capabilities as KeyCapabilities
from .golden_key_rectified_resadd import Plan as KeyPlan
from .golden_key_rectified_resadd import build as emit_key
from .golden_rectified_resadd import Capabilities, Plan
from .golden_rectified_resadd import build as emit
from .joint_residual_catalog import _identifier, _validate_record
from .no_fsm_audit import audit_elf
from .spad_fence_coalescing import OrderingContract


def _key_family(route):
    value = route.get("identity_i32_acc_dma_accumulate", False)
    if type(value) is not bool:
        raise ValueError("explicit identity ACC DMA RMW capability boolean required")
    key = route["proof"]["indicator_family"] == "predictor_key"
    if key and value is not True:
        raise ValueError("predictor-key family requires identity ACC DMA RMW")
    return key


def _plan(route):
    if _key_family(route):
        return KeyPlan(
            route["m"],
            route["n"],
            route["proof"],
            KeyCapabilities(Capabilities(**route["capabilities"]), True),
            panel_batch=route.get("panel_batch", 1),
        )
    return Plan(
        route["m"], route["n"], route["proof"], Capabilities(**route["capabilities"])
    )


def kernel(route):
    flag = route["coalesce_internal_spad"]
    panel_batch = route.get("panel_batch", 1)
    if type(flag) is not bool:
        raise ValueError("explicit SPAD coalescing boolean required")
    if _key_family(route):
        if flag is not True or not route["ordering_source"]:
            raise ValueError(
                "predictor-key family requires its pinned SPAD completion proof"
            )
        return emit_key(
            _plan(route),
            ordering_contract=OrderingContract(route["ordering_source"]),
        )
    return emit(
        _plan(route),
        coalesce_internal_spad=flag,
        ordering_contract=OrderingContract(route["ordering_source"])
        if flag or panel_batch == 4
        else None,
        panel_batch=panel_batch,
    )


def adapter(route):
    plan = _plan(route)
    symbol, function, count = route["symbol"], route["kernel"], plan.m * plan.n
    values = ",".join(str(x if x < 128 else x - 256) for x in plan.tables())
    if _key_family(route):
        extra_parameter = ",int8_t*"
        workspace = (
            f"int8_t scratch[{plan.scratch_bytes}] __attribute__((aligned(64)));"
        )
        workspace += (
            f"if({symbol}_overlap(pa,scratch,sizeof(scratch))||"
            f"{symbol}_overlap(pb,scratch,sizeof(scratch))||"
            f"{symbol}_overlap(pc,scratch,sizeof(scratch)))__builtin_trap();"
        )
        extra_argument = ",scratch"
    else:
        extra_parameter = workspace = extra_argument = ""
    return f"""#include <stdint.h>
#ifndef GEMMINI_RECTIFIER_RESIDUAL_ABI
#define GEMMINI_RECTIFIER_RESIDUAL_ABI
typedef struct{{void *allocated,*aligned;intptr_t offset,sizes[2],strides[2];}} rectifier_memref2;
#endif
static const int8_t {symbol}_tables[{len(plan.tables())}] __attribute__((aligned(64)))={{{values}}};
extern void {function}(const int8_t*,const int8_t*,int8_t*,const int8_t*{extra_parameter});
static int8_t *{symbol}_pointer(rectifier_memref2*d){{
 if(!d->aligned||d->offset<0||d->sizes[0]!={plan.m}||d->sizes[1]!={plan.n}||d->strides[0]!={plan.n}||d->strides[1]!=1)__builtin_trap();
 uintptr_t base=(uintptr_t)d->aligned,off=(uintptr_t)d->offset;
 if(base>UINTPTR_MAX-off||base+off>UINTPTR_MAX-{count})__builtin_trap();
 return(int8_t*)(base+off);
}}
static int {symbol}_overlap(const int8_t*a,const int8_t*b,uintptr_t bsize){{
 uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;return x<=y?y-x<{count}:x-y<bsize;
}}
void _mlir_ciface_{symbol}(rectifier_memref2*r,rectifier_memref2*a,rectifier_memref2*b,rectifier_memref2*c){{
 const int8_t*pa={symbol}_pointer(a),*pb={symbol}_pointer(b);int8_t*pc={symbol}_pointer(c);
 if({symbol}_overlap(pc,pa,{count})||{symbol}_overlap(pc,pb,{count})||{symbol}_overlap(pc,{symbol}_tables,{len(plan.tables())}))__builtin_trap();
 {workspace}{function}(pa,pb,pc,{symbol}_tables{extra_argument});*r=*c;
}}
"""


def oracle(route):
    """Simulate the actual certified bounded device arithmetic, not a lookup."""
    plan = _plan(route)
    predictor = plan.certificate["predictor"]
    low = 0 if plan.certificate["source"]["relu"] else -128
    stages = []
    if _key_family(route):
        key = plan.relation["key"]
        stages.append(
            f"int32_t key=acc+{key['seed']};"
            "int e=key==0;"
            f"out+={plan.relation['correction']}*e;"
        )
    elif plan.relation:
        stages.append("int e=1;")
        for offset in plan.relation["offsets"]:
            operand = "a[i]" if offset["axis"] == "lhs" else "b[i]"
            stages.append(
                f"part=(int){operand}*{offset['coefficient']}+{offset['seed']};"
                "if(part<0)part=0;if(part>127)part=127;"
                "e-=part;if(e<0)e=0;if(e>127)e=127;"
            )
        stages.append(f"out+={plan.relation['correction']}*e;")
    extra_parameter = ",int8_t*scratch" if _key_family(route) else ""
    extra_void = "(void)scratch;" if _key_family(route) else ""
    return f"""#include <math.h>
void {route["kernel"]}(const int8_t*a,const int8_t*b,int8_t*c,const int8_t*tables{extra_parameter}){{
 (void)tables;{extra_void}
 for(intptr_t i=0;i<{plan.m * plan.n};i++){{
  int32_t acc=(int32_t)a[i]*{predictor["p"]}+(int32_t)b[i]*{predictor["q"]};
  float predicted=nearbyintf((float)acc*{float(predictor["scale"]).hex()}f);
  if(predicted<{low})predicted={low};if(predicted>127)predicted=127;
  int out=(int)predicted,part;
  {"".join(stages)}
  if(out<{low})out={low};if(out>127)out=127;c[i]=(int8_t)out;
 }}
}}
"""


def build(
    original_bundle,
    certificates,
    llvm_bin,
    output,
    *,
    capabilities,
    coalesce_internal_spad=False,
    ordering_source=None,
    panel_batch=1,
    identity_i32_acc_dma_accumulate=False,
):
    if type(identity_i32_acc_dma_accumulate) is not bool:
        raise ValueError("explicit identity ACC DMA RMW capability boolean required")
    if type(coalesce_internal_spad) is not bool:
        raise ValueError("explicit SPAD coalescing boolean required")
    if type(panel_batch) is not int or panel_batch not in (1, 4):
        raise ValueError("explicit panel batch 1 or proved factor 4 required")
    capabilities.require()
    if coalesce_internal_spad or panel_batch == 4:
        if ordering_source is None:
            raise ValueError("pinned SPAD ordering source required")
        OrderingContract(str(ordering_source)).require()
    original_bundle, llvm_bin, output = map(Path, (original_bundle, llvm_bin, output))
    original = json.loads((original_bundle / "residual.json").read_text())
    if original["numeric_policy"]["max_output_lsb"] != 0:
        raise ValueError("original complete exact source policy required")
    proofs = [validate(json.loads(Path(path).read_text())) for path in certificates]
    sources = [canonical(p["source"]) for p in proofs]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("one certificate per explicit source relation required")
    selected = dict(zip(sources, proofs, strict=True))
    routes = []
    for old in original["routes"]:
        proof = selected.get(canonical(old["proof"]["source"]))
        if proof is None:
            continue
        symbol = _identifier(old["symbol"] + "__rectifier")
        route = {
            "original_symbol": old["symbol"],
            "symbol": symbol,
            "kernel": symbol + "_kernel",
            "m": old["m"],
            "n": old["n"],
            "original_route": old,
            "proof": proof,
            "proof_sha256": sha(
                certificates[sources.index(canonical(proof["source"]))]
            ),
            "result_argument": 2,
            "fully_written_arguments": [2],
            "rectifier_pipeline": True,
            "capabilities": vars(capabilities),
            "coalesce_internal_spad": coalesce_internal_spad,
            "ordering_source": str(Path(ordering_source).resolve())
            if ordering_source
            else None,
        }
        if proof["indicator_family"] == "predictor_key":
            route["identity_i32_acc_dma_accumulate"] = identity_i32_acc_dma_accumulate
        if panel_batch != 1:
            route["panel_batch"] = panel_batch
        kernel(route)
        routes.append(route)
    if {canonical(r["proof"]["source"]) for r in routes} != set(sources):
        raise ValueError("explicit source relation absent from bound catalog")
    record = {"schema": "gemmini_source_bound_rectifier_residual_v1", "routes": routes}
    _validate_record(original, record)
    output.mkdir(parents=True, exist_ok=False)
    objects, native = [], []
    for route in routes:
        directory = output / route["symbol"]
        module = kernel(route)
        module.body.block.first_op.properties["sym_name"] = StringAttr(route["kernel"])
        route["compilation"] = compile_module(module, llvm_bin, directory)
        code = adapter(route)
        source = directory / "adapter.c"
        source.write_text(code)
        route["adapter_compilation"] = compile_adapter(
            source, directory / "adapter.o", llvm_bin
        )
        objects.extend([directory / "kernel.o", directory / "adapter.o"])
        native.append(code + oracle(route))
    linker = llvm_bin / "ld.lld"
    if not linker.is_file():
        found = shutil.which("ld.lld")
        if found is None:
            raise ValueError("required target partial linker is unavailable")
        linker = Path(found)
    argv = [
        str(linker),
        "-r",
        *map(str, objects),
        "-o",
        str(output / "joint.o"),
    ]
    subprocess.run(argv, check=True, capture_output=True)
    audit = audit_elf((output / "joint.o").read_bytes())
    if audit["status"] != "pass":
        raise ValueError("rectifier catalog uses a forbidden target instruction")
    (output / "native_oracle.c").write_text("\n".join(native))
    record.update(
        original_manifest_path=str((original_bundle / "residual.json").resolve()),
        original_manifest_sha256=sha(original_bundle / "residual.json"),
        certificates=[
            {"path": str(Path(p).resolve()), "sha256": sha(p)} for p in certificates
        ],
        first_output_guard=False,
        object_sha256=sha(output / "joint.o"),
        native_oracle_sha256=sha(output / "native_oracle.c"),
        nofsm_audit=audit,
        linker_argv=argv,
        linker_sha256=sha(linker),
        numeric_policy=original["numeric_policy"],
        default_selection_enabled=False,
        target_cycle_cost="UNKNOWN until complete matched runtime and whole gates",
        workspace=(
            "private fresh C and bounded adapter-owned aligned scratch; immutable A/B/tables; "
            "completion before private scratch return; no retained pointer or CPU correction"
            if any(_key_family(route) for route in routes)
            else "private fresh C; immutable A/B/tables; no second writer or CPU correction"
        ),
    )
    (output / "joint.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
