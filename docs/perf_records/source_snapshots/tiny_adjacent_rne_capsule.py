"""Fixed source pair, original boundary oracle, five RV FP modes, actual GSIM."""
from dataclasses import asdict
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.late_quant_rne import rewrite as scalar
from merlin.llvmlower.bounded_rne_basic_block import rewrite as packet
from merlin.perf.layer_bench import build_program, run_on_gsim
from mlir_oot.no_fsm_audit import audit_elf

root = Path.cwd()
base = root / "out/artifacts/probes/tiny-pointwise-packet/capsule"
work = root / "out/artifacts/probes/tiny-rne-basic-block/capsule"
work.mkdir(parents=True, exist_ok=True)
llvm = Path("/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin")
flags = ["--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany",
         "-O3", "-ffreestanding", "-fno-builtin", "-ffp-contract=off"]
objects = []
source = (base / "packet2/model.ll").read_text()
expected = np.fromfile(base / "expected.bin", np.int8)
inputs = [np.fromfile(base / f"input{i}.bin", np.int32 if i < 2 else np.float32) for i in range(4)]
assert len(expected) == 2049 and all(len(x) == 2049 for x in inputs)
receipts = []
for name in ("control", "grouped"):
    text = source.replace("forward", name).replace("dealloc_helper", name + "_dealloc_helper")
    target, proof = packet(text, host_isa="rv64gc") if name == "grouped" else (text, {"routes": []})
    native, _ = packet(text, host_isa="portable") if name == "grouped" else (text, {})
    target, remaining = scalar(target, host_isa="rv64gc", combine_clamp=True)
    native, _ = scalar(native, host_isa="portable")
    if name == "grouped": assert len(proof["routes"]) == 1 and proof["routes"][0]["lanes"] == 2
    path = work / (name + ".ll"); path.write_text(target)
    npth = work / (name + ".native.ll"); npth.write_text(native)
    so = work / (name + ".so")
    subprocess.run([str(llvm / "clang"), "-O3", "-fPIC", "-shared", str(npth), str(mlir_runtime_c()), "-lm", "-o", str(so)], check=True)
    output = np.zeros(2049, np.int8)
    HostModel.load(str(so), name=name)([(x.ctypes.data, x.shape) for x in inputs] + [(output.ctypes.data, output.shape)])
    assert np.array_equal(output, expected)
    obj = work / (name + ".o")
    subprocess.run([str(llvm / "clang"), *flags, "-c", str(path), "-o", str(obj)], check=True)
    objects.append(obj)
    receipts.append({"name": name, "native2049_exact": True, "packet": proof,
                     "remaining_scalar": remaining, "llvm_sha256": hashlib.sha256(target.encode()).hexdigest()})

fixture = Path("/scratch/agustin/tmp/merlin-tiny-rne-basic-block-20261005/merlin/tests/ir/test_bounded_rne_basic_block.py")
spec = importlib.util.spec_from_file_location("fixture", fixture); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
for name in ("source_quant", "scalar_quant", "group_quant"):
    text = module.source(4).replace("@quant", "@" + name)
    if name == "group_quant": text, _ = packet(text, host_isa="rv64gc", width=4)
    if name != "source_quant": text, _ = scalar(text, host_isa="rv64gc", combine_clamp=True)
    path = work / (name + ".ll"); path.write_text(text); obj = path.with_suffix(".o")
    subprocess.run([str(llvm / "clang"), *flags, "-c", str(path), "-o", str(obj)], check=True)
    objects.append(obj)
halves = np.arange(-130.5, 130.5, dtype=np.float32)
boundary = np.concatenate([halves, np.nextafter(halves, np.float32(np.inf)),
    np.nextafter(halves, np.float32(-np.inf)), np.array([0., -0., 1e-40, -1e-40, np.inf, -np.inf], np.float32)])
boundary = np.pad(boundary, (0, (-len(boundary)) % 4)).astype(np.float32)
oracle = np.rint(np.clip(boundary, -128, 127)).astype(np.int8)
boundary.tofile(work / "boundary.bin"); oracle.tofile(work / "boundary_oracle.bin")
assembly = ".section .data\n"
for symbol, path in [*( (f"input{i}", base / f"input{i}.bin") for i in range(4)),
                     ("expected", base / "expected.bin"), ("boundary", work / "boundary.bin"),
                     ("boundary_oracle", work / "boundary_oracle.bin")]:
    assembly += f'.balign 64\n.global {symbol}\n{symbol}:\n.incbin "{path}"\n'
(work / "data.S").write_text(assembly)
subprocess.run([str(llvm / "clang"), *flags, "-c", str(work / "data.S"), "-o", str(work / "data.o")], check=True)
objects.append(work / "data.o")
c = r'''#include <stdint.h>
extern int printf(const char*,...);
#define N 2049
extern int32_t input0[],input1[];extern float input2[],input3[],boundary[];
extern int8_t expected[],boundary_oracle[];
struct d{void*a,*p;long off,size,stride;};
extern void _mlir_ciface_control(struct d*,struct d*,struct d*,struct d*,struct d*);
extern void _mlir_ciface_grouped(struct d*,struct d*,struct d*,struct d*,struct d*);
extern void source_quant(float*,int8_t*),scalar_quant(float*,int8_t*),group_quant(float*,int8_t*);
static int8_t out[N+128];static uint8_t arena[4*N+1024];static unsigned cursor;
void *malloc(unsigned long size){cursor=(cursor+63)&~63u;if(cursor+size>sizeof(arena))return 0;void*p=arena+cursor;cursor+=size;return p;}
void free(void*p){(void)p;}
static inline uint64_t tick(void){uint64_t t;asm volatile("csrr %0,mcycle":"=r"(t)::"memory");return t;}
int main(void){int8_t q[4];
for(unsigned mode=0;mode<5;mode++){
asm volatile("csrw frm,%0"::"r"(mode):"memory");
for(unsigned i=0;i<BOUNDARY;i+=4){
source_quant(boundary+i,q);for(unsigned j=0;j<4;j++)if(q[j]!=boundary_oracle[i+j]){printf("SOURCE_MODE_FAIL %u %u\n",mode,i+j);return 1;}
scalar_quant(boundary+i,q);for(unsigned j=0;j<4;j++)if(q[j]!=boundary_oracle[i+j]){printf("SCALAR_MODE_FAIL %u %u\n",mode,i+j);return 2;}
group_quant(boundary+i,q);for(unsigned j=0;j<4;j++)if(q[j]!=boundary_oracle[i+j]){printf("PACKET_MODE_FAIL %u %u\n",mode,i+j);return 3;}
}printf("RNE_MODE_PASS %u\n",mode);}
asm volatile("csrw frm,zero":::"memory");
struct d a={input0,input0,0,N,1},b={input1,input1,0,N,1},sa={input2,input2,0,N,1},sb={input3,input3,0,N,1},o={out,out,0,N,1};
for(unsigned i=N;i<N+128;i++)out[i]=73;
for(unsigned round=0;round<2;round++){
cursor=0;uint64_t t=tick();_mlir_ciface_control(&a,&b,&sa,&sb,&o);t=tick()-t;
for(unsigned i=0;i<N;i++)if(out[i]!=expected[i]){printf("CONTROL_FAIL %u\n",i);return 4;}
printf("CONTROL_CYCLES %u %lu\n",round,t);
cursor=0;t=tick();_mlir_ciface_grouped(&a,&b,&sa,&sb,&o);t=tick()-t;
for(unsigned i=0;i<N;i++)if(out[i]!=expected[i]){printf("GROUPED_FAIL %u\n",i);return 5;}
for(unsigned i=N;i<N+128;i++)if(out[i]!=73){printf("GUARD_FAIL\n");return 6;}
printf("GROUPED_CYCLES %u %lu\n",round,t);
}printf("RNE_PACKET PASS\n");return 0;}
'''.replace("BOUNDARY", str(len(boundary)))
(work / "main.c").write_text(c)
subprocess.run([str(llvm / "clang"), *flags, "-c", str(work / "main.c"), "-o", str(work / "main.o")], check=True)
objects.append(work / "main.o")
built = build_program(objects, work / "build", target="gemmini", max_loaded_bytes=None)
audit = audit_elf(built.elf.read_bytes()); assert audit["status"] == "pass"
spike = subprocess.run(["/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike", "--extension=gemmini", "--isa=rv64gc", str(built.elf)], capture_output=True, text=True, timeout=240)
console = spike.stdout + spike.stderr; (work / "spike.log").write_text(console)
assert spike.returncode == 0 and "RNE_PACKET PASS" in console and console.count("RNE_MODE_PASS") == 5
proof = {"schema": "rne_packet_capsule_v1", "elf_sha256": built.elf_sha256,
         "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "cases": receipts,
         "audit": audit, "five_actual_RV_rounding_modes_pass": True,
         "original_boundary_words": len(boundary), "all2049_outputs_and128_guard_bytes": True,
         "token_usage_available": False, "scope": "Actual2049sourcearithmetic capsule; not whole-model performance",
         "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()}
(work / "qualification.json").write_text(json.dumps(proof, indent=2) + "\n")
print("STRICT_SPIKE_PASS", console, flush=True)
run = run_on_gsim(built.elf, target="gemmini", timeout_s=1800, max_cycles=3000000, stdout_path=work / "gsim.log")
(work / "gsim_receipt.json").write_text(json.dumps(asdict(run), indent=2, default=str) + "\n")
print("GSIM", run, flush=True)
