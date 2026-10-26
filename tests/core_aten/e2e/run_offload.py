"""Whole-graph path: capture -> catalog -> offload rewrite -> standard MLIR lowering -> Spike.

Unlike ``run_e2e.py`` (which calls the kernels directly), this lowers the rewritten model with an
upstream ``mlir-opt`` pipeline, so the bound regions are reached through ``func.call`` and the
generated C-interface shim (``mlir_oot.core_aten.offload``) exactly as a host compiler would
reach them.  The model's own ``_mlir_ciface_forward`` is called from a bare-metal driver and its
result is compared with the exact-arithmetic reference. Eager-float divergence is informational.

    python run_offload.py CAPTURE_DIR [--only name ...] [--batch NAME]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "device"))
sys.path.insert(0, str(HERE.parent))

import runner  # noqa: E402
from exact_oracle import SourceOracle  # noqa: E402
from fixture_source import read  # noqa: E402
from mlir_oot.core_aten.catalog import merlin_builder  # noqa: E402
from mlir_oot.core_aten.emit import llvm_bin_dir  # noqa: E402
from mlir_oot.core_aten.offload import rewrite, shim_source  # noqa: E402
from mlir_oot.frontend.parse import parse_module  # noqa: E402

PIPELINE = ["--allow-unregistered-dialect", "--canonicalize",
            "--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map",
            "--convert-linalg-to-loops", "--expand-strided-metadata", "--lower-affine", "--convert-scf-to-cf",
            "--finalize-memref-to-llvm", "--convert-arith-to-llvm", "--convert-func-to-llvm", "--convert-cf-to-llvm",
            "--reconcile-unrealized-casts"]

RUNTIME = r"""
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
static uint8_t ca_heap[64u << 20] __attribute__((aligned(64)));
static size_t ca_top = 0;
void *malloc(size_t n) { size_t a = (ca_top + 63) & ~(size_t)63; ca_top = a + n; return ca_top <= sizeof ca_heap ? ca_heap + a : 0; }
void *aligned_alloc(size_t al, size_t n) { (void)al; return malloc(n); }
void free(void *p) { (void)p; }
void exit(int);
int puts(const char *s) { return printf("%s\n", s); }
void ca_abort(const char *why) { printf("FAIL abort %s\n", why); exit(1); }
"""


def _ctype(dt):
    return {"int8": "int8_t", "uint8": "uint8_t", "int32": "int32_t", "float32": "float"}[dt]


def build_case(case: Path, wd: Path, *, numeric_policy=None) -> Path:
    wd.mkdir(parents=True, exist_ok=True)
    source=read(case / "model.mlir")
    input_path=wd / "source.mlir"
    input_path.write_text(source+"\n")
    manifest_path, obj = merlin_builder(llvm_bin_dir(), numeric_policy=numeric_policy)(input_path, wd)
    manifest = json.loads(manifest_path.read_text())
    if obj is None:
        raise ValueError('Core ATen catalog admission declined: ' + manifest.get('reason', 'no exact kernels'))
    rewritten=rewrite(source,manifest)
    if 'quant_ext.' in rewritten:
        from mlir_oot.frontend.parse import context
        from xdsl.parser import Parser
        from m2m.ir.quant import Quant
        from m2m.transforms.expand_quant import expand_quant_to_standard
        ctx=context();ctx.load_dialect(Quant)
        rewritten=str(expand_quant_to_standard(Parser(ctx,rewritten).parse_module()))
    (wd / 'offloaded.mlir').write_text(rewritten+'\n')
    (wd / "shim.c").write_text(shim_source(manifest))
    original=parse_module(source)
    function=next(o for o in original.walk() if o.name=='func.func')
    arguments=list(function.body.blocks[0].args)
    inputs=dict(np.load(case/'inputs.npz'))
    reference=SourceOracle({v:np.asarray(inputs['arg'+str(i)],dtype=object) for i,v in enumerate(arguments)})
    returned=next(o for o in function.walk() if o.name=='func.return')
    exact_outputs=[]
    for value in returned.operands:
        dtype=str(value.type.get_element_type())
        array=reference.tensor(value)
        if dtype=='f32': array=np.asarray(array,dtype=np.float32)
        else: array=np.asarray(array,dtype=np.int64).astype(np.int32 if dtype=='i32' else np.int8)
        exact_outputs.append(array)
    bindir = llvm_bin_dir()
    subprocess.run([str(bindir / "mlir-opt"), str(wd / "offloaded.mlir"), *PIPELINE, "-o", str(wd / "host.mlir")],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(bindir / "mlir-translate"), "--mlir-to-llvmir", str(wd / "host.mlir"), "-o",
                    str(wd / "host.ll")], check=True, capture_output=True, text=True)
    subprocess.run([str(bindir / "llc"), "-mtriple=riscv64-unknown-elf", "-mattr=+m,+a,+f,+d,+c", "-target-abi=lp64d",
                    "-code-model=medium", "-O2", "-filetype=obj", str(wd / "host.ll"), "-o", str(wd / "host.o")],
                   check=True, capture_output=True, text=True)
    exps = exact_outputs
    names = sorted(inputs, key=lambda k: int(k[3:]))
    c = ['#include "ca_harness.h"', RUNTIME]
    asm = [".section .data"]
    blobs = {nm: inputs[nm] for nm in names} | {f"expected{k}": e for k, e in enumerate(exps)}
    for nm, arr in blobs.items():
        (wd / f"{nm}.bin").write_bytes(np.ascontiguousarray(arr).tobytes() or b"\0")
        asm += [f".global ca_{nm}", ".balign 64", f"ca_{nm}:", f'.incbin "{wd / (nm + ".bin")}"']
        c.append(f"extern {_ctype(str(arr.dtype))} ca_{nm}[];")
    ranks = sorted({inputs[n].ndim for n in names} | {e.ndim for e in exps})
    for r in ranks:
        c.append(f"typedef struct {{ void *allocated; void *aligned; int64_t offset; int64_t sizes[{r}]; "
                 f"int64_t strides[{r}]; }} md{r};")
    if len(exps) == 1:
        rtype = f"md{exps[0].ndim}"
    else:
        c.append("typedef struct { " + " ".join(f"md{e.ndim} r{k};" for k, e in enumerate(exps)) + " } results_t;")
        rtype = "results_t"
    c.append(f"extern void _mlir_ciface_forward({rtype} *res, "
             + ", ".join(f"md{inputs[n].ndim} *" for n in names) + ");")
    c.append("int main(void) {")
    for n in names:
        a = inputs[n]
        st = [int(np.prod(a.shape[i + 1:])) for i in range(a.ndim)]
        c.append(f"  md{a.ndim} d_{n} = {{ ca_{n}, ca_{n}, 0, {{{', '.join(map(str, a.shape))}}}, {{{', '.join(map(str, st))}}} }};")
    c.append(f"  {rtype} res;")
    c.append("  uint64_t t0 = ca_rdcycle();")
    c.append("  _mlir_ciface_forward(&res, " + ", ".join(f"&d_{n}" for n in names) + ");")
    c.append("  ca_fence();")
    c.append(f'  printf("PERF {case.name} cycles=%lu\\n", (unsigned long)(ca_rdcycle() - t0));')
    for k, e in enumerate(exps):
        field = "res" if len(exps) == 1 else f"res.r{k}"
        tag = case.name if len(exps) == 1 else f"{case.name}.out{k}"
        c.append(f'  ca_check_{"i8" if e.dtype == np.int8 else "i32"}("{tag}_offload_vs_exact_arithmetic", '
                 f"({"int8_t" if e.dtype == np.int8 else "int32_t"} *){field}.aligned + {field}.offset, "
                 f"(const {"int8_t" if e.dtype == np.int8 else "int32_t"} *)ca_expected{k}, {e.size}, 0);")
    c.append("  return ca_finish();")
    c.append("}")
    (wd / "main.c").write_text("\n".join(c) + "\n")
    (wd / "blobs.S").write_text("\n".join(asm) + "\n")
    elf = runner.link(wd / f"{case.name}.elf", [wd / f for f in ("main.c", "shim.c", "blobs.S", "host.o", "core_aten.o")],
                      ["-fno-builtin-malloc"])
    from mlir_oot.golden_device_catalog import final_elf_audit
    final_elf_audit(elf)
    return elf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--batch", default="offload")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--numeric-policy", type=Path)
    a = ap.parse_args()
    policy = None
    if a.numeric_policy is not None:
        from merlin.targetgen.software_spec import load_software_spec
        policy = load_software_spec(a.numeric_policy, target='gemmini')
    cases = sorted(p for p in a.capture.iterdir() if (p / "model.mlir").exists())
    if a.only:
        cases = [c for c in cases if c.name in a.only]
    rows = []
    for case in cases:
        try:
            elf = build_case(case, runner.work_root() / a.batch / case.name, numeric_policy=policy)
            r = runner.run_spike(elf)
            rows.append(dict(case=case.name, status=r.status, info=r.info, failed=r.failed, perf=r.perf,
                             tail=None if r.ok else r.log[-1500:]))
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(case=case.name, status="ERROR", error=f"{type(exc).__name__}: {str(exc)[-1500:]}"))
        print(f"{case.name:32s} {rows[-1]['status']} {rows[-1].get('info') or rows[-1].get('error', '')}", flush=True)
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1) + "\n")
    return 0 if all(r["status"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
