"""Execute every admitted pair for explicit source-domain residual choices.

This is a target arithmetic/guard qualification, not a whole-model performance
prediction. The original ordered scalar source supplies every expected byte.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from merlin.llvmlower import quantized_affine_domain as domain
from merlin.llvmlower.quantized_affine_pair import source_table
from merlin.perf.layer_bench import build_program, run_on_gsim
from xdsl.dialects.builtin import StringAttr

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_wide_resadd import build, tables
from mlir_oot.no_fsm_audit import audit_elf


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--spike", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    record = json.loads(args.manifest.read_text())
    selected = record["source_domain_derivation"]["changed"]
    sources = ["#include <stdint.h>", "#include <stdio.h>", "#include <string.h>"]
    objects, cases = [], []
    for index, choice in enumerate(selected):
        certificate = domain.validate(choice["choice"]["certificate"])
        bounds = certificate["operand_intervals"]
        av = np.arange(bounds[0][0], bounds[0][1] + 1, dtype=np.int16)
        bv = np.arange(bounds[1][0], bounds[1][1] + 1, dtype=np.int16)
        a = np.repeat(av, len(bv)).astype(np.int8)
        b = np.tile(bv, len(av)).astype(np.int8)
        count = len(a)
        if count % 1024:
            raise ValueError("complete admitted domain must fit whole 16x64 panels")
        expected = source_table(**certificate["source"])[
            a.astype(np.int16) + 128, b.astype(np.int16) + 128
        ].astype(np.int8)
        coeff = certificate["predictor"]
        target = build(
            count // 64,
            coeff["p"],
            coeff["q"],
            coeff["scale"],
            relu=certificate["source"]["relu"],
            prefetch_m=True,
            banked_accumulators=True,
        )
        symbol = f"admitted_residual_{index}"
        target.body.block.first_op.properties["sym_name"] = StringAttr(symbol)
        directory = work / f"case{index}"
        compilation = compile_module(target, args.llvm_bin, directory)
        objects.append(directory / "kernel.o")
        for name, array in (
            ("a", a),
            ("b", b),
            ("expected", expected),
            ("coeff", np.frombuffer(tables(coeff["p"], coeff["q"]), dtype=np.int8)),
        ):
            (directory / f"{name}.i8").write_bytes(array.tobytes())
            sources.append(
                f"static const int8_t {name}{index}[{len(array)}] __attribute__((aligned(64)))={{"
                + ",".join(map(str, array.tolist()))
                + "};"
            )
        sources.append(
            f"extern void {symbol}(const int8_t*,const int8_t*,int8_t*,const int8_t*);"
        )
        sources.append(
            f"static struct {{uint8_t pre[2048];int8_t values[{count}];uint8_t post[2048];}} aa{index},bb{index},cc{index} __attribute__((aligned(64)));"
        )
        sources.append(f"""static int case{index}(void) {{
 memset(&aa{index},0x5a,sizeof(aa{index}));memset(&bb{index},0x5a,sizeof(bb{index}));memset(&cc{index},0xa5,sizeof(cc{index}));
 memcpy(aa{index}.values,a{index},{count});memcpy(bb{index}.values,b{index},{count});
 uint64_t begin,end,frm,flags;__asm__ volatile("csrwi frm,0;csrwi fflags,0;csrr %0,mcycle":"=r"(begin)::"memory");
 {symbol}(aa{index}.values,bb{index}.values,cc{index}.values,coeff{index});
 __asm__ volatile("csrr %0,mcycle;csrr %1,frm;csrr %2,fflags":"=r"(end),"=r"(frm),"=r"(flags)::"memory");
 printf("ADMITTED_CASE {index} %lu %lu %lu\\n",(unsigned long)(end-begin),(unsigned long)frm,(unsigned long)flags);
 if(frm||flags)return 10;
 for(int i=0;i<{count};i++)if(cc{index}.values[i]!=expected{index}[i]||aa{index}.values[i]!=a{index}[i]||bb{index}.values[i]!=b{index}[i]){{printf("FAIL {index} %d\\n",i);return 11;}}
 for(int i=0;i<2048;i++)if(aa{index}.pre[i]!=0x5a||aa{index}.post[i]!=0x5a||bb{index}.pre[i]!=0x5a||bb{index}.post[i]!=0x5a||cc{index}.pre[i]!=0xa5||cc{index}.post[i]!=0xa5)return 12;
 return 0;
}}""")
        cases.append(
            {
                "index": index,
                "source": certificate["source"],
                "predictor": coeff,
                "complete_admitted_pairs": count,
                "certificate": certificate,
                "compilation": compilation,
                "fixture": {
                    name: pin(directory / f"{name}.i8")
                    for name in ("a", "b", "expected", "coeff")
                },
            }
        )
    sources.append(
        "int main(void){int rc;"
        + "".join(f"if((rc=case{i}()))return rc;" for i in range(len(cases)))
        + 'printf("ADMITTED_DOMAIN_ALL PASS\\n");return 0;}'
    )
    c = work / "probe.c"
    c.write_text("\n".join(sources) + "\n")
    built = build_program(
        [c, *objects],
        work,
        target="gemmini",
        max_loaded_bytes=None,
        extra_cflags=["-march=rv64gc", "-mabi=lp64d"],
    )
    audit = audit_elf(built.elf.read_bytes())
    if audit["status"] != "pass":
        raise ValueError("final admitted-domain ELF contains forbidden instructions")
    command = [
        str(args.spike),
        "--extension=gemmini",
        "--isa=rv64gc",
        "-m0x80000000:0x80000000",
        str(built.elf),
    ]
    strict = subprocess.run(command, capture_output=True, timeout=300, check=False)
    console = strict.stdout + strict.stderr
    (work / "spike.stdout").write_bytes(console)
    if strict.returncode or b"ADMITTED_DOMAIN_ALL PASS" not in console:
        raise ValueError("strict complete admitted-domain target execution failed")
    first = {
        "schema": "admitted_residual_target_domain_v1",
        "status": "STRICT_PASS",
        "manifest": pin(args.manifest),
        "cases": cases,
        "elf": pin(built.elf),
        "spike": pin(args.spike),
        "spike_command": command,
        "strict_stdout": pin(work / "spike.stdout"),
        "nofsm": audit,
        "scope": "complete admitted source pairs; private disjoint outputs/inputs; all guards; no performance extrapolation",
    }
    (work / "strict_result.json").write_text(json.dumps(first, indent=2) + "\n")
    run = run_on_gsim(
        built.elf,
        target="gemmini",
        max_cycles=8000000,
        timeout_s=900,
        backdoor=True,
        stdout_path=work / "gsim.stdout",
    )
    if (
        not run.completed
        or run.returncode
        or "ADMITTED_DOMAIN_ALL PASS" not in run.stdout_tail
    ):
        raise ValueError("GSIM complete admitted-domain execution failed")
    first.update(
        status="PASS",
        gsim_engine=run.engine,
        gsim_stdout=pin(work / "gsim.stdout"),
        observed_cycles=[
            line
            for line in run.stdout_tail.splitlines()
            if line.startswith("ADMITTED_CASE ")
        ],
    )
    (work / "result.json").write_text(json.dumps(first, indent=2) + "\n")
    print("ADMITTED_DOMAIN_STRICT_GSIM_PASS", flush=True)


if __name__ == "__main__":
    main()
