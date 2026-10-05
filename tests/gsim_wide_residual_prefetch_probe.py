"""Strict RV64GC numeric/guard screen for explicit residual M prefetch.

The optional source receipt provides an independently computed full source-pair
table. Otherwise the oracle is the declared i32 accumulation and f32 RNE store
contract. Both controls use the same generated input and expected bytes.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mlir_oot.golden_device_compile import compile_module
from mlir_oot.golden_wide_resadd import build, tables
from mlir_oot.no_fsm_audit import audit_elf
from merlin.perf.layer_bench import build_program, run_on_gsim


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m', type=int, default=1024)
    parser.add_argument('--p', type=int, required=True)
    parser.add_argument('--q', type=int, required=True)
    parser.add_argument('--scale', type=float, required=True)
    parser.add_argument('--no-relu', action='store_true')
    parser.add_argument('--prefetch-m', action='store_true')
    parser.add_argument('--banked-accumulators', action='store_true')
    parser.add_argument('--source-receipt', type=Path)
    parser.add_argument('--source-pairs', type=Path)
    parser.add_argument('--llvm-bin', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    count = args.m * 64
    pair = np.arange(count, dtype=np.int32) % 65536
    if count < 65536:
        # Cover both signs and mixed coefficients even for a single panel.
        pair = (pair * 40503 + 32768) % 65536
    a = (pair // 256 - 128).astype(np.int8)
    b = (pair % 256 - 128).astype(np.int8)
    acc = a.astype(np.int32) * args.p + b.astype(np.int32) * args.q
    expected = np.clip(np.rint(acc.astype(np.float32) * np.float32(args.scale)),
                       0 if not args.no_relu else -128, 127).astype(np.int8)
    source = None
    if args.source_receipt:
        receipt = json.loads(args.source_receipt.read_text())
        source = receipt['source']
        assert args.source_pairs and sha(args.source_pairs) == receipt['complete_pair_table_sha256']
        source_expected = np.frombuffer(args.source_pairs.read_bytes(), dtype='<i2')[pair]
        assert np.array_equal(source_expected, expected), 'numeric contract does not reproduce source'
        expected = source_expected.astype(np.int8)
    for name, array in (('a', a), ('b', b), ('expected', expected)):
        (work / (name + '.i8')).write_bytes(array.tobytes())
        (work / (name + '.h')).write_text(
            f'static const int8_t {name}[{count}] __attribute__((aligned(64)))={{'
            + ','.join(map(str, array.tolist())) + '};\n')
    (work / 'tables.h').write_text(
        'static const int8_t coefficients[768] __attribute__((aligned(64)))={'
        + ','.join(map(str, tables(args.p, args.q))) + '};\n')
    program = work / 'probe.c'
    program.write_text('''#include <stdint.h>
#include <stdio.h>
#include "a.h"
#include "b.h"
#include "expected.h"
#include "tables.h"
static struct {uint8_t before[2048];int8_t output[COUNT];uint8_t after[2048];} c __attribute__((aligned(64)));
extern void gemmini_golden_wide_resadd(const int8_t*,const int8_t*,int8_t*,const int8_t*);
int main(void) {
 for(int i=0;i<COUNT;i++)c.output[i]=-37;
 for(int i=0;i<2048;i++){c.before[i]=0x5a;c.after[i]=0xa5;}
 uint64_t begin,end;__asm__ volatile("csrr %0, mcycle":"=r"(begin)::"memory");
 gemmini_golden_wide_resadd(a,b,c.output,coefficients);
 __asm__ volatile("csrr %0, mcycle":"=r"(end)::"memory");
 printf("RESIDUAL_KERNEL_CYCLES %lu\\n",(unsigned long)(end-begin));
 for(int i=0;i<COUNT;i++)if(c.output[i]!=expected[i]){printf("FAIL %d got%d want%d\\n",i,c.output[i],expected[i]);return 1;}
 for(int i=0;i<2048;i++)if(c.before[i]!=0x5a||c.after[i]!=0xa5){printf("FAIL guard%d\\n",i);return 2;}
 printf("RESIDUAL_PREFETCH PASS all%d guards\\n",COUNT);return 0;
}
'''.replace('COUNT', str(count)))
    compilation = compile_module(build(args.m, args.p, args.q, args.scale,
        relu=not args.no_relu, prefetch_m=args.prefetch_m,
        banked_accumulators=args.banked_accumulators), args.llvm_bin, work)
    built = build_program([program, work / 'kernel.o'], work, target='gemmini',
        extra_cflags=[f'-I{work}', '-march=rv64gc', '-mabi=lp64d'], max_loaded_bytes=None)
    audit = audit_elf(built.elf.read_bytes())
    (work / 'nofsm_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    assert audit['status'] == 'pass'
    (work / 'elf_path.txt').write_text(str(built.elf) + '\n')
    run = run_on_gsim(built.elf, target='gemmini', max_cycles=10000000,
        timeout_s=600, backdoor=True, stdout_path=work / 'gsim.stdout')
    stdout = run.stdout_tail
    passed = bool(run.completed and run.returncode == 0
                  and f'RESIDUAL_PREFETCH PASS all{count} guards' in stdout)
    cycle_lines = [line.split()[-1] for line in stdout.splitlines()
                   if line.startswith('RESIDUAL_KERNEL_CYCLES ')]
    result = dict(schema='residual_m_prefetch_capsule_v1', status='pass' if passed else 'fail',
        elements=count, full_signed_i8_pairs=count >= 65536, m=args.m, p=args.p, q=args.q,
        scale=args.scale, relu=not args.no_relu, prefetch_m=args.prefetch_m,
        banked_accumulators=args.banked_accumulators,
        kernel_cycles=int(cycle_lines[-1]) if cycle_lines else None,
        source=source, source_receipt_sha256=sha(args.source_receipt) if args.source_receipt else None,
        source_pairs_sha256=sha(args.source_pairs) if args.source_pairs else None,
        input_sha256={name: sha(work / (name + '.i8')) for name in ('a', 'b')},
        expected_sha256=sha(work / 'expected.i8'), compilation=compilation,
        elf_path=str(built.elf), elf_sha256=sha(built.elf), nofsm_status=audit['status'],
        gsim_engine_sha256=run.engine.get('binary_sha256'), stdout=stdout, stderr=run.stderr_tail)
    (work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
