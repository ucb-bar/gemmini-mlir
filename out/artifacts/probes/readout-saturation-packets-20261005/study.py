"""Measure exact optional Merlin readout schedules on frozen source accumulators."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from merlin.llvmlower.integer_readout import emit_readout
from merlin.perf.layer_bench import build_program, run_on_gsim
from mlir_oot.no_fsm_audit import audit_elf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--llvm-bin', type=Path, required=True)
    parser.add_argument('--spike', type=Path, required=True)
    parser.add_argument('--gsim', action='store_true')
    parser.add_argument('--variants', nargs='+', default=['control','saturation','packet2','packet4','packet8','saturation2','saturation4','saturation8'])
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    policies = dict(control={}, saturation=dict(saturation_first=True),
        packet2=dict(packet=2), packet4=dict(packet=4), packet8=dict(packet=8),
        saturation2=dict(saturation_first=True,packet=2),
        saturation4=dict(saturation_first=True,packet=4),
        saturation8=dict(saturation_first=True,packet=8))
    capture_path=args.fixture/'capture_receipt.json'
    capture=json.loads(capture_path.read_text())
    routes=[r for r in json.loads(args.proofs.read_text())['routes'] if r.get('integer_readout')]
    c=['#include <stdio.h>\n#include <stdint.h>\n#include <stddef.h>\n#include <math.h>\n',
       'static uint64_t clock_now(void){\n#ifdef __riscv\nuint64_t x;__asm__ volatile("csrr %0,mcycle":"=r"(x)::"memory");return x;\n#else\nreturn 0;\n#endif\n}\n']
    asm=[]; body=[]; arrays=[]
    for index,row in enumerate(routes):
        name=row['symbol']+'_readout'; proof=row['integer_readout']
        input_path=(args.fixture/(name+'.i32')).resolve()
        expected_path=(args.fixture/(name+'.i8')).resolve()
        pin=capture['readout_arrays'][str(input_path)]
        assert sha(input_path)==pin['sha256']
        data=np.fromfile(input_path,dtype='<i4')
        expected=np.fromfile(expected_path,dtype=np.int8)
        assert len(data)==pin['elements']==len(expected)
        assert int(data.min())>=proof['accumulator_min'] and int(data.max())<=proof['accumulator_max']
        arrays.append(dict(input_sha256=sha(input_path),expected_sha256=sha(expected_path),
            elements=len(data),low_saturated=int(np.count_nonzero(expected==proof['output_min'])),
            high_saturated=int(np.count_nonzero(expected==127)),proof=proof))
        for symbol,path in [(f'input{index}',input_path),(f'expected{index}',expected_path)]:
            asm.append(f'.section .data\n.balign 64\n.global {symbol}\n{symbol}:\n.incbin "{path}"\n')
        c.append(f'extern const int32_t input{index}[];extern const int8_t expected{index}[];static int8_t output{index}[{len(data)}+2048] __attribute__((aligned(64)));\n')
        arithmetic='volatile float value=(float)x;'
        for scale in proof['source_scales']:
            arithmetic+=f'value=value*{float(scale).hex()}f;'
        c.append(f'static int8_t original{index}(int32_t x){{{arithmetic}float q=nearbyintf(value);if(q<{proof["output_min"]})q={proof["output_min"]};if(q>127)q=127;return (int8_t)q;}}\n')
        body.append(f'\n#ifndef __riscv\nfor(size_t i=0;i<{len(data)};i++)if(original{index}(input{index}[i])!=expected{index}[i])return 3;\n#endif\n')
        for variant in args.variants:
            symbol=f'{variant}_{index}'
            c.append(emit_readout(proof,symbol,fixedpoint=True,**policies[variant]))
            body.append(f'''for(size_t i=0;i<{len(data)}+2048;i++)output{index}[i]=-77;
{{uint64_t a=clock_now();{symbol}(input{index},output{index},{len(data)});uint64_t b=clock_now();
for(size_t i=0;i<{len(data)};i++)if(output{index}[i]!=expected{index}[i]){{printf("FAIL {index} {variant} %lu\\n",(unsigned long)i);return 1;}}
for(size_t i={len(data)};i<{len(data)}+2048;i++)if(output{index}[i]!=-77)return 2;
printf("READOUT {index} {variant} %llu\\n",(unsigned long long)(b-a));}}\n''')
    c.append('int main(void){'+''.join(body)+'printf("READOUT PASS\\n");return 0;}\n')
    source=work/'probe.c';assembly=work/'data.S'
    source.write_text(''.join(c));assembly.write_text(''.join(asm))
    native=work/'native'
    subprocess.run(['cc','-O2','-fno-builtin','-ffp-contract=off',str(source),str(assembly),'-lm','-o',str(native)],check=True,capture_output=True)
    native_result=subprocess.run([str(native)],check=True,capture_output=True,text=True)
    (work/'native.stdout').write_text(native_result.stdout)
    assert 'READOUT PASS' in native_result.stdout
    obj=work/'data.o'
    subprocess.run([str(args.llvm_bin/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-c',str(assembly),'-o',str(obj)],check=True)
    build=build_program([source,obj],work/'build',target='gemmini',extra_cflags=['-march=rv64gc','-fno-builtin'],max_loaded_bytes=None)
    audit=audit_elf(build.elf.read_bytes());assert audit['status']=='pass'
    target=subprocess.run([str(args.spike),'--extension=gemmini','--isa=rv64gc',str(build.elf)],capture_output=True,text=True,timeout=180)
    output=target.stdout+target.stderr;(work/'spike.stdout').write_text(output)
    assert target.returncode==0 and 'READOUT PASS' in output
    measures=[]
    for line in output.splitlines():
        fields=line.split()
        if len(fields)==4 and fields[0]=='READOUT':
            measures.append(dict(case=int(fields[1]),variant=fields[2],retired_instructions=int(fields[3])))
    assert len(measures)==len(routes)*len(args.variants)
    receipt=dict(schema='exact_readout_optional_schedule_study_v1',arrays=arrays,
        capture_receipt_sha256=sha(capture_path),proof_source_sha256=sha(args.proofs),
        source_sha256=sha(source),elf_sha256=sha(build.elf),native_full_source_exact=True,
        all_target_outputs_and_2048_guards_exact=True,no_fsm_audit=audit,
        strict_target_stdout_sha256=sha(work/'spike.stdout'),spike_measurements=measures,
        metric='Spike retired instructions per full readout invocation; not hardware cycles',
        gsim=None,full_model_hardware_measured=False)
    (work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(measures,indent=2),flush=True)
    print('ELF',build.elf,flush=True)
    if args.gsim:
        result=run_on_gsim(build.elf,target='gemmini',max_cycles=100000000,timeout_s=600,stdout_path=work/'gsim.stdout')
        raw=(work/'gsim.stdout').read_text()
        timings=[]
        for line in raw.splitlines():
            fields=line.split()
            if len(fields)==4 and fields[0]=='READOUT':
                timings.append(dict(case=int(fields[1]),variant=fields[2],cycles=int(fields[3])))
        passed=result.completed and 'READOUT PASS' in raw and len(timings)==len(measures)
        receipt['gsim']=dict(status='pass' if passed else 'incomplete_or_failed',
            returncode=result.returncode,wall_seconds=result.wall_seconds,
            engine=result.engine,command_sha256=result.command_sha256,
            stdout_sha256=sha(work/'gsim.stdout'),measurements=timings,
            scope='complete captured readouts, not whole model')
        (work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(result,flush=True)


if __name__=='__main__':
    main()
