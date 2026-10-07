"""Complete source M8 finite-scale preparation pair, source/cells unchanged."""
from __future__ import annotations
import importlib.util
import json
import os
import subprocess
from pathlib import Path

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.late_quant_rne import _tokens
from merlin.llvmlower.scaled_integer_finite_llvm import bind_finite_scale_helpers, emit_finite_scale_helper
from merlin.llvmlower.source_expression_interval import IntervalEffectContract, build_source_interval_table, emit_source_interval_i8_lookup, find_closed_scalar_i8_observers
from merlin.llvmlower.source_expression_interval_llvm import rewrite_source_interval_i8_lookup
from mlir_oot.late_quant_rne import merlin_host_llvm_transform
from mlir_oot.no_fsm_audit import audit_elf
from finite_scale_whole_prepare import OLD,HERE,LLVM,CORE,sha,save

OUT=HERE/'out/artifacts/probes/finite-scale-M8-v2-20261007'
P=OLD/'out/artifacts/probes/closed-i8-interval-result-20261007'
W=OLD/'out/artifacts/probes/source-interval-calibration-v2-20261006'
GCC=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc')


def rename(source,mapping):
    changes=[(t.start,t.end,mapping[t.text]) for t in _tokens(source) if t.text in mapping]
    for a,b,new in changes[::-1]:source=source[:a]+new+source[b:]
    return source


def main():
    OUT.mkdir(parents=True,exist_ok=False);commands=[]
    def run(argv):
        argv=list(map(str,argv));commands.append(argv)
        result=subprocess.run(argv,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        return result
    effects=IntervalEffectContract(True,True,True,True,True)
    typed=Path('/scratch/agustin/tmp/merlin-tiny-quant-consumer-main-20261006/out/artifacts/probes/source-expression-interval-promotion-20261006/normal_source_generic/typed_prepacket.generic.mlir')
    observers,refused=find_closed_scalar_i8_observers(parse_mlir_text(typed.read_text()),effects=effects);assert len(observers)==22 and not refused
    source=(P/'source.ll').read_text()
    old_report=json.loads((P/'source_binding.json').read_text())
    helpers=bind_finite_scale_helpers(source,routes=old_report['routes'],observers=observers,effects=effects,immutable_inputs=True,fresh_disjoint_output=True);assert len(helpers)==1
    table=build_source_interval_table(observers[0].expression,effects=effects,leading_bits=16,max_table_bytes=512*1024)
    changed,report=rewrite_source_interval_i8_lookup(source,proofs=observers,table=table,lookup_symbol='finite_lookup_activation',effects=effects)
    assert len(report['routes'])==2
    (OUT/'selected.ll').write_text(rename(changed,{'@integer_M8_b16_rne':'@finite_M8_rne'}))
    (OUT/'lookup.c').write_text(emit_source_interval_i8_lookup(table_name='table_b16',activation_name='finite_source_activation',quantizer_name='finite_quantize',lookup_name='finite_lookup_activation',leading_bits=16,finite_inputs=helpers))
    (OUT/'activation.ll').write_text(rename((P/'activation.ll').read_text(),{'@integer_source_activation':'@finite_source_activation'}))
    (OUT/'quantize.ll').write_text(rename((P/'quantize.ll').read_text(),{'@integer_quantize':'@finite_quantize'}))
    scan=emit_finite_scale_helper(helpers[0],wrapper_symbol='prepared_M8',finite_symbol='finite_M8_rne',fallback_symbol='control_M8')
    (OUT/'scale_guard.c').write_text(scan)
    flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off']
    run([LLVM/'clang',*flags,'-S','-emit-llvm',OUT/'lookup.c','-o',OUT/'lookup.ll'])
    run([LLVM/'clang',*flags,'-S','-emit-llvm',OUT/'scale_guard.c','-o',OUT/'scale_guard.ll'])
    run([LLVM/'llvm-link','-S',OUT/'selected.ll',OUT/'lookup.ll',OUT/'activation.ll',OUT/'quantize.ll',OUT/'scale_guard.ll','-o',OUT/'linked.ll'])
    run([LLVM/'opt','-S','-passes=always-inline',OUT/'linked.ll','-o',OUT/'inlined.ll'])
    merlin_host_llvm_transform(LLVM,combine_clamp=True)(OUT/'inlined.ll',OUT/'late_rne')
    run([LLVM/'clang',*flags,'-c',OUT/'late_rne/model.ll','-o',OUT/'candidate.o'])
    (OUT/'candidate.dump').write_text(run([LLVM/'llvm-objdump','-dr',OUT/'candidate.o']).stdout)
    assert sha(OUT/'candidate.o')!=sha(P/'candidate.o')
    modeguard='''#include <stdint.h>
extern void control_M8(int32_t*,float*,int32_t*,float*,int8_t*);
extern void prepared_M8(int32_t*,float*,int32_t*,float*,int8_t*);
void finite_M8(int32_t*a,float*sa,int32_t*b,float*sb,int8_t*out){unsigned frm;__asm__ volatile("csrr %0,frm":"=r"(frm)::"memory");if(frm)control_M8(a,sa,b,sb,out);else prepared_M8(a,sa,b,sb,out);}
'''
    (OUT/'target_guard.c').write_text(modeguard);run([LLVM/'clang',*flags,'-c',OUT/'target_guard.c','-o',OUT/'target_guard.o'])
    # Preserve existing source and current2070 helper objects. Harness pair changes
    # only function identities; all scans/table/continuation/store/frame are timed.
    main=(P/'main.c').read_text().replace('lazy_M8_b16(a,scale_a,b,scale_b,original+64)','integer_M8_b16(a,scale_a,b,scale_b,original+64)').replace('integer_M8_b16(a,scale_a,b,scale_b,guarded+64)','finite_M8(a,scale_a,b,scale_b,guarded+64)').replace('F pair[2]={lazy_M8_b16,integer_M8_b16}','F pair[2]={integer_M8_b16,finite_M8}')
    main=main.replace('extern void integer_M8_b16(int32_t*,float*,int32_t*,float*,int8_t*);','extern void integer_M8_b16(int32_t*,float*,int32_t*,float*,int8_t*);\nextern void finite_M8(int32_t*,float*,int32_t*,float*,int8_t*);')
    (OUT/'main_all_modes.c').write_text(main)
    declaration=json.loads((P/'target_v3/declaration.json').read_text());existing=[Path(p) for p in declaration['objects'] if not p.endswith('/main.o')]
    # Existing unused float-carrier objects remain pinned to preserve ordinary
    # data/table/context. Their code is not called by either timed arm.
    from merlin.perf.layer_bench import build_program
    for name,text in [('all_modes',main),('timing',main[:main.index(' for(unsigned frm=0;frm<5;frm++)')]+''' clear(guarded);mode(0,0);finite_M8(a,scale_a,b,scale_b,guarded+64);
 if(valid()){printf("INTEGER_RESULT_FAIL initialRNE\\n");return 1;}
'''+main[main.index(' printf("INTEGER_RESULT_SOURCE_GATE'):])]:
        case=OUT/name;case.mkdir()
        if name=='timing':text=text.replace('modes=5 presets=7 words=45056 flags guards PASS','initialRNE words=45056 original guards PASS')
        (case/'main.c').write_text(text)
        run([LLVM/'clang',*flags,'-c',case/'main.c','-o',case/'main.o'])
        objects=[*existing,OUT/'candidate.o',OUT/'target_guard.o',case/'main.o']
        before={str(p):sha(p) for p in objects}
        build=build_program(objects,case/'build',target='gemmini',max_loaded_bytes=None)
        assert before=={str(p):sha(p) for p in objects}
        audit=audit_elf(build.elf.read_bytes());assert audit['status']=='pass';save(case/'nofsm_audit.json',audit)
        result=run([GCC.with_name('spike'),'--isa=rv64gc','--extension=gemmini',build.elf])
        (case/'spike.stdout').write_text(result.stdout);(case/'spike.stderr').write_text(result.stderr)
        log=result.stdout+result.stderr
        assert 'INTEGER_RESULT_PASS original45056i8 allguards inputhashes rank0' in log and 'INTEGER_RESULT_FAIL' not in log
        rows=[dict(part.split('=',1) for part in line.split()[1:]) for line in log.splitlines() if line.startswith('INTEGER_RESULT_ROW ')]
        assert len(rows)==4
        save(case/'qualification.json',{'schema':'source_finite_broadcast_complete_M8_v1','status':'pass','mode_scope':'all5modes/7sticky' if name=='all_modes' else 'initialRNE plus ABBA','original_i8_words':45056,'rows':rows,'objects':before,'elf_path':str(build.elf),'elf_sha256':sha(build.elf),'all_sources_canonical_constants_order_table_and_finish_unchanged':True,'controls':'Exactcurrent2070 complete observer vs new complete scanner+finiteobserver; source fallback retained.','native_and_stock_cycles':'UNKNOWN','commands':commands,'pins':{str(p):sha(p) for p in [Path(__file__),typed,*OUT.rglob('*'),*objects] if p.is_file()},'token_usage_available':False})
        print('FINITE_SCALE_CAPSULE_STRICT_PASS',name,sha(build.elf),flush=True)

if __name__=='__main__':main()
