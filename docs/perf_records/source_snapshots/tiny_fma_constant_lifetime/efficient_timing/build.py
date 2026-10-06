from pathlib import Path
import hashlib,json,subprocess
from dataclasses import asdict
from merlin.perf.layer_bench import build_program,run_on_gsim
from mlir_oot.no_fsm_audit import audit_elf
W=Path(__file__).resolve().parent;P=W.parent;OLD=Path('/scratch/agustin/tmp/merlin-smol-encoded-zero-groups-20261005/out/artifacts/probes/reciprocal-rne-observer-20261006/cost_m2');R=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006');L=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=json.loads((P/'gsim_receipt.json').read_text());assert prior['returncode']==-1 and prior['finish'] is None,'Refuse redundant same-object timing: prior run must be incomplete.'
q=json.loads((P/'qualification.json').read_text());e=json.loads((P/'early_clobber/qualification.json').read_text());assert e['status']=='pass' and e['object_reclosure']['actual_object_bytes_identical']
assert sha(OLD/'control.o')==q['control_object_sha256'] and sha(P/'materialized.o')==q['selected_object_sha256']
flags=json.loads((P/'prepared.json').read_text())['flags'];argv=[str(L/'clang'),*flags,'-c',str(W/'main.c'),'-o',str(W/'main.o')];subprocess.run(argv,check=True,capture_output=True)
objects=[OLD/'control.o',P/'materialized.o',R/'capsule/data.o',P/'primitive_source.o',P/'primitive_selected.o',W/'main.o'];pins={str(p):sha(p)for p in objects}
b=build_program(objects,W/'build',target='gemmini',max_loaded_bytes=None)
a=audit_elf(b.elf.read_bytes());assert a['status']=='pass';(W/'nofsm_audit.json').write_text(json.dumps(a,indent=2)+'\n')
s=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(b.elf)],capture_output=True,text=True,timeout=120)
(W/'spike.stdout').write_text(s.stdout);(W/'spike.stderr').write_text(s.stderr);assert s.returncode==0 and'ORIGINAL_FMA_LIFETIME PASS 11264 128'in s.stdout,s.stdout+s.stderr
r=dict(schema='same_object_fma_timing_harness_v1',emission_sha256=sha(P/'prepared.json'),prior_incomplete_gsim_sha256=sha(P/'gsim_receipt.json'),objects=pins,main_compile_argv=argv,elf_sha256=sha(b.elf),strict_source11264_and128guards=True,sharedSSA5frm_proof_sha256=sha(P/'early_clobber/qualification.json'),scope='Same actual source/kernel objects. Remove duplicate independently closed primitive test from timing; faster defined8byte input hash outsideROI checks same complete405504bytes. Full timedcapsule and warmedABBA order unchanged.',no_FSM=True,token_usage_available=False)
(W/'qualification.json').write_text(json.dumps(r,indent=2)+'\n');print('SAME_OBJECT_STRICT11264_PASS',flush=True)
g=run_on_gsim(b.elf,target='gemmini',timeout_s=1800,max_cycles=50000000,stdout_path=W/'gsim.stdout');(W/'gsim_receipt.json').write_text(json.dumps(asdict(g),indent=2,default=str)+'\n');print(g,flush=True)
