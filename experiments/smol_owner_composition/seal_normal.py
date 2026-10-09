"""Seal new normal native/source/target build without claiming target execution."""
from pathlib import Path
import hashlib,json,subprocess
w=Path(__file__).resolve().parents[2]/'out/normal_composition'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
n=json.loads((w/'native_v3/validation.json').read_text());a=json.loads((w/'normal_final_nofsm.json').read_text())
assert n['allclose'] and n['elements']==1600 and n['bitwise_mismatches']==0 and n['product_calls']==23040
assert n['calls'][0:3]==[48,0,0] and n['calls'][4:7]==[12,0,0] and a['status']=='pass'
old=(w/'build_v5/lower/model.ll').read_bytes().splitlines(keepends=True);new=(w/'build_v6/lower/model.ll').read_bytes().splitlines(keepends=True)
assert old[1:]==new[1:]
paths=[w/'native_v3/validation.json',w/'native_v3/output.npy',w/'native_v3/model.o',w/'native_v3/model.so',w/'native_v3.log',w/'build_result.json',w/'normal_final_nofsm.json',w/'build_v6/model.elf',w/'build_v6/model.o',w/'build_v6/lower/model.ll',w/'build_v5/lower/model.ll',w/'build_v6/host_provider/host_provider.json',w/'build_v6/device_signatures.json',w/'source/preparation.json',w/'source/normal_prepared_original.mlir',w/'source/normal_prepared.mlir',w/'provider/build.json']
paths+=list((w/'capability_snapshot').glob('*'))+list((w/'provider').glob('*/*'))+list(w.glob('build_normal*.log'))
paths+=list((w.parents[1]/'experiments/smol_owner_composition').glob('*.py'))
r={'schema':'smol_normal_owner_current2072_composition_v1','policy':'Explicit approximate_source_roundoff_rms4; ordinary closed-group writer binding, not exact-observer theorem','native':n,'target_elf_sha256':sha(w/'build_v6/model.elf'),'target_no_fsm':a['status'],'source_llvm_equivalence':'Native build_v5 and target build_v6 LLVM differ only first ModuleID comment; all subsequent bytes identical','workspace':124061504,'owner_payload':30048912,'allocation_alignment':64,'configured_dram_bytes':17179869184,'target_execution':'UNMEASURED; no whole strict or stock admission','performance':'UNKNOWN. Original174.45B whole instruction evidence is not repriced by this composition.','historical_failures':'v2 missing capability; v3 missing explicit toolchain; v4 detected in-place prepared-source mutation from v3; v5 compiler pin absent. Logs and changed source preserved; v6 uses fresh per-build copy of original SHA82dce...23f7.','pins':{str(p.resolve()):sha(p) for p in paths if p.is_file()}}
(w/'qualification.json').write_text(json.dumps(r,indent=2)+'\n');print(len(r['pins']),r['target_elf_sha256'])
