"""Diagnostic boundaries for the exact stock1988 control; no arithmetic change."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from mlir_oot.golden_device_profile import emit,leaf_kernel_profile
from mlir_oot.no_fsm_audit import audit_elf
from tests.fused_whole_model_probe import spike_validate

w=Path(__file__).resolve().parent
old=Path('/scratch/agustin/tmp/gemmini-paired-resident-packets-20261006/out/paired_resident_packets/normal_source/controlled1930')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
record=json.loads((old/'controlled_link.json').read_text())
expected='f515a8bc4cf18ebc6b690da779e37ce0ceca2e0c7ec2110d069107037ecbf43e'
assert record['elf_sha256']==expected==sha(old/'candidate/model.elf')
catalog_path=Path(record['old_catalog_manifest']);catalog=json.loads(catalog_path.read_text())
assert sha(catalog_path)==record['old_catalog_manifest_sha256']
llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
leaf_dir=w/'source_leaves';leaf_dir.mkdir(exist_ok=False)
leaves,boundaries,reproductions=leaf_kernel_profile(catalog,old/'candidate',leaf_dir,llvm)
replacement={Path(r['original']):Path(r['selected'])for r in record['changed_leaves']}
assert len(replacement)==5 and set(replacement)<=set(leaves)
for r in record['changed_leaves']:
 assert sha(r['original'])==r['original_sha256'] and sha(r['selected'])==r['selected_sha256']
leaves=[replacement.get(p,p)for p in leaves]
assert len(leaves)==len(set(leaves)) and len(boundaries)==catalog['total_device_contractions']==70
graph=[]
for i,node in enumerate(record['partial_link_graph']):
 if node['arm']!='candidate':continue
 argv=list(node['argv']);original=Path(argv[-1]);out=w/f'graph_{i}_{original.name}'
 assert sha(original)==node['output_sha256'];argv[-1]=str(out)
 subprocess.run(argv,check=True,capture_output=True)
 assert sha(out)==node['output_sha256'];graph.append({'argv':argv,'output_sha256':sha(out)})
link=json.loads((old/'candidate/linker_argv.json').read_text())
reproduce=list(link);reproduce[-1]=str(w/'reproduced1988.elf')
subprocess.run(reproduce,check=True,capture_output=True)
assert sha(w/'reproduced1988.elf')==expected
merged=[p for p in link if p.endswith('/guarded_mean_mixed.o')];assert len(merged)==1
source=w/'device_profile.c';source.write_text(emit([(r['symbol'],r['pointer_arity'])for r in boundaries]))
gcc=link[0];rt=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/merlin/runtime')
flags=['-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin']
compile=[gcc,*flags,'-I',str(rt/'baremetal/spike'),'-I',str(rt/'c'),'-c',str(source),'-o',str(w/'device_profile.o')]
subprocess.run(compile,check=True,capture_output=True)
argv=[]
for arg in link:
 if arg==merged[0]:argv.extend(map(str,leaves))
 elif arg=='-lm':argv.extend([str(w/'device_profile.o'),*[f'-Wl,--wrap={r["symbol"]}'for r in boundaries],'-Wl,--wrap=merlin_run_multi','-Wl,--wrap=htif_exit',arg])
 elif arg==link[-1]:argv.append(str(w/'model.elf'))
 else:argv.append(arg)
subprocess.run(argv,check=True,capture_output=True)
audit=audit_elf((w/'model.elf').read_bytes());assert audit['status']=='pass'
(w/'model.nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
pins={str(Path(p).resolve()):sha(p)for p in [*leaves,catalog_path,old/'controlled_link.json',old/'candidate/linker_argv.json',source,w/'device_profile.o']}
pins.update({str(Path(p).resolve()):sha(p)for p in link if Path(p).is_file()})
profile={'schema':'golden_device_boundary_profile_build_v1','control_job':1988,'control_cycles':33633109,
 'base_elf':str(old/'candidate/model.elf'),'base_elf_sha256':expected,'control_reproduced_byteidentical':True,
 'elf_sha256':sha(w/'model.elf'),'source_sha256':sha(source),'linker_argv':argv,'compiler_argv':compile,
 'boundaries':[dict(r,id=i)for i,r in enumerate(boundaries)],'expected_device_calls':70,
 'leaf_component_reproduction':reproductions,'current_candidate_graph_reproduction':graph,'pins':pins,
 'semantics':'Diagnostic only; exact current1988 host/runtime/device arithmetic. Primitive callbacks measured; descriptor checks and CPU readout in host gaps. Instrumentation placement/overhead changes timing. Never assign1974/1903 costs to1988.'}
(w/'profile_build.json').write_text(json.dumps(profile,indent=2)+'\n')
print('CURRENT1988_CONTROL_REPRODUCED',expected,'PROFILE_ELF',profile['elf_sha256'],flush=True)
(w/'host').mkdir(exist_ok=False)
shutil.copyfile(old/'host/output.npy',w/'host/output.npy')
validation=spike_validate(old/'capture',w,Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike'),w)
assert validation['exact_equal'];(w/'reference_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print('STRICT_ORIGINAL1000_PASS',json.dumps(validation),flush=True)
