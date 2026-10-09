"""Reproduce current2101 exactly, then replace only its qualified key pair."""
from pathlib import Path
import hashlib,json,subprocess
from mlir_oot.no_fsm_audit import audit_elf
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/key_batch16_normal';out=W/'controlled2101';out.mkdir();old=Path('/scratch/agustin/tmp/gemmini-dense-stationary-tail-normal-20261007/out/dense_tail_normal/controlled2095');r=json.loads((old/'controlled_link.json').read_text());steps=[x for x in r['partial_link_graph']if x['arm']=='candidate'];assert len(steps)==14;final=json.loads((old/'candidate/linker_argv.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();expected='1aa4fa5a921153d10d951d3acea89b3e2d6ba1bba1f3f97fc02718ab749bf3d3';assert sha(old/'candidate/model.elf')==expected
new=json.loads((W/'key_bundle/joint.json').read_text())['routes'];assert len(new)==1;route=new[0];assert route['panel_batch']==16;kernel=Path(route['compilation']['compiler_argv'][-1][-1]);adapter=Path(route['adapter_compilation']['compiler_argv'][-1]);assert sha(kernel)==route['compilation']['object_sha256'];assert sha(adapter)==route['adapter_compilation']['object_sha256']
objcopy='/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-objcopy';bound=out/'selected_adapter_original_abi.o';rename=[objcopy,'--redefine-sym','_mlir_ciface_'+route['symbol']+'=_mlir_ciface_'+route['original_symbol'],str(adapter),str(bound)];subprocess.run(rename,check=True)
oldkernel='/scratch/agustin/tmp/gemmini-residual-domain-scale-20261007/out/key_rectifier_normal_binding_v1/key_bundle/gemmini_residual_0__rectifier/kernel.o';oldadapter='/scratch/agustin/tmp/gemmini-residual-domain-scale-20261007/out/key_rectifier_normal_binding_v1/controlled2071_v2/selected_adapter_original_abi.o';assert steps[-1]['argv'].count(oldkernel)==steps[-1]['argv'].count(oldadapter)==1
replays=[];pins={}
for arm in ['control','candidate']:
 d=out/arm;d.mkdir();mapping={}
 for i,step in enumerate(steps):
  oldout=step['argv'][-1];dest=d/Path(oldout).name;cmd=[str(dest)if p==oldout else mapping.get(p,p)for p in step['argv']]
  if arm=='candidate'and i==13:cmd=[str(kernel)if p==oldkernel else str(bound)if p==oldadapter else p for p in cmd]
  for p in cmd[:-1]:
   if Path(p).is_file():pins[str(Path(p).resolve())]=sha(p)
  subprocess.run(cmd,check=True,capture_output=True)
  if arm=='control' or i<13:assert sha(dest)==step['output_sha256'],(arm,i)
  mapping[oldout]=str(dest);replays.append(dict(arm=arm,index=i,argv=cmd,output_sha256=sha(dest),previous_output_sha256=step['output_sha256']))
 cmd=[str(d/'model.elf')if i==len(final)-1 else mapping.get(p,p)for i,p in enumerate(final)]
 for p in cmd:
  if Path(p).is_file():pins[str(Path(p).resolve())]=sha(p)
 subprocess.run(cmd,check=True,capture_output=True)
 if arm=='control':assert sha(d/'model.elf')==expected
 audit=audit_elf((d/'model.elf').read_bytes());assert audit['status']=='pass';(d/'model.nofsm_audit.json').write_text(json.dumps(audit,indent=2));(d/'linker_argv.json').write_text(json.dumps(cmd,indent=2))
 for p in (old/'candidate').iterdir():
  if p.name in ['model.elf','model.nofsm_audit.json','linker_argv.json']:continue
  if not(d/p.name).exists():(d/p.name).symlink_to(p.resolve())
 print(arm,sha(d/'model.elf'),flush=True)
(out/'controlled_link.json').write_text(json.dumps(dict(control_job=2101,control_reproduced=expected,candidate_sha256=sha(out/'candidate/model.elf'),partial_link_graph=replays,adapter_rename=rename,changed_objects=[dict(original=oldkernel,selected=str(kernel)),dict(original=oldadapter,selected=str(bound))],retained_input_pins=pins,scope='only selected first residual kernel and ordinary adapter; all previous partial components reproduced; full validation pending'),indent=2))
