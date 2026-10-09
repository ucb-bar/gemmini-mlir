"""Close the completed execution after a missing probe-native-reference path."""
import hashlib,json,shutil,struct
from pathlib import Path
import numpy as np
from mlir_oot.golden_device_profile import parse_profile
from tests.fused_whole_model_probe import quality,verify_output_digest
from mlir_oot.no_fsm_audit import audit_elf
w=Path(__file__).resolve().parent
old=Path('/scratch/agustin/tmp/gemmini-paired-resident-packets-20261006/out/paired_resident_packets/normal_source/controlled1930')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected=json.loads((old/'spike_validation.json').read_text())
assert sha(old/'host/output.npy')==expected['native_reference_sha256']
(w/'host').mkdir(exist_ok=False);shutil.copyfile(old/'host/output.npy',w/'host/output.npy')
text=(w/'spike.log').read_text();assert 'DONE' in text and 'METRIC memref_rank_mismatch 0' in text
lines=[x.split()for x in text.splitlines()if x.startswith('OUT ')];assert len(lines)==1
count=int(lines[0][1]);words=np.array([int(x)&0xffffffff for x in lines[0][2:]],np.uint32);assert count==len(words)
native=np.load(w/'host/output.npy').reshape(-1);reference=np.load(old/'capture/golden.npy').reshape(-1)
evidence=verify_output_digest(text,native);assert evidence and np.array_equal(words,native.view('u4')[:count])
assert native.size==reference.size==1000 and np.array_equal(native.view('u4'),reference.view('u4'))
profile=json.loads((w/'profile_build.json').read_text());conserved=parse_profile(text,profile)
audit=audit_elf((w/'model.elf').read_bytes());assert audit['status']=='pass'
r={'schema':'golden_current1988_profile_strict_gate_v1','scope':'Same completed strictRV64GC execution; reference path fix and saved-console parse. No second run or hardware timing.',
 'elf_sha256':sha(w/'model.elf'),'exact_equal':True,'elements':1000,'target_native_exact':True,
 'native_reference_sha256':sha(w/'host/output.npy'),'original_golden_sha256':sha(old/'capture/golden.npy'),
 'console_sha256':sha(w/'spike.log'),'conserved_instructions':conserved,'digest_evidence':evidence,'nofsm':audit,
 'probe_harness_failure':'First parser failed only because host/output.npy was absent; actual target completed. Original build.log retained.'}
(w/'reference_validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':'pass','elf_sha256':r['elf_sha256'],'elements':1000,'boundary_count':len(conserved['events'])}),flush=True)
