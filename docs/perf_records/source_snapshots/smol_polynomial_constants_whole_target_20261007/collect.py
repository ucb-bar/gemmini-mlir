from pathlib import Path
import hashlib,json,re
import numpy as np
from mlir_oot.no_fsm_audit import audit_elf

WORK=Path(__file__).resolve().parent/'smol_polynomial_constants_whole_strict'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
a_path=WORK/'root_admission.json';a=json.loads(a_path.read_text())
packet=Path(a['source_packet']);assert sha(packet)==a['source_packet_sha256']=='000366dadc1157c57ae00e046570a884ab7111eacd35e3adb6f5fcc254fa1a83'
p=json.loads(packet.read_text());assert len(p['pins'])==a['source_pins_revalidated']==171
for path,digest in p['pins'].items():assert sha(path)==digest,path
terminal_path=WORK/'terminal.json';terminal=json.loads(terminal_path.read_text())
assert terminal['returncode']==0 and terminal['argv']==a['argv']
elf=Path(a['elf']);assert sha(elf)==p['target_elf_sha256']==a['elf_sha256']==terminal['elf_sha256']
engine=Path(a['engine']);assert sha(engine)==a['engine_sha256']
assert terminal['argv'][3]=='7200s' and '-m0x80000000:0x400000000' in terminal['argv']
stdout=WORK/'spike.stdout';histogram=WORK/'spike.stderr'
assert sha(stdout)==terminal['stdout_sha256'] and sha(histogram)==terminal['histogram_sha256']
text=stdout.read_text().replace('\r','');lines=text.splitlines()
assert lines.count('DONE')==1 and lines.count('METRIC memref_rank_mismatch 0')==1
native=Path('/scratch/agustin/tmp/gemmini-smol-normal-composition-20261007/out/artifacts/probes/prepared-polynomial-constants/normal/native_v3/output.npy')
assert sha(native)==p['pins'][str(native)]==p['native']['original_golden_sha256']
g=np.load(native);assert g.size==1600 and g.dtype==np.float32
digest=hashlib.sha256(g.astype('<f4').tobytes()).hexdigest()
assert re.findall(r'^OUT_SHA256 f32le (\d+) (\d+) ([a-f0-9]{64})$',text,re.M)==[('1600','6400',digest)]
cycles,=re.findall(r'^METRIC cycles (\d+)$',text,re.M)
audit=audit_elf(elf.read_bytes());assert audit['status']=='pass' and not audit['forbidden'] and not audit['unknown']
record=dict(schema='root_smol_prepared_polynomial_constants_whole_target_qualification_v1',status='PASS',
 source_pins_revalidated=171,target_elf=str(elf),target_elf_sha256=sha(elf),original_elements=1600,
 original_gate=a['original_gate'],original_raw_output_sha256=digest,bitwise_exact=True,rank_mismatch=0,
 native_source_validation=p['native']['bitwise_mismatches']==0,terminal=terminal,final_nofsm=audit,
 spike_functional_cycles=int(cycles),whole_firesim_cycles='UNKNOWN',stock_capacity_proven=False,
 scope='Independent complete successor ELF run; all1600 original output words exact. Spike functional cycles are not stock timings. No target group counters or allocator highwater inferred from native instrumentation.',
 pins={str(x):sha(x) for x in [packet,a_path,terminal_path,stdout,histogram,elf,engine,native,Path(__file__),WORK.parent/'run_smol_polynomial_constants_whole_strict.py']})
(WORK/'qualification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('pins','final_nofsm','terminal')}))
