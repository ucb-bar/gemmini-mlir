"""Unchanged qualified candidate ELF: actual PC/symbol instruction attribution."""
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
import bisect, hashlib,json,re,subprocess
P=Path(__file__).resolve().parent
W=P.parent/'tiny-two-products-outline-composed-whole-20261006'/'whole'
L=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
save=lambda p,r:Path(p).write_text(json.dumps(r,indent=2)+'\n')
r=json.loads((W/'spike_validation.json').read_text());assert r['status']=='pass' and r['spike_full_output_match']
assert sha(W/'model.elf')==r['elf_sha256']=='d33239297b29c5d73f276ea0dfb579c7e62d8376f843a88c7366f078b2b40664'
assert sha(W/'spike_pc_histogram.log')==r['unchanged_ELF_PC_histogram_sha256']
argvs=[[str(L/'llvm-nm'),'--defined-only','--print-size','--numeric-sort',str(W/'model.elf')],[str(L/'llvm-objdump'),'--no-show-raw-insn','-d',str(W/'model.elf')]]
nm=subprocess.check_output(argvs[0],text=True);asm=subprocess.check_output(argvs[1],text=True)
(P/'symbols.txt').write_text(nm);(P/'disassembly.txt').write_text(asm)
functions={}
for line in nm.splitlines():
 fields=line.split()
 if len(fields)!=4 or fields[2] not in 'tTwW':continue
 start,size=int(fields[0],16),int(fields[1],16)
 if size==0:continue
 if start in functions:
  assert functions[start]['size']==size
  functions[start]['aliases'].append(fields[3])
 else:functions[start]=dict(start=start,size=size,aliases=[fields[3]])
starts=sorted(functions)
assert all(functions[a]['start']+functions[a]['size']<=b for a,b in zip(starts,starts[1:]))
parsed={}
for line in asm.splitlines():
 m=re.match(r'\s*([0-9a-f]+):\s+(.+)',line)
 if not m:continue
 pc=int(m[1],16);body=m[2].strip();parsed[pc]=body.split()[0] if body else '<empty>'
hist={int(a,16):int(b)for a,b in re.findall(r'^([0-9a-f]+) (\d+)$',(W/'spike_pc_histogram.log').read_text(),re.M)}
assert hist
counts={a:Counter()for a in starts};unresolved=[]
for pc,count in hist.items():
 i=bisect.bisect_right(starts,pc)-1
 if i<0 or pc>=functions[starts[i]]['start']+functions[starts[i]]['size']:
  unresolved.append(dict(pc=hex(pc),count=count));continue
 counts[starts[i]][parsed.get(pc,'<undecoded>')]+=count
rows=[]
for a in starts:
 c=counts[a]
 if not c:continue
 names=functions[a]['aliases']
 rows.append(dict(symbols=names,start=hex(a),code_bytes=functions[a]['size'],retired_instructions=sum(c.values()),classes=dict(c),scope='Whole-program counts for exactly this final ELF symbol. Runtime calls may occur both inside and outside timed model forward.'))
rows.sort(key=lambda row:row['retired_instructions'],reverse=True)
model=[row for row in rows if any(x=='forward' or x=='_mlir_ciface_forward' or x.startswith('forward.extracted.') for x in row['symbols'])]
aggregate=Counter()
for row in model:aggregate.update(row['classes'])
result=dict(schema='source_qualified_candidate_unchanged_ELF_PC_census_v1',recorded_utc=datetime.now(timezone.utc).isoformat(),qualified_ELF_sha256=r['elf_sha256'],full_original256000_digest=True,torch_original_gate=True,zero_FSM=True,functional_ROI_retired_instructions=r['functional_instructions_not_hardware_cycles'],model_forward_and_extracted_helpers_instructions=sum(row['retired_instructions']for row in model),model_forward_and_extracted_helpers_classes=dict(aggregate),whole_program_retired_instructions=sum(hist.values()),functions=rows,unknown_symbol_PC_instructions=sum(row['count']for row in unresolved),unresolved=unresolved,actual_argv=argvs,pins={str(p):sha(p)for p in [W/'model.elf',W/'spike_validation.json',W/'spike_pc_histogram.log',P/'symbols.txt',P/'disassembly.txt',Path(__file__),L/'llvm-nm',L/'llvm-objdump']},limitations=['Instruction counts are not cycles; hardware outcome for this candidate is pending.','This is the newlyqualified two-product+identity+outline candidate, not champion1975 or queued1983.','ELF symbol membership gives exact PC attribution; no individual source SSA/cost mapping is inferred.','Includes no machine-code instrumentation or counter probes.','Zero-sized boot/trap symbols and unknown ROM PCs remain unassigned.'],token_usage_available=False)
save(P/'census.json',result)
print('CURRENT_CANDIDATE_PC_CENSUS',result['model_forward_and_extracted_helpers_instructions'],r['functional_instructions_not_hardware_cycles'],flush=True)
for row in rows[:16]:print(row['symbols'],row['retired_instructions'],row['code_bytes'],{k:v for k,v in row['classes'].items()if k.startswith('f') or k in ['jal','jalr']},flush=True)
