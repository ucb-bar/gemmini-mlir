from pathlib import Path
import json,collections,bisect,subprocess,sys
w=Path(sys.argv[1]).resolve();r=json.loads((w/'receipt.json').read_text());elf=Path(r['command'][-1]);prefix=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin');symbols=[]
raw=subprocess.check_output([str(prefix/'riscv64-unknown-elf-nm'),'-S','-n',str(elf)],text=True)
for line in raw.splitlines():
 a=line.split()
 if len(a)==4 and a[2] in ['t','T','w','W']:symbols.append((int(a[0],16),int(a[0],16)+int(a[1],16),a[3]))
symbols.sort();starts=[x[0] for x in symbols];counts={};functions=collections.Counter()
for line in (w/'spike.log').read_text().splitlines():
 a=line.split()
 if len(a)!=2:continue
 try:pc,n=int(a[0],16),int(a[1])
 except ValueError:continue
 counts[pc]=n;i=bisect.bisect_right(starts,pc)-1;name=symbols[i][2] if i>=0 and pc<symbols[i][1] else '<unmapped>';functions[name]+=n
raw=subprocess.check_output([str(prefix/'riscv64-unknown-elf-objdump'),'-d',str(elf)],text=True);(w/'disassembly.txt').write_text(raw);ops=collections.Counter();by=collections.defaultdict(collections.Counter);function='unknown'
for line in raw.splitlines():
 if line.endswith('>:'):function=line.split('<',1)[1][:-2];continue
 a=line.split()
 if len(a)<3 or not a[0].endswith(':'):continue
 try:pc=int(a[0][:-1],16)
 except ValueError:continue
 if pc in counts:ops[a[2]]+=counts[pc];by[function][a[2]]+=counts[pc]
d={'scope':'Same-ELF complete-program PC counts including init/checks, not ROI-only or hardware cycles','elf_sha256':r['elf_sha256'],'total':sum(counts.values()),'functions':dict(functions.most_common()),'opcodes':dict(ops.most_common()),'function_opcodes':{k:dict(v.most_common()) for k,v in by.items()}};(w/'attribution.json').write_text(json.dumps(d,indent=2)+'\n');print(list(d['functions'].items())[:5])
