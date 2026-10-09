"""Debug-only companion of the immutable 2072 group, no target rerun."""
import json
import subprocess
from pathlib import Path
from merlin.perf.debug_companion import verify_debug_companion

root=Path(__file__).resolve().parents[2]
out=root/'out/current2072_attribution';out.mkdir(parents=True,exist_ok=False)
base=Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose')
cmd=json.loads((base/'candidate/target_numeric/compile.json').read_text())['commands'][0]
cmd=list(cmd);cmd[cmd.index('-MF')+1]=str(out/'provider.d');cmd[cmd.index('-o')+1]=str(out/'provider.o');cmd.append('-gline-tables-only')
subprocess.run(cmd,check=True)
obj=verify_debug_companion((base/'candidate/target_numeric/provider.o').read_bytes(),(out/'provider.o').read_bytes(),relocatable=True)
link=json.loads((base/'candidate/build.json').read_text())['link']
link=[str(out/'provider.o') if x==str(base/'candidate/target_numeric/provider.o') else x for x in link]
link[link.index('-o')+1]=str(out/'model.elf');subprocess.run(link,check=True,capture_output=True)
elf=verify_debug_companion((base/'candidate/model.elf').read_bytes(),(out/'model.elf').read_bytes())
counts={}
for line in (base/'histogram/spike.stderr').read_text().splitlines():
 t=line.split()
 if len(t)==2:
  try:counts[int(t[0],16)]=int(t[1])
  except ValueError:pass
pcs=sorted(counts)
symbolizer='/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/llvm-symbolizer'
result=subprocess.run([symbolizer,'--obj='+str(out/'model.elf'),'--output-style=JSON','--no-inlines'],input=''.join(hex(p)+'\n'for p in pcs),capture_output=True,text=True,check=True)
(out/'symbolized.jsonl').write_text(result.stdout)
records=[json.loads(line)for line in result.stdout.splitlines()]
rows={}
for pc,record in zip(pcs,records,strict=True):
 for symbol in record.get('Symbol',[]):
  key=(symbol.get('FileName',''),symbol.get('Line',0),symbol.get('FunctionName',''))
  rows[key]=rows.get(key,0)+counts[pc]
report=dict(scope='Whole executable histogram; provider-only source rows may establish ROI attribution, shared callees remain unscoped. No cycles.',object_identity=obj,elf_identity=elf,rows=[dict(file=f,line=l,function=n,instructions=c)for (f,l,n),c in sorted(rows.items(),key=lambda x:-x[1])])
(out/'attribution.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['rows'][:12]),flush=True)
