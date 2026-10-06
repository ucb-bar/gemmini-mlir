from pathlib import Path
import re,json,subprocess,hashlib,collections
w=Path(__file__).parent;assert json.loads((w/'closure.json').read_text())['allocated_sections_identical'];hist=w.parent/'current_word_i64_pc_histogram/spike.log';counts={}
for line in hist.read_text().splitlines():
 m=re.fullmatch(r'([0-9a-f]+) (\d+)',line)
 if m:counts[int(m[1],16)]=int(m[2])
cmd=['/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin/llvm-symbolizer','--obj='+str(w/'model.elf'),'--output-style=JSON','--inlining'];out=subprocess.check_output(cmd,input=''.join(hex(pc)+'\n'for pc in counts),text=True);(w/'symbolized.jsonl').write_text(out)
files=collections.Counter();lines=collections.Counter();functions=collections.Counter();sites=collections.Counter();mapped=0
for raw in out.splitlines():
 row=json.loads(raw);pc=int(row['Address'],16);n=counts[pc];frames=row['Symbol'];frame=frames[0]if frames else{};file=frame.get('FileName','');line=frame.get('Line',0);fn=frame.get('FunctionName','<unmapped>');files[file or'<no debug>']+=n;lines[f'{file}:{line}']+=n;functions[fn]+=n
 provider_frames=[f for f in frames if f.get('FileName','').endswith('/provider.c')]
 if provider_frames:
  # Outermost actual provider source call-site; full inline stack retained separately.
  f=provider_frames[-1];sites[f"{f['FunctionName']}:{f['Line']}"]+=n
 if file and line:mapped+=n
j={'scope':'Existing uninstrumented same-ELF PC histogram mapped only after all102allocated sections match debug-only relink. Inline source attribution, not isolated execution timers.','total':sum(counts.values()),'debug_mapped':mapped,'innermost_functions':dict(functions.most_common()),'innermost_files':dict(files.most_common()),'innermost_lines':dict(lines.most_common()),'outermost_provider_sites':dict(sites.most_common()),'command':cmd,'pins':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [w/'model.elf',w/'provider.o',w/'closure.json',hist,w/'symbolized.jsonl',w/'map.py']}}
(w/'attribution.json').write_text(json.dumps(j,indent=2)+'\n');print('functions',functions.most_common(16));print('sites',sites.most_common(14))
