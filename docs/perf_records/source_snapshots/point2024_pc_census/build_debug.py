from pathlib import Path
import json,subprocess,hashlib
import struct
w=Path(__file__).parent;old=Path('/scratch/agustin/tmp/gemmini-probability-point-20261006/out/probability_points/candidate');numeric=old.parent/'candidate_numeric';b=json.loads((old/'build.json').read_text());h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cmd=json.loads((numeric/'compile.json').read_text())['commands'][0].copy();cmd.insert(1,'-gline-tables-only');cmd[cmd.index('-MF')+1]=str(w/'provider.d');cmd[cmd.index('-o')+1]=str(w/'provider.o');subprocess.run(cmd,check=True)
link=b['link'].copy();link[link.index(str(numeric/'provider.o'))]=str(w/'provider.o');link[link.index('-o')+1]=str(w/'model.elf');subprocess.run(link,check=True)
def sections(p):
 data=p.read_bytes();assert data[:6]==b'\x7fELF\x02\x01';off=struct.unpack_from('<Q',data,40)[0];size,count,names=struct.unpack_from('<HHH',data,58);assert size==64 and count and names<count
 rows=[struct.unpack_from('<IIQQQQIIQQ',data,off+i*size)for i in range(count)];tab=rows[names];strings=data[tab[4]:tab[4]+tab[5]];out={}
 for row in rows:
  if not row[2]&2:continue
  name=strings[row[0]:].split(b'\0',1)[0].decode();body=b''if row[1]==8 else data[row[4]:row[4]+row[5]];assert row[1]==8 or len(body)==row[5]
  out[name]=(row[3],row[5],row[1],hashlib.sha256(body).hexdigest())
 return out
a=sections(old/'model.elf');c=sections(w/'model.elf');diff={k:[a.get(k),c.get(k)]for k in a.keys()|c.keys()if a.get(k)!=c.get(k)}
j={'scope':'Debug source map only, no execution. May map frozen same-ELF PC counts only if every allocated section address/size/data is identical.','compile':cmd,'link':link,'control_sha256':h(old/'model.elf'),'debug_elf_sha256':h(w/'model.elf'),'allocated_sections_identical':not diff,'differences':diff,'sections':a};(w/'closure.json').write_text(json.dumps(j,indent=2)+'\n');print('identical',not diff, len(a),list(diff)[:5])
