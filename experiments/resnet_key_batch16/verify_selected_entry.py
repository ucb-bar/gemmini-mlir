"""Exact selected implementation closure, including final branch relocations."""
from pathlib import Path
import hashlib,json,re,struct
root=Path(__file__).resolve().parents[2]/'out/artifacts/key_batch16_normal'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sections(path):
 data=Path(path).read_bytes();assert data[:6]==b'\x7fELF\x02\x01';assert struct.unpack_from('<H',data,18)[0]==243
 table=struct.unpack_from('<Q',data,40)[0];width,count,strings=struct.unpack_from('<HHH',data,58);assert width==64
 rows=[struct.unpack_from('<IIQQQQIIQQ',data,table+i*width) for i in range(count)]
 names=data[rows[strings][4]:rows[strings][4]+rows[strings][5]]
 return data,rows,[names[r[0]:].split(b'\0',1)[0].decode() for r in rows]
def symbol(path,name):
 data,rows,_=sections(path);found=[]
 for row in rows:
  if row[1]!=2:continue
  strings=rows[row[6]];names=data[strings[4]:strings[4]+strings[5]]
  for location in range(row[4],row[4]+row[5],row[9]):
   n,flags,vis,section,value,size=struct.unpack_from('<IBBHQQ',data,location)
   if names[n:].split(b'\0',1)[0].decode()==name and section:found.append((section,value,size))
 assert len(found)==1,(name,found)
 return found[0]
def signed(value,bits):return value-(1<<bits) if value&(1<<(bits-1)) else value
link=json.loads((root/'controlled2101/controlled_link.json').read_text());elf=root/'controlled2101/candidate/model.elf'
link['candidate_elf_sha256']=link['candidate_sha256']
route=json.loads((root/'key_bundle/joint.json').read_text())['routes'][0]
link['changed_leaves']=[dict(selected=route['compilation']['compiler_argv'][-1][-1],selected_sha256=route['compilation']['object_sha256'],kernel=route['kernel'])]
assert sha(elf)==link['candidate_elf_sha256']
stdout=root/'entry_closure/spike.stdout';stderr=root/'entry_closure/spike.stderr'
# Functional result from this exact -g replay must match the qualified complete stdout.
qualified=(root/'qualification_controlled/spike.log').read_text()
log=stdout.read_text()+stderr.read_text();digest=re.findall(r'OUT_SHA256 f32le 1000 4000 ([0-9a-f]{64})',log)
assert len(digest)==1 and digest[0]==re.search(r'OUT_SHA256 f32le 1000 4000 ([0-9a-f]{64})',qualified).group(1)
hist={int(pc,16):int(count) for pc,count in re.findall(r'^(?:0x)?([0-9a-fA-F]{8,16})\s+([0-9]+)\s*$',stderr.read_text(),re.M)}
assert hist
result=[]
for leaf in link['changed_leaves']:
 obj=Path(leaf['selected']);assert sha(obj)==leaf['selected_sha256'];name=leaf['kernel']
 od,orr,on=sections(obj);ei,ev,es=symbol(elf,name);oi,ov,os=symbol(obj,name)
 assert ov==0 and os==es and os>0 and hist.get(ev)==1,(name,ev,hist.get(ev),os,es)
 source=bytearray(od[orr[oi][4]+ov:orr[oi][4]+ov+os]);ed,err,en=sections(elf);section=err[ei]
 target=bytearray(ed[section[4]+ev-section[3]:section[4]+ev-section[3]+es]);rawsource=bytes(source);rawtarget=bytes(target)
 relocations=[];relax=[]
 for rr in orr:
  if rr[1]!=4 or rr[7]!=oi:continue
  assert rr[9]==24;symbols=orr[rr[6]];assert symbols[9]==24
  for location in range(rr[4],rr[4]+rr[5],24):
   offset,info,addend=struct.unpack_from('<QQq',od,location);kind,si=info&0xffffffff,info>>32
   if kind==51:relax.append(offset);continue
   assert kind in (16,17,44),(name,offset,kind)
   _,_,_,ssec,value,_=struct.unpack_from('<IBBHQQ',od,symbols[4]+si*24);assert ssec==oi
   expected=ev+value+addend;width=2 if kind==44 else 4;old=int.from_bytes(source[offset:offset+width],'little');final=int.from_bytes(target[offset:offset+width],'little')
   if kind==16:
    mask=0x1fff07f;assert final&0x7f==0x63
    immediate=((final>>31)<<12)|(((final>>7)&1)<<11)|(((final>>25)&63)<<5)|(((final>>8)&15)<<1);displacement=signed(immediate,13)
   elif kind==17:
    mask=0xfff;assert final&0x7f==0x6f
    immediate=((final>>31)<<20)|(((final>>21)&1023)<<1)|(((final>>20)&1)<<11)|(((final>>12)&255)<<12);displacement=signed(immediate,21)
   else:
    mask=0xffff^((1<<12)|(3<<10)|(3<<5)|(3<<3)|(1<<2));assert final&3==1 and final>>13 in (6,7)
    immediate=(((final>>12)&1)<<8)|(((final>>10)&3)<<3)|(((final>>5)&3)<<6)|(((final>>3)&3)<<1)|(((final>>2)&1)<<5);displacement=signed(immediate,9)
   assert old&mask==final&mask,(name,offset,'opcode/register changed');assert ev+offset+displacement==expected,(name,offset,'target mismatch')
   source[offset:offset+width]=(old&mask).to_bytes(width,'little');target[offset:offset+width]=(final&mask).to_bytes(width,'little')
   relocations.append({'offset':offset,'kind':kind,'target':expected,'width':width})
 assert source==target
 out=root/'entry_closure'/name;out.mkdir();(out/'source.raw.text').write_bytes(rawsource);(out/'candidate.raw.text').write_bytes(rawtarget);(out/'source.masked.text').write_bytes(source);(out/'candidate.masked.text').write_bytes(target)
 result.append({'symbol':name,'selected_object':str(obj),'selected_object_sha256':sha(obj),'symbol_start':ev,'body_bytes':es,'actual_entry_count':hist[ev],'all_nonrelocation_bytes_equal':True,'branch_opcode_registers_and_targets_exact':True,'relocations':relocations,'relax_offsets':relax})
assert len(result)==1
record={'schema':'current2101_key_batch16_selected_entry_closure_v1','status':'PASS','elf':str(elf),'elf_sha256':sha(elf),'strict_full1000_output_sha256':digest[0],'histogram_scope':'whole accepted image; entry counts, no cycle inference','selected':result,'proof_algorithm':'manual ELF64LE section/symbol/RELA parsing; only exact validated branch immediates masked','script_sha256':sha(__file__),'strict_stdout_sha256':sha(stdout),'strict_stderr_sha256':sha(stderr)}
(root/'entry_closure/selected_entries.json').write_text(json.dumps(record,indent=2)+'\n')
print('SELECTED_ENTRIES_PASS',[(r['symbol'],r['actual_entry_count'],r['body_bytes'],len(r['relocations'])) for r in result])
