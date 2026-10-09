from pathlib import Path
import struct,json,hashlib
T=Path(__file__).resolve().parent;W=T/'normal_whole_v3';out=W/'physical_table_closure.json';assert not out.exists()
expected=hashlib.sha256((W/'explicit_policy_reclosure/table.bin').read_bytes()).hexdigest()
def verify(path):
 with path.open('rb')as f:
  ident=f.read(16);assert ident[:4]==b'\x7fELF'and ident[4:6]==b'\x02\x01'
  h=struct.unpack('<HHIQQQIHHHHHH',f.read(48));kind,shoff,entsize,count=h[0],h[5],h[10],h[11];assert entsize==64
  f.seek(shoff);sections=[struct.unpack('<IIQQQQIIQQ',f.read(64))for _ in range(count)]
  matches=[]
  for section in sections:
   if section[1]!=2:continue
   strings=sections[section[6]];f.seek(strings[4]);names=f.read(strings[5]);f.seek(section[4]);raw=f.read(section[5]);assert section[9]==24
   for offset in range(0,len(raw),24):
    name,info,other,index,value,size=struct.unpack_from('<IBBHQQ',raw,offset)
    if names[name:names.find(b'\0',name)]==b'source_interval_table':matches.append((info,index,value,size))
  assert len(matches)==1;info,index,value,size=matches[0];assert info&15==1 and size==8388608
  section=sections[index];assert not(section[2]&1) and section[2]&2
  assert value%64==0;position=section[4]+value-section[3];assert position>=section[4]and position+size<=section[4]+section[5]
  f.seek(position);data=f.read(size);actual=hashlib.sha256(data).hexdigest();assert actual==expected
 return dict(path=str(path),ELF64_little_endian=True,symbols=1,bytes=size,alignment_atleast64=True,section_index=index,section_readonly=True,symbol_value=value,file_offset=position,data_sha256=actual,virtual_address_only_if_executable=value if kind==2 else None)
record=dict(schema='source_interval_shipped_physical_table_v1',status='pass',scope='Actualqualified normaltargetmodel.o and finalELF contain exactlyone readonly8MiBdata symbol, aligned64, bytes exactlycompilerproduced source-wide table. ActualRV64GC ELF little-endianIEEEbinary32 decoding contract; no cache miss or DRAM rate inference.',records=[verify(W/'target_v2/model.o'),verify(W/'target_v2/model.elf')])
out.write_text(json.dumps(record,indent=2)+'\n');print('SINGLE_SHIPPED_READONLY_TABLE_ACTUAL_OBJECT_ELF_BYTES_PASS',expected,flush=True)
