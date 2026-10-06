from pathlib import Path
import json,re,struct,hashlib,subprocess
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scalar_pointwise_packet import FOUR_FEATURE
from merlin.llvmlower.late_quant_rne import rewrite
W=Path(__file__).resolve().parent;CORE=Path('/scratch/agustin/tmp/merlin-smol-encoded-zero-groups-20261005/out/artifacts/probes/reciprocal-rne-observer-20261006');R=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006');L=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
source=(CORE/'cost_m2/source.mlir').read_text();(W/'source.mlir').write_text(source)
ll=lower_to_llvm_ir(source,workdir=W/'lower4',features={FOUR_FEATURE,'lower_fma_to_intrinsic'}).replace('forward','materialized').replace('dealloc_helper','materialized_dealloc_helper')
native,nr=rewrite(ll,host_isa='portable');target,tr=rewrite(ll,host_isa='rv64gc',combine_clamp=True);assert len(nr['routes'])==len(tr['routes'])==4
(W/'plain4.native.ll').write_text(native);(W/'plain4.target.ll').write_text(target)
lines=target.splitlines(keepends=True);pattern=re.compile(r'  (%[-A-Za-z0-9_.]+) = call float @llvm.fma.f32\(float ([^,]+), float ([^,]+), float ([^)]+)\)\n$')
def word(s):
 value=struct.unpack('>d',bytes.fromhex(s[2:]))[0] if s.startswith('0x') else float(s)
 return struct.unpack('<I',struct.pack('<f',value))[0]
rows=[];i=0;out=[]
while i<len(lines):
 group=[pattern.fullmatch(x)for x in lines[i:i+4]]
 mode=None
 if len(group)==4 and all(group):
  vals=[[m[j]for m in group]for j in range(2,5)]
  same=[len(set(v))==1 and not v[0].startswith('%')for v in vals]
  variable=[all(v.startswith('%')for v in vs)for vs in vals]
  if same[1] and variable[0] and variable[2]:mode='constant_rhs'
  elif same[0] and same[2] and variable[1]:mode='constant_lhs_addend'
  elif same[2] and variable[0] and variable[1]:mode='constant_addend'
 if mode:
  results=[m[1]for m in group];lhs=[m[2]for m in group];rhs=[m[3]for m in group];add=[m[4]for m in group];assert not set(results)&set(lhs+rhs+add)
  temp='%constant.packet.'+str(len(rows));ty='{ float, float, float, float }'
  if mode=='constant_rhs':
   coefs=[word(rhs[0])];params=', '.join([*(f'float {v}'for v in lhs),f'i32 {coefs[0]}',*(f'float {v}'for v in add)])
   asm='fmv.w.x ft0,$8'+''.join('\\0Afmadd.s $'+str(j)+',$'+str(j+4)+',ft0,$'+str(j)for j in range(4));constraints='=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}'
  elif mode=='constant_addend':
   coefs=[word(add[0])];params=', '.join([*(f'float {v}'for v in rhs),f'i32 {coefs[0]}',*(f'float {v}'for v in lhs)])
   asm='fmv.w.x ft0,$8'+''.join('\\0Afmadd.s $'+str(j)+',$'+str(j)+',$'+str(j+4)+',ft0'for j in range(4));constraints='=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}'
  else:
   coefs=[word(lhs[0]),word(add[0])];params=', '.join([*(f'i32 {v}'for v in coefs),*(f'float {v}'for v in rhs)])
   asm='fmv.w.x ft0,$4\\0Afmv.w.x ft1,$5'+''.join('\\0Afmadd.s $'+str(j)+',ft0,$'+str(j)+',ft1'for j in range(4));constraints='=f,=f,=f,=f,r,r,0,1,2,3,~{ft0},~{ft1}'
  out.append(f'  {temp} = call {ty} asm "{asm}", "{constraints}"({params})\n')
  for j,v in enumerate(results):out.append(f'  {v} = extractvalue {ty} {temp}, {j}\n')
  rows.append(dict(results=results,lhs=lhs,rhs=rhs,addend=add,constant_words=coefs,mode=mode,source_lines=[i+1,i+4],scope='four contiguous independent typed f32FMAs, unchanged dynamic rounding and operandorder; explicit RV64GC provider capability'));i+=4
 else:out.append(lines[i]);i+=1
assert len(rows)==8,len(rows)
selected=''.join(out);(W/'materialized.target.ll').write_text(selected);(W/'materialized.native.ll').write_text(native)
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off']
for name in ['plain4','materialized']:
 subprocess.run([str(L/'clang'),*flags,'-c',str(W/(name+'.target.ll')),'-o',str(W/(name+'.o'))],check=True,capture_output=True)
 subprocess.run([str(L/'llvm-objdump'),'-d',str(W/(name+'.o'))],check=True,stdout=(W/(name+'.disasm')).open('w'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=dict(schema='experimental_constant_fma_packet_isa_binding_v1',source_sha256=sha(W/'source.mlir'),plain4_target_sha256=sha(W/'plain4.target.ll'),selected_target_sha256=sha(W/'materialized.target.ll'),native_unchanged_sha256=sha(W/'plain4.native.ll'),groups=rows,emission_owner='Explicit OOT RV64GC FMA packet assembly; generic source liveness/group legality to be owned byMerlin ifperformancequalified.',flags=flags,production_policy=False,token_usage_available=False)
(W/'prepared.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='groups'}),flush=True)
