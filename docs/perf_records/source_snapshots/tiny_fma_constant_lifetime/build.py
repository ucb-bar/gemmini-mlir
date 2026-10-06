from pathlib import Path
import hashlib,json,subprocess,numpy as np
from merlin.llvmlower.abi import HostModel
from merlin.perf.layer_bench import build_program,run_on_gsim
from mlir_oot.no_fsm_audit import audit_elf
from dataclasses import asdict
W=Path(__file__).resolve().parent;OLD=Path('/scratch/agustin/tmp/merlin-smol-encoded-zero-groups-20261005/out/artifacts/probes/reciprocal-rne-observer-20261006/cost_m2');R=Path('/scratch/agustin/tmp/gemmini-golden-nofsm-20261004/out/artifacts/probes/tiny-broadcast-packet-20261006');L=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O3','-ffreestanding','-fno-builtin','-ffp-contract=off']
args=[np.load(R/'capture'/(n+'.npy'))for n in ['a','scale_a','b','scale_b']];args[0]=args[0][:,:2,:].copy();args[2]=args[2][:,:2,:].copy();expected=np.load(R/'capture/expected.npy')[:,:2,:].copy();actual=np.zeros_like(expected)
so=W/'materialized.so';subprocess.run([str(L/'clang'),'-O3','-fPIC','-shared',str(W/'materialized.native.ll'),'-lm','-o',str(so)],check=True,capture_output=True);HostModel.load(str(so),name='materialized')([(x.ctypes.data,x.shape)for x in args]+[(actual.ctypes.data,actual.shape)]);assert np.array_equal(actual,expected);print('NATIVE_ORIGINAL11264_PASS',flush=True)
# Derive independently called primitive functions from every ACTUAL emitted
# group mode/constant. Same IEEE f32 sourceFMA vs selected packet, all5frm+flags.
rows=json.loads((W/'prepared.json').read_text())['groups'];plain=[];selected=[]
for i,row in enumerate(rows):
 for target,body in [(False,plain),(True,selected)]:
  name=('selected'if target else'source')+'_probe'+str(i);ty='{ float, float, float, float }';mode=row['mode'];coeff=row['constant_words']
  lines=[f'define void @{name}(ptr %a,ptr %b,ptr %o) {{']
  for j in range(4):
   lines.extend([f'  %ap{j} = getelementptr float, ptr %a, i64 {j}',f'  %bp{j} = getelementptr float, ptr %b, i64 {j}',f'  %a{j} = load float, ptr %ap{j}',f'  %b{j} = load float, ptr %bp{j}'])
  for j,u in enumerate(coeff):lines.append(f'  %c{j} = bitcast i32 {u} to float')
  if not target:
   for j in range(4):
    operands=[f'%a{j}',f'%b{j}','%c0'] if mode=='constant_addend' else ([f'%a{j}','%c0',f'%b{j}'] if mode=='constant_rhs' else ['%c0',f'%a{j}','%c1'])
    lines.append(f'  %r{j} = call float @llvm.fma.f32('+', '.join('float '+v for v in operands)+')')
  else:
   if mode in ['constant_addend','constant_rhs']:
    asm='fmv.w.x ft0,$8'+''.join(('\\0Afmadd.s $'+str(j)+',$'+str(j)+',$'+str(j+4)+',ft0')if mode=='constant_addend' else('\\0Afmadd.s $'+str(j)+',$'+str(j+4)+',ft0,$'+str(j))for j in range(4))
    extra=[*(f'float %b{j}'if mode=='constant_addend' else f'float %a{j}'for j in range(4)),f'i32 {coeff[0]}',*(f'float %a{j}'if mode=='constant_addend' else f'float %b{j}'for j in range(4))];constraint='=f,=f,=f,=f,f,f,f,f,r,0,1,2,3,~{ft0}'
   else:
    asm='fmv.w.x ft0,$4\\0Afmv.w.x ft1,$5'+''.join('\\0Afmadd.s $'+str(j)+',ft0,$'+str(j)+',ft1'for j in range(4));extra=[f'i32 {coeff[0]}',f'i32 {coeff[1]}',*(f'float %a{j}'for j in range(4))];constraint='=f,=f,=f,=f,r,r,0,1,2,3,~{ft0},~{ft1}'
   lines.append(f'  %pack = call {ty} asm "{asm}", "{constraint}"('+', '.join(extra)+')')
   for j in range(4):lines.append(f'  %r{j} = extractvalue {ty} %pack, {j}')
  for j in range(4):lines.extend([f'  %op{j} = getelementptr float, ptr %o, i64 {j}',f'  store float %r{j}, ptr %op{j}'])
  lines.extend(['  ret void','}']);body.append('\n'.join(lines))
objects=[OLD/'control.o',W/'materialized.o',R/'capsule/data.o']
for name,body in [('primitive_source',plain),('primitive_selected',selected)]:
 ir=W/(name+'.ll');ir.write_text('\n\n'.join(body)+'\n\ndeclare float @llvm.fma.f32(float,float,float)\n');obj=W/(name+'.o');subprocess.run([str(L/'clang'),*flags,'-c',str(ir),'-o',str(obj)],check=True,capture_output=True);objects.append(obj)
main=(OLD/'main.c').read_text().replace('candidate','materialized').replace('RECIP_OBSERVER_CYCLES','FMA_LIFETIME_CYCLES').replace('ORIGINAL_RECIP_OBSERVER','ORIGINAL_FMA_LIFETIME')
protos='\n'.join(f'extern void {p}_probe{i}(float*,float*,float*);'for p in ['source','selected']for i in range(8))
check='''
static int check_primitives(void){
 typedef void(*fn)(float*,float*,float*);
 fn source[8]={SOURCE};fn selected[8]={SELECTED};
 unsigned rng=0x6f48a293;unsigned sa[4],sb[4],outa[4],outb[4];float a[4],b[4],oa[4],ob[4];
 for(unsigned frm=0;frm<5;frm++)for(unsigned op=0;op<8;op++)for(unsigned trial=0;trial<512;trial++){
  for(unsigned j=0;j<4;j++){rng=rng*1664525+1013904223;sa[j]=rng;rng=rng*1664525+1013904223;sb[j]=rng;__builtin_memcpy(a+j,sa+j,4);__builtin_memcpy(b+j,sb+j,4);}
  unsigned f0,f1;asm volatile("csrw frm,%0;csrw fflags,%1"::"r"(frm),"r"(8):"memory");source[op](a,b,oa);asm volatile("csrr %0,fflags":"=r"(f0)::"memory");
  asm volatile("csrw fflags,%0"::"r"(8):"memory");selected[op](a,b,ob);asm volatile("csrr %0,fflags":"=r"(f1)::"memory");
  for(unsigned j=0;j<4;j++){__builtin_memcpy(outa+j,oa+j,4);__builtin_memcpy(outb+j,ob+j,4);if(outa[j]!=outb[j]){printf("FMA_BITS_FAIL %u %u %u %u\\n",frm,op,trial,j);return 1;}}
  if(f0!=f1){printf("FMA_FLAGS_FAIL %u %u %u %u %u\\n",frm,op,trial,f0,f1);return 2;}
 }
 asm volatile("csrw frm,zero;csrw fflags,zero":::"memory");printf("FMA_PRIMITIVE PASS 81920 fivefrm stickyflags\\n");return 0;
}
'''.replace('SOURCE',','.join(f'source_probe{i}'for i in range(8))).replace('SELECTED',','.join(f'selected_probe{i}'for i in range(8)))
main=main.replace('int main(void){',protos+'\n'+check+'\nint main(void){if(check_primitives())return 10;')
(W/'main.c').write_text(main);obj=W/'main.o';subprocess.run([str(L/'clang'),*flags,'-c',str(W/'main.c'),'-o',str(obj)],check=True,capture_output=True);objects.append(obj)
b=build_program(objects,W/'build',target='gemmini',max_loaded_bytes=None);audit=audit_elf(b.elf.read_bytes());assert audit['status']=='pass';(W/'nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
s=subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','--isa=rv64gc','--extension=gemmini',str(b.elf)],capture_output=True,text=True,timeout=600);(W/'spike.stdout').write_text(s.stdout);(W/'spike.stderr').write_text(s.stderr);assert s.returncode==0 and'FMA_PRIMITIVE PASS' in s.stdout and'ORIGINAL_FMA_LIFETIME PASS'in s.stdout,s.stdout+s.stderr
r=dict(schema='exact_fma_constant_lifetime_capsule_v1',source_shape=[1,2,5632],original_i8_words=11264,primitives_raw_words=81920,rounding_modes=5,stickyflags_exact=True,all_original_tap_words_exact=True,scope='Full first2rows/all5632channels original preDown helper inclloads/poly/dequant/div/quant/store/alloc/finalcopy; compare existingpacket2 to packet4+constantmaterialization. Plain4spillcensus separate; coupled schedule choice explicit.',prepared_sha256=sha(W/'prepared.json'),elf_sha256=sha(b.elf),control_object_sha256=sha(OLD/'control.o'),selected_object_sha256=sha(W/'materialized.o'),zero_FSM=True,strict_pass=True,compiler_promotion=False,token_usage_available=False)
(W/'qualification.json').write_text(json.dumps(r,indent=2)+'\n');print('STRICT_FMA_LIFETIME_PASS',flush=True)
g=run_on_gsim(b.elf,target='gemmini',timeout_s=1800,max_cycles=100000000,stdout_path=W/'gsim.stdout');(W/'gsim_receipt.json').write_text(json.dumps(asdict(g),indent=2,default=str)+'\n');print(g,flush=True)
