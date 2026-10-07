"""One complete original group, exact sparse alternative plus dense fallback."""
from pathlib import Path
import ctypes, hashlib, json, shutil, subprocess, time
from xdsl.dialects.builtin import StringAttr
from mlir_oot.golden_gemm import Shape
from mlir_oot.golden_product_sum import GoldenProductSum
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf
from adapt_sparse_dyadic import adapt
base=Path(__file__).resolve().parents[2];w=base/'out/sparse_dyadic_group';d=w/'candidate';d.mkdir(parents=True,exist_ok=False)
normal=base/'out/normal_composition/provider';commands=[]
for role in ('native_numeric','target_numeric'):
 dest=d/role;dest.mkdir()
 for p in (normal/role).iterdir():
  if p.suffix in ('.h','.c'):shutil.copyfile(p,dest/p.name)
 (dest/'provider.c').write_text(adapt((dest/'provider.c').read_text(),dest))
 for old in json.loads((normal/'build.json').read_text())['roles'][role]['commands']:
  cmd=[x.replace('out/normal_composition/provider/'+role,str(dest)) for x in old];subprocess.run(cmd,check=True);commands.append(cmd)
lib=ctypes.CDLL(str(d/'native_numeric/provider.so'));lib.group_provider_workspace_bytes.restype=ctypes.c_size_t;capacity=lib.group_provider_workspace_bytes()
llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin');objects=[];catalog=[]
for name,n,k,bn in [('qk',512,64,16),('pv192',64,192,4),('pv128',64,128,4)]:
 for degree in range(3):
  pairs=tuple((a,degree-a) for a in range(2) if 0<=degree-a<2)
  module=GoldenProductSum(Shape(256,n,k,'i32',bm=4,bn=bn,reuse_b=True,wide_b=True),pairs=pairs,lhs_planes=2,rhs_planes=2,absolute_bound=len(pairs)*k*128**2).build()
  symbol=f'{name}_sparse_products_{degree}';module.body.block.first_op.properties['sym_name']=StringAttr(symbol)
  folder=w/'products'/symbol;receipt=compile_module(module,llvm,folder);objects.append(folder/'kernel.o');catalog.append(dict(symbol=symbol,pairs=pairs,shape=[256,n,k],bound=len(pairs)*k*128**2,receipt=receipt))
oldbridge=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/closed_group_endpoint/production_provider_frozen/bridge.c')
bridge=oldbridge.read_text().replace('121963584',str(capacity));declarations='';dispatch=''
for name,n,k in [('qk',512,64),('pv192',64,192),('pv128',64,128)]:
 for degree in range(3):
  symbol=f'{name}_sparse_products_{degree}';declarations+=f'extern void {symbol}(const int8_t*,const int8_t*,int32_t*);\n'
  dispatch+=f'if(m==256 && n=={n} && k=={k} && degree=={degree+5}){{{symbol}(a,b,c);return 1;}}\n'
bridge=bridge.replace('static int products(',declarations+'static int products(').replace('(void)opaque;','(void)opaque;\n'+dispatch)
(d/'bridge.c').write_text(bridge)
cmd=next(c for c in commands if str(d/'target_numeric/provider.c') in c)
cmd=[str(d/'bridge.c') if x==str(d/'target_numeric/provider.c') else str(d/'bridge.o') if x==str(d/'target_numeric/provider.o') else str(d/'bridge.d') if x==str(d/'target_numeric/provider.d') else x for x in cmd]
subprocess.run(cmd,check=True);commands.append(cmd)
b=Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose/candidate');link=json.loads((b/'build.json').read_text())['link'];new=[]
for x in link:
 if x==str(b/'model.elf'):x=str(d/'model.elf')
 elif x==str(b/'target_numeric/provider.o'):x=str(d/'target_numeric/provider.o')
 elif x.endswith('/bridge.o'):x=str(d/'bridge.o')
 new.append(x)
new[new.index('-lm'):new.index('-lm')]=list(map(str,objects));subprocess.run(new,check=True,capture_output=True);commands.append(new)
audit=audit_elf((d/'model.elf').read_bytes());assert audit['status']=='pass'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(d/'build.json').write_text(json.dumps(dict(commands=commands,catalog=catalog,capacity=capacity,audit=audit,contract='private explicit product ordinals0..4 original3digit;5..7 offset2byte degrees0..2; source roundoff policy unchanged',pins={str(p):sha(p) for p in w.rglob('*') if p.is_file()}),indent=2)+'\n')
t=w/'strict';t.mkdir();run=['timeout','--signal=TERM','--kill-after=15s','600s','/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','-g','--isa=rv64gc','--extension=gemmini',str(d/'model.elf')];start=time.time()
with (t/'stdout').open('w') as out,(t/'stderr').open('w') as err:p=subprocess.run(run,stdout=out,stderr=err)
(t/'terminal.json').write_text(json.dumps(dict(command=run,returncode=p.returncode,seconds=time.time()-start,elf_sha256=sha(d/'model.elf')),indent=2)+'\n');print((t/'stdout').read_text(),flush=True)
