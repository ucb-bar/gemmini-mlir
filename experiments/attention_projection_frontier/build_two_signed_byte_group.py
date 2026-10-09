"""Fixed two-byte product catalog and complete original consumer cost screen."""
from pathlib import Path
import ast,ctypes,hashlib,json,shutil,subprocess,time
from xdsl.dialects.builtin import StringAttr
from mlir_oot.golden_gemm import Shape
from mlir_oot.golden_product_sum import GoldenProductSum
from mlir_oot.golden_device_compile import compile_module
from mlir_oot.no_fsm_audit import audit_elf
base=Path(__file__).resolve().parents[2];w=base/'out/two_signed_byte_group';d=w/'candidate';d.mkdir(parents=True,exist_ok=False)
normal=base/'out/normal_composition/provider';dest=d/'target_numeric';dest.mkdir()
for p in (normal/'target_numeric').iterdir():
 if p.suffix in ('.c','.h'):shutil.copyfile(p,dest/p.name)
# Reuse the exact experiment adaptation already tested through all48 calls.
tree=ast.parse((base/'experiments/attention_projection_frontier/build_two_signed_byte_native.py').read_text())
replacement=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='replacement' for t in n.targets))
s=(dest/'provider.c').read_text();exec(compile(replacement,'sealed_two_byte_adaptation','exec'))
commands=[]
for old in json.loads((normal/'build.json').read_text())['roles']['target_numeric']['commands']:
 cmd=[x.replace('out/normal_composition/provider/target_numeric',str(dest)) for x in old];subprocess.run(cmd,check=True);commands.append(cmd)
llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin');objects=[];catalog=[]
for name,n,k,bn in [('qk',512,64,16),('pv192',64,192,4),('pv128',64,128,4)]:
 for degree in range(3):
  pairs=tuple((a,degree-a) for a in range(2) if 0<=degree-a<2)
  module=GoldenProductSum(Shape(256,n,k,'i32',bm=4,bn=bn,reuse_b=True,wide_b=True),pairs=pairs,lhs_planes=2,rhs_planes=2,absolute_bound=len(pairs)*k*128**2).build()
  symbol=f'{name}_products_{degree}';module.body.block.first_op.properties['sym_name']=StringAttr(symbol)
  folder=w/'products'/symbol;receipt=compile_module(module,llvm,folder);objects.append(folder/'kernel.o');catalog.append(dict(symbol=symbol,pairs=pairs,shape=[256,n,k],bound=len(pairs)*k*128**2,receipt=receipt))
lib=ctypes.CDLL(str(base/'out/observation_frontier/two_signed_byte_native/numeric/provider.so'));lib.group_provider_workspace_bytes.restype=ctypes.c_size_t;capacity=lib.group_provider_workspace_bytes()
oldbridge=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/closed_group_endpoint/production_provider_frozen/bridge.c')
bridge=oldbridge.read_text();bridge='\n'.join(line for line in bridge.splitlines() if not any(x in line for x in ('products_3','products_4')))+'\n';bridge=bridge.replace('121963584',str(capacity));(d/'bridge.c').write_text(bridge)
cmd=commands[0];cmd=[str(d/'bridge.c') if x==str(dest/'provider.c') else str(d/'bridge.o') if x==str(dest/'provider.o') else str(d/'bridge.d') if x==str(dest/'provider.d') else x for x in cmd];subprocess.run(cmd,check=True);commands.append(cmd)
b=Path('/scratch/agustin/tmp/gemmini-fused-encoder-radix-compose-20261007/out/encoder_compose/candidate');link=json.loads((b/'build.json').read_text())['link'];new=[]
for x in link:
 if x.endswith('/kernel.o'):continue
 if x==str(b/'model.elf'):x=str(d/'model.elf')
 elif x==str(b/'target_numeric/provider.o'):x=str(dest/'provider.o')
 elif x.endswith('/bridge.o'):x=str(d/'bridge.o')
 new.append(x)
new[new.index('-lm'):new.index('-lm')]=list(map(str,objects));subprocess.run(new,check=True,capture_output=True);commands.append(new)
audit=audit_elf((d/'model.elf').read_bytes());assert audit['status']=='pass'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(d/'build.json').write_text(json.dumps(dict(commands=commands,catalog=catalog,capacity=capacity,audit=audit,pins={str(p):sha(p) for p in w.rglob('*') if p.is_file()}),indent=2)+'\n')
t=w/'strict';t.mkdir();run=['timeout','--signal=TERM','--kill-after=15s','600s','/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike','-g','--isa=rv64gc','--extension=gemmini',str(d/'model.elf')];start=time.time()
with (t/'stdout').open('w') as out,(t/'stderr').open('w') as err:p=subprocess.run(run,stdout=out,stderr=err)
(t/'terminal.json').write_text(json.dumps(dict(command=run,returncode=p.returncode,seconds=time.time()-start,elf_sha256=sha(d/'model.elf')),indent=2)+'\n');print((t/'stdout').read_text(),flush=True)
