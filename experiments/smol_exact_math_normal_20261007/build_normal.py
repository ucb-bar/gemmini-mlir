"""Fresh normal source/provider binding; target build does not submit hardware."""
from pathlib import Path
import hashlib,json,subprocess,shutil
from merlin.llvmlower.device_build import DeviceRouting
from merlin.runtime.backends import spike_model
from merlin.runtime.host_provider import HostProviderObject,HostProviderBuild,read_host_function_abi
from mlir_oot.golden_device_catalog import merlin_builder,final_elf_audit
HERE=Path('/scratch/agustin/tmp/gemmini-smol-normal-composition-20261007')/'out/artifacts/probes/prepared-endpoint-normal'
WORK=Path('/scratch/agustin/tmp/merlin-latest-20261004/out/artifacts/probes/smol-exact-math-normal-20261007')
SOURCE=Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/smol_source_sum_fma_bundle')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
NM=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-nm')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof_path=HERE/'provider/build.json';proof=json.loads(proof_path.read_text())
for path,pin in proof['pins'].items():
 if sha(path)!=pin:raise ValueError('provider identity changed')
provider_dir=HERE/'provider/target_numeric'
provider_recipe=dict(compile_commands=proof['roles']['target_numeric']['commands'],compilation_pins=proof['pins'],compile_cwd=str(Path.cwd()))
bridge_dir=Path('/scratch/agustin/tmp/gemmini-prepared-rhs-owner-20261006/out/prepared_rhs_normal_materialized')
preparation=json.loads((HERE/'source/preparation.json').read_text())
effect_pin=sha(bridge_dir/'bridge.c')
prior=json.loads(Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005/bounded_native/build/device_prepared/source_control/source_group_calls.json').read_text())
def prepare(path,workdir):
 if sha(path)!=prior['original_prepared_sha256']:raise ValueError('normal source preparation input changed')
 selected=HERE/'source/normal_prepared_original.mlir'
 if sha(selected)!=preparation['selected_sha256']:raise ValueError('prepared source changed')
 workdir.mkdir(parents=True,exist_ok=True)
 copy=workdir/"normal_prepared.mlir";shutil.copyfile(selected,copy)
 return copy
def imported(path,ll,commands,pins,cwd):
 pins=dict(pins)
 for command in commands:
  compiler=Path(command[0]).resolve();pins[str(compiler)]=sha(compiler)
 text=subprocess.check_output([str(NM),'--format=posix','--extern-only',str(path)],text=True)
 defined=[];required=[]
 for row in text.splitlines():
  name,kind,*_=row.split();abi=read_host_function_abi(ll,name)
  (required if kind in ('U','w','v') else defined).append(abi)
 return HostProviderObject(Path(path),sha(path),Path(ll),sha(ll),tuple(defined),tuple(required),tuple((Path(p),pin)for p,pin in pins.items()),tuple(tuple(map(str,c))for c in commands),Path(cwd))

def host_provider(context):
 objects=[imported(provider_dir/'provider.o',provider_dir/'provider.ll',provider_recipe['compile_commands'],provider_recipe['compilation_pins'],provider_recipe['compile_cwd'])]
 # The compiler owns attention_frontier_writer; expose the qualified bridge
 # directly under its borrowed C-interface name, with unchanged C body.
 w=context.workdir;source=bridge_dir/'bridge.c';obj=w/'normal_bridge.o';ll=w/'normal_bridge.ll'
 common=[str(context.compiler),*context.compiler_flags,'-I'+str(provider_dir)]
 commands=[[*common,'-c',str(source),'-o',str(obj)],[*common,'-S','-emit-llvm',str(source),'-o',str(ll)]]
 for cmd in commands:context.compile(cmd,inputs=[source,provider_dir/'source_attention_frontier_api.h'],output=Path(cmd[cmd.index('-o')+1]))
 pins={str(source):sha(source),str(provider_dir/'source_attention_frontier_api.h'):sha(provider_dir/'source_attention_frontier_api.h'),str(Path(context.compiler).resolve()):sha(Path(context.compiler).resolve())}
 objects.append(imported(obj,ll,commands,pins,Path.cwd()))
 # Physical primitive imports preserve their source-derived target IR and object closure.
 for shape in ['qk','pv192','pv128']:
  for degree in range(5):
   d=Path('/scratch/agustin/tmp/gemmini-closed-bf16-certificate-20261005/out/artifacts/probes/closed-group-target-20261005/m256_h1')/f'{shape}_products_{degree}'
   r=json.loads((d/'device_compile.json').read_text());assert sha(d/'kernel.o')==r['object_sha256'];assert sha(d/'kernel.ll')==r['llvm_ir_sha256']
   pins={str(p):sha(p)for p in d.iterdir()if p.is_file()}
   for cmd in r['compiler_argv']:pins[str(Path(cmd[0]).resolve())]=sha(Path(cmd[0]).resolve())
   objects.append(imported(d/'kernel.o',d/'kernel.ll',r['compiler_argv'],pins,Path.cwd()))
 return HostProviderBuild(sha(context.prepared_path),sha(context.model_llvm_path),sha(context.model_object_path),tuple(objects),sha(proof_path),effect_pin)

(WORK/'binding_scope.json').write_text(json.dumps({'preparation':preparation,'policy':'explicit approximate_source_roundoff_rms4; exact observer binder NOT selected','original_whole_gate':{'atol':0.03125,'rtol':0.02},'validation':'pending'},indent=2)+'\n')
device=DeviceRouting(device='gemmini',package_dir=str(Path.cwd()),operand_dtype='i8',accum_dtype='i32',catalog_builder=merlin_builder(LLVM),final_elf_audit=final_elf_audit,prepared_transform=prepare)
result=spike_model.build(SOURCE,WORK/'build',arena_mb=8192,stack_bytes=16*1024**2,output_dump_cap=1,output_sha256=True,dram_bytes=16*1024**3,int8_compute=True,features=frozenset({'respect_captured_quantization_scope','lower_fma_to_intrinsic','outline_llvm_loops','lower_exact_math_inline'}),cflags_override=['-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin','-fno-vectorize','-fno-slp-vectorize'],device=device,host_math_policy='expf_via_double',host_provider_builder=host_provider)
(WORK/'build_result.json').write_text(json.dumps(result,default=str,indent=2)+'\n');print('NORMAL_ATTENTION_PROVIDER_BUILD_DONE',flush=True)
