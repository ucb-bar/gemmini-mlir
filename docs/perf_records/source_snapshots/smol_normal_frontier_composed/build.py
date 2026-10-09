"""Explicit ordinary whole-model source/consumer/provider integration experiment."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,subprocess
from merlin.llvmlower.closed_group_writer import ClosedGroupWriterContract
from merlin.llvmlower.fresh_tensor_writer import PrivateWorkspaceContract
from merlin.llvmlower.consumer_observed_group_writer import ConsumerObservationContract,ConsumerObservedGroupPreparation
from merlin.llvmlower.ordered_bf16_group_binding import SourceExactGroupPreparation
from merlin.llvmlower.device_build import DeviceRouting
from merlin.runtime.backends import spike_model
from merlin.runtime.host_provider import HostProviderObject,HostProviderBuild,read_host_function_abi
from mlir_oot.golden_device_catalog import merlin_builder,final_elf_audit

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'closed_group_endpoint'
TAPS=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/source-group-native-20261005')
SOURCE=Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/smol_source_sum_fma_bundle')
LLVM=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
NM=Path('/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-nm')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof_path=HERE/'numeric_witness.json'
proof=json.loads(proof_path.read_text())
for path,pin in proof['numeric_dependency_pins'].items():assert sha(path)==pin
from merlin.runtime.numeric_provider_identity import validate_numeric_provider_identity
import ctypes as C
identity_library=C.CDLL(str(HERE/'native_numeric_frozen/provider.so'))
identity_library.group_provider_workspace_bytes.restype=C.c_size_t
identity_library.group_provider_workspace_alignment.restype=C.c_size_t
identity_validation=validate_numeric_provider_identity(proof,manifest_path=HERE/'native_numeric_frozen/manifest.json',native_library_path=HERE/'native_numeric_frozen/provider.so',queried_workspace_bytes=identity_library.group_provider_workspace_bytes(),queried_workspace_alignment=identity_library.group_provider_workspace_alignment())
(HERE/'numeric_identity_validation.json').write_text(json.dumps(identity_validation,indent=2)+'\n')
complete=TAPS/'numeric_minmax_native_screen/complete_observation_pair.json'
observations=json.loads(complete.read_text());assert observations['integer_mismatches']==0 and observations['bf16_scale_mismatches']==0
native=TAPS/'numeric_minmax_native_screen/native_validation.json';assert json.loads(native.read_text())['bitwise_mismatches']==0
assert sha(SOURCE/'model.mlir')=='4814509b8e11a5c819b1f9ae63f01f89f72e9edf9c3d0ec9d5cd0ab6b35de2dc'
provider_dir=HERE/'numeric_frozen'
provider_recipe=json.loads((provider_dir/'reclosure.json').read_text())
for path,pin in provider_recipe['compilation_pins'].items():assert sha(path)==pin
bridge_dir=HERE/'bridge'
selected_native=Path('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed/native/validation.json')
assert json.loads(selected_native.read_text())['bitwise_mismatches']==0
implementation={'numeric_identity_validation':sha(HERE/'numeric_identity_validation.json'),'selected_numeric_full48_native':sha(selected_native),'provider_object':sha(provider_dir/'provider.o'),'provider_llvm':sha(provider_dir/'provider.ll'),'bridge_object':sha(bridge_dir/'bridge.o'),'bridge_llvm':sha(bridge_dir/'bridge.ll'),'complete_actual_group_receipt':sha(Path('/scratch/agustin/tmp/gemmini-frontier-composed-20261006/out/frontier_composed/candidate/spike_receipt.json')),'source_consumer_proof':sha(proof_path),'baseline_whole_native':sha(native),'baseline_complete_observations':sha(complete)}
implementation_pin=hashlib.sha256(json.dumps(implementation,sort_keys=True).encode()).hexdigest()
effect={'reads':11,'fresh_bf16_writer':True,'private_bytes':123012928,'alignment':64,'disjoint_workspace_inputs_output':True,'initializes_before_read':True,'borrowed_noescape':True,'synchronous':True,'publication':'only after complete source-consumer certification','refusal':'retained original compiler source body before publishing output','bridge_source':sha(bridge_dir/'bridge.c'),'baseline_native_full48_proof':sha(complete),'selected_normal_gate':'pending'}
effect_pin=hashlib.sha256(json.dumps(effect,sort_keys=True).encode()).hexdigest()
writer=ClosedGroupWriterContract(proof['complete_source_dag_semantic_sha256'],'attention_frontier_writer',implementation_pin,sha(proof_path),'exact_consumer_observations',effect_pin,64,True,True,True,True,'rne_returned_values','normal_actual_product_provider_pending_whole_target',private_workspaces=(PrivateWorkspaceContract(123012928,64,effect_pin,True,True,True),),source_fallback_symbol='source_attention_frontier_fallback')
consumer=ConsumerObservationContract(proof['complete_consumer_semantic_sha256'],('tensor<1x1024x768xi8>','tensor<1x1024xbf16>'),sha(proof_path),effect_pin,'rne_returned_values')
prior=json.loads((TAPS/'bounded_native/build/device_prepared/source_control/source_group_calls.json').read_text())
class ExactPreparation:
 def __call__(self,path,workdir):
  if sha(path)==prior['original_prepared_sha256'] and sha(prior['selected_path'])==prior['selected_sha256']:
   self.receipt=prior;return Path(prior['selected_path'])
  fresh=SourceExactGroupPreparation();result=fresh(path,workdir);self.receipt=fresh.receipt;return result
transform=ConsumerObservedGroupPreparation([writer],[consumer],source_preparation=ExactPreparation(),reuse_private_workspaces=True)
def prepare(path,workdir):
 selected=transform(path,workdir)
 assert transform.receipt['source_groups']==48 and transform.receipt['physical_writers']==1
 (HERE/'preparation.json').write_text(json.dumps(transform.receipt,indent=2)+'\n')
 return selected

def imported(path,ll,commands,pins,cwd):
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
 common=[str(context.compiler),*context.compiler_flags,'-I'+str(provider_dir),'-Dattention_frontier_writer=_mlir_ciface_attention_frontier_writer_borrowed']
 commands=[[*common,'-c',str(source),'-o',str(obj)],[*common,'-S','-emit-llvm',str(source),'-o',str(ll)]]
 for cmd in commands:context.compile(cmd,inputs=[source,provider_dir/'source_attention_frontier_api.h'],output=Path(cmd[cmd.index('-o')+1]))
 pins={str(source):sha(source),str(provider_dir/'source_attention_frontier_api.h'):sha(provider_dir/'source_attention_frontier_api.h'),str(Path(context.compiler).resolve()):sha(Path(context.compiler).resolve())}
 objects.append(imported(obj,ll,commands,pins,Path.cwd()))
 # Physical primitive imports preserve their source-derived target IR and object closure.
 for shape in ['qk','pv192','pv128']:
  for degree in range(5):
   d=HERE.parent/'artifacts/probes/closed-group-target-20261005/m256_h1'/f'{shape}_products_{degree}'
   r=json.loads((d/'device_compile.json').read_text());assert sha(d/'kernel.o')==r['object_sha256'];assert sha(d/'kernel.ll')==r['llvm_ir_sha256']
   pins={str(p):sha(p)for p in d.iterdir()if p.is_file()}
   for cmd in r['compiler_argv']:pins[str(Path(cmd[0]).resolve())]=sha(Path(cmd[0]).resolve())
   objects.append(imported(d/'kernel.o',d/'kernel.ll',r['compiler_argv'],pins,Path.cwd()))
 return HostProviderBuild(sha(context.prepared_path),sha(context.model_llvm_path),sha(context.model_object_path),tuple(objects),sha(proof_path),effect_pin)

(HERE/'declared_contracts.json').write_text(json.dumps({'writer':asdict(writer),'consumer':asdict(consumer),'implementation':implementation,'effect':effect},indent=2)+'\n')
device=DeviceRouting(device='gemmini',package_dir=str(Path.cwd()),operand_dtype='i8',accum_dtype='i32',catalog_builder=merlin_builder(LLVM),final_elf_audit=final_elf_audit,prepared_transform=prepare)
result=spike_model.build(SOURCE,HERE/'build',arena_mb=8192,stack_bytes=16*1024**2,output_dump_cap=1,output_sha256=True,dram_bytes=16*1024**3,int8_compute=True,features=frozenset({'respect_captured_quantization_scope','lower_fma_to_intrinsic','outline_llvm_loops'}),cflags_override=['-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin','-fno-vectorize','-fno-slp-vectorize'],device=device,host_math_policy='expf_via_double',host_provider_builder=host_provider)
(HERE/'build_result.json').write_text(json.dumps(result,default=str,indent=2)+'\n');print('NORMAL_ATTENTION_PROVIDER_BUILD_DONE',flush=True)
