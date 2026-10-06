from pathlib import Path
import json
from merlin.llvmlower.device_build import DeviceRouting
from merlin.runtime.backends import spike_model
from mlir_oot.golden_device_catalog import merlin_builder,final_elf_audit
w=Path(__file__).resolve().parent
llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
device=DeviceRouting(device='gemmini',package_dir='/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005',operand_dtype='i8',accum_dtype='i32',catalog_builder=merlin_builder(llvm),final_elf_audit=final_elf_audit)
original_run=spike_model._run
def native_only(cmd,**kwargs):
 args=list(map(str,cmd))
 if '-c' in args and args[args.index('-c')+1].endswith('.ll') and '-o' in args and args[args.index('-o')+1].endswith('/model.o'):
  (w/'native_ready.json').write_text(json.dumps({'target_command_deferred':args},indent=2));print('NATIVE_READY',flush=True);raise SystemExit(0)
 return original_run(cmd,**kwargs)
spike_model._run=native_only
spike_model.build(w/'bundle',w/'build',arena_mb=8192,stack_bytes=16*1024**2,output_dump_cap=1600,dram_bytes=16*1024**3,int8_compute=True,features=frozenset({'respect_captured_quantization_scope','lower_fma_to_intrinsic','outline_llvm_loops'}),cflags_override=['-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin','-fno-vectorize','-fno-slp-vectorize'],device=device,host_math_policy='expf_via_double')
