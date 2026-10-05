"""Build and validate fused ResNet catalog with native and actual Spike execution.

Requires MERLIN integration src on PYTHONPATH and its standard compiler environment.
Uses the supplied precompiled, pinned requant bundle; no fresh proof is inferred.
"""
import argparse,hashlib,json,os,subprocess,struct
from pathlib import Path
import numpy as np
from mlir_oot.fused_mixed_catalog import stage_capture,merlin_callbacks
from mlir_oot.golden_device_catalog import final_elf_audit
from merlin.runtime.backends.spike_model import build
from merlin.llvmlower.device_build import DeviceRouting
from merlin.runtime.dispatch_runtime import resolve_forward_args
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c


def native_validate(capture,build,host,llvm_bin):
    host.mkdir(parents=True,exist_ok=True);catalog=json.loads((build/'device_catalog/device_catalog.json').read_text())
    source=['#include <stdint.h>','#include <string.h>']
    for row in catalog['kernels']:
        d=row['dimensions'];b,m,n,k=(d[x] for x in ['batch','m','n','k']);sym=row['symbol']
        source.append(f'''void {sym}(const int8_t*a,const int8_t*b,int32_t*c) {{
 memset(c,0,{b*m*n}*sizeof(int32_t));
 for(int batch=0;batch<{b};++batch)for(int i=0;i<{m};++i)for(int q=0;q<{k};++q){{
 int32_t v=a[(batch*{m}+i)*{k}+q];for(int j=0;j<{n};++j)c[(batch*{m}+i)*{n}+j]+=v*(int32_t)b[(batch*{k}+q)*{n}+j];}} }}''')
    (host/'reference.c').write_text('\n'.join(source))
    subprocess.run([str(llvm_bin/'clang'),'-O0','-fPIC','-c',str(build/'lower/model.ll'),'-o',str(host/'model.o')],check=True)
    subprocess.run(['cc','-O3','-march=native','-fPIC','-shared',str(host/'model.o'),str(host/'reference.c'),*catalog['native_oracle_sources'],str(build/'device/device_catalog_shim.c'),str(mlir_runtime_c()),'-lm','-o',str(host/'model.so')],check=True)
    args=resolve_forward_args(capture);reference=np.load(capture/'golden.npy');actual=np.zeros_like(reference)
    model=HostModel.load(str(host/'model.so'));model([(a.ctypes.data,a.shape) for a in args]+[(actual.ctypes.data,actual.shape)])
    report={'scope':'native whole model with scalar device standins','elements':actual.size,'exact_equal':bool(np.array_equal(actual,reference)),'max_abs_error':float(np.max(np.abs(actual-reference)))}
    (host/'validation.json').write_text(json.dumps(report,indent=2)+'\n');np.save(host/'output.npy',actual)
    if not report['exact_equal']:raise ValueError('native whole model differs')
    return report


def spike_validate(capture,build,spike,work):
    command=[str(spike),'--extension=gemmini','--isa=rv64gc','-m0x80000000:0x80000000',str(build/'model.elf')]
    result=subprocess.run(command,capture_output=True,timeout=3600);console=(result.stdout+result.stderr).decode(errors='replace');(work/'spike.log').write_text(console)
    if result.returncode or 'DONE' not in console:raise ValueError('Spike did not finish correctly')
    line=next(x for x in console.splitlines() if x.startswith('OUT '));parts=line.split();count=int(parts[1]);bits=[int(x)&0xffffffff for x in parts[2:2+count]]
    actual=np.array([struct.unpack('<f',struct.pack('<I',x))[0] for x in bits],np.float32);reference=np.load(capture/'golden.npy').reshape(-1)
    report={'scope':'actual Gemmini Spike functional; cycles are retired instructions, not FireSim cycles','elf_sha256':hashlib.sha256((build/'model.elf').read_bytes()).hexdigest(),'elements':actual.size,'exact_equal':bool(np.array_equal(actual,reference)),'metrics':[x for x in console.splitlines() if x.startswith('METRIC ')]}
    (work/'spike_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['exact_equal']:raise ValueError('Spike whole model differs')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('bundle',type=Path);p.add_argument('--work',type=Path,required=True);p.add_argument('--llvm-bin',type=Path,required=True);p.add_argument('--spike',type=Path,required=True);p.add_argument('--validate-existing',action='store_true');p.add_argument('--packed-stem',action='store_true');p.add_argument('--pooled-stem',action='store_true');p.add_argument('--flat-spatial',action='store_true');a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    capture=a.work/'capture';builddir=a.work/'build_direct'
    if not a.validate_existing:
        (a.work/'host_compilation_policy.json').write_text(json.dumps({key:os.environ.get(key) for key in ['MERLIN_GENERALIZE_BEFORE_FUSE','MERLIN_FUSE_POST','MERLIN_CLANG']},indent=2)+'\n')
    if not a.validate_existing:
        stage_capture(a.capture,a.bundle,capture)
        callbacks=merlin_callbacks
        if a.packed_stem:
            from mlir_oot.stem_mixed_catalog import merlin_callbacks as callbacks
        if a.pooled_stem:
            if a.packed_stem:raise ValueError('choose packed or pooled stem')
            from mlir_oot.stem_pool_bundle import build as build_pool,apply_capture
            from mlir_oot.stem_pool_mixed_catalog import merlin_callbacks as pool_callbacks
            pool=a.work/'stem_pool_bundle';build_pool(capture,a.llvm_bin,pool);apply_capture(capture,pool)
            prepare,compile=pool_callbacks(a.llvm_bin,a.bundle,pool,flat_spatial=a.flat_spatial)
        else:
            if a.flat_spatial and a.packed_stem:raise ValueError('flat spatial option currently composes with pooled stem or ordinary fused catalog')
            prepare,compile=callbacks(a.llvm_bin,a.bundle,**({'flat_spatial':True} if a.flat_spatial else {}))
        result=build(capture,builddir,int8_compute=True,features=frozenset({'named_int8_contraction'}),cflags_override=['-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin'],device=DeviceRouting('gemmini',str(Path(__file__).resolve().parents[1]),'int8','i32',prepared_transform=prepare,catalog_builder=compile,final_elf_audit=final_elf_audit),dram_bytes=2*1024**3,arena_mb=256,stack_bytes=16*1024**2,console='htif')
        (a.work/'build_result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(native_validate(capture,builddir,a.work/'host',a.llvm_bin),flush=True)
    print(spike_validate(capture,builddir,a.spike,a.work),flush=True)

if __name__=='__main__':main()
