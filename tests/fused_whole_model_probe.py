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


def forward_args(capture, build):
    """Bind the same trailing hoisted tensors used by the bare-metal harness."""
    from merlin.llvmlower.quant_hoist import read_plan, read_values
    args=list(resolve_forward_args(capture))
    plan=read_plan(build)
    if plan:
        values=read_values(build)
        dtypes={'i8':np.dtype('int8'),'i32':np.dtype('int32'),'i64':np.dtype('int64'),
                'f16':np.dtype('float16'),'bf16':np.dtype('uint16'),
                'f32':np.dtype('float32'),'f64':np.dtype('float64')}
        try:
            for arg in plan:
                if arg.key not in values:
                    raise ValueError('hoisted native argument bytes are missing: '+arg.key)
                value=np.ascontiguousarray(values[arg.key])
                if (value.shape!=arg.shape or arg.dtype not in dtypes
                        or value.dtype!=dtypes[arg.dtype]):
                    raise ValueError('hoisted native argument shape or dtype differs: '+arg.key)
                args.append(value)
        finally:
            if hasattr(values,'close'):values.close()
    return args


def quality(actual, reference, *, allow_bounded=False, atol=0., rtol=0.):
    if actual.shape != reference.shape:raise ValueError('whole output shape differs')
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol<0 or rtol<0:raise ValueError('quality tolerances must be finite and nonnegative')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():raise ValueError('whole outputs and original golden must be finite')
    exact=bool(np.array_equal(actual.view(np.uint32),reference.astype(np.float32).view(np.uint32)))
    delta=actual.astype(np.float64)-reference.astype(np.float64)
    rel=float(np.linalg.norm(delta.reshape(-1))/max(np.linalg.norm(reference.astype(np.float64).reshape(-1)),np.finfo(np.float64).tiny))
    close=bool(np.allclose(actual,reference,atol=atol,rtol=rtol))
    return dict(elements=actual.size,exact_equal=exact,max_abs_error=float(np.max(np.abs(delta))),relative_l2=rel,atol=atol,rtol=rtol,allclose=close,
                quality_pass=close if allow_bounded else exact,numeric_policy='explicit_bounded_experiment' if allow_bounded else 'exact_source')


def native_validate(capture,build,host,llvm_bin,*,allow_bounded=False,atol=0.,rtol=0.,enforce_quality=True):
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
    args=forward_args(capture,build);reference=np.load(capture/'golden.npy');actual=np.zeros_like(reference)
    model=HostModel.load(str(host/'model.so'));model([(a.ctypes.data,a.shape) for a in args]+[(actual.ctypes.data,actual.shape)])
    report={'scope':'native whole model with scalar device standins','original_golden':str((capture/'golden.npy').resolve()),'original_golden_sha256':hashlib.sha256((capture/'golden.npy').read_bytes()).hexdigest(),**quality(actual,reference,allow_bounded=allow_bounded,atol=atol,rtol=rtol)}
    (host/'validation.json').write_text(json.dumps(report,indent=2)+'\n');np.save(host/'output.npy',actual)
    if enforce_quality and not report['quality_pass']:raise ValueError('native whole-model quality gate failed')
    return report


def spike_validate(capture,build,spike,work,*,allow_bounded=False,atol=0.,rtol=0.):
    command=[str(spike),'--extension=gemmini','--isa=rv64gc','-m0x80000000:0x80000000',str(build/'model.elf')]
    result=subprocess.run(command,capture_output=True,timeout=3600);console=(result.stdout+result.stderr).decode(errors='replace');(work/'spike.log').write_text(console)
    if result.returncode or 'DONE' not in console:raise ValueError('Spike did not finish correctly')
    lines=[x for x in console.splitlines() if x.startswith('OUT ')]
    if len(lines)!=1:raise ValueError('expected exactly one full OUT record')
    parts=lines[0].split();count=int(parts[1])
    if count<0 or len(parts)!=count+2:raise ValueError('Spike output record count differs')
    bits=[int(x)&0xffffffff for x in parts[2:]]
    actual=np.array([struct.unpack('<f',struct.pack('<I',x))[0] for x in bits],np.float32);reference=np.load(capture/'golden.npy').reshape(-1)
    native=np.load(work/'host/output.npy').reshape(-1)
    if count!=native.size or count!=reference.size or actual.size!=count:raise ValueError('Spike full output count differs')
    target_native_exact=bool(np.array_equal(actual.view(np.uint32),native.view(np.uint32)))
    report={'scope':'actual Gemmini Spike functional; cycles are retired instructions, not FireSim cycles','elf_sha256':hashlib.sha256((build/'model.elf').read_bytes()).hexdigest(),'target_native_exact':target_native_exact,'native_reference_sha256':hashlib.sha256((work/'host/output.npy').read_bytes()).hexdigest(),'original_golden_sha256':hashlib.sha256((capture/'golden.npy').read_bytes()).hexdigest(),**quality(actual,reference,allow_bounded=allow_bounded,atol=atol,rtol=rtol),'metrics':[x for x in console.splitlines() if x.startswith('METRIC ')]}
    if 'METRIC memref_rank_mismatch 0' not in console:raise ValueError('target descriptor rank mismatch')
    (work/'spike_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not target_native_exact:raise ValueError('Spike differs from same-candidate native output')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('bundle',type=Path);p.add_argument('--work',type=Path,required=True);p.add_argument('--llvm-bin',type=Path,required=True);p.add_argument('--spike',type=Path,required=True);p.add_argument('--validate-existing',action='store_true');p.add_argument('--packed-stem',action='store_true');p.add_argument('--pooled-stem',action='store_true');p.add_argument('--flat-spatial',action='store_true');p.add_argument('--hoist-weights',action='store_true');p.add_argument('--residual-add',action='store_true');p.add_argument('--residual-max-output-lsb',type=int,choices=(0,1),default=0);p.add_argument('--allow-bounded-output',action='store_true');p.add_argument('--atol',type=float,default=0.);p.add_argument('--rtol',type=float,default=0.);a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    if a.residual_max_output_lsb and (not a.residual_add or not a.allow_bounded_output):p.error('bounded residuals require --residual-add and explicit --allow-bounded-output')
    if not np.isfinite(a.atol) or not np.isfinite(a.rtol) or a.atol<0 or a.rtol<0:p.error('quality tolerances must be finite and nonnegative')
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
        if a.residual_add:
            from mlir_oot.captured_residual_bundle import build as build_residual
            from mlir_oot.residual_mixed_catalog import apply_capture as apply_residual,merlin_callbacks as residual_callbacks
            residual=a.work/'residual_bundle';build_residual(capture,a.llvm_bin,residual,max_output_lsb=a.residual_max_output_lsb)
            apply_residual(capture,residual)
            prepare,compile=residual_callbacks(a.llvm_bin,residual,(prepare,compile))
        (a.work/'whole_quality_policy.json').write_text(json.dumps({'allow_bounded_output':a.allow_bounded_output,'atol':a.atol,'rtol':a.rtol,'original_golden_sha256':hashlib.sha256((a.capture/'golden.npy').read_bytes()).hexdigest()},indent=2)+'\n')
        features={'named_int8_contraction'}
        if a.hoist_weights:features.add('hoist_weight_invariant_quantize')
        result=build(capture,builddir,int8_compute=True,features=frozenset(features),cflags_override=['-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin'],device=DeviceRouting('gemmini',str(Path(__file__).resolve().parents[1]),'int8','i32',prepared_transform=prepare,catalog_builder=compile,final_elf_audit=final_elf_audit),dram_bytes=2*1024**3,arena_mb=256,stack_bytes=16*1024**2,console='htif')
        (a.work/'build_result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    native=native_validate(capture,builddir,a.work/'host',a.llvm_bin,allow_bounded=a.allow_bounded_output,atol=a.atol,rtol=a.rtol,enforce_quality=False);print(native,flush=True)
    target=spike_validate(capture,builddir,a.spike,a.work,allow_bounded=a.allow_bounded_output,atol=a.atol,rtol=a.rtol);print(target,flush=True)
    if not native['quality_pass'] or not target['quality_pass']:raise ValueError('explicit whole-model quality gate failed; original-golden metrics retained')

if __name__=='__main__':main()
