"""Run the audited GEMM probe on explicitly supplied immutable captured operands."""
import argparse,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);args,remaining=p.parse_known_args();receipt=json.loads(args.receipt.read_text());assert hashlib.sha256(args.fixture.read_bytes()).hexdigest()==receipt['fixture_sha256']
data=np.load(args.fixture);a=data['a'];b=data['b'];assert a.dtype==b.dtype==np.int8 and a.shape[1]==b.shape[0]
assert hashlib.sha256(a.tobytes()).hexdigest()==receipt['activation_sha256'];assert hashlib.sha256(b.tobytes()).hexdigest()==receipt['weights_sha256'];expected=a.astype(np.int32)@b.astype(np.int32)
spec=importlib.util.spec_from_file_location('gsim_gemm',Path(__file__).with_name('gsim_gemm_probe.py'));probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

def check(shape):
    assert (shape.m,shape.n,shape.k)==(a.shape[0],b.shape[1],a.shape[1])
    assert shape.output_dtype=='i32' and not shape.bias


def write_expected(path,shape,amplitude):
    check(shape);path.write_text('static const int32_t expected_values[M][N] = {\n'+''.join('{'+','.join(map(str,row))+'},\n' for row in expected)+'};\n')


def static_inputs(workdir,llvm_bin,shape,amplitude):
    check(shape);text=[]
    for name,v in [('a',a),('b',b)]:
        path=workdir/(name+'.bin');v.tofile(path);text.append(f'.section .data\n.balign 64\n.global {name}\n{name}:\n.incbin "{path}"\n')
    asm=workdir/'input_data.S';asm.write_text(''.join(text));obj=workdir/'input_data.o';subprocess.run([str(llvm_bin/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-c',str(asm),'-o',str(obj)],check=True);(workdir/'capture_fixture_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return obj

probe._write_expected=write_expected;probe._static_inputs=static_inputs
sys.argv=[sys.argv[0],*remaining,'--embed-expected','--static-inputs']
raise SystemExit(probe.main())
