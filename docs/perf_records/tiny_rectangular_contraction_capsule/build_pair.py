from pathlib import Path
from dataclasses import asdict
import hashlib, json, subprocess
import numpy as np
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scalar_contraction import EIGHT_OUTPUTS_FEATURE, RECTANGULAR_FEATURE, REDUCTION_UNROLL_FEATURE
from merlin.llvmlower.bufferized_result_identity import FEATURE as IDENTITY
from merlin.llvmlower.abi import HostModel
from merlin.perf.layer_bench import build_program, run_on_gsim
from mlir_oot.no_fsm_audit import audit_elf

work = Path(__file__).resolve().parent
llvm = Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
frozen = Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
runtime = Path('/scratch/agustin/tmp/merlin-tiny-qualified-pointwise-20261005/merlin/runtime/abi/mlir_runtime.c')
flags = ['--target=riscv64-unknown-elf', '-march=rv64gc', '-mabi=lp64d', '-mcmodel=medany', '-O3', '-ffreestanding', '-fno-builtin']
shapes = {'qk': [(32,8,64), (32,64,8), (32,8,8)], 'pv': [(32,8,8), (32,8,64), (32,8,64)]}
objects, proofs = [], []
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

for family in ['qk', 'pv']:
    a = np.fromfile(work/'capture'/f'{family}_a.bin', dtype='<f4').reshape(shapes[family][0])
    b = np.fromfile(work/'capture'/f'{family}_b.bin', dtype='<f4').reshape(shapes[family][1])
    expected = np.fromfile(work/'capture'/f'{family}_expected.bin', dtype='<f4').reshape(shapes[family][2])
    assert (work/'capture'/f'{family}_initial.bin').read_bytes() == bytes(expected.nbytes)
    for arm, feature in [('control', EIGHT_OUTPUTS_FEATURE), ('rectangular', RECTANGULAR_FEATURE)]:
        name, directory = family+'_'+arm, work/(family+'_'+arm)
        text = lower_to_llvm_ir((work/f'{family}_source.mlir').read_text(), workdir=directory,
                               features={feature, REDUCTION_UNROLL_FEATURE, IDENTITY})
        text = text.replace('forward', name).replace('dealloc_helper', name+'_dealloc_helper')
        (directory/'target.ll').write_text(text)
        subprocess.run([str(llvm/'clang'), '-O3', '-shared', '-fPIC', '-ffp-contract=off', str(directory/'target.ll'),
                        str(runtime), '-lm', '-o', str(directory/'model.so')], check=True, capture_output=True)
        out = np.zeros(expected.shape, dtype='f4')
        HostModel.load(str(directory/'model.so'), name=name)([(x.ctypes.data,x.shape) for x in [a,b,out]])
        assert np.array_equal(out.view('u4'), expected.view('u4'))
        subprocess.run([str(llvm/'clang'), *flags, '-ffp-contract=off', '-c', str(directory/'target.ll'),
                        '-o', str(directory/'model.o')], check=True, capture_output=True)
        objects.append(directory/'model.o')
        proofs.append({'family': family, 'arm': arm, 'features': sorted([feature, REDUCTION_UNROLL_FEATURE, IDENTITY]),
                       'llvm_sha256': sha(directory/'target.ll'), 'object_sha256': sha(directory/'model.o'),
                       'source_sha256': sha(work/f'{family}_source.mlir'), 'native_all_original_words': True,
                       'native_output_sha256': hashlib.sha256(out.astype('<f4').tobytes()).hexdigest()})
        print('ORIGINAL_CONTRACTION_NATIVE_PASS', name, out.size, flush=True)

assembly = '.section .rodata\n'
for family in ['qk', 'pv']:
    for kind in ['a', 'b', 'expected']:
        assembly += f'.balign 64\n.global {family}_{kind}\n{family}_{kind}:\n.incbin "{work}/capture/{family}_{kind}.bin"\n'
(work/'data.S').write_text(assembly)
subprocess.run([str(llvm/'clang'), *flags, '-c', str(work/'data.S'), '-o', str(work/'data.o')], check=True, capture_output=True)
objects += [work/'data.o', frozen/'mlir_rt.o']
for arm in ['strict', 'timing']:
    main_object = work/(arm+'_main.o')
    extra = ['-DSTRICT'] if arm == 'strict' else []
    subprocess.run([str(llvm/'clang'), *flags, '-ffp-contract=off', *extra, '-c', str(work/'main.c'),
                    '-o', str(main_object)], check=True, capture_output=True)
    build = build_program([*objects, main_object], work/(arm+'_build'), target='gemmini', max_loaded_bytes=None)
    audit = audit_elf(build.elf.read_bytes())
    assert audit['status'] == 'pass'
    result = subprocess.run(['/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike', '--isa=rv64gc',
                             '--extension=gemmini', str(build.elf)], capture_output=True, text=True, timeout=300)
    console = result.stdout+result.stderr
    (work/(arm+'_spike.log')).write_text(console)
    assert result.returncode == 0 and 'ORIGINAL_CONTRACTION_COMPLETE PASS' in console, console[-2000:]
    print('ACTUAL_TARGET_CONTRACTION_PASS', arm, flush=True)
    if arm == 'timing': timed = build

qualification = {
    'schema': 'source_exact_original_contraction_rectangular_complete_capsule_v1', 'status': 'pass', 'proofs': proofs,
    'source_witness_sha256': sha(work/'typed_source_witness.json'), 'original_whole_native_capture_sha256': sha(work/'capture/validation.json'),
    'all_original_qk2048_and_pv16384_words_exact': True, 'all_five_target_rounding_modes_and_stickyflags_match': True,
    'poisoned_destination_tails': 128, 'original_inputs_preserved': True, 'timing_elf_sha256': timed.elf_sha256,
    'runtime_object_sha256': sha(frozen/'mlir_rt.o'), 'compile_flags': flags,
    'scope': 'Complete first original32head/QK8x8x64 andPV8x64x8 strict separate-multiply/add contractions, sourcepositivezero seed initialization/public result ABI/allocation/copy/output included. Both sourceexact objects present in same timedELF, same operands/destination/allocator addresses. Reference/hash checks outsideROI; no fullmodel cycleprediction. Original1989 native capture all256000 words+Torch gate retained.',
    'token_usage_available': False,
}
(work/'qualification.json').write_text(json.dumps(qualification, indent=2)+'\n')
(work/'ready.json').write_text(json.dumps({'elf_path': str(timed.elf), 'elf_sha256': timed.elf_sha256, 'qualification_path': str(work/'qualification.json')}, indent=2)+'\n')
print('RECTANGULAR_COMPLETE_TIMING_READY', flush=True)
receipt = run_on_gsim(timed.elf, target='gemmini', timeout_s=2400, max_cycles=80000000, stdout_path=work/'gsim.stdout')
(work/'gsim_receipt.json').write_text(json.dumps(asdict(receipt), indent=2, default=str)+'\n')
print('RECTANGULAR_GSIM_RESULT', receipt, flush=True)
