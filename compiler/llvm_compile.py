"""Optional LLVM pipeline retaining the authored pre-tested C loop structure.

This pipeline deliberately omits loop rotation, loop unswitching, loop inversion,
induction-variable rewriting, vectorization and unrolling. It is an ordinary
optimizing compiler path; task markers and all source computations remain.
"""
from pathlib import Path
import subprocess
import tempfile

# Promote locals and fold helper configuration while keeping loop headers. The
# Scalar conditional diamonds become selects; keep-loops protects loop headers.
# The last CFG transformation is loop-simplify; no later pass rotates headers.
CANONICAL_LOOP_PIPELINE = (
    'always-inline,'
    'function(sroa,mem2reg,early-cse,instsimplify,'
    'simplifycfg<keep-loops;speculate-blocks;'
    'no-hoist-common-insts;no-sink-common-insts>,'
    'instsimplify,loop-simplify,lcssa),globaldce,verify'
)


def compile_unoptimized_c_to_llvm(source, clang, optpath):
    """Compile authored C to verified LLVM IR using the explicit loop pipeline.

    ``clang`` and ``optpath`` must refer to the same selected pinned toolchain.
    The caller continues through its normal metadata, pointer-ABI, LLVM-dialect
    import and task-attribution paths. The default compiler mode is unaffected.
    """
    clang, optpath = Path(clang), Path(optpath)
    if clang.parent.resolve() != optpath.parent.resolve():
        raise ValueError('canonical-loop compiler and optimizer must share the selected installation')
    with tempfile.TemporaryDirectory(prefix='gm-canonical-loops-') as temporary:
        root = Path(temporary)
        cpath, rawpath = root/'kernel.c', root/'raw.ll'
        cpath.write_text(source)
        command = [str(clang), '--target=riscv64-unknown-elf',
                   '-march=rv64gc_zba_zbb_zbs_zfh', '-mabi=lp64d',
                   '-mcmodel=medany', '-O2', '-Xclang', '-disable-llvm-passes',
                   '-ffreestanding', '-fno-builtin', '-fno-vectorize',
                   '-fno-slp-vectorize', '-fno-unroll-loops',
                   '-fno-discard-value-names', '-S', '-emit-llvm',
                   str(cpath), '-o', str(rawpath)]
        compile_result = subprocess.run(command,text=True,capture_output=True)
        if compile_result.returncode:
            raise ValueError('canonical-loop C lowering failed:\n'+compile_result.stderr)
        optimized = subprocess.run([str(optpath),'-S',
                    '-passes='+CANONICAL_LOOP_PIPELINE, str(rawpath),'-o','-'],
                    text=True,capture_output=True)
        if optimized.returncode:
            raise ValueError('canonical-loop optimization failed:\n'+optimized.stderr)
        return optimized.stdout
