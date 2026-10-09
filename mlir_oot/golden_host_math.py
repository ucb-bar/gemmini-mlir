"""Exact CPU math support for the explicitly selected stock RV64GC target."""
from xdsl.dialects.builtin import Float32Type
from xdsl.dialects import func
from xdsl.ir import Region

SYMBOL = 'gemmini_golden_roundevenf'


def rewrite_roundeven(module) -> int:
    """Route scalar f32 roundeven to the target's RNE helper; f64 stays upstream."""
    selected = [op for op in module.walk() if op.name == 'math.roundeven'
                and len(op.results) == 1 and isinstance(op.results[0].type, Float32Type)]
    if not selected:
        return 0
    if any(op.name == 'func.func' and op.sym_name.data == SYMBOL for op in module.walk()):
        raise ValueError('host rounding helper symbol already exists')
    for op in selected:
        replacement = func.CallOp(SYMBOL, list(op.operands), [op.results[0].type])
        replacement.attributes.update(op.attributes)
        op.parent.insert_op_before(replacement, op)
        op.results[0].replace_all_uses_with(replacement.results[0])
        op.parent.erase_op(op)
    ty = Float32Type()
    module.body.block.add_op(func.FuncOp(SYMBOL, ([ty], [ty]), Region(), visibility='private'))
    module.verify()
    return len(selected)


def relink_roundeven(build_dir, runtime_dir, gcc, llvm_bin, work):
    """Select exact RNE at final link, retaining the optimized model object.

    This explicitly targets the stock RV64GC bare-metal model ABI. No calls are
    inserted before elementwise fusion. The generated receipt binds every reused
    object, target linker constants, support assembly and final no-FSM audit.
    """
    import hashlib
    import json
    import re
    import subprocess
    from pathlib import Path
    from .no_fsm_audit import audit_elf

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    build_dir, runtime_dir, llvm_bin, work = map(Path, (build_dir, runtime_dir, llvm_bin, work))
    work.mkdir(parents=True, exist_ok=False)
    base = build_dir / 'model.elf'
    if audit_elf(base.read_bytes())['status'] != 'pass':
        raise ValueError('roundeven support requires an audited base ELF')
    catalog = json.loads((build_dir / 'device_catalog/device_catalog.json').read_text())
    matches = [p for p in (build_dir / 'device_catalog').glob('*.o')
               if sha(p) == catalog['compilation']['object_sha256']]
    if len(matches) != 1:
        raise ValueError('expected one exact compiled device catalog object')
    symbols = subprocess.check_output([str(llvm_bin / 'llvm-nm'), str(base)], text=True)
    constants = {}
    for name in ('MERLIN_WEIGHTS_BASE', 'MERLIN_STACK_BYTES'):
        match = re.search(rf'^([0-9a-fA-F]+) A {name}$', symbols, re.M)
        if match is None:
            raise ValueError(f'missing target linker constant {name}')
        constants[name] = int(match[1], 16)
    if re.search(r'^\S+ [Tt] __wrap_roundevenf$', symbols, re.M):
        raise ValueError('base ELF already defines the selected rounding wrapper')
    assembly = Path(__file__).resolve().parent.parent / 'runtime/roundevenf_rv64gc.S'
    source = work / 'roundeven.S'
    source.write_text(assembly.read_text().replace(SYMBOL, '__wrap_roundevenf'))
    obj = work / 'roundeven.o'
    compiler = [str(llvm_bin / 'clang'), '--target=riscv64-unknown-elf', '-march=rv64gc',
                '-mabi=lp64d', '-c', str(source), '-o', str(obj)]
    subprocess.run(compiler, check=True, capture_output=True, text=True)
    names = ['model_call.o', 'merlin_model.o', 'model_main.o', 'mlir_rt.o', 'crt.o',
             'console.o', 'libc_min.o', 'malloc.o', 'model.o', 'weights_blob.o']
    objects = [build_dir / x for x in names] + matches + [build_dir / 'device/device_catalog_shim.o']
    flags = ['-march=rv64gc', '-mabi=lp64d', '-mcmodel=medany', '-O2', '-ffreestanding', '-fno-builtin']
    elf = work / 'model.elf'
    command = [str(gcc), *flags, '-nostdlib', '-nostartfiles',
               *[f'-Wl,--defsym,{k}={hex(v)}' for k, v in constants.items()],
               '-T', str(runtime_dir / 'baremetal/spike/model_link.ld'),
               *map(str, objects), str(obj), '-Wl,--wrap=roundevenf', '-lm', '-o', str(elf)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    audit = audit_elf(elf.read_bytes())
    (work / 'model.nofsm_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if audit['status'] != 'pass':
        raise ValueError('roundeven-linked model failed zero-FSM audit')
    receipt = dict(schema='gemmini_late_roundeven_link_v1', base_elf=str(base),
        base_elf_sha256=sha(base), elf_sha256=sha(elf),
        model_object_sha256=sha(build_dir / 'model.o'), source_sha256=sha(source),
        objects={str(p): sha(p) for p in objects}, compiler_argv=compiler, linker_argv=command,
        scope='Exact RV64GC RNE support selected at final link; optimized model object unchanged')
    (work / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Link exact RV64GC rounding after model optimization')
    for name in ('build-dir', 'runtime-dir', 'gcc', 'llvm-bin', 'workdir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    receipt = relink_roundeven(args.build_dir, args.runtime_dir, args.gcc, args.llvm_bin, args.workdir)
    print(json.dumps({'elf_sha256': receipt['elf_sha256']}, indent=2))
