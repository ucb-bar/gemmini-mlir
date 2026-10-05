"""Compatibility adapter and legacy target relink for Merlin's host RNE codegen.

Reusable typed recognition, CPU instruction emission, native oracle and normal
pre-object transformation belong to Merlin. This provider preserves historical
SSA printer identity. Legacy linked-object selection and no-FSM auditing remain
here; they are not the normal production build path.
"""
import hashlib
import re

def rewrite(text: str, *, native_oracle=False, combine_clamp=False):
    """Compatibility API; reusable recognition and host instructions live in Merlin."""
    from merlin.llvmlower.late_quant_rne import rewrite as host_rewrite

    return host_rewrite(
        text, host_isa='portable' if native_oracle else 'rv64gc',
        combine_clamp=combine_clamp and not native_oracle,
        # Preserve historical printed SSA identity without naming a target in core.
        temporary_prefix='gemmini.rne',
    )


def merlin_host_llvm_transform(llvm_bin, *, combine_clamp=False):
    """Select Merlin's explicit RV64GC host legalization before object identity."""
    from merlin.llvmlower.late_quant_rne import merlin_host_llvm_transform as host_transform

    return host_transform(llvm_bin, host_isa='rv64gc', combine_clamp=combine_clamp,
                          temporary_prefix='gemmini.rne')


def build(build_dir, work, llvm_bin, gcc, runtime_dir, *, combine_clamp=False):
    """Opt-in late model-object replacement; all other linked objects are pinned."""
    from pathlib import Path
    import json,subprocess
    from .no_fsm_audit import audit_elf
    build_dir,work,llvm_bin,runtime_dir=map(Path,(build_dir,work,llvm_bin,runtime_dir))
    work.mkdir(parents=True,exist_ok=False)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    source=build_dir/'lower/model.ll'
    # LLVM verifies the original and the replacement; textual matching is never
    # used as a substitute for SSA/type verification.
    subprocess.run([str(llvm_bin/'llvm-as'),str(source),'-o',str(work/'source.bc')],check=True)
    original=source.read_text();target,proof=rewrite(original,combine_clamp=combine_clamp)
    if not proof['routes']:raise ValueError('no proven bounded RNE chain')
    ll=work/'model.ll';ll.write_text(target)
    subprocess.run([str(llvm_bin/'llvm-as'),str(ll),'-o',str(work/'model.bc')],check=True)
    native,_=rewrite(original,native_oracle=True);(work/'model.native.ll').write_text(native)
    flags=['-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin']
    compile=[str(llvm_bin/'clang'),'--target=riscv64-unknown-elf',*flags,'-c',str(ll),'-o',str(work/'model.o')]
    subprocess.run(compile,check=True)
    base=build_dir/'model.elf'
    if audit_elf(base.read_bytes())['status']!='pass':raise ValueError('base ELF failed no-FSM audit')
    catalog=json.loads((build_dir/'device_catalog/device_catalog.json').read_text())
    matches=[p for p in (build_dir/'device_catalog').glob('*.o') if sha(p)==catalog['compilation']['object_sha256']]
    if len(matches)!=1:raise ValueError('ambiguous device catalog object')
    symbols=subprocess.check_output([str(llvm_bin/'llvm-nm'),str(base)],text=True)
    constants={}
    for name in ('MERLIN_WEIGHTS_BASE','MERLIN_STACK_BYTES'):
        m=re.search(rf'^([0-9a-fA-F]+) A {name}$',symbols,re.M)
        if m is None:raise ValueError('missing linker constant '+name)
        constants[name]=int(m[1],16)
    names=['model_call.o','merlin_model.o','model_main.o','mlir_rt.o','crt.o','console.o','libc_min.o','malloc.o','weights_blob.o']
    objects=[build_dir/n for n in names]+matches+[build_dir/'device/device_catalog_shim.o',work/'model.o']
    link=[str(gcc),*flags,'-nostdlib','-nostartfiles',*[f'-Wl,--defsym,{k}={hex(v)}' for k,v in constants.items()],'-T',str(runtime_dir/'baremetal/spike/model_link.ld'),*map(str,objects),'-lm','-o',str(work/'model.elf')]
    subprocess.run(link,check=True)
    audit=audit_elf((work/'model.elf').read_bytes())
    if audit['status']!='pass':raise ValueError('final ELF failed no-FSM audit')
    proof.update(combine_clamp=combine_clamp,base_elf_sha256=sha(base),elf_sha256=sha(work/'model.elf'),objects={str(p):sha(p) for p in objects},compiler_argv=compile,linker_argv=link,nofsm_audit=audit)
    (work/'receipt.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','work','llvm-bin','gcc','runtime-dir'):p.add_argument('--'+name,required=True)
    p.add_argument("--combine-clamp",action="store_true")
    a=p.parse_args();r=build(a.build_dir,a.work,a.llvm_bin,a.gcc,a.runtime_dir,combine_clamp=a.combine_clamp)
    print(json.dumps({'routes':len(r['routes']),'elf_sha256':r['elf_sha256']},indent=2))
