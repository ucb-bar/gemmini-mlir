"""Explicit RV64GC legalization of an exact bounded binary32 RNE-to-i8 idiom.

Runs after upstream tensor fusion/lowering. Match the entire typed SSA chain,
including its exact saturation endpoints. Unknown forms remain unchanged.
NaNs yield poison in the original fptosi; infinities clamp to finite endpoints.
This preserves every source-defined integer result, independently of frm.
The source uses unconstrained floating operations (no fflags contract).
"""
import hashlib
import re

V = r'%[-a-zA-Z$._0-9]+'


def _rewrite_function(text: str, *, native_oracle=False, combine_clamp=False):
    lines=text.splitlines(keepends=True); definitions={}
    for line in lines:
        m=re.match(r'\s*('+V+r') = (.*)',line)
        if m: definitions[m[1]]=m[2]
    proofs=[];out=[]
    # Match backwards from integer add; SSA values must connect at every step.
    def operands(value, pattern):
        return re.fullmatch(pattern,definitions.get(value,''))
    for line in lines:
        m=re.fullmatch(r'(\s*)('+V+r') = add i8 ('+V+r'), ('+V+r')\n?',line)
        if not m:out.append(line);continue
        indent,result,trunc,bump=m.groups()
        t=operands(trunc,r'fptosi float ('+V+r') to i8')
        b=operands(bump,r'select i1 ('+V+r'), i8 ('+V+r'), i8 0')
        if not t or not b:out.append(line);continue
        x=t[1];trigger,sign=b.groups()
        si=operands(sign,r'select i1 ('+V+r'), i8 -1, i8 1')
        tr=operands(trigger,r'or i1 ('+V+r'), ('+V+r')')
        hi=operands(x,r'call float @llvm.minimum.f32\(float ('+V+r'), float 1.270000e\+02\)')
        if not si or not tr or not hi:out.append(line);continue
        lo=operands(hi[1],r'call float @llvm.maximum.f32\(float ('+V+r'), float -1.280000e\+02\)')
        if not lo or definitions.get(si[1])!=f'fcmp olt float {x}, 0.000000e+00':out.append(line);continue
        gt=operands(tr[1],r'fcmp ogt float ('+V+r'), 5.000000e-01')
        tie=operands(tr[2],r'and i1 ('+V+r'), ('+V+r')')
        if not gt or not tie:out.append(line);continue
        absolute=operands(gt[1],r'call float @llvm.maximum.f32\(float ('+V+r'), float ('+V+r')\)')
        parity=operands(tie[2],r'icmp ne i8 ('+V+r'), 0')
        if not absolute or not parity or definitions.get(tie[1])!=f'fcmp oeq float {gt[1]}, 5.000000e-01':out.append(line);continue
        frac,negative=absolute.groups()
        diff=operands(frac,r'fsub float '+re.escape(x)+r', ('+V+r')')
        if (not diff or definitions.get(diff[1])!=f'sitofp i8 {trunc} to float'
            or definitions.get(negative)!=f'fneg float {frac}'
            or definitions.get(parity[1])!=f'and i8 {trunc}, 1'):
            out.append(line);continue
        temp=f'%gemmini.rne.{len(proofs)}'
        if temp in definitions:raise ValueError('reserved RNE temporary already exists')
        if native_oracle:
            out.append(f'{indent}{temp} = call float @llvm.roundeven.f32(float {x})\n')
            out.append(f'{indent}{result} = fptosi float {temp} to i8\n')
        elif combine_clamp:
            out.append(f'{indent}{temp} = call i32 asm "fmax.s ft0, $1, $2\\0Afmin.s ft0, ft0, $3\\0Afcvt.w.s $0, ft0, rne", "=r,f,f,f,~{{ft0}}"(float {lo[1]}, float -1.280000e+02, float 1.270000e+02)\n')
            out.append(f'{indent}{result} = trunc i32 {temp} to i8\n')
        else:
            out.append(f'{indent}{temp} = call i32 asm "fcvt.w.s $0, $1, rne", "=r,f"(float {x})\n')
            out.append(f'{indent}{result} = trunc i32 {temp} to i8\n')
        proofs.append(dict(result=result,clamped_input=x,raw_input=lo[1],dtype='f32',bounds=[-128,127],rounding='nearest_ties_even',source_nan='poison_from_fptosi'))
    rewritten=''.join(out)
    return rewritten,dict(schema='gemmini_late_bounded_rne_v1',source_sha256=hashlib.sha256(text.encode()).hexdigest(),rewritten_sha256=hashlib.sha256(rewritten.encode()).hexdigest(),routes=proofs)


def rewrite(text: str, *, native_oracle=False, combine_clamp=False):
    proofs=[]
    if re.search(r"\bstrictfp\b|llvm\.experimental\.constrained",text):
        return text,dict(schema="gemmini_late_bounded_rne_v1",routes=[],refusal="strict or constrained FP module",source_sha256=hashlib.sha256(text.encode()).hexdigest())
    def replace_function(match):
        body,receipt=_rewrite_function(match[0],native_oracle=native_oracle,combine_clamp=combine_clamp)
        proofs.extend(receipt['routes'])
        return body
    rewritten=re.sub(r'^define [^\n]*\{\n.*?^}[^\n]*',replace_function,text,flags=re.M|re.S)
    if native_oracle and proofs and not re.search(r'^declare float @llvm.roundeven.f32\(',rewritten,re.M):
        rewritten+='\ndeclare float @llvm.roundeven.f32(float)\n'
    return rewritten,dict(schema='gemmini_late_bounded_rne_v1',source_sha256=hashlib.sha256(text.encode()).hexdigest(),rewritten_sha256=hashlib.sha256(rewritten.encode()).hexdigest(),routes=proofs)


def merlin_host_llvm_transform(llvm_bin, *, combine_clamp=False):
    """Select proved legalization before Merlin hashes and links the model object.

    The normal backend owns the compiler flags, object build, harness identity
    and final linking. This package owns recognition, exact RNE instructions,
    boundary proofs and the paired portable native oracle.
    """
    from pathlib import Path
    import json, subprocess

    llvm_bin = Path(llvm_bin)

    def transform(source, work):
        source, work = Path(source), Path(work)
        work.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(llvm_bin/'llvm-as'),str(source),'-o',str(work/'source.bc')],check=True)
        original = source.read_text()
        target, proof = rewrite(original, combine_clamp=combine_clamp)
        if not proof['routes']:
            raise ValueError('no proven bounded RNE chain: '+str(proof.get('refusal', 'unrecognized source')))
        selected = work/'model.ll'
        selected.write_text(target)
        subprocess.run([str(llvm_bin/'llvm-as'),str(selected),'-o',str(work/'model.bc')],check=True)
        native, _ = rewrite(original, native_oracle=True)
        (work/'model.native.ll').write_text(native)
        subprocess.run([str(llvm_bin/'llvm-as'),str(work/'model.native.ll'),'-o',str(work/'model.native.bc')],check=True)
        proof.update(combine_clamp=combine_clamp,
                     llvm_as_sha256=hashlib.sha256((llvm_bin/'llvm-as').read_bytes()).hexdigest(),
                     native_oracle_sha256=hashlib.sha256(native.encode()).hexdigest(),
                     scope='pre-object host legalization; normal Merlin compilation and build identity')
        (work/'receipt.json').write_text(json.dumps(proof,indent=2)+'\n')
        return selected

    return transform


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
