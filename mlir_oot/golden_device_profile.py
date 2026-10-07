"""Opt-in final-link device attribution, preserving the optimized model object."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import re
import subprocess
from pathlib import Path
from .no_fsm_audit import audit_elf


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def emit(symbols: list[tuple[str, int]]) -> str:
    for symbol, arity in symbols:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', symbol) or arity not in (3, 4, 5):
            raise ValueError('profile supports only declared void pointer ABIs of arity3/4/5')
    out = ['''#include <stdint.h>
#include "merlin_model.h"
#include "htif.h"
#define CAP 4096
static uint64_t gaps[CAP], times[CAP], ids[CAP];
static uint64_t count, previous, total, device, tail;
static int active, overflow;
static inline uint64_t clock_now(void) {
 uint64_t v; __asm__ volatile("csrr %0, mcycle" : "=r"(v) :: "memory"); return v;
}
static void finish(uint64_t id, uint64_t begin, uint64_t end) {
 if (!active) return;
 if (count < CAP) { ids[count]=id; gaps[count]=begin-previous; times[count]=end-begin; }
 else overflow=1;
 ++count; device+=end-begin; previous=end;
}
extern void __real_merlin_run_multi(const merlin_arg_t*,int,const void*,void*const*,void*const*,merlin_descriptor_t*);
void __wrap_merlin_run_multi(const merlin_arg_t *a,int n,const void *w,void*const*i,void*const*o,merlin_descriptor_t*d) {
 count=0;device=0;overflow=0;active=1;uint64_t begin=clock_now();previous=begin;
 __real_merlin_run_multi(a,n,w,i,o,d);
 uint64_t end=clock_now();active=0;total=end-begin;tail=end-previous;
}
extern void __real_htif_exit(int) __attribute__((noreturn));
void __wrap_htif_exit(int code) {
 htif_line_flush(0);
 htif_puts("PROFILE_SUM ");htif_putd(total);htif_putc(' ');htif_putd(device);htif_putc(' ');
 htif_putd(total-device);htif_putc(' ');htif_putd(count);htif_putc(' ');htif_putd(overflow);htif_putc('\\n');
 for(uint64_t j=0;j<count && j<CAP;j++) {
  htif_puts("PROFILE_CALL ");htif_putd(j);htif_putc(' ');htif_putd(ids[j]);htif_putc(' ');
  htif_putd(gaps[j]);htif_putc(' ');htif_putd(times[j]);htif_putc('\\n');
 }
 htif_puts("PROFILE_TAIL ");htif_putd(tail);htif_putc('\\n');htif_line_flush(1);
 __real_htif_exit(code);
}
''']
    for idx, (symbol, arity) in enumerate(symbols):
        decl=', '.join(f'void *a{i}' for i in range(arity))
        call=', '.join(f'a{i}' for i in range(arity))
        out.append(f'extern void __real_{symbol}({decl});\n'
                   f'void __wrap_{symbol}({decl}) {{\n'
                   f' uint64_t begin=clock_now(); __real_{symbol}({call});\n'
                   f' uint64_t end=clock_now(); finish({idx},begin,end);\n}}\n')
    return ''.join(out)


def parse_profile(text: str, manifest: dict) -> dict:
    """Require complete boundary coverage and conserved measured intervals."""
    text = text.replace("\r", "")
    sums = re.findall(r"^PROFILE_SUM (\d+) (\d+) (\d+) (\d+) (\d+)$", text, re.M)
    tails = re.findall(r"^PROFILE_TAIL (\d+)$", text, re.M)
    if len(sums) != 1 or len(tails) != 1:
        raise ValueError("expected one complete forward profile")
    total, device, host, count, overflow = map(int, sums[0])
    events = [list(map(int, row)) for row in re.findall(
        r"^PROFILE_CALL (\d+) (\d+) (\d+) (\d+)$", text, re.M)]
    tail = int(tails[0])
    expected_ids = {row["id"] for row in manifest["boundaries"]}
    if (overflow or count != manifest["expected_device_calls"] or
            len(events) != count or [x[0] for x in events] != list(range(count)) or
            {x[1] for x in events} != expected_ids):
        raise ValueError("profile does not cover every expected device boundary")
    expected_counts=manifest.get("expected_symbol_calls")
    if expected_counts is not None and Counter(x[1] for x in events) != Counter({int(k):v for k,v in expected_counts.items()}):
        raise ValueError("profile does not cover expected per-symbol call counts")
    if (sum(x[3] for x in events) != device or
            sum(x[2] for x in events) + tail != host or total != device + host):
        raise ValueError("profile intervals do not conserve forward cycles")
    return {"forward_counter": total, "device_counter": device,
            "host_gap_counter": host, "tail_counter": tail, "events": events}


def leaf_kernel_profile(catalog: dict, build_dir: Path, work: Path, llvm_bin: Path):
    """Expose unresolved primitive calls without rebuilding any source object.

    Recreate each original partial-link component and require identical bytes
    before replacing it with its leaves. This is necessary because --wrap does
    not intercept already-resolved calls inside a merged relocatable object.
    """
    link_nodes={}; segmented_routes=[]; requant_manifest_hashes=set()
    def visit(value):
        if isinstance(value,dict):
            if 'requant_manifest_sha256' in value:
                requant_manifest_hashes.add(value['requant_manifest_sha256'])
            accepted=value.get('segmented_bindings')
            if accepted is not None:
                if accepted.get('schema')!='gemmini_segmented_input_bindings_v1':
                    raise ValueError('unsupported segmented profile binding')
                segmented_routes.extend(accepted['routes'])
            argv=value.get('linker_argv')
            if argv and '-r' in argv:
                out=Path(argv[argv.index('-o')+1]);inputs=[Path(x) for x in argv[argv.index('-r')+1:argv.index('-o')]]
                link_nodes[out]=(inputs,value['object_sha256'])
            for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    visit(catalog['compilation'])
    top_argv=catalog['compilation']['linker_argv'];top=Path(top_argv[top_argv.index('-o')+1])
    components={}
    for inputs,_ in link_nodes.values():
        for path in inputs:
            if path.name in ('requant.o','residual.o','stem_pool.o','segmented_inputs.o'):components[path.name]=path
    linker=llvm_bin/'ld.lld'
    if not linker.is_file():
        import shutil
        linker=Path(shutil.which('ld.lld'))
    receipt=[];boundaries=[];leaf_map={}
    def object_path(compilation,component_root):
        argv=compilation['compiler_argv']
        if isinstance(argv[0],list):argv=argv[-1]
        path=Path(argv[argv.index('-o')+1])
        if not path.is_file() and component_root.name in path.parts:
            path=component_root.joinpath(*path.parts[path.parts.index(component_root.name)+1:])
        if digest(path)!=compilation['object_sha256']:raise ValueError('leaf object identity mismatch')
        return path
    def reproduce(component,leaves):
        if not component.is_file() or not leaves:raise ValueError('missing profile partial-link component')
        target=work/('reproduced_'+component.name)
        argv=[str(linker),'-r',*map(str,leaves),'-o',str(target)]
        subprocess.run(argv,check=True,capture_output=True)
        if digest(target)!=digest(component):raise ValueError('leaf objects do not reproduce original partial-link bytes')
        leaf_map[component]=leaves;receipt.append(dict(component=str(component),sha256=digest(component),leaves={str(p):digest(p) for p in leaves},argv=argv))
    original_fused_routes=None
    for group,component_name,category in [('fused_requantizations','requant.o','unary'),('residual_additions','residual.o','residual')]:
        routes=catalog.get(group,[])
        if not routes:continue
        stored_routes=routes
        manifest_path=None
        if group=='fused_requantizations' and segmented_routes:
            # Semantic routes name the selected implementations. The stored
            # original component still contains the old leaves, including the
            # now-unused ones; its hash-bound manifest is the reconstruction
            # authority. Neither those leaves nor the component are rebuilt.
            manifest_path=components[component_name].parent/'requant.json'
            if not manifest_path.is_file() or digest(manifest_path) not in requant_manifest_hashes:
                raise ValueError('original requant manifest identity mismatch')
            original=json.loads(manifest_path.read_text())
            if (original.get('schema')!='gemmini_exact_captured_requant_bundle_v1' or
                    original.get('object_sha256')!=digest(components[component_name])):
                raise ValueError('original requant manifest component mismatch')
            original_fused_routes=stored_routes=original['routes']
            old_symbols=[r['symbol'] for r in stored_routes]
            selected_sources=[r.get('source_symbol',r['symbol']) for r in routes]
            if (len(set(old_symbols))!=len(old_symbols) or
                    len(set(selected_sources))!=len(selected_sources) or
                    set(old_symbols)!=set(selected_sources)):
                raise ValueError('selected requant route coverage differs from original')
        leaves=[]
        for route in stored_routes:
            leaves.extend([object_path(route['compilation'],components[component_name].parent),object_path(route['adapter_compilation'],components[component_name].parent)])
        for route in routes:
            arity=4 if category=='residual' and 'coefficients' in route['proof'] else 5 if category=='residual' else 3
            if arity==5:raise ValueError('leaf profiler currently requires wide residual ABI')
            boundaries.append(dict(symbol=route['kernel'],pointer_arity=arity,category=category,source_region=route.get('region'),shape=route.get('schedule',route.get('shape')),cpu_integer_readout_after=bool(route.get('integer_readout'))))
        reproduce(components[component_name],leaves)
        if manifest_path is not None:
            receipt[-1]['original_manifest']=str(manifest_path)
            receipt[-1]['original_manifest_sha256']=digest(manifest_path)
    if segmented_routes:
        accepted=catalog.get('segmented_input_acceptance',{})
        rewrites=accepted.get('source_rewrites',[])
        if len({r['source_symbol'] for r in segmented_routes})!=len(segmented_routes):
            raise ValueError('duplicate segmented profile source')
        if sorted((r['source_symbol'],r['accepted_symbol']) for r in segmented_routes)!=sorted(
                (r['source_symbol'],r['accepted_symbol']) for r in rewrites):
            raise ValueError('segmented profile source coverage differs')
        source=Path(catalog['source_snapshot'])
        if digest(source)!=accepted['selected_source_snapshot_sha256']:
            raise ValueError('segmented profile source identity changed')
        from .frontend.parse import parse_module
        calls=Counter(op.callee.root_reference.data for op in parse_module(source.read_text()).walk()
                      if op.name=='func.call')
        if any(type(r['calls']) is not int or r['calls']<=0 or
               calls[r['accepted_symbol']]!=r['calls'] or calls[r['source_symbol']] for r in rewrites):
            raise ValueError('segmented profile source call coverage differs')
        original={r['symbol']:r for r in original_fused_routes or []}
        selected={r.get('source_symbol',r['symbol']):r for r in catalog.get('fused_requantizations',[])}
        component=components['segmented_inputs.o']; leaves=[]
        for route in segmented_routes:
            old=original.get(route['source_symbol'])
            active=selected.get(route['source_symbol'])
            indices=[i for i,row in enumerate(boundaries) if active and row['symbol']==active['kernel']]
            if (old is None or active is None or len(indices)!=1 or
                    active['symbol']!=route['accepted_symbol'] or active['kernel']!=route['accepted_kernel'] or
                    any(active[key]['object_sha256']!=route[key]['object_sha256']
                        for key in ('compilation','adapter_compilation'))):
                raise ValueError('segmented profile requires one replaced primitive boundary')
            leaves.extend([object_path(route['compilation'],component.parent),
                           object_path(route['adapter_compilation'],component.parent)])
            boundaries[indices[0]]=dict(boundaries[indices[0]],symbol=route['accepted_kernel'],
                                       source_symbol=route['source_symbol'],
                                       accepted_symbol=route['accepted_symbol'],
                                       accepted_address=route['address'])
        reproduce(component,leaves)
    pool=catalog.get('pooled_stem')
    if pool:
        component=components['stem_pool.o'];leaves=[object_path(pool['compilation'],component.parent),component.parent/'adapter.o']
        reproduce(component,leaves)
        boundaries.append(dict(symbol=pool['symbol']+'_kernel',pointer_arity=3,category='pooled_stem',source_region='stem',shape=pool['shape']))
    for kernel in catalog['kernels']:
        boundaries.append(dict(symbol=kernel['symbol'],pointer_arity=3,category='dense',source_region=None,shape=kernel.get('shape')))
    if catalog.get('direct_convolutions'):raise ValueError('unfused direct components not supported in leaf mode')
    def flatten(path):
        if path in leaf_map:return leaf_map[path]
        if path in link_nodes:
            inputs,expected=link_nodes[path]
            if digest(path)!=expected:raise ValueError('partial-link identity changed')
            return [leaf for x in inputs for leaf in flatten(x)]
        return [path]
    leaves=flatten(top)
    if len(set(leaves))!=len(leaves):raise ValueError('duplicate link leaf')
    return leaves,sorted(boundaries,key=lambda x:x['symbol']),receipt


def build(build_dir: Path, runtime_dir: Path, gcc: Path, llvm_bin: Path, work: Path, *, leaf_kernels=False) -> dict:
    build_dir=build_dir.resolve();work=work.resolve();work.mkdir(parents=True,exist_ok=False)
    base=build_dir/'model.elf';base_audit=audit_elf(base.read_bytes())
    if base_audit['status']!='pass':raise ValueError('base ELF must pass zero-FSM audit')
    catalog=json.loads((build_dir/'device_catalog/device_catalog.json').read_text())
    if catalog['abi']['argument_order']!=['lhs','rhs','out']:raise ValueError('unsupported dense ABI')
    symbols=[(x['symbol'],3) for x in catalog['kernels']]
    symbols += [(f"_mlir_ciface_{x['symbol']}",4) for x in catalog.get('direct_convolutions',[])]
    symbols=sorted(set(symbols));leaf_objects=None;leaf_receipts=[];details=None
    if leaf_kernels:
        leaf_objects,details,leaf_receipts=leaf_kernel_profile(catalog,build_dir,work,llvm_bin)
        symbols=[(x['symbol'],x['pointer_arity']) for x in details]
    if not symbols:raise ValueError('no device boundaries')
    source=work/'device_profile.c';source.write_text(emit(symbols))
    h=runtime_dir/'baremetal/spike';rt=runtime_dir/'c'
    flags=['-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin']
    obj=work/'device_profile.o'
    subprocess.run([str(gcc),*flags,'-I',str(h),'-I',str(rt),'-c',str(source),'-o',str(obj)],check=True)
    nm=subprocess.check_output([str(llvm_bin/'llvm-nm'),str(base)],text=True)
    constants={}
    for key in ('MERLIN_WEIGHTS_BASE','MERLIN_STACK_BYTES'):
        m=re.search(rf'^([0-9a-fA-F]+) A {key}$',nm,re.M)
        if not m:raise ValueError(f'missing linker identity {key}')
        constants[key]=int(m[1],16)
    units=['model_call.o','merlin_model.o','model_main.o','mlir_rt.o','crt.o','console.o','libc_min.o','malloc.o','model.o','weights_blob.o']
    objects=[build_dir/x for x in units]
    if leaf_objects is None:
        kernel=build_dir/'device_catalog'/('mixed_kernel.o' if catalog.get('direct_convolutions') else 'kernel.o')
        if digest(kernel)!=catalog['compilation']['object_sha256']:raise ValueError('catalog object identity mismatch')
        leaf_objects=[kernel]
    objects += [*leaf_objects,build_dir/'device/device_catalog_shim.o']
    elf=work/'model.elf'
    wraps=[name for name,_ in symbols]+['merlin_run_multi','htif_exit']
    argv=[str(gcc),*flags,'-nostdlib','-nostartfiles',*[f'-Wl,--defsym,{k}={hex(v)}' for k,v in constants.items()],'-T',str(h/'model_link.ld'),*[str(x) for x in objects],str(obj),*[f'-Wl,--wrap={x}' for x in wraps],'-lm','-o',str(elf)]
    subprocess.run(argv,check=True)
    audit=audit_elf(elf.read_bytes());(work/'model.nofsm_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if audit['status']!='pass':raise ValueError('profile ELF failed zero-FSM audit')
    record={'schema':'golden_device_boundary_profile_build_v1','base_elf':str(base),'base_elf_sha256':digest(base),'elf_sha256':digest(elf),'model_object_sha256':digest(build_dir/'model.o'),'source_sha256':digest(source),'objects':{str(x):digest(x) for x in objects},'linker_argv':argv,'boundaries':[{'id':i,'symbol':name,'pointer_arity':arity} for i,(name,arity) in enumerate(symbols)],'expected_device_calls':catalog.get('total_device_contractions',catalog['covered_contractions']),'semantics':'PROFILE_SUM forward/device/host-gap cycles; each PROFILE_CALL ordinal/symbol-id/preceding-host-gap/device-cycles; instrumentation overhead belongs mostly to host gaps'}
    binding_counts=Counter(row['symbol'] for row in catalog.get('bindings',[]))
    if details is None and sum(binding_counts.values())==record['expected_device_calls'] and set(binding_counts)=={s for s,_ in symbols}:
        record['expected_symbol_calls']={str(i):binding_counts[symbol] for i,(symbol,_) in enumerate(symbols)}
    if details is not None:
        record['boundaries']=[dict(row,id=i) for i,row in enumerate(details)]
        record['leaf_component_reproduction']=leaf_receipts
        record['semantics']+='; primitive kernel calls only; descriptor checks and exact CPU integer readout remain in host gaps'
    (work/'profile_build.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir','runtime-dir','gcc','llvm-bin','workdir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--leaf-kernels',action='store_true')
    a=p.parse_args();r=build(a.build_dir,a.runtime_dir,a.gcc,a.llvm_bin,a.workdir,leaf_kernels=a.leaf_kernels)
    print(json.dumps({'elf_sha256':r['elf_sha256'],'boundaries':len(r['boundaries']),'expected_calls':r['expected_device_calls']},indent=2))
if __name__=='__main__':main()
