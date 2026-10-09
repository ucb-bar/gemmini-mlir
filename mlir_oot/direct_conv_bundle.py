"""Prepare source-bound direct-conv rewrite and linkable primitive device bundle.

Run before the ordinary dense-GEMM catalog builder, on its final prepared IR.
This leaves unmatched operations to the existing host/device path. Boundary
transposes are explicit and intentionally retained until layout propagation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
from xdsl.dialects.builtin import StringAttr
from .frontend.parse import parse_module
from .contraction_patterns import match_integer_gemm
from .direct_conv_binding import match,rewrite,serialize,emit_c_adapter,emit_scalar_oracle
from .conv_schedule import select_kernel
from .golden_device_compile import compile_module
from .no_fsm_audit import audit_elf


def build(source: Path, llvm_bin: Path, output: Path, *, flat_spatial=False, allow_empty=False):
    output.mkdir(parents=True,exist_ok=False)
    text=source.read_text();module=parse_module(text)
    selected=[]
    for op in list(module.walk()):
        if match_integer_gemm(op) is None:continue
        try:selected.append(match(op))
        except ValueError:pass
    if not selected and not allow_empty:raise ValueError('no structurally proven direct convolution')
    objects=[];declarations=[];routes=[];native=[]
    for i,b in enumerate(selected):
        # One unique symbol per source operation avoids assuming shared graph lifetimes.
        symbol=f'gemmini_direct_conv_{i}';kernel=f'{symbol}_kernel';work=output/symbol
        source_region=getattr(b.contraction.attributes.get('prov.region_id'),'data','')
        declaration=rewrite(b,symbol);declarations.append(declaration)
        generator,schedule_kind=select_kernel(b.shape,flat_spatial=flat_spatial)
        device=generator.build();device.body.block.first_op.properties['sym_name']=StringAttr(kernel)
        receipt=compile_module(device,llvm_bin,work)
        adapter=work/'adapter.c';adapter.write_text(emit_c_adapter(b.shape,symbol,kernel))
        obj=work/'adapter.o'
        subprocess.run([str(llvm_bin/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-O2','-ffreestanding','-fno-builtin','-c',str(adapter),'-o',str(obj)],check=True,capture_output=True)
        objects += [work/'kernel.o',obj]
        native.append(emit_c_adapter(b.shape,symbol,kernel)+emit_scalar_oracle(b.shape,kernel))
        routes.append(dict(region=source_region,symbol=symbol,kernel=kernel,shape=asdict(generator.conv),schedule_kind=schedule_kind,orientation=b.orientation,compilation=receipt))
    (output/'native_oracle.c').write_text('\n'.join(native))
    printed=serialize(module,declarations)
    rewritten=output/'rewritten.mlir';rewritten.write_text(printed)
    parse_module(printed).verify()
    linked=output/'direct_conv.o'
    linker=llvm_bin/'ld.lld'
    if not linker.is_file():
        found=shutil.which('ld.lld')
        if found is None:raise ValueError('ld.lld is required for the relocatable bundle')
        linker=Path(found)
    if not objects:
        empty=output/'empty.c';empty.write_text('/* Uncalled audited anchor for the explicitly empty component. */\nvoid gemmini_empty_direct_bundle_anchor(void) {}\n')
        obj=output/'empty.o'
        subprocess.run([str(llvm_bin/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-c',str(empty),'-o',str(obj)],check=True,capture_output=True)
        objects.append(obj)
    subprocess.run([str(linker),'-r',*[str(x) for x in objects],'-o',str(linked)],check=True,capture_output=True)
    audit=audit_elf(linked.read_bytes())
    if audit['status']!='pass':raise ValueError('direct-conv bundle has forbidden commands')
    manifest=dict(schema='gemmini_direct_conv_bundle_v1',source_sha256=hashlib.sha256(text.encode()).hexdigest(),rewritten_sha256=hashlib.sha256(printed.encode()).hexdigest(),object_sha256=hashlib.sha256(linked.read_bytes()).hexdigest(),routes=routes,nofsm_audit=audit,linker_sha256=hashlib.sha256(linker.read_bytes()).hexdigest(),scope='source-bound direct 3x3 contractions only; explicit host boundary layouts; other contractions remain; no whole-model verification')
    (output/'direct_conv.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--llvm-bin',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--flat-spatial',action='store_true');a=p.parse_args()
    result=build(a.source,a.llvm_bin,a.output,flat_spatial=a.flat_spatial);print(json.dumps({'routes':len(result['routes']),'object_sha256':result['object_sha256']},indent=2))

if __name__=='__main__':main()
