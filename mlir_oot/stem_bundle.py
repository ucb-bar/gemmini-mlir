"""Compile a structurally bound packed RGB stem, retaining exact i32 epilogues."""
import argparse,hashlib,json,subprocess,shutil
from dataclasses import asdict
from pathlib import Path
from xdsl.dialects.builtin import StringAttr
from .frontend.parse import parse_module
from .contraction_patterns import match_integer_gemm
from .stem_binding import match,rewrite,emit_adapter,emit_oracle
from .golden_stem import GoldenStem,command_counts
from .golden_device_compile import compile_module
from .direct_conv_binding import serialize
from .no_fsm_audit import audit_elf


def build(source,llvm_bin,output):
    source,llvm_bin,output=map(Path,(source,llvm_bin,output));output.mkdir(parents=True,exist_ok=False);text=source.read_text();module=parse_module(text);selected=[]
    for op in module.walk():
        if match_integer_gemm(op):
            try:selected.append(match(op))
            except ValueError:pass
    if len(selected)!=1:raise ValueError('exactly one proven packed stem required')
    binding=selected[0];symbol='gemmini_packed_stem';kernel=symbol+'_kernel';region=getattr(binding.contraction.attributes.get('prov.region_id'),'data','');declaration=rewrite(binding,symbol)
    device=GoldenStem(binding.shape).build();device.body.block.first_op.properties['sym_name']=StringAttr(kernel);compilation=compile_module(device,llvm_bin,output/'device')
    adapter=emit_adapter(binding.shape,symbol,kernel);(output/'adapter.c').write_text(adapter);(output/'native_oracle.c').write_text(adapter+emit_oracle(binding.shape,kernel))
    subprocess.run([str(llvm_bin/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-O2','-ffreestanding','-fno-builtin','-c',str(output/'adapter.c'),'-o',str(output/'adapter.o')],check=True,capture_output=True)
    linker=llvm_bin/'ld.lld'
    if not linker.is_file():linker=Path(shutil.which('ld.lld'))
    obj=output/'stem.o';subprocess.run([str(linker),'-r',str(output/'device/kernel.o'),str(output/'adapter.o'),'-o',str(obj)],check=True,capture_output=True)
    audit=audit_elf(obj.read_bytes())
    if audit['status']!='pass':raise ValueError('stem no-FSM audit failed')
    module.verify();printed=serialize(module,[declaration]);parse_module(printed).verify();(output/'rewritten.mlir').write_text(printed)
    result={'schema':'gemmini_packed_stem_bundle_v1','source_sha256':hashlib.sha256(text.encode()).hexdigest(),'rewritten_sha256':hashlib.sha256(printed.encode()).hexdigest(),'object_sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),'routes':[{'region':region,'symbol':symbol,'kernel':kernel,'shape':asdict(binding.shape)}],'commands':command_counts(binding.shape),'compilation':compilation,'nofsm_audit':audit,'scope':'exact i32 packed stem; host floating epilogue and maxpool preserved; explicit halo values preserved'}
    (output/'stem.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--llvm-bin',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=build(a.source,a.llvm_bin,a.output);print(json.dumps({'commands':r['commands'],'object_sha256':r['object_sha256']},indent=2))

if __name__=='__main__':main()
