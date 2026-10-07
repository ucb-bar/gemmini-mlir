from pathlib import Path
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--lanes',type=int,choices=(4,8),required=True);ap.add_argument('--workdir',type=Path,required=True);task_options=ap.parse_args()
import json,shutil
from mlir_oot.guarded_mean_bundle import build_and_apply,merlin_callbacks
from mlir_oot.stem_pool_mixed_catalog import merlin_callbacks as pool_callbacks
from mlir_oot.domain_residual_catalog import merlin_callbacks as residual_callbacks
from mlir_oot.golden_device_catalog import final_elf_audit
from mlir_oot.late_quant_rne import merlin_host_llvm_transform
from merlin.runtime.backends.spike_model import build
from merlin.llvmlower.device_build import DeviceRouting
import merlin.llvmlower.layout_propagation as layout
root=Path('/scratch/agustin/tmp/gemmini-key-batch16-normal-20261007');control=Path('/scratch/agustin/tmp/gemmini-prestem-fusion-20261005/out/prestem_classifier_identity')
work=task_options.workdir.resolve();work.mkdir(parents=True,exist_ok=True)
capture=work/'capture'
if not capture.exists():
 capture.mkdir()
 for p in (control/'capture').iterdir():
  if p.name in ('model.mlir','capture_receipt.json'):shutil.copyfile(p,capture/p.name)
  else:(capture/p.name).symlink_to(p.resolve())
llvm=Path('/scratch/agustin/projects/oscar-merlin/third_party/llvm-install/bin')
mean=work/'mean_bundle'
if not mean.exists():build_and_apply(capture,llvm,mean,packed_nhwc=True,device_integer_sum=True)
# Same exact bundle providers as1812, with only the proved mean replacement added.
requant=Path('/scratch/agustin/tmp/gemmini-dense-stationary-tail-normal-20261007/out/dense_tail_normal/candidate_bundle')
base=pool_callbacks(llvm,requant,control/'stem_pool_bundle',flat_spatial=True,propagate_layout=True,large_n=True)
domain_root=Path('/scratch/agustin/tmp/gemmini-residual-domain-20261007/out/source_domain_v1/closed_v3')
rectifier_bundle=Path('/scratch/agustin/tmp/gemmini-key-batch16-normal-20261007/out/artifacts/key_batch16_normal/key_bundle')
base=residual_callbacks(llvm,Path('/scratch/agustin/tmp/gemmini-residual-coefficient-20261005/out/artifacts/probes/residual-m-prefetch/residual_bundle'),domain_root/'bundle',base,joint_bundle=rectifier_bundle)
prepare,compile=merlin_callbacks(llvm,mean,base)
original=layout.rewrite_module
layout.rewrite_module=lambda module:original(module,reduction_channel_block=64)
# Only current prequalified ranked-descriptor adapters participate. Raw expanded
# core adapters remain unchanged. This is an owned prototype, not a global policy.
import importlib.util,hashlib,subprocess
import merlin.runtime.backends.zephyr_model as zm
from mlir_oot.frontend.parse import parse_module,context
from xdsl.dialects import bufferization
from xdsl.parser import Parser
from merlin.xdsl_dialects._common import text as portable_text
from mlir_oot.descriptor_writer import DescriptorWriterContract, transform as writer_transform, shim as writer_shim
# Contracts supplied by the exact emitter ABI; tensor type equality is never
# used to infer result identity. The raw expanded classifier is excluded.
contracts=[];contract_manifests=[]
for manifest,kind in [(requant/'requant.json','requant'),(domain_root/'bundle/residual.json','residual'),(control/'stem_pool_bundle/stem_pool.json','stem'),(mean/'mean.json','mean')]:
 payload=json.loads(manifest.read_text())
 contract_manifests.append(dict(path=str(manifest),sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),provider=kind))
 routes=payload.get('routes',[payload])
 for route in routes:
  if kind=='mean' and route.get('implementation')=='gemmini_integer_sum':result_arg,writers=2,(1,2)
  elif kind=='mean':result_arg,writers=1,(1,)
  elif kind=='requant' and route.get('integer_readout') is not None:result_arg,writers=3,(2,3)
  else:result_arg,writers=2,(2,)
  symbol=route.get('symbol','gemmini_exact_stem_pool' if kind=='stem' else None)
  if not symbol:raise ValueError('missing explicit adapter contract symbol')
  contracts.append(DescriptorWriterContract(symbol,result_arg,writers))
rectifier_record=json.loads((rectifier_bundle/'joint.json').read_text())
replacement_symbols={r['original_symbol']:r['symbol'] for r in rectifier_record['routes']}
contracts=[DescriptorWriterContract(replacement_symbols.get(c.symbol,c.symbol),c.result_argument,c.fully_written_arguments) for c in contracts]
contract_manifests.append(dict(path=str(rectifier_bundle/'joint.json'),sha256=hashlib.sha256((rectifier_bundle/'joint.json').read_bytes()).hexdigest(),provider='finite_source_rectifier'))
assert len(contracts)==70
from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers
from mlir_oot.segmented_input_binding import merlin_callbacks as segmented_callbacks
fresh_contracts=tuple(FreshTensorWriterContract(c.symbol,c.result_argument,c.fully_written_arguments,c.symbol+'__borrowed_write',64) for c in contracts)
prepare,compile,selected_writers=segmented_callbacks(llvm,json.loads((requant/'requant.json').read_text())['routes'],fresh_contracts,(prepare,compile),enabled=True)
original_prepare=zm.prepare_for_lowering
state={}
def fresh_prepare(*args,**kwargs):
 source,features=original_prepare(*args,**kwargs)
 module=parse_module(Path(source).read_text());routes=rewrite_fresh_tensor_writers(module,selected_writers())
 from xdsl.dialects.builtin import UnitAttr
 for op in module.body.block.ops:
  if op.name=='func.func' and op.body.blocks and getattr(op.sym_visibility,'data',None)!='private':op.attributes['llvm.emit_c_interface']=UnitAttr()
 from merlin.llvmlower.bounded_rne_maps import schedule_bounded_rne_maps
 quant_routes=schedule_bounded_rne_maps(module,lanes=task_options.lanes)
 (work/'quant_packet_source.json').write_text(json.dumps(quant_routes,indent=2)+'\n');print('QUANT_PACKET_MAPS',quant_routes,flush=True)
 target=work/'fresh.mlir';target.write_text(portable_text(module,generic=True));ctx=context();ctx.get_dialect('bufferization');Parser(ctx,target.read_text()).parse_module().verify()
 c=work/'borrowed.c';c.write_text(writer_shim(routes))
 record=dict(schema='fresh_descriptor_writer_prototype_v1',original_prepared_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),rewritten_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),routes=routes,contract_manifests=contract_manifests,contracts=[dict(symbol=c.symbol,result_argument=c.result_argument,fully_written_arguments=list(c.fully_written_arguments)) for c in selected_writers()],shim_sha256=hashlib.sha256(c.read_bytes()).hexdigest(),contract='Prequalified current adapters fully write and return the passed descriptor; all writable call operands are sole-use tensor.empty; fresh memref.alloc owns every writer; borrowed void shim suppresses duplicate return ownership')
 (work/'fresh_receipt.json').write_text(json.dumps(record,indent=2)+'\n');state.update(code=c)
 print('FRESH_ROUTES',len(routes),flush=True)
 return target,features
zm.prepare_for_lowering=fresh_prepare
base_transform=merlin_host_llvm_transform(llvm,combine_clamp=True)
def linked_host_transform(source,directory):
 selected=Path(base_transform(source,directory));directory=Path(directory)
 from merlin.llvmlower.bounded_rne_packet_llvm import rewrite_packet_helpers
 from merlin.llvmlower.late_quant_rne import rewrite as scalar_rne
 packet,packet_receipt=rewrite_packet_helpers(Path(source).read_text(),host_isa='rv64gc',max_lanes=task_options.lanes)
 packet,scalar_receipt=scalar_rne(packet,host_isa='rv64gc',combine_clamp=True,temporary_prefix='gemmini.rne')
 selected.write_text(packet)
 (directory/'packet_receipt.json').write_text(json.dumps(packet_receipt,indent=2)+'\n')
 print('QUANT_PACKET_LLVM',len(packet_receipt['routes']),flush=True)
 for native in (False,True):
  original=directory/'model.native.ll' if native else selected
  before=directory/('prebridge.native.ll' if native else 'prebridge.ll');shutil.copyfile(original,before)
  shim=directory/('borrowed.native.ll' if native else 'borrowed.ll')
  command=[str(llvm/'clang'),'-O2','-S','-emit-llvm',str(state['code']),'-o',str(shim)]
  if not native:command[1:1]=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany']
  subprocess.run(command,check=True,capture_output=True)
  linked=directory/('model.native.ll' if native else 'model.linked.ll')
  subprocess.run([str(llvm/'llvm-link'),'-S',str(before),str(shim),'-o',str(linked)],check=True,capture_output=True)
 return directory/'model.linked.ll'
result=build(capture,work/'build_direct',host_vectorize=True,int8_compute=True,host_llvm_transform=linked_host_transform,
 output_sha256=True,output_dump_cap=1,features=frozenset({'named_int8_contraction','hoist_weight_invariant_quantize','fuse_quantize_round_convert','reuse_tensor_destination'}),
 cflags_override=['-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin'],
 device=DeviceRouting('gemmini',str(root),'int8','i32',prepared_transform=prepare,catalog_builder=compile,final_elf_audit=final_elf_audit),
 dram_bytes=2*1024**3,arena_mb=256,stack_bytes=16*1024**2,console='htif')
(work/'build_result.json').write_text(json.dumps(result,indent=2,default=str)+'\n');print(json.dumps(result,default=str),flush=True)
