from pathlib import Path
from collections import Counter
import hashlib,json
from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.llvm import LLVM
from merlin.llvmlower.static_llvm_cfg import StaticInt,StaticPointer,trace_static_function
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.golden_gemm import Shape
from mlir_oot.golden_packed_rhs import PackedRhs,GoldenPackedRhsGemm
from mlir_oot.tables import rtl_facts as F
work=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
catalog_dir=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/device_catalog')
catalog=json.loads((catalog_dir/'device_catalog.json').read_text())
profile_path=Path('/scratch/agustin/tmp/gemmini-smol-target-numeric-20261005/docs/perf_records/firesim1901_tiny1880_profile_verified.json')
profile=json.loads(profile_path.read_text())
ids={'gemmini_golden_29c4e0a80e5abd97':0,'gemmini_golden_7b4380121ce5c4f3':1,'gemmini_golden_a5705ab56e324ba1':2,'gemmini_golden_e738f13ad92e3128':3,'gemmini_golden_ef297329ec9dee28':4}
ctx=Context();ctx.load_dialect(Builtin);ctx.load_dialect(LLVM);ctx.load_dialect(G.GEMMINI)
module=Parser(ctx,(catalog_dir/'kernel.gemmini.mlir').read_text()).parse_module()
functions={fn.sym_name.data:fn for fn in module.body.block.ops}
args=[StaticPointer(i,StaticInt(0,64))for i in range(3)]
def census(fn,bbase=0):
 counts=Counter();requests=Counter();bytes_=Counter();strides={};b_row_last=None;b_row_transitions=0;b_intervals=[]
 commands_hash=hashlib.sha256()
 for step in trace_static_function(fn,args,observe=lambda op:isinstance(op,G._GemminiOp),pointer_index_bits=64):
  op=step.operation;counts[op.name]+=1
  record=[op.name,{str(k):str(v)for k,v in op.attributes.items()},[(p.base,p.offset.value)for p in step.inputs if isinstance(p,StaticPointer)]]
  commands_hash.update((json.dumps(record,sort_keys=True)+'\n').encode())
  if isinstance(op,G.ConfigLdOp):strides[op.a('load_id')]=op.a('stride')
  if isinstance(op,G.MvinOp):
   p=step.inputs[0];requests[p.base]+=op.a('rows');bytes_[p.base]+=op.a('rows')*op.a('cols')
   if p.base==1:
    for r in range(op.a('rows')):
     start=p.offset.value+r*strides[op.a('load_id')];b_intervals.append((start,start+op.a('cols')))
     page=(bbase+start)>>12
     b_row_transitions+=page!=b_row_last;b_row_last=page
  if isinstance(op,G.MvoutOp):bytes_[2]+=op.a('rows')*op.a('cols')*4
 b_intervals.sort()
 assert b_intervals[0][0]==0 and all(a[1]==b[0]for a,b in zip(b_intervals,b_intervals[1:]))
 return {'commands':dict(counts),'command_total':sum(counts.values()),'requested_bytes':dict(bytes_),'input_row_requests':dict(requests),'B_written_extent_from_complete_read_partition':b_intervals[-1][1],'B_row_page_transitions_in_source_command_order':b_row_transitions,'command_trace_sha256':commands_hash.hexdigest()}
families=[]
for record in catalog['kernels']:
 symbol=record['symbol'];dims=record['dimensions'];events=[e for e in profile['boundary_profile']['events']if e[1]==ids[symbol]]
 facts=census(functions[symbol]);facts.update(symbol=symbol,dimensions=dims,logical_bindings=len(events),schedule=record['schedule'],older1901_same_device_call_cycles=sum(e[3]for e in events),requested_B_bytes_across_bindings=facts['requested_bytes'][1]*len(events))
 families.append(facts);print('CLOSED_COMMAND_CENSUS',symbol,facts['command_total'],flush=True)
symbol='gemmini_golden_a5705ab56e324ba1'
bbase=int(next(line.split()[0]for line in (work/'dense/symbols.txt').read_text().splitlines()if line.endswith(' input_b')),16)
selected_original=census(functions[symbol],bbase)
selected_packed=census(GoldenPackedRhsGemm(Shape(8,5632,2048,output_dtype='i32',bm=1,bn=64,cache_a=True,wide_b=True,prefetch_b=True),PackedRhs(2048,5632,64)).build().body.block.first_op,bbase)
assert selected_original['commands']==selected_packed['commands'] and selected_original['requested_bytes']==selected_packed['requested_bytes'] and selected_original['input_row_requests']==selected_packed['input_row_requests']
scala=Path('/scratch2/agustin/chipyard/generators/gemmini/src/main/scala/gemmini')
sources=[scala/'LoadController.scala',scala/'FrontendTLB.scala',scala/'Scratchpad.scala',scala/'Configs.scala',scala/'GemminiConfigs.scala',Path('/scratch2/agustin/chipyard/generators/gemmini/chipyard/GemminiConfigs.scala'),Path('/scratch2/agustin/chipyard/generators/firechip/chip/src/main/scala/TargetConfigs.scala'),Path('/scratch2/agustin/chipyard/generators/chipyard/src/main/scala/config/MerlinGemminiGsimConfigs.scala'),Path('/scratch2/agustin/chipyard/generators/gemmini/software/gemmini-rocc-tests/riscv-tests/benchmarks/common/crt.S')]
result={'schema':'current_tiny_device_exact_source_command_traffic_census_v1','families':families,'old_hardware_profile_path':str(profile_path),'old_hardware_profile_sha256':sha(profile_path),'catalog_path':str(catalog_dir/'device_catalog.json'),'catalog_sha256':sha(catalog_dir/'device_catalog.json'),'kernel_object_sha256':sha(catalog_dir/'kernel.o'),'typed_source_sha256':sha(catalog_dir/'kernel.gemmini.mlir'),'stock1901_scope':'155conserved events on1880host. Same current1983/1926/1967/1989/1997deviceobject bytes; actual current host/device split and cache/layout context UNKNOWN. Events include adapter/CPUcommand work, not pure mesh attribution.','requested_bytes_scope':'Complete exact static device command trace. Every B byte read exactlyonce per call; DRAM/physical backend traffic not observed by this census. No reduction or transfer-overlap performance claim.','resources':{'dim':F.DIM,'spad_rows':F.SPAD_ROWS,'spad_banks':F.SPAD_BANKS,'acc_rows':F.ACC_ROWS,'max_selected_output_block_tiles':64,'selected_output_columns_per_block':1024,'selected_last_output_block_columns':512,'selected_cached_A_rows':2048,'selected_prefetched_B_rows_per_slot':1024,'dma_row_max':16},'mesh_scope':'M8 requests half the16 mesh rows; actual dynamic PE occupancy/latency not measured. K/N tiles are complete for all five families.','selected_common_B_address':bbase,'selected_original':selected_original,'selected_packed':selected_packed,'frontend_hypothesis':'Only model transitions between4KiB pages of ordered B row addresses. Register-filter RTL can avoid shared frontend requests without pagewalks in bare Mmode. Actual DMA arbitration, filter events and latency are not measured here; packed candidate incurs+5.98% strict instructions.','rtl_and_startup_sources':{str(p):sha(p)for p in sources},'token_usage_available':False}
(work/'command_traffic_census.json').write_text(json.dumps(result,indent=2)+'\n')
print('CLOSED_PACKED_TRAFFIC_CENSUS',selected_original['B_row_page_transitions_in_source_command_order'],selected_packed['B_row_page_transitions_in_source_command_order'],flush=True)
