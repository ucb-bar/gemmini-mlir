from pathlib import Path
from collections import Counter
import hashlib,json
from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.llvm import LLVM
from merlin.llvmlower.static_llvm_cfg import StaticInt,StaticPointer,trace_static_function
from mlir_oot.ir import gemmini_dialect as G
from mlir_oot.tables import isa,rtl_facts as F
W=Path(__file__).resolve().parent
BASE=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/device_catalog')
ctx=Context();ctx.load_dialect(Builtin);ctx.load_dialect(LLVM);ctx.load_dialect(G.GEMMINI)
args=[StaticPointer(i,StaticInt(0,64))for i in range(3)]
mods=[Parser(ctx,(BASE/'kernel.gemmini.mlir').read_text()).parse_module(),Parser(ctx,(W/'transposed_device/kernel.gemmini.mlir').read_text()).parse_module()]
functions=[next(f for f in mods[0].body.block.ops if f.sym_name.data=='gemmini_golden_a5705ab56e324ba1'),mods[1].body.block.first_op]
records=[]
for arm,fn in zip(['dense','transposed'],functions):
 count=Counter();requested=Counter();rows=Counter();stationary=Counter();strides={};intervals={0:[],1:[]};h=hashlib.sha256()
 for step in trace_static_function(fn,args,observe=lambda op:isinstance(op,G._GemminiOp),pointer_index_bits=64):
  op=step.operation;count[op.name]+=1
  h.update((json.dumps([op.name,{str(k):str(v)for k,v in op.attributes.items()},[(p.base,p.offset.value)for p in step.inputs if isinstance(p,StaticPointer)]],sort_keys=True)+'\n').encode())
  if isinstance(op,G.ConfigLdOp):strides[op.a('load_id')]=op.a('stride')
  if isinstance(op,G.MvinOp):
   p=step.inputs[0];requested[p.base]+=op.a('rows')*op.a('cols');rows[p.base]+=op.a('rows')
   for r in range(op.a('rows')):
    start=p.offset.value+r*strides[op.a('load_id')];intervals[p.base].append((start,start+op.a('cols')))
  if isinstance(op,G.MvoutOp):requested[2]+=op.a('rows')*op.a('cols')*4;rows[2]+=op.a('rows')
  if isinstance(op,G.PreloadOp):
   keep=op.a('bd')==isa.GARBAGE_ADDR
   stationary['keep_operand_commands' if keep else 'reload_operand_commands']+=1
   if not keep:stationary['logical_stationary_operand_bytes']+=op.a('bd_rows')*op.a('bd_cols')
 for operand,ranges in intervals.items():
  ranges.sort();assert ranges[0][0]==0 and all(a[1]==b[0]for a,b in zip(ranges,ranges[1:]))
 record={'arm':arm,'command_counts':dict(count),'command_total':sum(count.values()),'requested_input_readout_bytes':dict(requested),'DMA_row_requests':dict(rows),'complete_exact_once_input_byte_partitions':{str(i):x[-1][1]for i,x in intervals.items()},'stationary_operand_source':dict(stationary),'exact_command_trace_sha256':h.hexdigest()};records.append(record)
 print('TRANSPOSED_SOURCE_COMMAND_CENSUS',arm,record,flush=True)
result={'schema':'exact_current_transposed_stationary_command_census_v1','arms':records,'logical_weight_parameter_bytes':11534336,'runtime_permutation_bytes':{'activation_read_and_write':2*8*2048,'output_read_and_write':2*8*5632*4},'resource_proof':{'DIM':F.DIM,'SPAD_ROWS':F.SPAD_ROWS,'SPAD_BANK_ROWS':F.SPAD_BANK_ROWS,'ACC_ROWS':F.ACC_ROWS,'ACC_BANK_ROWS':F.ACC_BANK_ROWS,'A_slot0_range_rows':[0,4096],'A_slot1_range_rows':[4096,8192],'cachedB_range_rows':[8192,10240],'ACC_slot0_row_extent':[0,32],'ACC_slot1_row_extent':[512,544],'disjoint_scratch_slots_and_B':True,'sequential_existing_schedule_dependency_authority':True},'scope':'Exact source command counts and requested byte/row ranges only. Garbage BD retains same cached operand for next independent A row tile; logical reload bytes do not imply physical mesh latency or DRAM savings. SameMAC count/gridoccupancy; runtime copies/readout row requests/issue costs can offset reuse. Actual hardware cost unknown; whole model prediction absent.','token_usage_available':False}
(W/'command_census.json').write_text(json.dumps(result,indent=2)+'\n')
