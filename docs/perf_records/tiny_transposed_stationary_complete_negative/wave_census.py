from pathlib import Path
import json,hashlib
from dataclasses import asdict
from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.llvm import LLVM
from mlir_oot.execute_wave_estimator import derive_facts,commands_from_function,estimate_commands
from mlir_oot.ir.gemmini_dialect import GEMMINI
w=Path(__file__).resolve().parent
rtl=Path('/scratch2/agustin/chipyard/generators/gemmini/src/main/scala/gemmini/ExecuteController.scala')
facts=derive_facts(rtl,dimension=16)
base=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/device_catalog/kernel.gemmini.mlir')
ctx=Context()
for d in [Builtin,LLVM,GEMMINI]:ctx.load_dialect(d)
records={}
for arm,path,symbol in [('dense',base,'gemmini_golden_a5705ab56e324ba1'),('transposed',w/'transposed_device/kernel.gemmini.mlir','gemmini_golden_gemm')]:
 module=Parser(ctx,path.read_text()).parse_module()
 function=next(f for f in module.body.block.ops if f.sym_name.data==symbol)
 r=estimate_commands(commands_from_function(function,pointer_index_bits=64),facts,retain_records=False)
 records[arm]=r
 print('ACTUAL_SOURCE_WAVE_SCENARIOS',arm,json.dumps(r),flush=True)
result={'schema':'source_stationary_transpose_wave_scenarios_v1','facts':asdict(facts),'records':records,'scope':'RTL-source geometry scenarios only, not mesh/CPU/DMA/whole timing. Source command order does not reveal actual cmd.valid/hazard dispatch or preload-only latency. Same mathematical MAC/active area does not establish equal executed feed-wave geometry. Existing target complete costs remain authoritative.','pins':{str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest()for p in [rtl,base,w/'transposed_device/kernel.gemmini.mlir',Path(__file__)]},'token_usage_available':False}
(w/'wave_census.json').write_text(json.dumps(result,indent=2)+'\n')
