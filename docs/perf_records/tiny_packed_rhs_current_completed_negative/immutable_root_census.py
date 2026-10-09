from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
from torch_mlir import ir
from merlin.llvmlower.quant_hoist import read_plan,read_values
work=Path(__file__).resolve().parent
build=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ctx=ir.Context()
with ctx:
 module=ir.Module.parse((build/'model.prepared.mlir').read_text())
 fn=next(op for op in module.body.operations if str(op.attributes['sym_name'])=='"forward"')
 args=list(fn.regions[0].blocks[0].arguments)
 plan=read_plan(build);first_hoisted=len(args)-len(plan)
 externs={str(op.attributes['sym_name']).strip('"'):op for op in module.body.operations if op.operation.name=='func.func' and len(op.regions[0].blocks)==0}
 result=[];selected=None
 for index,op in enumerate(fn.regions[0].blocks[0].operations):
  if op.name!='func.call' or len(op.operands)!=3:continue
  rhs=op.operands[1]
  if not any(rhs==arg for arg in args):continue
  argindex=next(i for i,arg in enumerate(args)if rhs==arg)
  if argindex<first_hoisted:continue
  entry=plan[argindex-first_hoisted]
  if entry.dtype!='i8' or len(entry.shape)!=2:continue
  callee=str(op.attributes['callee']).lstrip('@')
  declaration=externs[callee]
  argattrs=list(declaration.attributes['arg_attrs'])
  read_contract=str(argattrs[1])
  assert 'bufferization.access = "read"' in read_contract
  uses=[{'operation':use.owner.name,'operand':use.operand_number}for use in rhs.uses]
  assert len(uses)==1 and uses[0]['operation']=='func.call' and uses[0]['operand']==1
  record={'source_call_ordinal_in_function_body':index,'source_callee':callee,'public_argument_index':argindex,'logical_type':str(rhs.type),'hoisted_plan_index':argindex-first_hoisted,'hoisted_key':entry.key,'rhs_shape':list(entry.shape),'read_effect':read_contract,'complete_rhs_ssa_uses':uses,'source_immutable_tensor_argument':True,'ordinary_current_physical_layout':'dense_row_major'}
  result.append(record)
  if entry.shape==(2048,5632) and selected is None:selected=record
 assert len(result)==155 and selected is not None
 values=read_values(build);value=values[selected['hoisted_key']]
 captured=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/tiny-packed-rhs-current-20261006/capture/b.bin')
 assert value.dtype==np.int8 and value.shape==(2048,5632)
 selected['hoisted_value_raw_sha256']=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
 assert selected['hoisted_value_raw_sha256']==sha(captured)
 proof={'schema':'current_typed_immutable_rhs_root_census_v1','status':'pass','source_prepared_ir_path':str(build/'model.prepared.mlir'),'source_prepared_ir_sha256':sha(build/'model.prepared.mlir'),'quant_hoist_plan_path':str(build/'quant_hoist_args.json'),'quant_hoist_plan_sha256':sha(build/'quant_hoist_args.json'),'complete_source_contraction_immutable_roots':result,'selected_actual_capture_root':selected,'source_guarantee':'Each current RHS is a distinct immutable tensor argument with exactly one source use, the readonly RHS of a declared three-argument writer; original155catalog binding remains unchanged. This is legal source evidence for possible generic physical parameter materialization, not implemented packing routing.','normal_route_missing':'Existing weight_panel rewrite produces source panel-loop matmuls. The measured provider keeps the original completeN primitive. Normal use needs a generic immutable physical-argument plan, source logical-to-physical binding/ABI consistency, per-provider explicit layout, and original fullmodel native/strict gates.','selector_policy':'Source names/ordinals/keys are diagnostic bindings only; no production strategy routes on these fields.','token_usage_available':False}
 (work/'immutable_root_census.json').write_text(json.dumps(proof,indent=2)+'\n')
 print('CURRENT_TYPED_IMMUTABLE_RHS_ROOTS PASS',len(result),'firstroot',selected['public_argument_index'],flush=True)
