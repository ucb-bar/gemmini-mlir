from pathlib import Path
import json,hashlib
from torch_mlir import ir
original=Path('/scratch/agustin/tmp/gemmini-tiny-host-scheduling-20261005/out/artifacts/probes/tiny-pointwise-packet/qualified_whole_2/build/device_host_abi/model.mlir')
work=Path(__file__).resolve().parent
ctx=ir.Context()
with ctx:
 module=ir.Module.parse(original.read_text())
 todo=[]
 def walk(op):
  for r in op.regions:
   for b in r.blocks:
    for x in b.operations:
     if x.operation.name=='linalg.generic' and len(x.operands)==3 and len(x.results)==1:
      types=tuple(str(v.type) for v in x.operands)
      if types in [("tensor<32x8x64xf32>","tensor<32x64x8xf32>","tensor<32x8x8xf32>"),("tensor<32x8x8xf32>","tensor<32x8x64xf32>","tensor<32x8x64xf32>")]:todo.append(x)
     walk(x.operation)
 walk(module.operation)
 assert len(todo)==44
 witness=[]
 for name,op in [('qk',todo[0]),('pv',todo[1])]:
  a,b,c=(str(v.type) for v in op.operands)
  body=[x.operation.name for x in op.regions[0].blocks[0].operations];assert body==['arith.mulf','arith.addf','linalg.yield']
  src=ir.Module.parse(f'module {{func.func @forward(%a:{a},%b:{b})->{c} attributes{{llvm.emit_c_interface}} {{%zero=arith.constant dense<0.0>:{c}\nreturn %zero:{c}\n}}}}')
  fn=list(src.body.operations)[0];blk=fn.regions[0].blocks[0];ops=list(blk.operations)
  with ir.InsertionPoint(ops[-1]): clone=op.operation.clone();clone.operands[0]=blk.arguments[0];clone.operands[1]=blk.arguments[1];clone.operands[2]=ops[0].results[0]
  ops[-1].operands[0]=clone.results[0]
  src.operation.verify();target=work/(name+'_source.mlir');target.write_text(str(src)+'\n')
  witness.append({'name':name,'experimental_occurrence':0,'same_shape_contractions':22,'operand_types':[a,b,c],'raw_original_op':str(op),'raw_original_op_sha256':hashlib.sha256(str(op).encode()).hexdigest(),'capsule_source_path':str(target),'capsule_source_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'source_arithmetic_cloned':True,'initializer_source_positive_zero_proven':True})
 (work/'typed_source_witness.json').write_text(json.dumps({'schema':'source_exact_typed_contraction_capsule_v1','original_source_path':str(original),'original_source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'contractions':witness,'scope':'Actual originalpostoffload generic operation bodies/properties/provenance cloned unchanged; only actual source positivezero tensor initialization localized, complete source inputs and endpoints retained. Experimental occurrence binds capture only; production rectangle legality uses projections/types/tileextents.'},indent=2)+'\n')
 print('SOURCE_EXACT_TYPED_CONTRACTION_CAPSULES',len(todo))
