"""Reclose the extracted scalar SSA, maps, attributes and exact constants."""
from pathlib import Path
import hashlib
import json
import subprocess
from merlin.llvmlower.scalar_pointwise_packet import RUNNER_PRELUDE
from merlin.llvmlower.toolchain import m2m_python

W=Path(__file__).resolve().parent
script=W/'run_typed_source_witness.py'
body=r'''
from pathlib import Path
import hashlib,json
from torch_mlir import ir
W=Path(__file__).resolve().parent
src=W.parent/'tiny1880-host-classes-20261005/prebuffer.mlir'
ctx=ir.Context()
with ctx,ir.Location.unknown():
 original=ir.Module.parse(src.read_text())
 capsule=ir.Module.parse((W/'source_capsule.mlir').read_text())
 def selected(module):
  found=[]
  def walk(op):
   for region in op.regions:
    for block in region.blocks:
     for child in block.operations:
      if _pointwise_packet_spec(child,multiplication=True,two_products=True) is not None and str(child.results[0].type)=='tensor<1x8x2048xf32>':found.append(child)
      walk(child.operation)
  walk(module.operation)
  return found
 a,b=selected(original),selected(capsule)
 assert len(a)==44 and len(b)==1
 def witness(op):
  block=op.regions[0].blocks[0]
  ids={v:['argument',i,str(v.type)]for i,v in enumerate(block.arguments)}
  records=[]
  def identity(value):
   if value in ids:return ids[value]
   owner=getattr(value.owner,'operation',value.owner)
   assert isinstance(owner,ir.Operation) and owner.name=='arith.constant'
   assert not owner.operands and not owner.regions and len(owner.results)==1
   return ['constant',str(value.type),{name:str(owner.attributes[name])for name in sorted(owner.attributes)}]
  for i,scalar in enumerate(block.operations):
   records.append({'name':scalar.operation.name,'operands':[identity(v)for v in scalar.operands],
    'result_types':[str(v.type)for v in scalar.results],
    'attributes':{name:str(scalar.attributes[name])for name in sorted(scalar.attributes)}})
   ids.update((v,['result',i,j,str(v.type)])for j,v in enumerate(scalar.results))
  return {'body':records,'block_arguments':[str(v.type)for v in block.arguments],
   'operand_types':[str(v.type)for v in op.operands],
   'result_types':[str(v.type)for v in op.results],
   'attributes':{name:str(op.attributes[name])for name in sorted(op.attributes)}}
 left,right=witness(a[0]),witness(b[0])
 assert left==right
 payload=json.dumps(left,sort_keys=True,separators=(',',':')).encode()
 record={'schema':'complete_typed_dequant_residual_extraction_witness_v1','status':'pass',
  'source_path':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
  'source_capsule_sha256':hashlib.sha256((W/'source_capsule.mlir').read_bytes()).hexdigest(),
  'selected_original_source_ordinal':0,'matching_original_source_ops':44,
  'source_and_capsule_semantic_fingerprint':hashlib.sha256(payload).hexdigest(),
  'exact_complete_ssa_maps_types_constant_and_attribute_match':True,'typed_witness':left,
  'older_exact_original_body_field':'The extraction receipt field is a diagnostic Python Region repr. This file supplies the executable canonical SSA/constant witness; neither field selects production policy.'}
 (W/'typed_source_witness.json').write_text(json.dumps(record,indent=2)+'\n')
 print('COMPLETE_TYPED_SOURCE_CLONE_PASS',record['source_and_capsule_semantic_fingerprint'],flush=True)
'''
script.write_text('from torch_mlir import ir\n'+RUNNER_PRELUDE.split('_PP_MARKERS =',1)[0]+body)
subprocess.run([str(m2m_python()),str(script)],check=True)
