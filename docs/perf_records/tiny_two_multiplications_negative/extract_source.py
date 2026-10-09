"""Extract the original first closed signed-integer dequantization and residual body, without transcription."""
from pathlib import Path
import hashlib, json, subprocess
from merlin.llvmlower.scalar_pointwise_packet import RUNNER_PRELUDE
from merlin.llvmlower.toolchain import m2m_python

W = Path(__file__).resolve().parent
SOURCE = W.parent/'tiny1880-host-classes-20261005/prebuffer.mlir'
assert hashlib.file_digest(SOURCE.open('rb'), 'sha256').hexdigest() == '04a9a3eccb4578c446d51ad6ed23694911936bbbcc0bf00efd57b8c65f43a518'
body = r'''
from pathlib import Path
import json,hashlib
from torch_mlir import ir
W=Path(__file__).resolve().parent
ctx=ir.Context()
with ctx,ir.Location.unknown():
 source=ir.Module.parse(Path(SOURCE).read_text())
 candidates=[]
 def walk(operation):
  for region in operation.regions:
   for block in region.blocks:
    for op in block.operations:
     spec=_pointwise_packet_spec(op,multiplication=True,two_products=True)
     if spec is not None and str(op.results[0].type)=='tensor<1x8x2048xf32>':
      candidates.append((op,spec))
     walk(op.operation)
 walk(source.operation)
 assert len(candidates)==44,len(candidates)
 original,spec=candidates[0]
 out=ir.Module.create()
 inputs=list(original.operands[:-1]);ot=original.results[0].type
 with ir.InsertionPoint(out.body):
  function=ir.Operation.create('func.func',attributes={
   'sym_name':ir.StringAttr.get('forward'),
   'function_type':ir.TypeAttr.get(ir.FunctionType.get([v.type for v in inputs],[ot]))},regions=1)
  block=function.regions[0].blocks.append(*[v.type for v in inputs])
 with ir.InsertionPoint(block):
  mapping=dict(zip(inputs,block.arguments))
  constants={}
  def resolve(value):
   if value in mapping:return mapping[value]
   owner=getattr(value.owner,'operation',value.owner)
   assert isinstance(owner,ir.Operation) and owner.name=='arith.constant',str(owner)
   assert len(owner.results)==1 and not owner.operands and not owner.regions
   if value not in constants:
    constants[value]=ir.Operation.create('arith.constant',results=[value.type],
      attributes={a:owner.attributes[a] for a in owner.attributes}).results[0]
   return constants[value]
  # Resolve exact scalar captures in the function block before the generic.
  scalar=list(original.regions[0].blocks[0].operations)
  original_args=set(original.regions[0].blocks[0].arguments)
  local=set(original_args)
  for operation in scalar:
   for value in operation.operands:
    if value not in local:resolve(value)
   local.update(operation.results)
  empty=ir.Operation.create('tensor.empty',results=[ot])
  generic=ir.Operation.create('linalg.generic',results=[ot],
    operands=[*block.arguments,empty.results[0]],
    attributes={a:original.attributes[a] for a in original.attributes},regions=1)
  inner=generic.regions[0].blocks.append(*[v.type for v in original.regions[0].blocks[0].arguments])
  imap=dict(zip(original.regions[0].blocks[0].arguments,inner.arguments))
  with ir.InsertionPoint(inner):
   for operation in scalar:
    operands=[imap[v] if v in imap else resolve(v) for v in operation.operands]
    clone=ir.Operation.create(operation.operation.name,operands=operands,
      results=[v.type for v in operation.results],
      attributes={a:operation.attributes[a] for a in operation.attributes})
    imap.update(zip(operation.results,clone.results))
  ir.Operation.create('func.return',operands=[generic.results[0]])
 assert out.operation.verify()
 (W/'source_capsule.mlir').write_text(str(out)+'\n')
 record={'source':SOURCE,'source_sha256':hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest(),
  'matched_original_source_bodies':len(candidates),'selected_source_ordinal':0,
  'input_types':[str(v.type) for v in inputs],
  'indexing_maps':str(original.attributes['indexing_maps']),
  'exact_original_body':str(original.regions[0]),
  'source_capsule_sha256':hashlib.sha256((W/'source_capsule.mlir').read_bytes()).hexdigest(),
  'original_scalar_operations':len(scalar),'external_scalar_constants':len(constants),
  'selection_scope':'Experiment selects one actual source instance; production packet pass uses legality and broadcast maps only.'}
 (W/'source_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps({k:v for k,v in record.items() if k!='exact_original_body'}),flush=True)
'''
script = W/'run_extract.py'
script.write_text('SOURCE='+repr(str(SOURCE))+'\n'+RUNNER_PRELUDE.split('_PP_MARKERS =',1)[0]+body)
subprocess.run([str(m2m_python()),str(script)],check=True)
