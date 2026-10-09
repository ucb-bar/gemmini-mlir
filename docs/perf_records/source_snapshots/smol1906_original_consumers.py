from pathlib import Path
from collections import Counter
import json,hashlib,math
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.ordered_fma_rewrite import _match
from xdsl.dialects.linalg import ops as linalg
W=Path(__file__).resolve().parent
p=Path('/scratch/agustin/tmp/merlin-golden-language-models-20261005/out/language_models/smol_source_sum_fma_bundle/model.mlir')
s=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();assert s=='4814509b8e11a5c819b1f9ae63f01f89f72e9edf9c3d0ec9d5cd0ab6b35de2dc'
module=parse_mlir_text(p.read_text());module.verify();print('ORIGINAL_SOURCE_PARSED',flush=True)
shapes=Counter();consumers=Counter();records=[];examples=[]
for ordinal,op in enumerate(module.walk()):
 transposed=_match(op)
 if transposed is None:continue
 a,b=op.inputs;out=op.results[0];ls,rs,os=tuple(a.type.get_shape()),tuple(b.type.get_shape()),tuple(out.type.get_shape());macs=math.prod(os)*ls[-1];key=(ls,rs,os,transposed);shapes[key]+=1
 uses=[]
 for use in out.uses:
  user=use.operation
  body_ops=[]
  if isinstance(user,linalg.GenericOp):body_ops=[(x.name,[str(v.type)for v in x.results])for x in user.body.block.ops]
  info={'operation':user.name,'operand_index':use.index,'input_types':[str(v.type)for v in user.operands],'output_types':[str(v.type)for v in user.results],'scalar_body':body_ops}
  uses.append(info);consumers[(user.name,tuple(str(v.type.element_type)for v in user.results if hasattr(v.type,'element_type')),tuple(x[0]for x in body_ops))]+=1
 records.append({'source_ordinal':ordinal,'lhs':ls,'rhs':rs,'output':os,'rhs_transposed':transposed,'source_macs':macs,'result_users':uses})
 if len(examples)<10:examples.append(records[-1])
r={'schema':'original_ordered_fma_source_consumer_census_v1','source_path':str(p),'source_sha256':s,'source_ordinals_are_binding_identifiers_only':True,'matched_contractions':len(records),'source_macs':sum(x['source_macs']for x in records),'shapes':[{'lhs':ls,'rhs':rs,'output':os,'rhs_transposed':t,'sites':n}for(ls,rs,os,t),n in shapes.most_common()],'consumer_families':[{'operation':name,'output_element_types':dtype,'scalar_operations':ops,'sites':n}for(name,dtype,ops),n in consumers.most_common()],'first_examples':examples,'scope':'Typed source operation matches and direct SSA consumers; no route or physical-cost claim.','token_usage_available':False}
(W/'original_consumers.json').write_text(json.dumps(r,indent=2)+'\n');(W/'original_consumers_details.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
