from pathlib import Path
from collections import Counter
import hashlib,json,math as pymath
from merlin.frontends.linalg_mlir import parse_mlir_text
from xdsl.dialects import arith,math,scf
from xdsl.dialects.builtin import TensorType
from xdsl.dialects.linalg import ops as linalg
from xdsl.ir.affine import AffineDimExpr,AffineConstantExpr
W=Path(__file__).resolve().parent
SOURCE=Path('/scratch/agustin/tmp/merlin-golden-integration-20261004/out/artifacts/probes/smol-ordered-fma-schedule-20261005/bundle/model.mlir')
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
module=parse_mlir_text(SOURCE.read_text());module.verify();print('PARSED_SOURCE',flush=True)
def const(v):
 owner=v.owner
 if not isinstance(owner,arith.ConstantOp):raise ValueError('nonconstant loop bound')
 return owner.value.value.data

def generic_domain(op):
 maps=[x.data for x in op.indexing_maps];count=maps[0].num_dims;extents={}
 for affine,value in zip(maps,[*op.inputs,*op.outputs],strict=True):
  if not isinstance(value.type,TensorType):raise ValueError('nontensor generic domain')
  if affine.num_dims!=count or affine.num_symbols:raise ValueError('symbolic domain')
  for expr,n in zip(affine.results,value.type.get_shape(),strict=True):
   if n<0:raise ValueError('dynamic tensor domain')
   if isinstance(expr,AffineDimExpr):
    if expr.position in extents and extents[expr.position]!=n:raise ValueError('inconsistent domain')
    extents[expr.position]=n
   elif isinstance(expr,AffineConstantExpr):
    if not 0 <= expr.value < n:raise ValueError('out-of-range constant tensor index')
   else:raise ValueError('nonprojected domain')
 if sorted(extents)!=list(range(count)):raise ValueError('unbound generic dimension')
 return pymath.prod(extents.values())

def domain(op):
 factor=1;ancestor=op.parent_op();loops=[]
 while ancestor is not None:
  if isinstance(ancestor,scf.ForOp):
   lo,hi,step=const(ancestor.lb),const(ancestor.ub),const(ancestor.step)
   if step<=0:raise ValueError('nonpositive loop step')
   iterations=max(0,(hi-lo+step-1)//step);factor*=iterations;loops.append((lo,hi,step))
  elif isinstance(ancestor,linalg.GenericOp):factor*=generic_domain(ancestor)
  elif ancestor.name in ('scf.if','scf.while','tensor.generate','scf.parallel'):raise ValueError('conditional/unsupported source parent '+ancestor.name)
  ancestor=ancestor.parent_op()
 return factor,loops
counts=Counter();holes=Counter();fma_shapes=Counter();details=[]
for op in module.walk():
 if op.name not in ('math.fma','math.exp','math.rsqrt','math.tanh','math.log','math.erf','math.sin','math.cos','arith.mulf','arith.addf','arith.divf','arith.extf','arith.truncf'):continue
 dtype=str(op.results[0].type)
 try:factor,loops=domain(op)
 except ValueError as exc:holes[(op.name,dtype,str(exc))]+=1;continue
 counts[(op.name,dtype)]+=factor
 if op.name=='math.fma'and loops:
  a=op.operands[0].owner;b=op.operands[1].owner
  if a.name=='tensor.extract'and b.name=='tensor.extract':
   ancestor=op.parent_op();out=None
   while ancestor is not None:
    if isinstance(ancestor,scf.ForOp)and ancestor.results and isinstance(ancestor.results[0].type,TensorType):out=tuple(ancestor.results[0].type.get_shape());break
    ancestor=ancestor.parent_op()
   key=(tuple(a.operands[0].type.get_shape()),tuple(b.operands[0].type.get_shape()),out)
   fma_shapes[key]+=factor
 details.append({'operation':op.name,'dtype':dtype,'source_executions':factor,'enclosing_scf_ranges':loops})
r={'schema':'smol1906_typed_source_operation_census_v1','source_path':str(SOURCE),'source_sha256':sha(SOURCE),'scope':'Static source execution counts from typed projected linalg domains and positive constant SCF ranges; conditional or unsupported domains are holes. These are not machine instruction counts or hardware cycles.','counts':[{'operation':name,'dtype':dtype,'source_executions':value}for(name,dtype),value in counts.most_common()],'holes':[{'operation':name,'dtype':dtype,'reason':reason,'static_sites':value}for(name,dtype,reason),value in holes.items()],'explicit_ordered_fma_shapes':[{'lhs':ls,'rhs':rs,'output':out,'source_fmas':value}for(ls,rs,out),value in fma_shapes.most_common()],'token_usage_available':False}
(W/'source_census.json').write_text(json.dumps(r,indent=2)+'\n');(W/'source_census_details.json').write_text(json.dumps(details,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
