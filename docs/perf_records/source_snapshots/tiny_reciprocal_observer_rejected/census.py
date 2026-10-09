from pathlib import Path
import hashlib,json,math
from xdsl.dialects import arith,builtin,linalg
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text
W=Path(__file__).resolve().parent
P=Path('/scratch/agustin/tmp/gemmini-smol-encoded-zero-groups-20261005/out/artifacts/probes/tiny1880-host-classes-20261005/prebuffer.mlir')
assert hashlib.sha256(P.read_bytes()).hexdigest()=='04a9a3eccb4578c446d51ad6ed23694911936bbbcc0bf00efd57b8c65f43a518'
m=parse_mlir_text((W/"prebuffer.generic.mlir").read_text());before=text(m,generic=True);rows=[]
def value(op):
 if isinstance(op,arith.ConstantOp) and isinstance(op.value,builtin.FloatAttr):return op.value.value.data
for ordinal,op in enumerate(m.walk()):
 if not isinstance(op,arith.DivfOp):continue
 generic=op.parent_op()
 while generic is not None and not isinstance(generic,linalg.GenericOp):generic=generic.parent_op()
 assert generic is not None
 domain=generic.results[0].type.get_shape();constant_den=value(op.rhs.owner);constant_num=value(op.lhs.owner)
 chain=[];v=op.result
 for _ in range(3):
  users=[use.operation for use in v.uses]
  if len(users)!=1 or not isinstance(users[0],arith.MulfOp):break
  user=users[0];chain.append(user);v=user.result
 ending=[use.operation for use in v.uses]
 observer=constant_num==1 and len(chain)==3 and len(ending)==1 and isinstance(ending[0],arith.MaximumfOp) and generic.results[0].type.element_type==builtin.i8
 rows.append(dict(source_ordinal=ordinal,domain=domain,logical_elements=math.prod(domain),numerator_constant=constant_num,denominator_constant=constant_den,sole_successive_source_f32_multiplies=len(chain),observed_dtype=str(generic.results[0].type.element_type),classification='reciprocal_three_rounded_mul_to_i8_clamp' if observer else ('constant_divisor_folded_by_upstream' if constant_den is not None else 'live_float_or_other_DAG'),direct_division_integer_observer=False))
assert text(m,generic=True)==before
assert len([x for x in rows if x['classification']=='reciprocal_three_rounded_mul_to_i8_clamp'])==22
summary={k:dict(source_ops=len(a:=[x for x in rows if x['classification']==k]),logical_divisions=sum(x['logical_elements'] for x in a)) for k in sorted({x['classification']for x in rows})}
r=dict(schema='tiny_actual_typed_division_observer_census_v1',source_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),source_unchanged=True,summary=summary,rows=rows,scope='Typed source scalar operations over exact static domains; original constant divisions can disappear upstream. Logical counts are not machine retired instructions/DRAM/cycles. Direct division→RNE threshold refuses every reciprocal+three rounded multiply closure and live floating result.',profile1901_scope='Older frozen1880 conserved profile, notcurrent1926 normhoist; preDown22hostgaps127539158cycles include all surrounding loads/dequant/poly/divide/quant/store.',compiler_promotion=False,token_usage_available=False)
(W/'source_division_census.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(summary))
