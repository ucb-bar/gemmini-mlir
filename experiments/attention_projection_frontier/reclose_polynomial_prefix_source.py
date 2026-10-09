"""Reproduce immutable table emission from its exact evaluator and source theorem."""
from pathlib import Path
import ctypes as C,json,hashlib
from merlin.llvmlower.polynomial_prefix_table import prepare,c_header
from merlin.llvmlower.rounded_polynomial_monotonicity import RoundedPolynomialMonotonicity
from merlin.llvmlower.source_numeric_capability import SourceNumericContract
B=Path(__file__).resolve().parents[2];W=B/'out/artifacts/probes/source-polynomial-prefix-table';G=W/'generation';N=B/'out/artifacts/probes/prepared-polynomial-constants/candidate'
pv=json.load(open(N/'build.json'))['roles']['native_numeric'];args=pv['proof'];args['plan_words']=tuple(args['plan_words']);proof=RoundedPolynomialMonotonicity(**args);contract=SourceNumericContract(**pv['contract'])
lib=C.CDLL(str(G/'evaluator.so'));f=lib.evaluate;f.argtypes=[C.c_uint32];f.restype=C.c_uint32
original=(B/'out/normal_composition/provider/native_numeric/monotone_bit_polynomial.h').read_text()
table=prepare(original,proof,contract,lambda x:int(f(x)),low_bits=8,maximum_bytes=4*1024*1024)
header=c_header(table,original,proof,contract,lambda x:int(f(x)))
assert header==(G/'prefix_table.h').read_text()
for role in ('native_numeric','target_numeric'):
 assert header==(W/'candidate'/role/'polynomial_prefix_table.h').read_text()
receipt={'status':'PASS','entries':len(table.words),'bytes':4*len(table.words),'header_sha256':hashlib.sha256(header.encode()).hexdigest(),'proof':vars(proof),'contract':vars(contract)}
(G/'source_reclosure.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
