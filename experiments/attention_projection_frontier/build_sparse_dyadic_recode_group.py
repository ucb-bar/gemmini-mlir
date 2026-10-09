"""Same fixed sparse route with exact recode dyadic scaling implementation."""
from pathlib import Path
base=Path(__file__).resolve().parents[2]
s=(base/'experiments/attention_projection_frontier/build_sparse_dyadic_group.py').read_text()
s=s.replace("w=base/'out/sparse_dyadic_group'", "w=base/'out/sparse_dyadic_recode_group'")
s=s.replace('from adapt_sparse_dyadic import adapt', '''from adapt_sparse_dyadic import adapt as original_adapt
from merlin.llvmlower.sparse_dyadic_recode import c_header
from merlin.llvmlower.sparse_dyadic_products import plan_sparse_dyadic,SparseDyadicEffects
def adapt(source,destination):
 text=original_adapt(source,destination)
 for before,after in (("w->sparse_ai,w->sparse_am,epoch,&ao);","w->sparse_ai,w->sparse_am,epoch,&ao,&aproof,w->ap,w->astep);"),("w->sparse_bi,w->sparse_bm,epoch,&bo);","w->sparse_bi,w->sparse_bm,epoch,&bo,&bproof,bp,bstep);")):
  assert text.count(before)==1
  text=text.replace(before,after)
 (destination/'sparse_dyadic_products.h').write_text(c_header(plan_sparse_dyadic(192),SparseDyadicEffects(*([True]*6))))
 return text''')
start=s.index("llvm=Path(");end=s.index("oldbridge=",start)
s=s[:start]+'''previous=base/'out/sparse_dyadic_group'
old=json.loads((previous/'candidate/build.json').read_text())
catalog=old['catalog'];objects=[previous/'products'/entry['symbol']/'kernel.o' for entry in catalog]
for p in objects:assert hashlib.sha256(p.read_bytes()).hexdigest()==old['pins'][str(p)]
'''+s[end:]
(base/'out/sparse_dyadic_recode_group_driver.py').write_text(s)
exec(compile(s,str(__file__),'exec'))
