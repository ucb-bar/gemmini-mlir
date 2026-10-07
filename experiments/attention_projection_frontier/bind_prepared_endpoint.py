"""Bind the qualified private source grammar to generic endpoint preparation."""
from pathlib import Path
import ast
from merlin.llvmlower.prepared_endpoint_dag import PreparedEndpointContract,c_header
# Reuse the immutable, measured transformation literally. No second C mechanism.
SOURCE=Path(__file__).with_name('build_prepared_endpoint_dag.py')

def bind_source(source):
 tree=ast.parse(SOURCE.read_text());loop=next(n for n in tree.body if isinstance(n,ast.For))
 start=next(i for i,n in enumerate(loop.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='anchor'for t in n.targets))
 end=next(i for i,n in enumerate(loop.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='write_text'and i>start)
 module=ast.Module(body=loop.body[start:end],type_ignores=[]);scope={'s':source};exec(compile(module,str(SOURCE),'exec'),scope);return scope['s']

def bind_headers(directory):
 (directory/'prepared_endpoint_dag.h').write_text(c_header(PreparedEndpointContract(*([True]*9))))
