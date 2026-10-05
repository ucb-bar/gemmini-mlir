"""Explicit source-bound residual arithmetic experiment; exact by default.

A nonzero output-LSB policy is local to each i8 residual, never a whole-model
quality assertion. Native standins reproduce the two hardware load rounds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from xdsl.dialects import func, tensor
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, FloatAttr, IntegerAttr, StringAttr, TensorType, UnitAttr, f32 as F32, i8, i64
from xdsl.ir import Region
from .frontend.parse import parse_module
from .golden_resadd_proof import match, op_name, prove
from .golden_requant import f32
from .golden_resadd import build as build_kernel
from .golden_device_compile import compile_module
from .direct_conv_binding import serialize
from .no_fsm_audit import audit_elf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def policy(max_output_lsb):
    if type(max_output_lsb) is not int or max_output_lsb not in (0, 1):
        raise ValueError('max_output_lsb must be explicitly 0 or 1')
    return dict(kind='exact_source' if max_output_lsb == 0 else 'bounded_residual_output_lsb',
                max_output_lsb=max_output_lsb, domain='all 65536 signed-i8 operand pairs',
                scope='per residual output; not a whole-model quality guarantee')


def inspect(quantize, max_output_lsb=0, *, implementation="gemmini"):
    if implementation not in ("gemmini", "cpu_lut"):
        raise ValueError("unknown residual implementation")
    if implementation == "cpu_lut" and max_output_lsb != 0:
        raise ValueError("CPU lookup residual requires exact policy")
    numeric = policy(max_output_lsb)
    source = match(quantize)
    ratios = [f32(source[k] / source['output_scale']) for k in ('lhs_scale', 'rhs_scale')]
    factor = max(1.0, *ratios)
    proof = prove(**source, lhs_load=f32(ratios[0]/factor), rhs_load=f32(ratios[1]/factor), readout=factor)
    if implementation == 'cpu_lut':
        table = source_table(source)
        proof = dict(proof='exhaustive source float32 lookup table', pairs=65536,
                     exact=True, mismatched_pairs=0, max_output_lsb_error=0,
                     source=source, primitive={}, table_sha256=hashlib.sha256(table.tobytes()).hexdigest())
    if proof['max_output_lsb_error'] > max_output_lsb:
        raise ValueError('complete residual proof exceeds explicitly requested output-LSB policy')
    shape = list(quantize.results[0].type.get_shape())
    if not shape or any(x <= 0 for x in shape) or math.prod(shape) % 64:
        raise ValueError('residual flattening requires a positive static extent divisible by 64')
    addition = quantize.operands[0].owner
    cleanup_ops=[]
    if source['relu']:
        cleanup_ops.append(addition)
        addition = addition.operands[0].owner
    cleanup_ops.append(addition)
    cleanup_ops.extend(dq.owner for dq in addition.operands[:2])
    inputs = [dq.owner.operands[0] for dq in addition.operands[:2]]
    if any(v.type != quantize.results[0].type for v in inputs):
        raise ValueError('residual tensor layouts/types differ')
    return dict(inputs=inputs, cleanup_ops=cleanup_ops, shape=shape, m=math.prod(shape)//64, n=64,
                proof=proof, numeric_policy=numeric)


def reassociation(rank):
    return ArrayAttr([ArrayAttr([IntegerAttr(i,i64) for i in range(rank)])])


def reshape(value, shape):
    ops=[]
    if tuple(value.type.get_shape()) == tuple(shape):
        return ops, value
    if len(value.type.get_shape()) != 1:
        flat=tensor.CollapseShapeOp(operands=[value],result_types=[TensorType(i8,[math.prod(shape)])],properties={'reassociation':reassociation(len(value.type.get_shape()))})
        ops.append(flat);value=flat.results[0]
    if len(shape) != 1:
        expand=tensor.ExpandShapeOp(value,[],reassociation(len(shape)),list(shape),TensorType(i8,shape))
        ops.append(expand);value=expand.result
    return ops,value


def binding_attributes(route, source_sha, receipt_sha):
    proof_sha=hashlib.sha256(canonical(route['proof']).encode()).hexdigest()
    numeric=route['numeric_policy']
    qparams={k:FloatAttr(v,F32) for k,v in route['proof']['source'].items() if k!='relu'}
    qparams['relu']=IntegerAttr(int(route['proof']['source']['relu']),i64)
    qparams.update({k:FloatAttr(v,F32) for k,v in route['proof']['primitive'].items()})
    return {'gemmini.source_sha256':StringAttr(source_sha),
            'gemmini.capture_receipt_sha256':StringAttr(receipt_sha),
            'gemmini.residual_proof_sha256':StringAttr(proof_sha),
            'gemmini.qparams':DictionaryAttr(qparams),
            'gemmini.numeric_policy':DictionaryAttr({'kind':StringAttr(numeric['kind']),
                'max_output_lsb':IntegerAttr(numeric['max_output_lsb'],i64),
                'domain_pairs':IntegerAttr(65536,i64)})}


def rewrite(quantize, route, symbol, source_sha, receipt_sha):
    module=quantize
    while module.parent_op() is not None:module=module.parent_op()
    if any(o.name=='func.func' and o.sym_name.data==symbol for o in module.body.block.ops):
        raise ValueError('residual symbol already exists')
    ops=[];inputs=[]
    for value in route['inputs']:
        views,value=reshape(value,[route['m'],64]);ops+=views;inputs.append(value)
    outtype=TensorType(i8,[route['m'],64]);empty=tensor.EmptyOp([],outtype);ops.append(empty)
    call=func.CallOp(symbol,[*inputs,empty.tensor],[outtype]);ops.append(call)
    views,result=reshape(call.results[0],route['shape']);ops+=views
    if result.type != quantize.results[0].type:raise ValueError('residual output reshape mismatch')
    quantize.parent.insert_ops_before(ops,quantize)
    quantize.results[0].replace_all_uses_with(result)
    quantize.parent.erase_op(quantize)
    for old in route['cleanup_ops']:
        if old.parent is not None and all(not result.uses for result in old.results):
            old.parent.erase_op(old)
    accesses=ArrayAttr([DictionaryAttr({'bufferization.access':StringAttr(x)}) for x in ('read','read','write')])
    declaration=func.FuncOp(symbol,([outtype]*3,[outtype]),Region(),visibility='private',arg_attrs=accesses)
    declaration.attributes.update(binding_attributes(route,source_sha,receipt_sha))
    declaration.attributes['llvm.emit_c_interface']=UnitAttr();module.body.block.add_op(declaration)
    return declaration


def adapter(route, symbol, kernel):
    m=route['m']
    identity=','.join('1' if i==j else '0' for i in range(16) for j in range(16))
    checks=' || '.join(f'!{v}->aligned || {v}->offset<0 || {v}->sizes[0]!={m} || {v}->sizes[1]!=64 || {v}->strides[0]!=64 || {v}->strides[1]!=1' for v in ('a','b','c'))
    return f'''#include <stdint.h>
#ifndef GEMMINI_RESIDUAL_ABI
#define GEMMINI_RESIDUAL_ABI
typedef struct {{void *allocated,*aligned; intptr_t offset,sizes[2],strides[2];}} residual_memref2;
#endif
static const int8_t {symbol}_identity[256] __attribute__((aligned(64)))={{{identity}}};
static int8_t {symbol}_scratch[1024] __attribute__((aligned(64)));
extern void {kernel}(const int8_t*,const int8_t*,int8_t*,const int8_t*,int8_t*);
void _mlir_ciface_{symbol}(residual_memref2 *r,residual_memref2 *a,residual_memref2 *b,residual_memref2 *c) {{
 if({checks}) __builtin_trap();
 {kernel}((const int8_t*)a->aligned+a->offset,(const int8_t*)b->aligned+b->offset,(int8_t*)c->aligned+c->offset,{symbol}_identity,{symbol}_scratch);*r=*c;
}}
'''


def native_oracle(route,kernel):
    p=route['proof']['primitive'];low=0 if route['proof']['source']['relu'] else -128
    return f'''#include <math.h>
void {kernel}(const int8_t*a,const int8_t*b,int8_t*c,const int8_t*identity,int8_t*scratch) {{
 (void)identity;(void)scratch;
 for(int i=0;i<{route['m']*64};i++) {{
  float x=nearbyintf((float)a[i]*{p['lhs_load'].hex()}f);
  float y=nearbyintf((float)b[i]*{p['rhs_load'].hex()}f);
  if(x<-128)x=-128;if(x>127)x=127;
  if(y<-128)y=-128;if(y>127)y=127;
  float z=nearbyintf((float)((int)x+(int)y)*{p['readout'].hex()}f);
  if(z<{low})z={low};if(z>127)z=127;c[i]=(int8_t)z;
 }}
}}
'''


def source_table(source):
    import numpy as np
    if any(not math.isfinite(source[k]) or source[k] <= 0 for k in ('lhs_scale','rhs_scale','output_scale')):
        raise ValueError('positive finite source scales required')
    a=np.arange(-128,128,dtype=np.float32)[:,None]
    b=np.arange(-128,128,dtype=np.float32)[None,:]
    z=np.add(a*np.float32(source['lhs_scale']), b*np.float32(source['rhs_scale']),dtype=np.float32)
    if source['relu']:z=np.maximum(z,np.float32(0))
    return np.clip(np.rint(z*np.float32(f32(1.0/source['output_scale']))),0 if source['relu'] else -128,127).astype(np.int8)


def lookup_kernel(route,kernel):
    table=source_table(route['proof']['source'])
    values=','.join(str(int(x)) for x in table.ravel())
    return f'''#include <stdint.h>
static const int8_t {kernel}_table[65536] __attribute__((aligned(64)))={{{values}}};
void {kernel}(const int8_t*a,const int8_t*b,int8_t*c,const int8_t*identity,int8_t*scratch) {{
 (void)identity;(void)scratch;
 for(int i=0;i<{route['m']*64};i++)
  c[i]={kernel}_table[((unsigned)((int)a[i]+128)<<8)|(unsigned)((int)b[i]+128)];
}}
'''


def compile_adapter(source, output, llvm_bin):
    source,output,llvm_bin=map(Path,(source,output,llvm_bin))
    command=[str(llvm_bin/'clang'),'--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-O2','-ffreestanding','-fno-builtin','-c',str(source),'-o',str(output)]
    subprocess.run(command,check=True,capture_output=True)
    return dict(compiler_argv=command,compiler_sha256=sha(llvm_bin/'clang'),source_sha256=sha(source),object_sha256=sha(output))


def build(capture,llvm_bin,output,*,max_output_lsb=0,implementation="gemmini"):
    if implementation not in ("gemmini", "cpu_lut"):
        raise ValueError("unknown residual implementation")
    if implementation == "cpu_lut" and max_output_lsb != 0:
        raise ValueError("CPU lookup residual requires exact policy")
    capture,llvm_bin,output=map(Path,(capture,llvm_bin,output));numeric=policy(max_output_lsb)
    output.mkdir(parents=True,exist_ok=False)
    source=(capture/'model.mlir').read_text();source_sha=sha(capture/'model.mlir');receipt_sha=sha(capture/'capture_receipt.json')
    pins=json.loads((capture/'capture_receipt.json').read_text())['artifacts']
    for name in ('model.mlir','weights.safetensors','weights.safetensors.manifest.json'):
        if sha(capture/name)!=pins[name]['sha256']:raise ValueError('capture artifact changed: '+name)
    module=parse_module(source);objects=[];native=[];routes=[];refused=[];declarations=[]
    for op in list(module.walk()):
        if op_name(op)!='quant_ext.quantize_per_tensor':continue
        region=getattr(op.attributes.get('prov.region_id'),'data','')
        try:route=inspect(op,max_output_lsb,implementation=implementation)
        except ValueError as error:
            refused.append(dict(region=region,reason=str(error)));continue
        symbol=f'gemmini_residual_{len(routes)}';kernel=symbol+'_kernel';work=output/symbol
        declarations.append(rewrite(op,route,symbol,source_sha,receipt_sha))
        if implementation == 'cpu_lut':
            work.mkdir(parents=True)
            c=adapter(route,symbol,kernel)+lookup_kernel(route,kernel)
            (work/'adapter.c').write_text(c)
            adapter_compilation=compile_adapter(work/'adapter.c',work/'adapter.o',llvm_bin)
            compilation=dict(implementation='cpu_lut',table_bytes=65536)
            objects.append(work/'adapter.o');native.append(c)
        else:
            scales=route['proof']['primitive']
            device=build_kernel(route['m'],64,lhs_scale=scales['lhs_load'],rhs_scale=scales['rhs_load'],output_scale=scales['readout'],relu=route['proof']['source']['relu'])
            device.body.block.first_op.properties['sym_name']=StringAttr(kernel)
            compilation=compile_module(device,llvm_bin,work)
            c=adapter(route,symbol,kernel);(work/'adapter.c').write_text(c)
            adapter_compilation=compile_adapter(work/'adapter.c',work/'adapter.o',llvm_bin)
            objects.extend([work/'kernel.o',work/'adapter.o']);native.append(c+native_oracle(route,kernel))
        routes.append({k:v for k,v in route.items() if k not in ('inputs','cleanup_ops')}|dict(region=region,symbol=symbol,kernel=kernel,compilation=compilation,adapter_compilation=adapter_compilation,proof_sha256=hashlib.sha256(canonical(route['proof']).encode()).hexdigest()))
    module.verify();printed=serialize(module,declarations);parse_module(printed).verify();(output/'rewritten.mlir').write_text(printed)
    (output/'native_oracle.c').write_text('\n'.join(native))
    linked=output/'residual.o';audit=None
    if objects:
        linker=llvm_bin/'ld.lld'
        if not linker.is_file():linker=Path(shutil.which('ld.lld'))
        subprocess.run([str(linker),'-r',*[str(x) for x in objects],'-o',str(linked)],check=True,capture_output=True)
        audit=audit_elf(linked.read_bytes())
        if audit['status']!='pass':raise ValueError('residual object contains forbidden instruction')
    result=dict(schema='gemmini_captured_residual_bundle_v1',implementation=implementation,source_sha256=source_sha,capture_receipt_sha256=receipt_sha,weights_sha256=pins['weights.safetensors']['sha256'],manifest_sha256=pins['weights.safetensors.manifest.json']['sha256'],rewritten_sha256=sha(output/'rewritten.mlir'),object_sha256=sha(linked) if objects else None,native_oracle_sha256=sha(output/'native_oracle.c'),numeric_policy=numeric,routes=routes,refused=refused,nofsm_audit=audit,scope='Explicit local arithmetic experiment; unchanged original golden and whole-model quality gate still required; no default promotion.')
    (output/'residual.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('--llvm-bin',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-output-lsb',type=int,choices=(0,1),default=0);p.add_argument('--implementation',choices=('gemmini','cpu_lut'),default='gemmini');a=p.parse_args()
    result=build(a.capture,a.llvm_bin,a.output,max_output_lsb=a.max_output_lsb,implementation=a.implementation)
    print(json.dumps(dict(routes=len(result['routes']),refused=len(result['refused']),numeric_policy=result['numeric_policy']),indent=2))

if __name__=='__main__':main()
