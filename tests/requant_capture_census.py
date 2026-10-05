"""Census exact captured scalar scales/biases; prove unary requantizable channels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from xdsl.ir import Operation,BlockArgument
from mlir_oot.frontend.parse import parse_module
from mlir_oot.contraction_patterns import match_integer_gemm
from mlir_oot.golden_requant import f32,synthesize_bias
from merlin.runtime.captured_constants import verify_capture_constant


def scalar(value):
    op=value.owner
    if not isinstance(op,Operation) or op.name!='tensor.splat':raise ValueError('not scalar splat')
    c=op.operands[0].owner
    if not isinstance(c,Operation) or c.name!='arith.constant':raise ValueError('not constant scalar')
    return c.properties.get('value',c.attributes.get('value')).value.data


def chain(op):
    dims=match_integer_gemm(op);v=op.results[0];scales=[];bias=None;relu=False
    while True:
        uses=list(v.uses)
        if len(uses)!=1:raise ValueError('fanout before quantization')
        u=uses[0].operation;name=u.name
        if name=='builtin.unregistered':name=u.attributes['op_name__'].data
        if name=='quant_ext.quantize_per_tensor':
            if bias is None or len(scales)!=2:raise ValueError('expected two dequant scales and one bias')
            if scalar(u.operands[2])!=0:raise ValueError('nonzero output zero point')
            if u.properties['quant_min'].value.data!=-128 or u.properties['quant_max'].value.data!=127:raise ValueError('non-int8 quant limits')
            return dims,scales,bias,1.0/f32(scalar(u.operands[1])),relu
        if name in ('tensor.collapse_shape','tensor.expand_shape','linalg.transpose'):
            v=u.results[0];continue
        if name!='linalg.generic':raise ValueError('boundary '+name)
        ops=[x for x in u.regions[0].block.ops if x.name!='arith.constant'];names=[x.name for x in ops]
        if names==['arith.sitofp','linalg.yield'] and not scales and bias is None:pass
        elif names==['arith.mulf','linalg.yield'] and bias is None:
            other=next(x for x in u.operands[:-1] if x is not v);scales.append(f32(scalar(other)))
        elif names==['arith.addf','linalg.yield'] and bias is None:
            other=next(x for x in u.operands[:-1] if x is not v)
            while isinstance(other.owner,Operation) and other.owner.name in ('tensor.collapse_shape','tensor.expand_shape'):
                other=other.owner.operands[0]
            if not isinstance(other,BlockArgument) or tuple(other.type.get_shape())!=(dims.n,) or str(other.type.get_element_type())!='f32':raise ValueError('not captured channel bias')
            bias=other
        elif names==['arith.maximumf','linalg.yield'] and bias is not None:
            maximum=ops[0];other=next(x for x in maximum.operands if not isinstance(x,BlockArgument))
            if other.owner.name!='arith.constant' or other.owner.properties['value'].value.data!=0:raise ValueError('nonzero activation clamp')
            relu=True
        elif 'arith.addf' in names and bias is not None:raise ValueError('residual addition before quantization')
        else:raise ValueError('non-unary epilogue '+','.join(names))
        v=u.results[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('capture',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    source=(a.capture/'model.mlir').read_text();m=parse_module(source);receipt=json.loads((a.capture/'capture_receipt.json').read_text())['artifacts']
    rows=[]
    for op in m.walk():
        if match_integer_gemm(op) is None:continue
        rid=getattr(op.attributes.get('prov.region_id'),'data','');row={'region':rid}
        try:
            dims,scales,bias,reciprocal,relu=chain(op)
            constant=verify_capture_constant(manifest_path=a.capture/'weights.safetensors.manifest.json',manifest_sha256=receipt['weights.safetensors.manifest.json']['sha256'],safetensors_path=a.capture/'weights.safetensors',safetensors_sha256=receipt['weights.safetensors']['sha256'],entry_argument_index=bias.index,source_shape=[dims.n],source_dtype='f32',max_payload_bytes=dims.n*4)
            values=np.frombuffer(constant.logical_payload,dtype='<f4');bound=dims.k*16384
            proof=synthesize_bias(scales,values,reciprocal,max(-(1<<31),-bound),min((1<<31)-1,bound),relu)
            row.update(status='proved' if proof['accepted_channels']==dims.n else 'bias_refused',proof=proof,bias_argument=bias.index,bias_payload_sha256=constant.payload_sha256,source_bias_nonzero=int(np.count_nonzero(values)))
            print(rid,row['status'],proof['accepted_channels'],'/',dims.n,flush=True)
        except ValueError as e:row.update(status='structure_refused',reason=str(e));print(rid,str(e),flush=True)
        rows.append(row)
    report={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'weights_sha256':receipt['weights.safetensors']['sha256'],'scope':'Captured unary epilogue candidate census only; no graph rewrite or runtime freezing performed. Residuals and fanout refused. Proof retains original sequential f32 multiplication, bias, ReLU, reciprocal, nearest-even and clamp.','rows':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
