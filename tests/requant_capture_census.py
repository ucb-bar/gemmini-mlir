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


from mlir_oot.captured_requant import inspect_chain


def chain(op):
    c=inspect_chain(op)
    return c['dimensions'],c['scales'],c['bias'],c['reciprocal'],c['relu']


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
