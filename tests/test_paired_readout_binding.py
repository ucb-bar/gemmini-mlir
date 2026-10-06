from pathlib import Path
import subprocess
import pytest
from mlir_oot.paired_readout_binding import adapter,native_oracle
from mlir_oot.readout_store_plan import PairedReadoutPlan
from mlir_oot.golden_conv import ConvShape
from mlir_oot.captured_requant_bundle import scalar_oracle,build,rewrite_path


def test_adapter_owns_disjoint_byte_scratch_and_returns_output(tmp_path):
    s=ConvShape(3,3,16,16,output_dtype='i32',explicit_halo=True)
    plan=PairedReadoutPlan((.125,),(.125,.125),-(1<<31),(1<<31)-1)
    source=adapter(s,'boundary','kernel',plan)+native_oracle(s,'kernel',plan,scalar_oracle)
    source+='''
static int8_t a[400],b[2304],scratch[160],out[160];
int main(int argc,char**argv){for(int i=0;i<160;i++)scratch[i]=out[i]=-77;
memref2 r={0},ss={scratch,scratch,0,{9,16},{16,1}},c={out,out,0,{9,16},{16,1}};
memref4 aa={a,a,0,{1,5,5,16},{400,80,16,1}},bb={b,b,0,{3,3,16,16},{768,256,16,1}};
if(argc==2)ss.allocated=ss.aligned=out;
if(argc==3)ss.allocated=ss.aligned=a;
_mlir_ciface_boundary(&r,&aa,&bb,&ss,&c);
for(int i=0;i<144;i++)if(out[i]!=0||scratch[i]!=0)return 2;
for(int i=144;i<160;i++)if(out[i]!=-77||scratch[i]!=-77)return 3;
return r.aligned==out?0:4;}
'''
    c=tmp_path/'adapter.c';c.write_text(source);exe=c.with_suffix('')
    subprocess.run(['cc','-O2','-ffp-contract=off',str(c),'-lm','-o',str(exe)],check=True)
    assert subprocess.run([str(exe)],capture_output=True).returncode==0
    assert subprocess.run([str(exe),'alias-output'],capture_output=True).returncode<0
    assert subprocess.run([str(exe),'alias','input'],capture_output=True).returncode<0


def test_byte_scratch_rewrite_and_exact_contract():
    from mlir_oot.frontend.parse import parse_module
    from mlir_oot.contraction_patterns import match_integer_gemm
    from mlir_oot.captured_requant import inspect_chain
    from mlir_oot.exact_integer_readout import derive
    from mlir_oot.direct_conv_binding import serialize
    module=parse_module(Path(__file__).with_name('fixtures').joinpath('captured_requant.mlir').read_text())
    op=next(x for x in module.walk() if match_integer_gemm(x));chain=inspect_chain(op)
    proof=derive([*chain['scales'],chain['reciprocal']],relu=chain['relu'])
    pair=PairedReadoutPlan((.125,),(.125,.125),-(1<<31),(1<<31)-1).certificate()
    decl=rewrite_path(op,chain,'paired',None,integer_readout=proof,paired_readout=pair,source_sha='b'*64)
    parsed=parse_module(serialize(module,[decl]));parsed.verify()
    call=next(x for x in parsed.walk() if x.name=='func.call')
    assert call.operands[2].owner.name==call.operands[3].owner.name=='tensor.empty'
    assert call.operands[2].owner is not call.operands[3].owner
    assert all(str(x.type.get_element_type())=='i8' for x in call.operands[2:])
    assert 'gemmini.paired_readout' in decl.attributes


@pytest.mark.parametrize('policy,exact',[('unchecked',True),('source_proven',False),(True,True)])
def test_bad_policy_refuses_before_io(policy,exact):
    with pytest.raises(ValueError):build(Path('missing'),Path('missing'),Path('missing'),readout_pair_policy=policy,exact_integer_readout=exact)
