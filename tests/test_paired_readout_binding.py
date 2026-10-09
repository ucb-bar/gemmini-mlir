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


def legacy_scratch_fixture():
    from mlir_oot.frontend.parse import parse_module
    from mlir_oot.contraction_patterns import match_integer_gemm
    from mlir_oot.captured_requant import inspect_chain
    from mlir_oot.exact_integer_readout import derive
    from mlir_oot.direct_conv_binding import serialize
    from merlin.llvmlower.enclosed_readout import synthesize
    module=parse_module(Path(__file__).with_name('fixtures').joinpath('captured_requant.mlir').read_text())
    op=next(x for x in module.walk() if match_integer_gemm(x));chain=inspect_chain(op)
    proof=derive([*chain['scales'],chain['reciprocal']],relu=chain['relu'])
    pair=synthesize(proof['source_scales'],proof['accumulator_min'],proof['accumulator_max'],relu=proof['output_min']==0)
    assert pair['accepted']
    decl=rewrite_path(op,chain,'paired',None,integer_readout=proof,source_sha='b'*64)
    module=parse_module(serialize(module,[decl]))
    manifest={'source_sha256':'b'*64,'routes':[{'symbol':'paired','integer_readout':proof,'paired_readout':{'applied':True,'proof':pair['certificate']}}]}
    return module,manifest


def test_legacy_prepared_call_gets_fresh_byte_scratch():
    from mlir_oot.paired_readout_binding import prepare_paired_scratch
    module,manifest=legacy_scratch_fixture()
    assert prepare_paired_scratch(module,manifest)==1
    module.verify()
    call=next(x for x in module.walk() if x.name=='func.call')
    assert str(call.operands[2].type.get_element_type())=='i8'
    assert prepare_paired_scratch(module,manifest)==1
    module.verify()


def test_legacy_live_scratch_refuses_before_any_mutation():
    from mlir_oot.paired_readout_binding import prepare_paired_scratch
    from mlir_oot.direct_conv_binding import serialize
    from xdsl.dialects import tensor
    module,manifest=legacy_scratch_fixture()
    call=next(x for x in module.walk() if x.name=='func.call')
    clone=tensor.CastOp(call.operands[2],call.operands[2].type)
    call.parent.insert_op_before(clone,call)
    before=serialize(module,[])
    with pytest.raises(ValueError,match='sole-use'):
        prepare_paired_scratch(module,manifest)
    assert serialize(module,[])==before


def test_legacy_source_binding_refuses_without_mutation():
    from mlir_oot.paired_readout_binding import prepare_paired_scratch
    from mlir_oot.direct_conv_binding import serialize
    module,manifest=legacy_scratch_fixture();manifest['source_sha256']='c'*64
    before=serialize(module,[])
    with pytest.raises(ValueError,match='source/proof'):
        prepare_paired_scratch(module,manifest)
    assert serialize(module,[])==before


def test_all_routes_selected_before_legacy_scratch_mutation():
    import copy
    from mlir_oot.paired_readout_binding import prepare_paired_scratch
    from mlir_oot.direct_conv_binding import serialize
    module,manifest=legacy_scratch_fixture()
    later=copy.deepcopy(manifest['routes'][0]);later['symbol']='missing_binding'
    manifest['routes'].append(later);before=serialize(module,[])
    with pytest.raises(ValueError,match='one bound declaration'):
        prepare_paired_scratch(module,manifest)
    assert serialize(module,[])==before
