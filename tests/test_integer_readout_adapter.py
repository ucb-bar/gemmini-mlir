import ctypes
import subprocess
import numpy as np
from mlir_oot.captured_requant_bundle import integer_adapter
from mlir_oot.exact_integer_readout import derive,evaluate
from mlir_oot.golden_conv import ConvShape


def test_direct_adapter_uses_caller_scratch_and_exact_i8_output(tmp_path):
    shape=ConvShape(3,3,16,16,1,output_dtype='i32',explicit_halo=True)
    proof=derive([.1525018811225891,.0011392629239708185,4.235067367553711],relu=True)
    count=shape.oh*shape.ow*shape.cout
    source=integer_adapter(shape,'boundary','kernel',True,proof)
    source+=f'void kernel(int8_t*a,int8_t*b,int32_t*c){{for(int i=0;i<{count};i++)c[i]=(i-10)*1367;}}'
    (tmp_path/'adapter.c').write_text(source)
    subprocess.run(['cc','-O2','-fPIC','-shared',str(tmp_path/'adapter.c'),'-o',str(tmp_path/'adapter.so')],check=True)
    class M2(ctypes.Structure):_fields_=[('allocated',ctypes.c_void_p),('aligned',ctypes.c_void_p),('offset',ctypes.c_int64),('sizes',ctypes.c_int64*2),('strides',ctypes.c_int64*2)]
    class M4(ctypes.Structure):_fields_=[('allocated',ctypes.c_void_p),('aligned',ctypes.c_void_p),('offset',ctypes.c_int64),('sizes',ctypes.c_int64*4),('strides',ctypes.c_int64*4)]
    def desc(cls,a,sh):return cls(a.ctypes.data,a.ctypes.data,0,(ctypes.c_int64*len(sh))(*sh),(ctypes.c_int64*len(sh))(*[int(np.prod(sh[i+1:])) for i in range(len(sh))]))
    a=np.zeros((1,5,5,16),np.int8);b=np.zeros((3,3,16,16),np.int8);out=np.full(count+16,-77,np.int8);scratch=np.full(count+16,12345,np.int32)
    args=[M2(),desc(M4,a,a.shape),desc(M4,b,b.shape),desc(M2,scratch,(9,16)),desc(M2,out,(9,16))]
    lib=ctypes.CDLL(str(tmp_path/'adapter.so'));fn=lib._mlir_ciface_boundary;fn.argtypes=[ctypes.c_void_p]*5
    fn(*[ctypes.byref(x) for x in args])
    expected=np.array([evaluate((i-10)*1367,proof) for i in range(count)],np.int8)
    assert np.array_equal(out[:count],expected)
    assert np.all(out[count:]==-77) and np.all(scratch[count:]==12345)
    assert args[0].aligned==out.ctypes.data


def test_rewrite_exposes_unique_i32_scratch_before_i8_destination():
    from pathlib import Path
    from mlir_oot.frontend.parse import parse_module
    from mlir_oot.contraction_patterns import match_integer_gemm
    from mlir_oot.captured_requant import inspect_chain
    from mlir_oot.captured_requant_bundle import rewrite_path
    from mlir_oot.direct_conv_binding import serialize
    module=parse_module(Path(__file__).with_name('fixtures').joinpath('captured_requant.mlir').read_text())
    op=next(x for x in module.walk() if match_integer_gemm(x));chain=inspect_chain(op)
    proof=derive([*chain['scales'],chain['reciprocal']],relu=chain['relu'])
    declaration=rewrite_path(op,chain,'exact_integer',None,integer_readout=proof,source_sha='a'*64)
    parsed=parse_module(serialize(module,[declaration]));parsed.verify()
    decl=next(x for x in parsed.walk() if x.name=='func.func' and x.sym_name.data=='exact_integer')
    call=next(x for x in parsed.walk() if x.name=='func.call')
    assert [x.data['bufferization.access'].data for x in decl.arg_attrs]==['read','read','write','write']
    assert call.operands[2].owner.name=='tensor.empty'
    assert str(call.operands[2].type.get_element_type())=='i32'
    assert str(call.operands[3].type.get_element_type())=='i8'
    assert call.results[0].type==call.operands[3].type
    assert decl.attributes['gemmini.integer_readout'].data['source_sha256'].data=='a'*64


def test_bound_adapter_checks_scratch_output_disjointness_before_producer(tmp_path):
    from merlin.llvmlower.integer_producer_range import IntegerSumProductsRange
    shape=ConvShape(3,3,16,16,1,output_dtype='i32',explicit_halo=True)
    domain=IntegerSumProductsRange(144,-128,127,-128,127)
    proof=derive([.01],-2359296,2359296,relu=True)
    for bound in (False,True):
        source=integer_adapter(shape,'boundary','kernel',True,proof,readout_options={'saturation_first':True,'packet':8},producer_range=domain if bound else None)
        source+='''
static int8_t a[400],b[2304];static int32_t accum[144];
void kernel(int8_t*x,int8_t*y,int32_t*z){for(int i=0;i<144;i++)z[i]=0;}
int main(void){memref2 r={0},s={accum,accum,0,{9,16},{16,1}},c={accum,accum,0,{9,16},{16,1}};
memref4 aa={a,a,0,{1,5,5,16},{400,80,16,1}},bb={b,b,0,{3,3,16,16},{768,256,16,1}};
_mlir_ciface_boundary(&r,&aa,&bb,&s,&c);return 0;}
'''
        p=tmp_path/f'alias_{bound}.c';p.write_text(source);exe=p.with_suffix('')
        subprocess.run(['cc','-O2',str(p),'-o',str(exe)],check=True)
        result=subprocess.run([str(exe)],capture_output=True)
        assert (result.returncode<0) if bound else (result.returncode==0)


def test_source_proven_policy_rejects_unknown_or_missing_readout_before_io():
    import pytest
    from pathlib import Path
    from mlir_oot.captured_requant_bundle import build
    with pytest.raises(ValueError,match='unknown readout domain policy'):
        build(Path('missing'),Path('missing'),Path('missing'),readout_domain_policy='unchecked')
    with pytest.raises(ValueError,match='requires exact integer readout'):
        build(Path('missing'),Path('missing'),Path('missing'),readout_domain_policy='source_proven')


def test_default_adapter_source_is_byte_identical_to_prior_provider():
    import hashlib
    from mlir_oot.golden_gemm import Shape as GemmShape
    # Verified against provider baa0da7 before introducing the new option.
    expected=iter(['52f43315fb1263e1e68ad64b04a72b87aff70653619ce29e9f942bd939b1ee5b','d360b6515ea17047f431a9e00cf956776c01e806b60603e94ad4a763d5c1b8e2','d04cd2baa18ea63e0f946a2893d1c246726f439b14bb4644031fd46e0fb9b921','328f441d16c54d22e0b61b430ecd76f628e250de097e3627e0e0bc6b84dc2e8e'])
    proof=derive([.01],-10000000,10000000,relu=True)
    for shape,direct in [(ConvShape(3,5,16,32,output_dtype='i32'),True),(GemmShape(3,17,19,output_dtype='i32'),False)]:
        for options in (None,{'saturation_first':True,'packet':8}):
            assert hashlib.sha256(integer_adapter(shape,'boundary','kernel',direct,proof,readout_options=options).encode()).hexdigest()==next(expected)
