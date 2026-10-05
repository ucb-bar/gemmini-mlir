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
