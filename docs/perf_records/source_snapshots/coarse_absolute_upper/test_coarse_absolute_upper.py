"""Independent integer/rational proof of the positive magnitude envelope."""
from fractions import Fraction
import random
import pytest
from merlin.llvmlower.coarse_absolute_upper import CoarseAbsoluteUpperPlan, prepare_coarse_absolute_upper
from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier
from test_source_attention_frontier import PLAN

@pytest.mark.parametrize('bits', range(1,7))
def test_canonical_ceiling_and_exact_binary64_scale(bits):
    p=CoarseAbsoluteUpperPlan(bits)
    rng=random.Random(812)
    values=[0,1,127,128,16383,16384,2**21-1]+[rng.randrange(2**21) for _ in range(10000)]
    for n in values:
        for sign in [-1,1]:
            digits=[sign*((n>>(7*d))&127) for d in range(3)]
            magnitude=sum(abs(v)<<(7*d) for d,v in enumerate(digits))
            u=(magnitude+(1<<p.shift)-1)>>p.shift
            assert 0<=u<=1<<bits<=64
            assert n<=u*(1<<p.shift)
            assert n==0 or (u-1)*(1<<p.shift)<n
    # All finite BF16 source exponents plus zero's unit step are covered by
    # this broader source binary32 encoder-step range. Scaling is exact f64.
    for ea in range(-169,108):
        for eb in [-169,-148,-1,0,107]:
            count=(2**31-1)//((1<<bits)**2)
            integer=count*((1<<bits)**2)
            scaled=float(integer)*float(Fraction(2)**(2*p.shift))*float(Fraction(2)**(ea+eb))
            assert Fraction(scaled)==integer*Fraction(2)**(2*p.shift+ea+eb)

@pytest.mark.parametrize('value',[True,False,0,7,-1,1.5,None])
def test_bad_precision_refuses(value):
    with pytest.raises(ValueError):CoarseAbsoluteUpperPlan(value)

def test_accumulator_overflow_and_unknown_length_refuse():
    p=CoarseAbsoluteUpperPlan(6)
    p.validate_length((2**31-1)//4096)
    for k in [0,-1,True,1.5,2**31//4096]:
        with pytest.raises(ValueError):p.validate_length(k)

def test_default_identity_and_private_producer_contract():
    base=emit_source_attention_frontier(PLAN,symbol='p')
    assert base==emit_source_attention_frontier(PLAN,symbol='p',coarse_absolute_upper=None)
    p=CoarseAbsoluteUpperPlan(6)
    with pytest.raises(ValueError,match='exclusive canonical'):
        emit_source_attention_frontier(PLAN,symbol='p',coarse_absolute_upper=p)
    selected=emit_source_attention_frontier(PLAN,symbol='p',coarse_absolute_upper=p,
        integer_reconstruction=True,prepare_encoded_rows=True,prepare_required_norms=True,prepare_product_domain=True)
    assert 'merlin_absolute_upper_dot_prepare' in selected
    assert 'merlin_exact_absolute_dot_prepare' not in selected
    assert 'magnitude+32767u' in selected
    assert '*0x1p30' in selected
    assert 'double absolute_center[ROWS*CHUNK]' in selected
    # The existing structural producer matcher refuses unknown last-use text.
    with pytest.raises(ValueError,match='grammar changed'):
        prepare_coarse_absolute_upper('unknown',p)

@pytest.fixture(scope='module')
def upper_native(tmp_path_factory):
    import ctypes, shutil, subprocess
    from merlin.common.paths import merlin_dir
    from test_exact_absolute_dot_bounds import SOURCE
    cc=shutil.which('clang') or shutil.which('cc')
    if not cc:pytest.skip('native compiler required')
    root=tmp_path_factory.mktemp('upper-dot')
    source=SOURCE.replace('exact_absolute_dot_bounds.h','absolute_upper_dot_bounds.h').replace('merlin_exact_absolute_dot','merlin_absolute_upper_dot').replace('absolute[r*n+c]=t;','absolute[r*n+c]=2*t;')
    (root/'test.c').write_text(source)
    subprocess.run([cc,'-O2','-fno-fast-math','-ffp-contract=off','-shared','-fPIC','-I',str(merlin_dir()/'runtime/c'),str(root/'test.c'),'-lm','-o',str(root/'test.so')],check=True)
    lib=ctypes.CDLL(str(root/'test.so'))
    lib.compare.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    return lib

@pytest.mark.parametrize('case',['zero','signed','cancellation','tiny','rounding','overflow'])
def test_nonexact_upper_bound_encloses_source_and_refuses_invalid(upper_native,case):
    from test_exact_absolute_dot_bounds import test_exact_sums_enclose_original_fma
    test_exact_sums_enclose_original_fma(upper_native,(2,17,31),case)
    assert upper_native.refused()==1

@pytest.fixture(scope='module')
def coarse_group_native(tmp_path_factory):
    import ctypes as C, shutil, subprocess
    from merlin.common.paths import merlin_dir
    from test_source_attention_frontier import EXTRA, View
    cc=shutil.which('clang') or shutil.which('cc')
    if not cc:pytest.skip('native compiler required')
    root=tmp_path_factory.mktemp('coarse-group')
    text=emit_source_attention_frontier(PLAN,symbol='test_provider',coarse_absolute_upper=CoarseAbsoluteUpperPlan(6),integer_reconstruction=True,prepare_encoded_rows=True,prepare_required_norms=True,prepare_product_domain=True)
    (root/'test.c').write_text(text+EXTRA)
    subprocess.run([cc,'-O2','-fno-fast-math','-ffp-contract=off','-shared','-fPIC','-I',str(merlin_dir()/'runtime/c'),str(root/'test.c'),'-lm','-o',str(root/'test.so')],check=True)
    lib=C.CDLL(str(root/'test.so'))
    lib.test_provider_workspace_bytes.restype=C.c_size_t
    lib.run.argtypes=[C.POINTER(View),C.POINTER(View),C.c_void_p,C.c_size_t,C.c_int]
    lib.oracle.argtypes=[C.POINTER(View),C.c_void_p]
    return lib

@pytest.mark.parametrize('seed,masked,strided',[(1,False,False),(2,False,True),(3,True,True),(4,False,False)])
def test_rectangular_dirty_workspace_and_masks(coarse_group_native,seed,masked,strided):
    from test_source_attention_frontier import test_independent_source_quant_and_reused_dirty_workspace
    test_independent_source_quant_and_reused_dirty_workspace(coarse_group_native,seed,masked,strided)

@pytest.mark.parametrize('failure',['capacity','alignment','callback','nonfinite','stride','shape','overlapping_output'])
def test_coarse_group_refusal_preserves_output(coarse_group_native,failure):
    from test_source_attention_frontier import test_refusal_preserves_public_destination
    test_refusal_preserves_public_destination(coarse_group_native,failure)
