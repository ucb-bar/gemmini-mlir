from mlir_oot.exact_integer_readout import derive, emit_conv_wrapper


def test_explicit_scratch_wrapper_has_no_static_mutable_storage():
    source=emit_conv_wrapper(derive([.5]),'adapter','kernel',64)
    assert 'int32_t*scratch,size_t scratch_elements' in source
    assert 'kernel(a,b,scratch)' in source
    assert 'static int32_t' not in source

