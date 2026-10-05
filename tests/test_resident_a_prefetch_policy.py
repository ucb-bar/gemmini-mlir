"""Explicit overlap alternative preserves legality and source numeric contracts."""
from dataclasses import asdict, replace
from pathlib import Path
import runpy

import pytest

from mlir_oot.dense_schedule import select_kernel
from mlir_oot.golden_gemm import Shape
from mlir_oot.golden_tuning import estimate
from mlir_oot.golden_compiler_export import select_contraction_export


@pytest.mark.parametrize('m,n,k', [(8,256,2048), (7,259,2067), (1,35,69), (16,32,8192)])
def test_default_off_and_resource_legal_explicit_overlap(m,n,k):
    control=Shape(m,n,k,'i32',bm=1,bn=min(16,(n+15)//16),wide_b=True)
    assert select_kernel(control)[0].shape is control
    generator,kind=select_kernel(control,resident_a_prefetch_policy=True)
    assert generator.shape.cache_a and generator.shape.prefetch_b
    assert kind=='dense_gemm:resident_a_prefetch'
    assert generator.shape.m==m and generator.shape.n==n and generator.shape.k==k
    assert generator.shape.bm==control.bm and generator.shape.bn==control.bn
    generator.shape.validate()
    assert estimate(generator.shape)['dma_request_bytes_upper']<=estimate(control)['dma_request_bytes_upper']
    assert estimate(generator.shape)['primitive_command_count']<=estimate(control)['primitive_command_count']
    if (n+15)//16 <= control.bn:
        assert estimate(generator.shape)['dma_request_bytes_upper']==estimate(control)['dma_request_bytes_upper']
        assert estimate(generator.shape)['primitive_command_count']==estimate(control)['primitive_command_count']


def test_numeric_epilogue_fields_are_preserved():
    control=Shape(7,35,69,'i8',bm=1,bn=3,bias=True,scale=.03125,relu=True,wide_b=True)
    generated,_=select_kernel(control,resident_a_prefetch_policy=True)
    before=asdict(control);after=asdict(generated.shape)
    assert {k:v for k,v in after.items() if k not in ('cache_a','prefetch_b')}=={
        k:v for k,v in before.items() if k not in ('cache_a','prefetch_b')}


@pytest.mark.parametrize('shape', [Shape(17,35,69,bm=2,bn=3), Shape(8,35,8193,bm=1,bn=3),
    Shape(8,35,16,bm=1,bn=3), Shape(8,35,69,bm=1,bn=3,cache_b=True),
    Shape(8,35,69,bm=1,bn=3,separate_b_bank=True)])
def test_unsafe_or_competing_resource_choices_refuse(shape):
    with pytest.raises(ValueError):select_kernel(shape,resident_a_prefetch_policy=True)


def test_command_policy_and_explicit_policy_cannot_mix():
    control=Shape(8,256,2048,bm=1,bn=16)
    with pytest.raises(ValueError):
        select_kernel(control,resident_a_prefetch_policy=True,resident_a_command_policy=True)


def test_current_source_controls_alternative_without_name_selector(tmp_path):
    factory=runpy.run_path(str(Path(__file__).with_name('test_golden_compiler_export.py')))['source']
    path=tmp_path/'input.mlir';path.write_text(factory(7,259,2067))
    baseline=select_contraction_export(path,'operation',large_n=True)
    selected=select_contraction_export(path,'operation',large_n=True,dense_input_policy='resident_a_prefetch')
    assert baseline['source_sha256']==selected['source_sha256']
    assert baseline['binding']==selected['binding']
    assert selected['generator'].shape.prefetch_b and selected['generator'].shape.cache_a
    assert baseline['alternative'].id!=selected['alternative'].id
